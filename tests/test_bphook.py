# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import tempfile

import pytest
import torch
from hydra import compose, initialize

from lineas.hooks import BPLinearHook, PostprocessAndSaveHook
from lineas.models import get_model
from lineas.models.model_with_hooks import ModelWithHooks
from lineas.scripts.learn_intervention import learn_intervention


@pytest.mark.parametrize(
    "first_layer",
    [
        (True),
        (False),
    ],
)
def test_backprop(first_layer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Interventions for 2 layers
    hook0 = BPLinearHook(
        module_name="transformer.h.0.mlp.c_proj:0",
        dim=2,
        strength=1.0,
        device=device,
    )
    hook1 = BPLinearHook(
        module_name="transformer.h.1.mlp.c_proj:0",
        dim=2,
        strength=1.0,
        device=device,
    )

    # Hook to collect responses
    save_layer = (
        "transformer.h.0.mlp.c_proj:0"
        if first_layer
        else "transformer.h.1.mlp.c_proj:0"
    )
    hook_rx = PostprocessAndSaveHook(
        module_name=save_layer,
        pooling_op_names=[
            "mean",
        ],
        save_fields=[
            "id",
        ],
        return_outputs=True,  # return tensors
        output_path=None,  # do not save to disk
        keep_gradients=True,  # do not detach responses
    )

    model, tokenizer = get_model(
        model_path="sshleifer/tiny-gpt2",
        cache_dir="tests/data",
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_task="text_generation",
    )

    model_with_hooks = ModelWithHooks(
        module=model,
        hooks=[hook0, hook1, hook_rx],
        device=model.device,
    )
    model_with_hooks.register_hooks()

    # Save original parameters for test
    params_0 = {
        name: param.detach().clone() for name, param in hook0.named_parameters()
    }
    params_1 = {
        name: param.detach().clone() for name, param in hook1.named_parameters()
    }

    criterion = torch.nn.MSELoss()  # Mean squared error loss, to have something
    optimizer = torch.optim.Adam(model_with_hooks.get_hook_parameters(), lr=0.1)

    # Example data
    targets = 10 * torch.ones(
        (2,), requires_grad=False, device=device
    )  # Some arbitrary target output

    inputs = {
        "input_ids": torch.randint(
            10, 100, size=(1, 3), requires_grad=False, device=device
        ),
        "attention_mask": torch.ones((1, 3), requires_grad=False, device=device),
        "id": [0],
    }  # Single data point
    for _step in range(10):
        # Forward pass
        model_with_hooks.update_hooks(batch_idx=0, batch=inputs)
        model_with_hooks(batch=inputs)
        outputs = model_with_hooks.get_hook_outputs()
        rx = outputs[save_layer][0]["responses"]
        loss = criterion(rx, targets)
        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store new parameters
    params_0_new = {
        name: param.detach().clone() for name, param in hook0.named_parameters()
    }
    params_1_new = {
        name: param.detach().clone() for name, param in hook1.named_parameters()
    }

    for name in params_0_new.keys():
        # Make sure all parameters changed in the first layer
        assert not torch.allclose(params_0[name], params_0_new[name])
        # In the second layer, parameters should not change if the loss is applied on an earlier layer (first_layer=True)
        if first_layer:
            assert torch.allclose(params_1[name], params_1_new[name])
        else:
            assert not torch.allclose(params_1[name], params_1_new[name])


def test_learn_bphook():
    data_dir = "tests/data"

    # Assuming that the main function doesn't have any side effects and returns None when successful
    with (
        tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder,
        initialize(version_base=None, config_path="../lineas/configs"),
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # config is relative to a module
        cfg = compose(
            config_name="text_generation",
            overrides=[
                f"device={device}",
                "model=tiny-gpt2",
                "model.module_names=['transformer.h.0.mlp.c_fc', 'transformer.h.1.mlp.c_proj']",
                "+model.target_module_names=['transformer.h.1.mlp.c_proj']",
                "responses.tag=toxicity-responses",
                "responses.max_batches=2",
                "batch_size=4",
                "interventions.batch_size=8",
                "intervention_params=lineas",
                "intervention_params.name=lineas",
                "intervention_params.optimization_params.steps=10",
                "intervention_params.optimization_params.learning_rate=0.001",
                "intervention_params.optimization_params.init_identity=true",
                "intervention_params.optimization_params.criterion=wasserstein",
                f"data_dir={data_dir}",
                f"cache_dir={tempfolder}",
                f"interventions.cache_dir={tempfolder}",
                "responses.enabled=true",
                "wandb.mode=disabled",
            ],
        )

        learn_intervention(cfg)
