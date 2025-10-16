# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import tempfile

import numpy as np
import pytest
import torch
from hydra import compose, initialize

from lineas.hooks import GaussianAcTHook, get_hook
from lineas.scripts.learn_intervention import (
    BaseEstimator,
    learn_intervention,
)
from lineas.utils.utils import get_device


@pytest.mark.parametrize(
    "intervention_name,estimation_strategy",
    [
        ("none", "atonce"),
        ("aura", "atonce"),
        ("mean_act", "atonce"),
        ("mean_act", "incr"),
        ("gaussian_act", "atonce"),
        ("linear_act", "atonce"),
        ("linear_act", "incr"),
    ],
)
def test_hook_with_hook_args(intervention_name, estimation_strategy):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../lineas/configs"):
            cfg = compose(
                config_name="text_generation",
                overrides=[
                    "device=cpu",
                    "model.model_path=sshleifer/tiny-gpt2",
                    "model.module_names=['transformer.h.0.mlp.c_proj:0', 'transformer.h.1.mlp.c_proj:0']",
                    "responses.tag=toxicity-responses",
                    f"intervention_params={intervention_name}",
                    f"intervention_params.estimation_strategy={estimation_strategy}",
                    # setting to mean because we only have mean-pooled responses stored for tests
                    "intervention_params.pooling_op=mean",
                    "data_dir=tests/data",
                    "cache_dir=tests/data",
                    f"interventions.cache_dir={tempfolder}",
                    "responses.enabled=false",
                    "wandb.mode=disabled",
                ],
            )

            # interventions_manager = BaseEstimator(
            #     ResponsesManager.get_output_path(cfg.responses), cfg.interventions
            # )
            # # This call also tests the "fit()" method
            # interventions_manager.learn_intervention_all()
            learn_intervention(cfg)

        # Now create a new hook and load its state using the one learnt through "fit()"
        hook = get_hook(
            intervention_name,
            module_name="transformer.h.1.mlp.c_proj:0",
            device="cpu",
            std_eps=1e-7,  # <-- This one is not in statedict, will be updated.
            strength=0.9,
        )
        state_path = (
            BaseEstimator.get_output_path(cfg.interventions)
            / "transformer.h.1.mlp.c_proj:0.statedict"
        )
        hook.from_state_path(state_path)

        # Also testing the intervention forward
        zs = torch.randn(32, 3, 2)  # mu=0, std=1
        zs_post = hook(None, None, zs)
        assert torch.isnan(zs_post).sum() == 0
        assert torch.isinf(zs_post).sum() == 0

        if intervention_name == "none":
            assert torch.allclose(zs, zs_post)
            return

        # Check that values in statedict override those in constructor
        if intervention_name == "gaussian_act":
            assert hook.onlymean is False
        elif intervention_name == "mean_act":
            assert hook.onlymean is True
        if hasattr(hook, "hook_std_eps"):
            assert hook.std_eps == 1e-7

        assert hook.strength == 0.9


def test_gaussian_function():
    b, d = 1000, 5
    mu1_gt = 1.0
    std1_gt = 1.0
    mu2_gt = 5.0
    std2_gt = 0.5
    zs = torch.randn(b, d) * std1_gt + mu1_gt  # mu=1, std=1
    zd = torch.randn(b, d) * std2_gt + mu2_gt  # mu=5.0, std=0.5

    # Test in forward mode
    hook = GaussianAcTHook(
        module_name="test",
        dtype=torch.float32,
        intervention_position="all",
    )

    labels = torch.cat([torch.ones([b]), torch.zeros([b])]).to(torch.int64)
    hook.fit(
        responses=torch.cat([zs, zd], 0),
        labels=labels,
    )

    d = hook.state_dict()
    assert np.allclose(d["mu1"], mu1_gt, atol=0.1), f"mean1: {d['mu1']} !!"
    assert np.allclose(d["mu2"], mu2_gt, atol=0.1), f"mean2: {d['mu2']} !!"
    assert np.allclose(d["std1"], std1_gt, atol=0.1), f"std1: {d['std1']} !!"
    assert np.allclose(d["std2"], std2_gt, atol=0.1), f"std2: {d['std2']} !!"
    print(d["quantiles_dict_src"])
    print(zs.shape)
    z_transport = hook(None, None, zs)
    assert np.allclose(z_transport.mean(0), mu2_gt, atol=0.1), f"mean1: {d['mu1']} !!"
    assert np.allclose(z_transport.std(0), std2_gt, atol=0.1), f"std1: {d['std1']} !!"

    hook2 = GaussianAcTHook(
        module_name="test",
        dtype=torch.float32,
        intervention_position="all",
    )
    hook2.load_state_dict(d)
    d = hook2.state_dict()
    assert np.allclose(d["mu1"], 1.0, atol=0.1), f"mean: {d['mu1']} !!"
    assert np.allclose(d["mu2"], 5.0, atol=0.1), f"mean: {d['mu2']} !!"
    assert np.allclose(d["std1"], 1.0, atol=0.1), f"mean: {d['std1']} !!"
    assert np.allclose(d["std2"], 0.5, atol=0.1), f"mean: {d['std2']} !!"
