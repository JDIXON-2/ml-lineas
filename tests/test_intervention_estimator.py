# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import logging

import pytest
from hydra import compose, initialize

from lineas.estimators.get_estimator import ESTIMATORS_REGISTRY

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# List of strategies to test
strategies = ["incr", "atonce", "end2end"]


@pytest.mark.parametrize("strategy", strategies)
def test_evaluate_perplexity(strategy):
    with initialize(version_base=None, config_path="../lineas/configs"):
        # config is relative to a module
        cfg = compose(
            config_name="text_generation",
            overrides=[f"intervention_params.estimation_strategy={strategy}"],
        )
    assert cfg.intervention_params.state_path is None
    estimator = ESTIMATORS_REGISTRY[cfg.intervention_params.estimation_strategy]
    estimator.patch_output_path_(cfg.interventions)
    assert cfg.intervention_params.state_path is not None
