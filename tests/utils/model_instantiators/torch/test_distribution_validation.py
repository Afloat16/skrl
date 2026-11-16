from typing import Any

import pytest

import subprocess
import sys
import gymnasium

import torch

from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.utils.model_instantiators.torch import (
    categorical_model,
    gaussian_model,
    multicategorical_model,
    multivariate_gaussian_model,
)


@pytest.mark.parametrize("module", ["skrl.models.torch", "skrl.resources.noises.torch"])
@pytest.mark.parametrize("validate", [False, True])
def test_import_preserves_application_distribution_validation(module: str, validate: bool) -> None:
    program = """
import importlib
import sys
import torch

validate = sys.argv[2] == "True"
torch.distributions.Distribution.set_default_validate_args(validate)
importlib.import_module(sys.argv[1])
assert torch.distributions.Distribution._validate_args is validate
try:
    torch.distributions.Normal(torch.tensor(0.0), torch.tensor(-1.0))
except ValueError:
    assert validate
else:
    assert not validate
"""
    result = subprocess.run([sys.executable, "-c", program, module, str(validate)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize(
    "constructor,action_space,probabilities",
    [
        (gaussian_model, gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)), False),
        (multivariate_gaussian_model, gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)), False),
        (categorical_model, gymnasium.spaces.Discrete(3), False),
        (categorical_model, gymnasium.spaces.Discrete(3), True),
        (multicategorical_model, gymnasium.spaces.MultiDiscrete([2, 3]), False),
        (multicategorical_model, gymnasium.spaces.MultiDiscrete([2, 3]), True),
    ],
)
def test_policy_distributions_keep_validation_local(
    validate: bool, constructor: Any, action_space: gymnasium.Space, probabilities: bool
) -> None:
    previous = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(validate)
    try:
        options = (
            {"unnormalized_log_prob": not probabilities}
            if constructor in (categorical_model, multicategorical_model)
            else {}
        )
        policy = constructor(
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)),
            action_space=action_space,
            device="cpu",
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
            output="torch.softmax(ACTIONS, dim=-1)" if probabilities else "ACTIONS",
            **options,
        )
        policy.init_state_dict()
        _, outputs = policy.act({"observations": torch.tensor([[0.1, -0.2], [0.3, 0.4]])})
        distributions = getattr(policy, "_mc_distributions", [policy.distribution()])
        assert all(distribution._validate_args is False for distribution in distributions)
        assert torch.distributions.Distribution._validate_args is validate
        loss = outputs["log_prob"].sum() + policy.get_entropy().sum()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous)


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("correlated", [False, True])
def test_noise_distributions_keep_validation_local(validate: bool, correlated: bool) -> None:
    previous = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(validate)
    try:
        noise = (
            OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=0.5, device="cpu")
            if correlated
            else GaussianNoise(mean=0.1, std=0.7, device="cpu")
        )
        assert noise.distribution._validate_args is False
        assert torch.distributions.Distribution._validate_args is validate
        sample = noise.sample((3, 2))
        assert sample.shape == (3, 2)
        assert torch.isfinite(sample).all()
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous)
