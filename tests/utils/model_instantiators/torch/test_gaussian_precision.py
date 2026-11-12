import pytest

import math
from gymnasium import spaces

import torch

from skrl.utils.model_instantiators.torch import gaussian_model, multivariate_gaussian_model


@pytest.mark.parametrize("factory", [gaussian_model, multivariate_gaussian_model])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("log_std", [-20.0, -12.0, 2.0])
@pytest.mark.parametrize("clip_log_std", [False, True])
def test_gaussian_probability_math_preserves_precision(factory, dtype, log_std, clip_log_std):
    model = factory(
        observation_space=spaces.Box(-1, 1, (2,)),
        action_space=spaces.Box(-1, 1, (3,)),
        device="cpu",
        clip_log_std=clip_log_std,
        initial_log_std=log_std,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "elu"}],
        output="ACTIONS",
    )
    model.init_state_dict()
    model.to(dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "log_std" not in name:
                parameter.zero_()
    actions, outputs = model.act(
        {
            "observations": torch.zeros(2, 2, dtype=dtype),
            "taken_actions": torch.zeros(2, 3, dtype=dtype),
        }
    )
    result_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    expected_log_prob = torch.full((2, 1), -3 * (log_std + math.log(2 * math.pi) / 2), dtype=result_dtype)
    expected_entropy = torch.full((2, 1), 3 * (log_std + math.log(2 * math.pi * math.e) / 2), dtype=result_dtype)
    assert torch.isfinite(outputs["log_prob"]).all()
    assert torch.isfinite(model.get_entropy()).all()
    torch.testing.assert_close(outputs["log_prob"], expected_log_prob)
    torch.testing.assert_close(model.get_entropy(), expected_entropy)
    assert actions.dtype == result_dtype
    assert outputs["log_std"].dtype == result_dtype
    assert torch.isfinite(actions).all()
    (outputs["log_prob"].sum() + 0.5 * model.get_entropy().sum()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("factory", [gaussian_model, multivariate_gaussian_model])
def test_gaussian_model_runs_under_cpu_autocast(factory):
    model = factory(
        observation_space=spaces.Box(-1, 1, (2,)),
        action_space=spaces.Box(-1, 1, (3,)),
        device="cpu",
        initial_log_std=-12.0,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "elu"}],
        output="ACTIONS",
    )
    model.init_state_dict()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actions, outputs = model.act({"observations": torch.zeros(2, 2)})
        loss = outputs["log_prob"].mean() - 0.01 * model.get_entropy().mean()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
