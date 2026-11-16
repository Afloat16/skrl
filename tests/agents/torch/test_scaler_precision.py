import pytest

import numpy as np
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("no_grad", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_scaler_uses_double_statistics_for_double_observations(inverse: bool, no_grad: bool, temporal: bool) -> None:
    # A resumed world-coordinate normalizer must retain centimetre offsets in its stored means.
    means = np.array([600000.01, -300000.02], dtype=np.float64)
    variances = np.array([0.04, 0.25], dtype=np.float64)
    scaler = RunningStandardScaler(size=2, epsilon=0.0, device="cpu")
    state = {
        "running_mean": torch.from_numpy(means.copy()),
        "running_variance": torch.from_numpy(variances.copy()),
        "current_count": torch.tensor(10000.0, dtype=torch.float64),
    }
    scaler.load_state_dict(state)
    normalized = np.array([[0.1, -0.2], [-0.4, 0.6]], dtype=np.float64)
    original = normalized if inverse else means + normalized * np.sqrt(variances)
    expected = original * np.sqrt(variances) + means if inverse else (original - means) / np.sqrt(variances)
    shape = (1, 2, 2) if temporal else (2, 2)
    values = torch.tensor(original.reshape(shape), dtype=torch.float64, requires_grad=True)

    actual = scaler(values, inverse=inverse, no_grad=no_grad)

    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.detach().numpy(), expected.reshape(shape), rtol=1e-13, atol=1e-12)
    if no_grad:
        assert not actual.requires_grad
    else:
        actual.sum().backward()
        derivative = np.sqrt(variances) if inverse else 1 / np.sqrt(variances)
        np.testing.assert_allclose(values.grad.numpy(), np.broadcast_to(derivative, shape), rtol=1e-13, atol=1e-12)
    for name, value in state.items():
        torch.testing.assert_close(scaler.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int64])
@pytest.mark.parametrize("inverse", [False, True])
def test_scaler_preserves_existing_single_precision_arithmetic(dtype: torch.dtype, inverse: bool) -> None:
    scaler = RunningStandardScaler(size=2, device="cpu")
    scaler.running_mean.copy_(torch.tensor([0.3125, -0.75], dtype=torch.float64))
    scaler.running_variance.copy_(torch.tensor([0.04, 1.69], dtype=torch.float64))
    values = torch.tensor([[1.0, 2.0], [-1.0, 0.0]], dtype=dtype)
    mean = scaler.running_mean.float()
    scale = scaler.running_variance.float().sqrt()
    if inverse:
        expected = values.clamp(-5, 5) * scale + mean
    else:
        expected = ((values - mean) / (scale + scaler.epsilon)).clamp(-5, 5)

    actual = scaler(values, inverse=inverse)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
