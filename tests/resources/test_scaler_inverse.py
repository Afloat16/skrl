import pytest

import importlib

import numpy as np


def _make_scaler(backend, epsilon, tmp_path, monkeypatch):
    framework = pytest.importorskip(backend)
    if backend == "warp":
        monkeypatch.setattr(framework.config, "kernel_cache_dir", str(tmp_path / "warp"))
    scaler_type = importlib.import_module(f"skrl.resources.preprocessors.{backend}").RunningStandardScaler
    scaler = scaler_type(size=3, epsilon=epsilon, device="cpu")
    mean = np.array([0.0, 0.1, -0.2], dtype=np.float32)
    variance = np.array([0.0, 1e-12, 4.0] if epsilon else [0.1, 1.0, 4.0], dtype=np.float32)
    if backend == "torch":
        scaler.running_mean.copy_(framework.as_tensor(mean))
        scaler.running_variance.copy_(framework.as_tensor(variance))
        convert = framework.as_tensor
        to_numpy = lambda value: value.detach().cpu().numpy()
    elif backend == "jax":
        arrays = importlib.import_module("jax.numpy")
        scaler.running_mean = arrays.asarray(mean)
        scaler.running_variance = arrays.asarray(variance)
        convert, to_numpy = arrays.asarray, np.asarray
    else:
        scaler.load_state_dict({"running_mean": mean, "running_variance": variance, "current_count": 1.0})
        convert = lambda value: framework.array(value, dtype=framework.float32, device="cpu")
        to_numpy = lambda value: value.numpy()
    return scaler, convert, to_numpy, mean, variance


@pytest.mark.parametrize("backend", ["torch", "jax", "warp"])
@pytest.mark.parametrize("epsilon", [0.0, 1e-8, 0.1])
@pytest.mark.parametrize("shape", [(4, 3), (2, 2, 3)])
def test_standardization_inverse_round_trip_includes_epsilon(backend, epsilon, shape, tmp_path, monkeypatch):
    scaler, convert, to_numpy, mean, variance = _make_scaler(backend, epsilon, tmp_path, monkeypatch)
    standardized = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0], [0.5, -0.5, 1.0], [2.0, 2.0, -1.0]], dtype=np.float32)
    values = (mean + standardized * (np.sqrt(variance) + epsilon)).reshape(shape)
    restored = scaler(scaler(convert(values)), inverse=True)
    np.testing.assert_allclose(to_numpy(restored), values, rtol=5e-6, atol=1e-10)
    np.testing.assert_array_equal(to_numpy(scaler.running_mean), mean)
    np.testing.assert_allclose(to_numpy(scaler.running_variance), variance)


@pytest.mark.parametrize("backend", ["torch", "jax", "warp"])
def test_inverse_uses_the_forward_scale_at_the_clipping_boundary(backend, tmp_path, monkeypatch):
    scaler, convert, to_numpy, mean, variance = _make_scaler(backend, 0.1, tmp_path, monkeypatch)
    values = np.array([[10.0, -10.0, 3.0], [-5.0, 5.0, -6.0]], dtype=np.float32)
    expected = mean + np.clip(values, -5.0, 5.0) * (np.sqrt(variance) + 0.1)
    np.testing.assert_allclose(to_numpy(scaler(convert(values), inverse=True)), expected, rtol=5e-6, atol=1e-7)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("epsilon", [1e-8, 0.1])
def test_standardization_inverse_round_trip_has_identity_gradient(backend, epsilon, tmp_path, monkeypatch):
    scaler, convert, to_numpy, mean, variance = _make_scaler(backend, epsilon, tmp_path, monkeypatch)
    values = convert((mean + np.array([[1.0, -1.0, 2.0]], dtype=np.float32) * (np.sqrt(variance) + epsilon)))
    if backend == "torch":
        values.requires_grad_()
        scaler(scaler(values, no_grad=False), inverse=True, no_grad=False).sum().backward()
        gradient = values.grad
    else:
        import jax
        import jax.numpy as jnp

        gradient = jax.jit(jax.grad(lambda value: jnp.sum(scaler(scaler(value), inverse=True))))(values)
    np.testing.assert_allclose(to_numpy(gradient), np.ones((1, 3)), rtol=1e-5)


@pytest.mark.parametrize("epsilon", [1e-8, 0.1])
@pytest.mark.parametrize("shape", [(4, 3), (2, 2, 3)])
def test_warp_inplace_inverse_round_trip(epsilon, shape, tmp_path, monkeypatch):
    scaler, convert, to_numpy, mean, variance = _make_scaler("warp", epsilon, tmp_path, monkeypatch)
    values = np.broadcast_to(mean + (np.sqrt(variance) + epsilon), shape).copy()
    array = convert(values)
    normalized = scaler(array, inplace=True)
    restored = scaler(normalized, inverse=True, inplace=True)
    assert restored is array
    np.testing.assert_allclose(to_numpy(restored), values, rtol=5e-6, atol=1e-10)
