from typing import Any

import pytest

import gymnasium

import numpy as np
import torch

from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


@pytest.mark.parametrize("batch_size", [0, 1, 4])
@pytest.mark.parametrize("container", ["scalar", "tuple", "dict"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_scalar_box_batch_round_trip(batch_size: int, container: str, backend: str) -> None:
    scalar = gymnasium.spaces.Box(-100.0, 100.0, shape=())
    vector = gymnasium.spaces.Box(-100.0, 100.0, shape=(2,))
    scalar_values = np.arange(batch_size, dtype=np.float32)
    vector_values = np.arange(2 * batch_size, dtype=np.float32).reshape(batch_size, 2)
    expected = scalar_values.reshape(batch_size, 1)
    if backend == "torch":
        scalar_values = torch.from_numpy(scalar_values)
        vector_values = torch.from_numpy(vector_values)
    if container == "tuple":
        space = gymnasium.spaces.Tuple((vector, scalar))
        values = (vector_values, scalar_values)
        expected = np.concatenate([np.asarray(vector_values), expected], axis=-1)
    elif container == "dict":
        space = gymnasium.spaces.Dict({"z": scalar, "a": vector})
        values = {"z": scalar_values, "a": vector_values}
        expected = np.concatenate([np.asarray(vector_values), expected], axis=-1)
    else:
        space, values = scalar, scalar_values
    tensorized = tensorize_space(space, values, device="cpu")

    flattened = flatten_tensorized_space(tensorized)

    torch.testing.assert_close(flattened, torch.from_numpy(expected), rtol=0, atol=0)
    restored = unflatten_tensorized_space(space, flattened)
    actual_scalar = restored[1] if container == "tuple" else restored["z"] if container == "dict" else restored
    assert actual_scalar.shape == (batch_size,)
    torch.testing.assert_close(actual_scalar, torch.arange(batch_size, dtype=torch.float32))


class ScalarObservationEnv(gymnasium.Env):
    def __init__(self, initial: float) -> None:
        self.initial = initial
        self.value = initial
        self.observation_space = gymnasium.spaces.Box(-100.0, 100.0, shape=())
        self.action_space = gymnasium.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.value = self.initial
        return np.array(self.value, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.value += action
        return np.array(self.value, dtype=np.float32), 0.0, False, False, {}


@pytest.mark.parametrize("num_envs", [1, 3])
def test_vector_environment_keeps_one_scalar_feature_per_environment(num_envs: int) -> None:
    vector_env = gymnasium.vector.SyncVectorEnv(
        [lambda initial=initial: ScalarObservationEnv(float(initial)) for initial in range(num_envs)]
    )
    env = GymnasiumWrapper(vector_env)
    try:
        observations, _ = env.reset()
        torch.testing.assert_close(observations.cpu(), torch.arange(num_envs, dtype=torch.float32).reshape(-1, 1))
        observations, _, _, _, _ = env.step(torch.ones((num_envs, 1), device=env.device, dtype=torch.int64))
        torch.testing.assert_close(
            observations.cpu(), torch.arange(1, num_envs + 1, dtype=torch.float32).reshape(-1, 1)
        )
    finally:
        env.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flatten_scalar_batch_preserves_dtype_and_gradients(dtype: torch.dtype) -> None:
    source = torch.arange(8, dtype=dtype).requires_grad_()
    values = source[::2]
    flattened = flatten_tensorized_space(values)
    assert flattened.shape == (4, 1)
    assert flattened.dtype == dtype
    flattened.square().sum().backward()
    expected = torch.zeros_like(source)
    expected[::2] = 2 * values.detach()
    torch.testing.assert_close(source.grad, expected)


@pytest.mark.parametrize("value", [torch.tensor(2.0), torch.arange(12).reshape(2, 2, 3).transpose(1, 2)])
def test_flatten_preserves_scalar_and_multidimensional_controls(value: Any) -> None:
    expected = value.reshape(1, 1) if value.ndim == 0 else value.reshape(2, 6)
    torch.testing.assert_close(flatten_tensorized_space(value), expected)
