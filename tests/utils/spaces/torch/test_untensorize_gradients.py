import pytest

import gymnasium

import numpy as np
import torch

from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.utils.spaces.torch import untensorize_space


@pytest.mark.parametrize(
    "space",
    [
        gymnasium.spaces.Box(-3, 3, shape=(2, 2)),
        gymnasium.spaces.Box(-3, 3, shape=()),
        gymnasium.spaces.Discrete(3),
        gymnasium.spaces.MultiDiscrete([[2, 3], [4, 2]]),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_untensorize_detaches_conversion_without_changing_input_history(
    space: gymnasium.Space, batch_size: int, dtype: torch.dtype, requires_grad: bool
) -> None:
    shape = (1,) if isinstance(space, gymnasium.spaces.Discrete) else space.shape
    count = batch_size * int(np.prod(shape))
    leaf = torch.linspace(0.1, 0.9, count, dtype=dtype, requires_grad=requires_grad)
    tensor = (2 * leaf).reshape(batch_size, *shape)
    before = tensor.detach().clone()
    history = tensor.grad_fn

    result = untensorize_space(space, tensor)

    expected = tensor.detach().double().numpy().astype(space.dtype)
    if batch_size == 1:
        expected = expected.item() if isinstance(space, gymnasium.spaces.Discrete) else expected.reshape(space.shape)
    np.testing.assert_array_equal(result, expected)
    assert (
        isinstance(result, int)
        if isinstance(space, gymnasium.spaces.Discrete) and batch_size == 1
        else isinstance(result, np.ndarray)
    )
    if isinstance(result, np.ndarray):
        assert result.dtype == space.dtype
        result[...] = 0
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
    assert tensor.requires_grad is requires_grad
    assert tensor.grad_fn is history
    if requires_grad:
        tensor.square().sum().backward()
        torch.testing.assert_close(leaf.grad, 8 * leaf.detach())


@pytest.mark.parametrize("requires_grad", [False, True])
def test_untensorize_nested_spaces_preserves_the_original_graph(requires_grad: bool) -> None:
    space = gymnasium.spaces.Dict(
        {
            "continuous": gymnasium.spaces.Box(-3, 3, shape=(2,)),
            "command": gymnasium.spaces.Tuple((gymnasium.spaces.Discrete(3), gymnasium.spaces.MultiDiscrete([2, 3]))),
        }
    )
    leaf = torch.linspace(0.1, 0.9, 10, requires_grad=requires_grad)
    tensor = (2 * leaf).reshape(2, 5)
    values = {"continuous": tensor[:, :2], "command": (tensor[:, 2:3], tensor[:, 3:])}

    result = untensorize_space(space, values, squeeze_batch_dimension=False)

    expected = tensor.detach().numpy()
    np.testing.assert_array_equal(result["continuous"], expected[:, :2])
    np.testing.assert_array_equal(result["command"][0], expected[:, 2:3].astype(np.int64))
    np.testing.assert_array_equal(result["command"][1], expected[:, 3:].astype(np.int64))
    if requires_grad:
        tensor.square().sum().backward()
        torch.testing.assert_close(leaf.grad, 8 * leaf.detach())


@pytest.mark.parametrize("vectorized", [False, True])
def test_gymnasium_steps_accept_policy_outputs_with_autograd_history(vectorized: bool) -> None:
    torch.manual_seed(79)
    environment = (
        gymnasium.vector.SyncVectorEnv([lambda: gymnasium.make("Pendulum-v1") for _ in range(3)])
        if vectorized
        else gymnasium.make("Pendulum-v1")
    )
    wrapper = GymnasiumWrapper(environment)
    try:
        observations, _ = wrapper.reset()
        policy = torch.nn.Linear(3, 1, device=wrapper.device)
        actions = 2 * torch.tanh(policy(observations))
        expected_actions = actions.detach().clone()
        assert actions.requires_grad and actions.grad_fn is not None

        next_observations, rewards, terminated, truncated, _ = wrapper.step(actions)

        assert next_observations.shape == (wrapper.num_envs, 3)
        assert rewards.shape == terminated.shape == truncated.shape == (wrapper.num_envs, 1)
        assert torch.isfinite(next_observations).all() and torch.isfinite(rewards).all()
        assert torch.is_grad_enabled()
        torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
        actions.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
        )
        assert any(torch.count_nonzero(parameter.grad) for parameter in policy.parameters())
    finally:
        wrapper.close()
