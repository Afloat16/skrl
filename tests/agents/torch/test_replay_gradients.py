from typing import Any

import pytest

from pathlib import Path
import gymnasium

import numpy as np
import torch

from skrl.agents.torch.ddqn import DDQN
from skrl.agents.torch.dqn import DQN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model


class QNetwork(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(2,)),
            action_space=gymnasium.spaces.Discrete(2),
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.linear = torch.nn.Linear(2, 2)
        torch.nn.init.constant_(self.linear.weight, 0.2)
        torch.nn.init.constant_(self.linear.bias, 0.1)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.linear(inputs["observations"]), {}


@pytest.mark.parametrize("num_envs,shape", [(2, (2, 2)), (3, (2, 2)), (1, (3, 2)), (1, (2,))])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_replay_stores_values_without_collection_graph(
    num_envs: int, shape: tuple[int, ...], requires_grad: bool
) -> None:
    memory = RandomMemory(memory_size=4, num_envs=num_envs, device="cpu")
    memory.create_tensor("observations", size=2, dtype=torch.float32)
    source = torch.arange(1, 1 + int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    source.requires_grad_(requires_grad)
    observations = source.square()

    memory.add_samples(observations=observations)

    stored = memory.get_tensor_by_name("observations")
    torch.testing.assert_close(stored.view(-1, 2)[: len(memory)], observations.detach().reshape(-1, 2))
    assert not stored.requires_grad
    assert stored.grad_fn is None
    sampled = memory.sample(names=["observations"], batch_size=1)[0][0]
    assert not sampled.requires_grad
    assert source.grad is None


@pytest.mark.parametrize("agent_class", [DQN, DDQN])
def test_repeated_q_updates_do_not_backpropagate_through_collection(agent_class: Any) -> None:
    policy = QNetwork()
    memory = RandomMemory(memory_size=4, device="cpu")
    agent = agent_class(
        models={"q_network": policy, "target_q_network": QNetwork()},
        memory=memory,
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device="cpu",
        cfg={"batch_size": 1, "experiment": {"write_interval": 0, "checkpoint_interval": 0}},
    )
    agent.init()
    agent.enable_training_mode(True)
    source = torch.tensor([[1.0, 2.0]], requires_grad=True)
    observations = source.square()
    agent.record_transition(
        observations=observations,
        states=None,
        actions=torch.zeros((1, 1), dtype=torch.int64),
        rewards=torch.tensor([[1.5]]),
        next_observations=observations + 0.5,
        next_states=None,
        terminated=torch.zeros((1, 1), dtype=torch.bool),
        truncated=torch.zeros((1, 1), dtype=torch.bool),
        infos={},
        timestep=0,
        timesteps=3,
    )
    original = policy.linear.weight.detach().clone()

    for timestep in range(3):
        agent.update(timestep=timestep, timesteps=3)

    assert not torch.equal(policy.linear.weight, original)
    assert torch.isfinite(policy.linear.weight).all()
    assert source.grad is None


def test_replay_automatically_exports_grad_enabled_observations(tmp_path: Path) -> None:
    memory = RandomMemory(memory_size=1, device="cpu", export=True, export_format="npz", export_directory=str(tmp_path))
    memory.create_tensor("observations", size=2, dtype=torch.float32)
    source = torch.tensor([[1.0, 2.0]], requires_grad=True)

    memory.add_samples(observations=source.square())

    paths = list((tmp_path / "memories").glob("*.npz"))
    assert len(paths) == 1
    with np.load(paths[0]) as data:
        np.testing.assert_array_equal(data["observations"], np.array([[[1.0, 4.0]]]))
