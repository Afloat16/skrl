from typing import Any

import pytest

import gymnasium

import torch

from skrl.agents.torch.ddqn import DDQN
from skrl.agents.torch.dqn import DQN
from skrl.models.torch import DeterministicMixin, Model


class QNetwork(DeterministicMixin, Model):
    def __init__(self, asymmetric: bool) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1,)),
            state_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)) if asymmetric else None,
            action_space=gymnasium.spaces.Discrete(2),
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.values = torch.nn.Parameter(torch.tensor([3.0, 1.0]))
        self.calls = 0

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls += 1
        return self.values.expand(inputs["observations"].shape[0], -1), {}


@pytest.mark.parametrize("agent_class", [DQN, DDQN])
@pytest.mark.parametrize("epsilon", [None, 0.0, 1.0])
@pytest.mark.parametrize("timestep", [0, 4, 5])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_random_warmup_independent_of_scheduler(
    agent_class: Any, epsilon: float | None, timestep: int, asymmetric: bool
) -> None:
    policy = QNetwork(asymmetric)
    observations = torch.zeros((16, 1))
    states = torch.zeros((16, 2)) if asymmetric else None
    agent = agent_class(
        models={"q_network": policy},
        observation_space=policy.observation_space,
        state_space=policy.state_space,
        action_space=policy.action_space,
        device="cpu",
        cfg={
            "random_timesteps": 5,
            "exploration_scheduler": None if epsilon is None else lambda timestep, timesteps: epsilon,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    policy.action_space.seed(7)
    random_actions, _ = policy.random_act({"observations": observations, "states": states})
    assert random_actions.unique().numel() == 2
    policy.action_space.seed(7)

    actions, _ = agent.act(observations, states, timestep=timestep, timesteps=10)

    random_expected = timestep < 5 or epsilon == 1.0
    expected = random_actions if random_expected else torch.zeros_like(random_actions)
    torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    assert policy.calls == (0 if random_expected else 1)
