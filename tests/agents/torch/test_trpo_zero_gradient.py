from typing import Any

import pytest

import gymnasium

import torch

from skrl.agents.torch.trpo import trpo, trpo_rnn
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class LinearPolicy(GaussianMixin, Model):
    def __init__(self) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1,)),
            action_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(1,)),
            device="cpu",
        )
        GaussianMixin.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([0.2]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1]))
        self.log_std = torch.nn.Parameter(torch.tensor([-0.3]))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return inputs["observations"] * self.weight + self.bias, {"log_std": self.log_std}


class RecurrentPolicy(GaussianMixin, Model):
    def __init__(self) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1,)),
            action_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(1,)),
            device="cpu",
        )
        GaussianMixin.__init__(self)
        self.rnn = torch.nn.GRU(1, 1, batch_first=True)
        self.output = torch.nn.Linear(1, 1)
        self.log_std = torch.nn.Parameter(torch.tensor([-0.3]))

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 1, 1)]}}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        length = 2 if self.training else 1
        observations = inputs["observations"].reshape(-1, length, 1)
        state = inputs["rnn"][0][:, ::length]
        output, state = self.rnn(observations, state)
        return self.output(output.reshape(-1, 1)), {"log_std": self.log_std, "rnn": [state]}


class ConstantValue(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.value = torch.nn.Parameter(torch.zeros(1))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.value.expand(inputs["observations"].shape[0], 1), {}


@pytest.mark.parametrize("module", [trpo, trpo_rnn])
@pytest.mark.parametrize("tolerance", [0.0, 1e-10])
def test_conjugate_gradient_zero_rhs(module: Any, tolerance: float) -> None:
    policy = LinearPolicy().double()
    observations = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
    kwargs = {"rnn": {}} if module is trpo_rnn else {}

    result = module.conjugate_gradient(
        policy=policy,
        observations=observations,
        states=None,
        b=torch.zeros(3, dtype=torch.float64),
        residual_tolerance=tolerance,
        **kwargs,
    )

    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


@pytest.mark.parametrize("module", [trpo, trpo_rnn])
def test_conjugate_gradient_nonzero_rhs(module: Any) -> None:
    policy = LinearPolicy().double()
    observations = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float64)
    rhs = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    inverse_variance = torch.exp(-2 * policy.log_std.detach())[0]
    fisher = torch.diag(
        torch.stack([observations.square().mean() * inverse_variance, inverse_variance, rhs.new_tensor(2.0)])
    )
    damping = 0.1
    expected = torch.linalg.solve(fisher + damping * torch.eye(3, dtype=rhs.dtype), rhs)
    kwargs = {"rnn": {}} if module is trpo_rnn else {}

    result = module.conjugate_gradient(
        policy=policy,
        observations=observations,
        states=None,
        b=rhs,
        damping=damping,
        residual_tolerance=1e-20,
        **kwargs,
    )

    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("recurrent", [False, True])
def test_zero_advantages_preserve_policy_and_train_value(recurrent: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    policy = RecurrentPolicy() if recurrent else LinearPolicy()
    value = ConstantValue()
    module = trpo_rnn if recurrent else trpo
    agent_class = module.TRPO_RNN if recurrent else module.TRPO
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=2, num_envs=1, device="cpu"),
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device="cpu",
        cfg={
            "rollouts": 2,
            "learning_epochs": 1,
            "mini_batches": 1,
            "discount_factor": 0.0,
            "learning_rate": 0.05,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode()
    initial_parameters = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    original_compute = policy.compute

    def compute(inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
        return original_compute(inputs, role)

    monkeypatch.setattr(policy, "compute", compute)
    for timestep in range(2):
        observations = torch.tensor([[0.2 * timestep]])
        with torch.no_grad():
            actions, _ = agent.act(observations, None, timestep=timestep, timesteps=2)
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.ones((1, 1)),
                next_observations=observations,
                next_states=None,
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=2,
            )
        agent.post_interaction(timestep=timestep, timesteps=2)

    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0, atol=0)
    assert value.value.item() > 0.0
