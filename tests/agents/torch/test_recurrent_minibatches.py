from typing import Any

import pytest

import gymnasium

import torch

from skrl.agents.torch.ppo import PPO_RNN
from skrl.agents.torch.rpo import RPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def make_memory(memory_size: int = 6, num_envs: int = 2, replacement: bool = False) -> RandomMemory:
    memory = RandomMemory(memory_size=memory_size, num_envs=num_envs, device="cpu", replacement=replacement)
    memory.create_tensor("values", size=1, dtype=torch.float32)
    memory.create_tensor("paired", size=1, dtype=torch.float32)
    memory.create_tensor("hidden", size=(2, 3), dtype=torch.float32, keep_dimensions=True)
    for timestep in range(memory_size):
        values = (10 * timestep + torch.arange(num_envs, dtype=torch.float32)).reshape(-1, 1)
        memory.add_samples(values=values, paired=values + 100, hidden=values.reshape(-1, 1, 1).expand(-1, 2, 3))
    return memory


def check_batches(batches: list[list[torch.Tensor]], sequence_length: int, num_sequences: int) -> torch.Tensor:
    quotient, remainder = divmod(num_sequences, len(batches))
    for index, (values, paired, hidden, missing) in enumerate(batches):
        expected_rows = (quotient + (index < remainder)) * sequence_length
        assert values.shape == (expected_rows, 1)
        assert hidden.shape == (expected_rows, 2, 3)
        assert missing is None
        torch.testing.assert_close(paired, values + 100)
        torch.testing.assert_close(hidden, values.reshape(-1, 1, 1).expand(-1, 2, 3))
        if sequence_length > 1:
            sequences = values.reshape(-1, sequence_length)
            torch.testing.assert_close(sequences[:, 1:] - sequences[:, :-1], torch.full_like(sequences[:, 1:], 10))
    return torch.cat([batch[0] for batch in batches])


@pytest.mark.parametrize("sequence_length", [2, 3])
@pytest.mark.parametrize("mini_batches", [1, 3, 4])
def test_sample_all_splits_complete_recurrent_sequences(sequence_length: int, mini_batches: int) -> None:
    memory = make_memory()
    batches = memory.sample_all(
        names=["values", "paired", "hidden", "missing"], mini_batches=mini_batches, sequence_length=sequence_length
    )
    sampled = check_batches(batches, sequence_length, 12 // sequence_length)
    expected = memory.get_tensor_by_name("values").transpose(0, 1).reshape(-1, 1)
    torch.testing.assert_close(sampled, expected)


@pytest.mark.parametrize("sequence_length", [1, 2, 3])
@pytest.mark.parametrize("mini_batches", [1, 4])
@pytest.mark.parametrize("replacement", [False, True])
def test_random_samples_preserve_sequences_and_field_alignment(
    sequence_length: int, mini_batches: int, replacement: bool
) -> None:
    torch.manual_seed(17)
    memory = make_memory(memory_size=8, replacement=replacement)
    batches = memory.sample(
        names=["values", "paired", "hidden", "missing"],
        batch_size=5,
        mini_batches=mini_batches,
        sequence_length=sequence_length,
    )
    sampled = check_batches(batches, sequence_length, 5)
    indexes = memory.get_sampling_indexes()
    assert indexes.shape == (5 * sequence_length,)
    expected = memory.get_tensor_by_name("values").reshape(-1, 1)[indexes]
    torch.testing.assert_close(sampled, expected)
    if not replacement:
        assert indexes.reshape(-1, sequence_length)[:, 0].unique().numel() == 5


@pytest.mark.parametrize("num_envs", [1, 2])
def test_sample_all_rejects_incomplete_sequences_within_an_environment(num_envs: int) -> None:
    memory = make_memory(memory_size=5, num_envs=num_envs)
    with pytest.raises(ValueError, match="memory_size.*sequence_length"):
        memory.sample_all(names=["values"], mini_batches=2, sequence_length=2)


class Policy(GaussianMixin, Model):
    def __init__(self) -> None:
        Model.__init__(self, device="cpu")
        GaussianMixin.__init__(self)
        self.rnn = torch.nn.GRU(2, 3, batch_first=True)
        self.output = torch.nn.Linear(3, 1)
        self.log_std = torch.nn.Parameter(torch.tensor([-0.5]))
        self.training_batches = []

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 2, 3)]}}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        observations = inputs["observations"]
        length = 2 if self.training else 1
        if self.training:
            self.training_batches.append(observations.detach().clone())
        features, hidden = self.rnn(observations.reshape(-1, length, 2), inputs["rnn"][0][:, ::length])
        return self.output(features.reshape(-1, 3)), {"log_std": self.log_std, "rnn": [hidden]}


class Value(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.output = torch.nn.Linear(2, 1)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.output(inputs["observations"]), {}


@pytest.mark.parametrize("agent_class", [PPO_RNN, RPO_RNN])
@pytest.mark.parametrize("mini_batches", [3, 4])
def test_recurrent_agents_train_with_uneven_sequence_batches(agent_class: Any, mini_batches: int) -> None:
    torch.manual_seed(29)
    policy, value = Policy(), Value()
    memory = RandomMemory(memory_size=6, num_envs=2, device="cpu")
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=gymnasium.spaces.Box(-1, 1, shape=(2,)),
        action_space=gymnasium.spaces.Box(-1, 1, shape=(1,)),
        device="cpu",
        cfg={
            "rollouts": 6,
            "learning_epochs": 2,
            "mini_batches": mini_batches,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode()
    for timestep in range(6):
        observations = torch.tensor([[0.1 * timestep, 0.2], [0.1 * timestep, -0.2]])
        with torch.no_grad():
            actions, _ = agent.act(observations, None, timestep=timestep, timesteps=6)
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.tensor([[0.2 + timestep], [-0.1 - timestep]]),
                next_observations=observations + 0.05,
                next_states=None,
                terminated=torch.zeros((2, 1), dtype=torch.bool),
                truncated=torch.zeros((2, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=6,
            )
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    agent.enable_models_training_mode(True)
    agent.update(timestep=5, timesteps=6)
    assert len(policy.training_batches) == 2 * mini_batches
    for batch in policy.training_batches:
        sequences = batch.reshape(-1, 2, 2)
        torch.testing.assert_close(sequences[:, 1, 0] - sequences[:, 0, 0], torch.full_like(sequences[:, 0, 0], 0.1))
        torch.testing.assert_close(sequences[:, 0, 1], sequences[:, 1, 1])
    expected = memory.get_tensor_by_name("observations").transpose(0, 1).reshape(-1, 2)
    torch.testing.assert_close(torch.cat(policy.training_batches), expected.repeat(2, 1))
    assert any(not torch.equal(previous, parameter) for previous, parameter in zip(before, policy.parameters()))
    assert all(torch.isfinite(parameter).all() for model in (policy, value) for parameter in model.parameters())
