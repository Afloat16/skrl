from typing import Any

import pytest

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import gymnasium
from packaging import version

import torch
import torch.multiprocessing as mp

from skrl import config
from skrl.agents.torch.sac import SAC, SAC_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class Policy(GaussianMixin, Model):
    def __init__(self, recurrent: bool) -> None:
        Model.__init__(self, device="cpu")
        GaussianMixin.__init__(self)
        self.recurrent = recurrent
        self.rnn = torch.nn.GRU(2, 2, batch_first=True) if recurrent else None
        self.output = torch.nn.Linear(2, 1)
        self.log_std = torch.nn.Parameter(torch.tensor([-0.5]))

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 1, 2)]}} if self.recurrent else {}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        values = inputs["observations"]
        outputs = {"log_std": self.log_std + values[:, :1]}
        if self.recurrent:
            length = 2 if self.training else 1
            values, state = self.rnn(values.reshape(-1, length, 2), inputs["rnn"][0][:, ::length])
            values = values.reshape(-1, 2)
            outputs["rnn"] = [state]
        return self.output(values), outputs


class Critic(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.output = torch.nn.Linear(3, 1)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.output(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)), {}


def make_agent(recurrent: bool, learn_entropy: bool, mixed_precision: bool, rank: int) -> SAC | SAC_RNN:
    torch.manual_seed(11)
    agent = (SAC_RNN if recurrent else SAC)(
        models={
            "policy": Policy(recurrent),
            "critic_1": Critic(),
            "critic_2": Critic(),
            "target_critic_1": Critic(),
            "target_critic_2": Critic(),
        },
        memory=RandomMemory(memory_size=4, device="cpu"),
        observation_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(2,)),
        action_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(1,)),
        device="cpu",
        cfg={
            "batch_size": 2,
            "learn_entropy": learn_entropy,
            "initial_entropy_value": 0.2,
            "target_entropy": 1.0,
            "learning_rate": (0.001, 0.002, 0.01),
            "mixed_precision": mixed_precision,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode()
    for timestep in range(4):
        observations = torch.tensor([[2.0 * rank, -0.2]])
        with torch.no_grad():
            actions, _ = agent.act(observations, None, timestep=timestep, timesteps=4)
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.tensor([[0.2]]),
                next_observations=observations,
                next_states=None,
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=4,
            )
    agent.enable_models_training_mode(True)
    return agent


def gather(value: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    values = [torch.empty_like(value) for _ in range(world_size)]
    torch.distributed.all_gather(values, value)
    return values


def run_entropy_updates(rank: int, world_size: int, rendezvous: str, recurrent: bool, learn_entropy: bool) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    config.torch._world_size = world_size
    config.torch._rank = rank
    config.torch._is_distributed = True
    try:
        precisions = [False, True] if version.parse(torch.__version__) >= version.parse("2.4") else [False]
        for mixed_precision in precisions:
            agent = make_agent(recurrent, learn_entropy, mixed_precision, rank)
            if learn_entropy:
                expected = agent.log_entropy_coefficient.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([expected], lr=0.01)
                local_gradients = []
                agent.log_entropy_coefficient.register_hook(lambda gradient: local_gradients.append(gradient.clone()))

            # Evaluate a fixed sample from each distribution so local entropy gradients have opposite signs.
            with patch.object(torch.distributions.Normal, "rsample", lambda self: self.loc):
                for timestep in range(3):
                    scale = agent.scaler.get_scale()
                    agent.update(timestep=timestep, timesteps=3)
                    if learn_entropy:
                        gradients = gather(local_gradients[-1], world_size)
                        if timestep == 0:
                            assert gradients[0].item() < 0 < gradients[1].item()
                        optimizer.zero_grad()
                        expected.grad = torch.stack(gradients).mean(dim=0) / scale
                        optimizer.step()
                        torch.testing.assert_close(agent.log_entropy_coefficient, expected)
                        coefficients = gather(agent.log_entropy_coefficient.detach(), world_size)
                        torch.testing.assert_close(coefficients[0], coefficients[1], rtol=0, atol=0)
                        torch.testing.assert_close(agent._entropy_coefficient, expected.detach().exp())
                    else:
                        assert agent._entropy_coefficient == 0.2
                        assert not hasattr(agent, "entropy_optimizer")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo is required"
)
@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("learn_entropy", [False, True])
def test_sac_entropy_matches_average_gradient_on_two_workers(
    recurrent: bool, learn_entropy: bool, tmp_path: Path
) -> None:
    mp.spawn(
        run_entropy_updates,
        args=(2, (tmp_path / "rendezvous").as_uri(), recurrent, learn_entropy),
        nprocs=2,
        join=True,
    )
