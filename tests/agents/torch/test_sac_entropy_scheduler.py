from typing import Any

import pytest

from pathlib import Path
import gymnasium

import torch

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
        outputs = {"log_std": self.log_std}
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
        observations = inputs["observations"] if inputs["states"] is None else inputs["states"]
        return self.output(torch.cat([observations, inputs["taken_actions"]], dim=-1)), {}


def make_agent(recurrent: bool, asymmetric: bool, scheduling: str, learn_entropy: bool) -> SAC | SAC_RNN:
    torch.manual_seed(11)
    space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    scheduler = torch.optim.lr_scheduler.StepLR
    scheduler_kwargs = {"step_size": 2, "gamma": 0.5}
    cfg = {
        "batch_size": 2,
        "gradient_steps": 2,
        "learning_rate": (0.01, 0.02, 0.08),
        "learn_entropy": learn_entropy,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    if scheduling == "all":
        cfg.update(learning_rate_scheduler=scheduler, learning_rate_scheduler_kwargs=scheduler_kwargs)
    elif scheduling == "entropy":
        cfg.update(
            learning_rate_scheduler=(None, None, scheduler),
            learning_rate_scheduler_kwargs=({}, {}, scheduler_kwargs),
        )
    agent = (SAC_RNN if recurrent else SAC)(
        models={
            "policy": Policy(recurrent),
            "critic_1": Critic(),
            "critic_2": Critic(),
            "target_critic_1": Critic(),
            "target_critic_2": Critic(),
        },
        memory=RandomMemory(memory_size=4, device="cpu"),
        observation_space=space,
        state_space=space if asymmetric else None,
        action_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(1,)),
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode()
    for timestep in range(4):
        observations = torch.tensor([[0.1 * timestep, -0.2]])
        states = observations + 0.5 if asymmetric else None
        with torch.no_grad():
            actions, _ = agent.act(observations, states, timestep=timestep, timesteps=4)
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.tensor([[0.2 + 0.1 * timestep]]),
                next_observations=observations + 0.1,
                next_states=None if states is None else states + 0.1,
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=4,
            )
    agent.enable_models_training_mode(True)
    agent.write_interval = 1
    return agent


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("scheduling", ["none", "entropy", "all"])
@pytest.mark.parametrize("learn_entropy", [False, True])
def test_sac_entropy_schedule_updates_and_checkpoint(
    recurrent: bool, asymmetric: bool, scheduling: str, learn_entropy: bool, tmp_path: Path
) -> None:
    agent = make_agent(recurrent, asymmetric, scheduling, learn_entropy)
    for update in range(3):
        agent.update(timestep=4 + update, timesteps=8)
        if learn_entropy:
            factor = 0.5 ** (update + 1) if scheduling != "none" else 1.0
            assert agent.entropy_optimizer.param_groups[0]["lr"] == pytest.approx(0.08 * factor)
            if scheduling != "none":
                assert agent.tracking_data["Learning / Entropy learning rate"][-1] == pytest.approx(0.08 * factor)
        else:
            assert "entropy_scheduler" not in agent.checkpoint_modules
            assert "Learning / Entropy learning rate" not in agent.tracking_data
        factor = 0.5 ** (update + 1) if scheduling == "all" else 1.0
        assert agent.policy_optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * factor)
        assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(0.02 * factor)

    if learn_entropy and scheduling != "none":
        checkpoint = str(tmp_path / "agent.pt")
        agent.save(checkpoint)
        resumed = make_agent(recurrent, asymmetric, scheduling, learn_entropy)
        resumed.load(checkpoint)
        assert resumed.entropy_scheduler.optimizer is resumed.entropy_optimizer
        assert resumed.entropy_scheduler.state_dict() == agent.entropy_scheduler.state_dict()
        assert resumed.entropy_optimizer.param_groups[0]["lr"] == agent.entropy_optimizer.param_groups[0]["lr"]
