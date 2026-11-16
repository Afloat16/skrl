from typing import Any

import pytest

from pathlib import Path
import gymnasium

import torch

from skrl.agents.torch.a2c import A2C, A2C_RNN
from skrl.agents.torch.amp import AMP
from skrl.agents.torch.cem import CEM
from skrl.agents.torch.ddpg import DDPG, DDPG_RNN
from skrl.agents.torch.ddqn import DDQN
from skrl.agents.torch.dqn import DQN
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.agents.torch.rpo import RPO, RPO_RNN
from skrl.agents.torch.sac import SAC, SAC_RNN
from skrl.agents.torch.td3 import TD3, TD3_RNN
from skrl.agents.torch.trpo import TRPO, TRPO_RNN
from skrl.models.torch import DeterministicMixin, Model
from skrl.multi_agents.torch.ippo import IPPO
from skrl.multi_agents.torch.mappo import MAPPO


AGENTS = [
    A2C,
    A2C_RNN,
    AMP,
    CEM,
    DDPG,
    DDPG_RNN,
    DDQN,
    DQN,
    PPO,
    PPO_RNN,
    RPO,
    RPO_RNN,
    SAC,
    SAC_RNN,
    TD3,
    TD3_RNN,
    TRPO,
    TRPO_RNN,
    IPPO,
    MAPPO,
]


class SmallModel(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.3]))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return inputs["observations"] * self.weight, {}


def make_agent(agent_class: type, scheduler_class: type | None) -> Any:
    kwargs = (
        {"step_size": 3, "gamma": 0.5}
        if scheduler_class is torch.optim.lr_scheduler.StepLR
        else {"T_0": 3, "T_mult": 2}
    )
    cfg = {
        "learning_rate": 0.1,
        "learning_rate_scheduler": scheduler_class,
        "learning_rate_scheduler_kwargs": kwargs,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    if agent_class in (IPPO, MAPPO):
        names = ["first", "second"]
        return agent_class(
            possible_agents=names,
            models={name: {"policy": SmallModel(), "value": SmallModel()} for name in names},
            observation_spaces={name: space for name in names},
            state_spaces={name: space for name in names},
            action_spaces={name: space for name in names},
            device="cpu",
            cfg=cfg,
        )
    if agent_class in (DQN, DDQN):
        roles = ["q_network", "target_q_network"]
    elif agent_class in (DDPG, DDPG_RNN):
        roles = ["policy", "target_policy", "critic", "target_critic"]
    elif agent_class in (SAC, SAC_RNN, TD3, TD3_RNN):
        roles = ["policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]
        if agent_class in (TD3, TD3_RNN):
            roles.append("target_policy")
    else:
        roles = ["policy"] if agent_class is CEM else ["policy", "value"]
        if agent_class is AMP:
            roles.append("discriminator")
    return agent_class(
        models={role: SmallModel() for role in roles},
        observation_space=space,
        action_space=gymnasium.spaces.Discrete(2) if agent_class in (CEM, DQN, DDQN) else space,
        device="cpu",
        cfg=cfg,
    )


def schedulers(agent: Any) -> list[Any]:
    if isinstance(agent, (IPPO, MAPPO)):
        return list(agent.schedulers.values())
    return [
        getattr(agent, name)
        for name in ("scheduler", "policy_scheduler", "critic_scheduler", "value_scheduler")
        if hasattr(agent, name)
    ]


def step(agent: Any) -> None:
    for scheduler in schedulers(agent):
        optimizer = scheduler.optimizer
        optimizer.zero_grad()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = parameter.detach().clone()
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("agent_class", AGENTS)
@pytest.mark.parametrize(
    "scheduler_class", [torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts]
)
def test_checkpoint_resumes_schedule_and_parameter_updates(
    agent_class: type, scheduler_class: type, tmp_path: Path
) -> None:
    reference = make_agent(agent_class, scheduler_class)
    for _ in range(4):
        step(reference)
    checkpoint = str(tmp_path / "agent.pt")
    reference.save(checkpoint)
    resumed = make_agent(agent_class, scheduler_class)
    resumed.load(checkpoint)

    # Cross both a step-decay boundary and a cosine restart after resuming.
    for _ in range(6):
        step(reference)
        step(resumed)
        for expected, actual in zip(schedulers(reference), schedulers(resumed)):
            assert actual.get_last_lr() == expected.get_last_lr()
            for expected_group, actual_group in zip(expected.optimizer.param_groups, actual.optimizer.param_groups):
                for expected_parameter, actual_parameter in zip(expected_group["params"], actual_group["params"]):
                    torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
    for expected, actual in zip(schedulers(reference), schedulers(resumed)):
        assert actual.state_dict() == expected.state_dict()


@pytest.mark.parametrize("agent_class", AGENTS)
def test_legacy_checkpoint_keeps_constructor_scheduler_state(agent_class: type, tmp_path: Path) -> None:
    original = make_agent(agent_class, torch.optim.lr_scheduler.StepLR)
    for _ in range(4):
        step(original)
    groups = (
        original.checkpoint_modules.values() if isinstance(original, (IPPO, MAPPO)) else [original.checkpoint_modules]
    )
    for group in groups:
        for key in list(group):
            if "scheduler" in key:
                del group[key]
    checkpoint = str(tmp_path / "legacy.pt")
    original.save(checkpoint)
    resumed = make_agent(agent_class, torch.optim.lr_scheduler.StepLR)
    initial = [scheduler.state_dict() for scheduler in schedulers(resumed)]
    resumed.load(checkpoint)
    assert [scheduler.state_dict() for scheduler in schedulers(resumed)] == initial


@pytest.mark.parametrize("agent_class", AGENTS)
def test_disabled_schedulers_are_not_saved(agent_class: type, tmp_path: Path) -> None:
    agent = make_agent(agent_class, None)
    groups = agent.checkpoint_modules.values() if isinstance(agent, (IPPO, MAPPO)) else [agent.checkpoint_modules]
    assert all("scheduler" not in key for group in groups for key in group)
    checkpoint = str(tmp_path / "no-schedulers.pt")
    agent.save(checkpoint)
    make_agent(agent_class, None).load(checkpoint)
