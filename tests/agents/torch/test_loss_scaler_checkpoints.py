from typing import Any

import pytest

from pathlib import Path
import gymnasium
from packaging import version

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


def make_agent(agent_class: type, mixed_precision: bool) -> Any:
    cfg = {
        "learning_rate": 0.01,
        "mixed_precision": mixed_precision,
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
    if agent_class in (SAC, SAC_RNN):
        cfg["learn_entropy"] = False
    return agent_class(
        models={role: SmallModel() for role in roles},
        observation_space=space,
        action_space=gymnasium.spaces.Discrete(2) if agent_class in (CEM, DQN, DDQN) else space,
        device="cpu",
        cfg=cfg,
    )


def optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    if isinstance(agent, (IPPO, MAPPO)):
        return list(agent.optimizers.values())
    return [value for value in vars(agent).values() if isinstance(value, torch.optim.Optimizer)]


def checkpoint_groups(agent: Any) -> list[dict[str, Any]]:
    return list(agent.checkpoint_modules.values()) if isinstance(agent, (IPPO, MAPPO)) else [agent.checkpoint_modules]


def step(agent: Any, overflow: bool = False) -> None:
    for optimizer in optimizers(agent):
        optimizer.zero_grad()
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        before = [parameter.detach().clone() for parameter in parameters]
        loss = sum(parameter.square().sum() for parameter in parameters)
        if overflow:
            loss = loss * float("inf")
        agent.scaler.scale(loss).backward()
        agent.scaler.step(optimizer)
        if overflow:
            for parameter, previous in zip(parameters, before):
                torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
    agent.scaler.update()


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.4"), reason="CPU GradScaler requires Torch 2.4")
@pytest.mark.parametrize("agent_class", AGENTS)
def test_checkpoint_resumes_dynamic_loss_scale_and_optimizer_updates(agent_class: type, tmp_path: Path) -> None:
    reference = make_agent(agent_class, True)
    reference.scaler.set_growth_interval(3)
    for overflow in [False, False, True, False]:
        step(reference, overflow)
    assert reference.scaler.get_scale() == 32768
    assert reference.scaler.state_dict()["_growth_tracker"] == 1
    checkpoint = str(tmp_path / "agent.pt")
    reference.save(checkpoint)

    resumed = make_agent(agent_class, True)
    resumed.load(checkpoint)
    assert resumed.scaler.state_dict() == reference.scaler.state_dict()
    assert all(group["scaler"] is resumed.scaler for group in checkpoint_groups(resumed))
    # Cross a growth boundary, another backoff, and a second growth boundary.
    for overflow in [False, False, True, False, False, False]:
        step(reference, overflow)
        step(resumed, overflow)
        assert resumed.scaler.state_dict() == reference.scaler.state_dict()
        for actual, expected in zip(optimizers(resumed), optimizers(reference)):
            for actual_group, expected_group in zip(actual.param_groups, expected.param_groups):
                for actual_parameter, expected_parameter in zip(actual_group["params"], expected_group["params"]):
                    torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.4"), reason="CPU GradScaler requires Torch 2.4")
@pytest.mark.parametrize("agent_class", AGENTS)
def test_legacy_checkpoint_keeps_constructor_loss_scaler_state(agent_class: type, tmp_path: Path) -> None:
    original = make_agent(agent_class, True)
    step(original, overflow=True)
    for group in checkpoint_groups(original):
        group.pop("scaler", None)
    checkpoint = str(tmp_path / "legacy.pt")
    original.save(checkpoint)
    resumed = make_agent(agent_class, True)
    expected = resumed.scaler.state_dict()
    resumed.load(checkpoint)
    assert resumed.scaler.state_dict() == expected
    step(resumed)


@pytest.mark.parametrize("agent_class", AGENTS)
def test_disabled_loss_scalers_do_not_add_empty_checkpoint_state(agent_class: type, tmp_path: Path) -> None:
    original = make_agent(agent_class, False)
    assert all("scaler" not in group for group in checkpoint_groups(original))
    checkpoint = str(tmp_path / "disabled.pt")
    original.save(checkpoint)
    for mixed_precision in [False, True] if version.parse(torch.__version__) >= version.parse("2.4") else [False]:
        resumed = make_agent(agent_class, mixed_precision)
        expected = resumed.scaler.state_dict()
        resumed.load(checkpoint)
        assert resumed.scaler.state_dict() == expected
        step(resumed)


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.4"), reason="CPU GradScaler requires Torch 2.4")
@pytest.mark.parametrize("agent_class", AGENTS)
def test_enabled_scaler_checkpoint_can_load_with_mixed_precision_disabled(agent_class: type, tmp_path: Path) -> None:
    original = make_agent(agent_class, True)
    step(original, overflow=True)
    checkpoint = str(tmp_path / "enabled.pt")
    original.save(checkpoint)
    resumed = make_agent(agent_class, False)
    resumed.load(checkpoint)
    assert not resumed.scaler.is_enabled()
    assert resumed.scaler.state_dict() == {}
    step(resumed)
