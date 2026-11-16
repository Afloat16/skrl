from typing import Any

import pytest

import copy
import gymnasium
from packaging import version

import torch

from skrl.agents.torch.ddpg import DDPG, DDPG_RNN
from skrl.agents.torch.sac import SAC, SAC_RNN
from skrl.agents.torch.td3 import TD3, TD3_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class Policy(Model):
    def __init__(self, recurrent: bool) -> None:
        super().__init__(
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)),
            action_space=gymnasium.spaces.Box(-10.0, 10.0, shape=(1,)),
            device="cpu",
        )
        self.recurrent = recurrent
        self.encoder = torch.nn.Linear(2, 2)
        self.rnn = torch.nn.GRU(2, 2, batch_first=True) if recurrent else None
        self.output = torch.nn.Linear(2, 1)
        self.log_std = torch.nn.Parameter(torch.tensor([-0.5]), requires_grad=False)

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 1, 2)]}} if self.recurrent else {}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        values = self.encoder(inputs["observations"])
        outputs = {"log_std": self.log_std}
        if self.recurrent:
            length = 2 if self.training else 1
            values, state = self.rnn(values.reshape(-1, length, 2), inputs["rnn"][0][:, ::length])
            values = values.reshape(-1, 2)
            outputs["rnn"] = [state]
        return self.output(values), outputs


class DeterministicPolicy(DeterministicMixin, Policy):
    def __init__(self, recurrent: bool) -> None:
        Policy.__init__(self, recurrent)
        DeterministicMixin.__init__(self)


class GaussianPolicy(GaussianMixin, Policy):
    def __init__(self, recurrent: bool) -> None:
        Policy.__init__(self, recurrent)
        GaussianMixin.__init__(self)
        self.log_std.requires_grad_(True)


class Critic(DeterministicMixin, Model):
    def __init__(self, encoder: torch.nn.Module | None = None) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.encoder = torch.nn.Linear(2, 2) if encoder is None else encoder
        self.network = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.network(torch.cat([self.encoder(inputs["observations"]), inputs["taken_actions"]], dim=-1)), {}


def make_agent(agent_class: type, mixed_precision: bool, freeze: str, share_encoder: bool) -> Any:
    torch.manual_seed(23)
    recurrent = agent_class in (DDPG_RNN, SAC_RNN, TD3_RNN)
    policy = GaussianPolicy(recurrent) if agent_class in (SAC, SAC_RNN) else DeterministicPolicy(recurrent)
    policy.output.bias.requires_grad_(freeze != "bias")
    if freeze == "all":
        policy.freeze_parameters()
    models = {"policy": policy}
    if agent_class in (DDPG, DDPG_RNN):
        models.update(critic=Critic(policy.encoder if share_encoder else None))
        models.update(target_policy=copy.deepcopy(policy), target_critic=Critic())
    else:
        models.update(
            critic_1=Critic(policy.encoder if share_encoder else None),
            critic_2=Critic(),
            target_critic_1=Critic(),
            target_critic_2=Critic(),
        )
        if agent_class in (TD3, TD3_RNN):
            models["target_policy"] = copy.deepcopy(policy)
    cfg = {
        "batch_size": 4,
        "mixed_precision": mixed_precision,
        "grad_norm_clip": 0.5,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    if agent_class in (TD3, TD3_RNN):
        cfg["policy_delay"] = 1
    agent = agent_class(
        models=models,
        memory=RandomMemory(memory_size=8, device="cpu"),
        observation_space=policy.observation_space,
        action_space=policy.action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode()
    for timestep in range(8):
        observations = torch.tensor([[0.1 * timestep, -0.2]])
        with torch.no_grad():
            actions, _ = agent.act(observations, None, timestep=timestep, timesteps=8)
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.tensor([[0.2 + 0.1 * timestep]]),
                next_observations=observations + 0.1,
                next_states=None,
                terminated=torch.zeros((1, 1), dtype=torch.bool),
                truncated=torch.zeros((1, 1), dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=8,
            )
    agent.enable_models_training_mode(True)
    return agent


@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("agent_class", [DDPG, DDPG_RNN, SAC, SAC_RNN, TD3, TD3_RNN])
@pytest.mark.parametrize("mixed_precision,freeze", [(False, "all"), (True, "all"), (True, "none")])
def test_frozen_actor_keeps_critic_and_entropy_updates(
    agent_class: type, mixed_precision: bool, freeze: str, share_encoder: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mixed_precision and version.parse(torch.__version__) < version.parse("2.4"):
        pytest.skip("CPU GradScaler requires Torch 2.4")
    agent = make_agent(agent_class, mixed_precision, freeze, share_encoder)
    before_policy = {name: value.clone() for name, value in agent.policy.state_dict().items()}
    critics = [model for name, model in agent.models.items() if name.startswith("critic")]
    before_critics = [[parameter.detach().clone() for parameter in critic.parameters()] for critic in critics]
    entropy_before = agent.log_entropy_coefficient.detach().clone() if isinstance(agent, (SAC, SAC_RNN)) else None
    counts = {"policy": 0, "critic": 0}
    for name, optimizer in [("policy", agent.policy_optimizer), ("critic", agent.critic_optimizer)]:
        original = optimizer.step

        def step(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(optimizer, "step", step)

    for timestep in range(3):
        agent.update(timestep=timestep, timesteps=3)

    assert counts["critic"] == 3
    assert counts["policy"] == (0 if mixed_precision and freeze == "all" else 3)
    for critic, previous in zip(critics, before_critics):
        assert any(not torch.equal(parameter, old) for parameter, old in zip(critic.parameters(), previous))
        assert all(torch.isfinite(parameter).all() for parameter in critic.parameters())
    if freeze == "all":
        for name, value in agent.policy.state_dict().items():
            torch.testing.assert_close(value, before_policy[name], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in agent.policy.parameters())
    else:
        assert any(not torch.equal(value, before_policy[name]) for name, value in agent.policy.state_dict().items())
    if entropy_before is not None:
        assert not torch.equal(agent.log_entropy_coefficient, entropy_before)
        assert torch.isfinite(agent.log_entropy_coefficient)


@pytest.mark.parametrize("phase", ["fresh", "cleared", "frozen"])
def test_models_without_gradients_require_no_reduction(phase: str) -> None:
    model = Critic()
    if phase == "cleared":
        values, _ = model.act({"observations": torch.tensor([[0.1, 0.2]]), "taken_actions": torch.tensor([[0.3]])})
        values.sum().backward()
        model.zero_grad(set_to_none=True)
    elif phase == "frozen":
        model.freeze_parameters()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    # No process group is needed when there are no gradients to communicate.
    model.reduce_parameters()
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
