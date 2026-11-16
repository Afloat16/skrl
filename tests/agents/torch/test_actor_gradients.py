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
@pytest.mark.parametrize(
    "mixed_precision, freeze",
    [(False, "none"), (False, "bias"), (False, "all"), (True, "none"), (True, "bias")],
)
def test_actor_update_only_accumulates_policy_gradients(
    agent_class: type, mixed_precision: bool, freeze: str, share_encoder: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mixed_precision and version.parse(torch.__version__) < version.parse("2.4"):
        pytest.skip("The agent only enables its CPU GradScaler with PyTorch 2.4 and newer")
    agent = make_agent(agent_class, mixed_precision, freeze, share_encoder)
    reference = make_agent(agent_class, mixed_precision, freeze, share_encoder)
    critics = [model for name, model in agent.models.items() if name.startswith("critic")]
    if share_encoder:
        assert agent.policy.encoder is critics[0].encoder
    critic_gradients = {}
    original_step = agent.critic_optimizer.step

    def critic_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        critic_gradients.update(
            {
                id(parameter): parameter.grad.clone()
                for critic in critics
                for parameter in critic.parameters()
                if parameter.grad is not None
            }
        )
        return result

    monkeypatch.setattr(agent.critic_optimizer, "step", critic_step)
    torch.manual_seed(19)
    agent.update(timestep=8, timesteps=10)

    # Compare the complete update with an unrestricted backward pass, including
    # the gradient through each critic's actions and the subsequent target update.
    original_backward = torch.Tensor.backward

    def unrestricted_backward(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("inputs", None)
        original_backward(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "backward", unrestricted_backward)
        torch.manual_seed(19)
        reference.update(timestep=8, timesteps=10)

    for name, model in agent.models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.models[name].state_dict()[key], rtol=0, atol=0)
    for parameter, expected in zip(agent.policy.parameters(), reference.policy.parameters()):
        assert parameter.requires_grad == expected.requires_grad
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
    if agent_class in (SAC, SAC_RNN):
        torch.testing.assert_close(agent.log_entropy_coefficient, reference.log_entropy_coefficient, rtol=0, atol=0)
    if freeze != "all":
        policy_parameters = {id(parameter) for parameter in agent.policy.parameters()}
        for critic in critics:
            for parameter in critic.parameters():
                if id(parameter) not in policy_parameters:
                    torch.testing.assert_close(parameter.grad, critic_gradients[id(parameter)], rtol=0, atol=0)
