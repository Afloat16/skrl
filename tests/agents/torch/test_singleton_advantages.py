import pytest

import importlib
from gymnasium import spaces

import torch

from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


MODULES = [
    "agents.torch.a2c.a2c",
    "agents.torch.a2c.a2c_rnn",
    "agents.torch.amp.amp",
    "agents.torch.ppo.ppo",
    "agents.torch.ppo.ppo_rnn",
    "agents.torch.rpo.rpo",
    "agents.torch.rpo.rpo_rnn",
    "agents.torch.trpo.trpo",
    "agents.torch.trpo.trpo_rnn",
    "multi_agents.torch.ippo.ippo",
    "multi_agents.torch.mappo.mappo",
]


@pytest.mark.parametrize(
    "module,terminated,truncated,bootstrap",
    [
        (module, *flags)
        for module in MODULES
        for flags in (
            [(False, False, False), (True, False, False)]
            if module.endswith("_rnn")
            else [(False, False, False), (True, False, False), (False, True, False), (False, True, True)]
        )
    ],
)
def test_single_transition_gae_returns_finite_zero_normalized_advantage(module, terminated, truncated, bootstrap):
    compute_gae = importlib.import_module("skrl." + module).compute_gae
    arguments = dict(
        rewards=torch.tensor([[[2.0]]]),
        terminated=torch.tensor([[[terminated]]]),
        values=torch.tensor([[[0.5]]]),
        discount_factor=0.9,
    )
    if module.endswith("_rnn"):
        arguments["next_values"] = torch.tensor([[1.5]])
    else:
        arguments.update(
            truncated=torch.tensor([[[truncated]]]), last_values=torch.tensor([[1.5]]), time_limit_bootstrap=bootstrap
        )
    returns, advantages = compute_gae(**arguments)
    expected_return = 2.0 if terminated or (truncated and bootstrap) else 3.35
    torch.testing.assert_close(returns, torch.full_like(returns, expected_return))
    torch.testing.assert_close(advantages, torch.zeros_like(advantages))


@pytest.mark.parametrize("module", MODULES)
def test_multiple_transition_normalization_keeps_existing_sample_variance(module):
    compute_gae = importlib.import_module("skrl." + module).compute_gae
    rewards = torch.tensor([[[1.0], [3.0]], [[2.0], [7.0]]])
    arguments = dict(
        rewards=rewards,
        terminated=torch.zeros_like(rewards, dtype=torch.bool),
        values=torch.zeros_like(rewards),
        discount_factor=0.0,
    )
    if module.endswith("_rnn"):
        arguments["next_values"] = torch.zeros(2, 1)
    else:
        arguments.update(truncated=torch.zeros_like(rewards, dtype=torch.bool), last_values=torch.zeros(2, 1))
    returns, advantages = compute_gae(**arguments)
    torch.testing.assert_close(returns, rewards)
    torch.testing.assert_close(advantages, (rewards - rewards.mean()) / (rewards.std() + 1e-8))


@pytest.mark.parametrize("kind", ["a2c", "ppo", "rpo"])
@pytest.mark.parametrize("terminated", [False, True])
def test_native_agent_single_transition_update_keeps_parameters_finite(kind, terminated):
    agent_type = getattr(importlib.import_module(f"skrl.agents.torch.{kind}"), kind.upper())
    observation_space, action_space = spaces.Box(-1, 1, (2,)), spaces.Box(-1, 1, (1,))
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "elu"}]
    policy = gaussian_model(
        observation_space=observation_space, action_space=action_space, device="cpu", network=network, output="ACTIONS"
    )
    value = deterministic_model(
        observation_space=observation_space, action_space=action_space, device="cpu", network=network, output="ONE"
    )
    policy.init_state_dict()
    value.init_state_dict()
    memory = RandomMemory(memory_size=1, num_envs=1, device="cpu")
    cfg = {
        "rollouts": 1,
        "mini_batches": 1,
        "entropy_loss_scale": 0.01,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    if kind != "a2c":
        cfg["learning_epochs"] = 1
    agent = agent_type(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode(True)
    observations = torch.zeros(1, 2)
    with torch.no_grad():
        actions, _ = agent.act(observations, None, timestep=0, timesteps=1)
        agent.record_transition(
            observations=observations,
            states=None,
            actions=actions,
            rewards=torch.ones(1, 1),
            next_observations=observations,
            next_states=None,
            terminated=torch.tensor([[terminated]]),
            truncated=torch.zeros(1, 1, dtype=torch.bool),
            infos={},
            timestep=0,
            timesteps=1,
        )
    agent.post_interaction(timestep=0, timesteps=1)
    assert all(torch.isfinite(parameter).all() for model in (policy, value) for parameter in model.parameters())
    torch.testing.assert_close(memory.get_tensor_by_name("advantages"), torch.zeros(1, 1, 1))
