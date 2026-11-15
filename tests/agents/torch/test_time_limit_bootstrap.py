import pytest

import importlib
import gymnasium

import numpy as np
import torch

from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


def _models(observation_space, action_space):
    kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "device": "cpu",
        "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "elu"}],
    }
    models = {
        "policy": gaussian_model(**kwargs, output="ACTIONS"),
        "value": deterministic_model(**kwargs, output="ONE"),
    }
    for model in models.values():
        model.init_state_dict()
    models["value"].init_parameters("constant_", val=0)
    models["value"].init_biases("constant_", val=10)
    return models


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("name", ["A2C", "A2C_RNN", "AMP", "PPO", "PPO_RNN", "RPO", "RPO_RNN", "TRPO", "TRPO_RNN"])
def test_terminal_truncation_rewards(name, bootstrap):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    models = _models(observation_space, action_space)
    memory = RandomMemory(memory_size=1, num_envs=4, device="cpu")
    agent_class = getattr(importlib.import_module(f"skrl.agents.torch.{name.split('_')[0].lower()}"), name)
    kwargs = {}
    if name == "AMP":
        models["discriminator"] = _models(observation_space, action_space)["value"]
        kwargs["amp_observation_space"] = observation_space
    agent = agent_class(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "rollouts": 1,
            "discount_factor": 0.5,
            "time_limit_bootstrap": bootstrap,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
        **kwargs,
    )
    agent.init()
    agent.enable_training_mode(True)
    observations = torch.zeros((4, 1))
    # Ongoing, terminated, truncated, and simultaneously terminated/truncated.
    terminated = torch.tensor([[False], [True], [False], [True]])
    truncated = torch.tensor([[False], [False], [True], [True]])
    with torch.no_grad():
        actions, _ = agent.act(observations, None, timestep=0, timesteps=1)
        agent.record_transition(
            observations=observations,
            states=None,
            actions=actions,
            rewards=torch.ones((4, 1)),
            next_observations=observations,
            next_states=None,
            terminated=terminated,
            truncated=truncated,
            infos={"amp_obs": observations},
            timestep=0,
            timesteps=1,
        )
    expected = torch.tensor([1.0, 1.0, 6.0 if bootstrap else 1.0, 1.0]).view(1, 4, 1)
    torch.testing.assert_close(memory.get_tensor_by_name("rewards"), expected)


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("name", ["IPPO", "MAPPO"])
def test_multi_agent_terminal_truncation_rewards(name, bootstrap):
    possible_agents = ["first", "second"]
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    memories = {uid: RandomMemory(memory_size=1, num_envs=4, device="cpu") for uid in possible_agents}
    agent_class = getattr(importlib.import_module(f"skrl.multi_agents.torch.{name.lower()}"), name)
    agent = agent_class(
        possible_agents=possible_agents,
        models={uid: _models(observation_space, action_space) for uid in possible_agents},
        memories=memories,
        observation_spaces={uid: observation_space for uid in possible_agents},
        state_spaces={uid: None for uid in possible_agents},
        action_spaces={uid: action_space for uid in possible_agents},
        device="cpu",
        cfg={
            "rollouts": 1,
            "discount_factor": 0.5,
            "time_limit_bootstrap": bootstrap,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode(True)
    observations = {uid: torch.zeros((4, 1)) for uid in possible_agents}
    states = {uid: None for uid in possible_agents}
    terminated = {
        "first": torch.tensor([[False], [True], [False], [True]]),
        "second": torch.tensor([[True], [False], [True], [False]]),
    }
    truncated = {
        "first": torch.tensor([[False], [False], [True], [True]]),
        "second": torch.tensor([[True], [True], [False], [False]]),
    }
    with torch.no_grad():
        actions, _ = agent.act(observations, states, timestep=0, timesteps=1)
        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards={uid: torch.ones((4, 1)) for uid in possible_agents},
            next_observations=observations,
            next_states=states,
            terminated=terminated,
            truncated=truncated,
            infos={},
            timestep=0,
            timesteps=1,
        )
    expected = torch.tensor([1.0, 1.0, 6.0 if bootstrap else 1.0, 1.0]).view(1, 4, 1)
    torch.testing.assert_close(memories["first"].get_tensor_by_name("rewards"), expected)
    torch.testing.assert_close(memories["second"].get_tensor_by_name("rewards"), expected.flip(1))


class _TerminalEnv(gymnasium.Env):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))

    def __init__(self, terminal):
        self.terminal = terminal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 1.0, self.terminal, False, {}


@pytest.mark.parametrize("bootstrap", [False, True])
def test_ppo_time_limit_terminal_returns(bootstrap):
    from skrl.agents.torch.ppo import PPO

    envs = [
        gymnasium.wrappers.TimeLimit(_TerminalEnv(terminal), max_episode_steps=limit)
        for terminal, limit in [(False, 2), (True, 2), (False, 1), (True, 1)]
    ]
    observations = torch.tensor(np.stack([env.reset()[0] for env in envs]))
    models = _models(envs[0].observation_space, envs[0].action_space)
    memory = RandomMemory(memory_size=1, num_envs=4, device="cpu")
    agent = PPO(
        models=models,
        memory=memory,
        observation_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        device="cpu",
        cfg={
            "rollouts": 1,
            "learning_epochs": 1,
            "mini_batches": 1,
            "discount_factor": 0.5,
            "time_limit_bootstrap": bootstrap,
            "learning_rate": 0.01,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode(True)
    before = [parameter.detach().clone() for parameter in models["value"].parameters()]
    with torch.no_grad():
        actions, _ = agent.act(observations, None, timestep=0, timesteps=1)
        transitions = [env.step(action.numpy()) for env, action in zip(envs, actions)]
        terminated = torch.tensor([[transition[2]] for transition in transitions])
        truncated = torch.tensor([[transition[3]] for transition in transitions])
        assert terminated[:, 0].tolist() == [False, True, False, True]
        assert truncated[:, 0].tolist() == [False, False, True, True]
        agent.record_transition(
            observations=observations,
            states=None,
            actions=actions,
            rewards=torch.tensor([[transition[1]] for transition in transitions]),
            next_observations=torch.tensor(np.stack([transition[0] for transition in transitions])),
            next_states=None,
            terminated=terminated,
            truncated=truncated,
            infos={},
            timestep=0,
            timesteps=1,
        )
    agent.post_interaction(timestep=0, timesteps=1)
    # A time limit must not introduce a continuation value at a true terminal state.
    torch.testing.assert_close(memory.get_tensor_by_name("returns"), torch.tensor([6.0, 1.0, 6.0, 1.0]).view(1, 4, 1))
    assert any(not torch.equal(old, new) for old, new in zip(before, models["value"].parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in models["value"].parameters())
    for env in envs:
        env.close()
