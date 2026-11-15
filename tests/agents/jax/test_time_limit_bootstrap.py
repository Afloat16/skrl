import pytest

import importlib
import gymnasium

import jax.numpy as jnp
import numpy as np

from skrl.memories.jax import RandomMemory
from skrl.utils.model_instantiators.jax import deterministic_model, gaussian_model


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
    for role, model in models.items():
        model.init_state_dict(role=role)
    models["value"].init_parameters("zeros")
    models["value"].init_biases("constant", value=10)
    return models


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("flag_dtype", [jnp.bool_, jnp.int8])
@pytest.mark.parametrize("name", ["A2C", "PPO", "RPO"])
def test_terminal_truncation_rewards(name, bootstrap, flag_dtype):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    models = _models(observation_space, action_space)
    memory = RandomMemory(memory_size=1, num_envs=4, device="cpu")
    agent_class = getattr(importlib.import_module(f"skrl.agents.jax.{name.split('_')[0].lower()}"), name)
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
    )
    agent.init()
    agent.enable_training_mode(True)
    observations = jnp.zeros((4, 1))
    # Ongoing, terminated, truncated, and simultaneously terminated/truncated.
    terminated = jnp.array([[False], [True], [False], [True]], dtype=flag_dtype)
    truncated = jnp.array([[False], [False], [True], [True]], dtype=flag_dtype)
    actions, _ = agent.act(observations, None, timestep=0, timesteps=1)
    agent.record_transition(
        observations=observations,
        states=None,
        actions=actions,
        rewards=jnp.ones((4, 1)),
        next_observations=observations,
        next_states=None,
        terminated=terminated,
        truncated=truncated,
        infos={"amp_obs": observations},
        timestep=0,
        timesteps=1,
    )
    expected = jnp.array([1.0, 1.0, 6.0 if bootstrap else 1.0, 1.0]).reshape(1, 4, 1)
    np.testing.assert_allclose(memory.get_tensor_by_name("rewards"), expected)


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("flag_dtype", [jnp.bool_, jnp.int8])
@pytest.mark.parametrize("name", ["IPPO", "MAPPO"])
def test_multi_agent_terminal_truncation_rewards(name, bootstrap, flag_dtype):
    possible_agents = ["first", "second"]
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    memories = {uid: RandomMemory(memory_size=1, num_envs=4, device="cpu") for uid in possible_agents}
    agent_class = getattr(importlib.import_module(f"skrl.multi_agents.jax.{name.lower()}"), name)
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
    observations = {uid: jnp.zeros((4, 1)) for uid in possible_agents}
    states = {uid: None for uid in possible_agents}
    terminated = {
        "first": jnp.array([[False], [True], [False], [True]], dtype=flag_dtype),
        "second": jnp.array([[True], [False], [True], [False]], dtype=flag_dtype),
    }
    truncated = {
        "first": jnp.array([[False], [False], [True], [True]], dtype=flag_dtype),
        "second": jnp.array([[True], [True], [False], [False]], dtype=flag_dtype),
    }
    actions, _ = agent.act(observations, states, timestep=0, timesteps=1)
    agent.record_transition(
        observations=observations,
        states=states,
        actions=actions,
        rewards={uid: jnp.ones((4, 1)) for uid in possible_agents},
        next_observations=observations,
        next_states=states,
        terminated=terminated,
        truncated=truncated,
        infos={},
        timestep=0,
        timesteps=1,
    )
    expected = jnp.array([1.0, 1.0, 6.0 if bootstrap else 1.0, 1.0]).reshape(1, 4, 1)
    np.testing.assert_allclose(memories["first"].get_tensor_by_name("rewards"), expected)
    np.testing.assert_allclose(memories["second"].get_tensor_by_name("rewards"), expected[:, ::-1])

