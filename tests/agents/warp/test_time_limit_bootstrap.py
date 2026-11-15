import pytest

import gymnasium

import numpy as np
import warp as wp

from skrl.agents.warp.ppo import PPO
from skrl.memories.warp import RandomMemory
from skrl.utils.model_instantiators.warp import deterministic_model, gaussian_model


@pytest.mark.parametrize("bootstrap", [False, True])
def test_terminal_truncation_rewards(bootstrap):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
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
    for parameter in models["value"].parameters():
        parameter.fill_(0.5)

    memory = RandomMemory(memory_size=1, num_envs=4, device="cpu")
    agent = PPO(
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
    observations = wp.array(np.ones((4, 1), dtype=np.float32), device="cpu")
    # Ongoing, terminated, truncated, and simultaneously terminated/truncated.
    terminated = wp.array(np.array([[0], [1], [0], [1]], dtype=np.int8), device="cpu")
    truncated = wp.array(np.array([[0], [0], [1], [1]], dtype=np.int8), device="cpu")
    next_values, _ = models["value"].act({"observations": observations, "states": None}, role="value")
    next_values = next_values.numpy().copy()
    assert np.all(next_values > 0.1)
    actions, _ = agent.act(observations, None, timestep=0, timesteps=1)
    agent.record_transition(
        observations=observations,
        states=None,
        actions=actions,
        rewards=wp.array(np.ones((4, 1), dtype=np.float32), device="cpu"),
        next_observations=observations,
        next_states=None,
        terminated=terminated,
        truncated=truncated,
        infos={},
        timestep=0,
        timesteps=1,
    )
    expected = np.ones((1, 4, 1), dtype=np.float32)
    if bootstrap:
        expected[0, 2, 0] += 0.5 * next_values[2, 0]
    np.testing.assert_allclose(memory.get_tensor_by_name("rewards").numpy(), expected)
