import pytest

import importlib
from gymnasium import spaces

import numpy as np


CASES = {
    "vector_early_reward": (
        [[1, 10], [1, 0], [2, 0], [2, 0]],
        [[False, False], [True, False], [False, False], [True, True]],
        [1, 3, 5, 7],
    ),
    "vector_late_reward": (
        [[1, 0], [1, 0], [2, 0], [2, 10]],
        [[False, False], [True, False], [False, False], [True, True]],
        [4, 6],
    ),
    "discounted_single_env": (
        [[10], [0], [0], [6]],
        [[False], [True], [False], [True]],
        [0, 1],
    ),
    "first_step": ([[1, 4]], [[True, True]], [1]),
    "no_completed_episode": ([[1, 4], [2, 5]], [[False, False], [False, False]], []),
}


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_cem_optimizes_the_correct_completed_trajectories(backend, asymmetric, case, monkeypatch):
    framework = pytest.importorskip(backend)
    module = importlib.import_module(f"skrl.agents.{backend}.cem.cem")
    memory_type = importlib.import_module(f"skrl.memories.{backend}").RandomMemory
    factory = importlib.import_module(f"skrl.utils.model_instantiators.{backend}").categorical_model
    rewards, dones, expected_ids = CASES[case]
    rewards = np.asarray(rewards, dtype=np.float32)[..., None]
    dones = np.asarray(dones, dtype=bool)[..., None]
    rollouts, num_envs = rewards.shape[:2]
    observation_space = spaces.Box(-1000, 1000, (1,))
    state_space = spaces.Box(-1000, 1000, (1,)) if asymmetric else None
    action_space = spaces.Discrete(2)
    policy = factory(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "elu"}],
        output="ACTIONS",
    )
    policy.init_state_dict(role="policy")
    memory = memory_type(memory_size=rollouts, num_envs=num_envs, device="cpu")
    agent = module.CEM(
        models={"policy": policy},
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "rollouts": rollouts,
            "discount_factor": 0.5,
            "percentile": 0.7,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode(True)
    captured = []
    if backend == "torch":
        original = policy.act

        def capture(inputs, *, role=""):
            captured.append({k: None if v is None else v.detach().cpu().numpy().copy() for k, v in inputs.items()})
            return original(inputs, role=role)

        monkeypatch.setattr(policy, "act", capture)
        convert = framework.as_tensor
    else:
        original = module._update_policy

        def capture(policy_act, state_dict, inputs, actions, n):
            captured.append({k: None if v is None else np.asarray(v).copy() for k, v in inputs.items()})
            return original(policy_act, state_dict, inputs, actions, n)

        monkeypatch.setattr(module, "_update_policy", capture)
        convert = np.asarray

    for timestep in range(rollouts):
        observations = np.arange(timestep * num_envs, (timestep + 1) * num_envs, dtype=np.float32).reshape(-1, 1)
        states = convert(observations + 100) if asymmetric else None
        agent.record_transition(
            observations=convert(observations),
            states=states,
            actions=convert(observations.astype(np.int64) % 2),
            rewards=convert(rewards[timestep]),
            next_observations=convert(observations + num_envs),
            next_states=states,
            terminated=convert(dones[timestep]),
            truncated=convert(np.zeros_like(dones[timestep])),
            infos={},
            timestep=timestep,
            timesteps=rollouts,
        )
        agent.post_interaction(timestep=timestep, timesteps=rollouts)

    if not expected_ids:
        assert captured == []
        return
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0]["observations"].reshape(-1), expected_ids)
    if asymmetric:
        np.testing.assert_array_equal(captured[0]["states"].reshape(-1), np.asarray(expected_ids) + 100)
    else:
        assert captured[0]["states"] is None
    assert np.isfinite(agent.tracking_data["Loss / Policy loss"][-1])
