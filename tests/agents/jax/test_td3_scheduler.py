import pytest

import gymnasium

import numpy as np
import optax

from skrl.agents.jax.td3 import TD3
from skrl.memories.jax import RandomMemory
from skrl.utils.model_instantiators.jax import deterministic_model


@pytest.mark.parametrize("policy_delay", [1, 2, 3])
@pytest.mark.parametrize("gradient_steps", [1, 3])
def test_policy_scheduler_tracks_delayed_optimizer_steps(policy_delay, gradient_steps):
    observation_space = gymnasium.spaces.Box(-1, 1, (2,))
    action_space = gymnasium.spaces.Box(-1, 1, (1,))
    models = {}
    for role in ("policy", "target_policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        is_policy = "policy" in role
        models[role] = deterministic_model(
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            network=[
                {
                    "name": "net",
                    "input": "OBSERVATIONS" if is_policy else "concatenate([OBSERVATIONS, ACTIONS])",
                    "layers": [4],
                    "activations": "relu",
                }
            ],
            output="ACTIONS" if is_policy else "ONE",
        )
        models[role].init_state_dict(role=role)
    memory = RandomMemory(memory_size=4, num_envs=1, device="cpu")
    agent = TD3(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "batch_size": 2,
            "gradient_steps": gradient_steps,
            "policy_delay": policy_delay,
            "learning_rate": 0.01,
            "learning_rate_scheduler": optax.constant_schedule,
            "learning_rate_scheduler_kwargs": {"value": 0.5},
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    data = {name: np.ones_like(tensor[0]) for name, tensor in memory.tensors.items()}
    data["terminated"].fill(False)
    for _ in range(4):
        memory.add_samples(**data)

    for timestep in range(4):
        agent.update(timestep=timestep, timesteps=4)
        critic_steps = (timestep + 1) * gradient_steps
        actor_steps = critic_steps // policy_delay
        assert int(agent.policy_optimizer.state.count) == actor_steps
        assert int(agent.critic_1_optimizer.state.count) == critic_steps
        assert int(agent.critic_2_optimizer.state.count) == critic_steps
        assert float(agent.policy_learning_rate) == pytest.approx(0.01 * 0.5**actor_steps)
        assert float(agent.critic_learning_rate) == pytest.approx(0.01 * 0.5**critic_steps)
