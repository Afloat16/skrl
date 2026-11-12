import pytest
import warnings

import gymnasium

import torch

from skrl.agents.torch.td3 import TD3, TD3_RNN
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model


@pytest.mark.parametrize("agent_type", [TD3, TD3_RNN])
@pytest.mark.parametrize("policy_delay", [1, 2, 3])
@pytest.mark.parametrize("gradient_steps", [1, 3])
def test_policy_scheduler_tracks_delayed_optimizer_steps(agent_type, policy_delay, gradient_steps):
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
    agent = agent_type(
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
            "learning_rate_scheduler": torch.optim.lr_scheduler.ExponentialLR,
            "learning_rate_scheduler_kwargs": {"gamma": 0.5},
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    data = {name: torch.ones_like(tensor[0]) for name, tensor in memory.tensors.items()}
    data["terminated"].zero_()
    if "truncated" in data:
        data["truncated"].zero_()
    for _ in range(4):
        memory.add_samples(**data)

    actor_steps = []
    agent.policy_optimizer.register_step_post_hook(lambda *args: actor_steps.append(1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for timestep in range(4):
            agent.update(timestep=timestep, timesteps=4)
            critic_steps = (timestep + 1) * gradient_steps
            expected_actor_steps = critic_steps // policy_delay
            assert len(actor_steps) == expected_actor_steps
            assert agent.policy_scheduler.last_epoch == expected_actor_steps
            assert agent.critic_scheduler.last_epoch == critic_steps
            assert agent.policy_optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**expected_actor_steps)
            assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**critic_steps)
    assert not any("lr_scheduler.step()" in str(warning.message) for warning in caught)
