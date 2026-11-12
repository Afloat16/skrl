import pytest

import importlib
from gymnasium import spaces

import torch
from torch.nn.utils import parameters_to_vector

from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


@pytest.mark.parametrize("variant", ["trpo", "trpo_rnn"])
@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("step_fraction,accept_ratio", [(0.5, 0.99), (1.0, 0.99), (1.0, 0.5)])
def test_native_trpo_backtracking_satisfies_each_candidate_improvement_ratio(
    variant, seed, step_fraction, accept_ratio, monkeypatch
):
    torch.manual_seed(seed)
    module = importlib.import_module(f"skrl.agents.torch.trpo.{variant}")
    agent_type = getattr(module, variant.upper())
    observation_space, action_space = spaces.Box(-1, 1, (2,)), spaces.Box(-1, 1, (1,))
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}]
    policy = gaussian_model(
        observation_space=observation_space, action_space=action_space, device="cpu", network=network, output="ACTIONS"
    )
    value = deterministic_model(
        observation_space=observation_space, action_space=action_space, device="cpu", network=network, output="ONE"
    )
    policy.init_state_dict()
    value.init_state_dict()
    memory = RandomMemory(memory_size=4, num_envs=1, device="cpu")
    agent = agent_type(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "rollouts": 4,
            "learning_epochs": 1,
            "mini_batches": 1,
            "max_kl_divergence": 1.0,
            "step_fraction": step_fraction,
            "accept_ratio": accept_ratio,
            "max_backtrack_steps": 6,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode(True)
    initial = parameters_to_vector(policy.parameters()).detach().clone()
    losses, gradients, candidates, divergences = [], [], [], []
    original_loss, original_kl = module.surrogate_loss, module.kl_divergence

    def observe_loss(**kwargs):
        result = original_loss(**kwargs)
        if not losses:
            gradient = torch.autograd.grad(result, policy.parameters(), retain_graph=True)
            gradients.append(torch.cat([g.reshape(-1) for g in gradient]).detach())
        else:
            candidates.append(parameters_to_vector(policy.parameters()).detach().clone())
        losses.append(result.detach().item())
        return result

    def observe_kl(**kwargs):
        result = original_kl(**kwargs)
        if kwargs["policy_1"] is not kwargs["policy_2"]:
            divergences.append(result.detach().item())
        return result

    monkeypatch.setattr(module, "surrogate_loss", observe_loss)
    monkeypatch.setattr(module, "kl_divergence", observe_kl)
    for timestep, reward in enumerate([1.0, -1.0, 2.0, -2.0]):
        observations = torch.tensor([[timestep / 4, 1 - timestep / 4]])
        with torch.no_grad():
            actions, _ = agent.act(observations, None, timestep=timestep, timesteps=4)
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.tensor([[reward]]),
                next_observations=observations,
                next_states=None,
                terminated=torch.tensor([[timestep == 3]]),
                truncated=torch.zeros(1, 1, dtype=torch.bool),
                infos={},
                timestep=timestep,
                timesteps=4,
            )
        agent.post_interaction(timestep=timestep, timesteps=4)

    assert candidates
    full_step = (candidates[0] - initial) / step_fraction
    full_expected_improvement = torch.dot(gradients[0], full_step).item()
    assert full_expected_improvement > 0
    expected = initial
    ratios = []
    for index, (parameters, divergence, loss) in enumerate(zip(candidates, divergences, losses[1:])):
        alpha = step_fraction * 0.5**index
        ratio = (loss - losses[0]) / (alpha * full_expected_improvement)
        ratios.append(ratio)
        if divergence < agent.cfg.max_kl_divergence and ratio > accept_ratio:
            expected = parameters
            break
    torch.testing.assert_close(
        parameters_to_vector(policy.parameters()),
        expected,
        msg=f"Backtracking accepted a candidate below its improvement threshold: {ratios}",
    )
