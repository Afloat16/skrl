import pytest

import importlib
import gymnasium

import numpy as np
import torch

from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler


_CASES = [
    (name, shared, asymmetric)
    for name in ["A2C_RNN", "PPO_RNN", "RPO_RNN", "TRPO_RNN"]
    for shared, asymmetric in [(False, False), (False, True), (True, False)]
    if not (name == "TRPO_RNN" and shared)
]


class _LinearRecurrent(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, *, use_states=False, shared=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )
        GaussianMixin.__init__(self)
        DeterministicMixin.__init__(self)
        self.use_states = use_states
        self.shared = shared
        self.shared_cache = None
        # Positive inputs make this genuine RNN exactly h_next = h + observation.
        self.rnn = torch.nn.RNN(1, 1, nonlinearity="relu", bias=False, batch_first=True)
        self.head = torch.nn.Linear(1, 1, bias=False)
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(1))
        for parameter in self.rnn.parameters():
            torch.nn.init.ones_(parameter)
        torch.nn.init.ones_(self.head.weight)

    def get_specification(self):
        return {"rnn": {"sequence_length": 1, "sizes": [(1, 2, 1)]}}

    def act(self, inputs, *, role=""):
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        return GaussianMixin.act(self, inputs, role=role)

    def compute(self, inputs, role):
        if self.shared and role == "value" and self.shared_cache is not None:
            features, hidden = self.shared_cache
            self.shared_cache = None
        else:
            observations = inputs["states"] if self.use_states and role == "value" else inputs["observations"]
            features, hidden = self.rnn(observations.unsqueeze(1), inputs["rnn"][0].contiguous())
            features = features[:, 0]
            if self.shared and role == "policy":
                self.shared_cache = (features, hidden)
        output = self.head(features)
        if role == "policy":
            return 0.1 * output, {"log_std": self.log_std_parameter, "rnn": [hidden]}
        return output, {"rnn": [hidden]}


class _ObservationEnv(gymnasium.Env):
    observation_space = gymnasium.spaces.Box(-1000, 1000, shape=(1,))
    action_space = gymnasium.spaces.Box(-100, 100, shape=(1,))

    def __init__(self, index, next_episode_reward):
        self.index = index
        self.next_episode_reward = next_episode_reward
        self.episode = -1
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        observation = (1 if self.episode == 0 else 7) if self.index == 0 else 2
        return np.array([observation], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        if self.index == 0:
            observation = 3 if self.episode == 0 else 9
            reward = 1 if self.episode == 0 else self.next_episode_reward
        else:
            observation = 4 if self.steps == 1 else 5
            reward = 2 if self.steps == 1 else 200
        return np.array([observation], dtype=np.float32), float(reward), False, False, {}


def _rollout(name, shared, asymmetric, *, bootstrap=True, next_episode_reward=100, update=False, normalizers=False):
    torch.manual_seed(19)
    envs = [
        gymnasium.wrappers.TimeLimit(_ObservationEnv(index, next_episode_reward), max_episode_steps=limit)
        for index, limit in [(0, 1), (1, 3)]
    ]
    observation_space = envs[0].observation_space
    action_space = envs[0].action_space
    state_space = observation_space if asymmetric else None
    policy = _LinearRecurrent(observation_space, state_space, action_space, shared=shared)
    value = policy if shared else _LinearRecurrent(observation_space, state_space, action_space, use_states=asymmetric)
    memory = RandomMemory(memory_size=2, num_envs=2, device="cpu")
    cfg = {
        "rollouts": 2,
        "mini_batches": 1,
        "discount_factor": 0.5,
        "gae_lambda": 0.9,
        "time_limit_bootstrap": bootstrap,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    if name != "A2C_RNN":
        cfg["learning_epochs"] = 1
    if name == "RPO_RNN":
        cfg["alpha"] = 0
    if normalizers:
        for key in ["observation", "value"] + (["state"] if asymmetric else []):
            cfg[f"{key}_preprocessor"] = RunningStandardScaler
            cfg[f"{key}_preprocessor_kwargs"] = {"size": 1, "device": "cpu"}
    agent_class = getattr(importlib.import_module(f"skrl.agents.torch.{name.split('_')[0].lower()}"), name)
    agent = agent_class(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode(True)
    normalizer_states = {
        key: {name: tensor.clone() for name, tensor in module.state_dict().items()}
        for key, module in agent.checkpoint_modules.items()
        if key.endswith("_preprocessor")
    }
    observations = torch.tensor(np.stack([env.reset()[0] for env in envs]))
    policy_states = []
    for timestep in range(2):
        states = 10 * observations if asymmetric else None
        with torch.no_grad():
            actions, outputs = agent.act(observations, states, timestep=timestep, timesteps=2)
            policy_states.append(outputs["rnn"][0].clone())
            transitions = [env.step(action.numpy()) for env, action in zip(envs, actions)]
            next_observations = torch.tensor(np.stack([transition[0] for transition in transitions]))
            terminated = torch.tensor([[transition[2]] for transition in transitions])
            truncated = torch.tensor([[transition[3]] for transition in transitions])
            assert not terminated.any()
            assert truncated[:, 0].tolist() == [True, False]
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=torch.tensor([[transition[1]] for transition in transitions]),
                next_observations=next_observations,
                next_states=10 * next_observations if asymmetric else None,
                terminated=terminated,
                truncated=truncated,
                infos={},
                timestep=timestep,
                timesteps=2,
            )
        if timestep == 0 or update:
            agent.post_interaction(timestep=timestep, timesteps=2)
        observations = torch.tensor(
            np.stack(
                [
                    env.reset()[0] if transition[2] or transition[3] else transition[0]
                    for env, transition in zip(envs, transitions)
                ]
            )
        )
    for env in envs:
        env.close()
    return agent, memory, policy_states, normalizer_states


@pytest.mark.parametrize("name,shared,asymmetric", _CASES)
@pytest.mark.parametrize("bootstrap", [False, True])
def test_recurrent_bootstrap_uses_continued_critic_state(name, shared, asymmetric, bootstrap):
    agent, memory, policy_states, _ = _rollout(name, shared, asymmetric, bootstrap=bootstrap)
    factor = 10 if asymmetric else 1
    # Extra value look-ahead must not advance the real rollout a second time.
    torch.testing.assert_close(policy_states[0], torch.tensor([[[1.0], [2.0]]]))
    torch.testing.assert_close(policy_states[1], torch.tensor([[[7.0], [6.0]]]))
    torch.testing.assert_close(
        memory.get_tensor_by_name("values"),
        factor * torch.tensor([[[1.0], [2.0]], [[7.0], [6.0]]]),
    )
    expected = torch.tensor([[[1.0], [2.0]], [[100.0], [200.0]]])
    if bootstrap:
        expected[:, 0, 0] += factor * torch.tensor([2.0, 8.0])
    torch.testing.assert_close(memory.get_tensor_by_name("rewards"), expected)
    if shared:
        assert agent.policy.shared_cache is None


@pytest.mark.parametrize("name,shared,asymmetric", _CASES)
@pytest.mark.parametrize("next_episode_reward", [100, 1000])
def test_recurrent_returns_stop_at_truncation(name, shared, asymmetric, next_episode_reward):
    agent, memory, _, _ = _rollout(name, shared, asymmetric, next_episode_reward=next_episode_reward, update=True)
    factor = 10 if asymmetric else 1
    # The first episode's return is independent of the reward after its reset.
    # The continuing environment still includes its next value and GAE trace.
    expected = torch.tensor(
        [
            [[1.0 + 2 * factor], [92.0 + 2.775 * factor]],
            [[next_episode_reward + 8.0 * factor], [200.0 + 5.5 * factor]],
        ]
    )
    torch.testing.assert_close(memory.get_tensor_by_name("returns"), expected)
    assert all(torch.isfinite(parameter).all() for model in agent.models.values() for parameter in model.parameters())


@pytest.mark.parametrize("shared,asymmetric", [(False, False), (False, True), (True, False)])
def test_recurrent_look_ahead_preserves_normalizer_statistics(shared, asymmetric):
    agent, _, _, before = _rollout("PPO_RNN", shared, asymmetric, normalizers=True)
    assert before
    for key, state_dict in before.items():
        after = agent.checkpoint_modules[key].state_dict()
        for name, tensor in state_dict.items():
            torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)
    if shared:
        assert agent.policy.shared_cache is None
