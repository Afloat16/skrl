from typing import Any, Callable

import pytest

import gymnasium

from skrl.trainers.torch import StepTrainer

from ...utilities import AgentMock, SingleAgentEnv


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("budget", [None, 20])
@pytest.mark.parametrize("n_agents", [1, 2])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_step_budget_reaches_all_agent_callbacks(
    mode: str, budget: int | None, n_agents: int, asymmetric: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    env = SingleAgentEnv(
        observation_space=space,
        state_space=space if asymmetric else None,
        action_space=space,
        num_envs=2,
        device="cpu",
        ml_framework="torch",
        probability=0.0,
    )
    agents = [
        AgentMock(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            num_envs=2 // n_agents,
            device="cpu",
            ml_framework="torch",
        )
        for _ in range(n_agents)
    ]
    names = ["pre_interaction", "act", "record_transition", "post_interaction"]
    calls = {id(agent): {name: [] for name in names} for agent in agents}
    base = AgentMock.__bases__[0]

    def recording(method_name: str) -> Callable:
        original = getattr(base, method_name)

        def call(agent: Any, *args: Any, **kwargs: Any) -> Any:
            calls[id(agent)][method_name].append((kwargs["timestep"], kwargs["timesteps"]))
            return original(agent, *args, **kwargs)

        return call

    for name in names:
        monkeypatch.setattr(base, name, recording(name))
    trainer = StepTrainer(
        env=env,
        agents=agents,
        scopes=[2 // n_agents] * n_agents,
        cfg={
            "timesteps": 100,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )

    getattr(trainer, mode)(timestep=7, timesteps=budget)

    expected_budget = 100 if budget is None else budget
    for agent in agents:
        for name in names:
            assert calls[id(agent)][name] == [(7, expected_budget)], name
