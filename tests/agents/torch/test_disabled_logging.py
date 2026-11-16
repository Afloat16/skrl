from typing import Any

import pytest

from pathlib import Path
import gymnasium
from packaging import version

import torch

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.ippo import IPPO
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


def make_agent(
    multi_agent: bool, write_interval: int, checkpoint_interval: int, store_separately: bool, directory: Path
) -> PPO | IPPO:
    observations = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    actions = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,))

    def models() -> dict[str, Any]:
        options = {
            "observation_space": observations,
            "action_space": actions,
            "device": "cpu",
            "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
        }
        result = {
            "policy": gaussian_model(**options, output="ACTIONS"),
            "value": deterministic_model(**options, output="ONE"),
        }
        for model in result.values():
            model.init_state_dict()
        return result

    cfg = {
        "rollouts": 1024,
        "experiment": {
            "directory": str(directory),
            "experiment_name": "logging",
            "write_interval": write_interval,
            "checkpoint_interval": checkpoint_interval,
            "store_separately": store_separately,
        },
    }
    if multi_agent:
        names = ["first", "second"]
        agent = IPPO(
            possible_agents=names,
            models={name: models() for name in names},
            memories={name: RandomMemory(memory_size=4, device="cpu") for name in names},
            observation_spaces={name: observations for name in names},
            state_spaces={name: None for name in names},
            action_spaces={name: actions for name in names},
            device="cpu",
            cfg=cfg,
        )
    else:
        agent = PPO(
            models=models(),
            memory=RandomMemory(memory_size=4, device="cpu"),
            observation_space=observations,
            action_space=actions,
            device="cpu",
            cfg=cfg,
        )
    agent.init()
    agent.enable_training_mode()
    return agent


def record_step(agent: PPO | IPPO, timestep: int, reward: float, finished: bool) -> None:
    def inputs(value: Any) -> Any:
        return {name: value for name in agent.possible_agents} if isinstance(agent, IPPO) else value

    observations = inputs(torch.tensor([[0.1, -0.2]]))
    states = inputs(None)
    with torch.no_grad():
        actions, _ = agent.act(observations, states, timestep=timestep, timesteps=1024)
        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=inputs(torch.tensor([[reward]])),
            next_observations=observations,
            next_states=states,
            terminated=inputs(torch.tensor([[finished]])),
            truncated=inputs(torch.zeros((1, 1), dtype=torch.bool)),
            infos={},
            timestep=timestep,
            timesteps=1024,
        )
    agent.track_data("Stats / Environment time (ms)", float(timestep))
    agent.post_interaction(timestep=timestep, timesteps=1024)


@pytest.mark.parametrize("multi_agent", [False, True])
@pytest.mark.parametrize("store_separately", [False, True])
@pytest.mark.parametrize("write_interval", [-1, 0, 2])
def test_best_checkpoint_does_not_require_tensorboard(
    multi_agent: bool, store_separately: bool, write_interval: int, tmp_path: Path
) -> None:
    agent = make_agent(multi_agent, write_interval, 2, store_separately, tmp_path)
    policy = agent.models["first"]["policy"] if multi_agent else agent.models["policy"]
    expected = None
    try:
        for timestep, reward in enumerate([3.0, 3.0, 0.0, 0.0, -3.0, -3.0]):
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.fill_(0.1 * (1 + timestep // 2))
            if timestep == 1:
                expected = {name: value.clone() for name, value in policy.state_dict().items()}
            record_step(agent, timestep, reward, finished=bool(timestep % 2))

        filename = "best_first_policy.pt" if multi_agent else "best_policy.pt"
        checkpoint = Path(agent.experiment_dir) / "checkpoints" / (filename if store_separately else "best_agent.pt")
        assert checkpoint.is_file()
        options = {"weights_only": True} if version.parse(torch.__version__) >= version.parse("1.13") else {}
        saved = torch.load(checkpoint, map_location="cpu", **options)
        if not store_separately:
            saved = saved["first"]["policy"] if multi_agent else saved["policy"]
        for name, value in expected.items():
            torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
        assert agent.checkpoint_best_modules["reward"] == pytest.approx(12.0 if multi_agent else 6.0)
        assert agent.checkpoint_best_modules["timestep"] == 2
        if write_interval <= 0:
            assert not agent.tracking_data
            assert not hasattr(agent, "writer")
    finally:
        if hasattr(agent, "writer"):
            agent.writer.close()


@pytest.mark.parametrize("multi_agent", [False, True])
@pytest.mark.parametrize("write_interval", [-1, 0])
@pytest.mark.parametrize("checkpoint_interval", [0, 1000])
def test_disabled_logging_does_not_accumulate_per_step_metrics(
    multi_agent: bool, write_interval: int, checkpoint_interval: int, tmp_path: Path
) -> None:
    agent = make_agent(multi_agent, write_interval, checkpoint_interval, False, tmp_path)
    for timestep in range(105):
        record_step(agent, timestep, 1.0, finished=True)
    assert not agent.tracking_data
    assert len(agent._track_rewards) == (100 if checkpoint_interval else 0)
    assert not hasattr(agent, "writer")
