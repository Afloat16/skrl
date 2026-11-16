from typing import Any

import pytest

import copy
from pathlib import Path
import gymnasium

from skrl.agents.torch.ppo import PPO
from skrl.multi_agents.torch.ippo import IPPO
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model


def make_agent(multi_agent: bool, automatic: bool, directory: Path) -> PPO | IPPO:
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
        "experiment": {
            "directory": str(directory),
            "experiment_name": "initialization",
            "write_interval": "auto" if automatic else 7,
            "checkpoint_interval": "auto" if automatic else 11,
        }
    }
    if multi_agent:
        names = ["first", "second"]
        return IPPO(
            possible_agents=names,
            models={name: models() for name in names},
            observation_spaces={name: observations for name in names},
            state_spaces={name: None for name in names},
            action_spaces={name: actions for name in names},
            device="cpu",
            cfg=cfg,
        )
    return PPO(models=models(), observation_space=observations, action_space=actions, device="cpu", cfg=cfg)


@pytest.mark.parametrize("multi_agent", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("configuration_kind", ["dict", "dataclass", "none"])
def test_initialization_accepts_supported_trainer_configurations(
    multi_agent: bool, automatic: bool, configuration_kind: str, tmp_path: Path
) -> None:
    agent = make_agent(multi_agent, automatic, tmp_path)
    if configuration_kind == "dict":
        configuration = {"timesteps": 1000}
    elif configuration_kind == "dataclass":
        configuration = SequentialTrainerCfg(timesteps=1000)
    else:
        configuration = None
    original = copy.deepcopy(configuration)
    try:
        agent.init(trainer_cfg=configuration)
        horizon = 0 if configuration is None else 1000
        assert agent.write_interval == (horizon // 100 if automatic else 7)
        assert agent.checkpoint_interval == (horizon // 10 if automatic else 11)
        assert configuration == original
        if agent.checkpoint_interval:
            assert (Path(agent.experiment_dir) / "checkpoints").is_dir()
    finally:
        if hasattr(agent, "writer"):
            agent.writer.close()
