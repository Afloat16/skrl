import pytest

from datetime import timedelta
from pathlib import Path
from packaging import version

import torch
import torch.multiprocessing as mp

from skrl import config
from skrl.agents.torch.ddpg import DDPG, DDPG_RNN
from skrl.agents.torch.sac import SAC, SAC_RNN
from skrl.agents.torch.td3 import TD3, TD3_RNN
from tests.agents.torch.test_frozen_actor import make_agent


def run_frozen_actor_updates(rank: int, world_size: int, rendezvous: str, agent_class: type) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    config.torch._world_size = world_size
    config.torch._rank = rank
    config.torch._is_distributed = True
    try:
        precisions = [False, True] if version.parse(torch.__version__) >= version.parse("2.4") else [False]
        for mixed_precision in precisions:
            agent = make_agent(agent_class, mixed_precision, "all", share_encoder=True)
            agent.memory.get_tensor_by_name("rewards").add_(0.3 * rank)
            policy_before = {name: value.clone() for name, value in agent.policy.state_dict().items()}
            critics = [model for name, model in agent.models.items() if name.startswith("critic")]
            before = torch.cat([parameter.detach().flatten() for model in critics for parameter in model.parameters()])
            for timestep in range(3):
                agent.update(timestep=timestep, timesteps=3)
                for name, value in agent.policy.state_dict().items():
                    torch.testing.assert_close(value, policy_before[name], rtol=0, atol=0)
                assert all(parameter.grad is None for parameter in agent.policy.parameters())
                values = torch.cat(
                    [parameter.detach().flatten() for model in critics for parameter in model.parameters()]
                )
                gathered = [torch.empty_like(values) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, values)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
                assert torch.isfinite(values).all()
                assert not torch.equal(values, before)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo is required"
)
@pytest.mark.parametrize("agent_class", [DDPG, DDPG_RNN, SAC, SAC_RNN, TD3, TD3_RNN])
def test_two_workers_can_train_critics_with_frozen_actors(agent_class: type, tmp_path: Path) -> None:
    mp.spawn(
        run_frozen_actor_updates,
        args=(2, (tmp_path / "rendezvous").as_uri(), agent_class),
        nprocs=2,
        join=True,
    )
