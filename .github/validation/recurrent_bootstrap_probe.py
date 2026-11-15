import json

from tests.agents.torch.test_recurrent_time_limit_bootstrap import _rollout


results = []
for shared, asymmetric in [(False, False), (False, True), (True, False)]:
    for reward in [100, 1000]:
        _, memory, policy_states, _ = _rollout(
            "PPO_RNN", shared, asymmetric, next_episode_reward=reward, update=True
        )
        results.append({
            "shared": shared,
            "asymmetric": asymmetric,
            "next_episode_reward": reward,
            "rewards": memory.get_tensor_by_name("rewards").tolist(),
            "values": memory.get_tensor_by_name("values").tolist(),
            "returns": memory.get_tensor_by_name("returns").tolist(),
            "policy_states": [value.tolist() for value in policy_states],
        })
print("RECURRENT_BOOTSTRAP_PROBE=" + json.dumps(results), flush=True)
