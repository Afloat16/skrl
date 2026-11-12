import pytest

from itertools import combinations
from gymnasium import spaces

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.utils.model_instantiators.jax import multicategorical_model


@pytest.mark.parametrize("heads", [[2], [2, 2], [3, 3], [2, 3, 2]])
@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("compiled", [False, True])
def test_multicategorical_heads_sample_the_factorized_joint_distribution(heads, seed, compiled, monkeypatch):
    monkeypatch.setattr(config.jax, "key", jax.random.PRNGKey(seed))
    model = multicategorical_model(
        observation_space=spaces.Box(-1, 1, (2,)),
        action_space=spaces.MultiDiscrete(heads),
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [5], "activations": "elu"}],
        output="ACTIONS",
    )
    model.init_state_dict()

    def sample(observations):
        return model.act({"observations": observations})

    evaluate = jax.jit(sample) if compiled else sample
    batch_size = 8192
    actions, outputs = evaluate(jnp.zeros((batch_size, 2)))
    actions = np.asarray(actions)
    np.testing.assert_allclose(outputs["net_output"], 0, atol=0)
    np.testing.assert_allclose(outputs["log_prob"], -np.log(heads).sum(), atol=1e-6)
    for index, size in enumerate(heads):
        probabilities = np.bincount(actions[:, index], minlength=size) / batch_size
        np.testing.assert_allclose(probabilities, np.full(size, 1 / size), atol=0.03)
    for left, right in combinations(range(len(heads)), 2):
        joint = (
            np.bincount(
                actions[:, left] * heads[right] + actions[:, right], minlength=heads[left] * heads[right]
            ).reshape(heads[left], heads[right])
            / batch_size
        )
        expected = np.full_like(joint, 1 / (heads[left] * heads[right]))
        np.testing.assert_allclose(joint, expected, atol=0.03)
