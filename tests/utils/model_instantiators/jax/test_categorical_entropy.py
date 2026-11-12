import pytest

import importlib
from gymnasium import spaces

import jax
import jax.numpy as jnp
import numpy as np
import optax

from skrl.models.jax import CategoricalMixin, Model, MultiCategoricalMixin
from skrl.utils.model_instantiators.jax import categorical_model, gaussian_model, multicategorical_model


def _identity_policy(heads, unnormalized, reduction):
    mixin = CategoricalMixin if len(heads) == 1 else MultiCategoricalMixin

    class IdentityPolicy(mixin, Model):
        def __init__(self, **kwargs):
            kwargs.setdefault("observation_space", spaces.Box(-100, 100, (sum(heads),)))
            kwargs.setdefault(
                "action_space", spaces.Discrete(heads[0]) if len(heads) == 1 else spaces.MultiDiscrete(heads)
            )
            kwargs.setdefault("device", "cpu")
            Model.__init__(self, **kwargs)
            options = {} if len(heads) == 1 else {"reduction": reduction}
            mixin.__init__(self, unnormalized_log_prob=unnormalized, **options)

        def __call__(self, inputs, role=""):
            return inputs["observations"], {}

    model = IdentityPolicy()
    model.init_state_dict(inputs={"observations": jnp.ones((2, sum(heads)))})
    return model


def _reference(values, heads, unnormalized, reduction):
    values = np.asarray(values, dtype=np.float64)
    entropies = []
    for head in np.split(values, np.cumsum(heads)[:-1], axis=-1):
        weights = np.exp(head - head.max(axis=-1, keepdims=True)) if unnormalized else head
        probabilities = weights / weights.sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0)
        entropies.append(-terms.sum(axis=-1))
    entropy = np.stack(entropies, axis=-1)
    return entropy if reduction == "none" else getattr(np, reduction)(entropy, axis=-1, keepdims=True)


@pytest.mark.parametrize(
    "heads,reduction", [([3], "sum"), ([2, 3, 2], "sum"), ([2, 3, 2], "mean"), ([2, 3, 2], "prod"), ([2, 3, 2], "none")]
)
@pytest.mark.parametrize("unnormalized", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_categorical_entropy_matches_probability_reference_and_gradient(heads, reduction, unnormalized, compiled):
    model = _identity_policy(heads, unnormalized, reduction)
    values = np.linspace(0.2, 1.7, 2 * sum(heads), dtype=np.float32).reshape(2, -1)

    def entropy(data):
        _, outputs = model.act({"observations": data})
        return model.get_entropy(outputs.get("entropy_input", outputs["stddev"]))

    evaluate = jax.jit(entropy) if compiled else entropy
    actual = evaluate(jnp.asarray(values))
    expected = _reference(values, heads, unnormalized, reduction)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
    gradient = jax.grad(lambda data: evaluate(data).sum())(jnp.asarray(values))
    expected_gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.astype(np.float64), values.astype(np.float64)
        plus[index] += 1e-5
        minus[index] -= 1e-5
        expected_gradient[index] = (
            _reference(plus, heads, unnormalized, reduction).sum()
            - _reference(minus, heads, unnormalized, reduction).sum()
        ) / 2e-5
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("heads,values", [([3], [[0.0, 0.3, 0.7]]), ([2, 3], [[1.0, 0.0, 0.0, 0.4, 0.6]])])
@pytest.mark.parametrize("compiled", [False, True])
def test_zero_probability_actions_have_finite_entropy(heads, values, compiled):
    model = _identity_policy(heads, False, "sum")

    def entropy(data):
        _, outputs = model.act({"observations": data})
        return model.get_entropy(outputs.get("entropy_input", outputs["stddev"]))

    evaluate = jax.jit(entropy) if compiled else entropy
    np.testing.assert_allclose(evaluate(jnp.asarray(values)), _reference(values, heads, False, "sum"), atol=1e-6)


@pytest.mark.parametrize(
    "algorithm",
    [
        "agents.jax.a2c.a2c",
        "agents.jax.ppo.ppo",
        "agents.jax.rpo.rpo",
        "multi_agents.jax.ippo.ippo",
        "multi_agents.jax.mappo.mappo",
    ],
)
@pytest.mark.parametrize("kind", ["categorical", "multicategorical", "gaussian"])
def test_jitted_agent_entropy_update_has_finite_loss_and_gradients(algorithm, kind):
    factory = {
        "categorical": categorical_model,
        "multicategorical": multicategorical_model,
        "gaussian": gaussian_model,
    }[kind]
    action_space = {
        "categorical": spaces.Discrete(3),
        "multicategorical": spaces.MultiDiscrete([2, 3]),
        "gaussian": spaces.Box(-1, 1, (2,)),
    }[kind]
    model = factory(
        observation_space=spaces.Box(-1, 1, (4,)),
        action_space=action_space,
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [5], "activations": "elu"}],
        output="ACTIONS",
    )
    model.init_state_dict(role="policy")
    observations = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16
    actions, outputs = model.act({"observations": observations})
    update = importlib.import_module("skrl." + algorithm)._update_policy
    arguments = dict(
        policy_act=model.act,
        policy_state_dict=model.state_dict,
        inputs={"observations": observations, "taken_actions": actions},
        sampled_log_prob=outputs["log_prob"],
        sampled_advantages=jnp.asarray([[1.0], [-1.0], [0.5], [-0.5]]),
        get_entropy=model.get_entropy,
        entropy_loss_scale=0.1,
    )
    if not algorithm.endswith("a2c.a2c"):
        arguments["ratio_clip"] = 0.2
    gradient, policy_loss, entropy_loss, kl, _ = update(**arguments)
    assert np.isfinite([policy_loss, entropy_loss, kl]).all()
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(gradient))
    optimizer = optax.sgd(1e-3)
    updates, _ = optimizer.update(gradient, optimizer.init(model.state_dict.params))
    updated = optax.apply_updates(model.state_dict.params, updates)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(updated))
