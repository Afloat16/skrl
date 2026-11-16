from typing import Any, Generator

import pytest

import os
import random

import numpy as np
import torch

from skrl import config
from skrl.utils import set_seed


@pytest.fixture(autouse=True)
def restore_random_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None, Any, None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    for variable in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.setenv(variable, os.environ.get(variable, ""))
    for framework in (config.torch, config.jax, config.warp):
        monkeypatch.setattr(framework, "_key", framework._key)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


@pytest.mark.parametrize("modes", [(True, False), (False, True), (True, False, True, False)])
def test_seed_can_enable_and_disable_deterministic_algorithms(modes: tuple[bool, ...]) -> None:
    for enabled in modes:
        assert set_seed(17, deterministic=enabled) == 17
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.backends.cudnn.deterministic is enabled
        assert torch.backends.cudnn.benchmark is not enabled


def test_disabling_determinism_restores_normal_operation_support() -> None:
    set_seed(17, deterministic=True)
    set_seed(17, deterministic=False)
    result = torch.empty(2)
    result.put_(torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), accumulate=False)
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("deterministic", [False, True])
def test_seed_still_repeats_random_samples(deterministic: bool) -> None:
    set_seed(23, deterministic=deterministic)
    expected = random.random(), np.random.random(4), torch.rand(4)
    set_seed(23, deterministic=deterministic)
    actual = random.random(), np.random.random(4), torch.rand(4)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
