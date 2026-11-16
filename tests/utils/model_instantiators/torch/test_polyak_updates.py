from typing import Any

import pytest

import copy

import torch

from skrl.models.torch import DeterministicMixin, Model


class Network(DeterministicMixin, Model):
    def __init__(self, dtype: torch.dtype, encoder: torch.nn.Module | None = None) -> None:
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.encoder = torch.nn.Linear(3, 4, dtype=dtype) if encoder is None else encoder
        self.head = torch.nn.Linear(4, 2, dtype=dtype)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.head(self.encoder(inputs["observations"])), {}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("polyak", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("sharing", ["separate", "encoder", "self"])
def test_polyak_updates_match_snapshot_average_and_preserve_sources(
    dtype: torch.dtype, polyak: float, sharing: str
) -> None:
    torch.manual_seed(43)
    source = Network(dtype)
    target = source if sharing == "self" else Network(dtype, source.encoder if sharing == "encoder" else None)
    target_parameters = list(target.parameters())
    source_parameters = list(source.parameters())
    target_before = [parameter.detach().clone() for parameter in target_parameters]
    source_before = [parameter.detach().clone() for parameter in source_parameters]
    storage = [parameter.data_ptr() for parameter in target_parameters]
    for parameter in target_parameters:
        parameter.grad = torch.full_like(parameter, 0.25)

    target.update_parameters(source, polyak=polyak)

    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-6, torch.float64: 1e-14}[dtype]
    for parameter, source_parameter, before, source_value, pointer in zip(
        target_parameters, source_parameters, target_before, source_before, storage
    ):
        expected = ((1 - polyak) * before.double() + polyak * source_value.double()).to(dtype)
        torch.testing.assert_close(parameter, expected, rtol=tolerance, atol=tolerance)
        if parameter is source_parameter:
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        torch.testing.assert_close(source_parameter, source_value, rtol=0, atol=0)
        assert parameter.data_ptr() == pointer
        assert parameter.requires_grad
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25), rtol=0, atol=0)
    assert all(before is after for before, after in zip(target_parameters, target.parameters()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_soft_update_preserves_a_shared_encoder_after_source_optimization(dtype: torch.dtype) -> None:
    torch.manual_seed(53)
    source = Network(dtype)
    target = Network(dtype, source.encoder)
    inputs = {"observations": torch.randn(5, 3, dtype=dtype)}
    labels = torch.randn(5, 2, dtype=dtype)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        prediction, _ = source.act(inputs)
        torch.nn.functional.mse_loss(prediction, labels).backward()
        optimizer.step()

    reference = copy.deepcopy(target)
    reference.load_state_dict(
        {name: 0.8 * value + 0.2 * source.state_dict()[name] for name, value in target.state_dict().items()}
    )
    source_prediction = source.act(inputs)[0].detach().clone()
    encoder_before = {name: value.detach().clone() for name, value in source.encoder.state_dict().items()}

    target.update_parameters(source, polyak=0.2)

    assert target.encoder is source.encoder
    torch.testing.assert_close(target.act(inputs)[0], reference.act(inputs)[0])
    torch.testing.assert_close(source.act(inputs)[0], source_prediction, rtol=0, atol=0)
    for name, value in source.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[name], rtol=0, atol=0)
