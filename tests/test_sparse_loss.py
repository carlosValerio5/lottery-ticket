"""Pruebas de ``SparseLoss`` y desglose."""

from __future__ import annotations

import torch
from torch import nn

from pia.losses.sparse_loss import SparseLoss, SparseLossBreakdown


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_sparse_loss_decomposition_matches_total() -> None:
    modelo = _TinyModel()
    criterio = SparseLoss(lambda_weight=0.01, gamma_activation=0.02)
    logits = torch.randn(5, 3)
    y = torch.randint(0, 3, (5,))
    acts = {"a": torch.randn(5, 2, requires_grad=True)}
    total, br = criterio(logits, y, modelo, acts)
    esperado = br.task + 0.01 * br.weight_l1 + 0.02 * br.activation_l1
    assert torch.allclose(total, esperado)
    assert isinstance(br, SparseLossBreakdown)


def test_gamma_zero_ignores_empty_activations() -> None:
    modelo = _TinyModel()
    criterio = SparseLoss(lambda_weight=0.0, gamma_activation=0.0)
    logits = torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    total, br = criterio(logits, y, modelo, {})
    assert torch.allclose(total, br.task)
    assert float(br.activation_l1.item()) == 0.0


def test_weight_l1_mean_matches_global_mean() -> None:
    modelo = _TinyModel()
    criterio = SparseLoss(
        lambda_weight=1.0,
        gamma_activation=0.0,
        weight_l1_aggregation="mean",
    )
    logits = torch.randn(2, 3)
    y = torch.randint(0, 3, (2,))
    _, br = criterio(logits, y, modelo, {})
    w = modelo.lin.weight.detach().abs().flatten()
    b = modelo.lin.bias.detach().abs().flatten()
    flat = torch.cat([w, b])
    manual = flat.mean()
    assert torch.allclose(br.weight_l1, manual)


def test_backward_sparse_loss() -> None:
    modelo = _TinyModel()
    criterio = SparseLoss(lambda_weight=1e-3, gamma_activation=1e-3)
    x = torch.randn(3, 4)
    logits = modelo(x)
    y = torch.randint(0, 3, (3,))
    acts = {"layer2": torch.randn(3, 2, requires_grad=True)}
    loss, _ = criterio(logits, y, modelo, acts)
    loss.backward()
    assert modelo.lin.weight.grad is not None
