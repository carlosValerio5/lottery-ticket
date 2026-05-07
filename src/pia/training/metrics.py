"""
Métricas de evaluación y estadísticas de dispersidad de pesos.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def accuracy_top1(logits: Tensor, targets: Tensor) -> float:
    """Accuracy top-1 en [0, 1]."""
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item())


def weight_sparsity_ratio(model: nn.Module, *, eps: float = 1e-3) -> float:
    """
    Fracción de parámetros con |w| < eps (aproximación de sparsidad).

    Args:
        model: Red con parámetros en ``model.parameters()``.
        eps: Umbral absoluto.

    Returns:
        Escalar en [0, 1].
    """
    partes: list[Tensor] = [p.detach().abs().flatten() for p in model.parameters()]
    if not partes:
        return 0.0
    flat = torch.cat(partes)
    return float((flat < eps).float().mean().item())
