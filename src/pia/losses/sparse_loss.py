"""
Pérdida compuesta: entropía cruzada, L1 sobre pesos y L1 sobre activaciones.

La penalización de activaciones suma, por cada capa monitoreada, la media de
los valores absolutos de los tensores de activación (media sobre todos los
elementos de esa capa en el batch). Esas medias se suman entre capas. El
coeficiente ``gamma`` escala ese escalar; ajústalo junto con ``lambda`` porque
la suma L1 de pesos usa la suma global de ``|w|``, no la media.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class SparseLossBreakdown:
    """Desglose de la pérdida total (tensores escalares 0-D con grafo)."""

    task: Tensor
    weight_l1: Tensor
    activation_l1: Tensor
    total: Tensor


def _weight_l1_sum(model: nn.Module) -> Tensor:
    """Suma L1 de todos los parámetros con ``requires_grad``."""
    partes: list[Tensor] = [
        p.abs().sum() for p in model.parameters() if p.requires_grad
    ]
    if not partes:
        msg = "No hay parámetros entrenables para la penalización L1."
        raise ValueError(msg)
    return sum(partes[1:], start=partes[0])


def _activation_l1_sum_of_means(activations: dict[str, Tensor]) -> Tensor:
    """
    Suma de medias de |activación| por capa (orden estable por nombre de capa).
    """
    if not activations:
        msg = "Se esperaba un diccionario no vacío de activaciones."
        raise ValueError(msg)
    ordenados = [activations[k] for k in sorted(activations)]
    medias = [t.abs().mean() for t in ordenados]
    return sum(medias[1:], start=medias[0])


class SparseLoss(nn.Module):
    """
    Combina pérdida de tarea (CE), L1 global sobre pesos y L1 sobre activaciones.

    Args:
        lambda_weight: Coeficiente ``λ`` de la suma L1 de pesos.
        gamma_activation: Coeficiente ``γ`` de la penalización de activaciones.
    """

    def __init__(self, lambda_weight: float, gamma_activation: float) -> None:
        super().__init__()
        self.lambda_weight = float(lambda_weight)
        self.gamma_activation = float(gamma_activation)
        self._ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        model: nn.Module,
        activations: dict[str, Tensor],
    ) -> tuple[Tensor, SparseLossBreakdown]:
        """
        Calcula la pérdida total y el desglose.

        Args:
            logits: Salida lineal del clasificador.
            targets: Etiquetas enteras clase por muestra.
            model: Red cuyos parámetros entrenables entran en la L1 de pesos.
            activations: Mapa capa → tensor de activación con grafo (hooks).

        Returns:
            Tupla ``(total, breakdown)``.
        """
        task = self._ce(logits, targets)
        w_l1 = _weight_l1_sum(model)
        if self.gamma_activation != 0.0:
            act_l1 = _activation_l1_sum_of_means(activations)
        else:
            act_l1 = logits.new_zeros(())
        total = task + self.lambda_weight * w_l1 + self.gamma_activation * act_l1
        desglose = SparseLossBreakdown(
            task=task,
            weight_l1=w_l1,
            activation_l1=act_l1,
            total=total,
        )
        return total, desglose
