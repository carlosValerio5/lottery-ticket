"""
Poda estructurada por recorte de ancho en una CNN en cadena sin atajos residuales.

Cada índice de canal acopla la salida de ``conv1``, la entrada/salida de ``conv2``
y el bloque de columnas correspondiente en ``fc``. La puntuación global usa la
norma L1 agregada por índice para elegir qué canales eliminar físicamente.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class NarrowableChainCnn(nn.Module):
    """
    CNN mínima compatible con ``ActivationCapture`` (``layer2``/``layer3``).

    Dos convoluciones 3×3 con el mismo ancho ``C``, aplanado y cabeza lineal.
    Los canales ``0 .. C-1`` están alineados entre capas para poder recortar
    ancho sin romper dimensiones.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        spatial_hw: tuple[int, int],
        num_classes: int,
    ) -> None:
        """
        Args:
            in_channels: Canales de entrada (p. ej. 3 para RGB).
            width: Ancho compartido de salida de ``conv1`` y de ``conv2``.
            spatial_hw: ``(H, W)`` del mapa antes del aplanado (tras convs stride 1).
            num_classes: Clases de salida de ``fc``.
        """
        super().__init__()
        if width < 1:
            msg = "width debe ser >= 1."
            raise ValueError(msg)
        h, w = spatial_hw
        if h < 1 or w < 1:
            msg = "spatial_hw debe ser positivo."
            raise ValueError(msg)
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.spatial_hw = (int(h), int(w))
        self.num_classes = int(num_classes)
        self.conv1 = nn.Conv2d(
            self.in_channels, self.width, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            self.width, self.width, kernel_size=3, padding=1, bias=False
        )
        flat = self.width * h * w
        self.fc = nn.Linear(flat, self.num_classes)
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Aplica conv → ReLU → conv → ReLU → aplanado → logits."""
        x = torch.relu(self.conv1(x))
        x = self.layer2(x)
        x = torch.relu(self.conv2(x))
        x = self.layer3(x)
        return self.fc(x.reshape(x.size(0), -1))

    def flat_dim(self) -> int:
        """Producto ``C * H * W`` antes de ``fc``."""
        h, w = self.spatial_hw
        return int(self.width * h * w)


@torch.no_grad()
def chain_cnn_channel_l1_scores(model: NarrowableChainCnn) -> Tensor:
    """
    Devuelve un vector de puntuación L1 por índice de canal acoplado.

    Args:
        model: Instancia ``NarrowableChainCnn`` entrenada.

    Returns:
        Tensor 1-D de longitud ``model.width`` con puntuaciones no negativas.
    """
    c = model.width
    h, w = model.spatial_hw
    hw = h * w
    w1 = model.conv1.weight
    w2 = model.conv2.weight
    fc_w = model.fc.weight
    scores = w1.abs().sum(dim=(1, 2, 3))
    scores = scores + w2.abs().sum(dim=(0, 2, 3))
    scores = scores + w2.abs().sum(dim=(1, 2, 3))
    fc_reshaped = fc_w.reshape(fc_w.shape[0], c, hw)
    scores = scores + fc_reshaped.abs().sum(dim=(0, 2))
    return scores


@torch.no_grad()
def _narrow_chain_tensors(
    *,
    w1: Tensor,
    w2: Tensor,
    fc_w: Tensor,
    fc_b: Tensor | None,
    keep: list[int],
    spatial_hw: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Recorta tensores según índices de canal conservados (orden preservado)."""
    h, w = spatial_hw
    hw = h * w
    idx = torch.tensor(keep, dtype=torch.long, device=w1.device)
    new_w1 = w1.index_select(0, idx).clone()
    new_w2 = w2.index_select(0, idx).index_select(1, idx).clone()
    col_parts: list[Tensor] = []
    for i in keep:
        lo = int(i) * hw
        hi = (int(i) + 1) * hw
        col_parts.append(fc_w[:, lo:hi])
    new_fc_w = torch.cat(col_parts, dim=1).clone()
    new_fc_b = fc_b.clone() if fc_b is not None else None
    return new_w1, new_w2, new_fc_w, new_fc_b


def narrow_chain_cnn_by_fraction(
    model: NarrowableChainCnn,
    prune_fraction: float,
) -> tuple[NarrowableChainCnn, dict[str, Any]]:
    """
    Elimina una fracción de los canales supervivientes con menor puntuación L1.

    Args:
        model: Red en cadena actual (se leen pesos; no se muta in-place).
        prune_fraction: Fracción en ``(0, 1)`` del ancho actual a intentar podar.

    Returns:
        Tupla ``(nuevo_modelo, info)`` con ``info`` serializable (umbrales, k).

    Raises:
        TypeError: Si ``model`` no es ``NarrowableChainCnn``.
        ValueError: Si ``prune_fraction`` no está en ``(0, 1)`` o el ancho caería
            por debajo de 1.
    """
    if not isinstance(model, NarrowableChainCnn):
        msg = "narrow_chain_cnn_by_fraction solo admite NarrowableChainCnn."
        raise TypeError(msg)
    if not 0.0 < prune_fraction < 1.0:
        msg = "prune_fraction debe estar en (0, 1)."
        raise ValueError(msg)
    c = model.width
    if c <= 1:
        msg = "No quedan canales podables (width<=1)."
        raise ValueError(msg)
    k = int(prune_fraction * c)
    if k <= 0:
        return model, {
            "target_prune_fraction": float(prune_fraction),
            "k": 0,
            "width_before": c,
            "width_after": c,
            "removed_indices": [],
        }
    if k >= c:
        k = c - 1
    scores = chain_cnn_channel_l1_scores(model)
    _, worst = torch.topk(scores, k, largest=False)
    remove_set = {int(worst[j].item()) for j in range(k)}
    keep = [i for i in range(c) if i not in remove_set]
    if len(keep) < 1:
        msg = "La poda dejaría width<1."
        raise ValueError(msg)
    new_w1, new_w2, new_fc_w, new_fc_b = _narrow_chain_tensors(
        w1=model.conv1.weight,
        w2=model.conv2.weight,
        fc_w=model.fc.weight,
        fc_b=model.fc.bias,
        keep=keep,
        spatial_hw=model.spatial_hw,
    )
    nuevo = NarrowableChainCnn(
        model.in_channels,
        len(keep),
        model.spatial_hw,
        model.num_classes,
    )
    with torch.no_grad():
        nuevo.conv1.weight.data.copy_(new_w1)
        nuevo.conv2.weight.data.copy_(new_w2)
        nuevo.fc.weight.data.copy_(new_fc_w)
        if new_fc_b is not None and nuevo.fc.bias is not None:
            nuevo.fc.bias.data.copy_(new_fc_b)
    worst_list = sorted(remove_set)
    threshold = float(
        max(scores[i].item() for i in worst_list) if worst_list else float("nan")
    )
    info: dict[str, Any] = {
        "target_prune_fraction": float(prune_fraction),
        "k": k,
        "width_before": c,
        "width_after": len(keep),
        "removed_indices": worst_list,
        "threshold": threshold if not math.isnan(threshold) else None,
    }
    return nuevo.to(model.conv1.weight.device), info


def count_parameters(module: nn.Module) -> int:
    """
    Cuenta parámetros entrenables del módulo.

    Args:
        module: Red o submódulo.

    Returns:
        Número total de elementos con ``requires_grad=True``.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
