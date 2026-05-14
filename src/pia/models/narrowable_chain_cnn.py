"""
CNN en cadena de dos convoluciones con ancho compartido.

Alineada entre capas para poda estructurada. Expone ``layer2`` y ``layer3``
como identidades para compatibilidad con ``ActivationCapture`` en el bucle de
entrenamiento.
"""

from __future__ import annotations

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
