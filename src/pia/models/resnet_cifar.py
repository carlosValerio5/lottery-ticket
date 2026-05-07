"""
Construcción de ResNet-18 para CIFAR, inicialización He y API de modularidad.

Se centralizan aquí el stem adaptado a 32×32 y la inicialización Kaiming para
evitar duplicar lógica en scripts de entrenamiento (Fase 2) y mantener un único
punto de verdad ante cambios de arquitectura.
"""

from __future__ import annotations

from typing import Final, cast

from torch import nn
from torchvision.models import resnet18

MONITORED_MODULE_NAMES: Final[frozenset[str]] = frozenset({"layer2", "layer3"})


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """
    Instancia ResNet-18 sin pesos preentrenados, adaptada a imágenes pequeñas.

    Args:
        num_classes: Dimensión de la cabeza de clasificación (10 para CIFAR-10).

    Returns:
        Módulo listo para recibir tensores N×3×32×32 tras aplicar `apply_he_init`.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Los stubs tipan maxpool como MaxPool2d; en CIFAR se reemplaza por Identity.
    cast(nn.Module, model).maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def apply_he_init(module: nn.Module) -> None:
    """
    Aplica Kaiming He (normal) a convoluciones 2D y capas lineales.

    Las capas BatchNorm conservan su inicialización por defecto. Esta función
    existe para cumplir el criterio de Fase 1 de forma explícita e independiente
    de los detalles internos de torchvision.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_monitored_modules(model: nn.Module) -> dict[str, nn.Module]:
    """
    Resuelve los submódulos declarados para seguimiento en entrenamiento.

    Args:
        model: Red con atributos `layer2` y `layer3` (p. ej. salida de
            `build_resnet18_cifar`).

    Returns:
        Diccionario nombre → submódulo.

    Raises:
        AttributeError: Si falta algún nombre en `MONITORED_MODULE_NAMES`.
        TypeError: Si el atributo no es un `nn.Module`.
    """
    out: dict[str, nn.Module] = {}
    for name in sorted(MONITORED_MODULE_NAMES):
        if not hasattr(model, name):
            msg = f"El modelo carece del submódulo requerido '{name}'."
            raise AttributeError(msg)
        submodule = getattr(model, name)
        if not isinstance(submodule, nn.Module):
            msg = f"'{name}' no es un nn.Module."
            raise TypeError(msg)
        out[name] = submodule
    return out
