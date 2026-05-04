"""
Paquete principal: modelos para entrenamiento con restricciones de dispersión.
"""

from pia.models import (
    MONITORED_MODULE_NAMES,
    apply_he_init,
    build_resnet18_cifar,
    get_monitored_modules,
)

__all__ = [
    "MONITORED_MODULE_NAMES",
    "apply_he_init",
    "build_resnet18_cifar",
    "get_monitored_modules",
]
