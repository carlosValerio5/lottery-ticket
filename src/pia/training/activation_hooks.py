"""
Registro de hooks para capturar activaciones de submódulos monitoreados.

Se usa un diccionario reutilizable por paso de optimización: llamar a
``clear`` antes del forward y leer ``current`` después.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from pia.models.resnet_cifar import get_monitored_modules


class ActivationCapture:
    """
    Instala forward hooks en las capas devueltas por ``get_monitored_modules``.

    Args:
        model: Red compatible con la API de modularidad de Fase 1.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._handles: list[Any] = []
        self._current: dict[str, Tensor] = {}
        self._register()

    def _register(self) -> None:
        mods = get_monitored_modules(self._model)
        for nombre, modulo in mods.items():

            def _hook(
                _m: nn.Module,
                _inp: Any,
                salida: Tensor,
                n: str = nombre,
            ) -> None:
                self._current[n] = salida

            h = modulo.register_forward_hook(_hook)
            self._handles.append(h)

    def clear(self) -> None:
        """Vacía el caché antes de un nuevo forward."""
        self._current.clear()

    def current(self) -> dict[str, Tensor]:
        """Copia superficial del mapa nombre → tensor (referencias con grafo)."""
        return dict(self._current)

    def remove(self) -> None:
        """Elimina hooks; llamar al finalizar el entrenamiento."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._current.clear()
