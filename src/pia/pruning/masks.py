"""
Registro de máscaras booleanas por parámetro para poda no estructurada.

Mantiene la invariante ``w = m * w`` en los pesos rastreados y anula gradientes
en posiciones podadas para que el optimizador no reactive conexiones muertas.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class WeightMaskRegistry:
    """
    Máscaras 0/1 alineadas con tensores de parámetros seleccionados del modelo.

    Args:
        masks: Mapa ``nombre_completo`` → tensor booleano de la misma forma que
            el parámetro correspondiente.
    """

    def __init__(self, masks: dict[str, Tensor]) -> None:
        if not masks:
            msg = "Se requiere al menos una máscara."
            raise ValueError(msg)
        self._masks: dict[str, Tensor] = dict(masks)
        self._handles: list[Any] = []

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        selector: Callable[[nn.Module], list[tuple[str, nn.Parameter]]],
    ) -> WeightMaskRegistry:
        """
        Construye máscaras inicializadas a uno para los parámetros devueltos.

        Args:
            model: Red cuyos ``named_parameters`` deben contener los nombres.
            selector: Función que devuelve ``(nombre, parámetro)`` rastreables.

        Returns:
            Registro listo para ``apply_to_weights`` y poda posterior.
        """
        pares = selector(model)
        nombres_modelo = {n for n, _ in model.named_parameters()}
        mascaras: dict[str, Tensor] = {}
        for nombre, param in pares:
            if nombre not in nombres_modelo:
                msg = f"El nombre '{nombre}' no existe en model.named_parameters()."
                raise KeyError(msg)
            mascaras[nombre] = torch.ones_like(
                param, dtype=torch.bool, device=param.device
            )
        return cls(mascaras)

    def masks_dict(self) -> dict[str, Tensor]:
        """Copia superficial del mapa nombre → máscara (referencias a tensores)."""
        return self._masks

    def apply_to_weights(self, model: nn.Module) -> None:
        """
        Aplica ``param.data *= máscara`` in-place para cada parámetro rastreado.

        Args:
            model: Instancia que contiene los parámetros nombrados en el registro.
        """
        parametros = dict(model.named_parameters())
        for nombre, m in self._masks.items():
            if nombre not in parametros:
                msg = f"Falta el parámetro '{nombre}' en el modelo."
                raise KeyError(msg)
            p = parametros[nombre]
            p.data.mul_(m.to(dtype=p.dtype, device=p.device))

    def register_grad_hooks(self, model: nn.Module) -> None:
        """
        Registra ganchos que anulan gradientes donde la máscara es cero.

        Si ya existían ganchos, primero los elimina con ``remove_grad_hooks``.
        """
        self.remove_grad_hooks()
        parametros = dict(model.named_parameters())
        for nombre, m in self._masks.items():
            param = parametros[nombre]

            def _hook(grad: Tensor, mask: Tensor = m) -> Tensor:
                return grad * mask.to(dtype=grad.dtype, device=grad.device)

            self._handles.append(param.register_hook(_hook))

    def remove_grad_hooks(self) -> None:
        """Elimina los ganchos de gradiente registrados previamente."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def make_post_step_callback(self, model: nn.Module) -> Callable[[], None]:
        """
        Devuelve una función sin argumentos que reaplica máscaras a los pesos.

        Se invoca justo después de ``optimizer.step()`` para corregir deriva
        numérica o actualizaciones residuales en posiciones podadas.
        """

        def _cb() -> None:
            self.apply_to_weights(model)

        return _cb

    def total_elements(self) -> int:
        """Número total de elementos rastreados en todas las máscaras."""
        return int(sum(m.numel() for m in self._masks.values()))

    def current_sparsity(self) -> float:
        """
        Fracción de elementos con máscara cero respecto al total rastreado.

        Returns:
            Escalar en ``[0, 1]``.
        """
        total = self.total_elements()
        if total == 0:
            return 0.0
        ceros = int(sum((~m).sum().item() for m in self._masks.values()))
        return ceros / total

    def per_layer_sparsity(self) -> dict[str, float]:
        """
        Fracción de ceros por nombre de parámetro (solo capas rastreadas).

        Returns:
            Mapa ``nombre`` → fracción en ``[0, 1]``.
        """
        out: dict[str, float] = {}
        for nombre, m in self._masks.items():
            n = m.numel()
            out[nombre] = float((~m).sum().item()) / n if n else 0.0
        return out

    def save(self, path: str | Path) -> None:
        """
        Persiste las máscaras en disco (``torch.save`` de un dict serializable).

        Args:
            path: Ruta del fichero ``.pt``.
        """
        ruta = Path(path)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.detach().cpu() for k, v in self._masks.items()}
        torch.save(payload, ruta)

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | None = None
    ) -> WeightMaskRegistry:
        """
        Carga máscaras desde ``torch.save`` y opcionalmente las mueve de dispositivo.

        Args:
            path: Ruta del fichero guardado con ``save``.
            device: Si se indica, todas las máscaras se envían a ese dispositivo.

        Returns:
            Nuevo registro con las máscaras cargadas.
        """
        ruta = Path(path)
        try:
            raw = torch.load(ruta, map_location=device or "cpu", weights_only=True)
        except TypeError:
            raw = torch.load(ruta, map_location=device or "cpu")
        if not isinstance(raw, dict):
            msg = "El fichero de máscaras debe contener un dict nombre → tensor."
            raise TypeError(msg)
        mascaras: dict[str, Tensor] = {}
        for k, v in raw.items():
            if not isinstance(v, Tensor):
                msg = f"Valor inválido para la clave '{k}'."
                raise TypeError(msg)
            t = v.bool()
            if device is not None:
                t = t.to(device)
            mascaras[str(k)] = t
        return cls(mascaras)

    def named_masks(self) -> Iterable[tuple[str, Tensor]]:
        """Itera ``(nombre, máscara)`` en orden estable por nombre."""
        for nombre in sorted(self._masks):
            yield nombre, self._masks[nombre]
