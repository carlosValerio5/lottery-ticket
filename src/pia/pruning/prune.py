"""
Selección de pesos convolucionales y poda global por magnitud.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from pia.pruning.masks import WeightMaskRegistry


def select_imp_weight_params(
    model: nn.Module,
    *,
    exclude_conv1: bool = False,
    exclude_fc: bool = False,
) -> list[tuple[str, nn.Parameter]]:
    """
    Lista los ``weight`` de ``nn.Conv2d`` y ``nn.Linear`` (sin sesgos ni BatchNorm).

    Convención IMP en ResNet: podar solo matrices de conv/cabezal FC; dejar densos
    los afines de BatchNorm y los sesgos. Opcionalmente excluye el primer conv
    (``conv1``) y/o la cabeza ``fc`` por ser capas pequeñas de alto impacto.

    Args:
        model: Red a inspeccionar.
        exclude_conv1: Si True, no incluye el parámetro ``conv1.weight`` (raíz).
        exclude_fc: Si True, no incluye ``fc.weight``.

    Returns:
        Lista ``(nombre_completo, parámetro)`` alineada con ``named_parameters``.
    """
    salida: list[tuple[str, nn.Parameter]] = []
    for nombre, param in model.named_parameters():
        if not nombre.endswith(".weight"):
            continue
        if exclude_conv1 and nombre == "conv1.weight":
            continue
        if exclude_fc and nombre == "fc.weight":
            continue
        prefijo = nombre[: -len(".weight")]
        mod: nn.Module = model
        if prefijo:
            for parte in prefijo.split("."):
                mod = getattr(mod, parte)
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            salida.append((nombre, param))
    return salida


def make_imp_param_selector(
    *,
    exclude_conv1: bool = False,
    exclude_fc: bool = False,
) -> Callable[[nn.Module], list[tuple[str, nn.Parameter]]]:
    """
    Devuelve una función compatible con ``WeightMaskRegistry.from_model``.

    Args:
        exclude_conv1: Ver ``select_imp_weight_params``.
        exclude_fc: Ver ``select_imp_weight_params``.

    Returns:
        Callable que acepta un ``nn.Module`` y devuelve pares nombre/parámetro.
    """

    def _selector(m: nn.Module) -> list[tuple[str, nn.Parameter]]:
        return select_imp_weight_params(
            m, exclude_conv1=exclude_conv1, exclude_fc=exclude_fc
        )

    return _selector


def select_conv_weight_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """
    Lista los ``weight`` de cada ``nn.Conv2d`` con el nombre canónico del modelo.

    Args:
        model: Red a inspeccionar.

    Returns:
        Lista ``(nombre_completo, parámetro)`` alineada con ``named_parameters``.
    """
    nombres_conv: set[str] = set()
    for prefijo, modulo in model.named_modules():
        if isinstance(modulo, nn.Conv2d):
            base = f"{prefijo}." if prefijo else ""
            nombres_conv.add(f"{base}weight")
    salida: list[tuple[str, nn.Parameter]] = []
    for nombre, param in model.named_parameters():
        if nombre in nombres_conv:
            salida.append((nombre, param))
    return salida


def prune_globally_by_magnitude(
    masks: WeightMaskRegistry,
    model: nn.Module,
    fraction: float,
) -> dict[str, Any]:
    """
    Poda global: elimina una fracción de los supervivientes con menor ``|w|``.

    Tras la operación, las posiciones podadas quedan a cero en ``param.data`` y
    la máscara correspondiente pasa a ``False`` en esas posiciones.

    Args:
        masks: Registro de máscaras sincronizado con el modelo.
        model: Red cuyos pesos se podan.
        fraction: Fracción en ``(0, 1)`` del conjunto actual de supervivientes.

    Returns:
        Diccionario con fracciones y umbral usado para trazabilidad.
    """
    if not 0.0 < fraction < 1.0:
        msg = "fraction debe estar en el intervalo abierto (0, 1)."
        raise ValueError(msg)
    parametros = dict(model.named_parameters())
    mags: list[Tensor] = []
    name_ids: list[Tensor] = []
    flat_ixs: list[Tensor] = []
    nombres_orden = sorted(masks.masks_dict().keys())
    nombre_a_id = {n: i for i, n in enumerate(nombres_orden)}
    dispositivo: torch.device | None = None
    for nombre in nombres_orden:
        m = masks.masks_dict()[nombre]
        p = parametros[nombre]
        if dispositivo is None:
            dispositivo = p.device
        surv = m.to(device=p.device, dtype=torch.bool)
        idx = torch.nonzero(surv.reshape(-1), as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        flat = p.data.reshape(-1)
        mag = flat[idx].abs()
        mags.append(mag)
        name_ids.append(
            torch.full(
                (idx.numel(),),
                nombre_a_id[nombre],
                device=p.device,
                dtype=torch.long,
            )
        )
        flat_ixs.append(idx)
    if not mags:
        return {
            "target_prune_fraction": float(fraction),
            "survivors_pruned_fraction": 0.0,
            "achieved_cumulative_sparsity": masks.current_sparsity(),
            "threshold": float("nan"),
            "k": 0,
            "n_survivors": 0,
        }
    big = torch.cat(mags)
    name_cat = torch.cat(name_ids)
    flat_cat = torch.cat(flat_ixs)
    n = int(big.numel())
    k = int(fraction * n)
    if k <= 0:
        return {
            "target_prune_fraction": float(fraction),
            "survivors_pruned_fraction": 0.0,
            "achieved_cumulative_sparsity": masks.current_sparsity(),
            "threshold": float("nan"),
            "k": 0,
            "n_survivors": n,
        }
    _, idx_smallest = torch.topk(big, k, largest=False)
    umbrales = big[idx_smallest]
    threshold = float(umbrales.max().item())
    sel_name_ids = name_cat[idx_smallest]
    sel_flat_ix = flat_cat[idx_smallest]
    for nid in sel_name_ids.unique():
        nombre = nombres_orden[int(nid.item())]
        m = masks.masks_dict()[nombre]
        p = parametros[nombre]
        sel = sel_name_ids == nid
        ix_loc = sel_flat_ix[sel]
        m_flat = m.reshape(-1)
        p_flat = p.data.reshape(-1)
        m_flat[ix_loc] = False
        p_flat[ix_loc] = 0
    return {
        "target_prune_fraction": float(fraction),
        "survivors_pruned_fraction": float(k) / float(n),
        "achieved_cumulative_sparsity": float(masks.current_sparsity()),
        "threshold": threshold,
        "k": k,
        "n_survivors": n,
    }
