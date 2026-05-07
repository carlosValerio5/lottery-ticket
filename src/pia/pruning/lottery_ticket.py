"""
Orquestación de poda iterativa por magnitud con reinicio a ``θ₀`` (lottery ticket).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from pia.data.cifar10 import build_cifar10_loaders
from pia.losses.sparse_loss import SparseLoss
from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import prune_globally_by_magnitude, select_conv_weight_params
from pia.training.grid_search import build_default_observer
from pia.training.loop import fit
from pia.training.observers import TrainingObserver


class LotteryTicketRoundObserver:
    """
    Envuelve un observador e inyecta métricas fijas de poda en cada fin de época.

    Se creó para que TensorBoard y JSONL reciban escalares ``pruning/*`` en cada
    paso sin duplicar lógica en el bucle ``fit``.
    """

    def __init__(
        self,
        inner: TrainingObserver,
        *,
        imp_round: int,
        mask_sparsity: float,
        prune_per_round: float,
    ) -> None:
        self._inner = inner
        self._imp_round = int(imp_round)
        self._mask_sparsity = float(mask_sparsity)
        self._prune_per_round = float(prune_per_round)

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        base = dict(config)
        base.setdefault("pruning/imp_round", float(self._imp_round))
        base.setdefault("pruning/mask_sparsity", float(self._mask_sparsity))
        base.setdefault("pruning/prune_fraction_per_step", float(self._prune_per_round))
        self._inner.on_train_begin(base)

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        extra: dict[str, float] = {
            "pruning/imp_round": float(self._imp_round),
            "pruning/mask_sparsity": float(self._mask_sparsity),
            "pruning/prune_fraction_per_step": float(self._prune_per_round),
        }
        self._inner.on_epoch_end(epoch, {**dict(metrics), **extra})

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        self._inner.on_train_end(summary)


def iterative_magnitude_pruning(
    *,
    run_dir: Path | str,
    num_rounds: int,
    prune_per_round: float,
    epochs_per_round: int,
    lambda_weight: float,
    gamma_activation: float,
    data_root: str,
    batch_size: int,
    lr: float,
    num_workers: int = 0,
    device: torch.device | None = None,
    acc_floor: float = 0.1,
    custom_model: nn.Module | None = None,
    custom_train_loader: DataLoader[Any] | None = None,
    custom_val_loader: DataLoader[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Ejecuta IMP: entrenar, podar fracción de supervivientes, reiniciar a ``θ₀``.

    Hay ``num_rounds + 1`` fases de entrenamiento: ronda 0 densa y ``num_rounds``
    podas sucesivas. Tras cada ronda excepto la última se poda y se restaura el
    ``state_dict`` inicial; las máscaras persisten entre rondas.

    Args:
        run_dir: Directorio raíz del experimento (se crea si no existe).
        num_rounds: Número de pasos de poda (p. ej. 5 para el calendario del plan).
        prune_per_round: Fracción de supervivientes eliminada en cada paso.
        epochs_per_round: Épocas de entrenamiento por fase.
        lambda_weight: Coeficiente L1 de pesos en ``SparseLoss``.
        gamma_activation: Coeficiente de activación en ``SparseLoss``.
        data_root: Raíz de CIFAR-10 en disco.
        batch_size: Tamaño de batch.
        lr: Tasa de aprendizaje de Adam.
        num_workers: Hilos del ``DataLoader``.
        device: Dispositivo; por defecto CUDA si existe.
        acc_floor: Umbral bajo en ``val/acc`` para avisos del bucle ``fit``.
        custom_model: Si se indica, sustituye a ResNet-18 (p. ej. tests rápidos).
        custom_train_loader: Obligatorio junto a ``custom_model``.
        custom_val_loader: Obligatorio junto a ``custom_model``.

    Returns:
        Lista de entradas (una por fase) escrita también en ``imp_index.json``.
    """
    if num_rounds < 0:
        msg = "num_rounds no puede ser negativo."
        raise ValueError(msg)
    ruta = Path(run_dir)
    ruta.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if custom_model is not None:
        if custom_train_loader is None or custom_val_loader is None:
            msg = "custom_model requiere custom_train_loader y custom_val_loader."
            raise ValueError(msg)
        modelo = custom_model.to(device)
        train_loader = custom_train_loader
        val_loader = custom_val_loader
    else:
        modelo = build_resnet18_cifar(num_classes=10)
        apply_he_init(modelo)
        modelo.to(device)
        train_loader, val_loader = build_cifar10_loaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    theta_0 = copy.deepcopy(
        {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    )
    torch.save(theta_0, ruta / "theta_0.pt")
    mascaras = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    mascaras.register_grad_hooks(modelo)
    criterio = SparseLoss(
        lambda_weight=lambda_weight,
        gamma_activation=gamma_activation,
    )
    indice: list[dict[str, Any]] = []
    try:
        for r in range(num_rounds + 1):
            round_dir = ruta / f"round_{r:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            esparcidad_mascara = float(mascaras.current_sparsity())
            observador = LotteryTicketRoundObserver(
                build_default_observer(round_dir),
                imp_round=r,
                mask_sparsity=esparcidad_mascara,
                prune_per_round=prune_per_round,
            )
            optim = torch.optim.Adam(modelo.parameters(), lr=lr)
            post_cb = mascaras.make_post_step_callback(modelo)
            sha = _git_sha_short()
            run_config: dict[str, Any] = {
                "run_id": round_dir.name,
                "imp_parent": ruta.name,
                "imp_round": r,
                "lambda_weight": lambda_weight,
                "gamma_activation": gamma_activation,
                "epochs": epochs_per_round,
                "batch_size": batch_size,
                "lr": lr,
                "data_root": data_root,
                "device": str(device),
                "pruning/imp_round": float(r),
                "pruning/mask_sparsity": esparcidad_mascara,
                "pruning/prune_fraction_per_step": float(prune_per_round),
            }
            if sha:
                run_config["git_sha"] = sha
            ultimo = fit(
                modelo,
                train_loader,
                val_loader,
                criterio,
                optim,
                device,
                observador,
                epochs=epochs_per_round,
                run_config=run_config,
                acc_floor=acc_floor,
                live_progress_dir=round_dir,
                post_optimizer_step=post_cb,
            )
            teorico_fin = 1.0 - (1.0 - prune_per_round) ** min(r + 1, num_rounds)
            entrada: dict[str, Any] = {
                "round": r,
                "run_dir": str(round_dir.resolve()),
                "final_metrics": {k: float(v) for k, v in ultimo.items()},
                "target_sparsity": float(teorico_fin),
                "achieved_sparsity": float(mascaras.current_sparsity()),
            }
            resumen_ronda = {
                "config": run_config,
                "metrics": ultimo,
                "per_layer_sparsity": mascaras.per_layer_sparsity(),
            }
            (round_dir / "round_summary.json").write_text(
                json.dumps(resumen_ronda, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if r < num_rounds:
                info_poda = prune_globally_by_magnitude(
                    mascaras, modelo, prune_per_round
                )
                entrada["prune_info"] = _jsonable_prune_info(info_poda)
                modelo.load_state_dict({k: v.to(device) for k, v in theta_0.items()})
                mascaras.apply_to_weights(modelo)
                entrada["achieved_sparsity"] = float(mascaras.current_sparsity())
                mascaras.save(ruta / f"masks_round_{r + 1:02d}.pt")
            indice.append(entrada)
            (ruta / "imp_index.json").write_text(
                json.dumps(indice, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    finally:
        mascaras.remove_grad_hooks()
    return indice


def _jsonable_prune_info(info: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def _git_sha_short() -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None
