"""
Orquestación de poda estructurada por recorte de ancho (sin reinicio a θ₀).

El bucle replica la forma de ``iterative_magnitude_pruning`` pero sustituye
máscaras element-wise por un modelo físicamente más estrecho tras cada paso.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from pia.losses.sparse_loss import SparseLoss, WeightL1Agg
from pia.models.narrowable_chain_cnn import NarrowableChainCnn
from pia.pruning.lottery_ticket import LotteryTicketRoundObserver
from pia.pruning.prune_structured import (
    count_parameters,
    narrow_chain_cnn_by_fraction,
)
from pia.training.grid_search import build_default_observer
from pia.training.loop import fit
from pia.training.observers import TrainingObserver

LRSchedulerKind = Literal["none", "cosine", "step"]
RewindMode = Literal["theta0", "late_k", "none"]


def _persist_structured_index(
    ruta: Path,
    indice: list[dict[str, Any]],
    *,
    run_status: str,
    meta: dict[str, Any],
) -> None:
    """Escribe ``structured_index.json`` con el mismo envoltorio que IMP."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "run_status": run_status,
        "meta": meta,
        "rounds": indice,
    }
    (ruta / "structured_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _jsonable_prune_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Convierte valores de ``prune_info`` a tipos seguros para ``json.dumps``."""
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
    """Devuelve el hash corto de git del repo o ``None`` si no está disponible."""
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


def _layer_shapes_dict(model: nn.Module) -> dict[str, list[int]]:
    """Formas de peso relevantes para trazabilidad en JSON."""
    out: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        if name.endswith((".weight", ".bias")):
            out[name] = list(p.shape)
    return out


def iterative_structured_magnitude_pruning(
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
    rewind_mode: RewindMode = "none",
    lr_scheduler_kind: LRSchedulerKind = "none",
    lr_step_size: int = 10,
    lr_step_gamma: float = 0.1,
    weight_l1_aggregation: WeightL1Agg = "sum",
    abort_training_on_val_acc_drop: float | None = None,
    early_stopping_val_loss_relative: float | None = None,
    early_stopping_patience: int = 1,
    early_stopping_min_delta: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Entrena en rondas y recorta ancho de ``NarrowableChainCnn`` entre rondas.

    No soporta reinicio a θ₀ ni ``late_k``: los ``state_dict`` cambian de forma
    entre podas. ``rewind_mode`` distinto de ``\"none\"`` provoca error.

    Sin ``custom_model``, intenta ResNet-18 CIFAR y falla con mensaje explícito
    porque el recorte estructurado en cadena no aplica a residuales.

    Args:
        run_dir: Directorio raíz del experimento.
        num_rounds: Pasos de poda tras la ronda densa inicial.
        prune_per_round: Fracción del ancho de canal actual a eliminar por paso.
        epochs_per_round: Épocas por fase.
        lambda_weight: Coeficiente L1 de pesos en ``SparseLoss``.
        gamma_activation: Coeficiente de activación en ``SparseLoss``.
        data_root: Raíz CIFAR-10 (solo si se usa ResNet; hoy no soportado).
        batch_size: Tamaño de batch.
        lr: Tasa de aprendizaje Adam.
        num_workers: Hilos del ``DataLoader``.
        device: Dispositivo de cómputo.
        acc_floor: Umbral bajo de ``val/acc`` para avisos.
        custom_model: Debe ser ``NarrowableChainCnn`` con loaders asociados.
        custom_train_loader: Obligatorio con ``custom_model``.
        custom_val_loader: Obligatorio con ``custom_model``.
        rewind_mode: Solo ``\"none\"`` (otros valores lanzan ``ValueError``).
        lr_scheduler_kind: ``none``, ``cosine`` o ``step`` (por ronda).
        lr_step_size: Para ``StepLR``.
        lr_step_gamma: Para ``StepLR``.
        weight_l1_aggregation: Agregación L1 en ``SparseLoss``.
        abort_training_on_val_acc_drop: Parada anticipada por caída de val/acc.
        early_stopping_val_loss_relative: Si no es ``None`` y ``>= 0``, parada
            cuando ``val/loss`` rebota respecto al mejor valor de la fase.
        early_stopping_patience: Épocas consecutivas por encima del umbral.
        early_stopping_min_delta: Mejora mínima para registrar un nuevo mínimo.

    Returns:
        Lista de entradas por ronda (reflejada en ``structured_index.json``).

    Raises:
        ValueError: Configuración inválida o backbone no soportado.
    """
    if rewind_mode != "none":
        msg = (
            "iterative_structured_magnitude_pruning solo admite rewind_mode="
            "'none' (los recortes cambian las formas del state_dict)."
        )
        raise ValueError(msg)
    if num_rounds < 0:
        msg = "num_rounds no puede ser negativo."
        raise ValueError(msg)
    if lr_scheduler_kind not in ("none", "cosine", "step"):
        msg = "lr_scheduler_kind debe ser 'none', 'cosine' o 'step'."
        raise ValueError(msg)
    if early_stopping_val_loss_relative is not None:
        if early_stopping_val_loss_relative < 0.0:
            msg = "early_stopping_val_loss_relative debe ser >= 0 cuando no es None."
            raise ValueError(msg)
        if early_stopping_patience < 1:
            msg = "early_stopping_patience debe ser >= 1."
            raise ValueError(msg)

    ruta = Path(run_dir)
    ruta.mkdir(parents=True, exist_ok=True)

    if device is None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if custom_model is not None:
        if custom_train_loader is None or custom_val_loader is None:
            msg = "custom_model requiere custom_train_loader y custom_val_loader."
            raise ValueError(msg)
        if not isinstance(custom_model, NarrowableChainCnn):
            msg = (
                "custom_model debe ser NarrowableChainCnn para la poda "
                "estructurada por ancho en esta versión."
            )
            raise TypeError(msg)
        modelo = custom_model.to(device)
        train_loader = custom_train_loader
        val_loader = custom_val_loader
    else:
        msg = (
            "iterative_structured_magnitude_pruning: el recorte de ancho en "
            "cadena no está implementado para ResNet-18; pasa "
            "custom_model=NarrowableChainCnn(...) y loaders."
        )
        raise ValueError(msg)

    theta_0 = copy.deepcopy(
        {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    )
    torch.save(
        {k: v.cpu() for k, v in theta_0.items()},
        ruta / "theta_0.pt",
    )
    initial_width = int(modelo.width)
    indice: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "pipeline": "structured_width_chain_cnn",
        "rewind_mode": rewind_mode,
        "lr_scheduler_kind": lr_scheduler_kind,
        "lr_step_size": int(lr_step_size),
        "lr_step_gamma": float(lr_step_gamma),
        "weight_l1_aggregation": weight_l1_aggregation,
        "initial_conv_width": initial_width,
    }
    if early_stopping_val_loss_relative is not None:
        meta["early_stop_val_loss_relative"] = float(early_stopping_val_loss_relative)
        meta["early_stop_patience"] = int(early_stopping_patience)
        meta["early_stop_val_loss_min_delta"] = float(early_stopping_min_delta)
    _persist_structured_index(ruta, [], run_status="in_progress", meta=meta)
    criterio = SparseLoss(
        lambda_weight=lambda_weight,
        gamma_activation=gamma_activation,
        weight_l1_aggregation=weight_l1_aggregation,
    )
    try:
        for r in range(num_rounds + 1):
            round_dir = ruta / f"round_{r:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            width_now = int(modelo.width)
            structural_sparsity = 1.0 - float(width_now) / float(initial_width)
            observador: TrainingObserver = LotteryTicketRoundObserver(
                build_default_observer(round_dir),
                imp_round=r,
                mask_sparsity=structural_sparsity,
                prune_per_round=prune_per_round,
            )
            optim = torch.optim.Adam(modelo.parameters(), lr=lr)
            sched: torch.optim.lr_scheduler.LRScheduler | None = None
            if lr_scheduler_kind == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=epochs_per_round
                )
            elif lr_scheduler_kind == "step":
                sched = torch.optim.lr_scheduler.StepLR(
                    optim, step_size=max(1, lr_step_size), gamma=lr_step_gamma
                )
            sha = _git_sha_short()
            run_config: dict[str, Any] = {
                "run_id": round_dir.name,
                "structured_parent": ruta.name,
                "structured_round": r,
                "lambda_weight": lambda_weight,
                "gamma_activation": gamma_activation,
                "epochs": epochs_per_round,
                "batch_size": batch_size,
                "lr": lr,
                "data_root": data_root,
                "device": str(device),
                "pruning/imp_round": float(r),
                "pruning/mask_sparsity": float(structural_sparsity),
                "pruning/prune_fraction_per_step": float(prune_per_round),
                "structured/conv_width": float(width_now),
                "structured/initial_width": float(initial_width),
                "structured/lr_scheduler": lr_scheduler_kind,
                "structured/weight_l1_aggregation": weight_l1_aggregation,
            }
            if early_stopping_val_loss_relative is not None:
                run_config["structured/early_stop_val_loss_relative"] = float(
                    early_stopping_val_loss_relative
                )
                run_config["structured/early_stop_patience"] = float(
                    early_stopping_patience
                )
                run_config["structured/early_stop_val_loss_min_delta"] = float(
                    early_stopping_min_delta
                )
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
                post_optimizer_step=None,
                lr_scheduler=sched,
                epoch_end_callback=None,
                abort_training_on_val_acc_drop=abort_training_on_val_acc_drop,
                early_stopping_val_loss_relative=early_stopping_val_loss_relative,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
            )
            torch.save(
                {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
                round_dir / "model_state.pt",
            )
            teorico_fin = 1.0 - (1.0 - prune_per_round) ** min(r + 1, num_rounds)
            params_now = count_parameters(modelo)
            entrada: dict[str, Any] = {
                "round": r,
                "status": "complete",
                "run_dir": str(round_dir.resolve()),
                "final_metrics": {k: float(v) for k, v in ultimo.items()},
                "target_sparsity": float(teorico_fin),
                "achieved_sparsity": float(structural_sparsity),
                "structured_conv_width": width_now,
                "structured_param_count": params_now,
                "layer_shapes": _layer_shapes_dict(modelo),
            }
            resumen_ronda = {
                "config": run_config,
                "metrics": ultimo,
                "layer_shapes": entrada["layer_shapes"],
            }
            (round_dir / "round_summary.json").write_text(
                json.dumps(resumen_ronda, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if r < num_rounds:
                modelo_cpu = modelo.cpu()
                nuevo, info_poda = narrow_chain_cnn_by_fraction(
                    modelo_cpu, prune_per_round
                )
                modelo = nuevo.to(device)
                entrada["prune_info"] = _jsonable_prune_info(info_poda)
                entrada["structured_conv_width"] = int(modelo.width)
                entrada["achieved_sparsity"] = 1.0 - float(modelo.width) / float(
                    initial_width
                )
                entrada["structured_param_count"] = count_parameters(modelo)
                entrada["layer_shapes"] = _layer_shapes_dict(modelo)
            indice.append(entrada)
            _persist_structured_index(ruta, indice, run_status="in_progress", meta=meta)
    finally:
        pass
    _persist_structured_index(ruta, indice, run_status="complete", meta=meta)
    torch.save(
        {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
        ruta / "model_final.pt",
    )
    return indice
