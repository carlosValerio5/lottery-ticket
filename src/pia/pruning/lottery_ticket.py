"""
Orquestación de poda iterativa por magnitud con reinicio (lottery ticket / IMP).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from pia.data.cifar10 import build_cifar10_loaders
from pia.losses.sparse_loss import SparseLoss, WeightL1Agg
from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import (
    make_imp_param_selector,
    prune_globally_by_magnitude,
)
from pia.training.grid_search import build_default_observer
from pia.training.loop import fit
from pia.training.observers import TrainingObserver

RewindMode = Literal["theta0", "late_k", "none"]
LRSchedulerKind = Literal["none", "cosine", "step"]
EpochCb = Callable[[int, nn.Module, dict[str, float]], None]


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
        merged = {**dict(metrics), **extra}
        merged["sparsity/mask_zero_fraction"] = float(self._mask_sparsity)
        self._inner.on_epoch_end(epoch, merged)

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        self._inner.on_train_end(summary)


def _persist_imp_index(
    ruta: Path,
    indice: list[dict[str, Any]],
    *,
    run_status: str,
    meta: dict[str, Any],
) -> None:
    """Escribe ``imp_index.json`` con envoltorio versionado y estado del run."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "run_status": run_status,
        "meta": meta,
        "rounds": indice,
    }
    (ruta / "imp_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    rewind_mode: RewindMode = "theta0",
    rewind_epoch_k: int = 4,
    exclude_conv1_from_pruning: bool = False,
    exclude_fc_from_pruning: bool = False,
    lr_scheduler_kind: LRSchedulerKind = "none",
    lr_step_size: int = 10,
    lr_step_gamma: float = 0.1,
    weight_l1_aggregation: WeightL1Agg = "sum",
    abort_training_on_val_acc_drop: float | None = None,
) -> list[dict[str, Any]]:
    """
    Ejecuta IMP: entrenar, podar fracción de supervivientes, reinicio opcional.

    Hay ``num_rounds + 1`` fases de entrenamiento: ronda 0 densa y ``num_rounds``
    podas sucesivas. Tras cada ronda excepto la última se poda; el reinicio de
    pesos sigue ``rewind_mode`` (``theta0``, ``late_k`` o ``none``). Las máscaras
    persisten entre rondas.

    Args:
        run_dir: Directorio raíz del experimento (se crea si no existe).
        num_rounds: Número de pasos de poda.
        prune_per_round: Fracción de supervivientes eliminada en cada paso.
        epochs_per_round: Épocas de entrenamiento por fase.
        lambda_weight: Coeficiente L1 de pesos en ``SparseLoss``.
        gamma_activation: Coeficiente de activación en ``SparseLoss``.
        data_root: Raíz de CIFAR-10 en disco.
        batch_size: Tamaño de batch.
        lr: Tasa de aprendizaje inicial de Adam.
        num_workers: Hilos del ``DataLoader``.
        device: Dispositivo; por defecto MPS, CUDA o CPU.
        acc_floor: Umbral bajo en ``val/acc`` para avisos del bucle ``fit``.
        custom_model: Si se indica, sustituye a ResNet-18 (p. ej. tests rápidos).
        custom_train_loader: Obligatorio junto a ``custom_model``.
        custom_val_loader: Obligatorio junto a ``custom_model``.
        rewind_mode: ``theta0`` (reinicio a pesos iniciales), ``late_k`` (reinicio
            a checkpoint de la época ``rewind_epoch_k`` de la ronda 0), ``none``
            (sin reinicio; entrenar desde pesos tras poda).
        rewind_epoch_k: Época 1-based guardada en ronda 0 para ``late_k``.
        exclude_conv1_from_pruning: No podar ``conv1.weight``.
        exclude_fc_from_pruning: No podar ``fc.weight``.
        lr_scheduler_kind: ``none``, ``cosine`` (por ronda) o ``step``.
        lr_step_size: Para ``StepLR``.
        lr_step_gamma: Para ``StepLR``.
        weight_l1_aggregation: Modo de agregación L1 en ``SparseLoss``.
        abort_training_on_val_acc_drop: Parada anticipada si cae ``val/acc`` entre
            épocas más que este valor (``None`` = desactivado).

    Returns:
        Lista de entradas (una por fase), reflejada en ``imp_index.json`` bajo
        la clave ``rounds``.

    Tras cada ``fit`` se guarda ``round_XX/model_state.pt`` (``state_dict`` en
    CPU). Al terminar todas las fases se escribe ``model_final.pt`` en la raíz
    del run (mismo contenido que el último ``model_state.pt``).
    """
    if num_rounds < 0:
        msg = "num_rounds no puede ser negativo."
        raise ValueError(msg)
    if rewind_mode not in ("theta0", "late_k", "none"):
        msg = "rewind_mode debe ser 'theta0', 'late_k' o 'none'."
        raise ValueError(msg)
    if lr_scheduler_kind not in ("none", "cosine", "step"):
        msg = "lr_scheduler_kind debe ser 'none', 'cosine' o 'step'."
        raise ValueError(msg)
    if (
        num_rounds > 0
        and rewind_mode == "late_k"
        and not (1 <= rewind_epoch_k <= epochs_per_round)
    ):
        msg = "rewind_epoch_k debe cumplir 1 <= k <= epochs_per_round para late_k."
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
    torch.save(
        {k: v.cpu() for k, v in theta_0.items()},
        ruta / "theta_0.pt",
    )
    theta_late_k_path = ruta / "theta_late_k.pt"
    selector: Callable[[nn.Module], list[tuple[str, nn.Parameter]]] = (
        make_imp_param_selector(
            exclude_conv1=exclude_conv1_from_pruning,
            exclude_fc=exclude_fc_from_pruning,
        )
    )
    mascaras = WeightMaskRegistry.from_model(modelo, selector)
    mascaras.register_grad_hooks(modelo)
    criterio = SparseLoss(
        lambda_weight=lambda_weight,
        gamma_activation=gamma_activation,
        weight_l1_aggregation=weight_l1_aggregation,
    )
    indice: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "rewind_mode": rewind_mode,
        "rewind_epoch_k": int(rewind_epoch_k),
        "exclude_conv1_from_pruning": bool(exclude_conv1_from_pruning),
        "exclude_fc_from_pruning": bool(exclude_fc_from_pruning),
        "lr_scheduler_kind": lr_scheduler_kind,
        "lr_step_size": int(lr_step_size),
        "lr_step_gamma": float(lr_step_gamma),
        "weight_l1_aggregation": weight_l1_aggregation,
        "pruning_param_policy": "conv_and_linear_weights_only",
        "pass_fail_notes": (
            "Objetivo: val/acc ronda 01 no más de 5 puntos bajo ronda 00 "
            "(ver context/imp_baseline_large_epoch.md)."
        ),
    }
    _persist_imp_index(ruta, [], run_status="in_progress", meta=meta)
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
            sched: torch.optim.lr_scheduler.LRScheduler | None = None
            if lr_scheduler_kind == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=epochs_per_round
                )
            elif lr_scheduler_kind == "step":
                sched = torch.optim.lr_scheduler.StepLR(
                    optim, step_size=max(1, lr_step_size), gamma=lr_step_gamma
                )
            post_cb = mascaras.make_post_step_callback(modelo)
            sha = _git_sha_short()

            def _make_epoch_cb(ronda: int) -> EpochCb:
                def _epoch_cb(ep: int, mod: nn.Module, _m: dict[str, float]) -> None:
                    if ronda == 0 and rewind_mode == "late_k" and ep == rewind_epoch_k:
                        torch.save(
                            {k: v.detach().cpu() for k, v in mod.state_dict().items()},
                            theta_late_k_path,
                        )

                return _epoch_cb

            epoch_cb = _make_epoch_cb(r)

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
                "imp/rewind_mode": rewind_mode,
                "imp/rewind_epoch_k": int(rewind_epoch_k),
                "imp/lr_scheduler": lr_scheduler_kind,
                "imp/weight_l1_aggregation": weight_l1_aggregation,
                "imp/exclude_conv1": exclude_conv1_from_pruning,
                "imp/exclude_fc": exclude_fc_from_pruning,
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
                lr_scheduler=sched,
                epoch_end_callback=epoch_cb,
                abort_training_on_val_acc_drop=abort_training_on_val_acc_drop,
            )
            torch.save(
                {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
                round_dir / "model_state.pt",
            )
            if (
                rewind_mode == "late_k"
                and r == 0
                and num_rounds > 0
                and not theta_late_k_path.is_file()
            ):
                msg = (
                    "No se generó theta_late_k.pt; revisa rewind_epoch_k "
                    "y epochs_per_round."
                )
                raise RuntimeError(msg)
            teorico_fin = 1.0 - (1.0 - prune_per_round) ** min(r + 1, num_rounds)
            entrada: dict[str, Any] = {
                "round": r,
                "status": "complete",
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
                if rewind_mode == "theta0":
                    modelo.load_state_dict(
                        {k: v.to(device) for k, v in theta_0.items()}
                    )
                elif rewind_mode == "late_k":
                    try:
                        theta_k_raw = torch.load(
                            theta_late_k_path, map_location=device, weights_only=True
                        )
                    except TypeError:
                        theta_k_raw = torch.load(theta_late_k_path, map_location=device)
                    if not isinstance(theta_k_raw, dict):
                        msg = "theta_late_k.pt debe contener un state_dict."
                        raise TypeError(msg)
                    modelo.load_state_dict(
                        {k: v.to(device) for k, v in theta_k_raw.items()}
                    )
                elif rewind_mode == "none":
                    pass
                mascaras.apply_to_weights(modelo)
                entrada["achieved_sparsity"] = float(mascaras.current_sparsity())
                mascaras.save(ruta / f"masks_round_{r + 1:02d}.pt")
            indice.append(entrada)
            _persist_imp_index(ruta, indice, run_status="in_progress", meta=meta)
    finally:
        mascaras.remove_grad_hooks()
    _persist_imp_index(ruta, indice, run_status="complete", meta=meta)
    torch.save(
        {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
        ruta / "model_final.pt",
    )
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
