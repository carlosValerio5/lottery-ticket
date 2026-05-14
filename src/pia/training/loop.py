"""
Bucle de entrenamiento y evaluación con pérdida ``SparseLoss``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pia.losses.sparse_loss import SparseLoss
from pia.training.activation_hooks import ActivationCapture
from pia.training.early_stopping import val_loss_rebound_early_stop_step
from pia.training.metrics import accuracy_top1, weight_sparsity_ratio
from pia.training.observers import TrainingObserver

_log = logging.getLogger("pia.training")


def _append_live_batch(
    fh: TextIO,
    *,
    phase: str,
    epoch: int,
    batch: int,
    loss: float,
    loss_task: float,
    acc: float,
) -> None:
    """Escribe una línea JSON para el dashboard intra-época."""
    linea = json.dumps(
        {
            "phase": phase,
            "epoch": int(epoch),
            "batch": int(batch),
            "loss": float(loss),
            "loss_task": float(loss_task),
            "acc": float(acc),
        },
        ensure_ascii=False,
    )
    fh.write(linea + "\n")
    fh.flush()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: SparseLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    capture: ActivationCapture,
    *,
    epoch: int = 1,
    live_batches_path: Path | None = None,
    post_optimizer_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    """
    Una época de entrenamiento con barra de progreso y comprobaciones de sanidad.

    Args:
        epoch: Índice de época (1-based) para trazas ``live_batches.jsonl``.
        live_batches_path: Si se indica, se trunca al inicio y se appende una
            línea JSON por batch para observabilidad en tiempo casi real.
        post_optimizer_step: Invocado justo después de ``optimizer.step()`` si
            se proporciona (p. ej. reaplicar máscaras de poda).

    Returns:
        Diccionario con medias de pérdida y accuracy de train en la época.
    """
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_w = 0.0
    total_a = 0.0
    total_acc = 0.0
    n_batches = 0
    live_fh: TextIO | None = None
    if live_batches_path is not None:
        live_batches_path.parent.mkdir(parents=True, exist_ok=True)
        live_fh = live_batches_path.open("w", encoding="utf-8")
    try:
        pbar = tqdm(loader, desc="train", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            capture.clear()
            logits = model(x)
            acts = capture.current()
            loss, desglose = criterion(logits, y, model, acts)
            if not torch.isfinite(loss):
                _log.error("Pérdida no finita en entrenamiento.")
                raise RuntimeError("loss_nan")
            loss.backward()
            optimizer.step()
            if post_optimizer_step is not None:
                post_optimizer_step()
            acc = accuracy_top1(logits.detach(), y)
            loss_f = float(loss.detach().item())
            task_f = float(desglose.task.detach().item())
            total_loss += loss_f
            total_task += task_f
            total_w += float(desglose.weight_l1.detach().item())
            total_a += float(desglose.activation_l1.detach().item())
            total_acc += acc
            n_batches += 1
            if live_fh is not None:
                _append_live_batch(
                    live_fh,
                    phase="train",
                    epoch=epoch,
                    batch=n_batches,
                    loss=loss_f,
                    loss_task=task_f,
                    acc=acc,
                )
            pbar.set_postfix(loss=loss_f, task=task_f, acc=acc)
    finally:
        if live_fh is not None:
            live_fh.close()
    if n_batches == 0:
        msg = "DataLoader de entrenamiento vacío."
        raise ValueError(msg)
    return {
        "train/loss": total_loss / n_batches,
        "train/loss_task": total_task / n_batches,
        "train/loss_weight_l1": total_w / n_batches,
        "train/loss_activation_l1": total_a / n_batches,
        "train/acc": total_acc / n_batches,
        "train/weight_sparsity_ratio": weight_sparsity_ratio(model),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: SparseLoss,
    device: torch.device,
    capture: ActivationCapture,
    *,
    epoch: int = 1,
    live_batches_path: Path | None = None,
) -> dict[str, float]:
    """
    Evalúa en modo inferencia (sin gradientes) con la misma pérdida para logging.

    Args:
        epoch: Índice de época (1-based) para trazas ``live_batches.jsonl``.
        live_batches_path: Si se indica, se hace append por batch (fase val).
    """
    model.eval()
    total_loss = 0.0
    total_task = 0.0
    total_w = 0.0
    total_a = 0.0
    total_acc = 0.0
    n_batches = 0
    live_fh: TextIO | None = None
    if live_batches_path is not None:
        live_fh = live_batches_path.open("a", encoding="utf-8")
    try:
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            capture.clear()
            logits = model(x)
            acts = capture.current()
            loss, desglose = criterion(logits, y, model, acts)
            if not torch.isfinite(loss):
                _log.error("Pérdida no finita en validación.")
                raise RuntimeError("val_loss_nan")
            loss_f = float(loss.item())
            task_f = float(desglose.task.item())
            total_loss += loss_f
            total_task += task_f
            total_w += float(desglose.weight_l1.item())
            total_a += float(desglose.activation_l1.item())
            acc = accuracy_top1(logits, y)
            total_acc += acc
            n_batches += 1
            if live_fh is not None:
                _append_live_batch(
                    live_fh,
                    phase="val",
                    epoch=epoch,
                    batch=n_batches,
                    loss=loss_f,
                    loss_task=task_f,
                    acc=acc,
                )
    finally:
        if live_fh is not None:
            live_fh.close()
    if n_batches == 0:
        msg = "DataLoader de validación vacío."
        raise ValueError(msg)
    return {
        "val/loss": total_loss / n_batches,
        "val/loss_task": total_task / n_batches,
        "val/loss_weight_l1": total_w / n_batches,
        "val/loss_activation_l1": total_a / n_batches,
        "val/acc": total_acc / n_batches,
        "val/weight_sparsity_ratio": weight_sparsity_ratio(model),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    criterion: SparseLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    observer: TrainingObserver,
    *,
    epochs: int,
    run_config: Mapping[str, Any],
    acc_floor: float = 0.1,
    val_acc_drop_warn: float = 0.15,
    live_progress_dir: Path | None = None,
    post_optimizer_step: Callable[[], None] | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch_end_callback: (
        Callable[[int, nn.Module, dict[str, float]], None] | None
    ) = None,
    abort_training_on_val_acc_drop: float | None = None,
    early_stopping_val_loss_relative: float | None = None,
    early_stopping_patience: int = 1,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, float]:
    """
    Entrena ``epochs`` épocas y notifica al observador.

    Args:
        acc_floor: Si ``val/acc`` cae por debajo, se emite ``WARNING``.
        val_acc_drop_warn: Si la ``val/acc`` baja más que este margen respecto
            a la época anterior, se emite ``WARNING`` (posible inestabilidad).
        live_progress_dir: Si se indica, escribe ``live_batches.jsonl`` por
            batch (train y val) para dashboards en vivo.
        post_optimizer_step: Se reenvía a ``train_one_epoch`` en cada batch.
        lr_scheduler: Si se indica, ``step()`` tras cada época.
        epoch_end_callback: Invocado tras ``on_epoch_end`` con
            ``(época, modelo, métricas)`` (p. ej. checkpoints IMP).
        abort_training_on_val_acc_drop: Si no es ``None`` y la ``val/acc`` baja
            más que este margen respecto a la época anterior, corta el entrenamiento.
        early_stopping_val_loss_relative: Si no es ``None`` y ``>= 0``, corta
            cuando ``val/loss`` supere ``(1+r)`` veces el mejor ``val/loss`` visto
            durante ``early_stopping_patience`` épocas seguidas.
        early_stopping_patience: Épocas consecutivas por encima del umbral relativo.
        early_stopping_min_delta: Mejora mínima en ``val/loss`` para actualizar el
            mejor valor (evita ruido numérico).
    """
    if early_stopping_val_loss_relative is not None:
        if early_stopping_val_loss_relative < 0.0:
            msg = "early_stopping_val_loss_relative debe ser >= 0 cuando se usa."
            raise ValueError(msg)
        if early_stopping_patience < 1:
            msg = "early_stopping_patience debe ser >= 1."
            raise ValueError(msg)
    capture = ActivationCapture(model)
    observer.on_train_begin(dict(run_config))
    ultimo_val: dict[str, float] = {}
    val_acc_prev: float | None = None
    best_val_loss = math.inf
    val_loss_bad_epochs = 0
    early_stopped = False
    early_stop_reason: str | None = None
    live_path: Path | None = None
    if live_progress_dir is not None:
        live_path = Path(live_progress_dir) / "live_batches.jsonl"
    try:
        for ep in range(1, epochs + 1):
            train_m = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                capture,
                epoch=ep,
                live_batches_path=live_path,
                post_optimizer_step=post_optimizer_step,
            )
            val_m = evaluate(
                model,
                val_loader,
                criterion,
                device,
                capture,
                epoch=ep,
                live_batches_path=live_path,
            )
            ultimo_val = {**train_m, **val_m}
            observer.on_epoch_end(ep, ultimo_val)
            if epoch_end_callback is not None:
                epoch_end_callback(ep, model, ultimo_val)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if val_acc_prev is not None:
                caida = val_acc_prev - val_m["val/acc"]
                if caida > val_acc_drop_warn:
                    _log.warning(
                        "Caída fuerte de val/acc: %.4f -> %.4f (delta=%.4f)",
                        val_acc_prev,
                        val_m["val/acc"],
                        caida,
                    )
                if (
                    abort_training_on_val_acc_drop is not None
                    and caida > abort_training_on_val_acc_drop
                ):
                    _log.warning(
                        "Parada anticipada: caída val/acc %.4f > límite %.4f.",
                        caida,
                        abort_training_on_val_acc_drop,
                    )
                    early_stopped = True
                    early_stop_reason = "val_acc_drop"
                    break
            val_acc_prev = float(val_m["val/acc"])
            if val_m["val/acc"] < acc_floor:
                _log.warning(
                    "Accuracy de validación baja (%.4f < %.4f).",
                    val_m["val/acc"],
                    acc_floor,
                )
            if early_stopping_val_loss_relative is not None:
                vloss = float(ultimo_val["val/loss"])
                best_val_loss, val_loss_bad_epochs, stop_loss = (
                    val_loss_rebound_early_stop_step(
                        vloss,
                        best_val_loss,
                        relative_margin=early_stopping_val_loss_relative,
                        patience=early_stopping_patience,
                        min_delta=early_stopping_min_delta,
                        bad_epochs=val_loss_bad_epochs,
                    )
                )
                if stop_loss:
                    _log.warning(
                        "Parada anticipada (val/loss): actual=%.6f umbral relativo "
                        "sobre mejor=%.6f r=%.4f racha=%d/%d.",
                        vloss,
                        best_val_loss,
                        early_stopping_val_loss_relative,
                        val_loss_bad_epochs,
                        early_stopping_patience,
                    )
                    early_stopped = True
                    early_stop_reason = "val_loss_rebound"
                    break
    finally:
        capture.remove()
    if early_stopped:
        ultimo_val = {**ultimo_val, "train/early_stopped": 1.0}
    resumen: dict[str, Any] = {"last_metrics": ultimo_val}
    if early_stopped:
        resumen["early_stopped"] = True
        if early_stop_reason is not None:
            resumen["early_stop_reason"] = early_stop_reason
    observer.on_train_end(resumen)
    return ultimo_val
