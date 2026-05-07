"""
Búsqueda en rejilla de hiperparámetros ``lambda`` y ``gamma`` para ``SparseLoss``.

Orquesta runs completos (datos, modelo, observadores por defecto y resumen JSON)
sin acoplar PyTorch a ``GridSearchCV`` de scikit-learn: solo se usa
``ParameterGrid`` para iterar combinaciones.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import ParameterGrid

from pia.data.cifar10 import build_cifar10_loaders
from pia.losses.sparse_loss import SparseLoss
from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar
from pia.training.loop import fit
from pia.training.observers import (
    CompositeObserver,
    CsvMetricsObserver,
    JsonlEventsObserver,
    LoggingMetricsObserver,
    TensorBoardObserver,
    TrainingObserver,
)


def _git_sha_short() -> str | None:
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


def build_default_observer(run_dir: Path | str) -> TrainingObserver:
    """Observador compuesto estándar (CSV, JSONL, TensorBoard, logging)."""
    rd = Path(run_dir)
    return CompositeObserver(
        [
            CsvMetricsObserver(rd),
            JsonlEventsObserver(rd),
            TensorBoardObserver(rd),
            LoggingMetricsObserver(),
        ]
    )


def run_single_training(
    *,
    run_dir: Path | str,
    lambda_weight: float,
    gamma_activation: float,
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    observer_factory: Any | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
    acc_floor: float = 0.1,
) -> dict[str, float]:
    """
    Un run completo: datos, modelo, optimizador y ``fit``.

    Args:
        observer_factory: Callable ``(Path) -> TrainingObserver``; si es
            ``None``, se usa ``build_default_observer``.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = build_resnet18_cifar(num_classes=10)
    apply_he_init(modelo)
    modelo.to(device)
    train_loader, val_loader = build_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterio = SparseLoss(
        lambda_weight=lambda_weight, gamma_activation=gamma_activation
    )
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    obs_factory = observer_factory or build_default_observer
    observador = obs_factory(run_path)
    sha = _git_sha_short()
    run_config: dict[str, Any] = {
        "run_id": run_path.name,
        "lambda_weight": lambda_weight,
        "gamma_activation": gamma_activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "data_root": data_root,
        "device": str(device),
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
        epochs=epochs,
        run_config=run_config,
        acc_floor=acc_floor,
        live_progress_dir=run_path,
    )
    resumen_path = run_path / "summary.json"
    resumen_path.write_text(
        json.dumps({"metrics": ultimo, "config": run_config}, indent=2),
        encoding="utf-8",
    )
    return ultimo


def grid_search(
    *,
    base_logdir: Path | str,
    lambda_weights: Sequence[float],
    gamma_activations: Sequence[float],
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    observer_factory: Any | None = None,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    """
    Itera ``ParameterGrid`` sobre ``lambda_weights`` y ``gamma_activations``.

    Returns:
        Lista de dicts con ``run_dir``, ``config`` y ``metrics`` finales.
    """
    base = Path(base_logdir)
    base.mkdir(parents=True, exist_ok=True)
    grid = ParameterGrid(
        {
            "lambda_weight": list(lambda_weights),
            "gamma_activation": list(gamma_activations),
        }
    )
    resultados: list[dict[str, Any]] = []
    for params in grid:
        lw = float(params["lambda_weight"])
        ga = float(params["gamma_activation"])
        nombre = f"lw{lw:g}_ga{ga:g}".replace(".", "p")
        run_dir = base / nombre
        metrics = run_single_training(
            run_dir=run_dir,
            lambda_weight=lw,
            gamma_activation=ga,
            data_root=data_root,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            observer_factory=observer_factory,
            num_workers=num_workers,
        )
        resultados.append(
            {
                "run_dir": str(run_dir),
                "config": {"lambda_weight": lw, "gamma_activation": ga},
                "metrics": metrics,
            }
        )
    indice = base / "grid_index.json"
    indice.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    return resultados
