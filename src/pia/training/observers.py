"""
Observadores de entrenamiento: CSV, JSONL, TensorBoard y logging compuesto.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from torch.utils.tensorboard import SummaryWriter


@runtime_checkable
class TrainingObserver(Protocol):
    """Contrato para reaccionar a hitos del bucle de entrenamiento."""

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        """Inicio de un run (hiperparámetros, rutas, identificadores)."""

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Fin de época con métricas agregadas (escalares serializables)."""

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        """Cierre del run (p. ej. flush de writers)."""


class CompositeObserver:
    """Encadena varios observadores."""

    def __init__(self, observers: Sequence[TrainingObserver]) -> None:
        self._observers = list(observers)

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        for obs in self._observers:
            obs.on_train_begin(config)

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        for obs in self._observers:
            obs.on_epoch_end(epoch, metrics)

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        for obs in self._observers:
            obs.on_train_end(summary)


class CsvMetricsObserver(TrainingObserver):
    """Escribe una fila por época en ``metrics.csv`` (cabecera en la primera época)."""

    def __init__(self, run_dir: Path | str) -> None:
        self._path = Path(run_dir) / "metrics.csv"
        self._f: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._config: dict[str, Any] = {}

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._config = dict(config)
        self._f = None
        self._writer = None

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        fila: dict[str, Any] = {"epoch": epoch, **self._config, **dict(metrics)}
        if self._writer is None:
            self._f = self._path.open("w", newline="", encoding="utf-8")
            claves = sorted(fila.keys())
            self._writer = csv.DictWriter(self._f, fieldnames=claves)
            self._writer.writeheader()
        assert self._writer is not None and self._f is not None
        if set(fila.keys()) - set(self._writer.fieldnames):
            msg = "No se admiten claves nuevas de métricas a mitad de entrenamiento."
            raise ValueError(msg)
        self._writer.writerow({k: fila.get(k, "") for k in self._writer.fieldnames})
        self._f.flush()

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        self._writer = None


class JsonlEventsObserver(TrainingObserver):
    """Añade una línea JSON por época en ``events.jsonl``."""

    def __init__(self, run_dir: Path | str) -> None:
        self._path = Path(run_dir) / "events.jsonl"
        self._f: TextIO | None = None
        self._base: dict[str, Any] = {}

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self._path.open("a", encoding="utf-8")
        self._base = dict(config)

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if self._f is None:
            return
        evento = {
            **self._base,
            "epoch": epoch,
            **{k: float(v) for k, v in metrics.items()},
        }
        self._f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        self._f.flush()

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


class TensorBoardObserver(TrainingObserver):
    """Escalares de época en TensorBoard."""

    def __init__(self, run_dir: Path | str) -> None:
        self._run_dir = Path(run_dir)
        self._writer: SummaryWriter | None = None

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._run_dir / "tb"))

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if self._writer is None:
            return
        for clave, valor in metrics.items():
            self._writer.add_scalar(clave, valor, global_step=epoch)
        self._writer.flush()

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class LoggingMetricsObserver(TrainingObserver):
    """Registra métricas de época con el logger ``pia``."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("pia")

    def on_train_begin(self, config: Mapping[str, Any]) -> None:
        self._log.info("train_begin %s", json.dumps(dict(config), ensure_ascii=False))

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        self._log.info(
            "epoch_end epoch=%s metrics=%s",
            epoch,
            json.dumps(dict(metrics), ensure_ascii=False),
        )

    def on_train_end(self, summary: Mapping[str, Any]) -> None:
        self._log.info("train_end %s", json.dumps(dict(summary), ensure_ascii=False))
