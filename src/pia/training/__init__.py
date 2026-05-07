"""Bucles de entrenamiento, métricas y observadores."""

from pia.training.activation_hooks import ActivationCapture
from pia.training.loop import evaluate, fit, train_one_epoch
from pia.training.metrics import accuracy_top1, weight_sparsity_ratio
from pia.training.observers import (
    CompositeObserver,
    CsvMetricsObserver,
    JsonlEventsObserver,
    LoggingMetricsObserver,
    TensorBoardObserver,
    TrainingObserver,
)

__all__ = [
    "ActivationCapture",
    "CompositeObserver",
    "CsvMetricsObserver",
    "JsonlEventsObserver",
    "LoggingMetricsObserver",
    "TensorBoardObserver",
    "TrainingObserver",
    "accuracy_top1",
    "evaluate",
    "fit",
    "train_one_epoch",
    "weight_sparsity_ratio",
]
