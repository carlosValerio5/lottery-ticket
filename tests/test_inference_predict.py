"""Inferencia y latencia."""

from __future__ import annotations

import logging

import pytest
import torch

from pia.inference.predict import run_inference_batch
from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar


def test_run_inference_batch_logs_and_shape(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    modelo = build_resnet18_cifar()
    apply_he_init(modelo)
    x = torch.randn(2, 3, 32, 32)
    logits, stats = run_inference_batch(modelo, x)
    assert logits.shape == (2, 10)
    assert stats["batch_size"] == 2
    assert "latency_ms" in stats
    assert any("inference_batch" in r.message for r in caplog.records)
