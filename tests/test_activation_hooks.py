"""Pruebas de captura de activaciones."""

from __future__ import annotations

import torch

from pia.models.resnet_cifar import (
    MONITORED_MODULE_NAMES,
    apply_he_init,
    build_resnet18_cifar,
)
from pia.training.activation_hooks import ActivationCapture


def test_activation_capture_keys_match_monitored() -> None:
    modelo = build_resnet18_cifar()
    apply_he_init(modelo)
    cap = ActivationCapture(modelo)
    try:
        x = torch.randn(2, 3, 32, 32)
        cap.clear()
        _ = modelo(x)
        act = cap.current()
        assert set(act.keys()) == set(MONITORED_MODULE_NAMES)
        for t in act.values():
            assert t.ndim == 4
    finally:
        cap.remove()
