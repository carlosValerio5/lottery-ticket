"""Pruebas de humo para ResNet CIFAR y API de módulos monitoreados."""

import torch
from torch import nn

from pia.models.resnet_cifar import (
    MONITORED_MODULE_NAMES,
    apply_he_init,
    build_resnet18_cifar,
    get_monitored_modules,
)


def test_forward_shape_and_finite() -> None:
    """Un forward 32×32 debe producir logits finitos de la forma esperada."""
    model = build_resnet18_cifar(num_classes=10)
    apply_he_init(model)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert bool(torch.isfinite(logits).all())


def test_get_monitored_modules() -> None:
    """Los nombres acordados deben resolver a submódulos válidos."""
    model = build_resnet18_cifar()
    apply_he_init(model)
    mods = get_monitored_modules(model)
    assert set(mods.keys()) == set(MONITORED_MODULE_NAMES)
    for submodule in mods.values():
        assert isinstance(submodule, nn.Module)


def test_backward_smoke() -> None:
    """Un paso backward no debe producir gradientes no finitos al arranque."""
    model = build_resnet18_cifar()
    apply_he_init(model)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert bool(torch.isfinite(param.grad).all())
