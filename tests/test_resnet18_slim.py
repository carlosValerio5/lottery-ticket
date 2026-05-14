"""
Pruebas para ``slim_resnet18_cifar_from_state_dict``.

Recorte por columnas muertas en ``fc``.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar
from pia.pruning.resnet18_slim import (
    parameter_and_buffer_bytes,
    slim_resnet18_cifar_from_state_dict,
)


def _param_count(m: nn.Module) -> int:
    """Número total de elementos en los parámetros del módulo."""
    return sum(p.numel() for p in m.parameters())


def _gap512_before_fc(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Actividad N×512 tras ``layer4`` y GAP (sin ``fc``).

    Los stubs de torchvision tipan atributos como ``conv1``/``layer1`` como
    ``Tensor``; se usa ``cast`` para que el comprobador acepte la llamada.
    """
    t = cast(nn.Conv2d, model.conv1)(x)
    t = cast(nn.BatchNorm2d, model.bn1)(t)
    stem = torch.nn.functional.relu(t)
    pooled = stem  # CIFAR: ``maxpool`` es ``Identity``
    l1 = cast(nn.Module, model.layer1)(pooled)
    l2 = cast(nn.Module, model.layer2)(l1)
    l3 = cast(nn.Module, model.layer3)(l2)
    l4 = cast(nn.Module, model.layer4)(l3)
    return torch.nn.functional.adaptive_avg_pool2d(l4, (1, 1)).flatten(1)


def test_slim_noop_when_no_dead_fc_columns() -> None:
    """Sin columnas nulas en fc, el número de parámetros no cambia."""
    model = build_resnet18_cifar(num_classes=10)
    apply_he_init(model)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    slim, sd_out = slim_resnet18_cifar_from_state_dict(sd, eps=1e-12)
    assert _param_count(slim) == _param_count(model)
    assert sd_out["fc.weight"].shape == sd["fc.weight"].shape


def test_slim_reduces_memory_when_fc_columns_are_zero() -> None:
    """Columnas fc nulas: menos parámetros tras recortar ``layer4``."""
    model = build_resnet18_cifar(num_classes=10)
    apply_he_init(model)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["fc.weight"][:, 480:].zero_()
    slim, sd_out = slim_resnet18_cifar_from_state_dict(sd, eps=1e-12)
    assert sd_out["fc.weight"].shape[1] == 480
    assert _param_count(slim) < _param_count(model)
    assert parameter_and_buffer_bytes(slim) < parameter_and_buffer_bytes(model)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = slim(x)
    assert y.shape == (2, 10)


def test_slim_forward_reference_when_path_is_zeroed() -> None:
    """
    Si las columnas muertas de fc tienen activación residual nula en layer4,
    el recorte coincide con evaluar el modelo ancho y proyectar solo ``keep``.
    """
    model = build_resnet18_cifar(num_classes=10)
    apply_he_init(model)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    dead = list(range(500, 512))
    keep = [j for j in range(512) if j not in dead]
    # Anula el camino residual para los canales eliminados en layer4.1 (identidad).
    for j in dead:
        sd["layer4.1.conv1.weight"][j].zero_()
        sd["layer4.1.conv1.weight"][:, j, :, :].zero_()
        sd["layer4.1.conv2.weight"][j].zero_()
        sd["layer4.1.conv2.weight"][:, j, :, :].zero_()
    for j in dead:
        sd["layer4.0.conv2.weight"][j].zero_()
        sd["layer4.0.conv2.weight"][:, j, :, :].zero_()
        sd["layer4.0.conv1.weight"][j].zero_()
        sd["layer4.0.downsample.0.weight"][j].zero_()
    sd["fc.weight"][:, dead].zero_()

    slim, _ = slim_resnet18_cifar_from_state_dict(sd, eps=1e-12)
    model.load_state_dict(sd)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        full_logits = model(x)
        slim_logits = slim(x)
    # Referencia: mismos pesos fc en columnas vivas; activaciones muertas aportan 0.
    h = _gap512_before_fc(model, x)
    fc = cast(nn.Linear, model.fc)
    ref = h[:, keep] @ fc.weight[:, keep].T + fc.bias
    assert torch.allclose(slim_logits, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(slim_logits, full_logits, atol=1e-4, rtol=1e-4)
