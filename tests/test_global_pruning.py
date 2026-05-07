"""
Pruebas de poda global por magnitud sobre pesos convolucionales.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from pia.models.resnet_cifar import apply_he_init
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import prune_globally_by_magnitude, select_conv_weight_params


def _toy_model() -> nn.Module:
    """25 pesos conv totales para que ``int(0.2 * 25) = 5`` sea exactamente 20 %."""
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Conv2d(1, 5, 1, bias=False),
        nn.BatchNorm2d(5),
        nn.Conv2d(5, 4, 1, bias=False),
        nn.Linear(4, 2, bias=True),
    )
    apply_he_init(m)
    return m


def test_global_prune_twenty_percent_smallest_conv() -> None:
    modelo = _toy_model()
    snap = copy.deepcopy(modelo.state_dict())
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    info = prune_globally_by_magnitude(reg, modelo, 0.2)
    n_conv = modelo[0].weight.numel() + modelo[2].weight.numel()
    k = int(0.2 * n_conv)
    assert info["k"] == k
    assert abs(info["survivors_pruned_fraction"] - 0.2) < 1e-5
    claves_conv = {"0.weight", "2.weight"}
    for k_n, v in snap.items():
        cur = modelo.state_dict()[k_n]
        if k_n in claves_conv:
            continue
        assert torch.allclose(cur, v)
    flat0 = modelo[0].weight.reshape(-1)
    flat2 = modelo[2].weight.reshape(-1)
    conv_flat = torch.cat([flat0, flat2])
    assert int((conv_flat == 0).sum().item()) == k


def test_second_prune_of_survivors_cumulative() -> None:
    modelo = _toy_model()
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    prune_globally_by_magnitude(reg, modelo, 0.2)
    s1 = reg.current_sparsity()
    prune_globally_by_magnitude(reg, modelo, 0.2)
    s2 = reg.current_sparsity()
    assert s2 > s1
    esperado = 1.0 - (1.0 - 0.2) ** 2
    assert abs(s2 - esperado) < 1e-4
