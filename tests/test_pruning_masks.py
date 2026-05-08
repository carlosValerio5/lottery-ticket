"""
Pruebas del registro de máscaras y su interacción con gradientes.
"""

from __future__ import annotations

import torch
from torch import nn

from pia.models.resnet_cifar import apply_he_init, build_resnet18_cifar
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import (
    select_conv_weight_params,
    select_imp_weight_params,
)


def test_registry_selects_only_conv_weights() -> None:
    modelo = nn.Sequential(
        nn.Conv2d(3, 2, 1),
        nn.BatchNorm2d(2),
        nn.Conv2d(2, 2, 1),
        nn.Linear(2, 1),
    )
    pares = select_conv_weight_params(modelo)
    nombres = [n for n, _ in pares]
    assert nombres == ["0.weight", "2.weight"]
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    assert reg.total_elements() == modelo[0].weight.numel() + modelo[2].weight.numel()


def test_apply_to_weights_zeros_masked_positions() -> None:
    modelo = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Conv2d(2, 2, 1))
    apply_he_init(modelo)
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    reg.masks_dict()["0.weight"].reshape(-1)[0] = False
    reg.apply_to_weights(modelo)
    assert modelo[0].weight.reshape(-1)[0].item() == 0.0


def test_grad_hook_masks_pruned_positions() -> None:
    modelo = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False))
    modelo[0].weight.data.fill_(1.0)
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    reg.masks_dict()["0.weight"].fill_(True)
    reg.masks_dict()["0.weight"].reshape(-1)[0] = False
    reg.register_grad_hooks(modelo)
    try:
        loss = modelo(torch.zeros(1, 1, 1, 1)).sum()
        loss.backward()
        g = modelo[0].weight.grad
        assert g is not None
        assert float(g.reshape(-1)[0].item()) == 0.0
    finally:
        reg.remove_grad_hooks()


def test_imp_selector_includes_linear_and_excludes_bn() -> None:
    modelo = build_resnet18_cifar(num_classes=10)
    nombres = [n for n, _ in select_imp_weight_params(modelo)]
    assert "fc.weight" in nombres
    assert "conv1.weight" in nombres
    assert not any("bn" in n.lower() and n.endswith(".weight") for n in nombres)
    assert not any(".bias" in n for n in nombres)


def test_imp_selector_exclude_conv1_fc() -> None:
    modelo = build_resnet18_cifar(num_classes=10)
    nombres = [
        n
        for n, _ in select_imp_weight_params(
            modelo, exclude_conv1=True, exclude_fc=True
        )
    ]
    assert "conv1.weight" not in nombres
    assert "fc.weight" not in nombres
    assert any("layer1" in n for n in nombres)


def test_current_sparsity_matches_mask() -> None:
    modelo = nn.Sequential(nn.Conv2d(1, 4, 1), nn.Conv2d(4, 2, 1))
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    m0 = reg.masks_dict()["0.weight"]
    m0.reshape(-1)[:3] = False
    esperado = 3 / reg.total_elements()
    assert abs(reg.current_sparsity() - esperado) < 1e-6
