"""
Pruebas de integración del flujo IMP con modelo pequeño.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pia.models.resnet_cifar import apply_he_init
from pia.pruning.lottery_ticket import iterative_magnitude_pruning
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import prune_globally_by_magnitude, select_conv_weight_params


class _TinyImpNet(nn.Module):
    """Red mínima con ``layer2``/``layer3`` para ``ActivationCapture``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, 3, padding=1, bias=False)
        self.fc = nn.Linear(2 * 8 * 8, 2)
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.layer2(x)
        x = torch.relu(self.conv2(x))
        x = self.layer3(x)
        return self.fc(x.reshape(x.size(0), -1))


def _tiny_loaders() -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], ...]:
    torch.manual_seed(1)
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 2, (16,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def test_theta_zero_snapshot_and_reset_fidelity(tmp_path: Path) -> None:
    modelo = _TinyImpNet()
    apply_he_init(modelo)
    theta = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    reg = WeightMaskRegistry.from_model(modelo, select_conv_weight_params)
    prune_globally_by_magnitude(reg, modelo, 0.2)
    modelo.load_state_dict({k: v.clone() for k, v in theta.items()})
    reg.apply_to_weights(modelo)
    for nombre, m in reg.masks_dict().items():
        p = dict(modelo.named_parameters())[nombre]
        t0 = theta[nombre]
        assert torch.allclose(p * m.to(dtype=p.dtype), p)
        assert torch.allclose(p[m], t0[m])
        assert float(p[~m].abs().sum().item()) == 0.0


def test_imp_index_monotonic_and_theta_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "imp"
    train_ld, val_ld = _tiny_loaders()
    torch.manual_seed(42)
    modelo = _TinyImpNet()
    apply_he_init(modelo)
    theta_pre = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    iterative_magnitude_pruning(
        run_dir=run_dir,
        num_rounds=2,
        prune_per_round=0.2,
        epochs_per_round=1,
        lambda_weight=0.0,
        gamma_activation=0.0,
        data_root="unused",
        batch_size=4,
        lr=1e-2,
        num_workers=0,
        device=torch.device("cpu"),
        acc_floor=0.0,
        custom_model=modelo,
        custom_train_loader=train_ld,
        custom_val_loader=val_ld,
    )
    theta_path = run_dir / "theta_0.pt"
    assert theta_path.is_file()
    try:
        theta_disk = torch.load(theta_path, map_location="cpu", weights_only=True)
    except TypeError:
        theta_disk = torch.load(theta_path, map_location="cpu")
    for k in theta_pre:
        assert torch.allclose(theta_pre[k], theta_disk[k])
    for sub in ("round_00", "round_01", "round_02"):
        assert (run_dir / sub / "model_state.pt").is_file()
    assert (run_dir / "model_final.pt").is_file()
    indice_path = run_dir / "imp_index.json"
    raw = json.loads(indice_path.read_text(encoding="utf-8"))
    data = raw["rounds"] if isinstance(raw, dict) and "rounds" in raw else raw
    assert len(data) == 3
    prev = -1.0
    for fila in data:
        cur = float(fila["achieved_sparsity"])
        assert cur >= prev - 1e-6
        prev = cur
