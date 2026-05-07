"""Cableado mínimo de entrenamiento sin descargar CIFAR-10."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import pia.training.grid_search as gs


def _fake_loaders(*_a: object, **_kw: object) -> tuple[DataLoader, DataLoader]:
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8), DataLoader(ds, batch_size=8)


def test_run_single_training_smoke(tmp_path: Path) -> None:
    with patch.object(gs, "build_cifar10_loaders", _fake_loaders):
        ultimo = gs.run_single_training(
            run_dir=tmp_path / "r0",
            lambda_weight=1e-6,
            gamma_activation=1e-6,
            data_root=str(tmp_path),
            epochs=1,
            batch_size=8,
            lr=1e-3,
            num_workers=0,
        )
    assert "val/acc" in ultimo
    assert (tmp_path / "r0" / "summary.json").is_file()
