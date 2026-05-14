"""
Pruebas del flujo de poda estructurada por ancho y de ``NarrowableChainCnn``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pia.models.narrowable_chain_cnn import NarrowableChainCnn
from pia.models.resnet_cifar import apply_he_init
from pia.pruning.prune_structured import (
    chain_cnn_channel_l1_scores,
    count_parameters,
    narrow_chain_cnn_by_fraction,
)
from pia.pruning.structured_ticket import iterative_structured_magnitude_pruning


def _tiny_loaders() -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], ...]:
    torch.manual_seed(1)
    x = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 2, (16,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def test_chain_cnn_channel_scores_shape() -> None:
    m = NarrowableChainCnn(in_channels=1, width=5, spatial_hw=(8, 8), num_classes=2)
    s = chain_cnn_channel_l1_scores(m)
    assert s.shape == (5,)
    assert (s >= 0).all()


def test_narrow_chain_reduces_width_and_params() -> None:
    m = NarrowableChainCnn(in_channels=1, width=12, spatial_hw=(8, 8), num_classes=2)
    apply_he_init(m)
    n0 = count_parameters(m)
    m2, info = narrow_chain_cnn_by_fraction(m, 0.25)
    assert info["width_before"] == 12
    assert info["width_after"] == 9
    assert m2.width == 9
    assert count_parameters(m2) < n0
    x = torch.randn(2, 1, 8, 8)
    y = m(x)
    y2 = m2(x)
    assert y.shape == (2, 2)
    assert y2.shape == (2, 2)


def test_iterative_structured_index_and_rounds(tmp_path: Path) -> None:
    run_dir = tmp_path / "struct"
    train_ld, val_ld = _tiny_loaders()
    torch.manual_seed(42)
    modelo = NarrowableChainCnn(1, 16, (8, 8), 2)
    apply_he_init(modelo)
    iterative_structured_magnitude_pruning(
        run_dir=run_dir,
        num_rounds=2,
        prune_per_round=0.25,
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
    assert (run_dir / "theta_0.pt").is_file()
    for sub in ("round_00", "round_01", "round_02"):
        assert (run_dir / sub / "model_state.pt").is_file()
    assert (run_dir / "model_final.pt").is_file()
    raw = json.loads((run_dir / "structured_index.json").read_text(encoding="utf-8"))
    assert raw["run_status"] == "complete"
    assert raw["meta"]["pipeline"] == "structured_width_chain_cnn"
    data = raw["rounds"]
    assert len(data) == 3
    widths = [int(r["structured_conv_width"]) for r in data]
    assert widths[0] == 12
    assert widths[1] == 9
    assert widths[2] == 9
    for i in range(1, len(widths)):
        assert widths[i] <= widths[i - 1]
    params = [int(r["structured_param_count"]) for r in data]
    assert params[1] <= params[0]
    assert params[2] <= params[1]


def test_rewind_mode_non_none_rejected() -> None:
    run_dir = Path("/tmp/should_not_write_struct")
    train_ld, val_ld = _tiny_loaders()
    m = NarrowableChainCnn(1, 4, (8, 8), 2)
    with pytest.raises(ValueError, match="solo admite rewind_mode"):
        iterative_structured_magnitude_pruning(
            run_dir=run_dir,
            num_rounds=0,
            prune_per_round=0.2,
            epochs_per_round=1,
            lambda_weight=0.0,
            gamma_activation=0.0,
            data_root="unused",
            batch_size=4,
            lr=1e-2,
            device=torch.device("cpu"),
            acc_floor=0.0,
            custom_model=m,
            custom_train_loader=train_ld,
            custom_val_loader=val_ld,
            rewind_mode="theta0",
        )


def test_resnet_path_rejected() -> None:
    with pytest.raises(ValueError, match="NarrowableChainCnn"):
        iterative_structured_magnitude_pruning(
            run_dir=Path("/tmp/no"),
            num_rounds=0,
            prune_per_round=0.2,
            epochs_per_round=1,
            lambda_weight=0.0,
            gamma_activation=0.0,
            data_root="./data",
            batch_size=4,
            lr=1e-3,
            device=torch.device("cpu"),
            acc_floor=0.0,
        )
