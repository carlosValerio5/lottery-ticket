"""
Pruebas del criterio de parada por rebote de ``val/loss``.
"""

from __future__ import annotations

import math

import pytest

from pia.training.early_stopping import val_loss_rebound_early_stop_step


def test_first_epoch_always_updates_best() -> None:
    """Un primer valor finito siempre sustituye a ``inf`` como mejor pérdida."""
    b, s, stop = val_loss_rebound_early_stop_step(
        2.0,
        math.inf,
        relative_margin=0.1,
        patience=1,
        min_delta=0.0,
        bad_epochs=0,
    )
    assert b == 2.0
    assert s == 0
    assert not stop


def test_rebound_streak_then_stop() -> None:
    """Dos épocas por encima del umbral con paciencia 2 disparan la parada."""
    b, s, stop = val_loss_rebound_early_stop_step(
        1.2,
        1.0,
        relative_margin=0.1,
        patience=2,
        min_delta=0.0,
        bad_epochs=0,
    )
    assert b == 1.0
    assert s == 1
    assert not stop
    b2, s2, stop2 = val_loss_rebound_early_stop_step(
        1.25,
        b,
        relative_margin=0.1,
        patience=2,
        min_delta=0.0,
        bad_epochs=s,
    )
    assert b2 == 1.0
    assert s2 == 2
    assert stop2


def test_below_relative_band_resets_streak() -> None:
    """Pérdida dentro de la banda (no rebote fuerte) reinicia la racha."""
    b, s, stop = val_loss_rebound_early_stop_step(
        1.15,
        1.0,
        relative_margin=0.2,
        patience=1,
        min_delta=0.0,
        bad_epochs=1,
    )
    assert b == 1.0
    assert s == 0
    assert not stop


def test_min_delta_resets_streak_without_new_best() -> None:
    """Sin mejora >= min_delta y sin superar el umbral relativo, la racha vuelve a 0."""
    b, s, stop = val_loss_rebound_early_stop_step(
        0.99,
        1.0,
        relative_margin=0.5,
        patience=5,
        min_delta=0.02,
        bad_epochs=3,
    )
    assert b == 1.0
    assert s == 0
    assert not stop


def test_patience_one_stops_immediately_after_rebound() -> None:
    """Con margen 0, un empeoramiento estricto frente al best para en 1 época."""
    _, s, stop = val_loss_rebound_early_stop_step(
        1.5,
        1.0,
        relative_margin=0.0,
        patience=1,
        min_delta=0.0,
        bad_epochs=0,
    )
    assert s == 1
    assert stop


def test_structured_ticket_rejects_negative_relative() -> None:
    """La API de poda estructurada rechaza márgenes relativos negativos."""
    from pia.pruning.structured_ticket import iterative_structured_magnitude_pruning

    with pytest.raises(ValueError, match="early_stopping_val_loss_relative"):
        iterative_structured_magnitude_pruning(
            run_dir="/tmp/should_not_run",
            num_rounds=0,
            prune_per_round=0.2,
            epochs_per_round=1,
            lambda_weight=0.0,
            gamma_activation=0.0,
            data_root="unused",
            batch_size=4,
            lr=1e-3,
            device=None,
            early_stopping_val_loss_relative=-0.01,
            custom_model=None,
            custom_train_loader=None,
            custom_val_loader=None,
        )
