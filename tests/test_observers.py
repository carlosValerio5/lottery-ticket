"""Pruebas de observadores de entrenamiento."""

from __future__ import annotations

from pathlib import Path

from pia.training.observers import (
    CompositeObserver,
    CsvMetricsObserver,
    JsonlEventsObserver,
)


def test_csv_and_jsonl_write_rows(tmp_path: Path) -> None:
    csv_obs = CsvMetricsObserver(tmp_path)
    json_obs = JsonlEventsObserver(tmp_path)
    comp = CompositeObserver([csv_obs, json_obs])
    comp.on_train_begin({"run_id": "t1", "lambda_weight": 0.1})
    comp.on_epoch_end(1, {"train/loss": 1.0, "val/acc": 0.5})
    comp.on_train_end({})
    csv_text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert "epoch" in csv_text and "train/loss" in csv_text
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "val/acc" in lines[0]
