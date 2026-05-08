"""Pruebas de lectura JSONL del dashboard (sin Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pia.dashboard.io import (
    es_directorio_run_imp,
    load_events_jsonl,
    load_imp_index,
    to_metrics_dataframe,
)


def test_load_events_jsonl_skips_bad_and_truncated(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    filas = [
        {"epoch": 1, "train/acc": 0.1, "val/acc": 0.2},
        {"epoch": 2, "train/acc": 0.3, "val/acc": 0.4},
    ]
    incompleto = '\n{"epoch": 3, "incompleto"'
    contenido = "\n".join(json.dumps(x) for x in filas) + incompleto
    p.write_text(contenido, encoding="utf-8")
    out = load_events_jsonl(p)
    assert len(out) == 2
    assert out[0]["epoch"] == 1
    assert out[1]["epoch"] == 2


def test_load_events_jsonl_missing_file(tmp_path: Path) -> None:
    assert load_events_jsonl(tmp_path / "no.jsonl") == []


def test_load_live_batches_format(tmp_path: Path) -> None:
    """Formato intra-época (train/val por batch)."""
    p = tmp_path / "live_batches.jsonl"
    lineas = [
        {
            "phase": "train",
            "epoch": 1,
            "batch": 1,
            "loss": 2.0,
            "loss_task": 1.9,
            "acc": 0.1,
        },
        {
            "phase": "train",
            "epoch": 1,
            "batch": 2,
            "loss": 1.8,
            "loss_task": 1.7,
            "acc": 0.2,
        },
        {
            "phase": "val",
            "epoch": 1,
            "batch": 1,
            "loss": 1.5,
            "loss_task": 1.4,
            "acc": 0.3,
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lineas) + "\n", encoding="utf-8")
    df = to_metrics_dataframe(load_events_jsonl(p))
    assert len(df) == 3
    assert set(df["phase"]) == {"train", "val"}


def test_es_directorio_run_imp(tmp_path: Path) -> None:
    assert not es_directorio_run_imp(tmp_path / "no_existe")
    normal = tmp_path / "single"
    normal.mkdir()
    (normal / "events.jsonl").write_text("{}\n", encoding="utf-8")
    assert not es_directorio_run_imp(normal)
    imp = tmp_path / "lt"
    imp.mkdir()
    (imp / "round_00").mkdir()
    assert es_directorio_run_imp(imp)
    (imp / "imp_index.json").write_text("[]", encoding="utf-8")
    assert es_directorio_run_imp(imp)


def test_load_imp_index_schema_v2(tmp_path: Path) -> None:
    p = tmp_path / "imp_index.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_status": "complete",
                "rounds": [{"round": 0, "status": "complete"}],
            }
        ),
        encoding="utf-8",
    )
    rows = load_imp_index(p)
    assert len(rows) == 1
    assert rows[0]["round"] == 0


def test_to_metrics_dataframe_sorts_epoch(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"epoch": 2, "x": 1}\n{"epoch": 1, "x": 0}\n',
        encoding="utf-8",
    )
    df = to_metrics_dataframe(load_events_jsonl(p))
    assert isinstance(df, pd.DataFrame)
    assert list(df["epoch"]) == [1, 2]
