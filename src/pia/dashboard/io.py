"""
Lectura de artefactos JSON/JSONL generados durante entrenamiento y poda IMP.

Se centraliza el parseo tolerante para que las apps Streamlit puedan refrescar
en caliente aunque un archivo esté en escritura por otro proceso.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def es_directorio_run_imp(run_path: Path) -> bool:
    """
    Devuelve True si la ruta parece la raíz de un experimento IMP (lottery ticket).

    Los JSONL de entrenamiento están en subcarpetas ``round_XX/``, no en la raíz.
    """
    if not run_path.is_dir():
        return False
    if (run_path / "imp_index.json").is_file():
        return True
    return any(p.is_dir() for p in run_path.glob("round_*"))


def load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Carga eventos desde un archivo JSONL.

    Ignora líneas vacías y líneas que no decodifican como JSON (p. ej. la
    última línea truncada mientras otro proceso escribe el archivo).

    Args:
        path: Ruta a ``events.jsonl``.

    Returns:
        Lista ordenada de diccionarios (orden de aparición en el archivo).
    """
    if not path.is_file():
        return []
    texto = path.read_text(encoding="utf-8")
    filas: list[dict[str, Any]] = []
    for linea in texto.splitlines():
        s = linea.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            filas.append(obj)
    return filas


def load_imp_index(path: Path) -> list[dict[str, Any]]:
    """
    Carga el índice de rondas IMP desde ``imp_index.json``.

    Args:
        path: Ruta al índice global producido por ``iterative_magnitude_pruning``.

    Returns:
        Lista de entradas por ronda; si el archivo no existe o no es válido,
        devuelve una lista vacía para que el dashboard no falle.
    """
    if not path.is_file():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        rounds = obj.get("rounds")
        if isinstance(rounds, list):
            return [r for r in rounds if isinstance(r, dict)]
    return []


def to_metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte filas de eventos en un ``DataFrame`` homogéneo.

    Args:
        rows: Salida de ``load_events_jsonl`` o ``load_imp_index``.

    Returns:
        DataFrame con columnas unión de todas las claves; ``epoch`` numérico si
        existe, orden ascendente por ``epoch``.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df = df.sort_values("epoch", kind="stable").reset_index(drop=True)
    return df
