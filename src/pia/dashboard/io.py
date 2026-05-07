"""
Lectura de ``events.jsonl`` generado durante el entrenamiento (sin Streamlit).

Permite al dashboard y a las pruebas parsear líneas JSON de forma tolerante a
archivos aún en escritura (última línea incompleta).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def to_metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte filas de eventos en un ``DataFrame`` homogéneo.

    Args:
        rows: Salida de ``load_events_jsonl``.

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
