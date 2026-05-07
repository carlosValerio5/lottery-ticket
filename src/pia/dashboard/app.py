"""
Aplicación Streamlit: lectura periódica de ``events.jsonl`` y ``live_batches.jsonl``.

``live_batches.jsonl`` se actualiza **por batch** durante train y validación de
la época en curso; ``events.jsonl`` solo al cerrar cada época.

Requiere la variable de entorno ``PIA_STREAMLIT_DASHBOARD=1`` (u otros valores
aceptados) para evitar exponer la UI por accidente.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pia.dashboard.io import load_events_jsonl, to_metrics_dataframe

_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def _feature_enabled() -> bool:
    v = os.environ.get("PIA_STREAMLIT_DASHBOARD", "").strip().lower()
    return v in _FLAG_VALUES


def _default_run_dir() -> str:
    return os.environ.get("PIA_RUN_DIR", "./runs/single").strip() or "./runs/single"


def _metric_columns(df: pd.DataFrame) -> list[str]:
    """Columnas numéricas de métricas (excluye epoch y metadatos comunes)."""
    skip = {
        "epoch",
        "run_id",
        "data_root",
        "device",
        "git_sha",
    }
    out: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].dtype in ("float64", "int64", "float32", "int32") or str(
            df[c].dtype
        ).startswith("float"):
            out.append(str(c))
    return sorted(out)


def _render_intra_epoch(live_df: pd.DataFrame) -> None:
    """Gráficos por batch de la época en ejecución."""
    if "phase" not in live_df.columns:
        return
    st.divider()
    st.subheader("Intra-época en curso (cada batch)")
    if "epoch" in live_df.columns and len(live_df) > 0:
        st.caption(
            f"Mientras entrena la época **{int(live_df['epoch'].iloc[-1])}** "
            "(se actualiza al leer el disco cada pocos segundos)."
        )
    for phase, titulo in (
        ("train", "Train"),
        ("val", "Validación"),
    ):
        sub = live_df[live_df["phase"] == phase].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("batch", kind="stable")
        st.markdown(f"**{titulo}** — pérdida total y accuracy por batch")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(sub.set_index("batch")[["loss"]], height=260)
        with c2:
            st.line_chart(sub.set_index("batch")[["acc"]], height=260)
        with st.expander(f"CE (solo tarea) — {titulo}", expanded=False):
            st.line_chart(sub.set_index("batch")[["loss_task"]], height=220)


def _render_completed_epochs(df: pd.DataFrame) -> None:
    """Curvas agregadas por época ya cerradas."""
    if "epoch" not in df.columns:
        st.warning("El JSONL de épocas no contiene la columna `epoch`.")
        st.dataframe(df, use_container_width=True)
        return

    last = df.iloc[-1]
    idx = df.set_index("epoch")

    c1, c2, c3, c4 = st.columns(4)
    if "train/acc" in df.columns:
        c1.metric("train/acc (última época)", f"{float(last['train/acc']):.4f}")
    if "val/acc" in df.columns:
        c2.metric("val/acc (última época)", f"{float(last['val/acc']):.4f}")
    if "train/loss" in df.columns:
        c3.metric("train/loss (última época)", f"{float(last['train/loss']):.4f}")
    if "val/loss" in df.columns:
        c4.metric("val/loss (última época)", f"{float(last['val/loss']):.4f}")

    st.subheader("Accuracy por época (cerradas)")
    acc_cols = [c for c in ("train/acc", "val/acc") if c in df.columns]
    if acc_cols:
        st.line_chart(idx[acc_cols], height=320)
    else:
        st.caption("No hay columnas train/acc ni val/acc en el JSONL.")

    st.subheader("Pérdida total por época (cerradas)")
    loss_cols = [c for c in ("train/loss", "val/loss") if c in df.columns]
    if loss_cols:
        st.line_chart(idx[loss_cols], height=320)
    else:
        st.caption("No hay columnas train/loss ni val/loss en el JSONL.")

    task_cols = [c for c in ("train/loss_task", "val/loss_task") if c in df.columns]
    if task_cols:
        with st.expander("Pérdida solo CE (tarea) por época", expanded=False):
            st.line_chart(idx[task_cols], height=260)

    chart_cols = acc_cols + loss_cols
    conocidas = set(chart_cols) | set(task_cols)
    extra = [c for c in _metric_columns(df) if c not in conocidas]
    if extra:
        with st.expander("Más métricas por época"):
            st.line_chart(df.set_index("epoch")[extra])

    st.subheader("Tabla de eventos (épocas cerradas)")
    st.dataframe(df, use_container_width=True, height=min(400, 40 + 28 * len(df)))


def main() -> None:
    """Punto de entrada de la app Streamlit."""
    st.set_page_config(
        page_title="PIA — entrenamiento en vivo",
        layout="wide",
    )
    if not _feature_enabled():
        st.error(
            "El dashboard está desactivado. Exporta "
            "`PIA_STREAMLIT_DASHBOARD=1` (o `true` / `yes` / `on`) y vuelve a "
            "cargar la página."
        )
        st.stop()

    st.title("Seguimiento de entrenamiento en vivo")
    st.caption(
        "``live_batches.jsonl`` = métricas **por batch** de la época actual; "
        "``events.jsonl`` = resumen al **terminar** cada época."
    )

    run_dir_str = st.sidebar.text_input(
        "Directorio del run",
        value=_default_run_dir(),
        help="Carpeta con events.jsonl y live_batches.jsonl.",
    )
    poll_sec = st.sidebar.slider(
        "Mín. segundos entre lecturas en disco",
        min_value=1,
        max_value=10,
        value=2,
        help=(
            "El fragmento se ejecuta cada 1 s; la lectura de archivos respeta "
            "este mínimo (menor = gráficos más fluidos, más I/O)."
        ),
    )
    st.session_state["_poll_sec"] = int(poll_sec)
    run_path = Path(run_dir_str).expanduser().resolve()
    jsonl_path = run_path / "events.jsonl"
    live_path = run_path / "live_batches.jsonl"

    @st.fragment(run_every=timedelta(seconds=1))
    def live_panel() -> None:
        if "last_load_mono" not in st.session_state:
            st.session_state.last_load_mono = 0.0
        if "cached_events_df" not in st.session_state:
            st.session_state.cached_events_df = None
        if "cached_live_df" not in st.session_state:
            st.session_state.cached_live_df = None

        interval = float(st.session_state.get("_poll_sec", 2))
        now = time.monotonic()
        if now - st.session_state.last_load_mono >= interval:
            st.session_state.last_load_mono = now
            st.session_state.cached_events_df = to_metrics_dataframe(
                load_events_jsonl(jsonl_path)
            )
            st.session_state.cached_live_df = to_metrics_dataframe(
                load_events_jsonl(live_path)
            )

        events_df: pd.DataFrame | None = st.session_state.cached_events_df
        live_df: pd.DataFrame | None = st.session_state.cached_live_df

        ev_empty = events_df is None or events_df.empty
        live_empty = live_df is None or live_df.empty

        if ev_empty and live_empty:
            st.info(
                f"No hay datos aún en `{live_path}` ni en `{jsonl_path}`. "
                "Arranca el entrenamiento con este directorio de run."
            )
            return

        if not live_empty:
            _render_intra_epoch(live_df)

        if not ev_empty:
            _render_completed_epochs(events_df)
        elif not live_empty:
            st.caption(
                f"Cuando termine la época en curso aparecerán aquí las curvas "
                f"agregadas en `{jsonl_path.name}`."
            )

    live_panel()


if __name__ == "__main__":
    main()
