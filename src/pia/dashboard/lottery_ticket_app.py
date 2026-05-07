"""
Aplicación Streamlit para visualizar runs de lottery ticket (IMP).

Muestra progreso por ronda desde ``imp_index.json`` y permite inspeccionar
curvas por época de cada subdirectorio ``round_XX``.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st

from pia.dashboard.io import load_events_jsonl, load_imp_index, to_metrics_dataframe

_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def _feature_enabled() -> bool:
    """Indica si la UI está habilitada por variable de entorno explícita."""
    v = os.environ.get("PIA_STREAMLIT_DASHBOARD", "").strip().lower()
    return v in _FLAG_VALUES


def _default_run_dir() -> str:
    """Devuelve la ruta por defecto del run IMP desde entorno o fallback local."""
    return (
        os.environ.get("PIA_RUN_DIR", "./runs/lt/imp_default").strip()
        or "./runs/lt/imp_default"
    )


def _extract_imp_overview(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normaliza ``imp_index.json`` para análisis de sparsidad y accuracy final."""
    norm: list[dict[str, Any]] = []
    for r in rows:
        final = r.get("final_metrics")
        final_d = final if isinstance(final, dict) else {}
        norm.append(
            {
                "round": int(r.get("round", 0)),
                "target_sparsity": float(r.get("target_sparsity", 0.0)),
                "achieved_sparsity": float(r.get("achieved_sparsity", 0.0)),
                "val_acc": float(final_d.get("val/acc", float("nan"))),
                "train_acc": float(final_d.get("train/acc", float("nan"))),
                "val_loss": float(final_d.get("val/loss", float("nan"))),
                "run_dir": str(r.get("run_dir", "")),
            }
        )
    if not norm:
        return pd.DataFrame()
    return pd.DataFrame(norm).sort_values("round", kind="stable").reset_index(drop=True)


def _rounds_desde_directorios(run_path: Path) -> list[int]:
    """
    Descubre rondas existentes por subcarpetas ``round_XX`` (p. ej. durante un
    entrenamiento antes de que ``imp_index.json`` esté completo).
    """
    salida: list[int] = []
    for p in run_path.glob("round_*"):
        if not p.is_dir():
            continue
        partes = p.name.split("_", 1)
        if len(partes) < 2:
            continue
        try:
            salida.append(int(partes[1]))
        except ValueError:
            continue
    return sorted(set(salida))


def _chart_linea_por_batch(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    titulo: str,
    etiqueta_x: str,
    etiqueta_y: str,
    altura: int = 260,
) -> alt.Chart:
    """
    Gráfico de línea para métricas intra-época (eje X = batch o paso acumulado).

    Replica el estilo del dashboard principal para coherencia visual.
    """
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:Q", title=etiqueta_x),
            y=alt.Y(f"{y_col}:Q", title=etiqueta_y),
            tooltip=[x_col, y_col],
        )
        .properties(title=titulo, height=altura)
    )


def _line_chart(
    df: pd.DataFrame, x: str, y: str, title: str, y_label: str
) -> alt.Chart:
    """Construye un gráfico de línea simple para una métrica por ronda/época."""
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y_label), tooltip=[x, y]
        )
        .properties(title=title, height=260)
    )


def _render_round_overview(df: pd.DataFrame) -> None:
    """Renderiza resumen global del proceso IMP (sparsidad y calidad final)."""
    st.subheader("Resumen por ronda")
    if df.empty:
        st.info("No hay rondas en imp_index.json todavía.")
        return
    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ronda final", str(int(last["round"])))
    c2.metric("Sparsidad lograda", f"{float(last['achieved_sparsity']):.4f}")
    if pd.notna(last["val_acc"]):
        c3.metric("val/acc final", f"{float(last['val_acc']):.4f}")

    ch1 = (
        alt.Chart(df)
        .transform_fold(
            ["target_sparsity", "achieved_sparsity"], as_=["serie", "valor"]
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("round:Q", title="Ronda IMP"),
            y=alt.Y("valor:Q", title="Sparsidad"),
            color=alt.Color("serie:N", title="Curva"),
            tooltip=["round", "serie", "valor"],
        )
        .properties(title="Sparsidad objetivo vs lograda", height=300)
    )
    st.altair_chart(ch1, use_container_width=True)

    val_acc_col = df["val_acc"]
    if isinstance(val_acc_col, pd.Series) and bool(val_acc_col.notna().any()):
        mask_val = val_acc_col.notna()
        df_val = cast(pd.DataFrame, df.loc[mask_val])
        st.altair_chart(
            _line_chart(
                df_val,
                "round",
                "val_acc",
                "val/acc final por ronda",
                "val/acc",
            ),
            use_container_width=True,
        )

    st.dataframe(df, use_container_width=True)


def _render_live_batches_en_ronda(live_df: pd.DataFrame, round_id: int) -> None:
    """
    Gráficos por batch durante la ronda (``live_batches.jsonl``).

    El archivo se trunca al iniciar cada fase train de una época; las curvas
    reflejan el progreso en tiempo casi real de la época en curso.
    """
    if live_df.empty or "phase" not in live_df.columns:
        return
    st.divider()
    st.subheader(f"Ronda {round_id:02d} — métricas en vivo (por batch)")
    if "epoch" in live_df.columns and len(live_df) > 0:
        st.caption(
            f"Época en curso: **{int(live_df['epoch'].iloc[-1])}**. "
            "Los datos se refrescan leyendo el disco cada pocos segundos."
        )
    df_work = live_df.copy()
    if "epoch" in df_work.columns and "batch" in df_work.columns:
        orden_fase = {"train": 0, "val": 1}
        df_work["_ord_fase"] = (
            df_work["phase"]
            .astype(str)
            .replace(orden_fase)
            .fillna(99)
            .astype(int)
        )
        df_work = df_work.sort_values(
            by=["epoch", "_ord_fase", "batch"], kind="stable"
        ).reset_index(drop=True)
        df_work = df_work.drop(columns=["_ord_fase"], errors="ignore")
        df_work["paso_global"] = range(1, len(df_work) + 1)
        x_live = "paso_global"
        x_label = "Paso (train luego val, época actual)"
    else:
        x_live = "batch"
        x_label = "Número de batch (época actual)"

    for phase, titulo in (("train", "Train"), ("val", "Validación")):
        sub = cast(pd.DataFrame, df_work[df_work["phase"] == phase].copy())
        if sub.empty:
            continue
        sub = sub.sort_values(
            by=["epoch", "batch"] if "epoch" in sub.columns else ["batch"],
            kind="stable",
        )
        st.markdown(f"**{titulo}**")
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                _chart_linea_por_batch(
                    sub,
                    x_col=x_live,
                    y_col="loss",
                    titulo=f"Pérdida total por batch — {titulo}",
                    etiqueta_x=x_label,
                    etiqueta_y="Pérdida total",
                ),
                use_container_width=True,
            )
        with c2:
            st.altair_chart(
                _chart_linea_por_batch(
                    sub,
                    x_col=x_live,
                    y_col="acc",
                    titulo=f"Accuracy (top-1) — {titulo}",
                    etiqueta_x=x_label,
                    etiqueta_y="Accuracy",
                ),
                use_container_width=True,
            )
        if "loss_task" in sub.columns:
            with st.expander(
                f"Pérdida solo CE — {titulo}",
                expanded=False,
            ):
                st.altair_chart(
                    _chart_linea_por_batch(
                        sub,
                        x_col=x_live,
                        y_col="loss_task",
                        titulo=f"Pérdida CE — {titulo}",
                        etiqueta_x=x_label,
                        etiqueta_y="Entropía cruzada",
                        altura=220,
                    ),
                    use_container_width=True,
                )


def _render_selected_round(
    run_path: Path, round_id: int, live_df: pd.DataFrame
) -> None:
    """Muestra curvas en vivo y por época de una ronda (live + ``events.jsonl``)."""
    st.divider()
    round_dir = run_path / f"round_{round_id:02d}"
    st.subheader(f"Detalle de ronda {round_id:02d}")
    _render_live_batches_en_ronda(live_df, round_id)
    if live_df.empty:
        st.caption(
            f"Cuando arranque el entrenamiento aparecerán aquí curvas desde "
            f"``{round_dir / 'live_batches.jsonl'}``."
        )

    events = to_metrics_dataframe(load_events_jsonl(round_dir / "events.jsonl"))
    if events.empty:
        st.info(
            f"Aún no hay épocas cerradas en {round_dir / 'events.jsonl'}. "
            "Las curvas por época aparecerán al terminar cada época."
        )
        return
    if "epoch" not in events.columns:
        st.warning("events.jsonl no contiene columna epoch.")
        st.dataframe(events, use_container_width=True)
        return

    st.subheader("Épocas cerradas (events.jsonl)")
    cols = st.columns(2)
    if "val/acc" in events.columns:
        cols[0].altair_chart(
            _line_chart(
                events, "epoch", "val/acc", "Accuracy validación por época", "val/acc"
            ),
            use_container_width=True,
        )
    if "val/loss" in events.columns:
        cols[1].altair_chart(
            _line_chart(
                events, "epoch", "val/loss", "Pérdida validación por época", "val/loss"
            ),
            use_container_width=True,
        )

    series = [
        c
        for c in ("train/acc", "val/acc", "train/loss", "val/loss")
        if c in events.columns
    ]
    if series:
        pliegue: list[str | alt.FieldName] = list(series)
        ch = (
            alt.Chart(events)
            .transform_fold(pliegue, as_=["serie", "valor"])
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Época"),
                y=alt.Y("valor:Q", title="Valor"),
                color=alt.Color("serie:N", title="Métrica"),
                tooltip=["epoch", "serie", "valor"],
            )
            .properties(title="Curvas principales por época", height=320)
        )
        st.altair_chart(ch, use_container_width=True)


def main() -> None:
    """Punto de entrada Streamlit para monitorear entrenamientos IMP."""
    st.set_page_config(page_title="PIA — lottery ticket", layout="wide")
    if not _feature_enabled():
        st.error(
            "El dashboard está desactivado. Exporta PIA_STREAMLIT_DASHBOARD=1 "
            "(o true/yes/on) y recarga la página."
        )
        st.stop()

    st.title("Seguimiento IMP (lottery ticket)")
    st.caption(
        "``live_batches.jsonl`` = métricas por batch en la época actual; "
        "``events.jsonl`` = resumen al cerrar cada época; "
        "``imp_index.json`` = resumen por ronda."
    )

    run_dir_str = st.sidebar.text_input(
        "Directorio del run IMP", value=_default_run_dir()
    )
    poll_sec = st.sidebar.slider(
        "Segundos entre lecturas de imp_index.json",
        min_value=1,
        max_value=10,
        value=2,
        help=(
            "Las curvas en vivo (live_batches.jsonl) se actualizan cada ~1 s; "
            "este intervalo solo afecta al resumen global y a la tabla por ronda."
        ),
    )
    st.session_state["_imp_poll_sec"] = int(poll_sec)
    run_path = Path(run_dir_str).expanduser().resolve()
    imp_index_path = run_path / "imp_index.json"

    @st.fragment(run_every=timedelta(seconds=1))
    def live_panel() -> None:
        """Panel reactivo con caché temporal para evitar I/O excesivo."""
        if "imp_last_load_mono" not in st.session_state:
            st.session_state.imp_last_load_mono = 0.0
        if "imp_overview_df" not in st.session_state:
            st.session_state.imp_overview_df = pd.DataFrame()
        if "imp_live_df" not in st.session_state:
            st.session_state.imp_live_df = pd.DataFrame()
        if "imp_live_round" not in st.session_state:
            st.session_state.imp_live_round = -1
        if "imp_last_live_mono" not in st.session_state:
            st.session_state.imp_last_live_mono = 0.0

        interval = float(st.session_state.get("_imp_poll_sec", 2))
        now = time.monotonic()
        tick_index = now - st.session_state.imp_last_load_mono >= interval
        tick_live = now - st.session_state.imp_last_live_mono >= 1.0

        if tick_index:
            st.session_state.imp_last_load_mono = now
            rows = load_imp_index(imp_index_path)
            st.session_state.imp_overview_df = _extract_imp_overview(rows)

        df: pd.DataFrame = st.session_state.imp_overview_df
        rounds_fs = _rounds_desde_directorios(run_path)
        if not df.empty:
            rounds_idx = sorted(
                set(int(x) for x in df["round"].tolist()) | set(rounds_fs)
            )
        else:
            rounds_idx = rounds_fs

        if not df.empty:
            _render_round_overview(df)
        elif rounds_idx:
            st.info(
                f"Aún no hay entradas en {imp_index_path.name} o el índice va al día. "
                "Puedes seguir la ronda en vivo con ``live_batches.jsonl``."
            )
        else:
            st.info(
                f"No hay datos aún en {imp_index_path} ni carpetas round_XX. "
                "Ejecuta lottery_ticket para este run."
            )
            return

        if not rounds_idx:
            return

        # Los fragmentos no pueden crear widgets en ``st.sidebar`` (Streamlit >= 1.33).
        clave_sel = "lt_imp_round_choice"
        if clave_sel not in st.session_state:
            st.session_state[clave_sel] = int(rounds_idx[-1])
        elegido = int(st.session_state[clave_sel])
        if elegido not in rounds_idx:
            elegido = int(rounds_idx[-1])
            st.session_state[clave_sel] = elegido
        ix = rounds_idx.index(elegido)
        sel_i = int(
            st.selectbox(
                "Ronda a inspeccionar",
                options=rounds_idx,
                index=ix,
                help="Vista detallada y curvas en vivo de esta ronda IMP.",
            )
        )
        st.session_state[clave_sel] = sel_i
        live_path = run_path / f"round_{sel_i:02d}" / "live_batches.jsonl"
        round_changed = st.session_state.imp_live_round != sel_i
        if tick_live or round_changed:
            st.session_state.imp_last_live_mono = now
            st.session_state.imp_live_df = to_metrics_dataframe(
                load_events_jsonl(live_path)
            )
            st.session_state.imp_live_round = sel_i

        _render_selected_round(run_path, sel_i, st.session_state.imp_live_df)

    live_panel()


if __name__ == "__main__":
    main()
