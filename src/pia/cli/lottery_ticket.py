"""
Entrada CLI para poda iterativa por magnitud con reinicio (lottery ticket / IMP).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import subprocess
import sys
from pathlib import Path

from pia.cli.streamlit_child import (
    registrar_dashboard_streamlit,
    terminar_dashboard_streamlit_si_hay,
)
from pia.observability.logging_config import setup_pia_logging
from pia.pruning.lottery_ticket import iterative_magnitude_pruning

_log = logging.getLogger("pia.cli")


def main() -> None:
    """Parsea argumentos y ejecuta ``iterative_magnitude_pruning``."""
    parser = argparse.ArgumentParser(
        description=(
            "IMP (lottery ticket) con ResNet-18 CIFAR: poda global, máscaras y "
            "reinicio configurable (θ₀, late_k o ninguno)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/lt",
        help="Directorio base donde se crea --run-name.",
    )
    parser.add_argument("--run-name", type=str, default="imp_default")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--prune-per-round", type=float, default=0.2)
    parser.add_argument("--epochs-per-round", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-weight", type=float, default=1e-5)
    parser.add_argument("--gamma-activation", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--json-log",
        type=str,
        default="",
        help="Ruta opcional para log JSON del paquete pia.",
    )
    parser.add_argument(
        "--spawn-dashboard",
        action="store_true",
        help=(
            "Lanza el dashboard de lottery ticket en segundo plano "
            "apuntando al directorio del run."
        ),
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Puerto del dashboard si --spawn-dashboard.",
    )
    parser.add_argument(
        "--dashboard-log",
        type=str,
        default="",
        help=(
            "Ruta del log de Streamlit al usar --spawn-dashboard (stdout+stderr). "
            "Por defecto: <run_dir>/dashboard_streamlit.log"
        ),
    )
    parser.add_argument(
        "--rewind-mode",
        type=str,
        default="theta0",
        choices=("theta0", "late_k", "none"),
        help=("Reinicio tras poda: θ₀, checkpoint época k en ronda 0, o sin reinicio."),
    )
    parser.add_argument(
        "--rewind-epoch-k",
        type=int,
        default=4,
        help="Época 1-based en ronda 0 para late_k (debe ser <= epochs-per-round).",
    )
    parser.add_argument(
        "--exclude-conv1-from-pruning",
        action="store_true",
        help="No podar conv1.weight (stem de alto impacto).",
    )
    parser.add_argument(
        "--exclude-fc-from-pruning",
        action="store_true",
        help="No podar fc.weight (cabezal de clasificación).",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=("none", "cosine", "step"),
        help="Programación de LR por ronda (tras cada época).",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=10,
        help="StepLR: periodo en épocas (mínimo 1).",
    )
    parser.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.1,
        help="StepLR: factor multiplicativo.",
    )
    parser.add_argument(
        "--weight-l1-aggregation",
        type=str,
        default="sum",
        choices=("sum", "mean", "mean_per_param"),
        help="Agregación L1 de pesos en SparseLoss: sum, mean o mean_per_param.",
    )
    parser.add_argument(
        "--abort-on-val-acc-drop",
        type=float,
        default=-1.0,
        help=(
            "Si >= 0, detiene la fase actual si val/acc cae más que este delta "
            "entre épocas consecutivas."
        ),
    )
    args = parser.parse_args()
    log_path = args.json_log.strip() or None
    setup_pia_logging(json_file=log_path)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.spawn_dashboard:
        app = (
            Path(__file__).resolve().parent.parent
            / "dashboard"
            / "lottery_ticket_app.py"
        )
        if not app.is_file():
            _log.warning("No se encontró %s; no se lanza el dashboard.", app)
        else:
            env = os.environ.copy()
            env["PIA_STREAMLIT_DASHBOARD"] = "1"
            env["PIA_RUN_DIR"] = str(run_dir.resolve())
            dashboard_log_path = (
                Path(args.dashboard_log.strip()).expanduser()
                if args.dashboard_log.strip()
                else run_dir / "dashboard_streamlit.log"
            )
            dashboard_log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app),
                "--server.port",
                str(args.dashboard_port),
                "--logger.level",
                "info",
            ]
            proc = None
            log_f = None
            try:
                log_f = dashboard_log_path.open("a", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                registrar_dashboard_streamlit(proc, log_f)
                _log.info(
                    "Dashboard IMP: %s (no abras pia.cli.dashboard / app.py con esta "
                    "carpeta; son para train_cifar). PIA_RUN_DIR=%s URL: "
                    "http://localhost:%s — log Streamlit: %s",
                    app.resolve(),
                    env["PIA_RUN_DIR"],
                    args.dashboard_port,
                    dashboard_log_path.resolve(),
                )
            except OSError as exc:
                _log.warning("No se pudo lanzar Streamlit: %s", exc)
                if log_f is not None:
                    with contextlib.suppress(OSError):
                        log_f.close()
    _log.info("Iniciando IMP en %s", run_dir)
    try:
        abort_drop: float | None = (
            None if args.abort_on_val_acc_drop < 0 else args.abort_on_val_acc_drop
        )
        iterative_magnitude_pruning(
            run_dir=run_dir,
            num_rounds=args.num_rounds,
            prune_per_round=args.prune_per_round,
            epochs_per_round=args.epochs_per_round,
            lambda_weight=args.lambda_weight,
            gamma_activation=args.gamma_activation,
            data_root=args.data_root,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            rewind_mode=args.rewind_mode,
            rewind_epoch_k=args.rewind_epoch_k,
            exclude_conv1_from_pruning=args.exclude_conv1_from_pruning,
            exclude_fc_from_pruning=args.exclude_fc_from_pruning,
            lr_scheduler_kind=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_step_gamma=args.lr_step_gamma,
            weight_l1_aggregation=args.weight_l1_aggregation,
            abort_training_on_val_acc_drop=abort_drop,
        )
    finally:
        terminar_dashboard_streamlit_si_hay()


if __name__ == "__main__":
    main()
