"""
Entrada CLI para poda iterativa por magnitud con reinicio (lottery ticket / IMP).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from pia.observability.logging_config import setup_pia_logging
from pia.pruning.lottery_ticket import iterative_magnitude_pruning

_log = logging.getLogger("pia.cli")


def main() -> None:
    """Parsea argumentos y ejecuta ``iterative_magnitude_pruning``."""
    parser = argparse.ArgumentParser(
        description="IMP (lottery ticket) con ResNet-18 CIFAR y reinicio a θ₀.",
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
            try:
                log_f = dashboard_log_path.open("a", encoding="utf-8")
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                _log.info(
                    "Dashboard Streamlit desacoplado (PIA_RUN_DIR=%s). "
                    "URL: http://localhost:%s — salida del servidor en %s",
                    env["PIA_RUN_DIR"],
                    args.dashboard_port,
                    dashboard_log_path.resolve(),
                )
            except OSError as exc:
                _log.warning("No se pudo lanzar Streamlit: %s", exc)
    _log.info("Iniciando IMP en %s", run_dir)
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
    )


if __name__ == "__main__":
    main()
