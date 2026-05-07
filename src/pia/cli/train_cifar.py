"""
Entrada CLI para entrenar en CIFAR-10 con ``SparseLoss`` y observadores.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from pia.observability.logging_config import setup_pia_logging
from pia.training.grid_search import grid_search, run_single_training

_log = logging.getLogger("pia.cli")


def main() -> None:
    """Parsea argumentos y lanza un run o una rejilla de hiperparámetros."""
    parser = argparse.ArgumentParser(
        description="Entrenar ResNet CIFAR con SparseLoss.",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--logdir", type=str, default="./runs")
    parser.add_argument("--run-name", type=str, default="single")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-weight", type=float, default=1e-5)
    parser.add_argument("--gamma-activation", type=float, default=1e-5)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=[1e-5, 1e-4],
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="*",
        default=[1e-5, 1e-4],
    )
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
        help="Lanza Streamlit en segundo plano apuntando al directorio del run.",
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
    base = Path(args.logdir)
    if args.grid:
        if args.spawn_dashboard:
            _log.warning("--spawn-dashboard no aplica con --grid; se omite.")
        _log.info("Iniciando grid_search en %s", base)
        grid_search(
            base_logdir=base,
            lambda_weights=args.lambdas,
            gamma_activations=args.gammas,
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
        )
    else:
        run_dir = base / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.spawn_dashboard:
            app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
            if not app.is_file():
                _log.warning("No se encontró %s; no se lanza el dashboard.", app)
            else:
                env = os.environ.copy()
                env["PIA_STREAMLIT_DASHBOARD"] = "1"
                env["PIA_RUN_DIR"] = str(run_dir.resolve())
                log_path = (
                    Path(args.dashboard_log.strip()).expanduser()
                    if args.dashboard_log.strip()
                    else run_dir / "dashboard_streamlit.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
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
                    log_f = log_path.open("a", encoding="utf-8")
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
                        log_path.resolve(),
                    )
                except OSError as exc:
                    _log.warning("No se pudo lanzar Streamlit: %s", exc)
        _log.info("Iniciando run_single_training en %s", run_dir)
        run_single_training(
            run_dir=run_dir,
            lambda_weight=args.lambda_weight,
            gamma_activation=args.gamma_activation,
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
