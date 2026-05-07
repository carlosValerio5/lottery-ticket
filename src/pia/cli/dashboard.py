"""
Lanza la app Streamlit del dashboard si el feature flag de entorno está activo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def _flag_ok() -> bool:
    return os.environ.get("PIA_STREAMLIT_DASHBOARD", "").strip().lower() in _FLAG_VALUES


def _app_path() -> Path:
    return Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def main() -> None:
    """Valida el flag, construye el entorno y ejecuta ``streamlit run``."""
    if not _flag_ok():
        print(
            "Error: define PIA_STREAMLIT_DASHBOARD=1 (o true/yes/on) para habilitar "
            "el dashboard.",
            file=sys.stderr,
        )
        sys.exit(1)
    parser = argparse.ArgumentParser(
        description="Dashboard Streamlit para events.jsonl en vivo.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=os.environ.get("PIA_RUN_DIR", "./runs/single"),
        help="Directorio del run (events.jsonl).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Puerto del servidor Streamlit.",
    )
    args = parser.parse_args()
    run_resolved = Path(args.run_dir).expanduser().resolve()
    env = os.environ.copy()
    env["PIA_STREAMLIT_DASHBOARD"] = "1"
    env["PIA_RUN_DIR"] = str(run_resolved)
    app = _app_path()
    if not app.is_file():
        print(f"Error: no existe la app en {app}", file=sys.stderr)
        sys.exit(1)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
    ]
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
