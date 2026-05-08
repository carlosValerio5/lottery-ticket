"""
Gestión del proceso hijo ``streamlit run`` lanzado en segundo plano.

Sin registro explícito, al interrumpir el entrenamiento (p. ej. Ctrl+C) el
servidor Streamlit sigue vivo en otra sesión y mantiene el puerto ocupado.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import sys
from types import FrameType
from typing import Any, TextIO

_dashboard_proc: subprocess.Popen[Any] | None = None
_dashboard_log: TextIO | None = None
_atexit_registrado = False
_sigterm_registrado = False


def _terminar_arbol_streamlit() -> None:
    """Envía señal al grupo de procesos del dashboard y cierra el log."""
    global _dashboard_proc, _dashboard_log
    proc = _dashboard_proc
    log_f = _dashboard_log
    _dashboard_proc = None
    _dashboard_log = None
    if proc is not None and proc.poll() is None:
        if sys.platform == "win32":
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
    if log_f is not None:
        with contextlib.suppress(OSError):
            log_f.close()


def registrar_dashboard_streamlit(
    proc: subprocess.Popen[Any] | None,
    log_f: TextIO | None,
) -> None:
    """
    Guarda referencias al hijo y programa limpieza al salir o ante SIGTERM.

    Args:
        proc: Resultado de ``Popen`` para ``streamlit run``, o ``None``.
        log_f: Fichero abierto para stdout/stderr del hijo, o ``None``.
    """
    global _dashboard_proc, _dashboard_log, _atexit_registrado, _sigterm_registrado
    _dashboard_proc = proc
    _dashboard_log = log_f
    if proc is None:
        return
    if not _atexit_registrado:
        atexit.register(_terminar_arbol_streamlit)
        _atexit_registrado = True
    if not _sigterm_registrado and sys.platform != "win32":
        _sigterm_registrado = True

        def _sigterm(_signum: int, _frame: FrameType | None) -> None:
            _terminar_arbol_streamlit()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _sigterm)


def terminar_dashboard_streamlit_si_hay() -> None:
    """
    Termina el Streamlit registrado (p. ej. desde un bloque ``finally``).

    Es idempotente: llamadas repetidas no fallan.
    """
    _terminar_arbol_streamlit()
