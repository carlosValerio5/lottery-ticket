"""
Configuración de logging estructurado para el paquete ``pia``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serializa cada registro como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, Mapping):
            payload["extra"] = dict(extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_pia_logging(
    *,
    level: int = logging.INFO,
    json_file: str | None = None,
) -> logging.Logger:
    """
    Configura el logger ``pia`` con salida legible en stderr y JSON opcional.

    Args:
        level: Nivel mínimo de log.
        json_file: Si se indica, escribe JSON línea a línea en ese archivo.

    Returns:
        Logger raíz del paquete.
    """
    log = logging.getLogger("pia")
    log.handlers.clear()
    log.setLevel(level)
    consola = logging.StreamHandler(sys.stderr)
    consola.setLevel(level)
    consola.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(consola)
    if json_file:
        archivo = logging.FileHandler(json_file, encoding="utf-8")
        archivo.setLevel(level)
        archivo.setFormatter(JsonFormatter())
        log.addHandler(archivo)
    log.propagate = False
    return log
