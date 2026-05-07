"""
Inferencia por lotes con medición de latencia y eventos de log estructurados.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_log = logging.getLogger("pia.inference")


def _entropy_from_logits(logits: Tensor) -> float:
    """Entropía media de la distribución softmax por muestra (nats)."""
    p = F.softmax(logits, dim=1).clamp_min(1e-12)
    h = -(p * p.log()).sum(dim=1).mean()
    return float(h.item())


def run_inference_batch(
    model: nn.Module,
    x: Tensor,
    *,
    logger: logging.Logger | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """
    Ejecuta un forward en evaluación y emite un evento de observabilidad.

    Args:
        model: Red en ``eval`` (no se modifica el modo aquí).
        x: Lote de entrada ya en el dispositivo adecuado o CPU.
        logger: Logger para JSON; si es ``None``, usa ``pia.inference``.
        device: Si se indica, mueve ``x`` y el modelo antes del forward.

    Returns:
        Tupla ``(logits, stats)`` donde ``stats`` incluye latencia y entropía.
    """
    log = logger or _log
    prev_training = model.training
    model.eval()
    if device is not None:
        model.to(device)
        x = x.to(device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        if prev_training:
            model.train()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats: dict[str, Any] = {
        "event": "inference_batch",
        "batch_size": int(x.shape[0]),
        "latency_ms": float(elapsed_ms),
        "device": str(x.device),
        "mean_entropy_nats": _entropy_from_logits(logits),
    }
    if not torch.isfinite(logits).all():
        log.error("logits_no_finite %s", json.dumps(stats))
        raise RuntimeError("logits_non_finite")
    log.info("inference_batch %s", json.dumps(stats, ensure_ascii=False))
    return logits, stats
