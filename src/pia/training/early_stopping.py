"""
Parada anticipada por rebotes de val/loss frente al mejor valor visto.

Se extrae del bucle ``fit`` para poder probar la regla sin montar un
entrenamiento completo y para documentar en un solo lugar el criterio
relativo más racha.
"""

from __future__ import annotations


def val_loss_rebound_early_stop_step(
    val_loss: float,
    best_val_loss: float,
    *,
    relative_margin: float,
    patience: int,
    min_delta: float,
    bad_epochs: int,
) -> tuple[float, int, bool]:
    """
    Actualiza el mejor ``val/loss`` y decide si cortar por rebote sostenido.

    Tras alcanzar un mínimo (mejor valor hasta la época actual), considera parar
    cuando la pérdida supera ``(1 + relative_margin) * best`` durante ``patience``
    épocas consecutivas. Si la pérdida vuelve a la banda aceptable sin batir el
    mínimo, la racha se reinicia.

    Args:
        val_loss: Media de ``val/loss`` en la época actual.
        best_val_loss: Mejor ``val/loss`` observado hasta la época anterior
            (usar ``math.inf`` antes de la primera época).
        relative_margin: Umbral relativo ``r >= 0``; dispara si
            ``val_loss > best * (1 + r)``.
        patience: Épocas consecutivas por encima del umbral para parar (>= 1).
        min_delta: Mejora mínima respecto al mejor para registrar un nuevo mínimo.
        bad_epochs: Contador de épocas consecutivas ya acumuladas en la racha.

    Returns:
        Tupla ``(nuevo_mejor, nueva_racha, debe_parar)``. Si hay mejora clara,
        ``nuevo_mejor`` es ``val_loss`` y la racha vuelve a 0.
    """
    if val_loss < best_val_loss - min_delta:
        return val_loss, 0, False
    if val_loss > best_val_loss * (1.0 + relative_margin):
        nueva = bad_epochs + 1
        return best_val_loss, nueva, nueva >= patience
    return best_val_loss, 0, False
