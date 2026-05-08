# Línea base IMP: run `large-epoch`

Referencias tomadas de `runs/lt/large-epoch` (IMP con caída fuerte tras la primera poda).

## Métricas por ronda (resumen)

| Ronda | val/acc (aprox.) | train/acc (aprox.) | achieved_sparsity (máscara) |
|-------|------------------|--------------------|-----------------------------|
| 00    | 0.795            | 0.827              | ~0.02                       |
| 01    | 0.596            | 0.828              | ~0.04                       |

## Criterios de mejora (pass/fail)

- **Objetivo principal:** tras la primera poda (ronda 01), `Δ val/acc = val_acc_r01 - val_acc_r00` debe ser **≥ −0.05** (no más de 5 puntos de caída respecto a la ronda densa).
- **Estabilidad:** en una misma ronda, no más de **15 puntos** de caída consecutiva de `val/acc` entre épocas (misma heurística que `val_acc_drop_warn` en entrenamiento).
- **Sparsidad comparable:** comparar runs con el mismo `prune_per_round` y `num_rounds`, o reportar `achieved_sparsity` explícitamente.

## Notas

- `train/weight_sparsity_ratio` mide fracción global de pesos con \|w\| &lt; ε (no es la fracción de máscara IMP).
- `pruning/mask_sparsity` es la fracción de ceros en tensores bajo máscara IMP.
