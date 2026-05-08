#!/usr/bin/env bash
# Matriz compacta sugerida para barrer hiperparámetros IMP (plan de mejora).
# Uso desde la raíz del repo: bash scripts/imp_compact_sweep.sh
# Requiere: pipenv, datos CIFAR en ./data

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

run() {
  local name="$1"
  shift
  echo "=== $name ==="
  pipenv run python -m pia.cli.lottery_ticket --output-dir ./runs/lt_sweep --run-name "$name" "$@"
}

# 8 corridas de alta señal (ajusta flags según CLI actual)
run "s01_lam0_g0_p01" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.1 \
  --lambda-weight 0 --gamma-activation 0 --rewind-mode theta0
run "s02_lam0_g0_p02" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.2 \
  --lambda-weight 0 --gamma-activation 0 --rewind-mode theta0
run "s03_latek4_p01" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.1 \
  --lambda-weight 1e-6 --gamma-activation 1e-6 --rewind-mode late_k --rewind-epoch-k 4
run "s04_latek4_p02" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.2 \
  --lambda-weight 1e-6 --gamma-activation 1e-6 --rewind-mode late_k --rewind-epoch-k 4
run "s05_lr5e4_cos" --num-rounds 2 --epochs-per-round 15 --prune-per-round 0.1 \
  --lr 5e-4 --lr-scheduler cosine --lambda-weight 1e-6 --gamma-activation 0 \
  --rewind-mode late_k --rewind-epoch-k 3
run "s06_exclude_fc" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.1 \
  --exclude-fc-from-pruning --lambda-weight 1e-6 --gamma-activation 1e-6 \
  --rewind-mode late_k --rewind-epoch-k 4
run "s07_wl1_mean" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.1 \
  --lambda-weight 1e-3 --gamma-activation 1e-6 --weight-l1-aggregation mean \
  --rewind-mode late_k --rewind-epoch-k 4
run "s08_rewind_none" --num-rounds 2 --epochs-per-round 10 --prune-per-round 0.1 \
  --lambda-weight 1e-6 --gamma-activation 0 --rewind-mode none

echo "Listo. Revisa runs/lt_sweep/*/imp_index.json"
