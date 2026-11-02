#!/usr/bin/env bash
# Finer grid around the optimum for the parametric bootstrap and recovery tests: t_inv 775-900 ky in 25-ky steps,
# p_hist 0.68-0.78 in 0.02 steps, t_decline 100-250 ka and t_fall 50-200 ky in 50-ky steps, 48 reps per cell.
# Seeds are unique per cell (--seed-offset), so cells that repeat the coarse grid add independent replicates.
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] _pbootE"
$PY -m illex.scripts.refit_decline --reps 48 --workers ${W:-40} --tag _pbootE --seed-offset 700000000 \
    --t-inv 775000,800000,825000,850000,875000,900000 --p-hist 0.68,0.70,0.72,0.74,0.76,0.78 \
    --t-decline 100000,150000,200000,250000 --t-fall 50000,100000,150000,200000 \
    > results/illex/refit_decline_pbootE.log 2>&1 || echo "[$(date)] _pbootE FAILED"
echo "[$(date)] DONE"
