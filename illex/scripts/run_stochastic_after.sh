#!/usr/bin/env bash
# After the fine grid and its recovery runs: stochastic-trajectory check at the best cell (96 reps per version).
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
until grep -q "recovery ABCDE" results/illex/run_pboot_fine.log; do sleep 120; done
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m illex.scripts.stochastic_check --reps 96 --workers 24 > results/illex/stochastic_check.log 2>&1
echo "[$(date)] stochastic check exit $?" >> results/illex/stochastic_check.log
