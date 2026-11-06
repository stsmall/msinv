#!/usr/bin/env bash
# Older barriers (875-1000 ky), lower plateaus (p_hist 0.56-0.70) and recent fall ends (50-150 ka), 48 reps per
# cell, then recovery and calibrated intervals on ABCDEFG with trajectory variance from each best cell.
cd "$(dirname "$0")/../.."
PY=${PY:-../inversion_sims/files/.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] _pbootG"
$PY -m illex.scripts.refit_decline --reps 48 --workers ${W:-80} --tag _pbootG --seed-offset 900000000 \
    --t-inv 875000,900000,925000,950000,975000,1000000 --p-hist 0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70 \
    --t-decline 50000,75000,100000,150000 --t-fall 50000,100000,150000,200000 \
    > results/illex/refit_decline_pbootG.log 2>&1 || { echo "[$(date)] _pbootG FAILED"; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for t in 850 825; do
  $PY -m illex.scripts.param_recovery --tags ABCDEFG --boot 400 --traj results/illex/stochastic_check_traj$t.json \
      > results/illex/param_recovery_ABCDEFG_traj$t.log 2>&1
  for x in json txt; do /bin/mv -f results/illex/param_recovery_ABCDEFG_traj.$x results/illex/param_recovery_ABCDEFG_traj$t.$x; done
done
echo "[$(date)] DONE"
