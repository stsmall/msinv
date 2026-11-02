#!/usr/bin/env bash
# Fill the low-plateau gap left by _pbootG: p_hist 0.56-0.60 at t_inv 775-850 ky (t_decline 100-250 ka) and at
# 875-1000 ky with t_decline 200-250 ka, 48 reps per cell, then recovery and calibrated intervals on ABCDEFGHI (H and I) with
# trajectory variance from each best cell.
cd "$(dirname "$0")/../.."
PY=${PY:-../inversion_sims/files/.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] _pbootH"
$PY -m illex.scripts.refit_decline --reps 48 --workers ${W:-80} --tag _pbootH --seed-offset 1000000000 \
    --t-inv 775000,800000,825000,850000 --p-hist 0.56,0.58,0.60 \
    --t-decline 100000,150000,200000,250000 --t-fall 50000,100000,150000,200000 \
    > results/illex/refit_decline_pbootH.log 2>&1 || { echo "[$(date)] _pbootH FAILED"; exit 1; }
echo "[$(date)] _pbootI"
$PY -m illex.scripts.refit_decline --reps 48 --workers ${W:-80} --tag _pbootI --seed-offset 1100000000 \
    --t-inv 875000,900000,925000,950000,975000,1000000 --p-hist 0.56,0.58,0.60 \
    --t-decline 200000,250000 --t-fall 50000,100000,150000,200000 \
    > results/illex/refit_decline_pbootI.log 2>&1 || { echo "[$(date)] _pbootI FAILED"; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for t in 850 825; do
  $PY -m illex.scripts.param_recovery --tags ABCDEFGHI --boot 400 --traj results/illex/stochastic_check_traj$t.json \
      > results/illex/param_recovery_ABCDEFGHI_traj$t.log 2>&1
  for x in json txt; do /bin/mv -f results/illex/param_recovery_ABCDEFGHI_traj.$x results/illex/param_recovery_ABCDEFGHI_traj$t.$x; done
done
echo "[$(date)] DONE"
