#!/usr/bin/env bash
# Lower plateau values (p_hist 0.62-0.66) on the same t_inv, t_decline and t_fall grid as _pbootE, 48 reps per
# cell, then recovery and calibrated intervals on ABCDEF with trajectory variance from each best cell.
cd "$(dirname "$0")/../.."
PY=${PY:-../inversion_sims/files/.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] _pbootF"
$PY -m illex.scripts.refit_decline --reps 48 --workers ${W:-40} --tag _pbootF --seed-offset 800000000 \
    --t-inv 775000,800000,825000,850000,875000,900000 --p-hist 0.62,0.64,0.66 \
    --t-decline 100000,150000,200000,250000 --t-fall 50000,100000,150000,200000 \
    > results/illex/refit_decline_pbootF.log 2>&1 || { echo "[$(date)] _pbootF FAILED"; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for t in 850 825; do
  $PY -m illex.scripts.param_recovery --tags ABCDEF --boot 400 --traj results/illex/stochastic_check_traj$t.json \
      > results/illex/param_recovery_ABCDEF_traj$t.log 2>&1
  for x in json txt; do /bin/mv -f results/illex/param_recovery_ABCDEF_traj.$x results/illex/param_recovery_ABCDEF_traj$t.$x; done
done
echo "[$(date)] DONE"
