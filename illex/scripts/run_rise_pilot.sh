#!/usr/bin/env bash
# 2a (AA, p=0.626) as the derived inverted arrangement: broad pilot over barrier age, founding frequency
# (0 = single copy rising from t_inv), arrival time, rise duration and gene flux.
cd "$(dirname "$0")/../.."
PY=${PY:-../inversion_sims/files/.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] start"
$PY -m illex.scripts.refit_rise --reps 16 --workers ${W:-40} --tag _rise_pilot --seed-offset 2100000000 \
    --t-inv 850000,1000000,1300000,1700000,2200000,3000000 \
    --p-start 0,0.05,0.15,0.3,0.45 --t-arrive 0,100000,300000 --t-rise 50000,200000,500000 \
    --gamma 1e-15,1e-8,1e-7,1e-6,1e-5 > results/illex/refit_rise_pilot.log 2>&1
echo "[$(date)] exit $?"
