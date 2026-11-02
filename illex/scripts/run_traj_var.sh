#!/usr/bin/env bash
# Trajectory variance for the parametric bootstrap: deterministic vs Wright-Fisher drift at two best cells.
PY=${PY:-.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -m illex.scripts.stochastic_check --reps 384 --workers 24 --versions drift --tag _traj850 \
    > results/illex/stochastic_check_traj850.log 2>&1
$PY -m illex.scripts.stochastic_check --reps 384 --workers 24 --versions drift --tag _traj825 \
    --t-inv 825000 --p-hist 0.74 --t-decline 100000 --t-fall 200000 --seed 20261101 \
    > results/illex/stochastic_check_traj825.log 2>&1
echo "[$(date)] done" >> results/illex/stochastic_check_traj825.log
