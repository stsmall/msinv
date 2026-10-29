#!/usr/bin/env bash
# Rebuild the manuscript's final inversion grid (supplement "Fitting and profiles"), 48 reps per cell, with
# per-replicate output, for the parametric bootstrap. The original grid_extension*.json were lost with a scratchpad.
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=24
run(){ echo "[$(date)] $1"; $PY -m illex.scripts.refit_decline --reps 48 --workers $W --tag "$1" \
        --t-inv "$2" --p-hist "$3" --t-decline "$4" --t-fall "$5" > results/illex/refit_decline$1.log 2>&1 || echo "[$(date)] $1 FAILED"; }
run _pbootA 800000,850000,900000 0.70,0.74,0.78,0.82 100000,175000,250000 50000,100000,200000
run _pbootB 700000,750000 0.70,0.74 100000,175000,250000 50000,100000,200000
run _pbootC 800000,850000 0.70,0.74 0,25000,50000 50000,100000,200000
run _pbootD 800000,850000 0.70,0.74 100000,175000,250000 400000,600000
echo "[$(date)] DONE"
