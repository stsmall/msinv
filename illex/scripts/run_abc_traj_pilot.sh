#!/usr/bin/env bash
# Pilot rejection ABC on the 2a trajectory: 2.4M prior paths, accepted at p_now 0.626 +- 0.02, then msinv.
cd "$(dirname "$0")/../.."
PY=${PY:-../inversion_sims/files/.venv/bin/python}
export PYTHONPATH=$PWD RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] start"
$PY -m illex.scripts.abc_traj_pilot --batches 600 --batch 4000 --workers ${W:-40} --tag _p1 > results/illex/abc_traj_p1.log 2>&1
echo "[$(date)] exit $?"
