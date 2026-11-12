#!/usr/bin/env bash
# Run the parametric bootstrap once run_pboot_grid.sh has finished all four grids.
cd "$(dirname "$0")/../.."
until grep -q DONE results/illex/run_pboot_grid.log; do sleep 120; done
OMP_NUM_THREADS=8 ${PY:-.venv/bin/python} -m illex.scripts.param_bootstrap --boot 2000 > results/illex/param_bootstrap.log 2>&1
echo "[$(date)] bootstrap exit $?" >> results/illex/param_bootstrap.log
