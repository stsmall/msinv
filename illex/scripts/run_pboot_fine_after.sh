#!/usr/bin/env bash
# After run_pboot_fine.sh: covariance-aware fit and recovery on the coarse grid alone and on coarse + fine.
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
until grep -q DONE results/illex/run_pboot_fine.log; do sleep 120; done
for t in ABCD ABCDE; do
  OMP_NUM_THREADS=8 $PY -m illex.scripts.param_recovery --tags $t --boot 400 > results/illex/param_recovery_$t.log 2>&1
  echo "[$(date)] recovery $t exit $?" >> results/illex/run_pboot_fine.log
done
