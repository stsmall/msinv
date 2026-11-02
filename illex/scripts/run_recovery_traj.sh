#!/usr/bin/env bash
# Recovery and calibrated intervals with trajectory variance from each best cell (850 ky / 100-ky fall and
# 825 ky / 200-ky fall), ABCDE grid.
PY=${PY:-.venv/bin/python}
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for t in 850 825; do
  $PY -m illex.scripts.param_recovery --tags ABCDE --boot 400 --traj results/illex/stochastic_check_traj$t.json \
      > results/illex/param_recovery_ABCDE_traj$t.log 2>&1
  for x in json txt; do /bin/mv -f results/illex/param_recovery_ABCDE_traj.$x results/illex/param_recovery_ABCDE_traj$t.$x; done
done
echo "[$(date)] done" >> results/illex/param_recovery_ABCDE_traj825.log
