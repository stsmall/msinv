#!/bin/bash
# Run the SLiM-vs-msinv inversion benchmark locally with a small, fixed number of cores.
#   bash illex/slim_bench/run_local.sh            # defaults below
#   JOBS=8 REPS=200 CHUNK=10 OUT=results/slim_bench bash illex/slim_bench/run_local.sh
# Each task runs CHUNK replicates of one scenario on one core (OMP/BLAS threads = 1).
# Requires: SLiM >= 5.0 (SLIM_BIN), and msinv with the post-barrier fix importable
# from the repo root (run from the repo root).
set -uo pipefail   # a failed task must not stop the comparison step
JOBS=${JOBS:-8}
REPS=${REPS:-200}
CHUNK=${CHUNK:-10}
OUT=${OUT:-results/slim_bench}
PY=${PY:-python}
SLIM_BIN=${SLIM_BIN:-$(command -v slim)}
SCENARIOS=${SCENARIOS:-"short_barrier long_barrier low_recomb illex_like"}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/logs"
"$SLIM_BIN" -v | head -1
NT=$(( REPS / CHUNK ))
for sc in $SCENARIOS; do for t in $(seq 0 $((NT - 1))); do echo "$sc $t"; done; done |
  xargs -P "$JOBS" -L 1 bash -c '"$0" -m illex.slim_bench.run_bench --scenario "$1" --task-id "$2" \
      --reps '"$CHUNK"' --out '"$OUT"' --slim '"$SLIM_BIN"' > '"$OUT"'/logs/"$1"_"$2".log 2>&1' "$PY"
"$PY" -m illex.slim_bench.compare_bench "$OUT"
