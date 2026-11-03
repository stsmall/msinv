#!/bin/bash
# One SLURM array task = one (model, flux, class, task_id). Array index -> combination:
#   combo = idx / TASKS_PER_COMBO (over MODELS x FLUXES x CLASSES, model slowest), task_id = idx % TASKS_PER_COMBO.
# Submit via submit.sh (sets --array/--account/--partition/resources from config.sh).
#SBATCH --job-name=illex_traj_abc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.sh"
activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

idx=${SLURM_ARRAY_TASK_ID:?must run as a SLURM array}
combos=()
for m in $TABLE_MODELS; do for f in $TABLE_FLUXES; do for c in $TABLE_CLASSES; do combos+=("$m $f $c"); done; done; done
read -r M F C <<<"${combos[$((idx / TASKS_PER_COMBO))]}"
TID=$((idx % TASKS_PER_COMBO))

mkdir -p "$OUTROOT/table" logs
cd "$REPO"
echo "[idx $idx] model=$M flux=$F class=$C task_id=$TID K=$K_PER_TASK workers=$CPUS_PER_TASK host=$(hostname) $(date)"
python -m illex.abc.run_table --model "$M" --flux "$F" --class "$C" --n-accept "$K_PER_TASK" --task-id "$TID" \
    --out "$OUTROOT/table" --workers "$CPUS_PER_TASK" --max-drawn "$MAX_DRAWN"
echo "[idx $idx] done $(date)"
