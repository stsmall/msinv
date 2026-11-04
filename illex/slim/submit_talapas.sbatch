#!/bin/bash
# One SLURM array task = CHUNK inversion simulations. Submit via submit.sh, which
# sets --array/--account/--partition/resources from config.sh. Manual:
#   sbatch --array=0-1999%50 --account=kernlab --partition=compute \
#          illex/slim/submit_talapas.sbatch
#
# Task 9 validation array (fixed parameters, one set per submission; not yet submitted):
#   VALIDATION_SET=soft_B N_SIMS=200 CHUNK=2 bash illex/slim/submit.sh
# (submit.sh forwards the environment; sbatch exports it by default.)
#
# Mirrors 14_sweep_seqmodel/scripts/harness/talapas/gen_array.sbatch.
#SBATCH --job-name=illex_inv_abc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.sh"

# Seed base distinct from the sweep campaign's 1e8/2e8/3e8 bases, so seeds can
# never collide across campaigns.
BASE=400000000

activate_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [[ ! -x "${SLIM_BIN:-}" ]]; then
    echo "FATAL: SLiM not found at '${SLIM_BIN:-}'. Set SLIM_BIN in config.sh." >&2
    exit 1
fi
"$SLIM_BIN" -v | head -1

t=${SLURM_ARRAY_TASK_ID:?must run as a SLURM array}

# Node-local scratch for intermediate .trees; deleted per replicate by run_one.py,
# and the whole dir on exit.
SCRATCH=${SLURM_SCRATCH:-${TMPDIR:-/tmp}}/illex_inv_${SLURM_ARRAY_JOB_ID:-0}_${t}
mkdir -p "$SCRATCH" logs "$OUTROOT/sims"
trap 'rm -rf "$SCRATCH"' EXIT

cd "$REPO"
echo "[task $t] chunk=$CHUNK Q=$QSCALE seed_base=$((BASE + t * CHUNK)) host=$(hostname) $(date)"

# Run as a module: `illex` is only importable with the repo root on sys.path.
if [[ -n "${VALIDATION_SET:-}" ]]; then
    # Task 9 SLiM validation: fixed parameters from the sets file (Q comes from the set, not QSCALE).
    # Seed base 4.5e8, distinct from the prior-draw campaign's 4e8.
    python -m illex.slim.run_one \
        --params "${VALIDATION_SETS:-results/illex/abc_slim_sets.json}" \
        --set "$VALIDATION_SET" \
        --task-id "$t" \
        --reps "$CHUNK" \
        --seed-base 450000000 \
        --out-dir "$OUTROOT/slim_validation" \
        --slim "$SLIM_BIN" \
        --scratch "$SCRATCH"
else
python -m illex.slim.run_one \
    --task-id "$t" \
    --reps "$CHUNK" \
    --Q "$QSCALE" \
    --seed-base "$BASE" \
    --out-dir "$OUTROOT/sims" \
    --slim "$SLIM_BIN" \
    --scratch "$SCRATCH"
fi

echo "[task $t] done $(date)"
