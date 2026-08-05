#!/bin/bash
#SBATCH --job-name=illex_inv_abc
#SBATCH --array=1-500
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/abc_%A_%a.out
#SBATCH --error=logs/abc_%A_%a.err
#
# Talapas (University of Oregon, SLURM) array driver for the Illex inversion ABC.
#
# YOU MUST SET --account and --partition -- Talapas rejects jobs without an
# account. Either uncomment and edit the lines below, or pass them on the
# command line, which overrides anything in this file:
#
#   sbatch --account=<your_pirg> --partition=compute illex/slim/submit_talapas.sbatch
#
##SBATCH --account=CHANGE_ME
##SBATCH --partition=compute
#
# One array task runs $REPS replicates serially. 500 tasks x 4 reps = 2000
# simulations. Scale the array size, not the per-task rep count, to add
# simulations -- shorter tasks queue faster and lose less on preemption.
#
# Each task writes its own TSV (results/abc/sims_taskNNNNNN.tsv); nothing is
# appended to a shared file, so there is no cross-task race. Concatenate with
# collect.sh when the array finishes.

set -euo pipefail

REPS=${REPS:-4}
QSCALE=${QSCALE:-200}
OUTDIR=${OUTDIR:-results/abc}
REPO=${REPO:-$(pwd)}

# --- environment ------------------------------------------------------------
# Adjust to Talapas's module names; `module avail slim` / `module avail python`
# will show what is actually installed. If SLiM is not available as a module,
# build it once into $HOME and set SLIM_BIN to the binary.
module purge
module load slim    2>/dev/null || echo "NOTE: no 'slim' module; using SLIM_BIN"
module load python3 2>/dev/null || true

SLIM_BIN=${SLIM_BIN:-$(command -v slim || echo "$HOME/bin/slim")}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}

if [[ ! -x "$SLIM_BIN" ]]; then
    echo "FATAL: SLiM binary not found at '$SLIM_BIN'. Set SLIM_BIN." >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: python not found at '$PYTHON'. Set PYTHON." >&2
    exit 1
fi

# Node-local scratch for the intermediate .trees files. These are deleted after
# each replicate is summarized, but they must not go to shared storage -- 2000
# of them on a network filesystem will hurt everyone on the cluster.
SCRATCH=${SLURM_SCRATCH:-${TMPDIR:-/tmp}}/illex_abc_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$SCRATCH" logs "$OUTDIR"
trap 'rm -rf "$SCRATCH"' EXIT

cd "$REPO"
echo "task=${SLURM_ARRAY_TASK_ID:-0} reps=$REPS Q=$QSCALE slim=$SLIM_BIN"
"$SLIM_BIN" -v | head -1 || true

# Run as a module: the venv's msinv.pth points at a stale path, so `illex` is
# only importable with the repo root on sys.path.
exec "$PYTHON" -m illex.slim.run_one \
    --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --reps "$REPS" \
    --Q "$QSCALE" \
    --out-dir "$OUTDIR" \
    --slim "$SLIM_BIN" \
    --scratch "$SCRATCH"
