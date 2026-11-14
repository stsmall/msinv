#!/bin/bash
# One array task = one slice of DRAWS_PER_TASK posterior draws of one flux setting for ppc_blocks.py.
# Array index -> (flux, slice): flux = FLUXES[idx / TASKS_PER_FLUX], slice = idx % TASKS_PER_FLUX.
# Submit with submit_ppc_blocks.sh (it sets the array, account, partition and resources).
#SBATCH --job-name=illex_ppc_blocks
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -euo pipefail
# shellcheck disable=SC1091
source "${REPO:?}/illex/slim/config.sh"
activate_env
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
idx=${SLURM_ARRAY_TASK_ID:?must run as a SLURM array}
read -ra FL <<<"$PPC_FLUXES"
F=${FL[$((idx / TASKS_PER_FLUX))]}
T=$((idx % TASKS_PER_FLUX))
OUT="$PPC_OUTROOT/blocks_f${F}_t$(printf %03d "$T").csv"
if [[ -s "$OUT" ]]; then echo "[idx $idx] $OUT exists, skipping"; exit 0; fi
cd "$REPO"
echo "[idx $idx] flux=$F slice=$T draws $((T * DRAWS_PER_TASK))-$(((T + 1) * DRAWS_PER_TASK - 1)) host=$(hostname) $(date)"
PYTHONPATH="$REPO" python illex/abc/npe/ppc_blocks.py --draws "results/illex/npe_karyofix/f$F/posterior_samples.csv" \
    --flux "$F" --n "$PPC_N" --first $((T * DRAWS_PER_TASK)) --count "$DRAWS_PER_TASK" \
    --procs "$SLURM_CPUS_PER_TASK" --out "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "[idx $idx] done $(date)"
