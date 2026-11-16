#!/bin/bash
# Full-length NPE training run (msinv #30), one SLURM array task.
#   FULL_MODE=bank : control_bank.py, BANK_PER_TASK full-length controls -> $FULL_OUTROOT/bank/ctl_<idx>.jsonl.gz
#   FULL_MODE=rows : run_full.py, K_PER_TASK accepted rows -> $FULL_OUTROOT/table/B_full_<idx>.jsonl.gz
# Submit with illex/abc/npe/submit_full.sh (it sets the array, account, partition, resources and the dependency).
#SBATCH --job-name=illex_full
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -euo pipefail
# shellcheck disable=SC1091
source "${REPO:?}/illex/slim/config.sh"
ENV=${FULL_ENV:?}
activate_env
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
idx=${SLURM_ARRAY_TASK_ID:?must run as a SLURM array}
cd "$REPO"
python -c "from msinv import InversionSpec; assert 'long_flux_rate' in InversionSpec.__dataclass_fields__" || {
    echo "FATAL: msinv in env $ENV has no long-tract flux process (#30); rebuild it from $REPO" >&2; exit 3; }
echo "[idx $idx] mode=$FULL_MODE host=$(hostname) $(git log --oneline -1) $(date)"
case "$FULL_MODE" in
    bank)
        if [[ -s "$FULL_OUTROOT/bank/ctl_$idx.jsonl.gz" ]]; then echo "exists, skipping"; exit 0; fi
        python -m illex.abc.control_bank --task-id "$idx" --n "$BANK_PER_TASK" --out "$FULL_OUTROOT/bank" \
            --workers "$SLURM_CPUS_PER_TASK"
        ;;
    rows)
        if [[ -s "$FULL_OUTROOT/table/B_full_$idx.jsonl.gz" ]]; then echo "exists, skipping"; exit 0; fi
        python -m illex.abc.run_full --task-id "$idx" --n-accept "$K_PER_TASK" --bank "$FULL_OUTROOT/bank" \
            --out "$FULL_OUTROOT/table" --workers "$SLURM_CPUS_PER_TASK" --save-ts-every "$SAVE_TS_EVERY"
        ;;
    *) echo "FATAL: FULL_MODE must be bank or rows" >&2; exit 2 ;;
esac
echo "[idx $idx] done $(date)"
