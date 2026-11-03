#!/bin/bash
# Submit the reference-table array, sized from config.sh.
#   bash illex/abc/talapas/submit.sh            # submit
#   DRYRUN=1 bash illex/abc/talapas/submit.sh   # print the sbatch command only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.sh"

read -ra _m <<<"$TABLE_MODELS"; read -ra _f <<<"$TABLE_FLUXES"; read -ra _c <<<"$TABLE_CLASSES"
NCOMBO=$(( ${#_m[@]} * ${#_f[@]} * ${#_c[@]} ))
NTASKS=$(( NCOMBO * TASKS_PER_COMBO ))
mkdir -p logs

CMD=(sbatch
     --array="0-$((NTASKS - 1))%${MAX_CONCURRENT_TASKS}"
     --account="$ACCOUNT"
     --partition="$PARTITION"
     --cpus-per-task="$CPUS_PER_TASK"
     --mem="$MEM"
     --time="$WALLTIME"
     "$HERE/array.sbatch")

echo "$NCOMBO combinations x $TASKS_PER_COMBO tasks = $NTASKS array tasks; K=$K_PER_TASK per task"
echo "OUTROOT=$OUTROOT"
echo "${CMD[*]}"
if [[ -n "${DRYRUN:-}" ]]; then
    echo "(dry run; not submitted)"
    exit 0
fi
"${CMD[@]}"
