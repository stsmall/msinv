#!/bin/bash
# Submit the inversion-ABC array, sizing it from config.sh.
#   bash illex/slim/submit.sh            # submit
#   DRYRUN=1 bash illex/slim/submit.sh   # print the sbatch command only
#
# Mirrors 14_sweep_seqmodel/scripts/harness/talapas/submit.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/config.sh"

NTASKS=$(( N_SIMS / CHUNK ))
if (( NTASKS < 1 )); then
    echo "N_SIMS ($N_SIMS) must be >= CHUNK ($CHUNK)" >&2
    exit 2
fi

mkdir -p logs

CMD=(sbatch
     --array="0-$((NTASKS - 1))%${MAX_CONCURRENT_TASKS}"
     --account="$ACCOUNT"
     --partition="$PARTITION"
     --cpus-per-task="$CPUS_PER_TASK"
     --mem="$MEM"
     --time="$WALLTIME"
     "$HERE/submit_talapas.sbatch")

echo "N_SIMS=$N_SIMS CHUNK=$CHUNK -> $NTASKS array tasks (throttle %${MAX_CONCURRENT_TASKS})"
echo "Q=$QSCALE  OUTROOT=$OUTROOT"
echo "${CMD[*]}"

if [[ -n "${DRYRUN:-}" ]]; then
    echo "(dry run; not submitted)"
    exit 0
fi
"${CMD[@]}"
