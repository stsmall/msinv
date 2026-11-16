#!/bin/bash
# Full-length NPE training run on Talapas (msinv #30): balancing model, soft starts, short gene conversion plus a
# long-tract exchange process, the whole 18.96-Mb inversion per simulation (illex/abc/fulllen.py).
#
# Two arrays, the second waiting for the first:
#   1. bank: BANK_TASKS x BANK_PER_TASK full-length (45 Mb) neutral controls   -> $FULL_OUTROOT/bank/
#   2. rows: ROW_TASKS x K_PER_TASK accepted training rows                      -> $FULL_OUTROOT/table/
#
#   REPO=/gpfs/projects/kernlab/ssmall/msinv_illex_full bash illex/abc/npe/submit_full.sh
#   DRYRUN=1 ... bash illex/abc/npe/submit_full.sh       # preflight + print the sbatch commands only
#   STAGE=rows ... bash illex/abc/npe/submit_full.sh     # rows only (bank already complete)
#
# REPO must be a checkout of branch illex/fulllength with msinv built from it into FULL_ENV
# (ABC_ENV=$FULL_ENV REPO=... BRANCH=illex/fulllength bash illex/abc/talapas/install_env.sh). Use a separate env
# from the 40-kb runs, because `pip install -e` points the env's msinv at this checkout.
#
# Sizing: see the "Full-length run" section of illex/abc/talapas/TALAPAS_FULL.md. Each array task is resumable: a
# finished task file is skipped on resubmission, and a killed rows task keeps its finished rows in a .partial file.
set -euo pipefail
export REPO=${REPO:-/gpfs/projects/kernlab/$USER/msinv_illex_full}
export FULL_ENV=${FULL_ENV:-illex_full}
export FULL_OUTROOT=${FULL_OUTROOT:-/gpfs/projects/kernlab/$USER/illex_full}
export BANK_TASKS=${BANK_TASKS:-100}
export BANK_PER_TASK=${BANK_PER_TASK:-10}
export BANK_CPUS=${BANK_CPUS:-5}
export BANK_MEM=${BANK_MEM:-60G}
export BANK_WALLTIME=${BANK_WALLTIME:-06:00:00}
export ROW_TASKS=${ROW_TASKS:-10000}
export K_PER_TASK=${K_PER_TASK:-10}
export ROW_CPUS=${ROW_CPUS:-5}
export ROW_MEM=${ROW_MEM:-100G}
export ROW_WALLTIME=${ROW_WALLTIME:-08:00:00}
export MAX_CONCURRENT=${MAX_CONCURRENT:-200}
export SAVE_TS_EVERY=${SAVE_TS_EVERY:-500}
STAGE=${STAGE:-all}

# shellcheck disable=SC1091
source "$REPO/illex/slim/config.sh"
export ENV=$FULL_ENV
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M
for v in $(compgen -v | grep -E '^CONDA_PREFIX_[0-9]+$' || true); do unset "$v"; done
( activate_env && cd "$REPO" && PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" python - <<'EOF'
import msinv, illex.theory as T, illex.abc.constants as C
from msinv import InversionSpec
from illex.abc.error_model import LD_ERROR
from illex.abc.fulllen import FULL_BODY_BP, FULL_CONTROL_BP, LONG_PRIOR
ok = (round(T.N_ANC) == 507288 and round(T.N0) == 9112696 and round(T.T_GROW) == 382957
      and float(f"{C.REC_RATE:.4g}") == 5.507e-10 and LD_ERROR["control"]["I"] == (0.6768, 0.6645)
      and C.P_NOW == 0.602 and C.N_KARYO == 619 and "long_flux_rate" in InversionSpec.__dataclass_fields__)
print("msinv", msinv.__version__, msinv.__file__)
print("baseline", T.N_ANC, T.N0, T.T_GROW, C.REC_RATE, "body", FULL_BODY_BP, "control", FULL_CONTROL_BP,
      "long prior", LONG_PRIOR)
raise SystemExit(0 if ok else "FATAL: REPO/env is not illex/fulllength on the base2 baseline with msinv #30")
EOF
)
git -C "$REPO" log --oneline -1
git -C "$REPO" diff --quiet || { echo "FATAL: $REPO has uncommitted changes" >&2; exit 1; }
mkdir -p "$FULL_OUTROOT/bank" "$FULL_OUTROOT/table" "$REPO/logs"
SB="$REPO/illex/abc/talapas/full.sbatch"
BANK_CMD=(sbatch --parsable --array="0-$((BANK_TASKS - 1))%${MAX_CONCURRENT}" --account="$ACCOUNT"
          --partition="$PARTITION" --cpus-per-task="$BANK_CPUS" --mem="$BANK_MEM" --time="$BANK_WALLTIME"
          --job-name=illex_full_bank --export="ALL,FULL_MODE=bank" "$SB")
ROW_CMD=(sbatch --parsable --array="0-$((ROW_TASKS - 1))%${MAX_CONCURRENT}" --account="$ACCOUNT"
         --partition="$PARTITION" --cpus-per-task="$ROW_CPUS" --mem="$ROW_MEM" --time="$ROW_WALLTIME"
         --job-name=illex_full_rows --export="ALL,FULL_MODE=rows" "$SB")
echo "bank: $BANK_TASKS tasks x $BANK_PER_TASK controls; rows: $ROW_TASKS tasks x $K_PER_TASK rows"
echo "FULL_OUTROOT=$FULL_OUTROOT"
echo "${BANK_CMD[*]}"
echo "${ROW_CMD[*]} (after the bank)"
if [[ -n "${DRYRUN:-}" ]]; then echo "(dry run; not submitted)"; exit 0; fi
cd "$REPO"
if [[ "$STAGE" == all ]]; then
    BJ=$("${BANK_CMD[@]}")
    echo "bank job $BJ"
    RJ=$("${ROW_CMD[@]:0:2}" --dependency="afterok:$BJ" "${ROW_CMD[@]:2}")
elif [[ "$STAGE" == bank ]]; then
    "${BANK_CMD[@]}"; exit 0
else
    RJ=$("${ROW_CMD[@]}")
fi
echo "rows job $RJ"
echo "rows land in $FULL_OUTROOT/table/B_full_*.jsonl.gz; bring table/ and bank/ back to the box for collect + train."
