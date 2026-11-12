#!/bin/bash
# ABC-RF model-choice reference table on Talapas with the corrected karyotypes: N, D1, D2 and B, gene flux 1, hard and
# soft starts, 10 tasks x 250 accepted rows per combination (as results/illex/modelchoice_base2 on branch
# illex/trajectory-abc). Same preflight as submit_table.sh. Bring OUTROOT/table back to the box and collect with
# illex/abc/collect.py, then the model-choice part of illex/abc/abc_infer.R.
#
#   REPO=/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2 bash illex/abc/npe/submit_modelchoice.sh
#   DRYRUN=1 ... bash illex/abc/npe/submit_modelchoice.sh
set -euo pipefail
export REPO=${REPO:-/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2}
export ENV=${ENV:-illex_abc}
export ABC_OUTROOT=${ABC_OUTROOT:-/gpfs/projects/kernlab/$USER/illex_mc_karyofix}
export TABLE_MODELS=${TABLE_MODELS:-"N D1 D2 B"}
export TABLE_FLUXES=${TABLE_FLUXES:-1}
export TABLE_CLASSES=${TABLE_CLASSES:-"hard soft"}
export K_PER_TASK=${K_PER_TASK:-250}
export TASKS_PER_COMBO=${TASKS_PER_COMBO:-10}
export MAX_DRAWN=${MAX_DRAWN:-3000000}
export ABC_CPUS=${ABC_CPUS:-4}
export ABC_MEM=${ABC_MEM:-16G}
export ABC_WALLTIME=${ABC_WALLTIME:-03:00:00}
export ABC_MAX_CONCURRENT=${ABC_MAX_CONCURRENT:-60}

T="$REPO/illex/abc/talapas"
[[ -f "$T/submit.sh" && -f "$T/array.sbatch" ]] || { echo "FATAL: $T has no submit.sh / array.sbatch" >&2; exit 2; }
# shellcheck disable=SC1091
source "$T/config.sh"
export PYTHONNOUSERSITE=1   # ~/.local site-packages shadow the env's scipy on Talapas
# Submit from a clean shell. activate_env (here and in array.sbatch, which runs with --export=ALL) does
# `module load miniconda3; conda activate`; if the env is already active in the inherited environment, the module load
# puts base Python 3.13 first and `conda activate` is a no-op, so every task dies with
# "No module named msinv._msinv_core" (job 48451379). Drop any inherited conda state, and activate the env only in a
# subshell for the preflight, so submit.sh sees a shell with no env active.
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M
for v in $(compgen -v | grep -E '^CONDA_PREFIX_[0-9]+$' || true); do unset "$v"; done
( activate_env && cd "$REPO" && PYTHONPATH="$REPO" python - <<'EOF'
import illex.theory as T, illex.abc.constants as C
from illex.abc.smc import OBSERVED
from illex.abc.error_model import LD_ERROR
ok = (round(T.N_ANC) == 507288 and round(T.N0) == 9112696 and round(T.T_GROW) == 382957
      and float(f"{C.REC_RATE:.4g}") == 5.507e-10 and str(OBSERVED).endswith("abc_observed_ctl2_karyofix.json")
      and LD_ERROR["control"]["I"] == (0.6768, 0.6645) and C.P_NOW == 0.602 and C.N_KARYO == 619)
print("baseline", T.N_ANC, T.N0, T.T_GROW, C.REC_RATE, OBSERVED, "s prior", C.PRIOR["s"])
raise SystemExit(0 if ok else "FATAL: REPO is not on the base2 baseline with corrected karyotypes (branch illex/baseline-refit-karyofix)")
EOF
)
git -C "$REPO" log --oneline -1
mkdir -p "$ABC_OUTROOT"
cd "$REPO"
bash "$T/submit.sh"
echo "rows land in $ABC_OUTROOT/table/; bring it back to results/illex/modelchoice_karyofix/table/ on the box."
