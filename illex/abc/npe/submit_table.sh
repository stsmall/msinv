#!/bin/bash
# NPE training table on Talapas: prior draws of the balancing model (B), soft starts, flux 0 and 1, on the
# refitted (base2) neutral baseline. Reuses the reference-table array (illex/abc/run_table.py via
# illex/abc/talapas/array.sbatch) of the base2 checkout, so the simulator, baseline, error model and statistics are
# exactly those of the base2 ABC-SMC runs.
#
# Every row is a prior draw that passed the present-frequency filter (|p_now - 0.626| <= 0.02), the same
# conditioning as the SMC, with all 39 fitted statistics (MODEL_KEYS) plus the PPC keys.
#
#   REPO=/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2 bash illex/abc/npe/submit_table.sh
#   DRYRUN=1 ... bash illex/abc/npe/submit_table.sh      # preflight + print the sbatch command only
#
# REPO must be a checkout of branch illex/baseline-refit (the base2 constants are in its code). The preflight
# refuses any other baseline.
#
# Sizing (measured on the box, msinv 0.4.1): 1.3% of soft B prior draws pass the filter and an accepted row costs
# ~15 CPU-s (heavy tail to ~50 s for old barriers / high flux). Default 120 tasks x 500 rows per flux = 60,000 rows
# per flux, ~500 CPU-h in total; with 8 CPUs per task a task takes ~15-30 min.
set -euo pipefail
export REPO=${REPO:-/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2}
export ENV=${ENV:-illex_abc}
export ABC_OUTROOT=${ABC_OUTROOT:-/gpfs/projects/kernlab/$USER/illex_npe_base2}
export TABLE_MODELS=B
export TABLE_FLUXES=${TABLE_FLUXES:-"0 1"}
export TABLE_CLASSES=soft
export K_PER_TASK=${K_PER_TASK:-500}
export TASKS_PER_COMBO=${TASKS_PER_COMBO:-120}
export MAX_DRAWN=${MAX_DRAWN:-3000000}
export ABC_CPUS=${ABC_CPUS:-8}
export ABC_MEM=${ABC_MEM:-24G}
export ABC_WALLTIME=${ABC_WALLTIME:-03:00:00}
export ABC_MAX_CONCURRENT=${ABC_MAX_CONCURRENT:-60}

T="$REPO/illex/abc/talapas"
[[ -f "$T/submit.sh" && -f "$T/array.sbatch" ]] || { echo "FATAL: $T has no submit.sh / array.sbatch" >&2; exit 2; }
# shellcheck disable=SC1091
source "$T/config.sh"
activate_env
( cd "$REPO" && PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" python - <<'EOF'
import illex.theory as T, illex.abc.constants as C
from illex.abc.smc import OBSERVED
from illex.abc.error_model import LD_ERROR
ok = (round(T.N_ANC) == 507288 and round(T.N0) == 9112696 and round(T.T_GROW) == 382957
      and float(f"{C.REC_RATE:.4g}") == 5.507e-10 and str(OBSERVED).endswith("abc_observed_ctl2.json")
      and LD_ERROR["control"]["I"] == (0.6240, 0.6656))
print("baseline", T.N_ANC, T.N0, T.T_GROW, C.REC_RATE, OBSERVED, "s prior", C.PRIOR["s"])
raise SystemExit(0 if ok else "FATAL: REPO is not on the base2 baseline (branch illex/baseline-refit)")
EOF
)
git -C "$REPO" log --oneline -1
mkdir -p "$ABC_OUTROOT"
cd "$REPO"
bash "$T/submit.sh"
echo "rows land in $ABC_OUTROOT/table/B_{0,1}_soft_*.jsonl.gz; bring that table/ directory back to"
echo "results/illex/npe/table/ on the box and run illex/abc/npe/collect_npe.py, then train_npe.py."
