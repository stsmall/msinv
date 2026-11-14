#!/bin/bash
# Block-pooled posterior predictive check at the data's length (review M3a) on Talapas: ppc_blocks.py over the same
# 1000 NPE posterior draws as the 40-kb PPC (ppc_npe.py, seed 20261111), each with 474 x 40-kb inversion bodies
# (18.96 Mb) and 1125 x 40-kb neutral controls (45 Mb), pooled before the ratios are taken.
#
#   REPO=/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2 bash illex/abc/npe/submit_ppc_blocks.sh
#   DRYRUN=1 ... bash illex/abc/npe/submit_ppc_blocks.sh        # preflight + self-test + print the sbatch command
#   PPC_FLUXES="1 0" ...                                       # also the no-flux model (doubles the cost)
#
# REPO must be a checkout of branch illex/baseline-refit-karyofix (git pull first: the posterior draws are in
# results/illex/npe_karyofix/f{0,1}/). The preflight refuses any other baseline.
#
# Sizing (measured on the box, one core): ~0.5 CPU-s per 40-kb block with its statistics, 1599 blocks per draw,
# so ~13 CPU-min per draw and ~220 CPU-h per flux setting. Default 125 tasks x 8 draws x 8 CPUs per flux: one draw
# per CPU, ~15-20 min per task (old barriers and high flux run longer; 2 h wall leaves room). Rerunning the script
# skips slices whose CSV already exists, so failed tasks can be resubmitted as they are.
# Results: bring PPC_OUTROOT back to the box and run illex/abc/npe/collect_ppc_blocks.py on it.
set -euo pipefail
export REPO=${REPO:-/gpfs/projects/kernlab/ssmall/msinv_illex_abc_base2}
export ENV=${ENV:-illex_abc}
export PPC_OUTROOT=${PPC_OUTROOT:-/gpfs/projects/kernlab/$USER/illex_ppc_blocks}
export PPC_FLUXES=${PPC_FLUXES:-1}
export PPC_N=${PPC_N:-1000}
export DRAWS_PER_TASK=${DRAWS_PER_TASK:-8}
export PPC_CPUS=${PPC_CPUS:-8}
export PPC_MEM=${PPC_MEM:-16G}
export PPC_WALLTIME=${PPC_WALLTIME:-02:00:00}
export PPC_MAX_CONCURRENT=${PPC_MAX_CONCURRENT:-60}
export TASKS_PER_FLUX=$(((PPC_N + DRAWS_PER_TASK - 1) / DRAWS_PER_TASK))

# shellcheck disable=SC1091
source "$REPO/illex/slim/config.sh"
export PYTHONNOUSERSITE=1   # ~/.local site-packages shadow the env's scipy on Talapas
# Submit from a clean shell (see submit_table.sh: an inherited active env breaks activate_env in the tasks).
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M
for v in $(compgen -v | grep -E '^CONDA_PREFIX_[0-9]+$' || true); do unset "$v"; done
for f in $PPC_FLUXES; do
    [[ -f "$REPO/results/illex/npe_karyofix/f$f/posterior_samples.csv" ]] || {
        echo "FATAL: $REPO/results/illex/npe_karyofix/f$f/posterior_samples.csv missing (git pull the branch)" >&2; exit 2; }
done
# preflight (baseline) and the one-block self-test, in a subshell so this shell stays env-free
( activate_env && cd "$REPO" && PYTHONPATH="$REPO" OMP_NUM_THREADS=1 python illex/abc/npe/ppc_blocks.py --self-test )
git -C "$REPO" log --oneline -1
read -ra FL <<<"$PPC_FLUXES"
NTASKS=$((${#FL[@]} * TASKS_PER_FLUX))
mkdir -p "$PPC_OUTROOT" "$REPO/logs"
CMD=(sbatch --array="0-$((NTASKS - 1))%${PPC_MAX_CONCURRENT}" --account="$ACCOUNT" --partition="$PARTITION"
     --cpus-per-task="$PPC_CPUS" --mem="$PPC_MEM" --time="$PPC_WALLTIME" --chdir="$REPO" --export=ALL
     "$REPO/illex/abc/npe/ppc_blocks.sbatch")
echo "flux ${PPC_FLUXES}: ${#FL[@]} x $TASKS_PER_FLUX tasks x $DRAWS_PER_TASK draws, $PPC_CPUS CPUs each; out $PPC_OUTROOT"
echo "${CMD[*]}"
if [[ -n "${DRYRUN:-}" ]]; then echo "(dry run; not submitted)"; exit 0; fi
"${CMD[@]}"
echo "when done: bring $PPC_OUTROOT back to the box and run illex/abc/npe/collect_ppc_blocks.py on it"
