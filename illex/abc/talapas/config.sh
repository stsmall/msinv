#!/bin/bash
# Talapas config for the trajectory-ABC reference table. Account / partition / env / repo come from
# illex/slim/config.sh (same campaign conventions); only the table sizing is new. ABC_* variables override the
# sizing; they are separate from the slim config.sh names (CPUS_PER_TASK, MEM, ... are re-set here).
HERE_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE_CFG/../../slim/config.sh"
export OUTROOT=${ABC_OUTROOT:-/gpfs/projects/$ACCOUNT/$USER/illex_traj_abc}
# CHECK BEFORE SUBMITTING: the Talapas env (ENV from slim/config.sh, illex_slimsim) is not verified to hold msinv
# and msprime of the versions used on the box; array.sbatch fails fast with exit 3 if the import preflight fails.
export TABLE_MODELS=${TABLE_MODELS:-"N D1 D2 B"}
export TABLE_FLUXES=${TABLE_FLUXES:-"0 1"}
export TABLE_CLASSES=${TABLE_CLASSES:-"hard soft"}
export K_PER_TASK=${K_PER_TASK:-250}        # accepted rows per array task
export TASKS_PER_COMBO=${TASKS_PER_COMBO:-40}   # array tasks per (model, flux, class)
# Hard starts use importance sampling on establishment, so draws count thetas. Measured pilot conditional rates per theta
# (flux 0): N ~2e-5, D1 ~4e-5, D2 ~1.2e-4, B ~1.9e-2 (soft: 0.2-2%); path generation runs ~850 thetas/s per task. MAX_DRAWN
# 3e6 caps a task at ~1 h (N hard ~60 rows, D1 ~120, D2 ~250+/cap, B/soft finish at K in minutes); rows are
# importance-weighted by w_est, so partial tasks are fine.
export MAX_DRAWN=${MAX_DRAWN:-3000000}
export CPUS_PER_TASK=${ABC_CPUS:-4}
export MEM=${ABC_MEM:-16G}
export WALLTIME=${ABC_WALLTIME:-03:00:00}
export MAX_CONCURRENT_TASKS=${ABC_MAX_CONCURRENT:-50}
