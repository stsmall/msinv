#!/bin/bash
# Talapas config for the trajectory-ABC reference table. Account / partition / env / repo come from
# illex/slim/config.sh (same campaign conventions); only the table sizing is new. ABC_* variables override the
# sizing; they are separate from the slim config.sh names (CPUS_PER_TASK, MEM, ... are re-set here).
HERE_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE_CFG/../../slim/config.sh"
export OUTROOT=${ABC_OUTROOT:-/gpfs/projects/$ACCOUNT/$USER/illex_traj_abc}
export TABLE_MODELS=${TABLE_MODELS:-"N D1 D2 B"}
export TABLE_FLUXES=${TABLE_FLUXES:-"0 1"}
export TABLE_CLASSES=${TABLE_CLASSES:-"hard soft"}
export K_PER_TASK=${K_PER_TASK:-250}        # accepted rows per array task
export TASKS_PER_COMBO=${TASKS_PER_COMBO:-40}   # array tasks per (model, flux, class)
export MAX_DRAWN=${MAX_DRAWN:-20000000}
export CPUS_PER_TASK=${ABC_CPUS:-4}
export MEM=${ABC_MEM:-16G}
export WALLTIME=${ABC_WALLTIME:-12:00:00}
export MAX_CONCURRENT_TASKS=${ABC_MAX_CONCURRENT:-50}
