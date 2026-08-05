#!/bin/bash
# =====================================================================
# Talapas config for the Illex inversion ABC.
#
# Deliberately mirrors
#   14_sweep_seqmodel/scripts/harness/talapas/config.sh
# so the two campaigns share account, partition, env and scratch conventions.
# Values below are copied from that file, not guessed.
# =====================================================================
export ACCOUNT=${ACCOUNT:-kernlab}                 # sbatch --account (PIRG)
export PARTITION=${PARTITION:-compute}             # compute / computelong
export ENV=${ENV:-illex_slimsim}                   # conda env (has slim/pyslim/msprime)

# Repo checkout containing illex/ (this pipeline runs as `python -m illex.slim.*`)
export REPO=${REPO:-$HOME/inversion_sims/files}

# BIG scratch for intermediate .trees files. These are deleted per replicate, but
# must NOT live on shared storage while in use.
export OUTROOT=${OUTROOT:-/gpfs/projects/$ACCOUNT/$USER/illex_inv_abc}

# SLiM >= 5.0 REQUIRED: inversion_abc.slim uses the haplosome/tick API, exactly
# like the sweep recipes. SLiM 4.x will NOT run it.
export SLIM_BIN=${SLIM_BIN:-$(command -v slim)}

# ---- campaign sizing --------------------------------------------------
export N_SIMS=${N_SIMS:-20000}          # total simulations across the array
export CHUNK=${CHUNK:-10}               # sims per array task -> NTASKS = N_SIMS/CHUNK
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-50}   # the %N throttle
export CPUS_PER_TASK=${CPUS_PER_TASK:-1}
export MEM=${MEM:-8G}
export WALLTIME=${WALLTIME:-12:00:00}

# ---- model ------------------------------------------------------------
# Q is the cost knob AND a validity constraint: the recipe aborts at SEL*Q >= 0.1,
# which caps Q at 333 for the prior's s_max = 3e-4. Lower the s prior rather than
# raising Q past that (see README).
export QSCALE=${QSCALE:-200}

activate_env() {
  module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null || true
  # shellcheck disable=SC1091
  conda activate "$ENV" 2>/dev/null || {
    echo "config.sh: could not 'conda activate $ENV' -- edit activate_env()" >&2
    exit 1
  }
}
