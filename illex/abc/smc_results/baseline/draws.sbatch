#!/bin/bash
# Baseline prior draws, then the fit. Submit from the directory that should receive the outputs:
#   sbatch --export=ALL,REPO=<checkout> $REPO/illex/abc/smc_results/baseline/draws.sbatch
# Reads $REPO/results/illex/abc_observed_ctl2.json (representative control target and its LD error model).
#SBATCH --account=kernlab
#SBATCH --partition=compute,kern
#SBATCH --job-name=baseline_draws
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate illex_abc
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
OBS="$REPO/results/illex/abc_observed_ctl2.json"
S="$REPO/illex/abc/smc_results/baseline/baseline_refit.py"
echo "[draws] start $(date)"
python "$S" sim --n "${N:-20000}" --procs 32 --observed "$OBS" --out draws.csv
python "$S" fit --draws draws.csv --observed "$OBS" --out fit.json
echo "[draws] done $(date)"
