#!/bin/bash
# Posterior predictive replicates of one run-set-B SMC run under the refitted baseline. Submit from the directory
# holding fit.json (from draws.sbatch):
#   sbatch --job-name=ppcnb_B_f0 --export=ALL,REPO=<checkout>,DB=<runs_s1e-1/abc_smc_mc_0.db>,FLUX=0 ppc_newbase.sbatch
#SBATCH --account=kernlab
#SBATCH --partition=compute,kern
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate illex_abc
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
echo "[ppc-newbase] flux=$FLUX db=$DB $(date)"
python "$REPO/illex/abc/smc_results/baseline/ppc_new_baseline.py" --baseline fit.json --db "$DB" --flux "$FLUX" \
    --label "B_f${FLUX}" --observed "$REPO/results/illex/abc_observed_ctl2.json" \
    --out "ppc_newbase_B_f${FLUX}.csv" --n 1000 --procs 32
echo "[ppc-newbase] done $(date)"
