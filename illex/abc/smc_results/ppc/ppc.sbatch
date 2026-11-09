#!/bin/bash
# Posterior predictive replicates for one SMC run. Submit from the directory that should receive the CSV:
#   sbatch --job-name=ppc_A_f0 --export=ALL,REPO=<checkout>,DB=<abc_smc_mc_F.db>,LABEL=A,FLUX=0 ppc.sbatch
# REPO must be the checkout whose priors match the run (to_params clips s to constants.PRIOR["s"]).
#SBATCH --account=kernlab
#SBATCH --partition=compute,kern
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=ppc_%x_%j.out
set -euo pipefail
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate illex_abc
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[ppc] $LABEL flux=$FLUX repo=$REPO db=$DB $(date)"
python "$REPO/illex/abc/smc_results/ppc/ppc_run.py" --db "$DB" --flux "$FLUX" --label "$LABEL" \
    --out "ppc_${LABEL}_f${FLUX}.csv" --n 1000 --procs 32
echo "[ppc] done $(date)"
