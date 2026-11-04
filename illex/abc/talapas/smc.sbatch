#!/bin/bash
# Task 8b. Multi-node pyABC ABC-SMC on Talapas with the redis sampler.
#   Node 0 (the batch host) runs redis-server and the smc.py driver (--sampler redis).
#   Every node (node 0 included) runs abc-redis-worker with SMC_WORKER_PROCS processes.
# Account / partition / env / repo come from illex/slim/config.sh via illex/abc/talapas/config.sh.
# REDIS_PORT and REDIS_PASSWORD come from the environment (a random password is made if unset).
#
# Submit (from the repo root) with the variables that pick the run, for example
#   SMC_MODEL=B SMC_FLUX=0 SMC_CLS=mixed sbatch --account=kernlab --partition=compute \
#       --export=ALL,ABC_TALAPAS_DIR=$PWD/illex/abc/talapas illex/abc/talapas/smc.sbatch
#   SMC_ALL_MODELS=1 SMC_FLUX=0 sbatch ... illex/abc/talapas/smc.sbatch
# DRYRUN=1 bash illex/abc/talapas/smc.sbatch prints the commands without starting anything.
#
# CHECK BEFORE SUBMITTING. The env (ENV from slim/config.sh) must hold pyabc 0.13.1, redis-py, msprime, msinv and a
# redis-server binary (conda install -c conda-forge redis-server, or a site module). The preflight below exits 3 if
# not. The flux engine (msinv) still lacks msinv PR #19, so flux=1 runs wait for it.
#SBATCH --job-name=illex_smc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -euo pipefail
HERE=${ABC_TALAPAS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/illex/abc/talapas}
# shellcheck disable=SC1091
source "$HERE/config.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

SMC_FLUX=${SMC_FLUX:-0}
SMC_CLS=${SMC_CLS:-mixed}
SMC_POP=${SMC_POP:-1000}
SMC_MAX_GEN=${SMC_MAX_GEN:-15}
SMC_MIN_ACC=${SMC_MIN_ACC:-0.01}
SMC_WALLTIME_H=${SMC_WALLTIME_H:-22}       # driver stops before the SLURM limit so the outputs get written
SMC_WORKER_PROCS=${SMC_WORKER_PROCS:-${SLURM_CPUS_PER_TASK:-32}}
SMC_OUT=${SMC_OUT:-$OUTROOT/smc}
if [[ -n "${SMC_ALL_MODELS:-}" ]]; then
    # survivors of Task 8 from SMC_MODELCHOICE (prob_total >= SMC_MIN_MODEL_PROB) unless SMC_MODELS lists them
    TAG=mc; SEL=(--all-models --modelchoice "${SMC_MODELCHOICE:-$REPO/results/illex/abc_modelchoice.csv}"
                 --min-model-prob "${SMC_MIN_MODEL_PROB:-0.01}")
    [[ -n "${SMC_MODELS:-}" ]] && SEL+=(--models $SMC_MODELS)
else
    TAG=${SMC_MODEL:?set SMC_MODEL or SMC_ALL_MODELS}; SEL=(--model "$SMC_MODEL")
fi
export REDIS_PORT=${REDIS_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
export REDIS_PASSWORD=${REDIS_PASSWORD:-$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')}
export REDIS_HOST=${REDIS_HOST:-$(hostname)}
DB="$SMC_OUT/abc_smc_${TAG}_${SMC_FLUX}.db"

DRIVER=(python -m illex.abc.smc "${SEL[@]}" --flux "$SMC_FLUX" --cls "$SMC_CLS" --pop-size "$SMC_POP"
        --max-gen "$SMC_MAX_GEN" --min-acc "$SMC_MIN_ACC" --max-walltime-h "$SMC_WALLTIME_H" --sampler redis
        --db "$DB" --out-dir "$SMC_OUT")
WORKER="abc-redis-worker --host=$REDIS_HOST --port=$REDIS_PORT --password=\$REDIS_PASSWORD --processes=$SMC_WORKER_PROCS --runtime=${SMC_WALLTIME_H}h"
if [[ -n "${DRYRUN:-}" ]]; then
    echo "redis-server --port $REDIS_PORT --requirepass *** --bind 0.0.0.0 --protected-mode no --save '' --appendonly no"
    echo "srun --nodes=${SLURM_NNODES:-N} --ntasks-per-node=1 bash -c '$WORKER'"
    echo "cd $REPO && ${DRIVER[*]}"
    echo "abc-redis-manager stop --host=$REDIS_HOST --port=$REDIS_PORT --password=***"
    echo "(dry run, nothing started)"
    exit 0
fi

activate_env
mkdir -p "$SMC_OUT" logs
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! python -c "import pyabc, redis, msprime, msinv, illex.abc.smc" || ! command -v redis-server >/dev/null; then
    echo "FATAL: preflight failed: env '$ENV' needs pyabc, redis, msprime, msinv, illex.abc.smc and redis-server (REPO=$REPO)." >&2
    exit 3
fi

redis-server --port "$REDIS_PORT" --requirepass "$REDIS_PASSWORD" --bind 0.0.0.0 --protected-mode no \
    --save '' --appendonly no --daemonize no > "logs/redis_${SLURM_JOB_ID}.log" 2>&1 &
REDIS_PID=$!
cleanup() {
    abc-redis-manager stop --host="$REDIS_HOST" --port="$REDIS_PORT" --password="$REDIS_PASSWORD" || true
    [[ -n "${SRUN_PID:-}" ]] && wait "$SRUN_PID" 2>/dev/null || true
    kill "$REDIS_PID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 60); do
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q PONG && break
    sleep 1
done

# The workers expand REDIS_PASSWORD from the inherited environment, so it is not echoed into the job log (it does
# reach the worker argv, which other users on a shared node can see with ps).
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --export=ALL \
    bash -c "$WORKER" &
SRUN_PID=$!

echo "[smc] $TAG flux=$SMC_FLUX cls=$SMC_CLS pop=$SMC_POP nodes=$SLURM_NNODES procs/node=$SMC_WORKER_PROCS host=$REDIS_HOST:$REDIS_PORT $(date)"
"${DRIVER[@]}"
echo "[smc] done $(date)"
