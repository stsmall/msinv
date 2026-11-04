# Illex chr2 inversion: ABC-SMC (pyABC) on Talapas

This is the brief for the Claude session on Talapas. Follow it step by step. Stop and report back at each **STOP**.

## What this is

The project fits the history of the Illex chr2 inversion. The two arrangements are:

- **2a** (I): the AA cluster, present frequency 0.626, derived.
- **2+** (S): the BB cluster, the reference arrangement, ancestral.

There are four trajectory models: N (neutral drift), D1 and D2 (directional selection, with D2 switching selection off at t_stop) and B (balancing selection). Each runs without flux (flux 0) or with gene flux (flux 1), and with a hard start (a single copy) or a soft start (founding frequency p0).

Parameters and the posterior model probabilities come from pyABC ABC-SMC. The redis sampler spreads simulations over several nodes.

- **Simulator.** Each simulation draws a frequency path, simulates the 40 kb inversion body with msinv ≥ 0.4.0 and a 40 kb neutral control with msprime, then computes summary statistics.
- **Recombination in the body.** msinv suppresses crossovers in heterokaryotypes: a lineage of arrangement k recombines at r·p_k(t). This is why msinv is now used for every model, including no-flux.
- **Code.** Everything is on GitHub `stsmall/msinv`, branch `illex/trajectory-abc`. The `illex/` package is in the same repo as the msinv source.

## Rules

- Never cancel or modify Slurm jobs you did not submit. Cancel your own jobs only by job ID.
- Do not change model code, priors, statistics or the observed data (`results/illex/abc_observed.json`). If something looks wrong, STOP and report it.
- Do not commit or push to GitHub. Keep notes and logs on Talapas.
- No Claude or Anthropic attribution anywhere.
- Use account `kernlab`.
  - Partition `compute` has a 24 h limit.
  - Use `computelong` only if a run is shown to need more than 22 h, and ask first.
- Ignore the old Task 8 ABC-RF file `results/illex/abc_modelchoice.csv`. It came from a biased simulator. `smc.sbatch` refuses to use it, so always pass `SMC_MODELS` explicitly.

## Step 1: install

```bash
git clone --branch illex/trajectory-abc https://github.com/stsmall/msinv.git /gpfs/projects/kernlab/$USER/msinv_illex_abc
cd /gpfs/projects/kernlab/$USER/msinv_illex_abc
bash illex/abc/talapas/install_env.sh
```

`install_env.sh` does four things:

1. Creates the conda env `illex_abc` from conda-forge: Python 3.12, numpy 2.4, scipy 1.17, pandas 3.0, pyarrow 26, msprime 1.4.1, tskit 1.0.2, redis-server, rust and maturin.
2. pip-installs pyabc 0.13.1, redis-py 8.1.0 and pytest, constrained so the core stack cannot change.
3. Builds msinv from the checkout (`pip install -e .`, which compiles the Rust extension).
4. Verifies the versions, the `illex.abc.smc` import, `redis-server` and `abc-redis-worker`.

Run it on a compute node if the login node limits compiles:

```bash
srun -A kernlab -p compute -c 8 --mem=16G -t 1:00:00 --pty bash
```

The Rust build takes about 5–10 minutes.

If conda-forge has no `redis-server` for this platform, try `module avail redis`. Failing that, build redis from source into `$HOME/.local/bin` (`make` and `make PREFIX=$HOME/.local install`) and put it on PATH. Report what you did.

Check the msinv version the script prints. The branch now carries 0.4.1, which includes the flank-merge fix, and production (Step 4) requires 0.4.1 or later. If the script prints 0.4.0, run `git pull` in the repo and rerun `install_env.sh`.

**STOP:** report the version table the script printed, and any install problems.

## Step 2: local checks (on a compute node)

```bash
conda activate illex_abc
cd /gpfs/projects/kernlab/$USER/msinv_illex_abc
export PYTHONPATH=$PWD
python -m pytest tests/illex/test_abc_smc.py tests/illex/test_abc_engine_switch.py -m "not slow" -q
ENV=illex_abc REPO=$PWD SMC_ALL_MODELS=1 SMC_MODELS="N D1 D2 B" SMC_FLUX=0 DRYRUN=1 \
    bash illex/abc/talapas/smc.sbatch
```

Every test should pass. The dry run prints the redis, worker and driver commands without starting anything.

## Step 3: Talapas smoke run of the redis sampler

This path has never run on a cluster, so test it small first.

```bash
cd /gpfs/projects/kernlab/$USER/msinv_illex_abc
mkdir -p logs
ENV=illex_abc REPO=$PWD ABC_OUTROOT=/gpfs/projects/kernlab/$USER/illex_traj_abc_smoke \
SMC_MODEL=D1 SMC_FLUX=0 SMC_CLS=soft SMC_POP=50 SMC_MAX_GEN=2 SMC_WALLTIME_H=1.5 \
sbatch --account=kernlab --partition=compute --nodes=2 --cpus-per-task=32 --mem=64G --time=02:00:00 \
    --export=ALL,ABC_TALAPAS_DIR=$PWD/illex/abc/talapas illex/abc/talapas/smc.sbatch
```

On the box this smoke took about 35,000 simulations in generation 0 and 7,500 in generation 1, at 16 processes. With 64 workers it should finish in about 10 minutes.

When it finishes, check:

- `logs/illex_smc_<jobid>.out` ends with `[smc] done`.
- `$ABC_OUTROOT/smc/abc_smc_trace_D1_0.csv` has 2 generations, with eps, raw acceptance and valid-path acceptance.
- `abc_smc_post_D1_0.csv` has 1,000 rows.
- Workers on both nodes did work: the log shows simulations from both hostnames, or `abc-redis-manager info` while it runs.

Then do the same smoke with flux on: `SMC_MODEL=D2 SMC_FLUX=1`, otherwise identical.

**STOP:** report the wall times, the simulations per generation, the per-node CPU use (`seff <jobid>`), and any errors.

## Step 4: production (only with msinv ≥ 0.4.1, after the smoke runs pass)

Submit two jobs: model choice over all four models, mixed hard and soft starts, one job without flux and one with flux.

```bash
cd /gpfs/projects/kernlab/$USER/msinv_illex_abc
for FLUX in 0 1; do
  ENV=illex_abc REPO=$PWD ABC_OUTROOT=/gpfs/projects/kernlab/$USER/illex_traj_abc \
  SMC_ALL_MODELS=1 SMC_MODELS="N D1 D2 B" SMC_FLUX=$FLUX SMC_CLS=mixed \
  SMC_POP=1000 SMC_MAX_GEN=15 SMC_MIN_ACC=0.01 SMC_WALLTIME_H=22 \
  sbatch --account=kernlab --partition=compute --nodes=4 --cpus-per-task=32 --mem=64G --time=24:00:00 \
      --job-name=illex_smc_mc_f$FLUX \
      --export=ALL,ABC_TALAPAS_DIR=$PWD/illex/abc/talapas illex/abc/talapas/smc.sbatch
done
```

Before submitting, rescale the size estimate from the Step 3 timings. From the box, a full simulation costs about 1.4 s of one core and a path rejected on the frequency window costs about 0.3 s. At pop 1000:

- **Generation 0:** about 0.7 M draws for a soft-only run, roughly twice that mixed, so about 100–150 CPU-hours. That is about 1 h on 128 cores.
- **Later generations:** about 150k simulations each, roughly 30–60 CPU-hours.
- **Whole run:** 15 generations should fit in 22 h on 4 × 32 cores.

If Step 3 suggests the run will not fit, STOP and propose either more nodes or `computelong`. Do not change pop size or generations without approval.

### How the run stops

The driver stops at the first of:

- `max_gen` (15)
- valid-path acceptance below `min_acc` (0.01)
- ε changing by less than 5% between generations
- `--max-walltime-h` (22)

It writes its outputs before exiting. Note that ε is not comparable across generations, because the distance scales are refitted each generation. So the 5% rule can fire early. Report which rule stopped the run.

### Monitoring

- `squeue -u $USER`
- `tail -f logs/illex_smc_mc_f0_<jobid>.out`
- After each generation, the trace CSV gains a row. Watch the eps, acceptance and model-probability columns.

## Step 5: outputs to report back

The outputs are in `/gpfs/projects/kernlab/$USER/illex_traj_abc/smc/`:

- `abc_smc_mc_{0,1}.db`, the pyABC History (SQLite)
- `abc_smc_trace_mc_{0,1}.csv`
- `abc_smc_post_mc_{M}_{0,1}.csv`, with 1,000 resampled posterior rows per surviving model
- `abc_smc_modelchoice.csv`, `abc_smc_posteriors.csv` and `abc_rf_vs_smc.csv`. The RF comparison is stale. Ignore it.

**Report:**

- the per-generation trace for each run
- the final model probabilities, with hard and soft separated if available
- the posterior medians and 95% intervals per model (t_inv, s, p0, t_sel, t_stop, p_star, gamma_eff)
- which stop rule fired
- total CPU-hours (`sacct -j <id> --format=JobID,Elapsed,TotalCPU,MaxRSS`)

Tell the user the full paths. They will copy the directory back to the analysis box.

## Known caveats (do not try to fix them)

- **Gene-conversion floor.** msinv rejects a gene-conversion rate of exactly 0, so no-flux runs use γ = 1e-15. That is effectively zero.
- **Redis exposure.** The redis password is visible in the worker argv. The server binds 0.0.0.0 with protected mode off for the job's lifetime. This is acceptable on Talapas compute nodes, but do not reuse the password.
- **Wasted hard proposals.** In mixed runs, hard proposals keep taking about 30% of the simulations even after the hard class dies out.
- **Breakpoint boundary statistics.** The `bnd_*` boundary statistics are PPC-only and disabled (`BND_PPC_READY = False`). They are not part of this run.
