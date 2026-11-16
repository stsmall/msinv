# Illex chr2 inversion: full-length NPE training run on Talapas

This is the brief for the session on Talapas. Follow it step by step. Stop and report back at each **STOP**.

## What this is

The NPE fit of the balancing model was trained on 40-kb simulations of the inversion. This run simulates the whole
18.96-Mb inversion in every training simulation, so the training statistics scatter like the data's. It also adds a
second gene-flux process.

- **Model.** Balancing selection (B), soft start (founding frequency p0), with gene flux. 2a (I) is derived, 2+ (S)
  is ancestral. Paths are drawn from the B prior and kept when the present frequency is within 0.02 of 0.602.
- **Two flux processes (msinv #30).** Short gene conversion as before (per-site rate c, the prior variable
  `gamma_eff`, mean tract 300 bp) plus an independent long-tract exchange process (per-site rate `c_long`,
  log-uniform 1e-11 to 1e-6, and mean tract `l_long`, log-uniform 10 kb to 5 Mb), both geometric.
- **Simulator.** msinv 0.4.2 (with the #28 flux fix) plus the long-tract process, branch `illex/fulllength` of
  `stsmall/msinv`. Each row is the full inversion between two 10-kb collinear margins, with crossover suppression.
- **Control.** A bank of full-length (45 Mb) neutral msprime controls is simulated first. Each training row is
  divided by one bank control drawn from its own seed.
- **Saved per row.** The 42 statistics of the 40-kb fits (same functions), the raw values before division, LD by
  1-Mb block, LD in 12 fine bins (with and without the genotype-error model), LD at 20 kb to 2 Mb, the unfolded
  spectra, the exact joint count spectrum, per-Mb pi and dxy, the edge profile, the frequency path, all parameters
  and seeds. Mutated tree sequences are kept for every SAVE_TS_EVERY-th row (default 1000).

## Rules

- Never cancel or modify Slurm jobs you did not submit. Cancel your own jobs only by job ID.
- Do not change model code, priors, statistics or the observed data. If something looks wrong, STOP and report it.
- Do not commit or push to GitHub. Keep notes and logs on Talapas.
- No Claude or Anthropic attribution anywhere.
- Use account `kernlab`, partition `compute` (24 h limit).

## Step 1: install (a new checkout and a new env)

Use a separate env from the 40-kb runs: `pip install -e` points an env's msinv at one checkout.

```bash
git clone --branch illex/fulllength https://github.com/stsmall/msinv.git /gpfs/projects/kernlab/$USER/msinv_illex_full
cd /gpfs/projects/kernlab/$USER/msinv_illex_full
srun -A kernlab -p compute -c 8 --mem=16G -t 1:00:00 --pty bash
ABC_ENV=illex_full REPO=$PWD BRANCH=illex/fulllength bash illex/abc/talapas/install_env.sh
```

Then check that the long-tract process is in the build:

```bash
conda activate illex_full
PYTHONNOUSERSITE=1 python -c "import msinv; from msinv import InversionSpec; print(msinv.__version__, 'long_flux_rate' in InversionSpec.__dataclass_fields__)"
```

It must print `0.4.2 True`.

**STOP:** report the version table and the line above.

## Step 2: tests (on a compute node)

```bash
cd /gpfs/projects/kernlab/$USER/msinv_illex_full
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python -m pytest -q tests/hull/test_long_flux.py tests/illex/test_abc_fulllen.py
cd rust && cargo test --release -p msinv-core --lib flux && cd ..
```

All must pass.

**STOP:** report the test summaries.

## Step 3: dry run, then a small pilot

```bash
DRYRUN=1 bash illex/abc/npe/submit_full.sh
```

The preflight prints the msinv version, the baseline and the long-flux prior, and both sbatch commands.

Pilot (2 bank tasks, 2 row tasks, into a separate output directory):

```bash
FULL_OUTROOT=/gpfs/projects/kernlab/$USER/illex_full_pilot BANK_TASKS=2 BANK_PER_TASK=2 ROW_TASKS=2 K_PER_TASK=2 \
    SAVE_TS_EVERY=2 bash illex/abc/npe/submit_full.sh
```

When it finishes, report from the logs (`logs/illex_full_*`) the wall time and the peak memory of each control and
each row (`row k/K seed ... s ... GB` lines), and the sizes of `bank/ctl_*.jsonl.gz`, `table/B_full_*.jsonl.gz` and
`table/ts/*.trees`.

**STOP:** report these. The production sizes in Step 4 are set from the pilot.

## Step 4: production

```bash
bash illex/abc/npe/submit_full.sh
```

Defaults (override by environment variable): BANK_TASKS 100 x BANK_PER_TASK 8 controls (4 CPUs, 80G);
ROW_TASKS 10000 x K_PER_TASK 2 rows per array, ROW_CPUS 1 with ROW_MEM 200G and 24 h, at most 200 tasks at a time.
On the box (msinv 0.4.2) four full-length rows took 49, 62, 82 and 466 min with peaks of 61, 72, 91 and 152 GB (the
slow one had strong selection, s = 0.094), and one 45-Mb control 12 min with 13 GB. Rows drawn from the prior may
include more strong-selection draws than these posterior draws, so one row per task with room for about 3x the
slowest row. If a rows task runs out of memory or time, report its log tail and its parameters; do not change
ROW_MEM or ROW_WALLTIME without asking.

The full run is 40,000 row tasks (80,000 rows) in four arrays of 10,000. The row task id, which sets the seeds, is
ROW_OFFSET plus the array index:

```bash
bash illex/abc/npe/submit_full.sh                                  # bank, then rows 0-9999
STAGE=rows ROW_OFFSET=10000 bash illex/abc/npe/submit_full.sh      # rows 10000-19999
STAGE=rows ROW_OFFSET=20000 bash illex/abc/npe/submit_full.sh      # rows 20000-29999
STAGE=rows ROW_OFFSET=30000 bash illex/abc/npe/submit_full.sh      # rows 30000-39999
```

Submit the later arrays once the bank is complete. If Slurm refuses an array of 10000, use a smaller ROW_TASKS and
step ROW_OFFSET by the same amount. A task is skipped when its output file exists, so resubmitting the same command
resumes the run. A rows task that hits its wall time keeps its finished rows in
`table/.B_full_<task>.jsonl.gz.partial`.

Report progress as the number of `table/B_full_*.jsonl.gz` files, and any failed tasks with their log tails.

## Bringing the results back

Copy `bank/` and `table/` (and `table/ts/`) to the box under
`results/illex/npe_full/`. Training (collect, train_npe) and the full-length predictive check run on the box.
