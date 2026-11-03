# Illex chr2 trajectory ABC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rejection ABC with model choice and posterior estimation for the rise of 2a. The models are N, D1,
D2 and B, each with and without flux. Validation uses msprime and SLiM, and the PPCs are generated in SLiM.

**Architecture:**
- A new package `illex/abc/` holds four parts:
  - a forward trajectory sampler on the growth demography, giving one stochastic path per prior draw
  - an msinv simulator that turns each path into a statistic vector
  - an observed-statistic extractor that produces the same vector from the data
  - a reference-table driver that runs locally or as a Talapas array
- Inference is ABC-RF in R (`abcrf`, `abc`).
- SLiM, through the existing `illex/slim/` recipe, and msprime provide the independent checks.

**Tech Stack:**
- Python 3.12 venv at `../inversion_sims/files/.venv` (msinv, tskit, msprime, numpy, scipy)
- pg_gpu env `/home/ssmall/miniforge3/envs/varbuddy-pggpu` for the observed statistics
- ANGSD SAFs
- R at `/home/ssmall/R/bin/Rscript` with `abcrf` and `abc` (to be installed)
- SLiM ≥ 5.0 and pyslim
- Talapas Slurm (`illex/slim/config.sh` conventions: ACCOUNT=kernlab, PARTITION=compute)

**Spec:** `docs/superpowers/specs/2026-11-03-illex-chr2-trajectory-abc-design.md`

## Global Constraints

- The inverted class I is 2a (AA cluster, p_now 0.626), and S is 2+ (BB cluster).
- **Demography is fixed:** N_ANC 547,928 to N0 6,808,096, exponential over T_GROW 769,519 generations, μ 3e-9
  (`illex/theory.py`).
- **Recombination** is the chr2 collinear rate from the current male ReLERNN map (Task 1). Never 1.977e-9
  unless Task 1 confirms it.
- **Breakpoints** are 60.55 and 79.52 Mb. **Sample sizes** are 2aa 254 and 2++ 95 individuals (508 and 190
  chromosomes). Karyotypes are sampled as 633 binomial draws.
- **Priors:**
  - t_inv U(3e5, 1.5e6)
  - P(hard) 0.5, soft p0 logU(1e-3, 0.5)
  - t_sel U(0, t_inv)
  - s and s_het logU(1e-6, 1e-2)
  - t_stop U(0, t_sel)
  - p* U(0.3, 0.95)
  - γ_eff logU(1e-9, γ_max)
- **Pre-filter:** |p_now − 0.626| ≤ 0.02.
- **Run rules:**
  - Box load: our processes under 150, and under 100 when the total load is over 200.
  - Never touch `balanus` processes.
  - Never kill or stop a job unless the user says so.
  - Kill by PID only.
- **Commit rules:** no msinv commits without user approval. Each task ends with the files staged and the
  commit message drafted, and the commit happens when the user approves. Work on the new branch
  `illex/trajectory-abc` off `illex/pboot-recovery`.

## Review Focus

1. **Hard starts at weak s.** The establishment loop can spin for millions of attempts. Expect a bounded
   attempt cap and a recorded failure, not a hang. Test in Task 2.
2. **Paths with p crossing 0 or 1 within one diffusion step.** Expect loss to count as a rejection and
   fixation as a rejection, never a clipped frequency handed to msinv. Test in Task 2.
3. **Statistic NaN or inf when an arrangement has zero segregating sites,** as in a very young hard start.
   Expect the row kept with NaN flagged and dropped at ABC time, with the count reported. Test in Task 3.
4. **Observed and simulated statistic vectors drifting apart in keys or order.** Expect a single
   `STAT_KEYS` contract that both sides assert against. Tests in Tasks 3 and 4.
5. **Talapas array tasks dying mid-way.** Expect the per-task output to be atomic (write to a temp file,
   then rename) and the collector to report missing task IDs. Test in Task 6.

---

### Task 1: Recombination rate and branch setup

**Files:**
- Modify: `illex/slim/config.py` (REC_RATE)
- Create: `illex/abc/__init__.py`, `illex/abc/constants.py`
- Test: `tests/illex/test_abc_constants.py`

**Interfaces:**
- Produces: `illex.abc.constants` with `P_NOW=0.626`, `P_TOL=0.02`, `BP_LEFT=60_550_000`,
  `BP_RIGHT=79_520_000`, `N_I_CHROM=508`, `N_S_CHROM=190`, `N_IND=633`, `PRIOR` (dict, values above),
  `REC_RATE` (re-exported from `illex.slim.config`), and the TARGETS for the three ratios.

- [ ] **Step 1:** `git switch -c illex/trajectory-abc` from `illex/pboot-recovery`.
- [ ] **Step 2:** Find the chr2 collinear male rate in the current map. The source is in manuscript
  commit 7d69fd2, at `analysis/steps/11_relernn/`, bootstrap-corrected by `bs200_correct_genomewide.py`.
  Use the mean over chr2 outside 58.55–81.52 Mb, a 2 Mb buffer. Record the file path and value in a
  comment in `config.py`.
- [ ] **Step 3:** Write the test `test_constants_match_spec`. It asserts every constant above, and asserts
  that `REC_RATE` equals the value found in Step 2 to 4 significant figures.
- [ ] **Step 4:** Implement the constants and update REC_RATE. Run
  `pytest tests/illex/test_abc_constants.py -v`. Expect PASS.
- [ ] **Step 5:** Stage the files and draft the commit message. Tell the user the old and new REC_RATE.
  Commit on approval.

### Task 2: Forward trajectory sampler (four models)

**Files:**
- Create: `illex/abc/trajectory.py`, generalising `illex/scripts/abc_traj_pilot.py:traj_batch`/`establish`
- Test: `tests/illex/test_abc_trajectory.py`

**Interfaces:**
- Produces:
  - `draw_prior(model: str, flux: bool, rng, n: int) -> dict[str, np.ndarray]`, with `model` one of
    `"N","D1","D2","B"`, and keys `t_inv, hard, p0, t_sel, s, t_stop, p_star, gamma_eff`. Unused keys
    are NaN.
  - `simulate_paths(params: dict, rng, dt=10.0, rec_every=1000.0, max_attempts=2_000_000)
    -> tuple[list[Path], Counts]`
  - `Path` is a dataclass with `params: dict`, `times: np.ndarray` (backward, increasing from 0),
    `freqs: np.ndarray` and `p_now: float`.
  - `Counts` is a dataclass with `drawn`, `lost`, `fixed`, `est_failed`, `accepted`.

Selection per generation (the spec fixes these, so they are exact):
- **D1:** genic `s·p(1−p)` from t_sel to 0.
- **D2:** genic s from t_sel to t_stop, neutral after.
- **B:** overdominance toward p*, using `s_het·p(1−p)(p*−p)/p*`, the same form as
  `illex/balancing.py:_time_to_reach`. Applied from t_sel, and t_sel = t_inv for hard starts.
- **N:** none.
- **Hard starts:** exact binomial WF from 1 copy until 500 copies, then the diffusion.

- [ ] **Step 1: Write the failing tests**
  - `test_neutral_soft_mean_preserved`: p0 0.3, N model, 2,000 paths over 1e4 generations without the
    p_now filter. The mean final p is 0.3 ± 0.01.
  - `test_D1_rises`: s 1e-3 from p0 0.05 over 2e4 generations. Median final p > 0.9.
  - `test_B_settles_at_pstar`: s_het 1e-3, p* 0.5, from p0 0.05. Median final p is within 0.03 of 0.5.
  - `test_D2_freezes_after_tstop`: s 1e-2 with t_stop at 0.9 of the run. Path variance after t_stop
    matches neutral to within a factor of 2.
  - `test_hard_establishment_prob`: neutral, constant N 1e4, cap 1e5 attempts. The fraction reaching 500
    copies is 1/500 within 30%.
  - `test_attempt_cap_records_failure`: s = 0 and cap 10. `Counts.est_failed` > 0 and no hang (timeout 10 s).
  - `test_no_clipped_endpoints`: every accepted path has freqs strictly in (0, 1).
  - `test_filter`: every accepted `p_now` is within 0.626 ± 0.02.
- [ ] **Step 2:** Run `pytest tests/illex/test_abc_trajectory.py -v`. Expect FAIL (module missing).
- [ ] **Step 3:** Implement. The tests need an optional `n_e_fn` (default `N_growth`) and an optional
  `apply_filter=True`.
- [ ] **Step 4:** Run the tests. Expect PASS.
- [ ] **Step 5:** Stage the files and draft the commit message.

### Task 3: msinv simulator and statistic vector

**Files:**
- Create: `illex/abc/simulate.py`, `illex/abc/statvec.py`
- Test: `tests/illex/test_abc_simulate.py`

**Interfaces:**
- Consumes: `Path` (Task 2) and `illex.abc.constants`.
- Produces:
  - `STAT_KEYS: tuple[str, ...]` in `statvec.py`, the single contract
  - `simulate_stats(path: Path, seed: int) -> dict[str, float]`, whose keys are exactly `STAT_KEYS` plus
    `p_obs`, `n_nan`

What the simulation does:
- **Segment:** an inversion segment with collinear flanks on each side, at `MARGIN_FRACTION`.
- **Flux:** γ = γ_eff·(1−w)/w, with w = mean_tract_length / segment inversion length.
  mean_tract_length is fixed at 1,000 bp, the value to be confirmed from Korunes & Noor in Task 5.
- **Samples:** n_inv 508 and n_std 190. `p_obs` is Binomial(2·633, p_now)/(2·633).

The statistics (built in `statvec.py`, every one divided by its flank value where one exists):
- the three ratios
- the per-arrangement SFS, projected with `illex.scripts.sfs_shape.project_unfolded` to PROJ, as bins
  f1..f5 plus a pooled ≥6 bin
- the folded joint SFS on a 5×5 grid
- genotype r² decay within each class (diploids paired at random within class), in 6 distance bins from
  100 bp to 20 kb
- Tajima's D per arrangement and the edge-to-centre dxy/π, both flagged in `STAT_KEYS_PPC_ONLY` and used
  only in the PPCs

- [ ] **Step 1: Write the failing tests**
  - `test_keys_contract`: the output keys equal `set(STAT_KEYS) | {"p_obs","n_nan"}`.
  - `test_seed_deterministic`: the same seed gives identical output.
  - `test_no_barrier_limit`: t_inv tiny (100 generations), flat path at 0.5. The three ratios are 1.0 ±
    0.05 over 20 reps.
  - `test_zero_seg_sites_flagged`: a hard start with t_inv 2,000 generations gives `n_nan` > 0, with no
    exception.
  - `test_flux_scaling`: γ_eff 1e-4 lowers dxy/π_I against γ_eff 1e-9 by more than 5% over 20 reps, on a
    0.9 Ma soft path.
- [ ] **Step 2:** Run the tests and expect FAIL. **Step 3:** Implement. **Step 4:** Run and expect PASS.
- [ ] **Step 5:** Stage the files and draft the commit message.

### Task 4: Observed statistic vector

**Files:**
- Create: `illex/abc/observed.py`. It reuses `empirical_jackknife.py`, `sfs_shape_angsd.py`,
  `sfs2d_karyotype.py` and `empirical_windowed.py` logic, and runs in the pg_gpu env.
- Output: `results/illex/abc_observed.json` (values plus jackknife SEs)
- Test: `tests/illex/test_abc_observed.py`

**Interfaces:**
- Consumes: `STAT_KEYS` (Task 3).
- Produces: `abc_observed.json` with `{"stats": {key: value}, "se": {key: se}, "p_obs": 0.626}`.

Inputs:
- karyotypes `analysis/steps/03_karyotype/{AA,BB}_samples.txt`
- SAFs `analysis/steps/04_angsd_chr2/saf/{AA,BB}.2.saf.gz`
- the Earl Grey accessible mask
- flanks: chr2 collinear sequence matching the simulated margin. Use the same flank region as
  `empirical_windowed.py` REGIONS.

LD is computed with pg_gpu `GenotypeMatrix` on the accessible, depth-filtered sites, separately for the
2aa and 2++ samples.

- [ ] **Step 1: Write the failing tests**
  - `test_keys_match_sim`: the JSON keys equal `STAT_KEYS`.
  - `test_ratios_reproduce_targets`: π ratio 0.7377, dxy/π 1.8763 and f1 1.2307, each within 0.002.
- [ ] **Step 2:** Run and expect FAIL. **Step 3:** Implement. **Step 4:** Run with the pg_gpu python and
  expect PASS.
- [ ] **Step 5:** Stage the files and draft the commit message.

### Task 5: Flux calibration and γ_max

**Files:**
- Create: `illex/abc/calibrate_flux.py`
- Output: `results/illex/abc_flux_calibration.{csv,txt}`

**Interfaces:**
- Produces: `GAMMA_MAX` written into `illex/abc/constants.py`.

- [ ] **Step 1:** Confirm the tract length and the conversion rate in Korunes & Noor through CrossRef or the
  publisher. Record the DOI and the values in the constants comment. If the paper gives no tract length,
  keep 1,000 bp and say so.
- [ ] **Step 2:** Sweep γ_eff over 1e-9 to 1e-3 in 13 log steps. Use 3 fixed paths (pilot posterior
  median soft D1, a hard B and a 1.2 Ma soft N), with 32 reps each. Report dxy/π_I and π_I/π_S against
  γ_eff.
- [ ] **Step 3:** γ_eff* is the smallest value where dxy/π_I moves by more than 1 observed SE (0.0503).
  Set GAMMA_MAX = 100·γ_eff*. If no value moves it, report that to the user, because it changes the
  flux models.
- [ ] **Step 4:** Stage the files and draft the commit message.

### Task 6: Reference-table driver, local and Talapas

**Files:**
- Create: `illex/abc/run_table.py`, `illex/abc/talapas/array.sbatch`, `illex/abc/talapas/submit.sh`,
  `illex/abc/collect.py`
- Test: `tests/illex/test_abc_table.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces:
  - `run_table.py --model M --flux {0,1} --class {hard,soft} --n-accept K --task-id i --out DIR`, which
    writes `DIR/{M}_{flux}_{class}_{i}.jsonl.gz` atomically. Each row holds params, `STAT_KEYS`, `p_obs`
    and `n_nan`. The run also writes a `..._counts.json` with Counts.
  - `collect.py DIR -> results/illex/abc_table.parquet` plus `abc_acceptance.csv` (the acceptance rate
    per model × flux × class, with a binomial CI). Missing task IDs are reported.

- [ ] **Step 1: Write the failing tests**
  - `test_small_run_roundtrip`: K = 3, soft D1. Expect 3 rows with all keys.
  - `test_atomic_write`: a killed run leaves no partial `.jsonl.gz`. Simulate it with an exception after
    one row.
  - `test_collect_reports_missing`.
- [ ] **Step 2:** Run and expect FAIL. **Step 3:** Implement. sbatch follows `illex/slim/submit_talapas.sbatch`.
  **Step 4:** Run and expect PASS.
- [ ] **Step 5: Box pilot.** Run K = 250 per model × flux × class (16 combinations, 4,000 rows), using
  `W=40` workers within the load rule. Report the acceptance rates to the user.
- [ ] **Step 6:** Stage the files and draft the commit message.

### Task 7: msprime check

**Files:**
- Modify: `slim_validation/run_comparison.py` (add a scenario `illex_growth_noinv`)
- Output: `slim_validation/output/illex_growth_noinv_results.npz` and a short `.txt` verdict

- [ ] **Step 1:** Add the scenario: growth demography, Task 1 REC_RATE, 698 chromosomes, a 200 kb segment,
  msinv with no inversion against `msprime.sim_ancestry` with SMC′ off and Hudson, 200 reps each.
- [ ] **Step 2:** Run it. The pass criteria:
  - π, θ_W, Tajima's D and the projected SFS bins agree within 2 SE of the msprime mean
  - genotype r² bins agree within 10%
  - the verdict file says PASS or FAIL per statistic
- [ ] **Step 3:** Report FAIL lines to the user before Task 9. Stage the files and draft the commit message.

### Task 8: ABC inference in R

**Files:**
- Create: `illex/abc/abc_infer.R`, `illex/abc/abc_report.py`
- Output: `results/illex/abc_modelchoice.{csv,txt}`, `abc_posteriors.csv`, `abc_confusion.csv`,
  `abc_coverage.csv`

**Interfaces:**
- Consumes: `abc_table.parquet` (read via `arrow`), `abc_acceptance.csv` and `abc_observed.json`.
- Produces:
  - `abc_modelchoice.csv`: model, prob_stats (RF), p_now_term and prob_total
  - `abc_posteriors.csv`: model, parameter, median, 50% and 95% bounds, plus `hard_share` with and
    without the p_now term
  - per-model posterior draws, `abc_post_{M}_{flux}.csv`, 1,000 rows each, for Tasks 9 and 10

The combination rule, fixed by the spec: prob_total ∝ prior × acceptance × prob_stats.

- [ ] **Step 1:** Install `abcrf`, `abc` and `arrow` in the user R library. Record the versions.
- [ ] **Step 2:** Write the test `illex/abc/tests/test_abc_infer.R` on a toy table of two Gaussian models
  with means 0 and 3. RF model choice puts probability above 0.95 on the true model for an observed value
  of 3. Coverage of the 95% intervals is 0.95 ± 0.03 over 200 pods.
- [ ] **Step 3:** Run it and expect FAIL. Implement: `abcrf` with 1,000 trees, `regAbcrf` per parameter
  per model, an `abc` rejection plus loclinear cross-check at a tolerance of 0.01, a confusion matrix from
  the out-of-bag predictions, and coverage over 200 pods per model. Run and expect PASS.
- [ ] **Step 4:** Run it on the Task 6 pilot table. Report to the user:
  - the confusion matrix
  - posterior model probabilities
  - the hard/soft share both ways
  - posterior bounds
  - any parameter whose posterior piles up against a prior edge (the anti-goal: widen and rerun)
- [ ] **Step 5:** Stage the files and draft the commit message.

### Task 9: SLiM validation of msinv

**Files:**
- Modify: `illex/slim/inversion_abc.slim` (2a as I, models N/D1/D2/B with t_sel and t_stop, gene
  conversion in heterokaryotypes at γ_eff with 1 kb tracts), `illex/slim/run_one.py` (fixed-parameter
  mode `--params JSON`), `illex/slim/summarize.py` (emit `STAT_KEYS` via `illex.abc.statvec`)
- Create: `illex/abc/slim_vs_msinv.py`
- Output: `results/illex/abc_slim_validation.{csv,txt}`

- [ ] **Step 1:** Choose 6 parameter sets from the Task 8 posteriors: soft D1, hard D1, hard B, soft B, D2
  with flux and N soft.
- [ ] **Step 2:** For each, run 100 SLiM reps (Q chosen so s·Q < 0.1, Q ≤ 200) and 100 msinv reps whose
  paths come from `simulate_paths` with the same parameters. SLiM paths are not filtered on p_now. Instead
  both sets are filtered on the same ±0.02 window.
- [ ] **Step 3:** For every statistic, run a two-sample KS test and compare means in units of the pooled SD.
  Report PASS when |Δmean| < 0.25 SD. Write the table. Disagreements are reported, not tuned.
- [ ] **Step 4:** Run as a Talapas array via `illex/slim/submit.sh`. Stage the files and draft the commit
  message.

### Task 10: Production, PPCs, flux erasure and results

**Files:**
- Create: `illex/abc/ppc.py`, `illex/abc/flux_erasure.py`, `illex/abc/figures.py`
- Output: `results/illex/abc_ppc_{msinv,slim}.csv`, `abc_flux_erasure.csv`, `abc_results_table.csv`,
  `figures/abc_trajectories.png`, `figures/abc_confusion.png`

- [ ] **Step 1: Production on Talapas.** Run 2×10⁴ accepted rows per model × flux, with hard and soft
  classes run separately so their acceptance rates are measured. Collect, then rerun Task 8 on the full
  table.
- [ ] **Step 2: PPCs.** Use 500 posterior draws per surviving model (prob_total > 0.05). Simulate in SLiM
  (Task 9 machinery) and in msinv. For each statistic, including `STAT_KEYS_PPC_ONLY`, report the
  two-sided posterior predictive p-value.
- [ ] **Step 3: Flux erasure.** Use a hard start at t_inv in {1.5, 3, 5} Ma. Fit the age as the
  dxy-implied age with the Task 3 statistics at γ_eff over 1e-9 to GAMMA_MAX. Output the implied age
  against the true age.
- [ ] **Step 4: Results table.** One row per model × flux, with these columns:
  - prob_total
  - PPC failures
  - parameter medians and 95% bounds: t_inv, hard_share, p0, t_sel, s, t_stop, p*, γ_eff, p_max
  - the neutral establishment-and-rise probability (from `abc_acceptance.csv` for N)
  - a "requires" text column, written from the numbers, for example "s·2N > x" or "rise in < y ky"
- [ ] **Step 5: Figures.**
  - posterior trajectories, as the median and 50/95% bands of 2a frequency through time, for each
    surviving model
  - the confusion matrix
  Use the dataviz rules.
- [ ] **Step 6:** Report to the user. Stage the files and draft the commit message.
