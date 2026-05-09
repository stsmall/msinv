# msinv Validation Suite Design (publication-grade)

Date: 2026-05-09
Status: brainstorm-approved, pending writing-plans

**Supersedes:** `docs/superpowers/specs/2026-05-08-slim-rerun-design.md`
(scoped wrong: 3 reps at Ne=1000 cannot answer the actual validation
question. Old spec is kept for git-history record only — do not
implement against it.)

## Goal

Convince paper reviewers and future users that msinv is statistically
equivalent in simulation results to SLiM (gold-standard forward-time
simulator) and msprime + discoal (gold-standard coalescent simulators)
at realistic Anopheles-scale parameters, while being meaningfully
faster than SLiM. Produce a reproducible, statistically rigorous
validation harness covering 5 scenarios at n=100 reps each, with
pre-registered equivalence criteria.

## Five tracks

| # | Comparison | Shape | Scale | Compute target |
|---|---|---|---|---|
| 1 | msinv ↔ SLiM | Single inv (10–20% of L), 1 pop | 5 Mb, Ne=1e6, n=100 | HPC SLURM |
| 2 | msinv ↔ SLiM | Simplified Kir/Fol two-pop, 1 inv | 5 Mb, Ne=1e6 each, n=100 (50K + 50F) | HPC SLURM |
| 3 | msinv ↔ msprime | Neutral, no inv, no sweep | 5 Mb, Ne=1e6, n=100 | Local (≤50 cpu) |
| 4 | msinv ↔ discoal | Sweep, no inv: f=0, f>0, recurrent | L=1 Mb, Ne=1e6, n=100 each subscenario | Local (≤50 cpu) |
| 5 | msinv ↔ SLiM | Inversion + sweep | 5 Mb, Ne=1e6, n=100 | HPC SLURM |

Plus one **Q-bias characterization side-track** at Ne=1e3 (where SLiM
Q=1 is exact): re-run Track 1's stat panel at Q=1, 10, 100, 1000 to
bound per-stat bias. Pre-empts the reviewer Q-rescaling objection.

## Run ordering

User direction: "leave SLiM until last on the HPC; other sets use up
to 50 cpus to set up parallel runs."

Phase ordering:

0. **Pilot bench (local, 1 rep per track).** Measure msinv per-rep
   wall-time + peak RSS at L=5 Mb, Ne=1e6 to confirm full n=100 is
   tractable. If msinv per-rep > 4 hours, escalate before launching
   anything at scale. Also pilots msprime + discoal at the planned
   scenario shapes.
1. **Track 3 — msinv ↔ msprime, local.** Cheapest engine pair (msprime
   is fast). Establishes the basic neutral coalescent equivalence; if
   this fails, everything else is moot.
2. **Track 4 — msinv ↔ discoal, local.** Three subscenarios (f=0, f>0,
   recurrent). Runs alongside Track 3 if compute allows.
3. **Q-bias side-track, local.** Small Ne, fast. Methods-section data.
4. **Tracks 1, 2, 5 — msinv ↔ SLiM, HPC SLURM.** Defer to last because
   SLiM at Q=100 is the most expensive compute. Build SLURM job
   scripts after Tracks 3+4 confirm the harness + stats pipeline work.

## Shared parameters

| Parameter | Value |
|---|---|
| μ (mutation rate) | 1e-8 per bp per gen |
| r (recombination rate) | 1e-8 per bp per gen |
| Ne | 1e6 (Anopheles-scale) |
| L | 5 Mb (single chromosome arm scale) |
| Sample size n | 100 haploid lineages (Track 2: 50 + 50) |
| t_inv (inversion age) | 4·Ne = 4e6 generations ago |
| SLiM burnin | 8·Ne (rescaled, so 8e4 at Q=100) |
| SLiM Q-rescale | 100 (effective Ne=1e4) for Tracks 1, 2, 5 |
| Reps per scenario | 100 |

Seeds: deterministic offset by rep index. Per-track seed offset to
prevent cross-track correlation. Documented in the harness.

## SLiM Q-rescaling (transparent treatment)

At full Ne=1e6, SLiM forward-time at Q=1 is intractable (8·Ne = 8e6
generations × Ne individuals = 8e12 person-generations). Q=100
rescales to effective Ne=1e4 with all rates × Q (μ' = 1e-6, r' = 1e-6)
and burnin = 8·1e4 = 8e4 generations. Selection coefficient s in
sweep tracks is also rescaled (s' = s × Q).

Recap-via-msprime is **rejected** as a scaling shortcut for tracks
involving inversions: the structured-coalescent under polymorphic
inversions cannot be recapitated by msprime, which has no concept of
inversion-class barriers. (See `feedback_recap_rejected.md`.) SLiM
must run the full burnin + sim. The Q-rescaling is the only available
lever.

The Q-bias side-track (Ne=1e3 at Q=1, 10, 100, 1000) characterizes
the bias direction + magnitude per stat, supporting the methods-
section claim that Q=100 is acceptable for the main tracks.

## Demographic model — Track 2 simplified Kir/Fol

Per user spec (2026-05-09) and `project_kir_fol_model_design` memory:

- Two populations: Kir (K) and Folonzo (F) — both *Anopheles funestus*
  forms from Burkina Faso (`feedback_kir_fol_identity.md`)
- One inversion (e.g. 3Ra)
- **K is fixed for standard arrangement: p_inv_K = 0.0**
- **F is polymorphic: p_inv_F = 0.5** (balanced — exercises both arrangements equally)
- Both pops at Ne = 1e6
- Population split at t_split = 1·Ne = 1e6 generations ago
- t_inv = 4·Ne = 4e6 generations ago (older than split — inversion
  segregating in ancestral pop)
- **No post-split gene flow** (simplification per user)

This is a deliberate simplification of the published Kir/Fol
demographic model (which includes continuous deep expansion + minor
gene flow). The simplification keeps the validation harness clean
while exercising the multi-pop + structured-inversion code paths.

## Pre-registered equivalence criteria

For each (track × stat) cell, equivalence is declared when **both**:

1. **KS test:** `scipy.stats.ks_2samp(msinv_dist, ref_engine_dist)`
   yields p > 0.01 (paper-grade, not 0.05).
2. **Effect size:** Cohen's D between engine means < 0.2 (small
   effect bound per Cohen 1988).

Equivalence is **rejected** when KS p < 0.01 AND Cohen's D > 0.2.
A "needs investigation" verdict is reported in the asymmetric cases
(p < 0.01 but D < 0.2 = high statistical power found a tiny
difference; p > 0.01 but D > 0.2 = under-powered).

Reported per (track × stat × engine_pair):
- Per-rep stat values (input distributions to the KS test)
- KS p-value, Cohen's D
- Per-window mean ± SE side-by-side (visual)
- Verdict: ✅ equivalent / ❌ not equivalent / ⚠️ investigate

Aggregated to a single per-track summary table:
- track | engine_pair | n_stats_passed / n_stats_total | overall verdict

## Stats panel

Per-window (40-window grid across 5 Mb = 125 kb windows):
- π (within each subgroup: S, I; per-pop in Track 2)
- dxy (between subgroups, e.g. S↔I, K↔F)
- Fst (subgroup pairs)
- Tajima's D
- Folded SFS (1×n vector per rep, aggregated across windows)

Tree-level (per-rep distributions, "topology comparison"):
- TMRCA distribution (sample 1000 random tree positions per rep)
- Total branch length per tree (mean across trees in rep)
- Colless imbalance index (mean across trees in rep)
- Number of distinct trees per rep (`ts.num_trees`)

LD:
- r² as a function of pairwise distance, binned into 10 distance bins
  (log-spaced from 1 kb to 5 Mb), mean ± SE across reps

Selection-specific (Tracks 4 + 5 only):
- H1 (homozygosity of most common haplotype)
- H12 (combined H1 + H2)
- H2/H1 ratio (distinguishes hard vs soft sweeps)
- Spatial π profile around `x_sel`, normalized to neutral 4Nμ baseline
  (captures Kim-Stephan recovery curve)

Implementation: prefer `scikit-allel` for SFS, LD, H-stats; fall back
to hand-rolled if unavailable. tskit native for π/dxy/Fst/Tajima's D
and tree stats.

## Speed comparison framing

Headline claim: **msinv is meaningfully faster than SLiM at
realistic Anopheles-scale parameters.**

Honest comparison:
- Per-rep wall-clock at the SAME hardware (HPC node spec documented)
- Wall-clock under SLiM at Q=100 (the rescale used) — NOT Q=1
- Per-rep peak RSS (msinv tends to use more memory; report it)
- Total compute hours per track (n=100 reps)
- Speedup factor: per-rep msinv wall / per-rep SLiM-Q100 wall

This must be reported HONESTLY. If SLiM-Q100 happens to beat msinv at
some scenario, say so; the speedup story is empirical.

## Compute estimate

Order-of-magnitude per-rep wall-clock targets (1 cpu unless noted):

| Track | Engine A | Engine B | Per-rep ETA (A / B) |
|---|---|---|---|
| 1 | msinv (HEAD, 5 Mb, Ne=1e6) | SLiM Q=100 | 3-30 min / 20-60 min |
| 2 | msinv (5 Mb, 2-pop) | SLiM Q=100 (2-pop) | 5-45 min / 40-120 min |
| 3 | msinv (5 Mb neutral) | msprime (5 Mb neutral) | 3-30 min / seconds |
| 4 | msinv (sweep, no inv) | discoal | 1-15 min / 1-15 min |
| 5 | msinv (5 Mb, inv+sweep) | SLiM Q=100 (inv+sweep) | 10-60 min / 30-120 min |
| Q-bias | SLiM Q=1,10,100,1000 at Ne=1e3 | (self-comparison) | minutes/rep |

The msinv per-rep ranges are wide because L=5 Mb at Ne=1e6 hasn't been
benched at HEAD. The pilot in Phase 0 will pin these.

Total compute (rough):
- Local (50× parallel): Tracks 3 + 4 + Q-bias ≈ 1-3 days wall
- HPC SLURM: Tracks 1 + 2 + 5 ≈ 1-3 days wall at 100× parallel

Pilot must complete and pass sanity checks before any track is
launched at full n=100.

## Pilot phase 0 — bench msinv at scale

**Single rep per track, no SLiM/discoal/msprime side runs.**
Purpose: confirm msinv is tractable at n=100 before committing the
compute budget.

Measurements:
- Per-rep wall-clock (median over 3 reps)
- Per-rep peak RSS
- Per-rep iters_max consumed (any "barrier era INCOMPLETE" warnings = scale-fail)
- Per-rep tree count (sanity that the simulation actually completed)

Pilot pass/fail:
- ✅ per-rep wall < 4 hours AND peak RSS < 8 GB → proceed to full
- ⚠️ per-rep wall 4-8 hours OR peak RSS 8-32 GB → discuss with user
  before full launch (consider Q-rescale on msinv side, smaller L,
  or reduced n)
- ❌ per-rep wall > 8 hours OR peak RSS > 32 GB OR iters_max
  exhaustion → escalate; the realistic-scale claim may need rethinking

## Output / artifacts

Directory structure:
```
validation/
  _lib/
    stats.py       # tskit + scikit-allel wrappers
    equivalence.py # KS test + Cohen's D
    plot.py        # side-by-side histograms / CDFs / per-window means
    seeds.py       # deterministic seed generation per (track, rep)
  pilot/
    bench_msinv.py # phase 0 pilot
  track1_single_inv/
    run.py
    msinv_runner.py
    slim_runner.py + scenario.slim
  track2_kir_fol/
    run.py
    ...
  track3_msprime/
  track4_discoal/
  track5_inv_sweep/
  qbias/
  hpc/
    slurm_track1.sbatch
    slurm_track2.sbatch
    slurm_track5.sbatch

results/validation/
  track{1..5}/
    rep_NNN/
      msinv.trees
      <other_engine>.trees   (or .ms / .vcf)
      stats.parquet          # per-rep summary stats
  pilot/
    timing.csv

figures/validation/
  track1_pi.pdf
  track1_dxy.pdf
  ...
  equivalence_summary.pdf
  speed_comparison.pdf

docs/validation/
  equivalence_table.csv
  equivalence_table.tex
  methods.md      # for paper supplement
```

## Out of scope

- ABC inference pilot (see `project_abc_pilot_prep.md` and
  `feedback_abc_pushing.md`; user initiates separately)
- Extending the Kir/Fol model beyond the simplified 2-pop split
- Cross-engine bit-equivalence (statistical equivalence only)
- Anything outside the 5 tracks + Q-bias side-track
- Updating msinv source code; the harness is read-only on msinv

## Risks

- **msinv at L=5 Mb + Ne=1e6 may be slower than the optimistic
  estimate.** Mitigation: pilot phase 0 is a hard gate.
- **SLiM Q=100 bias on some stats may be large enough that "msinv ≠
  SLiM" doesn't mean msinv is wrong — it means the SLiM-Q100 reference
  is biased.** Mitigation: Q-bias side-track quantifies the bias;
  reviewer accepts the bias bound as part of the methods.
- **discoal sweep validation may reproduce existing D2/D3/D5 issues.**
  Mitigation: those are calibrated to 3·SE in CLAUDE.md; track 4
  carries forward those calibrations and reports the residual gap.
- **HPC SLURM scripting is new for this project.** Mitigation: write
  scripts after Tracks 3+4 confirm the harness; reviewer SLURM
  scripts in Track 1 first, then promote pattern to Tracks 2+5.
- **Topology comparison via tree-shape distributions is an approximate
  surrogate for direct topology-level equivalence.** Mitigation:
  document explicitly in methods that direct cross-engine topology
  comparison (e.g. RF distance) is not meaningful given different
  sample lineages; tree-shape statistic distributions are the
  defensible alternative.

## Deliverables (final, when all 5 tracks complete)

1. Reproducible validation harness in `validation/` (committed)
2. Raw simulation outputs in `results/validation/` (large; archived
   per project's data policy, not committed)
3. Per-rep summary stats `.parquet` (committed; small)
4. Equivalence-test summary table (`.csv` + `.tex`)
5. Plot panels per track (PDF, committed)
6. Methods section draft for the paper supplement
7. Speed comparison table (paper-ready)
