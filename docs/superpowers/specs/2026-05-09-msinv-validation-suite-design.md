# msinv Validation Suite Design (publication-grade)

Date: 2026-05-09 (revised 2026-05-10)
Status: brainstorm-approved, pending writing-plans

**Supersedes:** `docs/superpowers/specs/2026-05-08-slim-rerun-design.md`
(scoped wrong: 3 reps at Ne=1000 cannot answer the actual validation
question. Old spec is kept for git-history record only — do not
implement against it.)

**Revision 2026-05-10:** demography corrected from generic Ne=1e6 +
t_inv=4·Ne (which hit the remnant-ratchet at 180 GB RSS in pilot) to
the **v12 derivation** of the v11 Kir/Fol model — drop Ghost and Moz
pops from v11, keep K + F + KF + Anc with their Ne(t) trajectories
and the ancestral Anc size change. No K↔F migrations (the agreed
simplification).

## Goal

Convince paper reviewers and future users that msinv is statistically
equivalent in simulation results to SLiM (gold-standard forward-time
simulator) and msprime + discoal (gold-standard coalescent simulators)
at the realistic Kir/Fol-derived Anopheles parameters, while being
meaningfully faster than SLiM. Produce a reproducible, statistically
rigorous validation harness covering 5 scenarios at n=100 reps each,
with pre-registered equivalence criteria.

## v12 demography (used by ALL tracks)

Derived from `examples/kir_fol_demography.py` "LOCKED v11" by dropping
Ghost and Moz pops and the K↔F migrations. The Anc size change is
kept.

**Populations (forward-time):**

| Pop | Active from | Initial size | Notes |
|---|---|---|---|
| K | 0 | 126,772 | Kiribina; stair-step Ne(t) below |
| F | 0 | 2,496,632 | Folonzo; exponential growth from split |
| KF | merge of K+F at t_split | 86,000 | Ancestral merged pop |
| Anc | KF→Anc rename at deep change | 450,000 | Deep ancestral plateau |

**Events (backward-time, generations ago):**

```
t=0          K = 126,772; F = 2,496,632
t=200        (K: skip — t=200 dip dropped per v11 lock)
t=400        K = 161,546                           F = 1,157,768
t=600        K = 152,453                           F = 205,260
t=1,000                                            F = 1,374,810
t=1,400      K = 174,800                           F = 674,766
t=3,000      K = 182,180                           F = 340,074
t=6,200      K = 159,861                           F = 158,711
t=9,194      K-F split → KF (Ne = 86,000)
t=13,000     KF = 81,072
t=20,000     KF = 95,546
t=30,000     KF = 73,250
t=40,000     KF = 50,000   (floor, lifted from raw stairwayplot)
t=50,000     KF = 50,000
t=60,000     KF = 50,000
t=70,000     KF = 50,000
t=87,163     KF → Anc      (Ne = 450,000, deep plateau)
```

**Migrations:** NONE between K and F (the user-agreed simplification).

(The F exponential growth from 158,711 at t=9,194 to 2,496,632 at t=0
is captured by the stair-step Ne(t) in F prior to the split. Implement
either as `add_population_parameters_change` events as listed, or
equivalently as `growth_rate` + a single capping event at t=9,194 —
the math is identical.)

## Five tracks

All tracks use the **same v12 demography**. They differ in (a) which
populations are sampled, (b) whether an inversion is present, and (c)
whether a sweep is applied.

| # | Comparison | Sample structure | Inversion | Sweep | Compute target |
|---|---|---|---|---|---|
| 1 | msinv ↔ SLiM | 100 from F (~27 F_S + ~73 F_I per p_inv=0.73) | 3Ra | — | HPC SLURM |
| 2 | msinv ↔ SLiM | 50 K + 50 F (~14 F_S + ~36 F_I) | 3Ra | — | HPC SLURM |
| 3 | msinv ↔ msprime | 100 from F | — | — | Local (≤50 cpu) |
| 4 | msinv ↔ discoal | 100 from F (panmictic) | — | f=0 / f>0 / recurrent | Local (≤50 cpu) |
| 5 | msinv ↔ SLiM | 100 from F | 3Ra | hard inside inversion | HPC SLURM |

**Track 4 + discoal caveat:** discoal supports `-en` for single-pop
Ne(t) changes; it can replay the F → KF → Anc Ne(t) trajectory as a
panmictic series of size changes. The K-F split event has no discoal
analogue — for Track 4 we collapse the demography to "the F trajectory
backward to the split, then continue with the KF and Anc trajectories
as if F's lineages experienced them directly." This is the standard
collapsed-history representation used in single-pop comparisons. msinv
runs the full v12 (with K samples = 0); both engines see the same
effective Ne(t) on the F-lineage side.

## Inversion (Tracks 1, 2, 5)

- 3Ra, position 0.18·L (start), width 0.20·L. At L=10 Mb: 1.8 Mb – 3.8 Mb.
- t_inv = 330,000 generations (Small 2023, "3Ra age")
- p_inv_K = 0.0 (K is fixed for standard arrangement)
- p_inv_F = 0.73 (F is polymorphic for 3Ra)
- γ (gene conversion rate) = 1.0e-7
- mean tract length = 0.05·inv_width = 100 kb
- tract distribution: fixed (per `kir_fol_pilot.py`)

## Sweep (Tracks 4 + 5)

- **Track 4 (panmictic discoal):** three subscenarios at the same s
  and τ but different f₀:
    - Hard sweep: `mode='Deterministic'` msinv / `-ws` discoal,
      f₀ = 1/(2·N_eff)
    - Soft sweep: `mode='Stochastic'` msinv / `-wd … -f 0.05` discoal,
      f₀ = 0.05
    - Recurrent: `mode='StochasticConditioned'` msinv / `-uA <rate>`
      discoal, calibration per CLAUDE.md "discoal recurrent" notes
      (msinv `recurrent_mutation_rate = discoal_uA / (2N)`)
  Selection coefficient s = 0.05, τ = 0 (sweep ends at present),
  t_origin set so sweep completes by sampling. Matches the existing
  D2/D3/D5 calibration framework.

- **Track 5 (inversion + sweep in v12):** hard sweep inside 3Ra, on
  the F-population's I karyotype (the polymorphic one). x_sel at
  inversion midpoint (0.28·L = 2.8 Mb). s = 0.05, τ = 0, t_origin set
  so sweep completes within the F-only era (post-split, 0 ≤ t ≤ 9,194).

## Shared parameters

| Parameter | Value |
|---|---|
| μ (mutation rate) | 1.0e-8 per bp per gen |
| r (recombination rate) | 1.0e-8 per bp per gen |
| L (sequence length) | 10 Mb |
| Sample size n | 100 (Track 2: 50 + 50) |
| Reps per scenario | 100 |
| Seeds | deterministic from (track, scenario, engine, rep) via SHA-256 |

## SLiM Q-rescaling (transparent treatment)

SLiM forward-time at v12's largest Ne (F current = 2.5M) is intractable
without rescaling for n=100 reps × L=10 Mb. Q=100 rescaling: all Ne
values / Q, all rates × Q, all selection coefficients × Q, all times
scaled appropriately. Effective F_present becomes 25,000 (tractable).

**Q-bias side-track (methods support):** rerun a single-pop subset of
the v12 demography (just K's stair-step trajectory) at Q=1, 10, 100,
1000 in SLiM. Compare per-stat to characterize the bias direction and
magnitude. Pre-empts the reviewer Q-rescaling objection.

Recap-via-msprime is **rejected** as a scaling shortcut for tracks
involving inversions: the structured-coalescent under polymorphic
inversions cannot be recapitated by msprime. (See
`feedback_recap_rejected.md`.) SLiM must run the full burnin + sim.

## Run ordering

User direction: "leave SLiM until last on the HPC; other sets use up
to 50 cpus to set up parallel runs locally."

Phase ordering:

0. **Pilot bench (local, 3 reps msinv at v12).** Measure msinv per-rep
   wall + peak RSS at L=10 Mb on full v12. If per-rep > 4 hours OR
   RSS > 8 GB, escalate before launching anything at scale.
1. **Track 3 — msinv ↔ msprime, local.** Cheapest engine pair
   (msprime is fast). Establishes basic neutral coalescent equivalence
   on v12; if this fails, everything else is moot.
2. **Track 4 — msinv ↔ discoal, local.** Three sweep subscenarios.
3. **Q-bias side-track, local.** Small Ne, fast. Methods-section data.
4. **Tracks 1, 2, 5 — msinv ↔ SLiM, HPC SLURM.** SLURM scripts built
   after Tracks 3+4 confirm the harness works.

## Pre-registered equivalence criteria

For each (track × stat) cell, equivalence is declared when **both**:

1. **KS test:** `scipy.stats.ks_2samp(msinv_dist, ref_engine_dist)`
   yields p > 0.01 (paper-grade, not 0.05).
2. **Effect size:** Cohen's D between engine means < 0.2 (small
   effect bound per Cohen 1988).

Equivalence is **rejected** when KS p < 0.01 AND |D| > 0.2.
Asymmetric cases (one but not both): verdict = "investigate".

Reported per (track × stat × engine_pair):
- Per-rep stat values (input distributions to the KS test)
- KS p-value, Cohen's D
- Per-window mean ± SE side-by-side (visual)
- Verdict: ✅ equivalent / ❌ not equivalent / ⚠️ investigate

Aggregated to a single per-track summary table:
- track | engine_pair | n_stats_passed / n_stats_total | overall verdict

## Stats panel

Per-window (40-window grid across 10 Mb = 250 kb windows):
- π (per pop; per karyotype subgroup in tracks with the inversion)
- dxy (subgroup pairs)
- Fst (subgroup pairs)
- Tajima's D
- Folded SFS (1×n vector per rep)

Tree-level (per-rep distributions, "topology comparison"):
- TMRCA distribution (sample 1000 random tree positions per rep)
- Total branch length per tree
- Colless imbalance index
- Number of distinct trees per rep (`ts.num_trees`)

LD:
- r² as a function of pairwise distance, 10 log-spaced bins from
  1 kb to 10 Mb, mean ± SE across reps

Selection-specific (Tracks 4 + 5 only):
- H1, H12, H2/H1 (Garud et al. 2015)
- Spatial π profile around `x_sel`, normalized to neutral 4Nμ baseline

Implementation: tskit native for π/dxy/Fst/TajD/SFS/tree-stats.
H-stats hand-rolled (no scikit-allel dep). LD r²-decay hand-rolled.

## Speed comparison framing

Headline claim: **msinv is meaningfully faster than SLiM at the
realistic Kir/Fol v12 parameters.**

Honest comparison:
- Per-rep wall-clock at the SAME hardware (HPC node spec documented)
- Wall-clock under SLiM at Q=100 (the rescale used) — NOT Q=1
- Per-rep peak RSS (msinv tends to use more memory; report it)
- Total compute hours per track (n=100 reps)
- Speedup factor: per-rep msinv wall / per-rep SLiM-Q100 wall

This must be reported honestly. If SLiM-Q100 happens to beat msinv at
some scenario, say so; the speedup story is empirical.

## Compute estimate

Per-rep wall-clock targets are pinned by the pilot in phase 0. The
existing `examples/kir_fol_pilot.py` at L=10 Mb medium scale runs
msinv in ~80 ms/rep at the smaller Ne it uses; v12 with msinv's HEAD
hull index should land in the same order of magnitude unless deep
stair-step events introduce remnant-ratchet behavior the pilot will
expose.

SLiM at Q=100 on v12: rough estimate 20–60 min/rep at L=10 Mb based
on the April baseline (22 s/rep at L=100 kb at Q=1 implied scaling).

Total compute (rough, conditional on pilot):
- Local (≤50× parallel): Tracks 3 + 4 + Q-bias ≈ hours to days wall
- HPC SLURM: Tracks 1 + 2 + 5 ≈ 1–3 days wall at 100× parallel

## Pilot phase 0 — bench msinv on v12

**3 reps msinv at v12 demography, L=10 Mb, n=100 from F, single 3Ra
inversion at t_inv=330k.** No SLiM/discoal/msprime side runs.

Measurements per rep:
- Wall-clock
- Peak RSS
- iters_max consumed
- `ts.num_trees`, `ts.num_sites`

Pass / fail:
- ✅ per-rep wall < 4 h AND peak RSS < 8 GB → proceed to full
- ⚠️ wall 4–8 h OR RSS 8–32 GB → discuss with user before full launch
- ❌ wall > 8 h OR RSS > 32 GB OR iters_max exhaustion → escalate

## Output / artifacts

Directory structure:
```
validation/
  _lib/
    seeds.py
    stats.py
    equivalence.py
    io.py
    demography.py     # NEW: v12 builder for msinv, msprime, SLiM, discoal
  pilot/
    bench_msinv.py    # phase 0 — UPDATED to use v12
  track1_single_inv/  (msinv↔SLiM, 100 from F, 3Ra)
  track2_kir_fol/     (msinv↔SLiM, 50K + 50F, 3Ra)
  track3_msprime/     (msinv↔msprime, 100 from F, no inv)
  track4_discoal/     (msinv↔discoal, 100 from F panmictic, sweep)
  track5_inv_sweep/   (msinv↔SLiM, 100 from F, 3Ra + sweep)
  qbias/
  hpc/                # SLURM scripts for tracks 1/2/5

results/validation/
  pilot/
    rep_{000..002}/{stats.npz, timing.json}
  track{1..5}/
    rep_{000..099}/{msinv.trees, <engine>.trees, stats.npz}

figures/validation/
  track1_pi.pdf
  ...
  equivalence_summary.pdf
  speed_comparison.pdf

docs/validation/
  equivalence_table.csv
  equivalence_table.tex
  methods.md
```

## Risks

- **msinv on v12 may hit unexpected slow paths.** Mitigation: pilot
  phase 0 is the hard gate.
- **SLiM Q=100 bias.** Mitigation: Q-bias side-track quantifies it.
- **discoal sweep validation may reproduce known D2/D3/D5 residuals.**
  Mitigation: those calibrations carry forward; report residuals.
- **HPC SLURM scripting is new for this project.** Mitigation: write
  Track 1 SLURM script first; promote pattern to Tracks 2+5.
- **Topology comparison is a surrogate** (cross-engine direct topology
  distance not meaningful). Mitigation: tree-shape statistic
  distributions are the defensible alternative; documented in methods.

## Out of scope

- ABC inference pilot (user initiates separately)
- Kir/Fol model extensions beyond v12 (e.g. Ghost / Moz)
- Cross-engine bit-equivalence (statistical equivalence only)
- Updating msinv source code; the harness is read-only on msinv

## Deliverables (final, when all 5 tracks complete)

1. Reproducible validation harness in `validation/` (committed)
2. Raw simulation outputs in `results/validation/` (large; archived
   per project's data policy, not committed)
3. Per-rep summary stats `.npz` (committed; small)
4. Equivalence-test summary table (`.csv` + `.tex`)
5. Plot panels per track (PDF, committed)
6. Methods section draft for the paper supplement
7. Speed comparison table (paper-ready)
