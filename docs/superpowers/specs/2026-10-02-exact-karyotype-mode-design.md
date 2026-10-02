# Exact karyotype mode and tiered inference — design spec

Status: written 2026-10-02 for review. Sections 1–5 agreed in conversation on
2026-10-02. Supersedes the working notes
`2026-10-02-exact-karyotype-mode-notes.md` (kept for history; two statements
there about current msinv behaviour are corrected in §3.3 below).

Scope of this spec: build steps 1 and 2 (single inversion). Step 3 (multiple
and nested inversions, Balanus) gets its own spec; §6 lists what steps 1–2
must provide so step 3 needs no rework.

## 1. Motivation

msinv (v0.3.5 and current main) coalesces each class's material in a separate
event. Flank material still linked to an inverted chromosome is tagged
panmictic and coalesces as if panmictic, which weakens linkage across the
breakpoints. The SLiM forward benchmark with matched realised trajectories
(200 replicates × 4 scenarios, `illex/slim_bench/` on branch
`bench/slim-inversion`) shows means agree, but between-arrangement
coalescence times have only 42–57% of SLiM's replicate SD, even with
`smc_prime=True`. Flank SD is 75–87% with SMC' (33–36% without).

Next application: Balanus, with multiple and possibly nested inversions.
Targets there: inversion age, selection (trajectory; one population first),
migration, gene flux and per-population inversion frequencies. Speed has been
the limiting factor, so fast tiers must reject histories before slower exact
runs.

## 2. Decisions

- Single inversion first (with migration, per-population frequencies, flux
  and trajectories). Data model, recombination rule and frequency interface
  are designed for multiple and nested inversions from the start.
- Exact mode is opt-in. Default behaviour, output and speed are unchanged.
  Exact mode may cost up to 2× wall time and peak RSS. Exact mode always uses
  SMC'.
- Approach: karyotype state per lineage (Hudson & Kaplan 1988; Guerrero et
  al. 2012 background model), not per-segment tags. Rejected:
  distance-based approximate tagging (arbitrary, approximate); per-locus
  two-deme coalescent (loses the joint genealogy and tree-sequence output).
- Model assumption, stated explicitly: random mating (Hardy–Weinberg) within
  each population, so a lineage's homologous chromosome in its parent has
  karyotype K' with probability f_d(K', t).
- Out of scope: inferring trajectories inside msinv. Selection is estimated
  by fitting trajectory parameters (scan with tier 0, confirm with tier 2).

## 3. Model

### 3.1 Three tiers

| Tier | What | Speed | Use |
|---|---|---|---|
| 0 | Expectation calculator (new, `msinv.expect`). Single-site Markov chain over lineage counts by (population, state); forward equations integrated with an adaptive ODE solver. | ms–s per history | Grid scans, likelihood surfaces, rejecting histories |
| 1 | Current default engine (`karyotype_mode="segment"`), unchanged | fast | Genealogies when breakpoint linkage does not matter |
| 2 | Exact mode (new, `karyotype_mode="exact"`): lineage-level karyotype state + SMC' | ≤ 2× tier 1 | Linkage-dependent statistics (LD near breakpoints, haplotype structure, divergence profile) and final fits |

All tiers read one history description (§3.2).

Consistency contract (built-in correctness test):
- Tier 0 expectations = tier 2 means, everywhere (inside and flank, any
  number of populations, with flux).
- Tier 0 expectations = tier 1 means only for sites inside the inversion
  and only when inversion frequencies are equal in all populations. Tier 1
  treats flank material as panmictic and migrates lineages regardless of
  karyotype (§3.3), so flank sites and migration between populations with
  different frequencies are not expected to match tier 1.

Build order: (1) tier 0, single inversion; (2) tier 2, single inversion;
(3) both extended to multiple/nested inversions (separate spec).

### 3.2 History description, states and frequencies

- Populations: msinv `Demography` (sizes over time, splits/merges,
  migration matrix `M[dst][src]`, "fraction of dst absorbing from src", i.e.
  the backward rate of a lineage in dst moving to src).
- Inversion (`InversionSpec`): breakpoints; age `t_inv` (barrier forms;
  before it, backward, the locus is panmictic); gene-flux rate γ and tract
  length with the Peischl φ(x) profile. A `parent` field is added now for
  nesting; in steps 1–2 setting it raises `NotImplementedError`.
- Frequency trajectories per population: f_d(state, t) for each allowed
  karyotype combination (one inversion: p_d(t) for I, 1 − p_d(t) for S).
  Inputs only: constant, deterministic (selection; one population first) or
  precomputed. Existing trajectory types are reused.
- States: allowed karyotype combinations. One inversion: {S, I}. (Step 3,
  B nested in A: {S.S, I.S, I.I}.) Encoded with the existing 2-bit-per-
  inversion `BranchClass`.
- Frequency interface, used by both tiers:
  - `f(pop, state, t)`;
  - `draw_conditional(pop, t, known)` — given the karyotypes already known
    for a chromosome, draw the rest from the joint frequencies. Tier 2 calls
    it at recombination; tier 0 uses the same conditionals as transition
    rates. For one inversion with nothing known it reduces to "I with
    probability p_d(t)".
- Setup checks: allowed-state frequencies sum to 1 (tolerance 1e-9) in every
  population at every trajectory time point; a lineage sampled in state K
  requires f_d(K, 0) > 0.
- At t_inv the inversion's entry is dropped from every state (consistent with
  the post-barrier fix, PR #11).

### 3.3 Corrections to the working notes (current tier 1 behaviour)

Verified in source on 2026-10-02:
- Tier 1 recombination is not frequency-scaled inside the inversion: rate is
  `total_material * recombination_rate` (`simulator.rs:390, 705`) and
  `apply_recombination` does not consult the inversion. The notes' "same
  r·p_c rule as current msinv" was wrong; the r·f_d(K) rule (§3.4, rule 2)
  is new behaviour. It does not change single-site means inside the
  inversion in tier 1, because a site's class there never changes by
  recombination.
- Tier 1 continuous migration ignores karyotype: rate
  `migration_matrix[dst][src]` per lineage (`simulator.rs:409–424`). Only
  the instantaneous `apply_class_mig` pulse is karyotype-specific.

### 3.4 Tier 2 event rules

Backward time. Lineage l in population d with state K; N_d(t), f_d(·, t).
Written for one inversion; for several, the same rules apply per inversion
with the conditional draws of §3.2.

1. Coalescence: two lineages in the same population and the same state
   whose hulls overlap or touch (SMC'): rate 1 / (2 N_d f_d(K)) per pair;
   full Hudson merge of the two lineages.
2. Recombination at x, uniform over the lineage's material at rate r per
   base.
   - x inside the inversion: effective rate r · f_d(K) (only homokaryotypes
     cross over there). Both halves keep K.
   - x outside the inversion: the half on the inversion's side of x keeps K;
     the other half sits on the parent's other chromosome and draws its
     state with `draw_conditional` (one inversion: I with probability
     p_d(t)).
   - General case (step 3): each half keeps the karyotypes of inversions on
     its own side of x and draws the rest conditional on what it kept; an
     inversion spanning x forces both halves to share its karyotype.
3. Migration d → d' (backward): rate M[d][d'] · f_{d'}(K) / f_d(K)
   (migrants carry source karyotype frequencies; first order in m, by Bayes'
   rule). State kept. A lineage in state K cannot move into a population with
   f_{d'}(K) = 0. As f_d(K) → 0 with f_{d'}(K) > 0 the rate becomes large;
   this is correct (the lineage must leave) and is handled by the time-step
   cap in §4.4.
4. Gene flux: a conversion tract covering y inside the inversion at rate
   γ · φ(y) · f_d(other) (heterokaryotypes only). The tract becomes a new
   lineage in the other state; the rest of the lineage keeps K.
5. At t_inv: the inversion's entry is dropped from every lineage's state.

### 3.5 Tier 0 single-site chain

At position y; lineages labelled by (population, state).
- Coalescence within (population, state) cell with c lineages:
  C(c, 2) / (2 N_d f_d(K)).
- Migration: as tier 2 rule 3.
- State switching (linkage at one site):
  - flank site at distance δ from the nearer breakpoint: rate
    r · δ · f_d(other);
  - inside the inversion: flux only, γ · φ(y) · f_d(other).
- At t_inv states merge (cells of the same population combine).
- Time-varying rates (trajectories, demography): the forward (Kolmogorov)
  equations are integrated with an adaptive ODE solver at a stated tolerance
  (default rtol 1e-8, atol 1e-12), with demographic and trajectory
  breakpoints passed as forced step boundaries. No piecewise-constant
  approximation of trajectories.

Outputs:
- Expected pairwise coalescence times within and between states and
  populations at any y (pair chain: about ten states).
- Expected singleton fraction per arrangement. Computed with one focal
  lineage, by exchangeability within a sampled (population, state) group: the
  chain tracks the counts of the other lineages per (population, state) cell
  plus the focal lineage's cell; expected external length of the group =
  group size × E[time to the focal lineage's first coalescence]. State-space
  size is C(n + c, c) · c for c cells (1 population, n = 40: ≈ 1.7 k;
  2 populations, n = 40: ≈ 540 k).
- Not the full SFS (use tier 1/2).
- The state space is counted at setup; above `max_states` (default 10⁶) the
  call raises a `ValueError` naming the count and suggesting tier 1/2.

## 4. Implementation mapping (tier 2)

### 4.1 Core representation

In exact mode every segment of a lineage carries the lineage's state,
including flank segments. No segment is tagged panmictic until t_inv clears
the inversion's bits. The lineage's state is therefore the (uniform) class of
its segments; a debug assertion checks uniformity after every event that
touches segments.

Consequences, reusing existing machinery:
- `compute_overlap` (`rate_index.rs`) already matches material class by
  class, so a non-empty overlap means same population and same state.
- `p_class_for_tag` already returns f_d(K), so the rate
  1 / (2 N_d f_d(K)) is unchanged.
- Coalescence in exact mode calls `apply_coalescence` (full merge), not
  `apply_coalescence_partial`.
- SMC' eligibility: today hulls that only touch are eligible only when both
  lineages are all-panmictic (`segs_all_pan`, `rate_index.rs:1581`). In exact
  mode this extends to two lineages each of uniform class, the same class.
- t_inv: `cross_barriers_static` already clears the inversion's bits from
  every segment; with uniform tags this drops the lineage's state.

### 4.2 Recombination

The total recombination rate stays `r · material`.
- Breakpoint inside the inversion: accept with probability f_d(K, t);
  rejection is a null event (thinning; exact because the bound r is
  constant). The rate aggregate is unchanged.
- Breakpoint outside: the half on the inversion's side keeps K; the far half
  draws a new state and has its segments retagged (O(segments)), followed by
  `rebuild_segs_for` on that lineage.

### 4.3 Migration

Rate M[d][d'] · f_{d'}(K) / f_d(K). No thinning (the ratio is unbounded), so
the aggregate is computed from counts per (population, state) and the
migrating lineage is picked from per-(population, state) buckets, extending
the current `pop_buckets`.

### 4.4 Time-varying rates

As in tier 1's class coalescence, rates are re-evaluated at each event and
held fixed between events. Additional safeguard in exact mode: a time-step
cap that applies only while a trajectory is moving fast — when the relative
change |Δf/f| of any f_d(K) in use over the proposed step exceeds a
tolerance (default 0.05), the step is truncated and rates re-evaluated. The
cap is inactive under constant frequencies.

### 4.5 Gene flux

The tract becomes a new lineage whose segments take the other state; the
rest of the lineage keeps K. Existing flux path, with the new lineage given a
single state.

### 4.6 Unsupported combinations in v1

Each raises an error at setup, never silently ignored:
- `karyotype_mode="exact"` with sweeps;
- with `compound_rate=True` (exact mode runs only on the main `run_loop`;
  `run_loop_compound` is untouched);
- with the pure-Python reference simulator (`msinv/hull/simulator.py`);
- with `smc_prime=False`.

The sweep stack and its `has_sv_phase` gating are untouched.

### 4.7 Python interface

- `HullSimulator(..., karyotype_mode="segment" | "exact")`, default
  `"segment"`; matching `karyotype_mode` argument on `simulate_raw`
  (`rust/msinv-py/src/lib.rs`). `"exact"` sets SMC'. Unknown values raise.
- New module `msinv/expect/` (numpy/scipy), tier 0:
  - built from the same `Demography`, `InversionSpec` and trajectory objects
    as `HullSimulator`;
  - `pairwise_times(y)` → dict of expected T within/between (population,
    state);
  - `singleton_fraction(y, sample_config)` → per arrangement;
  - `profile(ys, ...)` → vectorised over positions;
  - `max_states` argument (default 10⁶).
  - Ported to Rust only if profiling shows a need.

## 5. Validation and performance

### 5.A Default behaviour unchanged

- `karyotype_mode="segment"` output identical for the same seed: one
  golden tree sequence per existing scenario type (panmictic, inversion,
  inversion + migration, flux, sweep), tables compared byte-for-byte against
  HEAD-before-change; existing Rust and Python suites pass.
- Default-mode wall time within ±2% of HEAD (median of 20 reps), measured as
  `bench-off-path` does.

### 5.B Tier 0 correctness, independent of msinv

1. Closed forms, constant f, no flux, inside the inversion:
   T_II = 2N·p, T_SS = 2N(1 − p), T_IS = t_inv + 2N. Plus a hand-solved
   two-state pair chain at a flank site.
2. msprime as exact single-site reference: each (population, state) is an
   msprime deme of size N_d · f_d(K) (ploidy 1, pass 2·N per CLAUDE.md);
   switching r · δ · f_d(other) and the scaled real migration enter as
   migration rates; time-varying f as stepwise parameter changes. Checks
   pairwise times and the singleton fraction (from msprime's SFS). msprime
   cannot model background-dependent recombination, so it is not a tier 2
   reference.

### 5.C Tier 0 vs simulation engines

Means only; n = 100 reps; KS + Cohen's d via
`validation/_lib/aggregator.py` (α = 0.01, |d| < 0.2).
- vs tier 2: inside and flank; one and two populations; with and without
  flux.
- vs tier 1: inside the inversion with equal frequencies across populations
  only (§3.1).

### 5.D Tier 2 vs SLiM (main test)

- Rerun `illex/slim_bench` (4 scenarios × 200 reps, matched realised
  trajectories) with `karyotype_mode="exact"`.
- Pass per statistic: KS p > 0.01 and |d| < 0.2 (existing), plus variance:
  Levene p > 0.01 and SD ratio (msinv/SLiM) in [0.85, 1.15]. SD ratio
  reported for every statistic.
- Additions to the benchmark:
  1. linkage statistics: T_IS and T_flank in 3 bins of distance from the
     nearer breakpoint; r² between flank sites and the inversion marker;
  2. a two-deme scenario: inversion frequencies differ between demes
     (selection in one deme only), migration between them; per-deme
     frequencies logged by SLiM and passed to msinv as precomputed
     per-population trajectories. This is the test of the migration rule
     (§3.4 rule 3).
- Tier 1 (with `smc_prime=True`) run alongside as the documented baseline.

### 5.E Performance targets

Pilot 1 rep at production L before any batch (CLAUDE.md).

| Measure | Target |
|---|---|
| Tier 2 vs tier 1, wall and peak RSS | ≤ 2× on SLiM-bench settings and on L = 1 Mb v12 + inversion |
| Tier 0 pairwise profile, 100 positions | < 1 s |
| Tier 0 singleton fraction, 1 population, n = 40 | < 1 s |
| Tier 0 above `max_states` | clear `ValueError` |

Batches use `ProcessPoolExecutor(max_workers=50)`. Runs ≥ 1 h wall or ≥ 10
cores go overnight. Results to `results/validation/exact_karyotype/`
(large `.trees` gitignored).

## 6. Route to Balanus (step 3, separate spec)

Step 3 adds:
- states as tuples over inversions, allowed combinations only (nested:
  {S.S, I.S, I.I}); partially overlapping, non-nested inversions remain out
  of scope;
- a joint frequency input per population, defaulting to the product of
  single-inversion frequencies for unlinked inversions, read by
  `draw_conditional`;
- nesting checks at setup: an inner inversion's t_inv is younger than its
  parent's and it lies on the parent's I background;
- a nested-inversion SLiM benchmark scenario.

Steps 1–2 provide now: the frequency interface (`f`, `draw_conditional`)
used by both tiers, and the `parent` field on `InversionSpec` (raises if
set).

## 7. Relevant code

- Rate cache: `rust/msinv-core/src/rate_index.rs` (`RateCache`,
  `compute_overlap`, `segs_all_pan`, `apply_recomb_split`).
- Event loop: `rust/msinv-core/src/simulator.rs` (`run_loop`,
  `CoalAggregate`/`CoalPair`, `MigrationAggregate`, `cross_barriers_static`,
  `p_class_for_tag`).
- Merges: `rust/msinv-core/src/events.rs` (`apply_coalescence`,
  `apply_coalescence_partial`, `apply_recombination`).
- Class tag: `rust/msinv-core/src/class_tag.rs` (`BranchClass`).
- PyO3 bridge: `rust/msinv-py/src/lib.rs` (`simulate_raw`); wrapper
  `msinv/hull/_rust_bridge.py`.
- SLiM benchmark: branch `bench/slim-inversion`, `illex/slim_bench/`.
- Known residual: ~4% excess distinct trees for n > 2 in SMC and SMC'
  (xfail in `tests/hull/test_smc_prime.py`); not addressed here.
