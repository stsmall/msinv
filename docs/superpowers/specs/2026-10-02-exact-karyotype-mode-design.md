# Exact karyotype mode and tiered inference — design spec

Status: written 2026-10-02; revised 2026-10-03 after two external reviews
(round 1: `t_inv` semantics, integrated hazards, zero-frequency boundaries,
singleton statistic, corrected closed forms; round 2: global `t_inv`,
open-boundary hazards, origin-merge semantics, msprime scope,
development vs release criteria). Sections 1–5 agreed in conversation. Supersedes the working notes
`2026-10-02-exact-karyotype-mode-notes.md` (kept for history; two statements
there about current msinv behaviour are corrected in §3.3).

Scope: build steps 1 and 2 (single inversion). Step 3 (multiple and nested
inversions, Balanus) gets its own spec; §6 lists what steps 1–2 must provide
so step 3 needs no rework.

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

## 2. Decisions and definitions

- Single inversion first (with migration, per-population frequencies, flux
  and trajectories). Data model, recombination rule and frequency interface
  are designed for multiple and nested inversions from the start.
- Exact mode is opt-in. Default behaviour, output and speed are unchanged.
- Approach: karyotype state per lineage (Hudson & Kaplan 1988; Guerrero et
  al. 2012 background model), not per-segment tags. Rejected:
  distance-based approximate tagging; per-locus two-deme coalescent (loses
  the joint genealogy and tree-sequence output).
- Meaning of "exact": `karyotype_mode="exact"` conditions each lineage's
  ancestry exactly on its karyotype state, within the SMC' approximation to
  the ARG and given the supplied frequency histories. Event timing under
  time-varying rates is exact up to stated numerical tolerances (§4.4). It
  is not a full ARG and not a forward Wright–Fisher simulation.
- Frequency trajectories are exogenous conditioning histories. msinv does
  not require them to be the deterministic solution of the supplied
  migration/selection model; inconsistent histories can imply extreme
  conditional hazards, and impossible ones raise errors (§4.4).
- Random mating within each population: conditional on a chromosome being
  in state K, its homolog is an independent draw from the population's
  chromosomes, so it is in state K' with probability f_d(K', t). (A
  chromosome-level probability, not a genotype frequency such as p².)
- Out of scope: inferring trajectories inside msinv. Selection is estimated
  by fitting trajectory parameters (scan with tier 0, confirm with tier 2).

## 3. Model

### 3.1 Three tiers

| Tier | What | Speed | Use |
|---|---|---|---|
| 0 | Expectation calculator (new, `msinv.expect`). Single-site Markov chains over lineages labelled by (population, state); forward equations integrated with an adaptive ODE solver. | ms–s per history | Grid scans, likelihood surfaces, rejecting histories |
| 1 | Current default engine (`karyotype_mode="segment"`), unchanged | fast | Genealogies when breakpoint linkage does not matter |
| 2 | Exact mode (new, `karyotype_mode="exact"`): lineage-level karyotype state + SMC' + integrated hazards | target ≤ 2× tier 1 | Linkage-dependent statistics and final fits |

All tiers read one history description (§3.2).

Consistency contract (built-in correctness test):
- Tier 0 expectations = tier 2 means, everywhere (inside and flank, any
  number of populations, with flux, either `origin` setting).
- Tier 0 expectations = tier 1 means only for sites inside the inversion,
  with inversion frequencies equal in all populations, and with
  `origin="barrier"`. Tier 1 treats flank material as panmictic, migrates
  lineages regardless of karyotype (§3.3), and has no single-origin merge.

Build order: (1) tier 0, single inversion; (2) tier 2, single inversion;
(3) both extended to multiple/nested inversions (separate spec).

### 3.2 History description, states and frequencies

- Populations: msinv `Demography` (sizes over time, splits/merges,
  migration matrix). Convention, verified in source 2026-10-03
  (`simulator.rs:521, 887`; `Demography` docstring): backward in time, a
  lineage currently in population d moves to population d' at hazard
  `migration_matrix[d'][d]` (code names: `m[dst][src]`, src = the
  lineage's current population). Below, m(d → d') denotes that hazard.
  msprime's `migration_matrix[j][k]` (lineage in j moves to k) is the
  transpose.
- Inversion (`InversionSpec`): breakpoints; `t_inv`; gene-flux rate γ
  (per bp per generation, as in the existing `gene_conversion_rate`) and
  mean tract length with the Peischl φ(x) profile. A `parent` field is added
  now for nesting; in steps 1–2 setting it raises `NotImplementedError`.
- Meaning of `t_inv`, set by a new `origin` argument (exact mode and tier 0
  only; tier 1 is always `"barrier"`):
  - `origin="barrier"` (default; matches tier 1 and all fits to date):
    `t_inv` is the onset of the recombination barrier. At `t_inv` the
    inversion's entry is dropped from every state and surviving
    I-background lineages continue independently in the panmictic
    population (consistent with the post-barrier fix, PR #11).
  - `origin="single"`: `t_inv` is the origin of a single inversion
    mutation. At `t_inv` every lineage still in state I (inside or flank
    material) is merged into one lineage — the origin chromosome — and that
    lineage then drops the inversion's entry. Use this when `t_inv` is to be
    reported as inversion age.
  - `t_inv` is global to the inversion (`t_inv_max()`, as tier 1 already
    uses for `barrier_active`). Population-specific absence, arrival, loss
    or reintroduction of the inversion is represented only by the support
    of f_d(I, t); the per-population values of coupled trajectories are
    read as support boundaries (p_d = 0 beyond them), not as separate
    inversion ages. An I lineage in a population whose I frequency reaches
    zero before `t_inv` must leave through the modelled ancestry process
    before that boundary, or the history is impossible (§4.4).
  - `origin="single"` additionally requires that at most one population has
    f_d(I, t) > 0 immediately below `t_inv` (the origin population; checked
    at setup); the origin chromosome is placed there.
- Frequency trajectories per population: f_d(state, t) for each allowed
  karyotype combination (one inversion: p_d(t) for I, 1 − p_d(t) for S).
  Inputs only: constant, deterministic (selection; one population first) or
  precomputed. Existing trajectory types are reused.
- States: allowed karyotype combinations. One inversion: {S, I}. Encoded
  with the existing 2-bit-per-inversion `BranchClass`. (Step 3, B nested in
  A: {S.S, I.S, I.I}, first component the parent inversion.)
- Frequency interface, used by both tiers:
  - `f(pop, state, t)`;
  - `draw_conditional(pop, t, known)` — given the karyotypes already known
    for a chromosome, draw the rest from the joint frequencies. Tier 2 calls
    it at recombination; tier 0 uses the same conditionals as transition
    rates. For one inversion with nothing known it reduces to "I with
    probability p_d(t)".
  - Invariants (checked at setup and in debug builds): the trajectory
    representation preserves the simplex at all times, not only at knots
    (for one inversion, f_S = 1 − p_I is derived, never stored; step 3 must
    derive one state's frequency from the others or interpolate in a
    simplex-preserving parameterisation); `f` in [0, 1] within 1e-12; allowed-state frequencies sum to 1 within 1e-9 in every
    population at every trajectory knot; `draw_conditional` returns only
    allowed states, derives its probabilities from the same joint table as
    `f`, and raises if the conditioning event has zero probability; no
    frequency is read at a time other than the one asked for (no stale
    caching).
- Setup checks: a lineage sampled in state K requires f_d(K, 0) > 0.

### 3.3 Corrections to the working notes (current tier 1 behaviour)

Verified in source on 2026-10-02:
- Tier 1 recombination is not frequency-scaled inside the inversion: rate is
  `total_material * recombination_rate` (`simulator.rs:390, 705`) and
  `apply_recombination` does not consult the inversion. The r·f_d(K) rule
  (§3.4 rule 2) is new behaviour. It does not change single-site means
  inside the inversion in tier 1, because a site's class there never changes
  by recombination.
- Tier 1 continuous migration ignores karyotype: rate
  `migration_matrix[dst][src]` per lineage (`simulator.rs:409–424`). Only
  the instantaneous `apply_class_mig` pulse is karyotype-specific.
- Tier 1 `t_inv` is a barrier onset: every trajectory type returns
  `p_inv = 0` for `t ≥ t_inv` (`trajectory.rs`) and `cross_barriers_static`
  clears the tags; nothing forces I lineages to coalesce at `t_inv`.
  Consequence outside this spec: the Illex "arose ~850 ka" estimate is a
  barrier-onset time under this model, not a single-origin age; whether
  `origin="single"` would move it has not been checked.

### 3.4 Tier 2 event rules

Backward time. Lineage l in population d with state K; N_d(t), f_d(·, t).
Written for one inversion; for several, the same rules apply per inversion
with the conditional draws of §3.2. All hazards are time-dependent and are
timed by §4.4.

1. Coalescence: two lineages in the same population and the same state
   whose hulls overlap or touch (SMC'): hazard 1 / (2 N_d(t) f_d(K, t)) per
   pair; full Hudson merge of the two lineages.
2. Recombination at x, uniform over the lineage's material at rate r per
   base.
   - x inside the inversion: accepted hazard r · f_d(K, t). By §2's
     random-mating statement the homolog is also K with probability
     f_d(K, t), and only homokaryotypes cross over inside the inversion.
     Both halves keep K.
   - x outside the inversion: the half on the inversion's side of x keeps K;
     the other half sits on the homolog and draws its state with
     `draw_conditional` (one inversion: I with probability p_d(t)).
   - General case (step 3): each half keeps the karyotypes of inversions on
     its own side of x and draws the rest conditional on what it kept; an
     inversion spanning x forces both halves to share its karyotype.
3. Migration d → d' (backward): conditional on observing a lineage in state
   K in population d, the backward migration hazard to source population d'
   is the unconditioned hazard times the source/destination frequency ratio,
   m(d → d', t) · f_{d'}(K, t) / f_d(K, t). State kept. A state with zero
   frequency in d' cannot be entered by migration. As f_d(K, t) → 0 with
   f_{d'}(K, t) > 0 the hazard can diverge; this is a genuine
   time-inhomogeneous boundary, handled by §4.4, not a large finite rate.
4. Gene flux: γ is a per-bp per-generation initiation rate. A lineage's
   flux hazard is γ · f_d(other, t) · ∫ φ over its material inside the
   inversion (`flux_lineage_rate_arena`, `simulator.rs:2346`); the event
   position x is drawn with density ∝ φ over that material
   (`sample_flux_position`) and a tract of length l (fixed or exponential,
   capped at 0.99 · inversion length) is placed uniformly among positions
   containing x, clipped to the inversion (`draw_tract`). The tract becomes
   a new lineage in the other state; the rest keeps K. Corrected
   2026-10-03: the per-site conversion hazard is not γ · φ(y) · f; for a
   lineage carrying the whole inversion it is γ · f_d(other, t) · C(y),
   where C(y) = E_l[∫ φ(x) · P(tract from x of length l covers y) dx]
   ≈ φ(y) · mean tract length away from the breakpoints. For a lineage
   whose material near y is fragmentary, the current rule (event positions
   drawn only within ancestral material) gives a smaller per-site hazard.
   Decided 2026-10-03: exact mode initiates tracts wherever they would
   overlap ancestral material (Wiuf & Hein 2000), so every ancestral site
   is converted at γ · f_d(other, t) · C(y) regardless of the surrounding
   material, matching tier 0. Tier 1 keeps its current rule. Recombination
   cannot enter a zero-frequency state either: if f_d(other, t) = 0 the
   flux hazard is zero.
5. At `t_inv`: `origin="barrier"` drops the inversion's entry from every
   lineage's state; `origin="single"` first applies the origin merge (§4.6),
   then drops the entry.

### 3.5 Tier 0 single-site chains

At position y; lineages labelled by (population, state).
- Coalescence within a (population, state) cell with c lineages:
  C(c, 2) / (2 N_d(t) f_d(K, t)).
- Migration: as tier 2 rule 3.
- State switching (linkage at one site):
  - flank site at distance δ from the nearer breakpoint: rate
    r · δ · f_d(other, t) (uniform recombination rate over that interval);
  - inside the inversion: flux only, γ · C(y) · f_d(other, t), with C(y)
    from rule 4.
- At `t_inv`: states merge (`origin="barrier"`), or all I lineages first
  merge into one (`origin="single"`).
- Time-varying rates: the forward (Kolmogorov) equations are integrated
  with an adaptive ODE solver (default rtol 1e-8, atol 1e-12), with
  demographic and trajectory knots and `t_inv` as forced step boundaries.
  An interval ending at a zero-support boundary t_b is integrated on
  [t₀, t_b − ε) only; rates are never evaluated at t_b. Where f_d(K, t)
  declines continuously to zero, exit hazards diverge only
  logarithmically, so mass left at t_b − ε is not negligible; it is
  resolved by the limit chain: all hazards out of the cell scale as
  1/f_d(K), so the remaining mass is redistributed by the absorption
  probabilities of the rescaled chain (coalescence within the cell at
  1/(2N_d), migration at m(d → d') f_{d'}(K, t_b)). Mass that this chain
  cannot move (no exit), or that sits in a cell whose frequency jumps to
  zero (no divergence), is the impossible-history case (§4.4) and raises
  if above the solver tolerance.

Outputs:
- Pairwise coalescence times within and between states and populations at
  any y: mean, variance and survival function P(T > t) (pair chain, about
  ten states).
- Expected site frequency spectrum of a sample from one arrangement (and
  population), by the block-counting process (Hobolth, Siri-Jégousse &
  Bladt 2019): states are multisets of (cell, number of sampled
  descendants) for the lineages; E[ξ_i] ∝ expected time-integrated total
  length of branches subtending i sampled chromosomes. Computed directly at
  the projection size (projecting an expected SFS by hypergeometric
  sampling equals computing it at the smaller size).
- `f1`, defined exactly as the Illex statistic (`illex/scripts/sfs_shape.py`
  `project_unfolded`, `refit_decline.py:191`): the folded expected SFS at
  projection size n (default 20 haploid chromosomes), first class divided by
  the total: f1 = (E[ξ_1] + E[ξ_{n−1}]) / Σ_i E[ξ_i]. A ratio of
  expectations, not E[S_1/S].
- Not the full joint SFS across arrangements or populations (use tier 1/2).
- State-space size of the SFS chain (partitions of n into cell-coloured
  parts): one population, two states, n = 20: 24,842; two populations,
  n = 20: 2,778,517, above the default limit. The SFS chain's state space
  is counted at setup; above `max_states` (default 10⁶, an argument of the
  SFS functions only) the call raises a `ValueError` naming the count and
  suggesting tier 1/2. Pairwise calculations are unaffected at any number
  of populations.

## 4. Implementation mapping (tier 2)

### 4.1 Core representation

In exact mode every segment of a lineage carries the lineage's state,
including flank segments. No segment is tagged panmictic until `t_inv`. The
lineage's state is the (uniform) class of its segments.

Consequences, reusing existing machinery:
- `compute_overlap` (`rate_index.rs`) matches material class by class, so a
  non-empty overlap means same population and same state.
- `p_class_for_tag` returns f_d(K, t) for the coalescence hazard.
- Coalescence in exact mode calls `apply_coalescence` (full merge), not
  `apply_coalescence_partial`.
- SMC' eligibility: today hulls that only touch are eligible only when both
  lineages are all-panmictic (`segs_all_pan`, `rate_index.rs:1581`). In
  exact mode this extends to two lineages each of uniform class, the same
  class.
- `t_inv`: `cross_barriers_static` clears the inversion's bits from every
  segment; with `origin="single"` it is preceded by the I-lineage merge
  (§4.6).

### 4.2 Recombination

The proposal rate stays `r · material` (constant between events).
- Breakpoint inside the inversion: accept with probability f_d(K, t) at the
  event time; rejection is a null event (thinning; exact because the
  proposal rate is constant). The proposal rate, not the accepted rate, is
  the term in the total hazard of §4.4.
- Breakpoint outside: the half on the inversion's side keeps K; the far half
  draws a new state and has every segment retagged (O(segments)) before
  `rebuild_segs_for` on that lineage.

### 4.3 Migration

Hazard m(d → d', t) · f_{d'}(K, t) / f_d(K, t). The migrating lineage is
picked from per-(population, state) buckets, extending the current
`pop_buckets`.

### 4.4 Time-varying hazards (integrated)

Between events, lineage counts per cell, eligible pair counts per
(population, state) bucket, total material and per-(population, state) flux
weights (∫φ over material, without the frequency factor) are fixed. The total
hazard is therefore

  Λ(t) = Σ_k c_k · g_k(t) + r · material,

where each g_k is one of a small set of known functions of time:
1 / (N_d(t) f_d(K, t)) (coalescence), m(d → d', t) f_{d'}(K, t) / f_d(K, t)
(migration), f_d(other, t) (flux). At setup, the cumulative integrals
G_k(t) = ∫₀ᵗ g_k are tabulated on a grid containing every demographic knot,
trajectory knot and `t_inv`; within a grid cell the integral is analytic
where the form is known (constant; linear interpolation) and by adaptive
quadrature otherwise (rel. tol. 1e-10). Grids end at `t_inv` (beyond it no
rate depends on frequency).

Event step from t₀: draw E ~ Exp(1); solve Σ_k c_k (G_k(t₁) − G_k(t₀)) +
r · material · (t₁ − t₀) = E for t₁ by bracketed root search on this
monotone function, stopping at the next forced boundary if it comes first
(apply the boundary, redraw). At t₁ choose the event class with probability
proportional to its instantaneous hazard. Every event ends the current
waiting interval, including a rejected recombination proposal (a null
ancestry event): the scheduler restarts from its time with a fresh
exponential variate. Constant-frequency, constant-size epochs use the
closed form (fast path).

Zero-support boundaries: a grid cell ending at a time t_b where some
f_d(K, t_b) = 0 (other than `t_inv`) is open at its upper end. If
lim_{t→t_b⁻} G_k(t) = +∞ the endpoint is stored as an explicit divergent
marker, not a finite value; root finding is restricted to t < t_b, and
f(K, t_b) is never used as a denominator. Where an occupied cell's exit
hazard diverges (migration to a supported population, or coalescence with
another lineage in the cell), the root lands before t_b and the lineage
leaves or coalesces first.

Impossible history (operational definition, both tiers): an occupied
(population, state) cell reaches a zero-support boundary that is not
`t_inv`, with ancestry still in it — in tier 2, the scheduler reaches t_b
with a lineage still in the cell; in tier 0, mass above tolerance remains
as t → t_b. Gene flux and flank recombination do not count as exits (each
leaves part of the lineage in K). The simulator raises an error naming the
population, state and time. Exogenous trajectories that are dynamically
inconsistent with the migration/selection model are otherwise valid
conditioning histories as long as no such boundary is hit.

The drop to zero at `t_inv` is a forced boundary, not a singularity: rule 5
is applied before any frequency is read as zero.

Tier 1 keeps its current scheduler (rates re-evaluated at events).

### 4.5 Gene flux

The tract becomes a new lineage whose segments take the other state; the
rest of the lineage keeps K. Existing flux path, with the new lineage given a
single state.

### 4.6 Single-origin merge

Semantic operation `force_origin_merge(I, t_inv)`, applied when two or more
lineages are in state I at `t_inv`. Invariants:
- exactly one lineage in state I remains afterwards (the origin
  chromosome), in the origin population, before the state entry is dropped;
- at most one new node, at time `t_inv`, created only if some position is
  carried by two or more I lineages;
- at each position carried by k ≥ 2 I lineages, the k child segments get
  edges to that node (a k-way multifurcation; no arbitrary binary
  resolution, no chain of nodes at equal or near-equal times);
- at positions carried by one I lineage, the segment keeps its existing
  node (no unary edge); disjoint material stays disjoint ancestral material
  on the one origin lineage;
- no zero-length edges; total material is conserved.

It is a dedicated k-way merge, not a reuse of the sweep force-coalesce
(`coalesce_uid_group`, `simulator.rs:3248`): that function skips disjoint
pairs (which would leave I lineages unmerged) and gives each binary merge a
staggered time (`next_sweep_merge_t`). Whether `apply_coalescence`
internals can be shared is an implementation choice, provided the
invariants hold.

### 4.7 Unsupported combinations in v1

Each raises an error at setup, never silently ignored:
- `karyotype_mode="exact"` with sweeps;
- with `compound_rate=True` (exact mode runs only on the main `run_loop`;
  `run_loop_compound` is untouched);
- with the pure-Python reference simulator (`msinv/hull/simulator.py`);
- with `smc_prime=False`;
- `origin` other than `"barrier"` in `karyotype_mode="segment"`.

The sweep stack and its `has_sv_phase` gating are untouched.

### 4.8 Python interface

- `HullSimulator(..., karyotype_mode="segment" | "exact",
  origin="barrier" | "single")`, defaults `"segment"`, `"barrier"`;
  matching arguments on `simulate_raw` (`rust/msinv-py/src/lib.rs`).
  `"exact"` sets SMC'. Unknown values raise.
- New package `msinv/expect/` (numpy/scipy), tier 0:
  - built from the same `Demography`, `InversionSpec` and trajectory objects
    as `HullSimulator`, plus `origin`;
  - `pairwise_times(y)` → mean, variance and survival function per
    (population, state) pair;
  - `expected_sfs(y, pop, state, n)` → unfolded expected SFS as E[L_b]
    (generations per unit mutation rate; E[ξ_b] = μ·E[L_b]);
  - `f1(y, pop, state, n=20)` → folded ratio of expectations (§3.5);
  - `profile(ys, ...)` → vectorised over positions;
  - `max_states` argument on the SFS functions only (default 10⁶).
  - Ported to Rust only if profiling shows a need.

## 5. Validation and performance

Release pass rule (simulation comparisons):
- per statistic, bootstrap 95% CI of Cohen's d within ±0.2;
- where variance is tested, bootstrap 95% CI of the SD ratio within
  [0.85, 1.15];
- KS and Brown–Forsythe/Levene reported as diagnostics only;
- quantiles 0.1, 0.5, 0.9 reported with bootstrap CIs.
Release replicate count is chosen from pilot precision so that both CIs can
fit their bands under equality (≈ 1000; at n = 200 the SD-ratio CI
half-width is ≈ ±14%), subject to a 1-rep pilot showing it is affordable.

Development smoke tests use 100 replicates and report the same quantities,
but do not apply the release CI gates (at n = 100 the 95% CI of d under
equality is already ≈ ±0.28). They fail only if |d| > 0.3 or the SD ratio
is outside [0.7, 1.4], or on disagreement with an analytic value. Where the same
realised trajectory is used across engines it is kept as a blocking factor
in summaries; replicates are not called paired.

### 5.A Default behaviour unchanged

- `karyotype_mode="segment"` with the same seed gives identical
  node/edge/site/mutation/population/individual/migration tables (provenance
  excluded): one golden tree sequence per existing scenario type
  (panmictic, inversion, inversion + migration, flux, sweep) against
  HEAD-before-change. Existing Rust and Python suites pass.
- Default-mode wall time within ±2% of HEAD (median of 20 reps), measured as
  `bench-off-path` does.

### 5.B Tier 0 correctness, independent of msinv

1. Closed forms, constant f, no flux, no migration, inside the inversion,
   with a_K = 2N f_K (f_I = p, f_S = 1 − p):
   - `origin="barrier"`:
     P(T_KK > t) = e^{−t/a_K} for t < t_inv, and
     e^{−t_inv/a_K} e^{−(t − t_inv)/(2N)} for t ≥ t_inv;
     E[T_KK] = a_K (1 − e^{−t_inv/a_K}) + 2N e^{−t_inv/a_K}.
     P(T_IS > t) = 1 for t < t_inv, e^{−(t − t_inv)/(2N)} after;
     E[T_IS] = t_inv + 2N.
   - `origin="single"`: E[T_II] = a_I (1 − e^{−t_inv/a_I}) (I/I pairs
     coalesce at `t_inv` at the latest); S/S and I/S as above.
   - Limits: t_inv → ∞ gives 2N f_K; t_inv → 0 gives 2N (barrier);
     p → 1 gives E[T_II] → 2N (barrier).
   Means and full survival functions are tested.
2. Plus a hand-solved two-state pair chain at a flank site.
3. msprime single-site reference, `origin="barrier"` only. Each
   (population, state) is an msprime deme with `ploidy=1` and size
   `2 · N_d(t) · f_d(K, t)`; switching r · δ · f_d(other) and the
   conditional scaled migration m(d → d') f_{d'}(K) / f_d(K) enter as
   migration rates (msprime matrix entry [cell(d, K)][cell(d', K)], i.e.
   transposed relative to msinv's `migration_matrix`).
   - Exact reference when tier 0 is given the same piecewise-constant
     history (sizes, frequencies, switching and migration rates).
   - Convergence reference for continuous trajectories: step widths 100,
     50, 20, 10, 5 generations, showing convergence to tier 0.
   Checks pairwise times and the expected SFS / f1. `origin="single"` is
   validated by the analytic results (item 1), tier 0 boundary transitions
   and SLiM, not by this construction. msprime cannot model
   background-dependent recombination, so it is not a tier 2 reference.
4. Identity cases: two identical demes with symmetric migration collapse to
   the expected structured result; equal frequencies across populations
   reduce the migration rule to plain m(d → d'); r = 0 and γ = 0 give no switching;
   p = 0 or 1 outside the inversion gives no stochastic state change.
5. Migration indexing: f_{d'}(K) = 0 gives exactly zero hazard into d';
   a small-Δt forward calculation of P(source d' | state K, now in d) ∝
   f_{d'}(K) · P(d' | d) reproduces the implemented conditional rates to
   first order, with source and destination not swapped.

### 5.C Tier 0 vs tier 2

At single sites, constant and time-varying trajectories; development smoke
tests at n = 100, release at the release count, under the §5 rules:
- pairwise mean, variance and survival curve; state occupancy through time;
  migration and flux event probabilities; expected SFS / f1;
- inside and flank; one and two populations; with and without flux; both
  `origin` settings.
Tier 0 vs tier 1: inside the inversion, equal frequencies, `origin="barrier"`
only (§3.1).

### 5.D Tier 2 vs SLiM (main test)

- Rerun `illex/slim_bench` (4 scenarios, matched realised trajectories) with
  `karyotype_mode="exact"`, at the release replicate count.
- Additions to the benchmark:
  1. linkage statistics: T_IS and T_flank in 3 bins of distance from the
     nearer breakpoint; r² between flank sites and the inversion marker;
  2. a two-deme scenario: inversion frequencies differ between demes
     (selection in one deme only), migration between them; per-deme
     frequencies logged by SLiM and passed to msinv as precomputed
     per-population trajectories (tests §3.4 rule 3);
  3. single-origin comparison: SLiM's inversion arises once, so
     `origin="single"` is compared against SLiM for T_II, T_IS, π within I
     inside the inversion, and the flank distance-bin statistics (means and
     SDs). The one-I-ancestor-at-origin property is trivially true in SLiM
     and is tested as an msinv invariant instead (§5.E).
- Tier 1 (`smc_prime=True`) run alongside as the documented baseline.

### 5.E Exact-mode unit and invariant tests

- SMC' eligibility, same state vs different state, for: overlapping
  material; disjoint material with overlapping hulls; hulls touching at one
  coordinate; and transition through `t_inv` (different states become
  compatible).
- After every exact-mode event in debug/test builds: every lineage has one
  uniform state; no lineage occupies a zero-support state; rate-cache
  bucket counts equal a direct recomputation; total ancestral material is
  conserved across recombination and flux as in the existing simulator.
- Integrated-hazard scheduler: event-time distribution under a known
  time-varying hazard matches the analytic inverse; zero-frequency
  boundary: a lineage in a state whose frequency falls linearly to zero
  always leaves before the boundary; the error path fires for an impossible
  history.
- `draw_conditional` empirical frequencies agree with `f`.
- Origin merge: every §4.6 invariant on hand-built lineage sets (all
  overlapping; all disjoint; mixed; a single I lineage; none).
- Null recombination proposals restart the scheduler (§4.4).

### 5.F Performance

Pilot 1 rep at production L before any batch (CLAUDE.md). The ≤ 2× figure
is an engineering target, reported, not a release gate.

| Measure | Target |
|---|---|
| Tier 2 vs tier 1, wall and peak RSS | ≤ 2× on SLiM-bench settings and on L = 1 Mb v12 + inversion |
| Tier 2 stress cases (reported) | p ≈ 0.5; p ≈ 0.05; p declining 0.05 → 0.001 without reaching zero; fast-changing trajectory; two demes with strongly different frequencies; flux on/off |
| Tier 0 pairwise profile, 100 positions | < 1 s |
| Tier 0 f1, 1 population, n = 20 | < 5 s |
| Tier 0 above `max_states` | clear `ValueError` |

Batches use `ProcessPoolExecutor(max_workers=50)`. Runs ≥ 1 h wall or ≥ 10
cores go overnight. Results to `results/validation/exact_karyotype/`
(large `.trees` gitignored).

## 6. Route to Balanus (step 3, separate spec)

Step 3 adds:
- states as tuples over inversions, allowed combinations only (nested:
  {S.S, I.S, I.I}, parent first); partially overlapping, non-nested
  inversions remain out of scope;
- a joint frequency input per population, defaulting to the product of
  single-inversion frequencies for unlinked inversions, read by
  `draw_conditional`;
- nesting checks at setup: an inner inversion's `t_inv` is younger than its
  parent's and it lies on the parent's I background;
- a recombination map, if needed (replace r·δ by integrated map distance);
- a nested-inversion SLiM benchmark scenario.

Steps 1–2 provide now: the frequency interface (`f`, `draw_conditional`)
with its invariants, the `origin` argument, and the `parent` field on
`InversionSpec` (raises if set).

## 7. Relevant code

- Rate cache: `rust/msinv-core/src/rate_index.rs` (`RateCache`,
  `compute_overlap`, `segs_all_pan`, `apply_recomb_split`).
- Event loop: `rust/msinv-core/src/simulator.rs` (`run_loop`,
  `CoalAggregate`/`CoalPair`, `MigrationAggregate`, `cross_barriers_static`,
  `p_class_for_tag`, `flux_lineage_rate_arena`; `coalesce_uid_group`
  for contrast only, see §4.6).
- Trajectories: `rust/msinv-core/src/trajectory.rs` (`Trajectory` trait,
  `p_inv_at`, `t_inv`).
- Merges: `rust/msinv-core/src/events.rs` (`apply_coalescence`,
  `apply_coalescence_partial`, `apply_recombination`).
- Class tag: `rust/msinv-core/src/class_tag.rs` (`BranchClass`).
- PyO3 bridge: `rust/msinv-py/src/lib.rs` (`simulate_raw`); wrapper
  `msinv/hull/_rust_bridge.py`.
- Illex f1 definition: `illex/scripts/sfs_shape.py` (`project_unfolded`,
  `PROJ = 20`), `illex/scripts/refit_decline.py:191`.
- SLiM benchmark: branch `bench/slim-inversion`, `illex/slim_bench/`.
- Known residual: ~4% excess distinct trees for n > 2 in SMC and SMC'
  (xfail in `tests/hull/test_smc_prime.py`); not addressed here.
