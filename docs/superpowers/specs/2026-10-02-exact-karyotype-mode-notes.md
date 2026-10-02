# Exact karyotype mode and tiered inference — design notes (WORKING DRAFT)

Status: brainstorming in progress. Sections 1–3 agreed in conversation on
2026-10-02; section 3 had open questions when the session paused. Not an
approved spec. Next step: finish sections 4–5, write the spec, review, then
writing-plans.

## Motivation

msinv (v0.3.5 and current main) coalesces each class's material in a separate
event. Flank material still linked to an inverted chromosome therefore coalesces
as if panmictic, which weakens linkage across the breakpoints. A SLiM forward
benchmark with matched realised trajectories (200 replicates x 4 scenarios,
`illex/slim_bench/` on branch `bench/slim-inversion`) shows means agree, but
between-arrangement coalescence times have only 42-57% of SLiM's replicate SD,
even with `smc_prime=True`. Flank SD is 75-87% with SMC' (33-36% without).

Next application: Balanus, with multiple and possibly nested inversions. The
goal there is to estimate inversion age, selection (trajectory, one population
first), migration, gene flux and per-population inversion frequencies. Speed
and scalability were the past problem, so fast tiers are needed to reject
histories before slower exact models are run.

## Decisions so far

- Scope: single inversion first (with migration, per-population frequencies,
  flux and trajectories). Data model, recombination rule and frequency
  interface designed for multiple and nested inversions from the start.
- Exact mode is opt-in (default behaviour and speed unchanged), tolerating up
  to about 2x slowdown. Exact mode implies SMC' eligibility.
- Approach: karyotype state per lineage (Hudson & Kaplan 1988; Guerrero et al.
  2012 background model), not per-segment tags. Rejected: distance-based
  approximate tagging (arbitrary, approximate); per-locus two-deme coalescent
  (loses the joint genealogy and tree-sequence output).

## Section 1 — Architecture: three tiers (agreed)

- Tier 0, expectation calculator (new, fastest, no simulation). Exact
  single-site expectations from a small continuous-time Markov chain over
  sample counts by (population, state), using matrix exponentials with
  time-varying rates (cf. Rousset, Kirkpatrick & Guerrero 2014). Milliseconds
  to seconds per history; for scanning grids or likelihood surfaces.
- Tier 1, current default engine (SMC, per-class coalescence), unchanged. Fast
  genealogies and tree sequences when breakpoint linkage does not matter.
- Tier 2, exact mode (new, opt-in): lineage-level karyotype state + SMC'. For
  linkage-dependent statistics (LD near breakpoints, haplotype structure,
  divergence profile along the chromosome) and final fits.
- Consistency contract: tier 0 expectations must equal tier 1 and tier 2 means
  for the same history (built-in correctness test).
- One shared history description used by all tiers.
- Build order: (1) tier 0, single inversion; (2) tier 2, single inversion;
  (3) both extended to multiple/nested inversions (Balanus).

## Section 2 — History description and state/frequency model (agreed)

- Populations: msinv `Demography` (sizes over time, splits/merges, migration).
- Inversions: breakpoints; age `t_inv` (barrier forms; before it the locus is
  panmictic); gene-flux rate and tract length with the Peischl phi(x) profile;
  for nesting later, a parent inversion (inner exists only on the outer's
  inverted background).
- Frequency trajectories per population: f_pop(state, t) for each allowed
  karyotype combination (one inversion: p_I(t), 1 - p_I(t)). Inputs only:
  constant, deterministic (selection; one population first) or precomputed.
- States: allowed karyotype combinations given nesting. One inversion {S, I};
  B nested in A: {S.S, I.S, I.I}. Fits msinv's existing 2-bit-per-inversion
  class tag.
- Frequency interface: f_pop(state, t) and a conditional draw "given the
  karyotypes known for this chromosome, draw the rest". Tier 2 uses the draw at
  recombination; tier 0 uses the same conditionals as transition rates.
- Validation: allowed-state frequencies sum to 1 per population at every time
  (checked at setup).
- Before t_inv: the inversion's entry is dropped from the state (consistent
  with the post-barrier fix, PR #11).
- Out of scope: inferring trajectories inside msinv. Selection is estimated by
  fitting trajectory parameters (scan in tier 0, confirm in tier 2).

## Section 3 — Event rules (presented; open questions below)

Backward time. Lineage l in population d with state K; N_d(t), f_d(., t).
Written for one inversion; for several, the same rules apply per inversion
with section 2's conditional draws.

Tier 2 per-lineage events:
1. Coalescence: same population, same state, hulls overlap or touch (SMC'):
   rate 1 / (2 N_d f_d(K)); full Hudson merge.
2. Recombination at x, uniform over the lineage hull.
   - x outside the inversion (rate r per base): the half on the inversion's
     side keeps K; the other half draws a new state (I with probability
     p_d(t)) because it sits on the parent's other chromosome.
   - x inside the inversion (rate r * f_d(K) per base; only homokaryotypes
     cross over there): both halves keep K. Same r * p_c rule as current msinv.
   - General case: each half keeps the karyotypes of inversions on its own side
     of x and draws the rest from the joint frequencies conditional on what it
     kept; an inversion spanning x forces both halves to share its karyotype.
3. Migration d -> d': base rate m_{d->d'} scaled by f_{d'}(K) / f_d(K)
   (migrants carry source karyotype frequencies; an inverted lineage cannot
   migrate into a population with no inverted chromosomes). State kept.
4. Gene flux: conversion tract covering y in the inversion at rate
   gamma * phi(y) * f_d(other) (heterokaryotypes); the tract becomes a new
   lineage in the other state; the rest keeps K.
5. At t_inv: the inversion's entry is dropped from every state.

Tier 0 single-site chain at position y (lineages labelled by (population, state)):
- Coalescence within (population, state): C(n, 2) / (2 N_d f_d(K)).
- Migration as in tier 2.
- State switching (linkage at one site): flank site at distance delta from the
  nearer breakpoint switches at rate r * delta * f_d(other); inside the
  inversion only by flux, gamma * phi(y) * f_d(other).
- States merge at t_inv.
- Outputs: expected pairwise coalescence times within and between states and
  populations at any y; expected singleton fraction per arrangement (track the
  number of still-external lineages; small for up to ~20 chromosomes per
  state). Not the full SFS (use tier 1/2).

Open questions for section 3 (not yet confirmed by the user):
- Migration scaling f_{d'}(K) / f_d(K) — confirm this is the intended model.
- Recombination inside the inversion at rate r * f_d(K) — confirm.
- Tier 0 scope limited to pairwise times and the singleton fraction.

## Remaining sections to present

- Section 4: how tier 2 maps onto the rate cache (coalescence cells keyed by
  (population, lineage state); SMC' hull eligibility within state; recombination
  split handling; interaction with existing per-class code paths) and the Python
  interface (e.g. a mode switch on HullSimulator plus a separate tier 0 API).
- Section 5: validation (tier 0 vs tier 1/2 means; tier 2 vs SLiM benchmark for
  replicate variance, especially between-arrangement times; msprime limits;
  performance targets) and the Balanus extension path.

## Relevant code and results

- Rate cache: `rust/msinv-core/src/rate_index.rs` (RateCache, pair_overlap,
  compute_overlap, apply_recomb_split); event loop and merges:
  `rust/msinv-core/src/simulator.rs` (run_loop, CoalAggregate/CoalPair,
  cross_barriers_static); merge: `rust/msinv-core/src/events.rs`
  (apply_coalescence_partial).
- SMC' option (PR #13): `HullSimulator(smc_prime=False)`.
- SLiM benchmark: branch `bench/slim-inversion`, `illex/slim_bench/`.
- Known residual: ~4% excess distinct trees for n > 2 in both SMC and SMC'
  modes (xfail in `tests/hull/test_smc_prime.py`).
- msinv CI: README restored in PR #14 (maturin needs README.md).
