# Progressive coalescence sweep extension — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Branch:** `feat/sweep-progressive-coalescence`
**Predecessor:** `2026-04-30-sweep-per-segment-hitchhiking-design.md` —
that extension fixed the spatial profile (segments at distance `d`
hitchhike with prob `exp(-r·d·T_eff)`). Surfaced when the spatial
profile worked but global pi_branch and n_trees still differed from
discoal by 70% / 64% on D2 hard sweep — the temporal coalescence model
is wrong.

## Goal

Replace msinv's "all A-tagged lineages force-coalesce at `t_origin`"
behavior with **progressive coalescence during the sweep window**
matching discoal's allele-conditional rate model:

- A-tagged pairs coalesce at rate `1 / (2·N·p_A(t))` per generation.
- a-tagged pairs coalesce at rate `1 / (2·N·(1-p_A(t)))` per generation.
- Mixed (A × a) pairs do not coalesce during the sweep window (rate 0).

Going backward through the window, p_A drops from ~1 at τ to f0 at
t_origin, so the A-tagged rate explodes (small subpop) and the a-tagged
rate normalizes (subpop grows). Most A-tagged lineages coalesce within
the window; any stragglers at `t_origin` are force-coalesced into the
single founder (the A allele didn't exist before t_origin).

This brings msinv's sweep model into temporal alignment with discoal's
canonical coalescent (verified at
`/home/adkern/discoal/src/core/discoalFunctions.c:1981-1984`).

## Scope (locked decisions)

**Q1 → B (both A and a).** Progressive rate enhancement applies
symmetrically to A-tagged and a-tagged lineage pairs. Even though the
a-tagged effect is small for typical hard-sweep params (few a-tagged
lineages near τ when p_A ≈ 1), keeping symmetry avoids drift under
partial sweeps and soft sweeps where the a/A balance is different.

**Q2 → A (endpoint as A-only founder convergence at t_origin).**
- Hard sweep (`target_freq=1.0`): redundant in practice (progressive
  collapses A pool by t_origin). Keep as defensive backstop.
- Partial sweep (`target_freq=0.5`): **required for correctness**.
  All A-tagged lineages still trace back to the single founder allele
  at t_origin even though sample-time freq is partial. Endpoint is
  the only mechanism that converges them.

The endpoint **only force-coalesces A-tagged** lineages, never
a-tagged. a-tagged lineages just continue past t_origin as normal
neutral-coalescent lineages.

## Out of scope

- A/a-aware `BranchClass` encoding. Currently the A/a tag is per-
  lineage (`HashMap<LinUid, bool>`). Encoding it into BranchClass
  bits would let `iter_class_totals` differentiate without a
  HashMap walk, but that's a class_tag rewrite. Defer.
- Recurrent + progressive interaction. Recurrent sweeps (D5) add
  new A-tagged lineages mid-sweep via the `recurrent_mutation_rate`
  parameter. Combined with progressive coalescence the dynamics get
  more involved. Document the combination as a known approximate
  area; D5 validation will surface any drift.
- Per-position A/a tagging. The A tag is on the lineage as a whole;
  recombination during the sweep can separate the x_sel-overlapping
  segment from the rest, but the lineage retains its tag.
  Approximation: lineage-level tag is good enough for typical sweep
  params (s ~ 0.05, T_eff ~ 200-500 gens, r·d·T_eff < 1 for
  positions within the chromosome). Future refinement could re-tag
  on each recombination event.

## Architecture

### Where the rate applies

`rust/msinv-core/src/simulator.rs:1697` — the existing
`emit_coal_events_from_cache` is the sweep-aware rate emitter. It
iterates `cache.iter_class_totals()` returning `(pop, cls, count)`.

Currently during the sweep window, a single rate event per cell is
emitted: `count / (2·N·p_kary(t,pop,kary))`. We replace this with
**three rate events per cell** (during the sweep window):

| Subgroup | Pair count | Rate denom | Notes |
|---|---|---|---|
| A-tagged | `n_A choose 2` | `2·N·p_kary·p_A(t)` | `p_A(t)` from trajectory |
| a-tagged | `n_a choose 2` | `2·N·p_kary·(1-p_A(t))` | |
| Untagged | `n_U choose 2 + n_U·(n_A + n_a)` | `2·N·p_kary` | normal Hudson; pairs include U-with-A and U-with-a |

The existing kary-conditional `p_kary` factor stays in for inversion
scenarios; for pure panmictic it degenerates to 1 (per the prior
spec's piece 2 fallback).

**Critical: A × a pairs (one A-tagged lineage with one a-tagged
lineage) contribute ZERO rate during the sweep window.** Same as
discoal — they're on different allele backgrounds at x_sel and can't
coalesce there. They'll coalesce after t_origin if at all.

A × untagged and a × untagged pairs use the **normal** rate (no
enhancement, no zero-out). The untagged pool is "outside the sweep"
in the sense that `apply_sweep` decided not to put them in either
allele subpop.

### How counts are obtained

Currently `cache.iter_class_totals()` returns total counts per
(pop, cls). To get per-allele subgroups, we walk `active` lineages
once at the start of `emit_coal_events_from_cache` (or at the start
of each rate-recompute), looking up each lineage's UID in `a_tag`
and bucketizing into `(pop, cls, A|a|U)` counters.

Cost: O(|active|) per rate emit during the sweep window. With
typical `|active| ≈ n = 10` the cost is negligible. For larger n
(hundreds of lineages) the per-emit cost would grow but is bounded.

### Rate emission

Per (pop, cls) cell during the sweep window:

```pseudo
n_A, n_a, n_U = bucketize lineages by allele tag
# A-tagged pairs (only matters if n_A >= 2)
if n_A >= 2:
    pairs = n_A * (n_A - 1) / 2
    denom = 2 * N * p_kary(t, pop, kary) * p_A(t, pop, kary)
    emit Coal((pop, cls, allele=A), rate = pairs / denom)
# a-tagged pairs
if n_a >= 2:
    pairs = n_a * (n_a - 1) / 2
    denom = 2 * N * p_kary(t, pop, kary) * (1 - p_A(t, pop, kary))
    emit Coal((pop, cls, allele=a), rate = pairs / denom)
# untagged pairs + untagged-with-A + untagged-with-a
n_normal_pairs = n_U*(n_U-1)/2 + n_U*n_A + n_U*n_a
if n_normal_pairs > 0:
    denom = 2 * N * p_kary(t, pop, kary)  # standard
    emit Coal((pop, cls, allele=Mixed), rate = n_normal_pairs / denom)
```

Outside the sweep window: emit one event per cell as before
(no progressive split). The check `sw.covers(t)` already gates this.

### Allele tag access pattern

The `a_tag: HashMap<LinUid, bool>` is currently in `apply_sweep` and
`apply_sweep_finalize`. To use it in the rate emitter, threadit
through to `emit_coal_events_from_cache`. Two paths:

1. **Pass `&a_tag` into `emit_coal_events_from_cache`.** Touches
   ~6 call sites in the run_loop. Mechanical.
2. **Move `a_tag` to a field on `HullSimulator`.** Cleaner long-term
   but a bigger change to the simulator state model.

Path 1 is simpler and reversible. Path 2 is right if `a_tag` becomes
load-bearing for many other rate paths. For now, Path 1.

### Coal event consumer

When a Coal event fires for `(pop, cls, allele=A)`, we need to pick
TWO A-tagged lineages from the (pop, cls) cell to coalesce. Current
event handler picks "any two lineages" via `coal_aggregate`. We need
to filter to A-tagged-only (or a-tagged-only) when consuming the
event. Implementation: pass the allele tag through the event payload
and filter in the consumer.

Same for a-tagged. For Mixed events, the existing "any two" logic
works (subject to the standard segments-overlap check).

### Endpoint at t_origin

`apply_sweep_finalize` (per the prior per-segment hitchhiking spec)
walks A-tagged lineages, partitions segments by `p_hh`, force-
coalesces the linked group. With progressive coalescence active,
most A-tagged lineages have already coalesced into the founder by
t_origin. The endpoint catches the residual:

- For hard sweeps: usually 0-2 stragglers; endpoint is idempotent.
- For partial sweeps: more stragglers (sample-time freq is partial,
  fewer of the original A-tagged lineages got progressive-coalesced).
  Endpoint forces them into the single founder.

The per-segment hitchhiking partition still happens — escapees from
distant segments still split off as untagged lineages.

### a-tagged at t_origin

a-tagged lineages do NOT get force-coalesced at t_origin. They just
continue. Their A/a tag becomes meaningless past the sweep window
(no sweep in effect; no rate enhancement). They re-enter the normal
coalescent.

Implementation detail: at t_origin, after the apply_sweep_finalize
endpoint, the simulator should **drop a_tag entries** for both A-
and a-tagged lineages. The A-tagged are gone (force-coalesced into
the founder); the a-tagged continue but no longer need their tag.

## Test strategy

**Existing tests must continue to pass:**
- T1-T5 (`tests/hull/test_phase6_sweep.py`): joint-WF Sweep API.
- J1-J9 (`tests/hull/test_phase6b_sweep_joint.py`): trajectory.
- A1-A4 (`rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`):
  closed-form anchors.
- PS1, PS2, PS3 (`tests/hull/test_phase6c_per_segment_hitchhiking.py`):
  spatial profile from per-segment hitchhiking — these may shift
  numerically (PS2's 67% reduction may become 50% or 80% under
  progressive — needs empirical measurement, possibly relax bound)
  but the qualitative shape (monotone, low-at-x_sel) must hold.

**New tests:**

- **PG1 (Rust smoke):** confirms simulator runs to completion with
  progressive coalescence, no panic.
- **PG2 (amplitude anchor):** for D2-equivalent params (s=0.05,
  Ne=10000, n=10, L=100kb, hard det sweep), mean pi_branch over 30
  reps should match the analytical Kim-Stephan reduction:
  E[pi] ≈ E[pi_neutral] · (1 - α^(2/(α+1))) where α = 2·s/r.
  Tolerance: ±20% (model approximation).
- **PG3 (cross-engine — depends on resumption of discoal D2):**
  the validation harness D2 test should pass after this lands.
  Specifically `pi_branch` and `n_trees` agree within 3·SE between
  msinv and discoal.

PG3 is the load-bearing external validation — same role as
discoal D2 was supposed to be.

## Risks and rollback

**Risk 1 — PS2/PS3 numerics shift unacceptably.** The per-segment
hitchhiking spec produced 67% pi reduction at sweep center. With
progressive coalescence, the reduction should be similar in shape
but potentially deeper (progressive enhances the rate during the
window, not just at the endpoint). If PS2's 30%-reduction threshold
is tighter under progressive, relax to 25%. PS3's at-x_sel anchor
should pass either way (the founder convergence still happens).

**Risk 2 — A × a pair zero-out breaks lineages that need to
coalesce off-x_sel.** Specifically, two lineages tagged A and a at
x_sel might have segments at distant positions that should coalesce
neutrally. Setting their pair rate to zero in the cell denies them
this. Mitigation: the per-segment hitchhiking already splits off
escaped segments into untagged lineages, so distant-position
coalescence happens via the untagged subgroup. The A-vs-a-tagged
pair is correct to be zero at the rate level.

**Risk 3 — runtime overhead.** Per-emit O(|active|) walk for
allele bucketization. For n=10 sims this is negligible; for n=100+
it could be noticeable. Future optimization: cache the bucket
counts in the rate cache and update incrementally on coal/recomb
events. Defer.

**Rollback:** if existing tests regress unfixably, the rollback is
to revert this change and accept that msinv's sweep model is
endpoint-only at the cost of distributional accuracy vs discoal.

## Files to change

**Primary:**
- `rust/msinv-core/src/simulator.rs:1697` —
  `emit_coal_events_from_cache` per-allele bucketization
- `rust/msinv-core/src/simulator.rs` (event consumer) — Coal handler
  needs to filter by allele tag when consuming
- `rust/msinv-core/src/events.rs` — extend Coal event payload with
  allele tag (`A`, `a`, `Mixed`)
- `rust/msinv-core/src/simulator.rs:apply_sweep_finalize` — drop
  a_tag entries at end of sweep window (cleanup)

**Tests:**
- `rust/msinv-core/tests/sweep_progressive_coalescence.rs` (new) —
  PG1 smoke
- `tests/hull/test_phase6d_progressive_coalescence.py` (new) —
  PG2 amplitude anchor

**Docs:**
- `CLAUDE.md` — note the progressive coalescence model

## References

- discoal source: `/home/adkern/discoal/src/core/discoalFunctions.c:1981-1984`
  (the canonical sweep coal-rate formulas).
- `docs/superpowers/specs/2026-04-30-sweep-per-segment-hitchhiking-design.md`
  — predecessor (spatial profile fix).
- `rust/msinv-core/src/simulator.rs:1697-1708` — current rate gate
  (post-piece-2; only kary-conditional, not allele-conditional).
- `rust/msinv-core/src/sweep_trajectory.rs:p_allele_given_kary` —
  the `p_A(t, pop, kary)` query already exists; just needs to be
  consumed by the rate emitter.
