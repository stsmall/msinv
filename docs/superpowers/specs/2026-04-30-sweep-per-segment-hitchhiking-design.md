# Per-segment hitchhiking sweep extension — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Branch:** `feat/sweep-per-segment-hitchhiking`
**Surfaced by:** discoal validation track D2 first run — msinv produced
panmictic π ≈ neutral (3.83e+04) while discoal produced strongly reduced
π (4.9e+03) for the same s=0.05, tau=1000g hard sweep. Root cause traced
to msinv's endpoint-only Hudson-Kaplan sweep operator at
`rust/msinv-core/src/simulator.rs:1697-1708, 2418-2465`.

## Goal

Extend msinv's `Sweep` operator to produce a discoal-comparable
spatial hitchhiking footprint for sweeps in **colinear regions**
(panmictic at the sweep locus). Two main use cases:

1. Sweep in a no-inversion (purely panmictic) scenario. Currently
   produces no footprint — sweep falls through to plain Hudson during
   the window and only force-coalesces lineages that physically
   cover `x_sel` at `t_origin` (regardless of distance).
2. Sweep in a with-inversion scenario where `x_sel` falls **outside**
   any active inversion. Same behavior as case 1 at that locus.

Both cases need: at distance `d` from `x_sel`, ancestral segments
should be force-coalesced with probability `exp(-r·d·T_eff)`, matching
the discoal/Kim-Stephan continuous-time approximation.

## Non-goals

- Inversion-internal sweeps (`x_sel` inside an active inversion's
  range). The existing trajectory-based ne_cell gate already produces
  a karyotype-conditioned footprint for that case; the per-segment
  refinement is the same model and will lift its quality, but the
  primary goal of this spec is colinear-region sweeps.
- Trajectory-integrated hitchhiking probability. Current
  `hitchhiking_prob(x, r) = exp(-r·d·T_eff)` uses the full
  `T_eff = t_origin - tau`. The "proper integral over trajectory
  shape" is a TODO refinement (already noted in `sweep.rs`).
- Multi-position sweeps. Lifting the
  `simulator.rs:236` single-sweep assertion is its own spec.

## Scope (three pieces)

| # | Piece | Where | What it does |
|---|---|---|---|
| **1** | Per-segment `p_hh` at `apply_sweep_finalize` | `simulator.rs:2430-2465` | For each A-tagged lineage, decide per-segment whether it hitchhikes. Linked segments force-coalesce; escaped segments split off. |
| **2** | Panmictic-aware coal-rate gate | `simulator.rs:1697-1708` | When `cls.get_inv(target_inv).is_none()` (panmictic at sweep locus), fall back to `origin_kary` so the trajectory's `ne_cell` engages. |
| **3** | Probabilistic A-tag for non-x_sel lineages | `simulator.rs:2410-2423` | For lineages NOT overlapping `x_sel`, sample whether they're "implicitly A-linked" with probability `p_A_at_τ * exp(-r·d_nearest·T_eff)`. |

All three needed for full discoal-faithful behavior. Piece 1 is the
dominant effect; pieces 2 and 3 are smaller secondary contributions.

## Piece 1 — Per-segment hitchhiking (load-bearing)

### Current behavior (`simulator.rs:2451-2461`)

```rust
let p_hh = sweep.hitchhiking_prob(sweep.x_sel, recomb_rate);
if rng.random::<f64>() < p_hh {
    a_uids.push(uid);
}
```

`hitchhiking_prob(sweep.x_sel, ...)` evaluates with `d = 0` because
the call passes `x = sweep.x_sel`, so `p_hh = exp(0) = 1.0`. Every
A-tagged lineage gets force-coalesced. The lineage's segments at
positions far from `x_sel` are dragged in regardless of distance.

### New behavior

Walk each A-tagged lineage's segment list. For each segment
`[seg_lo, seg_hi]`:

1. Compute `d = distance from segment to x_sel`:
   - If `seg_lo ≤ x_sel ≤ seg_hi`: d = 0 (segment spans x_sel)
   - Else if `seg_hi < x_sel`: d = x_sel - seg_hi (segment to the left)
   - Else (`seg_lo > x_sel`): d = seg_lo - x_sel (segment to the right)

2. Compute `p_hh_seg = exp(-r·d·T_eff)`.

3. Sample `u ~ Uniform(0,1)`. If `u < p_hh_seg`: segment is **linked**
   (will be force-coalesced). Else: segment **escapes** (continues
   independently after the sweep).

4. Partition the lineage's segments into a `linked` group and an
   `escaped` group. The linked group stays with the lineage's UID
   (force-coalesced with all other linked groups across all A-tagged
   lineages). The escaped group is detached into a new lineage with
   a fresh UID, untagged, free to recombine and coalesce normally.

### Coalescence after partition

After all lineages have been split, collect the union of all linked
segments across all lineages and force-coalesce them into a single
ancestral lineage at time `t_origin`. This is the same coalescent
operation the current code does on whole lineages, but applied to a
filtered set of segments.

The escaped segments form one or more "escapee" lineages that
continue past the sweep window. They re-enter the normal coal+recomb
event loop at the next iteration.

### Edge cases

- **Lineage with all segments escape:** the lineage drops its A flag
  entirely; nothing force-coalesces for that lineage.
- **Lineage with all segments link:** equivalent to current behavior
  (whole lineage force-coalesced).
- **Lineage with mixed:** non-trivial split; both linked and escaped
  groups exist after the operation.

### Independence assumption

Each segment's `p_hh_seg` is sampled independently. This is the
discoal "linkage approximation": correlation between adjacent
segments (via shared recombination history during the sweep window)
is dropped. Matches the Kim-Stephan single-locus reduction's
underlying assumption.

## Piece 2 — Panmictic-aware coal-rate gate

### Current behavior (`simulator.rs:1697-1708`)

```rust
let denom = match active_sweep {
    Some(sw) if sw.covers(t) && sw.origin_pop == pop => {
        if let Some(kary) = cls.get_inv(sw.target_inv) {
            2.0 * sw.ne_cell_or_fallback(t, pop, kary, ne, p_class).max(1e-9)
        } else {
            2.0 * ne * p_class
        }
    }
    _ => 2.0 * ne * p_class,
};
```

For panmictic at the sweep locus, `cls.get_inv(target_inv)` returns
`None`, and the inner `else` falls back to plain Hudson (no
trajectory effect during the sweep window).

### New behavior

```rust
if let Some(kary) = cls.get_inv(sw.target_inv).or(Some(sw.origin_kary)) {
    2.0 * sw.ne_cell_or_fallback(t, pop, kary, ne, p_class).max(1e-9)
} else { /* unreachable */ }
```

Or equivalently, drop the `if let Some(kary)` and unconditionally use
`cls.get_inv(...).unwrap_or(sw.origin_kary)`. This treats the
panmictic class as if it were `origin_kary` for the duration of the
sweep window.

### Interpretation

For pure panmictic (no inversions active anywhere on the genome at
the sweep locus), `origin_kary` is a placeholder picked by the user.
The trajectory's `p_kary(t, pop, S)` and `p_kary(t, pop, I)` are
both 1.0 throughout (since there's no active inversion to split the
population). The ne_cell reduction is then `N * 1.0 = N`, identical
to plain Hudson — **no effective change for pure panmictic**.

For with-inversion scenarios where x_sel is outside the inversion:
the lineages at x_sel are panmictic at this locus (no kary tag), but
are S or I at the inversion locus elsewhere. `origin_kary` is the
karyotype the user chose to associate with the sweep (typically the
ancestral or majority kary). Using it as a proxy reduces the coal
rate by `1/p_origin_kary(t)` during the sweep window — a proper
hitchhiking effect.

So Piece 2 is meaningful for the with-inversion-but-outside case; for
pure panmictic it's a no-op (trajectory is degenerate at p_kary=1).
We include it anyway because the gate is wrong on principle and the
fix is one line.

### Caveat (out of scope to fix here)

The fully correct panmictic ne_cell would need A/a-aware class
encoding so the gate can use `p_A` instead of `p_kary`. Currently
`BranchClass` only encodes karyotype. Extending it is a class_tag
rewrite, deferred. Piece 2 here makes the existing machinery engage
in colinear regions; A/a-aware encoding is a separate spec when it
becomes load-bearing.

## Piece 3 — Probabilistic A-tag for non-x_sel lineages

### Current behavior (`simulator.rs:2410-2423`)

```rust
for lin in active.iter() {
    if !lineage_overlaps_position(lin.head, sweep.x_sel, arena) {
        continue;
    }
    // ... tag with A or a using trajectory frequency
}
```

Only lineages that physically cover `x_sel` get tagged. Distant
lineages — those whose segments are entirely off `x_sel` — never
become eligible for hitchhiking.

### New behavior

For each lineage, regardless of x_sel coverage:

1. Compute `d_nearest = min over segments of distance to x_sel`.
2. Compute `p_link = exp(-r·d_nearest·T_eff)`.
3. With probability `p_link`, treat the lineage as if it covers
   x_sel for tagging purposes: sample A/a from the trajectory.
4. With probability `1 - p_link`, leave untagged.

### Why this is needed

A lineage whose nearest segment is 5kb from x_sel might still be on
the A background (no recombination has separated its 5kb-distant
segment from the swept allele during the window). Without piece 3,
that lineage never enters the A pool, so its segments — even those
that would have hitchhiked — pass through the sweep neutrally.

This is the same exp(-r·d·T_eff) used in piece 1, applied at the
**tagging** step (whether the lineage even gets considered) rather
than at the **finalize** step (which segments hitchhike given the
lineage is tagged). Both stages use the same distance-decay; they
multiply for segments far from x_sel:

- P(segment at distance `d` is force-coalesced) =
  `exp(-r·d_lineage_nearest·T_eff) · exp(-r·d_segment·T_eff)`

For segments where `d_segment ≈ d_lineage_nearest` (i.e., the
nearest segment IS the segment we're checking), the two exponentials
collapse to one (you don't double-count the same recombination
event). The clean formulation: tag the lineage with the
"nearest-segment" probability; then at finalize, each segment uses
its OWN distance for the second draw, but conditioned on the
lineage being tagged. The net effect for the nearest segment is
`exp(-r·d_nearest·T_eff)` (single exponential, single Bernoulli).

To avoid double-counting, **piece 3 doesn't apply piece 1's
per-segment p_hh redundantly**. Implementation note: in
`apply_sweep_finalize`, for each segment, use the segment's own
distance d for the per-segment Bernoulli; piece 3 only gates whether
the lineage entered the pool at all.

### Computational cost

Piece 3 makes apply_sweep iterate over **all** active lineages, not
just those overlapping x_sel. For typical n=10 and active≈n during
the sweep window, this is O(n). The added cost is one Bernoulli draw
per untagged lineage. Negligible.

## T_eff convention

All three pieces use the existing `Sweep::hitchhiking_prob` formula:

```rust
T_eff = sweep.joint.t_origin - sweep.tau
```

The full sweep window duration. Stochastic mode trajectories may
have actual durations slightly different (the sweep can fail or
take longer/shorter than expected); the `T_eff = t_origin - tau`
approximation matches discoal's deterministic-trajectory formula.
True trajectory-integrated hitchhiking is the deferred TODO.

## Test strategy

Existing tests must continue to pass unchanged:

- **T1-T5** (`tests/hull/test_phase6_sweep.py`): joint-WF Sweep API
  tests. Most measure trajectory or single-locus stats; these are
  unchanged. Any spatial assertions (currently none) would change.
- **J1-J9** (`tests/hull/test_phase6b_sweep_joint.py`): trajectory
  integration tests. Single-locus / count-based; unchanged.
- **A1-A4** (`rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`):
  closed-form analytical anchors at single locus. Unchanged.

New tests (added with this spec):

- **PS1**: panmictic sweep produces non-flat π profile across
  positions in a single rep (smoke test that piece 1 produces a
  spatial gradient).
- **PS2**: spatial profile decays monotonically from x_sel toward L/2
  in mean over reps (≥30 reps, single-pop, hard sweep).
- **PS3**: at x_sel itself, mean π ≈ pre-existing Kim-Stephan anchor
  expectation (no regression on the anchor case).
- **PS4 (resume the discoal track's D2)**: cross-engine 200-rep MC
  comparison vs discoal `-ws`. Pass = pi_branch + n_trees agree
  within 3·SE; windowed-π bins agree within Bonferroni z·SE.

PS4 lives in the discoal validation harness (the deferred D2 test).
PS1-PS3 live as new internal tests in the sweep test suite. The
discoal validation track (paused at D1) resumes after this spec lands.

## Files to change

**Primary:**
- `rust/msinv-core/src/simulator.rs` — update `apply_sweep_finalize`
  (piece 1), `apply_sweep` (piece 3), `emit_coal_events_from_cache`
  gate (piece 2)
- `rust/msinv-core/src/sweep.rs` — possibly add helper methods on
  `Sweep` for per-segment p_hh and "lineage nearest-segment distance"

**Tests:**
- `rust/msinv-core/tests/sweep_per_segment_hitchhiking.rs` (new) —
  PS1, PS2, PS3 anchors

**Docs:**
- `CLAUDE.md` — bump test count and note the per-segment sweep model

**Out of scope (touched but not modified semantically):**
- `rust/msinv-core/src/sweep_trajectory.rs` — read-only
- `rust/msinv-core/src/class_tag.rs` — A/a-aware encoding deferred

## Risk and rollback

Risk: this changes sweep semantics for **all** sweeps, not just
panmictic. The existing inversion-aware sweep tests pass under the
endpoint-only model; under per-segment hitchhiking, lineages that
were force-coalesced as a whole are now partitioned. Distributional
results at single-locus stats should be unchanged (segments AT x_sel
have d=0 → p_hh=1 → still force-coalesced); spatial stats off x_sel
are new and weren't measured before.

If a regression hits T1-T5 / J1-J9 / Kim-Stephan anchors that we
can't trace to a real bug, the rollback is the panmictic-aware gate
(piece 2 only) plus a TODO marker: gives correct Ne path during
sweep window for the new colinear use case, leaves endpoint behavior
unchanged. This wouldn't validate against discoal D2 but would not
break existing tests.

## References

- `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md` — the
  joint-WF Sweep API design (the model being extended).
- `docs/superpowers/specs/2026-04-30-discoal-validation-design.md` —
  the discoal validation track that surfaced this gap.
- `rust/msinv-core/src/sweep.rs:hitchhiking_prob` — existing per-x
  formula, currently called only with x = x_sel.
- `rust/msinv-core/src/simulator.rs:2418` —
  `lineage_class_for_inv_id_arena(...).unwrap_or(sweep.origin_kary)`
  — same fallback pattern this spec applies to the coal-rate gate.
