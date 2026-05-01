# Tag-aware recombination during sweep window — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Branch:** `feat/sweep-tag-aware-recomb`
**Predecessors:** `2026-04-30-sweep-progressive-coalescence-design.md`,
`2026-04-30-sweep-standing-variation-phase-design.md`. The progressive
coal model + SV phase brought msinv's *temporal* sweep model in line
with discoal; this spec adds the missing *spatial* mechanism.

## Context

D3 (soft sweep, `f0=0.05`) is currently `@pytest.mark.skip`-marked
on `feat/discoal-validation`. After progressive coal + SV phase +
T_eff extension shipped:

| State | msinv π | discoal π |
|---|---|---|
| pre-progressive | 8176 | 16630 |
| post-progressive (no SV) | 8176 | 16630 |
| post-SV phase (unconditional WF) | 4469 | 16630 |
| post-conditional drift | 4447 | 16630 |
| post-T_eff extension | 5989 | 16630 |

The remaining gap is structural: discoal's
`recombineAtTimePopnSweep` (`discoalFunctions.c:2569-2583`)
*continuously* thins the A-tagged pool during the sweep window via
recombination-as-tag-shedding. msinv's one-shot per-segment partition
at `t_de_novo` is a coarser approximation that under-counts the
shedding rate.

## Goal

Replicate discoal's per-recombination tag rejection-sampling. When
a recombination splits a lineage during `[tau, t_de_novo]`, the child
not containing `x_sel` keeps its sweep-group tag with probability
proportional to the trajectory's current `p_A(t)`; otherwise it
switches to the opposite group.

This produces the continuous shedding behavior that preserves
diversity in soft sweeps. D3 should converge to discoal's π.

## discoal reference

`/home/adkern/discoal/src/core/discoalFunctions.c:2569-2583`:

```c
if(sweepSite < (float) xOver / nSites){
    // sweepSite on left; lParent inherits child's group, rParent is sampled
    lParent->sweepPopn = sp;
    r = ranf();
    if (r < popnFreq) rParent->sweepPopn = sp;
    else rParent->sweepPopn = (sp == 0) ? 1 : 0;
}
else{
    // sweepSite on right; rParent inherits, lParent is sampled
    rParent->sweepPopn = sp;
    r = ranf();
    if (r < popnFreq) lParent->sweepPopn = sp;
    else lParent->sweepPopn = (sp == 0) ? 1 : 0;
}
```

`popnFreq` arg is `x` for the B-group (sp=1) and `(1-x)` for the
b-group (sp=0). I.e., the non-`x_sel` parent rejection-samples
*against the same-group's bgkd freq*: it stays in its current group
with prob equal to the group's frequency at time `t`.

For sp=1 (A-tagged child) at low x: `popnFreq = x` is small,
`r < x` is rare → most of these recombinations push the rParent
to b-group.

## Locked decisions

**Q1: Tag swap fires when `sweep.covers(t)`.** Both selection and SV
phases trigger the swap. Outside any active sweep window, recombination
keeps the existing tag-inheritance behavior (both children copy the
parent's `a_tag` entry).

**Q2: Untagged lineages during the sweep window are treated as
"a-tagged" for rejection sampling.** discoal assigns every lineage in
the swept population a sweep-group tag at `apply_sweep`. msinv may
have lineages without an `a_tag` entry — typically distant-from-`x_sel`
lineages that aren't probabilistically tagged at sample time. For the
recombination-time rejection sampling, we treat their effective tag as
"a" (sp=0 in discoal terms): they stay untagged with prob `1 - p_A(t)`,
or become A-tagged with prob `p_A(t)`. This matches the limiting
behavior in discoal where an a-tagged lineage's recombination can
switch the non-`x_sel` parent to A.

**Q3: Tag swap places "switched" lineages as `a_tag = false`, not
removed from the map.** Keeping them in the map (as a-tagged) lets
the per-allele rate model count them in `n_a_lower` via the existing
filter; removing them entirely would put them in the `n_untagged`
bucket, which has different semantics. Symmetric for a→A: insert
`a_tag[uid] = true`.

**Q4: The post-recomb tag swap runs even when the parent didn't have
an entry in `a_tag`.** The non-`x_sel` child can become A-tagged (per
Q2) — the swap creates a fresh entry.

**Q5: When neither child contains `x_sel` (rare; happens when the
crossover happens on a segment chain that doesn't span `x_sel` at
all).** Both children's tags are sampled independently against
`p_A(t)`. This catches the "lineage doesn't even cover `x_sel`" case
which discoal treats by tagging via `popnFreq` for both halves.

## Out of scope

- **Gene flux during sweep.** Same conceptual question (flux can move
  segments between karyotypes during the sweep window) but separate
  mechanism. Defer to a future spec.
- **Multi-pop sweep recombination dynamics.** Tag swap operates only
  on the origin pop — non-swept pops have no per-allele structure.
- **Bidirectional rejection sampling for non-A→A transitions.** discoal's
  `pCoalB`/`pRecB` distinction lets b-tagged lineages also recombine
  via a separate code path (line 2042). Our msinv-side equivalent
  symmetrically samples both directions in one pass (Q2/Q4 above).

## Architecture

### Where the swap fires

`rust/msinv-core/src/simulator.rs` — both `run_loop_simple` and
`run_loop_with_caches` consume `Event::Recombination`. After
`apply_recombination` returns the new children, immediately:

1. Check whether any `finalized_sweeps[i].covers(t)`. If not, return.
2. For the swept sweep, query `p_A(t)` from its trajectory.
3. For each newly-created child lineage (typically 2):
   - Walk its segment chain, check whether `x_sel ∈ [seg.left, seg.right)`.
   - If yes: child keeps its tag (no change).
   - If no: rejection-sample against `p_A(t)` and update `a_tag`.

The new helper:

```rust
/// Per-discoal recombineAtTimePopnSweep semantics. After a
/// recombination during the sweep window splits a parent lineage
/// into children, the child(ren) not containing x_sel rejection-
/// sample their sweep-group tag against the trajectory's current
/// p_A(t).
fn apply_sweep_recomb_tag_swap(
    active: &[Lineage],
    new_indices: &[usize],
    sweep: &Sweep,
    t: f64,
    arena: &SegmentArena,
    rng: &mut Xoshiro256PlusPlus,
    a_tag: &mut HashMap<LinUid, bool>,
) {
    let traj = match sweep.trajectory.as_ref() {
        Some(t) => t,
        None => return,
    };
    // Use origin_kary as the kary key; for panmictic sweeps the
    // trajectory's p_A query is degenerate w.r.t. kary.
    let p_a = traj.p_allele_given_kary(t, sweep.origin_pop, sweep.origin_kary);
    for &idx in new_indices {
        if active[idx].population != sweep.origin_pop { continue; }
        let contains_x_sel = lineage_overlaps_position(
            active[idx].head, sweep.x_sel, arena);
        if contains_x_sel { continue; }
        let uid = active[idx].uid;
        let was_a = a_tag.get(&uid).copied().unwrap_or(false);
        // discoal: stays in current group with prob popnFreq; here
        // popnFreq = p_a if was_a, else (1 - p_a).
        let stay_prob = if was_a { p_a } else { 1.0 - p_a };
        if rng.random::<f64>() < stay_prob {
            // Tag unchanged. Ensure entry exists so future swaps
            // can flip it correctly (Q4 / Q2 invariant).
            a_tag.entry(uid).or_insert(was_a);
        } else {
            // Switch.
            a_tag.insert(uid, !was_a);
        }
    }
}
```

### Determining `new_indices`

`apply_recombination` returns 2 new lineages appended to `active`
(replacing 1 parent via swap_remove). The simulator's existing
recombination consumer already knows the parent index and post-call
length. The two new indices are `[pre_len - 1, pre_len]` after the
swap_remove + push pattern (consistent with the coalescence consumer
pattern at simulator.rs:931).

### Interaction with existing per-segment partition

`apply_sweep_finalize`'s per-segment partition still runs at
`t_de_novo`. Most A-tagged segments will have already shed via
recombination during the SV phase, so the partition has fewer linked
segments to process. The de novo merge (PG-D1 / SV-B3) still fires
for any residual A-tagged lineages.

PS2/PS3 use `f0=1/(2N)` → `t_de_novo == t_origin` → no SV-phase
recombinations happen inside the sweep window beyond the existing
selection-phase ones. Numerical impact: small.

## Test strategy

**Existing tests must continue to pass.** Hard sweeps (T1-T5, J1-J9,
A1-A4, PS1-PS3, PG1-PG2, SV1-SV2) all use `f0 = 1/(2N)` → no
SV-phase recomb → behavior unchanged.

**New tests:**

- **TR1 (Rust unit, recomb math):** with `p_A=0.9`, recombine an
  A-tagged lineage 1000 times; expect ~90% of non-`x_sel` children to
  keep A tag. With `p_A=0.1`, expect ~10%.
- **TR2 (Rust simulator, soft sweep smoke):** a soft-sweep run with
  `f0=0.05` completes without panic and produces non-trivial output.
  (Refines SV2 with the new logic.)
- **TR3 (Python, D3 against discoal):** flip `@pytest.mark.skip` on
  `test_discoal_validation_d3_soft_sweep`; expect both stats OK at
  3·SE.

## Risks and rollback

**Risk 1 — performance.** Per-recombination segment walk to check
`x_sel` overlap. For typical `n=10` sims with rho=40, total
recombination count is small (~tens per rep). Walk is O(n_segs/lin),
~ms-scale. Negligible.

**Risk 2 — tag bookkeeping drift.** Lineages that are split-then-merged
multiple times during the SV phase accumulate tag flips. Each flip
preserves the rejection-sampling semantics independently — no drift.
But edge cases (e.g., a recombination producing a child containing no
segments) need defensive handling: if `lineage_overlaps_position`
returns false because the segment chain is empty, treat as "doesn't
contain `x_sel`" and rejection-sample the tag.

**Risk 3 — D3 still fails.** If after this change msinv π is still
materially below discoal π, the remaining gap is one of:
- Trajectory shape differences (variance, drift parameterization).
- discoal's `pRecurMut` (line 1987) — recurrent adaptive mutation
  during the sweep, which msinv's `recurrent_mutation_rate` may
  apply differently.

We can iterate on these only if D3 still fails after TR3.

**Rollback:** if TR1/TR2 pass but PS-PG tests regress unexpectedly,
the tag swap can be gated via a runtime flag or simply reverted
(it's additive; existing behavior is the no-op default).

## Files to change

**Primary:**
- `rust/msinv-core/src/simulator.rs` —
  - Add `apply_sweep_recomb_tag_swap` helper.
  - Call it in both `run_loop_simple` and `run_loop_with_caches`'s
    Recombination consumers, after `apply_recombination` returns.

**Tests:**
- `rust/msinv-core/tests/sweep_tag_aware_recomb.rs` (new) —
  TR1 (recomb math) + TR2 (simulator smoke).
- `tests/hull/test_validation_discoal.py` — flip skip on
  `test_discoal_validation_d3_soft_sweep` (TR3, on
  `feat/discoal-validation` after merge).

**Docs:**
- `CLAUDE.md` — append tag-aware recomb to the Sweep model entry.

## References

- discoal source: `/home/adkern/discoal/src/core/discoalFunctions.c:2569-2583`
  (`recombineAtTimePopnSweep`); `:2042-2051` (call sites in the sweep
  loop).
- Predecessor specs:
  - `docs/superpowers/specs/2026-04-30-sweep-progressive-coalescence-design.md`
  - `docs/superpowers/specs/2026-04-30-sweep-standing-variation-phase-design.md`
- Failing test: `tests/hull/test_validation_discoal.py::test_discoal_validation_d3_soft_sweep`.
