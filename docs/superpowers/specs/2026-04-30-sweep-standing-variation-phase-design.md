# Sweep standing-variation phase — design

**Date:** 2026-04-30
**Status:** spec, awaiting user approval
**Branch:** `feat/sweep-standing-variation`
**Predecessor:** `2026-04-30-sweep-progressive-coalescence-design.md` —
that extension brought the in-window per-allele rate model. This spec
extends the *post-window* phase to match discoal's actual model.

## Context

D3 (`f0=0.05` standing-variation soft sweep, `tests/hull/test_validation_discoal.py`)
is currently `@pytest.mark.skip`-marked. msinv π=8176 vs discoal π=16630 (~half).

Direct cause: at `t_origin` (going backward in time), msinv's `apply_sweep_finalize`
force-coalesces every A-tagged lineage into a single founder. Past `t_origin` the
A allele has no model — A-tagged lineages have already been merged.

discoal does not collapse to a single founder. It runs a *backward neutral
Wright-Fisher drift trajectory* of the A allele frequency past `t_origin`,
starting at `p_A = f0` and continuing until the trajectory extinction-times
out at `1/(2N)`. During that drift phase, A-tagged lineages keep coalescing
at the elevated rate `1/(2N·p_A(t))` from the per-allele rate model. At the
extinction time (the de novo origin), any remaining A-tagged lineages merge
back into the broader pool.

The fix: extend msinv's sweep window past `t_origin` with the same per-allele
rate machinery, drawing `p_A(t)` from a stochastic neutral-drift trajectory.

## Goal

Replace msinv's at-`t_origin` "all A-tagged collapse to one founder" behavior
with discoal's two-phase model:

1. **Selection phase** (`tau ≥ t > t_origin`): per-allele rates with
   `p_A(t)` from the existing logistic / WF trajectory. *(unchanged)*
2. **Standing-variation phase** (`t_origin ≥ t > t_de_novo`): per-allele rates
   with `p_A(t)` from a *stochastic backward neutral WF drift* starting at `f0`
   and ending when `x ≤ 1/(2N)`. *(new)*
3. **De novo origin** (`t = t_de_novo`): force-merge any remaining A-tagged
   lineages into the a/untagged pool. *(replaces current "all-collapse" endpoint)*
4. **Pre-origin phase** (`t > t_de_novo`): standard neutral coalescent on the
   merged pool. *(unchanged)*

Reference: `/home/adkern/discoal/src/core/discoalFunctions.c:1764-1875` —
discoal's `proposeTrajectory` shows the dual-phase trajectory build (sweep
phase + neutral pre-sweep phase, both written into the same trajectory file).

## Locked decisions

**Q1: Standing-variation drift is always stochastic.** Both `mode='Deterministic'`
and `mode='Stochastic'` use stochastic backward neutral WF drift for the
standing-variation phase. The Det/Stoch choice only affects the *sweep
phase*. This matches discoal: `-wd` and `-ws` differ in the sweep phase
but both use `neutralStochasticOptimized` past `t_origin`.

**Q2: For `f0 = 1/(2N)` there is no standing-variation phase.** The drift
extinction threshold is `1/(2N)`, so a trajectory starting at `f0 = 1/(2N)`
is at the threshold immediately. `t_de_novo == t_origin` and the existing
endpoint behavior is preserved (single force-coalesce). D2 and D4 unaffected.

**Q3: `t_de_novo` is determined by the trajectory, not user-specified.** It's
whatever backward time the stochastic drift first hits `x ≤ 1/(2N)`. Per-rep
random; not a parameter on `Sweep`.

**Q4: At `t_de_novo`, force-merge each remaining A-tagged lineage with one
randomly-chosen non-A lineage.** This represents the moment the A allele
first arose — the A founder is a single mutation on a single chromosome that
otherwise looks like the rest of the population. We model this by pairing
each surviving A lineage with a random a/untagged target and applying a
standard coalescence event.

If there are 0 non-A lineages at `t_de_novo` (e.g., all lineages were
A-tagged after a fixed sweep), force-coalesce the A-tagged among themselves
into a single founder. (Edge case; should be rare with `f0 > 1/(2N)`.)

## Out of scope

- **Conditional drift trajectory.** discoal's stochastic neutral drift starts
  at `f0` and runs until extinction without conditioning. We follow the same
  approach. Some implementations condition on the extinction time being
  finite or matching some prior — we don't.
- **Multi-population standing variation.** The drift is computed for the
  origin population only. Multi-pop variants of the standing-variation phase
  (e.g., the variant exists at `f0` in pop A but `f0'` in pop B) are out
  of scope; defer to a multi-pop sweep extension.
- **Recurrent mutation in the standing-variation phase** (`-uA` style).
  D5 already needs a separate units-audit fix; recurrent mutation interacts
  with the standing-variation phase in non-trivial ways and is deferred.
- **Cap on standing-variation phase length.** The expected backward time to
  extinction starting from `f0` is roughly `2·f0·(1-f0)/(1/(2N)) ≈ 4N·f0`
  for small `f0` — bounded but for very small `f0` this can be long. We
  don't add a hard timeout in v1; if runtime becomes an issue we can add
  one in a follow-up.

## Architecture

### Trajectory module changes

`rust/msinv-core/src/sweep_trajectory.rs` currently builds a forward-time
trajectory from `t_origin` (selection onset) up to `tau` (sample time).

New: extend the trajectory backward past `t_origin` with neutral WF drift.
The trajectory data structure stays the same (sample points keyed by time),
but the time range expands to `[t_de_novo, tau]`. Queries for `p_kary` and
`p_allele_given_kary` past the existing `t_origin` boundary read the new
drift segment.

API additions on `JointSweepTrajectory`:
- `t_de_novo() -> f64` — the backward time at which the trajectory hit the
  extinction threshold. Equal to `t_origin` for `f0 = 1/(2N)`.
- Existing `p_kary(t, pop, kary)` and `p_allele_given_kary(t, pop, kary)`
  return the drift values for `t` in `(t_origin, t_de_novo]`.

The drift is a discrete WF step:
```
x_{i+1} = Binomial(2N_eff, x_i) / (2N_eff)
```
with `N_eff = floor(N·sizeRatio)` matching discoal's
`neutralStochasticOptimized` and time step `tInc = 1/(deltaTMod·N_eff)`. Match
discoal's `deltaTMod = 400` so the trajectory increments are commensurate.

For multi-pop sweeps the drift is computed for the origin pop only; other
pops have their `p_kary` and `p_allele_given_kary` continue at the boundary
values from `t_origin`. (Documented as a limitation; future extension.)

### Sweep window extension

`Sweep::covers(t)` currently returns `t >= self.tau && t <= self.joint.t_origin`.

Update to:
```rust
pub fn covers(&self, t: f64) -> bool {
    let upper = self.trajectory.as_ref()
        .map(|tr| tr.t_de_novo())
        .unwrap_or(self.joint.t_origin);
    t >= self.tau && t <= upper
}
```

Inside the new extended window, the existing per-allele rate emitter
(`emit_coal_events_from_cache`) and consumer (CoalAggregate handler) operate
unchanged. They just see a longer covering window. The `p_A(t)` query
returns the drift value naturally because the trajectory now spans further.

### Endpoint at `t_de_novo`

Replaces the existing `apply_sweep_finalize` semantics. New behavior:

1. **Per-segment partition** (carried over): walk each A-tagged lineage,
   roll Bernoulli `p_hh` per segment, separate linked vs. escaped. Escaped
   segments split off as fresh untagged lineages. *(unchanged from
   per-segment-hitchhiking spec)*
2. **De novo merge**: for each remaining A-tagged lineage, pick a random
   non-A-tagged lineage from the same population, apply a standard
   coalescence event at `t_de_novo`. The merged ancestor is no longer
   A-tagged.
3. **Edge case** (all lineages A-tagged): force-coalesce A-tagged among
   themselves into a single founder; the lone founder loses its tag.
4. **Cleanup**: clear `a_tag` entries for the surviving lineages
   (no progressive logic past `t_de_novo`).

Existing `apply_sweep_finalize` is renamed to `apply_sweep_de_novo` to
reflect the new semantics; the old name was misleading because the work
no longer happens at the sweep window's nominal end.

### Boundary scheduling

`apply_boundary` currently fires `apply_sweep_finalize` when
`(finalized_sweeps[0].joint.t_origin - t).abs() < 1e-9`. Update to fire when
`(finalized_sweeps[0].t_de_novo() - t).abs() < 1e-9`, where `t_de_novo()` is
the trajectory-derived bound.

The next-boundary scheduler in `run_loop_with_caches` pulls
`finalized_sweeps[0].joint.t_origin` for `next_boundary`. Update to use
`t_de_novo()` instead.

## Test strategy

**Existing tests must still pass:**
- T1-T5, J1-J9, A1-A4 (all unchanged — they use `f0 = 1/(2N)` so
  `t_de_novo == t_origin`).
- PS1, PS2, PS3 (per-segment hitchhiking; same `f0` — unchanged).
- PG1, PG2 (progressive coalescence; same `f0` — unchanged).
- D2, D4 (discoal validation; deterministic Det+`f0=1/(2N)` — unchanged).

**New tests:**

- **SV1 (Rust trajectory smoke):** build a Sweep with `f0 = 0.05`, confirm
  `t_de_novo() > t_origin`, confirm `p_allele_given_kary` queries past
  `t_origin` return values in `[0, f0]`, monotonically non-increasing on
  expectation (drift is stochastic so individual reps can fluctuate).

- **SV2 (Rust simulator smoke):** run a full simulation with `f0 = 0.05`,
  confirm it runs to completion without panic and produces non-trivial
  output. Equivalent to PG1 but with `f0 > 1/(2N)`.

- **SV3 (D3 unblocked):** flip the `@pytest.mark.skip` on the
  `test_discoal_validation_d3_soft_sweep` test. Both stats must pass at
  3·SE against discoal v2.0.0-beta.

## Risks and rollback

**Risk 1 — runtime cost.** Standing-variation phase adds ~`4N·f0`
generations of coalescent simulation per rep. For `f0=0.05, N=10000`
that's ~2000 extra generations. Should still be sub-second per rep.
For very small `f0` (e.g. `f0=1e-4`, `4N·f0 = 4`) negligible. For
`f0` close to 1 (immediate origin), no drift phase. Mid-range `f0`
is the worst case.

**Risk 2 — drift trajectory variance.** Per-rep stochastic drift means
per-rep variance in coalescence outcomes is higher than a deterministic
model. This is a feature (matches discoal's behavior) but means more reps
are needed for tight 3·SE bounds. Existing harness uses 20 reps; D3 may
benefit from 50 reps. Tunable on the test side.

**Risk 3 — De novo merge target selection.** Picking a random non-A
lineage to merge with might pick a lineage with no genomic overlap (no
shared segments), in which case the merge is a no-op. The existing
`apply_coalescence` with `skip_if_no_overlap=True` handles this; if there's
no overlap the A-tagged lineage just continues (and presumably coalesces
later via the normal coalescent). Acceptable.

**Risk 4 — discoal source-code drift.** discoal's
`neutralStochasticOptimized` may have implementation details we're missing
(e.g., variance correction). v1 implementation matches the surface of the
algorithm; if D3 is close-but-not-passing, audit details against the C
source and tighten.

**Rollback:** if the new endpoint behavior breaks D2/D4 (which use
`f0=1/(2N)`), the trajectory extension is gated on `f0 > 1/(2N)` so they
shouldn't regress. If they do, disabling the standing-variation phase
restores the prior endpoint behavior.

## Files to change

**Primary:**
- `rust/msinv-core/src/sweep_trajectory.rs` — extend `build_joint_trajectory`
  to append a backward neutral drift phase past `t_origin`. Add
  `t_de_novo()` query.
- `rust/msinv-core/src/sweep.rs` — update `Sweep::covers(t)` to use
  `t_de_novo()` as the upper bound.
- `rust/msinv-core/src/simulator.rs` — `apply_sweep_finalize` →
  `apply_sweep_de_novo`. Replace single-founder collapse with per-lineage
  merge into random non-A target. Update boundary scheduling to use
  `t_de_novo()`.

**Tests:**
- `rust/msinv-core/tests/sweep_standing_variation.rs` (new) — SV1, SV2.
- `tests/hull/test_validation_discoal.py` — flip skip on D3 (SV3).

**Docs:**
- `CLAUDE.md` — note the dual-phase trajectory model.

## References

- discoal source: `/home/adkern/discoal/src/core/discoalFunctions.c:1764-1875`
  (`proposeTrajectory`); `:1817-1837` (the dual-phase loop with
  `insweepphase` flip); `:1814` (`minF` clamp at `1/(2N)`).
- Predecessor: `docs/superpowers/specs/2026-04-30-sweep-progressive-coalescence-design.md`.
- Failing test today: `tests/hull/test_validation_discoal.py::test_discoal_validation_d3_soft_sweep`.
