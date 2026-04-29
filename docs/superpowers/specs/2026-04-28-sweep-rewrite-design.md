# Sweep rewrite — joint forward WF over (kary × allele × pop)

**Branch:** `feat/sweep-rewrite`
**Status:** design approved 2026-04-28
**Replaces:** Hudson-Kaplan endpoint-only `Sweep` operator (`rust/msinv-core/src/sweep.rs`).

## Problem

The current `Sweep` operator is an instantaneous outcome-conditioning
coalescent operator — three modes (window, hard, soft), all forced
coalescence at `t_event` with a hitchhiking probability decay. It does
not simulate the beneficial allele's frequency trajectory, cannot model
partial sweeps with a meaningful terminal frequency, and cannot capture
the interplay between an inversion's frequency and the selected allele
that gives the RDL biology its character (allele locked to inversion
background while recomb-suppressed; transferred via flux; both
backgrounds eventually carrying allele; inversion frequency rising and
then plateauing as flux equilibrates).

The motivating cases are:

1. **RDL gambiae**: 296G allele in 2L+a/2La inversion system, multi-pop
   *An. gambiae* with differential 2La frequencies and migration.
   Selection acts on 296G regardless of background; gene flux exchanges
   it between S and I; observe 2La frequency rising as it carries the
   allele, plateauing post-flux.
2. **Kir-Fol-3Ra**: Kir-only soft sweep within 3Ra inversion from
   standing variation. Folonzo experiences no selection on those
   alleles; migration between Kir and Fol may spread allele to Fol.

Both cases require joint dynamics over (population × karyotype ×
allele) — the simulator must compute the coupling, not be told it.

## Target model: discoal-style stochastic-then-deterministic, joint

Forward-time integer-WF on **4 haplotype classes per population**:

| Class | Karyotype at inversion | Allele at x_sel |
|---|---|---|
| `(S, a)` | colinear | ancestral |
| `(S, A)` | colinear | beneficial |
| `(I, a)` | inverted | ancestral |
| `(I, A)` | inverted | beneficial |

Per generation forward, in order:

1. **Selection within each pop**: relative fitness `1 + s` for any
   class with `A`, `1` otherwise. Fitness-weighted next-gen frequencies
   in each pop.
2. **Recurrent de novo origins** (if `recurrent_mutation_rate > 0`): per
   pop, mutate `a → A` at rate `uA · 2·N_pop(t)`. New `A` mutation
   placed on a random `(pop, kary)` class with probability proportional
   to current `a`-class counts in that pop. Same site, same allele,
   independent of any earlier origin.
3. **Wright-Fisher resampling within each pop**: multinomial(`2·N_pop(t)`,
   weighted frequencies). Drift on all four classes jointly.
   Stoch+det hybrid: integer-WF until any class crosses `5/(2N)` and
   det-logistic stability conditions, then deterministic logistic for
   that class. (Same boundary as `StochasticDeterministicTrajectory`,
   commit `1c7c8b2`.)
4. **Flux within each pop**: rate γ per coalescent unit. Per generation,
   transfer `A` between `(I,A) ↔ (S,A)` and `a` between `(I,a) ↔ (S,a)`
   at rate scaled by mean tract length and overlap at `x_sel`. Same
   physics as the existing flux operator, applied to bookkeeping rather
   than lineage events.
5. **Migration across pops**: redistribute haplotype-class counts using
   the demography's migration matrix `m_ij(t)`. Standard mass action.

Result: a `JointSweepTrajectory` of `freq[pop][class]` per generation
across `[tau, t_origin]`. Both `p_inv(t, pop)`, `p_allele(t, pop)`, and
all per-class frequencies emerge naturally.

The backward-time coalescent simulator queries this trajectory to drive
its rates; nothing about kary frequency is hand-specified by the user.

## Scope

### In scope (v1)

- One sweep event.
- One inversion (the karyotype axis).
- Any number of populations (≥1; degenerate at 1).
- Migration during sweep window (consumed from existing demography
  migration matrix, no separate sweep-time migration spec).
- Demographic events during sweep window (`-en`-equivalent): `N_pop(t)`
  is queried per generation from the demography. Sweep dynamics scale
  with size ratio (discoal-style; matches discoal `discoalFunctions.c`
  lines 1794, 1901, 1958).
- Origin: allele appears at one (`origin_pop`, `origin_kary`) pair at
  generation `t_origin` with frequency `f0` on that background.
- Modes: `Stochastic` (stoch+det hybrid; matches discoal `-ws`),
  `Deterministic` (logistic only; `-wd`), `Neutral` (no selection;
  `-wn`; for tests).
- Partial sweep: `partial_sweep_final_freq` (matches discoal `-c`).
  Default 1.0 (complete). At sample time, lineages assigned to swept
  fraction with probability equal to per-pop A frequency.
- **Soft sweep from standing variation** (matches discoal `-f`):
  `f0 > 1/(2·N·p_kary_origin)`. The allele exists pre-sweep at f0
  frequency, joint WF handles K-founder partitioning naturally — at
  t_origin, K ≈ ceil(2·N·p_kary_origin·f0) copies are seeded on
  distinct haplotype backgrounds within `(origin_pop, origin_kary)`.
- **Recurrent de novo origins during sweep phase** (matches discoal
  `-uA`): per generation in the sweep window, mutate `a → A` at rate
  `uA · 2·N_pop_total`. New `A` mutation placed on a random
  `(pop, kary)` class with probability proportional to current `a`-
  class counts. Same site, same allele, just multiple independent
  origins — the joint WF accommodates this as an extra step between
  selection and drift.

### Out of scope (v1)

- **Multiple concurrent sweeps at different sites** (i.e., a second
  selective site with a different allele, simultaneously). The joint
  state space would scale combinatorially. Soft sweeps and recurrent
  de novo origins are *not* this — same site, same allele.
- **Multiple inversions interacting**: the karyotype axis is exactly
  one inversion. A second inversion can be present in the simulation
  but doesn't enter the joint state of the sweep.
- "Sweep to the side of the window" mode (discoal `-ls/-ld/-ln`).

## Public API

```rust
pub struct Sweep {
    pub x_sel: f64,                    // bp; discoal -x analog
    pub tau: f64,                      // gen ago when sample is taken
    pub origin_pop: u32,
    pub origin_kary: Karyotype,        // S or I
    pub target_inv: u16,               // which inversion provides kary axis
    pub joint: JointSweepSpec,
}

pub struct JointSweepSpec {
    pub mode: SweepMode,                       // Stochastic | Deterministic | Neutral
    pub s: f64,                                // selection coefficient on A
    pub t_origin: f64,                         // gen ago; A first appears
    pub f0: f64,                               // freq on origin_kary at t_origin
    pub partial_sweep_final_freq: f64,         // discoal -c; default 1.0
    pub recurrent_mutation_rate: f64,          // discoal -uA; per individual per gen; default 0.0
    pub gamma_flux: f64,                       // gene conversion rate (coal units)
    pub mean_tract_length: f64,                // bp
    pub seed: u64,
    pub dt_scalar: f64,                        // discoal -i; default 400.0
}

pub enum SweepMode { Stochastic, Deterministic, Neutral }
```

Notes on parameterization:

- `s` over `alpha`. `alpha = 2·Ne_cell·s` is computed internally where
  needed (theory anchors). Modellers set `s` directly.
- `Ne_cell` is **not a user input**. It emerges from `N_pop(t) ·
  p_kary(t, pop)` where `p_kary` comes from the joint trajectory and
  `N_pop` from demography. Same idea as discoal scaling sweep N by
  current size ratio, generalized to per-(pop, kary).
- `f0` is interpreted as a frequency *on `origin_kary` in `origin_pop`*.
  De novo origin = `1 / (2 · N_pop · p_kary_initial)`. Soft sweep =
  larger.

## Module structure

```
rust/msinv-core/src/
  sweep.rs                     # rewritten: API + backward-time operator
  sweep_trajectory.rs          # NEW: joint forward WF (sweep+inv+flux+mig)
  sweep_kim_stephan.rs         # NEW: closed-form theory anchors (test-only)
rust/msinv-py/src/lib.rs       # PyO3 bindings updated
msinv/hull/sweep.py            # Python helpers; rewritten

tests/hull/test_phase6_sweep.py             # rewritten
tests/hull/test_phase6b_sweep_joint.py      # NEW
rust/msinv-core/tests/sweep_kim_stephan_anchors.rs  # NEW
```

`sweep_trajectory.rs` is parallel-but-independent from `trajectory.rs`
(the inversion trajectory module). Same math, separate evolution paths
— per user direction, this keeps each cleanly tunable for its own
biology.

## JointSweepTrajectory output

```rust
pub struct JointSweepTrajectory {
    pub t_origin: f64,
    pub tau: f64,
    pub n_pops: u32,
    pub samples: Vec<JointSample>,    // one per generation in [tau, t_origin]
}

pub struct JointSample {
    pub t: f64,                       // generations ago
    pub freq: Vec<[f64; 4]>,          // freq[pop] = [(S,a), (S,A), (I,a), (I,A)]
}
```

Query API used by the coalescent operator:

- `p_kary(t, pop, kary) → f64`
- `p_allele_given_kary(t, pop, kary) → f64`
- `ne_cell(t, pop, kary, n_pop_t) → f64` = `n_pop_t · p_kary(t, pop, kary)`
- `is_class_present(t, pop, class) → bool` (avoid divide-by-zero
  conditions on extinct classes)

Lookup is `O(log n)` (binary search by `t`); samples are
`(t_origin - tau)` long which for typical RDL parameters is a few
hundred to a few thousand entries.

## Backward-time operator wiring

When the coalescent reaches the sweep window `[tau, t_origin]`:

1. **Per-class coalescent rates** in `rate_index.rs`: existing
   structured-coal already partitions lineages by `(pop, kary)`. During
   the sweep window, the per-(pop, kary) effective size becomes
   time-varying via `ne_cell(t, ...)`. Hook is a function pointer or
   match on whether a `JointSweepTrajectory` is active.
2. **Hitchhiking for `A`-bearing lineages**: a lineage of class `(pop,
   kary, A)` at distance `d` from `x_sel` has hitchhiking probability
   `exp(-r · d · T_local)` where `T_local` is the integrated local
   sweep duration on its (pop, kary) trajectory. Lineages of class `a`
   pass through the sweep window without forced coalescence (they're
   the unswept fraction).
3. **Lineage class assignment at sample time**: at `tau`, each sample
   lineage is randomly assigned `A` with probability `p_allele(tau,
   pop, kary)` from the trajectory (matches discoal mechanics:
   `discoalFunctions.c:1914`).
4. **Flux events during sweep**: the existing flux operator continues
   to fire at its usual rate. The joint trajectory was computed knowing
   γ, so the lineage-level flux events are statistically consistent
   with the trajectory's class frequencies.
5. **Migration during sweep**: existing backward-time migration
   continues to fire. Forward-time migration was incorporated into the
   joint trajectory, so the rates the coalescent sees match.

## Theory anchors (validation only, not in simulator)

`sweep_kim_stephan.rs` provides closed-form predictions used as test
anchors. Same role as Andolfatto for flux: order-of-magnitude
expectation, catch regressions; tight quantitative match isn't
expected.

| Anchor | Formula | Validates |
|---|---|---|
| Sojourn time | `T_fix ≈ (2/s)·ln(2·Ne_cell)` | Trajectory rises in ~the right number of gens |
| Fixation probability | `~ 2s/(1+s)` | f0 = 1/(2·Ne_cell) trajectories fix at expected rate |
| Hitchhiking footprint width | `~ s / (r · ln(2·Ne))` | π reduction at distance d matches Kim-Stephan within 25% |
| Flux mixing time | `~ 1 / (γ · L_tract)` per A lineage | Fraction (I,A) → (S,A) at τ matches expected mix |

These are pure functions over `JointSweepTrajectory` outputs; no
simulator state involved.

## Test plan

### `tests/hull/test_phase6_sweep.py` (rewritten)

| ID | Test | Tolerance |
|---|---|---|
| T1 | DetOnly mode, single kary, panmictic, no flux: trajectory matches discrete logistic | `1e-6` per-gen |
| T2 | Stoch mode, de novo (`f0 = 1/(2N)`): fixation proportion ≈ `2s/(1+s)` over 100 reps | MC error |
| T3 | Hitchhiking footprint: π reduction at multiple `d` matches Kim-Stephan | 25% rel |
| T4 | Soft sweep: `f0 > 1/(2N)`, π reduction at `x_sel` ≈ `1 - 1/K`, K = round(1/f0) | 25% rel |
| T5 | Partial sweep: `c = 0.5`, ~50% of lineages in swept fraction | binomial CI |

### `tests/hull/test_phase6b_sweep_joint.py` (new)

| ID | Test |
|---|---|
| J1 | γ=0: A locked to origin_kary; `(S,A)` stays exactly 0 if origin_kary=I |
| J2 | RDL lifecycle: γ>0, origin on I; verify `(S,A)` grows over time, p_I plateaus, total A frequency hits `partial_sweep_final_freq` |
| J3 | Symmetry: origin on S vs origin on I produce mirror trajectories |
| J4 | Bottleneck through sweep: `-en` event during window, trajectory speed responds |
| J5 | Backward flux events fire at right rate during sweep window |
| J6 | Migration spreads sweep: 2-pop, origin in pop 0, m>0 → A appears in pop 1 |
| J7 | m=0 degenerate: pop 1 stays unaffected |
| J8 | Soft sweep from standing variation: f0=0.05, K≈ceil(2·N·p_kary·f0) origins seeded across distinct lineages within `(origin_pop, origin_kary)` at t_origin |
| J9 | Recurrent de novo: uA>0, multiple A-origins fire across the sweep window; verify count matches Poisson(uA·2N·duration) within MC error and origins distribute across kary backgrounds proportional to a-class freqs |

### `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs` (new)

| ID | Test | Tolerance |
|---|---|---|
| A1 | Sojourn time | 25% rel |
| A2 | Fixation probability over reps | MC error |
| A3 | Hitchhiking footprint | 25% rel |
| A4 | Flux mixing time | 25% rel |

Acceptance: T1–T5 + J1–J7 pass; A1–A4 within 25% relative (Tier-1
looseness, same as Andolfatto closed-form anchor for flux).

## Migration plan

1. Delete legacy `Sweep` Hudson-Kaplan code; rewrite from scratch.
2. Delete entire existing `tests/hull/test_phase6_sweep.py`. The 17
   panmictic-target failures listed in `CLAUDE.md` are no longer
   "pre-existing" because the file is replaced.
3. Update `rust/msinv-py/src/lib.rs` PyO3 bridge for new `Sweep` and
   `JointSweepSpec`. Update `msinv/hull/sweep.py`.
4. Update `CLAUDE.md` post-merge: drop the 17-failures paragraph;
   mention new sweep test files; bump test counts.
5. Implementation order (the writing-plans step turns this into the
   actual plan):
   - `sweep_trajectory.rs` joint forward WF — testable in isolation
     against §A1–A4 anchors first.
   - `sweep_kim_stephan.rs` theory anchor module.
   - Backward-time `Sweep` operator wiring into rate_index / event loop.
   - PyO3 + Python helpers.
   - Full integration tests T1–T5, J1–J7.

## Open questions / deferred

1. **Tight quantitative match against theory anchors.** v1 ships with
   25% relative tolerance (Tier-1). Like the Andolfatto closed-form
   for flux (`project_andolfatto_closed_form_correction.md`), tight
   match would require accounting for finite-Ne corrections,
   trajectory fragmentation effects, etc. Deferred theory work; not
   blocking.
2. **Multi-site / multi-inversion joint sweep dynamics.** A second
   selective site with a different allele simultaneously, or a sweep
   that operates on the kary axes of two inversions at once.
   Combinatorial state-space explosion. Out of scope for v1; revisit
   if a use case requires it. (Note: soft sweeps and recurrent de novo
   origins at the *same* site / same allele are in scope and work
   today.)

## References

- discoal source: `/home/adkern/discoal/src/core/discoalFunctions.c`,
  particularly `sweepPhaseEventsGeneralPopNumber` (lines ~1880–2000)
  and the `partialSweepMode` mechanic at line 1912.
- discoal docs: `/home/adkern/discoal/discoaldoc.pdf`.
- partialSHIC: trains on `discoal -c` partial-sweep simulations.
- Existing trajectory machinery (parallel codepath, not reused):
  `rust/msinv-core/src/trajectory.rs`, particularly
  `StochasticDeterministicTrajectory` and `IntegerWFTrajectory`
  (commit `1c7c8b2`).
- Existing flux operator (used as-is during sweep window):
  `rust/msinv-core/src/simulator.rs::apply_gene_flux`.
- Theory: Kim & Stephan 2002, Hudson & Kaplan 1988, Stephan 2019 review.
- Memory: `project_kir_fol_3ra_sweep.md`, `project_rdl_abc_todo.md`,
  `project_msinv_todo.md` (item 3).
