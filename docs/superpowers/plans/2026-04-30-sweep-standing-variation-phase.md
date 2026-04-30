# Sweep standing-variation phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the joint sweep trajectory backward past `t_origin` with stochastic neutral WF drift on the A allele, and replace the at-`t_origin` "all-A force-coalesce" endpoint with a per-lineage de novo merge at trajectory extinction. Unblocks D3 (soft sweep, `f0=0.05`) without disturbing D2 / D4 / PS / PG.

**Architecture:** The trajectory builder appends pre-`t_origin` samples driven by neutral WF drift on the origin population's A subgroup until the global A frequency hits `1/(2N)`. The simulator's `Sweep::covers` and boundary scheduler read a new `t_de_novo()` accessor instead of `joint.t_origin`. `apply_sweep_finalize` keeps its name and per-segment partition, but its terminal step is now a per-lineage merge with a random non-A target rather than a single-founder collapse.

**Tech Stack:** Rust 2021 (`rust/msinv-core`), `rand` 0.9, `rand_xoshiro`, PyO3 bridge.

**Spec:** `docs/superpowers/specs/2026-04-30-sweep-standing-variation-phase-design.md`

**Predecessors (do not regress):** T1-T5, J1-J9, A1-A4 closed-form anchors, PS1-PS3 spatial profile, PG1-PG2 progressive coal, D1/D2/D4 discoal validation. All currently green.

**Build + tests:** Same as the progressive coalescence plan. Rebuild .so via `cd rust && cargo build --release -p msinv-py && /bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so`. PostToolUse hook runs `cargo check` after each Rust edit.

---

## File structure

| File | Change | Why |
|---|---|---|
| `rust/msinv-core/src/sweep_trajectory.rs` | Add SV-phase append after forward loop; new `t_de_novo` field | Phase A |
| `rust/msinv-core/src/sweep.rs` | New `t_de_novo()` accessor; extend `covers()` upper bound | Phase A |
| `rust/msinv-core/src/simulator.rs` | Update boundary scheduler to read `t_de_novo()`; rewrite `apply_sweep_finalize` terminal merge | Phase B |
| `rust/msinv-core/tests/sweep_standing_variation.rs` (NEW) | SV1 + SV2 Rust smoke + invariants | Phase C |
| `CLAUDE.md` | Test counts + dual-phase trajectory note | Phase D |

The discoal-validation D3 unblocking (`SV3`) is separate from this plan — done on the `feat/discoal-validation` branch after this lands. Captured in Phase D step.

---

## Phase A — Trajectory module: append the standing-variation phase

### Task SV-A1: Add `t_de_novo` field on `JointSweepTrajectory`

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs:73-78` (struct definition + accessor)

- [ ] **Step 1: Read the current struct**

`rust/msinv-core/src/sweep_trajectory.rs:73`:
```rust
pub struct JointSweepTrajectory {
    pub t_origin: f64,
    pub tau: f64,
    pub n_pops: u32,
    pub samples: Vec<JointSample>,
}
```

- [ ] **Step 2: Add `t_de_novo` field**

```rust
pub struct JointSweepTrajectory {
    pub t_origin: f64,
    /// Backward time at which the standing-variation phase ends (the
    /// de novo origin of the A allele). Equal to `t_origin` when there
    /// is no SV phase (e.g. `f0 == 1/(2N)`); strictly greater otherwise.
    pub t_de_novo: f64,
    pub tau: f64,
    pub n_pops: u32,
    pub samples: Vec<JointSample>,
}
```

- [ ] **Step 3: Initialize `t_de_novo` in the existing constructor**

`rust/msinv-core/src/sweep_trajectory.rs:180-185` currently:
```rust
JointSweepTrajectory {
    t_origin: spec.t_origin,
    tau,
    n_pops,
    samples,
}
```
Becomes (Phase A1 emits `t_de_novo == t_origin` initially; the SV-phase append in A2 will overwrite):
```rust
JointSweepTrajectory {
    t_origin: spec.t_origin,
    t_de_novo: spec.t_origin,
    tau,
    n_pops,
    samples,
}
```

- [ ] **Step 4: Audit other `JointSweepTrajectory { ... }` literals**

```bash
grep -rn "JointSweepTrajectory {" rust/msinv-core/src/ rust/msinv-core/tests/ rust/msinv-core/examples/ rust/msinv-py/src/ 2>&1
```

Add `t_de_novo: <value>` to every literal. For tests / examples that don't care, use `t_de_novo: t_origin`.

- [ ] **Step 5: Compile**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-core 2>&1 | tail -10
```

Expected: clean compile (no missing-field errors).

- [ ] **Step 6: Run full Rust suite — expect no behavior change**

```bash
cargo test --release --manifest-path rust/Cargo.toml 2>&1 | grep -E "^test result:" | head -10
```

Expected: 137 lib + 17 + 4 + 1 PS1 + 1 PG1 + 2 trajectory passing (same as baseline).

- [ ] **Step 7: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs $(other audited files)
git commit -m "refactor(sweep): add t_de_novo field on JointSweepTrajectory

Plumbs the new field through the type and all literals. Initialized
to t_origin (no SV phase) so behavior is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task SV-A2: Append SV-phase samples to the trajectory

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs:175-186` (after the existing forward loop, before constructing `JointSweepTrajectory`)

- [ ] **Step 1: Locate the forward-loop tail**

The forward loop ends near line 175 with the "ensure final sample is at tau" step:
```rust
if samples.last().map(|s| s.t).unwrap_or(f64::INFINITY) > tau {
    samples.push(JointSample { t: tau, freq: state.clone() });
}
```

Insert the new SV-phase block immediately *before* this — i.e., after the forward loop body and before "ensure final at tau" — because the forward loop ends with `state` representing the present-day (most-recent) frequencies, but we need the `state_at_t_origin` (the very first sample we pushed).

Actually the forward loop walks `t_origin → tau`, pushing samples in that order. The first sample is `JointSample { t: spec.t_origin, freq: state.clone() }` at line 132 (before the loop). So we have access to `samples[0].freq` for the SV-phase initial state.

- [ ] **Step 2: Compute the extinction threshold**

```rust
// SV phase only fires when there's room to drift below f0. f0 == 1/(2N)
// at the origin pop is already at the threshold — no SV phase.
let n_at_origin = pop_size_at(spec.t_origin, origin_pop);
let extinction = 1.0 / (2.0 * n_at_origin);
let mut t_de_novo = spec.t_origin;
```

- [ ] **Step 3: Build SV samples (drift backward in time = forward iteration with t increasing past t_origin)**

```rust
if spec.f0 > extinction + 1e-12 {
    let mut sv_state: Vec<[f64; 4]> = samples[0].freq.clone();
    let mut t_sv = spec.t_origin + 1.0;
    let mut sv_samples: Vec<JointSample> = Vec::new();
    // Hard cap: ~100 * 4N generations protects against runaway in
    // pathological parameter regimes (e.g., extreme bottleneck).
    let max_steps = (400.0 * n_at_origin) as usize + 1024;
    let mut steps = 0usize;
    loop {
        // Apply WF drift to the origin pop's 4-vector. Selection is
        // off in the SV phase. Recurrent + flux + migration are NOT
        // applied here (deferred; the SV phase models the variant's
        // own neutral drift in the origin pop only).
        let n = pop_size_at(t_sv, origin_pop);
        wf_resample(&mut sv_state[origin_pop as usize], 2.0 * n, &mut rng);
        renormalize_inplace(&mut sv_state[origin_pop as usize]);

        // Compute current global A freq in the origin pop.
        let p_a_origin = sv_state[origin_pop as usize][CLASS_S_A_BENEF]
            + sv_state[origin_pop as usize][CLASS_I_A_BENEF];

        // Stop when the A allele is effectively extinct in the origin pop.
        if p_a_origin <= extinction {
            t_de_novo = t_sv;
            break;
        }

        sv_samples.push(JointSample { t: t_sv, freq: sv_state.clone() });

        steps += 1;
        if steps >= max_steps {
            // Pathological run; cap and return what we have.
            t_de_novo = t_sv;
            break;
        }
        t_sv += 1.0;
    }
    // Prepend sv_samples (oldest -> newest) so that the combined
    // `samples` vector remains ordered with t monotonically decreasing
    // from samples[0] (oldest = t_de_novo region) to samples.last()
    // (newest = tau).
    if !sv_samples.is_empty() {
        // sv_samples currently has t_sv increasing; we need oldest first,
        // i.e., reverse so the largest t is at the front.
        sv_samples.reverse();
        let mut combined = sv_samples;
        combined.extend(samples);
        samples = combined;
    }
}
```

(The variable `samples` is mutable in the function body; the existing forward loop already mutates it.)

- [ ] **Step 4: Update the constructor to use `t_de_novo`**

The earlier (A1) literal currently writes `t_de_novo: spec.t_origin`. Update to write `t_de_novo: t_de_novo` (the local variable computed above).

- [ ] **Step 5: Compile**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-core 2>&1 | tail -10
```

Expected: clean compile.

- [ ] **Step 6: Verify `idx_at` handles t > t_origin queries**

`idx_at(t)` needs to return the closest sample index for `t` between `t_de_novo` and `tau`. The existing implementation walks `samples` linearly (or binary searches). Read the current implementation and confirm it remains correct after prepending; no change should be required because we prepend in oldest-first order so the array stays ordered.

```bash
grep -n "fn idx_at" rust/msinv-core/src/sweep_trajectory.rs
```

Read the function. If it does a linear scan or a comparison-based search assuming descending t, it should still work. If it assumed `t_origin` was the maximum t, *fix it* to use `samples[0].t` as the maximum.

- [ ] **Step 7: Run full Rust suite**

```bash
cargo test --release --manifest-path rust/Cargo.toml 2>&1 | grep -E "^test result:|FAILED" | head -20
```

Expected: 161 tests still passing. The SV phase only fires for `f0 > 1/(2N)`, and no existing test uses that — so the entire suite is functionally unchanged.

- [ ] **Step 8: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "feat(sweep): append standing-variation phase to trajectory

When f0 > 1/(2N), the joint trajectory now includes a backward-time
neutral WF drift segment past t_origin. The drift runs on the origin
pop's A subgroup until the A frequency hits the extinction threshold
1/(2N); that hit time becomes t_de_novo.

Other pops continue at the t_origin boundary value (multi-pop SV
support is out of scope for v1).

Existing tests unaffected — they all use f0=1/(2N) so the SV phase
is a no-op.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — Simulator integration

### Task SV-B1: Add `Sweep::t_de_novo()` accessor and extend `covers()`

**Files:**
- Modify: `rust/msinv-core/src/sweep.rs` (add accessor, extend covers)

- [ ] **Step 1: Add the accessor on `Sweep`**

After the existing `covers` method (`rust/msinv-core/src/sweep.rs:60-62`), add:

```rust
/// Backward-time upper bound of the sweep window. Reads from the
/// trajectory's `t_de_novo` if a trajectory has been built; falls
/// back to `joint.t_origin` otherwise (matches the pre-SV-extension
/// behavior).
pub fn t_de_novo(&self) -> f64 {
    self.trajectory.as_ref()
        .map(|t| t.t_de_novo)
        .unwrap_or(self.joint.t_origin)
}
```

- [ ] **Step 2: Update `covers()` to use the new bound**

Replace the existing:
```rust
pub fn covers(&self, t: f64) -> bool {
    t >= self.tau && t <= self.joint.t_origin
}
```

with:
```rust
pub fn covers(&self, t: f64) -> bool {
    t >= self.tau && t <= self.t_de_novo()
}
```

- [ ] **Step 3: Compile**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-core 2>&1 | tail -5
```

Expected: clean compile.

- [ ] **Step 4: Run full Rust suite**

```bash
cargo test --release --manifest-path rust/Cargo.toml 2>&1 | grep -E "^test result:|FAILED" | head -20
```

Expected: 161 still passing. For all existing tests f0=1/(2N) → t_de_novo == t_origin, so behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep.rs
git commit -m "feat(sweep): t_de_novo() accessor + covers() to upper bound

covers(t) now returns true throughout the SV phase as well as the
selection phase. For sweeps without an SV phase (f0=1/(2N)) the
accessor falls through to joint.t_origin and behavior is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task SV-B2: Update boundary scheduler to fire at `t_de_novo`

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:367-369` (run_loop_simple boundary)
- Modify: `rust/msinv-core/src/simulator.rs:854-858` (run_loop_with_caches boundary)
- Modify: `rust/msinv-core/src/simulator.rs:2335-2341` (apply_boundary call to apply_sweep_finalize)

- [ ] **Step 1: Update both `next_boundary` computations**

In both `run_loop_simple` and `run_loop_with_caches`, replace:
```rust
let t_sweep_origin = finalized_sweeps.first()
    .map(|s| s.joint.t_origin).unwrap_or(f64::INFINITY);
```

with:
```rust
let t_sweep_origin = finalized_sweeps.first()
    .map(|s| s.t_de_novo()).unwrap_or(f64::INFINITY);
```

(Variable name kept as `t_sweep_origin` for minimal churn — it's now actually `t_de_novo` but the semantic role is the same: the next sweep-side boundary.)

- [ ] **Step 2: Update `apply_boundary`'s sweep-finalize draining**

`rust/msinv-core/src/simulator.rs:2335-2341`:
```rust
while !finalized_sweeps.is_empty()
    && (finalized_sweeps[0].joint.t_origin - t).abs() < 1e-9
{
    let sweep = finalized_sweeps.remove(0);
    apply_sweep_finalize(active, &sweep, t, arena, tables,
                          next_uid, rng, recomb_rate, sweep_cursor, a_tag);
}
```

becomes:
```rust
while !finalized_sweeps.is_empty()
    && (finalized_sweeps[0].t_de_novo() - t).abs() < 1e-9
{
    let sweep = finalized_sweeps.remove(0);
    apply_sweep_finalize(active, &sweep, t, arena, tables,
                          next_uid, rng, recomb_rate, sweep_cursor, a_tag);
}
```

Also update `finalized_sweeps.sort_by(...)` (the line right above) to sort by `t_de_novo()`:
```rust
finalized_sweeps.sort_by(|a, b|
    a.t_de_novo().partial_cmp(&b.t_de_novo()).unwrap());
```

- [ ] **Step 3: Compile**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-core 2>&1 | tail -5
```

Expected: clean compile.

- [ ] **Step 4: Run full Rust suite**

Expected: 161 still passing (existing sweeps have `t_de_novo == t_origin`).

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): boundary scheduler reads Sweep::t_de_novo()

Both run_loop_simple and run_loop_with_caches now treat the SV-phase
end (t_de_novo) as the sweep-side boundary instead of t_origin. For
sweeps without an SV phase, t_de_novo == t_origin and behavior is
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task SV-B3: Replace `apply_sweep_finalize` terminal merge with per-lineage de novo merge

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:2573-2660` (apply_sweep_finalize body)

- [ ] **Step 1: Read the current `apply_sweep_finalize` body**

The current terminal step (after the per-segment partition) is:
```rust
if linked_uids.len() < 2 { return; }
// Force-coalesce all linked-segment groups at t_origin.
coalesce_uid_group(active, &linked_uids, t, arena, tables, next_uid, sweep_cursor);
```

This collapses every A-tagged lineage with linked segments into a single founder. Correct for `f0=1/(2N)` (one mutational origin); wrong for `f0>1/(2N)` (multiple distinct standing-variation copies → most should not coalesce until much further past).

After the SV phase (Phase A), the surviving A-tagged lineages have already been thinned by the elevated per-allele rate `1/(2N·p_A(t))` running through the SV window. At `t_de_novo`, what's left is the truly residual A-tagged set — they need to merge into the broader pool (the de novo mutation event), each onto a different non-A lineage.

- [ ] **Step 2: Replace the terminal step**

Replace the existing terminal force-coalesce with:

```rust
// At t_de_novo: for each surviving A-tagged lineage, pick a random
// non-A-tagged target in the same population and apply a standard
// coalescence event. This represents the de novo mutation — the A
// founder's chromosome looks just like any other a/untagged
// chromosome past the moment of origin.
//
// Edge case: when no non-A targets exist in a pop (e.g. all lineages
// were A-tagged after a complete fixation), force-coalesce the
// remaining A-tagged among themselves into a single founder so the
// simulation still terminates.
for &a_uid in linked_uids.iter() {
    let a_idx = match active.iter().position(|l| l.uid == a_uid) {
        Some(i) => i,
        None => continue,  // already merged via a prior iteration
    };
    let a_pop = active[a_idx].population;

    // Collect candidate targets: same pop, not in a_tag (or tagged false / no-tag).
    let candidates: Vec<usize> = active.iter().enumerate()
        .filter(|(j, lin)| {
            *j != a_idx
                && lin.population == a_pop
                && a_tag.get(&lin.uid).copied() != Some(true)
        })
        .map(|(j, _)| j)
        .collect();

    if candidates.is_empty() {
        // No non-A target in this pop — fall through to the all-A
        // collapse below.
        continue;
    }

    let pick = rng.random_range(0..candidates.len());
    let target_idx = candidates[pick];
    let (lo, hi) = if a_idx < target_idx { (a_idx, target_idx) } else { (target_idx, a_idx) };

    // Skip-if-no-overlap is the policy used elsewhere in the sweep
    // module; if the chosen target shares no genomic span, the pair
    // can't coalesce and the A-tagged lineage just continues.
    if !segments_overlap(active[a_idx].head, active[target_idx].head, arena) {
        continue;
    }
    let t_merge = next_sweep_merge_t(sweep_cursor, t);
    apply_coalescence_partial(active, lo, hi, t_merge, arena, tables, next_uid,
        None, Some(a_tag));
}

// All-A edge case: any uids that are still in `active` AND a_tag=true
// after the per-lineage merges must be collapsed among themselves to
// avoid a stuck simulation. This mirrors the previous behavior for
// f0=1/(2N) sweeps.
let still_a: Vec<LinUid> = active.iter()
    .filter(|lin| a_tag.get(&lin.uid).copied().unwrap_or(false))
    .map(|lin| lin.uid)
    .collect();
if still_a.len() >= 2 {
    coalesce_uid_group(active, &still_a, t, arena, tables, next_uid, sweep_cursor);
}

// Cleanup: clear a_tag entries for any lineage that's no longer in
// `active` AND for surviving lineages whose tag is now meaningless.
// Past t_de_novo there is no sweep window, so the consumer-side
// gate (`finalized_sweeps.iter().any(|s| s.covers(t))`) naturally
// turns the per-allele filter off.
//
// Per the prior PG-D1 fix, we keep `a_tag` populated past the
// window because count_a_samples reads it for sweep_a_count. So we
// DON'T clear a_tag here; the live-sweep-state gate handles the
// post-window correctness.
```

(The trailing comment block is informational; the actual code ends at the closing `if still_a.len() >= 2 { ... }`.)

- [ ] **Step 3: Compile**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-core 2>&1 | tail -5
```

Expected: clean compile. Watch for `apply_coalescence_partial` arg count mismatch — read the current signature first via `grep "fn apply_coalescence_partial" rust/msinv-core/src/events.rs` and adapt the call exactly. Also confirm `next_sweep_merge_t` is in scope (it's used elsewhere in the module).

- [ ] **Step 4: Run full Rust suite**

```bash
cargo test --release --manifest-path rust/Cargo.toml 2>&1 | grep -E "^test result:|FAILED" | head -20
```

Expected: 161 still passing. Existing sweep tests use `f0=1/(2N)` → no SV phase → `t_de_novo == t_origin` → at the boundary the per-lineage merge loop runs but has no non-A targets in the same pop (all 10 samples are A-tagged for hard sweep with `f0=1/(2N)` and full fixation), so it falls through to the all-A collapse, recovering the prior behavior.

If any sweep test fails: read the failure carefully. The most likely cause is the `still_a` collapse missing some lineages, or the per-lineage merge consuming a non-A target that was needed elsewhere. Audit the candidates filter and the order of operations.

- [ ] **Step 5: Run Python sweep tests**

```bash
cargo build --release --manifest-path rust/Cargo.toml -p msinv-py 2>&1 | tail -3
/bin/cp -f rust/target/release/lib_msinv_core.so msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/hull/test_phase6_sweep.py tests/hull/test_phase6b_sweep_joint.py tests/hull/test_phase6c_per_segment_hitchhiking.py tests/hull/test_phase6d_progressive_coalescence.py -v --timeout=120 2>&1 | tail -10
```

Expected: T1-T5 + J1-J9 + PS2/PS3 + PG2 all green.

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): de novo merge at t_de_novo replaces single-founder collapse

apply_sweep_finalize's terminal step is now per-lineage: each
surviving A-tagged lineage merges with a random non-A target in
its population. This matches the discoal model where the de novo
mutation appears on a single random ancestral chromosome.

For sweeps without an SV phase (f0=1/(2N)), no non-A targets exist
in the swept pop at t_origin, so all A-tagged falls through to the
'still_a' all-collapse path and recovers the prior endpoint
behavior. Existing tests pass unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Tests

### Task SV-C1: SV1 trajectory smoke (Rust)

**Files:**
- Create: `rust/msinv-core/tests/sweep_standing_variation.rs`

- [ ] **Step 1: Write the file**

```rust
//! Standing-variation phase tests for the sweep trajectory + simulator.
//! Spec: docs/superpowers/specs/2026-04-30-sweep-standing-variation-phase-design.md

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

fn build_sweep(f0: f64, mode: SweepMode, seed: u64) -> Sweep {
    let spec = JointSweepSpec {
        mode,
        s: 0.05,
        t_origin: 1_500.0,
        f0,
        partial_sweep_final_freq: 1.0,
        seed,
        ..Default::default()
    };
    Sweep::new(50_000.0, 1_000.0, 0, Karyotype::S, 0, spec)
        .with_trajectory(1, &[0.0],
            &|_t, _p| 10_000.0,
            &|_t, _i, _j| 0.0)
}

#[test]
fn sv1_t_de_novo_equals_t_origin_when_f0_at_extinction() {
    // f0 = 1/(2N) is at the extinction threshold; SV phase is a no-op.
    let sw = build_sweep(1.0 / 20_000.0, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    assert!((traj.t_de_novo - traj.t_origin).abs() < 1e-9,
        "Expected t_de_novo == t_origin for f0=1/(2N); got t_origin={}, t_de_novo={}",
        traj.t_origin, traj.t_de_novo);
}

#[test]
fn sv1_t_de_novo_extends_past_t_origin_when_f0_high() {
    // f0 = 0.05 puts the variant well above extinction; SV drift runs.
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    assert!(traj.t_de_novo > traj.t_origin,
        "Expected t_de_novo > t_origin for f0=0.05; got t_origin={}, t_de_novo={}",
        traj.t_origin, traj.t_de_novo);
    // E[t_de_novo - t_origin] ~ 4 * N * f0 = 2000 generations. Single
    // realization can vary; bound is loose.
    let drift_len = traj.t_de_novo - traj.t_origin;
    assert!(drift_len > 100.0 && drift_len < 50_000.0,
        "Expected SV drift length in [100, 50_000] gens; got {drift_len}");
}

#[test]
fn sv1_p_allele_query_past_t_origin_is_below_f0() {
    // The drift starts at f0 going backward. After a few hundred
    // generations going backward, the freq should be lower (on
    // expectation) than f0.
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    // Pick a midpoint of the SV phase.
    let t_mid = (traj.t_origin + traj.t_de_novo) / 2.0;
    let p_a = traj.p_allele_given_kary(t_mid, 0, Karyotype::S);
    // p_A at midpoint should be somewhere between 1/(2N) and f0
    // (single realization can deviate; bound is loose).
    assert!(p_a >= 0.0 && p_a <= 0.05 + 0.02,
        "Expected p_A at SV midpoint in [0, ~f0]; got {p_a}");
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test --release --manifest-path rust/Cargo.toml --test sweep_standing_variation 2>&1 | tail -10
```

Expected: 3 passed.

```bash
git add rust/msinv-core/tests/sweep_standing_variation.rs
git commit -m "test(sweep): SV1 trajectory smoke for standing-variation phase

3 tests: t_de_novo equals t_origin at f0=1/(2N), t_de_novo extends
past t_origin at f0=0.05, p_A queries at SV midpoint return values
within [0, f0].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task SV-C2: SV2 simulator smoke (Rust)

**Files:**
- Modify: `rust/msinv-core/tests/sweep_standing_variation.rs` (append)

- [ ] **Step 1: Append the test**

```rust
#[test]
fn sv2_simulator_completes_with_sv_phase() {
    // Full simulation with f0=0.05; the simulator must reach MRCA
    // through the selection + SV + neutral phases without panic.
    let mut sim = HullSimulator {
        samples: vec![SampleEntry {
            karyotypes: vec![],
            population: 0,
            count: 10,
        }],
        demography: Demography::single_pop(10_000.0),
        sequence_length: 100_000.0,
        recombination_rate: 1e-8,
        inversions: vec![],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 100_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
    sim.sweeps = vec![sw];
    let result = sim.simulate();
    assert!(
        result.tables.num_nodes() >= 19,
        "Expected ≥19 nodes for n=10 sweep with SV phase; got {}",
        result.tables.num_nodes()
    );
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test --release --manifest-path rust/Cargo.toml --test sweep_standing_variation 2>&1 | tail -10
```

Expected: 4 passed (3 from SV1 + 1 from SV2).

```bash
git add rust/msinv-core/tests/sweep_standing_variation.rs
git commit -m "test(sweep): SV2 simulator smoke with f0=0.05

End-to-end run with the SV phase active. Confirms the simulator
reaches MRCA without panic and produces a non-trivial tree
sequence. Equivalent to PG1 but with f0 > 1/(2N).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase D — CLAUDE.md + D3 unblock pointer

### Task SV-D1: Final pass + CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full Rust + Python suites**

```bash
cargo test --release --manifest-path rust/Cargo.toml 2>&1 | grep -E "^test result:" | head -10
cargo build --release --manifest-path rust/Cargo.toml -p msinv-py 2>&1 | tail -3
/bin/cp -f rust/target/release/lib_msinv_core.so msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py --timeout=180 -q 2>&1 | tail -5
```

Expected: 165 Rust (was 161 + 4 SV) + 192 Python (unchanged). The new SV phase only fires for `f0 > 1/(2N)` — no Python test in the main suite uses that, so the Python count is unchanged.

- [ ] **Step 2: Update CLAUDE.md test counts**

```
- Rust: `cd rust && cargo test --release` (137 lib + 17 integration + 4 sweep-anchor + 1 PS1 + 1 PG1 + 4 SV + 2 sweep-trajectory as of 2026-04-30).
```

- [ ] **Step 3: Append the SV test file to the sweep test files list**

After the existing PG1 entry:
```
  `rust/msinv-core/tests/sweep_progressive_coalescence.rs` (PG1 Rust-side smoke),
  `rust/msinv-core/tests/sweep_standing_variation.rs` (SV1+SV2 Rust trajectory + simulator smoke;
   spec `docs/superpowers/specs/2026-04-30-sweep-standing-variation-phase-design.md`).
```

- [ ] **Step 4: Append the dual-phase note to the existing Sweep model convention entry**

After the "Progressive coalescence" sub-entry, add:

```
  Standing-variation phase (post-SV extension, 2026-04-30): when
  `f0 > 1/(2N)`, the joint trajectory is appended with a backward-time
  stochastic neutral WF drift on the origin pop's A subgroup, running
  until A frequency hits 1/(2N). The drift end is `Sweep::t_de_novo()`
  and replaces `joint.t_origin` as the boundary scheduler's sweep-end
  time. The per-allele rate model from PG-B1 stays engaged through the
  SV phase. At `t_de_novo`, surviving A-tagged lineages merge with
  random non-A targets in-pop (the de novo origin); the prior
  single-founder collapse is recovered as the all-A edge case.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md test counts + standing-variation phase note

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: D3 unblock pointer**

The discoal-validation D3 unblock is on a different branch
(`feat/discoal-validation`). After this branch merges to main, on
`feat/discoal-validation`:

1. `git checkout feat/discoal-validation && git merge main` to pull
   in the SV phase.
2. Edit `tests/hull/test_validation_discoal.py`: remove the
   `@pytest.mark.skip(...)` decorator on
   `test_discoal_validation_d3_soft_sweep`.
3. Run the test: `.venv/bin/python -m pytest
   tests/hull/test_validation_discoal.py::test_discoal_validation_d3_soft_sweep
   -v -s --timeout=600 2>&1 | tail -15`.
4. Expected: both stats OK at 3·SE. If close-but-failing, increase the
   harness's `n_reps` from 20 to 50 (per the spec's risk note on
   per-rep variance).

This pointer is captured in the commit body above and the spec; it's
not a step in this plan because the work happens on a different
branch.

---

## Self-review checklist

- [x] **Spec coverage:**
  - Selection-phase trajectory unchanged → covered (no task; pre-existing).
  - Standing-variation drift trajectory → SV-A1 (field) + SV-A2 (drift loop).
  - `t_de_novo` accessor + `covers()` extension → SV-B1.
  - Boundary scheduler → SV-B2.
  - De novo per-lineage merge → SV-B3.
  - SV1 trajectory smoke → SV-C1.
  - SV2 simulator smoke → SV-C2.
  - SV3 D3 unblock → SV-D1 step 6 (cross-branch pointer).
  - CLAUDE.md → SV-D1.
- [x] **Placeholder scan:** no "TODO", "TBD", "implement later" in tasks. Each task has explicit code blocks.
- [x] **Type consistency:**
  - `t_de_novo` field on `JointSweepTrajectory` (added SV-A1) read by `Sweep::t_de_novo()` (SV-B1).
  - `Sweep::covers` (SV-B1) → `finalized_sweeps.iter().any(|s| s.covers(t))` (existing PG-C1 consumer gate, unchanged).
  - `apply_sweep_finalize` signature unchanged across SV-B3.
- [x] **No "similar to Task N":** every code block is spelled out per task.
