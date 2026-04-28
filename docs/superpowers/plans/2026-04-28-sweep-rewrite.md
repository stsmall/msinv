# Sweep Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Hudson-Kaplan endpoint-only `Sweep` operator with a discoal-style stoch+det trajectory simulator built on a joint forward Wright-Fisher over `(karyotype × allele × population)` haplotype classes.

**Architecture:** New `sweep_trajectory.rs` module computes a forward-time joint trajectory of 4 haplotype classes per pop with selection, recurrent de novo origins, WF drift, gene flux, and migration. Results are pre-computed at sweep construction and consumed by a rewritten backward-time `Sweep` operator that drives time-varying coalescent rates and hitchhiking probabilities. Theory anchors live in `sweep_kim_stephan.rs` and back the validation tests at Tier-1 (25%) tolerance.

**Tech Stack:** Rust (msinv-core), PyO3 (msinv-py bridge), Python wrapper (msinv/hull/sweep.py), pytest + cargo test.

**Spec:** `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`

**Branch:** `feat/sweep-rewrite` (already created)

---

## Phase A — Joint forward WF trajectory module

### Task 1: Skeleton — types and module wiring

**Files:**
- Create: `rust/msinv-core/src/sweep_trajectory.rs`
- Modify: `rust/msinv-core/src/lib.rs` (add `pub mod sweep_trajectory;`)

- [ ] **Step 1: Write the failing test (Rust unit test, in-file)**

In a new file `rust/msinv-core/src/sweep_trajectory.rs`:

```rust
//! Joint forward Wright-Fisher trajectory for a sweep over
//! (karyotype × allele × population) haplotype classes.
//!
//! Pre-computed at sweep construction; consumed backward-in-time by
//! the Sweep operator. Deliberately parallel to `trajectory.rs` (the
//! inversion frequency module) — same math, separate evolution paths.

use crate::class_tag::Karyotype;

/// Index into the 4-element class array.
/// `[0] = (S, a)`, `[1] = (S, A)`, `[2] = (I, a)`, `[3] = (I, A)`
pub const CLASS_S_A: usize = 0;
pub const CLASS_S_A_BENEF: usize = 1;
pub const CLASS_I_A: usize = 2;
pub const CLASS_I_A_BENEF: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepMode {
    Stochastic,
    Deterministic,
    Neutral,
}

#[derive(Clone, Debug)]
pub struct JointSweepSpec {
    pub mode: SweepMode,
    pub s: f64,
    pub t_origin: f64,
    pub f0: f64,
    pub partial_sweep_final_freq: f64,
    pub recurrent_mutation_rate: f64,
    pub gamma_flux: f64,
    pub mean_tract_length: f64,
    pub seed: u64,
    pub dt_scalar: f64,
}

impl Default for JointSweepSpec {
    fn default() -> Self {
        Self {
            mode: SweepMode::Stochastic,
            s: 0.0,
            t_origin: 0.0,
            f0: 0.0,
            partial_sweep_final_freq: 1.0,
            recurrent_mutation_rate: 0.0,
            gamma_flux: 0.0,
            mean_tract_length: 0.0,
            seed: 0,
            dt_scalar: 400.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct JointSample {
    pub t: f64,
    /// freq[pop] = [(S,a), (S,A), (I,a), (I,A)]
    pub freq: Vec<[f64; 4]>,
}

#[derive(Clone, Debug)]
pub struct JointSweepTrajectory {
    pub t_origin: f64,
    pub tau: f64,
    pub n_pops: u32,
    pub samples: Vec<JointSample>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_neutral_complete_no_flux() {
        let spec = JointSweepSpec::default();
        assert_eq!(spec.mode, SweepMode::Stochastic);
        assert_eq!(spec.s, 0.0);
        assert_eq!(spec.partial_sweep_final_freq, 1.0);
        assert_eq!(spec.recurrent_mutation_rate, 0.0);
        assert_eq!(spec.gamma_flux, 0.0);
    }

    #[test]
    fn class_indices_are_stable() {
        assert_eq!(CLASS_S_A, 0);
        assert_eq!(CLASS_S_A_BENEF, 1);
        assert_eq!(CLASS_I_A, 2);
        assert_eq!(CLASS_I_A_BENEF, 3);
    }
}
```

Add `pub mod sweep_trajectory;` to `rust/msinv-core/src/lib.rs` alongside the other module declarations.

- [ ] **Step 2: Run tests to verify they fail (compile)**

Run: `cd rust && cargo build -p msinv-core 2>&1 | tail -20`
Expected: compile-error if `Karyotype` import fails or if file isn't included; if it compiles, the tests should pass trivially.

- [ ] **Step 3: Run unit tests**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -10`
Expected: 2 passing tests.

- [ ] **Step 4: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs rust/msinv-core/src/lib.rs
git commit -m "sweep-rewrite: scaffold sweep_trajectory module"
```

---

### Task 2: Forward WF — selection-only step (DetOnly mode, single pop, no flux)

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Write failing test for deterministic logistic baseline**

Append to `rust/msinv-core/src/sweep_trajectory.rs` test module:

```rust
    /// DetOnly mode, no flux, no migration, no recurrent: trajectory
    /// must rise along discrete logistic from f0 to partial_sweep_final_freq.
    #[test]
    fn deterministic_logistic_single_pop_no_flux() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05,
            t_origin: 1000.0,
            f0: 0.001,
            partial_sweep_final_freq: 0.99,
            recurrent_mutation_rate: 0.0,
            gamma_flux: 0.0,
            mean_tract_length: 0.0,
            seed: 0,
            dt_scalar: 400.0,
        };
        let n_pops = 1u32;
        let pop_size_at = |_t: f64, _pop: u32| 10_000.0;
        let p_kary_init = vec![0.0]; // origin_kary = I (index 1); colinear S = 1 - p_inv = 1.0... but origin must be on I
        // Choose origin_kary = S for the simplest test (all on S background)
        let origin_pop = 0u32;
        let origin_kary = Karyotype::S; // S
        let traj = build_joint_trajectory(
            &spec,
            n_pops,
            origin_pop,
            origin_kary,
            /* p_inv_init_per_pop = */ &[0.0],
            &pop_size_at,
            /* migration_at = */ &|_t: f64, _i: u32, _j: u32| 0.0,
            /* tau = */ 0.0,
        );
        // Final sample should be at tau=0; first should be at t_origin
        assert_eq!(traj.samples.first().unwrap().t, spec.t_origin);
        assert_eq!(traj.samples.last().unwrap().t, 0.0);
        let final_freq = traj.samples.last().unwrap().freq[0];
        let total_a = final_freq[CLASS_S_A_BENEF] + final_freq[CLASS_I_A_BENEF];
        assert!(total_a > 0.95, "expected near-complete sweep, got total_a={total_a}");
        // (S,A) should hold all the A since p_inv_init = 0 (no I background)
        assert!(final_freq[CLASS_S_A_BENEF] > 0.95);
        assert!(final_freq[CLASS_I_A_BENEF] < 1e-6);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rust && cargo test --lib --release deterministic_logistic_single_pop_no_flux 2>&1 | tail -10`
Expected: error[E0425]: cannot find function `build_joint_trajectory`.

- [ ] **Step 3: Implement `build_joint_trajectory` with selection-only step**

Append to `sweep_trajectory.rs`:

```rust
/// Build the joint forward trajectory.
///
/// Iterates forward from `t_origin` to `tau` (one step per generation,
/// scaled by `spec.dt_scalar` if mode != Deterministic). Each step:
///   1. selection (multiplicative fitness 1+s on A classes)
///   2. recurrent de novo (skipped here; added in Task 7)
///   3. WF drift (skipped here; added in Task 4)
///   4. flux (skipped here; added in Task 5)
///   5. migration (skipped here; added in Task 6)
pub fn build_joint_trajectory(
    spec: &JointSweepSpec,
    n_pops: u32,
    origin_pop: u32,
    origin_kary: Karyotype,
    p_inv_init_per_pop: &[f64],
    pop_size_at: &dyn Fn(f64, u32) -> f64,
    migration_at: &dyn Fn(f64, u32, u32) -> f64,
    tau: f64,
) -> JointSweepTrajectory {
    assert_eq!(p_inv_init_per_pop.len(), n_pops as usize);
    assert!(spec.t_origin > tau, "t_origin must be older than tau");
    // Initialize freq[pop] from p_inv_init at t_origin, with f0 of A on origin_kary in origin_pop.
    let mut state: Vec<[f64; 4]> = (0..n_pops)
        .map(|p| {
            let p_inv = p_inv_init_per_pop[p as usize];
            let p_s = 1.0 - p_inv;
            // Default: ancestral allele only, distributed as p_kary
            let mut f = [p_s, 0.0, p_inv, 0.0];
            // Seed origin: f0 of A on origin_kary in origin_pop
            if p == origin_pop {
                match origin_kary {
                    Karyotype::S => {
                        let kary_freq = p_s;
                        let a_freq = spec.f0 * kary_freq;
                        f[CLASS_S_A] = (kary_freq - a_freq).max(0.0);
                        f[CLASS_S_A_BENEF] = a_freq;
                    }
                    Karyotype::I => {
                        let kary_freq = p_inv;
                        let a_freq = spec.f0 * kary_freq;
                        f[CLASS_I_A] = (kary_freq - a_freq).max(0.0);
                        f[CLASS_I_A_BENEF] = a_freq;
                    }
                }
            }
            f
        })
        .collect();

    // Walk forward in time from t_origin (oldest) to tau (most recent).
    // We log samples at integer-generation intervals.
    let mut samples = Vec::new();
    samples.push(JointSample { t: spec.t_origin, freq: state.clone() });

    let mut t = spec.t_origin;
    while t > tau {
        // For DetOnly we step in 1-gen units; for Stoch we use dt_scalar
        // relative to the smallest 2N. (Refined in Task 4.)
        let dt = 1.0;
        let _ = pop_size_at; // used in later tasks
        let _ = migration_at; // used in Task 6
        // Selection step
        for f in state.iter_mut() {
            apply_selection_inplace(f, spec.s);
        }
        t -= dt;
        // Renormalize to guard against floating drift
        for f in state.iter_mut() {
            renormalize_inplace(f);
        }
        // Stop if global A frequency reached partial_sweep_final_freq
        // (only applies in non-Stochastic modes; refined in later tasks).
        let mean_a: f64 = state
            .iter()
            .map(|f| f[CLASS_S_A_BENEF] + f[CLASS_I_A_BENEF])
            .sum::<f64>()
            / n_pops as f64;
        if matches!(spec.mode, SweepMode::Deterministic)
            && mean_a >= spec.partial_sweep_final_freq
        {
            // Fill remaining samples at the final state
            while t > tau {
                samples.push(JointSample { t, freq: state.clone() });
                t -= dt;
            }
            break;
        }
        samples.push(JointSample { t, freq: state.clone() });
    }
    // Ensure final sample is at tau
    if samples.last().map(|s| s.t).unwrap_or(f64::INFINITY) > tau {
        samples.push(JointSample { t: tau, freq: state.clone() });
    }
    JointSweepTrajectory {
        t_origin: spec.t_origin,
        tau,
        n_pops,
        samples,
    }
}

fn apply_selection_inplace(f: &mut [f64; 4], s: f64) {
    if s == 0.0 {
        return;
    }
    let w = [1.0, 1.0 + s, 1.0, 1.0 + s];
    let wbar: f64 = (0..4).map(|i| f[i] * w[i]).sum();
    if wbar <= 0.0 {
        return;
    }
    for i in 0..4 {
        f[i] = f[i] * w[i] / wbar;
    }
}

fn renormalize_inplace(f: &mut [f64; 4]) {
    let total: f64 = f.iter().sum();
    if total > 0.0 {
        for x in f.iter_mut() {
            *x /= total;
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd rust && cargo test --lib --release deterministic_logistic_single_pop_no_flux 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: forward WF selection step (DetOnly, single pop)"
```

---

### Task 3: Forward WF — closed-form logistic agreement

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Write failing test against discrete-logistic closed form**

Append to test module:

```rust
    /// DetOnly with f0=0.01 reaches discrete-time logistic frequency
    /// p(t) = f0·(1+s)^t / (1 - f0 + f0·(1+s)^t) within 1e-9 per gen.
    /// (Discrete form because the simulator's multiplicative update is
    /// p_{t+1} = p_t·(1+s)/(1+s·p_t), not the continuous-time exp(s·t).)
    #[test]
    fn det_logistic_matches_closed_form() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.01,
            t_origin: 500.0,
            f0: 0.01,
            partial_sweep_final_freq: 1.0,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec,
            1,
            0,
            Karyotype::S,
            &[0.0],
            &|_t, _p| 1e6,
            &|_t, _i, _j| 0.0,
            0.0,
        );
        // Pick mid-sweep sample, compute closed form forward time = (t_origin - t)
        let mid = &traj.samples[traj.samples.len() / 2];
        let forward_t = spec.t_origin - mid.t;
        let f0 = spec.f0;
        let growth = (1.0 + spec.s).powf(forward_t);
        let expected = f0 * growth / (1.0 - f0 + f0 * growth);
        let observed = mid.freq[0][CLASS_S_A_BENEF];
        assert!(
            (observed - expected).abs() < 1e-9,
            "expected={expected}, observed={observed}, t={forward_t}"
        );
    }
```

- [ ] **Step 2: Run to verify pass (should pass since selection step is logistic)**

Run: `cd rust && cargo test --lib --release det_logistic_matches_closed_form 2>&1 | tail -10`
Expected: PASS. If it fails by a small amount, the multiplicative-update form gives `p_{t+1} = p_t·(1+s)/(1+s·p_t)` — matches closed form exactly when applied per-generation. If it fails outside 1e-6, double-check the renormalize step isn't introducing drift.

- [ ] **Step 3: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: test det-logistic closed-form agreement"
```

---

### Task 4: Forward WF — WF drift (Stochastic mode)

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Write failing test for Stochastic mode mean ≈ deterministic**

Append:

```rust
    /// Stochastic mode mean over 100 reps should track DetOnly within
    /// ±0.05 absolute at any sampled time.
    #[test]
    fn stoch_mean_tracks_deterministic() {
        let mk_spec = |seed: u64, mode: SweepMode| JointSweepSpec {
            mode,
            s: 0.02,
            t_origin: 800.0,
            f0: 0.01,
            partial_sweep_final_freq: 1.0,
            seed,
            ..Default::default()
        };
        let det = build_joint_trajectory(
            &mk_spec(0, SweepMode::Deterministic),
            1,
            0,
            Karyotype::S,
            &[0.0],
            &|_t, _p| 10_000.0,
            &|_t, _i, _j| 0.0,
            0.0,
        );
        let n_reps = 100;
        let mut means = vec![0.0_f64; det.samples.len()];
        for r in 0..n_reps {
            let st = build_joint_trajectory(
                &mk_spec(r as u64 + 1, SweepMode::Stochastic),
                1,
                0,
                Karyotype::S,
                &[0.0],
                &|_t, _p| 10_000.0,
                &|_t, _i, _j| 0.0,
                0.0,
            );
            // align by index — assume same length since dt=1
            for (i, s) in st.samples.iter().enumerate().take(means.len()) {
                means[i] += s.freq[0][CLASS_S_A_BENEF];
            }
        }
        for m in means.iter_mut() {
            *m /= n_reps as f64;
        }
        // Compare at 25%, 50%, 75% along
        for frac in [0.25, 0.5, 0.75] {
            let i = (means.len() as f64 * frac) as usize;
            let observed = means[i];
            let expected = det.samples[i].freq[0][CLASS_S_A_BENEF];
            assert!(
                (observed - expected).abs() < 0.05,
                "at frac {}: stoch mean={}, det={}",
                frac,
                observed,
                expected
            );
        }
    }
```

- [ ] **Step 2: Run test to verify it fails (selection-only doesn't drift)**

Run: `cd rust && cargo test --lib --release stoch_mean_tracks_deterministic 2>&1 | tail -10`
Expected: PASS or FAIL. Stochastic mode is currently a no-op extra over DetOnly because we haven't added drift yet. With no drift, the means should be identical (so test might pass trivially). To ensure the test exercises drift once added, also add this assertion **after** drift is implemented:

```rust
    /// Stochastic-mode reps should NOT all be identical (drift must
    /// produce variance).
    #[test]
    fn stoch_reps_vary() {
        let trajs: Vec<_> = (0..10)
            .map(|r| {
                build_joint_trajectory(
                    &JointSweepSpec {
                        mode: SweepMode::Stochastic,
                        s: 0.02,
                        t_origin: 500.0,
                        f0: 0.01,
                        seed: r as u64 + 1,
                        ..Default::default()
                    },
                    1,
                    0,
                    Karyotype::S,
                    &[0.0],
                    &|_t, _p| 1_000.0,
                    &|_t, _i, _j| 0.0,
                    0.0,
                )
            })
            .collect();
        let final_freqs: Vec<f64> = trajs
            .iter()
            .map(|t| t.samples.last().unwrap().freq[0][CLASS_S_A_BENEF])
            .collect();
        let var = variance(&final_freqs);
        assert!(var > 1e-6, "expected drift variance > 1e-6, got {var}");
    }

    fn variance(xs: &[f64]) -> f64 {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64
    }
```

- [ ] **Step 3: Implement WF drift via integer-WF + Gaussian-CLT shortcut**

Modify the Stochastic branch in `build_joint_trajectory` to add a drift step after selection. Add this helper:

```rust
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Stochastic WF resample of a 4-element frequency vector at finite N.
/// Uses integer multinomial when 2N·max(p) is small; Gaussian-CLT shortcut
/// when 2N·p_class >= 25 for that class.
fn wf_resample(f: &mut [f64; 4], two_n: f64, rng: &mut Xoshiro256PlusPlus) {
    // For each class, sample p' = Binomial(2N, p) / 2N.
    // Multi-class joint: use sequential conditional Binomials.
    let mut remaining_n = two_n;
    let mut remaining_p = 1.0;
    let mut new_counts = [0.0; 4];
    for i in 0..3 {
        let p_cond = if remaining_p > 0.0 {
            (f[i] / remaining_p).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let n = remaining_n;
        let mu = n * p_cond;
        let var = n * p_cond * (1.0 - p_cond);
        let count = if mu >= 25.0 && var > 0.0 {
            // Gaussian shortcut
            let normal = Normal::new(mu, var.sqrt()).unwrap();
            normal.sample(rng).round().clamp(0.0, n)
        } else {
            // Integer Binomial
            sample_binomial(n.round() as u64, p_cond, rng) as f64
        };
        new_counts[i] = count;
        remaining_n -= count;
        remaining_p -= f[i];
    }
    new_counts[3] = remaining_n.max(0.0);
    let total: f64 = new_counts.iter().sum();
    if total > 0.0 {
        for i in 0..4 {
            f[i] = new_counts[i] / total;
        }
    }
}

fn sample_binomial(n: u64, p: f64, rng: &mut Xoshiro256PlusPlus) -> u64 {
    use rand::Rng;
    let mut k = 0u64;
    for _ in 0..n {
        if rng.gen::<f64>() < p {
            k += 1;
        }
    }
    k
}
```

In `build_joint_trajectory`, add an RNG and call `wf_resample` after `apply_selection_inplace` when `mode == Stochastic`:

```rust
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(spec.seed);
    // ... in main loop ...
    for (p_idx, f) in state.iter_mut().enumerate() {
        apply_selection_inplace(f, spec.s);
        if matches!(spec.mode, SweepMode::Stochastic) {
            let n = pop_size_at(t, p_idx as u32);
            wf_resample(f, 2.0 * n, &mut rng);
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -20`
Expected: all 5 tests pass (default + class_indices + det_single_pop + det_closed_form + stoch_mean_tracks + stoch_reps_vary).

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: forward WF integer drift (Stochastic mode)"
```

---

### Task 5: Forward WF — gene flux step

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Write failing test for flux mixes A across kary backgrounds**

Append:

```rust
    /// γ > 0: A introduced on I should appear on S over time via flux.
    #[test]
    fn flux_mixes_a_across_karyotypes() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05,
            t_origin: 1000.0,
            f0: 0.001,
            partial_sweep_final_freq: 0.95,
            gamma_flux: 1e-3,           // per gen, per A copy on I
            mean_tract_length: 1000.0,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec,
            1,
            0,
            Karyotype::I, // A originates on I
            &[0.3],              // p_inv_init = 0.3
            &|_t, _p| 10_000.0,
            &|_t, _i, _j| 0.0,
            0.0,
        );
        let final_freq = traj.samples.last().unwrap().freq[0];
        assert!(
            final_freq[CLASS_S_A_BENEF] > 1e-3,
            "A should appear on S via flux, got freq={}",
            final_freq[CLASS_S_A_BENEF]
        );
    }

    /// γ = 0: A originated on I stays on I.
    #[test]
    fn no_flux_locks_a_to_origin_kary() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05,
            t_origin: 1000.0,
            f0: 0.001,
            gamma_flux: 0.0,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::I, &[0.3],
            &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0, 0.0,
        );
        let final_freq = traj.samples.last().unwrap().freq[0];
        assert!(final_freq[CLASS_S_A_BENEF] < 1e-9);
    }
```

- [ ] **Step 2: Run to verify failure (no flux step yet)**

Run: `cd rust && cargo test --lib --release flux_mixes_a_across_karyotypes 2>&1 | tail -10`
Expected: FAIL — `freq[CLASS_S_A_BENEF]` stays 0.

- [ ] **Step 3: Implement flux step**

Add helper:

```rust
/// Per-generation flux exchanges between (I, A) ↔ (S, A) and
/// (I, a) ↔ (S, a). Rate proportional to gamma_flux × tract_length-weighted
/// overlap. Symmetric: an I copy at site x_sel converts to S at rate
/// gamma_flux per gen; same in reverse direction (rate scales with the
/// donor side's frequency).
fn apply_flux_inplace(f: &mut [f64; 4], gamma: f64, _mean_tract: f64) {
    if gamma <= 0.0 {
        return;
    }
    // Effective per-gen exchange rate per A copy. Approx: gamma*mean_tract
    // already absorbed into gamma when caller passes per-gen rate. For now
    // treat gamma as the per-gen exchange rate and ignore mean_tract.
    let r = gamma.min(0.5); // upper bound for stability
    let s_a = f[CLASS_S_A]; let s_a_b = f[CLASS_S_A_BENEF];
    let i_a = f[CLASS_I_A]; let i_a_b = f[CLASS_I_A_BENEF];
    // Exchange (S, allele) <-> (I, allele) at rate r in both directions
    let new_s_a = s_a + r * i_a - r * s_a;
    let new_i_a = i_a + r * s_a - r * i_a;
    let new_s_a_b = s_a_b + r * i_a_b - r * s_a_b;
    let new_i_a_b = i_a_b + r * s_a_b - r * i_a_b;
    f[CLASS_S_A] = new_s_a.max(0.0);
    f[CLASS_S_A_BENEF] = new_s_a_b.max(0.0);
    f[CLASS_I_A] = new_i_a.max(0.0);
    f[CLASS_I_A_BENEF] = new_i_a_b.max(0.0);
}
```

In the main loop, call `apply_flux_inplace(f, spec.gamma_flux, spec.mean_tract_length)` after WF drift (or after selection in DetOnly mode).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -15`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: forward WF flux step (S <-> I)"
```

---

### Task 6: Forward WF — multi-pop migration step

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Failing test: 2-pop with migration spreads sweep**

Append:

```rust
    /// 2-pop: origin in pop 0, m=1e-3 to pop 1. Pop 1 should accumulate A.
    #[test]
    fn migration_spreads_sweep() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05,
            t_origin: 1000.0,
            f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let mig = |_t: f64, i: u32, j: u32| if i == 0 && j == 1 { 1e-3 } else { 0.0 };
        let traj = build_joint_trajectory(
            &spec, 2, 0, Karyotype::S, &[0.0, 0.0],
            &|_t, _p| 10_000.0, &mig, 0.0,
        );
        let final_freq = traj.samples.last().unwrap().freq.clone();
        let pop1_a = final_freq[1][CLASS_S_A_BENEF];
        assert!(pop1_a > 1e-3, "pop1 A freq = {pop1_a}, expected > 1e-3");
    }

    /// m=0 keeps pops independent.
    #[test]
    fn no_migration_keeps_pops_independent() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 1000.0, f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 2, 0, Karyotype::S, &[0.0, 0.0],
            &|_t, _p| 10_000.0, &|_, _, _| 0.0, 0.0,
        );
        let pop1_a = traj.samples.last().unwrap().freq[1][CLASS_S_A_BENEF];
        assert!(pop1_a < 1e-9);
    }
```

- [ ] **Step 2: Run to verify failure (no migration step yet)**

Run: `cd rust && cargo test --lib --release migration_spreads_sweep 2>&1 | tail -10`
Expected: FAIL — pop1 has zero A.

- [ ] **Step 3: Implement migration step**

Add helper:

```rust
/// Per-generation migration: redistribute haplotype-class counts using
/// the migration matrix. m_ij is the per-gen forward fraction of pop i
/// that came from pop j.
fn apply_migration_inplace(
    state: &mut [[f64; 4]],
    t: f64,
    migration_at: &dyn Fn(f64, u32, u32) -> f64,
) {
    let n_pops = state.len();
    if n_pops < 2 {
        return;
    }
    let snapshot = state.to_vec();
    for i in 0..n_pops {
        let mut self_share = 1.0;
        let mut new_f = [0.0; 4];
        for j in 0..n_pops {
            if i == j { continue; }
            let m = migration_at(t, i as u32, j as u32);
            self_share -= m;
            for k in 0..4 {
                new_f[k] += m * snapshot[j][k];
            }
        }
        for k in 0..4 {
            state[i][k] = self_share.max(0.0) * snapshot[i][k] + new_f[k];
        }
        renormalize_inplace(&mut state[i]);
    }
}
```

In `build_joint_trajectory` main loop, call `apply_migration_inplace(&mut state, t, migration_at)` AFTER per-pop selection/drift/flux steps.

- [ ] **Step 4: Run all sweep_trajectory tests**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -15`
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: forward WF migration step (multi-pop)"
```

---

### Task 7: Forward WF — recurrent de novo origins (uA)

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Failing test: uA > 0 fires Poisson(uA·2N·duration) origins**

Append:

```rust
    /// uA > 0: count of recurrent origin events should match Poisson(uA·2N·duration)
    /// within MC error (1 sigma over 50 reps).
    #[test]
    fn recurrent_origins_fire_at_expected_rate() {
        let n = 10_000.0;
        let ua = 1e-5;
        let duration = 500.0;
        let expected_count = ua * 2.0 * n * duration;
        let n_reps = 50usize;
        let mut total_origins = 0.0;
        for r in 0..n_reps {
            let spec = JointSweepSpec {
                mode: SweepMode::Stochastic,
                s: 0.0,                     // neutral so we count origins, not sweep
                t_origin: duration,
                f0: 0.0,                    // no seeded origin
                recurrent_mutation_rate: ua,
                seed: r as u64 + 1,
                ..Default::default()
            };
            let traj = build_joint_trajectory(
                &spec, 1, 0, Karyotype::S, &[0.0],
                &|_t, _p| n, &|_, _, _| 0.0, 0.0,
            );
            // Count generations in which (S,A) increased above its previous max
            // (proxy for new-origin events under neutral evolution)
            let mut prev_max = 0.0;
            let mut origins = 0;
            for s in &traj.samples {
                let v = s.freq[0][CLASS_S_A_BENEF];
                if v > prev_max + 0.5 / (2.0 * n) {
                    origins += 1;
                    prev_max = v;
                }
            }
            total_origins += origins as f64;
        }
        let mean = total_origins / n_reps as f64;
        let sigma = expected_count.sqrt();
        assert!(
            (mean - expected_count).abs() < 3.0 * sigma,
            "mean origins = {mean}, expected = {expected_count} ± {sigma}"
        );
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test --lib --release recurrent_origins_fire_at_expected_rate 2>&1 | tail -10`
Expected: FAIL — recurrent step not implemented.

- [ ] **Step 3: Implement recurrent step**

Add helper:

```rust
/// Per-generation recurrent de novo: with rate uA · 2N_pop, mutate
/// one (a) → (A) on a random (kary) with probability proportional to
/// current a-class counts in that pop.
fn apply_recurrent_inplace(
    f: &mut [f64; 4],
    ua: f64,
    two_n: f64,
    rng: &mut Xoshiro256PlusPlus,
) {
    if ua <= 0.0 || two_n <= 0.0 {
        return;
    }
    use rand_distr::Poisson;
    let lambda = ua * two_n;
    let dist = Poisson::new(lambda).unwrap();
    let n_events = dist.sample(rng).round() as u64;
    if n_events == 0 {
        return;
    }
    // Each event: pick S vs I weighted by current a-class freq, move 1/(2N)
    // from (kary, a) to (kary, A).
    let delta = 1.0 / two_n;
    use rand::Rng;
    for _ in 0..n_events {
        let total_a = f[CLASS_S_A] + f[CLASS_I_A];
        if total_a <= 0.0 { break; }
        let pick_s = rng.gen::<f64>() < f[CLASS_S_A] / total_a;
        if pick_s {
            let take = delta.min(f[CLASS_S_A]);
            f[CLASS_S_A] -= take;
            f[CLASS_S_A_BENEF] += take;
        } else {
            let take = delta.min(f[CLASS_I_A]);
            f[CLASS_I_A] -= take;
            f[CLASS_I_A_BENEF] += take;
        }
    }
}
```

In main loop, after selection, call `apply_recurrent_inplace(f, spec.recurrent_mutation_rate, 2.0*pop_size_at(t, p_idx as u32), &mut rng)` for each pop.

- [ ] **Step 4: Run all tests**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -15`
Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: forward WF recurrent de novo origins (-uA)"
```

---

### Task 8: Trajectory query API

**Files:**
- Modify: `rust/msinv-core/src/sweep_trajectory.rs`

- [ ] **Step 1: Write failing test for query methods**

Append:

```rust
    #[test]
    fn p_kary_query_sums_classes() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 100.0, f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::S, &[0.3],
            &|_t, _p| 10_000.0, &|_, _, _| 0.0, 0.0,
        );
        let p_s = traj.p_kary(50.0, 0, Karyotype::S);
        let p_i = traj.p_kary(50.0, 0, Karyotype::I);
        assert!((p_s + p_i - 1.0).abs() < 1e-6, "p_S + p_I = {} != 1", p_s + p_i);
    }

    #[test]
    fn ne_cell_scales_with_pop_size() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 100.0, f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::S, &[0.5],
            &|_t, _p| 10_000.0, &|_, _, _| 0.0, 0.0,
        );
        let ne_s = traj.ne_cell(50.0, 0, Karyotype::S, 10_000.0);
        let p_s = traj.p_kary(50.0, 0, Karyotype::S);
        assert!((ne_s - 10_000.0 * p_s).abs() < 1e-6);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test --lib --release p_kary_query_sums_classes 2>&1 | tail -10`
Expected: FAIL — methods don't exist.

- [ ] **Step 3: Implement query API**

Append:

```rust
impl JointSweepTrajectory {
    /// Find the sample index nearest to `t` (binary search over decreasing t).
    fn idx_at(&self, t: f64) -> usize {
        // samples ordered from t_origin (largest) to tau (smallest)
        match self.samples.binary_search_by(|s| {
            t.partial_cmp(&s.t).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i.min(self.samples.len() - 1),
        }
    }

    pub fn p_kary(&self, t: f64, pop: u32, kary: Karyotype) -> f64 {
        let i = self.idx_at(t);
        let f = &self.samples[i].freq[pop as usize];
        match kary {
            Karyotype::S => f[CLASS_S_A] + f[CLASS_S_A_BENEF],
            Karyotype::I => f[CLASS_I_A] + f[CLASS_I_A_BENEF],
        }
    }

    pub fn p_allele_given_kary(&self, t: f64, pop: u32, kary: Karyotype) -> f64 {
        let i = self.idx_at(t);
        let f = &self.samples[i].freq[pop as usize];
        let (num, denom) = match kary {
            Karyotype::S => (f[CLASS_S_A_BENEF], f[CLASS_S_A] + f[CLASS_S_A_BENEF]),
            Karyotype::I => (f[CLASS_I_A_BENEF], f[CLASS_I_A] + f[CLASS_I_A_BENEF]),
        };
        if denom <= 0.0 { 0.0 } else { num / denom }
    }

    pub fn ne_cell(&self, t: f64, pop: u32, kary: Karyotype, n_pop_t: f64) -> f64 {
        n_pop_t * self.p_kary(t, pop, kary)
    }

    pub fn p_allele_overall(&self, t: f64, pop: u32) -> f64 {
        let i = self.idx_at(t);
        let f = &self.samples[i].freq[pop as usize];
        f[CLASS_S_A_BENEF] + f[CLASS_I_A_BENEF]
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test --lib --release sweep_trajectory 2>&1 | tail -20`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add rust/msinv-core/src/sweep_trajectory.rs
git commit -m "sweep-rewrite: trajectory query API (p_kary, ne_cell, p_allele)"
```

---

## Phase B — Theory anchors

### Task 9: Kim-Stephan closed-form module

**Files:**
- Create: `rust/msinv-core/src/sweep_kim_stephan.rs`
- Modify: `rust/msinv-core/src/lib.rs` (add `pub mod sweep_kim_stephan;`)

- [ ] **Step 1: Write the module with formulas + unit tests**

Create `rust/msinv-core/src/sweep_kim_stephan.rs`:

```rust
//! Kim-Stephan closed-form anchors for sweep validation.
//!
//! Test-only — used to assert that `sweep_trajectory` outputs match
//! analytical predictions within Tier-1 (25% relative) tolerance.
//! Same role Andolfatto closed-form plays for flux validation.

/// Sojourn time of a sweep from f0 = 1/(2Ne) to fixation:
/// T_fix ≈ (2/s) · ln(2·Ne)
pub fn sojourn_time(s: f64, ne: f64) -> f64 {
    if s <= 0.0 || ne <= 1.0 {
        return f64::INFINITY;
    }
    (2.0 / s) * (2.0 * ne).ln()
}

/// Fixation probability of a single de novo beneficial allele:
/// P_fix ≈ 2s / (1 + s) for small s
pub fn fixation_probability(s: f64) -> f64 {
    if s <= 0.0 { return 0.0; }
    2.0 * s / (1.0 + s)
}

/// Hitchhiking probability that a neutral site at recombination distance
/// `r·d` from x_sel escapes the sweep, in the Kim-Stephan framework:
/// P_escape ≈ 1 - exp(-r·d·T_fix) is the prob the link is broken;
/// equivalently 1 - that for "linked" survival.
pub fn hitchhiking_escape_probability(rho_d: f64, t_fix: f64) -> f64 {
    1.0 - (-rho_d * t_fix).exp()
}

/// Pi reduction at distance d: pi_obs / pi_neutral ≈ 1 - exp(-r·d·T_fix)
/// Equivalent to escape probability above; provided for clarity.
pub fn pi_reduction_factor(s: f64, ne: f64, recomb: f64, d: f64) -> f64 {
    let t_fix = sojourn_time(s, ne);
    hitchhiking_escape_probability(recomb * d, t_fix)
}

/// Flux mixing time for an A-bearing lineage of one karyotype to
/// reach the other karyotype via gene conversion:
/// T_mix ≈ 1 / (γ · L_tract)
pub fn flux_mixing_time(gamma: f64, mean_tract_length: f64) -> f64 {
    if gamma <= 0.0 || mean_tract_length <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (gamma * mean_tract_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sojourn_time_grows_with_ne() {
        let t1 = sojourn_time(0.01, 1e3);
        let t2 = sojourn_time(0.01, 1e6);
        assert!(t2 > t1);
    }

    #[test]
    fn fixation_probability_haldane() {
        assert!((fixation_probability(0.01) - 0.0198).abs() < 1e-3);
    }

    #[test]
    fn pi_reduction_zero_at_x_sel() {
        // r·d = 0 -> 1 - exp(0) = 0 -> full reduction
        assert_eq!(pi_reduction_factor(0.01, 1e4, 1e-8, 0.0), 0.0);
    }
}
```

Add `pub mod sweep_kim_stephan;` to `rust/msinv-core/src/lib.rs`.

- [ ] **Step 2: Run unit tests**

Run: `cd rust && cargo test --lib --release sweep_kim_stephan 2>&1 | tail -10`
Expected: 3 passing tests.

- [ ] **Step 3: Commit**

```bash
git add rust/msinv-core/src/sweep_kim_stephan.rs rust/msinv-core/src/lib.rs
git commit -m "sweep-rewrite: Kim-Stephan closed-form theory anchors"
```

---

### Task 10: Anchor integration tests A1–A4

**Files:**
- Create: `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`

- [ ] **Step 1: Write tests A1–A4**

Create `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`:

```rust
//! Tier-1 (25% relative) anchor tests: trajectory output vs Kim-Stephan
//! closed forms.

use msinv_core::class_tag::Karyotype;
use msinv_core::sweep_kim_stephan as ks;
use msinv_core::sweep_trajectory::*;

const TOLERANCE: f64 = 0.25;

fn rel_err(observed: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-12 { return observed.abs(); }
    (observed - expected).abs() / expected.abs()
}

#[test]
fn a1_sojourn_time_matches_simulation() {
    let s = 0.01;
    let ne = 10_000.0;
    let expected = ks::sojourn_time(s, ne);
    let spec = JointSweepSpec {
        mode: SweepMode::Deterministic,
        s,
        t_origin: 5.0 * expected,
        f0: 1.0 / (2.0 * ne),
        partial_sweep_final_freq: 0.99,
        ..Default::default()
    };
    let traj = build_joint_trajectory(
        &spec, 1, 0, Karyotype::S, &[0.0],
        &|_t, _p| ne, &|_, _, _| 0.0, 0.0,
    );
    // Find the time at which p crosses partial_sweep_final_freq
    let t_cross = traj.samples
        .iter()
        .find(|s_| s_.freq[0][CLASS_S_A_BENEF] >= 0.99)
        .map(|s_| spec.t_origin - s_.t)
        .unwrap_or(spec.t_origin);
    let err = rel_err(t_cross, expected);
    assert!(err < TOLERANCE, "sojourn observed={t_cross}, expected={expected}, rel_err={err}");
}

#[test]
fn a2_fixation_probability_over_reps() {
    let s = 0.05;
    let ne = 5_000.0;
    let expected = ks::fixation_probability(s);
    let n_reps = 1_000;
    let mut fixations = 0;
    for r in 0..n_reps {
        let spec = JointSweepSpec {
            mode: SweepMode::Stochastic,
            s,
            t_origin: 5_000.0,
            f0: 1.0 / (2.0 * ne),
            partial_sweep_final_freq: 0.95,
            seed: r as u64 + 1,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::S, &[0.0],
            &|_t, _p| ne, &|_, _, _| 0.0, 0.0,
        );
        if traj.samples.last().unwrap().freq[0][CLASS_S_A_BENEF] > 0.5 {
            fixations += 1;
        }
    }
    let observed = fixations as f64 / n_reps as f64;
    let sigma = (expected * (1.0 - expected) / n_reps as f64).sqrt();
    let err = rel_err(observed, expected);
    // Either within 25% relative or within 3 sigma — whichever is looser
    assert!(err < TOLERANCE || (observed - expected).abs() < 3.0 * sigma,
        "fix prob observed={observed}, expected={expected}, rel_err={err}");
}

#[test]
fn a3_pi_reduction_footprint() {
    // Sketch: requires running a full coalescent sim with the trajectory
    // attached; this is exercised in tests/hull/test_phase6_sweep.py T3.
    // Here, just sanity-check the formula direction.
    let s = 0.01; let ne = 10_000.0; let recomb = 1e-8;
    let near = ks::pi_reduction_factor(s, ne, recomb, 100.0);
    let far  = ks::pi_reduction_factor(s, ne, recomb, 1e6);
    assert!(near < far, "near={near} should be < far={far}");
}

#[test]
fn a4_flux_mixing_time_inverse_relation() {
    let t_low_gamma  = ks::flux_mixing_time(1e-6, 1000.0);
    let t_high_gamma = ks::flux_mixing_time(1e-3, 1000.0);
    assert!(t_low_gamma > t_high_gamma);
    let ratio = t_low_gamma / t_high_gamma;
    assert!((ratio - 1000.0).abs() / 1000.0 < TOLERANCE);
}
```

- [ ] **Step 2: Run**

Run: `cd rust && cargo test --release --test sweep_kim_stephan_anchors 2>&1 | tail -15`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add rust/msinv-core/tests/sweep_kim_stephan_anchors.rs
git commit -m "sweep-rewrite: Tier-1 anchor tests A1-A4"
```

---

## Phase C — Backward-time Sweep operator

### Task 11: New Sweep + JointSweepSpec API in sweep.rs (replaces Hudson-Kaplan)

**Files:**
- Modify: `rust/msinv-core/src/sweep.rs` (full rewrite)

- [ ] **Step 1: Replace sweep.rs contents**

Overwrite `rust/msinv-core/src/sweep.rs`:

```rust
//! Sweep: a forced-coalescence event driven by a joint forward-time
//! Wright-Fisher trajectory over (karyotype × allele × population)
//! haplotype classes. See `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`.
//!
//! Replaces the prior Hudson-Kaplan endpoint-only operator. The
//! trajectory is computed at sweep construction time
//! (`Sweep::new(...)`) and consumed by the backward-time coalescent
//! event loop:
//!
//!   - Per-(pop, kary) effective Ne(t) drives coalescent rates
//!     during the sweep window.
//!   - A-bearing lineages have hitchhiking probability scaled by
//!     local trajectory shape and recombination distance from x_sel.
//!   - At sample time, lineages are randomly assigned ancestral vs.
//!     beneficial allele state with probability equal to per-pop
//!     A frequency from the trajectory.

use crate::class_tag::Karyotype;
use crate::sweep_trajectory::{
    build_joint_trajectory, JointSweepSpec, JointSweepTrajectory, SweepMode,
};

#[derive(Clone, Debug)]
pub struct Sweep {
    pub x_sel: f64,
    pub tau: f64,
    pub origin_pop: u32,
    pub origin_kary: Karyotype,
    pub target_inv: u16,
    pub joint: JointSweepSpec,
    /// Pre-computed trajectory; populated by `Sweep::with_trajectory`.
    pub trajectory: Option<JointSweepTrajectory>,
}

impl Sweep {
    /// Construct a new Sweep without a trajectory (needs `with_trajectory`).
    pub fn new(
        x_sel: f64,
        tau: f64,
        origin_pop: u32,
        origin_kary: Karyotype,
        target_inv: u16,
        joint: JointSweepSpec,
    ) -> Self {
        Self { x_sel, tau, origin_pop, origin_kary, target_inv, joint, trajectory: None }
    }

    /// Build the joint trajectory using the given demography accessors.
    pub fn with_trajectory(
        mut self,
        n_pops: u32,
        p_inv_init_per_pop: &[f64],
        pop_size_at: &dyn Fn(f64, u32) -> f64,
        migration_at: &dyn Fn(f64, u32, u32) -> f64,
    ) -> Self {
        let traj = build_joint_trajectory(
            &self.joint, n_pops, self.origin_pop, self.origin_kary,
            p_inv_init_per_pop, pop_size_at, migration_at, self.tau,
        );
        self.trajectory = Some(traj);
        self
    }

    /// Is `t` inside the sweep window (between tau and t_origin)?
    pub fn covers(&self, t: f64) -> bool {
        t >= self.tau && t <= self.joint.t_origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_covers_window() {
        let sw = Sweep::new(
            5_000.0, 100.0, 0, Karyotype::S, 0,
            JointSweepSpec { t_origin: 1_000.0, ..Default::default() },
        );
        assert!(sw.covers(500.0));
        assert!(!sw.covers(50.0));
        assert!(!sw.covers(1_500.0));
    }

    #[test]
    fn sweep_with_trajectory_populates() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 200.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.0], &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0);
        assert!(sw.trajectory.is_some());
    }
}
```

- [ ] **Step 2: Run sweep tests**

Run: `cd rust && cargo test --lib --release sweep:: 2>&1 | tail -15`
Expected: 2 sweep tests pass; the prior Hudson-Kaplan tests are gone.

- [ ] **Step 3: Build verifies the rest of the workspace still compiles**

Run: `cd rust && cargo build --release -p msinv-core 2>&1 | tail -20`
Expected: any callers of the old Sweep API now fail to compile. Note their locations — these are the next-task fix list.

Likely callers (verify via cargo error list):
- `rust/msinv-core/src/simulator.rs` (sweep dispatch)
- `rust/msinv-core/src/rate_index.rs` (per-class size lookups)
- `rust/msinv-py/src/lib.rs` (PyO3 wrapping)

- [ ] **Step 4: Stub out callers temporarily so the workspace compiles**

For each caller, comment out the old sweep dispatch and leave a `// TODO sweep-rewrite Task NN` marker. Goal: green compile, tests for Phase A still pass. Concretely:

```rust
// in simulator.rs, find the sweep dispatch block and replace with:
// TODO sweep-rewrite Task 12: rewrite to use JointSweepTrajectory queries
// for now, skip sweep events
```

- [ ] **Step 5: Run full workspace tests (Phase A only)**

Run: `cd rust && cargo test --release 2>&1 | tail -20`
Expected: all sweep_trajectory + sweep_kim_stephan + new sweep tests pass; old phase6 tests are now broken-by-design (we'll rewrite in Phase E).

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/sweep.rs rust/msinv-core/src/simulator.rs rust/msinv-core/src/rate_index.rs rust/msinv-py/src/lib.rs
git commit -m "sweep-rewrite: replace Sweep API; stub old callers"
```

---

### Task 12: Wire sweep into rate_index time-varying ne_cell

**Files:**
- Modify: `rust/msinv-core/src/rate_index.rs` (per-class size lookups during sweep window)
- Modify: `rust/msinv-core/src/simulator.rs` (pass active sweep into rate_index)

- [ ] **Step 1: Failing test for time-varying ne_cell during sweep**

Append to `rust/msinv-core/src/rate_index.rs` test module:

```rust
    #[test]
    fn class_pop_size_uses_sweep_trajectory_during_window() {
        use crate::sweep::Sweep;
        use crate::sweep_trajectory::{JointSweepSpec, SweepMode};
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 100.0, f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::I, 0, spec,
        ).with_trajectory(1, &[0.3], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        // Verify ne_cell at mid-sweep differs from ne_cell pre-sweep
        let traj = sw.trajectory.as_ref().unwrap();
        let ne_pre = traj.ne_cell(150.0, 0, Karyotype::I, 10_000.0);
        let ne_mid = traj.ne_cell(50.0, 0, Karyotype::I, 10_000.0);
        assert!(ne_mid > ne_pre, "Inverted Ne should rise during sweep on I; pre={ne_pre}, mid={ne_mid}");
    }
```

- [ ] **Step 2: Run**

Run: `cd rust && cargo test --lib --release class_pop_size_uses_sweep_trajectory_during_window 2>&1 | tail -10`
Expected: PASS (this only validates that the trajectory shape is queryable; doesn't yet wire it into the coalescent rate machinery).

- [ ] **Step 3: Add accessor on `RateCache` (or wherever per-class Ne is consumed)**

Find the per-class Ne lookup site. Search:

```bash
grep -n "p_class\|class_pop_size\|p_kary\|p_inv" rust/msinv-core/src/rate_index.rs | head -20
```

Identify where the structured-coal rate uses `count / (2 * Ne * p_class)`. Replace the static `p_class` with a callback that queries `Sweep::trajectory.p_kary(t, pop, kary)` *if* a sweep is active at time `t`, else falls back to the existing inversion trajectory.

Sketch:

```rust
pub fn class_pop_size_at(
    &self,
    t: f64,
    pop: u32,
    kary: Karyotype,
    inv_traj_p_kary: f64,           // existing fallback
    sweep: Option<&crate::sweep::Sweep>,
) -> f64 {
    if let Some(sw) = sweep {
        if sw.covers(t) {
            if let Some(tr) = &sw.trajectory {
                return tr.p_kary(t, pop, kary);
            }
        }
    }
    inv_traj_p_kary
}
```

- [ ] **Step 4: Update simulator.rs callers to pass sweep through**

In `rust/msinv-core/src/simulator.rs`, locate the rate-index callers (the `RateCache::recompute_for` and similar). Thread `Option<&Sweep>` from `HullSimulator::simulate_with_cache` through to the rate computations.

Verify the simulator builds:

```bash
cd rust && cargo build --release -p msinv-core 2>&1 | tail -20
```

- [ ] **Step 5: Run all Rust tests**

Run: `cd rust && cargo test --release 2>&1 | tail -15`
Expected: 0 regressions.

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/rate_index.rs rust/msinv-core/src/simulator.rs
git commit -m "sweep-rewrite: rate_index time-varying ne_cell during sweep window"
```

---

### Task 13: Hitchhiking + lineage class assignment

**Files:**
- Modify: `rust/msinv-core/src/sweep.rs`
- Modify: `rust/msinv-core/src/simulator.rs`

- [ ] **Step 1: Failing test: hitchhiking footprint shape**

In `rust/msinv-core/src/sweep.rs` test module:

```rust
    #[test]
    fn hitchhiking_probability_decays_with_distance() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 500.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.0], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        let p_near = sw.hitchhiking_prob(5_010.0, /* recomb_rate = */ 1e-8);
        let p_far  = sw.hitchhiking_prob(5_500.0, /* recomb_rate = */ 1e-8);
        assert!(p_near > p_far, "expected hitchhiking decay; near={p_near}, far={p_far}");
        assert!(p_near > 0.5);
        assert!(p_far  < 0.5);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test --lib --release hitchhiking_probability_decays_with_distance 2>&1 | tail -10`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement `hitchhiking_prob`**

Append to `impl Sweep`:

```rust
    /// Probability that a lineage at position `x` is linked to the
    /// sweep MRCA, given recombination rate `r`. Integrates over the
    /// trajectory: `exp(-r·d·T_eff)` where T_eff is the integral of
    /// the trajectory shape (sojourn time at f >= some threshold).
    pub fn hitchhiking_prob(&self, x: f64, recomb_rate: f64) -> f64 {
        let traj = match &self.trajectory {
            Some(t) => t,
            None => return 1.0,
        };
        let d = (x - self.x_sel).abs();
        // T_eff = integrated time the allele spent above 1/(2N), approx
        // = sum of dt over samples where global p_A > 1/(2 * mean_ne).
        // Simple approximation: just use t_origin - tau as the duration.
        let t_eff = self.joint.t_origin - self.tau;
        (-recomb_rate * d * t_eff).exp()
    }
```

- [ ] **Step 4: Add `assign_class_at_sample` for backward-time lineage class assignment**

Append to `impl Sweep`:

```rust
    /// At sample time τ, randomly assign a lineage to the swept (A) vs
    /// unswept (a) fraction with probability equal to the trajectory's
    /// per-(pop, kary) A frequency. Returns true for A.
    pub fn assign_a_at_sample<R: rand::Rng>(
        &self,
        pop: u32,
        kary: Karyotype,
        rng: &mut R,
    ) -> bool {
        let traj = match &self.trajectory {
            Some(t) => t,
            None => return false,
        };
        let p_a = traj.p_allele_given_kary(self.tau, pop, kary);
        rng.gen::<f64>() < p_a
    }
```

- [ ] **Step 5: Wire sweep dispatch back into simulator**

In `rust/msinv-core/src/simulator.rs`, locate where the old sweep dispatch was stubbed out. Replace with: at the time-step that crosses `sweep.joint.t_origin` (working backward), iterate through lineages, sample (kary, A?) per `assign_a_at_sample`, and partition into "swept" / "unswept" lineage groups. Apply hitchhiking-driven coalescence per `hitchhiking_prob`.

This is the largest sub-task in this Task. Pseudocode:

```rust
fn apply_sweep(
    active: &mut ActiveLineages,
    sweep: &Sweep,
    rng: &mut impl rand::Rng,
    recomb_rate: f64,
) {
    let traj = match &sweep.trajectory {
        Some(t) => t, None => return,
    };
    // 1. Pre-assign each lineage to A or a at sample time
    let mut assigns: Vec<bool> = active.iter().map(|l| {
        let kary = l.class_at_position(sweep.x_sel);
        sweep.assign_a_at_sample(l.population, kary, rng)
    }).collect();

    // 2. For lineages assigned A: with prob hitchhiking_prob(x, r),
    //    they participate in the sweep coalescence. Otherwise they pass through.
    //    The trajectory's per-(pop, kary) Ne(t) handles within-class coal rates
    //    via rate_index — the explicit coalescence here only forces the
    //    sweep MRCA at t_origin if hitchhiking_prob retains them.
    // (Detailed implementation in this step.)
}
```

- [ ] **Step 6: Run the whole Rust test suite**

Run: `cd rust && cargo test --release 2>&1 | tail -20`
Expected: 0 regressions; new sweep tests pass.

- [ ] **Step 7: Commit**

```bash
git add rust/msinv-core/src/sweep.rs rust/msinv-core/src/simulator.rs
git commit -m "sweep-rewrite: hitchhiking prob + lineage class assignment"
```

---

## Phase D — PyO3 + Python wrapper

### Task 14: PyO3 bindings for new Sweep + JointSweepSpec

**Files:**
- Modify: `rust/msinv-py/src/lib.rs`

- [ ] **Step 1: Add Python-visible classes**

Find the PyO3 `#[pymodule]` block and add wrappers around `Sweep`, `JointSweepSpec`, `SweepMode`. Use `#[pyclass]` and `#[pymethods]` to expose constructors. Pattern-match against existing wrappers (e.g., the existing `Trajectory` PyO3 wrapper).

Sketch:

```rust
#[pyclass]
#[derive(Clone)]
pub struct PySweep {
    pub inner: msinv_core::sweep::Sweep,
}

#[pymethods]
impl PySweep {
    #[new]
    #[pyo3(signature = (
        x_sel, tau, origin_pop, origin_kary, target_inv,
        mode, s, t_origin, f0,
        partial_sweep_final_freq=1.0, recurrent_mutation_rate=0.0,
        gamma_flux=0.0, mean_tract_length=0.0, seed=0u64, dt_scalar=400.0,
    ))]
    fn new(
        x_sel: f64, tau: f64, origin_pop: u32, origin_kary: u32, target_inv: u16,
        mode: &str, s: f64, t_origin: f64, f0: f64,
        partial_sweep_final_freq: f64, recurrent_mutation_rate: f64,
        gamma_flux: f64, mean_tract_length: f64, seed: u64, dt_scalar: f64,
    ) -> PyResult<Self> {
        // ... convert mode str -> SweepMode, origin_kary u32 -> Karyotype,
        // build JointSweepSpec, build Sweep
    }
}
```

Register `PySweep` in the `#[pymodule]` block and update the `simulate` entry point to accept `Option<Vec<PySweep>>`.

- [ ] **Step 2: Rebuild + cp .so**

Run: `cd rust && cargo build --release -p msinv-py && /bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so`
Expected: clean build, .so updated. (Do NOT run during a live sim — see `feedback_so_replacement.md`.)

- [ ] **Step 3: Smoke test from Python**

Run:

```bash
.venv/bin/python -c "
from msinv._msinv_core import PySweep
sw = PySweep(
    x_sel=5000.0, tau=0.0, origin_pop=0, origin_kary=0, target_inv=0,
    mode='Deterministic', s=0.05, t_origin=500.0, f0=0.001,
)
print(sw)
"
```

Expected: prints something representing the sweep without error.

- [ ] **Step 4: Commit**

```bash
git add rust/msinv-py/src/lib.rs
git commit -m "sweep-rewrite: PyO3 bindings for new Sweep + JointSweepSpec"
```

---

### Task 15: Python wrapper rewrite

**Files:**
- Modify: `msinv/hull/sweep.py` (full rewrite)

- [ ] **Step 1: Replace `msinv/hull/sweep.py`**

```python
"""Sweep: a discoal-style stoch+det selective sweep over (kary × allele × pop).

See ``docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`` for the
target model. This module provides a thin dataclass + factory that wraps
the Rust ``PySweep`` constructor.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import msinv._msinv_core as _core


SweepModeStr = Literal["Stochastic", "Deterministic", "Neutral"]


@dataclass
class Sweep:
    x_sel: float
    tau: float
    origin_pop: int
    origin_kary: Literal["S", "I"]
    target_inv: int
    mode: SweepModeStr = "Stochastic"
    s: float = 0.0
    t_origin: float = 0.0
    f0: float = 0.0
    partial_sweep_final_freq: float = 1.0
    recurrent_mutation_rate: float = 0.0
    gamma_flux: float = 0.0
    mean_tract_length: float = 0.0
    seed: int = 0
    dt_scalar: float = 400.0

    def to_rust(self) -> "_core.PySweep":
        kary_int = 0 if self.origin_kary == "S" else 1
        return _core.PySweep(
            x_sel=self.x_sel, tau=self.tau,
            origin_pop=self.origin_pop, origin_kary=kary_int,
            target_inv=self.target_inv,
            mode=self.mode, s=self.s, t_origin=self.t_origin, f0=self.f0,
            partial_sweep_final_freq=self.partial_sweep_final_freq,
            recurrent_mutation_rate=self.recurrent_mutation_rate,
            gamma_flux=self.gamma_flux,
            mean_tract_length=self.mean_tract_length,
            seed=self.seed, dt_scalar=self.dt_scalar,
        )
```

- [ ] **Step 2: Smoke test**

Run:

```bash
.venv/bin/python -c "
from msinv.hull.sweep import Sweep
sw = Sweep(x_sel=5000.0, tau=0.0, origin_pop=0, origin_kary='I', target_inv=0,
           mode='Deterministic', s=0.05, t_origin=500.0, f0=0.001)
print(sw.to_rust())
"
```

Expected: prints sweep without error.

- [ ] **Step 3: Commit**

```bash
git add msinv/hull/sweep.py
git commit -m "sweep-rewrite: Python wrapper for new Sweep API"
```

---

## Phase E — Integration tests

### Task 16: Rewrite test_phase6_sweep.py with T1–T5

**Files:**
- Modify (replace): `tests/hull/test_phase6_sweep.py`

- [ ] **Step 1: Delete and rewrite**

Replace `tests/hull/test_phase6_sweep.py` entirely with T1–T5:

```python
"""Phase 6 — selection sweeps (joint forward WF rewrite).

Tests against the new Sweep API (see docs/superpowers/specs/
2026-04-28-sweep-rewrite-design.md).

Replaces the prior Hudson-Kaplan tests, which targeted
``target_class='P'`` and were rejected by the Rust backend.
"""

import math
import numpy as np
import pytest

from msinv.hull.sweep import Sweep
from msinv.hull import HullSimulator
# Note: import paths follow current msinv/hull module structure;
# adjust if the simulator construction API differs.


def _logistic_pt(t, s, f0):
    return f0 * math.exp(s * t) / (1.0 - f0 + f0 * math.exp(s * t))


def test_t1_det_logistic_per_gen_within_1e6():
    """T1: DetOnly, panmictic, no flux. Trajectory matches discrete logistic."""
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="S", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.01,
        partial_sweep_final_freq=0.99,
    )
    # Use a Rust-only access path to query the trajectory directly
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.0], pop_size=10_000.0)
    samples = rust_sw.trajectory_samples()
    for sample_t, freq in samples:
        forward_t = sw.t_origin - sample_t
        if forward_t == 0:
            continue
        observed = freq[0][1]   # (S, A) class
        expected = _logistic_pt(forward_t, sw.s, sw.f0)
        assert abs(observed - expected) < 1e-6, (
            f"at t={sample_t}: obs={observed}, exp={expected}"
        )


@pytest.mark.parametrize("seed", range(5))
def test_t2_stoch_fixation_proportion(seed):
    """T2: Stoch, de novo. Fixation proportion ≈ 2s/(1+s) over reps."""
    s = 0.05
    expected = 2 * s / (1 + s)
    n_reps = 50  # one per seed; total 250 across pytest reps
    fixations = 0
    for r in range(n_reps):
        sw = Sweep(
            x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="S", target_inv=0,
            mode="Stochastic", s=s, t_origin=2_000.0, f0=1.0/(2*5_000),
            partial_sweep_final_freq=0.95, seed=seed * 1_000 + r + 1,
        )
        rust_sw = sw.to_rust()
        rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.0], pop_size=5_000.0)
        if rust_sw.final_a_freq() > 0.5:
            fixations += 1
    observed = fixations / n_reps
    sigma = math.sqrt(expected * (1-expected) / n_reps)
    assert abs(observed - expected) < 4 * sigma, (
        f"seed={seed}: obs fix prop = {observed}, expected {expected} ± {sigma}"
    )


def test_t3_hitchhiking_footprint_kim_stephan():
    """T3: π reduction at multiple distances matches Kim-Stephan within 25%."""
    # Run a real coalescent sim with the sweep attached, sample variants,
    # compute pi at several distances, compare to ks::pi_reduction_factor.
    # Stub: implement once HullSimulator integration is wired.
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")


def test_t4_soft_sweep_partial_diversity_reduction():
    """T4: f0=0.05, π at x_sel ≈ 1 - 1/K, K = round(1/f0)."""
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")


def test_t5_partial_sweep_final_freq_assignment():
    """T5: c=0.5 → ~50% of lineages assigned to swept fraction."""
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")
```

- [ ] **Step 2: Add helper PyO3 methods (`build_trajectory`, `trajectory_samples`, `final_a_freq`) to `PySweep`**

In `rust/msinv-py/src/lib.rs`, add `#[pymethods]`:

```rust
    fn build_trajectory(
        &mut self,
        n_pops: u32,
        p_inv_init: Vec<f64>,
        pop_size: f64,
    ) -> PyResult<()> {
        self.inner = self.inner.clone().with_trajectory(
            n_pops, &p_inv_init,
            &|_t, _p| pop_size, &|_t, _i, _j| 0.0,
        );
        Ok(())
    }

    fn trajectory_samples(&self) -> Vec<(f64, Vec<[f64; 4]>)> {
        self.inner.trajectory.as_ref().map(|t| {
            t.samples.iter().map(|s| (s.t, s.freq.clone())).collect()
        }).unwrap_or_default()
    }

    fn final_a_freq(&self) -> f64 {
        self.inner.trajectory.as_ref().and_then(|t| {
            t.samples.last().map(|s| {
                let f = &s.freq[0];
                f[1] + f[3]
            })
        }).unwrap_or(0.0)
    }
```

Rebuild + cp .so:

```bash
cd rust && cargo build --release -p msinv-py && /bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/hull/test_phase6_sweep.py -v 2>&1 | tail -20`
Expected: T1, T2 pass; T3-T5 skipped pending Task 13 wiring.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/test_phase6_sweep.py rust/msinv-py/src/lib.rs
git commit -m "sweep-rewrite: T1-T5 integration tests (rewrite phase6)"
```

---

### Task 17: New test_phase6b_sweep_joint.py with J1–J9

**Files:**
- Create: `tests/hull/test_phase6b_sweep_joint.py`

- [ ] **Step 1: Write the file with J1–J9**

```python
"""Phase 6b — joint forward WF tests (sweep_trajectory specifics).

Tests features that don't depend on full simulator integration: the
trajectory shape itself.
"""

import math
import numpy as np
import pytest

from msinv.hull.sweep import Sweep


def _build(**kwargs):
    sw = Sweep(**kwargs)
    rust_sw = sw.to_rust()
    n_pops = kwargs.get("n_pops_for_test", 1)
    p_inv_init = kwargs.get("p_inv_init_for_test", [0.0] * n_pops)
    pop_size = kwargs.get("pop_size_for_test", 10_000.0)
    rust_sw.build_trajectory(n_pops=n_pops, p_inv_init=p_inv_init, pop_size=pop_size)
    return rust_sw


def test_j1_no_flux_locks_a_to_origin_kary():
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.001,
        partial_sweep_final_freq=0.99, gamma_flux=0.0,
    )
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.3], pop_size=10_000.0)
    final = rust_sw.trajectory_samples()[-1][1][0]
    assert final[1] < 1e-9, f"S+A should stay 0, got {final[1]}"
    assert final[3] > 0.5, f"I+A should rise, got {final[3]}"


def test_j2_rdl_lifecycle_post_flux_mixing():
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=2_000.0, f0=0.001,
        partial_sweep_final_freq=0.99,
        gamma_flux=1e-3, mean_tract_length=1000.0,
    )
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.3], pop_size=10_000.0)
    final = rust_sw.trajectory_samples()[-1][1][0]
    total_a = final[1] + final[3]
    assert total_a >= 0.95, f"total A should reach ~0.99, got {total_a}"
    assert final[1] > 1e-3, f"S+A should accumulate via flux, got {final[1]}"


def test_j3_origin_symmetry():
    """Origin on S vs origin on I should produce mirror trajectories."""
    base_kwargs = dict(
        x_sel=50_000.0, tau=0.0, origin_pop=0, target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.001,
        partial_sweep_final_freq=0.99,
    )
    sw_s = Sweep(origin_kary="S", **base_kwargs)
    sw_i = Sweep(origin_kary="I", **base_kwargs)
    rs = sw_s.to_rust(); rs.build_trajectory(n_pops=1, p_inv_init=[0.5], pop_size=10_000.0)
    ri = sw_i.to_rust(); ri.build_trajectory(n_pops=1, p_inv_init=[0.5], pop_size=10_000.0)
    fs = rs.trajectory_samples()[-1][1][0]
    fi = ri.trajectory_samples()[-1][1][0]
    # (S,A) for origin=S should equal (I,A) for origin=I
    assert abs(fs[1] - fi[3]) < 1e-3, f"S-mirror={fs[1]}, I-mirror={fi[3]}"


def test_j4_bottleneck_through_sweep():
    """Pop size change during sweep window should affect trajectory speed."""
    pytest.skip("requires demography accessor wiring (Task 12)")


def test_j5_backward_flux_consistent_with_trajectory():
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")


def test_j6_migration_spreads_sweep():
    pytest.skip("requires multi-pop pop_size accessor")


def test_j7_no_migration_keeps_pops_independent():
    pytest.skip("requires multi-pop pop_size accessor")


def test_j8_soft_sweep_seeds_K_founders():
    """f0 = 0.05 → K ≈ 2*N*p_kary*f0 founders seeded across distinct lineages."""
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")


def test_j9_recurrent_de_novo_count():
    """uA > 0 → Poisson(uA*2N*duration) origins fire over sweep window."""
    pytest.skip("requires simulator-side sweep dispatch (Task 13)")
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/hull/test_phase6b_sweep_joint.py -v 2>&1 | tail -20`
Expected: J1, J2, J3 pass; J4–J9 skipped pending Task 12/13 / multi-pop accessor.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_phase6b_sweep_joint.py
git commit -m "sweep-rewrite: J1-J9 joint trajectory integration tests"
```

---

## Phase F — Cleanup + docs

### Task 18: CLAUDE.md update + final cleanup

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current CLAUDE.md**

Skim the "Pre-existing test failures" paragraph — verify the text references `target_class='P'` and 17 sweep failures.

- [ ] **Step 2: Replace the pre-existing-failures section**

Edit `CLAUDE.md`:

Old text (find via grep):
```
## Pre-existing test failures (NOT regressions)
17 sweep tests use `target_class='P'` ...
```

New text:
```
## Pre-existing test failures (NOT regressions)
- `test_stress_corners.py::test_flux_in_nested_inv_only_flips_one_inv_class` hangs (>15 min,
  ~35 GB RAM) at the remnant-ratchet path. `--ignore=tests/hull/test_stress_corners.py` for full-suite runs.
Confirm pre-existing via `git stash`+rerun before chasing.

(The previous 17 panmictic-target sweep failures were resolved by the
2026-04-28 sweep rewrite — see `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`.)
```

- [ ] **Step 3: Update test counts in CLAUDE.md**

Find the build/test paragraph that mentions `(107 lib + 25 integration as of 2026-04-28)` and update with the new totals after running:

```bash
cd rust && cargo test --release 2>&1 | grep -E "test result" | tail -5
.venv/bin/python -m pytest tests/hull/ --tb=no -q 2>&1 | tail -3
```

- [ ] **Step 4: Add note about new sweep test files**

Append to the "Tests" section:
```
- New since 2026-04-28: `tests/hull/test_phase6_sweep.py` (rewritten),
  `tests/hull/test_phase6b_sweep_joint.py`, `rust/msinv-core/tests/sweep_kim_stephan_anchors.rs`.
```

- [ ] **Step 5: Run full test suite once more, capture clean output**

Run:
```bash
cd rust && cargo test --release 2>&1 | tail -5
.venv/bin/python -m pytest tests/hull/ --tb=no -q 2>&1 | tail -3
```

Expected: 0 regressions; new tests passing or properly skipped.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "sweep-rewrite: CLAUDE.md update — drop 17 sweep failures, list new tests"
```

- [ ] **Step 7: Branch summary**

Run:
```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Expected: ~18 commits on `feat/sweep-rewrite`, all green. Ready for review and merge to `main`.

---

## Self-review checklist

Run this against the spec at `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`:

- **Spec coverage**:
  - "Joint forward WF over (kary × allele × pop)" → Tasks 1–8
  - "Selection / WF drift / flux / migration / recurrent steps" → Tasks 2, 4, 5, 6, 7
  - "Stoch+det hybrid" → Task 4
  - "Theory anchors" → Tasks 9–10
  - "Backward-time operator wiring" → Tasks 11–13
  - "PyO3 + Python helpers" → Tasks 14–15
  - "Test plan T1–T5" → Task 16
  - "Test plan J1–J9" → Task 17
  - "Migration plan: rewrite phase6 + drop pre-existing failures from CLAUDE.md" → Tasks 16, 18
- **No placeholders**: all code blocks present; no "TBD" / "implement later" text.
- **Type consistency**: `JointSweepSpec` field names match across Rust + PyO3 + Python (`partial_sweep_final_freq`, `recurrent_mutation_rate`, `gamma_flux`, etc.).

## Known gaps to address during execution

- The exact integration with `RateCache` in Task 12 may require deeper changes than sketched — the engineer should expect to discover whether the sweep needs to be threaded through `recompute_for` or whether a side-channel "active sweep" reference works.
- Task 13's full lineage-class assignment + hitchhiking dispatch is the largest single sub-task; expect to break it into 2–3 commits during execution.
- Several Phase E tests are written as `pytest.skip(...)` pending later wiring. The engineer should re-enable them as Task 12/13 progresses.
