//! Joint forward Wright-Fisher trajectory for a sweep over
//! (karyotype × allele × population) haplotype classes.
//!
//! Pre-computed at sweep construction; consumed backward-in-time by
//! the Sweep operator. Deliberately parallel to `trajectory.rs` (the
//! inversion frequency module) — same math, separate evolution paths.

use crate::class_tag::Karyotype;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(spec.seed);
    let mut t = spec.t_origin;
    while t > tau {
        // For DetOnly we step in 1-gen units; for Stoch we use dt_scalar
        // relative to the smallest 2N. (Refined in Task 4.)
        let dt = 1.0;
        let _ = migration_at; // used in Task 6
        // Selection + drift step
        for (p_idx, f) in state.iter_mut().enumerate() {
            apply_selection_inplace(f, spec.s);
            if matches!(spec.mode, SweepMode::Stochastic) {
                let n = pop_size_at(t, p_idx as u32);
                wf_resample(f, 2.0 * n, &mut rng);
            }
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

/// Stochastic WF resample of a 4-element frequency vector at finite N.
/// Uses sequential conditional binomials with a Gaussian-CLT shortcut
/// when both tails satisfy `mu >= 25` AND `n - mu >= 25` AND var > 0.
/// Otherwise falls back to an integer-Binomial draw via `sample_binomial`.
fn wf_resample(f: &mut [f64; 4], two_n: f64, rng: &mut Xoshiro256PlusPlus) {
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
        let count = if mu >= 25.0 && (n - mu) >= 25.0 && var > 0.0 {
            let normal = Normal::new(mu, var.sqrt()).unwrap();
            normal.sample(rng).round().clamp(0.0, n)
        } else {
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
        let u: f64 = rng.random();
        if u < p {
            k += 1;
        }
    }
    k
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
        let origin_pop = 0u32;
        let origin_kary = Karyotype::S;
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
            1, 0, Karyotype::S, &[0.0],
            &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0, 0.0,
        );
        let n_reps = 100;
        let mut means = vec![0.0_f64; det.samples.len()];
        for r in 0..n_reps {
            let st = build_joint_trajectory(
                &mk_spec(r as u64 + 1, SweepMode::Stochastic),
                1, 0, Karyotype::S, &[0.0],
                &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0, 0.0,
            );
            for (i, s) in st.samples.iter().enumerate().take(means.len()) {
                means[i] += s.freq[0][CLASS_S_A_BENEF];
            }
        }
        for m in means.iter_mut() {
            *m /= n_reps as f64;
        }
        for frac in [0.25, 0.5, 0.75] {
            let i = (means.len() as f64 * frac) as usize;
            let observed = means[i];
            let expected = det.samples[i].freq[0][CLASS_S_A_BENEF];
            assert!(
                (observed - expected).abs() < 0.05,
                "at frac {}: stoch mean={}, det={}",
                frac, observed, expected
            );
        }
    }

    /// Stochastic-mode reps should NOT all be identical (drift must
    /// produce variance).
    #[test]
    fn stoch_reps_vary() {
        let trajs: Vec<_> = (0..10)
            .map(|r| {
                build_joint_trajectory(
                    &JointSweepSpec {
                        mode: SweepMode::Stochastic,
                        s: 0.02, t_origin: 500.0, f0: 0.01,
                        seed: r as u64 + 1,
                        ..Default::default()
                    },
                    1, 0, Karyotype::S, &[0.0],
                    &|_t, _p| 1_000.0, &|_t, _i, _j| 0.0, 0.0,
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
}
