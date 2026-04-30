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
    /// Gene-conversion rate per generation. In v1 this is treated as
    /// the per-gen exchange rate directly; `mean_tract_length` is
    /// reserved (see note on that field).
    pub gamma_flux: f64,
    /// Reserved for v1: not currently consumed by `apply_flux_inplace`
    /// (gamma is interpreted as already-per-generation). Kept in the
    /// public API for forward compatibility with a future tract-length
    /// scaling.
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
    /// Backward time at which the standing-variation phase ends (the
    /// de novo origin of the A allele). Equal to `t_origin` when there
    /// is no SV phase (e.g. `f0 == 1/(2N)`); strictly greater otherwise.
    pub t_de_novo: f64,
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
        // Selection + drift + flux step (per pop)
        for (p_idx, f) in state.iter_mut().enumerate() {
            apply_selection_inplace(f, spec.s);
            let n = pop_size_at(t, p_idx as u32);
            apply_recurrent_inplace(f, spec.recurrent_mutation_rate, 2.0 * n, &mut rng);
            if matches!(spec.mode, SweepMode::Stochastic) {
                wf_resample(f, 2.0 * n, &mut rng);
            }
            apply_flux_inplace(f, spec.gamma_flux, spec.mean_tract_length);
        }
        // Migration step (across pops)
        apply_migration_inplace(&mut state, t, migration_at);
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

    // Standing-variation phase: backward-time stochastic neutral WF
    // drift on the origin pop's A subgroup, starting at f0 and running
    // until A frequency hits the extinction threshold 1/(2N). Other
    // pops are held at their t_origin boundary value (multi-pop SV is
    // out of scope for v1).
    //
    // The trajectory data structure is oldest-first (samples[0].t is
    // the largest). We build the SV samples with t increasing past
    // t_origin (i.e., each step is further into the past), then
    // reverse + prepend so the combined `samples` stays oldest-first.
    let n_at_origin = pop_size_at(spec.t_origin, origin_pop);
    let extinction = 1.0 / (2.0 * n_at_origin);
    let mut t_de_novo = spec.t_origin;
    if spec.f0 > extinction + 1e-12 {
        // Start from the t_origin freq (samples[0] is the t_origin sample).
        let mut sv_state: Vec<[f64; 4]> = samples[0].freq.clone();
        let mut t_sv = spec.t_origin + 1.0;
        let mut sv_samples: Vec<JointSample> = Vec::new();
        // Hard cap so a runaway drift can't loop forever in pathological
        // demographies.
        let max_steps = (400.0 * n_at_origin) as usize + 1024;
        let mut steps = 0usize;
        loop {
            let n = pop_size_at(t_sv, origin_pop);
            wf_resample(&mut sv_state[origin_pop as usize], 2.0 * n, &mut rng);
            renormalize_inplace(&mut sv_state[origin_pop as usize]);

            let p_a_origin = sv_state[origin_pop as usize][CLASS_S_A_BENEF]
                + sv_state[origin_pop as usize][CLASS_I_A_BENEF];

            if p_a_origin <= extinction {
                t_de_novo = t_sv;
                break;
            }

            sv_samples.push(JointSample { t: t_sv, freq: sv_state.clone() });

            steps += 1;
            if steps >= max_steps {
                t_de_novo = t_sv;
                break;
            }
            t_sv += 1.0;
        }
        if !sv_samples.is_empty() {
            // Prepend in oldest-first order: reverse so the largest t
            // (deepest in the past) is at the front, then chain with
            // the existing samples.
            sv_samples.reverse();
            let mut combined = sv_samples;
            combined.extend(samples);
            samples = combined;
        }
    }

    JointSweepTrajectory {
        t_origin: spec.t_origin,
        t_de_novo,
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

/// Per-generation flux exchanges between (I, A) ↔ (S, A) and
/// (I, a) ↔ (S, a) at rate `gamma`. Symmetric: each direction copies
/// at rate `gamma`. `mean_tract_length` is currently unused — kept in
/// the signature for API stability and possible future scaling.
fn apply_flux_inplace(f: &mut [f64; 4], gamma: f64, _mean_tract: f64) {
    if gamma <= 0.0 {
        return;
    }
    let r = gamma.min(0.5);
    let s_a = f[CLASS_S_A];
    let s_a_b = f[CLASS_S_A_BENEF];
    let i_a = f[CLASS_I_A];
    let i_a_b = f[CLASS_I_A_BENEF];
    let new_s_a = s_a + r * i_a - r * s_a;
    let new_i_a = i_a + r * s_a - r * i_a;
    let new_s_a_b = s_a_b + r * i_a_b - r * s_a_b;
    let new_i_a_b = i_a_b + r * s_a_b - r * i_a_b;
    f[CLASS_S_A] = new_s_a.max(0.0);
    f[CLASS_S_A_BENEF] = new_s_a_b.max(0.0);
    f[CLASS_I_A] = new_i_a.max(0.0);
    f[CLASS_I_A_BENEF] = new_i_a_b.max(0.0);
}

/// Per-generation migration: redistribute haplotype-class counts using
/// the migration matrix `m_ij(t)` — the per-gen forward fraction of
/// pop `i` that came from pop `j`.
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

/// Per-generation recurrent de novo: with rate uA · 2N_pop, mutate
/// one (a) → (A) on a random kary background with probability
/// proportional to current a-class counts in that pop.
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
    let dist = match Poisson::new(lambda) {
        Ok(d) => d,
        Err(_) => return,
    };
    let n_events = dist.sample(rng).round() as u64;
    if n_events == 0 {
        return;
    }
    let delta = 1.0 / two_n;
    use rand::Rng;
    for _ in 0..n_events {
        let total_a = f[CLASS_S_A] + f[CLASS_I_A];
        if total_a <= 0.0 { break; }
        let pick_s = rng.random::<f64>() < f[CLASS_S_A] / total_a;
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

impl JointSweepTrajectory {
    /// Find the sample index nearest to `t`. Samples are stored in
    /// order from oldest (first) to most recent (last), so `t` decreases
    /// as the index increases.
    fn idx_at(&self, t: f64) -> usize {
        if self.samples.is_empty() {
            return 0;
        }
        // Find the index where samples[i].t <= t < samples[i-1].t.
        // Since t is decreasing, we look for the largest i with samples[i].t >= t.
        // Linear scan is fine — sample count is bounded by t_origin generations.
        let mut best = 0usize;
        for (i, s) in self.samples.iter().enumerate() {
            if s.t >= t {
                best = i;
            } else {
                break;
            }
        }
        best
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
        // Selection-phase entry should be at t_origin; final at tau=0.
        // (For f0 > 1/(2N) the first sample may be deeper in the SV
        // phase, so search for the t_origin sample by value.)
        let sel_entry = traj.samples.iter()
            .find(|s| (s.t - spec.t_origin).abs() < 1e-9)
            .expect("expected a sample at t_origin");
        assert_eq!(sel_entry.t, spec.t_origin);
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
        // Pick a mid-selection-phase sample by t-value (the selection
        // phase spans [tau, t_origin]; SV-phase samples sit deeper).
        let t_mid_sel = (spec.t_origin + 0.0) / 2.0;
        let mid = traj.samples.iter()
            .min_by(|a, b| {
                (a.t - t_mid_sel).abs()
                    .partial_cmp(&(b.t - t_mid_sel).abs()).unwrap()
            })
            .expect("trajectory must have samples");
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
        // Sample at three times within the selection phase [tau,
        // t_origin]; align stochastic reps to the same wall-clock
        // times via p_allele_overall queries (the trajectories may
        // have different SV-phase lengths so direct index alignment
        // doesn't work).
        let t_origin = mk_spec(0, SweepMode::Deterministic).t_origin;
        let n_reps = 100;
        for frac in [0.25, 0.5, 0.75] {
            let t_query = t_origin * (1.0 - frac);
            let det_p = det.p_allele_given_kary(t_query, 0, Karyotype::S);
            let mut stoch_sum = 0.0;
            for r in 0..n_reps {
                let st = build_joint_trajectory(
                    &mk_spec(r as u64 + 1, SweepMode::Stochastic),
                    1, 0, Karyotype::S, &[0.0],
                    &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0, 0.0,
                );
                stoch_sum += st.p_allele_given_kary(t_query, 0, Karyotype::S);
            }
            let stoch_mean = stoch_sum / n_reps as f64;
            assert!(
                (stoch_mean - det_p).abs() < 0.05,
                "at frac {}: stoch mean={}, det={}",
                frac, stoch_mean, det_p
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

    /// γ > 0: A introduced on I should appear on S over time via flux.
    #[test]
    fn flux_mixes_a_across_karyotypes() {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05,
            t_origin: 1000.0,
            f0: 0.001,
            partial_sweep_final_freq: 0.95,
            gamma_flux: 1e-3,
            mean_tract_length: 1000.0,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::I, &[0.3],
            &|_t, _p| 10_000.0, &|_t, _i, _j| 0.0, 0.0,
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

    /// 2-pop: origin in pop 0, m(1,0)=1e-3 means pop 1 absorbs 1e-3
    /// of its gene pool from pop 0 each gen (matches simulator's
    /// `Demography::migration_matrix[dst][src]` convention). Pop 1
    /// should accumulate A.
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
        let mig = |_t: f64, i: u32, j: u32| if i == 1 && j == 0 { 1e-3 } else { 0.0 };
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

    /// uA > 0: count of recurrent origin events should match Poisson(uA·2N·duration)
    /// within MC error (3 sigma over 50 reps).
    ///
    /// Uses Neutral mode (no drift) so the prev-max heuristic isn't fooled by
    /// drift dropping freq below earlier maxima. With s=0 + no drift, the only
    /// thing that can raise freq[(S,A)] is the recurrent step. Heuristic still
    /// undercounts when 2+ events fire in the same gen (rare for λ=0.2/gen).
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
                mode: SweepMode::Neutral,
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
}
