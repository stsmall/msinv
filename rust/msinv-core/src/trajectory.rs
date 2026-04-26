//! Inversion frequency trajectories through time.
//!
//! Ports the four trajectory types from the legacy SMC engine
//! (`msinv/simulator.py`, removed in commit `eb47504` "v0.3.0:
//! Hull-only — full SMC removal").  Each trajectory exposes
//! `p_inv_at(t, pop)` returning the inversion frequency in
//! population `pop` at backward time `t` (generations).
//!
//! - [`ConstantTrajectory`]: fixed `p_inv` per pop, with optional
//!   finite `t_inv` (returns 0 for t >= t_inv).  Drop-in replacement
//!   for the old `InversionSpec.p_inv: Vec<f64>` behaviour.
//! - [`DeterministicTrajectory`]: logistic forward sweep from
//!   1/(2N) to `p_final` under selection `s`.  `t_inv` is derived
//!   from the logistic equation.
//! - [`StochasticTrajectory`]: Wright-Fisher diffusion backward
//!   from `p_final` with reflecting boundary at `p=0` (recurrent
//!   origins).  `t_inv` is the first time `p` reaches 1/(2N).
//!   Pre-computes (times, freqs) at construction; queries linearly
//!   interpolate.
//! - [`CoupledTrajectory`]: per-population coupled 2D diffusion
//!   with selection `s_i` per pop and symmetric migration `m`.
//!   `t_inv` is when ALL pops have reached 1/(2N_i).
//!
//! All trajectories implement [`Trajectory`].  Coupled and
//! Stochastic trajectories cache their pre-computed frequency paths.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Per-population inversion frequency over backward time.
///
/// Implementations MUST be `Send + Sync` so the simulator can
/// hold `Box<dyn Trajectory + Send + Sync>` in `InversionSpec`.
pub trait Trajectory: Send + Sync + std::fmt::Debug {
    /// Inversion frequency at backward time `t` (generations) in
    /// population `pop`.  Returns 0.0 when `t >= t_inv(pop)`.
    fn p_inv_at(&self, t: f64, pop: u32) -> f64;

    /// Time at which the inversion arose in population `pop`
    /// (backward generations).  Beyond this time the inversion
    /// did not exist; class barrier dissolves.
    fn t_inv(&self, pop: u32) -> f64;

    /// Maximum `t_inv` across all known populations.  Used by
    /// the simulator to know when the barrier era ends globally.
    fn t_inv_max(&self) -> f64;

    /// Number of populations this trajectory tracks (0 = single).
    fn n_pops(&self) -> usize;

    /// Clone into a new boxed Trajectory.
    fn clone_boxed(&self) -> Box<dyn Trajectory + Send + Sync>;
}

impl Clone for Box<dyn Trajectory + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

// =====================================================================
// Constant trajectory
// =====================================================================

/// Fixed inversion frequency per population.  Drop-in replacement for
/// the original `InversionSpec.p_inv: Vec<f64>`.
#[derive(Clone, Debug)]
pub struct ConstantTrajectory {
    /// Per-population frequency.  `p_inv[pop]` is used; falls back to
    /// `p_inv[0]` if `pop` is out of bounds.
    pub p_inv: Vec<f64>,
    /// Time at which the inversion arose (same for all pops).  Beyond
    /// this time, returns 0.0 and barrier lifts.
    pub t_inv: f64,
}

impl ConstantTrajectory {
    pub fn new(p_inv: Vec<f64>, t_inv: f64) -> Self {
        Self { p_inv, t_inv }
    }

    /// Single-pop convenience.
    pub fn single(p_inv: f64, t_inv: f64) -> Self {
        Self {
            p_inv: vec![p_inv],
            t_inv,
        }
    }

    pub fn p_inv_for(&self, pop: u32) -> f64 {
        self.p_inv
            .get(pop as usize)
            .copied()
            .unwrap_or_else(|| self.p_inv[0])
    }

    pub fn set_p_inv_for(&mut self, pop: u32, val: f64) {
        let idx = pop as usize;
        if idx >= self.p_inv.len() {
            let fill = self.p_inv[0];
            self.p_inv.resize(idx + 1, fill);
        }
        self.p_inv[idx] = val;
    }
}

impl Trajectory for ConstantTrajectory {
    #[inline]
    fn p_inv_at(&self, t: f64, pop: u32) -> f64 {
        if t >= self.t_inv {
            0.0
        } else {
            self.p_inv_for(pop)
        }
    }

    #[inline]
    fn t_inv(&self, _pop: u32) -> f64 {
        self.t_inv
    }

    #[inline]
    fn t_inv_max(&self) -> f64 {
        self.t_inv
    }

    #[inline]
    fn n_pops(&self) -> usize {
        self.p_inv.len()
    }

    fn clone_boxed(&self) -> Box<dyn Trajectory + Send + Sync> {
        Box::new(self.clone())
    }
}

// =====================================================================
// Deterministic logistic trajectory
// =====================================================================

/// Logistic sweep from 1/(2N) to `p_final` under selection `s`.
///
/// Forward equation: p(t) = p0 * exp(s_scaled * t) / (1 - p0 + p0 * exp(s_scaled * t))
/// where s_scaled = 2*N*s.  Going backward, `p_inv_at(t, pop)` returns
/// p evaluated at forward time `t_inv - t`.
///
/// `t_inv` is derived analytically from the logistic equation as the
/// time required to grow from p0 = 1/(2N) up to `p_final` under s.
/// If s <= 0 or p_final <= p0, falls back to `t_inv = 20.0` coalescent
/// units (matching the legacy Python behavior).
#[derive(Clone, Debug)]
pub struct DeterministicTrajectory {
    pub p_final: f64,
    pub n_e: f64,
    pub s: f64,
    s_scaled: f64,
    p0: f64,
    t_inv_cached: f64,
}

impl DeterministicTrajectory {
    pub fn new(p_final: f64, n_e: f64, s: f64) -> Self {
        let s_scaled = 2.0 * n_e * s;
        let p0 = 1.0 / (2.0 * n_e);
        let t_inv = if s_scaled > 0.0 && p_final > p0 {
            ((p_final / (1.0 - p_final)).ln() - (p0 / (1.0 - p0)).ln()) / s_scaled
        } else {
            20.0
        };
        Self {
            p_final,
            n_e,
            s,
            s_scaled,
            p0,
            t_inv_cached: t_inv,
        }
    }
}

impl Trajectory for DeterministicTrajectory {
    fn p_inv_at(&self, t: f64, _pop: u32) -> f64 {
        if t >= self.t_inv_cached {
            return 0.0;
        }
        let t_fwd = self.t_inv_cached - t;
        if self.s_scaled <= 0.0 {
            return self.p0;
        }
        let exp_st = (self.s_scaled * t_fwd).exp();
        let p = self.p0 * exp_st / (1.0 - self.p0 + self.p0 * exp_st);
        p.min(self.p_final)
    }

    fn t_inv(&self, _pop: u32) -> f64 {
        self.t_inv_cached
    }

    fn t_inv_max(&self) -> f64 {
        self.t_inv_cached
    }

    fn n_pops(&self) -> usize {
        0
    }

    fn clone_boxed(&self) -> Box<dyn Trajectory + Send + Sync> {
        Box::new(self.clone())
    }
}

// =====================================================================
// Stochastic Wright-Fisher diffusion trajectory
// =====================================================================

/// Backward Wright-Fisher diffusion from `p_final` toward 1/(2N) with
/// drift + selection and reflecting boundary at p=0 (recurrent origins
/// at the same breakpoints).  The inversion "age" `t_inv` is when p
/// first reaches 1/(2N) going backward.
///
/// Pre-computes the trajectory at construction; `p_inv_at` interpolates
/// from the cached arrays.
#[derive(Clone, Debug)]
pub struct StochasticTrajectory {
    pub p_final: f64,
    pub n_e: f64,
    pub s: f64,
    pub seed: u64,
    times: Vec<f64>,
    freqs: Vec<f64>,
    t_inv_cached: f64,
}

impl StochasticTrajectory {
    pub fn new(p_final: f64, n_e: f64, s: f64, seed: u64) -> Self {
        let p0 = 1.0 / (2.0 * n_e);
        let dt = 1.0 / (2.0 * n_e);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut p = p_final;
        let mut times = vec![0.0_f64];
        let mut freqs = vec![p];
        let mut t = 0.0_f64;
        // Run until p reaches the founding-copy floor or we hit the cap.
        while p > p0 && t < 100.0 {
            let dp_sel = -s * p * (1.0 - p) * dt;
            let var = (p * (1.0 - p) * dt).max(0.0);
            let sd = var.sqrt();
            let dp_drift = if sd > 0.0 {
                Normal::new(0.0, sd).unwrap().sample(&mut rng)
            } else {
                0.0
            };
            let mut p_new = p + dp_sel + dp_drift;
            if p_new <= 0.0 {
                p_new = p_new.abs() + p0;
            }
            if p_new >= 1.0 {
                p_new = 2.0 - p_new;
            }
            p = p_new.clamp(p0, 1.0 - p0);
            t += dt;
            times.push(t);
            freqs.push(p);
        }
        Self {
            p_final,
            n_e,
            s,
            seed,
            times,
            freqs,
            t_inv_cached: t,
        }
    }

    /// Linear interpolation of (times, freqs) at backward time `t`.
    fn interp(&self, t: f64) -> f64 {
        // times is monotone increasing from 0 to t_inv_cached.
        if t <= self.times[0] {
            return self.freqs[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.freqs.last().unwrap();
        }
        // Binary search for the bracketing interval.
        let i = match self
            .times
            .binary_search_by(|x| x.partial_cmp(&t).unwrap())
        {
            Ok(i) => return self.freqs[i],
            Err(i) => i,
        };
        let t0 = self.times[i - 1];
        let t1 = self.times[i];
        let f0 = self.freqs[i - 1];
        let f1 = self.freqs[i];
        let frac = (t - t0) / (t1 - t0);
        f0 + frac * (f1 - f0)
    }
}

impl Trajectory for StochasticTrajectory {
    fn p_inv_at(&self, t: f64, _pop: u32) -> f64 {
        if t >= self.t_inv_cached {
            return 0.0;
        }
        self.interp(t)
    }

    fn t_inv(&self, _pop: u32) -> f64 {
        self.t_inv_cached
    }

    fn t_inv_max(&self) -> f64 {
        self.t_inv_cached
    }

    fn n_pops(&self) -> usize {
        0
    }

    fn clone_boxed(&self) -> Box<dyn Trajectory + Send + Sync> {
        Box::new(self.clone())
    }
}

// =====================================================================
// Coupled per-population trajectory
// =====================================================================

/// Per-population WF diffusion with local selection s_i and symmetric
/// pairwise migration m.  Backward in time:
/// dp_i = -s_i p_i (1-p_i) dt + m * sum_{j!=i}(p_j - p_i) dt
///        + sqrt(p_i(1-p_i) (N_ref/N_i) dt) * dW_i
///
/// `t_inv(pop)` is when pop `pop` reached 1/(2 N[pop]).  `t_inv_max` is
/// when ALL pops have reached the founding-copy floor (the last to go).
/// Frequencies are pre-computed and stored in `freqs[step][pop]`.
#[derive(Clone, Debug)]
pub struct CoupledTrajectory {
    pub p_final: Vec<f64>,
    pub n_e: Vec<f64>,
    pub s: Vec<f64>,
    pub m: f64,
    pub seed: u64,
    n_pops: usize,
    n_ref: f64,
    p0: Vec<f64>,
    times: Vec<f64>,
    freqs: Vec<Vec<f64>>, // freqs[step][pop]
    t_inv_per_pop: Vec<f64>,
    t_inv_global: f64,
}

impl CoupledTrajectory {
    pub fn new(p_final: Vec<f64>, n_e: Vec<f64>, s: Vec<f64>, m: f64, seed: u64) -> Self {
        let n_pops = p_final.len();
        assert_eq!(n_e.len(), n_pops, "n_e must match p_final length");
        assert_eq!(s.len(), n_pops, "s must match p_final length");
        let n_ref = n_e[0];
        let dt = 1.0 / (2.0 * n_ref);
        let p0: Vec<f64> = n_e.iter().map(|ni| 1.0 / (2.0 * ni)).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut p = p_final.clone();
        let mut alive: Vec<bool> = vec![true; n_pops];
        let mut t_inv_per_pop = vec![0.0_f64; n_pops];
        let mut times = vec![0.0_f64];
        let mut freqs = vec![p.clone()];
        let mut t = 0.0_f64;
        while alive.iter().any(|&a| a) && t < 100.0 {
            let mut p_new = p.clone();
            for i in 0..n_pops {
                if !alive[i] {
                    continue;
                }
                let dp_sel = -s[i] * p[i] * (1.0 - p[i]) * dt;
                let mig_sum: f64 = (0..n_pops)
                    .filter(|&j| j != i)
                    .map(|j| p[j] - p[i])
                    .sum();
                let dp_mig = m * mig_sum * dt;
                let var = (p[i] * (1.0 - p[i]) * dt * n_ref / n_e[i]).max(0.0);
                let sd = var.sqrt();
                let dp_drift = if sd > 0.0 {
                    Normal::new(0.0, sd).unwrap().sample(&mut rng)
                } else {
                    0.0
                };
                let mut pi_new = p[i] + dp_sel + dp_mig + dp_drift;
                if pi_new <= 0.0 {
                    pi_new = pi_new.abs() + p0[i];
                }
                if pi_new >= 1.0 {
                    pi_new = 2.0 - pi_new;
                }
                pi_new = pi_new.clamp(p0[i], 1.0 - p0[i]);
                if pi_new <= p0[i] * 1.01 {
                    alive[i] = false;
                    pi_new = 0.0;
                    t_inv_per_pop[i] = t + dt;
                }
                p_new[i] = pi_new;
            }
            p = p_new;
            t += dt;
            times.push(t);
            freqs.push(p.clone());
        }
        // Any pop that never died: assign the global cap as t_inv.
        for i in 0..n_pops {
            if t_inv_per_pop[i] == 0.0 {
                t_inv_per_pop[i] = t;
            }
        }
        let t_inv_global = t;
        Self {
            p_final,
            n_e,
            s,
            m,
            seed,
            n_pops,
            n_ref,
            p0,
            times,
            freqs,
            t_inv_per_pop,
            t_inv_global,
        }
    }

    fn interp(&self, t: f64, pop: u32) -> f64 {
        let pop = pop as usize;
        if pop >= self.n_pops {
            return self.interp(t, 0);
        }
        let pop_traj: Vec<f64> = self.freqs.iter().map(|step| step[pop]).collect();
        if t <= self.times[0] {
            return pop_traj[0];
        }
        if t >= *self.times.last().unwrap() {
            return *pop_traj.last().unwrap();
        }
        let i = match self
            .times
            .binary_search_by(|x| x.partial_cmp(&t).unwrap())
        {
            Ok(i) => return pop_traj[i],
            Err(i) => i,
        };
        let t0 = self.times[i - 1];
        let t1 = self.times[i];
        let f0 = pop_traj[i - 1];
        let f1 = pop_traj[i];
        let frac = (t - t0) / (t1 - t0);
        f0 + frac * (f1 - f0)
    }
}

impl Trajectory for CoupledTrajectory {
    fn p_inv_at(&self, t: f64, pop: u32) -> f64 {
        let pop_idx = (pop as usize).min(self.n_pops - 1);
        if t >= self.t_inv_per_pop[pop_idx] {
            return 0.0;
        }
        self.interp(t, pop)
    }

    fn t_inv(&self, pop: u32) -> f64 {
        let pop_idx = (pop as usize).min(self.n_pops - 1);
        self.t_inv_per_pop[pop_idx]
    }

    fn t_inv_max(&self) -> f64 {
        self.t_inv_global
    }

    fn n_pops(&self) -> usize {
        self.n_pops
    }

    fn clone_boxed(&self) -> Box<dyn Trajectory + Send + Sync> {
        Box::new(self.clone())
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_basic() {
        let traj = ConstantTrajectory::single(0.5, 1000.0);
        assert_eq!(traj.p_inv_at(0.0, 0), 0.5);
        assert_eq!(traj.p_inv_at(500.0, 0), 0.5);
        assert_eq!(traj.p_inv_at(1000.0, 0), 0.0);
        assert_eq!(traj.p_inv_at(2000.0, 0), 0.0);
        assert_eq!(traj.t_inv(0), 1000.0);
    }

    #[test]
    fn constant_per_pop() {
        let traj = ConstantTrajectory::new(vec![0.0, 0.73], 1000.0);
        assert_eq!(traj.p_inv_at(0.0, 0), 0.0);
        assert_eq!(traj.p_inv_at(0.0, 1), 0.73);
        // OOB pop falls back to pop 0
        assert_eq!(traj.p_inv_at(0.0, 2), 0.0);
    }

    #[test]
    fn deterministic_logistic_growth() {
        // p_final=0.5, N=1000, s=0.01 -> s_scaled=20, t_inv ≈ ln(...) / 20
        let traj = DeterministicTrajectory::new(0.5, 1000.0, 0.01);
        // Going backward from t=0 (p=p_final), p should decrease.
        let p0 = traj.p_inv_at(0.0, 0);
        let p_mid = traj.p_inv_at(traj.t_inv(0) * 0.5, 0);
        assert!(p_mid < p0, "p should decrease going backward");
        assert!(p_mid > 0.0);
        // At t_inv, p = 0.
        assert_eq!(traj.p_inv_at(traj.t_inv(0) + 1.0, 0), 0.0);
    }

    #[test]
    fn stochastic_reaches_founder() {
        let traj = StochasticTrajectory::new(0.5, 1000.0, 0.0, 42);
        // p_final at t=0
        assert!((traj.p_inv_at(0.0, 0) - 0.5).abs() < 1e-9);
        // p reaches founder by t_inv (or close)
        let p_at_tinv = traj.p_inv_at(traj.t_inv(0) - 1e-9, 0);
        assert!(p_at_tinv > 0.0 && p_at_tinv < 0.5);
        // Beyond t_inv, p = 0
        assert_eq!(traj.p_inv_at(traj.t_inv(0) + 1.0, 0), 0.0);
    }

    #[test]
    fn stochastic_reproducible_seed() {
        let a = StochasticTrajectory::new(0.5, 1000.0, 0.001, 12345);
        let b = StochasticTrajectory::new(0.5, 1000.0, 0.001, 12345);
        assert_eq!(a.t_inv_cached, b.t_inv_cached);
        for t in [0.0, 0.001, 0.005, 0.01] {
            assert_eq!(a.p_inv_at(t, 0), b.p_inv_at(t, 0));
        }
    }

    #[test]
    fn coupled_two_pops() {
        // Two pops, no migration, different s.
        let traj = CoupledTrajectory::new(
            vec![0.5, 0.3],
            vec![1000.0, 1000.0],
            vec![0.01, 0.0],
            0.0,
            42,
        );
        // Pop 1 (no selection) should reach founder at a different time
        // than pop 0.
        let t0 = traj.t_inv(0);
        let t1 = traj.t_inv(1);
        // At t=0 each pop has its present-day frequency
        assert!((traj.p_inv_at(0.0, 0) - 0.5).abs() < 1e-9);
        assert!((traj.p_inv_at(0.0, 1) - 0.3).abs() < 1e-9);
        // Both eventually hit zero
        assert_eq!(traj.p_inv_at(traj.t_inv_max() + 1.0, 0), 0.0);
        assert_eq!(traj.p_inv_at(traj.t_inv_max() + 1.0, 1), 0.0);
        // Different per-pop t_inv
        assert!(t0 > 0.0 && t1 > 0.0);
    }
}
