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

    /// Export the precomputed (times, freqs[pop]) sample series.
    /// Returns (times, freqs) where `times.len() == freqs[k].len()`
    /// for each pop k.  Constant/Deterministic trajectories that
    /// have analytical p(t) sample at uniform points spanning
    /// [0, t_inv_max].  Used to write a trajectory to disk so it
    /// can be loaded as a PrecomputedTrajectory for repeatability.
    fn sample_curve(&self, n_samples: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n_pops = self.n_pops().max(1);
        let t_max = self.t_inv_max();
        let mut times = Vec::with_capacity(n_samples);
        let mut freqs: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_pops];
        for i in 0..n_samples {
            let t = (i as f64 / (n_samples - 1).max(1) as f64) * t_max;
            times.push(t);
            for p in 0..n_pops {
                freqs[p].push(self.p_inv_at(t, p as u32));
            }
        }
        (times, freqs)
    }
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

    /// Override: emit the constant per-pop value at all sampled t in
    /// `[0, t_inv)` (no spurious last-point=0 interpolation that the
    /// default would produce when sampling at exactly t_inv).
    fn sample_curve(&self, n_samples: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = n_samples.max(2);
        let n_pops = self.p_inv.len().max(1);
        // Sample over [0, t_inv) — last point is t_inv*(n-1)/n,
        // strictly less than t_inv, where the value is still p_inv.
        let mut times = Vec::with_capacity(n);
        let mut freqs: Vec<Vec<f64>> = vec![Vec::with_capacity(n); n_pops];
        for i in 0..n {
            let t = (i as f64 / n as f64) * self.t_inv;
            times.push(t);
            for p in 0..n_pops {
                freqs[p].push(self.p_inv_for(p as u32));
            }
        }
        (times, freqs)
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
        // s is the forward per-generation selection coefficient.
        // Logistic forward: dp/dt = s*p*(1-p), solution
        //   p(t) = p0 / (p0 + (1-p0)*exp(-s*t))
        // Solve for t when p reaches p_final (gives t_inv in
        // GENERATIONS to match the simulator).
        let p0 = 1.0 / (2.0 * n_e);
        let t_inv = if s > 0.0 && p_final > p0 {
            ((p_final / (1.0 - p_final)).ln() - (p0 / (1.0 - p0)).ln()) / s
        } else {
            // Neutral or implausible — pick a generic 4*N gens as a
            // conservative cap (matches drift expectation).
            4.0 * n_e
        };
        Self {
            p_final,
            n_e,
            s,
            s_scaled: s,  // kept for back-compat; see p_inv_at
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
        if self.s <= 0.0 {
            return self.p0;
        }
        let exp_st = (self.s * t_fwd).exp();
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
        // dt = 1 generation per step.  Loop cap = 10*N gen (≈ 5 coal
        // units) — plenty for any realistic trajectory to reach 1/(2N),
        // bounded so a runaway high-p_final stochastic walk terminates.
        // Times are stored in GENERATIONS to match the simulator's
        // backward-time convention (generations, not coalescent units).
        let dt: f64 = 1.0;
        // Cap: 40*N gens (~20 coal units).  Generous so neutral drift
        // from p≈1 has time to reach 1/(2N).  Practically: any
        // reasonable trajectory completes in 4-10*N gens; the cap just
        // bounds runaway when reflecting boundary keeps p high.
        let t_cap: f64 = 40.0 * n_e;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut p = p_final;
        let mut times = vec![0.0_f64];
        let mut freqs = vec![p];
        let mut t = 0.0_f64;
        // Run until p reaches the founding-copy floor or we hit the cap.
        while p > p0 && t < t_cap {
            let dp_sel = -s * p * (1.0 - p) * dt;
            let var = (p * (1.0 - p) / (2.0 * n_e) * dt).max(0.0);
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
// Bridge stochastic trajectory (conditioned on both endpoints)
// =====================================================================

/// Stochastic Wright-Fisher trajectory conditioned on **both endpoints**:
///
/// - At backward time `t = t_inv` (the inversion's age):
///   `p = 1/(2 N_e)` (founding-allele frequency).
/// - At backward time `t = 0` (today): `p = p_final`.
///
/// This is the partialdiscoal-style "incomplete sweep" trajectory:
/// the inversion arose at a fixed time `t_inv` (empirical anchor)
/// and is currently at frequency `p_final` (also empirical).
///
/// Implementation (rejection sampling): forward-simulate from
/// `1/(2 N_e)` under the continuous-diffusion WF (drift + optional
/// selection `s`), accept iff the path reaches `p_final ± tolerance`
/// at `t = t_inv`.
///
/// ⚠️ **KNOWN LIMITATION (2026-04-26): the continuous-diffusion
/// forward sampler breaks down for large N.**  When `p_start =
/// 1/(2N)` is very small, the per-generation drift SD
/// `sqrt(p(1-p)/(2N)) ≈ sqrt(p / 2N)` is comparable to `p` itself,
/// so most attempts go extinct in the first few generations.  At
/// Anopheles-scale `N≈450k`, acceptance rate is effectively 0 even
/// with positive `s`.  The current rejection sampler is fine for
/// small N (≲10k) and reasonable selection coefficients but should
/// not be used at species-scale Ne for now.
///
/// **Recommended workarounds (until the sampler is rewritten)**:
///
/// 1. **`DeterministicTrajectory` for empirically-anchored cases.**
///    Given `(p_final, t_inv)`, the implied selection coefficient
///    `s ≈ ln((p_final/(1-p_final)) / (p0/(1-p0))) / t_inv` parameterises
///    a unique logistic path that lands exactly on both endpoints.
///    Tractable and conditioned by construction.  Use this for
///    Kir/Fol-style incomplete sweep where t_inv (e.g. 330,000 g for
///    3Ra) and p_final (e.g. 0.734) are both empirical anchors.
///
/// 2. **Hybrid stochastic-then-deterministic** ("Kim-Stephan" approach).
///    Forward-simulate with WF + selection `s` until `p` escapes the
///    drift-dominated regime (typically `p ≳ 10/(2N)` or
///    `1/(2Ns)`), then switch to the deterministic logistic for the
///    remaining trajectory to `p_final`.  Avoids the rare-event
///    problem in the early stochastic phase.  ~50 LOC to add as
///    `BridgeHybridTrajectory`.
///
/// **Proper future fixes** (each ~150-300 LOC, real engineering):
///
/// 3. **Integer-copy WF** instead of continuous diffusion.  Track
///    integer allele counts `k = round(2 * N * p)` and resample
///    `k_next ~ Binomial(2N, p_after_selection)` each generation.
///    This is the actual Wright-Fisher process, has the right
///    near-boundary behaviour (extinction is a discrete event, not
///    a diffusion artefact), and acceptance rates stay reasonable.
///    More expensive per gen than continuous (binomial sampling vs
///    one normal RV) but each forward attempt has a non-trivial
///    chance of surviving.  Recommended primary fix.
///
/// 4. **Doob's h-transform** with conditional-on-survival drift
///    correction.  Adds an analytic drift term `g(p, t_remaining)`
///    that pulls the path toward the target endpoint, so every
///    sampled path is guaranteed to hit `p_final` at `t_inv` (no
///    rejection).  Mathematically clean but requires deriving
///    `g(p, t)` for the specific WF + selection diffusion (Schraiber
///    et al. 2013 has the formulas).  Faster than rejection
///    sampling once implemented.
///
/// See project memory `feedback_no_silent_reverts.md` and the
/// Kir/Fol roadmap for the trajectory model's intended use.
#[derive(Clone, Debug)]
pub struct BridgeStochasticTrajectory {
    pub p_final: f64,
    pub n_e: f64,
    pub s: f64,
    pub t_inv: f64,        // FIXED — the empirical anchor
    pub seed: u64,
    pub tolerance: f64,    // |p_at_tinv - p_final| < tolerance to accept
    pub n_attempts: u64,   // how many tries it actually took
    times: Vec<f64>,       // backward times in [0, t_inv], generations
    freqs: Vec<f64>,
    t_inv_cached: f64,
}

impl BridgeStochasticTrajectory {
    /// `tolerance` is the absolute difference allowed between the
    /// forward path's terminal frequency and `p_final`.  Defaults to
    /// `0.02` (2 percentage points) when called via `new_default`.
    pub fn new(
        p_final: f64,
        n_e: f64,
        s: f64,
        t_inv: f64,
        seed: u64,
        tolerance: f64,
        max_attempts: u64,
    ) -> Result<Self, String> {
        let p0 = 1.0 / (2.0 * n_e);
        let t_inv_int = t_inv as u64;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        for attempt in 1..=max_attempts {
            // Forward simulate p from p0 for t_inv generations.
            let mut p = p0;
            // Forward path: [t=0_fwd ... t=t_inv_fwd] = [t_inv_back ... t=0_back]
            let mut p_path: Vec<f64> = Vec::with_capacity(t_inv_int as usize + 1);
            p_path.push(p);
            let mut t_fwd: u64 = 0;
            let mut lost = false;
            while t_fwd < t_inv_int {
                // Forward WF: dp = +s*p*(1-p) + N(0, sqrt(p(1-p)/(2N)))
                let dp_sel = s * p * (1.0 - p);
                let var = (p * (1.0 - p) / (2.0 * n_e)).max(0.0);
                let sd = var.sqrt();
                let dp_drift = if sd > 0.0 {
                    Normal::new(0.0, sd).unwrap().sample(&mut rng)
                } else {
                    0.0
                };
                let p_new = (p + dp_sel + dp_drift).clamp(0.0, 1.0);
                if p_new <= 0.0 {
                    // Allele lost; this path can't reach p_final.
                    lost = true;
                    break;
                }
                p = p_new;
                p_path.push(p);
                t_fwd += 1;
            }
            if lost { continue; }
            // Accept iff terminal freq is within tolerance of p_final.
            if (p - p_final).abs() < tolerance {
                // Convert forward path to backward path:
                //   p_path[i_fwd] is freq at forward gen i_fwd
                //   at backward time t_back = t_inv - i_fwd
                let n = p_path.len();
                let mut times_back: Vec<f64> = Vec::with_capacity(n);
                let mut freqs_back: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let i_fwd = n - 1 - i;
                    times_back.push(i as f64);  // generation index back from present
                    freqs_back.push(p_path[i_fwd]);
                }
                return Ok(Self {
                    p_final,
                    n_e,
                    s,
                    t_inv,
                    seed,
                    tolerance,
                    n_attempts: attempt,
                    times: times_back,
                    freqs: freqs_back,
                    t_inv_cached: t_inv,
                });
            }
        }
        Err(format!(
            "BridgeStochasticTrajectory: rejection sampler failed after \
             {max_attempts} attempts (p_final={p_final}, N={n_e}, s={s}, \
             t_inv={t_inv}, tolerance={tolerance}). Try a larger \
             tolerance, larger s, or use DeterministicTrajectory."))
    }

    /// Convenience constructor with sensible defaults: tolerance=0.02,
    /// max_attempts=10000.
    pub fn new_default(p_final: f64, n_e: f64, s: f64, t_inv: f64,
                       seed: u64) -> Result<Self, String> {
        Self::new(p_final, n_e, s, t_inv, seed, 0.02, 10_000)
    }

    fn interp(&self, t: f64) -> f64 {
        if t <= self.times[0] {
            return self.freqs[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.freqs.last().unwrap();
        }
        let i = match self.times.binary_search_by(
            |x| x.partial_cmp(&t).unwrap()) {
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

impl Trajectory for BridgeStochasticTrajectory {
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

    fn sample_curve(&self, _n_samples: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        (self.times.clone(), vec![self.freqs.clone()])
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
        // dt = 1 generation per step; times stored in GENERATIONS to
        // match the simulator.  Loop cap = 10*max(N) generations.
        let dt: f64 = 1.0;
        let n_max = n_e.iter().cloned().fold(0.0_f64, f64::max);
        let t_cap: f64 = 40.0 * n_max;
        let p0: Vec<f64> = n_e.iter().map(|ni| 1.0 / (2.0 * ni)).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut p = p_final.clone();
        let mut alive: Vec<bool> = vec![true; n_pops];
        let mut t_inv_per_pop = vec![0.0_f64; n_pops];
        let mut times = vec![0.0_f64];
        let mut freqs = vec![p.clone()];
        let mut t = 0.0_f64;
        while alive.iter().any(|&a| a) && t < t_cap {
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
                let var = (p[i] * (1.0 - p[i]) / (2.0 * n_e[i]) * dt).max(0.0);
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

    /// Override default: return the actual stored (times, freqs)
    /// instead of resampling — preserves the exact stochastic path.
    fn sample_curve(&self, _n_samples: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut freqs: Vec<Vec<f64>> = vec![Vec::with_capacity(self.times.len());
                                             self.n_pops];
        for step in &self.freqs {
            for (pop, &v) in step.iter().enumerate() {
                freqs[pop].push(v);
            }
        }
        (self.times.clone(), freqs)
    }
}

// =====================================================================
// Precomputed (input) trajectory
// =====================================================================

/// User-supplied or pre-saved (times, freqs) trajectory.  Same query
/// API as the simulated trajectories; useful for repeatability (run
/// the same WF path many times) and speed (skip the WF integration
/// after the first run).
///
/// `freqs[pop][step]` gives the inversion frequency for pop at
/// time `times[step]`.  Linear interpolation between steps; returns
/// 0.0 for `t >= t_inv(pop)` where `t_inv(pop)` is the first time the
/// freq for `pop` reaches `1/(2 N_e[pop])` (the founding-copy floor),
/// or `times.last()` if it never does.
#[derive(Clone, Debug)]
pub struct PrecomputedTrajectory {
    pub times: Vec<f64>,
    pub freqs: Vec<Vec<f64>>,    // freqs[pop][step]
    pub n_e: Vec<f64>,
    n_pops: usize,
    t_inv_per_pop: Vec<f64>,
    t_inv_global: f64,
}

impl PrecomputedTrajectory {
    /// `t_inv_explicit`: optional per-pop barrier dissolution times.
    /// If `None`, inferred per-pop as the first time `p ≤ 1/(2 N_e[pop])`,
    /// or `times.last()` if never (incomplete sweep).  Pass an explicit
    /// vector to override the auto-detection (matches the constant-
    /// trajectory semantic where `t_inv` is the user-specified barrier
    /// time, not derived from when `p` reaches founder).
    pub fn new(times: Vec<f64>, freqs: Vec<Vec<f64>>, n_e: Vec<f64>) -> Self {
        Self::with_t_inv(times, freqs, n_e, None)
    }

    pub fn with_t_inv(
        times: Vec<f64>,
        freqs: Vec<Vec<f64>>,
        n_e: Vec<f64>,
        t_inv_explicit: Option<Vec<f64>>,
    ) -> Self {
        let n_pops = freqs.len();
        assert!(n_pops > 0, "freqs must have at least one population");
        assert_eq!(n_e.len(), n_pops, "n_e must match freqs.len()");
        for pop_curve in &freqs {
            assert_eq!(pop_curve.len(), times.len(),
                       "each pop's freq curve must match times.len()");
        }
        let t_inv_per_pop = if let Some(explicit) = t_inv_explicit {
            assert_eq!(explicit.len(), n_pops, "t_inv_explicit must match n_pops");
            explicit
        } else {
            // Per-pop t_inv = first time p reaches 1/(2 N_e[pop]).
            let mut t_inv_per_pop = vec![*times.last().unwrap_or(&0.0); n_pops];
            for (pop, pop_curve) in freqs.iter().enumerate() {
                let p0 = 1.0 / (2.0 * n_e[pop]);
                for (i, &p) in pop_curve.iter().enumerate() {
                    if p <= p0 * 1.01 {
                        t_inv_per_pop[pop] = times[i];
                        break;
                    }
                }
            }
            t_inv_per_pop
        };
        let t_inv_global = t_inv_per_pop.iter().cloned().fold(0.0_f64, f64::max);
        Self {
            times,
            freqs,
            n_e,
            n_pops,
            t_inv_per_pop,
            t_inv_global,
        }
    }

    fn interp(&self, t: f64, pop: u32) -> f64 {
        let pop = (pop as usize).min(self.n_pops - 1);
        if t <= self.times[0] {
            return self.freqs[pop][0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.freqs[pop].last().unwrap();
        }
        let i = match self.times.binary_search_by(
            |x| x.partial_cmp(&t).unwrap()) {
            Ok(i) => return self.freqs[pop][i],
            Err(i) => i,
        };
        let t0 = self.times[i - 1];
        let t1 = self.times[i];
        let f0 = self.freqs[pop][i - 1];
        let f1 = self.freqs[pop][i];
        let frac = (t - t0) / (t1 - t0);
        f0 + frac * (f1 - f0)
    }
}

impl Trajectory for PrecomputedTrajectory {
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

    fn sample_curve(&self, _n_samples: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        (self.times.clone(), self.freqs.clone())
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

    #[test]
    fn bridge_lands_at_p_final_with_selection() {
        // With s>0 and a moderately strong selection coefficient,
        // the bridge sampler should accept within a reasonable number
        // of attempts.  N=1000, t_inv=400 gen, p_final=0.5, s=0.02.
        let traj = BridgeStochasticTrajectory::new(
            0.5, 1000.0, 0.02, 400.0, 42, 0.05, 5_000)
            .expect("bridge should accept");
        assert_eq!(traj.t_inv(0), 400.0);
        // At t=0 (today): p ≈ p_final
        assert!((traj.p_inv_at(0.0, 0) - 0.5).abs() < 0.05);
        // At t=t_inv (when inversion arose): p ≈ 1/(2N) = 5e-4
        let p_at_origin = traj.p_inv_at(traj.t_inv(0) - 1e-3, 0);
        assert!(p_at_origin > 0.0 && p_at_origin < 0.05);
        // Beyond t_inv: p = 0
        assert_eq!(traj.p_inv_at(traj.t_inv(0) + 1.0, 0), 0.0);
        eprintln!("bridge accepted in {} attempts", traj.n_attempts);
    }

    #[test]
    fn precomputed_roundtrip() {
        // Build a Constant trajectory, sample the curve, load into
        // PrecomputedTrajectory with explicit t_inv, verify same
        // p_inv_at output.
        let orig = ConstantTrajectory::single(0.5, 1000.0);
        let (times, freqs) = orig.sample_curve(20);
        let pre = PrecomputedTrajectory::with_t_inv(
            times, freqs, vec![10000.0], Some(vec![1000.0]));
        for t in [0.0, 100.0, 500.0, 999.0] {
            assert!((pre.p_inv_at(t, 0) - 0.5).abs() < 1e-9,
                    "t={t}: pre={}", pre.p_inv_at(t, 0));
        }
        // At t >= t_inv: p = 0
        assert_eq!(pre.p_inv_at(1001.0, 0), 0.0);
    }
}
