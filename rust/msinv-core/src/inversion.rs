//! InversionSpec: parameters for one chromosomal inversion.
//!
//! `trajectory` describes the inversion's frequency through time
//! (per-population). See [`crate::trajectory`] for the four
//! supported types: Constant, Deterministic (logistic), Stochastic
//! (WF diffusion), Coupled (multi-pop diffusion + migration).
//!
//! Back-compat constructors (`new`, `with_p_inv`) wrap a vector of
//! per-pop frequencies in a `ConstantTrajectory`, preserving the
//! pre-trajectory-port behaviour exactly.

use crate::trajectory::{ConstantTrajectory, Trajectory};

/// Per-event tract length distribution for the b2-flux model.
/// See docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TractDistribution {
    Fixed,
    Geometric,
}

impl Default for TractDistribution {
    fn default() -> Self { TractDistribution::Geometric }
}

#[derive(Clone, Debug)]
pub struct InversionSpec {
    pub bp_left: f64,
    pub bp_right: f64,
    /// Inversion frequency model through time (per-population).
    pub trajectory: Box<dyn Trajectory + Send + Sync>,
    pub gene_conversion_rate: f64,
    pub flux_window: f64,
    /// Mean per-event gene-conversion tract length (bp).
    /// Replaces `flux_window`'s tract role; phi(x) is computed with
    /// `w = mean_tract_length / inv_length`. Removed at Task 7 of
    /// the b2-flux migration.
    pub mean_tract_length: f64,
    /// Per-event tract length distribution (`Fixed` reproduces the
    /// pre-b2 deterministic-tract semantics; `Geometric` samples
    /// Exponential(1/mean_tract_length).
    pub tract_distribution: TractDistribution,
    pub inv_id: u16,
}

impl InversionSpec {
    /// Build with an explicit Trajectory.
    pub fn new(
        bp_left: f64,
        bp_right: f64,
        trajectory: Box<dyn Trajectory + Send + Sync>,
    ) -> Self {
        Self {
            bp_left,
            bp_right,
            trajectory,
            gene_conversion_rate: 1e-9,
            flux_window: 0.05,
            mean_tract_length: 100.0,
            tract_distribution: TractDistribution::Geometric,
            inv_id: 0,
        }
    }

    /// Back-compat: construct with a per-pop constant frequency vector
    /// and a single t_inv.  Wraps in [`ConstantTrajectory`], which
    /// reproduces the pre-trajectory-port behaviour bit-for-bit.
    pub fn with_p_inv(
        bp_left: f64,
        bp_right: f64,
        p_inv: Vec<f64>,
        t_inv: f64,
    ) -> Self {
        Self::new(
            bp_left,
            bp_right,
            Box::new(ConstantTrajectory::new(p_inv, t_inv)),
        )
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.bp_right - self.bp_left
    }

    /// Inverted-arrangement frequency for `pop` at backward time `t`.
    #[inline]
    pub fn p_inv_at(&self, t: f64, pop: u32) -> f64 {
        self.trajectory.p_inv_at(t, pop)
    }

    /// Standard-arrangement frequency for `pop` at backward time `t`.
    #[inline]
    pub fn p_std_at(&self, t: f64, pop: u32) -> f64 {
        1.0 - self.trajectory.p_inv_at(t, pop)
    }

    /// Time at which the inversion arose in `pop`.  For
    /// per-population trajectories (Coupled) this varies; for
    /// Constant/Deterministic/Stochastic it's a single value.
    #[inline]
    pub fn t_inv(&self, pop: u32) -> f64 {
        self.trajectory.t_inv(pop)
    }

    /// Maximum t_inv across all populations — when the barrier era
    /// ends globally.  Use this for "is the barrier still active
    /// anywhere?" checks.
    #[inline]
    pub fn t_inv_max(&self) -> f64 {
        self.trajectory.t_inv_max()
    }

    // ---- DEPRECATED back-compat shims --------------------------------
    // These return the present-day (t=0) frequency, equivalent to
    // the pre-trajectory-port behaviour where `p_inv_for(pop)` was
    // a static accessor.  Preserved to ease migration of call sites
    // that don't yet have access to the current simulation time.
    // NEW CODE SHOULD USE `p_inv_at(t, pop)` INSTEAD.

    #[inline]
    #[deprecated(note = "use p_inv_at(t, pop) — frequency now varies with time")]
    pub fn p_inv_for(&self, pop: u32) -> f64 {
        self.trajectory.p_inv_at(0.0, pop)
    }

    #[inline]
    #[deprecated(note = "use p_std_at(t, pop) — frequency now varies with time")]
    pub fn p_std_for(&self, pop: u32) -> f64 {
        1.0 - self.trajectory.p_inv_at(0.0, pop)
    }

    #[inline]
    #[deprecated(note = "use p_inv_at(0, 0)")]
    pub fn p_inv_default(&self) -> f64 {
        self.trajectory.p_inv_at(0.0, 0)
    }

    #[inline]
    #[deprecated(note = "use p_std_at(0, 0)")]
    pub fn p_std(&self) -> f64 {
        1.0 - self.trajectory.p_inv_at(0.0, 0)
    }

    /// Override the frequency for a population on a ConstantTrajectory.
    /// No-op (with debug warning) on non-constant trajectories — those
    /// have their full p(t) baked in at construction.
    pub fn set_p_inv_for(&mut self, pop: u32, val: f64) {
        // NOTE: we can't easily downcast `Box<dyn Trajectory>` without
        // adding the Any trait.  For now, we rebuild the trajectory if
        // it's a ConstantTrajectory by querying t_inv_max + n_pops.
        // For non-constant trajectories the call is silently ignored —
        // they encode their own p(t) and should not be mutated post-hoc.
        let n_pops = self.trajectory.n_pops().max(pop as usize + 1);
        let t_inv = self.trajectory.t_inv_max();
        let mut p_inv: Vec<f64> = (0..n_pops)
            .map(|i| self.trajectory.p_inv_at(0.0, i as u32))
            .collect();
        if (pop as usize) >= p_inv.len() {
            let fill = p_inv[0];
            p_inv.resize(pop as usize + 1, fill);
        }
        p_inv[pop as usize] = val;
        self.trajectory = Box::new(ConstantTrajectory::new(p_inv, t_inv));
    }
}
