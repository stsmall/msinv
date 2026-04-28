//! Sweep: a forced-coalescence event driven by a joint forward-time
//! Wright-Fisher trajectory over (karyotype × allele × population)
//! haplotype classes. See `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`.
//!
//! Replaces the prior Hudson-Kaplan endpoint-only operator. The
//! trajectory is computed at sweep construction time
//! (`Sweep::with_trajectory(...)`) and consumed by the backward-time
//! coalescent event loop in subsequent tasks (Task 12+).

use crate::class_tag::Karyotype;
use crate::sweep_trajectory::{
    build_joint_trajectory, JointSweepSpec, JointSweepTrajectory,
};

#[allow(unused_imports)]
use crate::sweep_trajectory::SweepMode;

#[derive(Clone, Debug)]
pub struct Sweep {
    pub x_sel: f64,
    pub tau: f64,
    pub origin_pop: u32,
    pub origin_kary: Karyotype,
    pub target_inv: u16,
    pub joint: JointSweepSpec,
    /// Pre-computed trajectory; populated by `with_trajectory`.
    pub trajectory: Option<JointSweepTrajectory>,
}

impl Sweep {
    /// Construct a new Sweep without a trajectory (call `with_trajectory` to populate).
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
