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

    /// Backward-time upper bound of the sweep window. Reads from the
    /// trajectory's `t_de_novo` if a trajectory has been built; falls
    /// back to `joint.t_origin` otherwise (matches the pre-SV-extension
    /// behavior).
    pub fn t_de_novo(&self) -> f64 {
        self.trajectory.as_ref()
            .map(|t| t.t_de_novo)
            .unwrap_or(self.joint.t_origin)
    }

    /// Is `t` inside the sweep window? Spans the selection phase
    /// (`[tau, t_origin]`) plus the standing-variation phase
    /// (`(t_origin, t_de_novo]`) when one exists.
    pub fn covers(&self, t: f64) -> bool {
        t >= self.tau && t <= self.t_de_novo()
    }

    /// Sweep-aware effective Ne for a (pop, kary) cell at backward
    /// time `t`. If `t` is inside the sweep window AND the trajectory
    /// is built, returns `n_pop_t * trajectory.p_kary(t, pop, kary)`.
    /// Otherwise returns `n_pop_t * fallback_p_kary` — the caller's
    /// pre-sweep p_kary value (typically from the inversion trajectory).
    pub fn ne_cell_or_fallback(
        &self,
        t: f64,
        pop: u32,
        kary: Karyotype,
        n_pop_t: f64,
        fallback_p_kary: f64,
    ) -> f64 {
        if self.covers(t) {
            if let Some(traj) = &self.trajectory {
                return n_pop_t * traj.p_kary(t, pop, kary);
            }
        }
        n_pop_t * fallback_p_kary
    }

    /// Probability that a lineage at position `x` is linked to the
    /// sweep MRCA, given recombination rate `r`. Approximation:
    /// `exp(-r·d·T_eff)` where `T_eff = t_origin - tau` (full sweep
    /// duration). The proper integral over the trajectory shape is
    /// a TODO refinement.
    pub fn hitchhiking_prob(&self, x: f64, recomb_rate: f64) -> f64 {
        if self.trajectory.is_none() {
            return 1.0;
        }
        let d = (x - self.x_sel).abs();
        let t_eff = self.joint.t_origin - self.tau;
        (-recomb_rate * d * t_eff).exp()
    }

    /// Per-segment hitchhiking probability: for an ancestral segment
    /// `[seg_left, seg_right)`, the probability that NO recombination
    /// has occurred between `x_sel` and the closest edge of the segment
    /// during the sweep window.  d_min = 0 if the segment spans `x_sel`,
    /// else the distance from `x_sel` to the nearest edge.
    pub fn p_hh_for_segment(
        &self, seg_left: f64, seg_right: f64, recomb_rate: f64,
    ) -> f64 {
        if self.trajectory.is_none() {
            return 1.0;
        }
        let d_min = if self.x_sel >= seg_left && self.x_sel < seg_right {
            0.0
        } else if seg_right <= self.x_sel {
            self.x_sel - seg_right
        } else {
            seg_left - self.x_sel
        };
        let t_eff = self.joint.t_origin - self.tau;
        (-recomb_rate * d_min * t_eff).exp()
    }

    /// Nearest-segment distance: walks a lineage's segment chain and
    /// returns the minimum distance from any segment to `x_sel`.
    /// Returns `f64::INFINITY` for an empty chain (SEG_NIL head).
    pub fn lineage_nearest_distance(
        &self,
        head: crate::segment::SegIdx,
        arena: &crate::segment::SegmentArena,
    ) -> f64 {
        use crate::segment::SEG_NIL;
        let mut cur = head;
        let mut best = f64::INFINITY;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            let d = if self.x_sel >= seg.left && self.x_sel < seg.right {
                0.0
            } else if seg.right <= self.x_sel {
                self.x_sel - seg.right
            } else {
                seg.left - self.x_sel
            };
            if d < best { best = d; }
            if d == 0.0 { return 0.0; }
            cur = seg.next;
        }
        best
    }

    /// At sample time τ, randomly assign a lineage to the swept (A) vs
    /// unswept (a) fraction with probability equal to the trajectory's
    /// per-(pop, kary) A frequency. Returns true for A.
    pub fn assign_a_at_sample<R: rand::Rng>(
        &self,
        pop: u32,
        kary: Karyotype,
        rng: &mut R,
    ) -> bool {
        let p_a = match &self.trajectory {
            Some(t) => t.p_allele_given_kary(self.tau, pop, kary),
            None => return false,
        };
        rng.random::<f64>() < p_a
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

    #[test]
    fn ne_cell_rises_during_sweep_on_inverted() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::I, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 100.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.3], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        // Outside the sweep window: trajectory query returns the closest sample (extrapolation).
        // Strictly: pre-sweep-start is t > t_origin = 100, so should hold p_inv at initial 0.3.
        // Inside window (t = 50): I-class should have grown via selection.
        let traj = sw.trajectory.as_ref().unwrap();
        let p_i_old = traj.p_kary(99.0, 0, Karyotype::I);
        let p_i_mid = traj.p_kary(50.0, 0, Karyotype::I);
        assert!(p_i_mid > p_i_old, "Inverted Ne should rise during sweep on I; old={p_i_old}, mid={p_i_mid}");
        // ne_cell scales as p_kary * N_pop
        let ne_mid = sw.trajectory.as_ref().unwrap().ne_cell(50.0, 0, Karyotype::I, 10_000.0);
        assert!((ne_mid - 10_000.0 * p_i_mid).abs() < 1e-6);
    }

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
        // T_eff = t_origin - tau = 500. To straddle p = 0.5 we need
        //   r * d * T_eff ≈ ln(2) ≈ 0.69
        // Pick recomb_rate = 1e-5 so the threshold distance is
        //   d* = ln(2) / (r * T_eff) = 0.69 / (1e-5 * 500) ≈ 138.6 bp.
        // Original task spec used r = 1e-3 which is mathematically
        // inconsistent with `p_near > 0.5` (would give exp(-5) ≈ 0.007).
        let p_near = sw.hitchhiking_prob(5_010.0, 1e-5);
        let p_far  = sw.hitchhiking_prob(5_500.0, 1e-5);
        assert!(p_near > p_far, "expected hitchhiking decay; near={p_near}, far={p_far}");
        assert!(p_near > 0.5, "p_near={p_near}");
        assert!(p_far  < 0.5, "p_far={p_far}");
    }

    #[test]
    fn p_hh_for_segment_zero_distance_at_x_sel() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 500.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.0], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        // Segment spans x_sel: d=0, p=1
        let p_at = sw.p_hh_for_segment(4_900.0, 5_100.0, 1e-5);
        assert!((p_at - 1.0).abs() < 1e-9);
        // Segment to the right of x_sel
        let p_right = sw.p_hh_for_segment(5_138.6, 5_500.0, 1e-5);
        // d=138.6, T_eff=500, exp(-1e-5*138.6*500) = exp(-0.693) ≈ 0.5
        assert!(p_right > 0.45 && p_right < 0.55,
            "expected ~0.5, got {p_right}");
        // Segment to the left of x_sel: same distance, same prob
        let p_left = sw.p_hh_for_segment(4_500.0, 4_861.4, 1e-5);
        assert!(p_left > 0.45 && p_left < 0.55,
            "expected ~0.5, got {p_left}");
    }

    #[test]
    fn p_hh_for_segment_no_trajectory_returns_one() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0, JointSweepSpec::default());
        // No trajectory built ⇒ degenerate, returns 1.0
        assert_eq!(sw.p_hh_for_segment(0.0, 100.0, 1e-3), 1.0);
    }

    #[test]
    fn assign_a_at_sample_uses_trajectory_freq() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 500.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.0], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        use rand_xoshiro::Xoshiro256PlusPlus;
        use rand::SeedableRng;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut a_count = 0;
        let n_trials = 1000;
        for _ in 0..n_trials {
            if sw.assign_a_at_sample(0, Karyotype::S, &mut rng) {
                a_count += 1;
            }
        }
        let observed = a_count as f64 / n_trials as f64;
        let traj = sw.trajectory.as_ref().unwrap();
        let expected = traj.p_allele_given_kary(0.0, 0, Karyotype::S);
        // Within 3 sigma of binomial
        let sigma = (expected * (1.0 - expected) / n_trials as f64).sqrt();
        assert!((observed - expected).abs() < 3.0 * sigma,
            "observed={observed}, expected={expected}, sigma={sigma}");
    }

    #[test]
    fn ne_cell_or_fallback_uses_trajectory_inside_window_only() {
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::I, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 100.0, f0: 0.001,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            },
        ).with_trajectory(1, &[0.3], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
        // Outside window (t > t_origin): falls back
        let ne_pre = sw.ne_cell_or_fallback(200.0, 0, Karyotype::I, 10_000.0, 0.3);
        assert!((ne_pre - 3_000.0).abs() < 1e-6);
        // Inside window: uses trajectory
        let traj_p = sw.trajectory.as_ref().unwrap().p_kary(50.0, 0, Karyotype::I);
        let ne_mid = sw.ne_cell_or_fallback(50.0, 0, Karyotype::I, 10_000.0, 0.3);
        assert!((ne_mid - 10_000.0 * traj_p).abs() < 1e-6);
        assert!(ne_mid > ne_pre, "expected ne to rise during sweep; pre={ne_pre}, mid={ne_mid}");
    }

    #[test]
    fn lineage_nearest_distance_walks_chain() {
        use crate::segment::{SegmentArena, SEG_NIL};
        use crate::class_tag::BranchClass;
        let sw = Sweep::new(
            5_000.0, 0.0, 0, Karyotype::S, 0, JointSweepSpec::default());
        let mut arena = SegmentArena::new();
        // Empty chain: infinity
        assert_eq!(sw.lineage_nearest_distance(SEG_NIL, &arena), f64::INFINITY);
        // Single segment far from x_sel
        let s1 = arena.alloc(7_000.0, 8_000.0, 0, BranchClass::PANMICTIC);
        assert_eq!(sw.lineage_nearest_distance(s1, &arena), 2_000.0);
        // Build chain: [7000,8000) -> [4500,4900) (closer to x_sel=5000)
        let s2 = arena.alloc(4_500.0, 4_900.0, 0, BranchClass::PANMICTIC);
        arena.get_mut(s1).next = s2;
        assert_eq!(sw.lineage_nearest_distance(s1, &arena), 100.0);
        // Add a segment that spans x_sel: d=0, returns immediately
        let s3 = arena.alloc(4_950.0, 5_050.0, 0, BranchClass::PANMICTIC);
        arena.get_mut(s2).next = s3;
        assert_eq!(sw.lineage_nearest_distance(s1, &arena), 0.0);
    }
}
