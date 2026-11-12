//! Sweep: a forced-coalescence event driven by a joint forward-time
//! Wright-Fisher trajectory over (karyotype × allele × population)
//! haplotype classes. See `docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`.
//!
//! Replaces the prior Hudson-Kaplan endpoint-only operator. The
//! trajectory is computed at sweep construction time
//! (`Sweep::with_trajectory(...)`) and consumed by the backward-time
//! coalescent event loop in subsequent tasks (Task 12+).

use crate::class_tag::Karyotype;
use crate::inversion::InversionSpec;
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
    /// Crossover-suppression exposure over the selection phase, per
    /// inversion: `(inv_id, ∫ w_S dt, ∫ w_I dt)` over
    /// `[tau, supp_t]`, where `w_k(t)` is the arrangement frequency
    /// `p_k(t)` in `origin_pop` while the inversion exists and 1 after
    /// `t_inv`. Accumulated piecewise at every deterministic boundary
    /// (`accumulate_suppression`), so frequency changes from demographic
    /// events are integrated at their own times.
    pub supp_int: Vec<(u16, f64, f64)>,
    /// Backward time up to which `supp_int` has been accumulated.
    pub supp_t: f64,
    /// True when the selected site lies inside inversion `target_inv`:
    /// the allele is carried on an arrangement, so its trajectory and
    /// the inversion frequency are coupled. False for a sweep outside
    /// every configured inversion (or with no matching inversion): an
    /// ordinary discoal-style sweep, independent of arrangement. Set by
    /// `populate_sweep_trajectories`; `new` defaults to true.
    pub tied: bool,
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
        Self {
            x_sel, tau, origin_pop, origin_kary, target_inv, joint,
            trajectory: None, supp_int: Vec::new(), supp_t: tau,
            tied: true,
        }
    }

    /// Integrate the crossover-suppression weights from `supp_t` up to
    /// `min(t_to, t_origin)` (midpoint rule, steps of at most one
    /// generation) using each inversion's current frequency model.
    /// Call before any event at `t_to` changes an inversion frequency.
    pub fn accumulate_suppression(&mut self, t_to: f64, inversions: &[InversionSpec]) {
        let lo = self.supp_t;
        let hi = t_to.min(self.joint.t_origin);
        if hi <= lo {
            return;
        }
        let n = ((hi - lo).ceil() as usize).clamp(1, 1_000_000);
        let h = (hi - lo) / n as f64;
        for inv in inversions {
            let t_inv = inv.t_inv_max();
            let (mut int_s, mut int_i) = (0.0, 0.0);
            for j in 0..n {
                let tm = lo + (j as f64 + 0.5) * h;
                if tm >= t_inv {
                    int_s += h;
                    int_i += h;
                } else {
                    let p = inv.p_inv_at(tm, self.origin_pop);
                    int_i += p * h;
                    int_s += (1.0 - p) * h;
                }
            }
            match self.supp_int.iter_mut().find(|e| e.0 == inv.inv_id) {
                Some(e) => { e.1 += int_s; e.2 += int_i; }
                None => self.supp_int.push((inv.inv_id, int_s, int_i)),
            }
        }
        self.supp_t = hi;
    }

    /// Crossover exposure `∫∫ w(x, t) dx dt` (bp × generations) between
    /// `x_sel` and the nearest edge of `[seg_left, seg_right)` over the
    /// selection phase. `w = 1` outside inversions; inside inversion `k`
    /// it is the accumulated `p_k` of the lineage's arrangement there
    /// (`kary_at(inv_id)`), or of `origin_kary` at `target_inv` when the
    /// lineage carries no tagged material there (it is linked to the A
    /// allele, which arose on `origin_kary`), and 1 at other untagged
    /// inversions, as in the main-loop thinning. Where inversions nest,
    /// the time-averaged weights are multiplied (exact when at most one
    /// of them has a time-varying frequency).
    /// With no inversions this is `d_min · (t_origin - tau)`, the
    /// exposure in `p_hh_for_segment`.
    pub fn escape_exposure(
        &self,
        seg_left: f64,
        seg_right: f64,
        inversions: &[InversionSpec],
        kary_at: &dyn Fn(u16) -> Option<Karyotype>,
    ) -> f64 {
        let fallback = if self.tied { Some(self.origin_kary) } else { None };
        self.escape_exposure_with(seg_left, seg_right, inversions, kary_at, fallback)
    }

    /// Probability that segment `[seg_left, seg_right)` of an A lineage
    /// escapes the sweep by recombination: `exp(-r * exposure)`. For a
    /// tied sweep and a lineage with no tagged material at `target_inv`,
    /// the arrangement it carries there is unknown; it is I with
    /// probability `P(I | A)` at `tau` (1 or 0 without gene flux, i.e.
    /// `origin_kary`), and the escape probability is the mixture over
    /// the two arrangements.
    pub fn escape_prob(
        &self,
        seg_left: f64,
        seg_right: f64,
        inversions: &[InversionSpec],
        kary_at: &dyn Fn(u16) -> Option<Karyotype>,
        r: f64,
    ) -> f64 {
        let has_target = inversions.iter().any(|inv| inv.inv_id == self.target_inv);
        let mixture = self.tied && has_target && kary_at(self.target_inv).is_none();
        if !mixture {
            return (-r * self.escape_exposure(seg_left, seg_right, inversions, kary_at)).exp();
        }
        let w_i = self.p_kary_given_a_at_tau(Karyotype::I);
        let e_s = self.escape_exposure_with(
            seg_left, seg_right, inversions, kary_at, Some(Karyotype::S));
        let e_i = self.escape_exposure_with(
            seg_left, seg_right, inversions, kary_at, Some(Karyotype::I));
        (1.0 - w_i) * (-r * e_s).exp() + w_i * (-r * e_i).exp()
    }

    /// `P(kary | A)` in `origin_pop` at `tau` from the trajectory;
    /// `origin_kary` with certainty when there is no trajectory or no A.
    pub fn p_kary_given_a_at_tau(&self, kary: Karyotype) -> f64 {
        let certain = if kary == self.origin_kary { 1.0 } else { 0.0 };
        let traj = match &self.trajectory { Some(t) => t, None => return certain };
        let p_a = traj.p_allele_overall(self.tau, self.origin_pop);
        if p_a <= 0.0 {
            return certain;
        }
        traj.p_allele_given_kary(self.tau, self.origin_pop, kary)
            * traj.p_kary(self.tau, self.origin_pop, kary) / p_a
    }

    /// `escape_exposure` with an explicit arrangement for a lineage
    /// untagged at `target_inv` (`None`: weight 1 there, as at any other
    /// untagged inversion).
    pub fn escape_exposure_with(
        &self,
        seg_left: f64,
        seg_right: f64,
        inversions: &[InversionSpec],
        kary_at: &dyn Fn(u16) -> Option<Karyotype>,
        target_fallback: Option<Karyotype>,
    ) -> f64 {
        let (a, b) = if self.x_sel >= seg_left && self.x_sel < seg_right {
            return 0.0;
        } else if seg_right <= self.x_sel {
            (seg_right, self.x_sel)
        } else {
            (self.x_sel, seg_left)
        };
        let t_eff = self.joint.t_origin - self.tau;
        if t_eff <= 0.0 {
            return 0.0;
        }
        let mut cuts: Vec<f64> = vec![a, b];
        for inv in inversions {
            for bp in [inv.bp_left, inv.bp_right] {
                if bp > a && bp < b { cuts.push(bp); }
            }
        }
        cuts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut exposure = 0.0;
        for w in cuts.windows(2) {
            let len = w[1] - w[0];
            if len <= 0.0 { continue; }
            let mid = 0.5 * (w[0] + w[1]);
            let mut factor = 1.0;
            for inv in inversions {
                if mid < inv.bp_left || mid >= inv.bp_right { continue; }
                let kary = kary_at(inv.inv_id).or(
                    if inv.inv_id == self.target_inv { target_fallback } else { None });
                let entry = self.supp_int.iter().find(|e| e.0 == inv.inv_id);
                if let (Some(k), Some(e)) = (kary, entry) {
                    let int_k = match k { Karyotype::S => e.1, Karyotype::I => e.2 };
                    factor *= int_k / t_eff;
                }
            }
            exposure += len * t_eff * factor;
        }
        exposure
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

    /// Does this sweep have a standing-variation phase
    /// (`f0 > 1/(2N)`, i.e. `t_de_novo` strictly past `t_origin`)?
    pub fn has_sv_phase(&self) -> bool {
        self.t_de_novo() > self.joint.t_origin + 1e-9
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
    /// sweep MRCA, given recombination rate `r`. Uses the same
    /// `T_eff = t_origin - tau` convention as `p_hh_for_segment`.
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
        // T_eff = selection phase only (spec
        // `2026-04-30-sweep-per-segment-hitchhiking-design.md`). The
        // SV phase shedding is handled by the in-window per-allele
        // rate divergence, not by this deterministic partition;
        // adding it here double-counts.
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
            Some(t) if self.tied => t.p_allele_given_kary(self.tau, pop, kary),
            Some(t) => t.p_allele_overall(self.tau, pop),
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
