/// Sweep: forced-coalescence event modelling a selective sweep.
///
/// ⚠️ APPROXIMATION-ONLY MODEL — NOT a frequency trajectory simulator.
/// All three modes below are *outcome-conditioning* coalescent operators
/// (Hudson-Kaplan family): they apply an instantaneous forced-coalescence
/// at `t_event` rather than simulating the beneficial allele's frequency
/// trajectory through time. For the discoal-style stochastic-then-
/// deterministic trajectory model, see (TODO: port from legacy SMC engine
/// at git `eb47504~1`, classes `StochasticTrajectory` /
/// `DeterministicTrajectory`). Validation tests pending: at what (sweep
/// age, selection coefficient, time-of-fixation) regimes does this
/// approximation match a true-trajectory simulation? See project memory
/// `feedback_no_silent_reverts.md`.
///
/// Three modes:
///
/// 1. **Window mode** (`selection_coefficient == 0`): all lineages
///    with material in `[x_sel - sweep_window, x_sel + sweep_window]`
///    are deterministically coalesced (Hudson-Kaplan approximation).
///
/// 2. **Hard sweep / hitchhiking mode** (`selection_coefficient > 0`,
///    `starting_frequency == 0`): each lineage's inclusion is
///    probabilistic, decaying with recombination distance from x_sel.
///    All swept lineages coalesce to a single ancestor.
///
/// 3. **Soft sweep from standing variation** (`selection_coefficient > 0`,
///    `starting_frequency > 0`): hitchhiking mode, but swept lineages
///    are randomly partitioned among K ≈ 1/f0 "founding copies" of the
///    beneficial allele (discoal model). Within each group lineages
///    coalesce; the K surviving ancestors continue at the normal
///    coalescent rate. Produces partial diversity reduction at x_sel.

use crate::class_tag::{BranchClass, Karyotype};

#[derive(Clone, Debug)]
pub struct Sweep {
    /// Genomic position of the selected site.
    pub x_sel: f64,
    /// Time (generations backward) of the sweep MRCA.
    pub t_event: f64,
    /// Target inversion + karyotype. None = any class ("hard sweep on
    /// all carriers"). Some((inv_id, kary)) = only lineages that are
    /// `kary` at inversion `inv_id` at position `x_sel`.
    pub target: Option<(u16, Karyotype)>,
    /// Restrict to lineages in this population (None = any).
    pub population: Option<u32>,
    /// Half-width of the sweep window (bp). The force-coalescence
    /// applies to [x_sel - window, x_sel + window].
    pub sweep_window: f64,
    /// Selection coefficient for the swept allele. When > 0, enables
    /// hitchhiking mode: inclusion probability decays with recombination
    /// distance from x_sel. Default 0 (window mode).
    pub selection_coefficient: f64,
    /// Starting frequency of the beneficial allele (standing variation).
    /// When > 0, enables soft-sweep mode: swept lineages are partitioned
    /// among K ≈ 1/f0 founding copies instead of coalescing to 1.
    /// 0.0 = hard sweep (single origin). Must be in [0, 1).
    pub starting_frequency: f64,
}

impl Default for Sweep {
    fn default() -> Self {
        Self {
            x_sel: 0.0,
            t_event: 0.0,
            target: None,
            population: None,
            sweep_window: 0.0,
            selection_coefficient: 0.0,
            starting_frequency: 0.0,
        }
    }
}

impl Sweep {
    /// None ≡ any class; Some((inv, kary)) requires exact karyotype
    /// at `inv` (panmictic fails).
    pub fn class_matches(&self, cls: BranchClass) -> bool {
        match self.target {
            None => true,
            Some((inv_id, kary)) => cls.get_inv(inv_id) == Some(kary),
        }
    }

    /// Number of founding copies for soft sweep partitioning.
    /// Returns 1 for hard sweeps (starting_frequency == 0).
    pub fn num_founders(&self) -> usize {
        if self.starting_frequency <= 0.0 {
            return 1;
        }
        (1.0 / self.starting_frequency).round().max(1.0) as usize
    }

    /// Hitchhiking probability that position `x` is linked to the sweep.
    /// Requires `selection_coefficient > 0` and `recomb_rate > 0`.
    /// `ne` is the effective population size for sweep duration.
    pub fn hitchhiking_probability(&self, x: f64, recomb_rate: f64, ne: f64) -> f64 {
        let s = self.selection_coefficient;
        if s <= 0.0 || recomb_rate <= 0.0 {
            return 1.0;
        }
        let t_dur = ((2.0 * ne * s).max(2.0)).ln() / s;
        let dist = (x - self.x_sel).abs();
        (-recomb_rate * dist * t_dur).exp()
    }
}
