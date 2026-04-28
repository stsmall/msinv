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
