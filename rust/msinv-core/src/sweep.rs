/// Sweep: forced-coalescence event modelling a selective sweep.

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
}

impl Sweep {
    /// Check if a BranchClass at x_sel qualifies for this sweep.
    pub fn class_matches(&self, cls: BranchClass) -> bool {
        match self.target {
            None => true, // 'any'
            Some((inv_id, kary)) => {
                match cls.get_inv(inv_id) {
                    Some(k) => k == kary,
                    None => true, // panmictic at this inv → qualifies
                }
            }
        }
    }
}
