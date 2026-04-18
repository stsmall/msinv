/// InversionSpec: parameters for one chromosomal inversion.
///
/// `p_inv` is per-population: `p_inv[pop]` gives the inverted-arrangement
/// frequency in population `pop`.  When a lineage's population has no
/// entry (index out of bounds), falls back to `p_inv[0]`.

#[derive(Clone, Debug)]
pub struct InversionSpec {
    pub bp_left: f64,
    pub bp_right: f64,
    /// Per-population inverted-arrangement frequency.
    pub p_inv: Vec<f64>,
    pub t_inv: f64,
    pub gene_conversion_rate: f64,
    pub flux_window: f64,
    pub inv_id: u16,
}

impl InversionSpec {
    pub fn new(bp_left: f64, bp_right: f64, p_inv: Vec<f64>, t_inv: f64) -> Self {
        Self {
            bp_left,
            bp_right,
            p_inv,
            t_inv,
            gene_conversion_rate: 1e-9,
            flux_window: 0.05,
            inv_id: 0,
        }
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.bp_right - self.bp_left
    }

    /// Inverted-arrangement frequency for `pop`.
    #[inline]
    pub fn p_inv_for(&self, pop: u32) -> f64 {
        self.p_inv.get(pop as usize).copied().unwrap_or(self.p_inv[0])
    }

    /// Standard-arrangement frequency for `pop`.
    #[inline]
    pub fn p_std_for(&self, pop: u32) -> f64 {
        1.0 - self.p_inv_for(pop)
    }

    // Legacy single-value accessors — use the first entry.
    #[inline]
    pub fn p_inv_default(&self) -> f64 {
        self.p_inv[0]
    }

    #[inline]
    pub fn p_std(&self) -> f64 {
        1.0 - self.p_inv[0]
    }

    /// Set `p_inv` for a specific population.  Grows the vector if needed.
    pub fn set_p_inv_for(&mut self, pop: u32, val: f64) {
        let idx = pop as usize;
        if idx >= self.p_inv.len() {
            self.p_inv.resize(idx + 1, self.p_inv[0]);
        }
        self.p_inv[idx] = val;
    }
}
