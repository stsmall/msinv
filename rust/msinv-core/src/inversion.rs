/// InversionSpec: parameters for one chromosomal inversion.

#[derive(Clone, Debug)]
pub struct InversionSpec {
    pub bp_left: f64,
    pub bp_right: f64,
    pub p_inv: f64,
    pub t_inv: f64,
    pub gene_conversion_rate: f64,
    pub flux_window: f64,
    pub inv_id: u16,
}

impl InversionSpec {
    pub fn new(bp_left: f64, bp_right: f64, p_inv: f64, t_inv: f64) -> Self {
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

    #[inline]
    pub fn p_std(&self) -> f64 {
        1.0 - self.p_inv
    }
}
