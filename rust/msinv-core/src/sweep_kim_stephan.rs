//! Kim-Stephan closed-form anchors for sweep validation.
//!
//! Test-only — used to assert that `sweep_trajectory` outputs match
//! analytical predictions within Tier-1 (25% relative) tolerance.
//! Same role Andolfatto closed-form plays for flux validation.

/// Sojourn time of a sweep from f0 = 1/(2Ne) to fixation:
/// T_fix ≈ (2/s) · ln(2·Ne)
pub fn sojourn_time(s: f64, ne: f64) -> f64 {
    if s <= 0.0 || ne <= 1.0 {
        return f64::INFINITY;
    }
    (2.0 / s) * (2.0 * ne).ln()
}

/// Fixation probability of a single de novo beneficial allele:
/// P_fix ≈ 2s / (1 + s) for small s
pub fn fixation_probability(s: f64) -> f64 {
    if s <= 0.0 { return 0.0; }
    2.0 * s / (1.0 + s)
}

/// Hitchhiking probability that a neutral site at recombination distance
/// `r·d` from x_sel escapes the sweep, in the Kim-Stephan framework:
/// P_escape ≈ 1 - exp(-r·d·T_fix) is the prob the link is broken.
pub fn hitchhiking_escape_probability(rho_d: f64, t_fix: f64) -> f64 {
    1.0 - (-rho_d * t_fix).exp()
}

/// Pi reduction at distance d: pi_obs / pi_neutral ≈ 1 - exp(-r·d·T_fix).
pub fn pi_reduction_factor(s: f64, ne: f64, recomb: f64, d: f64) -> f64 {
    let t_fix = sojourn_time(s, ne);
    hitchhiking_escape_probability(recomb * d, t_fix)
}

/// Flux mixing time for an A-bearing lineage of one karyotype to
/// reach the other karyotype via gene conversion:
/// T_mix ≈ 1 / (γ · L_tract)
pub fn flux_mixing_time(gamma: f64, mean_tract_length: f64) -> f64 {
    if gamma <= 0.0 || mean_tract_length <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (gamma * mean_tract_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sojourn_time_grows_with_ne() {
        let t1 = sojourn_time(0.01, 1e3);
        let t2 = sojourn_time(0.01, 1e6);
        assert!(t2 > t1);
    }

    #[test]
    fn fixation_probability_haldane() {
        assert!((fixation_probability(0.01) - 0.0198).abs() < 1e-3);
    }

    #[test]
    fn pi_reduction_zero_at_x_sel() {
        // r·d = 0 -> 1 - exp(0) = 0 -> full reduction
        assert_eq!(pi_reduction_factor(0.01, 1e4, 1e-8, 0.0), 0.0);
    }
}
