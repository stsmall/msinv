/// Peischl et al. (2013) phi(x) gene-flux profile.
///
/// phi(x) = min(x, 1-x, w) / (1 - w)
///
/// where x is the inversion-relative position (0 to 1) and w is the
/// gene-conversion tract width as a fraction of inversion length.
///
/// Shape: triangular roof — zero at breakpoints, flat peak in the
/// middle. Gene flux concentrates in the centre of the inversion.

/// phi(x) at a single position x in [0, 1].
#[inline]
pub fn phi(x: f64, w: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    if w >= 1.0 {
        return 1.0;
    }
    let val = x.min(1.0 - x).min(w) / (1.0 - w);
    val.clamp(0.0, 1.0)
}

/// Integrate phi over [a, b] in inversion-relative coordinates.
///
/// The triangular-roof shape has three pieces:
///   - rising: x in [0, w] → phi = x / (1 - w)
///   - flat:   x in [w, 1-w] → phi = w / (1 - w)
///   - falling: x in [1-w, 1] → phi = (1 - x) / (1 - w)
pub fn phi_integral(a: f64, b: f64, w: f64) -> f64 {
    if w >= 1.0 {
        return (b.min(1.0) - a.max(0.0)).max(0.0);
    }
    let a = a.max(0.0);
    let b = b.min(1.0);
    if b <= a {
        return 0.0;
    }
    let one_minus_w = 1.0 - w;
    let peak = w / one_minus_w;

    // Helper: antiderivative of the rising part (x / (1-w))
    let rising = |x: f64| -> f64 { x * x / (2.0 * one_minus_w) };
    // Helper: antiderivative of the flat part (peak)
    let flat = |x: f64| -> f64 { peak * x };
    // Helper: antiderivative of the falling part ((1-x) / (1-w))
    let falling = |x: f64| -> f64 { (x - x * x / 2.0) / one_minus_w };

    let mut total = 0.0;

    // Rising piece: [a, b] ∩ [0, w]
    let r_lo = a.max(0.0);
    let r_hi = b.min(w);
    if r_hi > r_lo {
        total += rising(r_hi) - rising(r_lo);
    }

    // Flat piece: [a, b] ∩ [w, 1-w]
    let f_lo = a.max(w);
    let f_hi = b.min(1.0 - w);
    if f_hi > f_lo {
        total += flat(f_hi) - flat(f_lo);
    }

    // Falling piece: [a, b] ∩ [1-w, 1]
    let d_lo = a.max(1.0 - w);
    let d_hi = b.min(1.0);
    if d_hi > d_lo {
        total += falling(d_hi) - falling(d_lo);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_at_breakpoints_is_zero() {
        assert_eq!(phi(0.0, 0.05), 0.0);
        assert_eq!(phi(1.0, 0.05), 0.0);
    }

    #[test]
    fn phi_peak_in_centre() {
        let w = 0.05;
        let peak = w / (1.0 - w);
        // At x = 0.5 (well inside [w, 1-w]), phi should equal peak.
        assert!((phi(0.5, w) - peak).abs() < 1e-12);
    }

    #[test]
    fn phi_symmetric() {
        let w = 0.10;
        assert!((phi(0.3, w) - phi(0.7, w)).abs() < 1e-12);
    }

    #[test]
    fn integral_full_range_matches_average() {
        // Average of phi over [0, 1] for triangular roof should be
        // approximately w * peak + (1 - 2w) * peak = ... well,
        // let's just check it's positive and reasonable.
        let w = 0.05;
        let integral = phi_integral(0.0, 1.0, w);
        let peak = w / (1.0 - w);
        // The area under the triangular roof:
        // two triangles of width w, height peak → area = 2 * w * peak / 2 = w * peak
        // one rectangle of width (1-2w), height peak → area = (1-2w) * peak
        // total = peak * (1 - w)
        let expected = peak * (1.0 - w);
        assert!((integral - expected).abs() < 1e-10,
            "integral={integral}, expected={expected}");
    }

    #[test]
    fn integral_centre_higher_than_edges() {
        let w = 0.10;
        let centre = phi_integral(0.4, 0.6, w);
        let edge = phi_integral(0.0, 0.2, w);
        assert!(centre > edge,
            "centre integral ({centre}) should exceed edge ({edge})");
    }
}
