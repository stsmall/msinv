//! Gene-flux events seen by one lineage.
//!
//! Forward model: inside an active inversion, a gene-conversion event
//! starts at a point `y` at rate `gcr * phi(y)` per bp per generation,
//! and copies a tract of length `l` (`mean_tract_length`, fixed or
//! exponential, capped at 0.99 of the inversion) placed uniformly among
//! the positions that contain `y` and fit in the inversion. The donor
//! carries the other arrangement with probability `p_other`.
//!
//! Backward, a lineage is hit by every such tract that overlaps the
//! material it carries, wherever the tract starts. A site `x` it
//! carries is therefore covered at about `gcr * p_other * phi(x) * l`,
//! however little of the material around `x` the lineage still holds.
//!
//! The rate is split into components:
//! - `Seg`: events starting on a carried segment. Every such tract
//!   overlaps the lineage; rate `gcr * p_other * ∫_seg phi`.
//! - `Edge`: events starting in a gap next to a piece of material,
//!   nearer to this edge than to any other piece. Proposed at
//!   `gcr * p_other * phi_max * mean_tract_length` with the start
//!   distance `d` drawn Exp(mean) (geometric tracts) or U(0, mean)
//!   (fixed), then thinned by `phi(y) / phi_max` and by whether the
//!   tract reaches the material. For geometric tracts the length is
//!   drawn as `d + Exp(mean)`, which is the length law given `l > d`
//!   (memoryless), so the thinning is exact.
//!
//! In a gap, `p_other` uses the arrangement of the nearest piece of
//! material. A lineage carrying the whole inversion has no edges.

use rand::Rng;
use rand_distr::{Distribution, Exp};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::Karyotype;
use crate::inversion::{InversionSpec, TractDistribution};
use crate::phi::{phi, phi_integral};

/// Material touching within this distance is one piece.
const TOUCH_EPS: f64 = 1e-9;

/// A carried segment clipped to the inversion, with its arrangement.
pub type ClassedSeg = (f64, f64, Karyotype);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Component {
    Seg { left: f64, right: f64 },
    /// Gap events proposed from `edge` towards `limit` (exclusive).
    Edge { edge: f64, limit: f64 },
}

fn p_other(kary: Karyotype, p_inv: f64) -> f64 {
    match kary {
        Karyotype::S => p_inv,
        Karyotype::I => 1.0 - p_inv,
    }
}

/// Peak of `phi(., w)`.
pub fn phi_max(w: f64) -> f64 {
    if w >= 1.0 {
        1.0
    } else {
        (w.min(0.5) / (1.0 - w)).clamp(0.0, 1.0)
    }
}

/// Components and their rates. `segs` must be sorted by position and
/// clipped to the inversion.
fn components(segs: &[ClassedSeg], inv: &InversionSpec, p_inv: f64) -> Vec<(Component, f64)> {
    let mut out = Vec::new();
    let inv_len = inv.length();
    if inv_len <= 0.0 || inv.gene_conversion_rate <= 0.0 || segs.is_empty() {
        return out;
    }
    let ell = inv.mean_tract_length;
    let w = ell / inv_len;
    let edge_rate_unit = if ell > 0.0 {
        inv.gene_conversion_rate * phi_max(w) * ell
    } else {
        0.0
    };

    for &(l, r, k) in segs {
        let po = p_other(k, p_inv);
        if po <= 0.0 || r <= l {
            continue;
        }
        let a = (l - inv.bp_left) / inv_len;
        let b = (r - inv.bp_left) / inv_len;
        let rate = inv.gene_conversion_rate * po * phi_integral(a, b, w) * inv_len;
        if rate > 0.0 {
            out.push((Component::Seg { left: l, right: r }, rate));
        }
    }

    if edge_rate_unit <= 0.0 {
        return out;
    }
    // Pieces: maximal runs of touching segments, with the arrangement at
    // each end.
    let mut pieces: Vec<(f64, f64, Karyotype, Karyotype)> = Vec::new();
    for &(l, r, k) in segs {
        if r <= l {
            continue;
        }
        match pieces.last_mut() {
            Some(last) if l <= last.1 + TOUCH_EPS => {
                if r > last.1 {
                    last.1 = r;
                    last.3 = k;
                }
            }
            _ => pieces.push((l, r, k, k)),
        }
    }
    for (i, &(pl, pr, kl, kr)) in pieces.iter().enumerate() {
        if pl > inv.bp_left + TOUCH_EPS {
            let limit = if i > 0 { 0.5 * (pieces[i - 1].1 + pl) } else { inv.bp_left };
            let rate = edge_rate_unit * p_other(kl, p_inv);
            if rate > 0.0 {
                out.push((Component::Edge { edge: pl, limit }, rate));
            }
        }
        if pr < inv.bp_right - TOUCH_EPS {
            let limit = if i + 1 < pieces.len() {
                0.5 * (pr + pieces[i + 1].0)
            } else {
                inv.bp_right
            };
            let rate = edge_rate_unit * p_other(kr, p_inv);
            if rate > 0.0 {
                out.push((Component::Edge { edge: pr, limit }, rate));
            }
        }
    }
    out
}

/// Total flux rate for a lineage's carried segments in `inv`.
pub fn lineage_flux_rate(segs: &[ClassedSeg], inv: &InversionSpec, p_inv: f64) -> f64 {
    lineage_flux_rate_iter(segs.iter().copied(), inv, p_inv)
}

/// `lineage_flux_rate` in one pass without allocating: the rate needs
/// only the edges' arrangements, not their limits. Must sum the same
/// components as `components`.
pub fn lineage_flux_rate_iter<I>(segs: I, inv: &InversionSpec, p_inv: f64) -> f64
where
    I: IntoIterator<Item = ClassedSeg>,
{
    let inv_len = inv.length();
    if inv_len <= 0.0 || inv.gene_conversion_rate <= 0.0 {
        return 0.0;
    }
    let ell = inv.mean_tract_length;
    let w = ell / inv_len;
    let unit = if ell > 0.0 { inv.gene_conversion_rate * phi_max(w) * ell } else { 0.0 };
    let edges = |(pl, pr, kl, kr): (f64, f64, Karyotype, Karyotype)| -> f64 {
        let mut e = 0.0;
        if unit > 0.0 {
            if pl > inv.bp_left + TOUCH_EPS {
                e += unit * p_other(kl, p_inv);
            }
            if pr < inv.bp_right - TOUCH_EPS {
                e += unit * p_other(kr, p_inv);
            }
        }
        e
    };
    let mut rate = 0.0;
    let mut piece: Option<(f64, f64, Karyotype, Karyotype)> = None;
    for (l, r, k) in segs {
        if r <= l {
            continue;
        }
        let po = p_other(k, p_inv);
        if po > 0.0 {
            let a = (l - inv.bp_left) / inv_len;
            let b = (r - inv.bp_left) / inv_len;
            rate += inv.gene_conversion_rate * po * phi_integral(a, b, w) * inv_len;
        }
        piece = match piece {
            Some(mut p) if l <= p.1 + TOUCH_EPS => {
                if r > p.1 {
                    p.1 = r;
                    p.3 = k;
                }
                Some(p)
            }
            Some(p) => {
                rate += edges(p);
                Some((l, r, k, k))
            }
            None => Some((l, r, k, k)),
        };
    }
    if let Some(p) = piece {
        rate += edges(p);
    }
    rate
}

/// Draw a tract length from the inversion's tract distribution.
pub fn draw_tract_len(inv: &InversionSpec, rng: &mut Xoshiro256PlusPlus) -> f64 {
    let l = match inv.tract_distribution {
        TractDistribution::Fixed => inv.mean_tract_length,
        TractDistribution::Geometric => Exp::new(1.0 / inv.mean_tract_length)
            .expect("mean > 0")
            .sample(rng),
    };
    l.min(inv.length() * 0.99)
}

/// Place a tract of length `l` uniformly among the positions that
/// contain `x` and fit in the inversion.
pub fn place_tract(x: f64, l: f64, inv: &InversionSpec, rng: &mut Xoshiro256PlusPlus) -> (f64, f64) {
    let inv_len = inv.length();
    let x_rel = x - inv.bp_left;
    let b1_lo = (x_rel - l).max(0.0);
    let b1_hi = x_rel.min(inv_len - l);
    let b1 = if b1_hi <= b1_lo {
        (x_rel - l / 2.0).clamp(0.0, inv_len - l)
    } else {
        rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
    };
    let tl = (inv.bp_left + b1).max(inv.bp_left);
    let tr = (tl + l).min(inv.bp_right);
    (tl, tr)
}

/// Sample one flux event for a lineage, proportional to the components
/// of `lineage_flux_rate`. Returns `(x_event, tract_left, tract_right)`,
/// or `None` when the proposal is thinned (no event happens).
pub fn sample_flux_tract(
    segs: &[ClassedSeg],
    inv: &InversionSpec,
    p_inv: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<(f64, f64, f64)> {
    let comps = components(segs, inv, p_inv);
    let total: f64 = comps.iter().map(|c| c.1).sum();
    if total <= 0.0 {
        return None;
    }
    let u = rng.random::<f64>() * total;
    let mut running = 0.0;
    let mut chosen = comps.last().unwrap().0;
    for &(c, rate) in &comps {
        running += rate;
        if u < running {
            chosen = c;
            break;
        }
    }
    let inv_len = inv.length();
    let ell = inv.mean_tract_length;
    let w = ell / inv_len;
    let pmax = phi_max(w);
    let rel = |y: f64| (y - inv.bp_left) / inv_len;

    match chosen {
        Component::Seg { left, right } => {
            let mut y = None;
            for _ in 0..1000 {
                let yy = rng.random::<f64>() * (right - left) + left;
                if rng.random::<f64>() * pmax < phi(rel(yy), w) {
                    y = Some(yy);
                    break;
                }
            }
            let y = y.unwrap_or_else(|| rng.random::<f64>() * (right - left) + left);
            if ell <= 0.0 {
                return Some((y, y, y));
            }
            let l = draw_tract_len(inv, rng);
            let (tl, tr) = place_tract(y, l, inv, rng);
            Some((y, tl, tr))
        }
        Component::Edge { edge, limit } => {
            let d = match inv.tract_distribution {
                TractDistribution::Fixed => rng.random::<f64>() * ell,
                TractDistribution::Geometric => Exp::new(1.0 / ell).expect("mean > 0").sample(rng),
            };
            let y = if limit < edge { edge - d } else { edge + d };
            let inside = if limit < edge { y > limit } else { y < limit };
            if !inside {
                return None;
            }
            if rng.random::<f64>() * pmax >= phi(rel(y), w) {
                return None;
            }
            let l = match inv.tract_distribution {
                TractDistribution::Fixed => ell,
                TractDistribution::Geometric => d + Exp::new(1.0 / ell).expect("mean > 0").sample(rng),
            }
            .min(inv_len * 0.99);
            let (tl, tr) = place_tract(y, l, inv, rng);
            let hits = segs.iter().any(|&(sl, sr, _)| tl < sr && tr > sl);
            if hits { Some((y, tl, tr)) } else { None }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn inv(dist: TractDistribution) -> InversionSpec {
        let mut v = InversionSpec::with_p_inv(0.0, 100_000.0, vec![0.4], 1e9);
        v.gene_conversion_rate = 1e-6;
        v.mean_tract_length = 500.0;
        v.tract_distribution = dist;
        v
    }

    /// Rate at which tracts cover `x`, by Monte Carlo of the sampler.
    fn cover_rate(segs: &[ClassedSeg], v: &InversionSpec, x: f64, n: usize, seed: u64) -> f64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let total = lineage_flux_rate(segs, v, 0.4);
        let mut hits = 0usize;
        for _ in 0..n {
            if let Some((_, tl, tr)) = sample_flux_tract(segs, v, 0.4, &mut rng) {
                if tl <= x && x < tr {
                    hits += 1;
                }
            }
        }
        total * hits as f64 / n as f64
    }

    #[test]
    fn per_site_rate_does_not_depend_on_carried_material() {
        for dist in [TractDistribution::Fixed, TractDistribution::Geometric] {
            let v = inv(dist);
            let x = 40_000.0;
            let full = vec![(0.0, 100_000.0, Karyotype::S)];
            let sliver = vec![(x - 1.0, x + 1.0, Karyotype::S)];
            let split = vec![
                (10_000.0, x - 200.0, Karyotype::S),
                (x - 1.0, x + 3.0, Karyotype::S),
                (x + 150.0, 90_000.0, Karyotype::S),
            ];
            let n = 400_000;
            let c_full = cover_rate(&full, &v, x, n, 1);
            let c_sliver = cover_rate(&sliver, &v, x, n, 2);
            let c_split = cover_rate(&split, &v, x, n, 3);
            // Forward: gcr * p_other * phi(x) * mean_tract_length.
            let w = v.mean_tract_length / v.length();
            let expect = v.gene_conversion_rate * 0.4 * phi(0.4, w) * v.mean_tract_length;
            for (name, c) in [("full", c_full), ("sliver", c_sliver), ("split", c_split)] {
                assert!(
                    (c / expect - 1.0).abs() < 0.05,
                    "{:?} {}: cover rate {} vs forward {}",
                    dist, name, c, expect
                );
            }
        }
    }

    /// Exact forward rate at which tracts cover `x` (Monte Carlo of the
    /// forward process: start `y` at density gcr * phi, tract by the
    /// forward rule), times `p_other`.
    fn forward_cover_rate(v: &InversionSpec, p_other: f64, x: f64, n: usize, seed: u64) -> f64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let len = v.length();
        let w = v.mean_tract_length / len;
        let mut acc = 0.0;
        for _ in 0..n {
            let y = v.bp_left + rng.random::<f64>() * len;
            let l = draw_tract_len(v, &mut rng);
            let (tl, tr) = place_tract(y, l, v, &mut rng);
            if tl <= x && x < tr {
                acc += phi((y - v.bp_left) / len, w);
            }
        }
        v.gene_conversion_rate * p_other * acc / n as f64 * len
    }

    #[test]
    fn per_site_rate_matches_forward_near_a_breakpoint() {
        for dist in [TractDistribution::Fixed, TractDistribution::Geometric] {
            let v = inv(dist);
            for x in [150.0, 400.0] {
                let exact = forward_cover_rate(&v, 0.4, x, 4_000_000, 9);
                // Material within 2 kb of the breakpoint keeps the share
                // of events covering x large enough for a tight check.
                let layouts = [
                    vec![(0.0, 2_000.0, Karyotype::S)],
                    vec![(x - 1.0, x + 1.0, Karyotype::S)],
                    vec![(x - 1.0, 2_000.0, Karyotype::S)],
                    vec![(0.0, x + 1.0, Karyotype::S)],
                ];
                for (i, segs) in layouts.iter().enumerate() {
                    let c = cover_rate(segs, &v, x, 400_000, 20 + i as u64);
                    assert!(
                        (c / exact - 1.0).abs() < 0.05,
                        "{:?} x={} layout {}: cover rate {} vs forward {}",
                        dist, x, i, c, exact
                    );
                }
            }
        }
    }

    #[test]
    fn whole_inversion_has_no_edge_rate() {
        let v = inv(TractDistribution::Geometric);
        let full = vec![(0.0, 60_000.0, Karyotype::I), (60_000.0, 100_000.0, Karyotype::S)];
        let w = v.mean_tract_length / v.length();
        let expect = v.gene_conversion_rate * v.length()
            * (0.6 * phi_integral(0.0, 0.6, w) + 0.4 * phi_integral(0.6, 1.0, w));
        assert!((lineage_flux_rate(&full, &v, 0.4) / expect - 1.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_rate_matches_components() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        for dist in [TractDistribution::Fixed, TractDistribution::Geometric] {
            let v = inv(dist);
            for _ in 0..2000 {
                let n = rng.random_range(1..8);
                let mut cuts: Vec<f64> = (0..2 * n).map(|_| rng.random::<f64>() * 100_000.0).collect();
                if rng.random::<f64>() < 0.2 { cuts[0] = 0.0; }
                cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut segs = Vec::new();
                for i in 0..n {
                    let (l, mut r) = (cuts[2 * i], cuts[2 * i + 1]);
                    // Some touching neighbours.
                    if i + 1 < n && rng.random::<f64>() < 0.3 { r = cuts[2 * i + 2]; }
                    let k = if rng.random::<f64>() < 0.5 { Karyotype::S } else { Karyotype::I };
                    segs.push((l, r, k));
                }
                let p = rng.random::<f64>();
                let a = lineage_flux_rate(&segs, &v, p);
                let b: f64 = components(&segs, &v, p).iter().map(|c| c.1).sum();
                assert!((a - b).abs() <= 1e-12 * b.max(1e-30), "{:?}: {} vs {}", segs, a, b);
            }
        }
    }

    #[test]
    fn sampled_tracts_always_touch_material() {
        let v = inv(TractDistribution::Geometric);
        let segs = vec![(20_000.0, 20_010.0, Karyotype::S), (70_000.0, 70_400.0, Karyotype::I)];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for _ in 0..50_000 {
            if let Some((_, tl, tr)) = sample_flux_tract(&segs, &v, 0.3, &mut rng) {
                assert!(segs.iter().any(|&(l, r, _)| tl < r && tr > l));
                assert!(tl >= v.bp_left && tr <= v.bp_right);
            }
        }
    }
}
