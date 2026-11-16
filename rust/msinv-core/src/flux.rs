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
//!
//! An inversion can carry two independent flux processes (#30): the
//! short process (`gene_conversion_rate`, `mean_tract_length`) and an
//! optional long one (`long_flux_rate`, `long_mean_tract_length`). Each
//! follows the forward model above with its own `w`, rates add, and an
//! event is drawn from the components of both, so a lineage meets the
//! long process at its own forward rate. With the long process off the
//! components, the rate and the random draws are those of one process.

use rand::Rng;
use rand_distr::{Distribution, Exp};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::Karyotype;
use crate::inversion::{FluxProc, InversionSpec, TractDistribution};
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

/// Components of one process and their rates, tagged with the process
/// index `pi`. `segs` must be sorted by position and clipped to the
/// inversion.
fn components_proc(
    segs: &[ClassedSeg],
    pr: &FluxProc,
    pi: usize,
    p_inv: f64,
    out: &mut Vec<(Component, usize, f64)>,
) {
    let inv_len = pr.length();
    if inv_len <= 0.0 || pr.gcr <= 0.0 || segs.is_empty() {
        return;
    }
    let ell = pr.mean;
    let w = ell / inv_len;
    let edge_rate_unit = if ell > 0.0 { pr.gcr * phi_max(w) * ell } else { 0.0 };

    for &(l, r, k) in segs {
        let po = p_other(k, p_inv);
        if po <= 0.0 || r <= l {
            continue;
        }
        let a = (l - pr.bp_left) / inv_len;
        let b = (r - pr.bp_left) / inv_len;
        let rate = pr.gcr * po * phi_integral(a, b, w) * inv_len;
        if rate > 0.0 {
            out.push((Component::Seg { left: l, right: r }, pi, rate));
        }
    }

    if edge_rate_unit <= 0.0 {
        return;
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
    for (i, &(pl, prt, kl, kr)) in pieces.iter().enumerate() {
        if pl > pr.bp_left + TOUCH_EPS {
            let limit = if i > 0 { 0.5 * (pieces[i - 1].1 + pl) } else { pr.bp_left };
            let rate = edge_rate_unit * p_other(kl, p_inv);
            if rate > 0.0 {
                out.push((Component::Edge { edge: pl, limit }, pi, rate));
            }
        }
        if prt < pr.bp_right - TOUCH_EPS {
            let limit = if i + 1 < pieces.len() {
                0.5 * (prt + pieces[i + 1].0)
            } else {
                pr.bp_right
            };
            let rate = edge_rate_unit * p_other(kr, p_inv);
            if rate > 0.0 {
                out.push((Component::Edge { edge: prt, limit }, pi, rate));
            }
        }
    }
}

/// Components of every flux process of `inv`, short process first.
fn components(segs: &[ClassedSeg], inv: &InversionSpec, p_inv: f64) -> Vec<(Component, usize, f64)> {
    let mut out = Vec::new();
    let procs = inv.flux_procs();
    for (pi, pr) in procs.as_slice().iter().enumerate() {
        components_proc(segs, pr, pi, p_inv, &mut out);
    }
    out
}

/// Total flux rate for a lineage's carried segments in `inv`.
pub fn lineage_flux_rate(segs: &[ClassedSeg], inv: &InversionSpec, p_inv: f64) -> f64 {
    lineage_flux_rate_iter(segs.iter().copied(), inv, p_inv)
}

/// `lineage_flux_rate` in one pass without allocating: the rate needs
/// only the edges' arrangements, not their limits. Must sum the same
/// components as `components`, over every process.
pub fn lineage_flux_rate_iter<I>(segs: I, inv: &InversionSpec, p_inv: f64) -> f64
where
    I: IntoIterator<Item = ClassedSeg>,
{
    let inv_len = inv.length();
    if inv_len <= 0.0 {
        return 0.0;
    }
    let procs = inv.flux_procs();
    let ps = procs.as_slice();
    let mut w = [0.0f64; 2];
    let mut unit = [0.0f64; 2];
    let mut any = false;
    for (i, pr) in ps.iter().enumerate() {
        if pr.gcr <= 0.0 {
            continue;
        }
        any = true;
        w[i] = pr.mean / inv_len;
        unit[i] = if pr.mean > 0.0 { pr.gcr * phi_max(w[i]) * pr.mean } else { 0.0 };
    }
    if !any {
        return 0.0;
    }
    let edges = |(pl, prt, kl, kr): (f64, f64, Karyotype, Karyotype)| -> f64 {
        let mut e = 0.0;
        for i in 0..ps.len() {
            if unit[i] > 0.0 {
                if pl > inv.bp_left + TOUCH_EPS {
                    e += unit[i] * p_other(kl, p_inv);
                }
                if prt < inv.bp_right - TOUCH_EPS {
                    e += unit[i] * p_other(kr, p_inv);
                }
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
            for (i, pr) in ps.iter().enumerate() {
                if pr.gcr > 0.0 {
                    rate += pr.gcr * po * phi_integral(a, b, w[i]) * inv_len;
                }
            }
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

/// Draw a tract length from one process's tract distribution.
pub fn draw_tract_len_proc(pr: &FluxProc, rng: &mut Xoshiro256PlusPlus) -> f64 {
    let l = match pr.dist {
        TractDistribution::Fixed => pr.mean,
        TractDistribution::Geometric => Exp::new(1.0 / pr.mean).expect("mean > 0").sample(rng),
    };
    l.min(pr.length() * 0.99)
}

/// Draw a tract length from the inversion's short-process distribution.
pub fn draw_tract_len(inv: &InversionSpec, rng: &mut Xoshiro256PlusPlus) -> f64 {
    draw_tract_len_proc(&inv.flux_procs().as_slice()[0], rng)
}

/// Place a tract of length `l` uniformly among the positions that
/// contain `x` and fit in the inversion `[bp_left, bp_right)`.
fn place_tract_in(x: f64, l: f64, bp_left: f64, bp_right: f64, rng: &mut Xoshiro256PlusPlus) -> (f64, f64) {
    let inv_len = bp_right - bp_left;
    let x_rel = x - bp_left;
    let b1_lo = (x_rel - l).max(0.0);
    let b1_hi = x_rel.min(inv_len - l);
    let b1 = if b1_hi <= b1_lo {
        (x_rel - l / 2.0).clamp(0.0, inv_len - l)
    } else {
        rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
    };
    let tl = (bp_left + b1).max(bp_left);
    let tr = (tl + l).min(bp_right);
    (tl, tr)
}

/// Place a tract of length `l` uniformly among the positions that
/// contain `x` and fit in the inversion.
pub fn place_tract(x: f64, l: f64, inv: &InversionSpec, rng: &mut Xoshiro256PlusPlus) -> (f64, f64) {
    place_tract_in(x, l, inv.bp_left, inv.bp_right, rng)
}

/// Rate per generation at which tracts of one process cover a carried
/// site `x`, before the factor `p_other`: `gcr * mean * phi(x)`.
pub fn site_cover_rate_proc(pr: &FluxProc, x: f64) -> f64 {
    let len = pr.length();
    if len <= 0.0 || pr.gcr <= 0.0 || pr.mean <= 0.0 {
        return 0.0;
    }
    pr.gcr * pr.mean * phi((x - pr.bp_left) / len, pr.mean / len)
}

/// `site_cover_rate_proc` summed over the inversion's processes.
pub fn site_cover_rate(inv: &InversionSpec, x: f64) -> f64 {
    inv.flux_procs().as_slice().iter().map(|pr| site_cover_rate_proc(pr, x)).sum()
}

/// Draw a tract that covers `x`, from the process chosen in proportion
/// to its cover rate at `x`. One process: no extra random draw.
pub fn draw_tract_covering(x: f64, inv: &InversionSpec, rng: &mut Xoshiro256PlusPlus) -> (f64, f64) {
    let procs = inv.flux_procs();
    let ps = procs.as_slice();
    let mut pr = ps[0];
    if ps.len() > 1 {
        let r: Vec<f64> = ps.iter().map(|p| site_cover_rate_proc(p, x)).collect();
        let tot: f64 = r.iter().sum();
        if tot > 0.0 && rng.random::<f64>() * tot >= r[0] {
            pr = ps[1];
        }
    }
    if pr.mean <= 0.0 {
        return (x, x);
    }
    let l = draw_tract_len_proc(&pr, rng);
    place_tract_in(x, l, pr.bp_left, pr.bp_right, rng)
}

/// Sample one flux event for a lineage, proportional to the components
/// of `lineage_flux_rate` over every process. Returns
/// `(x_event, tract_left, tract_right)`, or `None` when the proposal is
/// thinned (no event happens).
pub fn sample_flux_tract(
    segs: &[ClassedSeg],
    inv: &InversionSpec,
    p_inv: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<(f64, f64, f64)> {
    let comps = components(segs, inv, p_inv);
    let total: f64 = comps.iter().map(|c| c.2).sum();
    if total <= 0.0 {
        return None;
    }
    let u = rng.random::<f64>() * total;
    let mut running = 0.0;
    let last = comps.last().unwrap();
    let (mut chosen, mut pi) = (last.0, last.1);
    for &(c, i, rate) in &comps {
        running += rate;
        if u < running {
            chosen = c;
            pi = i;
            break;
        }
    }
    let procs = inv.flux_procs();
    let pr = procs.as_slice()[pi];
    let inv_len = pr.length();
    let ell = pr.mean;
    let w = ell / inv_len;
    let pmax = phi_max(w);
    let rel = |y: f64| (y - pr.bp_left) / inv_len;

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
            let l = draw_tract_len_proc(&pr, rng);
            let (tl, tr) = place_tract_in(y, l, pr.bp_left, pr.bp_right, rng);
            Some((y, tl, tr))
        }
        Component::Edge { edge, limit } => {
            let d = match pr.dist {
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
            let l = match pr.dist {
                TractDistribution::Fixed => ell,
                TractDistribution::Geometric => d + Exp::new(1.0 / ell).expect("mean > 0").sample(rng),
            }
            .min(inv_len * 0.99);
            let (tl, tr) = place_tract_in(y, l, pr.bp_left, pr.bp_right, rng);
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
                let b: f64 = components(&segs, &v, p).iter().map(|c| c.2).sum();
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

    fn two_proc(dist: TractDistribution) -> InversionSpec {
        let mut v = inv(dist);
        v.long_flux_rate = 2e-8;
        v.long_mean_tract_length = 20_000.0;
        v.long_tract_distribution = dist;
        v
    }

    /// The long process alone, as a one-process spec.
    fn long_only(v: &InversionSpec) -> InversionSpec {
        let mut l = InversionSpec::with_p_inv(v.bp_left, v.bp_right, vec![0.4], 1e9);
        l.gene_conversion_rate = v.long_flux_rate;
        l.mean_tract_length = v.long_mean_tract_length;
        l.tract_distribution = v.long_tract_distribution;
        l
    }

    #[test]
    fn long_process_off_by_default() {
        let v = inv(TractDistribution::Geometric);
        assert_eq!(v.flux_procs().as_slice().len(), 1);
        let mut w = v.clone();
        w.long_mean_tract_length = 5_000.0; // rate still 0
        assert_eq!(w.flux_procs().as_slice().len(), 1);
        // Same draws as a spec that never had the field set.
        let segs = vec![(1_000.0, 1_050.0, Karyotype::S), (30_000.0, 31_000.0, Karyotype::I)];
        let mut r1 = Xoshiro256PlusPlus::seed_from_u64(3);
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(3);
        for _ in 0..5_000 {
            assert_eq!(sample_flux_tract(&segs, &v, 0.4, &mut r1), sample_flux_tract(&segs, &w, 0.4, &mut r2));
        }
    }

    #[test]
    fn two_process_rate_is_the_sum() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        for dist in [TractDistribution::Fixed, TractDistribution::Geometric] {
            let v = two_proc(dist);
            let s = inv(dist);
            let l = long_only(&v);
            for _ in 0..500 {
                let a = rng.random::<f64>() * 90_000.0;
                let b = a + rng.random::<f64>() * 10_000.0;
                let segs = vec![(a, b, Karyotype::S), (b, (b + 500.0).min(100_000.0), Karyotype::I)];
                let p = rng.random::<f64>();
                let both = lineage_flux_rate(&segs, &v, p);
                let sum = lineage_flux_rate(&segs, &s, p) + lineage_flux_rate(&segs, &l, p);
                assert!((both / sum - 1.0).abs() < 1e-12, "{} vs {}", both, sum);
                let c: f64 = components(&segs, &v, p).iter().map(|c| c.2).sum();
                assert!((both / c - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn two_process_per_site_rate_matches_forward() {
        for dist in [TractDistribution::Fixed, TractDistribution::Geometric] {
            let v = two_proc(dist);
            let s = inv(dist);
            let l = long_only(&v);
            for x in [400.0, 50_000.0] {
                let exact = forward_cover_rate(&s, 0.4, x, 2_000_000, 9)
                    + forward_cover_rate(&l, 0.4, x, 2_000_000, 10);
                let layouts = [
                    vec![(x - 1.0, x + 1.0, Karyotype::S)],
                    vec![(x - 1_000.0_f64.min(x), x + 1_000.0, Karyotype::S)],
                    vec![(0.0, 100_000.0, Karyotype::S)],
                ];
                for (i, segs) in layouts.iter().enumerate() {
                    let c = cover_rate(segs, &v, x, 1_000_000, 30 + i as u64);
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
    fn site_cover_rate_sums_processes() {
        let v = two_proc(TractDistribution::Geometric);
        let x = 37_000.0;
        let s = inv(TractDistribution::Geometric);
        let l = long_only(&v);
        let a = site_cover_rate(&v, x);
        let b = site_cover_rate(&s, x) + site_cover_rate(&l, x);
        assert!((a / b - 1.0).abs() < 1e-12);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut long_hits = 0usize;
        let n = 200_000;
        for _ in 0..n {
            let (tl, tr) = draw_tract_covering(x, &v, &mut rng);
            assert!(tl <= x && x <= tr);
            if tr - tl > 5_000.0 { long_hits += 1; }
        }
        // Share of tracts from the long process, against its cover-rate share
        // (geometric short tracts of mean 500 bp almost never exceed 5 kb).
        let share = site_cover_rate(&l, x) / a;
        let p_long_gt = (-5_000.0f64 / 20_000.0).exp();
        let got = long_hits as f64 / n as f64;
        assert!((got / (share * p_long_gt) - 1.0).abs() < 0.05, "{} vs {}", got, share * p_long_gt);
    }
}
