//! Path 2 stage 1: compound per-pair coalescence rate.
//!
//! Pure function — no simulator state, no event loop hooks. Computes
//! the total coalescence rate for a pair (i, j) of lineages by walking
//! both segment chains once and accumulating per-position rate
//! contributions, weighted by overlap length and the appropriate
//! per-class sub-population frequency.
//!
//! See `compound_pair_rate.md` for the model.

use crate::class_tag::{BranchClass, Karyotype};
use crate::inversion::InversionSpec;
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};

/// Total coalescence rate for the pair (heads_a, heads_b) in pop `pop`
/// at effective Ne `ne`, sequence length `seq_len`. Zero if the pair
/// has no coalescence-eligible overlap (no material in common, or all
/// shared positions blocked by class barriers with mismatched S / I).
///
/// Rate formula, per overlap interval [l, r):
///   contribution = (r - l) / (2 · ne · p_eff(a_cls, b_cls) · seq_len)
///
/// where `p_eff` is the product over active inversions of:
///   - 1 if both sides are panmictic at that inv (no barrier)
///   - p_std(pop) if both sides carry S at that inv
///   - p_inv(pop) if both sides carry I at that inv
///   - +∞ (returns rate 0) if one side S and other I at an active
///     barrier — that overlap position does NOT coalesce.
///
/// `barrier_active[k]` = false means inv k has crossed t_inv
/// (barrier lifted, treat as panmictic regardless of bit).
pub fn compute_pair_rate(
    head_a: SegIdx,
    head_b: SegIdx,
    arena: &SegmentArena,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    pop: u32,
    ne: f64,
    seq_len: f64,
) -> f64 {
    if head_a == SEG_NIL || head_b == SEG_NIL { return 0.0; }
    if ne <= 0.0 || seq_len <= 0.0 { return 0.0; }

    let two_ne_l = 2.0 * ne * seq_len;
    let mut accum_inv_p_over_length: f64 = 0.0;

    let mut sa = head_a;
    let mut sb = head_b;
    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let b = arena.get(sb);
        if a.right <= b.left { sa = a.next; continue; }
        if b.right <= a.left { sb = b.next; continue; }
        let l = a.left.max(b.left);
        let r = a.right.min(b.right);
        if r > l {
            match p_eff(a.branch_class, b.branch_class,
                         inversions, barrier_active, pop) {
                Some(p) if p > 0.0 => {
                    accum_inv_p_over_length += (r - l) / p;
                }
                _ => {}  // blocked by barrier OR p_eff = 0 → no rate
            }
        }
        if a.right < b.right { sa = a.next; } else { sb = b.next; }
    }

    accum_inv_p_over_length / two_ne_l
}

/// Effective sub-population frequency product for a shared overlap
/// position. Returns None if the pair is barrier-blocked at this
/// position (S vs I at an active inv).
fn p_eff(
    a_cls: BranchClass,
    b_cls: BranchClass,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    pop: u32,
) -> Option<f64> {
    let mut p = 1.0;
    for (k, inv) in inversions.iter().enumerate() {
        if !barrier_active.get(k).copied().unwrap_or(false) { continue; }
        let a_k = a_cls.get_inv(inv.inv_id);
        let b_k = b_cls.get_inv(inv.inv_id);
        match (a_k, b_k) {
            (None, None) => {},                                // both pan — no barrier
            (None, Some(Karyotype::S)) | (Some(Karyotype::S), None) =>
                p *= inv.p_std_for(pop),
            (None, Some(Karyotype::I)) | (Some(Karyotype::I), None) =>
                p *= inv.p_inv_for(pop),
            (Some(Karyotype::S), Some(Karyotype::S)) =>
                p *= inv.p_std_for(pop),
            (Some(Karyotype::I), Some(Karyotype::I)) =>
                p *= inv.p_inv_for(pop),
            (Some(Karyotype::S), Some(Karyotype::I)) |
            (Some(Karyotype::I), Some(Karyotype::S)) => return None,
        }
    }
    Some(p)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_tag::BranchClass;

    const L: f64 = 1_000.0;
    const NE: f64 = 1_000.0;

    fn mk_chain(arena: &mut SegmentArena, segs: &[(f64, f64, BranchClass)])
        -> SegIdx
    {
        let mut head = SEG_NIL;
        let mut tail = SEG_NIL;
        for (i, &(l, r, cls)) in segs.iter().enumerate() {
            let s = arena.alloc(l, r, i as i32, cls);
            if head == SEG_NIL { head = s; }
            else { arena.get_mut(tail).next = s; }
            tail = s;
        }
        head
    }

    #[test]
    fn no_overlap_returns_zero() {
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let a = mk_chain(&mut arena, &[(0.0, 100.0, pan)]);
        let b = mk_chain(&mut arena, &[(200.0, 300.0, pan)]);
        let rate = compute_pair_rate(a, b, &arena, &[], &[], 0, NE, L);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn panmictic_full_overlap_matches_hudson() {
        // Two lineages spanning [0, L), all panmictic. Expected rate:
        //   L / (2·Ne·L) = 1 / (2·Ne) = 5e-4
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let b = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let rate = compute_pair_rate(a, b, &arena, &[], &[], 0, NE, L);
        let expected = 1.0 / (2.0 * NE);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }

    #[test]
    fn structured_full_overlap_at_S() {
        // Both lineages S at inv 0 over entire sequence. p_std = 0.5.
        // Rate: L / (2·Ne·p_std·L) = 1 / (2·Ne·0.5) = 1/Ne.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let s = BranchClass::single(0, Karyotype::S);
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: L,
            p_inv: vec![0.5], t_inv: 1e9,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, s)]);
        let rate = compute_pair_rate(
            a, b, &arena, std::slice::from_ref(&inv), &[true],
            0, NE, L);
        let expected = 1.0 / NE;
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }

    #[test]
    fn s_vs_i_blocked_at_inv_overlap() {
        // A is S at inv 0, B is I at inv 0. Overlap entirely inside
        // inv. Barrier blocks coalescence → rate 0.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let s = BranchClass::single(0, Karyotype::S);
        let i = BranchClass::single(0, Karyotype::I);
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: L,
            p_inv: vec![0.3], t_inv: 1e9,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, i)]);
        let rate = compute_pair_rate(
            a, b, &arena, std::slice::from_ref(&inv), &[true],
            0, NE, L);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn mixed_pan_and_S_sums_cleanly() {
        // A and B each have [0, 500) panmictic + [500, 1000) at S-inv0.
        // Inv 0 at [500, 1000), p_std = 0.5.
        // Rate:
        //   pan contribution: 500 / (2·Ne·1·L)         = 500/(2e6)
        //   S   contribution: 500 / (2·Ne·0.5·L)       = 500/(1e6)
        //   sum = 2.5e-4 + 5e-4 = 7.5e-4
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let inv = InversionSpec {
            bp_left: 500.0, bp_right: L,
            p_inv: vec![0.5], t_inv: 1e9,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let a = mk_chain(&mut arena, &[(0.0, 500.0, pan), (500.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, 500.0, pan), (500.0, L, s)]);
        let rate = compute_pair_rate(
            a, b, &arena, std::slice::from_ref(&inv), &[true],
            0, NE, L);
        let expected = 500.0 / (2.0 * NE * 1.0 * L)
                     + 500.0 / (2.0 * NE * 0.5 * L);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }

    #[test]
    fn barrier_lifted_ignores_class_tags() {
        // Same setup as s_vs_i_blocked, but barrier_active = false.
        // Now inv 0 is effectively panmictic; rate = 1/(2Ne).
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let s = BranchClass::single(0, Karyotype::S);
        let i = BranchClass::single(0, Karyotype::I);
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: L,
            p_inv: vec![0.3], t_inv: 1e9,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, i)]);
        let rate = compute_pair_rate(
            a, b, &arena, std::slice::from_ref(&inv), &[false],
            0, NE, L);
        let expected = 1.0 / (2.0 * NE);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }

    #[test]
    fn panmictic_vs_S_uses_s_partner_freq() {
        // A panmictic, B has S at inv 0. At overlap positions both
        // can coalesce (B's class restricts the pair to S-carriers).
        // Rate uses p_std.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: L,
            p_inv: vec![0.5], t_inv: 1e9,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let b = mk_chain(&mut arena, &[(0.0, L, s)]);
        let rate = compute_pair_rate(
            a, b, &arena, std::slice::from_ref(&inv), &[true],
            0, NE, L);
        let expected = 1.0 / (2.0 * NE * 0.5);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }
}
