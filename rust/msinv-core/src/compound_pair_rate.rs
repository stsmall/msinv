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
/// at effective Ne `ne`. Hudson per-pair-per-class rate: for each
/// distinct merged class that has any shared overlap between the
/// lineages, the pair contributes `1 / (2 · ne · p_eff(class))`. Classes
/// blocked by an active S-vs-I barrier contribute 0.
///
/// This matches `compute_coal_rates_structured` on the bucket path:
/// coalescence rate is per-pair-per-class with any eligible overlap,
/// not overlap-length-weighted. Overlap-weighting would make rates
/// vanish as recombination fragments lineages into tiny pieces — the
/// "remnant ratchet" that hangs on realistic Kir/Fol scale.
///
/// `seq_len` is unused here (kept for API symmetry with the earlier
/// overlap-weighted formulation); retained so callers don't break.
pub fn compute_pair_rate(
    head_a: SegIdx,
    head_b: SegIdx,
    arena: &SegmentArena,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    t: f64,
    pop: u32,
    ne: f64,
    _seq_len: f64,
) -> f64 {
    if head_a == SEG_NIL || head_b == SEG_NIL { return 0.0; }
    if ne <= 0.0 { return 0.0; }

    // Dedup key: the merged class (a_bc | b_bc) for each overlap
    // interval. Walk pairs with sufficient shared material and record
    // each distinct merged class's p_eff. Final rate sums 1/(2 Ne p_k)
    // over unique keys.
    let mut seen: SmallSet = SmallSet::new();
    let mut rate = 0.0;

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
            if let Some(p) = p_eff(a.branch_class, b.branch_class,
                                     inversions, barrier_active, t, pop) {
                if p > 0.0 {
                    let key = a.branch_class.bits() | b.branch_class.bits();
                    if seen.insert(key) {
                        rate += 1.0 / (2.0 * ne * p);
                    }
                }
            }
        }
        if a.right < b.right { sa = a.next; } else { sb = b.next; }
    }

    rate
}

/// Tiny inline-friendly set of u64 keys. Coal classes per pair are
/// usually 1–3; heap allocation would dominate `compute_pair_rate`.
struct SmallSet {
    data: [u64; 8],
    len: usize,
}

impl SmallSet {
    fn new() -> Self { Self { data: [0; 8], len: 0 } }
    /// Returns true if the key was newly inserted.
    fn insert(&mut self, k: u64) -> bool {
        for i in 0..self.len {
            if self.data[i] == k { return false; }
        }
        if self.len < self.data.len() {
            self.data[self.len] = k;
            self.len += 1;
        }
        // In the unlikely case of > 8 distinct classes the extras just
        // aren't deduped — rate may over-count. Two inversions = 9
        // possible classes (3 karyotypes^2), so this is a soft cap.
        true
    }
}

/// Effective sub-population frequency product for a shared overlap
/// position. Returns None if the pair is barrier-blocked at this
/// position (S vs I at an active inv).
///
/// When one side is PAN (uncommitted at this inv) and the other is
/// typed, the inv's contribution to p_eff is 1 — NOT the partner's
/// class frequency. Rationale: an uncommitted lineage marginalises
/// over its possible karyotype, yielding Hudson rate 1/(2·Ne) rather
/// than the structured rate 1/(2·Ne·p_std). Multiplying by p_std
/// here would make PAN-vs-S coalesce faster than the marginal
/// expectation (the uncommitted side adds uncertainty, not
/// restriction).
fn p_eff(
    a_cls: BranchClass,
    b_cls: BranchClass,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    t: f64,
    pop: u32,
) -> Option<f64> {
    let mut p = 1.0;
    for (k, inv) in inversions.iter().enumerate() {
        if !barrier_active.get(k).copied().unwrap_or(false) { continue; }
        let a_k = a_cls.get_inv(inv.inv_id);
        let b_k = b_cls.get_inv(inv.inv_id);
        match (a_k, b_k) {
            (None, _) | (_, None) => {}, // either side PAN → no restriction
            (Some(Karyotype::S), Some(Karyotype::S)) =>
                p *= inv.p_std_at(t, pop),
            (Some(Karyotype::I), Some(Karyotype::I)) =>
                p *= inv.p_inv_at(t, pop),
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
        let rate = compute_pair_rate(a,
            b,
            &arena,
            &[],
            &[],
            0.0,
            0,
            NE,
            L);
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
        let rate = compute_pair_rate(a,
            b,
            &arena,
            &[],
            &[],
            0.0,
            0,
            NE,
            L);
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
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, L, vec![0.5], 1e9);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, s)]);
        let rate = compute_pair_rate(a,
            b,
            &arena,
            std::slice::from_ref(&inv),
            &[true],
            0.0,
            0,
            NE,
            L);
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
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, L, vec![0.3], 1e9);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, i)]);
        let rate = compute_pair_rate(a,
            b,
            &arena,
            std::slice::from_ref(&inv),
            &[true],
            0.0,
            0,
            NE,
            L);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn mixed_pan_and_s_sums_cleanly() {
        // A and B each have [0, 500) panmictic + [500, 1000) at S-inv0.
        // Hudson per-pair-per-class: 1/(2Ne) for the PAN class (overlap
        // present) plus 1/(2Ne·p_std) for the S class. Overlap lengths
        // don't enter — only "is there any overlap in this class".
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let inv = { let mut s = InversionSpec::with_p_inv(500.0, L, vec![0.5], 1e9);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
        let a = mk_chain(&mut arena, &[(0.0, 500.0, pan), (500.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, 500.0, pan), (500.0, L, s)]);
        let rate = compute_pair_rate(a,
            b,
            &arena,
            std::slice::from_ref(&inv),
            &[true],
            0.0,
            0,
            NE,
            L);
        let expected = 1.0 / (2.0 * NE) + 1.0 / (2.0 * NE * 0.5);
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
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, L, vec![0.3], 1e9);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
        let a = mk_chain(&mut arena, &[(0.0, L, s)]);
        let b = mk_chain(&mut arena, &[(0.0, L, i)]);
        let rate = compute_pair_rate(a,
            b,
            &arena,
            std::slice::from_ref(&inv),
            &[false],
            0.0,
            0,
            NE,
            L);
        let expected = 1.0 / (2.0 * NE);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }

    #[test]
    fn panmictic_vs_S_is_hudson_rate() {
        // A panmictic (uncommitted), B has S at inv 0. Marginalising
        // over A's unknown karyotype gives Hudson rate 1/(2·Ne), NOT
        // the S-restricted rate 1/(2·Ne·p_std): P(A=S)·1/(2Ne·p_std)
        // + P(A=I)·0 = 1/(2Ne).
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, L, vec![0.5], 1e9);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
        let a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let b = mk_chain(&mut arena, &[(0.0, L, s)]);
        let rate = compute_pair_rate(a,
            b,
            &arena,
            std::slice::from_ref(&inv),
            &[true],
            0.0,
            0,
            NE,
            L);
        let expected = 1.0 / (2.0 * NE);
        assert!((rate - expected).abs() < 1e-12,
            "rate={} expected={}", rate, expected);
    }
}
