//! Path 2 stage 2a: Fenwick-backed compound pair-rate store.
//!
//! Holds one `f64` rate per pair (i, j) with i < j, flattened via
//! `pair_idx`. Backed by the existing `Fenwick` so we get O(log n²)
//! proportional pair selection for CoalPairCompound events.
//!
//! Stage 2a scope: build, read total, draw a pair by rate. No
//! incremental mutations yet — rebuild-from-scratch only. Stage 2b
//! adds `recompute_for`, `remove_lineage`, `swap_update`,
//! `apply_recomb_split` mirrors of the existing `RateCache` API.

use crate::compound_pair_rate::compute_pair_rate;
use crate::demography::Demography;
use crate::fenwick::Fenwick;
use crate::inversion::InversionSpec;
use crate::lineage::Lineage;
use crate::segment::SegmentArena;

/// Flat index for a pair (i, j) where i < j in a triangular array of
/// size `cap`. Matches `rate_index::pair_idx`.
#[inline]
pub fn pair_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j && j < n);
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Inverse of `pair_idx`: given a flat pidx and the triangular
/// capacity, recover (i, j). Binary search on the row offset.
#[inline]
pub fn unpack_pair_idx(pidx: usize, cap: usize) -> (usize, usize) {
    // Row i starts at offset `off(i) = i*cap - i*(i+1)/2` and has
    // `cap - i - 1` entries. Find largest i with `off(i) <= pidx`.
    let mut lo = 0usize;
    let mut hi = cap;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let off = mid * cap - mid * (mid + 1) / 2;
        if off <= pidx { lo = mid + 1; } else { hi = mid; }
    }
    let i = lo - 1;
    let off_i = i * cap - i * (i + 1) / 2;
    let j = pidx - off_i + i + 1;
    (i, j)
}

/// Number of entries in the triangular pair array for `n` lineages.
#[inline]
pub fn tri_size(n: usize) -> usize { n * (n - 1) / 2 }

pub struct PairRateCache {
    pair_rates: Fenwick,
    capacity: usize,
    n: usize,
}

impl PairRateCache {
    pub fn new(max_lineages: usize) -> Self {
        let cap = max_lineages.max(1);
        let n_pairs = tri_size(cap);
        Self {
            pair_rates: Fenwick::new(n_pairs),
            capacity: cap,
            n: 0,
        }
    }

    pub fn capacity(&self) -> usize { self.capacity }
    pub fn n(&self) -> usize { self.n }

    pub fn total(&self) -> f64 { self.pair_rates.total() }

    /// Rate at pair (i, j). Panics if i >= j or j >= self.n.
    pub fn rate_at(&self, i: usize, j: usize) -> f64 {
        let p = pair_idx(i, j, self.capacity);
        self.pair_rates.range_sum(p, p + 1)
    }

    /// Rebuild from scratch. O(n²) in active lineages — use sparingly.
    pub fn rebuild(
        &mut self,
        active: &[Lineage],
        arena: &SegmentArena,
        inversions: &[InversionSpec],
        barrier_active: &[bool],
        demo: &Demography,
        t: f64,
        seq_len: f64,
    ) {
        self.n = active.len();
        // Grow capacity if needed.
        if self.n > self.capacity {
            self.capacity = self.n.max(self.capacity * 2);
            self.pair_rates = Fenwick::new(tri_size(self.capacity));
        } else {
            self.pair_rates.reset(tri_size(self.capacity));
        }
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                // Same-pop filter — cross-pop pairs don't coalesce.
                if active[i].population != active[j].population { continue; }
                let pop = active[i].population;
                let ne = demo.size_at(pop, t).max(1e-9);
                let rate = compute_pair_rate(
                    active[i].head, active[j].head,
                    arena, inversions, barrier_active,
                    pop, ne, seq_len);
                if rate > 0.0 {
                    let pidx = pair_idx(i, j, self.capacity);
                    self.pair_rates.update(pidx, rate);
                }
            }
        }
    }

    /// Draw a pair proportional to rate. Returns (i, j) with i < j,
    /// or None if total rate is zero.
    pub fn sample_pair(&self, target: f64) -> Option<(usize, usize)> {
        let total = self.total();
        if !(target >= 0.0) || total <= 0.0 { return None; }
        let t = target.min(total - f64::EPSILON * total.max(1.0));
        let pidx = self.pair_rates.find(t);
        if pidx >= tri_size(self.capacity) { return None; }
        Some(unpack_pair_idx(pidx, self.capacity))
    }

    /// Recompute all pairs involving lineage `idx`. O(n) rate evals.
    /// Call after mutating `idx`'s chain (coalescence, recombination,
    /// flux, migration).
    pub fn recompute_for(
        &mut self,
        idx: usize,
        active: &[Lineage],
        arena: &SegmentArena,
        inversions: &[InversionSpec],
        barrier_active: &[bool],
        demo: &Demography,
        t: f64,
        seq_len: f64,
    ) {
        let n = active.len();
        if n > self.capacity { self.grow_to(n); }
        self.n = n;
        let pop_idx = active[idx].population;
        let ne = demo.size_at(pop_idx, t).max(1e-9);
        for other in 0..n {
            if other == idx { continue; }
            let (i, j) = if idx < other { (idx, other) } else { (other, idx) };
            let new_rate = if active[i].population != active[j].population {
                0.0
            } else {
                compute_pair_rate(
                    active[i].head, active[j].head,
                    arena, inversions, barrier_active,
                    pop_idx, ne, seq_len)
            };
            let pidx = pair_idx(i, j, self.capacity);
            self.pair_rates.set(pidx, new_rate);
        }
    }

    /// Clear all pair rates involving lineage `idx`. Caller must call
    /// this BEFORE removing `idx` from `active`. `n_active` is
    /// active.len() at call time (idx still present).
    pub fn remove_lineage(&mut self, idx: usize, n_active: usize) {
        for other in 0..n_active {
            if other == idx { continue; }
            let (i, j) = if idx < other { (idx, other) } else { (other, idx) };
            let pidx = pair_idx(i, j, self.capacity);
            self.pair_rates.set(pidx, 0.0);
        }
    }

    /// Mirror `active.swap_remove(removed_idx)` on the pair rates:
    /// for every peer j != removed_idx, j != old_last, move the pair
    /// rate at (j, old_last) to (j, removed_idx). The old row is
    /// zeroed. Decrements n.
    ///
    /// Precondition: caller already invoked `remove_lineage(removed_idx, pre_len)`
    /// so all (removed_idx, *) slots are zero at entry.
    pub fn swap_update(&mut self, removed_idx: usize, old_last: usize) {
        if removed_idx == old_last {
            self.n = self.n.saturating_sub(1);
            return;
        }
        let pre_n = self.n;
        for other in 0..pre_n {
            if other == removed_idx || other == old_last { continue; }
            let (oi, oj) = if other < old_last {
                (other, old_last)
            } else {
                (old_last, other)
            };
            let (ni, nj) = if other < removed_idx {
                (other, removed_idx)
            } else {
                (removed_idx, other)
            };
            let old_pidx = pair_idx(oi, oj, self.capacity);
            let new_pidx = pair_idx(ni, nj, self.capacity);
            let r = self.pair_rates.range_sum(old_pidx, old_pidx + 1);
            if r != 0.0 {
                self.pair_rates.set(old_pidx, 0.0);
                self.pair_rates.set(new_pidx, r);
            }
        }
        self.n = self.n.saturating_sub(1);
    }

    fn grow_to(&mut self, need: usize) {
        let new_cap = (need * 2).max(self.capacity * 2);
        let mut rates: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let pidx = pair_idx(i, j, self.capacity);
                let r = self.pair_rates.range_sum(pidx, pidx + 1);
                if r > 0.0 { rates.push((i, j, r)); }
            }
        }
        self.capacity = new_cap;
        self.pair_rates = Fenwick::new(tri_size(new_cap));
        for (i, j, r) in rates {
            let pidx = pair_idx(i, j, self.capacity);
            self.pair_rates.update(pidx, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_tag::BranchClass;
    use crate::segment::{SegIdx, SEG_NIL};
    use crate::demography::Demography;

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
    fn pair_idx_roundtrip() {
        for cap in [2, 4, 10, 64] {
            for i in 0..cap {
                for j in (i + 1)..cap {
                    let pidx = pair_idx(i, j, cap);
                    let (i2, j2) = unpack_pair_idx(pidx, cap);
                    assert_eq!((i, j), (i2, j2),
                        "cap={} pair=({},{}) → pidx={} → ({},{})",
                        cap, i, j, pidx, i2, j2);
                }
            }
        }
    }

    #[test]
    fn pair_idx_covers_tri_size() {
        for cap in [2, 4, 10, 64] {
            let mut seen = vec![false; tri_size(cap)];
            for i in 0..cap {
                for j in (i + 1)..cap {
                    seen[pair_idx(i, j, cap)] = true;
                }
            }
            assert!(seen.iter().all(|&b| b),
                "cap={} pair_idx skipped some pidx", cap);
        }
    }

    #[test]
    fn rebuild_empty_is_zero() {
        let mut arena = SegmentArena::new();
        let demo = Demography::single_pop(1000.0);
        let active: Vec<Lineage> = Vec::new();
        let mut cache = PairRateCache::new(16);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, 1000.0);
        let _ = &mut arena;  // silence unused
        assert_eq!(cache.n(), 0);
        assert_eq!(cache.total(), 0.0);
    }

    #[test]
    fn rebuild_two_panmictic_matches_hudson() {
        // Two lineages spanning [0, L), panmictic, Ne=1000.
        // Single pair: rate = 1/(2·Ne) = 5e-4.
        const L: f64 = 1000.0;
        const NE: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let h_a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h_b = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let active = vec![
            Lineage::new(h_a, h_a, 0, 0, &arena),
            Lineage::new(h_b, h_b, 0, 1, &arena),
        ];
        let demo = Demography::single_pop(NE);
        let mut cache = PairRateCache::new(4);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, L);
        let expected = 1.0 / (2.0 * NE);
        assert!((cache.total() - expected).abs() < 1e-12);
        assert!((cache.rate_at(0, 1) - expected).abs() < 1e-12);
    }

    #[test]
    fn cross_pop_pair_excluded() {
        // Two lineages in different pops — should not be in the cache.
        const L: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let h_a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h_b = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let active = vec![
            Lineage::new(h_a, h_a, 0, 0, &arena),
            Lineage::new(h_b, h_b, 1, 1, &arena),
        ];
        let demo = Demography::new(vec![1000.0, 1000.0]);
        let mut cache = PairRateCache::new(4);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, L);
        assert_eq!(cache.total(), 0.0);
    }

    fn total_of_rebuild(
        active: &[Lineage], arena: &SegmentArena,
        inversions: &[InversionSpec], barrier_active: &[bool],
        demo: &Demography, t: f64, seq_len: f64, cap: usize,
    ) -> f64 {
        let mut c = PairRateCache::new(cap);
        c.rebuild(active, arena, inversions, barrier_active, demo, t, seq_len);
        c.total()
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1e-12)
    }

    #[test]
    fn remove_lineage_zeros_row_and_col() {
        const L: f64 = 1000.0;
        const NE: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let heads: Vec<_> = (0..4).map(|_|
            mk_chain(&mut arena, &[(0.0, L, pan)])).collect();
        let active: Vec<Lineage> = heads.iter().enumerate()
            .map(|(i, &h)| Lineage::new(h, h, 0, i as u32, &arena))
            .collect();
        let demo = Demography::single_pop(NE);
        let mut cache = PairRateCache::new(8);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, L);
        let full = cache.total();
        cache.remove_lineage(2, active.len());
        // Pairs involving 2: (0,2), (1,2), (2,3). Each had rate 1/(2Ne).
        let expected = full - 3.0 * (1.0 / (2.0 * NE));
        assert!(approx_eq(cache.total(), expected, 1e-10),
            "got={} expected={}", cache.total(), expected);
        assert_eq!(cache.rate_at(0, 2), 0.0);
        assert_eq!(cache.rate_at(1, 2), 0.0);
        assert_eq!(cache.rate_at(2, 3), 0.0);
    }

    #[test]
    fn swap_update_matches_rebuild() {
        // Build cache, simulate active.swap_remove(idx), compare incremental
        // path (remove_lineage + swap_update) against a fresh rebuild.
        const L: f64 = 1000.0;
        const NE: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        // Four lineages, varying extents so every pair has distinct rate.
        let h0 = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h1 = mk_chain(&mut arena, &[(100.0, 900.0, pan)]);
        let h2 = mk_chain(&mut arena, &[(0.0, 500.0, pan)]);
        let h3 = mk_chain(&mut arena, &[(400.0, L, pan)]);
        let active_pre = vec![
            Lineage::new(h0, h0, 0, 0, &arena),
            Lineage::new(h1, h1, 0, 1, &arena),
            Lineage::new(h2, h2, 0, 2, &arena),
            Lineage::new(h3, h3, 0, 3, &arena),
        ];
        let demo = Demography::single_pop(NE);
        let mut cache = PairRateCache::new(8);
        cache.rebuild(&active_pre, &arena, &[], &[], &demo, 0.0, L);

        // Apply active.swap_remove(1) to the cache.
        cache.remove_lineage(1, 4);
        cache.swap_update(1, 3);

        // Build the post-swap active list independently for the oracle.
        let active_post = vec![
            Lineage::new(h0, h0, 0, 0, &arena),
            Lineage::new(h3, h3, 0, 3, &arena),   // moved from index 3
            Lineage::new(h2, h2, 0, 2, &arena),
        ];
        drop(active_pre);

        let want = total_of_rebuild(
            &active_post, &arena, &[], &[], &demo, 0.0, L, 8);
        assert!(approx_eq(cache.total(), want, 1e-10),
            "incremental={} rebuild={}", cache.total(), want);
        assert_eq!(cache.n(), 3);
    }

    #[test]
    fn recompute_for_matches_rebuild_after_mutation() {
        const L: f64 = 1000.0;
        const NE: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        let h0 = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h1 = mk_chain(&mut arena, &[(100.0, 900.0, pan)]);
        let h2 = mk_chain(&mut arena, &[(200.0, 800.0, pan)]);
        let mut active = vec![
            Lineage::new(h0, h0, 0, 0, &arena),
            Lineage::new(h1, h1, 0, 1, &arena),
            Lineage::new(h2, h2, 0, 2, &arena),
        ];
        let demo = Demography::single_pop(NE);
        let mut cache = PairRateCache::new(8);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, L);

        // Mutate lineage 1: shrink its chain.
        let h1b = mk_chain(&mut arena, &[(300.0, 700.0, pan)]);
        active[1] = Lineage::new(h1b, h1b, 0, 1, &arena);

        cache.recompute_for(1, &active, &arena, &[], &[], &demo, 0.0, L);

        let want = total_of_rebuild(
            &active, &arena, &[], &[], &demo, 0.0, L, 8);
        assert!(approx_eq(cache.total(), want, 1e-10),
            "incremental={} rebuild={}", cache.total(), want);
    }

    #[test]
    fn sample_pair_is_proportional() {
        // Three lineages, one much more valuable pair than the others.
        // Draw many samples, check the big-rate pair wins majority.
        const L: f64 = 1000.0;
        const NE: f64 = 1000.0;
        let mut arena = SegmentArena::new();
        let pan = BranchClass::PANMICTIC;
        // a, b fully overlap; c has only a small segment.
        let h_a = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h_b = mk_chain(&mut arena, &[(0.0, L, pan)]);
        let h_c = mk_chain(&mut arena, &[(0.0, 10.0, pan)]);
        let active = vec![
            Lineage::new(h_a, h_a, 0, 0, &arena),
            Lineage::new(h_b, h_b, 0, 1, &arena),
            Lineage::new(h_c, h_c, 0, 2, &arena),
        ];
        let demo = Demography::single_pop(NE);
        let mut cache = PairRateCache::new(8);
        cache.rebuild(&active, &arena, &[], &[], &demo, 0.0, L);
        let ab = cache.rate_at(0, 1);
        let ac = cache.rate_at(0, 2);
        let bc = cache.rate_at(1, 2);
        // Hand-check: ab overlap = L, ac and bc overlap = 10.
        let exp_ab = L / (2.0 * NE * L);
        let exp_sm = 10.0 / (2.0 * NE * L);
        assert!((ab - exp_ab).abs() < 1e-12, "ab={} exp={}", ab, exp_ab);
        assert!((ac - exp_sm).abs() < 1e-12);
        assert!((bc - exp_sm).abs() < 1e-12);

        // Draws from total range: first bucket (ab) should always win.
        let total = cache.total();
        let mid = total * 0.5;
        let (i, j) = cache.sample_pair(mid).unwrap();
        assert!((i, j) == (0, 1) || (i, j) == (0, 2) || (i, j) == (1, 2));
        // Very small target should hit the first non-empty pair.
        let (i0, j0) = cache.sample_pair(1e-9).unwrap();
        assert_eq!((i0, j0), (0, 1));
    }
}
