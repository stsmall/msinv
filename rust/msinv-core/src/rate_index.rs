/// Incremental rate cache for O(n) per-event coalescence rate updates.
///
/// Instead of recomputing all n^2/2 pair overlaps every iteration,
/// maintain a cache of per-pair overlap-by-class results. After an
/// event that changes lineage `idx`, only recompute O(n) pairs
/// involving `idx`.
///
/// Uses array indices (not LinUid) following msprime's swap-update
/// protocol: when a lineage at position `idx` is removed via
/// `swap_remove`, the last lineage moves to `idx` and all cache
/// entries referencing the old last index are patched.

use crate::class_tag::BranchClass;
use crate::lineage::Lineage;
use crate::segment::SegmentArena;

use smallvec::SmallVec;

/// Per-pair overlap: list of (BranchClass, overlap_length) entries.
/// SmallVec inline size 2 fits the common 1-2 class case while keeping
/// the per-slot footprint small (matters at n² scale at rho ≥ 500).
type PairOverlap = SmallVec<[(BranchClass, f64); 2]>;

/// Flat index for a pair (i, j) where i < j, into a triangular array.
#[inline]
pub fn pair_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j && j < n);
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Number of entries in the triangular pair cache for n lineages.
#[inline]
pub fn tri_size(n: usize) -> usize {
    n * (n - 1) / 2
}

pub struct RateCache {
    /// Per-pair overlap cache. Indexed by pair_idx(i, j, capacity).
    overlaps: Vec<PairOverlap>,
    /// Bitmap of non-empty pair slots (one bit per pair_idx). Allows
    /// O(m + n^2/64) iteration over occupied pairs without reading each
    /// SmallVec header.
    nonempty_bits: Vec<u64>,
    /// Current number of active lineages.
    n: usize,
    /// Max capacity (determines pair_idx mapping).
    capacity: usize,
}

#[inline(always)]
fn bit_set(bits: &mut [u64], i: usize) {
    bits[i >> 6] |= 1u64 << (i & 63);
}
#[inline(always)]
fn bit_clear(bits: &mut [u64], i: usize) {
    bits[i >> 6] &= !(1u64 << (i & 63));
}
#[inline(always)]
fn bit_get(bits: &[u64], i: usize) -> bool {
    (bits[i >> 6] >> (i & 63)) & 1 != 0
}

#[inline(always)]
fn nbits_words(n_bits: usize) -> usize { (n_bits + 63) / 64 }

impl RateCache {
    pub fn new(max_lineages: usize) -> Self {
        let cap = max_lineages;
        let n_pairs = tri_size(cap);
        Self {
            overlaps: vec![SmallVec::new(); n_pairs],
            nonempty_bits: vec![0u64; nbits_words(n_pairs)],
            n: 0,
            capacity: cap,
        }
    }

    fn ensure_capacity(&mut self, need: usize) {
        if need > self.capacity {
            self.capacity = need * 2;
            let n_pairs = tri_size(self.capacity);
            self.overlaps.resize(n_pairs, SmallVec::new());
            self.nonempty_bits.resize(nbits_words(n_pairs), 0u64);
        }
    }

    /// Build the full cache from scratch. O(n^2 * segments).
    pub fn rebuild(
        &mut self,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.n = active.len();
        self.ensure_capacity(self.n);
        // Clear all entries.
        for entry in &mut self.overlaps {
            entry.clear();
        }
        for w in &mut self.nonempty_bits {
            *w = 0;
        }
        // Compute all pairs.
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if active[i].population != active[j].population {
                    continue;
                }
                let ovl = compute_overlap(active[i].head, active[j].head, arena);
                if !ovl.is_empty() {
                    let pidx = pair_idx(i, j, self.capacity);
                    self.overlaps[pidx] = ovl;
                    bit_set(&mut self.nonempty_bits, pidx);
                }
            }
        }
    }

    /// Recompute all pairs involving lineage `idx`. O(n * segments).
    /// Call this after a lineage at `idx` changes (new lineage placed
    /// at `idx` after coalescence/recombination).
    pub fn recompute_for(
        &mut self,
        idx: usize,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.n = active.len();
        self.ensure_capacity(self.n);
        for other in 0..self.n {
            if other == idx { continue; }
            let (i, j) = if other < idx { (other, idx) } else { (idx, other) };
            let pidx = pair_idx(i, j, self.capacity);
            self.overlaps[pidx].clear();
            bit_clear(&mut self.nonempty_bits, pidx);
            if active[i].population != active[j].population {
                continue;
            }
            let ovl = compute_overlap(active[i].head, active[j].head, arena);
            if !ovl.is_empty() {
                self.overlaps[pidx] = ovl;
                bit_set(&mut self.nonempty_bits, pidx);
            }
        }
    }

    /// Remove all pairs involving lineage `idx`. Call before removing
    /// the lineage from active.
    pub fn remove_lineage(&mut self, idx: usize) {
        for other in 0..self.n {
            if other == idx { continue; }
            let (i, j) = if other < idx { (other, idx) } else { (idx, other) };
            let pidx = pair_idx(i, j, self.capacity);
            self.overlaps[pidx].clear();
            bit_clear(&mut self.nonempty_bits, pidx);
        }
    }

    /// After `active.swap_remove(idx)`, the last lineage moved to `idx`.
    /// Patch cache entries: old references to `last` become `idx`.
    pub fn swap_update(&mut self, removed_idx: usize, old_last: usize) {
        if removed_idx == old_last {
            self.n -= 1;
            return;
        }
        // Callers must have invoked `remove_lineage(removed_idx)` first,
        // so every (removed_idx, *) slot is already empty with bit = 0.
        // Move (old_last, *) overlap data into those slots.
        for other in 0..old_last {
            if other == removed_idx { continue; }
            let old_pidx = pair_idx(
                other.min(old_last), other.max(old_last), self.capacity);
            let new_pidx = pair_idx(
                other.min(removed_idx), other.max(removed_idx), self.capacity);
            let was_nonempty = bit_get(&self.nonempty_bits, old_pidx);
            let data = std::mem::take(&mut self.overlaps[old_pidx]);
            bit_clear(&mut self.nonempty_bits, old_pidx);
            self.overlaps[new_pidx] = data;
            if was_nonempty {
                bit_set(&mut self.nonempty_bits, new_pidx);
            }
        }
        self.n -= 1;
    }

    /// Get the overlap for pair (i, j).
    pub fn get_pair(&self, i: usize, j: usize) -> &PairOverlap {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        &self.overlaps[pair_idx(a, b, self.capacity)]
    }

    /// Iterate all non-empty pairs using the bitmap at word granularity:
    /// for each row we load 64-bit chunks and use `trailing_zeros` to
    /// step directly to the next set bit. Empty words cost a single
    /// load + compare.
    pub fn iter_pairs(&self) -> NonEmptyPairIter<'_> {
        let mut it = NonEmptyPairIter {
            cache: self,
            row: 0,
            base_pidx: 0,
            row_end_pidx: 0,
            pidx_word: 0,
            bits: 0,
            done: false,
        };
        it.prime_row(0);
        it
    }
}

pub struct NonEmptyPairIter<'a> {
    cache: &'a RateCache,
    row: usize,
    // pair_idx(row, row+1, cap); base of current row's bit range.
    base_pidx: usize,
    // Exclusive end of current row's pair_idx range.
    row_end_pidx: usize,
    pidx_word: usize,
    // Remaining set bits in the currently-loaded word, masked to the
    // current row's range. Cleared bits are the ones already yielded.
    bits: u64,
    done: bool,
}

impl<'a> NonEmptyPairIter<'a> {
    fn prime_row(&mut self, row: usize) {
        let n = self.cache.n;
        if row + 1 >= n {
            self.done = true;
            return;
        }
        let cap = self.cache.capacity;
        self.row = row;
        self.base_pidx = pair_idx(row, row + 1, cap);
        self.row_end_pidx = self.base_pidx + (n - row - 1);
        self.pidx_word = self.base_pidx >> 6;
        self.load_current_word_masked();
    }

    #[inline]
    fn load_current_word_masked(&mut self) {
        let word_start = self.pidx_word << 6;
        if word_start >= self.row_end_pidx {
            self.bits = 0;
            return;
        }
        let raw = self.cache.nonempty_bits
            .get(self.pidx_word).copied().unwrap_or(0);
        // Mask off bits before row start (only relevant on first word).
        let lo_mask = if self.base_pidx > word_start {
            !((1u64 << (self.base_pidx - word_start)) - 1)
        } else { !0u64 };
        // Mask off bits past row end (only relevant on last word).
        let end_offset = self.row_end_pidx - word_start;
        let hi_mask = if end_offset >= 64 { !0u64 }
            else { (1u64 << end_offset) - 1 };
        self.bits = raw & lo_mask & hi_mask;
    }

    #[inline]
    fn advance_word(&mut self) -> bool {
        self.pidx_word += 1;
        let word_start = self.pidx_word << 6;
        if word_start >= self.row_end_pidx {
            return false;
        }
        self.load_current_word_masked();
        true
    }
}

impl<'a> Iterator for NonEmptyPairIter<'a> {
    type Item = (usize, usize, &'a PairOverlap);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done { return None; }
            if self.bits != 0 {
                let bit = self.bits.trailing_zeros() as usize;
                self.bits &= self.bits - 1;
                let pidx = (self.pidx_word << 6) + bit;
                let j = self.row + 1 + (pidx - self.base_pidx);
                let i = self.row;
                return Some((i, j, &self.cache.overlaps[pidx]));
            }
            if self.advance_word() {
                continue;
            }
            // Next row.
            self.prime_row(self.row + 1);
        }
    }
}

/// Compute overlap-by-class between two segment chains.
/// Same logic as the existing `overlap_by_class` but returns SmallVec.
fn compute_overlap(
    head_a: crate::segment::SegIdx,
    head_b: crate::segment::SegIdx,
    arena: &SegmentArena,
) -> PairOverlap {
    use crate::segment::SEG_NIL;
    let mut result = PairOverlap::new();
    let mut sa = head_a;
    let mut sb = head_b;
    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let b = arena.get(sb);
        if a.right <= b.left { sa = a.next; continue; }
        if b.right <= a.left { sb = b.next; continue; }
        let l = a.left.max(b.left);
        let r = a.right.min(b.right);
        if r > l && a.branch_class == b.branch_class {
            let cls = a.branch_class;
            if let Some(entry) = result.iter_mut().find(|(c, _)| *c == cls) {
                entry.1 += r - l;
            } else {
                result.push((cls, r - l));
            }
        }
        if a.right < b.right { sa = a.next; } else { sb = b.next; }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_tag::BranchClass;
    use crate::segment::SegmentArena;

    #[test]
    fn pair_idx_mapping() {
        // For n=4: pairs (0,1)(0,2)(0,3)(1,2)(1,3)(2,3) = indices 0..6
        assert_eq!(pair_idx(0, 1, 4), 0);
        assert_eq!(pair_idx(0, 2, 4), 1);
        assert_eq!(pair_idx(0, 3, 4), 2);
        assert_eq!(pair_idx(1, 2, 4), 3);
        assert_eq!(pair_idx(1, 3, 4), 4);
        assert_eq!(pair_idx(2, 3, 4), 5);
        assert_eq!(tri_size(4), 6);
    }

    #[test]
    fn rebuild_and_query() {
        let mut arena = SegmentArena::new();
        let cls = BranchClass::PANMICTIC;

        // Two lineages: [0,100) each, same class, same pop.
        let s0 = arena.alloc(0.0, 100.0, 0, cls);
        let s1 = arena.alloc(0.0, 100.0, 1, cls);
        let lin0 = Lineage::new(s0, s0, 0, 0, &arena);
        let lin1 = Lineage::new(s1, s1, 0, 1, &arena);
        let active = vec![lin0, lin1];

        let mut cache = RateCache::new(10);
        cache.rebuild(&active, &arena);

        let ovl = cache.get_pair(0, 1);
        assert_eq!(ovl.len(), 1);
        assert_eq!(ovl[0].0, cls);
        assert!((ovl[0].1 - 100.0).abs() < 1e-9);
    }
}
