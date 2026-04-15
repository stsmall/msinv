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
use crate::fenwick::Fenwick;
use crate::lineage::Lineage;
use crate::segment::SegmentArena;

use smallvec::SmallVec;

/// Per-pair overlap: list of (BranchClass, overlap_length) entries.
/// SmallVec avoids heap allocation for the common 1-2 class case.
type PairOverlap = SmallVec<[(BranchClass, f64); 4]>;

/// Flat index for a pair (i, j) where i < j, into a triangular array.
#[inline]
fn pair_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j && j < n);
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Number of entries in the triangular pair cache for n lineages.
#[inline]
fn tri_size(n: usize) -> usize {
    n * (n - 1) / 2
}

pub struct RateCache {
    /// Per-pair overlap cache. Indexed by pair_idx(i, j, capacity).
    overlaps: Vec<PairOverlap>,
    /// Current number of active lineages.
    n: usize,
    /// Max capacity (determines pair_idx mapping).
    capacity: usize,
}

impl RateCache {
    pub fn new(max_lineages: usize) -> Self {
        let cap = max_lineages;
        Self {
            overlaps: vec![SmallVec::new(); tri_size(cap)],
            n: 0,
            capacity: cap,
        }
    }

    /// Build the full cache from scratch. O(n^2 * segments).
    pub fn rebuild(
        &mut self,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.n = active.len();
        // Grow capacity if needed.
        if self.n > self.capacity {
            self.capacity = self.n * 2;
            self.overlaps.resize(tri_size(self.capacity), SmallVec::new());
        }
        // Clear all entries.
        for entry in &mut self.overlaps {
            entry.clear();
        }
        // Compute all pairs.
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if active[i].population != active[j].population {
                    continue;
                }
                let ovl = compute_overlap(active[i].head, active[j].head, arena);
                if !ovl.is_empty() {
                    self.overlaps[pair_idx(i, j, self.capacity)] = ovl;
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
        if self.n > self.capacity {
            self.capacity = self.n * 2;
            self.overlaps.resize(tri_size(self.capacity), SmallVec::new());
        }
        for other in 0..self.n {
            if other == idx { continue; }
            let (i, j) = if other < idx { (other, idx) } else { (idx, other) };
            let pidx = pair_idx(i, j, self.capacity);
            self.overlaps[pidx].clear();
            if active[i].population != active[j].population {
                continue;
            }
            let ovl = compute_overlap(active[i].head, active[j].head, arena);
            self.overlaps[pidx] = ovl;
        }
    }

    /// Remove all pairs involving lineage `idx`. Call before removing
    /// the lineage from active.
    pub fn remove_lineage(&mut self, idx: usize) {
        for other in 0..self.n {
            if other == idx { continue; }
            let (i, j) = if other < idx { (other, idx) } else { (idx, other) };
            self.overlaps[pair_idx(i, j, self.capacity)].clear();
        }
    }

    /// After `active.swap_remove(idx)`, the last lineage moved to `idx`.
    /// Patch cache entries: old references to `last` become `idx`.
    pub fn swap_update(&mut self, removed_idx: usize, old_last: usize) {
        if removed_idx == old_last {
            self.n -= 1;
            return;
        }
        // Clear all entries for `removed_idx` (already done by remove_lineage).
        // Copy entries for `old_last` → `removed_idx`.
        for other in 0..old_last {
            if other == removed_idx { continue; }
            // old pair: (other, old_last) or (old_last, other)
            let old_pidx = pair_idx(
                other.min(old_last), other.max(old_last), self.capacity);
            let new_pidx = pair_idx(
                other.min(removed_idx), other.max(removed_idx), self.capacity);
            let data = std::mem::take(&mut self.overlaps[old_pidx]);
            self.overlaps[new_pidx] = data;
        }
        self.n -= 1;
    }

    /// Get the overlap for pair (i, j).
    pub fn get_pair(&self, i: usize, j: usize) -> &PairOverlap {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        &self.overlaps[pair_idx(a, b, self.capacity)]
    }

    /// Iterate all non-empty pairs. Returns (i, j, &overlaps).
    pub fn iter_pairs(&self) -> impl Iterator<Item = (usize, usize, &PairOverlap)> {
        let cap = self.capacity;
        let n = self.n;
        (0..n).flat_map(move |i| {
            ((i + 1)..n).filter_map(move |j| {
                let pidx = pair_idx(i, j, cap);
                let ovl = &self.overlaps[pidx];
                if ovl.is_empty() { None } else { Some((i, j, ovl)) }
            })
        })
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
