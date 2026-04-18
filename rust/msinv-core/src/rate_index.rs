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
    /// Per-lineage population; maintained in lockstep with the outer
    /// `active` vector. Needed so that class_totals diffs on per-pair
    /// updates can attribute overlap to the right (pop, class) bucket.
    lineage_pop: Vec<u32>,
    /// Dense (pop, class, pair_count) table. Counts the number of
    /// cached pairs whose overlap touches `class` in `pop`. Coalescence
    /// hazard between any two lineages in the same (pop, class) bucket
    /// is 1/(2*Ne*p_class), so the aggregate rate is just count × that.
    /// Maintained by increment/decrement whenever a pair's overlap
    /// entries change.
    class_totals: SmallVec<[(u32, BranchClass, f64); 8]>,
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
            lineage_pop: Vec::with_capacity(cap),
            class_totals: SmallVec::new(),
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

    /// Apply `delta` to the (pop, class) total; insert if new.
    #[inline]
    fn totals_add(&mut self, pop: u32, cls: BranchClass, delta: f64) {
        if delta == 0.0 { return; }
        for entry in self.class_totals.iter_mut() {
            if entry.0 == pop && entry.1 == cls {
                entry.2 += delta;
                return;
            }
        }
        self.class_totals.push((pop, cls, delta));
    }

    /// Rebuild class_totals from the authoritative per-pair data. Used
    /// periodically to bound any drift from subtle swap/migration
    /// ordering issues — totals-based rate emission only needs to stay
    /// close to truth to keep the waiting-time distribution correct.
    pub fn reconcile_class_totals(&mut self, active: &[Lineage]) {
        // Collect first to decouple from the immutable borrow on iter_pairs.
        let mut snapshot: SmallVec<[(u32, BranchClass); 8]> = SmallVec::new();
        for (i, _j, overlaps) in self.iter_pairs() {
            let pop = active[i].population;
            for (cls, _ov) in overlaps {
                snapshot.push((pop, *cls));
            }
        }
        self.class_totals.clear();
        for (pop, cls) in snapshot {
            self.totals_add(pop, cls, 1.0);
        }
    }

    /// Subtract pair (i, j)'s current class contributions from totals.
    /// Each stored (class, _) entry in the pair counts as one hazard
    /// slot in its (pop, class) bucket.
    fn totals_sub_pair(&mut self, i: usize, j: usize) {
        let pidx = pair_idx(i, j, self.capacity);
        if !bit_get(&self.nonempty_bits, pidx) { return; }
        let pop = self.lineage_pop[i];
        // Snapshot classes to avoid aliasing the borrow during totals_add.
        let classes: SmallVec<[BranchClass; 2]> =
            self.overlaps[pidx].iter().map(|(c, _)| *c).collect();
        for cls in classes {
            self.totals_add(pop, cls, -1.0);
        }
    }

    /// Iterate the maintained (pop, class, total_overlap) table.
    pub fn iter_class_totals(
        &self,
    ) -> impl Iterator<Item = (u32, BranchClass, f64)> + '_ {
        self.class_totals.iter().copied()
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
        self.class_totals.clear();
        self.lineage_pop.clear();
        self.lineage_pop.extend(active.iter().map(|l| l.population));
        // Compute all pairs.
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if active[i].population != active[j].population {
                    continue;
                }
                let ovl = compute_overlap(active[i].head, active[j].head, arena);
                if !ovl.is_empty() {
                    let pop = active[i].population;
                    for (cls, _ov) in ovl.iter() {
                        self.totals_add(pop, *cls, 1.0);
                    }
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
        if self.lineage_pop.len() < self.n {
            self.lineage_pop.resize(self.n, 0);
        }
        self.lineage_pop[idx] = active[idx].population;
        // Cache the changed lineage's hull — cheap extent bounds used
        // to skip the full segment walk when hulls don't intersect.
        let changed_head = active[idx].head;
        let changed_tail = active[idx].tail;
        let changed_hull_l = if changed_head == crate::segment::SEG_NIL {
            f64::INFINITY
        } else {
            arena.get(changed_head).left
        };
        let changed_hull_r = if changed_tail == crate::segment::SEG_NIL {
            f64::NEG_INFINITY
        } else {
            arena.get(changed_tail).right
        };

        let changed_pop = active[idx].population;
        for other in 0..self.n {
            if other == idx { continue; }
            let (i, j) = if other < idx { (other, idx) } else { (idx, other) };
            let pidx = pair_idx(i, j, self.capacity);
            let was_nonempty = bit_get(&self.nonempty_bits, pidx);
            let other_pop = self.lineage_pop.get(other).copied()
                .unwrap_or(active[other].population);
            let pops_match = changed_pop == other_pop;
            // Hull prescreen cost: two arena reads. Still cheap compared
            // to compute_overlap's segment walk.
            let (hulls_overlap, other_head_is_nil) = if !pops_match {
                (false, false)
            } else {
                let other_head = active[other].head;
                if other_head == crate::segment::SEG_NIL {
                    (false, true)
                } else {
                    let other_tail = active[other].tail;
                    let other_l = arena.get(other_head).left;
                    let other_r = arena.get(other_tail).right;
                    (other_r > changed_hull_l && changed_hull_r > other_l,
                     false)
                }
            };
            // Fast path: pair was empty and will stay empty — skip
            // the totals/overlap/bit dance entirely.
            if !was_nonempty && (!pops_match || !hulls_overlap || other_head_is_nil) {
                continue;
            }
            // Clear old.
            self.totals_sub_pair(i, j);
            self.overlaps[pidx].clear();
            bit_clear(&mut self.nonempty_bits, pidx);
            if !pops_match || !hulls_overlap { continue; }
            // Compute new overlap (pops match, hulls intersect).
            let ovl = compute_overlap(active[i].head, active[j].head, arena);
            if !ovl.is_empty() {
                for (cls, _ov) in ovl.iter() {
                    self.totals_add(changed_pop, *cls, 1.0);
                }
                self.overlaps[pidx] = ovl;
                bit_set(&mut self.nonempty_bits, pidx);
            }
        }
    }

    /// Specialised recomb update: split at `split_pos` on lineage
    /// `idx`, producing a new lineage at `new_idx` (= active.len()-1).
    ///
    /// For each other lineage we have three cases:
    /// * Other is entirely left of the split → its old pair with `idx`
    ///   used only left-of-split material and stays valid; new pair
    ///   with `new_idx` is guaranteed empty. No-op.
    /// * Other is entirely right of the split → old pair used only
    ///   right-of-split material; move the slot to (other, new_idx)
    ///   and clear the old slot. Totals unchanged.
    /// * Other spans the split → both halves can overlap; fall back to
    ///   the full recompute path for both rows.
    ///
    /// At rho ≥ 1000 the vast majority of `other`s are in the first
    /// two categories because most lineages hold only small fragments
    /// of the sequence. Avoids a full segment-walk for each non-
    /// spanning pair.
    pub fn apply_recomb_split(
        &mut self,
        idx: usize,
        new_idx: usize,
        split_pos: f64,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.n = active.len();
        self.ensure_capacity(self.n);
        if self.lineage_pop.len() < self.n {
            self.lineage_pop.resize(self.n, 0);
        }
        self.lineage_pop[idx] = active[idx].population;
        self.lineage_pop[new_idx] = active[new_idx].population;
        let changed_pop = active[idx].population;
        // Left (idx) and right (new_idx) hull extents for the fall-back
        // spanning-case computations.
        let left_head = active[idx].head;
        let left_tail = active[idx].tail;
        let left_hull_l = if left_head == crate::segment::SEG_NIL {
            f64::INFINITY
        } else { arena.get(left_head).left };
        let left_hull_r = if left_tail == crate::segment::SEG_NIL {
            f64::NEG_INFINITY
        } else { arena.get(left_tail).right };
        let right_head = active[new_idx].head;
        let right_tail = active[new_idx].tail;
        let right_hull_l = if right_head == crate::segment::SEG_NIL {
            f64::INFINITY
        } else { arena.get(right_head).left };
        let right_hull_r = if right_tail == crate::segment::SEG_NIL {
            f64::NEG_INFINITY
        } else { arena.get(right_tail).right };

        for other in 0..self.n {
            if other == idx || other == new_idx { continue; }
            let other_pop = self.lineage_pop[other];
            if other_pop != changed_pop {
                // Nothing to do — cross-pop pairs were never stored.
                continue;
            }
            let other_head = active[other].head;
            if other_head == crate::segment::SEG_NIL { continue; }
            let other_tail = active[other].tail;
            let other_l = arena.get(other_head).left;
            let other_r = arena.get(other_tail).right;

            // Old pair: (min(idx, other), max(idx, other)).
            let (oi, oj) = if other < idx { (other, idx) } else { (idx, other) };
            let old_pidx = pair_idx(oi, oj, self.capacity);
            // New-side pair slot: (min(new_idx, other), max(new_idx, other)).
            let (ni, nj) = if other < new_idx
                { (other, new_idx) } else { (new_idx, other) };
            let new_pidx = pair_idx(ni, nj, self.capacity);

            // Case A: other entirely left of split_pos.
            if other_r <= split_pos {
                // Old pair with the left-half is unchanged. `new_idx`
                // is the just-pushed slot, and the swap_update protocol
                // guarantees that every pair slot involving `new_idx`
                // was scrubbed before any subsequent push — so no stale
                // data lingers. Release builds skip the check entirely;
                // debug builds assert the invariant.
                debug_assert!(
                    !bit_get(&self.nonempty_bits, new_pidx),
                    "apply_recomb_split Case A: new_pidx should be empty",
                );
                continue;
            }

            // Case B: other entirely right of split_pos.
            if other_l >= split_pos {
                // Old pair content becomes the new pair (right-half
                // ∩ other = idx ∩ other). Move slot; totals unchanged.
                if bit_get(&self.nonempty_bits, old_pidx) {
                    let data = std::mem::take(&mut self.overlaps[old_pidx]);
                    bit_clear(&mut self.nonempty_bits, old_pidx);
                    // If new_pidx happened to be non-empty (unlikely
                    // but a defensive check for re-used slot history),
                    // clear its totals first.
                    if bit_get(&self.nonempty_bits, new_pidx) {
                        self.totals_sub_pair(ni, nj);
                        self.overlaps[new_pidx].clear();
                        bit_clear(&mut self.nonempty_bits, new_pidx);
                    }
                    self.overlaps[new_pidx] = data;
                    bit_set(&mut self.nonempty_bits, new_pidx);
                } else if bit_get(&self.nonempty_bits, new_pidx) {
                    // Old empty, new stale — scrub.
                    self.totals_sub_pair(ni, nj);
                    self.overlaps[new_pidx].clear();
                    bit_clear(&mut self.nonempty_bits, new_pidx);
                }
                continue;
            }

            // Case C: other spans split_pos. Recompute both halves.
            // Old slot (idx, other): clear, recompute with left half.
            if bit_get(&self.nonempty_bits, old_pidx) {
                self.totals_sub_pair(oi, oj);
                self.overlaps[old_pidx].clear();
                bit_clear(&mut self.nonempty_bits, old_pidx);
            }
            // Hull prescreen for left-half ∩ other.
            if other_r > left_hull_l && left_hull_r > other_l {
                let ovl = compute_overlap(
                    active[oi].head, active[oj].head, arena);
                if !ovl.is_empty() {
                    for (cls, _) in ovl.iter() {
                        self.totals_add(changed_pop, *cls, 1.0);
                    }
                    self.overlaps[old_pidx] = ovl;
                    bit_set(&mut self.nonempty_bits, old_pidx);
                }
            }
            // New slot (new_idx, other): clear, recompute with right half.
            if bit_get(&self.nonempty_bits, new_pidx) {
                self.totals_sub_pair(ni, nj);
                self.overlaps[new_pidx].clear();
                bit_clear(&mut self.nonempty_bits, new_pidx);
            }
            if other_r > right_hull_l && right_hull_r > other_l {
                let ovl = compute_overlap(
                    active[ni].head, active[nj].head, arena);
                if !ovl.is_empty() {
                    for (cls, _) in ovl.iter() {
                        self.totals_add(changed_pop, *cls, 1.0);
                    }
                    self.overlaps[new_pidx] = ovl;
                    bit_set(&mut self.nonempty_bits, new_pidx);
                }
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
            // Fast path: empty pairs don't contribute to totals and
            // have nothing to clear — skip them without touching memory.
            if !bit_get(&self.nonempty_bits, pidx) { continue; }
            self.totals_sub_pair(i, j);
            self.overlaps[pidx].clear();
            bit_clear(&mut self.nonempty_bits, pidx);
        }
    }

    /// After `active.swap_remove(idx)`, the last lineage moved to `idx`.
    /// Patch cache entries: old references to `last` become `idx`.
    pub fn swap_update(&mut self, removed_idx: usize, old_last: usize) {
        if removed_idx == old_last {
            if !self.lineage_pop.is_empty() {
                self.lineage_pop.pop();
            }
            self.n -= 1;
            return;
        }
        // Callers must have invoked `remove_lineage(removed_idx)` first,
        // so every (removed_idx, *) slot is already empty with bit = 0.
        // Move (old_last, *) overlap data into those slots.
        // Iterate the last row's bitmap at word granularity so we only
        // pay for nonempty pairs. Empty pairs need no slot move and no
        // bit maintenance, which is the common case at high rho.
        for other in 0..old_last {
            if other == removed_idx { continue; }
            let old_pidx = pair_idx(
                other.min(old_last), other.max(old_last), self.capacity);
            if !bit_get(&self.nonempty_bits, old_pidx) { continue; }
            let new_pidx = pair_idx(
                other.min(removed_idx), other.max(removed_idx), self.capacity);
            let data = std::mem::take(&mut self.overlaps[old_pidx]);
            bit_clear(&mut self.nonempty_bits, old_pidx);
            self.overlaps[new_pidx] = data;
            bit_set(&mut self.nonempty_bits, new_pidx);
        }
        // Mirror the active-side swap_remove on lineage_pop so later
        // totals diffs see the right pop at `removed_idx`.
        if old_last < self.lineage_pop.len() {
            let moved_pop = self.lineage_pop[old_last];
            self.lineage_pop[removed_idx] = moved_pop;
            self.lineage_pop.pop();
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
