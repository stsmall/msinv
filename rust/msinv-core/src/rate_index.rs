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

/// Reverse-index entry: (bucket_slot, pos_in_bucket). Parallel to
/// overlaps[pidx] order — overlaps[pidx][k].0 (class) is the class
/// whose (pop, class) bucket is pair_buckets[refs[pidx][k].0], and
/// refs[pidx][k].1 is that pair's position in the bucket's Vec.
type PairBucketRefs = SmallVec<[(u32, u32); 2]>;

/// Pack (i, j) lineage indices into a single u32 for dense bucket storage.
/// Max n = 65535 lineages per pop, far above any realistic sample size.
#[inline]
pub fn pack_ij(i: usize, j: usize) -> u32 {
    debug_assert!(i < 65536 && j < 65536);
    (i as u32) | ((j as u32) << 16)
}

#[inline]
pub fn unpack_ij(packed: u32) -> (usize, usize) {
    ((packed & 0xFFFF) as usize, (packed >> 16) as usize)
}

/// Flat per-lineage segment view: (left, right, class). Stored contiguously
/// per lineage so `compute_overlap`'s two-pointer walk hits sequential
/// memory instead of chasing arena indices scattered by free-list recycling.
pub type FlatSeg = (f64, f64, BranchClass);
type LineageSegs = SmallVec<[FlatSeg; 4]>;

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
    /// `active` vector. Needed so pair-bucket inserts on per-pair
    /// updates can attribute overlap to the right (pop, class) bucket.
    lineage_pop: Vec<u32>,
    /// Per-lineage flat segment list: materialised copy of each lineage's
    /// linked segment chain in contiguous memory. `compute_overlap` reads
    /// these slices instead of walking the arena, which killed ~15% of
    /// wall time at rho=2000 to arena random-index reads.
    lineage_segs: Vec<LineageSegs>,
    /// Per-lineage 64-bin positional bitmap. Bin `b` is set iff some
    /// segment covers any position in [b * seq_len / 64, (b+1) * seq_len
    /// / 64). Two lineages whose bitmaps AND to 0 share no positional
    /// overlap, so `compute_overlap` would return empty — skip it. Hull
    /// prescreen already filters the worst cases; this catches the
    /// common rho ≥ 16000 pattern where both hulls span the full
    /// sequence via a few scattered fragments that don't actually
    /// intersect.
    lineage_pos_bits: Vec<u64>,
    /// Sequence length. Used to compute `lineage_pos_bits` bin indices.
    /// Must stay stable across rebuilds/updates so bit AND is valid.
    seq_len: f64,
    /// Per-(pop, class) bucket of pair_idx packed (i, j) entries.
    /// `bucket.len()` doubles as the (pop, class) pair count that the
    /// coalescence-rate aggregator needs, so no parallel class_totals
    /// table is required. CoalAggregate dispatch picks the kth pair in
    /// O(1) via direct indexing instead of walking iter_pairs until the
    /// target (the old ~15%-of-wall cost at rho=2000).
    pair_buckets: SmallVec<[(u32, BranchClass, Vec<u32>); 8]>,
    /// Reverse index per pair_idx: list of (bucket_slot, pos_in_bucket)
    /// entries, one per class stored in overlaps[pidx], in the same
    /// order. Used to patch bucket positions during swap_remove and to
    /// rewrite packed (i, j) during swap_update.
    pair_bucket_refs: Vec<PairBucketRefs>,
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

/// Allocate `Vec<SmallVec<[T; N]>>` of length `n` using calloc-backed
/// zero-init instead of `vec![SmallVec::new(); n]`'s per-element clone
/// loop. For SmallVec 1.13 the struct layout is `{ capacity: usize,
/// data: union { inline: MaybeUninit<[T; N]>, heap: (ptr, cap) } }`.
/// `capacity == 0` selects the inline variant with length 0, and the
/// inline payload is `MaybeUninit::uninit()` (all bit patterns valid),
/// so all-zero bytes represent a valid empty inline SmallVec.
///
/// Cuts the ~500ms first-rep ensure_capacity cost observed on the
/// 4096-cap (~8.4M-slot) triangular arrays — the `extend_with` /
/// `SmallVec::clone` chain that appeared twice in the post-compact
/// flame (combined ~5% of wall at rho=2000).
fn zeroed_smallvec_vec<T, const N: usize>(n: usize) -> Vec<SmallVec<[T; N]>>
where
    [T; N]: smallvec::Array<Item = T>,
{
    use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
    if n == 0 {
        return Vec::new();
    }
    let layout = Layout::array::<SmallVec<[T; N]>>(n)
        .expect("SmallVec Vec allocation size overflow");
    // SAFETY: SmallVec<[T; N]> with all-zero bytes is
    // `SmallVec { capacity: 0, data: union-inline-uninit }`, which is
    // a valid empty inline SmallVec. capacity == 0 flags inline
    // storage; the inline payload is MaybeUninit so any bytes (zeroes
    // included) are a valid representation. Length is stored in
    // `capacity` and is 0, so no uninitialised elements will ever be
    // read. Safe only for SmallVec 1.13 — if smallvec is upgraded,
    // re-audit this path.
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut SmallVec<[T; N]>;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, n, n)
    }
}

impl RateCache {
    pub fn new(max_lineages: usize, seq_len: f64) -> Self {
        let cap = max_lineages;
        let n_pairs = tri_size(cap);
        Self {
            overlaps: zeroed_smallvec_vec(n_pairs),
            nonempty_bits: vec![0u64; nbits_words(n_pairs)],
            lineage_pop: Vec::with_capacity(cap),
            lineage_segs: Vec::with_capacity(cap),
            lineage_pos_bits: Vec::with_capacity(cap),
            seq_len,
            pair_buckets: SmallVec::new(),
            pair_bucket_refs: zeroed_smallvec_vec(n_pairs),
            n: 0,
            capacity: cap,
        }
    }

    /// Reset this cache for a new simulation while keeping all heap
    /// allocations. Subsequent `rebuild` populates the cleared state;
    /// `ensure_capacity` grows if `max_lineages` exceeds current cap.
    /// Used by pooled callers (e.g. multi-rep benches / ABC drivers)
    /// so the large triangular `overlaps` Vec is allocated once per
    /// process rather than per rep.
    pub fn reset(&mut self, max_lineages: usize, seq_len: f64) {
        self.seq_len = seq_len;
        self.n = 0;
        for entry in self.pair_buckets.iter_mut() {
            entry.2.clear();
        }
        self.lineage_pop.clear();
        self.lineage_segs.clear();
        self.lineage_pos_bits.clear();
        // Walk only non-empty overlap slots via the bitmap — skips
        // the O(n_pairs) sweep over the 300k+ typically-empty
        // triangular array that dominated reset cost at rho=2000.
        for (widx, w) in self.nonempty_bits.iter_mut().enumerate() {
            let mut word = *w;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let pidx = widx * 64 + bit;
                if pidx < self.overlaps.len() {
                    self.overlaps[pidx].clear();
                    if pidx < self.pair_bucket_refs.len() {
                        self.pair_bucket_refs[pidx].clear();
                    }
                }
                word &= word - 1;
            }
            *w = 0;
        }
        self.ensure_capacity(max_lineages);
    }

    /// Compute the 64-bin positional bitmap for a segment list.
    /// Bin b = floor(pos / seq_len * 64). Segments spanning multiple
    /// bins set all covered bins (floor(l)..=ceil(r)-1 clamped to 63).
    #[inline]
    fn seg_bits_for(seq_len: f64, segs: &[FlatSeg]) -> u64 {
        if seq_len <= 0.0 || segs.is_empty() { return !0u64; }
        let inv = 64.0 / seq_len;
        let mut bits = 0u64;
        for (l, r, _) in segs {
            let bl = (*l * inv).floor() as i64;
            let br = (*r * inv).ceil() as i64;
            let bl = bl.max(0) as u64;
            let br = br.min(64).max(0) as u64;
            if bl >= br { continue; }
            let span = br - bl;
            let mask = if span >= 64 { !0u64 }
                else { ((1u64 << span) - 1) << bl };
            bits |= mask;
        }
        bits
    }

    /// Grow capacity and reindex existing pair data. `pair_idx` layout
    /// is capacity-dependent, so a raw `Vec::resize` invalidates every
    /// previously-stored pidx — old overlaps land at positions that now
    /// encode different (i, j) under the new mapping. Reindex-in-place
    /// walks only the `walk_n × walk_n` triangle actually populated
    /// under old capacity, moving each non-empty slot to its new pidx.
    /// Amortised O(1) per slot across the geometric doubling schedule;
    /// far cheaper than re-running `compute_overlap` on every pair.
    fn ensure_capacity(&mut self, need: usize) {
        if need > self.capacity {
            let old_cap = self.capacity;
            let new_cap = need * 2;
            let new_n_pairs = tri_size(new_cap);
            let mut new_overlaps: Vec<PairOverlap> =
                zeroed_smallvec_vec(new_n_pairs);
            let mut new_bits = vec![0u64; nbits_words(new_n_pairs)];
            let mut new_refs: Vec<PairBucketRefs> =
                zeroed_smallvec_vec(new_n_pairs);
            let walk_n = self.n.min(old_cap);
            for i in 0..walk_n {
                for j in (i + 1)..walk_n {
                    let old_pidx = pair_idx(i, j, old_cap);
                    if bit_get(&self.nonempty_bits, old_pidx) {
                        let new_pidx = pair_idx(i, j, new_cap);
                        new_overlaps[new_pidx] =
                            std::mem::take(&mut self.overlaps[old_pidx]);
                        new_refs[new_pidx] =
                            std::mem::take(&mut self.pair_bucket_refs[old_pidx]);
                        bit_set(&mut new_bits, new_pidx);
                    }
                }
            }
            self.overlaps = new_overlaps;
            self.nonempty_bits = new_bits;
            self.pair_bucket_refs = new_refs;
            self.capacity = new_cap;
            // Bucket entries' packed (i, j) are capacity-independent,
            // so pair_buckets need no remap.
        }
    }

    /// Materialise lineage `idx`'s segment chain into `lineage_segs[idx]`.
    /// Clear-then-extend to reuse the SmallVec's allocation. Callers must
    /// invoke this whenever `idx`'s segment list changes (coalescence,
    /// recombination split, gene flux mutation).
    fn rebuild_lineage_segs(
        &mut self,
        idx: usize,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        use crate::segment::SEG_NIL;
        if self.lineage_segs.len() <= idx {
            self.lineage_segs.resize_with(idx + 1, SmallVec::new);
        }
        if self.lineage_pos_bits.len() <= idx {
            self.lineage_pos_bits.resize(idx + 1, 0u64);
        }
        let slot = &mut self.lineage_segs[idx];
        slot.clear();
        let mut sa = active[idx].head;
        while sa != SEG_NIL {
            let s = arena.get(sa);
            slot.push((s.left, s.right, s.branch_class));
            sa = s.next;
        }
        self.lineage_pos_bits[idx] = Self::seg_bits_for(self.seq_len, slot);
    }

    /// Hull [left, right] derived from the flat segs; returns an empty
    /// interval sentinel when the lineage has no segments.
    #[inline]
    fn hull_from_segs(segs: &[FlatSeg]) -> (f64, f64) {
        if segs.is_empty() {
            (f64::INFINITY, f64::NEG_INFINITY)
        } else {
            (segs[0].0, segs[segs.len() - 1].1)
        }
    }

    /// Locate or allocate the `pair_buckets` slot for (pop, class).
    /// Slots are append-only; slot index is therefore stable across a
    /// simulation and can be recorded in `pair_bucket_refs` without
    /// needing to patch on later inserts.
    fn find_or_create_pair_bucket_slot(
        &mut self, pop: u32, cls: BranchClass) -> usize {
        for (k, entry) in self.pair_buckets.iter().enumerate() {
            if entry.0 == pop && entry.1 == cls { return k; }
        }
        let k = self.pair_buckets.len();
        self.pair_buckets.push((pop, cls, Vec::new()));
        k
    }

    /// Swap-remove `pos` from bucket `slot` and patch the reverse index
    /// of whatever pair got moved in (if any).
    fn bucket_swap_remove(&mut self, slot: usize, pos: usize) {
        let bucket = &mut self.pair_buckets[slot].2;
        let last = bucket.len() - 1;
        if pos < last {
            let moved_packed = bucket[last];
            bucket[pos] = moved_packed;
            bucket.pop();
            let (mi, mj) = unpack_ij(moved_packed);
            let moved_pidx = pair_idx(mi, mj, self.capacity);
            for e in self.pair_bucket_refs[moved_pidx].iter_mut() {
                if e.0 == slot as u32 && e.1 == last as u32 {
                    e.1 = pos as u32;
                    return;
                }
            }
            debug_assert!(false, "bucket_swap_remove: moved pidx ref missing");
        } else {
            bucket.pop();
        }
    }

    /// Store a fully-computed pair overlap into the cache: writes
    /// `overlaps[pidx]`, sets the nonempty bit, and pushes packed
    /// (i, j) into each matching (pop, class) bucket with matching
    /// reverse-index entries in `pair_bucket_refs[pidx]`. Bucket
    /// lengths double as the (pop, class) pair counts consumed by
    /// `emit_coal_events_from_cache`, so no extra totals bookkeeping.
    fn store_pair(&mut self, i: usize, j: usize, ovl: PairOverlap) {
        let pidx = pair_idx(i, j, self.capacity);
        let pop = self.lineage_pop[i];
        debug_assert!(!bit_get(&self.nonempty_bits, pidx));
        debug_assert!(self.pair_bucket_refs[pidx].is_empty());
        let packed = pack_ij(i, j);
        for (cls, _) in ovl.iter() {
            let slot = self.find_or_create_pair_bucket_slot(pop, *cls);
            let pos = self.pair_buckets[slot].2.len() as u32;
            self.pair_buckets[slot].2.push(packed);
            self.pair_bucket_refs[pidx].push((slot as u32, pos));
        }
        self.overlaps[pidx] = ovl;
        bit_set(&mut self.nonempty_bits, pidx);
    }

    /// Clear pair (i, j): swap-remove each bucket entry via the reverse
    /// index, empty overlaps, clear bit. No-op if already empty. Bucket
    /// length decrement is the only count maintenance needed.
    fn clear_pair(&mut self, i: usize, j: usize) {
        let pidx = pair_idx(i, j, self.capacity);
        if !bit_get(&self.nonempty_bits, pidx) { return; }
        let refs = std::mem::take(&mut self.pair_bucket_refs[pidx]);
        for (slot, pos) in refs.iter().copied() {
            self.bucket_swap_remove(slot as usize, pos as usize);
        }
        self.overlaps[pidx].clear();
        bit_clear(&mut self.nonempty_bits, pidx);
    }

    /// Move pair data from (old_i, old_j) to (new_i, new_j) without
    /// recomputing overlap. totals unchanged (same classes); packed
    /// (i, j) in each referenced bucket entry is rewritten.
    /// Precondition: old slot nonempty, new slot empty.
    fn move_pair(
        &mut self,
        old_i: usize, old_j: usize,
        new_i: usize, new_j: usize,
    ) {
        let old_pidx = pair_idx(old_i, old_j, self.capacity);
        let new_pidx = pair_idx(new_i, new_j, self.capacity);
        debug_assert!(bit_get(&self.nonempty_bits, old_pidx));
        debug_assert!(!bit_get(&self.nonempty_bits, new_pidx));
        let data = std::mem::take(&mut self.overlaps[old_pidx]);
        bit_clear(&mut self.nonempty_bits, old_pidx);
        self.overlaps[new_pidx] = data;
        bit_set(&mut self.nonempty_bits, new_pidx);
        let refs = std::mem::take(&mut self.pair_bucket_refs[old_pidx]);
        let new_packed = pack_ij(new_i, new_j);
        for (slot, pos) in refs.iter().copied() {
            self.pair_buckets[slot as usize].2[pos as usize] = new_packed;
        }
        self.pair_bucket_refs[new_pidx] = refs;
    }

    /// Iterate per-(pop, class) pair counts as f64. Count equals the
    /// size of the (pop, class) bucket — no separate totals table.
    pub fn iter_class_totals(
        &self,
    ) -> impl Iterator<Item = (u32, BranchClass, f64)> + '_ {
        self.pair_buckets.iter()
            .map(|e| (e.0, e.1, e.2.len() as f64))
    }

    /// Build the full cache from scratch. O(n^2 * segments).
    pub fn rebuild(
        &mut self,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.n = active.len();
        self.ensure_capacity(self.n);
        // Clear only populated slots via the bitmap — the full triangular
        // sweep was ~17% self-time at rho=2000 because max_lins is pre-
        // sized to 2048 (→ 2M pair slots) and rebuild fires on every
        // cache_dirty transition.
        for (widx, w) in self.nonempty_bits.iter_mut().enumerate() {
            let mut word = *w;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let pidx = widx * 64 + bit;
                if pidx < self.overlaps.len() {
                    self.overlaps[pidx].clear();
                    if pidx < self.pair_bucket_refs.len() {
                        self.pair_bucket_refs[pidx].clear();
                    }
                }
                word &= word - 1;
            }
            *w = 0;
        }
        for entry in self.pair_buckets.iter_mut() {
            entry.2.clear();
        }
        self.lineage_pop.clear();
        self.lineage_pop.extend(active.iter().map(|l| l.population));
        if self.lineage_segs.len() < self.n {
            self.lineage_segs.resize_with(self.n, SmallVec::new);
        } else {
            self.lineage_segs.truncate(self.n);
        }
        if self.lineage_pos_bits.len() < self.n {
            self.lineage_pos_bits.resize(self.n, 0u64);
        } else {
            self.lineage_pos_bits.truncate(self.n);
        }
        for i in 0..self.n {
            self.rebuild_lineage_segs(i, active, arena);
        }
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if active[i].population != active[j].population {
                    continue;
                }
                if self.lineage_pos_bits[i] & self.lineage_pos_bits[j] == 0 {
                    continue;
                }
                let ovl = compute_overlap(
                    &self.lineage_segs[i], &self.lineage_segs[j]);
                if !ovl.is_empty() {
                    self.store_pair(i, j, ovl);
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
        if self.lineage_segs.len() < self.n {
            self.lineage_segs.resize_with(self.n, SmallVec::new);
        }
        if self.lineage_pos_bits.len() < self.n {
            self.lineage_pos_bits.resize(self.n, 0u64);
        }
        self.lineage_pop[idx] = active[idx].population;
        // Refresh the flat segment view for `idx`. Callers invoke
        // recompute_for after any mutation to `idx`'s chain.
        self.rebuild_lineage_segs(idx, active, arena);
        let (changed_hull_l, changed_hull_r) =
            Self::hull_from_segs(&self.lineage_segs[idx]);

        let changed_pop = active[idx].population;
        let changed_bits = self.lineage_pos_bits[idx];
        let cap = self.capacity;
        let n = self.n;
        // Step 1: clear every currently-nonempty (idx, *) pair via bitmap
        // row + column walk. Old classes get removed from buckets;
        // overlaps[pidx] emptied.
        if idx + 1 < n {
            let base_row = pair_idx(idx, idx + 1, cap);
            let row_end = base_row + (n - idx - 1);
            let mut w = base_row >> 6;
            let words_end = (row_end + 63) >> 6;
            while w < words_end {
                let word_start = w << 6;
                let raw = self.nonempty_bits[w];
                let lo = if base_row > word_start {
                    !((1u64 << (base_row - word_start)) - 1)
                } else { !0u64 };
                let end_off = row_end - word_start;
                let hi = if end_off >= 64 { !0u64 }
                    else { (1u64 << end_off) - 1 };
                let mut bits = raw & lo & hi;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let pidx = word_start + b;
                    let j = idx + 1 + (pidx - base_row);
                    self.clear_pair(idx, j);
                }
                w += 1;
            }
        }
        for i in 0..idx {
            let pidx = pair_idx(i, idx, cap);
            if bit_get(&self.nonempty_bits, pidx) {
                self.clear_pair(i, idx);
            }
        }
        // Step 2: compute new (idx, other) pairs. Skip same-pop / hull /
        // pos-bits prescreens like before but without the bit_get + pair_idx
        // + max/min overhead of the old interleaved path.
        for other in 0..n {
            if other == idx { continue; }
            let other_pop = self.lineage_pop[other];
            if other_pop != changed_pop { continue; }
            let other_segs: &[FlatSeg] = self.lineage_segs[other].as_slice();
            if other_segs.is_empty() { continue; }
            let (other_l, other_r) = Self::hull_from_segs(other_segs);
            if !(other_r > changed_hull_l && changed_hull_r > other_l) {
                continue;
            }
            let other_bits = self.lineage_pos_bits[other];
            if changed_bits & other_bits == 0 { continue; }
            let i = other.min(idx);
            let j = other.max(idx);
            let ovl = compute_overlap(
                &self.lineage_segs[i], &self.lineage_segs[j]);
            if !ovl.is_empty() {
                self.store_pair(i, j, ovl);
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
        // Refresh both halves' flat segment views before any compute_overlap.
        self.rebuild_lineage_segs(idx, active, arena);
        self.rebuild_lineage_segs(new_idx, active, arena);
        let (left_hull_l, left_hull_r) =
            Self::hull_from_segs(&self.lineage_segs[idx]);
        let (right_hull_l, right_hull_r) =
            Self::hull_from_segs(&self.lineage_segs[new_idx]);

        // Split can only shrink idx's hull (left/right halves are
        // subsets); empty old pairs stay empty. Iterate only nonempty
        // pairs (idx, other) via the bitmap: row walk for j > idx and
        // column walk for i < idx. new_pidx slot is empty by invariant
        // (new_idx was just pushed), so no defensive scrub needed.
        let cap = self.capacity;
        let n = self.n;
        let _ = changed_pop;
        if idx + 1 < n {
            let base_row = pair_idx(idx, idx + 1, cap);
            let row_end = base_row + (n - idx - 1);
            let mut w = base_row >> 6;
            let words_end = (row_end + 63) >> 6;
            while w < words_end {
                let word_start = w << 6;
                let raw = self.nonempty_bits[w];
                let lo = if base_row > word_start {
                    !((1u64 << (base_row - word_start)) - 1)
                } else { !0u64 };
                let end_off = row_end - word_start;
                let hi = if end_off >= 64 { !0u64 }
                    else { (1u64 << end_off) - 1 };
                let mut bits = raw & lo & hi;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let pidx = word_start + b;
                    let other = idx + 1 + (pidx - base_row);
                    if other == new_idx { continue; }
                    self.apply_recomb_split_body(
                        idx, other, new_idx, split_pos,
                        left_hull_l, left_hull_r,
                        right_hull_l, right_hull_r);
                }
                w += 1;
            }
        }
        for i in 0..idx {
            let pidx = pair_idx(i, idx, cap);
            if !bit_get(&self.nonempty_bits, pidx) { continue; }
            // i < idx < new_idx (new_idx was just pushed), so never hits new_idx.
            self.apply_recomb_split_body(
                idx, i, new_idx, split_pos,
                left_hull_l, left_hull_r,
                right_hull_l, right_hull_r);
        }
    }

    /// Per-pair handler for apply_recomb_split: called only for
    /// nonempty old pairs (idx, other). Dispatches Case A/B/C based on
    /// other's hull vs split_pos.
    #[inline]
    fn apply_recomb_split_body(
        &mut self,
        idx: usize, other: usize, new_idx: usize, split_pos: f64,
        left_hull_l: f64, left_hull_r: f64,
        right_hull_l: f64, right_hull_r: f64,
    ) {
        let cap = self.capacity;
        let oi = other.min(idx);
        let oj = other.max(idx);
        let other_segs: &[FlatSeg] = self.lineage_segs
            .get(other).map(|s| s.as_slice()).unwrap_or(&[]);
        let (other_l, other_r) = Self::hull_from_segs(other_segs);
        let ni = other.min(new_idx);
        let nj = other.max(new_idx);
        debug_assert!(!bit_get(&self.nonempty_bits, pair_idx(ni, nj, cap)));

        // Case A: other entirely left of split_pos — old pair with the
        // left-half is unchanged; nothing to do.
        if other_r <= split_pos {
            return;
        }
        // Case B: other entirely right of split_pos — old overlap used
        // right-half only; move slot to (new_idx, other).
        if other_l >= split_pos {
            self.move_pair(oi, oj, ni, nj);
            return;
        }
        // Case C: other spans split_pos. Clear old; recompute each half.
        self.clear_pair(oi, oj);
        let left_bits = self.lineage_pos_bits
            .get(idx).copied().unwrap_or(0);
        let other_bits = self.lineage_pos_bits
            .get(other).copied().unwrap_or(0);
        if other_r > left_hull_l && left_hull_r > other_l
            && left_bits & other_bits != 0 {
            let ovl = compute_overlap(
                &self.lineage_segs[oi], &self.lineage_segs[oj]);
            if !ovl.is_empty() {
                self.store_pair(oi, oj, ovl);
            }
        }
        let right_bits = self.lineage_pos_bits
            .get(new_idx).copied().unwrap_or(0);
        if other_r > right_hull_l && right_hull_r > other_l
            && right_bits & other_bits != 0 {
            let ovl = compute_overlap(
                &self.lineage_segs[ni], &self.lineage_segs[nj]);
            if !ovl.is_empty() {
                self.store_pair(ni, nj, ovl);
            }
        }
    }

    /// Remove all pairs involving lineage `idx`. Call before removing
    /// the lineage from active.
    pub fn remove_lineage(&mut self, idx: usize) {
        let cap = self.capacity;
        let n = self.n;
        // Row walk (j > idx): pidxs are contiguous from base_row to
        // base_row + (n - idx - 1); walk the bitmap word-wise so empty
        // 64-entry stretches cost one load + compare. trailing_zeros
        // jumps directly to each set bit.
        if idx + 1 < n {
            let base_row = pair_idx(idx, idx + 1, cap);
            let row_end = base_row + (n - idx - 1);
            let mut w = base_row >> 6;
            let words_end = (row_end + 63) >> 6;
            while w < words_end {
                let word_start = w << 6;
                let raw = self.nonempty_bits[w];
                let lo = if base_row > word_start {
                    !((1u64 << (base_row - word_start)) - 1)
                } else { !0u64 };
                let end_off = row_end - word_start;
                let hi = if end_off >= 64 { !0u64 }
                    else { (1u64 << end_off) - 1 };
                let mut bits = raw & lo & hi;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let pidx = word_start + b;
                    let j = idx + 1 + (pidx - base_row);
                    self.clear_pair(idx, j);
                }
                w += 1;
            }
        }
        // Column walk (i < idx): pidx(i, idx, cap) is scattered — one
        // per row. Linear scan checks bit per i; most empty so fast.
        for i in 0..idx {
            let pidx = pair_idx(i, idx, cap);
            if bit_get(&self.nonempty_bits, pidx) {
                self.clear_pair(i, idx);
            }
        }
    }

    /// After `active.swap_remove(idx)`, the last lineage moved to `idx`.
    /// Patch cache entries: old references to `last` become `idx`.
    pub fn swap_update(&mut self, removed_idx: usize, old_last: usize) {
        if removed_idx == old_last {
            if !self.lineage_pop.is_empty() {
                self.lineage_pop.pop();
            }
            if !self.lineage_segs.is_empty() {
                self.lineage_segs.pop();
            }
            if !self.lineage_pos_bits.is_empty() {
                self.lineage_pos_bits.pop();
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
        // other < old_last always, so (min, max) = (other, old_last).
        // Only removed_idx's position relative to other needs ordering.
        let cap = self.capacity;
        for other in 0..old_last {
            if other == removed_idx { continue; }
            let old_pidx = pair_idx(other, old_last, cap);
            if !bit_get(&self.nonempty_bits, old_pidx) { continue; }
            let (ni, nj) = if other < removed_idx {
                (other, removed_idx)
            } else {
                (removed_idx, other)
            };
            self.move_pair(other, old_last, ni, nj);
        }
        // Mirror the active-side swap_remove on lineage_pop so later
        // totals diffs see the right pop at `removed_idx`.
        if old_last < self.lineage_pop.len() {
            let moved_pop = self.lineage_pop[old_last];
            self.lineage_pop[removed_idx] = moved_pop;
            self.lineage_pop.pop();
        }
        // Same swap_remove for the flat segment view. Move allocation
        // from old_last into removed_idx instead of cloning.
        if old_last < self.lineage_segs.len() {
            let moved = std::mem::take(&mut self.lineage_segs[old_last]);
            self.lineage_segs[removed_idx] = moved;
            self.lineage_segs.pop();
        }
        if old_last < self.lineage_pos_bits.len() {
            let moved = self.lineage_pos_bits[old_last];
            self.lineage_pos_bits[removed_idx] = moved;
            self.lineage_pos_bits.pop();
        }
        self.n -= 1;
    }

    /// Get the overlap for pair (i, j).
    pub fn get_pair(&self, i: usize, j: usize) -> &PairOverlap {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        &self.overlaps[pair_idx(a, b, self.capacity)]
    }

    /// O(1) access to the (pop, class) pair bucket — packed (i, j) list.
    /// Returns an empty slice if no such bucket exists yet.
    /// CoalAggregate dispatch picks the kth pair directly from here,
    /// skipping the iter_pairs walk that dominated rho=2000 wall time.
    pub fn pair_bucket_for(
        &self, pop: u32, cls: BranchClass) -> &[u32] {
        for entry in self.pair_buckets.iter() {
            if entry.0 == pop && entry.1 == cls {
                return &entry.2;
            }
        }
        &[]
    }

    /// Flat segment slice for lineage `idx`. Empty slice if `idx` is out
    /// of range (caller should invoke `refresh_lineage_segs` beforehand
    /// for any lineage whose segs were not already maintained by an
    /// incremental path like `recompute_for` / `apply_recomb_split`).
    #[inline]
    pub fn lineage_segs(&self, idx: usize) -> &[FlatSeg] {
        self.lineage_segs.get(idx).map_or(&[][..], |v| v.as_slice())
    }

    /// Rebuild every lineage's flat-segs view (and its positional bitmap)
    /// without touching the pair-overlap cache. Cheap — O(n · avg_segs) —
    /// so the flux path can safely run it before `flux_rebuild_full`
    /// regardless of whether the overlap cache is about to be rebuilt.
    pub fn refresh_lineage_segs(
        &mut self,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        let n = active.len();
        if self.lineage_segs.len() < n {
            self.lineage_segs.resize_with(n, SmallVec::new);
        } else {
            self.lineage_segs.truncate(n);
        }
        if self.lineage_pos_bits.len() < n {
            self.lineage_pos_bits.resize(n, 0u64);
        } else {
            self.lineage_pos_bits.truncate(n);
        }
        for i in 0..n {
            self.rebuild_lineage_segs(i, active, arena);
        }
    }

    /// Refresh flat-segs view for a single lineage `idx`. Used by flux
    /// call sites that mutate `idx` without otherwise calling
    /// `recompute_for` / `apply_recomb_split` (e.g. CoalPanmicticPop,
    /// FluxAggregate paths where cache_dirty is set instead of
    /// incremental).
    pub fn rebuild_segs_for(
        &mut self,
        idx: usize,
        active: &[Lineage],
        arena: &SegmentArena,
    ) {
        self.rebuild_lineage_segs(idx, active, arena);
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
    // Yield the overlap as a plain slice — callers avoid repeated
    // SmallVec::spilled() branches when looping through classes.
    type Item = (usize, usize, &'a [(BranchClass, f64)]);

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
                return Some((i, j, self.cache.overlaps[pidx].as_slice()));
            }
            if self.advance_word() {
                continue;
            }
            // Next row.
            self.prime_row(self.row + 1);
        }
    }
}

/// Compute overlap-by-class between two lineages' flat segment slices.
/// Two-pointer walk over contiguous `(left, right, class)` tuples —
/// same algorithm as the arena-chain version, but without arena
/// random-access reads (the segments are recycled via free-list, so
/// their indices are scattered even when the owning lineage's list is
/// logically contiguous).
fn compute_overlap(a: &[FlatSeg], b: &[FlatSeg]) -> PairOverlap {
    let mut result = PairOverlap::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (al, ar, ac) = a[i];
        let (bl, br, bc) = b[j];
        if ar <= bl { i += 1; continue; }
        if br <= al { j += 1; continue; }
        let l = al.max(bl);
        let r = ar.min(br);
        if r > l && ac == bc {
            if let Some(entry) = result.iter_mut().find(|(c, _)| *c == ac) {
                entry.1 += r - l;
            } else {
                result.push((ac, r - l));
            }
        }
        if ar < br { i += 1; } else { j += 1; }
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

        let mut cache = RateCache::new(10, 100.0);
        cache.rebuild(&active, &arena);

        let ovl = cache.get_pair(0, 1);
        assert_eq!(ovl.len(), 1);
        assert_eq!(ovl[0].0, cls);
        assert!((ovl[0].1 - 100.0).abs() < 1e-9);
    }
}
