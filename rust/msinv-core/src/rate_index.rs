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
    /// Per-lineage hull `(left, right)` — first segment's left and last
    /// segment's right, computed alongside `lineage_pos_bits` at every
    /// `rebuild_lineage_segs`. Direct-indexed, same size as `lineage_pop`.
    /// Lets the recompute_for step-2 prescreen read hulls without
    /// chasing the SmallVec header (spilled check + pointer-to-data
    /// dereference) — that chase was 45% of recompute_for self-time
    /// at rho=16000 via the `ucomisd (%rcx), %xmm0` hull-overlap branch.
    lineage_hulls: Vec<(f64, f64)>,
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
    /// Per-lineage peer bitmap. `peer_bit(i, j)` is set iff pair
    /// `(min(i,j), max(i,j))` has a cached overlap. Stored flat with
    /// `peer_word_stride` u64 words per lineage; `peer_bits[i *
    /// stride + w]` holds bits `64w..64w+64` of lineage `i`'s peer
    /// set. Replaces the `for other in 0..idx / 0..old_last`
    /// scattered-pidx column walks in remove_lineage /
    /// swap_update / recompute_for / apply_recomb_split with a
    /// `trailing_zeros` walk over set bits (O(peers) vs O(n)).
    peer_bits: Vec<u64>,
    peer_word_stride: usize,
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

/// Iterator over set bits in a single u64 word via `trailing_zeros`.
/// Matches the pattern used throughout the rate_index module — each
/// step clears the LSB (`word &= word - 1`) then returns the zero-
/// count of the previous word.
struct BitWordIter(u64);
impl Iterator for BitWordIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 { return None; }
        let b = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(b)
    }
}

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
        let peer_stride = nbits_words(cap);
        Self {
            overlaps: zeroed_smallvec_vec(n_pairs),
            nonempty_bits: vec![0u64; nbits_words(n_pairs)],
            lineage_pop: Vec::with_capacity(cap),
            lineage_segs: Vec::with_capacity(cap),
            lineage_pos_bits: Vec::with_capacity(cap),
            lineage_hulls: Vec::with_capacity(cap),
            seq_len,
            pair_buckets: SmallVec::new(),
            pair_bucket_refs: zeroed_smallvec_vec(n_pairs),
            peer_bits: vec![0u64; cap * peer_stride],
            peer_word_stride: peer_stride,
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
        self.lineage_hulls.clear();
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
        // peer_bits: full sweep of `cap * stride` u64 words. At
        // cap=4096 that's ~512KB = ~128k u64 memset — fast (<1ms).
        // Avoids tracking old_n just to target the per-lineage rows.
        for w in self.peer_bits.iter_mut() { *w = 0; }
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
            // Reindex peer_bits. Per-lineage layout: each active row
            // occupies `stride` u64 words. On cap grow, stride grows
            // too (nbits_words doubles with cap), so each row moves
            // to a new start offset. Bits within a row stay at the
            // same bit index (peer index j unchanged by cap).
            let old_stride = self.peer_word_stride;
            let new_stride = nbits_words(new_cap);
            let mut new_peer_bits = vec![0u64; new_cap * new_stride];
            for i in 0..walk_n {
                let src_start = i * old_stride;
                let dst_start = i * new_stride;
                new_peer_bits[dst_start..dst_start + old_stride]
                    .copy_from_slice(
                        &self.peer_bits[src_start..src_start + old_stride]);
            }
            self.overlaps = new_overlaps;
            self.nonempty_bits = new_bits;
            self.pair_bucket_refs = new_refs;
            self.peer_bits = new_peer_bits;
            self.peer_word_stride = new_stride;
            self.capacity = new_cap;
            // Bucket entries' packed (i, j) are capacity-independent,
            // so pair_buckets need no remap.
        }
    }

    /// Set the (i, j) and (j, i) peer bits — call from `store_pair`.
    #[inline(always)]
    fn peer_set_pair(&mut self, i: usize, j: usize) {
        let stride = self.peer_word_stride;
        let (iw, ib) = (j >> 6, 1u64 << (j & 63));
        self.peer_bits[i * stride + iw] |= ib;
        let (jw, jb) = (i >> 6, 1u64 << (i & 63));
        self.peer_bits[j * stride + jw] |= jb;
    }

    /// Clear the (i, j) and (j, i) peer bits — call from `clear_pair`.
    #[inline(always)]
    fn peer_clear_pair(&mut self, i: usize, j: usize) {
        let stride = self.peer_word_stride;
        let (iw, ib) = (j >> 6, 1u64 << (j & 63));
        self.peer_bits[i * stride + iw] &= !ib;
        let (jw, jb) = (i >> 6, 1u64 << (i & 63));
        self.peer_bits[j * stride + jw] &= !jb;
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
        if self.lineage_hulls.len() <= idx {
            self.lineage_hulls.resize(idx + 1,
                (f64::INFINITY, f64::NEG_INFINITY));
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
        self.lineage_hulls[idx] = if slot.is_empty() {
            (f64::INFINITY, f64::NEG_INFINITY)
        } else {
            (slot[0].0, slot[slot.len() - 1].1)
        };
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
        self.peer_set_pair(i, j);
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
        self.peer_clear_pair(i, j);
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
        // Keep peer_bits in sync: (old_i, old_j) is no longer a
        // nonempty pair; (new_i, new_j) is. Without this the peer
        // walks in recompute_for / apply_recomb_split / etc. would
        // visit stale peers.
        self.peer_clear_pair(old_i, old_j);
        self.peer_set_pair(new_i, new_j);
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
        if self.lineage_hulls.len() < self.n {
            self.lineage_hulls.resize(self.n,
                (f64::INFINITY, f64::NEG_INFINITY));
        } else {
            self.lineage_hulls.truncate(self.n);
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
        if self.lineage_hulls.len() < self.n {
            self.lineage_hulls.resize(self.n,
                (f64::INFINITY, f64::NEG_INFINITY));
        }
        self.lineage_pop[idx] = active[idx].population;
        // Refresh the flat segment view for `idx`. Callers invoke
        // recompute_for after any mutation to `idx`'s chain.
        self.rebuild_lineage_segs(idx, active, arena);
        let (changed_hull_l, changed_hull_r) = self.lineage_hulls[idx];

        let changed_pop = active[idx].population;
        let changed_bits = self.lineage_pos_bits[idx];
        let n = self.n;
        // Step 1: walk peer_bits[idx] directly — every set bit is a
        // peer that currently has a nonempty pair with `idx`. Replaces
        // the old bitmap row + column walks; single O(peers) loop.
        let stride = self.peer_word_stride;
        let mut peers: SmallVec<[usize; 16]> = SmallVec::new();
        let row_start = idx * stride;
        for w in 0..stride {
            let word = self.peer_bits[row_start + w];
            let base = w << 6;
            for bit in BitWordIter(word) {
                peers.push(base + bit);
            }
        }
        for peer in peers {
            let (i, j) = if peer < idx { (peer, idx) } else { (idx, peer) };
            self.clear_pair(i, j);
        }
        // Step 2: compute new (idx, other) pairs. Prescreens read from
        // direct-indexed side tables (lineage_pop / lineage_hulls /
        // lineage_pos_bits) so the hot filter is three sequential
        // Vec-index loads with no SmallVec header chase.
        for other in 0..n {
            if other == idx { continue; }
            if self.lineage_pop[other] != changed_pop { continue; }
            let (other_l, other_r) = self.lineage_hulls[other];
            if !(other_r > changed_hull_l && changed_hull_r > other_l) {
                continue;
            }
            if changed_bits & self.lineage_pos_bits[other] == 0 { continue; }
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
        let (left_hull_l, left_hull_r) = self.lineage_hulls[idx];
        let (right_hull_l, right_hull_r) = self.lineage_hulls[new_idx];

        // Split can only shrink idx's hull (left/right halves are
        // subsets); empty old pairs stay empty. Walk peer_bits[idx]
        // to visit only nonempty pairs — single unified loop replaces
        // the old bitmap row walk + column walk. new_pidx slot is
        // empty by invariant (new_idx was just pushed), so no
        // defensive scrub needed.
        let _ = changed_pop;
        let stride = self.peer_word_stride;
        let mut peers: SmallVec<[usize; 16]> = SmallVec::new();
        let row_start = idx * stride;
        for w in 0..stride {
            let word = self.peer_bits[row_start + w];
            let base = w << 6;
            for bit in BitWordIter(word) {
                peers.push(base + bit);
            }
        }
        for other in peers {
            if other == new_idx { continue; }
            self.apply_recomb_split_body(
                idx, other, new_idx, split_pos,
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
        let (other_l, other_r) = self.lineage_hulls
            .get(other).copied()
            .unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
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
        // Drain peer_bits[idx] directly — each set bit is another
        // lineage that currently has a nonempty pair with idx. One
        // unified walk replaces the old bitmap row walk (j > idx,
        // word-scan over nonempty_bits) + column walk (i < idx,
        // scattered pidxs with per-i bit_get). The peer bitmap is
        // already keyed by lineage-pair, so walk cost is O(peers)
        // independent of which side of `idx` they sit on.
        //
        // Snapshot into a local SmallVec: `clear_pair` mutates
        // peer_bits via peer_clear_pair, so we can't iterate the row
        // while borrowing it mutably.
        let stride = self.peer_word_stride;
        let mut peers: SmallVec<[usize; 16]> = SmallVec::new();
        let row_start = idx * stride;
        for w in 0..stride {
            let word = self.peer_bits[row_start + w];
            let base = w << 6;
            for bit in BitWordIter(word) {
                peers.push(base + bit);
            }
        }
        for peer in peers {
            let (i, j) = if peer < idx { (peer, idx) } else { (idx, peer) };
            self.clear_pair(i, j);
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
            if !self.lineage_hulls.is_empty() {
                self.lineage_hulls.pop();
            }
            // Clear the just-freed peer_bits row so stale bits can't
            // surface if this slot is reused later.
            let stride = self.peer_word_stride;
            let start = old_last * stride;
            for w in 0..stride { self.peer_bits[start + w] = 0; }
            self.n -= 1;
            return;
        }
        // Callers must have invoked `remove_lineage(removed_idx)` first,
        // so every (removed_idx, *) slot is already empty with bit = 0.
        // Walk peer_bits[old_last] for the set of lineages that have
        // a nonempty pair with old_last — much cheaper than the old
        // `for other in 0..old_last` scan with bit_get per slot.
        let stride = self.peer_word_stride;
        let mut peers: SmallVec<[usize; 16]> = SmallVec::new();
        let row_start = old_last * stride;
        for w in 0..stride {
            let word = self.peer_bits[row_start + w];
            let base = w << 6;
            for bit in BitWordIter(word) {
                peers.push(base + bit);
            }
        }
        for other in peers {
            if other == removed_idx { continue; }
            let (ni, nj) = if other < removed_idx {
                (other, removed_idx)
            } else {
                (removed_idx, other)
            };
            // move_pair now handles peer_bits too: peer_clear_pair
            // (other, old_last) + peer_set_pair(ni, nj). No extra
            // row-copy needed — every peer touched here flips the
            // same pair of bits, so peer_bits[old_last] drains to
            // zero as each peer is processed and peer_bits[removed_idx]
            // accumulates the new (other, removed_idx) bits.
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
        if old_last < self.lineage_hulls.len() {
            let moved = self.lineage_hulls[old_last];
            self.lineage_hulls[removed_idx] = moved;
            self.lineage_hulls.pop();
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
        if self.lineage_hulls.len() < n {
            self.lineage_hulls.resize(n,
                (f64::INFINITY, f64::NEG_INFINITY));
        } else {
            self.lineage_hulls.truncate(n);
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

    /// Debug-only consistency check for every invariant the incremental
    /// protocol relies on. Panics with a descriptive message on first
    /// violation. Call before/after suspect mutations; runs in O(n² +
    /// pairs) so gate with `debug_assertions` or `#[cfg(test)]` at hot
    /// call sites.
    ///
    /// Invariants verified:
    ///  1. `nonempty_bits[pidx]` set ⇔ `overlaps[pidx]` non-empty.
    ///  2. For every nonempty pair (i, j): `pair_bucket_refs[pidx]` has
    ///     one entry per class in `overlaps[pidx]`, in matching order.
    ///  3. Each ref `(slot, pos)` indexes the correct (pop, class)
    ///     bucket — pop from `lineage_pop[i]`, class from `overlaps[pidx][k].0`
    ///     — and `pair_buckets[slot].2[pos]` unpacks back to (i, j).
    ///  4. Every bucket entry `(slot, pos)` has exactly one refs entry
    ///     pointing back to it (bijection between nonempty pair/class
    ///     slots and bucket positions).
    ///  5. No two pair_buckets slots share the same (pop, class) key.
    #[cfg(any(test, debug_assertions))]
    pub fn debug_check_invariants(&self, label: &str) {
        use std::collections::HashSet;
        // Invariant 5: (pop, class) bucket keys unique.
        let mut seen: HashSet<(u32, BranchClass)> = HashSet::new();
        for (k, entry) in self.pair_buckets.iter().enumerate() {
            if !seen.insert((entry.0, entry.1)) {
                panic!("[{}] duplicate pair_buckets key ({:?}, {:?}) at slot {}",
                    label, entry.0, entry.1, k);
            }
        }

        // Invariants 1–3: walk every pair slot.
        // Separately track how many bucket positions are referenced.
        let mut ref_hits: Vec<Vec<bool>> = self.pair_buckets.iter()
            .map(|e| vec![false; e.2.len()])
            .collect();
        let cap = self.capacity;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let pidx = pair_idx(i, j, cap);
                let bit = bit_get(&self.nonempty_bits, pidx);
                let ovl = &self.overlaps[pidx];
                if bit != !ovl.is_empty() {
                    panic!("[{}] nonempty_bits mismatch at ({},{}) \
                           pidx={} bit={} len={}",
                           label, i, j, pidx, bit, ovl.len());
                }
                if ovl.is_empty() {
                    if !self.pair_bucket_refs[pidx].is_empty() {
                        panic!("[{}] empty overlap but refs non-empty at ({},{})",
                            label, i, j);
                    }
                    continue;
                }
                let refs = &self.pair_bucket_refs[pidx];
                if refs.len() != ovl.len() {
                    panic!("[{}] refs/overlap length mismatch at ({},{}): \
                           refs={} overlap={}",
                           label, i, j, refs.len(), ovl.len());
                }
                let expected_pop = self.lineage_pop[i];
                let expected_packed = pack_ij(i, j);
                for (k, (class, _)) in ovl.iter().enumerate() {
                    let (slot, pos) = (refs[k].0 as usize, refs[k].1 as usize);
                    let bucket = &self.pair_buckets[slot];
                    if bucket.0 != expected_pop || bucket.1 != *class {
                        panic!("[{}] ref ({},{}) class idx {}: slot key \
                               ({:?}, {:?}) ≠ expected ({}, {:?})",
                               label, i, j, k,
                               bucket.0, bucket.1, expected_pop, class);
                    }
                    if pos >= bucket.2.len() {
                        panic!("[{}] ref ({},{}) class idx {}: pos {} \
                               out of bucket len {}",
                               label, i, j, k, pos, bucket.2.len());
                    }
                    if bucket.2[pos] != expected_packed {
                        panic!("[{}] ref ({},{}) class idx {}: bucket[{}]={:#x} \
                               ≠ pack_ij({},{})={:#x}",
                               label, i, j, k, pos,
                               bucket.2[pos], i, j, expected_packed);
                    }
                    if ref_hits[slot][pos] {
                        panic!("[{}] bucket slot {} pos {} referenced twice",
                               label, slot, pos);
                    }
                    ref_hits[slot][pos] = true;
                }
            }
        }

        // Invariant 4: every bucket position had a back-reference.
        for (slot, hits) in ref_hits.iter().enumerate() {
            for (pos, hit) in hits.iter().enumerate() {
                if !hit {
                    panic!("[{}] bucket slot {} pos {} packed={:#x} \
                           has no refs entry",
                           label, slot, pos, self.pair_buckets[slot].2[pos]);
                }
            }
        }

        // Invariant 6: peer_bits consistency — for every lineage i in
        // 0..n, peer_bits[i] has bit j set iff j in 0..n, j != i,
        // and pair (min(i,j), max(i,j)) is nonempty.
        let stride = self.peer_word_stride;
        for i in 0..self.n {
            for j in 0..self.capacity {
                let word = self.peer_bits[i * stride + (j >> 6)];
                let bit = (word >> (j & 63)) & 1 != 0;
                let expected = if j >= self.n || j == i {
                    false
                } else {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    bit_get(&self.nonempty_bits, pair_idx(lo, hi, self.capacity))
                };
                if bit != expected {
                    panic!("[{}] peer_bits[{}] bit {} = {} but expected {}",
                           label, i, j, bit, expected);
                }
            }
        }
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

    /// Build a lineage spanning [l, r) in a single segment.
    fn mk_lin(arena: &mut SegmentArena, l: f64, r: f64,
              pop: u32, cls: BranchClass,
              uid: crate::lineage::LinUid) -> Lineage {
        let s = arena.alloc(l, r, uid as i32, cls);
        Lineage::new(s, s, pop, uid, arena)
    }

    /// Pre-peer-bitmap invariant fence: exercises rebuild, recompute_for,
    /// apply_recomb_split, swap_update, remove_lineage across a random
    /// stream and calls `debug_check_invariants` after every mutation.
    /// If pair_buckets / overlaps / nonempty_bits / pair_bucket_refs
    /// drift apart, this fires immediately with the mutation label.
    #[test]
    fn incremental_invariants_random_ops() {
        use crate::class_tag::Karyotype;
        use rand::{Rng, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let seq_len = 1000.0;
        let classes = [
            BranchClass::PANMICTIC,
            BranchClass::single(0, Karyotype::S),
            BranchClass::single(0, Karyotype::I),
        ];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut arena = SegmentArena::new();
        let mut cache = RateCache::new(64, seq_len);
        let mut active: Vec<Lineage> = Vec::new();
        let mut next_uid: crate::lineage::LinUid = 0;

        // Seed: 8 lineages, random intervals, 2 pops, 3 classes.
        for _ in 0..8 {
            let l = rng.random_range(0..seq_len as u64 / 2) as f64;
            let r = l + rng.random_range(1..seq_len as u64 / 2) as f64;
            let pop = rng.random_range(0..2u32);
            let cls = classes[rng.random_range(0..classes.len())];
            active.push(mk_lin(&mut arena, l, r, pop, cls, next_uid));
            next_uid += 1;
        }
        cache.rebuild(&active, &arena);
        cache.debug_check_invariants("after initial rebuild");

        for step in 0..200usize {
            if active.len() < 2 {
                // Replenish so ops stay meaningful.
                let pop = rng.random_range(0..2u32);
                let cls = classes[rng.random_range(0..classes.len())];
                active.push(mk_lin(&mut arena, 0.0, seq_len, pop, cls, next_uid));
                next_uid += 1;
                cache.rebuild(&active, &arena);
                cache.debug_check_invariants(&format!("step {}: rebuild", step));
                continue;
            }
            let op = rng.random_range(0..4u32);
            match op {
                // recompute_for: replace a lineage at idx with a fresh one.
                0 => {
                    let idx = rng.random_range(0..active.len());
                    let l = rng.random_range(0..seq_len as u64 / 2) as f64;
                    let r = l + rng.random_range(1..seq_len as u64 / 2) as f64;
                    let pop = rng.random_range(0..2u32);
                    let cls = classes[rng.random_range(0..classes.len())];
                    active[idx] = mk_lin(&mut arena, l, r, pop, cls, next_uid);
                    next_uid += 1;
                    cache.recompute_for(idx, &active, &arena);
                    cache.debug_check_invariants(
                        &format!("step {}: recompute_for({})", step, idx));
                }
                // apply_recomb_split: push a new lineage (simulating the
                // right-half of a recomb split) then update both.
                1 => {
                    let idx = rng.random_range(0..active.len());
                    let split_pos = rng.random_range(1..seq_len as u64 - 1) as f64;
                    let pop = active[idx].population;
                    let cls = classes[rng.random_range(0..classes.len())];
                    // Replace idx with left half, push right half.
                    let old_hl = active[idx].cached_hull_l;
                    let old_hr = active[idx].cached_hull_r;
                    if !(old_hl < split_pos && split_pos < old_hr) {
                        continue;
                    }
                    active[idx] = mk_lin(
                        &mut arena, old_hl, split_pos, pop, cls, next_uid);
                    next_uid += 1;
                    active.push(mk_lin(
                        &mut arena, split_pos, old_hr, pop, cls, next_uid));
                    next_uid += 1;
                    let new_idx = active.len() - 1;
                    cache.apply_recomb_split(
                        idx, new_idx, split_pos, &active, &arena);
                    cache.debug_check_invariants(
                        &format!("step {}: apply_recomb_split({},{})",
                                 step, idx, new_idx));
                }
                // swap_update: simulate swap_remove(idx) on the active list.
                2 => {
                    let idx = rng.random_range(0..active.len());
                    let old_last = active.len() - 1;
                    cache.remove_lineage(idx);
                    cache.debug_check_invariants(
                        &format!("step {}: remove_lineage({})", step, idx));
                    active.swap_remove(idx);
                    cache.swap_update(idx, old_last);
                    cache.debug_check_invariants(
                        &format!("step {}: swap_update({},{})",
                                 step, idx, old_last));
                }
                // push a new lineage + recompute for it.
                _ => {
                    let l = rng.random_range(0..seq_len as u64 / 2) as f64;
                    let r = l + rng.random_range(1..seq_len as u64 / 2) as f64;
                    let pop = rng.random_range(0..2u32);
                    let cls = classes[rng.random_range(0..classes.len())];
                    active.push(mk_lin(&mut arena, l, r, pop, cls, next_uid));
                    next_uid += 1;
                    let idx = active.len() - 1;
                    cache.recompute_for(idx, &active, &arena);
                    cache.debug_check_invariants(
                        &format!("step {}: push + recompute({})", step, idx));
                }
            }
        }
    }
}
