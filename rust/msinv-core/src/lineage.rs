/// Lineage: a (segment chain, population) record in the active list.
///
/// Each lineage carries a chain of segments representing the genomic
/// intervals it is ancestral to. The segment chain is stored in the
/// shared `SegmentArena`; the lineage only holds the head index.
///
/// `cached_len` is maintained incrementally so that `total_length()`
/// is O(1) instead of O(segments).

use crate::class_tag::BranchClass;
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};

/// Unique lineage identifier (monotonically increasing per simulation).
pub type LinUid = u32;

/// Sweep-allele tag map: `lin.uid → bool` (true = `A`, false = `ALower`).
/// Backed by `rustc_hash::FxHashMap` so the per-iteration lookup inside
/// `emit_coal_events_from_cache`'s per-cell active walk doesn't pay
/// SipHash13's ~600 ns/lookup overhead. Flamegraph (Phase F L=100kb,
/// 2026-05-03) showed ~13% of total wall in `BuildHasher::hash_one` on
/// this exact map; FxHash on a 4-byte LinUid is ~5× faster.
///
/// Switching the backing hasher is byte-equivalent semantically: the
/// map's only callers do `.get/.insert/.contains_key` keyed on uid;
/// no caller iterates in a particular order.
pub type ATagMap = rustc_hash::FxHashMap<LinUid, bool>;

pub struct Lineage {
    pub head: SegIdx,
    pub tail: SegIdx,
    pub population: u32,
    pub uid: LinUid,
    /// Cached total ancestral material length. Kept in sync by
    /// `new_with_len`, `split_at`, and the event handlers.
    pub cached_len: f64,
    /// Cached hull bounds — [cached_hull_l, cached_hull_r) covers the
    /// leftmost-segment.left to rightmost-segment.right span. Avoids the
    /// per-call `arena.get(head).left` + `arena.get(tail).right` bounds-
    /// checked deref on every CoalPanmicticPop prescreen (~2.8% of run_loop
    /// wall at rho=2000). Kept in sync by `new` / `new_with_len` (arena
    /// lookup) and `split_at` (lookup after arena mutation).
    pub cached_hull_l: f64,
    pub cached_hull_r: f64,
}

impl Lineage {
    /// Create a lineage and compute its length from the arena.
    pub fn new(head: SegIdx, tail: SegIdx, population: u32, uid: LinUid,
               arena: &SegmentArena) -> Self {
        let cached_len = arena.total_length(head);
        let (cached_hull_l, cached_hull_r) = Self::hull_from_arena(head, tail, arena);
        Self { head, tail, population, uid, cached_len,
               cached_hull_l, cached_hull_r }
    }

    /// Create a lineage with a pre-computed length (avoids the walk
    /// when the caller already knows the length).
    pub fn new_with_len(head: SegIdx, tail: SegIdx, population: u32,
                         uid: LinUid, cached_len: f64,
                         arena: &SegmentArena) -> Self {
        let (cached_hull_l, cached_hull_r) = Self::hull_from_arena(head, tail, arena);
        Self { head, tail, population, uid, cached_len,
               cached_hull_l, cached_hull_r }
    }

    #[inline]
    fn hull_from_arena(head: SegIdx, tail: SegIdx, arena: &SegmentArena) -> (f64, f64) {
        if head == SEG_NIL {
            (f64::INFINITY, f64::NEG_INFINITY)
        } else {
            let l = arena.get(head).left;
            let r = arena.get(tail).right;
            (l, r)
        }
    }

    /// Leftmost genomic position covered by this lineage. O(1).
    #[inline]
    pub fn hull_left(&self, _arena: &SegmentArena) -> f64 {
        self.cached_hull_l
    }

    /// Rightmost genomic position covered by this lineage. O(1).
    #[inline]
    pub fn hull_right(&self, _arena: &SegmentArena) -> f64 {
        self.cached_hull_r
    }

    /// O(1) hull overlap check: do the genomic extents of two lineages
    /// overlap at all? Rejects clearly non-overlapping pairs before the
    /// full segment walk.
    #[inline]
    pub fn hulls_overlap(&self, other: &Lineage, _arena: &SegmentArena) -> bool {
        self.cached_hull_l < other.cached_hull_r
            && other.cached_hull_l < self.cached_hull_r
    }

    /// Total length of ancestral material (O(1) — cached).
    #[inline]
    pub fn total_length(&self, _arena: &SegmentArena) -> f64 {
        self.cached_len
    }

    /// The branch class of this lineage. If all segments have the same
    /// class, returns that class; otherwise returns None.
    pub fn branch_class(&self, arena: &SegmentArena) -> Option<BranchClass> {
        if self.head == SEG_NIL {
            return None;
        }
        let first_class = arena.get(self.head).branch_class;
        let mut cur = arena.get(self.head).next;
        while cur != SEG_NIL {
            if arena.get(cur).branch_class != first_class {
                return None;
            }
            cur = arena.get(cur).next;
        }
        Some(first_class)
    }

    /// Branch class at a specific genomic position, or None if the
    /// lineage has no material at that position.
    pub fn class_at(&self, pos: f64, arena: &SegmentArena) -> Option<BranchClass> {
        let mut cur = self.head;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            if pos < seg.left {
                return None;
            }
            if pos < seg.right {
                return Some(seg.branch_class);
            }
            cur = seg.next;
        }
        None
    }

    /// Splice `other`'s chain onto the tail of this lineage and
    /// fold its cached length / hull right extent. `other`'s segments
    /// must sort strictly after this lineage's tail (caller-enforced,
    /// typical for a same-node split around a central tract).
    pub fn append_chain(&mut self, other: Lineage, arena: &mut SegmentArena) {
        if other.head == SEG_NIL { return; }
        if self.tail != SEG_NIL {
            arena.get_mut(self.tail).next = other.head;
        } else {
            self.head = other.head;
            self.cached_hull_l = other.cached_hull_l;
        }
        self.tail = other.tail;
        self.cached_len += other.cached_len;
        self.cached_hull_r = other.cached_hull_r;
    }

    /// Split this lineage at genomic position `x`. Returns the right
    /// half as a new Lineage (with a new uid); this lineage is
    /// truncated to [head, x). Returns None if x is past the end.
    /// Both lineages' cached_len are updated.
    pub fn split_at(&mut self, x: f64, arena: &mut SegmentArena,
                     new_uid: LinUid) -> Option<Lineage> {
        let (left_head, left_tail, right_head, right_tail) =
            arena.split_at(self.head, x);
        if right_head == SEG_NIL {
            // Update tail in case split_at changed it (x past end).
            self.tail = left_tail;
            return None;
        }
        let right_len = arena.total_length(right_head);
        let right = Lineage::new_with_len(
            right_head, right_tail, self.population, new_uid, right_len, arena);

        // Update self to be the left portion (no find_tail needed).
        self.head = left_head;
        self.tail = left_tail;
        self.cached_len -= right_len;
        if self.cached_len < 0.0 { self.cached_len = 0.0; }
        let (hl, hr) = Self::hull_from_arena(left_head, left_tail, arena);
        self.cached_hull_l = hl;
        self.cached_hull_r = hr;
        Some(right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_tag::Karyotype;

    fn build_lineage(arena: &mut SegmentArena, intervals: &[(f64, f64)],
                      bc: BranchClass, uid: LinUid) -> Lineage {
        let mut head = SEG_NIL;
        let mut tail = SEG_NIL;
        for (i, &(l, r)) in intervals.iter().enumerate() {
            let idx = arena.alloc(l, r, i as i32, bc);
            if head == SEG_NIL {
                head = idx;
            } else {
                arena.get_mut(tail).next = idx;
            }
            tail = idx;
        }
        Lineage::new(head, tail, 0, uid, arena)
    }

    #[test]
    fn total_length_two_segments() {
        let mut arena = SegmentArena::new();
        let lin = build_lineage(&mut arena,
            &[(0.0, 10.0), (20.0, 30.0)], BranchClass::PANMICTIC, 0);
        assert!((lin.total_length(&arena) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn branch_class_uniform() {
        let mut arena = SegmentArena::new();
        let bc = BranchClass::single(0, Karyotype::S);
        let lin = build_lineage(&mut arena, &[(0.0, 100.0)], bc, 0);
        assert_eq!(lin.branch_class(&arena), Some(bc));
    }

    #[test]
    fn class_at_position() {
        let mut arena = SegmentArena::new();
        let bc = BranchClass::single(0, Karyotype::I);
        let lin = build_lineage(&mut arena, &[(10.0, 50.0)], bc, 0);
        assert_eq!(lin.class_at(25.0, &arena), Some(bc));
        assert_eq!(lin.class_at(5.0, &arena), None);
        assert_eq!(lin.class_at(55.0, &arena), None);
    }

    #[test]
    fn split_lineage() {
        let mut arena = SegmentArena::new();
        let mut lin = build_lineage(&mut arena,
            &[(0.0, 50.0), (50.0, 100.0)], BranchClass::PANMICTIC, 0);
        let right = lin.split_at(30.0, &mut arena, 1).unwrap();
        assert!((lin.total_length(&arena) - 30.0).abs() < 1e-12);
        assert!((right.total_length(&arena) - 70.0).abs() < 1e-12);
    }

    #[test]
    fn split_at_first_seg_left_leaves_empty_zombie() {
        // Reproduces the bug where apply_gene_flux (or apply_recombination
        // with offset==0) calls split_at(x) with x equal to the first
        // segment's left boundary. The current lineage becomes head=NIL,
        // cached_len=0 — a zombie entry that must be removed by the caller.
        let mut arena = SegmentArena::new();
        let mut lin = build_lineage(&mut arena,
            &[(100.0, 200.0)], BranchClass::PANMICTIC, 0);
        assert!((lin.cached_len - 100.0).abs() < 1e-12);

        let right = lin.split_at(100.0, &mut arena, 1);
        assert!(right.is_some());
        let right = right.unwrap();

        assert!((right.cached_len - 100.0).abs() < 1e-12);
        assert_eq!(lin.head, SEG_NIL,
            "BUG: self.head != SEG_NIL after split at first-seg left?");
        assert_eq!(lin.cached_len, 0.0,
            "self went from 100.0 to 0.0 — zombie lineage");
        // If callers don't detect this and swap_remove it, the zombie
        // stays in `active` contributing nothing but still picked by
        // index-walks, bucket iteration, etc.
    }

    #[test]
    fn cached_len_consistent_after_split() {
        let mut arena = SegmentArena::new();
        let mut lin = build_lineage(&mut arena,
            &[(0.0, 100.0)], BranchClass::PANMICTIC, 0);
        assert!((lin.cached_len - 100.0).abs() < 1e-12);
        let right = lin.split_at(40.0, &mut arena, 1).unwrap();
        assert!((lin.cached_len - 40.0).abs() < 1e-12);
        assert!((right.cached_len - 60.0).abs() < 1e-12);
    }
}
