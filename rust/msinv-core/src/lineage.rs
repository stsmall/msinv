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

pub struct Lineage {
    pub head: SegIdx,
    pub tail: SegIdx,
    pub population: u32,
    pub uid: LinUid,
    /// Cached total ancestral material length. Kept in sync by
    /// `new_with_len`, `split_at`, and the event handlers.
    pub cached_len: f64,
}

impl Lineage {
    /// Create a lineage and compute its length from the arena.
    pub fn new(head: SegIdx, tail: SegIdx, population: u32, uid: LinUid,
               arena: &SegmentArena) -> Self {
        let cached_len = arena.total_length(head);
        Self { head, tail, population, uid, cached_len }
    }

    /// Create a lineage with a pre-computed length (avoids the walk
    /// when the caller already knows the length).
    pub fn new_with_len(head: SegIdx, tail: SegIdx, population: u32,
                         uid: LinUid, cached_len: f64) -> Self {
        Self { head, tail, population, uid, cached_len }
    }

    /// Leftmost genomic position covered by this lineage. O(1).
    #[inline]
    pub fn hull_left(&self, arena: &SegmentArena) -> f64 {
        if self.head == SEG_NIL { return f64::INFINITY; }
        arena.get(self.head).left
    }

    /// Rightmost genomic position covered by this lineage. O(1).
    #[inline]
    pub fn hull_right(&self, arena: &SegmentArena) -> f64 {
        if self.tail == SEG_NIL { return f64::NEG_INFINITY; }
        arena.get(self.tail).right
    }

    /// O(1) hull overlap check: do the genomic extents of two lineages
    /// overlap at all? Rejects clearly non-overlapping pairs before the
    /// full segment walk.
    #[inline]
    pub fn hulls_overlap(&self, other: &Lineage, arena: &SegmentArena) -> bool {
        self.hull_left(arena) < other.hull_right(arena)
            && other.hull_left(arena) < self.hull_right(arena)
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
            right_head, right_tail, self.population, new_uid, right_len);

        // Update self to be the left portion (no find_tail needed).
        self.head = left_head;
        self.tail = left_tail;
        self.cached_len -= right_len;
        if self.cached_len < 0.0 { self.cached_len = 0.0; }
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
