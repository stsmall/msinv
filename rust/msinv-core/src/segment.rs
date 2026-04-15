/// Arena-allocated segment storage.
///
/// Each `Segment` represents one ancestral genomic interval [left, right)
/// carried by a lineage. Segments form a singly-linked list via `next`
/// indices into the arena. The arena provides O(1) allocation (with
/// free-list recycling) and cache-friendly iteration.

use crate::class_tag::BranchClass;

/// Index into the segment arena. u32 is sufficient for ~4 billion
/// segments; typical simulations create a few thousand.
pub type SegIdx = u32;

/// Sentinel value for "no segment".
pub const SEG_NIL: SegIdx = u32::MAX;

#[derive(Clone, Debug)]
pub struct Segment {
    pub left: f64,
    pub right: f64,
    pub node_id: i32,
    pub branch_class: BranchClass,
    pub next: SegIdx, // SEG_NIL if tail
}

impl Segment {
    #[inline]
    pub fn length(&self) -> f64 {
        self.right - self.left
    }
}

/// Arena allocator for segments. Owns all segment storage.
pub struct SegmentArena {
    segs: Vec<Segment>,
    free: Vec<SegIdx>,
}

impl SegmentArena {
    pub fn new() -> Self {
        Self {
            segs: Vec::with_capacity(1024),
            free: Vec::new(),
        }
    }

    /// Allocate a new segment, returning its index.
    pub fn alloc(&mut self, left: f64, right: f64, node_id: i32,
                  branch_class: BranchClass) -> SegIdx {
        let seg = Segment {
            left,
            right,
            node_id,
            branch_class,
            next: SEG_NIL,
        };
        if let Some(idx) = self.free.pop() {
            self.segs[idx as usize] = seg;
            idx
        } else {
            let idx = self.segs.len() as SegIdx;
            self.segs.push(seg);
            idx
        }
    }

    /// Return a segment to the free list for reuse.
    #[inline]
    pub fn free(&mut self, idx: SegIdx) {
        self.free.push(idx);
    }

    /// Access a segment by index.
    #[inline]
    pub fn get(&self, idx: SegIdx) -> &Segment {
        &self.segs[idx as usize]
    }

    /// Mutably access a segment by index.
    #[inline]
    pub fn get_mut(&mut self, idx: SegIdx) -> &mut Segment {
        &mut self.segs[idx as usize]
    }

    /// Total length of ancestral material in a segment chain starting
    /// at `head`.
    pub fn total_length(&self, head: SegIdx) -> f64 {
        let mut len = 0.0;
        let mut cur = head;
        while cur != SEG_NIL {
            let seg = self.get(cur);
            len += seg.length();
            cur = seg.next;
        }
        len
    }

    /// Split a segment chain at position `x`.
    /// Returns (left_head, left_tail, right_head, right_tail).
    /// Left chain covers [original_left, x); right chain covers [x, ...).
    /// Any of these may be SEG_NIL if that chain is empty.
    pub fn split_at(&mut self, head: SegIdx, x: f64)
        -> (SegIdx, SegIdx, SegIdx, SegIdx)
    {
        if head == SEG_NIL {
            return (SEG_NIL, SEG_NIL, SEG_NIL, SEG_NIL);
        }

        // Walk to find the segment containing x, tracking tail.
        let mut prev_idx = SEG_NIL;
        let mut cur = head;
        while cur != SEG_NIL {
            let seg = self.get(cur);
            if x <= seg.left {
                // x is at or before this segment — everything from
                // cur onward goes to the right chain.
                let right_tail = self.find_tail(cur);
                if prev_idx != SEG_NIL {
                    self.get_mut(prev_idx).next = SEG_NIL;
                    return (head, prev_idx, cur, right_tail);
                }
                return (SEG_NIL, SEG_NIL, head, right_tail);
            }
            if x < seg.right {
                // x falls inside this segment — split it.
                let right_half = self.alloc(
                    x, seg.right, seg.node_id, seg.branch_class,
                );
                let next_after = self.get(cur).next;
                self.get_mut(right_half).next = next_after;
                self.get_mut(cur).right = x;
                self.get_mut(cur).next = SEG_NIL;
                let right_tail = if next_after == SEG_NIL {
                    right_half
                } else {
                    self.find_tail(right_half)
                };
                if prev_idx == SEG_NIL {
                    return (cur, cur, right_half, right_tail);
                }
                return (head, cur, right_half, right_tail);
            }
            prev_idx = cur;
            cur = self.get(cur).next;
        }
        // x is past the end of the chain — everything is on the left.
        (head, prev_idx, SEG_NIL, SEG_NIL)
    }

    /// Append segment chain `suffix_head` to the end of chain ending
    /// at `tail`. Updates `tail`'s next pointer. Returns the new tail.
    pub fn append_chain(&mut self, tail: SegIdx, suffix_head: SegIdx) -> SegIdx {
        if tail == SEG_NIL {
            return self.find_tail(suffix_head);
        }
        if suffix_head == SEG_NIL {
            return tail;
        }
        self.get_mut(tail).next = suffix_head;
        self.find_tail(suffix_head)
    }

    /// Walk to the tail of a chain.
    pub fn find_tail(&self, head: SegIdx) -> SegIdx {
        if head == SEG_NIL {
            return SEG_NIL;
        }
        let mut cur = head;
        loop {
            let next = self.get(cur).next;
            if next == SEG_NIL {
                return cur;
            }
            cur = next;
        }
    }

    /// Iterate over segment indices in a chain.
    pub fn iter_chain(&self, head: SegIdx) -> ChainIter<'_> {
        ChainIter { arena: self, cur: head }
    }

    /// Number of segments currently allocated (including free-list).
    pub fn capacity(&self) -> usize {
        self.segs.len()
    }
}

impl Default for SegmentArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over segment indices in a linked chain.
pub struct ChainIter<'a> {
    arena: &'a SegmentArena,
    cur: SegIdx,
}

impl<'a> Iterator for ChainIter<'a> {
    type Item = SegIdx;

    #[inline]
    fn next(&mut self) -> Option<SegIdx> {
        if self.cur == SEG_NIL {
            return None;
        }
        let idx = self.cur;
        self.cur = self.arena.get(idx).next;
        Some(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(arena: &mut SegmentArena, intervals: &[(f64, f64)]) -> SegIdx {
        let bc = BranchClass::PANMICTIC;
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
        head
    }

    #[test]
    fn alloc_and_access() {
        let mut arena = SegmentArena::new();
        let idx = arena.alloc(0.0, 10.0, 0, BranchClass::PANMICTIC);
        assert_eq!(arena.get(idx).left, 0.0);
        assert_eq!(arena.get(idx).right, 10.0);
    }

    #[test]
    fn free_and_reuse() {
        let mut arena = SegmentArena::new();
        let a = arena.alloc(0.0, 10.0, 0, BranchClass::PANMICTIC);
        arena.free(a);
        let b = arena.alloc(5.0, 15.0, 1, BranchClass::PANMICTIC);
        assert_eq!(a, b); // reused same slot
        assert_eq!(arena.get(b).left, 5.0);
    }

    #[test]
    fn total_length() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(0.0, 10.0), (20.0, 30.0)]);
        assert!((arena.total_length(head) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn split_at_boundary() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(0.0, 10.0), (10.0, 20.0)]);
        let (lh, lt, rh, rt) = arena.split_at(head, 10.0);
        assert_ne!(lh, SEG_NIL);
        assert_ne!(rh, SEG_NIL);
        assert_ne!(lt, SEG_NIL);
        assert_ne!(rt, SEG_NIL);
        assert!((arena.total_length(lh) - 10.0).abs() < 1e-12);
        assert!((arena.total_length(rh) - 10.0).abs() < 1e-12);
        // Tail pointers are correct.
        assert_eq!(arena.get(lt).next, SEG_NIL);
        assert_eq!(arena.get(rt).next, SEG_NIL);
    }

    #[test]
    fn split_inside_segment() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(0.0, 20.0)]);
        let (lh, lt, rh, rt) = arena.split_at(head, 7.0);
        assert_ne!(lh, SEG_NIL);
        assert_ne!(rh, SEG_NIL);
        assert!((arena.get(lh).right - 7.0).abs() < 1e-12);
        assert!((arena.get(rh).left - 7.0).abs() < 1e-12);
        assert!((arena.get(rh).right - 20.0).abs() < 1e-12);
        assert_eq!(lh, lt); // single-segment left chain
        assert_eq!(rh, rt); // single-segment right chain
    }

    #[test]
    fn split_before_start() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(5.0, 10.0)]);
        let (lh, _lt, rh, rt) = arena.split_at(head, 3.0);
        assert_eq!(lh, SEG_NIL);
        assert_eq!(rh, head);
        assert_eq!(rt, head); // single segment
    }

    #[test]
    fn split_past_end() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(0.0, 10.0)]);
        let (lh, lt, rh, _rt) = arena.split_at(head, 15.0);
        assert_eq!(lh, head);
        assert_eq!(lt, head); // single segment
        assert_eq!(rh, SEG_NIL);
    }

    #[test]
    fn iter_chain() {
        let mut arena = SegmentArena::new();
        let head = make_chain(&mut arena, &[(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]);
        let indices: Vec<_> = arena.iter_chain(head).collect();
        assert_eq!(indices.len(), 3);
    }
}
