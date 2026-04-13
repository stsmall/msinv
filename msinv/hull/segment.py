"""Segment: one ancestral genomic interval owned by a lineage.

A lineage's ancestral material is a doubly-linked list of segments,
each pointing to the tskit node id that the segment descends from.
Segments are kept sorted by ``left``.
"""


class Segment:
    """One ancestral interval [left, right) referring to tskit node_id."""

    __slots__ = ('left', 'right', 'node_id', 'prev', 'next')

    def __init__(self, left: float, right: float, node_id: int,
                 prev: 'Segment' = None, next: 'Segment' = None):
        if right <= left:
            raise ValueError(
                f"Segment must have right > left, got [{left}, {right})")
        self.left = left
        self.right = right
        self.node_id = node_id
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"Segment([{self.left:.4f}, {self.right:.4f}) -> n{self.node_id})"


def make_segment_list(intervals, node_ids):
    """Build a doubly-linked list of segments.

    intervals: iterable of (left, right) tuples, sorted ascending by left.
    node_ids: iterable of tskit node ids (parallel to intervals).
    Returns (head, tail) of the linked list.
    """
    head = tail = None
    for (l, r), nid in zip(intervals, node_ids):
        seg = Segment(l, r, nid, prev=tail)
        if head is None:
            head = seg
        if tail is not None:
            tail.next = seg
        tail = seg
    return head, tail


def split_segment_list(head, tail, x):
    """Split a segment list at position x.

    Returns ((left_head, left_tail), (right_head, right_tail)).
    Either side may be (None, None) if no segments fall there.
    The segment containing x (if any) is split into two new Segment
    objects sharing the same ``node_id``.
    """
    left_head = left_tail = None
    right_head = right_tail = None

    seg = head
    while seg is not None:
        nxt = seg.next
        # Detach seg from any neighbours; we'll re-link below.
        seg.prev = None
        seg.next = None
        if seg.right <= x:
            # Whole segment is to the left of x.
            seg.prev = left_tail
            if left_head is None:
                left_head = seg
            if left_tail is not None:
                left_tail.next = seg
            left_tail = seg
        elif seg.left >= x:
            # Whole segment is to the right of x.
            seg.prev = right_tail
            if right_head is None:
                right_head = seg
            if right_tail is not None:
                right_tail.next = seg
            right_tail = seg
        else:
            # Segment straddles x: split into two new segments.
            left_part = Segment(seg.left, x, seg.node_id, prev=left_tail)
            if left_head is None:
                left_head = left_part
            if left_tail is not None:
                left_tail.next = left_part
            left_tail = left_part
            right_part = Segment(x, seg.right, seg.node_id, prev=right_tail)
            if right_head is None:
                right_head = right_part
            if right_tail is not None:
                right_tail.next = right_part
            right_tail = right_part
        seg = nxt

    return (left_head, left_tail), (right_head, right_tail)


def total_length(head):
    """Sum of segment lengths in a list rooted at ``head``."""
    s = 0.0
    seg = head
    while seg is not None:
        s += seg.right - seg.left
        seg = seg.next
    return s
