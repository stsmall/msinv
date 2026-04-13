"""Segment: one ancestral genomic interval owned by a lineage.

A lineage's ancestral material is a doubly-linked list of segments,
each pointing to the tskit node id that the segment descends from.
Segments are kept sorted by ``left``.
"""


class Segment:
    """One ancestral interval [left, right) referring to tskit node_id.

    branch_class:
      'S' — standard arrangement (inside an inversion)
      'I' — inverted arrangement (inside an inversion)
      'P' — panmictic / outside any inversion (no karyotype constraint)

    A panmictic segment can coalesce with any other panmictic segment
    at the standard rate. An 'S' segment can only coalesce with another
    'S' segment (same inversion); same for 'I'. The class barrier is
    enforced per-position via the segment's class.
    """

    __slots__ = ('left', 'right', 'node_id', 'branch_class', 'prev', 'next')

    def __init__(self, left: float, right: float, node_id: int,
                 branch_class: str = 'P',
                 prev: 'Segment' = None, next: 'Segment' = None):
        if right <= left:
            raise ValueError(
                f"Segment must have right > left, got [{left}, {right})")
        self.left = left
        self.right = right
        self.node_id = node_id
        self.branch_class = branch_class if branch_class is not None else 'P'
        self.prev = prev
        self.next = next

    def __repr__(self):
        return (f"Segment([{self.left:.4f}, {self.right:.4f}) "
                f"-> n{self.node_id} cls={self.branch_class})")


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
            left_part = Segment(seg.left, x, seg.node_id,
                                branch_class=seg.branch_class,
                                prev=left_tail)
            if left_head is None:
                left_head = left_part
            if left_tail is not None:
                left_tail.next = left_part
            left_tail = left_part
            right_part = Segment(x, seg.right, seg.node_id,
                                 branch_class=seg.branch_class,
                                 prev=right_tail)
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


def make_initial_segments(L: float, node_id: int,
                          inversions=None, sample_class: str = None):
    """Build the initial segment list for one sample lineage.

    Parameters
    ----------
    L : sequence length.
    node_id : tskit node id for the sample.
    inversions : list of objects with ``bp_left``, ``bp_right``, and
        (for multi-inversion) ``inv_id`` attributes. If multi-inv,
        each inversion's segment is tagged with class ``'S<inv_id>'``
        or ``'I<inv_id>'`` so class barriers stay independent.
    sample_class : 'S' or 'I' for the sample's karyotype, applied to
        every inversion the sample crosses (linked karyotype). ``None``
        ⇒ sample is purely panmictic ('P').

    Returns ``(head, tail)`` of the linked list.
    """
    if not inversions:
        seg = Segment(0.0, L, node_id, branch_class='P')
        return seg, seg
    # Sort inversions by bp_left and split the chromosome into
    # alternating outside / inside intervals.
    sorted_invs = sorted(inversions, key=lambda inv: inv.bp_left)
    intervals = []  # list of (left, right, class)
    cursor = 0.0
    for inv in sorted_invs:
        if inv.bp_left > cursor:
            intervals.append((cursor, inv.bp_left, 'P'))
        # Inside-inv class is per-inversion when inv_id is set,
        # otherwise plain 'S' or 'I' for back-compat with single-inv.
        inv_id = getattr(inv, 'inv_id', -1)
        if sample_class is None:
            cls = 'P'
        elif inv_id is None or inv_id < 0:
            cls = sample_class  # 'S' or 'I'
        else:
            cls = f'{sample_class}{inv_id}'
        intervals.append((max(cursor, inv.bp_left), inv.bp_right, cls))
        cursor = inv.bp_right
    if cursor < L:
        intervals.append((cursor, L, 'P'))
    head = tail = None
    for (l, r, cls) in intervals:
        if r <= l:
            continue
        seg = Segment(l, r, node_id, branch_class=cls, prev=tail)
        if head is None:
            head = seg
        if tail is not None:
            tail.next = seg
        tail = seg
    return head, tail
