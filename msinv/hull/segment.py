"""Segment: one ancestral genomic interval owned by a lineage.

A lineage's ancestral material is a doubly-linked list of segments,
each pointing to the tskit node id that the segment descends from.
Segments are kept sorted by ``left``.
"""


class Segment:
    """One ancestral interval [left, right) referring to tskit node_id.

    branch_class:
      Phase 1-5c.1: a single string tag.
        'S' / 'I'        — inside a single (back-compat) inversion.
        'S<k>' / 'I<k>'  — inside inversion k of a multi-inversion list.
        'P'              — panmictic / outside any inversion.

      Phase 5c.2 (nested): a frozenset of string tags. A position
      inside multiple inversions carries one tag per containing
      inversion (e.g. ``frozenset({'S0', 'I1'})`` for S in inv 0 +
      I in inv 1). Empty frozenset is equivalent to ``'P'``.

    Compatibility (used by ``_overlap_by_class`` and the partial-coal
    handler) requires EXACT equality of branch_class. Tags removed by
    a class-barrier flip simply disappear from the frozenset.
    """

    __slots__ = ('left', 'right', 'node_id', 'branch_class', 'prev', 'next')

    def __init__(self, left: float, right: float, node_id: int,
                 branch_class='P',
                 prev: 'Segment' = None, next: 'Segment' = None):
        if right <= left:
            raise ValueError(
                f"Segment must have right > left, got [{left}, {right})")
        self.left = left
        self.right = right
        self.node_id = node_id
        # Normalise None → 'P'. Preserve frozenset and string types.
        if branch_class is None:
            branch_class = 'P'
        self.branch_class = branch_class
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
                          inversions=None, sample_class=None):
    """Build the initial segment list for one sample lineage.

    Parameters
    ----------
    L : sequence length.
    node_id : tskit node id for the sample.
    inversions : list of objects with ``bp_left``, ``bp_right``, and
        (for multi-inversion) ``inv_id`` attributes. Each inversion
        gets its own per-segment class tag ``'S<inv_id>'`` or
        ``'I<inv_id>'`` so class barriers stay independent.
    sample_class : per-inversion karyotype assignment. May be:

        - ``None``: purely panmictic ('P' for every position).
        - A single character ``'S'`` or ``'I'``: linked karyotype —
          the sample is that karyotype at EVERY inversion it crosses
          (Phase 5b semantics).
        - A sequence (tuple/list/str of length n_inv) of ``'S'``,
          ``'I'``, or ``None``: independent karyotype per inversion.
          ``sample_class[k]`` is the karyotype for inversion ``k`` in
          the (sorted-by-inv_id) inversions list. Length must match
          the number of inversions.

    Returns ``(head, tail)`` of the linked list.
    """
    if not inversions:
        seg = Segment(0.0, L, node_id, branch_class='P')
        return seg, seg

    sorted_invs = sorted(inversions, key=lambda inv: inv.bp_left)
    per_inv_class = _resolve_per_inv_class(sample_class, sorted_invs)

    # Detect nesting / overlap: if any inversion's [bp_left, bp_right)
    # overlaps another's, we use the multi-tag (frozenset) class
    # representation. Non-overlapping sticks with the cheaper string
    # representation (back-compat with Phase 5b semantics).
    has_overlap = False
    for i, a in enumerate(sorted_invs):
        for b in sorted_invs[i + 1:]:
            if b.bp_left < a.bp_right:
                has_overlap = True
                break
        if has_overlap:
            break

    if not has_overlap:
        # Phase 5b path: alternating outside / inside intervals.
        intervals = []
        cursor = 0.0
        for inv in sorted_invs:
            inv_id = getattr(inv, 'inv_id', -1)
            if inv.bp_left > cursor:
                intervals.append((cursor, inv.bp_left, 'P'))
            kary = per_inv_class.get(inv_id)
            if kary is None:
                cls = 'P'
            elif inv_id is None or inv_id < 0:
                cls = kary
            else:
                cls = f'{kary}{inv_id}'
            intervals.append((max(cursor, inv.bp_left), inv.bp_right, cls))
            cursor = inv.bp_right
        if cursor < L:
            intervals.append((cursor, L, 'P'))
    else:
        # Phase 5c.2 path: build per-position tag sets by scanning the
        # union of all inversion breakpoints.
        breakpoints = set([0.0, L])
        for inv in sorted_invs:
            breakpoints.add(inv.bp_left)
            breakpoints.add(inv.bp_right)
        sorted_bps = sorted(b for b in breakpoints if 0.0 <= b <= L)
        intervals = []
        for a, b in zip(sorted_bps, sorted_bps[1:]):
            if b <= a:
                continue
            tags = []
            for inv in sorted_invs:
                if inv.bp_left <= a and b <= inv.bp_right:
                    inv_id = getattr(inv, 'inv_id', -1)
                    kary = per_inv_class.get(inv_id)
                    if kary is None:
                        continue
                    tags.append(f'{kary}{inv_id}'
                                if inv_id is not None and inv_id >= 0
                                else kary)
            if not tags:
                cls = 'P'
            elif len(tags) == 1:
                cls = tags[0]
            else:
                cls = frozenset(tags)
            intervals.append((a, b, cls))

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


def _resolve_per_inv_class(sample_class, sorted_invs) -> dict:
    """Resolve a ``sample_class`` value to ``{inv_id: 'S'|'I'|None}``."""
    if sample_class is None:
        return {inv.inv_id: None for inv in sorted_invs}
    # Single-character string: linked karyotype across all inversions.
    if isinstance(sample_class, str) and len(sample_class) == 1:
        return {inv.inv_id: sample_class for inv in sorted_invs}
    # String or sequence of length n_inv: independent karyotype.
    if (isinstance(sample_class, str) or
            hasattr(sample_class, '__iter__')):
        seq = list(sample_class)
        if len(seq) != len(sorted_invs):
            raise ValueError(
                f"sample_class has {len(seq)} entries but there are "
                f"{len(sorted_invs)} inversions; lengths must match.")
        return {inv.inv_id: seq[i] for i, inv in enumerate(sorted_invs)}
    raise TypeError(
        f"sample_class must be None, 'S'/'I', or a sequence of "
        f"karyotypes; got {type(sample_class).__name__} {sample_class!r}.")
