"""Lineage: a (segment_list, branch_class, population) record.

Lineages are the units of the structured-coalescent state. Each lineage
owns a doubly-linked list of :class:`Segment` objects representing the
genomic intervals it is currently ancestral to.
"""

from itertools import count

from .segment import Segment, total_length, split_segment_list


_uid_gen = count()


def reset_uids():
    """Reset the lineage UID generator (call at the start of each rep)."""
    global _uid_gen
    _uid_gen = count()


class Lineage:
    """A lineage in the hull simulator.

    Class is *both* a per-segment attribute (``Segment.branch_class``)
    AND a lineage-level summary (``Lineage.branch_class``). When all
    segments have the same class, the lineage's class equals that
    common class; mixed-class lineages return ``'mixed'`` (e.g. after
    gene flux affects part of the lineage's ancestral material).

    For Phase 5: outside-inv segments carry ``'P'`` (panmictic), in-inv
    segments carry ``'S'`` or ``'I'``. The per-pair coalescence rate
    walks segments and sums contributions per class-compatible overlap.

    Attributes
    ----------
    head, tail : Segment
    branch_class : str
        Read/write summary; on write, the class is propagated to all
        segments (backwards-compat with phase 1-4 code).
    population : int
    uid : int
    """

    __slots__ = ('head', 'tail', '_branch_class_override',
                 'population', 'uid')

    def __init__(self, head: Segment, tail: Segment,
                 branch_class: str = None, population: int = 0):
        self.head = head
        self.tail = tail
        self._branch_class_override = None
        self.population = population
        self.uid = next(_uid_gen)
        if branch_class is not None:
            self.branch_class = branch_class  # propagates to segments

    @property
    def branch_class(self) -> str:
        """Summary class: single class if all segments agree, else 'mixed'."""
        if self._branch_class_override is not None:
            return self._branch_class_override
        seg = self.head
        if seg is None:
            return None
        cls = seg.branch_class
        seg = seg.next
        while seg is not None:
            if seg.branch_class != cls:
                return 'mixed'
            seg = seg.next
        return cls

    @branch_class.setter
    def branch_class(self, value: str):
        """Setting branch_class on a lineage propagates it to ALL
        segments. (Backwards-compat with phase 1-4 init patterns.)"""
        seg = self.head
        while seg is not None:
            seg.branch_class = value
            seg = seg.next
        # Drop any override so the property returns the segment-derived value.
        self._branch_class_override = None

    @property
    def hull_left(self) -> float:
        return self.head.left if self.head is not None else float('nan')

    @property
    def hull_right(self) -> float:
        return self.tail.right if self.tail is not None else float('nan')

    @property
    def total_length(self) -> float:
        return total_length(self.head)

    def covers(self, x: float) -> bool:
        """True if x is in any of this lineage's segments."""
        seg = self.head
        while seg is not None and seg.left <= x:
            if seg.left <= x < seg.right:
                return True
            seg = seg.next
        return False

    def class_at(self, x: float) -> str:
        """Class of the segment containing x, or None if x not covered."""
        seg = self.head
        while seg is not None:
            if seg.left <= x < seg.right:
                return seg.branch_class
            if seg.left > x:
                return None
            seg = seg.next
        return None

    def split_at(self, x: float) -> 'tuple[Lineage, Lineage]':
        """Split this lineage at position x. Returns two new lineages."""
        (lh, lt), (rh, rt) = split_segment_list(self.head, self.tail, x)
        left = Lineage(lh, lt, population=self.population) if lh else None
        right = Lineage(rh, rt, population=self.population) if rh else None
        return left, right

    def __repr__(self):
        n_segs = 0
        seg = self.head
        while seg is not None:
            n_segs += 1
            seg = seg.next
        return (f"Lineage(uid={self.uid}, cls={self.branch_class}, "
                f"pop={self.population}, {n_segs} segs, "
                f"len={self.total_length:.4f})")
