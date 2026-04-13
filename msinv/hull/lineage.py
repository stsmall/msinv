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

    Attributes
    ----------
    head, tail : Segment
        Doubly-linked list of ancestral material intervals.
    branch_class : str
        'S' or 'I' (karyotype). May be None outside an inversion or
        in panmictic mode.
    population : int
        Population id (per the ``demes`` graph).
    uid : int
        Unique identifier for this lineage.
    """

    __slots__ = ('head', 'tail', 'branch_class', 'population', 'uid')

    def __init__(self, head: Segment, tail: Segment,
                 branch_class: str = None, population: int = 0):
        self.head = head
        self.tail = tail
        self.branch_class = branch_class
        self.population = population
        self.uid = next(_uid_gen)

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

    def split_at(self, x: float) -> 'tuple[Lineage, Lineage]':
        """Split this lineage at position x. Returns two new lineages."""
        (lh, lt), (rh, rt) = split_segment_list(self.head, self.tail, x)
        left = Lineage(lh, lt, self.branch_class, self.population) if lh else None
        right = Lineage(rh, rt, self.branch_class, self.population) if rh else None
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
