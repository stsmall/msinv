"""Event handlers for the hull simulator.

Each handler mutates the active-lineage list and the table builder,
keeping the ARG state consistent. SMC' is correct by construction in
this representation (Kelleher, Etheridge, McVean 2016).
"""

import numpy as np

from .lineage import Lineage
from .segment import Segment


def apply_coalescence(active, lin_a, lin_b, t, tables):
    """Coalesce two lineages.

    A new tskit node is added at time t. For each genomic interval
    where both lin_a and lin_b have ancestral material, an edge is
    added from the new node to each of their existing node_ids. Where
    only one is ancestral, the offspring lineage carries that lineage's
    node_id. Where both are ancestral, the offspring lineage carries
    the new node's id (because it is now their MRCA at that interval).

    The merged lineage replaces lin_a and lin_b in ``active``.
    """
    new_node = tables.add_internal(time=t, population=lin_a.population)
    new_head = new_tail = None

    sa = lin_a.head
    sb = lin_b.head

    def _append(seg):
        nonlocal new_head, new_tail
        seg.prev = new_tail
        seg.next = None
        if new_head is None:
            new_head = seg
        if new_tail is not None:
            new_tail.next = seg
        new_tail = seg

    # Sweep both segment lists left-to-right, splitting at every
    # boundary, and emit at most one merged segment per interval.
    while sa is not None and sb is not None:
        if sa.right <= sb.left:
            _append(Segment(sa.left, sa.right, sa.node_id))
            sa = sa.next
        elif sb.right <= sa.left:
            _append(Segment(sb.left, sb.right, sb.node_id))
            sb = sb.next
        else:
            # Overlap [max(left), min(right))
            l = max(sa.left, sb.left)
            r = min(sa.right, sb.right)
            # Pre-overlap solo bits
            if sa.left < l:
                _append(Segment(sa.left, l, sa.node_id))
            if sb.left < l:
                _append(Segment(sb.left, l, sb.node_id))
            # Overlap → coalesces here. Add edges from new_node to both.
            tables.add_edge(l, r, new_node, sa.node_id)
            tables.add_edge(l, r, new_node, sb.node_id)
            _append(Segment(l, r, new_node))
            # Advance whichever ended at r; keep the other's tail
            if sa.right == r:
                sa = sa.next
            else:
                sa = Segment(r, sa.right, sa.node_id, next=sa.next)
                if sa.next is not None:
                    sa.next.prev = sa
            if sb.right == r:
                sb = sb.next
            else:
                sb = Segment(r, sb.right, sb.node_id, next=sb.next)
                if sb.next is not None:
                    sb.next.prev = sb
    while sa is not None:
        _append(Segment(sa.left, sa.right, sa.node_id))
        sa = sa.next
    while sb is not None:
        _append(Segment(sb.left, sb.right, sb.node_id))
        sb = sb.next

    active.remove(lin_a)
    active.remove(lin_b)
    if new_head is not None:
        merged = Lineage(new_head, new_tail,
                         branch_class=lin_a.branch_class,
                         population=lin_a.population)
        active.append(merged)
    return new_node


def apply_recombination(active, lineage, x):
    """Split ``lineage`` at position x. Returns the two new lineages."""
    left, right = lineage.split_at(x)
    active.remove(lineage)
    if left is not None:
        active.append(left)
    if right is not None:
        active.append(right)
    return left, right


def apply_gene_flux(active, lineage, tract_left: float, tract_right: float):
    """Split a tract [tract_left, tract_right) out of ``lineage`` and
    flip the tract's class (gene-conversion event).

    Models gene conversion in a heterokaryote: at this tract, the
    chromosome's karyotype-of-origin going BACKWARD in time is the
    OTHER karyotype. The tract's ancestral material becomes a new
    lineage in the flipped class; the lineage's other material
    (everything outside the tract) stays in the original class on the
    original lineage.

    The lineage's segment list may end up with a "hole" where the
    tract was — those positions are now traced by the flipped-class
    lineage, not by this one.

    Returns
    -------
    (outside_lineage, tract_lineage) : tuple
        ``outside_lineage`` carries the original lineage's material
        outside the tract (may be ``None`` if the tract covered all of
        the lineage's material).
        ``tract_lineage`` carries material inside the tract, in the
        flipped class (may be ``None`` if the lineage didn't have any
        ancestral material in the tract — in which case no flux event
        happens; this is a no-op).
    """
    if tract_right <= tract_left:
        raise ValueError(
            f"Tract must have right > left, got [{tract_left}, "
            f"{tract_right}).")

    # Step 1: split lineage at tract_left → (A, BC)
    A, BC = lineage.split_at(tract_left)
    if BC is None:
        # Lineage has no material at or after tract_left — no event.
        return lineage, None
    # Step 2: split BC at tract_right → (B, C)
    B, C = BC.split_at(tract_right)
    if B is None:
        # No material inside the tract — re-merge A and C, no event.
        if A is not None and C is not None:
            A.tail.next = C.head
            C.head.prev = A.tail
            A.tail = C.tail
            return A, None
        return (A or C), None

    # B is the converted tract — flip its class.
    flipped = 'I' if lineage.branch_class == 'S' else 'S'
    B.branch_class = flipped

    # Re-merge A and C into the outside-tract lineage (same class as
    # the original). The lineage now has a "hole" where the tract was.
    if A is not None and C is not None:
        A.tail.next = C.head
        C.head.prev = A.tail
        A.tail = C.tail
        outside = A
    elif A is not None:
        outside = A
    else:
        outside = C  # may be None

    # Update active list: remove original, add outside (if any) + tract.
    active.remove(lineage)
    if outside is not None:
        active.append(outside)
    active.append(B)
    return outside, B


def apply_migration(lineage, new_pop):
    """Move ``lineage`` into ``new_pop``. (Demography handler.)"""
    lineage.population = new_pop
