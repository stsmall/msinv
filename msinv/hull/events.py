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


def apply_gene_flux(active, lineage, x, w):
    """Split a small tract [x, x+w] out of ``lineage`` with class flipped.

    Models gene conversion: at this position the chromosome's
    karyotype-of-origin is the OTHER karyotype for a tract of length w.
    The tract becomes its own lineage in the other class; the
    surrounding material stays on the original lineage.
    """
    # Phase 3 implementation. Sketch:
    # 1. Split lineage at x → (A, BC)
    # 2. Split BC at x+w → (B, C)
    # 3. Reattach A and C to a single lineage in the original class
    # 4. B becomes a new lineage in the OTHER class
    raise NotImplementedError("gene flux: phase 3 work")


def apply_migration(lineage, new_pop):
    """Move ``lineage`` into ``new_pop``. (Demography handler.)"""
    lineage.population = new_pop
