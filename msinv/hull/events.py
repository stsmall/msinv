"""Event handlers for the hull simulator.

Each handler mutates the active-lineage list and the table builder,
keeping the ARG state consistent. SMC' is correct by construction in
this representation (Kelleher, Etheridge, McVean 2016).
"""

from .lineage import Lineage
from .segment import Segment


def apply_coalescence(active, lin_a, lin_b, t, tables,
                      skip_if_no_overlap=False):
    """Coalesce two lineages.

    For each genomic interval where both lin_a and lin_b have ancestral
    material, an edge is added from a new internal node to each of
    their existing node_ids.  The merged lineage carries the new node's
    id in the overlap and retains the original node_ids elsewhere.

    If *skip_if_no_overlap* is True and the lineages have no
    overlapping material, no merge occurs — matching msprime's Hudson
    algorithm where non-overlapping pairs are a no-op.
    """
    if skip_if_no_overlap:
        sa_check = lin_a.head
        sb_check = lin_b.head
        has_overlap = False
        while sa_check is not None and sb_check is not None:
            if sa_check.right <= sb_check.left:
                sa_check = sa_check.next
            elif sb_check.right <= sa_check.left:
                sb_check = sb_check.next
            else:
                has_overlap = True
                break
        if not has_overlap:
            return None

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
            _append(Segment(sa.left, sa.right, sa.node_id,
                            branch_class=sa.branch_class))
            sa = sa.next
        elif sb.right <= sa.left:
            _append(Segment(sb.left, sb.right, sb.node_id,
                            branch_class=sb.branch_class))
            sb = sb.next
        else:
            # Overlap [max(left), min(right))
            l = max(sa.left, sb.left)
            r = min(sa.right, sb.right)
            # Pre-overlap solo bits
            if sa.left < l:
                _append(Segment(sa.left, l, sa.node_id,
                                branch_class=sa.branch_class))
            if sb.left < l:
                _append(Segment(sb.left, l, sb.node_id,
                                branch_class=sb.branch_class))
            # Overlap → coalesces here. Add edges from new_node to both.
            tables.add_edge(l, r, new_node, sa.node_id)
            tables.add_edge(l, r, new_node, sb.node_id)
            _append(Segment(l, r, new_node,
                            branch_class=sa.branch_class))
            # Advance whichever ended at r; keep the other's tail
            if sa.right == r:
                sa = sa.next
            else:
                sa = Segment(r, sa.right, sa.node_id,
                             branch_class=sa.branch_class, next=sa.next)
                if sa.next is not None:
                    sa.next.prev = sa
            if sb.right == r:
                sb = sb.next
            else:
                sb = Segment(r, sb.right, sb.node_id,
                             branch_class=sb.branch_class, next=sb.next)
                if sb.next is not None:
                    sb.next.prev = sb
    while sa is not None:
        _append(Segment(sa.left, sa.right, sa.node_id,
                        branch_class=sa.branch_class))
        sa = sa.next
    while sb is not None:
        _append(Segment(sb.left, sb.right, sb.node_id,
                        branch_class=sb.branch_class))
        sb = sb.next

    active.remove(lin_a)
    active.remove(lin_b)
    if new_head is not None:
        # Do NOT pass branch_class: each segment already carries the
        # correct per-position class from the merge.  Passing a
        # lineage-level class would overwrite every segment (the Lineage
        # constructor propagates branch_class to all segments).
        merged = Lineage(new_head, new_tail,
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


def _flip_class_tag(bc, cls_S: str, cls_I: str):
    """Return a new branch_class with the S/I tag for one inversion
    flipped. Handles both string tags ('S0', 'I', ...) and frozenset
    tags (nested inversions, where a segment carries multiple tags).
    Tags for OTHER inversions are left alone.
    """
    if isinstance(bc, frozenset):
        new = set(bc)
        if cls_S in new:
            new.remove(cls_S); new.add(cls_I)
        elif cls_I in new:
            new.remove(cls_I); new.add(cls_S)
        if len(new) == 1:
            return next(iter(new))
        return frozenset(new)
    if bc == cls_S:
        return cls_I
    if bc == cls_I:
        return cls_S
    return bc  # not in this inversion (e.g., 'P' or another inv's tag)


def apply_gene_flux(active, lineage, tract_left: float, tract_right: float,
                     inv=None):
    """Split a tract [tract_left, tract_right) out of ``lineage`` and
    flip the tract's class for the given ``inv`` (gene-conversion event).

    Models gene conversion in a heterokaryote: at this tract, the
    chromosome's karyotype-of-origin going BACKWARD in time is the
    OTHER karyotype. The tract's ancestral material becomes a new
    lineage with that inversion's class flipped; tags for other
    inversions (in the nested case) are preserved. The lineage's
    other material (outside the tract) stays in the original class on
    the original lineage.

    Parameters
    ----------
    inv : InversionSpec, optional
        The inversion whose class tag this conversion event flips.
        If ``None``, falls back to the legacy single-inv behaviour
        (flip the lineage's branch_class between 'S' and 'I').

    Returns
    -------
    (outside_lineage, tract_lineage) : tuple
        ``outside_lineage`` carries the original lineage's material
        outside the tract (may be ``None`` if the tract covered all of
        the lineage's material).
        ``tract_lineage`` carries material inside the tract, with
        ``inv``'s S/I tag flipped (may be ``None`` if the lineage
        didn't have any ancestral material in the tract — no-op).
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

    # B is the converted tract — flip its class for `inv`.
    if inv is None:
        # Legacy single-inv path: flip 'S' <-> 'I' at the lineage level.
        flipped = 'I' if lineage.branch_class == 'S' else 'S'
        B.branch_class = flipped
    else:
        # Multi-inv (or single-inv via inversions=[...]): flip just
        # this inversion's tag on each segment of B, leaving any other
        # inversion's tags intact.
        cls_S = inv.class_S()
        cls_I = inv.class_I()
        seg = B.head
        while seg is not None:
            seg.branch_class = _flip_class_tag(seg.branch_class, cls_S, cls_I)
            seg = seg.next
        # Reset the lineage-level summary so it recomputes from segments.
        B._branch_class_override = None

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
