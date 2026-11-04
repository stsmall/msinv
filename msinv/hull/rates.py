"""Rate computations for the hull simulator.

Wraps ``demestats`` (Ragsdale 2026, doi:10.64898/2026.04.09.717519)
when available for arbitrary structured-coalescent rates over a demes
graph. Falls back to direct calculation for the simple cases.

Currently a STUB — future work will hook in demestats as the rate engine.
"""

try:
    import demestats  # noqa: F401

    _HAS_DEMESTATS = True
except ImportError:
    _HAS_DEMESTATS = False


def coalescence_rate_same_class_pop(k: int, p_class: float, ne: float) -> float:
    """Pairwise coalescence rate among k same-(class, pop) lineages.

    rate = k*(k-1)/2 / (p_class * Ne)
    """
    if k < 2:
        return 0.0
    return k * (k - 1) / 2.0 / max(p_class * ne, 1e-300)


def crossover_weight(branch_class, tag_p) -> float:
    """Crossover weight w of one segment while inversion barriers are active.

    A crossover inside an inversion needs a homokaryotypic partner, so a
    lineage carrying arrangement k there recombines at ``r * p_k``. The
    weight is the product of ``tag_p[tag]`` over the segment's class tags
    (one tag per containing inversion, a frozenset for nested inversions).
    Panmictic segments ('P': collinear flanks, untagged material, or
    inversions whose barrier has lifted) and tags missing from ``tag_p``
    get weight 1. Untagged material inside an inversion (samples with a
    ``None`` karyotype) is treated as panmictic, as in the coalescence rate.
    The product over nested or overlapping inversions assumes the
    arrangements are independent, again the coalescence convention.

    Mirrors the Rust ``p_class_for_tag`` applied to the class at the
    breakpoint (``simulator.rs``, Event::Recombination).
    """
    if branch_class is None or branch_class == "P":
        return 1.0
    if isinstance(branch_class, frozenset):
        w = 1.0
        for tag in branch_class:
            w *= tag_p.get(tag, 1.0)
        return w
    return tag_p.get(branch_class, 1.0)


def recombination_rate_for_lineage(lineage, rho_per_unit: float, tag_p) -> float:
    """Effective recombination rate for one lineage.

    ``rho_per_unit`` times the integral over the lineage's ancestral
    material of the crossover weight (see ``crossover_weight``): the full
    rate on panmictic material and ``rho_per_unit * p_k`` on material
    tagged with arrangement k of an inversion whose barrier is active.
    ``tag_p`` maps class tags ('S', 'I', 'S0', 'I1', ...) to the
    arrangement frequency in the lineage's population; pass an empty dict
    for the unsuppressed rate.
    """
    rate = 0.0
    seg = lineage.head
    while seg is not None:
        rate += (seg.right - seg.left) * crossover_weight(seg.branch_class, tag_p)
        seg = seg.next
    return rho_per_unit * rate


def gene_flux_rate_for_lineage(
    lineage,
    gamma: float,
    phi_func,
    bp_left: float,
    bp_right: float,
    p_other_in_pop: float,
) -> float:
    """Gene-flux rate for in-inv segments of one lineage.

    Integrates gamma * phi(x) * p_other across in-inv portions of the
    lineage's segments.
    """
    if gamma <= 0 or p_other_in_pop <= 0:
        return 0.0
    rate = 0.0
    seg = lineage.head
    inv_len = bp_right - bp_left
    if inv_len <= 0:
        return 0.0
    while seg is not None:
        l = max(seg.left, bp_left)
        r = min(seg.right, bp_right)
        if r > l:
            # Approximate ∫ phi((x - bp_left)/inv_len) dx by midpoint rule
            # Phase 1: trapezoidal with endpoints; Phase 2: numerical quad.
            xl = (l - bp_left) / inv_len
            xr = (r - bp_left) / inv_len
            phi_l = phi_func(max(0.02, min(0.98, xl)))
            phi_r = phi_func(max(0.02, min(0.98, xr)))
            rate += gamma * 0.5 * (phi_l + phi_r) * (r - l) * p_other_in_pop
        seg = seg.next
    return rate
