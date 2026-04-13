"""Rate computations for the hull simulator.

Wraps ``demestats`` (Ragsdale 2026, doi:10.64898/2026.04.09.717519)
when available for arbitrary structured-coalescent rates over a demes
graph. Falls back to direct calculation for the simple cases.

Currently a STUB — Phase 4 will hook in demestats as the rate engine.
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


def recombination_rate_for_lineage(lineage, rho_per_unit: float,
                                   bp_left: float, bp_right: float,
                                   p_class_in_pop: float) -> float:
    """Effective recombination rate for one lineage.

    Out-of-inv intervals: full rho_per_unit per unit length.
    In-inv intervals: rho_per_unit * p_class_in_pop (only homokaryotype
        recombines normally).
    """
    rate = 0.0
    seg = lineage.head
    while seg is not None:
        out_left = max(0.0, min(seg.right, bp_left) - seg.left)
        in_mid = max(0.0, min(seg.right, bp_right) - max(seg.left, bp_left))
        out_right = max(0.0, seg.right - max(seg.left, bp_right))
        rate += rho_per_unit * (out_left + out_right)
        rate += rho_per_unit * p_class_in_pop * in_mid
        seg = seg.next
    return rate


def gene_flux_rate_for_lineage(lineage, gamma: float, phi_func,
                               bp_left: float, bp_right: float,
                               p_other_in_pop: float) -> float:
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
