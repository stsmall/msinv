"""Hull-algorithm simulator main loop.

Phase 1 (✓): panmictic, no inversion, no recombination.
Phase 2 (✓): karyotype class barrier (S/I, t_inv).
Phase 3 (✓): gene-flux events with class flip.
Phase 4 (✓): multi-population structure + migration + ms-style
demographic events (en/eN/eg/eG/em/eM/ej). ``HullSimulator`` now
accepts a ``Demography`` instance and a per-(class, pop)
``sample_config``. Migration moves individual lineages between pops
at rate ``M[dst][src]`` per src lineage per generation. ``ej`` events
move all lineages of a population in bulk and zero its migration.
Per-(class, pop) coalescence rate uses the structured-coalescent
scaling ``k(k-1)/2 / (p_class · Ne_pop(t))`` with growth-rate-aware
``Ne_pop(t)``.

Subsequent phases:
  Phase 5: multiple inversions.
  Phase 6: sweep model integration.
  Phase 7: Cython/C inner loop.
"""

import math

import numpy as np

from .lineage import Lineage, reset_uids
from .segment import Segment
from .tables import TableBuilder
from .events import (apply_coalescence, apply_recombination,
                     apply_gene_flux, apply_migration)
from .demography import Demography


# ---------------------------------------------------------------------------
# Gene-flux geometry (Peischl 2013): phi(x) for a fixed-window model.
# ---------------------------------------------------------------------------

def _phi(x: float, w: float) -> float:
    """Probability that the inversion-relative position ``x`` (in (0, 1))
    is covered by a random gene-conversion tract of width ``w`` (also in
    (0, 1)). Peischl et al. 2013 closed form:

        phi(x) = min(x, 1-x, w) / (1 - w)

    Vanishes at the breakpoints (x → 0 or 1) and peaks at x ∈ [w, 1-w].
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    if w >= 1.0:
        return 1.0
    return max(0.0, min(1.0, min(x, 1.0 - x, w) / (1.0 - w)))


def _phi_integral(a: float, b: float, w: float) -> float:
    """Integrate phi over [a, b] (in inversion-relative coordinates).

    Triangular-roof shape with closed form. We split the interval
    into three pieces — rising (x ∈ [0, w]), flat (x ∈ [w, 1-w]),
    falling (x ∈ [1-w, 1]) — and sum the contributions."""
    if w >= 1.0:
        return max(0.0, min(1.0, b) - max(0.0, a))
    a = max(0.0, a)
    b = min(1.0, b)
    if b <= a:
        return 0.0
    denom = 1.0 - w
    total = 0.0
    # Rising part: phi = x / denom on [0, w]
    lo = max(a, 0.0); hi = min(b, w)
    if hi > lo:
        total += 0.5 * (hi * hi - lo * lo) / denom
    # Flat part: phi = w / denom on [w, 1-w]
    lo = max(a, w); hi = min(b, 1.0 - w)
    if hi > lo:
        total += w * (hi - lo) / denom
    # Falling part: phi = (1 - x) / denom on [1-w, 1]
    lo = max(a, 1.0 - w); hi = min(b, 1.0)
    if hi > lo:
        total += ((hi - lo) - 0.5 * (hi * hi - lo * lo)) / denom
    return total


# ---------------------------------------------------------------------------
# Per-pair overlap helpers (Phase 5)
# ---------------------------------------------------------------------------

def _overlap_by_class(lin_a, lin_b) -> dict:
    """Total overlap length between lin_a and lin_b, bucketed by
    matching segment class. Returns ``{'S': ..., 'I': ..., 'P': ...}``.

    Only counts overlap at positions where BOTH lineages have ancestral
    material AND their classes at that position match. This is the
    mathematically correct compatibility filter for the per-pair
    structured-coalescent rate.
    """
    out = {'S': 0.0, 'I': 0.0, 'P': 0.0}
    sa = lin_a.head
    sb = lin_b.head
    while sa is not None and sb is not None:
        if sa.right <= sb.left:
            sa = sa.next
            continue
        if sb.right <= sa.left:
            sb = sb.next
            continue
        l = max(sa.left, sb.left)
        r = min(sa.right, sb.right)
        if r > l and sa.branch_class == sb.branch_class:
            out[sa.branch_class] = out.get(sa.branch_class, 0.0) + (r - l)
        if sa.right < sb.right:
            sa = sa.next
        else:
            sb = sb.next
    return out


def _coalesce_partial(active, lin_a, lin_b, t, tables, allowed_class):
    """Coalesce ``lin_a`` and ``lin_b`` ONLY at overlap positions where
    BOTH segments have class ``allowed_class``. Other overlap and
    non-overlap positions remain on the original lineages.

    Returns the new internal node id.
    """
    from .events import Lineage as _Lineage  # avoid circular import
    new_node = tables.add_internal(time=t, population=lin_a.population)
    # Build new merged segment list (allowed-class overlap) + leftover
    # segment lists (everything else for each lineage).
    merged_head = merged_tail = None
    a_remain_head = a_remain_tail = None
    b_remain_head = b_remain_tail = None

    def _append(head_attr, tail_attr, seg):
        head, tail = head_attr
        seg.prev = tail
        seg.next = None
        if head is None:
            head = seg
        if tail is not None:
            tail.next = seg
        return seg, seg if head is None else head, seg

    def _emit_merged(l, r):
        nonlocal merged_head, merged_tail
        from .segment import Segment
        seg = Segment(l, r, new_node, branch_class=allowed_class,
                      prev=merged_tail)
        if merged_head is None:
            merged_head = seg
        if merged_tail is not None:
            merged_tail.next = seg
        merged_tail = seg

    def _emit_a(l, r, src_seg):
        nonlocal a_remain_head, a_remain_tail
        from .segment import Segment
        seg = Segment(l, r, src_seg.node_id,
                      branch_class=src_seg.branch_class,
                      prev=a_remain_tail)
        if a_remain_head is None:
            a_remain_head = seg
        if a_remain_tail is not None:
            a_remain_tail.next = seg
        a_remain_tail = seg

    def _emit_b(l, r, src_seg):
        nonlocal b_remain_head, b_remain_tail
        from .segment import Segment
        seg = Segment(l, r, src_seg.node_id,
                      branch_class=src_seg.branch_class,
                      prev=b_remain_tail)
        if b_remain_head is None:
            b_remain_head = seg
        if b_remain_tail is not None:
            b_remain_tail.next = seg
        b_remain_tail = seg

    sa = lin_a.head; sb = lin_b.head
    while sa is not None or sb is not None:
        if sa is None:
            _emit_b(sb.left, sb.right, sb)
            sb = sb.next; continue
        if sb is None:
            _emit_a(sa.left, sa.right, sa)
            sa = sa.next; continue
        if sa.right <= sb.left:
            _emit_a(sa.left, sa.right, sa)
            sa = sa.next; continue
        if sb.right <= sa.left:
            _emit_b(sb.left, sb.right, sb)
            sb = sb.next; continue
        # Some overlap.
        l = max(sa.left, sb.left)
        r = min(sa.right, sb.right)
        # Bits before overlap stay on their lineage.
        if sa.left < l:
            _emit_a(sa.left, l, sa)
        if sb.left < l:
            _emit_b(sb.left, l, sb)
        # Overlap: if classes match `allowed_class`, MERGE into new
        # node; otherwise it stays on both lineages.
        if (sa.branch_class == allowed_class and
                sb.branch_class == allowed_class):
            tables.add_edge(l, r, new_node, sa.node_id)
            tables.add_edge(l, r, new_node, sb.node_id)
            _emit_merged(l, r)
        else:
            _emit_a(l, r, sa)
            _emit_b(l, r, sb)
        # Advance.
        if sa.right == r:
            sa = sa.next
        else:
            from .segment import Segment as _S
            sa = _S(r, sa.right, sa.node_id,
                    branch_class=sa.branch_class, next=sa.next)
            if sa.next is not None:
                sa.next.prev = sa
        if sb.right == r:
            sb = sb.next
        else:
            from .segment import Segment as _S
            sb = _S(r, sb.right, sb.node_id,
                    branch_class=sb.branch_class, next=sb.next)
            if sb.next is not None:
                sb.next.prev = sb

    # Replace lin_a, lin_b in active with whatever remains + the merged
    # lineage, where each is non-empty.
    from .lineage import Lineage as _L
    active.remove(lin_a)
    active.remove(lin_b)
    if merged_head is not None:
        active.append(_L(merged_head, merged_tail,
                          population=lin_a.population))
    if a_remain_head is not None:
        active.append(_L(a_remain_head, a_remain_tail,
                          population=lin_a.population))
    if b_remain_head is not None:
        active.append(_L(b_remain_head, b_remain_tail,
                          population=lin_b.population))
    return new_node


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class HullSimulator:
    """Hull-algorithm simulator.

    Parameters
    ----------
    samples : int, optional
        Total samples (panmictic mode). Mutually exclusive with
        ``n_std``/``n_inv``.
    n_std, n_inv : int, optional
        S- and I-class samples (structured mode). Requires ``p_inv`` and
        ``t_inv``.
    population_size : float
        Effective population size (scales coalescent times to generations).
    sequence_length : float
        Sequence length, in the same units as ``bp_left`` and
        ``bp_right``.
    recombination_rate : float
        Per-bp per-generation recombination rate. (Phase 4+; not yet
        used here.)
    p_inv : float, optional
        Inverted-arrangement frequency in (0, 1). Required when
        ``n_inv > 0``.
    t_inv : float, optional
        Inversion age in generations. Required when ``n_inv > 0``.
    bp_left, bp_right : float, optional
        Inversion breakpoints in genomic coordinates. Required when
        ``n_inv > 0``.
    gene_conversion_rate : float, optional
        Per-bp per-generation gene-conversion rate (γ_per_bp). Defaults
        to 0 (no gene flux). Combined with ``flux_window`` and
        per-position ``phi(x)`` to give the per-lineage flux rate.
    flux_window : float, optional
        Gene-conversion tract length as a fraction of the inversion's
        genomic length (Peischl model). Default 0.05 (i.e. ~5% of the
        inversion length per tract; for a 100 kb inversion, ~5 kb).
    seed : int, optional
    """

    def __init__(self, *, samples: int = None,
                 n_std: int = None, n_inv: int = None,
                 sample_config: dict = None,
                 population_size: float = 1.0,
                 demography: Demography = None,
                 sequence_length: float = 1.0,
                 recombination_rate: float = 0.0,
                 p_inv: float = None,
                 t_inv: float = None,
                 bp_left: float = None,
                 bp_right: float = None,
                 gene_conversion_rate: float = 0.0,
                 flux_window: float = 0.05,
                 seed: int = None):
        """Resolve sample counts (Phase 1-3 args still supported for
        single-pop work; Phase 4 introduces ``sample_config`` and
        ``demography`` for multi-pop work).
        """
        # ---- Sample resolution ----
        # sample_config: {(class_or_None, pop): n}
        if sample_config is not None:
            if (samples is not None or n_std is not None or
                    n_inv is not None):
                raise ValueError(
                    "Use either `sample_config` OR the simpler "
                    "`samples`/`n_std`/`n_inv` API, not both.")
            # sample_config keys are tuples; first element class, second pop.
            self.sample_config = dict(sample_config)
            self.n_std = sum(c for (cls, _), c in self.sample_config.items()
                             if cls in (None, 'S'))
            self.n_inv = sum(c for (cls, _), c in self.sample_config.items()
                             if cls == 'I')
        elif samples is not None:
            if n_std is not None or n_inv is not None:
                raise ValueError(
                    "Pass either `samples` (panmictic) or "
                    "`n_std`/`n_inv` (structured), not both.")
            self.n_std = samples
            self.n_inv = 0
            self.sample_config = {(None, 0): samples}
        else:
            self.n_std = n_std if n_std is not None else 0
            self.n_inv = n_inv if n_inv is not None else 0
            if self.n_std + self.n_inv == 0:
                raise ValueError(
                    "Must pass `samples`, `sample_config`, or "
                    "non-zero `n_std`/`n_inv`.")
            self.sample_config = {}
            if self.n_std:
                self.sample_config[('S', 0)] = self.n_std
            if self.n_inv:
                self.sample_config[('I', 0)] = self.n_inv
        self.samples = self.n_std + self.n_inv

        # ---- Demography ----
        if demography is not None:
            self.demography = demography.copy()
        else:
            self.demography = Demography([population_size])
        # Sanity: every sample's pop must exist.
        for (cls, pop), n in self.sample_config.items():
            if pop >= self.demography.n_pops:
                raise ValueError(
                    f"Sample pop {pop} not in demography "
                    f"(n_pops={self.demography.n_pops}).")

        # Ne for backward compatibility (only meaningful when single pop).
        self.Ne = self.demography.pop_sizes[0]

        # ---- Inversion ----
        if self.n_inv > 0:
            if p_inv is None or not (0.0 < p_inv < 1.0):
                raise ValueError(
                    "p_inv must be in (0, 1) when n_inv > 0.")
            if t_inv is None or t_inv <= 0.0:
                raise ValueError(
                    "t_inv > 0 must be given when n_inv > 0.")
            if bp_left is None or bp_right is None:
                bp_left = 0.0
                bp_right = sequence_length
            if bp_right <= bp_left:
                raise ValueError(
                    f"bp_right must be > bp_left, got "
                    f"({bp_left}, {bp_right}).")
            self.p_inv = p_inv
            self.t_inv = t_inv
            self.bp_left = bp_left
            self.bp_right = bp_right
        else:
            self.p_inv = None
            self.t_inv = None
            self.bp_left = None
            self.bp_right = None

        self.L = sequence_length
        self.r = recombination_rate
        self.g_per_bp = float(gene_conversion_rate)
        if not (0.0 < flux_window < 1.0):
            raise ValueError(
                f"flux_window must be in (0, 1), got {flux_window}.")
        self.flux_window = flux_window
        self.rng = np.random.default_rng(seed)

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _overlap_by_class_static(a, b):
        return _overlap_by_class(a, b)

    def _initial_lineages(self, tables: TableBuilder):
        """Create one sample lineage per sample, honouring sample_config
        (per-(class, pop) initial sample counts). Iterates the dict in
        insertion order so user controls sample-id mapping.

        Phase 5: each sample's initial segments are split at inversion
        boundaries — inside-inv segments carry the sample's karyotype
        class ('S' or 'I'), outside-inv segments carry 'P' (panmictic).
        """
        from .segment import make_initial_segments

        # Build a tiny "inversion" record for make_initial_segments.
        inversions = []
        if self.bp_left is not None and self.bp_right is not None:
            class _InvSpec:
                pass
            inv = _InvSpec()
            inv.bp_left = self.bp_left
            inv.bp_right = self.bp_right
            inversions.append(inv)

        active = []
        for (cls, pop), count in self.sample_config.items():
            # Sample has a karyotype assignment for the inversion.
            # ``cls`` of None means "no inversion" → all segments 'P'.
            sample_cls = cls if cls in ('S', 'I') else None
            for _ in range(count):
                nid = tables.add_sample(time=0.0, population=pop)
                head, tail = make_initial_segments(
                    self.L, nid, inversions=inversions,
                    sample_class=sample_cls)
                active.append(Lineage(head, tail, population=pop))
        return active

    # -- rate helpers ------------------------------------------------------

    def _coal_rates(self, active, t: float):
        """Coalescence rates at time t (Phase 5: per-pair, per-event-type).

        Each pair (A, B) of same-pop lineages has up to three event
        types:

          'outside'  — coalesce ONLY at outside-inv overlap (both 'P').
                       Rate = 1/(2·Ne_pop(t)) if any 'P-P' overlap.
          'inside_S' — coalesce ONLY at S-S overlap inside the inv.
                       Rate = 1/(2·Ne_pop(t)·p_std) if any 'S-S' overlap.
          'inside_I' — coalesce ONLY at I-I overlap inside the inv.
                       Rate = 1/(2·Ne_pop(t)·p_inv) if any 'I-I' overlap.

        Each event type's rate has the standard per-pair structured-
        coalescent form (1/(2·Ne·p_class)). When an event fires, the
        pair coalesces ONLY at overlap positions of that event type;
        other positions stay separate.

        After t_inv the class barrier lifts: every pair has a single
        'panmictic' event with rate 1/(2·Ne_pop(t)).

        Returns list of (kind, rate, payload) where payload =
        ``(i, j, event_kind)`` for pair-events, or
        ``(idx_list, panmictic_kind)`` for the post-t_inv catch-all.
        """
        rates = []
        inv_active = self.p_inv is not None and (
            self.t_inv is None or t < self.t_inv)

        if not inv_active:
            # Post-t_inv: panmictic-by-pop bucket.
            buckets = {}
            for i, lin in enumerate(active):
                buckets.setdefault(lin.population, []).append(i)
            for pop, idx_list in buckets.items():
                k = len(idx_list)
                if k < 2:
                    continue
                ne_pop = max(self.demography.size_at(pop, t), 1e-9)
                rate = k * (k - 1) / 2.0 / (2.0 * ne_pop)
                rates.append((f'coal_panmictic_{pop}', rate, idx_list))
            return rates

        # Inversion is still active: enumerate pairs in same-pop.
        # Per-pair rate split by event type (outside / inside_S / inside_I).
        p_std = 1.0 - self.p_inv
        for i in range(len(active)):
            lin_i = active[i]
            for j in range(i + 1, len(active)):
                lin_j = active[j]
                if lin_i.population != lin_j.population:
                    continue
                ovl = _overlap_by_class(lin_i, lin_j)
                ne_pop = max(
                    self.demography.size_at(lin_i.population, t), 1e-9)
                if ovl['P'] > 0:
                    rates.append((
                        'pair_outside',
                        1.0 / (2.0 * ne_pop),
                        (i, j, 'outside')))
                if ovl['S'] > 0:
                    rates.append((
                        'pair_inside_S',
                        1.0 / (2.0 * ne_pop * p_std),
                        (i, j, 'inside_S')))
                if ovl['I'] > 0:
                    rates.append((
                        'pair_inside_I',
                        1.0 / (2.0 * ne_pop * self.p_inv),
                        (i, j, 'inside_I')))
        return rates

    def _migration_rates(self, active, t: float):
        """Per-lineage migration rates at time t.

        Returns list of ('mig', rate, (lineage_idx, dst_pop)) where
        each tuple is one (lineage, destination) pair contributing
        rate ``M[dst][src]`` per generation.
        """
        rates = []
        if self.demography.n_pops < 2:
            return rates
        M = self.demography.migration_matrix
        for i, lin in enumerate(active):
            src = lin.population
            for dst in range(self.demography.n_pops):
                if dst == src:
                    continue
                m = M[dst][src]
                if m > 0:
                    rates.append(('mig', m, (i, dst)))
        return rates

    def _flux_lineage_weight(self, lineage):
        """Per-lineage gene-flux weight: ∫_inv phi(x) dx over the
        lineage's in-inv ancestral material, in inversion-relative
        coordinates (so the resulting weight × g_per_bp × inv_len ×
        p_other gives a per-generation rate in 1/gen).
        """
        if self.bp_left is None:
            return 0.0
        inv_len = self.bp_right - self.bp_left
        if inv_len <= 0:
            return 0.0
        w = self.flux_window
        weight = 0.0
        seg = lineage.head
        while seg is not None:
            l = max(seg.left, self.bp_left)
            r = min(seg.right, self.bp_right)
            if r > l:
                a = (l - self.bp_left) / inv_len
                b = (r - self.bp_left) / inv_len
                weight += _phi_integral(a, b, w) * inv_len
            seg = seg.next
        return weight

    def _flux_rates(self, active):
        """List of (kind, rate, lineage_idx) for gene-flux events.

        Each entry corresponds to ONE lineage's gene-flux rate.
        """
        if self.p_inv is None or self.g_per_bp <= 0:
            return []
        p_std = 1.0 - self.p_inv
        rates = []
        for idx, lin in enumerate(active):
            if lin.branch_class == 'S':
                p_other = self.p_inv
            elif lin.branch_class == 'I':
                p_other = p_std
            else:
                continue
            if p_other <= 0:
                continue
            w_lin = self._flux_lineage_weight(lin)
            if w_lin <= 0:
                continue
            rate = self.g_per_bp * p_other * w_lin
            if rate > 0:
                rates.append(('flux', rate, idx))
        return rates

    def _flip_to_panmictic(self, active):
        for lin in active:
            lin.branch_class = 'S'
        self.p_inv = None
        self.t_inv = None
        # Gene flux is also gone after t_inv (no class barrier → no
        # heterokaryotypes → no gene-conversion events).
        self.g_per_bp = 0.0

    # -- gene-flux event helper -------------------------------------------

    def _sample_flux_position(self, lineage):
        """Sample a gene-flux event position uniformly weighted by
        phi(x) over ``lineage``'s in-inv ancestral material.

        Returns the genomic position where the conversion CENTRES
        (call it x_event). The tract is then drawn around it via the
        Peischl b1-uniform construction.
        """
        inv_len = self.bp_right - self.bp_left
        w = self.flux_window
        # Walk segments, build CDF over phi-weighted in-inv material.
        intervals = []
        cum = 0.0
        seg = lineage.head
        while seg is not None:
            l = max(seg.left, self.bp_left)
            r = min(seg.right, self.bp_right)
            if r > l:
                a = (l - self.bp_left) / inv_len
                b = (r - self.bp_left) / inv_len
                weight = _phi_integral(a, b, w) * inv_len
                intervals.append((l, r, a, b, weight))
                cum += weight
            seg = seg.next
        if cum <= 0.0:
            return None
        # Pick an interval by weight.
        u = self.rng.random() * cum
        running = 0.0
        chosen = intervals[-1]
        for entry in intervals:
            running += entry[4]
            if u < running:
                chosen = entry
                break
        l, r, a, b, weight = chosen
        # Within this interval, sample x by phi-density via rejection.
        # Triangular bound for phi: max value is min(1, w/(1-w)).
        phi_max = w / (1.0 - w) if w < 1.0 else 1.0
        for _ in range(1000):
            xx = self.rng.uniform(a, b)
            if self.rng.random() * phi_max < _phi(xx, w):
                # Convert back to genomic coords.
                return self.bp_left + xx * inv_len
        # Fallback — sample uniformly in the chosen segment.
        return self.rng.uniform(l, r)

    def _draw_tract(self, x_event):
        """Given a conversion-event centre ``x_event`` in genomic
        coords, draw a tract [tract_left, tract_right) in genomic
        coords using the Peischl b1-uniform construction.

        b1 is uniform in [max(0, x-w_g), min(L_inv-w_g, x)] (with
        w_g = flux_window * inv_len). Tract is [b1, b1 + w_g] within
        the inversion, clipped to inv bounds.
        """
        inv_len = self.bp_right - self.bp_left
        w_g = self.flux_window * inv_len
        x_rel = x_event - self.bp_left
        b1_lo = max(0.0, x_rel - w_g)
        b1_hi = min(inv_len - w_g, x_rel)
        if b1_hi <= b1_lo:
            # x_event near edge — clip
            b1 = max(0.0, min(inv_len - w_g, x_rel - w_g / 2.0))
        else:
            b1 = self.rng.uniform(b1_lo, b1_hi)
        tract_left = self.bp_left + b1
        tract_right = tract_left + w_g
        # Clip to inversion bounds.
        tract_left = max(self.bp_left, tract_left)
        tract_right = min(self.bp_right, tract_right)
        return tract_left, tract_right

    # -- main loop ---------------------------------------------------------

    def simulate(self):
        """Run one replicate. Returns a tskit ``TreeSequence``."""
        reset_uids()
        tables = TableBuilder(sequence_length=self.L,
                               num_populations=self.demography.n_pops)
        active = self._initial_lineages(tables)

        t = 0.0
        t_inv = self.t_inv

        max_iters = 10_000_000
        for _ in range(max_iters):
            if len(active) <= 1:
                if len(active) == 0 or active[0].total_length >= self.L - 1e-9:
                    break
                break

            # Compute event rates.
            coal = self._coal_rates(active, t)
            flux = self._flux_rates(active)
            mig = self._migration_rates(active, t)
            all_events = coal + flux + mig
            total = sum(r for _, r, _ in all_events)

            # Time of the next demographic event (or +inf).
            t_demo = self.demography.next_event_time(t)
            t_class = t_inv if t_inv is not None else float('inf')

            # If no per-event rate, advance to the next scheduled
            # event boundary (demographic or class barrier).
            if total <= 0:
                next_boundary = min(t_demo, t_class)
                if next_boundary == float('inf'):
                    raise RuntimeError(
                        "No events possible and no scheduled boundaries "
                        f"to advance to — stuck with {len(active)} "
                        "active lineages.")
                t = next_boundary
                if next_boundary == t_class:
                    self._flip_to_panmictic(active)
                    t_inv = None
                else:
                    self.demography.apply_event_at(t, active)
                continue

            dt = self.rng.exponential(1.0 / total)
            t_event = t + dt

            # Class-barrier crossing
            if t_class < t_event:
                t = t_class
                self._flip_to_panmictic(active)
                t_inv = None
                continue
            # Demographic event crossing
            if t_demo < t_event:
                t = t_demo
                self.demography.apply_event_at(t, active)
                continue
            t = t_event

            # Pick which event.
            u = self.rng.random() * total
            cum = 0.0
            chosen_kind = None
            chosen_payload = None
            for kind, rate, payload in all_events:
                cum += rate
                if u < cum:
                    chosen_kind = kind
                    chosen_payload = payload
                    break

            if chosen_kind is None:
                continue  # numerical-precision miss

            if chosen_kind.startswith('coal_'):
                # Post-t_inv panmictic catch-all: payload is a list of
                # lineage indices; pick two to coalesce.
                pool = chosen_payload
                ii, jj = self.rng.choice(len(pool), size=2, replace=False)
                i, j = pool[ii], pool[jj]
                apply_coalescence(active, active[i], active[j], t, tables)
            elif chosen_kind.startswith('pair_'):
                # Per-pair, per-event-type Phase-5 dispatch. Payload is
                # ``(i, j, event_kind)`` with event_kind in
                # {'outside', 'inside_S', 'inside_I'}.
                i, j, ek = chosen_payload
                allowed = {'outside': 'P',
                           'inside_S': 'S',
                           'inside_I': 'I'}[ek]
                _coalesce_partial(active, active[i], active[j], t,
                                   tables, allowed)
            elif chosen_kind == 'flux':
                idx = chosen_payload
                lineage = active[idx]
                x_event = self._sample_flux_position(lineage)
                if x_event is None:
                    continue
                tract_left, tract_right = self._draw_tract(x_event)
                if tract_right <= tract_left:
                    continue
                apply_gene_flux(active, lineage, tract_left, tract_right)
            elif chosen_kind == 'mig':
                idx, dst = chosen_payload
                apply_migration(active[idx], dst)
            else:
                raise RuntimeError(f"Unknown event kind: {chosen_kind}")
        else:
            raise RuntimeError(
                f"max_iters ({max_iters}) exceeded — likely a runaway "
                f"event loop.")

        return tables.finalize()
