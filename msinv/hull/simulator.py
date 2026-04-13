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
from .inversion import InversionSpec
from .sweep import Sweep


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
    matching segment class. Returns a dict keyed by class.

    Only counts overlap at positions where BOTH lineages have ancestral
    material AND their classes at that position match (segment classes
    compare with ``==``, which works for both string tags and
    frozensets used by Phase 5c.2 nested inversions).
    """
    out = {}
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
            key = sa.branch_class
            out[key] = out.get(key, 0.0) + (r - l)
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
                 inversions: list = None,
                 sweeps: list = None,
                 seed: int = None):
        """Resolve sample counts (Phase 1-3 args still supported for
        single-pop work; Phase 4 introduces ``sample_config`` and
        ``demography`` for multi-pop work).
        """
        # ---- Sample resolution ----
        # sample_config: {(karyotype, pop): n} where karyotype is
        # None, 'S', 'I', or a per-inv sequence.
        if sample_config is not None:
            if (samples is not None or n_std is not None or
                    n_inv is not None):
                raise ValueError(
                    "Use either `sample_config` OR the simpler "
                    "`samples`/`n_std`/`n_inv` API, not both.")
            self.sample_config = dict(sample_config)
            # n_std/n_inv counts: a sample is "n_std"-counted if it is
            # NOT 'I' at the FIRST inversion (back-compat heuristic for
            # the n_std/n_inv accessors that single-inv code uses).
            def _is_inv_sample(kary):
                if kary is None or kary == 'S':
                    return False
                if kary == 'I':
                    return True
                # Sequence/multi-char string: look at first entry.
                if hasattr(kary, '__iter__'):
                    first = next(iter(kary), None)
                    return first == 'I'
                return False
            self.n_std = sum(c for (cls, _), c in self.sample_config.items()
                             if not _is_inv_sample(cls))
            self.n_inv = sum(c for (cls, _), c in self.sample_config.items()
                             if _is_inv_sample(cls))
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

        # ---- Inversions ----
        # Two acceptable APIs:
        # 1) inversions=[InversionSpec(...), ...]  — multi-inv (Phase 5b)
        # 2) bp_left/bp_right/p_inv/t_inv/gene_conversion_rate +
        #    flux_window — back-compat single-inv (Phases 2-5a)
        if inversions:
            if any(x is not None for x in (p_inv, t_inv, bp_left, bp_right)):
                raise ValueError(
                    "Pass either `inversions=[...]` OR the legacy "
                    "single-inv args (bp_left/bp_right/p_inv/t_inv), "
                    "not both.")
            self.inversions = []
            for i, spec in enumerate(inversions):
                # Allow tuple shorthand or full InversionSpec.
                if not isinstance(spec, InversionSpec):
                    spec = InversionSpec(*spec) if isinstance(spec, tuple) \
                        else InversionSpec(**dict(spec))
                spec.inv_id = i
                self.inversions.append(spec)
            # Sort by bp_left; nested/overlapping inversions ARE
            # supported (Phase 5c.2) via per-position tag sets in
            # ``Segment.branch_class``.
            self.inversions.sort(key=lambda inv: inv.bp_left)
            # Back-compat single-inv attributes (used in some helpers).
            inv0 = self.inversions[0]
            self.p_inv = inv0.p_inv
            self.t_inv = inv0.t_inv
            self.bp_left = inv0.bp_left
            self.bp_right = inv0.bp_right
            # Gene flux: use first inversion's γ for back-compat. Per-inv
            # γ is read directly from each spec where it matters.
            self.g_per_bp = float(inv0.gene_conversion_rate)
            self.flux_window = inv0.flux_window
        elif self.n_inv > 0:
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
            self.g_per_bp = float(gene_conversion_rate)
            if not (0.0 < flux_window < 1.0):
                raise ValueError(
                    f"flux_window must be in (0, 1), got {flux_window}.")
            self.flux_window = flux_window
            # Single-inv mode: build a single InversionSpec with inv_id=-1
            # (sentinel) so initial-segment classes use plain 'S'/'I'
            # without inv_id suffix — preserves Phase 1-5a semantics.
            single = InversionSpec(
                bp_left=bp_left, bp_right=bp_right,
                p_inv=p_inv, t_inv=t_inv,
                gene_conversion_rate=self.g_per_bp,
                flux_window=flux_window, inv_id=-1)
            self.inversions = [single]
        else:
            self.p_inv = None
            self.t_inv = None
            self.bp_left = None
            self.bp_right = None
            self.g_per_bp = float(gene_conversion_rate)
            self.flux_window = flux_window
            self.inversions = []

        self.L = sequence_length
        self.r = recombination_rate
        # Sweeps: list of Sweep objects, sorted by t_event.
        self.sweeps = []
        if sweeps:
            for s in sweeps:
                if not isinstance(s, Sweep):
                    s = Sweep(*s) if isinstance(s, tuple) else Sweep(**dict(s))
                self.sweeps.append(s)
            self.sweeps.sort(key=lambda s: s.t_event)
        self.rng = np.random.default_rng(seed)

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _overlap_by_class_static(a, b):
        return _overlap_by_class(a, b)

    def _initial_lineages(self, tables: TableBuilder):
        """Create one sample lineage per sample, honouring sample_config.

        sample_config keys are ``(karyotype, pop)`` tuples where
        ``karyotype`` is:
          - ``None``                 — purely panmictic (no inversion).
          - ``'S'`` or ``'I'``       — linked karyotype across all invs.
          - tuple/list of len n_inv  — independent karyotype per inv;
            entries are ``'S'``, ``'I'``, or ``None``.

        For multi-inversion linked-karyotype back-compat: a 2-character
        string like ``'SI'`` is interpreted as a tuple ``('S', 'I')``
        (i.e. S in inv 0, I in inv 1). Plain ``'S'`` / ``'I'`` remain
        the linked-karyotype shorthand.
        """
        from .segment import make_initial_segments

        n_inv_specs = len(self.inversions)
        active = []
        for (karyotype, pop), count in self.sample_config.items():
            # Normalise the karyotype value.
            if karyotype is None:
                sample_cls = None
            elif isinstance(karyotype, str) and len(karyotype) == 1:
                # 'S' or 'I' — linked karyotype.
                sample_cls = karyotype
            elif (isinstance(karyotype, str)
                    and len(karyotype) == n_inv_specs
                    and n_inv_specs > 1):
                # 'SI' / 'IS' / 'II' / 'SS' for multi-inv shorthand.
                sample_cls = tuple(karyotype)
            elif hasattr(karyotype, '__iter__'):
                sample_cls = tuple(karyotype)
            else:
                raise ValueError(
                    f"Unrecognized sample karyotype: {karyotype!r}")
            for _ in range(count):
                nid = tables.add_sample(time=0.0, population=pop)
                head, tail = make_initial_segments(
                    self.L, nid, inversions=self.inversions,
                    sample_class=sample_cls)
                active.append(Lineage(head, tail, population=pop))
        return active

    # -- rate helpers ------------------------------------------------------

    def _coal_rates(self, active, t: float):
        """Coalescence rates at time t (Phase 5b: multi-inversion).

        Each pair (A, B) of same-pop lineages has up to (1 + 2·n_inv)
        event types:
          'outside' (P-P), and per-inversion ('S<k>'-'S<k>') and
          ('I<k>'-'I<k>') events. Each event's rate uses standard
          per-pair structured-coalescent scaling.

        After ALL inversions' barriers have lifted (their t_inv passed),
        the simulator falls back to a single panmictic-by-pop bucket
        for efficiency. While ANY inversion is still active we use the
        per-pair, per-class enumeration.

        Returns list of (kind, rate, payload).
        """
        rates = []
        # Determine the set of currently active inversion classes.
        active_inv_classes = set()
        any_inv_active = False
        for inv in self.inversions:
            if t < inv.t_inv:
                any_inv_active = True
                active_inv_classes.add(inv.class_S())
                active_inv_classes.add(inv.class_I())
                # Single-inv back-compat alias
                if inv.inv_id == -1:
                    active_inv_classes.add('S')
                    active_inv_classes.add('I')

        if not any_inv_active:
            # All inversions retired → bucket by pop, single panmictic
            # event per pop.
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

        # Build a tag → effective sub-pop frequency lookup for rate
        # scaling. A position's class is a string tag (single inv) or
        # a frozenset of tags (nested invs). The per-pair coal rate
        # at that class is 1/(2·Ne·product_of_p_class).
        inv_p_class = {}  # tag → p_class for currently-active inversions
        for inv in self.inversions:
            if t >= inv.t_inv:
                continue  # barrier lifted, segments will be retagged
            p_std = 1.0 - inv.p_inv
            cls_S = inv.class_S() if inv.inv_id != -1 else 'S'
            cls_I = inv.class_I() if inv.inv_id != -1 else 'I'
            inv_p_class[cls_S] = p_std
            inv_p_class[cls_I] = inv.p_inv

        def _p_class_for(cls):
            """Effective sub-pop frequency for a class (string or
            frozenset). 'P' / empty / unknown → panmictic (1.0)."""
            if cls == 'P' or cls is None:
                return 1.0
            if isinstance(cls, frozenset):
                if not cls:
                    return 1.0
                p = 1.0
                for tag in cls:
                    p *= inv_p_class.get(tag, 1.0)
                return p
            # Single string tag
            return inv_p_class.get(cls, 1.0)

        # Per-pair, per-class enumeration.
        for i in range(len(active)):
            lin_i = active[i]
            for j in range(i + 1, len(active)):
                lin_j = active[j]
                if lin_i.population != lin_j.population:
                    continue
                ovl = _overlap_by_class(lin_i, lin_j)
                ne_pop = max(
                    self.demography.size_at(lin_i.population, t), 1e-9)
                for cls_key, ov_len in ovl.items():
                    if ov_len <= 0:
                        continue
                    p_class = _p_class_for(cls_key)
                    if p_class <= 0:
                        continue
                    if cls_key == 'P':
                        kind = 'pair_outside'
                    elif isinstance(cls_key, frozenset):
                        kind = 'pair_inside_nested'
                    else:
                        kind = f'pair_inside_{cls_key}'
                    rates.append((
                        kind,
                        1.0 / (2.0 * ne_pop * p_class),
                        (i, j, cls_key)))
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

    def _apply_sweep(self, active, sweep, t, tables):
        """Force-coalesce all qualifying lineages at sweep.x_sel.

        A "qualifying" lineage is one with ancestral material at
        ``sweep.x_sel`` whose class at that position matches
        ``sweep.target_class`` (or any class if ``target_class``
        is 'any'), and whose population matches ``sweep.population``
        (or any pop if ``None``).

        The merge happens at positions in
        ``[x_sel - sweep_window, x_sel + sweep_window]``. Material
        outside this window remains on the original lineages.
        """
        x_lo = sweep.x_sel - sweep.sweep_window
        x_hi = sweep.x_sel + sweep.sweep_window
        if x_hi <= x_lo:
            # Single-point sweep: use a tiny epsilon window so we
            # actually have something to merge.
            eps = max(1e-9, self.L * 1e-12)
            x_lo = sweep.x_sel
            x_hi = sweep.x_sel + eps

        qualifying = []
        for lin in active:
            if (sweep.population is not None
                    and lin.population != sweep.population):
                continue
            cls_at_x = lin.class_at(sweep.x_sel)
            if cls_at_x is None:
                continue  # no material at x_sel
            if sweep.target_class != 'any':
                # The segment may carry a single string tag ('S0', 'S',
                # 'P', ...) or a frozenset of tags (Phase 5c.2 nested
                # inversions). For a string target_class, accept either
                # equality with the segment class OR membership in the
                # segment's frozenset. For a frozenset target_class,
                # require subset. This lets sweeps targeting "S in
                # inv 0" fire correctly at positions inside multiple
                # nested inversions.
                if isinstance(cls_at_x, frozenset):
                    if isinstance(sweep.target_class, frozenset):
                        if not sweep.target_class.issubset(cls_at_x):
                            continue
                    else:
                        if sweep.target_class not in cls_at_x:
                            continue
                else:
                    if cls_at_x != sweep.target_class:
                        continue
            qualifying.append(lin)

        if len(qualifying) < 2:
            return None  # nothing to coalesce

        # Force-coalesce all qualifying lineages into a single sweep
        # ancestor at time t. We do this by sequentially merging via
        # _coalesce_partial restricted to the sweep window.
        # Simplification: we use the existing apply_coalescence (which
        # merges all overlap), but first split each lineage at x_lo
        # and x_hi so the merged piece is exactly the sweep window.
        from .events import apply_coalescence

        # Split each qualifying lineage at x_lo and x_hi so the sweep
        # window is its own segment.
        windowed_lineages = []
        for lin in qualifying:
            # Split at x_lo: (left of lo, rest)
            a, rest = lin.split_at(x_lo)
            if rest is None:
                # Material doesn't extend to x_lo; restore lin
                continue
            # Split rest at x_hi: (window, right of hi)
            window, b = rest.split_at(x_hi)
            # Re-add the non-window pieces back to active as separate lineages
            active.remove(lin)
            if a is not None:
                active.append(a)
            if b is not None:
                active.append(b)
            if window is not None:
                active.append(window)
                windowed_lineages.append(window)

        # Now sequentially coalesce all windowed lineages. Each
        # successive pair-merge needs a strictly-greater time than the
        # previous one (tskit requires parent.time > child.time);
        # nudge by a tiny epsilon per merge so they're all "at" the
        # sweep time but strictly ordered.
        if len(windowed_lineages) < 2:
            return None
        eps = max(1e-9, t * 1e-12)
        merged = windowed_lineages[0]
        for k_idx, other in enumerate(windowed_lineages[1:], start=1):
            t_merge = t + k_idx * eps
            apply_coalescence(active, merged, other, t_merge, tables)
            merged = active[-1]
        return merged

    def _flip_to_panmictic(self, active, inv_id=None):
        """Flip per-segment classes to 'P' for inversion ``inv_id`` (or
        for ALL inversions if ``inv_id`` is None — used by the
        legacy single-inv code path).
        """
        if inv_id is None:
            # Legacy single-inv flip: every segment becomes 'P', and
            # the back-compat single-inv attrs are zeroed.
            for lin in active:
                seg = lin.head
                while seg is not None:
                    seg.branch_class = 'P'
                    seg = seg.next
            self.p_inv = None
            self.t_inv = None
            self.g_per_bp = 0.0
            return
        # Multi-inv flip: drop the inv_id's tag from every segment.
        # For a string tag exactly matching this inversion's S/I, the
        # segment becomes 'P'. For a frozenset (nested inversions),
        # remove just the matching tags; the segment retains other
        # tags. If after removal the frozenset is empty → 'P'. If
        # exactly one tag remains, collapse to a string.
        cls_S = f'S{inv_id}' if inv_id != -1 else 'S'
        cls_I = f'I{inv_id}' if inv_id != -1 else 'I'
        targets = {cls_S, cls_I}
        for lin in active:
            seg = lin.head
            while seg is not None:
                bc = seg.branch_class
                if isinstance(bc, frozenset):
                    new_bc = bc - targets
                    if not new_bc:
                        seg.branch_class = 'P'
                    elif len(new_bc) == 1:
                        seg.branch_class = next(iter(new_bc))
                    else:
                        seg.branch_class = new_bc
                elif bc == cls_S or bc == cls_I:
                    seg.branch_class = 'P'
                seg = seg.next

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
        # Schedule of remaining inversion-barrier events: list of
        # (t_inv, inv_id) sorted by time. Each fires once.
        pending_barriers = sorted(
            [(inv.t_inv, inv.inv_id) for inv in self.inversions],
            key=lambda x: x[0])
        # Pending sweeps (already sorted by t_event in __init__).
        pending_sweeps = list(self.sweeps)

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
            t_class = pending_barriers[0][0] if pending_barriers else float('inf')
            t_sweep = pending_sweeps[0].t_event if pending_sweeps else float('inf')

            # If no per-event rate, advance to the next scheduled
            # event boundary.
            if total <= 0:
                next_boundary = min(t_demo, t_class, t_sweep)
                if next_boundary == float('inf'):
                    raise RuntimeError(
                        "No events possible and no scheduled boundaries "
                        f"to advance to — stuck with {len(active)} "
                        "active lineages.")
                t = next_boundary
                if next_boundary == t_class:
                    _, inv_id = pending_barriers.pop(0)
                    self._flip_to_panmictic(active, inv_id=inv_id)
                elif next_boundary == t_sweep:
                    sweep = pending_sweeps.pop(0)
                    self._apply_sweep(active, sweep, t, tables)
                else:
                    self.demography.apply_event_at(t, active)
                continue

            dt = self.rng.exponential(1.0 / total)
            t_event = t + dt

            # Class-barrier crossing (per-inversion barrier).
            if t_class < t_event and t_class <= t_sweep and t_class <= t_demo:
                t = t_class
                _, inv_id = pending_barriers.pop(0)
                self._flip_to_panmictic(active, inv_id=inv_id)
                continue
            # Sweep crossing
            if t_sweep < t_event and t_sweep <= t_demo:
                t = t_sweep
                sweep = pending_sweeps.pop(0)
                self._apply_sweep(active, sweep, t, tables)
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
                # Per-pair, per-class dispatch. Payload is
                # ``(i, j, class_string)`` where class_string is the
                # exact segment class to merge ('P', 'S', 'I', 'S0',
                # 'I0', 'S1', 'I1', ...). Other classes' overlap
                # remains on the original lineages.
                i, j, allowed = chosen_payload
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
