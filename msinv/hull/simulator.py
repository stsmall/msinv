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
from .segment import Segment, total_length
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
# Lineage GC — remove "fully coalesced" lineages
# ---------------------------------------------------------------------------

def _gc_sole_lineages(active):
    """Remove lineages whose material doesn't overlap with any other
    lineage at any position (endpoint-sweep, O(S log S)).

    Collects all segment start/end events, sweeps left-to-right
    tracking per-position coverage count. A lineage is "sole" if
    every position it covers has coverage == 1. Sole lineages can
    never produce more edges and are removed.
    """
    if len(active) <= 1:
        return

    # Build sweep events: (position, +1/-1, lineage_index).
    events = []
    for i, lin in enumerate(active):
        seg = lin.head
        while seg is not None:
            events.append((seg.left, 1, i))
            events.append((seg.right, -1, i))
            seg = seg.next
    if not events:
        return
    # Sort by position; at ties, ends (-1) before starts (+1) so
    # count drops before it rises at the same coordinate.
    events.sort(key=lambda e: (e[0], e[1]))

    # Sweep: track which lineages have overlap (coverage > 1 at
    # any of their positions).
    n = len(active)
    has_overlap = [False] * n  # True if lineage i shares a position
    coverage = 0               # current total coverage count
    # Also track which lineages are currently active at this position.
    active_set = set()

    prev_pos = events[0][0]
    for pos, delta, lin_idx in events:
        if pos > prev_pos and coverage > 1:
            # The interval [prev_pos, pos) has coverage > 1 —
            # every lineage active in this interval has overlap.
            for idx in active_set:
                has_overlap[idx] = True
        prev_pos = pos
        if delta > 0:
            active_set.add(lin_idx)
            coverage += 1
        else:
            active_set.discard(lin_idx)
            coverage -= 1

    # Remove lineages with no overlap (in reverse order).
    to_remove = [i for i in range(n) if not has_overlap[i]]
    for idx in reversed(to_remove):
        active.pop(idx)


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
        Per-bp per-generation recombination rate. Fires recombination
        events in the main loop alongside coalescence/flux/migration (
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
        to 0 (no gene flux). Combined with ``mean_tract_length`` and
        per-position ``phi(x)`` to give the per-lineage flux rate.
    mean_tract_length : float, optional
        Mean gene-conversion tract length in bp (Peischl b2 flux model).
        Default 100.0.
    tract_distribution : str, optional
        Tract-length distribution: 'geometric' (default) or 'fixed'.
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
                 gene_conversion_rate: float = 1e-9,
                 mean_tract_length: float = 100.0,
                 tract_distribution: str = 'geometric',
                 inversions: list = None,
                 sweeps: list = None,
                 seed: int = None,
                 stop_at: float = float('inf'),
                 compound_rate: bool = False,
                 iters_max: int = 10_000_000,
                 gc_stride: int = 160,
                 record_events: bool = False):
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
        #    mean_tract_length/tract_distribution — back-compat single-inv (Phases 2-5a)
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
            # Validate every inversion fits within the sequence.
            for inv in self.inversions:
                if inv.bp_left < 0 or inv.bp_right > sequence_length + 1e-9:
                    raise ValueError(
                        f"Inversion {inv} extends outside the sequence "
                        f"[0, {sequence_length}). Either widen "
                        f"`sequence_length` or shrink the inversion "
                        f"breakpoints.")
            # Back-compat single-inv attributes (used in some helpers).
            inv0 = self.inversions[0]
            self.p_inv = inv0.p_inv
            self.t_inv = inv0.t_inv
            self.bp_left = inv0.bp_left
            self.bp_right = inv0.bp_right
            # Gene flux: use first inversion's γ for back-compat. Per-inv
            # γ is read directly from each spec where it matters.
            self.g_per_bp = float(inv0.gene_conversion_rate)
            self.mean_tract_length = inv0.mean_tract_length
            self.tract_distribution = inv0.tract_distribution
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
            self.mean_tract_length = mean_tract_length
            self.tract_distribution = tract_distribution
            # Single-inv mode: build a single InversionSpec with inv_id=-1
            # (sentinel) so initial-segment classes use plain 'S'/'I'
            # without inv_id suffix — preserves Phase 1-5a semantics.
            single = InversionSpec(
                bp_left=bp_left, bp_right=bp_right,
                p_inv=p_inv, t_inv=t_inv,
                gene_conversion_rate=self.g_per_bp,
                mean_tract_length=mean_tract_length,
                tract_distribution=tract_distribution,
                inv_id=-1)
            self.inversions = [single]
        else:
            self.p_inv = None
            self.t_inv = None
            self.bp_left = None
            self.bp_right = None
            self.g_per_bp = float(gene_conversion_rate)
            self.mean_tract_length = mean_tract_length
            self.tract_distribution = tract_distribution
            self.inversions = []

        self.L = sequence_length
        self.r = recombination_rate
        # rho = 0 is forbidden globally. Without recombination,
        # partial coalescence fragments lineages that can never
        # recombine back together (hangs with inversions) and the
        # ARG collapses to a single tree across the whole sequence
        # (no genealogical resolution within the locus). For
        # independent non-recombining loci, simulate each locus
        # separately with its own short sequence_length.
        if self.r <= 0:
            raise ValueError(
                f"recombination_rate must be > 0 (got {self.r}). "
                "rho=0 is not supported. For non-recombining loci, "
                "simulate each locus separately.")
        # Sweeps: the new joint forward WF sweep API (msinv.hull.sweep.Sweep)
        # is implemented only in the Rust backend (see
        # docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md).
        # The legacy Python fallback simulator no longer supports the
        # old t_event/target_class Hudson-Kaplan sweep model; pass
        # ``sweeps=`` through the Rust path (HullSimulator via
        # msinv/hull/_rust_bridge.py) instead.
        self.sweeps = list(sweeps) if sweeps else []
        self.rng = np.random.default_rng(seed)
        self.stop_at = stop_at
        self.compound_rate = compound_rate
        self.iters_max = int(iters_max)
        self.gc_stride = int(gc_stride)
        self._record_events = record_events
        self.event_log = None  # populated after simulate() when record_events=True
        self.sweep_a_count = 0  # count of A-tagged sample lineages after last simulate()
        # Sanity: cross-population reachability. Without a path between
        # populations (via migration or 'ej'), lineages in disjoint pops
        # never coalesce and downstream msprime recap hangs. Warn now
        # rather than mystery-fail later.
        if hasattr(self, 'demography') and self.demography is not None:
            try:
                self.demography.check_connectivity(warn=True)
            except AttributeError:
                pass

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

    # Below this rho, use exact per-pair overlap-by-class coalescence
    # rates (O(n^2)).  Above, use Hudson-style (class, pop) buckets
    # (O(n)) with segment-walking classification and non-overlapping
    # pair rejection (skip_if_no_overlap in apply_coalescence).
    _RHO_THRESHOLD = 100.0

    def _coal_rates(self, active, t: float):
        """Coalescence rates at time t.

        Two regimes:
        1. No active inversions → Hudson per-pop buckets, O(n).
        2. Active inversions → per-pair overlap-by-class with
           _coalesce_partial, O(n^2). Exact.

        Returns list of (kind, rate, payload).
        """
        rates = []
        any_inv_active = any(t < inv.t_inv for inv in self.inversions)

        if not any_inv_active:
            # All inversions retired → Hudson per-pop buckets.
            buckets = {}
            for i, lin in enumerate(active):
                buckets.setdefault(lin.population, []).append(i)
            for pop, idx_list in buckets.items():
                k = len(idx_list)
                if k < 2:
                    continue
                ne_pop = max(self.demography.size_at(pop, t), 1e-9)
                rate = k * (k - 1) / 2.0 / (2.0 * ne_pop)
                rates.append((f'coal_{pop}', rate, idx_list))
            return rates

        # Build per-(class, pop) p_class lookup for active inversions.
        # inv_p_class_map[(tag, pop)] → frequency
        active_inversions = []
        sample_positions = []
        for inv in self.inversions:
            if t >= inv.t_inv:
                continue
            active_inversions.append(inv)
            sample_positions.append(
                ((inv.bp_left + inv.bp_right) / 2.0, inv))

        # Pre-build (tag, pop) → p lookup for O(1) access in inner loops.
        _tag_p_cache = {}
        pops = {lin.population for lin in active}
        for inv in active_inversions:
            cls_S = inv.class_S() if inv.inv_id != -1 else 'S'
            cls_I = inv.class_I() if inv.inv_id != -1 else 'I'
            for pop in pops:
                _tag_p_cache[(cls_S, pop)] = inv.p_std_for(pop)
                _tag_p_cache[(cls_I, pop)] = inv.p_inv_for(pop)

        def _tag_p(tag, pop):
            """Return p for a single class tag in a given population."""
            return _tag_p_cache.get((tag, pop), 1.0)

        def _p_class_for(cls, pop=0):
            if cls == 'P' or cls is None:
                return 1.0
            if isinstance(cls, frozenset):
                if not cls:
                    return 1.0
                p = 1.0
                for tag in cls:
                    p *= _tag_p(tag, pop)
                return p
            return _tag_p(cls, pop)

        # Compute rho to decide which algorithm to use.
        rho = 4.0 * self.demography.pop_sizes[0] * self.r * self.L

        if rho > self._RHO_THRESHOLD and self.r > 0:
            # HIGH RHO: Hudson per-(class, pop) buckets.
            #
            # Classify each lineage by walking ALL segments (not just
            # the midpoint) so fragments retain class identity after
            # recombination.  The rate k*(k-1)/2 / (2*Ne*p_class)
            # overestimates when fragments don't overlap, but the
            # event handler uses rejection sampling to skip non-
            # overlapping pairs (see 'coal_' handler in main loop).
            def _classify(lin):
                """Classify by reading branch_class tags directly.

                Each segment's branch_class is 'S0', 'I1', 'P', or a
                frozenset for nested inversions.  We extract the per-
                inversion class from the tag itself — no position
                check needed.  This avoids misclassification of small
                segments near breakpoints whose midpoint might fall
                outside the inversion bounds.
                """
                inv_tags = {}
                seg = lin.head
                while seg is not None:
                    bc = seg.branch_class
                    if bc != 'P' and bc is not None:
                        if isinstance(bc, frozenset):
                            for tag in bc:
                                for _, inv in sample_positions:
                                    if tag == inv.class_S() or tag == inv.class_I():
                                        inv_tags[inv.inv_id] = tag
                        else:
                            for _, inv in sample_positions:
                                if bc == inv.class_S() or bc == inv.class_I():
                                    inv_tags[inv.inv_id] = bc
                    seg = seg.next
                tags = []
                for _, inv in sample_positions:
                    tags.append(inv_tags.get(inv.inv_id, 'P'))
                return tuple(tags) if tags else ('P',)

            buckets = {}
            for i, lin in enumerate(active):
                key = (_classify(lin), lin.population)
                buckets.setdefault(key, []).append(i)
            for (cls_tuple, pop), idx_list in buckets.items():
                k = len(idx_list)
                if k < 2:
                    continue
                ne_pop = max(self.demography.size_at(pop, t), 1e-9)
                p_class = 1.0
                for tag in cls_tuple:
                    p_class *= _p_class_for(tag, pop)
                if p_class <= 0:
                    continue
                rate = k * (k - 1) / 2.0 / (2.0 * ne_pop * p_class)
                rates.append((f'coal_{pop}', rate, idx_list))
        else:
            # LOW RHO: exact per-pair overlap-by-class.
            for i in range(len(active)):
                lin_i = active[i]
                for j in range(i + 1, len(active)):
                    lin_j = active[j]
                    if lin_i.population != lin_j.population:
                        continue
                    ovl = _overlap_by_class(lin_i, lin_j)
                    ne_pop = max(
                        self.demography.size_at(lin_i.population, t),
                        1e-9)
                    for cls_key, ov_len in ovl.items():
                        if ov_len <= 0:
                            continue
                        p_class = _p_class_for(cls_key, lin_i.population)
                        if p_class <= 0:
                            continue
                        rates.append((
                            f'coal_{cls_key}_{lin_i.population}',
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

    def _recomb_rates(self, active):
        """Per-lineage recombination rates.

        Returns list of ('recomb', rate, lineage_idx) where
        rate = total_length(lineage) * self.r.
        """
        if self.r <= 0:
            return []
        rates = []
        for idx, lin in enumerate(active):
            mat = total_length(lin.head)
            if mat > 0:
                rates.append(('recomb', mat * self.r, idx))
        return rates

    def _offset_to_position(self, lineage, offset):
        """Convert an offset within a lineage's ancestral material
        to a genomic position."""
        remaining = offset
        seg = lineage.head
        while seg is not None:
            seg_len = seg.right - seg.left
            if remaining < seg_len:
                return seg.left + remaining
            remaining -= seg_len
            seg = seg.next
        return self.L

    def _flux_lineage_rate(self, lineage, inv):
        """Per-lineage flux rate with class resolved per segment.

        Sums over each in-inv segment of the lineage:
            γ · p_other(seg.class) · phi_integral(seg-bounds, w) · inv_len
        where p_other = p_inv(pop) for class S segments, 1 - p_inv for class I.
        Panmictic segments (no S/I tag for this inversion) contribute 0.

        Replaces the prior "one karyotype per lineage" model: mixed-class
        lineages — which b2-flux's partial-tract events create regularly —
        now contribute the correct non-zero rate from BOTH their S and I
        segments, instead of being zero-blocked by ``_lineage_class_for_inv``
        returning None.

        Mirrors Rust's ``flux_lineage_rate_arena``.
        """
        if inv.bp_left is None:
            return 0.0
        inv_len = inv.bp_right - inv.bp_left
        if inv_len <= 0:
            return 0.0
        # b2-flux: w_phi is mean_tract_length / inv_length.
        w_phi = inv.mean_tract_length / inv_len
        pop = lineage.population
        p_inv_v = inv.p_inv_for(pop)
        cls_S = inv.class_S()
        cls_I = inv.class_I()
        rate = 0.0
        seg = lineage.head
        while seg is not None:
            l = max(seg.left, inv.bp_left)
            r = min(seg.right, inv.bp_right)
            if r > l:
                bc = seg.branch_class
                seg_class = None
                if isinstance(bc, frozenset):
                    if cls_S in bc:
                        seg_class = 'S'
                    elif cls_I in bc:
                        seg_class = 'I'
                else:
                    if bc == cls_S:
                        seg_class = 'S'
                    elif bc == cls_I:
                        seg_class = 'I'
                if seg_class is not None:
                    p_other = p_inv_v if seg_class == 'S' else 1.0 - p_inv_v
                    if p_other > 0.0:
                        a = (l - inv.bp_left) / inv_len
                        b = (r - inv.bp_left) / inv_len
                        rate += (inv.gene_conversion_rate * p_other
                                 * _phi_integral(a, b, w_phi) * inv_len)
            seg = seg.next
        return rate

    def _lineage_class_for_inv(self, lineage, inv):
        """Return 'S', 'I', or None for the lineage's karyotype at
        inversion ``inv``. Inspects each in-inv segment's class tag
        (a string like 'S0' or a frozenset for nested inversions).
        Returns None if the lineage has no in-inv material, has been
        flipped to panmictic ('P'), or carries inconsistent classes
        across its segments.

        Used by class-conditional migration (cmig). NOT used for flux
        rate computation since the per-segment refactor — the flux rate
        path now uses ``_flux_lineage_rate`` to handle mixed-class
        lineages correctly.
        """
        cls_S = inv.class_S()
        cls_I = inv.class_I()
        seen = set()
        seg = lineage.head
        while seg is not None:
            l = max(seg.left, inv.bp_left)
            r = min(seg.right, inv.bp_right)
            if r > l:
                bc = seg.branch_class
                if isinstance(bc, frozenset):
                    if cls_S in bc:
                        seen.add('S')
                    if cls_I in bc:
                        seen.add('I')
                else:
                    if bc == cls_S:
                        seen.add('S')
                    elif bc == cls_I:
                        seen.add('I')
            seg = seg.next
        if len(seen) == 1:
            return next(iter(seen))
        return None

    def _flux_rates(self, active):
        """List of (kind, rate, payload) for gene-flux events.

        One entry per (lineage, inversion) combination with non-zero
        flux rate. Rate is computed per-segment via ``_flux_lineage_rate``
        so mixed-class lineages contribute correctly (unlike the prior
        per-lineage class lookup which zero-blocked them).
        Payload is ``(lineage_idx, inv_id)``.
        """
        rates = []
        for inv in self.inversions:
            if inv.gene_conversion_rate <= 0:
                continue
            if inv.bp_left is None or inv.bp_right <= inv.bp_left:
                continue
            for idx, lin in enumerate(active):
                rate = self._flux_lineage_rate(lin, inv)
                if rate > 0:
                    rates.append(('flux', rate, (idx, inv.inv_id)))
        return rates

    def _apply_sweep(self, active, sweep, t, tables):
        """Force-coalesce qualifying lineages near sweep.x_sel.

        A "qualifying" lineage is one with ancestral material at
        ``sweep.x_sel`` whose class at that position matches
        ``sweep.target_class`` (or any class if ``target_class``
        is 'any'), and whose population matches ``sweep.population``
        (or any pop if ``None``).

        Two modes:

        **Hitchhiking mode** (``sweep.selection_coefficient > 0``):
        For each qualifying lineage, each segment is included with
        probability ``exp(-r * |midpoint - x_sel| * t_dur)`` where
        ``t_dur = ln(2*Ne*s)/s``.  This produces the classic smooth
        hitchhiking valley, deep at x_sel and decaying with distance.

        **Window mode** (``sweep.selection_coefficient == 0``):
        The merge happens at positions in
        ``[x_sel - sweep_window, x_sel + sweep_window]``. Material
        outside this window remains on the original lineages.
        """
        # ---- identify qualifying lineages ----
        qualifying = []
        for lin in active:
            if (sweep.population is not None
                    and lin.population != sweep.population):
                continue
            cls_at_x = lin.class_at(sweep.x_sel)
            if cls_at_x is None:
                continue  # no material at x_sel
            if sweep.target_class != 'any':
                if isinstance(cls_at_x, frozenset):
                    if isinstance(sweep.target_class, frozenset):
                        if not sweep.target_class.issubset(cls_at_x):
                            continue
                    else:
                        if sweep.target_class not in cls_at_x:
                            continue
                else:
                    if cls_at_x != sweep.target_class:
                        # Allow 'S' to match 'S0', 'S1', etc. (and 'I'
                        # to match 'I0', 'I1') so users don't have to
                        # know the inv_id suffix for single-inversion
                        # cases.
                        tc = sweep.target_class
                        if not (len(tc) == 1 and isinstance(cls_at_x, str)
                                and cls_at_x.startswith(tc)):
                            continue
            qualifying.append(lin)

        if len(qualifying) < 2:
            return None  # nothing to coalesce

        from .events import apply_coalescence

        # ---- hitchhiking mode ----
        if sweep.selection_coefficient > 0:
            r = getattr(self, 'recombination_rate', 0.0) or 0.0
            Ne = self._get_Ne_for_sweep()
            swept_lineages = []
            for lin in qualifying:
                swept_segs = []
                unswept_segs = []
                seg = lin.head
                while seg is not None:
                    mid = (seg.left + seg.right) / 2.0
                    p = sweep.hitchhiking_probability(mid, r, Ne)
                    if self.rng.random() < p:
                        swept_segs.append(seg)
                    else:
                        unswept_segs.append(seg)
                    seg = seg.next
                if not swept_segs:
                    continue
                # Build new lineage from swept segments
                active.remove(lin)
                from .segment import Segment as Seg
                s_head = s_tail = None
                for s in swept_segs:
                    ns = Seg(s.left, s.right, s.node_id,
                             branch_class=s.branch_class, prev=s_tail)
                    if s_head is None:
                        s_head = ns
                    if s_tail is not None:
                        s_tail.next = ns
                    s_tail = ns
                from .lineage import Lineage
                swept_lin = Lineage(s_head, s_tail,
                                    population=lin.population)
                active.append(swept_lin)
                swept_lineages.append(swept_lin)
                # Build lineage from unswept segments (if any)
                if unswept_segs:
                    u_head = u_tail = None
                    for s in unswept_segs:
                        ns = Seg(s.left, s.right, s.node_id,
                                 branch_class=s.branch_class, prev=u_tail)
                        if u_head is None:
                            u_head = ns
                        if u_tail is not None:
                            u_tail.next = ns
                        u_tail = ns
                    unsw_lin = Lineage(u_head, u_tail,
                                       population=lin.population)
                    active.append(unsw_lin)
            if len(swept_lineages) < 2:
                return None

            # Soft sweep: partition into K founder groups.
            k = sweep.num_founders
            if k <= 1:
                # Hard sweep: coalesce all to one ancestor.
                groups = [swept_lineages]
            else:
                # Randomly assign each swept lineage to one of K groups.
                groups = [[] for _ in range(k)]
                for lin in swept_lineages:
                    g = int(self.rng.random() * k)
                    g = min(g, k - 1)  # clamp edge case
                    groups[g].append(lin)
                groups = [g for g in groups if len(g) >= 2]

            merged = None
            for gi, group in enumerate(groups):
                if len(group) < 2:
                    continue
                m = group[0]
                for k_idx, other in enumerate(group[1:], start=1):
                    t_merge = self._next_sweep_merge_time(t)
                    result = apply_coalescence(active, m, other,
                                               t_merge, tables,
                                               skip_if_no_overlap=True)
                    if result is not None:
                        m = active[-1]
                merged = m
            return merged

        # ---- window mode (original) ----
        x_lo = sweep.x_sel - sweep.sweep_window
        x_hi = sweep.x_sel + sweep.sweep_window
        if x_hi <= x_lo:
            eps = max(1e-9, self.L * 1e-12)
            x_lo = sweep.x_sel
            x_hi = sweep.x_sel + eps

        windowed_lineages = []
        for lin in qualifying:
            a, rest = lin.split_at(x_lo)
            if rest is None:
                continue
            window, b = rest.split_at(x_hi)
            active.remove(lin)
            if a is not None:
                active.append(a)
            if b is not None:
                active.append(b)
            if window is not None:
                active.append(window)
                windowed_lineages.append(window)

        if len(windowed_lineages) < 2:
            return None
        merged = windowed_lineages[0]
        for other in windowed_lineages[1:]:
            t_merge = self._next_sweep_merge_time(t)
            apply_coalescence(active, merged, other, t_merge, tables)
            merged = active[-1]
        return merged

    def _get_Ne_for_sweep(self):
        """Get effective population size for sweep duration calculation."""
        if hasattr(self, 'demography') and self.demography is not None:
            return self.demography.pop_sizes[0]
        return getattr(self, 'population_size', 10_000) or 10_000

    def _next_sweep_merge_time(self, t):
        # Monotone counter shared across all sweep merges at the same
        # base t. Prevents TSK_ERR_BAD_NODE_TIME_ORDERING when multiple
        # sweeps fire simultaneously (same t_event) and a lineage
        # produced by an earlier merge is then touched by a later one.
        eps = max(1e-9, t * 1e-12)
        if getattr(self, '_sweep_base_t', None) != t:
            self._sweep_base_t = t
            self._sweep_merge_k = 0
        self._sweep_merge_k += 1
        return t + self._sweep_merge_k * eps

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

    def _sample_flux_position(self, lineage, inv):
        """Sample a gene-flux event position weighted by phi(x) over
        ``lineage``'s in-inv ancestral material under inversion ``inv``.

        Returns the genomic position where the conversion centres.
        """
        inv_len = inv.bp_right - inv.bp_left
        # b2-flux: w is mean_tract_length / inv_length.
        w = inv.mean_tract_length / inv_len
        intervals = []
        cum = 0.0
        seg = lineage.head
        while seg is not None:
            l = max(seg.left, inv.bp_left)
            r = min(seg.right, inv.bp_right)
            if r > l:
                a = (l - inv.bp_left) / inv_len
                b = (r - inv.bp_left) / inv_len
                weight = _phi_integral(a, b, w) * inv_len
                intervals.append((l, r, a, b, weight))
                cum += weight
            seg = seg.next
        if cum <= 0.0:
            return None
        u = self.rng.random() * cum
        running = 0.0
        chosen = intervals[-1]
        for entry in intervals:
            running += entry[4]
            if u < running:
                chosen = entry
                break
        l, r, a, b, weight = chosen
        phi_max = w / (1.0 - w) if w < 1.0 else 1.0
        for _ in range(1000):
            xx = self.rng.uniform(a, b)
            if self.rng.random() * phi_max < _phi(xx, w):
                return inv.bp_left + xx * inv_len
        return self.rng.uniform(l, r)

    def _draw_tract(self, x_event, inv):
        """Draw a gene-conversion tract centred at ``x_event`` for
        inversion ``inv``, using the b2 flux model.

        Tract length L is drawn per-event from the distribution
        configured via ``inv.tract_distribution``:
            * 'fixed':     L = inv.mean_tract_length
            * 'geometric': L ~ Exponential(1 / inv.mean_tract_length)
              (continuous-coordinate analog of geometric).
        """
        inv_len = inv.bp_right - inv.bp_left
        mean_L = inv.mean_tract_length

        # Defensive: rate-zero short-circuit upstream should prevent
        # reaching here with mean_L == 0, but guard so we never
        # divide by zero in the Exponential sampler.
        if mean_L <= 0.0:
            return float(x_event), float(x_event)

        if inv.tract_distribution == 'fixed':
            L = mean_L
        else:  # 'geometric'
            L = self.rng.exponential(mean_L)
        L = min(L, inv_len * 0.99)

        x_rel = x_event - inv.bp_left
        b1_lo = max(0.0, x_rel - L)
        b1_hi = min(inv_len - L, x_rel)
        if b1_hi <= b1_lo:
            b1 = max(0.0, min(inv_len - L, x_rel - L / 2.0))
        else:
            b1 = self.rng.uniform(b1_lo, b1_hi)
        tract_left = inv.bp_left + b1
        tract_right = min(tract_left + L, inv.bp_right)
        return tract_left, tract_right

    # -- main loop ---------------------------------------------------------

    def simulate(self, use_rust=None):
        """Run one replicate. Returns a tskit ``TreeSequence``.

        Parameters
        ----------
        use_rust : bool or None
            If True, use the Rust backend (requires the compiled
            extension). If False, use the pure-Python backend. If None
            (default), auto-detect: use Rust if available.
        """
        if use_rust is None:
            try:
                from ._rust_bridge import RUST_AVAILABLE
                use_rust = RUST_AVAILABLE
            except ImportError:
                use_rust = False
        if use_rust:
            from ._rust_bridge import rust_simulate
            ts, event_log, sweep_a_count = rust_simulate(self)
            self.event_log = event_log
            self.sweep_a_count = sweep_a_count
            return ts
        if self.sweeps:
            raise NotImplementedError(
                "The Python fallback simulator does not support sweeps. "
                "Use the Rust backend (default when available). See docs/"
                "superpowers/specs/2026-04-28-sweep-rewrite-design.md."
            )
        reset_uids()
        self._sweep_base_t = None
        self._sweep_merge_k = 0
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
            recomb = self._recomb_rates(active)
            all_events = coal + flux + mig + recomb
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
                    inv_changes = self.demography.apply_event_at(t, active)
                    for (inv_id, pop, p_inv_val) in inv_changes:
                        for inv in self.inversions:
                            if inv.inv_id == inv_id:
                                inv.set_p_inv_for(pop, p_inv_val)
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
                inv_changes = self.demography.apply_event_at(t, active)
                for (inv_id, pop, p_inv_val) in inv_changes:
                    for inv in self.inversions:
                        if inv.inv_id == inv_id:
                            inv.set_p_inv_for(pop, p_inv_val)
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
                payload = chosen_payload
                if isinstance(payload, list):
                    # Hudson bucket: pick two random lineages and merge.
                    # Following msprime (Kelleher 2016), the rate uses
                    # k*(k-1)/2 as an upper bound; non-overlapping
                    # pairs produce no edges and no lineage merge (a
                    # cheap no-op inside apply_coalescence).
                    pool = payload
                    ii, jj = self.rng.choice(len(pool), size=2,
                                              replace=False)
                    i, j = pool[ii], pool[jj]
                    apply_coalescence(active, active[i], active[j],
                                       t, tables,
                                       skip_if_no_overlap=True)
                else:
                    # Per-pair structured event: payload is
                    # (i, j, allowed_class). Partial coalescence —
                    # only merge at positions where both segments
                    # have this class.
                    i, j, allowed = payload
                    _coalesce_partial(active, active[i], active[j],
                                       t, tables, allowed)
            elif chosen_kind == 'flux':
                idx, inv_id = chosen_payload
                lineage = active[idx]
                # Find the InversionSpec for this event.
                inv = next((iv for iv in self.inversions
                            if iv.inv_id == inv_id), None)
                if inv is None:
                    continue
                x_event = self._sample_flux_position(lineage, inv)
                if x_event is None:
                    continue
                tract_left, tract_right = self._draw_tract(x_event, inv)
                if tract_right <= tract_left:
                    continue
                apply_gene_flux(active, lineage, tract_left,
                                 tract_right, inv=inv)
            elif chosen_kind == 'recomb':
                idx = chosen_payload
                lineage = active[idx]
                # Pick a breakpoint within this lineage's material.
                mat_len = total_length(lineage.head)
                x_offset = self.rng.random() * mat_len
                x = self._offset_to_position(lineage, x_offset)
                apply_recombination(active, lineage, x)
            elif chosen_kind == 'mig':
                idx, dst = chosen_payload
                apply_migration(active[idx], dst)
            else:
                raise RuntimeError(f"Unknown event kind: {chosen_kind}")

            # GC after recombination: remove sole-carrier lineages
            # to bound n for the O(n^2) structured rate computation.
            if chosen_kind == 'recomb':
                _gc_sole_lineages(active)
        else:
            raise RuntimeError(
                f"max_iters ({max_iters}) exceeded — likely a runaway "
                f"event loop.")

        return tables.finalize()
