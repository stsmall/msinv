"""Hull-algorithm simulator main loop.

Phase 1 (✓): panmictic, no inversion, no recombination. Validates
ancestral-material bookkeeping.

Phase 2 (✓): karyotype class barrier. Adds S/I lineage classes. Cross-
class coalescence is forbidden before t_inv; at t_inv all lineages
flip to a single 'S' class and the simulation proceeds panmictically.
Single-site marginals match the structured coalescent of
``msinv.simulator.build_structured_tree``.

Phase 3 (✓): gene flux events with class flip. Each lineage carries
a per-generation flux rate ``g_per_bp * p_other * sum(in-inv segment
length × phi(x))``. When an event fires, a small tract is split out
of the chosen lineage and its class is flipped to the OTHER karyotype.
This is the gene-conversion model; LD inside the inversion now decays
gradually with γ and follows the empirical phi(x) gradient (more flux
near centre, less near breakpoints).

Subsequent phases (per ``docs/hull_algorithm_design.md``) layer on:
  Phase 4: population structure + demestats rate engine.
  Phase 5: multiple inversions.
  Phase 6: sweep model integration.
  Phase 7: Cython/C inner loop.
"""

import math

import numpy as np

from .lineage import Lineage, reset_uids
from .segment import Segment
from .tables import TableBuilder
from .events import apply_coalescence, apply_recombination, apply_gene_flux


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
                 population_size: float = 1.0,
                 sequence_length: float = 1.0,
                 recombination_rate: float = 0.0,
                 p_inv: float = None,
                 t_inv: float = None,
                 bp_left: float = None,
                 bp_right: float = None,
                 gene_conversion_rate: float = 0.0,
                 flux_window: float = 0.05,
                 seed: int = None):
        # Resolve sample counts.
        if samples is not None:
            if n_std is not None or n_inv is not None:
                raise ValueError(
                    "Pass either `samples` (panmictic) or "
                    "`n_std`/`n_inv` (structured), not both.")
            self.n_std = samples
            self.n_inv = 0
        else:
            self.n_std = n_std if n_std is not None else 0
            self.n_inv = n_inv if n_inv is not None else 0
            if self.n_std + self.n_inv == 0:
                raise ValueError(
                    "Must pass `samples` or non-zero `n_std`/`n_inv`.")
        self.samples = self.n_std + self.n_inv

        # Inversion parameters.
        if self.n_inv > 0:
            if p_inv is None or not (0.0 < p_inv < 1.0):
                raise ValueError(
                    "p_inv must be in (0, 1) when n_inv > 0.")
            if t_inv is None or t_inv <= 0.0:
                raise ValueError(
                    "t_inv > 0 must be given when n_inv > 0.")
            if bp_left is None or bp_right is None:
                # Default: whole sequence is inside the inversion.
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

        self.Ne = population_size
        self.L = sequence_length
        self.r = recombination_rate
        # γ in per-bp per-generation units (analogous to recomb rate).
        self.g_per_bp = float(gene_conversion_rate)
        if not (0.0 < flux_window < 1.0):
            raise ValueError(
                f"flux_window must be in (0, 1), got {flux_window}.")
        self.flux_window = flux_window
        self.rng = np.random.default_rng(seed)

    # -- internal helpers --------------------------------------------------

    def _initial_lineages(self, tables: TableBuilder):
        """One sample lineage per sample with assigned class."""
        active = []
        for sid in range(self.n_std):
            nid = tables.add_sample(time=0.0)
            seg = Segment(0.0, self.L, nid)
            active.append(Lineage(seg, seg, branch_class='S', population=0))
        for sid in range(self.n_inv):
            nid = tables.add_sample(time=0.0)
            seg = Segment(0.0, self.L, nid)
            active.append(Lineage(seg, seg, branch_class='I', population=0))
        return active

    # -- rate helpers ------------------------------------------------------

    def _coal_rates(self, active):
        """List of (kind, rate, pool_indices) for coalescence events."""
        s_idx = [i for i, lin in enumerate(active) if lin.branch_class == 'S']
        i_idx = [i for i, lin in enumerate(active) if lin.branch_class == 'I']
        rates = []
        if self.p_inv is None:
            k = len(active)
            if k >= 2:
                rates.append((
                    'coal_panmictic',
                    k * (k - 1) / 2.0 / (2.0 * self.Ne),
                    list(range(k))))
            return rates
        p_std = 1.0 - self.p_inv
        ks = len(s_idx); ki = len(i_idx)
        if ks >= 2:
            rates.append((
                'coal_S',
                ks * (ks - 1) / 2.0 / (2.0 * self.Ne * p_std),
                s_idx))
        if ki >= 2:
            rates.append((
                'coal_I',
                ki * (ki - 1) / 2.0 / (2.0 * self.Ne * self.p_inv),
                i_idx))
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
        tables = TableBuilder(sequence_length=self.L)
        active = self._initial_lineages(tables)

        t = 0.0
        t_inv = self.t_inv

        max_iters = 10_000_000
        for _ in range(max_iters):
            if len(active) <= 1:
                # Check we've actually built a complete ARG (single
                # ancestor at every position).
                if len(active) == 0 or active[0].total_length >= self.L - 1e-9:
                    break
                # Else: we have one lineage but it doesn't cover the
                # whole sequence — must continue (e.g. after gene flux
                # spawned a tract lineage that has since coalesced).
                # Shouldn't happen if accounting is correct.
                break

            coal = self._coal_rates(active)
            flux = self._flux_rates(active)
            all_events = coal + flux
            total = sum(r for _, r, _ in all_events)
            if total <= 0:
                if t_inv is not None and t < t_inv:
                    t = t_inv
                    self._flip_to_panmictic(active)
                    t_inv = None
                    continue
                raise RuntimeError(
                    "No events possible and no t_inv to advance to — "
                    f"stuck with {len(active)} active lineages.")

            dt = self.rng.exponential(1.0 / total)
            if t_inv is not None and t + dt >= t_inv:
                t = t_inv
                self._flip_to_panmictic(active)
                t_inv = None
                continue
            t += dt

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

            if chosen_kind in ('coal_S', 'coal_I', 'coal_panmictic'):
                pool = chosen_payload
                ii, jj = self.rng.choice(len(pool), size=2, replace=False)
                i, j = pool[ii], pool[jj]
                apply_coalescence(active, active[i], active[j], t, tables)
            elif chosen_kind == 'flux':
                idx = chosen_payload
                lineage = active[idx]
                x_event = self._sample_flux_position(lineage)
                if x_event is None:
                    continue  # zero-coverage edge case
                tract_left, tract_right = self._draw_tract(x_event)
                if tract_right <= tract_left:
                    continue
                apply_gene_flux(active, lineage, tract_left, tract_right)
            else:
                raise RuntimeError(f"Unknown event kind: {chosen_kind}")
        else:
            raise RuntimeError(
                f"max_iters ({max_iters}) exceeded — likely a runaway "
                f"flux + coalescence loop.")

        return tables.finalize()
