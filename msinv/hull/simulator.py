"""Hull-algorithm simulator main loop.

Phase 1 (✓): panmictic, no inversion, no recombination. Validates
ancestral-material bookkeeping.

Phase 2 (✓): karyotype class barrier. Adds S/I lineage classes. Cross-
class coalescence is forbidden before t_inv; at t_inv all lineages
flip to a single 'S' class and the simulation proceeds panmictically
(this is the inversion's age, before which all chromosomes shared a
single arrangement). Single-site marginals match the structured
coalescent of ``msinv.simulator.build_structured_tree``.

Subsequent phases (per ``docs/hull_algorithm_design.md``) layer on:
  Phase 3: gene flux events with class flip.
  Phase 4: population structure + demestats rate engine.
  Phase 5: multiple inversions.
  Phase 6: sweep model integration.
  Phase 7: Cython/C inner loop.
"""

import numpy as np

from .lineage import Lineage, reset_uids
from .segment import Segment
from .tables import TableBuilder
from .events import apply_coalescence, apply_recombination


class HullSimulator:
    """Hull-algorithm simulator.

    Parameters
    ----------
    samples : int, optional
        Total number of samples (panmictic mode). If given, ``n_std``
        and ``n_inv`` are ignored. Equivalent to Phase 1.
    n_std : int, optional
        Number of S-class (standard arrangement) samples.
    n_inv : int, optional
        Number of I-class (inverted arrangement) samples.
    population_size : float
        Effective population size (scales coalescent times).
    sequence_length : float
        Length in base pairs (or any consistent unit).
    recombination_rate : float
        Per-bp per-generation recombination rate. Currently unused
        (Phase 2 has no recombination; Phase 3+ adds it).
    p_inv : float, optional
        Frequency of the inverted (I) arrangement, in (0, 1). Required
        when ``n_inv > 0``. Sets the structured-coalescent rate
        scaling: I lineages coalesce at rate
        ``k(k-1)/2 / (p_inv * Ne)`` and S at
        ``k(k-1)/2 / (p_std * Ne)``.
    t_inv : float, optional
        Age of the inversion in generations (the time before which all
        chromosomes shared a single arrangement). At ``t >= t_inv`` all
        lineages flip to a single 'S' class and coalesce panmictically.
        Required when ``n_inv > 0``.
    seed : int, optional
    """

    def __init__(self, *, samples: int = None,
                 n_std: int = None, n_inv: int = None,
                 population_size: float = 1.0,
                 sequence_length: float = 1.0,
                 recombination_rate: float = 0.0,
                 p_inv: float = None,
                 t_inv: float = None,
                 seed: int = None):
        # Resolve sample counts and inversion mode.
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

        if self.n_inv > 0:
            if p_inv is None or not (0.0 < p_inv < 1.0):
                raise ValueError(
                    "p_inv must be in (0, 1) when n_inv > 0.")
            if t_inv is None or t_inv <= 0.0:
                raise ValueError(
                    "t_inv > 0 must be given when n_inv > 0.")
            self.p_inv = p_inv
            self.t_inv = t_inv
        else:
            self.p_inv = None
            self.t_inv = None

        self.Ne = population_size
        self.L = sequence_length
        self.r = recombination_rate
        self.rng = np.random.default_rng(seed)

    # -- internal helpers --------------------------------------------------

    def _initial_lineages(self, tables: TableBuilder):
        """Create one sample lineage per sample with assigned class."""
        active = []
        # S samples first (sample_id 0 .. n_std-1), then I samples.
        for sid in range(self.n_std):
            nid = tables.add_sample(time=0.0)
            seg = Segment(0.0, self.L, nid)
            active.append(Lineage(seg, seg, branch_class='S', population=0))
        for sid in range(self.n_inv):
            nid = tables.add_sample(time=0.0)
            seg = Segment(0.0, self.L, nid)
            active.append(Lineage(seg, seg, branch_class='I', population=0))
        return active

    def _structured_rates(self, active):
        """Return list of (event_kind, rate, payload) tuples.

        event_kind: 'coal_S' | 'coal_I' | 'coal_panmictic'
        payload: list of indices in ``active`` of lineages of that class.
        """
        s_idx = [i for i, lin in enumerate(active) if lin.branch_class == 'S']
        i_idx = [i for i, lin in enumerate(active) if lin.branch_class == 'I']
        rates = []
        if self.p_inv is None:
            # Panmictic mode (Phase 1) — class barrier not active.
            k = len(active)
            if k >= 2:
                rates.append((
                    'coal_panmictic',
                    k * (k - 1) / 2.0 / (2.0 * self.Ne),
                    list(range(k))))
            return rates
        p_std = 1.0 - self.p_inv
        # Structured rates per class. Effective sub-pop size Ne * p_class.
        ks = len(s_idx)
        ki = len(i_idx)
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

    def _flip_to_panmictic(self, active):
        """At t_inv: classes merge — every lineage becomes 'S'."""
        for lin in active:
            lin.branch_class = 'S'
        # From now on we're in panmictic mode; setting p_inv=None makes
        # _structured_rates take the panmictic branch.
        self.p_inv = None
        self.t_inv = None

    # -- main loop ---------------------------------------------------------

    def simulate(self):
        """Run one replicate. Returns a tskit ``TreeSequence``."""
        reset_uids()
        tables = TableBuilder(sequence_length=self.L)
        active = self._initial_lineages(tables)

        t = 0.0
        # Snapshot t_inv so the flip can null it out without losing the value.
        t_inv = self.t_inv

        while len(active) > 1:
            rates = self._structured_rates(active)
            total = sum(r for _, r, _ in rates)
            if total <= 0:
                # No allowed events (e.g. only one S and one I lineage,
                # both pre-t_inv with no flux yet). Jump to t_inv if it
                # exists; otherwise we're stuck.
                if t_inv is not None and t < t_inv:
                    t = t_inv
                    self._flip_to_panmictic(active)
                    t_inv = None
                    continue
                raise RuntimeError(
                    "No coalescent events possible and no t_inv to "
                    "advance to — simulation stuck.")
            dt = self.rng.exponential(1.0 / total)
            # Class barrier: if event would fire past t_inv, snap to
            # t_inv, flip classes, and resample on the panmictic state.
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
            chosen_pool = None
            for kind, rate, pool in rates:
                cum += rate
                if u < cum:
                    chosen_kind = kind
                    chosen_pool = pool
                    break
            # Pick two lineages from the pool to coalesce.
            ii, jj = self.rng.choice(len(chosen_pool), size=2, replace=False)
            i, j = chosen_pool[ii], chosen_pool[jj]
            apply_coalescence(active, active[i], active[j], t, tables)

        return tables.finalize()
