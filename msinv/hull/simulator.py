"""Hull-algorithm simulator main loop.

Phase 1: panmictic, no inversion, no demography. Validates the
ancestral-material bookkeeping by comparing tree-sequence output to
``msprime`` for matching parameters.

Subsequent phases (per ``docs/hull_algorithm_design.md``) layer on:
  Phase 2: class barrier (S/I, t_inv).
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
    """Phase 1 (panmictic) skeleton.

    Parameters
    ----------
    samples : int
    population_size : float
        Effective population size (scales coalescent times).
    sequence_length : float
        Length in base pairs (or any consistent unit).
    recombination_rate : float
        Per-bp per-generation recombination rate.
    seed : int
    """

    def __init__(self, samples: int, population_size: float,
                 sequence_length: float, recombination_rate: float = 0.0,
                 seed: int = None):
        self.samples = samples
        self.Ne = population_size
        self.L = sequence_length
        self.r = recombination_rate
        self.rng = np.random.default_rng(seed)

    def simulate(self):
        """Run one replicate. Returns a tskit ``TreeSequence``.

        PHASE 1 SCAFFOLD — only panmictic with no recombination is
        implemented. Recombination handler is in events.py but the
        rate selection here doesn't yet wire it up. See design doc.
        """
        reset_uids()
        tables = TableBuilder(sequence_length=self.L)

        # Initialise: each sample is a lineage with one segment over
        # the full chromosome.
        active = []
        for i in range(self.samples):
            nid = tables.add_sample(time=0.0)
            seg = Segment(0.0, self.L, nid)
            active.append(Lineage(seg, seg))

        t = 0.0
        # Per-generation rate scaling: 1/(2 Ne) per pair-coalescence opportunity.
        while len(active) > 1:
            k = len(active)
            coal_rate = k * (k - 1) / 2.0 / (2.0 * self.Ne)
            # TODO Phase 1: add recombination rate using
            # rates.recombination_rate_for_lineage on each active
            # lineage and dispatch to apply_recombination.
            dt = self.rng.exponential(1.0 / coal_rate)
            t += dt
            i, j = self.rng.choice(k, size=2, replace=False)
            apply_coalescence(active, active[i], active[j], t, tables)

        return tables.finalize()
