"""Sweep: a forced-coalescence event modelling a recent selective sweep.

A sweep at position ``x_sel`` reaching fixation at time
``t_event`` (in generations going backward) is modelled by
force-coalescing lineages carrying ancestral material near ``x_sel``
that are of a specified target class into a single sweep ancestor at
``t_event``.

Three modes are supported:

1. **Hitchhiking mode** (``selection_coefficient > 0``): each lineage's
   inclusion in the sweep is probabilistic, with probability decaying
   exponentially with recombination distance from ``x_sel``::

       P(linked) = exp(-r * |x - x_sel| * t_dur)

   where ``t_dur = ln(2*Ne*s) / s`` is the sweep duration.  This
   produces the classic Maynard Smith & Haigh hitchhiking valley — deep
   at ``x_sel``, eroding with distance.  Requires
   ``recombination_rate`` and ``Ne`` to be set on the simulator.

2. **Window mode** (``selection_coefficient == 0``): all lineages with
   material in ``[x_sel - sweep_window, x_sel + sweep_window]`` are
   deterministically coalesced (the original Hudson-Kaplan
   approximation).

3. **Soft sweep from standing variation** (``selection_coefficient > 0``
   and ``starting_frequency > 0``): hitchhiking mode, but swept
   lineages are randomly partitioned among K ≈ 1/f0 "founding copies"
   of the beneficial allele (discoal model, Kern & Schrider 2016).
   Lineages within each group coalesce; K surviving ancestors continue
   at the normal coalescent rate.  Produces the characteristic partial
   diversity reduction of a sweep from standing variation.

For a sweep that started on the S background, transferred to I via
gene conversion, and fixed on both — model with two sweep events, one
per class, at different times.  Use a ``FluxTransfer`` to explicitly
model the S→I transfer event at a specific time.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sweep:
    """A single forced-coalescence event.

    Attributes
    ----------
    x_sel : float
        Genomic position where the selected allele resides.
    t_event : float
        Time (generations going backward) of the sweep MRCA. All
        target-class lineages carrying ancestral material at
        ``x_sel`` will coalesce to a single ancestor at this time.
    target_class : str
        Segment class to act on at ``x_sel``: 'S', 'I', 'P', or
        per-inversion-tagged ('S0', 'S1', etc.). If 'any', acts on
        all classes (rare — mostly for sweeps after t_inv).
    population : int
        Restrict the sweep to lineages currently in this population
        (allows pop-specific sweeps). ``None`` = any population.
    sweep_window : float
        Half-width of the sweep window in genomic coordinates. The
        force-coalescence is applied to ancestral material in
        [x_sel - sweep_window, x_sel + sweep_window]. Default 0
        (single-point coalescence at exactly x_sel).  Ignored when
        ``selection_coefficient > 0`` (hitchhiking mode).
    selection_coefficient : float
        Selection coefficient for the swept allele.  When > 0,
        enables hitchhiking mode: inclusion probability decays with
        recombination distance from x_sel.  Default 0 (window mode).
    starting_frequency : float
        Starting frequency of the beneficial allele (standing variation).
        When > 0, enables soft-sweep mode (discoal model): swept
        lineages are partitioned among K ≈ 1/f0 founding copies instead
        of coalescing to a single ancestor.  0.0 = hard sweep (single
        origin).  Must be in [0, 1).
    """

    x_sel: float
    t_event: float
    target_class: str = 'any'
    population: Optional[int] = None
    sweep_window: float = 0.0
    selection_coefficient: float = 0.0
    starting_frequency: float = 0.0

    @property
    def num_founders(self) -> int:
        """Number of founding copies for soft sweep partitioning.

        Returns 1 for hard sweeps (starting_frequency == 0).
        """
        if self.starting_frequency <= 0.0:
            return 1
        return max(1, round(1.0 / self.starting_frequency))

    def hitchhiking_probability(self, x: float, r: float, Ne: float) -> float:
        """Probability that position *x* is linked to the sweep.

        Parameters
        ----------
        x : float
            Genomic position to test.
        r : float
            Per-bp per-generation recombination rate.
        Ne : float
            Effective population size (for sweep duration).

        Returns
        -------
        float
            Probability in [0, 1].
        """
        s = self.selection_coefficient
        if s <= 0 or r <= 0:
            return 1.0
        # Sweep duration: time for allele to go from 1/(2N) to fixation
        t_dur = math.log(max(2 * Ne * s, 2.0)) / s
        dist = abs(x - self.x_sel)
        return math.exp(-r * dist * t_dur)
