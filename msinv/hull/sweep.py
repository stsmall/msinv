"""Sweep: a forced-coalescence event modelling a recent selective sweep.

A sweep at position ``x_sel`` reaching fixation at time
``t_event`` (in generations going backward) is modelled by
force-coalescing all lineages carrying ancestral material at ``x_sel``
that are of a specified target class into a single sweep ancestor at
``t_event``.

This is the standard "Hudson-Kaplan-style" approximation for a hard
sweep in a structured coalescent: within a tiny sweep window, a
single ancestor founded the carrier sub-population. Going forward,
the sweep took the carriers from one chromosome to fixation; going
backward, all carriers collapse to that single ancestor at
``t_event``.

For a sweep that started on the S background, transferred to I via
gene conversion, and fixed on both — model with a single sweep event
that targets both classes (or with two events, one per class). The
gene-flux machinery on the hull (Phase 3) handles the S↔I transfer
naturally.
"""

from dataclasses import dataclass


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
        (single-point coalescence at exactly x_sel).
    """

    x_sel: float
    t_event: float
    target_class: str = 'any'
    population: int = None
    sweep_window: float = 0.0
