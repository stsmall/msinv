"""Tier 0 expectation calculator (exact karyotype mode spec, §3.5).

Exact single-site expectations under an inversion history: pairwise
coalescence times and the expected site frequency spectrum, from a Markov
chain over lineages labelled by (population, karyotype state). Reads the same
``Demography`` and ``InversionSpec`` objects as ``HullSimulator``. Requires
scipy: ``pip install msinv[expect]``.
"""

from __future__ import annotations

from ._history import Epochs, FrequencyHistory, flux_coverage

__all__ = ["Epochs", "FrequencyHistory", "flux_coverage"]
