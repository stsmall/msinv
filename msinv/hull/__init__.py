"""msinv hull algorithm — per-position ancestral material tracking.

ARG-style simulator that replaces the single-tree SMC. Each lineage
carries the genomic intervals it's ancestral to. Recombination splits
intervals; coalescence merges. SMC' is correct by construction; the
karyotype barrier and inversion-internal LD both fall out of the model.

See ``docs/hull_algorithm_design.md`` for the full design.

Status: phased implementation. The public entry point is
:class:`HullSimulator`. Phase 4 (multi-population structure +
demography) is the current validated frontier — see ``__phase__`` and
the design doc for which features are implemented.
"""

__phase__ = 6  # last fully-validated phase (5c.1 + 5c.2 also done)

from .segment import Segment
from .lineage import Lineage
from .demography import Demography
from .inversion import InversionSpec
from .sweep import Sweep
from .simulator import HullSimulator
from ._event_log import (
    filter_cmig,
    filter_flux,
    tract_lengths,
    survival_curve,
    coverage_count,
    samples_converted_at,
)

__all__ = [
    "Segment",
    "Lineage",
    "Demography",
    "InversionSpec",
    "Sweep",
    "HullSimulator",
    "__phase__",
    "filter_cmig",
    "filter_flux",
    "tract_lengths",
    "survival_curve",
    "coverage_count",
    "samples_converted_at",
]
