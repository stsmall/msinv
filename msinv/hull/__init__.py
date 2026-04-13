"""msinv hull algorithm — per-position ancestral material tracking.

ARG-style simulator that replaces the single-tree SMC. Each lineage
carries the genomic intervals it's ancestral to. Recombination splits
intervals; coalescence merges. SMC' is correct by construction; the
karyotype barrier and inversion-internal LD both fall out of the model.

See ``docs/hull_algorithm_design.md`` for the full design.

Status: phased implementation. The public entry point is
:class:`HullSimulator`, but only Phase 1 (panmictic, no inversion) is
guaranteed working at any given checkpoint. See ``__phase__`` for the
current frontier.
"""

__phase__ = 0  # bumped as each phase is validated

from .segment import Segment
from .lineage import Lineage
from .simulator import HullSimulator

__all__ = ['Segment', 'Lineage', 'HullSimulator', '__phase__']
