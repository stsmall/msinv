"""Sweep: a discoal-style stoch+det selective sweep over (kary × allele × pop).

See ``docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md`` for the
target model. This module provides a thin dataclass + factory that wraps
the Rust ``PySweep`` constructor.
"""

from dataclasses import dataclass
from typing import Literal

import msinv._msinv_core as _core


SweepModeStr = Literal[
    "Stochastic",
    "StochasticConditioned",
    "Deterministic",
    "Neutral",
]


@dataclass
class Sweep:
    """One selective sweep (Rust engine only).

    With an inversion ``target_inv`` that exists at ``t_origin``, the sweep
    is linked to it. The trajectory tracks the beneficial allele on each
    arrangement, starting from the inversion frequency at ``t_origin``, and
    the association between allele and arrangement decays at a rate ``c``
    per generation:

    - ``x_sel`` inside the inversion: the allele arises on ``origin_kary``
      at overall frequency ``f0`` (at most that arrangement's frequency)
      and moves to the other arrangement only by the inversion's gene flux;
      ``c = gene_conversion_rate * mean_tract_length * phi(x_sel)``.
    - ``x_sel`` outside it: ``c = r * d``, with ``d`` the distance to the
      nearer breakpoint. A hard sweep's founder arrangement is drawn from
      the arrangement frequencies at ``t_origin``. Standing variation
      starts independent of arrangement, at overall frequency ``f0``.
      ``origin_kary`` is ignored.

    Selection on the allele changes the inversion frequency in proportion
    to their linkage. The inversion frequency follows the trajectory from
    ``t_origin`` to ``tau`` and stays at its final value after ``tau``.
    Backward in time, lineages carrying an arrangement coalesce at that
    arrangement's allele frequency. A lineage's association with the
    allele moves between arrangements at the exact backward rate implied
    by the trajectory. A sweep far from the inversion therefore behaves as
    in discoal.

    Without such an inversion (or if it arises after ``t_origin``) the
    sweep is an ordinary discoal-style sweep, and inversion frequencies
    are unchanged.
    """

    x_sel: float
    tau: float
    origin_pop: int
    origin_kary: Literal["S", "I"]
    target_inv: int
    mode: SweepModeStr = "Stochastic"
    s: float = 0.0
    t_origin: float = 0.0
    f0: float = 0.0
    partial_sweep_final_freq: float = 1.0
    recurrent_mutation_rate: float = 0.0
    seed: int = 0
    dt_scalar: float = 400.0

    def to_rust(self) -> "_core.PySweep":
        kary_int = 0 if self.origin_kary == "S" else 1
        return _core.PySweep(
            x_sel=self.x_sel,
            tau=self.tau,
            origin_pop=self.origin_pop,
            origin_kary=kary_int,
            target_inv=self.target_inv,
            mode=self.mode,
            s=self.s,
            t_origin=self.t_origin,
            f0=self.f0,
            partial_sweep_final_freq=self.partial_sweep_final_freq,
            recurrent_mutation_rate=self.recurrent_mutation_rate,
            seed=self.seed,
            dt_scalar=self.dt_scalar,
        )
