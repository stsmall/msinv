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
    gamma_flux: float = 0.0
    mean_tract_length: float = 0.0
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
            gamma_flux=self.gamma_flux,
            mean_tract_length=self.mean_tract_length,
            seed=self.seed,
            dt_scalar=self.dt_scalar,
        )
