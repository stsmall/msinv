"""Generic engine runners for the msinv validation suite.

Each runner takes a unified set of parameters (demography, samples, L,
r, mu, seed) and engine-specific extras (inversions / sweeps for msinv;
sweep CLI args for discoal). Returns a mutated tskit.TreeSequence ready
for the validation/_lib/stats panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import msprime
import tskit

from msinv import HullSimulator, InversionSpec, Sweep
from msinv import Demography as MsinvDemography


def msinv_run(
    *,
    demography: MsinvDemography,
    sample_config: dict,
    L: float,
    r: float,
    mu: float,
    seed: int,
    inversions: list[InversionSpec] | None = None,
    sweeps: list[Sweep] | None = None,
    iters_max: int = 1_000_000_000,
) -> tskit.TreeSequence:
    """Run msinv at the given config and overlay neutral mutations."""
    sim = HullSimulator(
        sample_config=sample_config,
        demography=demography,
        sequence_length=float(L),
        recombination_rate=float(r),
        inversions=inversions or [],
        sweeps=sweeps or [],
        seed=int(seed),
        iters_max=iters_max,
    )
    ts_raw = sim.simulate()
    ts = msprime.sim_mutations(
        ts_raw, rate=float(mu), random_seed=int(seed) + 1, keep=True,
    )
    return ts
