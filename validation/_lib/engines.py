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


def msprime_run(
    *,
    demography: msprime.Demography,
    samples_by_pop: dict[str, int],
    L: float,
    r: float,
    mu: float,
    seed: int,
) -> tskit.TreeSequence:
    """Run msprime at the given config and overlay neutral mutations.

    `samples_by_pop` keys must match the v12 population names ("K", "F").
    msprime is called with ploidy=1 (haploid) to match msinv's sample
    convention (per CLAUDE.md `population_size` / sample convention).
    """
    ts_raw = msprime.sim_ancestry(
        samples=samples_by_pop,
        demography=demography,
        sequence_length=float(L),
        recombination_rate=float(r),
        random_seed=int(seed),
        ploidy=1,
    )
    ts = msprime.sim_mutations(
        ts_raw, rate=float(mu), random_seed=int(seed) + 1, keep=True,
    )
    return ts
