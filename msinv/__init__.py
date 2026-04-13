"""
msinv — Coalescent simulator with chromosomal inversions.

Two simulator engines are available:

* ``HullSimulator`` (recommended) — ARG-based per-position ancestral
  material tracking. Architecturally correct: cross-karyotype barriers
  preserved, multi-pop demographies match msprime, multiple
  inversions (including nested/overlapping) supported.

* ``MsinvSimulator`` (legacy) — single-tree SMC implementation.
  Faster for inversion-only single-pop scenarios but has a known
  two-pop bug (Scenario 5 of ``examples/bakeoff.py`` shows
  cross-pop dxy is ~half of the msprime ground truth). Kept for
  backwards compatibility.

Usage (HullSimulator, recommended):
    from msinv import HullSimulator, InversionSpec, Sweep

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=10000,
        sequence_length=100_000,
        p_inv=0.5, t_inv=200_000,
        bp_left=30_000, bp_right=70_000,
        gene_conversion_rate=1e-9,
        seed=42)
    ts = sim.simulate()  # returns a tskit TreeSequence

Usage (MsinvSimulator, legacy):
    from msinv import MsinvSimulator
    sim = MsinvSimulator(samples=10, population_size=10000, ...)
    positions, haplotypes = sim.simulate_one()

References:
    Kelleher J et al. (2016) PLOS Comp Bio 12:e1004842 (msprime hull algorithm)
    Peischl S et al. (2013) Heredity 111:200–209 (gene flux model)
    Guerrero RF et al. (2012) Phil Trans R Soc B 367:430–438
"""

# Hull simulator (recommended)
from .hull import (
    HullSimulator,
    InversionSpec as HullInversionSpec,
    Sweep,
    Demography as HullDemography,
)

# Legacy SMC simulator
from .simulator import (
    # Simulator
    MsinvSimulator,
    # Inversion specification
    InversionSpec,
    GeneFluxModel,
    # Frequency trajectories
    ConstantFrequency,
    DeterministicTrajectory,
    StochasticTrajectory,
    CoupledTrajectory,
    # Demography
    Demography,
    # Tree utilities
    Node,
    EdgeRecorder,
    # n=2 functions
    build_initial_tree,
    smc_step,
    simulate_one_n2,
    phi,
    # Helper functions
    build_structured_tree,
    get_all_nodes,
    get_branches,
    find_root,
)

__version__ = "0.1.0"
