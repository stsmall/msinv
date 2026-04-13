"""
msinv — Coalescent simulator with chromosomal inversions.

A sequential Markov coalescent (SMC) simulator for modeling chromosomal
inversions with structured coalescence, gene flux, and demographic history.

Usage (msprime-compatible real units):
    from msinv import MsinvSimulator, InversionSpec

    sim = MsinvSimulator(
        samples=10, population_size=10000,
        mutation_rate=1e-8, recombination_rate=1e-8,
        sequence_length=100000,
        n_std=5, n_inv=5, p_inv=0.5, c=0.01,
        t_inv=200000,  # generations
        bp_left=0.3, bp_right=0.7, seed=42)
    positions, haplotypes = sim.simulate_one()

Usage (coalescent-scaled, ms-style):
    sim = MsinvSimulator(nsam=10, theta=10, rho=50, nsites=1000,
                         n_std=5, n_inv=5, p_inv=0.5, c=0.01,
                         t_inv=10.0, seed=42)
    positions, haplotypes = sim.simulate_one()

If libmsinv.so is compiled and present, the C inner loop is used
automatically for ~12x speedup. Otherwise, pure Python is used.

References:
    Peischl S et al. (2013) Heredity 111:200–209
    Guerrero RF et al. (2012) Phil Trans R Soc B 367:430–438
"""

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
