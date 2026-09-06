"""msinv — Coalescent simulator with chromosomal inversions.

ARG-based per-position ancestral material tracking (msprime hull
algorithm extended with karyotype-class barriers).

Usage:
    from msinv import HullSimulator, InversionSpec, Sweep, Demography

    sim = HullSimulator(
        n_std=5, n_inv=5,
        population_size=10_000,
        sequence_length=100_000,
        p_inv=0.5, t_inv=200_000,
        bp_left=30_000, bp_right=70_000,
        gene_conversion_rate=1e-9,
        seed=42,
    )
    ts = sim.simulate()  # returns a tskit TreeSequence

References:
    Kelleher J et al. (2016) PLOS Comp Bio 12:e1004842 (hull algorithm)
    Peischl S et al. (2013) Heredity 111:200-209 (gene flux model)
    Guerrero RF et al. (2012) Phil Trans R Soc B 367:430-438
"""

from .hull import (
    HullSimulator,
    InversionSpec,
    Sweep,
    Demography,
)

__all__ = ["HullSimulator", "InversionSpec", "Sweep", "Demography"]
__version__ = "0.3.5"
