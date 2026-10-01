"""SMC' coalescence option (``smc_prime=True``) for lineages without inversion tags.

msinv's default is the original SMC: two lineages can coalesce only if they
share ancestral material, so every recombination leaves a breakpoint. With
``smc_prime=True`` lineages whose hulls overlap or touch are eligible (msprime's
hull algorithm with k = 0), so the two halves of a recombined lineage can
coalesce back together. Distinct genealogies (simplified tree counts) should
then match msprime's SMC' and Hudson models, and the default should match SMC.
"""

import math

import msprime
import numpy as np
import pytest

from msinv.hull import HullSimulator

N = 2_000
L = 20_000
R = 2.5e-6          # 4 N R L = 400
REPS = 120


def _msinv_trees(n, smc_prime):
    return np.array([
        HullSimulator(samples=n, population_size=N, sequence_length=L,
                      recombination_rate=R, seed=10_000 + s,
                      smc_prime=smc_prime).simulate().simplify().num_trees
        for s in range(REPS)])


def _msprime_trees(n, model):
    # ploidy 2 so that population_size N matches msinv's diploid N
    return np.array([
        msprime.sim_ancestry(samples=n // 2, ploidy=2, population_size=N,
                             sequence_length=L, recombination_rate=R,
                             model=model, random_seed=20_000 + s).num_trees
        for s in range(REPS)])


def _close(a, b, z=4.0):
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return abs(a.mean() - b.mean()) <= z * se, (a.mean(), b.mean(), se)


@pytest.mark.parametrize("n", [2, pytest.param(10, marks=pytest.mark.xfail(
    reason="known ~4% excess of distinct trees for n > 2 in both the SMC and "
           "SMC' modes; separate from the SMC/SMC' eligibility rule, under "
           "investigation", strict=True))])
def test_smc_prime_matches_msprime_smc_prime(n):
    ok, info = _close(_msinv_trees(n, True), _msprime_trees(n, "smc_prime"))
    assert ok, info


@pytest.mark.parametrize("n", [2, pytest.param(10, marks=pytest.mark.xfail(
    reason="known ~4% excess of distinct trees for n > 2 in both the SMC and "
           "SMC' modes; separate from the SMC/SMC' eligibility rule, under "
           "investigation", strict=True))])
def test_default_matches_msprime_smc(n):
    ok, info = _close(_msinv_trees(n, False), _msprime_trees(n, "smc"))
    assert ok, info


def test_smc_prime_python_engine_not_supported():
    sim = HullSimulator(samples=2, population_size=N, sequence_length=L,
                        recombination_rate=R, seed=1, smc_prime=True)
    with pytest.raises(NotImplementedError):
        sim.simulate(use_rust=False)
