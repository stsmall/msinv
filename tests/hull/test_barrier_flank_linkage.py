"""Flank (margin) material linked to an inversion while its barrier is active.

Under explicit recombination a lineage that carries arrangement material is
one chromosome of that arrangement, flanks included. An I-carrying and an
S-carrying lineage have no common ancestor while the barrier is active, so
their coalescence rate is 0, also where they overlap in the untagged flank.
Flank material leaves its arrangement only through a recombination in the
flank, which makes an untagged lineage that can coalesce with any other.

Regression test for msinv <= 0.3.5 (also present after PR #19), where I x S
pairs partial-merged their shared margin at 1/(2N): flank material behaved as
panmictic right up to the breakpoints, so dxy collapsed ~3x at the breakpoint even with the margin
fully linked to the body (r = 1e-12), and pi within an arrangement was
inflated in the margin. With the fix:

* fully linked margin: margin dxy and pi_I equal the body values and the
  exact two-deme expectations, with and without gene flux (flux tracts lie
  inside the body, so the margins follow the arrangement backbone);
* realistic flank r: dxy decays from the body level toward the panmictic
  value with distance from the breakpoint;
* the two engines agree, including the Python high-rho bucket path, which
  now pairs untagged lineages with tagged ones.
"""

import math

import numpy as np
import pytest

from msinv.hull import HullSimulator
from msinv.hull.inversion import InversionSpec
from msinv.hull.simulator import _pair_karyotype

N = 10_000
P = 0.6
T_INV = 4 * N
Z_TOL = 3.5

# Two-deme expectations in branch units (2 x mean pairwise TMRCA).
_A = 2 * N * P
PI_I_EXP = 2 * (_A * (1 - math.exp(-T_INV / _A)) + 2 * N * math.exp(-T_INV / _A))
DXY_EXP = 2 * (T_INV + 2 * N)
DXY_PAN = 2 * 2 * N


def _fully_linked(seed, use_rust, gamma=1e-15, tract=1.0):
    """n = 5 + 5, 10 kb margins around a 50 kb body, r = 1e-12."""
    spec = InversionSpec(bp_left=10_000, bp_right=60_000, p_inv=P, t_inv=T_INV,
                         gene_conversion_rate=gamma, mean_tract_length=tract)
    ts = HullSimulator(n_std=5, n_inv=5, population_size=N,
                       sequence_length=70_000, recombination_rate=1e-12,
                       inversions=[spec], seed=seed).simulate(use_rust=use_rust)
    s_nodes, i_nodes = list(range(5)), list(range(5, 10))
    w = [0, 10_000, 60_000, 70_000]
    d = np.ravel(ts.divergence([i_nodes, s_nodes], windows=w, mode="branch"))
    pi = np.ravel(ts.diversity([i_nodes], windows=w, mode="branch"))
    # margins (both sides pooled), body
    return np.array([(d[0] + d[2]) / 2, d[1], (pi[0] + pi[2]) / 2, pi[1]])


def _z_vs(x, expected):
    se = x.std(ddof=1) / math.sqrt(len(x))
    return (x.mean() - expected) / se


@pytest.mark.parametrize("use_rust,reps", [(True, 300), (False, 80)],
                         ids=["rust", "python"])
def test_fully_linked_margin_matches_body(use_rust, reps):
    a = np.array([_fully_linked(10 + k, use_rust) for k in range(reps)])
    dxy_m, dxy_b, pi_m, pi_b = a.mean(axis=0)
    # Before the fix: dxy margin ~38k vs body ~121k.
    assert abs(dxy_m - dxy_b) < 0.05 * dxy_b, (dxy_m, dxy_b)
    assert abs(pi_m - pi_b) < 0.05 * pi_b, (pi_m, pi_b)
    z = [_z_vs(a[:, 0], DXY_EXP), _z_vs(a[:, 2], PI_I_EXP)]
    assert all(abs(v) < Z_TOL for v in z), (dxy_m, DXY_EXP, pi_m, PI_I_EXP, z)


@pytest.mark.parametrize("use_rust,reps", [(True, 200), (False, 40)],
                         ids=["rust", "python"])
def test_fully_linked_margin_with_gene_flux(use_rust, reps):
    """Gene flux moves tracts inside the body to the other arrangement on a
    separate lineage; the margins stay on the arrangement backbone, which
    is a two-deme coalescent with no exchange. Before the fix the
    margin dxy here was ~35k."""
    a = np.array([_fully_linked(10 + k, use_rust, gamma=1e-6, tract=500.0)
                  for k in range(reps)])
    z = [_z_vs(a[:, 0], DXY_EXP), _z_vs(a[:, 2], PI_I_EXP)]
    assert all(abs(v) < Z_TOL for v in z), (a.mean(axis=0), z)


# Realistic flank: 100 kb flanks around a 50 kb body, p = 0.5.
DECAY_D = [0, 1_000, 5_000, 20_000, 50_000, 100_000]


def _decay(seed, use_rust, r=1e-8):
    m, body = 100_000, 50_000
    spec = InversionSpec(bp_left=m, bp_right=m + body, p_inv=0.5, t_inv=T_INV,
                         gene_conversion_rate=1e-15, mean_tract_length=1.0)
    ts = HullSimulator(n_std=5, n_inv=5, population_size=N,
                       sequence_length=2 * m + body, recombination_rate=r,
                       inversions=[spec], seed=seed).simulate(use_rust=use_rust)
    left = sorted(m - d for d in DECAY_D)
    right = [m + body + d for d in DECAY_D]
    w = left + right
    d = np.ravel(ts.divergence([list(range(5, 10)), list(range(5))],
                               windows=w, mode="branch"))
    nb = len(DECAY_D) - 1
    # Distance bins outward from each breakpoint, both sides pooled.
    left_bins = d[:nb][::-1]
    right_bins = d[nb + 1:]
    return np.r_[(left_bins + right_bins) / 2, d[nb]]


def test_flank_dxy_decays_with_distance_rust():
    a = np.array([_decay(100 + k, True) for k in range(150)])
    m = a.mean(axis=0)
    se = a.std(axis=0, ddof=1) / math.sqrt(len(a))
    flank, body = m[:-1], m[-1]
    assert abs(_z_vs(a[:, -1], 2 * (T_INV + 2 * N))) < Z_TOL, body
    # Non-increasing with distance (up to noise).
    for k in range(len(flank) - 1):
        tol = Z_TOL * math.hypot(se[k], se[k + 1])
        assert flank[k + 1] <= flank[k] + tol, (k, flank, se)
    # Near the breakpoint close to the body level; far out near panmictic.
    # Before the fix every flank bin sat at ~38-40k.
    assert flank[0] > 0.75 * body, (flank, body)
    assert flank[0] > 2.0 * flank[-1], flank
    assert abs(flank[-1] - DXY_PAN) < 0.15 * DXY_PAN, flank


def test_flank_dxy_decay_python_matches_rust():
    """Python per-pair path (4 N r L = 100) against the Rust engine."""
    a = np.array([_decay(300 + k, False) for k in range(20)])
    b = np.array([_decay(400 + k, True) for k in range(200)])
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    z = (a.mean(axis=0) - b.mean(axis=0)) / se
    assert np.all(np.abs(z) < Z_TOL), (a.mean(axis=0), b.mean(axis=0), z)


def test_python_high_rho_pairs_untagged_with_tagged():
    """High-rho bucket path: an untagged (margin-only) lineage coalesces
    with I and S lineages at 1/(2N); I x S gets no event. Before the fix only
    same-bucket pairs had events, so the untagged lineage was stranded."""
    spec = InversionSpec(bp_left=10_000, bp_right=60_000, p_inv=P, t_inv=T_INV,
                         gene_conversion_rate=1e-15, mean_tract_length=1.0)
    sim = HullSimulator(n_std=1, n_inv=1, population_size=N,
                        sequence_length=70_000, recombination_rate=1e-6,
                        inversions=[spec], seed=1)
    assert 4 * N * sim.r * sim.L > sim._RHO_THRESHOLD
    from msinv.hull.tables import TableBuilder

    active = sim._initial_lineages(TableBuilder(sim.L))
    assert len(active) == 2
    # Recombination in the right margin of the I sample (node 1): the right
    # piece carries margin material only.
    i_lin = next(lin for lin in active if lin.head.node_id == 1)
    left, right = i_lin.split_at(65_000)
    active = [lin for lin in active if lin is not i_lin] + [left, right]
    rates = sim._coal_rates(active, 0.0)
    total = sum(r for _, r, _ in rates)
    assert math.isclose(total, 2 / (2 * N)), rates
    s_lin = next(lin for lin in active if lin.head.node_id == 0)
    assert _pair_karyotype(left, s_lin)[0] is False
    assert _pair_karyotype(right, s_lin) == (True, "P")
