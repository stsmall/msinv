"""Crossover suppression in heterokaryotypes (``crossover_suppression=True``).

While an inversion's barrier is active, a crossover at a position inside the
inversion needs a homokaryotypic partner. A lineage carrying arrangement k
there recombines at ``r * p_k(t)``; collinear flanks, untagged material and
times older than ``t_inv`` keep ``r``. SLiM, the forward reference, drops every
crossover inside the inversion in a heterokaryotype. Up to msinv 0.3.5 both
engines used the full ``r`` everywhere, which over-estimated within-arrangement
rho by 1/p_k (about 2.7x in the minority and 1.6x in the majority arrangement
at p = 0.626) and so under-estimated within-arrangement LD.

Tests:
- exactness: an inversion over the whole sequence, constant p, barrier older
  than every TMRCA, one arrangement sampled. This is a single population of
  size p N with recombination rate r p, which msprime (SMC, n = 2) and msinv's
  own panmictic mode (any n) simulate directly.
- compensator checks for n = 2: in the SMC every recombination on the
  two-branch tree is a tree change, so E[#breakpoints in a region] is
  E[integral over the region of r * w * (weighted branch length)], with the
  branch length read from the same marginal trees. This tests the realised
  crossover rate position by position: the flanks (w = 1), the body (w = p)
  and the epoch older than t_inv (w = 1 again).
- Python and Rust engines agree.
- ``crossover_suppression=False`` restores the old rate.
"""

import math

import msprime
import numpy as np
import pytest

from msinv.hull import HullSimulator
from msinv.hull.inversion import InversionSpec

P = 0.6
Z_TOL = 4.0


def _spec(left, right, t_inv, p=P):
    return InversionSpec(bp_left=left, bp_right=right, p_inv=p, t_inv=t_inv,
                         gene_conversion_rate=1e-15, mean_tract_length=1.0)


def _sim(kary, n, N, L, r, inv, seed, use_rust=True, **kw):
    n_std, n_inv = (0, n) if kary == "I" else (n, 0)
    return HullSimulator(n_std=n_std, n_inv=n_inv, population_size=N,
                         sequence_length=L, recombination_rate=r,
                         inversions=[inv], seed=seed, **kw
                         ).simulate(use_rust=use_rust).simplify()


def _z(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se


def _mean_root_time(ts):
    return sum(t.span * t.time(t.root) for t in ts.trees()) / ts.sequence_length


# ---------------------------------------------------------------------------
# (a) Exactness: inverted chromosome == one deme of size qN with rate r q
# ---------------------------------------------------------------------------

EX_N, EX_L, EX_R = 2_000, 20_000, 2.5e-6      # 4 N r L = 400


@pytest.mark.parametrize("kary", ["I", "S"])
def test_whole_sequence_inversion_matches_msprime_rescaled_deme(kary):
    """n = 2, whole sequence inverted, barrier never lifts: the tree count
    matches msprime SMC for a population of size qN and rate r q."""
    q = P if kary == "I" else 1 - P
    reps = 300
    inv = _spec(0.0, EX_L, 1e12)
    a = [_sim(kary, 2, EX_N, EX_L, EX_R, inv, 30_000 + s).num_trees
         for s in range(reps)]
    b = [msprime.sim_ancestry(samples=1, ploidy=2, population_size=q * EX_N,
                              sequence_length=EX_L, recombination_rate=EX_R * q,
                              model="smc", random_seed=40_000 + s).num_trees
         for s in range(reps)]
    z = _z(a, b)
    assert abs(z) < Z_TOL, (np.mean(a), np.mean(b), z)
    # The 0.3.5 rate (r instead of r q) gives 1/q times as many trees.
    old = [_sim(kary, 2, EX_N, EX_L, EX_R, inv, 30_000 + s,
                crossover_suppression=False).num_trees for s in range(reps)]
    assert _z(old, b) > 10


@pytest.mark.parametrize("kary", ["I", "S"])
def test_whole_sequence_inversion_matches_msinv_panmictic_rescaled(kary):
    """n = 10: inverted chromosome vs msinv's own panmictic engine with
    population size qN and rate r q (exact identity of the two models)."""
    q = P if kary == "I" else 1 - P
    reps = 150
    inv = _spec(0.0, EX_L, 1e12)
    inv_ts = [_sim(kary, 10, EX_N, EX_L, EX_R, inv, 50_000 + s)
              for s in range(reps)]
    pan_ts = [HullSimulator(samples=10, population_size=q * EX_N,
                            sequence_length=EX_L, recombination_rate=EX_R * q,
                            seed=60_000 + s).simulate().simplify()
              for s in range(reps)]
    for stat in (lambda ts: ts.num_trees, _mean_root_time):
        a = [stat(ts) for ts in inv_ts]
        b = [stat(ts) for ts in pan_ts]
        z = _z(a, b)
        assert abs(z) < Z_TOL, (np.mean(a), np.mean(b), z)


def test_option_off_restores_full_rate():
    """crossover_suppression=False: the inverted I chromosome is a deme of
    size pN with the full rate r (msinv <= 0.3.5)."""
    reps = 150
    inv = _spec(0.0, EX_L, 1e12)
    a = [_sim("I", 10, EX_N, EX_L, EX_R, inv, 70_000 + s,
              crossover_suppression=False).num_trees for s in range(reps)]
    b = [HullSimulator(samples=10, population_size=P * EX_N,
                       sequence_length=EX_L, recombination_rate=EX_R,
                       seed=80_000 + s).simulate().simplify().num_trees
         for s in range(reps)]
    z = _z(a, b)
    assert abs(z) < Z_TOL, (np.mean(a), np.mean(b), z)


# ---------------------------------------------------------------------------
# (b, c, d) Compensator checks, n = 2
# ---------------------------------------------------------------------------

def _region_counts(ts, lo, hi, r, w_of_T):
    """(#breakpoints strictly inside (lo, hi), r * integral of w_of_T(T(x)))."""
    bps = np.asarray(ts.breakpoints(as_array=True))
    n_bp = int(((bps > lo) & (bps < hi)).sum())
    comp = 0.0
    for tree in ts.trees():
        a, b = max(tree.interval.left, lo), min(tree.interval.right, hi)
        if b > a:
            comp += (b - a) * r * w_of_T(tree.time(tree.root))
    return n_bp, comp


def _branch_weight(p, t_inv):
    """Suppression-weighted branch length of a two-leaf tree of height T:
    both branches recombine at r p below t_inv and at r above it."""
    return lambda T: 2.0 * (p * min(T, t_inv) + max(T - t_inv, 0.0))


def _flank(T):
    return 2.0 * T


def _ratios(kary, N, L, r, t_inv, reps, seed0, use_rust=True, **kw):
    """Observed / expected breakpoints for the left flank, body, right flank
    of an inversion over the middle third, plus the raw body counts."""
    q = P if kary == "I" else 1 - P
    lo, hi = L / 3, 2 * L / 3
    inv = _spec(lo, hi, t_inv)
    w_body = _branch_weight(q, t_inv)
    obs = np.zeros(3)
    exp = np.zeros(3)
    exp_full = 0.0
    body = []
    for s in range(reps):
        ts = _sim(kary, 2, N, L, r, inv, seed0 + s, use_rust=use_rust, **kw)
        for k, (a, b, w) in enumerate([(0, lo, _flank), (lo, hi, w_body),
                                       (hi, L, _flank)]):
            o, e = _region_counts(ts, a, b, r, w)
            obs[k] += o
            exp[k] += e
            if k == 1:
                body.append(o)
                exp_full += _region_counts(ts, a, b, r, _flank)[1]
    # Poisson-scale SE of each ratio.
    return obs / exp, np.sqrt(obs) / exp, obs[1] / exp_full, np.array(body)


CMP = dict(N=1_000, L=300_000, r=1e-6)        # ~1000 breakpoints per rep


@pytest.mark.parametrize("kary", ["I", "S"])
def test_flanks_full_rate_body_suppressed(kary):
    """Barrier never lifts: flanks recombine at r, the body at r q."""
    ratio, se, ratio_vs_full, _ = _ratios(kary, t_inv=1e12, reps=40,
                                          seed0=100_000, **CMP)
    z = (ratio - 1.0) / se
    assert np.all(np.abs(z) < Z_TOL), (ratio, z)
    q = P if kary == "I" else 1 - P
    # Measured against the full-r compensator, the body is at q.
    assert abs(ratio_vs_full - q) < 0.03, ratio_vs_full


@pytest.mark.parametrize("kary", ["I", "S"])
def test_body_full_rate_older_than_t_inv(kary):
    """t_inv inside the coalescent time scale: the body recombines at r q
    below t_inv and at r above it (the compensator splits each branch)."""
    q = P if kary == "I" else 1 - P
    t_inv = 2 * CMP["N"] * q                     # ~E[T] of a same-class pair
    ratio, se, ratio_vs_full, _ = _ratios(kary, t_inv=t_inv, reps=40,
                                          seed0=200_000, **CMP)
    z = (ratio - 1.0) / se
    assert np.all(np.abs(z) < Z_TOL), (ratio, z)
    # Above t_inv the rate is r again: the body is well above q but below 1.
    assert q + 0.1 < ratio_vs_full < 0.97, ratio_vs_full


def test_young_barrier_is_full_rate():
    """t_inv far younger than the TMRCA: almost all branch length is older
    than t_inv, so the body recombines at ~r."""
    ratio, se, ratio_vs_full, _ = _ratios("I", t_inv=5.0, reps=30,
                                          seed0=300_000, **CMP)
    assert np.all(np.abs((ratio - 1.0) / se) < Z_TOL), ratio
    assert abs(ratio_vs_full - 1.0) < 0.03, ratio_vs_full


def test_option_off_body_full_rate():
    ratio, se, ratio_vs_full, _ = _ratios("I", t_inv=1e12, reps=30,
                                          seed0=400_000,
                                          crossover_suppression=False, **CMP)
    assert abs(ratio_vs_full - 1.0) < 0.03, ratio_vs_full
    assert ratio[1] > 1.4          # vs the suppressed expectation, 1/p


# Python engine: 4 N r L <= 100 keeps it on the exact per-pair path.
PYCMP = dict(N=500, L=20_000, r=2.5e-6)


@pytest.mark.parametrize("kary", ["I", "S"])
def test_python_engine_matches_rust(kary):
    py = _ratios(kary, t_inv=1e12, reps=150, seed0=500_000, use_rust=False,
                 **PYCMP)
    rs = _ratios(kary, t_inv=1e12, reps=600, seed0=600_000, **PYCMP)
    for ratio, se, _, _ in (py, rs):
        assert np.all(np.abs((ratio - 1.0) / se) < Z_TOL), (ratio, se)
    z = _z(py[3], rs[3])
    assert abs(z) < Z_TOL, (py[3].mean(), rs[3].mean(), z)


def test_python_engine_option_off():
    q = P
    ratio, se, ratio_vs_full, _ = _ratios("I", t_inv=1e12, reps=150,
                                          seed0=700_000, use_rust=False,
                                          crossover_suppression=False,
                                          **PYCMP)
    assert abs(ratio_vs_full - 1.0) < 0.15, ratio_vs_full
    assert ratio_vs_full > q + 0.2


# ---------------------------------------------------------------------------
# Unit: weight helper
# ---------------------------------------------------------------------------

def test_crossover_weight_helper():
    from msinv.hull.rates import crossover_weight
    tag_p = {"S0": 0.4, "I0": 0.6, "S1": 0.7, "I1": 0.3}
    assert crossover_weight("P", tag_p) == 1.0
    assert crossover_weight(None, tag_p) == 1.0
    assert crossover_weight("I0", tag_p) == 0.6
    assert crossover_weight(frozenset({"I0", "S1"}), tag_p) == pytest.approx(0.42)
    assert crossover_weight("I2", tag_p) == 1.0   # no active barrier


# ---------------------------------------------------------------------------
# Time-varying trajectory (precomputed, the Illex configuration)
# ---------------------------------------------------------------------------

TV_T_INV = 3_000.0
TV_TIMES = np.linspace(0.0, TV_T_INV, 61)
TV_P_I = 0.6 + (0.05 - 0.6) * TV_TIMES / TV_T_INV     # p_I falls 0.6 -> 0.05


def _cum_p(p, T):
    """integral_0^min(T, t_inv) p(s) ds for the piecewise-linear p."""
    T = min(T, TV_T_INV)
    g = np.append(TV_TIMES[TV_TIMES < T], T)
    return np.trapezoid(np.interp(g, TV_TIMES, p), g)


def _tv_ratios(kary, reps, seed0, N=1_000, L=300_000, r=1e-6):
    """Observed / expected breakpoints [left flank, body, right flank] for
    n = 2 same-arrangement samples under a precomputed p(t); the body
    compensator integrates p over each branch below t_inv. Also returns the
    body ratio against a compensator that freezes p at its t = 0 value."""
    p = TV_P_I if kary == "I" else 1.0 - TV_P_I
    lo, hi = L / 3, 2 * L / 3
    regions = [(0, lo), (lo, hi), (hi, L)]
    obs, exp = np.zeros(3), np.zeros(3)
    exp_frozen = 0.0
    for s in range(reps):
        spec = InversionSpec(
            bp_left=lo, bp_right=hi, gene_conversion_rate=1e-15,
            mean_tract_length=1.0,
            trajectory={"type": "precomputed", "times": list(TV_TIMES),
                        "freqs": [list(TV_P_I)], "n_e": [float(N)],
                        "t_inv": [TV_T_INV]})
        n_std, n_inv = (0, 2) if kary == "I" else (2, 0)
        ts = HullSimulator(n_std=n_std, n_inv=n_inv, population_size=N,
                           sequence_length=L, recombination_rate=r,
                           inversions=[spec], seed=seed0 + s
                           ).simulate().simplify()
        bps = np.asarray(ts.breakpoints(as_array=True))
        for k, (a, b) in enumerate(regions):
            obs[k] += ((bps > a) & (bps < b)).sum()
        for tree in ts.trees():
            T = tree.time(tree.root)
            for k, (a, b) in enumerate(regions):
                span = min(tree.interval.right, b) - max(tree.interval.left, a)
                if span <= 0:
                    continue
                if k == 1:
                    above = max(T - TV_T_INV, 0.0)
                    exp[k] += span * r * 2 * (_cum_p(p, T) + above)
                    exp_frozen += span * r * 2 * (p[0] * min(T, TV_T_INV) + above)
                else:
                    exp[k] += span * r * 2 * T
    return obs / exp, np.sqrt(obs) / exp, obs[1] / exp_frozen


def _check_tv(kary, reps, seed0):
    ratio, se, vs_frozen = _tv_ratios(kary, reps, seed0)
    z = (ratio - 1.0) / se
    assert np.all(np.abs(z) < Z_TOL), (ratio, se, z)
    return vs_frozen


@pytest.mark.parametrize("kary", ["I", "S"])
def test_time_varying_trajectory_compensator(kary):
    """Fast version (40 reps): flanks at r, body at r * p(t) with p read at
    the event time, for a precomputed trajectory."""
    _check_tv(kary, 40, 900)


@pytest.mark.slow
@pytest.mark.parametrize("kary", ["I", "S"])
def test_time_varying_trajectory_compensator_many_reps(kary):
    """400 reps: SE ~0.4% on the body. Freezing p at p(0) would be off by
    ~20% (I, p falls) and ~30% (S, p rises), far outside the band."""
    vs_frozen = _check_tv(kary, 400, 10_000)
    if kary == "I":
        assert vs_frozen < 0.9, vs_frozen
    else:
        assert vs_frozen > 1.15, vs_frozen


# ---------------------------------------------------------------------------
# On / off byte identity where the flag must not matter
# ---------------------------------------------------------------------------

def _table_bytes(ts):
    t = ts.dump_tables()
    return (t.nodes.time.tobytes(), t.edges.left.tobytes(),
            t.edges.right.tobytes(), t.edges.parent.tobytes(),
            t.edges.child.tobytes())


@pytest.mark.parametrize("seed", range(5))
def test_flag_is_byte_identical_without_inversion(seed):
    ts = [HullSimulator(samples=8, population_size=1_000, sequence_length=1e5,
                        recombination_rate=1e-7, seed=seed,
                        crossover_suppression=flag).simulate()
          for flag in (True, False)]
    assert _table_bytes(ts[0]) == _table_bytes(ts[1])


@pytest.mark.parametrize("seed", range(5))
def test_flag_is_byte_identical_for_untagged_samples(seed):
    """Samples with a None karyotype carry no arrangement tag, so their
    material has weight 1 even under an active barrier."""
    ts = []
    for flag in (True, False):
        spec = _spec(3e4, 7e4, 1e9, p=0.4)
        ts.append(HullSimulator(sample_config={(None, 0): 8},
                                population_size=1_000, sequence_length=1e5,
                                recombination_rate=1e-7, inversions=[spec],
                                seed=seed, crossover_suppression=flag
                                ).simulate())
    assert _table_bytes(ts[0]) == _table_bytes(ts[1])
