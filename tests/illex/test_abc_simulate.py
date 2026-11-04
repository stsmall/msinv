"""Task 3: simulator and statistic vector for the trajectory ABC.

Engine dispatch is ``engine="auto"``: no-flux paths (gamma_eff NaN) run on msprime, flux paths on msinv
(``test_abc_msprime_engine.py`` covers the msprime engine itself).

The rep-heavy tests (no-barrier limit, flux scaling) run 20 sims each and are marked slow.
``gamma_eff`` in a path's params is c, the per-site conversion probability per generation in heterokaryotypes.
"""
import numpy as np
import pytest

from illex.abc.simulate import simulate_stats
from illex.abc.statvec import BND_KEYS, STAT_KEYS, STAT_KEYS_PPC_ONLY
from illex.abc.trajectory import Path
from illex.theory import N_growth

NAN = float("nan")
RATIOS = ("pi_ratio", "dxy_pi", "f1_ratio")


def path(t_inv, p_now=0.626, p_start=None, gamma_eff=NAN, hard=False):
    """A flat (or two-point linear) path, built by hand so the tests do not depend on the sampler."""
    p_start = p_now if p_start is None else p_start
    prm = dict(model="N", flux=not np.isnan(gamma_eff), t_inv=float(t_inv), hard=hard,
               p0=NAN if hard else p_start, t_sel=NAN, s=NAN, t_stop=NAN, p_star=NAN, gamma_eff=gamma_eff)
    return Path(params=prm, times=np.array([0.0, float(t_inv)]), freqs=np.array([p_now, p_start]), p_now=p_now)


def test_keys_contract():
    out = simulate_stats(path(2e4), seed=11)
    assert set(out) == set(STAT_KEYS) | {"p_obs", "n_nan"}
    assert set(STAT_KEYS_PPC_ONLY) - set(BND_KEYS) <= set(STAT_KEYS)   # boundary keys come from boundary_ppc
    assert len(set(STAT_KEYS)) == len(STAT_KEYS)
    assert all(isinstance(v, float) for k, v in out.items() if k != "n_nan")
    assert out["n_nan"] == sum(np.isnan(out[k]) for k in STAT_KEYS)
    assert 0.0 < out["p_obs"] < 1.0


def test_seed_deterministic():
    p = path(2e4, gamma_eff=1e-5)
    a, b = simulate_stats(p, seed=5), simulate_stats(p, seed=5)
    assert a.keys() == b.keys()
    for k in a:
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k], k
    c = simulate_stats(p, seed=6)
    assert any(a[k] != c[k] for k in STAT_KEYS if not np.isnan(a[k]))


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_no_barrier_limit():
    p = path(100.0, p_now=0.5)
    rows = [simulate_stats(p, seed=100 + r) for r in range(20)]
    for k in RATIOS:
        m = np.nanmean([r[k] for r in rows])
        assert abs(m - 1.0) <= 0.05, (k, m)


def test_zero_seg_sites_flagged():
    # A hard start 2,000 generations ago, rising exponentially from one copy. A two-point (linear-in-p) path
    # would NOT do: msinv interpolates p linearly, so p stays O(1) until the last generation, the 2a lineages
    # barely coalesce, and they are released into the panmictic pool at t_inv (no forced monophyly).
    t_inv, p_now = 2000.0, 0.626
    p_start = 1.0 / (2.0 * float(N_growth(t_inv)))
    times = np.linspace(0.0, t_inv, 201)
    freqs = p_now * np.exp(np.log(p_start / p_now) * times / t_inv)
    p = path(t_inv, p_now=p_now, p_start=p_start, hard=True)
    p.times, p.freqs = times, freqs
    out = simulate_stats(p, seed=3)
    assert out["n_nan"] > 0


@pytest.mark.slow
@pytest.mark.timeout(0)
def test_flux_scaling():
    # c = 1e-9 per site per generation is effectively no flux; c = 1e-5 (the Korunes & Noor scale) nearly
    # homogenises the arrangements at 0.9 Ma. c = 1e-4 would be a stronger contrast but one call takes ~18 min
    # and 40 GB RSS, so it is not used here.
    lo = [simulate_stats(path(9e5, gamma_eff=1e-9), seed=200 + r)["dxy_pi"] for r in range(20)]
    hi = [simulate_stats(path(9e5, gamma_eff=1e-5), seed=200 + r)["dxy_pi"] for r in range(20)]
    assert np.nanmean(hi) < 0.95 * np.nanmean(lo), (np.nanmean(hi), np.nanmean(lo))
