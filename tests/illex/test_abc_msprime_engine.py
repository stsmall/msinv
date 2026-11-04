"""Task 3b: the exact msprime engine for the no-flux trajectory ABC rows (and the msprime neutral control).

References: the LD debug report (.superpowers/sdd/2026-11-03-illex-chr2-trajectory-abc/ld-debug-report.md, Q1):
an msprime two-deme split (I = p N(t), S = (1 - p) N(t), merging at t_inv) at flat p = 0.626, t_inv = 8e5 gives
error-free within-class r^2 of 0.185 +- 0.003 (I) and 0.222 +- 0.003 (S) in LD bin 1 (100-240 bp), 0.120 / 0.150
in bin 2.
"""
import numpy as np
import pytest

from illex.abc import simulate as SIM
from illex.abc import statvec as SV
from illex.abc.constants import BODY_BP
from illex.abc.simulate import simulate_stats
from illex.abc.statvec import STAT_KEYS
from illex.abc.trajectory import Path
from illex.theory import N_growth

NAN = float("nan")
NO_ERR = {"body": None, "control": None}


def path(t_inv, p_now=0.626, p_start=None, gamma_eff=NAN, hard=False):
    p_start = p_now if p_start is None else p_start
    prm = dict(model="N", flux=not np.isnan(gamma_eff), t_inv=float(t_inv), hard=hard,
               p0=NAN if hard else p_start, t_sel=NAN, s=NAN, t_stop=NAN, p_star=NAN, gamma_eff=gamma_eff)
    return Path(params=prm, times=np.array([0.0, float(t_inv)]), freqs=np.array([p_now, p_start]), p_now=p_now)


def hard_path(t_inv=6e5, g_est=2000.0, p_now=0.626):
    """A hard-start path in the sampler's format: a 1000-generation grid from the present back to the end of
    establishment (t_inv - g_est, at ESTABLISHED / 2N), then the single first copy (t_inv, 1 / 2N). The grid part
    rises logistically from 500 / 2N to p_now."""
    two_n = 2.0 * float(N_growth(t_inv))
    t_start = t_inv - g_est
    p_est = 500.0 / two_n
    tt = np.arange(0.0, t_start, 1000.0)
    lo, hi = np.log(p_est / (1 - p_est)), np.log(p_now / (1 - p_now))
    z = hi + (lo - hi) * tt / t_start
    ff = 1.0 / (1.0 + np.exp(-z))
    prm = dict(model="D1", flux=False, t_inv=float(t_inv), hard=True, p0=NAN, t_sel=float(t_inv), s=1e-4,
               t_stop=NAN, p_star=NAN, gamma_eff=NAN)
    return Path(params=prm, times=np.concatenate([tt, [t_start, t_inv]]),
                freqs=np.concatenate([ff, [p_est, 1.0 / two_n]]), p_now=float(ff[0]))


def test_engine_dispatch():
    assert SIM.engine_for(path(2e4)) == "msinv"                  # auto: msinv for every inversion body
    assert SIM.engine_for(path(2e4), "msprime") == "msprime"     # validation engine, no-flux only
    assert SIM.engine_for(path(2e4, gamma_eff=1e-6)) == "msinv"
    with pytest.raises(ValueError):
        SIM.engine_for(path(2e4, gamma_eff=1e-6), "msprime")      # msprime has no flux
    assert SIM.engine_for(path(2e4), "msinv") == "msinv"
    with pytest.raises(ValueError):
        SIM.engine_for(path(2e4), "bogus")


def test_keys_contract_msprime():
    out = simulate_stats(path(2e4), seed=11, engine="msprime")
    assert set(out) == set(STAT_KEYS) | {"p_obs", "n_nan"}
    assert all(isinstance(v, float) for k, v in out.items() if k != "n_nan")
    assert out["n_nan"] == sum(np.isnan(out[k]) for k in STAT_KEYS)
    assert out["n_nan"] == 0


def test_seed_deterministic_msprime():
    p = path(8e5, p_start=0.3)
    a, b = simulate_stats(p, seed=5, engine="msprime"), simulate_stats(p, seed=5, engine="msprime")
    for k in a:
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k], k
    c = simulate_stats(p, seed=6, engine="msprime")
    assert any(a[k] != c[k] for k in STAT_KEYS if not np.isnan(a[k]))


def test_auto_uses_msprime_body_window():
    """The msprime validation engine simulates a body-only BODY_BP sequence."""
    ts, i_nodes, s_nodes = SIM._inversion_ts_msprime(path(2e4), seed=3)
    assert ts.sequence_length == BODY_BP
    assert len(i_nodes) == 100 and len(s_nodes) == 100
    assert set(ts.samples(population=ts.population(0).id)) == set(i_nodes)


def test_demography_sizes():
    p = path(8e5, p_start=0.3)
    d = SIM.inversion_demography(p)
    dbg = d.debug()
    for t in (0.0, 5e3, 3e5, 7.9e5):
        sz = dbg.population_size_trajectory(np.array([t]))[0]
        pt = np.interp(t, p.times, p.freqs)
        n = float(N_growth(t))
        assert sz[0] == pytest.approx(pt * n, rel=0.01), t
        assert sz[1] == pytest.approx((1 - pt) * n, rel=0.01), t
    sz = dbg.population_size_trajectory(np.array([9e5]))[0]
    assert sz[2] == pytest.approx(float(N_growth(9e5)), rel=1e-6)


def test_no_barrier_limit_msprime():
    p = path(100.0, p_now=0.5)
    rows = [simulate_stats(p, seed=100 + r, engine="msprime") for r in range(20)]
    for k in ("pi_ratio", "dxy_pi", "f1_ratio"):
        m = np.nanmean([r[k] for r in rows])
        assert abs(m - 1.0) <= 0.05, (k, m)


def _body_raw(p, seed):
    ts, i_nodes, s_nodes = SIM._inversion_ts_msprime(p, seed)
    mts = SIM._mutate(ts, seed + 1)
    return SV.region_stats(mts, i_nodes, s_nodes, (0.0, BODY_BP), np.random.default_rng(seed + 2))


@pytest.mark.timeout(180)
def test_flat_ld_matches_reference():
    p = path(8e5, p_now=0.626)
    rows = [_body_raw(p, 1000 + 7 * r) for r in range(24)]
    for key, ref in (("ld_I_1", 0.185), ("ld_S_1", 0.222), ("ld_I_2", 0.120), ("ld_S_2", 0.150)):
        v = np.array([r[key] for r in rows])
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
        assert abs(m - ref) <= max(4 * se, 0.012), (key, m, se, ref)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_hard_path_tmrca_before_t_inv(seed):
    p = hard_path()
    ts, i_nodes, _ = SIM._inversion_ts_msprime(p, seed)
    sub = ts.simplify(samples=list(i_nodes))
    root_t = max(sub.node(t.root).time for t in sub.trees())
    assert sub.num_trees >= 1 and all(t.num_roots == 1 for t in sub.trees())
    assert root_t <= p.params["t_inv"], root_t


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_msprime_matches_msinv_pi_dxy():
    """msinv without crossover suppression (msprime has none) must agree with msprime on pi and dxy (soft path);
    test_abc_engine_switch.py runs the full 200-rep equivalence over every body statistic."""
    p = path(8e5, p_now=0.626, p_start=0.3)
    res = {}
    for eng in ("msprime", "msinv"):
        res[eng] = [simulate_stats(p, seed=300 + r, engine=eng, ld_error=NO_ERR, crossover_suppression=False)
                    for r in range(20)]
    for k in ("pi_ratio", "dxy_pi"):
        a = np.array([r[k] for r in res["msprime"]])
        b = np.array([r[k] for r in res["msinv"]])
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        assert abs(a.mean() - b.mean()) <= 3 * se, (k, a.mean(), b.mean(), se)
