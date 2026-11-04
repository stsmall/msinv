"""The msinv engine switch: every inversion body runs on msinv >= 0.4.0 with crossover suppression.

Fast tests: the msinv call passes crossover_suppression=True explicitly, a no-flux row uses GAMMA_FLOOR (msinv
rejects gamma = 0), and an msinv older than 0.4.0 is refused.

Slow tests (one fixed soft D1 path, the Task 9 ``soft_D1`` set, N_REPS reps per engine, error-free LD, raw body
statistics from ``statvec.region_stats`` plus the edge/centre ratio):
  (a) msinv with crossover_suppression=False matches the old msprime no-flux engine in distribution. A statistic
      passes when |delta| < 0.25 pooled SD (``slim_vs_msinv.compare_stat``) OR its KS p-value exceeds a Bonferroni
      0.01 / n_stats (at n = 200 per side the 0.25 SD rule alone has a ~1.3% null FAIL rate per statistic).
  (b) suppression on raises the within-arrangement LD bins ld_I_* and ld_S_*.
Both print their per-statistic tables (run with ``-s``). Workers: env ENGINE_SWITCH_WORKERS (default 16).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from illex.abc import simulate as SIM
from illex.abc import statvec as SV
from illex.abc.constants import BODY_BP, GAMMA_FLOOR, MARGIN_BP
from illex.abc.trajectory import Path

NAN = float("nan")
N_REPS = 200
SETS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "illex", "abc_slim_sets.json")


def _soft_path(t_inv=5e5, gamma_eff=NAN):
    prm = dict(model="N", flux=not np.isnan(gamma_eff), t_inv=float(t_inv), hard=False, p0=0.05, t_sel=NAN, s=NAN,
               t_stop=NAN, p_star=NAN, gamma_eff=gamma_eff)
    return Path(params=prm, times=np.array([0.0, float(t_inv)]), freqs=np.array([0.62, 0.05]), p_now=0.62)


class _Captured(Exception):
    pass


def _capture_msinv(monkeypatch):
    import msinv

    seen = {}

    class FakeSim:
        def __init__(self, **kw):
            seen.update(kw)
            raise _Captured

    monkeypatch.setattr(msinv, "HullSimulator", FakeSim)
    return seen


@pytest.mark.parametrize("gamma_eff", [NAN, 1e-7])
def test_msinv_call_passes_suppression_explicitly(monkeypatch, gamma_eff):
    seen = _capture_msinv(monkeypatch)
    with pytest.raises(_Captured):
        SIM._inversion_ts(_soft_path(gamma_eff=gamma_eff), seed=3)
    assert seen["crossover_suppression"] is True
    assert isinstance(seen["seed"], int)
    gamma = seen["inversions"][0].gene_conversion_rate
    if np.isnan(gamma_eff):
        assert gamma == GAMMA_FLOOR                                  # smallest rate msinv accepts (no flux)
    else:
        assert gamma == pytest.approx(SIM.gamma_from_c(gamma_eff))


def test_suppression_switch_reaches_msinv(monkeypatch):
    seen = _capture_msinv(monkeypatch)
    with pytest.raises(_Captured):
        SIM.simulate_stats(_soft_path(), seed=3, crossover_suppression=False)
    assert seen["crossover_suppression"] is False
    with pytest.raises(_Captured):
        SIM.simulate_stats(_soft_path(), seed=3)                     # production default
    assert seen["crossover_suppression"] is True


def test_auto_is_msinv_everywhere():
    assert SIM.engine_for(_soft_path()) == "msinv"
    assert SIM.engine_for(_soft_path(gamma_eff=1e-7)) == "msinv"
    assert SIM.engine_for(_soft_path(), "msprime") == "msprime"
    with pytest.raises(ValueError):
        SIM.engine_for(_soft_path(gamma_eff=1e-7), "msprime")


def test_old_msinv_refused(monkeypatch):
    import msinv

    assert SIM._version_tuple(msinv.__version__) >= SIM.MSINV_MIN
    monkeypatch.setattr(msinv, "__version__", "0.3.5")
    with pytest.raises(RuntimeError, match="0.4.0"):
        SIM._inversion_ts(_soft_path(), seed=3)


def test_version_tuple():
    assert SIM._version_tuple("0.4.0") == (0, 4, 0)
    assert SIM._version_tuple("0.4.1.dev3") == (0, 4, 1)
    assert SIM._version_tuple("1.0") == (1, 0, 0)


# ------------------------------------------------------------------------------------------ slow: (a), (b) ----
def _body(args):
    """Raw body statistics (error-free LD) and the edge/centre ratio of one replicate on one engine."""
    from illex import stats as S

    path, seed, engine, cs = args
    s_inv, _, s_mi, _, s_pi, _, _, _, _ = SIM._seed_words(seed)
    if engine == "msprime":
        ts, i_nodes, s_nodes = SIM._inversion_ts_msprime(path, s_inv)
        win = (0.0, BODY_BP)
    else:
        sim, ts = SIM._inversion_ts(path, s_inv, crossover_suppression=cs)
        i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
        win = (MARGIN_BP, MARGIN_BP + BODY_BP)
    out = SV.region_stats(SIM._mutate(ts, s_mi), i_nodes, s_nodes, win, np.random.default_rng(s_pi))
    out["edge_centre"] = SV.edge_centre(ts, i_nodes, s_nodes, win)
    return out


@pytest.fixture(scope="module")
def engine_rows():
    from illex.abc.slim_vs_msinv import draw_paths, load_sets

    (path,), _ = draw_paths(load_sets(SETS)["soft_D1"], 1, np.random.default_rng(20261104))
    jobs = {"msprime": ("msprime", False), "msinv_off": ("msinv", False), "msinv_on": ("msinv", True)}
    workers = int(os.environ.get("ENGINE_SWITCH_WORKERS", "16"))
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, (eng, cs) in jobs.items():
            rows[name] = list(ex.map(_body, [(path, 70_000 + r, eng, cs) for r in range(N_REPS)], chunksize=4))
    return path, rows


def _table(a_rows, b_rows):
    from illex.abc.slim_vs_msinv import compare_stat

    keys = [k for k in a_rows[0] if k in b_rows[0]]
    out = {}
    for k in keys:
        out[k] = compare_stat([r[k] for r in a_rows], [r[k] for r in b_rows])
    return out


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_a_msinv_without_suppression_matches_msprime(engine_rows):
    path, rows = engine_rows
    tab = _table(rows["msinv_off"], rows["msprime"])
    n = sum(1 for v in tab.values() if v["verdict"] != "NA")
    ks_min = 0.01 / n
    print(f"\n(a) msinv (suppression off) vs msprime, soft D1 path t_inv={path.params['t_inv']:.0f} "
          f"p_now={path.p_now:.3f}, {N_REPS} reps each; delta = (msinv - msprime) / pooled SD")
    bad = []
    for k, v in tab.items():
        flag = v["verdict"] == "FAIL" and not (v["ks_p"] > ks_min)
        print(f"  {k:14s} msinv {v['mean_slim']:.5g}  msprime {v['mean_sim']:.5g}  delta {v['delta_sd']:+.3f}  "
              f"KS p {v['ks_p']:.3g}  {v['verdict']}{'  <-- FLAG' if flag else ''}")
        if flag:
            bad.append(k)
    assert not bad, bad


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_b_suppression_raises_within_arrangement_ld(engine_rows):
    _, rows = engine_rows
    tab = _table(rows["msinv_on"], rows["msinv_off"])
    print("\n(b) msinv suppression on vs off; delta = (on - off) / pooled SD, rel = on / off - 1")
    ld = [k for k in tab if k.startswith(("ld_I_", "ld_S_"))]
    for k, v in tab.items():
        rel = v["mean_slim"] / v["mean_sim"] - 1 if v["mean_sim"] else NAN
        print(f"  {k:14s} on {v['mean_slim']:.5g}  off {v['mean_sim']:.5g}  rel {rel:+.1%}  "
              f"delta {v['delta_sd']:+.3f}  KS p {v['ks_p']:.3g}")
    assert ld
    for k in ld:
        assert tab[k]["mean_slim"] > tab[k]["mean_sim"], k
    for k in ("ld_I_1", "ld_S_1"):                                     # the short-range bins rise clearly
        assert tab[k]["delta_sd"] > 0.25 and tab[k]["ks_p"] < 1e-3, (k, tab[k])
