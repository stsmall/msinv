"""Task 9: SLiM validation of the trajectory-ABC simulator (illex/abc/slim_vs_msinv.py, illex/slim/run_one.py fixed
mode, illex/slim/summarize.py summarize_abc, the ABC models of illex/slim/inversion_abc.slim).

The two tests that run SLiM need the slim binary ($SLIM_BIN, PATH, or the box's slim_sim env) and are slow.
"""
import math
import subprocess
from pathlib import Path

import msprime
import numpy as np
import pandas as pd
import pytest
import tskit

from illex.abc import simulate
from illex.abc import slim_vs_msinv as V
from illex.abc import statvec as SV
from illex.abc.constants import P_NOW, P_TOL
from illex.abc.trajectory import Path as TPath
from illex.slim import run_one as R
from illex.slim.summarize import abc_stats_from_ts, karyotype_sample_nodes, summarize_abc

NAN = float("nan")
SLIM = R.default_slim()
HAVE_SLIM = Path(SLIM).exists() or bool(__import__("shutil").which(SLIM))


# ------------------------------------------------------------------ compare_stat / compare ----
def test_compare_stat_hand_values():
    r = V.compare_stat([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert r["n_slim"] == 3 and r["n_sim"] == 3
    assert r["sd_pooled"] == pytest.approx(1.0)
    assert r["delta_sd"] == pytest.approx(-1.0)
    assert r["ks_D"] == pytest.approx(1.0 / 3.0)
    assert r["verdict"] == "FAIL"


@pytest.mark.parametrize("shift, verdict", [(0.0, "PASS"), (0.1, "PASS"), (0.5, "FAIL"), (1.0, "FAIL")])
def test_compare_stat_known_shift(shift, verdict):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4000) * 2.0 + 5.0 + 2.0 * shift      # shift in SD units (SD = 2)
    b = rng.standard_normal(4000) * 2.0 + 5.0
    r = V.compare_stat(a, b)
    assert r["delta_sd"] == pytest.approx(shift, abs=0.06)
    assert r["verdict"] == verdict
    if shift >= 0.5:
        assert r["ks_p"] < 1e-10
    if shift == 0.0:
        assert r["ks_p"] > 1e-3


def test_compare_stat_na():
    assert V.compare_stat([1.0], [1.0, 2.0])["verdict"] == "NA"
    assert V.compare_stat([NAN, NAN, 1.0], [1.0, 2.0, 3.0])["verdict"] == "NA"      # 1 finite SLiM value
    assert V.compare_stat([2.0, 2.0], [2.0, 2.0])["verdict"] == "NA"                # pooled SD 0
    r = V.compare_stat([1.0, NAN, 3.0, 2.0], [2.0, 3.0, 4.0])                       # NaN dropped
    assert r["n_slim"] == 3 and r["verdict"] == "FAIL"


def test_same_window_filter_on_both_sides():
    """Rows outside |p_now - P_NOW| <= P_TOL (and SLiM rows that did not finish) are dropped on BOTH sides."""
    inside = [P_NOW - 0.9 * P_TOL, P_NOW, P_NOW + 0.9 * P_TOL]
    outside = [P_NOW - 1.5 * P_TOL, P_NOW + 1.5 * P_TOL, NAN]
    slim = pd.DataFrame(dict(set="x", status=["ok"] * 6 + ["slim_error"],
                             p_now=inside + outside + [P_NOW], pi_ratio=[1.0, 2.0, 3.0, 1e3, 1e3, 1e3, 1e3]))
    sim = pd.DataFrame(dict(set="x", status="ok", p_now=inside + outside,
                            pi_ratio=[1.0, 2.0, 3.0, -1e3, -1e3, -1e3]))
    t = V.compare(slim, sim, keys=("pi_ratio",))
    r = t.iloc[0]
    assert r["n_slim_rows"] == 3 and r["n_sim_rows"] == 3
    assert r["mean_slim"] == pytest.approx(2.0) and r["mean_sim"] == pytest.approx(2.0)
    assert r["delta_sd"] == pytest.approx(0.0) and r["verdict"] == "PASS"
    assert list(V.in_window([P_NOW + P_TOL - 1e-9, P_NOW + P_TOL + 1e-9, NAN])) == [True, False, False]


def test_compare_sets_and_report():
    rng = np.random.default_rng(3)
    rows = []
    for name, shift in (("a", 0.0), ("b", 3.0)):
        for side in ("slim", "sim"):
            v = rng.standard_normal(50) + (shift if side == "slim" else 0.0)
            rows.append(pd.DataFrame(dict(set=name, status="ok", p_now=P_NOW, side=side, pi_ratio=v, dxy_pi=v)))
    df = pd.concat(rows)
    slim, sim = df[df.side == "slim"], df[df.side == "sim"]
    t = V.compare_sets(slim, sim, {"a": {"model": "N"}, "b": {"model": "B"}})
    assert list(t["set"].unique()) == ["a", "b"]
    assert set(t["stat"]) == set(V.COMPARE_KEYS)
    got = t.set_index(["set", "stat"])["verdict"]
    assert got[("a", "pi_ratio")] == "PASS" and got[("b", "pi_ratio")] == "FAIL"
    assert got[("a", "ld_I_1")] == "NA"                     # absent column -> no values -> NA
    txt = V.report_text(t, slim, sim, {"a": {"model": "N"}, "b": {"model": "B"}})
    assert "FAIL pi_ratio" in txt and "[b] model=B" in txt


# ------------------------------------------------------------------ simulator side ----
def test_path_params_hard_and_flux():
    p = V.path_params(dict(model="B", hard=True, t_inv=5e5, p0=0.05, t_sel=1e5, s=1e-3, p_star=0.62), 3)
    assert np.all(p["hard"]) and np.all(np.isnan(p["p0"])) and np.all(p["t_sel"] == 5e5)
    assert p["flux"] is False and p["model"] == "B"
    q = V.path_params(dict(model="D2", hard=False, t_inv=5e5, p0=0.05, t_sel=3e5, s=1e-4, t_stop=2e5,
                           gamma_eff=1e-7), 2)
    assert q["flux"] is True and np.all(q["t_stop"] == 2e5) and np.all(q["p0"] == 0.05)


def _tpath(gamma_eff):
    prm = dict(model="N", flux=math.isfinite(gamma_eff), t_inv=5e5, hard=False, p0=0.05, t_sel=NAN, s=NAN,
               t_stop=NAN, p_star=NAN, gamma_eff=gamma_eff)
    return TPath(params=prm, times=np.array([0.0, 5e5]), freqs=np.array([0.62, 0.05]), p_now=0.62)


@pytest.mark.parametrize("c, engine", [(NAN, "msprime"), (1e-7, "msinv")])
def test_simulator_engine_from_engine_for(monkeypatch, c, engine):
    seen = {}

    def fake(path, seed, engine="auto"):
        seen["engine"] = engine
        return dict.fromkeys(SV.STAT_KEYS, 1.0)

    monkeypatch.setattr(simulate, "simulate_stats", fake)
    row = V._sim_one((_tpath(c), 7, "x"))
    assert seen["engine"] == engine == row["engine"] == simulate.engine_for(_tpath(c))
    assert row["status"] == "ok" and row["p_now"] == 0.62
    if not simulate.BND_PPC_READY:                 # the boundary keys wait for the msinv fixes
        assert all(math.isnan(row[k]) for k in SV.BND_KEYS)


def test_draw_paths_window():
    pset = dict(model="B", hard=False, t_inv=480411.0, p0=0.0515, t_sel=432374.0, s=2.65e-4, p_star=0.622)
    paths, counts = V.draw_paths(pset, 5, np.random.default_rng(0), batch=50)
    assert len(paths) == 5 and counts["drawn"] >= 50
    assert all(abs(p.p_now - P_NOW) <= P_TOL for p in paths)


# ------------------------------------------------------------------ run_one fixed mode ----
def test_check_params_guards():
    ok = dict(model="D1", hard=False, t_inv=5e5, p0=0.07, t_sel=3e5, s=7e-6, Q=200.0)
    assert R.check_params(ok)["t_stop"] != R.check_params(ok)["t_stop"]           # NaN
    with pytest.raises(ValueError, match="s \\* Q"):
        R.check_params(dict(ok, s=6e-4))                                            # s Q = 0.12
    with pytest.raises(ValueError, match="Q ="):
        R.check_params(dict(ok, Q=400.0))
    with pytest.raises(ValueError, match="p0"):
        R.check_params(dict(ok, p0=None))
    R.check_params(dict(model="N", hard=True, t_inv=5e5, Q=200.0))                  # N needs no s
    assert R.load_params('{"model": "N", "hard": false, "t_inv": 5e5, "p0": 0.07, "Q": 200}')["Q"] == 200.0


def test_slim_command_abc():
    hard = R.check_params(dict(model="B", hard=True, t_inv=6e5, t_sel=1e5, s=1e-3, p_star=0.62, Q=80.0))
    cmd = " ".join(R.slim_command_abc("slim", hard, Path("x.trees"), 5))
    assert "-d T_SEL=600000.0" in cmd and "-d HARD=T" in cmd and '-d MODEL="B"' in cmd
    assert "-d P_STAR=0.62" in cmd and "-d GC_C=0.0" in cmd and "-d INV_LEN=40000" in cmd
    assert "-d FLANK_LEN=100000" in cmd and "-d R=2.158e-09" in cmd
    flux = R.check_params(dict(model="D2", hard=False, t_inv=5e5, p0=0.05, t_sel=3e5, s=8e-5, t_stop=2e5,
                               gamma_eff=1e-7, Q=200.0))
    cmd = " ".join(R.slim_command_abc("slim", flux, Path("x.trees"), 5))
    assert "-d GC_C=1e-07" in cmd and "-d T_STOP=200000.0" in cmd and "-d HARD=F" in cmd
    assert "-d P_START=0.05" in cmd and "-d GC_TRACT=300.0" in cmd


def test_sets_file_valid():
    """Every set in the Task 9 sets file passes the fixed-mode guards (s Q < 0.1, Q <= 200)."""
    f = Path(__file__).resolve().parents[2] / "results/illex/abc_slim_sets.json"
    if not f.exists():
        pytest.skip("results/illex/abc_slim_sets.json not present")
    sets = V.load_sets(f)
    assert {"soft_D1", "hard_D1", "hard_B", "soft_B", "D2_flux", "N_soft"} <= set(sets)
    for name in sets:
        p = R.load_params(str(f), name)
        assert p["model"] == "N" or p["s"] * p["Q"] < 0.1
    assert math.isfinite(sets["D2_flux"]["gamma_eff"]) and sets["D2_flux"]["provisional"]


# ------------------------------------------------------------------ summarize ----
def _marked_ts(n_hap=320, frac_i=0.6, seed=4):
    """msprime ts over a 40 kb body between 100 kb margins, with a SLiM-like marker site at the body's start
    carried by the first round(frac_i * n_hap) samples (one mutation per carrier leaf)."""
    ts = msprime.sim_ancestry(samples=n_hap, ploidy=1, population_size=2_000, sequence_length=240_000,
                              recombination_rate=1e-8, random_seed=seed)
    t = ts.dump_tables()
    s = t.sites.add_row(position=100_000, ancestral_state="0")
    carriers = np.arange(int(round(frac_i * n_hap)))
    for u in carriers:
        t.mutations.add_row(site=s, node=int(u), derived_state="1", time=0.0)
    t.sort()
    t.build_index()
    t.compute_mutation_parents()
    return t.tree_sequence(), set(carriers.tolist())


def test_karyotype_and_stats_keys_from_marked_ts():
    ts, carriers = _marked_ts()
    i_nodes, s_nodes = karyotype_sample_nodes(ts, 100_000)
    assert set(i_nodes.tolist()) == carriers and len(s_nodes) == ts.num_samples - len(carriers)
    out = abc_stats_from_ts(ts, 100_000, 139_999, q=1.0, seed=11, mu=1e-7, mask=None)
    assert set(SV.STAT_KEYS) | set(SV.BND_KEYS) <= set(out)
    assert out["p_now"] == pytest.approx(len(carriers) / ts.num_samples)
    assert out["n_i_pop"] == len(carriers)
    for k in ("pi_ratio", "dxy_pi", "f1_ratio", "edge_centre", "bnd_piratio_b3", "bnd_dxypi_b3"):
        assert math.isfinite(out[k]), k
    # Exchangeable classes (the marker is on arbitrary leaves): no barrier, so the body dxy / pi_I is ~ the
    # control's and the flank ratios are ~ 1. Loose bounds: one replicate.
    assert 0.7 < out["dxy_pi"] < 1.4
    assert 0.7 < out["bnd_dxypi_b3"] < 1.4
    again = abc_stats_from_ts(ts, 100_000, 139_999, q=1.0, seed=11, mu=1e-7, mask=None)
    assert all((out[k] == again[k]) or (math.isnan(out[k]) and math.isnan(again[k])) for k in SV.STAT_KEYS)
    with pytest.raises(RuntimeError, match="too few"):
        abc_stats_from_ts(*_marked_ts(n_hap=150)[:1], 100_000, 139_999, q=1.0, seed=1, mu=1e-7, mask=None)


# ------------------------------------------------------------------ SLiM end to end (slow) ----
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SLIM, reason="no slim binary")
def test_slim_recipe_models(tmp_path):
    """The ABC models of inversion_abc.slim at a tiny Q: genotype fitnesses, selection window, one hard copy."""
    prm = R.check_params(dict(name="t", model="B", hard=True, t_inv=3e5, s=4e-5, p_star=0.62, Q=2000.0),
                         q_max=5000.0)
    trees = tmp_path / "b.trees"
    proc = subprocess.run(R.slim_command_abc(SLIM, prm, trees, 3), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "on 1 of" in proc.stdout                               # one copy at t_inv
    meta = tskit.load(str(trees)).metadata["SLiM"]["user_metadata"]
    sq = 4e-5 * 2000.0
    assert meta["w_het"][0] == pytest.approx(1 + sq) and meta["w_hom"][0] == pytest.approx(1 + sq * (2 - 1 / 0.62))
    assert meta["ts_on"][0] == meta["tintro"][0] and meta["ts_off"][0] == meta["g_total"][0] - 1
    out = summarize_abc(str(trees), 3, mask="data")
    assert set(SV.STAT_KEYS) | set(SV.BND_KEYS) <= set(out)
    assert out["p_now"] == pytest.approx(meta["p_final"][0], abs=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SLIM, reason="no slim binary")
def test_run_replicate_abc_end_to_end(tmp_path):
    prm = R.check_params(dict(name="t", model="D2", hard=False, t_inv=3e5, p0=0.08, t_sel=2e5, t_stop=1e5,
                              s=2e-5, gamma_eff=1e-6, Q=2000.0), q_max=5000.0)
    row = R.run_replicate_abc(prm, 0, 12, SLIM, tmp_path, 600)
    assert row["status"] == "ok", row.get("stderr_tail")
    assert row["slim_wall_s"] > 0 and row["slim_maxrss_mb"] > 0
    assert all(k in row for k in SV.STAT_KEYS + SV.BND_KEYS)
    assert not list(tmp_path.glob("*.trees*"))                    # scratch cleaned
