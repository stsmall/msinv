"""Full-length rows with two flux processes (illex/abc/fulllen.py, run_full.py, control_bank.py).

The full-length region statistics must equal ``statvec.region_stats`` on the same tree sequence and seeds, so the
42 fitted statistics mean the same thing as in the 40-kb fits. The end-to-end row runs at a reduced length.
"""
import numpy as np
import pytest

import illex.abc.fulllen as F
from illex.abc import statvec as SV
from illex.abc.constants import BODY_BP, MARGIN_BP, TRACT_BP
from illex.abc.error_model import LD_ERROR
from illex.abc.simulate import _mutate, gamma_from_c, simulate_stats
from illex.abc.trajectory import Path

NAN = float("nan")


def path(t_inv=2e4, p_now=0.6, p0=0.05, c=3e-7):
    prm = dict(model="B", flux=True, t_inv=float(t_inv), hard=False, p0=p0, t_sel=t_inv / 2, s=1e-3, t_stop=NAN,
               p_star=0.6, gamma_eff=c)
    return Path(params=prm, times=np.array([0.0, float(t_inv)]), freqs=np.array([p_now, p0]), p_now=p_now)


def test_gcr_matches_40kb_conversion():
    assert F.gcr_from_c(3e-7, TRACT_BP, BODY_BP) == pytest.approx(gamma_from_c(3e-7), rel=1e-12)


def test_long_prior_bounds():
    rng = np.random.default_rng(0)
    d = np.array([F.draw_long(rng) for _ in range(2000)])
    assert (d[:, 0] >= 1e-11).all() and (d[:, 0] <= 1e-6).all()
    assert (d[:, 1] >= 1e4).all() and (d[:, 1] <= 5e6).all()
    # log-uniform: median near the geometric midpoint
    assert np.log10(np.median(d[:, 1])) == pytest.approx(0.5 * (4 + np.log10(5e6)), abs=0.15)


def test_region_full_reproduces_region_stats():
    ts = _mutate(_short_control(7), 8)
    smp = ts.samples()
    i_nodes, s_nodes = smp[100:200], smp[:100]
    win = (0.0, ts.sequence_length)
    ref = SV.region_stats(ts, i_nodes, s_nodes, win, np.random.default_rng(1), ld_err=LD_ERROR["control"],
                          err_rng=np.random.default_rng(2))
    raw, ext = F.region_full(ts, i_nodes, s_nodes, win, np.random.default_rng(1), LD_ERROR["control"],
                             np.random.default_rng(2), np.random.default_rng(3))
    assert set(raw) == set(ref)
    for k in ref:
        if np.isnan(ref[k]):
            assert np.isnan(raw[k]), k
        else:
            assert raw[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12), k
    assert sum(ext["sfs_unf_I"]) == ts.num_sites
    assert sum(c for _, _, c in ext["jsfs_raw"]) == ts.num_sites
    s, c = (np.asarray(v) for v in ext["ld_mb_I"])
    assert s.shape == (1, 6) and c.sum() > 0


def _short_control(seed, L=300_000.0):
    old = F.FULL_CONTROL_BP
    F.FULL_CONTROL_BP = L
    try:
        return F.control_ts(seed)
    finally:
        F.FULL_CONTROL_BP = old


@pytest.mark.timeout(1200)
def test_simulate_full_short(monkeypatch):
    monkeypatch.setattr(F, "FULL_BODY_BP", 300_000.0)
    monkeypatch.setattr(F, "FULL_CONTROL_BP", 300_000.0)
    monkeypatch.setattr(F, "LONG_PRIOR", dict(c_long=(1e-11, 1e-6), l_long=(1e4, 1e5)))   # tracts < body
    bank = [F.control_entry(11), F.control_entry(12)]
    out = F.simulate_full(path(), 5, bank)
    for k in SV.STAT_KEYS:
        assert k in out
    assert out["n_nan"] == sum(np.isnan(out[k]) for k in SV.MODEL_KEYS)
    assert 1e-11 <= out["c_long"] <= 1e-6 and 1e4 <= out["l_long"] <= 1e5
    assert out["ctl_seed"] in (11, 12)
    x = out["x"]
    assert len(x["pi_mb_I"]) == 1 and len(x["edge_profile"]) == SV.N_WIN
    again = F.simulate_full(path(), 5, bank)
    for k in SV.STAT_KEYS:
        assert (np.isnan(out[k]) and np.isnan(again[k])) or out[k] == again[k], k
    # Same seed words as the 40-kb row: p_obs identical.
    assert out["p_obs"] == simulate_stats(path(), 5)["p_obs"]
