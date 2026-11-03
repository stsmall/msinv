import numpy as np
import pytest

from illex.abc.trajectory import Counts, Path, draw_prior, simulate_paths

NAN = float("nan")


def const_ne(n):
    return lambda t: np.full_like(np.asarray(t, dtype=float), float(n))


def prm(model, n, t_inv, hard=False, p0=NAN, t_sel=NAN, s=NAN, t_stop=NAN, p_star=NAN):
    f = lambda x: np.full(n, x, dtype=float)
    return dict(model=model, flux=False, t_inv=f(t_inv), hard=np.full(n, hard), p0=f(p0), t_sel=f(t_sel),
                s=f(s), t_stop=f(t_stop), p_star=f(p_star), gamma_eff=f(NAN))


def finals(paths):
    return np.array([p.p_now for p in paths])


def test_draw_prior_shapes_and_nans():
    rng = np.random.default_rng(1)
    for model in ("N", "D1", "D2", "B"):
        d = draw_prior(model, True, rng, 500)
        assert all(len(d[k]) == 500 for k in ("t_inv", "hard", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff"))
        assert np.isnan(d["p0"][d["hard"]]).all() and not np.isnan(d["p0"][~d["hard"]]).any()
        assert np.isnan(d["s"]).all() == (model == "N")
        assert np.isnan(d["t_stop"]).all() == (model != "D2")
        assert np.isnan(d["p_star"]).all() == (model != "B")
        assert not np.isnan(d["gamma_eff"]).any()
        if model != "N":
            assert (d["t_sel"][d["hard"]] == d["t_inv"][d["hard"]]).all()
            assert (d["t_sel"] <= d["t_inv"]).all()
    assert np.isnan(draw_prior("D1", False, rng, 5)["gamma_eff"]).all()


def test_neutral_soft_mean_preserved():
    p = prm("N", 2000, 1e4, p0=0.3)
    paths, c = simulate_paths(p, np.random.default_rng(2), n_e_fn=const_ne(1e5), apply_filter=False)
    assert isinstance(c, Counts) and isinstance(paths[0], Path)
    assert abs(finals(paths).mean() - 0.3) < 0.01


def test_D1_rises():
    # 2e4 generations at N s = 100 would fix every path (rejected); 8e3 stops short of fixation
    p = prm("D1", 300, 8e3, p0=0.05, t_sel=8e3, s=1e-3)
    paths, _ = simulate_paths(p, np.random.default_rng(3), n_e_fn=const_ne(1e5), apply_filter=False)
    assert np.median(finals(paths)) > 0.9


def test_B_settles_at_pstar():
    p = prm("B", 300, 2e4, p0=0.05, t_sel=2e4, s=1e-3, p_star=0.5)
    paths, _ = simulate_paths(p, np.random.default_rng(4), n_e_fn=const_ne(1e5), apply_filter=False)
    assert abs(np.median(finals(paths)) - 0.5) < 0.03


def test_D2_freezes_after_tstop():
    T, ts = 200.0, 100.0

    def z(model, **kw):
        paths, _ = simulate_paths(prm(model, 1000, T, p0=0.3, **kw), np.random.default_rng(5), dt=10.0,
                                  rec_every=100.0, n_e_fn=const_ne(2000), apply_filter=False)
        out = []
        for p in paths:
            assert p.times[1] == ts and p.times[2] == T
            out.append((p.freqs[0] - p.freqs[1]) / np.sqrt(p.freqs[1] * (1 - p.freqs[1])))
        return np.array(out)

    zd = z("D2", t_sel=T, s=1e-2, t_stop=ts)   # selection for the first 100 forward generations only
    zn = z("N")
    assert 0.5 < zd.var() / zn.var() < 2.0
    assert abs(zd.mean()) < 0.03


def test_hard_establishment_prob():
    n = 100_000
    p = prm("N", n, 5000.0, hard=True)
    _, c = simulate_paths(p, np.random.default_rng(6), max_attempts=1, n_e_fn=const_ne(1e4), apply_filter=False)
    frac = 1.0 - c.est_failed / n
    assert abs(frac - 1 / 500) < 0.3 / 500


@pytest.mark.timeout(10)
def test_attempt_cap_records_failure():
    p = prm("D1", 50, 5000.0, hard=True, t_sel=5000.0, s=0.0)
    _, c = simulate_paths(p, np.random.default_rng(7), max_attempts=10, n_e_fn=const_ne(1e4))
    assert c.est_failed > 0


def test_no_clipped_endpoints():
    p = draw_prior("D1", False, np.random.default_rng(8), 300)
    p["t_inv"] = np.full(300, 3e4)
    p["t_sel"] = np.minimum(p["t_sel"], 3e4)
    p["s"] = np.full(300, 1e-4)
    paths, c = simulate_paths(p, np.random.default_rng(9), n_e_fn=const_ne(1e4), apply_filter=False)
    assert c.fixed > 0 and len(paths) > 0
    for q in paths:
        assert (q.freqs > 0).all() and (q.freqs < 1).all()
        assert q.times[0] == 0 and (np.diff(q.times) > 0).all()


def test_filter():
    p = prm("N", 3000, 2e4, p0=0.626)
    paths, c = simulate_paths(p, np.random.default_rng(10), n_e_fn=const_ne(1e5))
    assert c.accepted == len(paths) > 0
    assert all(abs(q.p_now - 0.626) <= 0.02 for q in paths)
