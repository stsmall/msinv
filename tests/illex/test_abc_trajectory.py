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
    assert isinstance(c["soft"], Counts) and isinstance(paths[0], Path)
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
    frac = 1.0 - c["hard"].lost_establishing / n
    assert abs(frac - 1 / 500) < 0.3 / 500
    assert c["hard"].est_failed == 0


def test_hard_acceptance_includes_establishment():
    # default max_attempts=1: a lost single copy is a rejection, so established fraction = establishment probability
    n = 100_000
    _, c = simulate_paths(prm("N", n, 5000.0, hard=True), np.random.default_rng(11), n_e_fn=const_ne(1e4),
                          apply_filter=False)
    assert abs(1 - c["hard"].lost_establishing / n - 1 / 500) < 0.3 / 500
    s = 1e-2
    p = prm("D1", n, 5000.0, hard=True, t_sel=5000.0, s=s)
    _, c = simulate_paths(p, np.random.default_rng(12), n_e_fn=const_ne(1e4), apply_filter=False)
    assert abs(1 - c["hard"].lost_establishing / n - (1 - np.exp(-2 * s))) < 0.3 * (1 - np.exp(-2 * s))


def test_counts_split_by_class():
    rng = np.random.default_rng(13)
    p = draw_prior("N", False, rng, 400)
    p["t_inv"] = np.full(400, 5000.0)
    _, c = simulate_paths(p, rng, n_e_fn=const_ne(1e4), apply_filter=False)
    assert c["hard"].drawn + c["soft"].drawn == 400
    assert c["hard"].drawn == p["hard"].sum() and 0 < c["hard"].drawn < 400
    assert draw_prior("D1", False, rng, 50, cls="hard")["hard"].all()
    assert not draw_prior("D1", False, rng, 50, cls="soft")["hard"].any()


@pytest.mark.timeout(10)
def test_attempt_cap_records_failure():
    p = prm("D1", 50, 5000.0, hard=True, t_sel=5000.0, s=0.0)
    _, c = simulate_paths(p, np.random.default_rng(7), max_attempts=10, n_e_fn=const_ne(1e4))
    assert c["hard"].est_failed > 0 and c["hard"].lost_establishing == 0


def test_no_clipped_endpoints():
    p = draw_prior("D1", False, np.random.default_rng(8), 300)
    p["t_inv"] = np.full(300, 3e4)
    p["t_sel"] = np.minimum(p["t_sel"], 3e4)
    p["s"] = np.full(300, 1e-4)
    paths, c = simulate_paths(p, np.random.default_rng(9), n_e_fn=const_ne(1e4), apply_filter=False)
    assert c["soft"].fixed > 0 and len(paths) > 0
    for q in paths:
        assert (q.freqs > 0).all() and (q.freqs < 1).all()
        assert q.times[0] == 0 and (np.diff(q.times) > 0).all()


def test_filter():
    p = prm("N", 3000, 2e4, p0=0.626)
    paths, c = simulate_paths(p, np.random.default_rng(10), n_e_fn=const_ne(1e5))
    assert c["soft"].accepted == len(paths) > 0
    assert all(abs(q.p_now - 0.626) <= 0.02 for q in paths)


# ---- exact importance sampling on establishment (Task 6 fix round 1) -------------------------------------------

def _const_ne(n):
    return lambda t: np.full_like(np.asarray(t, dtype=float), float(n))


@pytest.mark.parametrize("s", [0.0, 1e-4, 1e-3])
def test_establishment_prob_matches_wf(s):
    """u(theta) (Kimura) vs the fraction of single-copy WF runs reaching ESTABLISHED, same _establish code path."""
    from illex.abc.trajectory import ESTABLISHED, _establish, establishment_prob
    n, m = 10_000, 400_000
    t_inv = np.full(m, 1e5)
    prm = dict(t_inv=t_inv, s=np.full(m, s), t_sel=t_inv.copy(), t_stop=np.full(m, np.nan),
               p_star=np.full(m, np.nan))
    g, k, two_n, att = _establish("D1", prm, np.arange(m), np.random.default_rng(5), _const_ne(n), 1)
    assert att.sum() == m and (k[g >= 0] >= ESTABLISHED).all()
    u = float(establishment_prob("D1", s, 1e5, _const_ne(n))[0])
    frac = float((g >= 0).mean())
    se = np.sqrt(u * (1 - u) / m)
    assert abs(frac - u) < 3 * se, (s, frac, u, se)


def test_conditional_attempts_and_weights():
    from illex.abc.trajectory import draw_prior, simulate_paths
    rng = np.random.default_rng(3)
    prm = draw_prior("D1", False, rng, 300, "hard")
    paths, cnt = simulate_paths(prm, np.random.default_rng(4), condition_on_establishment=True)
    h = cnt["hard"]
    assert h.drawn == 300 and h.lost_establishing == 0 and h.attempts >= 300
    assert all(0 < q.params["w_est"] <= 1 for q in paths)
    assert h.w_acc == pytest.approx(sum(q.params["w_est"] for q in paths))


def test_weighted_acceptance_matches_bruteforce(monkeypatch):
    """Hard D1 at a smaller constant N (2e4): (1/n) sum w_est * accepted from the conditional run == brute-force rate."""
    from illex.abc import trajectory as T
    monkeypatch.setitem(T.PRIOR, "t_inv", (5e3, 2e4))
    monkeypatch.setitem(T.PRIOR, "s", (2e-4, 6e-4))
    ne = _const_ne(20_000)
    nb, nc = 2_400_000, 20_000
    prm = T.draw_prior("D1", False, np.random.default_rng(1), nb, "hard")
    _, c1 = T.simulate_paths(prm, np.random.default_rng(2), n_e_fn=ne)
    p1 = c1["hard"].accepted / nb
    se1 = np.sqrt(p1 * (1 - p1) / nb)
    prm = T.draw_prior("D1", False, np.random.default_rng(11), nc, "hard")
    _, c2 = T.simulate_paths(prm, np.random.default_rng(12), n_e_fn=ne, condition_on_establishment=True)
    h = c2["hard"]
    p2 = h.w_acc / nc
    se2 = np.sqrt((h.w_acc_sq / nc - p2 ** 2) / nc)
    assert c1["hard"].accepted > 40 and h.accepted > 100
    assert abs(p1 - p2) < 3 * np.hypot(se1, se2), (p1, p2, se1, se2)


def test_establishment_prob_exact_reviewer_values():
    """Exact first-passage u at 2N = 1.1e6 (independent linear solve by the reviewer), to 4 significant figures."""
    from illex.abc.establishment import direct_u, u_grid
    exact = {0.0: 0.0019511, 1e-4: 0.0020526, 1e-3: 0.0031155, 1e-2: 0.0197371}
    for s, v in exact.items():
        assert direct_u(1_100_000, s) == pytest.approx(v, rel=5e-5)
        assert float(u_grid([s], [1_100_000])[0]) == pytest.approx(v, rel=1e-4)


def test_establishment_grid_matches_direct_offgrid():
    """Grid interpolation vs a direct linear solve at off-grid s, 2N and (for B) p*: within 1e-4 relative."""
    from illex.abc.establishment import direct_u, establishment_prob
    from illex.theory import N_growth
    rng = np.random.default_rng(0)
    for t in (4e5, 6.1e5, 1.2e6):
        two_n = int(np.floor(2 * float(N_growth(t))))
        for s in np.exp(rng.uniform(np.log(1.1e-6), np.log(9e-3), 3)).tolist() + [3e-7, 0.0]:
            d = establishment_prob("D1", s, t)[0]
            assert d == pytest.approx(direct_u(two_n, s), rel=1e-4), (t, s)
            for ps in (0.31, 0.5, 0.77, 0.94):
                b = establishment_prob("B", s, t, None, ps)[0]
                assert b == pytest.approx(direct_u(two_n, s, ps), rel=1e-4), (t, s, ps)
    assert establishment_prob("N", 0.0, 7e5)[0] == pytest.approx(direct_u(int(2 * float(N_growth(7e5))), 0.0), rel=1e-4)


def test_hard_draws_have_t_sel_equal_t_inv():
    from illex.abc.trajectory import draw_prior
    for m in ("D1", "D2", "B"):
        p = draw_prior(m, False, np.random.default_rng(1), 200, "hard")
        assert np.array_equal(p["t_sel"], p["t_inv"])
