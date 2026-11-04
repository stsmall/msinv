"""Task 8b: ABC-SMC (pyABC) machinery. Toy posterior, the +inf sentinel, the prior fraction mapping, an Illex smoke run."""
import math

import numpy as np
import pytest

pyabc = pytest.importorskip("pyabc")

from illex.abc import smc  # noqa: E402
from illex.abc.constants import PRIOR  # noqa: E402


class _Toy:
    """y ~ N(theta, 1). Returns the sentinel when theta < cut."""

    def __init__(self, cut=-math.inf):
        self.cut = cut

    def __call__(self, par):
        if par["theta"] < self.cut:
            return dict(smc.SENTINEL)
        return {"y": float(par["theta"] + np.random.standard_normal()), "n_nan": 0}


def _toy_run(tmp_path, model, pop=1000, gens=10):
    prior = pyabc.Distribution(theta=pyabc.RV("uniform", -5.0, 10.0))
    abc = smc.build_abc([model], [prior], keys=("y",), pop_size=pop, sampler=pyabc.sampler.SingleCoreSampler(),
                        continuous=[("theta",)], discrete=())
    abc.new("sqlite:///" + str(tmp_path / "toy.db"), {"y": 1.0})
    np.random.seed(7)
    return abc.run_stop_rules(max_gen=gens, min_acc=0.01, min_eps_rel=0.05)


def test_toy_gaussian_posterior(tmp_path):
    h = _toy_run(tmp_path, _Toy())
    df, w = h.get_distribution(m=0, t=h.max_t)
    mean = np.average(df["theta"], weights=w)
    sd = math.sqrt(np.average((df["theta"] - mean) ** 2, weights=w))
    assert abs(mean - 1.0) < 0.15, (mean, h.max_t)
    assert abs(sd - 1.0) < 0.25, (sd, h.max_t)


def test_inf_sentinel(tmp_path):
    h = _toy_run(tmp_path, _Toy(cut=0.0), pop=200, gens=4)
    assert h.max_t >= 1
    for t in range(h.max_t + 1):
        df, w = h.get_distribution(m=0, t=t)
        assert (df["theta"] >= 0.0).all(), t
    dist = smc.NanMadPNormDistance(keys=("y",))
    dist.initialize(t=0, get_sample=None, x_0={"y": 1.0}, total_sims=0)
    assert dist(dict(smc.SENTINEL), {"y": 1.0}, t=0) == math.inf
    acc = smc.FiniteAcceptor()
    eps = pyabc.epsilon.ConstantEpsilon(math.inf)
    assert not acc(dist, eps, dict(smc.SENTINEL), {"y": 1.0}, 0, None).accept


def test_nan_handling():
    keys = ("a", "b", "c", "d", "e")
    dist = smc.NanMadPNormDistance(keys=keys)
    x0 = {k: 0.0 for k in keys}
    dist.initialize(t=0, get_sample=None, x_0=x0, total_sims=0)
    x = {k: 1.0 for k in keys}
    full = dist(x, x0, t=0)
    x["a"] = math.nan
    assert dist(dict(x, n_nan=1), x0, t=0) == pytest.approx(math.sqrt(4.0))
    assert full == pytest.approx(math.sqrt(5.0))
    for k in ("b", "c", "d"):
        x[k] = math.nan
    assert dist(dict(x, n_nan=4), x0, t=0) == math.inf


@pytest.mark.parametrize("model", ["N", "D1", "D2", "B"])
@pytest.mark.parametrize("cls", ["mixed", "soft", "hard"])
def test_prior_fraction_mapping(model, cls):
    prior = smc.make_prior(model, flux=True, cls=cls)
    np.random.seed(3)
    draws = [prior.rvs() for _ in range(10_000)]
    hard_n = 0
    for par in draws:
        p = smc.to_params(par, model, flux=True, cls=cls)
        assert PRIOR["t_inv"][0] <= p["t_inv"] <= PRIOR["t_inv"][1]
        assert smc.GAMMA_MIN <= p["gamma_eff"] <= smc.GAMMA_MAX
        hard_n += p["hard"]
        if p["hard"]:
            assert math.isnan(p["p0"])
        else:
            assert PRIOR["p0"][0] <= p["p0"] <= PRIOR["p0"][1]
        if model == "N":
            assert math.isnan(p["t_sel"]) and math.isnan(p["s"])
            continue
        if p["hard"]:
            assert p["t_sel"] == p["t_inv"]
        else:
            assert p["t_sel"] < p["t_inv"]
        assert PRIOR["s"][0] <= p["s"] <= PRIOR["s"][1]
        if model == "D2":
            assert p["t_stop"] < p["t_sel"]
        else:
            assert math.isnan(p["t_stop"])
        if model == "B":
            assert PRIOR["p_star"][0] <= p["p_star"] <= PRIOR["p_star"][1]
    frac = hard_n / len(draws)
    want = {"mixed": PRIOR["p_hard"], "soft": 0.0, "hard": 1.0}[cls]
    assert abs(frac - want) < 0.02


@pytest.mark.slow
@pytest.mark.timeout(5400)
def test_smoke_illex(tmp_path):
    db = tmp_path / "smoke.db"
    h = smc.run(models=["D1"], flux=False, cls="soft", pop_size=8, max_gen=2, min_acc=0.01,
                sampler="multicore", n_procs=4, db=str(db), out_dir=str(tmp_path), write=True, n_post=50)
    assert h.max_t == 1
    assert h.n_populations == 2
    assert (tmp_path / "abc_smc_post_D1_0.csv").exists()
    assert (tmp_path / "abc_smc_trace_D1_0.csv").exists()
