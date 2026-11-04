"""Task 8b. ABC-SMC (pyABC) parameter posteriors for the surviving trajectory models.

Prior. The Task 2 priors (``trajectory.draw_prior``) as independent pyABC marginals. Bounded parameters are fractions
(t_sel = f_sel t_inv, t_stop = f_stop t_sel) and logU parameters are uniform on log10, mapped back in ``to_params``.
The hard/soft class is the discrete parameter ``hard`` (P(hard) = PRIOR["p_hard"]) in a mixed run, and fixed in a
soft-only or hard-only run. A hard draw ignores p0 and f_sel (t_sel = t_inv).

Model. One trajectory per particle with ``max_attempts=1``, so a lost hard copy is a rejection and the establishment
probability enters through the acceptance. A lost, fixed or off-target path (|p_now - P_NOW| > P_TOL) returns
``SENTINEL``, whose distance is +inf. Otherwise the particle carries ``simulate_stats`` restricted to MODEL_KEYS,
plus ``p_obs``, ``p_now`` and ``n_nan``.

Distance. Adaptive weighted L2 over MODEL_KEYS with NaN-aware MAD scales (Prangle 2017, all particles of the last
generation, sentinels excluded). A NaN statistic gets weight 0 in that particle, and more than MAX_NAN NaNs give +inf.

Epsilon and stopping. Quantile epsilon (alpha 0.5) from an infinite start, so generation 0 is a prior sample of valid
paths. ``FiniteAcceptor`` never accepts +inf. The run stops at max_gen, when the epsilon acceptance among valid paths
(accepted / non-sentinel simulations) drops below min_acc, or when epsilon changes by less than min_eps_rel
(relative). The raw pyABC acceptance (accepted / all simulations) includes the p_now filter and the establishment
loss (about 1e-3 soft and 1e-7 hard at the prior), so it cannot carry the min_acc rule. It is reported in the trace,
and raw_min_acc is only a guard against a run that cannot progress.

Transitions. Multivariate normal on the continuous parameters and a discrete jump (stay probability 0.7) on ``hard``.

Usage (from the repo root)
    python -m illex.abc.smc --model D1 --flux 0 --cls soft --pop-size 200 --max-gen 6 --n-procs 16 --db PATH
    python -m illex.abc.smc --all-models --flux 0 --sampler redis ...
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path as FsPath

import numpy as np
import pandas as pd
import pyabc
from pyabc.acceptor import AcceptorResult, UniformAcceptor
from pyabc.distance import AdaptivePNormDistance
from pyabc.sumstat import IdentitySumstat

from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR
from illex.abc.statvec import MODEL_KEYS
from illex.abc.trajectory import KEYS, MODELS

log = logging.getLogger("illex.abc.smc")

SENTINEL = {"sentinel": 1.0}
MAX_NAN = 3                  # a particle with more NaN statistics than this has distance +inf
P_STAY = 0.7                 # discrete jump kernel on hard/soft
RESULTS = FsPath("results/illex")
OBSERVED = RESULTS / "abc_observed.json"
RF_POSTERIORS = RESULTS / "abc_posteriors.csv"


# ------------------------------------------------------------------ prior and parameter mapping

def _log_bounds(lo, hi):
    return math.log10(lo), math.log10(hi) - math.log10(lo)


def param_names(model: str, flux: bool, cls: str) -> tuple[list[str], list[str]]:
    """(continuous, discrete) pyABC parameter names of one model."""
    if model not in MODELS:
        raise ValueError(model)
    if cls not in ("mixed", "soft", "hard"):
        raise ValueError(cls)
    cont = ["t_inv", "log10_p0"]
    if model != "N":
        cont += ["f_sel", "log10_s"]
    if model == "D2":
        cont.append("f_stop")
    if model == "B":
        cont.append("p_star")
    if flux:
        cont.append("log10_gamma")
    return cont, (["hard"] if cls == "mixed" else [])


def make_prior(model: str, flux: bool, cls: str = "mixed") -> pyabc.Distribution:
    """The Task 2 prior of ``model`` as a product of independent pyABC marginals."""
    cont, disc = param_names(model, flux, cls)
    lo, hi = PRIOR["t_inv"]
    spec = dict(t_inv=pyabc.RV("uniform", lo, hi - lo),
                log10_p0=pyabc.RV("uniform", *_log_bounds(*PRIOR["p0"])),
                f_sel=pyabc.RV("uniform", 0.0, 1.0),
                log10_s=pyabc.RV("uniform", *_log_bounds(*PRIOR["s"])),
                f_stop=pyabc.RV("uniform", 0.0, 1.0),
                p_star=pyabc.RV("uniform", PRIOR["p_star"][0], PRIOR["p_star"][1] - PRIOR["p_star"][0]),
                log10_gamma=pyabc.RV("uniform", *_log_bounds(GAMMA_MIN, GAMMA_MAX)),
                hard=pyabc.RV("rv_discrete", values=([0, 1], [1.0 - PRIOR["p_hard"], PRIOR["p_hard"]])))
    return pyabc.Distribution(**{k: spec[k] for k in cont + disc})


def _pow10(x, lo, hi):
    return float(min(max(10.0 ** float(x), lo), hi))   # clip only the 1-ulp round trip of log10


def to_params(par, model: str, flux: bool, cls: str) -> dict:
    """pyABC parameters to the ``draw_prior`` row (scalars, unused keys NaN). Mirrors trajectory.draw_prior."""
    nan = math.nan
    hard = bool(round(float(par["hard"]))) if cls == "mixed" else cls == "hard"
    t_inv = float(par["t_inv"])
    out = dict(t_inv=t_inv, hard=hard, p0=nan if hard else _pow10(par["log10_p0"], *PRIOR["p0"]),
               t_sel=nan, s=nan, t_stop=nan, p_star=nan, gamma_eff=nan)
    if model != "N":
        out["t_sel"] = t_inv if hard else float(par["f_sel"]) * t_inv
        out["s"] = _pow10(par["log10_s"], *PRIOR["s"])
    if model == "D2":
        out["t_stop"] = float(par["f_stop"]) * out["t_sel"]
    if model == "B":
        out["p_star"] = float(par["p_star"])
    if flux:
        out["gamma_eff"] = _pow10(par["log10_gamma"], GAMMA_MIN, GAMMA_MAX)
    return out


# ------------------------------------------------------------------ model

class IllexModel:
    """pyABC model of one trajectory model. Picklable (multicore fork and redis cloudpickle)."""

    def __init__(self, model: str, flux: bool, cls: str, engine: str = "auto"):
        self.model, self.flux, self.cls, self.engine = model, bool(flux), cls, engine

    def __call__(self, par) -> dict:
        from illex.abc.simulate import simulate_stats
        from illex.abc.trajectory import simulate_paths

        p = to_params(par, self.model, self.flux, self.cls)
        arr = {k: np.array([p[k]], dtype=bool if k == "hard" else float) for k in KEYS}
        arr.update(model=self.model, flux=self.flux)
        # pyABC reseeds np.random in every worker, so the legacy stream seeds this particle's generator
        rng = np.random.default_rng(np.random.randint(1, 2**31 - 1))
        paths, _ = simulate_paths(arr, rng, max_attempts=1)
        if not paths:
            return dict(SENTINEL)
        path = paths[0]
        out = simulate_stats(path, int(rng.integers(1, 2**31 - 1)), engine=self.engine)
        res = {k: float(out[k]) for k in MODEL_KEYS}
        res.update(p_obs=float(out["p_obs"]), p_now=float(path.p_now), n_nan=float(out["n_nan"]))
        return res


# ------------------------------------------------------------------ distance, acceptor, ABCSMC

def nan_mad(*, samples: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG001
    """Median absolute deviation per statistic, ignoring NaN entries (pyabc.distance.mad with nanmedian)."""
    if samples.size == 0:
        return np.full(samples.shape[1] if samples.ndim > 1 else 1, np.nan)
    med = np.nanmedian(samples, axis=0)
    return np.nanmedian(np.abs(samples - med), axis=0)


class ModelKeySumstat(IdentitySumstat):
    """Fixed key order. A missing key (the sentinel) maps to NaN instead of a KeyError."""

    def __init__(self, keys):
        super().__init__()
        self.keys = list(keys)

    def initialize(self, t, get_sample, x_0, total_sims):
        super().initialize(t=t, get_sample=get_sample, x_0=x_0, total_sims=total_sims)
        self.x_keys = list(self.keys)

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            return data.flatten()
        return np.array([float(data.get(k, math.nan)) for k in self.keys])

    def get_ids(self):
        return list(self.keys)


def _is_sentinel(x) -> bool:
    return isinstance(x, dict) and x.get("sentinel", 0.0) == 1.0


class NanMadPNormDistance(AdaptivePNormDistance):
    """Adaptive L2 over ``keys``. NaN components weigh 0, more than ``max_nan`` NaNs or the sentinel give +inf.

    Scales are refitted every generation by ``nan_mad`` over all non-sentinel particles (accepted and rejected) with
    at most ``max_nan`` NaNs. A statistic whose scale is 0 or not finite gets weight 0. Generation 0 uses unit
    weights, which do not matter because its epsilon is +inf. ``gen_counts[t]`` records, for generation t, the
    simulations recorded by the sampler, the non-sentinel ones and the accepted ones.
    """

    def __init__(self, keys, max_nan: int = MAX_NAN, p: float = 2):
        super().__init__(p=p, scale_function=nan_mad, initial_scale_weights={k: 1.0 for k in keys},
                         all_particles_for_scale=True, sumstat=ModelKeySumstat(keys))
        self.max_nan = max_nan
        self.gen_counts: dict[int, dict] = {}

    def __call__(self, x, x_0, t=None, par=None) -> float:
        if _is_sentinel(x):
            return math.inf
        s, s0 = self.sumstat(x), self.sumstat(x_0)
        bad = ~np.isfinite(s)
        if bad.sum() > self.max_nan:
            return math.inf
        w = np.broadcast_to(self.get_weights(t=t), s.shape)
        d = np.where(bad, 0.0, np.abs(w * np.where(bad, 0.0, s - s0)))
        if self.p == np.inf:
            return float(d.max())
        return float((d ** self.p).sum() ** (1.0 / self.p))

    def fit_scales(self, t, sample) -> None:
        parts = sample.all_particles
        keep = [q for q in parts if not _is_sentinel(q.sum_stat)]
        self.gen_counts[t - 1] = dict(n_recorded=len(parts), n_valid=len(keep),
                                      n_accepted=len(sample.accepted_particles))
        ss = np.array([self.sumstat(q.sum_stat) for q in keep]).reshape(-1, len(self.sumstat.keys))
        ss = ss[np.isnan(ss).sum(axis=1) <= self.max_nan]
        scales = nan_mad(samples=ss)
        ok = np.isfinite(scales) & ~np.isclose(scales, 0.0)
        w = np.zeros_like(scales)
        w[ok] = 1.0 / scales[ok]
        self.scale_weights[t] = w


class FiniteAcceptor(UniformAcceptor):
    """Uniform acceptance d <= eps(t) that never accepts d = +inf (inf <= inf would pass at eps = +inf)."""

    def __call__(self, distance_function, eps, x, x_0, t, par):
        d = distance_function(x, x_0, t, par)
        return AcceptorResult(distance=d, accept=bool(np.isfinite(d) and d <= eps(t)))


class StopRuleABCSMC(pyabc.ABCSMC):
    """ABCSMC with the Task 8b stopping rules (valid-path epsilon acceptance and relative epsilon change)."""

    def run_stop_rules(self, max_gen: int, min_acc: float, min_eps_rel: float, raw_min_acc: float = 0.0,
                       max_walltime_h: float | None = None):
        self._min_acc_valid = float(min_acc)
        self._min_eps_rel = float(min_eps_rel)
        self.stop_reason = None
        wt = timedelta(hours=max_walltime_h) if max_walltime_h else None
        h = self.run(max_nr_populations=max_gen, min_acceptance_rate=raw_min_acc, max_walltime=wt)
        if self.stop_reason is None:
            self.stop_reason = "pyabc (max_gen, raw acceptance guard, walltime, or a failed sample)"
        return h

    def check_terminate(self, t, acceptance_rate) -> bool:
        if super().check_terminate(t=t, acceptance_rate=acceptance_rate):
            self.stop_reason = "max_gen" if t >= self.max_t else "pyabc criterion"
            return True
        c = getattr(self.distance_function, "gen_counts", {}).get(t)
        if c and c["n_valid"] > 0 and c["n_accepted"] / c["n_valid"] < self._min_acc_valid:
            self.stop_reason = f"valid-path acceptance {c['n_accepted'] / c['n_valid']:.3g} < {self._min_acc_valid}"
            return True
        if t >= 1:
            e, e0 = self.eps(t), self.eps(t - 1)
            if np.isfinite(e) and np.isfinite(e0) and e0 > 0 and abs(e - e0) / e0 < self._min_eps_rel:
                self.stop_reason = f"relative epsilon change {abs(e - e0) / e0:.3g} < {self._min_eps_rel}"
                return True
        return False


def make_transition(cont, disc):
    mvn = pyabc.MultivariateNormalTransition()
    if not disc:
        return mvn
    mapping = {tuple(cont): mvn}
    for k in disc:
        mapping[k] = pyabc.DiscreteJumpTransition(domain=np.array([0, 1]), p_stay=P_STAY)
    return pyabc.AggregatedTransition(mapping=mapping)


def build_abc(models, priors, keys, pop_size, sampler, continuous, discrete=(), names=None) -> StopRuleABCSMC:
    """ABCSMC with the Task 8b distance, epsilon, acceptor and transitions.

    continuous and discrete hold one tuple of parameter names per model (discrete may be empty for all models).
    """
    discrete = list(discrete) or [()] * len(models)
    names = names or [f"m{i}" for i in range(len(models))]
    mods = [pyabc.model.FunctionModel(f, name=n) for f, n in zip(models, names)]
    return StopRuleABCSMC(
        models=mods, parameter_priors=list(priors), distance_function=NanMadPNormDistance(keys),
        population_size=pop_size, transitions=[make_transition(c, d) for c, d in zip(continuous, discrete)],
        eps=pyabc.QuantileEpsilon(initial_epsilon=np.inf, alpha=0.5), sampler=sampler, acceptor=FiniteAcceptor())


# ------------------------------------------------------------------ outputs

def _wq(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x)
    x, w = x[m], w[m]
    if x.size == 0 or w.sum() <= 0:
        return [math.nan] * len(q)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return list(np.interp(q, cw, x))


def _ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size else 0.0


def _natural(df, model, flux, cls):
    rows = [to_params(r, model, flux, cls) for r in df.to_dict("records")]
    return pd.DataFrame(rows, columns=list(KEYS))


def _upsert(path: FsPath, new: pd.DataFrame, on: list[str]):
    if path.exists():
        old = pd.read_csv(path, comment="#")
        key_new = set(map(tuple, new[on].astype(str).values))
        old = old[[tuple(r) not in key_new for r in old[on].astype(str).values]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


PAR_SCALE = dict(t_inv="linear", s="log10", p0="log10", t_sel="linear", t_stop="linear", p_star="linear",
                 gamma_eff="log10")


def write_outputs(h, models, flux, cls, out_dir, abc=None, n_post=1000, seed=1, rf_path=RF_POSTERIORS):
    """Posterior draws, summaries, the per-generation trace, model choice and the ABC-RF comparison."""
    out_dir = FsPath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fl = int(bool(flux))
    tag = models[0] if len(models) == 1 else "all"
    rng = np.random.default_rng(seed)
    pops = h.get_all_populations()
    pops = pops[pops.t >= 0].set_index("t")
    counts = getattr(abc.distance_function, "gen_counts", {}) if abc is not None else {}
    trace = []
    for t in range(h.max_t + 1):
        probs = h.get_model_probabilities(t).iloc[0] if len(models) > 1 else None
        row = dict(t=t, eps=float(pops.loc[t, "epsilon"]), samples=int(pops.loc[t, "samples"]),
                   accepted=int(pops.loc[t, "particles"]))
        row["acc_raw"] = row["accepted"] / row["samples"]
        c = counts.get(t, {})
        row["n_valid"] = c.get("n_valid", math.nan)
        row["n_recorded"] = c.get("n_recorded", math.nan)
        row["acc_valid"] = row["accepted"] / c["n_valid"] if c.get("n_valid") else math.nan
        for mi, m in enumerate(models):
            if probs is not None:
                row[f"prob_{m}"] = float(probs.get(mi, 0.0))
            try:
                df, w = h.get_distribution(m=mi, t=t)
            except Exception:  # noqa: BLE001  the model died in this generation
                continue
            if len(df) == 0:
                continue
            row[f"ess_{m}"] = _ess(w)
            nat = _natural(df, m, flux, cls)
            row[f"hard_share_{m}"] = float(np.sum(w * nat["hard"].astype(float).values) / np.sum(w))
            for k in ("p0", "t_inv", "s", "t_sel"):
                row[f"median_{k}_{m}"] = _wq(nat[k], w, [0.5])[0]
        trace.append(row)
    pd.DataFrame(trace).to_csv(out_dir / f"abc_smc_trace_{tag}_{fl}.csv", index=False)

    summ = []
    for mi, m in enumerate(models):
        try:
            df, w = h.get_distribution(m=mi, t=h.max_t)
        except Exception:  # noqa: BLE001
            continue
        if len(df) == 0:
            continue
        w = np.asarray(w, float) / np.sum(w)
        nat = _natural(df, m, flux, cls)
        idx = rng.choice(len(nat), size=n_post, replace=True, p=w)
        post = nat.iloc[idx].reset_index(drop=True)
        post.insert(0, "cls", np.where(post["hard"], "hard", "soft"))
        post.insert(0, "flux", fl)
        post.insert(0, "model", m)
        post["src"] = idx
        post.to_csv(out_dir / f"abc_smc_post_{m}_{fl}.csv", index=False)
        hard = nat["hard"].astype(bool).values
        for k in ("t_inv", "s", "p0", "t_sel", "t_stop", "p_star", "gamma_eff"):
            x = nat[k].values.astype(float)
            sel = ~hard if k in ("p0", "t_sel") else np.ones(len(x), bool)
            if not np.isfinite(x[sel]).any():
                continue
            ww = w[sel]
            q = _wq(x[sel], ww, [0.025, 0.25, 0.5, 0.75, 0.975])
            summ.append(dict(model=m, flux=fl, cls_run=cls, parameter=k, scale=PAR_SCALE[k],
                             rows="soft" if k in ("p0", "t_sel") else "all", n=int(sel.sum()), ess=_ess(ww),
                             median=q[2], q025=q[0], q25=q[1], q75=q[3], q975=q[4], n_gen=h.n_populations,
                             eps_final=float(pops.loc[h.max_t, "epsilon"]),
                             post_hard_mass=float(np.sum(w[hard])),
                             stop_reason=getattr(abc, "stop_reason", None)))
    summ = pd.DataFrame(summ)
    if len(summ):
        _upsert(out_dir / "abc_smc_posteriors.csv", summ, ["model", "flux", "cls_run"])

    if len(models) > 1:
        probs = h.get_model_probabilities(h.max_t).iloc[0]
        mc = pd.DataFrame([dict(flux=fl, cls_run=cls, model=m, prob=float(probs.get(mi, 0.0)), t=h.max_t)
                           for mi, m in enumerate(models)])
        _upsert(out_dir / "abc_smc_modelchoice.csv", mc, ["flux", "cls_run"])

    if len(summ) and FsPath(rf_path).exists():
        rf = pd.read_csv(rf_path, comment="#")
        rf = rf[rf["rows"].isin(["all", "soft"])]
        cmp = summ.merge(rf[["model", "flux", "parameter", "median", "q025", "q975"]].rename(
            columns=dict(median="rf_median", q025="rf_q025", q975="rf_q975")), on=["model", "flux", "parameter"],
            how="left")
        cmp = cmp.rename(columns=dict(median="smc_median", q025="smc_q025", q975="smc_q975"))
        lg = cmp["scale"] == "log10"
        cmp["median_diff"] = np.where(lg, np.log10(cmp["smc_median"]) - np.log10(cmp["rf_median"]),
                                      cmp["smc_median"] - cmp["rf_median"])
        cols = ["model", "flux", "cls_run", "parameter", "scale", "rf_median", "rf_q025", "rf_q975", "smc_median",
                "smc_q025", "smc_q975", "median_diff", "ess", "n_gen"]
        _upsert(out_dir / "abc_rf_vs_smc.csv", cmp[cols], ["model", "flux", "cls_run"])
    return summ


# ------------------------------------------------------------------ driver

def observed(path=OBSERVED) -> dict:
    with open(path) as f:
        st = json.load(f)["stats"]
    return {k: float(st[k]) for k in MODEL_KEYS}


def make_sampler(kind: str, n_procs: int = 4, redis_batch: int = 10):
    if kind == "multicore":
        return pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs)
    if kind == "redis":
        return pyabc.sampler.RedisEvalParallelSampler(
            host=os.environ.get("REDIS_HOST", "localhost"), port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None, batch_size=redis_batch)
    if kind == "single":
        return pyabc.sampler.SingleCoreSampler()
    raise ValueError(kind)


def run(models, flux, cls="mixed", pop_size=1000, max_gen=15, min_acc=0.01, min_eps_rel=0.05, sampler="multicore",
        n_procs=4, redis_batch=10, db=None, out_dir=RESULTS, write=True, n_post=1000,
        engine="auto", raw_min_acc=0.0, max_walltime_h=None, overwrite=False, seed=None, obs_path=OBSERVED):
    """Run ABC-SMC for ``models`` (one model, or model choice over several) and write the outputs."""
    flux = bool(flux)
    fl = int(flux)
    tag = models[0] if len(models) == 1 else "all"
    db = FsPath(db or FsPath(out_dir) / f"abc_smc_{tag}_{fl}.db")
    if db.exists():
        if not overwrite:
            raise FileExistsError(f"{db} exists (pass overwrite=True / --overwrite)")
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
    names = [param_names(m, flux, cls) for m in models]
    abc = build_abc([IllexModel(m, flux, cls, engine) for m in models], [make_prior(m, flux, cls) for m in models],
                    keys=MODEL_KEYS, pop_size=pop_size, sampler=make_sampler(sampler, n_procs, redis_batch),
                    continuous=[tuple(c) for c, _ in names], discrete=[tuple(d) for _, d in names], names=models)
    abc.new("sqlite:///" + str(db.resolve()), observed(obs_path))
    h = abc.run_stop_rules(max_gen=max_gen, min_acc=min_acc, min_eps_rel=min_eps_rel, raw_min_acc=raw_min_acc,
                           max_walltime_h=max_walltime_h)
    log.info("stop: %s", abc.stop_reason)
    if write and h.n_populations > 0:
        write_outputs(h, models, flux, cls, out_dir, abc=abc, n_post=n_post)
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--all-models", action="store_true", help="model choice over --models (default N D1 D2 B)")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--flux", type=int, choices=(0, 1), required=True)
    ap.add_argument("--cls", choices=("mixed", "soft", "hard"), default="mixed")
    ap.add_argument("--pop-size", type=int, default=1000)
    ap.add_argument("--max-gen", type=int, default=15)
    ap.add_argument("--min-acc", type=float, default=0.01, help="stop below this valid-path epsilon acceptance")
    ap.add_argument("--min-eps-rel", type=float, default=0.05)
    ap.add_argument("--raw-min-acc", type=float, default=0.0, help="pyABC raw acceptance guard (0 = off)")
    ap.add_argument("--max-walltime-h", type=float, default=None)
    ap.add_argument("--sampler", choices=("multicore", "redis", "single"), default="multicore")
    ap.add_argument("--n-procs", type=int, default=16)
    ap.add_argument("--redis-batch", type=int, default=10)
    ap.add_argument("--engine", choices=("auto", "msprime", "msinv"), default="auto")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--n-post", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not a.all_models and not a.model:
        ap.error("--model or --all-models is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    models = a.models if a.all_models else [a.model]
    h = run(models, bool(a.flux), cls=a.cls, pop_size=a.pop_size, max_gen=a.max_gen, min_acc=a.min_acc,
            min_eps_rel=a.min_eps_rel, sampler=a.sampler, n_procs=a.n_procs, redis_batch=a.redis_batch, db=a.db,
            out_dir=a.out_dir, n_post=a.n_post, engine=a.engine, raw_min_acc=a.raw_min_acc,
            max_walltime_h=a.max_walltime_h, overwrite=a.overwrite, seed=a.seed)
    print(f"done: {h.n_populations} generations")


if __name__ == "__main__":
    main()
