"""Refit the neutral baseline (exponential growth N_anc -> N0 over T_grow, recombination rate r; MU fixed) to the
collinear chr2 10-30 Mb statistics, by ABC rejection with local-linear regression adjustment.

Observed (--observed, default results/illex/abc_observed_ctl2.json from build_observed_ctl2.py): meta["control"],
the representative collinear chr2 control 40-55 + 85-115 Mb (SFS shape, LD by distance, Tajima's D; the I and S
labels are random sample sets in a collinear region and are averaged), plus meta["control_pi"]["mean"], the collinear
pi per site of the same regions from the within-class ANGSD spectra (0.0062). The draws apply that control's LD
genotype error model (meta["ld_error_model"]["control"]).

Each draw simulates SIM_BP of sequence with the ABC control's sampling (N_STD + N_INV haploid samples, ploidy 2,
binary mutations at MU, the control genotype error model on the LD diploids) and computes statvec.region_stats.

    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python baseline_refit.py sim --n 20000 --procs 24 --out draws.csv
    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python baseline_refit.py fit --draws draws.csv --out fit.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

SIM_BP = 200_000.0
OBS_DEFAULT = "results/illex/abc_observed_ctl2.json"
PRIOR = dict(n_anc=(1e5, 3e6), fold=(1.0, 200.0), t_grow=(1e4, 3e6), r=(1e-10, 1e-8))   # all log-uniform
BINS = ["1", "2", "3", "4", "5", "6p"]
FEATS = [f"sfs_{b}" for b in BINS] + [f"ld_{k}" for k in range(1, 7)] + ["tajd", "log10_pi"]


def control_ts(n_anc, n0, t_grow, r, seed, length=SIM_BP):
    import msprime
    from illex.abc.constants import N_INV, N_STD
    alpha = np.log(n0 / n_anc) / t_grow
    d = msprime.Demography()
    d.add_population(name="A", initial_size=float(n0), growth_rate=float(alpha))
    d.add_population_parameters_change(time=float(t_grow), initial_size=float(n_anc), growth_rate=0.0, population="A")
    return msprime.sim_ancestry(samples=[msprime.SampleSet(N_STD + N_INV, population="A", ploidy=1)],
                                demography=d, sequence_length=length, recombination_rate=float(r),
                                ploidy=2, model="hudson", random_seed=int(seed))


def features(raw: dict, pi: float) -> dict:
    out = {f"sfs_{b}": 0.5 * (raw[f"sfs_I_{b}"] + raw[f"sfs_S_{b}"]) for b in BINS}
    out.update({f"ld_{k}": 0.5 * (raw[f"ld_I_{k}"] + raw[f"ld_S_{k}"]) for k in range(1, 7)})
    out["tajd"] = 0.5 * (raw["tajd_I"] + raw["tajd_S"])
    out["log10_pi"] = float(np.log10(pi)) if pi > 0 else np.nan
    return out


def one(task):
    n_anc, n0, t_grow, r, seed, length = task
    from illex.abc import statvec as SV
    from illex.abc.constants import N_INV, N_STD
    from illex.abc.error_model import LD_ERROR
    from illex.abc.simulate import _mutate
    ss = np.random.SeedSequence(seed).generate_state(4)
    ts = _mutate(control_ts(n_anc, n0, t_grow, r, ss[0], length), int(ss[1]))
    smp = ts.samples()
    raw = SV.region_stats(ts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, length), np.random.default_rng(ss[2]),
                          ld_err=LD_ERROR["control"], err_rng=np.random.default_rng(ss[3]))
    pi = float(ts.diversity(mode="site"))
    return dict(n_anc=n_anc, n0=n0, t_grow=t_grow, r=r, seed=seed, **features(raw, pi), pi=pi)


def observed(path: str = OBS_DEFAULT) -> dict:
    m = json.load(open(path))["meta"]
    return features(m["control"], m["control_pi"]["mean"])


def set_ld_error(path: str = OBS_DEFAULT):
    """The control's LD genotype error model -> error_model.LD_ERROR["control"] (before the pool forks)."""
    from illex.abc import error_model as E
    em = json.load(open(path))["meta"]["ld_error_model"]["control"]
    E.LD_ERROR["control"] = {c: (float(em[c]["call_rate"]), float(em[c]["h"])) for c in ("I", "S")}
    return E.LD_ERROR["control"]


def draw_prior(n, rng):
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    n_anc = lu(*PRIOR["n_anc"])
    return n_anc, n_anc * lu(*PRIOR["fold"]), lu(*PRIOR["t_grow"]), lu(*PRIOR["r"])


def cmd_sim(a):
    print("LD error model (control):", set_ld_error(a.observed), flush=True)
    rng = np.random.default_rng(a.seed)
    n_anc, n0, t, r = draw_prior(a.n, rng)
    if a.fixed:                      # replicate one parameter set (validation), length = ABC control
        p = json.loads(a.fixed)
        n_anc, n0, t, r = (np.full(a.n, p[k]) for k in ("n_anc", "n0", "t_grow", "r"))
    seeds = rng.integers(1, 2**63 - 1, size=a.n)
    tasks = [(float(n_anc[i]), float(n0[i]), float(t[i]), float(r[i]), int(seeds[i]), a.length) for i in range(a.n)]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(one, tasks, chunksize=2)
    pd.DataFrame(res).to_csv(a.out, index=False)


def cmd_fit(a):
    d = pd.read_csv(a.draws).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
    obs = pd.Series(observed(a.observed))
    X = d[FEATS].to_numpy(float)
    mad = np.median(np.abs(X - np.median(X, 0)), 0)
    dist = np.sqrt((((X - obs[FEATS].to_numpy(float)) / mad) ** 2).sum(1))
    k = max(int(a.accept * len(d)), 50)
    idx = np.argsort(dist)[:k]
    h = dist[idx].max()
    w = 1 - (dist[idx] / h) ** 2                                   # Epanechnikov
    theta = np.log10(d[["n_anc", "n0", "t_grow", "r"]].to_numpy(float)[idx])
    Z = np.column_stack([np.ones(k), (X[idx] - obs[FEATS].to_numpy(float)) / mad])
    W = np.diag(w)
    beta = np.linalg.lstsq(np.sqrt(W) @ Z, np.sqrt(W) @ theta, rcond=None)[0]
    adj = theta - Z[:, 1:] @ beta[1:]
    def wq(x, q):
        o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return float(x[o][np.searchsorted(c, q)])
    res = {"n_draws": int(len(d)), "n_accepted": k, "observed_file": a.observed, "observed": obs.to_dict(), "params": {}}
    for j, name in enumerate(["n_anc", "n0", "t_grow", "r"]):
        res["params"][name] = {lab: 10 ** wq(adj[:, j], q) for lab, q in (("q025", .025), ("median", .5), ("q975", .975))}
        res["params"][name]["rejection_median"] = 10 ** wq(theta[:, j], .5)
    fold = adj[:, 1] - adj[:, 0]
    res["params"]["fold"] = {lab: 10 ** wq(fold, q) for lab, q in (("q025", .025), ("median", .5), ("q975", .975))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["params"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sim")
    s.add_argument("--n", type=int, default=20000); s.add_argument("--procs", type=int, default=24)
    s.add_argument("--seed", type=int, default=20261109); s.add_argument("--out", required=True)
    s.add_argument("--length", type=float, default=SIM_BP); s.add_argument("--fixed", default=None)
    s.add_argument("--observed", default=OBS_DEFAULT)
    f = sp.add_parser("fit")
    f.add_argument("--draws", required=True); f.add_argument("--out", required=True)
    f.add_argument("--accept", type=float, default=0.01); f.add_argument("--observed", default=OBS_DEFAULT)
    a = ap.parse_args()
    {"sim": cmd_sim, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
