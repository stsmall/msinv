"""Summarise the posterior predictive check CSVs written by ppc_run.py against the observed statistics.

    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python ppc_analyze.py ppc_A_f0.csv ppc_A_f1.csv [--observed OBS.json]

The observed statistics default to results/illex/abc_observed_ctl2.json at the repository root, the vector the base2
SMC and the NPE were fitted to (divided by the representative collinear control). Pass --observed for another one,
e.g. the pre-refit results/illex/abc_observed.json. The file used is printed.

Writes ppc_per_stat.csv (each run x model x statistic: observed value, predictive median and 95% range, the observed
value's predictive quantile and a two-sided predictive p-value), ppc_joint.csv (a joint check over the 39 fitted
statistics: the observed vector's MAD-scaled L2 distance from the predictive median against the replicates' own) and ppc_separation.csv
(how well each statistic separates B from the directional models: overlap of their predictive distributions).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from illex.abc.statvec import MODEL_KEYS, ROW_PPC_KEYS

OBS = Path(__file__).resolve().parents[4] / "results" / "illex" / "abc_observed_ctl2.json"


def scaled_distance_check(sim: np.ndarray, obs: np.ndarray):
    """Joint check in the ABC's own metric: L2 distance after scaling each statistic by its predictive MAD (the
    full Mahalanobis form is unusable: the JSFS cells are proportions that sum to 1, so the covariance is singular).
    Returns the observed distance from the predictive median and the share of replicates at least as far."""
    ok = np.isfinite(sim).all(axis=1)
    x = sim[ok]
    mu = np.median(x, axis=0)
    mad = np.median(np.abs(x - mu), axis=0)
    w = np.where(mad > 0, 1.0 / np.where(mad > 0, mad, 1.0), 0.0)
    d = np.sqrt((((x - mu) * w) ** 2).sum(axis=1))
    d0 = float(np.sqrt((((obs - mu) * w) ** 2).sum()))
    return d0, float(np.mean(d >= d0)), int(ok.sum())


def main(paths, obs_path=OBS):
    print(f"observed statistics: {obs_path}", flush=True)
    st = json.load(open(obs_path))["stats"]
    keys = list(MODEL_KEYS) + list(ROW_PPC_KEYS)
    obs = {k: float(st[k]) for k in keys}
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d = d[d.ok == 1]
    per, joint, sep = [], [], []
    for (run, flux, model), g in d.groupby(["run", "flux", "model"]):
        for k in keys:
            x = g[k].to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            lo, hi = float(np.mean(x < obs[k])), float(np.mean(x <= obs[k]))
            q = 0.5 * (lo + hi)
            per.append(dict(run=run, flux=flux, model=model, stat=k, fitted=k in MODEL_KEYS, n=len(x),
                            n_distinct_particles=int(g.n_distinct.iloc[0]), observed=obs[k],
                            pred_median=float(np.median(x)), pred_q025=float(np.quantile(x, 0.025)),
                            pred_q975=float(np.quantile(x, 0.975)), obs_quantile=q,
                            ppp_two_sided=float(min(1.0, 2 * min(hi, 1 - lo)))))
        sim = g[list(MODEL_KEYS)].to_numpy(float)
        d0, p, n = scaled_distance_check(sim, np.array([obs[k] for k in MODEL_KEYS]))
        joint.append(dict(run=run, flux=flux, model=model, n=n, n_distinct_particles=int(g.n_distinct.iloc[0]),
                          scaled_dist_obs=d0, joint_ppp=p))
    for (run, flux), g in d.groupby(["run", "flux"]):
        b = g[g.model == "B"]
        for other in ("D1", "D2"):
            o = g[g.model == other]
            if len(o) == 0 or len(b) == 0:
                continue
            for k in keys:
                xb, xo = b[k].to_numpy(float), o[k].to_numpy(float)
                xb, xo = xb[np.isfinite(xb)], xo[np.isfinite(xo)]
                # AUC: P(stat_B > stat_other), 0.5 = no separation
                auc = float(np.mean(xb[:, None] > xo[None, :]) + 0.5 * np.mean(xb[:, None] == xo[None, :]))
                sep.append(dict(run=run, flux=flux, other=other, stat=k, fitted=k in MODEL_KEYS,
                                auc_B_gt_other=auc, separation=abs(auc - 0.5) * 2,
                                obs_q_in_B=float(np.mean(xb <= obs[k])), obs_q_in_other=float(np.mean(xo <= obs[k]))))
    pd.DataFrame(per).to_csv("ppc_per_stat.csv", index=False)
    pd.DataFrame(joint).to_csv("ppc_joint.csv", index=False)
    pd.DataFrame(sep).to_csv("ppc_separation.csv", index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--observed", default=str(OBS))
    a = ap.parse_args()
    main(a.csv, Path(a.observed))
