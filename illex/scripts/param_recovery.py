#!/usr/bin/env python
"""Covariance-aware fit and parameter recovery for the inversion grid.

    .venv/bin/python -m illex.scripts.param_recovery --tags ABCD --boot 400

1. DISTANCE. The published fit scores cells with a diagonal chi2, sum_k [(m_k - x_k)/SE_k]^2, although
   pi_I/pi_S and dxy/pi_I are correlated across the inversion (1-Mb block jackknife). Here every cell is
   also scored with the Mahalanobis distance D = (m - x)' S^-1 (m - x), where S has the jackknife SEs and
   that correlation, and the singleton ratio (ANGSD, a separate estimator) is independent of the other two.
   The best cell, the profiles and the parametric-bootstrap cutoffs (95th percentile of the profile
   Delta at the true value, truth = best cell) are reported for both distances.

2. RECOVERY. Every grid cell in turn is the truth. For each, B pseudo-observed datasets are drawn as
   m(truth) + MVN(0, S), every cell's statistics are recomputed from a resample of its replicates, and the
   data are refitted on the whole grid. The script reports how often the best cell recovers each true
   parameter value, how often the calibrated 95% set (cutoffs from step 1) contains it, and the confusion
   between the true and recovered barrier age. The last answers whether a younger inversion with another
   frequency trajectory could be mistaken for the fitted old one.

TRAJECTORY VARIANCE (--traj). The grid conditions each cell on one deterministic frequency trajectory, but the
inversion's history is a single realisation of a stochastic process, and that part of the variance does not
average out along the 19 Mb. stochastic_check.py estimates it at a best cell as the covariance of the per-replicate
statistics under Wright-Fisher drift minus that under the deterministic trajectory (the genealogical part).
With --traj the averaged difference (clipped to positive semi-definite) is added to S, for the distance weights and
for the pseudo-data noise alike, and outputs carry the suffix _traj.

The test is conditional on the model and on the grid. Cell resamples are shared across truths, and the
coarse grid has cells with common random numbers (the legacy seed scheme), so cell-to-cell differences
are, if anything, noisier here than in the simulations.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from illex.scripts.param_bootstrap import (PARS, R, SE, TARGETS, load_cells, obs_corr, profiles,
                                           stats)


def distance(m, x, w):
    """Quadratic distance with weight matrix w; m [..., cells, 3], x [..., 3] -> [..., cells]."""
    d = m - x[..., None, :]
    return np.einsum("...ck,kl,...cl->...c", d, w, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="ABCD")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--traj", nargs="*", default=[], help="stochastic_check JSONs with per-replicate statistics")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cells = load_cells(a.tags)
    keys = np.array(sorted(cells))
    reps = [cells[tuple(k)] for k in keys]
    means = np.array([stats(np.full(len(c[0]), 1 / len(c[0])), c) for c in reps])
    rho, nblk = obs_corr()
    cov = np.diag(SE ** 2)
    cov[0, 1] = cov[1, 0] = rho * SE[0] * SE[1]
    meas = cov.copy()
    traj = np.zeros((3, 3))
    for f in a.traj:
        v = json.load(open(f))["versions"]
        traj += (np.cov(np.array(v["drift"]["per_rep"]).T) - np.cov(np.array(v["deterministic"]["per_rep"]).T)) / len(a.traj)
    ev, evec = np.linalg.eigh(traj)
    traj = evec @ np.diag(np.clip(ev, 0, None)) @ evec.T
    cov = meas + traj
    sfx = "_traj" if a.traj else ""
    W = {"diag": np.diag(1 / np.diag(cov)), "cov": np.linalg.inv(cov)}

    B, nc = a.boot, len(keys)
    mstar = np.empty((B, nc, 3))
    for i, c in enumerate(reps):
        n = len(c[0])
        mstar[:, i] = stats(rng.multinomial(n, np.full(n, 1 / n), size=B) / n, c)
    noise = rng.multivariate_normal(np.zeros(3), cov, size=B)

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    out = {"tags": a.tags, "boot": B, "n_cells": nc, "obs_corr_r_d": rho, "n_jackknife_blocks": nblk,
           "traj_files": a.traj, "traj_cov": traj.tolist(), "total_cov": cov.tolist()}
    emit(f"Inversion grid ({a.tags}, {nc} cells): covariance-aware fit and parameter recovery, {B} pseudo-datasets")
    emit(f"corr(pi_I/pi_S, dxy/pi_I) = {rho:+.2f} from {nblk} 1-Mb jackknife blocks; singleton ratio independent")
    if a.traj:
        emit("trajectory SD " + " ".join(f"{x:.4f}" for x in np.sqrt(np.diag(traj))) + "  measurement SE "
             + " ".join(f"{x:.4f}" for x in SE) + "  total SD " + " ".join(f"{x:.4f}" for x in np.sqrt(np.diag(cov))))
        emit("trajectory correlation\n" + np.array2string(traj / np.sqrt(np.outer(np.diag(traj), np.diag(traj)) + 1e-30), precision=2))
    for name, w in W.items():
        emit()
        emit(f"=== distance: {name} ===")
        obs = distance(means, TARGETS, w)
        ti = int(obs.argmin())
        truth = keys[ti]
        oprof = profiles(keys, obs)
        # step 1: bootstrap cutoffs with the best cell as truth
        D = distance(mstar, means[ti] + noise, w)
        bprof = profiles(keys, D)
        c95 = {p: float(np.quantile(bprof[p][truth[j]], 0.95)) for j, p in enumerate(PARS)}
        res = {"best": dict(zip(PARS, truth.tolist())), "obs_min": float(obs[ti]), "params": {}}
        emit(f"best cell {dict(zip(PARS, truth.tolist()))}, D = {obs[ti]:.2f}")
        for p in PARS:
            vals = sorted(oprof[p])
            ci = [v for v in vals if oprof[p][v] <= c95[p]]
            emit(f"  {p}: c95 {c95[p]:.2f}  profile " + "  ".join(f"{v:g}:{oprof[p][v]:.1f}" for v in vals))
            emit(f"      calibrated 95% set {[f'{v:g}' for v in ci]}")
            res["params"][p] = {"c95": c95[p], "profile": {f"{v:g}": float(oprof[p][v]) for v in vals},
                                "calibrated_95_set": [float(v) for v in ci]}
        # step 2: recovery with every cell as the truth
        hit = np.zeros((nc, 4))
        cover = np.zeros((nc, 4))
        conf = collections.defaultdict(collections.Counter)
        young = keys[:, 0] <= truth[0] - 100_000
        fooled = []
        for t in range(nc):
            D = distance(mstar, means[t] + noise, w)
            best = keys[D.argmin(axis=1)]
            if young[t]:
                fooled.append(np.mean(best[:, 0] >= truth[0] - 50_000))
            bp = profiles(keys, D)
            for j, p in enumerate(PARS):
                hit[t, j] = np.mean(best[:, j] == keys[t, j])
                cover[t, j] = np.mean(bp[p][keys[t, j]] <= c95[p])
            for v in best[:, 0]:
                conf[keys[t, 0]][v] += 1
        emit("  recovery over all truths (median and 10th percentile across truths):")
        rec = {}
        for j, p in enumerate(PARS):
            emit(f"    {p}: best cell has the true value in {100 * np.median(hit[:, j]):.0f}% "
                 f"(p10 {100 * np.quantile(hit[:, j], 0.1):.0f}%), calibrated set covers it in "
                 f"{100 * np.median(cover[:, j]):.0f}% (p10 {100 * np.quantile(cover[:, j], 0.1):.0f}%)")
            rec[p] = {"hit_median": float(np.median(hit[:, j])), "hit_p10": float(np.quantile(hit[:, j], 0.1)),
                      "cover_median": float(np.median(cover[:, j])),
                      "cover_p10": float(np.quantile(cover[:, j], 0.1))}
        tv = sorted(conf)
        emit("  barrier-age confusion (rows = true t_inv, columns = recovered best t_inv, % of pseudo-datasets):")
        emit("    " + "".join(f"{v / 1e3:>8.0f}" for v in tv))
        cm = {}
        for t in tv:
            n = sum(conf[t].values())
            emit(f"    {t / 1e3:>4.0f}" + "".join(f"{100 * conf[t][v] / n:>8.1f}" for v in tv))
            cm[f"{t:g}"] = {f"{v:g}": conf[t][v] / n for v in tv}
        if fooled:
            emit(f"  truths at least 100 ky younger than the best fit ({len(fooled)} cells) recovered within 50 ky "
                 f"of it: {100 * np.mean(fooled):.1f}% (worst cell {100 * max(fooled):.1f}%)")
            res["young_recovered_as_old"] = {"mean": float(np.mean(fooled)), "max": float(max(fooled)),
                                             "n_cells": len(fooled)}
        res["recovery"] = rec
        res["t_inv_confusion"] = cm
        res["per_truth"] = [{**dict(zip(PARS, keys[t].tolist())), "hit": hit[t].tolist(), "cover": cover[t].tolist()}
                            for t in range(nc)]
        out[name] = res
    (R / f"param_recovery_{a.tags}{sfx}.json").write_text(json.dumps(out, indent=1))
    (R / f"param_recovery_{a.tags}{sfx}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
