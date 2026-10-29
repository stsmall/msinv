#!/usr/bin/env python
"""Parametric bootstrap of the inversion grid fit: calibrated Delta-chi-square thresholds.

    .venv/bin/python -m illex.scripts.param_bootstrap --boot 2000

WHAT IT ANSWERS
---------------
The fit scores each grid cell with chi2 = sum_k [(xhat_k - x_k)/SE_k]^2 and bounds each parameter with a
Delta-chi2 profile (others re-optimised). The manuscript reads Delta-chi2 < 2 as "not excluded", but with
three statistics, four parameters, a coarse irregular grid, and simulated cell means that are themselves
noisy, that threshold has no known coverage. This script measures it.

THE GRID
--------
The manuscript's final grid (supplement "Fitting and profiles"), rebuilt by run_pboot_grid.sh with
refit_decline.py at 48 replicates per cell and per-replicate output (refit_decline_pboot?_reps.jsonl.gz):
t_inv 800/850/900 ky x p_hist 0.70-0.82, t_inv 700/750 ky x p_hist 0.70/0.74, t_decline 100-250 ka and
t_fall 50-200 ky, plus the two extensions around the best cells (t_decline 0-50 ka, t_fall 400/600 ky).
Cell statistics are computed exactly as refit_decline.py does (mean ratios, singleton ratio from the
cell-mean branch spectra projected to 20 chromosomes).

THE BOOTSTRAP
-------------
Truth is the best-fitting cell. Each pseudo-observed dataset is
    x* = m(truth) + e,   e ~ MVN(0, Sigma_obs)
where Sigma_obs holds the empirical block-jackknife SEs (the measurement noise of the 19-Mb observation),
with the pi_I/pi_S--dxy/pi_I correlation from the same 1-Mb block jackknife of the per-window statistics,
and the ANGSD singleton ratio independent of the other two. Every cell's statistics are recomputed from a
resample (with replacement) of its own replicates, so the refit sees the simulation noise of a 48-replicate
cell. Each pseudo-dataset is refitted on the same grid with the same weighted chi2. For every parameter the
bootstrap records the profile Delta-chi2 at its TRUE value. Its 95th percentile is the calibrated threshold
c95, and the calibrated 95% interval is every grid value whose OBSERVED profile Delta-chi2 is <= c95.

WHAT IT DOES NOT DO
-------------------
The noise model is the jackknife SE, so the bootstrap inherits whatever the jackknife misses, and the
demography, mu and r stay fixed at their fitted values. It calibrates the fit given the model.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
from pathlib import Path

import numpy as np

from illex.scripts.sfs_shape import PROJ, project_unfolded

R = Path("results/illex")
PARS = ("t_inv", "p_hist", "t_decline", "t_fall")
TARGETS = np.array([1.3556, 1.3848, 0.8125])
SE = np.array([0.0481, 0.0214, 0.0076])
BP_L, BP_R = 60_540_000, 79_500_000


def load_cells():
    """{params: (r[n], d[n], PI[n, PROJ+1], PS[n, PROJ+1])} from every *_pboot*_reps.jsonl.gz."""
    raw = collections.defaultdict(lambda: ([], [], [], []))
    for f in sorted(glob.glob(str(R / "refit_decline_pboot*_reps.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            x = json.loads(line)
            k = tuple(float(x[p]) for p in PARS)
            c = raw[k]
            c[0].append(x["pi_i_over_pi_s"])
            c[1].append(x["dxy_over_pi_i"])
            c[2].append(project_unfolded(np.asarray(x["I"], float), x["I_n"], PROJ))
            c[3].append(project_unfolded(np.asarray(x["S"], float), x["S_n"], PROJ))
    return {k: tuple(np.asarray(v) for v in c) for k, c in raw.items()}


def stats(w, c):
    """Cell statistics under replicate weights w ([..., n], rows sum to 1)."""
    r, d, pi, ps = c
    si, ss = w @ pi, w @ ps
    f1 = (si[..., 0] / si.sum(-1)) / (ss[..., 0] / ss.sum(-1))
    return np.stack([w @ r, w @ d, f1], axis=-1)


def obs_corr():
    """Correlation of pi_I/pi_S and dxy/pi_I from a delete-one 1-Mb block jackknife over the inversion."""
    w = [r for r in csv.DictReader(open(R / "empirical_jackknife_windows.csv"))
         if BP_L <= float(r["center"]) <= BP_R]
    blk = np.array([int(float(r["window_start"]) // 1_000_000) for r in w])
    piS = np.array([float(r["pi_AA"]) for r in w])
    piI = np.array([float(r["pi_BB"]) for r in w])
    dxy = np.array([float(r["dxy"]) for r in w])
    loo = [(piI[blk != b].sum() / piS[blk != b].sum(), dxy[blk != b].sum() / piI[blk != b].sum())
           for b in np.unique(blk)]
    return float(np.corrcoef(np.array(loo).T)[0, 1]), len(np.unique(blk))


def profiles(keys, chi):
    """Delta-chi2 profile per parameter: {par: {value: delta}} (chi has shape [..., n_cells])."""
    gmin = chi.min(axis=-1)
    return {p: {v: chi[..., keys[:, j] == v].min(axis=-1) - gmin for v in np.unique(keys[:, j])}
            for j, p in enumerate(PARS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    cells = load_cells()
    keys = np.array(sorted(cells))
    reps = [cells[tuple(k)] for k in keys]
    means = np.array([stats(np.full(len(c[0]), 1 / len(c[0])), c) for c in reps])
    obs_chi = (((means - TARGETS) / SE) ** 2).sum(axis=1)
    ti = int(obs_chi.argmin())
    truth = tuple(keys[ti])
    obs_prof = profiles(keys, obs_chi)

    rho, nblk = obs_corr()
    cov = np.diag(SE ** 2)
    cov[0, 1] = cov[1, 0] = rho * SE[0] * SE[1]
    B = a.boot
    xstar = means[ti] + rng.multivariate_normal(np.zeros(3), cov, size=B)
    mstar = np.empty((B, len(keys), 3))
    for i, c in enumerate(reps):
        n = len(c[0])
        w = rng.multinomial(n, np.full(n, 1 / n), size=B) / n
        mstar[:, i] = stats(w, c)
    chi = (((mstar - xstar[:, None, :]) / SE) ** 2).sum(axis=2)
    best = keys[chi.argmin(axis=1)]
    bprof = profiles(keys, chi)
    nrep = sorted({len(c[0]) for c in reps})

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    res = {"boot": B, "seed": a.seed, "truth": dict(zip(PARS, truth)), "n_cells": len(keys),
           "reps_per_cell": nrep, "obs_chi2_truth": float(obs_chi[ti]), "obs_corr_r_d": rho,
           "n_jackknife_blocks": nblk, "params": {}}
    emit(f"Parametric bootstrap of the inversion grid fit ({B} pseudo-datasets, {len(keys)} cells, "
         f"{nrep} reps/cell)")
    emit(f"truth = best cell {dict(zip(PARS, truth))}; observed chi2 there = {obs_chi[ti]:.2f}")
    emit(f"noise: jackknife SE {SE.tolist()}, corr(pi_I/pi_S, dxy/pi_I) = {rho:+.2f} ({nblk} 1-Mb blocks); "
         "cell means resampled from their replicates")
    dtrue = chi[:, ti] - chi.min(axis=1)
    emit(f"joint Delta-chi2 at the truth: median {np.median(dtrue):.2f}, 95th pct {np.quantile(dtrue, 0.95):.2f}")
    emit()
    for j, p in enumerate(PARS):
        d = bprof[p][truth[j]]
        c95 = float(np.quantile(d, 0.95))
        cover2 = float(np.mean(d < 2))
        vals = sorted(obs_prof[p])
        ci = [v for v in vals if obs_prof[p][v] <= c95]
        nominal = [v for v in vals if obs_prof[p][v] < 2]
        rec = collections.Counter(best[:, j])
        emit(f"{p}: c95 = {c95:.2f} (Delta-chi2 < 2 covers the truth in {100 * cover2:.1f}% of pseudo-datasets)")
        emit("   observed profile    " + "  ".join(f"{v:g}:{obs_prof[p][v]:.1f}" for v in vals))
        emit(f"   calibrated 95% set  {[f'{v:g}' for v in ci]}   (Delta-chi2 < 2: {[f'{v:g}' for v in nominal]})")
        emit("   refitted best value " + "  ".join(f"{v:g}:{100 * rec[v] / B:.1f}%" for v in sorted(rec)))
        res["params"][p] = {"c95": c95, "coverage_delta_lt_2": cover2,
                            "observed_profile": {f"{v:g}": float(obs_prof[p][v]) for v in vals},
                            "calibrated_95_set": [float(v) for v in ci],
                            "delta_lt_2_set": [float(v) for v in nominal],
                            "refit_best_freq": {f"{v:g}": rec[v] / B for v in sorted(rec)}}
    (R / "param_bootstrap.json").write_text(json.dumps(res, indent=1))
    (R / "param_bootstrap.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
