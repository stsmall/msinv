"""Collinear (control) region check: the observed chr2 10-30 Mb collinear statistics against the ABC's own neutral
control simulation (simulate._control_ts: msprime, growth demography from illex.theory, no inversion), with the
data's genotype error model applied to the LD diploids exactly as in every ABC row.

    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python collinear_check.py --n 1000 --procs 32 [--obs <observed json>]

The observed values are meta["control"] of the observed JSON (default results/illex/abc_observed.json, the chr2
10-30 Mb control; results/illex/abc_observed_ctl2.json is the 40-55 + 85-115 Mb control). The simulated LD diploids
take that JSON's control genotype error model (meta["ld_error_model"]["control"]).
Writes collinear_sims.csv (one row per replicate) and collinear_check.csv (observed vs predictive
median, 95% range, quantile and two-sided p per statistic). The JSFS has no observed control (realSFS 2D was run on
the body only), so the shared low-frequency cells cannot be checked here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd


def _one(task):
    from illex.abc.simulate import _seed_words, control_stats

    seed, ld_err = task
    w = _seed_words(seed)
    return control_stats(w[1], w[3], w[5], w[8], ld_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--obs", default=None)
    a = ap.parse_args()

    import illex
    obs_file = a.obs or Path(illex.__file__).resolve().parents[1] / "results" / "illex" / "abc_observed.json"
    meta = json.load(open(obs_file))["meta"]
    obs = meta["control"]
    ld_err = {c: (float(v["call_rate"]), float(v["h"])) for c, v in meta["ld_error_model"]["control"].items()}
    seeds = np.random.default_rng(a.seed).integers(1, 2**31 - 1, size=a.n)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_one, [(int(s), ld_err) for s in seeds], chunksize=4)
    sims = pd.DataFrame(res)
    sims.to_csv("collinear_sims.csv", index=False)
    rows = []
    for k, o in obs.items():
        x = sims[k].to_numpy(float)
        x = x[np.isfinite(x)]
        lo, hi = float(np.mean(x < o)), float(np.mean(x <= o))
        rows.append(dict(stat=k, n=len(x), observed=o, pred_median=float(np.median(x)),
                         pred_q025=float(np.quantile(x, 0.025)), pred_q975=float(np.quantile(x, 0.975)),
                         obs_over_median=o / float(np.median(x)), obs_quantile=0.5 * (lo + hi),
                         ppp_two_sided=float(min(1.0, 2 * min(hi, 1 - lo)))))
    pd.DataFrame(rows).to_csv("collinear_check.csv", index=False)


if __name__ == "__main__":
    main()
