"""Posterior predictive checks for the Illex trajectory ABC-SMC runs.

For every model alive at the final generation of one run (pyABC History DB), draw particles by posterior weight,
re-simulate each with the same simulator the SMC used (smc.IllexModel: frequency path, msinv body, msprime control,
statistics), retrying the path until it lands in the p_now window, and record every statistic in STAT_KEYS (the 39
fitted MODEL_KEYS plus the held-out ROW_PPC_KEYS tajd_I, tajd_S, edge_centre).

Run with PYTHONPATH set to the checkout whose priors match the run (to_params clips s to constants.PRIOR["s"]):

    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python ppc_run.py --db <abc_smc_mc_F.db> --flux F --label <run> --out <csv> --n 1000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyabc

NAMES = ("N", "D1", "D2", "B")
MAX_TRIES = 200


def _one(task):
    model, flux, par, seed = task
    from illex.abc.smc import IllexModel
    from illex.abc.statvec import STAT_KEYS
    from illex.abc.simulate import simulate_stats
    from illex.abc.smc import to_params, KEYS
    from illex.abc.trajectory import simulate_paths

    rng = np.random.default_rng(seed)
    p = to_params(par, model, flux, "mixed")
    arr = {k: np.array([p[k]], dtype=bool if k == "hard" else float) for k in KEYS}
    arr.update(model=model, flux=flux)
    for tries in range(1, MAX_TRIES + 1):
        paths, _ = simulate_paths(arr, rng, max_attempts=1)
        if paths:
            break
    else:
        return dict(ok=0, tries=MAX_TRIES)
    out = simulate_stats(paths[0], int(rng.integers(1, 2**31 - 1)))
    res = {k: float(out[k]) for k in STAT_KEYS}
    res.update(ok=1, tries=tries, p_now=float(paths[0].p_now), n_nan=float(out["n_nan"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--flux", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()

    h = pyabc.History(f"sqlite:///{a.db}", _id=1)
    t = h.max_t
    rng = np.random.default_rng(a.seed)
    tasks, meta = [], []
    for mi in h.alive_models(t):
        df, w = h.get_distribution(m=mi, t=t)
        idx = rng.choice(len(df), size=a.n, p=w / w.sum())
        for j, i in enumerate(idx):
            par = df.iloc[i].to_dict()
            tasks.append((NAMES[mi], bool(a.flux), par, int(rng.integers(1, 2**31 - 1))))
            meta.append(dict(run=a.label, flux=a.flux, model=NAMES[mi], particle=int(i), n_distinct=len(df),
                             **{f"par_{k}": v for k, v in par.items()}))
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_one, tasks, chunksize=4)
    pd.concat([pd.DataFrame(meta), pd.DataFrame(res)], axis=1).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
