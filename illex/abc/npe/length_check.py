"""Does the 40-kb simulated inversion represent the 19-Mb inversion? Paired length check under the final model.

The ABC-SMC and the NPE simulate a BODY_BP = 40 kb inversion body (between two 10-kb collinear margins) and compare
its statistics with those of the whole 19-Mb inversion. Here each of --n NPE posterior draws (balancing model with
gene flux) gets one frequency path, and the same path is simulated at every length in --lengths (default 40 kb to
the full 18.96 Mb), so differences between lengths are paired within a draw. For each length the script records
all fitted statistics (MODEL_KEYS) and the held-out ones, the wall time and the peak memory.

Run against a tree of branch illex/baseline-refit (base2 baseline), like ppc_npe.py:

    PYTHONNOUSERSITE=1 PYTHONPATH=<base2 tree> OMP_NUM_THREADS=1 python illex/abc/npe/length_check.py \
        --draws results/illex/npe/f1/posterior_samples.csv --out results/illex/npe/length_check --n 48 --procs 16

Writes OUT/length_check.jsonl (one record per draw x length, appended as each finishes). Summarise with
length_check_summary.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import time

import numpy as np
import pandas as pd

LENGTHS = (40_000, 160_000, 640_000, 2_560_000, 18_960_000)
MAX_TRIES = 200


def path_for(par, seed):
    from illex.abc.trajectory import KEYS, simulate_paths
    rng = np.random.default_rng(seed)
    p = dict(t_inv=par["t_inv"], hard=False, p0=par["p0"], t_sel=par["t_sel"], s=par["s"], t_stop=np.nan,
             p_star=par["p_star"], gamma_eff=par["gamma"])
    arr = {k: np.array([p[k]], dtype=bool if k == "hard" else float) for k in KEYS}
    arr.update(model="B", flux=True)
    for _ in range(MAX_TRIES):
        paths, _ = simulate_paths(arr, rng, max_attempts=1)
        if paths:
            return paths[0]
    return None


def one(task):
    draw, par, seed, L = task
    import illex.abc.simulate as SIM
    from illex.abc.statvec import STAT_KEYS
    pth = path_for(par, seed)          # same seed -> same path at every length
    if pth is None:
        return dict(draw=draw, L=L, ok=0)
    SIM.BODY_BP = float(L)
    t = time.time()
    out = SIM.simulate_stats(pth, seed + 1)
    res = {k: float(out[k]) for k in STAT_KEYS}
    res.update(draw=draw, L=L, ok=1, wall_s=time.time() - t,
               maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, **{f"par_{k}": v for k, v in par.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.draws).sample(n=a.n, replace=False, random_state=a.seed).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    seeds = rng.integers(1, 2**30, size=len(d))
    # longest first so the slow tasks start early; short ones fill the gaps
    tasks = [(i, d.iloc[i].to_dict(), int(seeds[i]), L) for L in sorted(a.lengths, reverse=True) for i in range(len(d))]
    f = os.path.join(a.out, "length_check.jsonl")
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(one, tasks):
            with open(f, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"draw {res['draw']} L {res['L']} ok {res['ok']} {res.get('wall_s', float('nan')):.0f}s", flush=True)


if __name__ == "__main__":
    main()
