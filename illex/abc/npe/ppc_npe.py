"""Posterior predictive checks of the balancing model from NPE posterior draws (train_npe.py output).

Same simulator and scoring as illex/abc/smc_results/ppc/ppc_run.py, but the parameter sets are draws from the NPE
posterior (natural units, soft starts) instead of SMC particles. Each draw's frequency path is retried until it lands
in the p_now window, then the inversion and control are simulated and every STAT_KEYS statistic recorded. The output
CSV has the columns ppc_analyze.py reads (run, flux, model, n_distinct, ok, statistics).

Run with PYTHONPATH set to a tree of branch illex/baseline-refit-karyofix (the base2 baseline, error model and observed
statistics are in its code); the preflight refuses any other baseline:

    PYTHONNOUSERSITE=1 PYTHONPATH=<base2 tree> OMP_NUM_THREADS=1 python illex/abc/npe/ppc_npe.py \
        --draws results/illex/npe/f1/posterior_samples.csv --flux 1 --out ppc_npe_f1.csv --n 1000 --procs 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

MAX_TRIES = 200


def preflight():
    import illex.theory as T
    import illex.abc.constants as C
    from illex.abc.smc import OBSERVED
    from illex.abc.error_model import LD_ERROR
    ok = (round(T.N_ANC) == 507288 and round(T.N0) == 9112696 and round(T.T_GROW) == 382957
          and float(f"{C.REC_RATE:.4g}") == 5.507e-10 and str(OBSERVED).endswith("abc_observed_ctl2_karyofix.json")
          and LD_ERROR["control"]["I"] == (0.6768, 0.6645) and C.P_NOW == 0.602 and C.N_KARYO == 619)
    if not ok:
        raise SystemExit("FATAL: PYTHONPATH is not a base2-karyofix (illex/baseline-refit-karyofix) tree")


def _one(task):
    par, flux, seed = task
    from illex.abc.simulate import simulate_stats
    from illex.abc.statvec import STAT_KEYS
    from illex.abc.trajectory import KEYS, simulate_paths
    rng = np.random.default_rng(seed)
    p = dict(t_inv=par["t_inv"], hard=False, p0=par["p0"], t_sel=par["t_sel"], s=par["s"], t_stop=np.nan,
             p_star=par["p_star"], gamma_eff=par["gamma"] if flux else np.nan)
    arr = {k: np.array([p[k]], dtype=bool if k == "hard" else float) for k in KEYS}
    arr.update(model="B", flux=bool(flux))
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
    ap.add_argument("--draws", required=True)
    ap.add_argument("--flux", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="npe")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    preflight()
    d = pd.read_csv(a.draws).sample(n=a.n, replace=False, random_state=a.seed).reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    tasks = [(r._asdict(), a.flux, int(rng.integers(1, 2**31 - 1))) for r in d.itertuples(index=False)]
    meta = pd.DataFrame({"run": a.label, "flux": a.flux, "model": "B", "particle": np.arange(len(d)),
                         "n_distinct": len(d)})
    meta = pd.concat([meta, d.add_prefix("par_")], axis=1)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_one, tasks, chunksize=2)
    out = pd.concat([meta, pd.DataFrame(res)], axis=1)
    out.to_csv(a.out, index=False)
    print(f"flux {a.flux}: {int(out.ok.sum())} of {len(out)} draws simulated", flush=True)


if __name__ == "__main__":
    main()
