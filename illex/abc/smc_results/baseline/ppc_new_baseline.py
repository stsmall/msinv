"""Posterior predictive check of the run-set-B ABC-SMC posteriors (s prior up to 1e-1, now constants.PRIOR) under a
refitted neutral baseline and the representative collinear control. The ABC code is unchanged: the baseline constants
are swapped at run time, before the worker pool forks, in every module that holds a copy (illex.theory,
illex.demography, illex.abc.constants / simulate), and the control's LD genotype error model is set from the observed
file (abc_observed_ctl2.json). ppc/ppc_run.py's per-particle simulator is then reused unchanged. Summarise with
analyze_new_baseline.py, which scores against the same observed file.

    OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python ppc_new_baseline.py \
        --baseline fit.json --db abc_smc_mc_F.db --flux F --label B_fF --out ppc_newbase_B_fF.csv --n 1000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


def apply_baseline(n_anc, n0, t_grow, r):
    import illex.abc.constants as C
    import illex.abc.simulate as S
    import illex.demography as D
    import illex.slim.config as SC
    import illex.theory as T
    alpha = float(np.log(n0 / n_anc) / t_grow)
    for m in (T, D):
        m.N_ANC, m.N0, m.T_GROW, m.ALPHA = float(n_anc), float(n0), float(t_grow), alpha
    D.PRESENT_NE_GROWTH = float(n0)
    C.REC_RATE = S.REC_RATE = SC.REC_RATE = float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="fit.json from baseline_refit.py (posterior medians used)")
    ap.add_argument("--db", required=True); ap.add_argument("--flux", type=int, required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--observed", default="results/illex/abc_observed_ctl2.json")
    a = ap.parse_args()

    p = json.load(open(a.baseline))["params"]
    base = {k: p[k]["median"] for k in ("n_anc", "n0", "t_grow", "r")}
    apply_baseline(**base)
    from baseline_refit import set_ld_error
    print("baseline", base, "LD error (control)", set_ld_error(a.observed), flush=True)
    spec = importlib.util.spec_from_file_location("ppc_run", Path(__file__).resolve().parents[1] / "ppc" / "ppc_run.py")
    pr = importlib.util.module_from_spec(spec); sys.modules["ppc_run"] = pr; spec.loader.exec_module(pr)   # picklable for the pool

    import pyabc
    h = pyabc.History(f"sqlite:///{a.db}", _id=1)
    t = h.max_t
    rng = np.random.default_rng(a.seed)
    tasks, meta = [], []
    for mi in h.alive_models(t):
        df, w = h.get_distribution(m=mi, t=t)
        idx = rng.choice(len(df), size=a.n, p=w / w.sum())
        for i in idx:
            par = df.iloc[i].to_dict()
            tasks.append((pr.NAMES[mi], bool(a.flux), par, int(rng.integers(1, 2**31 - 1))))
            meta.append(dict(run=a.label, flux=a.flux, model=pr.NAMES[mi], particle=int(i), n_distinct=len(df),
                             **{f"par_{k}": v for k, v in par.items()}, **{f"base_{k}": v for k, v in base.items()}))
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(pr._one, tasks, chunksize=4)
    pd.concat([pd.DataFrame(meta), pd.DataFrame(res)], axis=1).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
