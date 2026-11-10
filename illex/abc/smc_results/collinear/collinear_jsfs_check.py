"""Control JSFS check: meta["jsfs_control"] of an observed JSON (the AA40 x BB40 2D realSFS on the collinear control)
against the 12 JSFS cells of collinear_check.py's baseline replicates, next to the inversion's observed cells and
model B's predictive median for them (a ppc_per_stat.csv row set).

    PYTHONNOUSERSITE=1 python collinear_jsfs_check.py --obs <abc_observed_ctl2.json> --body <abc_observed.json> \\
        --sims collinear_sims.csv --ppc ppc_per_stat.csv --run B --flux 1

Writes collinear_jsfs_check.csv.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--sims", required=True)
    ap.add_argument("--ppc", required=True)
    ap.add_argument("--run", default="B")
    ap.add_argument("--flux", type=int, default=1)
    ap.add_argument("--out", default="collinear_jsfs_check.csv")
    a = ap.parse_args()

    ctl = json.load(open(a.obs))["meta"]["jsfs_control"]
    body = json.load(open(a.body))["stats"]
    sims = pd.read_csv(a.sims)
    ppc = pd.read_csv(a.ppc)
    ppc = ppc[(ppc.model == "B") & (ppc.run == a.run) & (ppc.flux == a.flux)].set_index("stat")
    rows = []
    for k in sorted(ctl):
        x = sims[k].to_numpy(float)
        x = x[np.isfinite(x)]
        o = ctl[k]
        lo, hi = np.mean(x < o), np.mean(x <= o)
        rows.append(dict(stat=k, obs_ctl=o, sim_ctl_med=np.median(x), sim_q025=np.quantile(x, .025),
                         sim_q975=np.quantile(x, .975), ppp=min(1.0, 2 * min(hi, 1 - lo)), obs_body=body[k],
                         B_pred_body=ppc.loc[k, "pred_median"], B_ppp_body=ppc.loc[k, "ppp_two_sided"]))
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
