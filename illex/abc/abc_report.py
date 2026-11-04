"""Python side of the ABC-RF inference (Task 8). R (``abc_infer.R``) does the inference only; this module owns the
definitions (statistic keys, prior bounds, which parameters each model has) and formats R's raw output.

    python -m illex.abc.abc_report spec   --out results/illex/abc_rf/spec.json
    Rscript illex/abc/abc_infer.R --table T.parquet --acceptance A.csv --observed results/illex/abc_observed.json \
        --spec results/illex/abc_rf/spec.json --out-dir results/illex/abc_rf
    python -m illex.abc.abc_report report --raw-dir results/illex/abc_rf --out-dir results/illex

``report`` writes abc_modelchoice.{csv,txt}, abc_posteriors.csv, abc_confusion.csv, abc_coverage.csv,
abc_crosscheck.csv and copies the per-model posterior draws abc_post_{M}_{flux}.csv.

Prior-edge rule: on the fitting scale (log10 for log parameters), a parameter is flagged ``upper`` (``lower``) when its
posterior median lies in the top (bottom) 10% of the prior range, or its 95% interval reaches the top (bottom) 2%.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil

import pandas as pd

from illex.abc.constants import GAMMA_MAX, PRIOR
from illex.abc.statvec import MODEL_KEYS

FIT_KEYS = tuple(MODEL_KEYS) + ("p_obs",)   # p_obs (binomial-sampled present frequency) is a fitting statistic
LOG_PARAMS = ("s", "p0", "gamma_eff")
GAMMA_MIN = 1e-10                           # lower end of the Task 5 c sweep (logU prior on c)
# rows: which rows of a model's table carry the parameter. "soft" = soft rows only (hard rows have p0 = NaN and
# t_sel = t_inv by construction). models: which models define it; flux: 1 = only flux models.
PARAMS = {
    "t_inv": dict(rows="all", models=["N", "D1", "D2", "B"], lo=PRIOR["t_inv"][0], hi=PRIOR["t_inv"][1]),
    "s": dict(rows="all", models=["D1", "D2", "B"], lo=PRIOR["s"][0], hi=PRIOR["s"][1]),
    "p0": dict(rows="soft", models=["N", "D1", "D2", "B"], lo=PRIOR["p0"][0], hi=PRIOR["p0"][1]),
    "t_sel": dict(rows="soft", models=["D1", "D2", "B"], lo=0.0, hi=PRIOR["t_inv"][1]),
    "t_stop": dict(rows="all", models=["D2"], lo=0.0, hi=PRIOR["t_inv"][1]),
    "p_star": dict(rows="all", models=["B"], lo=PRIOR["p_star"][0], hi=PRIOR["p_star"][1]),
    "gamma_eff": dict(rows="all", models=["N", "D1", "D2", "B"], flux=1, lo=GAMMA_MIN, hi=GAMMA_MAX),
}


def spec() -> dict:
    return dict(fit_keys=list(FIT_KEYS), log_params=list(LOG_PARAMS),
                params={k: dict(v, log=k in LOG_PARAMS) for k, v in PARAMS.items()},
                p_hard=PRIOR["p_hard"])


def edge_flag(row) -> str:
    p = PARAMS.get(row["parameter"])
    if p is None:
        return ""
    f = (lambda x: math.log10(x)) if row["parameter"] in LOG_PARAMS else (lambda x: x)
    lo, hi = f(p["lo"]), f(p["hi"])
    r = hi - lo
    pos = lambda x: (f(x) - lo) / r
    med, q_lo, q_hi = pos(row["median"]), pos(row["q025"]), pos(row["q975"])
    flags = []
    if med > 0.9 or q_hi > 0.98:
        flags.append("upper")
    if med < 0.1 or q_lo < 0.02:
        flags.append("lower")
    return "+".join(flags)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.4g}")


def report(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rd = lambda n: pd.read_csv(os.path.join(raw_dir, n))
    meta = json.load(open(os.path.join(raw_dir, "run_meta.json")))
    mc = rd("modelchoice_raw.csv")
    mc.insert(mc.columns.get_loc("acc") + 1, "p_now_term", mc["acc"])
    mc.to_csv(os.path.join(out_dir, "abc_modelchoice.csv"), index=False)

    post = rd("posteriors_raw.csv")
    post["edge"] = post.apply(edge_flag, axis=1)
    hs = rd("hardshare_raw.csv")
    hs_rows = []
    for _, h in hs.iterrows():
        for name, col in (("hard_share_noacc", "hard_share_stats"), ("hard_share_acc", "hard_share_total"),
                          ("hard_share_mix", "hard_share_mix")):
            hs_rows.append(dict(model=h["model"], flux=h["flux"], parameter=name, scale="prob", median=h[col]))
    post_all = pd.concat([post, pd.DataFrame(hs_rows)], ignore_index=True)
    post_all.to_csv(os.path.join(out_dir, "abc_posteriors.csv"), index=False)

    conf = [rd(n).assign(method=m) for n, m in (("confusion_oob.csv", "oob"), ("confusion_cv.csv", "grouped_cv"))
            if os.path.exists(os.path.join(raw_dir, n))]
    conf = pd.concat(conf, ignore_index=True)
    conf.to_csv(os.path.join(out_dir, "abc_confusion.csv"), index=False)
    cov = rd("coverage_raw.csv")
    cov.to_csv(os.path.join(out_dir, "abc_coverage.csv"), index=False)
    xc = rd("crosscheck_raw.csv")
    xc.to_csv(os.path.join(out_dir, "abc_crosscheck.csv"), index=False)
    for f in glob.glob(os.path.join(raw_dir, "abc_post_*.csv")):
        shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))

    lines = ["ABC-RF inference (illex/abc/abc_infer.R; formatted by illex/abc/abc_report.py)", ""]
    lines += [f"{k}: {v}" for k, v in meta.items()]
    lines += ["", "Model choice: prob_total(M) ∝ prior(M) x acc(M) x prob_stats(M); acc(M) = 0.5 acc_hard + "
              "0.5 acc_soft (= p_now_term); prob_stats from abcrf on the per-model balanced, weight-resampled table",
              _fmt(mc), ""]
    for m in ("oob", "grouped_cv"):
        c = conf[conf.method == m]
        if len(c):
            tab = c.pivot_table(index="true", columns="predicted", values="n", aggfunc="sum", fill_value=0)
            lines += [f"Confusion matrix ({m}; rows = true model):", tab.to_string(), ""]
    lines += ["Hard share: noacc = P(hard | stats) on a class-balanced table (prior 0.5); acc = combined with the "
              "class acceptances, acc_h p / (acc_h p + acc_s (1 - p)); mix = regAbcrf on the hard indicator over the "
              "acceptance-weighted mixture table", _fmt(hs), ""]
    cols = ["model", "flux", "parameter", "scale", "n_train", "median", "q025", "q25", "q75", "q975", "edge"]
    lines += ["Posteriors (regAbcrf per parameter per model; 50% and 95% intervals; edge = prior-edge flag)",
              _fmt(post[cols]), ""]
    lines += ["Coverage (grouped CV pods; nominal 0.50 and 0.95)", _fmt(cov), ""]
    lines += ["Cross-check: abc rejection / loclinear medians (tol in meta)", _fmt(xc), ""]
    flagged = post[post.edge != ""]
    lines += ["Prior-edge flags: " + ("none" if flagged.empty else
              "; ".join(f"{r.model}/{r.parameter}: {r.edge}" for r in flagged.itertuples()))]
    with open(os.path.join(out_dir, "abc_modelchoice.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("spec")
    a.add_argument("--out", required=True)
    b = sub.add_parser("report")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--out-dir", default="results/illex")
    args = ap.parse_args(argv)
    if args.cmd == "spec":
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(spec(), fh, indent=1)
    else:
        report(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
