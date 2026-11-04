"""Python side of the ABC-RF inference (Task 8). R (``abc_infer.R``) does the inference only; this module owns the
definitions (statistic keys, prior bounds, which parameters each model has) and formats R's raw output.

    python -m illex.abc.abc_report spec   --out results/illex/abc_rf/spec.json
    Rscript illex/abc/abc_infer.R --table T.parquet --acceptance A.csv --observed results/illex/abc_observed.json \
        --spec results/illex/abc_rf/spec.json --out-dir results/illex/abc_rf
    python -m illex.abc.abc_report report --raw-dir results/illex/abc_rf --out-dir results/illex

``report`` writes abc_modelchoice.{csv,txt}, abc_posteriors.csv, abc_confusion.csv, abc_coverage.csv,
abc_crosscheck.csv and copies the per-model posterior draws abc_post_{M}_{flux}.csv.

Prior bounds have a single source: constants.PRIOR / GAMMA_MIN / GAMMA_MAX -> spec.json -> R (which copies it to
spec_used.json and writes prior_lo / prior_hi per posterior row); ``report`` warns when spec_used.json no longer matches
the constants. Prior-edge rule: on the fitting scale (log10 for log parameters), a parameter is flagged ``upper`` (``lower``) when its
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

from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR
from illex.abc.statvec import MODEL_KEYS

FIT_KEYS = tuple(MODEL_KEYS) + ("p_obs",)   # p_obs (binomial-sampled present frequency) is a fitting statistic
LOG_PARAMS = ("s", "p0", "gamma_eff")
# rows: which rows of a model's table carry the parameter. "soft" = soft rows only (hard rows have p0 = NaN and
# t_sel = t_inv by construction). models: which models define it; flux: 1 = only flux models.
PARAMS = {
    "t_inv": dict(rows="all", models=["N", "D1", "D2", "B"], lo=PRIOR["t_inv"][0], hi=PRIOR["t_inv"][1]),
    "s": dict(rows="all", models=["D1", "D2", "B"], lo=PRIOR["s"][0], hi=PRIOR["s"][1]),
    "p0": dict(rows="soft", models=["N", "D1", "D2", "B"], lo=PRIOR["p0"][0], hi=PRIOR["p0"][1]),
    # t_sel = U(0,1) t_inv, t_stop = U(0,1) t_sel: lower bound 0 is natural (selection recent / ongoing), not a
    # truncation, so edge_lo=False keeps it out of the prior-edge check
    "t_sel": dict(rows="soft", models=["D1", "D2", "B"], lo=0.0, hi=PRIOR["t_inv"][1], edge_lo=False),
    "t_stop": dict(rows="all", models=["D2"], lo=0.0, hi=PRIOR["t_inv"][1], edge_lo=False),
    "p_star": dict(rows="all", models=["B"], lo=PRIOR["p_star"][0], hi=PRIOR["p_star"][1]),
    "gamma_eff": dict(rows="all", models=["N", "D1", "D2", "B"], flux=1, lo=GAMMA_MIN, hi=GAMMA_MAX),
}


def spec() -> dict:
    return dict(fit_keys=list(FIT_KEYS), log_params=list(LOG_PARAMS),
                params={k: dict(v, log=k in LOG_PARAMS) for k, v in PARAMS.items()},
                p_hard=PRIOR["p_hard"])


def edge_flag(row, log_params=LOG_PARAMS, params=PARAMS) -> str:
    """Prior-edge flag from the row's own prior_lo / prior_hi (written by R from the spec it ran with)."""
    if not (isinstance(row.get("prior_lo"), (int, float)) and math.isfinite(row["prior_lo"])):
        return ""
    f = (lambda x: math.log10(x)) if row["parameter"] in log_params else (lambda x: x)
    lo, hi = f(row["prior_lo"]), f(row["prior_hi"])
    r = hi - lo
    pos = lambda x: (f(x) - lo) / r
    med, q_lo, q_hi = pos(row["median"]), pos(row["q025"]), pos(row["q975"])
    flags = []
    if med > 0.9 or q_hi > 0.98:
        flags.append("upper")
    if params.get(row["parameter"], {}).get("edge_lo", True) and (med < 0.1 or q_lo < 0.02):
        flags.append("lower")
    return "+".join(flags)


def check_spec(raw_dir: str) -> str:
    """'' if the spec R ran with equals the spec dumped from the current constants, else a warning line."""
    path = os.path.join(raw_dir, "spec_used.json")
    if not os.path.exists(path):
        return "WARNING: no spec_used.json in raw dir; prior bounds unverified"
    used = json.load(open(path))
    cur = json.loads(json.dumps(spec()))
    return "" if used == cur else ("WARNING: spec_used.json differs from illex/abc/constants.py (stale run); "
                                   "rerun `abc_report spec` and abc_infer.R")


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.4g}")


def report(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rd = lambda n: pd.read_csv(os.path.join(raw_dir, n))
    meta = json.load(open(os.path.join(raw_dir, "run_meta.json")))
    stale = check_spec(raw_dir)
    used = json.load(open(os.path.join(raw_dir, "spec_used.json"))) if not stale.startswith("WARNING: no") else {}
    log_params = tuple(used.get("log_params", LOG_PARAMS))
    mc = rd("modelchoice_raw.csv")
    mc.insert(mc.columns.get_loc("acc") + 1, "p_now_term", mc["acc"])
    mc.to_csv(os.path.join(out_dir, "abc_modelchoice.csv"), index=False)

    post = rd("posteriors_raw.csv")
    post["edge"] = post.apply(lambda r: edge_flag(r, log_params, used.get("params", PARAMS)), axis=1)
    hs = rd("hardshare_raw.csv")
    hs_rows = []
    for _, h in hs.iterrows():
        for name, col in (("hard_share_noacc", "hard_share_stats"), ("hard_share_acc", "hard_share_total"),
                          ("hard_share_mix", "hard_share_mix")):
            hs_rows.append(dict(model=h["model"], flux=h["flux"], parameter=name, scale="prob", median=h[col]))
    post_all = pd.concat([post, pd.DataFrame(hs_rows)], ignore_index=True)
    post_all.to_csv(os.path.join(out_dir, "abc_posteriors.csv"), index=False)

    conf = [rd(n).assign(method=m) for n, m in (("confusion_cv.csv", "grouped_cv"), ("confusion_oob.csv", "oob"))
            if os.path.exists(os.path.join(raw_dir, n))]
    conf = pd.concat(conf, ignore_index=True) if conf else pd.DataFrame(columns=["true", "predicted", "n", "w",
                                                                                 "method"])
    conf.to_csv(os.path.join(out_dir, "abc_confusion.csv"), index=False)
    cov = rd("coverage_raw.csv")
    cov.to_csv(os.path.join(out_dir, "abc_coverage.csv"), index=False)
    xc = rd("crosscheck_raw.csv")
    xc.to_csv(os.path.join(out_dir, "abc_crosscheck.csv"), index=False)
    for f in glob.glob(os.path.join(raw_dir, "abc_post_*.csv")):
        shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))

    lines = ["ABC-RF inference (illex/abc/abc_infer.R; formatted by illex/abc/abc_report.py)", ""]
    if stale:
        lines += [stale, ""]
    lines += [f"{k}: {v}" for k, v in meta.items()]
    lines += ["", f"HEADLINE model-choice accuracy (grouped 5-fold CV, mixture-weighted): "
                  f"{meta.get('accuracy_grouped_cv_weighted')}  (OOB: {meta.get('accuracy_oob_weighted')})", ""]
    lines += ["Model choice: prob_total(M) ∝ prior(M) x acc(M) x prob_stats(M); acc(M) = 0.5 acc_hard + "
              "0.5 acc_soft (= p_now_term); prob_stats from abcrf on unique rows with case.weights = mixture "
              "weights (models balanced)", _fmt(mc), ""]
    for m in ("grouped_cv", "oob"):
        c = conf[conf.method == m]
        if len(c):
            for col, lab in (("n", "unique rows"), ("w", "mixture weight; rows sum to 1")):
                tab = c.pivot_table(index="true", columns="predicted", values=col, aggfunc="sum", fill_value=0)
                lines += [f"Confusion matrix ({m}, {lab}; rows = true model):",
                          tab.to_string(float_format=lambda x: f"{x:.3f}"), ""]
    lines += ["Hard share: noacc = P(hard | stats) with class-balanced weights (prior 0.5); acc = combined with the "
              "class acceptances, acc_h p / (acc_h p + acc_s (1 - p)); mix = regAbcrf on the hard indicator with "
              "the acceptance-weighted mixture weights", _fmt(hs), ""]
    cols = ["model", "flux", "parameter", "scale", "n_train", "ess_train", "median", "q025", "q25", "q75", "q975",
            "prior_lo", "prior_hi", "edge"]
    lines += ["Posteriors (regAbcrf per parameter per model; 50% and 95% intervals; edge = prior-edge flag)",
              _fmt(post[[c for c in cols if c in post.columns]]), ""]
    lines += ["Coverage (grouped CV pods; nominal 0.50 and 0.95)", _fmt(cov), ""]
    lines += ["Cross-check: abc rejection / loclinear medians (tol = abc_accept / n)", _fmt(xc), ""]
    flagged = post[post.edge != ""]
    lines += ["Prior-edge flags: " + ("none" if flagged.empty else
              "; ".join(f"{r.model}/{r.parameter}: {r.edge}" for r in flagged.itertuples()))]
    with open(os.path.join(out_dir, "abc_modelchoice.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if stale:
        print(stale)


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
