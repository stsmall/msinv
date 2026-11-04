"""Collect reference-table shards.

    python -m illex.abc.collect DIR [--expect N] [--out-dir results/illex]

Writes ``abc_table.parquet`` (pyarrow if importable, else ``abc_table.csv.gz``) and ``abc_acceptance.csv``:
acceptance = the importance-weighted prior acceptance (1 / drawn) * sum(w_est over accepted) per model x flux x class
(normal 95% CI from the weighted variance; Wilson when every weight is 1), plus ``acceptance_attempts`` = accepted /
attempts as a cross-check (a ratio of sums, so it differs from the prior acceptance by a Jensen term when u varies with
theta), and the lost / fixed / lost_establishing breakdown (summed Counts over tasks). Shards from before the
importance sampling (no w_acc) count as w = 1, attempts = drawn. ``--expect N`` reports task IDs 0..N-1 missing for each
(model, flux, class) combination present.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
from collections import defaultdict

import pandas as pd

PAT = re.compile(r"^(?P<m>[A-Za-z0-9]+)_(?P<f>\d)_(?P<c>hard|soft)_(?P<i>\d+)\.jsonl\.gz$")
CFIELDS = ("drawn", "accepted", "lost", "fixed", "lost_establishing", "est_failed", "attempts", "w_acc", "w_acc_sq")


def wilson(k, n, z=1.959964):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def collect(dirpath, expect=None, out_dir="results/illex"):
    shards = defaultdict(set)
    frames = []
    csum = defaultdict(lambda: dict.fromkeys(CFIELDS, 0.0))
    for p in sorted(glob.glob(os.path.join(dirpath, "*.jsonl.gz"))):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        key = (m["m"], int(m["f"]), m["c"])
        i = int(m["i"])
        cj = os.path.join(dirpath, f"{m['m']}_{m['f']}_{m['c']}_{i}_counts.json")
        if not os.path.exists(cj):
            continue                                    # a shard without its counts is incomplete
        with gzip.open(p, "rt") as f:
            rows = [json.loads(x) for x in f]
        if rows:
            frames.append(pd.DataFrame(rows))
        shards[key].add(i)
        cnt = json.load(open(cj))["counts"][m["c"]]
        legacy = "w_acc" not in cnt                       # pre-importance-sampling shard: w = 1, attempts = drawn
        for k in CFIELDS:
            if k == "w_acc" or k == "w_acc_sq":
                csum[key][k] += cnt[k] if not legacy else cnt["accepted"]
            elif k == "attempts":
                csum[key][k] += cnt[k] if not legacy else cnt["drawn"]
            else:
                csum[key][k] += int(cnt.get(k, 0))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "w_est" in df:
        df["w_est"] = df["w_est"].fillna(1.0)               # older soft shards carry no weight
    os.makedirs(out_dir, exist_ok=True)
    fmt = "parquet"
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(os.path.join(out_dir, "abc_table.parquet"), index=False)
    except ImportError:
        fmt = "csv.gz"
        df.to_csv(os.path.join(out_dir, "abc_table.csv.gz"), index=False)
    acc = []
    for (m, f, c), s in sorted(csum.items()):
        n, k, w, w2 = s["drawn"], s["accepted"], s["w_acc"], s["w_acc_sq"]
        if abs(w - k) < 1e-9 * max(1.0, k):               # all weights 1: plain binomial, Wilson interval
            lo, hi = wilson(k, n)
        else:                                              # importance-weighted mean of w * 1[accepted]
            mean = w / n
            se = math.sqrt(max(w2 / n - mean * mean, 0.0) / n)
            lo, hi = max(0.0, mean - 1.959964 * se), mean + 1.959964 * se
            if k == 0:
                lo, hi = 0.0, float("nan")
        acc.append(dict(model=m, flux=f, cls=c, n_tasks=len(shards[(m, f, c)]),
                        **{q: int(v) if q not in ("w_acc", "w_acc_sq") else v for q, v in s.items()},
                        acceptance=w / n if n else float("nan"), ci_lo=lo, ci_hi=hi,
                        acceptance_attempts=k / s["attempts"] if s["attempts"] else float("nan")))
    acc_df = pd.DataFrame(acc)
    acc_df.to_csv(os.path.join(out_dir, "abc_acceptance.csv"), index=False)
    missing = {}
    if expect is not None:
        for (m, f, c), ids in sorted(shards.items()):
            miss = [i for i in range(expect) if i not in ids]
            if miss:
                missing[f"{m}_{f}_{c}"] = miss
    return {"n_rows": len(df), "format": fmt, "missing": missing, "acceptance": acc_df}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--out-dir", default="results/illex")
    a = ap.parse_args(argv)
    info = collect(a.dir, a.expect, a.out_dir)
    print(f"{info['n_rows']} rows -> {a.out_dir}/abc_table.{info['format']}")
    if info["format"] != "parquet":
        print("NOTE: pyarrow not available in this environment; wrote .csv.gz instead of .parquet")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(info["acceptance"].to_string(index=False))
    if info["missing"]:
        print("MISSING task ids:")
        for k, v in info["missing"].items():
            print(f"  {k}: {v}")
    elif a.expect is not None:
        print(f"all task ids 0..{a.expect - 1} present")


if __name__ == "__main__":
    main()
