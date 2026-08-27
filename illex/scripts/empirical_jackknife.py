#!/usr/bin/env python
"""Block-jackknife standard errors on the two fitted empirical targets.

    CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
        illex/scripts/empirical_jackknife.py

WHY THIS EXISTS
---------------
``illex.empirical`` carries pi_I/pi_S = 0.744 and dxy/pi_I = 1.846 as bare
point values with no uncertainty, and the balancing-selection fit
(NOTES sec 7.5) turns them into an age. Its reported sensitivity is
d(dxy/pi_I)/d(t_inv) = 0.283 per 1e5 generations, i.e. a 1% error in the
target moves the age ~6,500 years -- so quoting the age to five figures while
the target has no error bar at all is unjustified. This puts an error bar on
the targets.

THE ESTIMATOR
-------------
Both targets are ratios of region-wide quantities, so the jackknife has to
delete blocks from the NUMERATOR AND DENOMINATOR TOGETHER and recompute the
ratio -- not average per-window ratios, which is a different and biased
estimator. With equal-size windows and span normalisation by nominal window
length, the region-wide pi is exactly the unweighted mean of the window pi
values (both are total pairwise differences over total span), so a
delete-one-block region estimate is the mean over retained windows.

    theta_jack = B*theta_hat - (B-1)*mean(theta_(-b))          (bias-corrected)
    SE^2       = (B-1)/B * sum_b (theta_(-b) - mean(theta_(-b)))^2

Accessibility cancels: pi_AA, pi_BB and dxy all carry the same denominator, so
the two ratios are mask-free. That is why they were chosen as the targets
(NOTES sec 8.3) and it means this run needs no mask.

BLOCK SIZE IS ITSELF THE DIAGNOSTIC
-----------------------------------
A block jackknife is only valid once blocks are larger than the correlation
length. Rather than assert a block size, several are run: if the SE has
plateaued, the blocks are big enough; if it is still climbing, they are not.

THE CAVEAT THAT MUST TRAVEL WITH THE dxy NUMBER
-----------------------------------------------
Blocks inside a non-recombining inversion are **not independent replicates of
the evolutionary process**, and the two targets are affected differently:

* pi_AA and pi_BB are within-arrangement. Recombination is unsuppressed within
  a homokaryotype (NOTES sec 8.0), so at >=250 kb the blocks are close to
  independent and their jackknife SE is meaningful.
* dxy is across the barrier. Every block shares the *same* single origin and
  therefore the same t_inv. The jackknife measures how much dxy varies along
  the region, NOT how much the coalescent process could have produced. It is a
  measurement error, and it is a LOWER bound on the uncertainty that should be
  propagated into an age.

The process variance is the larger term and is already available from the
model side (per-replicate SD in ``results/illex/fit_balancing.csv``); this
script reports both so the age interval can be built from the right one.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
from pg_gpu import HaplotypeMatrix, windowed_analysis

T = Path(".tmp/illex_chr2")
OUT = Path("results/illex")
REGION = "2:60000000-80000000"
VCF = T / "inv.vcf.gz"

BASE_WINDOW = 100_000          # fine grid; blocks are built by grouping these
BLOCK_SIZES = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]

# Nominal breakpoints. The differentiated body is narrower (NOTES sec 4.2), so
# the targets are also recomputed over the Fst-defined extent, which is what
# the interval-restricted simulations actually correspond to.
INV_START, INV_STOP = 60_040_617, 79_995_597
FST_DIFF_CUTOFF = 0.15


def windowed() -> pd.DataFrame:
    h = HaplotypeMatrix.from_vcf(str(VCF), region=REGION)
    h.load_pop_file(str(T / "pops.tsv"))
    n_aa = len(h.sample_sets["AA"])
    n_bb = len(h.sample_sets["BB"])
    print(f"loaded {h.num_variants:,} variants x {h.num_haplotypes:,} "
          f"haplotypes  (AA {n_aa}, BB {n_bb})  device={h.device}")

    kw = dict(window_size=BASE_WINDOW, step_size=BASE_WINDOW,
              missing_data="include")
    df = windowed_analysis(h, statistics=["fst", "dxy"],
                           populations=["AA", "BB"], **kw)
    # pi per population separately: requesting 'pi' alongside dxy/fst silently
    # returns pi for populations[0] only (pg_gpu bug, NOTES sec 11).
    for pop in ("AA", "BB"):
        p = windowed_analysis(h, statistics=["pi"], populations=[pop], **kw)
        df = df.merge(p[["window_id", "pi"]].rename(columns={"pi": f"pi_{pop}"}),
                      on="window_id", how="left")
    return df.rename(columns={"start": "window_start", "end": "window_stop"})


def ratios(sub: pd.DataFrame) -> tuple[float, float]:
    """(pi_I/pi_S, dxy/pi_I) from a set of windows, as ratios of sums.

    **I = the BB cluster, S = the AA cluster.** The polarization is reversed
    from what this project recorded for most of its history: the reference
    genome was assembled from an inverted individual, so the arrangement it
    carries -- BB -- is the inverted, derived one, at frequency 0.374
    (NOTES sec 8.15). The cluster labels are unchanged; only the interpretation
    is swapped. Before 2026-08-27 this function had pi_i = pi_AA, which is why
    every fit in sec 7 and 8.6-8.8 is invalid.
    """
    pi_i = sub.pi_BB.mean()
    pi_s = sub.pi_AA.mean()
    dxy = sub.dxy.mean()
    return float(pi_i / pi_s), float(dxy / pi_i)


def jackknife(df: pd.DataFrame, block_bp: int):
    """Delete-one-block jackknife over both ratios."""
    blk = (df.window_start // block_bp).to_numpy()
    ids = np.unique(blk)
    b = len(ids)
    full = np.array(ratios(df))
    partial = np.array([ratios(df[blk != i]) for i in ids])
    mean_partial = partial.mean(axis=0)
    est = b * full - (b - 1) * mean_partial
    se = np.sqrt((b - 1) / b * ((partial - mean_partial) ** 2).sum(axis=0))
    return b, full, est, se


def report(df: pd.DataFrame, label: str, fh) -> dict:
    def emit(s):
        print(s, flush=True)
        fh.write(s + "\n")

    emit(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    emit(f"windows: {len(df)}  span {int(df.window_start.min()):,}-"
         f"{int(df.window_stop.max()):,}  "
         f"variants {int(df.n_variants.sum()):,}")
    r, d = ratios(df)
    emit(f"point estimates:  pi_I/pi_S = {r:.4f}   dxy/pi_I = {d:.4f}   "
         f"(pi_I=pi_BB {df.pi_BB.mean():.6f}  pi_S=pi_AA {df.pi_AA.mean():.6f}  "
         f"dxy {df.dxy.mean():.6f})")
    emit("")
    emit(f"{'block':>9s} {'n_blk':>6s} {'pi_I/pi_S':>22s} {'dxy/pi_I':>22s}")
    rows = {}
    for bp in BLOCK_SIZES:
        if len(df) * BASE_WINDOW < 3 * bp:
            continue
        b, full, est, se = jackknife(df, bp)
        rows[bp] = {"n_blocks": b, "r": float(est[0]), "r_se": float(se[0]),
                    "d": float(est[1]), "d_se": float(se[1])}
        emit(f"{bp / 1e6:8.2f}M {b:6d} "
             f"{est[0]:10.4f} +- {se[0]:.4f} ({100 * se[0] / est[0]:4.1f}%) "
             f"{est[1]:10.4f} +- {se[1]:.4f} ({100 * se[1] / est[1]:4.1f}%)")
    return {"point": {"r": r, "d": d}, "blocks": rows,
            "n_windows": len(df),
            "pi_AA": float(df.pi_AA.mean()),
            "pi_BB": float(df.pi_BB.mean()),
            "dxy": float(df.dxy.mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = windowed()
    df.to_csv(OUT / "empirical_jackknife_windows.csv", index=False)

    # Nominal span: what illex.empirical's targets were measured over.
    nominal = df[(df.window_start >= INV_START - BASE_WINDOW)
                 & (df.window_stop <= INV_STOP + BASE_WINDOW)]
    # Differentiated body: what the interval-restricted simulations model.
    body = df[df.fst > FST_DIFF_CUTOFF]

    with (OUT / "empirical_jackknife.txt").open("w") as fh:
        fh.write("Block-jackknife SEs on the fitted empirical targets\n")
        res_nom = report(nominal, "NOMINAL SPAN 60.0-80.0 Mb "
                                  "(what illex.empirical quotes)", fh)
        res_body = report(body, f"DIFFERENTIATED BODY (Fst > {FST_DIFF_CUTOFF}) "
                                "-- the model's interval", fh)

        def emit(s):
            print(s, flush=True)
            fh.write(s + "\n")

        emit(f"\n{'=' * 74}\nWHAT THIS MEANS FOR THE AGE\n{'=' * 74}")
        emit("Age sensitivity from the fit: d(dxy/pi_I)/d(t_inv) = 0.283 per "
             "1e5 generations.")
        big = max(res_nom["blocks"])
        for name, res in (("nominal span", res_nom),
                          ("differentiated body", res_body)):
            if big not in res["blocks"]:
                big_l = max(res["blocks"])
            else:
                big_l = big
            b = res["blocks"][big_l]
            emit(f"\n{name} ({big_l / 1e6:.0f} Mb blocks, n={b['n_blocks']}):")
            emit(f"  dxy/pi_I = {b['d']:.4f} +- {b['d_se']:.4f}  "
                 f"-> +-{b['d_se'] / 0.283 * 1e5:,.0f} generations "
                 "(measurement only)")
            emit(f"  pi_I/pi_S = {b['r']:.4f} +- {b['r_se']:.4f}")
        emit("\nThis is a LOWER bound on the age uncertainty: blocks inside a "
             "non-recombining\ninversion share one origin, so they are not "
             "independent replicates of the\nprocess. The per-replicate model "
             "SD is the larger term -- see the header.")

        shift_d = res_body["point"]["d"] - res_nom["point"]["d"]
        emit(f"\nInterval choice moves dxy/pi_I by {shift_d:+.4f} "
             f"({100 * shift_d / res_nom['point']['d']:+.1f}%), i.e. "
             f"{shift_d / 0.283 * 1e5:+,.0f} generations of age.")
        emit("The simulations are interval-restricted to the inversion body, "
             "so the\ndifferentiated-body value is the like-for-like target.")

    pd.DataFrame([
        {"region": reg, "block_bp": bp, **v}
        for reg, res in (("nominal", res_nom), ("body", res_body))
        for bp, v in res["blocks"].items()
    ]).to_csv(OUT / "empirical_jackknife.csv", index=False)
    print(f"\nwrote {OUT}/empirical_jackknife.{{txt,csv}} and "
          f"{OUT}/empirical_jackknife_windows.csv")


if __name__ == "__main__":
    main()
