#!/usr/bin/env python
"""Growth-arm L-invariance check (spec amendment A8).

THE PREMISE UNDER TEST. The whole design rescales the inversion: a 20 Mb
inversion is simulated at 30-75 kb on the assumption that per-site pi and dxy
depend on Ne, mu and t_inv but NOT on L. If that fails, every fitted ratio --
and therefore the age -- is biased by an unknown amount.

WHY THIS RE-RUN. Task 5's rho ladder reported the premise "empirically
verified", but only on the CONSTANT arm. On the GROWTH arm -- the arm the
fitted statistics must use, since the null has to carry the expansion -- the
ladder found |r| ~ 0.65-0.74 for both ratios against L, driven almost entirely
by the smallest rung. With ONE replicate per rung it is impossible to say
whether that is Monte Carlo noise shrinking as L grows or a real systematic
bias. This script settles it with many replicates per L and CIs.

WHAT WOULD CAUSE A GENUINE SMALL-L BIAS, and why it is not expected here.
Sites within a recombination "escape length" of a breakpoint can recombine onto
the other arrangement's background, leaking across the barrier and pulling the
ratios toward panmixia. That length is where recombination outruns coalescence,
d ~ 1/(2*Ne*r) = 29.4 bp on the growth arm. The smallest rung's inversion body
is 2,350 bp -- 80x that -- so leakage should be negligible at every L tested.
The prediction is therefore FLAT MEANS with variance shrinking in L. Stated in
advance so the test can fail.

Two things are measured, and only the second is decision-relevant:

1. Is there a detectable slope of each ratio against log10(L)?
2. **The extrapolation that actually matters.** Production fits at L ~ 30-75 kb
   but the claim is about a 20 Mb inversion, i.e. 2.4 decades beyond the largest
   rung. The slope's 95% CI is converted into a bound on the induced bias over
   that extrapolation. A slope statistically indistinguishable from zero is NOT
   the same as a slope tight enough to extrapolate 2.4 decades; a wide CI can be
   "not significant" and still permit a large bias. Report the bound, not the
   p-value.

Interval-restricted throughout (A4): the pilot ladder used interval=None and
its ratio columns are flank-diluted, so its numbers are not comparable to these.

Run from the repo root AS A MODULE -- the venv's ``msinv.pth`` still points at
the pre-move ``/home/ssmall/inversion_sims/files``, so ``illex`` is only
importable when the repo root is on sys.path, which ``-m`` guarantees and
``python path/to/script.py`` does not:
  .venv/bin/python -m illex.scripts.l_invariance_growth
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats as sps

from illex import model, stats
from illex.demography import PRESENT_NE_GROWTH

ARM = "growth"
RHO_RUNGS = [200, 500, 1000, 2000, 5000]
REPS = 40
# Current best growth-arm point (spec A1/A9): t_inv ~ 7-8e5, p_start ~ 0.15.
T_INV = 8.0e5
P_START = 0.15
GAMMA = 1e-15
R = 2.5e-9
MAX_WORKERS = 24              # shared device; ~1 GB/worker at the largest L
TARGET_L = 20_000_000.0       # the real inversion, for the extrapolation bound

OUT_CSV = Path("results/illex/l_invariance_growth.csv")
OUT_JSON = Path("results/illex/l_invariance_growth.json")
RATIOS = ["pi_i_over_pi_s", "dxy_over_pi_i"]
LEVELS = ["pi_i", "pi_s", "dxy", "fst"]


def seq_length_for(rho: float, present_ne: float = PRESENT_NE_GROWTH) -> int:
    return int(round(rho / (4.0 * present_ne * R)))


def run_one(job) -> dict:
    rho, rep = job
    L = seq_length_for(rho)
    # Seeds unique across (rung, rep) so no realization is reused. Replicates
    # are independent, not paired across L -- msinv realizations at different
    # L are not comparable rep-for-rep anyway.
    seed = 100_000 + 1_000 * RHO_RUNGS.index(rho) + rep
    t0 = time.time()
    sim = model.build_inversion_sim(
        arm=ARM, seq_length=L, t_inv=T_INV, gamma=GAMMA,
        p_start=P_START, seed=seed,
    )
    ts = sim.simulate()
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    st = stats.arrangement_stats(
        ts, i_nodes, s_nodes, interval=model.inversion_interval(sim))
    return {
        "rho": rho, "seq_length": L, "rep": rep, "seed": seed,
        "num_trees": ts.num_trees, "wall_s": round(time.time() - t0, 2),
        **{k: st[k] for k in RATIOS + LEVELS},
    }


def ols_vs_logL(L, y):
    """OLS of y on log10(L), with HC3 heteroscedasticity-robust SE.

    Regressing on log10(L) rather than L because the rungs are geometrically
    spaced and the extrapolation of interest (75 kb -> 20 Mb) is in decades.

    The residual variance is expected to SHRINK with L (more independent trees
    per replicate), which is the whole reason a single replicate per rung was
    uninformative. That heteroscedasticity makes the classical OLS SE wrong, so
    HC3 is reported alongside it and drives the CI. HC3 rather than HC0 because
    it is the better-behaved small-sample variant.
    """
    x = np.log10(np.asarray(L, dtype=float))
    y = np.asarray(y, dtype=float)
    n = len(x)
    res = sps.linregress(x, y)

    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    resid = y - (res.intercept + res.slope * x)
    # HC3: leverage-corrected sandwich estimator for the slope.
    h = 1.0 / n + xc ** 2 / sxx
    se_hc3 = float(np.sqrt(((xc ** 2) * (resid / (1.0 - h)) ** 2).sum()) / sxx)

    dof = n - 2
    tcrit = float(sps.t.ppf(0.975, dof))
    return {
        "slope_per_decade": float(res.slope),
        "slope_se_classical": float(res.stderr),
        "slope_se_hc3": se_hc3,
        "t_hc3": float(res.slope / se_hc3) if se_hc3 > 0 else float("nan"),
        "p_value_classical": float(res.pvalue),
        "p_value_hc3": float(
            2.0 * sps.t.sf(abs(res.slope / se_hc3), dof)) if se_hc3 > 0
            else float("nan"),
        "ci95": [float(res.slope - tcrit * se_hc3),
                 float(res.slope + tcrit * se_hc3)],
        "n": int(n),
    }


def wls_on_means(seq_lengths, means, sems):
    """WLS slope from the per-L MEANS, weighted by 1/sem^2.

    Independent cross-check on ols_vs_logL: it collapses each L to one point,
    so it is immune to the within-L variance structure entirely. If the two
    disagree, the heteroscedasticity is driving the result and neither should
    be trusted without more replicates.
    """
    x = np.log10(np.asarray(seq_lengths, dtype=float))
    y = np.asarray(means, dtype=float)
    w = 1.0 / np.asarray(sems, dtype=float) ** 2
    sw = w.sum()
    xbar = float((w * x).sum() / sw)
    ybar = float((w * y).sum() / sw)
    sxx = float((w * (x - xbar) ** 2).sum())
    slope = float((w * (x - xbar) * (y - ybar)).sum() / sxx)
    se = float(np.sqrt(1.0 / sxx))
    dof = len(x) - 2
    tcrit = float(sps.t.ppf(0.975, dof)) if dof > 0 else float("nan")
    return {
        "slope_per_decade": slope, "slope_se": se,
        "ci95": [slope - tcrit * se, slope + tcrit * se],
        "p_value": float(2.0 * sps.t.sf(abs(slope / se), dof)) if dof > 0
        else float("nan"),
        "n_points": int(len(x)),
    }


def summarize(rows: list[dict]) -> dict:
    out = {
        "config": {
            "arm": ARM, "t_inv": T_INV, "p_start": P_START, "gamma": GAMMA,
            "reps_per_L": REPS, "rho_rungs": RHO_RUNGS,
            "seq_lengths": [seq_length_for(r) for r in RHO_RUNGS],
            "interval_restricted": True, "target_L": TARGET_L,
        },
        "per_L": {}, "trend": {}, "extrapolation": {}, "endpoints": {},
    }

    by_rho = {r: [d for d in rows if d["rho"] == r] for r in RHO_RUNGS}
    for rho, grp in by_rho.items():
        entry = {"seq_length": grp[0]["seq_length"], "n": len(grp),
                 "mean_num_trees": float(np.mean([g["num_trees"] for g in grp])),
                 "mean_wall_s": float(np.mean([g["wall_s"] for g in grp]))}
        for k in RATIOS + LEVELS:
            v = np.array([g[k] for g in grp], dtype=float)
            entry[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                        "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}
        out["per_L"][str(rho)] = entry

    Lall = [d["seq_length"] for d in rows]
    # Production range only: the rungs the fit will actually use. A trend
    # confined to the smallest rungs is harmless if production never goes there.
    prod = [d for d in rows if d["seq_length"] >= 14_000]
    for k in RATIOS + LEVELS:
        out["trend"][k] = {
            "all_L": ols_vs_logL(Lall, [d[k] for d in rows]),
            "production_L_ge_14kb": ols_vs_logL(
                [d["seq_length"] for d in prod], [d[k] for d in prod]),
            "wls_on_means": wls_on_means(
                [out["per_L"][str(r)]["seq_length"] for r in RHO_RUNGS],
                [out["per_L"][str(r)][k]["mean"] for r in RHO_RUNGS],
                [out["per_L"][str(r)][k]["sem"] for r in RHO_RUNGS]),
        }

    # Decision-relevant: how much bias could the slope's CI permit over the
    # 75 kb -> 20 Mb extrapolation, as a fraction of the mean at the largest L?
    biggest = by_rho[RHO_RUNGS[-1]]
    L_big = biggest[0]["seq_length"]
    decades = float(np.log10(TARGET_L / L_big))
    for k in RATIOS:
        tr = out["trend"][k]["all_L"]
        base = float(np.mean([g[k] for g in biggest]))
        lo, hi = (c * decades for c in tr["ci95"])
        worst = max(abs(lo), abs(hi))
        out["extrapolation"][k] = {
            "decades_to_20Mb": decades,
            "mean_at_largest_L": base,
            "point_bias": float(tr["slope_per_decade"] * decades),
            "bias_ci95": [float(lo), float(hi)],
            "worst_case_abs_bias": float(worst),
            "worst_case_pct_of_mean": float(100.0 * worst / base),
        }

    # Smallest vs largest L, Welch (unequal variances expected by design).
    small = by_rho[RHO_RUNGS[0]]
    for k in RATIOS + LEVELS:
        a = np.array([g[k] for g in small], dtype=float)
        b = np.array([g[k] for g in biggest], dtype=float)
        t, p = sps.ttest_ind(a, b, equal_var=False)
        out["endpoints"][k] = {
            "small_L": small[0]["seq_length"], "large_L": L_big,
            "small_mean": float(a.mean()), "large_mean": float(b.mean()),
            "small_sd": float(a.std(ddof=1)), "large_sd": float(b.std(ddof=1)),
            "welch_t": float(t), "welch_p": float(p),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    jobs = [(rho, rep) for rho in RHO_RUNGS for rep in range(args.reps)]
    print(f"growth-arm L-invariance: {len(RHO_RUNGS)} L values x {args.reps} "
          f"reps = {len(jobs)} sims, {args.workers} workers", flush=True)
    print(f"  t_inv={T_INV:,.0f}  p_start={P_START}  gamma={GAMMA}  "
          f"interval-restricted", flush=True)
    for rho in RHO_RUNGS:
        print(f"  rho={rho:>5} -> L={seq_length_for(rho):>8,} bp", flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(run_one, jobs), start=1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"    {i}/{len(jobs)} done "
                      f"({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda d: (d["rho"], d["rep"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    with OUT_JSON.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"{'L (bp)':>9} {'trees':>8} "
          f"{'pi_I/pi_S mean+-sem':>24} {'dxy/pi_I mean+-sem':>24}")
    for rho in RHO_RUNGS:
        e = summary["per_L"][str(rho)]
        a, b = e["pi_i_over_pi_s"], e["dxy_over_pi_i"]
        print(f"{e['seq_length']:>9,} {e['mean_num_trees']:>8,.0f} "
              f"{a['mean']:>15.4f} +-{a['sem']:.4f} "
              f"{b['mean']:>15.4f} +-{b['sem']:.4f}")

    # Levels, not just ratios: pi_S is the near-panmictic class, so a drift in
    # it with L would indicate the per-site calibration itself moves, which
    # would be a worse problem than a drift in the ratios.
    print(f"\n{'L (bp)':>9} {'pi_I':>12} {'pi_S':>12} {'dxy':>12} {'Fst':>10}")
    for rho in RHO_RUNGS:
        e = summary["per_L"][str(rho)]
        print(f"{e['seq_length']:>9,} {e['pi_i']['mean']:>12.6f} "
              f"{e['pi_s']['mean']:>12.6f} {e['dxy']['mean']:>12.6f} "
              f"{e['fst']['mean']:>10.4f}")

    print("\nslope vs log10(L)  [0 = invariant]  (CIs are HC3-robust)")
    for k in RATIOS:
        for scope in ("all_L", "production_L_ge_14kb"):
            t = summary["trend"][k][scope]
            print(f"  {k:<18} {scope:<22} "
                  f"slope={t['slope_per_decade']:+.4f}/decade  "
                  f"95% CI [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  "
                  f"p={t['p_value_hc3']:.3f}")
        w = summary["trend"][k]["wls_on_means"]
        print(f"  {k:<18} {'wls_on_means (check)':<22} "
              f"slope={w['slope_per_decade']:+.4f}/decade  "
              f"95% CI [{w['ci95'][0]:+.4f}, {w['ci95'][1]:+.4f}]  "
              f"p={w['p_value']:.3f}")

    print(f"\nEXTRAPOLATION to L={TARGET_L:,.0f} "
          f"({summary['extrapolation'][RATIOS[0]]['decades_to_20Mb']:.2f} "
          f"decades beyond the largest rung)")
    for k in RATIOS:
        x = summary["extrapolation"][k]
        print(f"  {k:<18} worst-case bias {x['worst_case_abs_bias']:.4f} "
              f"= {x['worst_case_pct_of_mean']:.1f}% of the mean "
              f"({x['mean_at_largest_L']:.4f})")

    print("\nsmallest vs largest L (Welch)")
    for k in RATIOS:
        e = summary["endpoints"][k]
        print(f"  {k:<18} {e['small_mean']:.4f} (sd {e['small_sd']:.4f}) vs "
              f"{e['large_mean']:.4f} (sd {e['large_sd']:.4f})  "
              f"p={e['welch_p']:.3f}")
    print("=" * 78)
    print(f"total wall {time.time() - t0:.0f}s -> {OUT_CSV}, {OUT_JSON}")


if __name__ == "__main__":
    main()
