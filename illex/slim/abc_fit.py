#!/usr/bin/env python
"""ABC inference from the simulation table.

Rejection ABC with local-linear regression adjustment (Beaumont et al. 2002),
plus the model comparisons the manuscript needs:

  Q1  Can the inversion persist without selection?
      -> P(s = 0 | data) from the posterior mass on the neutral atom, versus the
         prior weight. Reported as a Bayes factor.
  Q2  How old is the inversion?
      -> marginal posterior for t_inv, with the caveat that it scales inversely
         with mu = 3e-9.
  Q3  Why is p_inv intermediate?
      -> joint posterior for (s, h); h > 1 is overdominance. Only meaningful on
         the s > 0 subset, since h is unidentifiable when s = 0.

Distances are computed on statistics standardized by their PRIOR-PREDICTIVE
median absolute deviation, not standard deviation: the SFS-shape bins are
strongly non-normal and a few extreme simulations would otherwise dominate the
scaling and silently reweight the whole statistic vector.

Usage:
  python -m illex.slim.abc_fit --sims results/abc/sims_all.tsv \\
      --observed results/illex/abc_observed.json --tol 0.005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def robust_scale(x: np.ndarray) -> np.ndarray:
    """MAD-based scale per column, with a floor so constant columns can't blow up."""
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    floor = np.nanmax(np.abs(x), axis=0) * 1e-6 + 1e-12
    return np.maximum(mad, floor)


def load_sims(path: Path, use_absolute: bool) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, sep="\t")
    n_all = len(df)
    df = df[df.status == "ok"].copy()
    stats = list(C.STAT_NAMES)
    if use_absolute:
        stats += list(C.STAT_NAMES_ABSOLUTE)
    df = df.dropna(subset=stats + C.PARAM_NAMES)
    print(f"simulations: {n_all:,} total, {len(df):,} usable "
          f"({100 * len(df) / max(1, n_all):.1f}%)")
    if n_all - len(df):
        drop = pd.read_csv(path, sep="\t")
        vc = drop[drop.status != "ok"].status.value_counts()
        print("  dropped by status:")
        for k, v in vc.items():
            print(f"    {k}: {v:,}")
    return df, stats


def abc(df: pd.DataFrame, stats: list[str], obs: dict, tol: float,
        regression: bool = True) -> pd.DataFrame:
    """Rejection ABC. Returns the accepted table with regression-adjusted params."""
    S = df[stats].to_numpy(dtype=float)
    y = np.array([obs[s] for s in stats], dtype=float)
    scale = robust_scale(S)
    d = np.sqrt(np.nansum(((S - y) / scale) ** 2, axis=1))

    n_acc = max(20, int(round(tol * len(df))))
    idx = np.argsort(d)[:n_acc]
    acc = df.iloc[idx].copy()
    acc["distance"] = d[idx]
    print(f"accepted {n_acc:,} of {len(df):,} (tol={tol}) "
          f"distance <= {d[idx].max():.4f}")

    if regression:
        # Epanechnikov-weighted local linear adjustment toward the observed point.
        Sa = (S[idx] - y) / scale
        h = d[idx].max()
        w = np.maximum(0.0, 1.0 - (d[idx] / h) ** 2)
        X = np.column_stack([np.ones(len(idx)), Sa])
        W = np.diag(w)
        for p in C.PARAM_NAMES:
            v = acc[p].to_numpy(dtype=float)
            # Adjust log-scale parameters in log space; s and p_flux have atoms
            # at 0, so those are left alone (a regression through an atom is
            # meaningless).
            if p in ("t_inv", "p_start"):
                z = np.log(v)
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ z, rcond=None)[0]
                adj = np.exp(z - Sa @ beta[1:])
            elif p == "h":
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ v, rcond=None)[0]
                adj = v - Sa @ beta[1:]
            else:
                adj = v
            # Clip to the prior support. Local-linear adjustment is a first-order
            # extrapolation and will happily push draws OUTSIDE the prior --
            # observed on synthetic data, where p_start ran to 0.61 against a
            # prior ceiling of 0.4. Reporting a posterior interval wider than the
            # prior would be an artifact, not an inference.
            kind, lo, hi = C.PRIORS[p]
            # Priors with an atom at zero have 0 in their support, so the lower
            # clip bound is 0, NOT the log-uniform floor. Clipping those up to
            # the floor would silently convert every s = 0 draw into
            # s = 1e-7 and destroy the neutral atom that question 1 rests on.
            if kind.endswith("_atom0"):
                lo = 0.0
            n_out = int(np.sum((adj < lo) | (adj > hi)))
            if n_out:
                print(f"  {p}: clipped {n_out}/{len(adj)} regression-adjusted "
                      f"draws to prior support [{lo:.4g}, {hi:.4g}]")
            acc[p + "_adj"] = np.clip(adj, lo, hi)
    return acc


def report(acc: pd.DataFrame, df: pd.DataFrame) -> dict:
    out = {"n_accepted": int(len(acc))}
    print("\n" + "=" * 74)
    print("MARGINAL POSTERIORS (regression-adjusted where applicable)")
    for p in C.PARAM_NAMES:
        col = p + "_adj" if (p + "_adj") in acc else p
        v = acc[col].to_numpy(dtype=float)
        q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        out[p] = {"median": float(q[2]),
                  "ci95": [float(q[0]), float(q[4])],
                  "iqr": [float(q[1]), float(q[3])]}
        print(f"  {p:<9} median={q[2]:<12.4g} 95% CI [{q[0]:.4g}, {q[4]:.4g}]")

    # Q1: neutrality. Bayes factor for s == 0 versus s > 0.
    post0 = float((acc.s == 0).mean())
    prior0 = float((df.s == 0).mean())
    print("\nQ1  NEUTRALITY")
    print(f"  prior  P(s=0) = {prior0:.3f}")
    print(f"  post   P(s=0) = {post0:.3f}")
    if 0 < post0 < 1 and 0 < prior0 < 1:
        bf = (post0 / (1 - post0)) / (prior0 / (1 - prior0))
        out["bayes_factor_neutral"] = float(bf)
        verdict = ("favours NEUTRAL" if bf > 3 else
                   "favours SELECTION" if bf < 1 / 3 else "INCONCLUSIVE")
        print(f"  BF(neutral vs selected) = {bf:.2f}  -> {verdict}")
    else:
        out["bayes_factor_neutral"] = None
        print("  BF undefined (posterior mass entirely on one model)")
    out["post_p_neutral"] = post0
    out["prior_p_neutral"] = prior0

    # Q2: age. Generations == years for this annual species.
    col = "t_inv_adj" if "t_inv_adj" in acc else "t_inv"
    tq = np.percentile(acc[col].to_numpy(dtype=float), [2.5, 50, 97.5])
    print("\nQ2  AGE")
    print(f"  t_inv = {tq[1]:,.0f} generations "
          f"[{tq[0]:,.0f}, {tq[2]:,.0f}] 95% CI")
    print(f"  = {tq[1] / 1000:,.0f} ky (1 generation = 1 year)")
    print("  NOTE: inversely proportional to mu = 3e-9; quote mu with the age.")

    # Q3: overdominance, on the selected subset only.
    sel = acc[acc.s > 0]
    print("\nQ3  MECHANISM (selected subset only; h is unidentifiable at s=0)")
    if len(sel) >= 20:
        hcol = "h_adj" if "h_adj" in sel else "h"
        hv = sel[hcol].to_numpy(dtype=float)
        p_over = float((hv > 1.0).mean())
        hq = np.percentile(hv, [2.5, 50, 97.5])
        out["h_selected"] = {"median": float(hq[1]),
                             "ci95": [float(hq[0]), float(hq[2])],
                             "p_overdominant": p_over}
        print(f"  n={len(sel):,}  h median={hq[1]:.3f} "
              f"[{hq[0]:.3f}, {hq[2]:.3f}]")
        print(f"  P(h > 1 | s > 0, data) = {p_over:.3f}  "
              f"({'overdominance supported' if p_over > 0.9 else 'not resolved'})")
    else:
        out["h_selected"] = None
        print(f"  only {len(sel)} accepted draws with s>0 -- not enough to say")

    # Flux, reported as a bound per the design's standing instruction.
    fl = acc.p_flux.to_numpy(dtype=float)
    p_zero = float((fl == 0).mean())
    upper = float(np.percentile(fl, 95))
    out["p_flux"] = {"post_p_zero": p_zero, "upper95": upper}
    print("\nFLUX (reported as a bound, never a point estimate)")
    print(f"  P(p_flux = 0 | data) = {p_zero:.3f}; 95th pct = {upper:.3g}")
    print("=" * 74)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.005,
                    help="accepted fraction of usable simulations")
    ap.add_argument("--use-absolute", action="store_true",
                    help="include absolute pi/dxy; needs the nuisance scale in "
                         "the observed file (see NOTES sec 8.3)")
    ap.add_argument("--no-regression", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df, stats = load_sims(args.sims, args.use_absolute)
    obs = json.loads(args.observed.read_text())["stats"]
    missing = [s for s in stats if s not in obs]
    if missing:
        raise SystemExit(f"observed file lacks statistics: {missing}")

    acc = abc(df, stats, obs, args.tol, regression=not args.no_regression)
    summary = report(acc, df)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        acc.to_csv(args.out.with_suffix(".accepted.tsv"), sep="\t", index=False)
        print(f"\nwrote {args.out} and {args.out.with_suffix('.accepted.tsv')}")


if __name__ == "__main__":
    main()
