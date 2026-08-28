#!/usr/bin/env python
"""Block-jackknife SE on f1, the third fitted target.

    .venv/bin/python -m illex.scripts.sfs_f1_jackknife

WHY THIS EXISTS
---------------
The decline fit (NOTES sec 8.16) is scored against three statistics. Two of
them carry 1 Mb block-jackknife SEs from ``empirical_jackknife.py`` -- 3.5% on
pi_I/pi_S and 1.5% on dxy/pi_I. The third, the ANGSD singleton ratio
f1(I)/f1(S) = 0.8256, is a bare point estimate with no error bar at all.

That gap did not matter much while f1 was the third of three roughly
concordant statistics. It matters now, because the question "is the decline
still ongoing?" is answered almost entirely by f1: across the fitted grid,
moving t_decline from 175 ka to 100 ka moves pi_I/pi_S by +2.5% but f1 by
+4.6%, so f1 is the statistic carrying the recent-time information. A
delta-chi-square on t_decline is uninterpretable until f1 has a variance.

THE ADDITIVITY GATE -- READ THIS BEFORE TRUSTING THE SE
-------------------------------------------------------
``realSFS -r`` accepts ONE contiguous region, so a delete-one-block spectrum
(the span minus a hole) cannot be requested directly. This script instead runs
each 1 Mb block separately (``.tmp/angsd_jack/run.sh``) and forms the
delete-one-block spectrum by summing the others.

That substitution is NOT free. realSFS estimates the SFS by EM over all sites
in its region, so the per-block estimates each converge to their own prior and
their sum is not identically the global estimate. Whether the difference
matters is an empirical question, so it is GATED: the summed spectrum is
compared against the global one already computed by ``run.sh`` in
``.tmp/angsd_sfs``. If f1 from the sum disagrees with f1 from the global EM by
more than GATE_TOL, the block route is not a valid stand-in and the script
refuses to report an SE rather than quoting one that is measuring EM drift.

THE ESTIMATOR
-------------
Same discipline as ``empirical_jackknife.py``: f1(I)/f1(S) is a ratio, so
blocks are deleted from NUMERATOR AND DENOMINATOR TOGETHER and the ratio
recomputed -- not averaged over per-block ratios, which is a different and
biased estimator.

    theta_jack = B*theta_hat - (B-1)*mean(theta_(-b))
    SE^2       = (B-1)/B * sum_b (theta_(-b) - mean(theta_(-b)))^2

THE CAVEAT THAT TRAVELS WITH IT
-------------------------------
Identical to the dxy caveat in ``empirical_jackknife.py``: blocks inside a
non-recombining inversion share one origin, so this is a MEASUREMENT error, a
lower bound on the uncertainty that should be propagated into a fit. It is
still far better than the nothing f1 has now.

I = the BB cluster, S = the AA cluster (polarization corrected, sec 8.15).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from illex.scripts.sfs_shape import PROJ, project_unfolded

JACK = Path(".tmp/angsd_jack")
GLOBAL = Path(".tmp/angsd_sfs")
OUT = Path("results/illex")

N_DIP = {"AA": 254, "BB": 95}
REGIONS = {"body": (60_500_000, 79_500_000), "control": (10_000_000, 30_000_000)}
BLOCK = 1_000_000
GATE_TOL = 0.02          # relative disagreement in f1 between sum-of-blocks
                         # and the global EM that we are willing to tolerate


def _read(f: Path, cls: str) -> np.ndarray:
    v = np.array([float(x) for x in f.read_text().split()], dtype=float)
    if v.size != 2 * N_DIP[cls] + 1:
        raise SystemExit(f"{f}: {v.size} entries, expected {2 * N_DIP[cls] + 1}")
    return v


def blocks(cls: str, region: str) -> dict[int, np.ndarray]:
    lo, hi = REGIONS[region]
    out = {}
    for start in range(lo, hi, BLOCK):
        f = JACK / f"{cls}.{region}.{start}.sfs"
        if not f.exists() or f.stat().st_size == 0:
            raise SystemExit(f"missing block {f} -- has run.sh finished?")
        out[start] = _read(f, cls)
    return out


def f1_of(raw: np.ndarray, cls: str) -> float:
    """Singleton fraction after exact projection to PROJ and folding."""
    p = project_unfolded(raw, 2 * N_DIP[cls], PROJ)
    return float(p[0] / p.sum())


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Block-jackknife SE on the ANGSD singleton ratio f1(I)/f1(S)")
    emit("I = BB (inverted, derived), S = AA (standard).  1 Mb blocks.")
    emit()

    blk = {(c, r): blocks(c, r) for c in ("AA", "BB") for r in REGIONS}

    # ---- the gate ------------------------------------------------------
    emit("=" * 74)
    emit("GATE: does the sum of per-block EM estimates reproduce the global EM?")
    emit("=" * 74)
    emit("  realSFS optimises over its whole region, so per-block estimates need")
    emit("  not sum to the global one. If they do not, the block route is not a")
    emit("  valid stand-in for a true leave-one-out and no SE is reported.")
    emit()
    emit(f"  {'arm':<12s} {'f1 global':>10s} {'f1 sum-of-blk':>14s} {'rel diff':>10s}")
    worst = 0.0
    for c in ("AA", "BB"):
        for r in REGIONS:
            g = _read(GLOBAL / f"{c}.{r}.sfs", c)
            s = np.sum(list(blk[(c, r)].values()), axis=0)
            fg, fs = f1_of(g, c), f1_of(s, c)
            d = abs(fs - fg) / fg
            worst = max(worst, d)
            emit(f"  {c + '/' + r:<12s} {fg:10.4f} {fs:14.4f} {d:9.2%}")
    emit()
    if worst > GATE_TOL:
        emit(f"  GATE FAILED: worst relative disagreement {worst:.2%} exceeds "
             f"{GATE_TOL:.0%}.")
        emit("  Per-block EM does not reconstruct the global spectrum, so the")
        emit("  delete-one-block sums are not leave-one-out estimates. Either run")
        emit("  true leave-one-out realSFS jobs (2 intervals each, needs a region")
        emit("  file realSFS does not support) or widen the blocks. NO SE REPORTED.")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "sfs_f1_jackknife.txt").write_text("\n".join(lines) + "\n")
        raise SystemExit(1)
    emit(f"  GATE PASSED (worst {worst:.2%} <= {GATE_TOL:.0%}). Proceeding.")
    emit()

    # ---- the jackknife --------------------------------------------------
    res = {}
    for r in REGIONS:
        starts = sorted(blk[("BB", r)])
        b = len(starts)

        def ratio(drop=None, reg=r):
            i = np.sum([v for s, v in blk[("BB", reg)].items() if s != drop],
                       axis=0)
            s_ = np.sum([v for s, v in blk[("AA", reg)].items() if s != drop],
                        axis=0)
            return f1_of(i, "BB") / f1_of(s_, "AA")

        full = ratio()
        part = np.array([ratio(s) for s in starts])
        mp = part.mean()
        est = b * full - (b - 1) * mp
        se = float(np.sqrt((b - 1) / b * ((part - mp) ** 2).sum()))
        res[r] = {"n_blocks": b, "point": float(full), "jack": float(est),
                  "se": se}

        # per-class f1 as well -- free, and the body/control contrast uses it
        cls_f1 = {}
        for c in ("AA", "BB"):
            fu = f1_of(np.sum(list(blk[(c, r)].values()), axis=0), c)
            pa = np.array([f1_of(np.sum([v for s, v in blk[(c, r)].items()
                                         if s != d], axis=0), c)
                           for d in starts])
            m = pa.mean()
            cls_f1[c] = {"jack": float(b * fu - (b - 1) * m),
                         "se": float(np.sqrt((b - 1) / b * ((pa - m) ** 2).sum()))}
        res[r]["per_class"] = cls_f1

    emit("=" * 74)
    emit("RESULT")
    emit("=" * 74)
    for r in REGIONS:
        v = res[r]
        emit(f"\n{r.upper()}  ({v['n_blocks']} blocks of 1 Mb)")
        emit(f"  f1(BB) = {v['per_class']['BB']['jack']:.4f} +- "
             f"{v['per_class']['BB']['se']:.4f}")
        emit(f"  f1(AA) = {v['per_class']['AA']['jack']:.4f} +- "
             f"{v['per_class']['AA']['se']:.4f}")
        emit(f"  ratio f1(I)/f1(S) = {v['jack']:.4f} +- {v['se']:.4f}  "
             f"({100 * v['se'] / v['jack']:.1f}%)   [point {v['point']:.4f}]")
    emit()
    c, b_ = res["control"], res["body"]
    emit("  The CONTROL ratio is the sanity check: AA and BB are exchangeable")
    emit("  outside the inversion, so it must sit at 1 within its own SE.")
    emit(f"  control ratio {c['jack']:.4f} +- {c['se']:.4f} -> "
         f"{abs(c['jack'] - 1) / c['se']:.2f} SE from 1")
    emit()

    # ---- the control does NOT sit at 1, and that is informative ---------
    emit("=" * 74)
    emit("THE CONTROL FAILS ITS OWN CHECK -- and the fix is a ratio of ratios")
    emit("=" * 74)
    emit("  1.6% is small, but the jackknife SE is smaller still (0.1%), so the")
    emit("  offset is a SYSTEMATIC, not noise: it is near-identical in every")
    emit("  block, which is exactly why the spatial jackknife cannot see it.")
    emit("  AA and BB are exchangeable in the collinear region by construction,")
    emit("  so a persistent f1 difference there is a property of the two SAMPLE")
    emit("  SETS, not of the genome. The obvious candidate is n: 254 vs 95")
    emit("  individuals, so realSFS's EM estimates the two source spectra with")
    emit("  different bias before either is projected to n = 20. Coverage")
    emit("  differences between the two bamlists would do the same. This has NOT")
    emit("  been diagnosed -- confirming it needs an AA SAF rebuilt at n = 95")
    emit("  over the control region, which is a new ANGSD run.")
    emit()
    emit("  It does not need to be diagnosed to be removed. The model side has")
    emit("  n_i = n_s = 100, so it carries NO class-size asymmetry; the empirical")
    emit("  ratio carries one. Dividing the body ratio by the control ratio")
    emit("  cancels any class-level systematic shared by the two regions,")
    emit("  whatever its cause -- the same logic already used for the per-class")
    emit("  body-vs-control comparison (NOTES sec 8.5).")
    emit()
    rr = b_["jack"] / c["jack"]
    rel = float(np.hypot(b_["se"] / b_["jack"], c["se"] / c["jack"]))
    rr_se = rr * rel
    res["ratio_of_ratios"] = {"value": float(rr), "se": float(rr_se)}
    emit(f"  body ratio          {b_['jack']:.4f} +- {b_['se']:.4f}")
    emit(f"  control ratio       {c['jack']:.4f} +- {c['se']:.4f}")
    emit(f"  CALIBRATED TARGET   {rr:.4f} +- {rr_se:.4f}  ({100 * rel:.1f}%)")
    emit()
    emit(f"  This shifts the fitted f1 target from {b_['jack']:.4f} to {rr:.4f},")
    emit(f"  i.e. by {100 * (rr / b_['jack'] - 1):+.1f}% -- about "
         f"{abs(rr - b_['jack']) / b_['se']:.1f} of the spatial SE. The decline")
    emit("  fit was scored against the UNCALIBRATED 0.8256, so it should be")
    emit("  rescored. Per-class body-vs-control L1 values are unaffected: they")
    emit("  compare each class against itself and the systematic cancels there")
    emit("  already.")
    emit()
    emit("  Blocks are disjoint between body and control, so the two jackknives")
    emit("  are treated as independent and their relative SEs added in quadrature.")
    emit("  The assumption that does the work is that the class-level offset is")
    emit("  the SAME in both regions. That is plausible -- same individuals, same")
    emit("  bamlists, same n -- but it is an assumption, not a measurement.")
    emit()
    emit("=" * 74)
    emit("WHAT THIS MEANS FOR THE t_decline TEST")
    emit("=" * 74)
    emit(f"  f1 target = {rr:.4f} +- {rr_se:.4f} (calibrated)  "
         f"[uncalibrated {b_['jack']:.4f} +- {b_['se']:.4f}]")
    emit("  Fitted sensitivity (t_inv = 850 ka, t_fall = 100 ky):")
    emit("    t_decline 175 ka -> f1 0.827   (the fit)")
    emit("    t_decline 100 ka -> f1 0.865   (+0.038)")
    emit(f"  so 75 ky of t_decline moves f1 by 0.038 = "
         f"{0.038 / rr_se:.1f} jackknife SE.")
    emit("  Resolution on t_decline is therefore roughly "
         f"{75 * rr_se / 0.038:,.0f} ky per SE, which is what sets whether an")
    emit("  ongoing decline (t_decline -> 0) is separable from the fitted one.")
    emit()
    emit("  CAVEAT, same as dxy in empirical_jackknife.py: blocks inside a")
    emit("  non-recombining inversion share one origin, so this is a measurement")
    emit("  error and a LOWER bound on the uncertainty for a fit.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sfs_f1_jackknife.txt").write_text("\n".join(lines) + "\n")
    json.dump(res, (OUT / "sfs_f1_jackknife.json").open("w"), indent=2)
    print(f"\nwrote {OUT}/sfs_f1_jackknife.{{txt,json}}")


if __name__ == "__main__":
    main()
