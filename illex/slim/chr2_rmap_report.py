#!/usr/bin/env python
"""Summarize a chr2 ReLERNN map and quantify the inversion's LD artifact.

Ready to run the moment the chr2 ReLERNN results exist:

  .venv/bin/python -m illex.slim.chr2_rmap_report \\
      --rmap /path/to/chr2.kept.PREDICT.BSCORRECTED.txt

Reports three things:

1. **The collinear rate** — chr2 outside the inversion (plus a buffer). This is
   the meiotic rate the simulation should use, and the number to compare against
   the autosomal proxy currently in `config.REC_RATE` (2.52e-9).
2. **The interior rate** — inside 60–80 Mb. Expected to be BIASED LOW, because
   ReLERNN infers recombination from LD decay and the inversion elevates LD in
   heterokaryotypes. This is a validation target, never a simulation input:
   feeding it to SLiM would double-count a barrier the model already imposes.
3. **The interior/collinear ratio** — an empirical measure of apparent
   recombination suppression, which the fitted model predicts independently.

If the interior rate comes out at or ABOVE the collinear rate, something is wrong
with the run (or the inversion is not suppressing recombination), and that is
worth understanding before the map is used for anything.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmap", required=True,
                    help="ReLERNN PREDICT (ideally .BSCORRECTED) file for chr2")
    ap.add_argument("--margin", type=int, default=2_000_000,
                    help="buffer excluded either side of the breakpoints, since "
                         "LD spills beyond them (default 2 Mb)")
    args = ap.parse_args()

    d = C._load_chr2_rmap(args.rmap)
    d = d.sort_values("start")
    wlen = (d.end - d.start)
    print(f"chr2 windows: {len(d):,}  span {d.start.min():,}-{d.end.max():,}")
    print(f"modal window: {int(wlen.mode().iloc[0]):,} bp")
    print(f"inversion:    {C.INV_START_REAL:,}-{C.INV_STOP_REAL:,} "
          f"({C.INV_LEN_REAL / 1e6:.2f} Mb), buffer +/-{args.margin:,}")
    print()

    coll = C.rec_rate_for_inversion(args.rmap, margin=args.margin)
    interior = C.rec_rate_inversion_interior(args.rmap)

    body = d[(d.start >= C.INV_START_REAL) & (d.end <= C.INV_STOP_REAL)]
    outside = d[(d.end < C.INV_START_REAL - args.margin)
                | (d.start > C.INV_STOP_REAL + args.margin)]

    print(f"COLLINEAR chr2 (n={len(outside):,} windows)  r = {coll:.4g}")
    print("  --> this is the SIMULATION's r "
          "(config.rec_rate_for_inversion)")
    print(f"  autosomal proxy currently in config.REC_RATE = {C.REC_RATE:.4g}"
          f"  ratio {coll / C.REC_RATE:.3f}")
    print(f"  male/female bracket {C.REC_RATE_BRACKET[0]:.4g}"
          f"-{C.REC_RATE_BRACKET[1]:.4g}")
    print()
    print(f"INTERIOR 60-80Mb (n={len(body):,} windows)  r = {interior:.4g}")
    print("  --> VALIDATION TARGET ONLY. Do not pass this to SLiM: the model "
          "already\n      imposes the barrier, so this would suppress twice.")
    if coll > 0:
        ratio = interior / coll
        print(f"  interior/collinear = {ratio:.3f}")
        if ratio >= 1.0:
            print("  UNEXPECTED: interior >= collinear. Either the run has a "
                  "problem or the\n  inversion is not suppressing recombination "
                  "-- investigate before using.")
        else:
            print(f"  apparent suppression = {100 * (1 - ratio):.1f}% "
                  "(elevated LD read as low r)")

    # Spatial profile: suppression should be strongest mid-inversion and relax
    # toward the breakpoints, mirroring the barrier's geometry.
    if len(body) >= 6:
        rel = ((body.start + body.end) / 2.0 - C.INV_START_REAL) / C.INV_LEN_REAL
        edges = np.linspace(0, 1, 6)
        print("\n  interior profile (relative position -> r):")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rel >= lo) & (rel < hi)
            if m.sum():
                print(f"    {lo:.1f}-{hi:.1f}  n={int(m.sum()):>5}  "
                      f"r={_wm(body[m.to_numpy()]):.4g}")

    print("\nNext: set config.CHR2_RMAP to this path. Do not edit REC_RATE by "
          "hand.")


def _wm(d) -> float:
    w = (d.end - d.start).to_numpy(dtype=float)
    return float((d.recombRate.to_numpy(dtype=float) * w).sum() / w.sum())


if __name__ == "__main__":
    main()
