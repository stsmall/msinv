#!/usr/bin/env python
"""Compare the four chr2 ReLERNN maps; report the rate the simulation uses.

    .venv/bin/python -m illex.slim.chr2_rmap_report

Four maps exist for chr2 (2026-08-07). Two were trained on chr2 subsets and
carry their own absolute calibration; two are the genome-wide autosomal network
applied to chr2 and are therefore the only ones commensurable with the
genome-wide rates in NOTES sec 8.1:

    run_chr2_AA     AA homokaryotypes only (n=254), own network
    run_chr2_all    all 633 samples, own network
    chr2_male       genome-wide autosomal network -> chr2   <- config.REC_RATE
    chr2_female     genome-wide autosomal network -> chr2

Because the absolute levels are not comparable across maps, the statistic to
read is each map's own **interior/collinear ratio**.

The headline result is that no map detects recombination suppression inside the
inversion, and the ratios are not even ordered the way a barrier signal would
order them. See ``config.rec_rate_inversion_interior`` for why (ReLERNN's ~19 kb
window is the wrong scale) and for what that costs us.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config as C

MAPS = [
    ("autonet male", "CHR2_RMAP", "the simulation's r"),
    ("autonet female", "CHR2_RMAP_FEMALE", "sex sensitivity arm"),
    ("all 633 pooled", "CHR2_RMAP_ALL", "own network; every heterokaryotype"),
    ("AA only", "CHR2_RMAP_AA", "own network; no barrier possible"),
]


def _wm(d) -> float:
    w = (d.end - d.start).to_numpy(dtype=float)
    return float((d.recombRate.to_numpy(dtype=float) * w).sum() / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmap", help="override: report this map alone")
    ap.add_argument("--margin", type=int, default=2_000_000,
                    help="buffer excluded either side of the breakpoints, since "
                         "LD spills beyond them (default 2 Mb)")
    args = ap.parse_args()

    todo = ([("(given)", args.rmap, "")] if args.rmap
            else [(n, getattr(C, a), w) for n, a, w in MAPS])

    print(f"inversion {C.INV_START_REAL:,}-{C.INV_STOP_REAL:,} "
          f"({C.INV_LEN_REAL / 1e6:.2f} Mb), collinear buffer "
          f"+/-{args.margin:,} bp\n")
    print(f"{'map':16s} {'n_in':>6s} {'n_out':>6s} {'r_interior':>11s} "
          f"{'r_collinear':>12s} {'ratio':>7s}  note")

    rows = []
    for name, path, note in todo:
        d = C._load_chr2_rmap(path).sort_values("start")
        body = d[(d.start >= C.INV_START_REAL) & (d.end <= C.INV_STOP_REAL)]
        coll = d[(d.end < C.INV_START_REAL - args.margin)
                 | (d.start > C.INV_STOP_REAL + args.margin)]
        ri, rc = _wm(body), _wm(coll)
        rows.append((name, ri, rc, body))
        print(f"{name:16s} {len(body):6d} {len(coll):6d} {ri:11.4g} "
              f"{rc:12.4g} {ri / rc:7.4f}  {note}")

    print(f"\nconfig.REC_RATE = {C.REC_RATE:.4g} "
          f"(male collinear; superseded proxy was "
          f"{C.REC_RATE_PROXY_OLD:.4g}, ratio "
          f"{C.REC_RATE_PROXY_OLD / C.REC_RATE:.3f})")

    # Spatial profile. A barrier is strongest mid-inversion and relaxes toward
    # the breakpoints, so a real signal is U-shaped. Flat means no signal.
    edges = np.linspace(0, 1, 6)
    print("\ninterior profile by relative position (U-shape = barrier, "
          "flat = none):")
    for name, _, rc, body in rows:
        rel = ((body.start + body.end) / 2.0
               - C.INV_START_REAL) / C.INV_LEN_REAL
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ((rel >= lo) & (rel < hi)).to_numpy()
            cells.append(f"{_wm(body[m]) / rc:6.3f}" if m.sum() else "    --")
        print(f"  {name:16s} " + " ".join(cells) + "   (/collinear)")

    if not args.rmap:
        print("\nNo map shows interior suppression, and the ordering is wrong "
              "for a barrier:\n  the AA-only map cannot have one yet shows the "
              "largest deficit, while the\n  pooled map, which contains every "
              "heterokaryotype, shows a slight excess.\n  ReLERNN's ~19 kb "
              "window measures within-arrangement LD decay, which the\n  "
              "barrier does not suppress. See config.rec_rate_inversion_interior.")


if __name__ == "__main__":
    main()
