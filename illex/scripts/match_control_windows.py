#!/usr/bin/env python
"""Regenerate results/illex/control_windows_matched.csv (I5).

Density-matches the collinear control region (chr2:10-30 Mb) windows to the
inversion body's SNP-density distribution, so the held-out control
comparison isn't confounded by a systematic density difference between the
two regions. This is a straight re-implementation of the exact recipe used
to originally produce the committed CSV (see
``.superpowers/sdd/2026-08-03-illex-chr2-neutral-sufficiency/task-6-report.md``,
"Step 4: density-matched control windows") -- it previously existed only as
an inline snippet in that report, with no committed script.

Density is defined per window as n_variants / window_width. A control
window is "matched" if its density falls within the inversion body's
[10th, 90th] percentile density range.

Run with:
    .venv/bin/python illex/scripts/match_control_windows.py

Reads ``results/illex/empirical_windowed.csv`` (committed, not regenerated
by this script) and writes ``results/illex/control_windows_matched.csv``.
Verified to reproduce the existing 5 committed rows exactly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IN = Path("results/illex/empirical_windowed.csv")
OUT = Path("results/illex/control_windows_matched.csv")


def main() -> None:
    d = pd.read_csv(IN)
    d["density"] = d.n_variants / (d.window_stop - d.window_start)
    inv = d[d.region == "inversion"]
    ctl = d[d.region == "control"]

    lo, hi = inv.density.quantile([0.1, 0.9])
    matched = ctl[(ctl.density >= lo) & (ctl.density <= hi)]

    print(f"inversion density 10-90 pct: {lo:.4f}-{hi:.4f} /bp")
    print(f"control windows total {len(ctl)}, density-matched {len(matched)}")
    if len(matched) < 5:
        print(
            "WARNING: fewer than 5 density-matched control windows -- "
            "see task-6-report.md caveat; the control region may not be "
            "usable as a density-matched control at all."
        )
    fst_col = [c for c in d.columns if c.startswith("fst")][0]
    print(f"matched-control mean Fst: {matched[fst_col].mean():.4f}")

    matched.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
