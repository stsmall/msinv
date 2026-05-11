"""Per-stat equivalence verdict across two engine output dirs.

Loads per-rep .npz files via validation._lib.io.aggregate_track and
applies validation._lib.equivalence.equivalence_verdict to each
flat-stat key. Returns a dict keyed by stat name with KS + Cohen's D
+ verdict per the pre-registered spec criteria.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation._lib.equivalence import equivalence_verdict
from validation._lib.io import aggregate_track


def track_equivalence_table(
    dir_a: str | Path, dir_b: str | Path,
    *, alpha: float = 0.01, d_threshold: float = 0.2,
) -> dict[str, dict]:
    """Aggregate per-rep stats from two engine directories and compute
    the equivalence verdict per stat.

    Returns a dict keyed by stat name:
        {stat: {"ks_stat", "ks_p", "cohens_d", "verdict",
                "n_reps_a", "n_reps_b", "mean_a", "mean_b"}}

    Only stats that appear in BOTH directories are reported. Stats with
    different array shapes between engines are skipped (with a
    `__skipped__` entry in the returned dict noting the stat name).
    """
    agg_a = aggregate_track(dir_a)
    agg_b = aggregate_track(dir_b)
    shared = set(agg_a) & set(agg_b)
    shared.discard("__rep_indices__")
    table: dict[str, dict] = {}
    skipped: list[str] = []
    for stat in sorted(shared):
        a_vals = agg_a[stat]
        b_vals = agg_b[stat]
        if a_vals.shape[1:] != b_vals.shape[1:]:
            skipped.append(stat)
            continue
        # Flatten across reps and (if applicable) windows to a 1-D
        # distribution per engine for the KS test. For window stats
        # this gives a per-window-per-rep mixed distribution; that
        # matches the per-rep distributions framing in the spec.
        a_flat = np.asarray(a_vals).ravel()
        b_flat = np.asarray(b_vals).ravel()
        v = equivalence_verdict(
            a_flat, b_flat, alpha=alpha, d_threshold=d_threshold,
        )
        v["n_reps_a"] = int(a_vals.shape[0])
        v["n_reps_b"] = int(b_vals.shape[0])
        v["mean_a"] = float(np.nanmean(a_flat))
        v["mean_b"] = float(np.nanmean(b_flat))
        table[stat] = v
    if skipped:
        table["__skipped__"] = {"stats": skipped}
    return table
