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


# Stats with documented cross-engine convention differences. They are
# still computed and saved per rep, and appear in the table marked
# verdict="convention_diff", but they do not count toward a track's
# overall pass/fail.
#
# num_trees: msinv records every recombination boundary as a tree
# split; msprime default simplifies non-topology-changing ones. This
# is closable with record_full_arg=True but costs 4x wall + 4x RAM
# at L=5 Mb (verified 2026-05-11 L=100kb pilot).
KNOWN_CONVENTION_DIFFS: set[str] = {"num_trees"}


def track_equivalence_table(
    dir_a: str | Path, dir_b: str | Path,
    *, alpha: float = 0.01, d_threshold: float = 0.2,
) -> dict[str, dict]:
    """Aggregate per-rep stats from two engine directories and compute
    the equivalence verdict per stat.

    Returns a dict keyed by stat name:
        {stat: {"ks_stat", "ks_p", "cohens_d", "verdict",
                "n_reps_a", "n_reps_b", "mean_a", "mean_b"}}

    Stats in ``KNOWN_CONVENTION_DIFFS`` are computed and reported but
    marked verdict="convention_diff" rather than passing through the
    KS/Cohen's-D test. They do not count toward the track's pass/fail.

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
        a_flat = np.asarray(a_vals).ravel()
        b_flat = np.asarray(b_vals).ravel()
        if stat in KNOWN_CONVENTION_DIFFS:
            v = {
                "ks_stat": float("nan"),
                "ks_p": float("nan"),
                "cohens_d": float("nan"),
                "verdict": "convention_diff",
            }
        else:
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
