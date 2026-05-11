"""Track 3: msinv ↔ msprime, v12 demography, no inv, no sweep, L=1 Mb.

Per-rep: run both engines with deterministically-paired seeds, compute
the validation-suite stats panel, save to .npz. After all reps, compute
the per-stat equivalence verdict via the aggregator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation._lib import io, stats
from validation._lib.aggregator import track_equivalence_table
from validation._lib.demography import v12_msinv, v12_msprime
from validation._lib.engines import msinv_run, msprime_run
from validation._lib.seeds import seed_for


def _compute_and_save_stats(ts, out_dir, seed, label):
    """Compute the validation-suite stats panel and save to {out_dir}/stats.npz."""
    samples = list(ts.samples())
    # Track 3 has only F samples (single sub-pop in stats land).
    sset = {"F": samples}
    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_F = stats.sfs(ts, sample_set=samples, folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(ts.sequence_length), 11)
    ld_d = stats.ld_decay(
        ts, distance_bins=bins, max_pairs=2000, seed=seed + 3,
    )
    flat: dict[str, np.ndarray] = {}
    for name, arr in win["pi"].items():
        flat[f"pi__{name}"] = arr
    for name, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{name}"] = arr
    flat["sfs__F"] = sfs_F
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)


def run_track3(
    *,
    out_root: str | Path,
    n_reps: int,
    L: float = 1_000_000,
    n_samples: int = 100,
    mu: float = 1.0e-8,
    r: float = 1.0e-8,
) -> dict:
    """Run Track 3 end-to-end. Returns a dict with per-rep timings and
    the final equivalence table."""
    out_root = Path(out_root)
    msinv_dir = out_root / "msinv"
    msprime_dir = out_root / "msprime"
    msinv_dir.mkdir(parents=True, exist_ok=True)
    msprime_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(n_reps):
        seed_a = seed_for(track="track3", scenario="v12", engine="msinv", rep=rep)
        seed_b = seed_for(track="track3", scenario="v12", engine="msprime", rep=rep)
        # msinv side
        ts_a = msinv_run(
            demography=v12_msinv(),
            sample_config={("S", 0): 0, ("S", 1): n_samples},
            L=L, r=r, mu=mu, seed=seed_a,
        )
        _compute_and_save_stats(
            ts_a, msinv_dir / f"rep_{rep:03d}", seed=seed_a, label="msinv",
        )
        # msprime side
        ts_b = msprime_run(
            demography=v12_msprime(),
            samples_by_pop={"F": n_samples},
            L=L, r=r, mu=mu, seed=seed_b,
        )
        _compute_and_save_stats(
            ts_b, msprime_dir / f"rep_{rep:03d}", seed=seed_b, label="msprime",
        )

    table = track_equivalence_table(msinv_dir, msprime_dir)
    return {"equivalence_table": table}


def _cli_main():
    import json
    out_root = Path("results/validation/track3")
    res = run_track3(out_root=out_root, n_reps=100)
    print(json.dumps(res["equivalence_table"], indent=2, default=float))


if __name__ == "__main__":
    _cli_main()
