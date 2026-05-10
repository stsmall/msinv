"""Phase-0 pilot bench: msinv at the validation-suite scale.

Runs a single rep, measures wall + peak RSS + iters consumed, computes
the full validation-suite stats panel, and persists everything to
`out_dir / {stats.npz, timing.json}`.

Used to gate the full n=100 launch: per-rep wall < 4h AND peak RSS < 8GB
must both hold.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import msprime
import tskit

from msinv import HullSimulator, InversionSpec
from validation._lib import io, stats


def run_pilot_rep(
    *,
    out_dir: str | Path,
    rep: int,
    L: float,
    Ne: float,
    n_samples: int,
    inv_bp_left: float,
    inv_bp_right: float,
    t_inv: float,
    mu: float,
    r: float,
    gc_rate: float,
    seed: int,
    iters_max: int = 200_000_000,
) -> dict[str, float]:
    """Run one msinv pilot rep at the given parameters and persist outputs.

    Returns a small dict with the timing info that's also written to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = InversionSpec(
        bp_left=float(inv_bp_left),
        bp_right=float(inv_bp_right),
        p_inv=0.5,
        t_inv=float(t_inv),
        gene_conversion_rate=float(gc_rate),
        inv_id=0,
    )

    n_std = n_samples // 2
    n_inv = n_samples - n_std
    sim = HullSimulator(
        n_std=n_std, n_inv=n_inv,
        population_size=float(Ne),
        sequence_length=float(L),
        recombination_rate=float(r),
        inversions=[inv],
        seed=int(seed),
        iters_max=iters_max,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    ts_raw = sim.simulate()
    wall = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux; report bytes for clarity
    peak_rss = max(rss_before, rss_after) * 1024

    # Overlay neutral mutations
    ts = msprime.sim_mutations(ts_raw, rate=float(mu),
                                random_seed=seed + 1, keep=True)

    # Sample-set partition: first n_std = "S", rest = "I"
    samples = list(ts.samples())
    sset = {"S": samples[:n_std], "I": samples[n_std:]}

    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_S = stats.sfs(ts, sample_set=sset["S"], folded=True)
    sfs_I = stats.sfs(ts, sample_set=sset["I"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(L), 11)
    ld_d = stats.ld_decay(ts, distance_bins=bins, max_pairs=2000, seed=seed + 3)

    flat: dict[str, np.ndarray] = {}
    for sname, arr in win["pi"].items():
        flat[f"pi__{sname}"] = arr
    for pname, arr in win["dxy"].items():
        flat[f"dxy__{pname}"] = arr
    for pname, arr in win["fst"].items():
        flat[f"fst__{pname}"] = arr
    for sname, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{sname}"] = arr
    flat["sfs__S"] = sfs_S
    flat["sfs__I"] = sfs_I
    flat["tree_tmrca"] = tree_d["tmrca"]
    flat["tree_total_branch"] = tree_d["total_branch"]
    flat["tree_colless"] = tree_d["colless"]
    flat["ld_bin_edges"] = ld_d["bin_edges"]
    flat["ld_mean_r2"] = ld_d["mean_r2"]
    flat["ld_count"] = ld_d["count"]
    flat["num_trees"] = np.asarray(ts.num_trees)
    flat["num_sites"] = np.asarray(ts.num_sites)
    io.save_rep_stats(out_dir / "stats.npz", **flat)

    timing = {
        "wall_seconds": float(wall),
        "peak_rss_bytes": int(peak_rss),
        "iters_consumed": int(getattr(sim, "iters_used", -1)),
        "rep": int(rep),
        "L": float(L),
        "Ne": float(Ne),
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def _cli_main():
    """Run the production-scale pilot: 3 reps at L=5 Mb, Ne=1e6, n=100."""
    import sys
    from validation._lib.seeds import seed_for

    out_root = Path("results/validation/pilot")
    n_reps = 3
    timings = []
    for rep in range(n_reps):
        out_dir = out_root / f"rep_{rep:03d}"
        seed = seed_for(track="pilot", scenario="default",
                          engine="msinv", rep=rep)
        print(f"Pilot rep {rep}: seed={seed}, out={out_dir}", flush=True)
        t = run_pilot_rep(
            out_dir=out_dir, rep=rep,
            L=5_000_000, Ne=1_000_000, n_samples=100,
            inv_bp_left=2_000_000.0, inv_bp_right=3_000_000.0,
            t_inv=4_000_000.0,
            mu=1e-8, r=1e-8, gc_rate=1e-9, seed=seed,
        )
        timings.append(t)
        print(f"  wall={t['wall_seconds']:.1f}s, "
              f"peak_rss={t['peak_rss_bytes'] / 1e9:.2f} GB",
              flush=True)
    walls = [t["wall_seconds"] for t in timings]
    rsses = [t["peak_rss_bytes"] for t in timings]
    print(f"\nPilot summary over {n_reps} reps:")
    print(f"  wall: median={np.median(walls):.1f}s, "
          f"min={min(walls):.1f}s, max={max(walls):.1f}s")
    print(f"  rss : median={np.median(rsses) / 1e9:.2f}GB, "
          f"max={max(rsses) / 1e9:.2f}GB")
    if max(walls) > 4 * 3600:
        print("  GATE: ❌ per-rep wall > 4h — escalate before full launch")
        sys.exit(2)
    if max(rsses) > 8 * 1e9:
        print("  GATE: ⚠️ per-rep RSS > 8GB — discuss before full launch")
        sys.exit(1)
    print("  GATE: ✅ within pilot pass criteria")


if __name__ == "__main__":
    _cli_main()
