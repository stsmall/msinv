"""Phase-0 pilot bench: msinv on v12 demography.

Runs a single rep at L=10 Mb (or smaller for smoke tests) on the v12
Kir/Fol demography with a 3Ra inversion in F. Measures wall + peak RSS
+ iters consumed, computes the full validation-suite stats panel, and
persists everything to `out_dir / {stats.npz, timing.json}`.

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

from msinv import HullSimulator, InversionSpec
from validation._lib import io, stats
from validation._lib.demography import (
    v12_msinv,
    T_INV_3RA,
    P_INV_F_3RA,
    P_INV_K_3RA,
    GAMMA_3RA,
)

# 3Ra geometry: position 0.18·L start, width 0.20·L.
INV_LEFT_FRAC = 0.18
INV_WIDTH_FRAC = 0.20
MEAN_TRACT_FRAC = 0.05  # fraction of inv_width

# Rates per spec
MU = 1.0e-8
R = 1.0e-8

# F-only sampling at n=100 with p_inv_F = 0.73 → 27 F_S + 73 F_I.
N_F_S = 27
N_F_I = 73
N_TOTAL = N_F_S + N_F_I


def run_pilot_rep(
    *,
    out_dir: str | Path,
    rep: int,
    L: float,
    seed: int,
    iters_max: int = 1_000_000_000,
) -> dict[str, float]:
    """Run one msinv pilot rep on v12 + 3Ra at the given L and persist outputs.

    Returns a small dict with the timing info that's also written to disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_bp_left = INV_LEFT_FRAC * L
    inv_bp_right = inv_bp_left + INV_WIDTH_FRAC * L
    inv_width = inv_bp_right - inv_bp_left

    inv_3ra = InversionSpec(
        bp_left=int(inv_bp_left),
        bp_right=int(inv_bp_right),
        p_inv={0: P_INV_K_3RA, 1: P_INV_F_3RA},
        t_inv=float(T_INV_3RA),
        gene_conversion_rate=GAMMA_3RA,
        mean_tract_length=MEAN_TRACT_FRAC * inv_width,
        tract_distribution="fixed",
        inv_id=0,
    )

    sim = HullSimulator(
        sample_config={("S", 0): 0, ("S", 1): N_F_S, ("I", 1): N_F_I},
        demography=v12_msinv(),
        sequence_length=float(L),
        recombination_rate=R,
        inversions=[inv_3ra],
        sweeps=[],
        seed=int(seed),
        iters_max=iters_max,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    ts_raw = sim.simulate()
    wall = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux; convert to bytes
    peak_rss = max(rss_before, rss_after) * 1024

    # Overlay neutral mutations
    ts = msprime.sim_mutations(
        ts_raw,
        rate=MU,
        random_seed=seed + 1,
        keep=True,
    )

    # Sample-set partition: F_S = first N_F_S, F_I = next N_F_I.
    # (sample_config above produces samples in order (S,1)x27, (I,1)x73.)
    samples = list(ts.samples())
    sset = {"F_S": samples[:N_F_S], "F_I": samples[N_F_S:]}

    win = stats.window_stats(ts, sample_sets=sset, n_windows=40)
    sfs_S = stats.sfs(ts, sample_set=sset["F_S"], folded=True)
    sfs_I = stats.sfs(ts, sample_set=sset["F_I"], folded=True)
    tree_d = stats.tree_shape_stats(ts, n_samples=200, seed=seed + 2)
    bins = np.logspace(2, np.log10(L), 11)
    ld_d = stats.ld_decay(
        ts,
        distance_bins=bins,
        max_pairs=2000,
        seed=seed + 3,
    )

    flat: dict[str, np.ndarray] = {}
    for sname, arr in win["pi"].items():
        flat[f"pi__{sname}"] = arr
    for pname, arr in win["dxy"].items():
        flat[f"dxy__{pname}"] = arr
    for pname, arr in win["fst"].items():
        flat[f"fst__{pname}"] = arr
    for sname, arr in win["tajimas_d"].items():
        flat[f"tajimas_d__{sname}"] = arr
    flat["sfs__F_S"] = sfs_S
    flat["sfs__F_I"] = sfs_I
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
        "num_trees": int(ts.num_trees),
        "num_sites": int(ts.num_sites),
        "rep": int(rep),
        "L": float(L),
        "seed": int(seed),
        "n_samples": int(N_TOTAL),
    }
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def _cli_main():
    """Run the production-scale pilot: 3 reps at L=10 Mb on v12."""
    import sys
    from validation._lib.seeds import seed_for

    out_root = Path("results/validation/pilot")
    n_reps = 3
    L = 10_000_000
    timings = []
    for rep in range(n_reps):
        out_dir = out_root / f"rep_{rep:03d}"
        seed = seed_for(
            track="pilot",
            scenario="v12",
            engine="msinv",
            rep=rep,
        )
        print(
            f"Pilot rep {rep}: seed={seed}, L={L}, out={out_dir}",
            flush=True,
        )
        t = run_pilot_rep(out_dir=out_dir, rep=rep, L=L, seed=seed)
        timings.append(t)
        print(
            f"  wall={t['wall_seconds']:.1f}s, "
            f"peak_rss={t['peak_rss_bytes'] / 1e9:.2f} GB, "
            f"trees={t['num_trees']}, sites={t['num_sites']}",
            flush=True,
        )

    walls = [t["wall_seconds"] for t in timings]
    rsses = [t["peak_rss_bytes"] for t in timings]
    print(f"\nPilot summary over {n_reps} reps:")
    print(
        f"  wall: median={np.median(walls):.1f}s, "
        f"min={min(walls):.1f}s, max={max(walls):.1f}s"
    )
    print(
        f"  rss : median={np.median(rsses) / 1e9:.2f}GB, max={max(rsses) / 1e9:.2f}GB"
    )
    if max(walls) > 4 * 3600:
        print("  GATE: per-rep wall > 4h — escalate before full launch")
        sys.exit(2)
    if max(rsses) > 8 * 1e9:
        print("  GATE: per-rep RSS > 8GB — discuss before full launch")
        sys.exit(1)
    print("  GATE: within pilot pass criteria")


if __name__ == "__main__":
    _cli_main()
