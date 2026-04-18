#!/usr/bin/env python3
"""Run SLIM + msinv for a validation scenario, time both, save stats.

Usage (from project root):
    .venv/bin/python slim_validation/run_comparison.py --scenario 1 --reps 5

Outputs to slim_validation/output/:
    scenario{N}_rep{i}_slim.trees   — SLIM tree sequence
    scenario{N}_rep{i}_msinv.trees  — msinv tree sequence
    scenario{N}_results.npz         — per-rep + aggregated stats + timing
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import msprime
import tskit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from msinv import HullSimulator, InversionSpec  # noqa: E402
from msinv.hull.sweep import Sweep  # noqa: E402

SLIM_BIN = os.environ.get(
    "SLIM_BIN", "/home/ssmall/miniforge3/envs/popgen/bin/slim")

# --- Shared constants ---
Ne = 1000
L = 100_000
r_rate = 1e-7
gc_rate = 1e-8
mu_rate = 1e-8
burnin_factor = 8
t_inv_factor = 4
s_bal = 0.01           # balancing selection in SLIM (keeps inv polymorphic)
n_window = 40

# Scenario-specific
SCENARIO_PARAMS = {
    1: dict(inv_list=[(30_000, 70_000)]),
    2: dict(inv_list=[(15_000, 45_000), (55_000, 85_000)]),
    3: dict(inv_list=[(30_000, 70_000)],
            x_sel=50_000, t_sweep_factor=0.2, s_coef=0.05,
            starting_frequency=20.0 / (2 * Ne)),
}


def slim_script(scenario):
    name = {1: "scenario1_single_inv.slim",
            2: "scenario2_multi_inv.slim",
            3: "scenario3_sweep_in_inv.slim"}[scenario]
    return HERE / "scenarios" / name


def _slim_cmd(scenario, seed, out_trees):
    script = slim_script(scenario)
    cmd = [SLIM_BIN,
           "-d", f'trees_path="{out_trees}"',
           "-d", f"Ne={Ne}",
           "-d", f"L={L}",
           "-d", f"r_rate={r_rate}",
           "-d", f"gc_rate={gc_rate}",
           "-d", f"burnin_factor={burnin_factor}",
           "-d", f"t_inv_factor={t_inv_factor}",
           "-d", f"s_bal={s_bal}",
           "-d", f"seed={seed}"]
    if scenario == 3:
        p = SCENARIO_PARAMS[3]
        cmd += ["-d", f"x_sel={p['x_sel']}",
                "-d", f"t_sweep_factor={p['t_sweep_factor']}",
                "-d", f"s_coef={p['s_coef']}",
                "-d", f"n_sweep_copies=20"]
    cmd.append(str(script))
    return cmd


def run_slim(scenario, rep, out_trees, max_retries=30):
    """Run SLIM scenario, retrying on sweep loss (scenario 3).
    Returns (elapsed, n_attempts)."""
    total_elapsed = 0.0
    for attempt in range(max_retries):
        seed = 1000 + rep * 100 + attempt
        cmd = _slim_cmd(scenario, seed, out_trees)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=HERE.parent)
        total_elapsed += time.time() - t0
        if proc.returncode != 0:
            print(proc.stdout[-500:])
            print(proc.stderr[-500:])
            raise RuntimeError(f"SLIM failed (exit {proc.returncode})")
        # Scenario 3: retry if sweep was lost.
        if scenario == 3 and "m4 lost" in proc.stdout:
            if attempt < max_retries - 1:
                continue
            raise RuntimeError(
                f"scenario 3 rep {rep}: sweep lost in {max_retries} attempts")
        return total_elapsed, attempt + 1
    raise RuntimeError("unreachable")


def run_msinv(scenario, rep, out_trees):
    """Run msinv equivalent and save tree sequence. Returns elapsed sec."""
    params = SCENARIO_PARAMS[scenario]
    inv_list = params["inv_list"]
    t_inv = t_inv_factor * Ne
    invs = [InversionSpec(bp_left=float(lo), bp_right=float(hi),
                          p_inv=0.5, t_inv=float(t_inv),
                          gene_conversion_rate=gc_rate,
                          inv_id=i)
            for i, (lo, hi) in enumerate(inv_list)]

    sweeps = []
    if scenario == 3:
        sweeps = [Sweep(
            x_sel=float(params["x_sel"]),
            t_event=float(params["t_sweep_factor"] * Ne),
            target_class="S",
            selection_coefficient=float(params["s_coef"]),
            starting_frequency=float(params.get("starting_frequency", 0.0)))]

    # Match 10 S + 10 I haploid samples.
    sim = HullSimulator(
        n_std=10, n_inv=10,
        population_size=Ne, sequence_length=float(L),
        recombination_rate=r_rate,
        inversions=invs, sweeps=sweeps,
        seed=2000 + rep)

    t0 = time.time()
    ts = sim.simulate()
    elapsed = time.time() - t0

    # Overlay neutral mutations at mu_rate for stats.
    ts = msprime.sim_mutations(ts, rate=mu_rate, random_seed=3000 + rep,
                                keep=True)
    ts.dump(out_trees)
    return elapsed


def load_slim_trees(path, rep, scenario):
    """Load SLIM tree seq, overlay mu, pick 10 S-genomes + 10 I-genomes.

    Uses the marker mutation at bp_left of the FIRST inversion to
    classify samples as S (no m2) vs I (has m2). For scenarios with
    multiple inversions we still classify by inv 0 so the stat
    comparison matches msinv's n_std/n_inv semantics (karyotype of
    first inv).
    """
    ts = tskit.load(path)
    inv_list = SCENARIO_PARAMS[scenario]["inv_list"]
    marker_pos = float(inv_list[0][0])
    marker_tree = ts.at(marker_pos)
    i_nodes = set()
    for site in ts.sites():
        if abs(site.position - marker_pos) > 1.0:
            continue
        for mut in site.mutations:
            for n in marker_tree.leaves(mut.node):
                if ts.node(n).is_sample():
                    i_nodes.add(n)
    all_samples = list(ts.samples())
    s_nodes = [n for n in all_samples if n not in i_nodes]
    i_nodes = [n for n in all_samples if n in i_nodes]
    rng = np.random.default_rng(5000 + rep)
    if len(s_nodes) < 10 or len(i_nodes) < 10:
        raise RuntimeError(
            f"rep {rep}: insufficient haps S={len(s_nodes)} I={len(i_nodes)}")
    s_pick = list(rng.choice(s_nodes, 10, replace=False))
    i_pick = list(rng.choice(i_nodes, 10, replace=False))
    keep = s_pick + i_pick
    ts = ts.simplify(samples=keep, filter_sites=True)
    ts = msprime.sim_mutations(ts, rate=mu_rate, random_seed=6000 + rep,
                                keep=True)
    return ts, list(range(10)), list(range(10, 20))


def load_msinv_trees(path):
    """Load msinv tree seq. Sample indices 0..9 = S, 10..19 = I (by
    construction in HullSimulator when using n_std/n_inv)."""
    ts = tskit.load(path)
    assert ts.num_samples == 20, f"expected 20 samples, got {ts.num_samples}"
    return ts, list(range(10)), list(range(10, 20))


def per_window_stats(ts, s_nodes, i_nodes):
    """Return (mid, pi_S, pi_I, dxy_SI, fst_SI) in windows."""
    wins = np.linspace(0, ts.sequence_length, n_window + 1)
    mid = (wins[:-1] + wins[1:]) / 2
    pi_s = ts.diversity([s_nodes], windows=wins, mode="site").reshape(-1)
    pi_i = ts.diversity([i_nodes], windows=wins, mode="site").reshape(-1)
    dxy = ts.divergence([s_nodes, i_nodes], windows=wins, mode="site").reshape(-1)
    pi_w = (pi_s + pi_i) / 2
    fst = np.where(dxy > 0, 1.0 - pi_w / dxy, 0.0)
    fst = np.clip(fst, 0.0, 1.0)
    return mid, pi_s, pi_i, dxy, fst


def run_rep(scenario, rep, outdir):
    print(f"  rep {rep}: SLIM...", end=" ", flush=True)
    slim_trees = outdir / f"scenario{scenario}_rep{rep}_slim.trees"
    slim_time, attempts = run_slim(scenario, rep, str(slim_trees))
    msg = f"{slim_time:.1f}s"
    if attempts > 1:
        msg += f" ({attempts} attempts)"
    print(msg + "  msinv...", end=" ", flush=True)

    msinv_trees = outdir / f"scenario{scenario}_rep{rep}_msinv.trees"
    msinv_time = run_msinv(scenario, rep, str(msinv_trees))
    print(f"{msinv_time:.2f}s", flush=True)

    slim_ts, s_s, s_i = load_slim_trees(str(slim_trees), rep, scenario)
    msinv_ts, m_s, m_i = load_msinv_trees(str(msinv_trees))

    mid_s, pi_s_s, pi_i_s, dxy_s, fst_s = per_window_stats(slim_ts, s_s, s_i)
    mid_m, pi_s_m, pi_i_m, dxy_m, fst_m = per_window_stats(msinv_ts, m_s, m_i)

    return dict(
        rep=rep,
        slim_time=slim_time, msinv_time=msinv_time,
        slim_pi_s=pi_s_s, slim_pi_i=pi_i_s,
        slim_dxy=dxy_s, slim_fst=fst_s,
        msinv_pi_s=pi_s_m, msinv_pi_i=pi_i_m,
        msinv_dxy=dxy_m, msinv_fst=fst_m,
        slim_num_trees=slim_ts.num_trees,
        msinv_num_trees=msinv_ts.num_trees,
        mid=mid_m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True,
                        choices=[1, 2, 3])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    outdir = HERE / "output"
    outdir.mkdir(exist_ok=True)
    print(f"Scenario {args.scenario}: {args.reps} reps")
    print(f"  Ne={Ne}, L={L}, r={r_rate}, gc={gc_rate}, "
          f"mu={mu_rate}, t_inv={t_inv_factor}Ne, burnin={burnin_factor}Ne")

    reps = []
    for rep in range(args.reps):
        try:
            reps.append(run_rep(args.scenario, rep, outdir))
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    if not reps:
        print("No reps succeeded.")
        return

    # Aggregate
    agg = {k: np.mean([r[k] for r in reps], axis=0)
           for k in ("slim_pi_s", "slim_pi_i", "slim_dxy", "slim_fst",
                     "msinv_pi_s", "msinv_pi_i", "msinv_dxy", "msinv_fst")}
    slim_times = np.array([r["slim_time"] for r in reps])
    msinv_times = np.array([r["msinv_time"] for r in reps])

    npz_path = outdir / f"scenario{args.scenario}_results.npz"
    np.savez(
        npz_path,
        mid=reps[0]["mid"],
        slim_times=slim_times, msinv_times=msinv_times,
        scenario=args.scenario, n_reps=len(reps),
        Ne=Ne, L=L, r_rate=r_rate, gc_rate=gc_rate, mu_rate=mu_rate,
        burnin_factor=burnin_factor, t_inv_factor=t_inv_factor,
        **agg)
    print(f"\nSaved: {npz_path}")
    print(f"  SLIM:  mean {slim_times.mean():.1f}s  "
          f"(min {slim_times.min():.1f}, max {slim_times.max():.1f})")
    print(f"  msinv: mean {msinv_times.mean():.3f}s  "
          f"(min {msinv_times.min():.3f}, max {msinv_times.max():.3f})")
    print(f"  speedup: {slim_times.mean() / msinv_times.mean():.0f}x")


if __name__ == "__main__":
    main()
