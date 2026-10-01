"""One chunk of SLiM-vs-msinv inversion benchmark replicates.

For each replicate: run inversion_bench.slim, recapitate the pre-inversion history
(panmictic, constant N), sample NS inverted and NS standard haplosomes from the final
generation, and compute mean pairwise coalescence times (branch-mode diversity / 2)
within each arrangement and between arrangements inside the inversion, and in the
collinear flanks. Then simulate the same history with msinv, passing SLiM's realised
inverted-frequency trajectory, and compute the same statistics. One JSON line per
replicate.

Run from the repo root as a module:
  python -m illex.slim_bench.run_bench --scenario long_barrier --reps 10 \
      --task-id 0 --out results/slim_bench --slim $(command -v slim)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyslim
import tskit

HERE = Path(__file__).resolve().parent
RECIPE = HERE / "inversion_bench.slim"

N = 2000
L = 100_000
INV_START, INV_END = 10_000, 89_999        # inversion = [10,000, 90,000)
NS = 20                                    # haplosomes sampled per arrangement
R_DEFAULT = 2.5e-6                         # 4 N R L = 2000, as in the Illex fits

SCENARIOS = {
    # barrier present for 0.1N generations: most lineages still span inversion and
    # flanks when the barrier is removed (the regime of the post-barrier fix)
    "short_barrier": dict(P0=0.5, T_INV=201, R=R_DEFAULT, A_SS=0.0, B_II=0.0),
    # neutral, barrier for N generations
    "long_barrier": dict(P0=0.5, T_INV=2001, R=R_DEFAULT, A_SS=0.0, B_II=0.0),
    # neutral, 100-fold lower recombination (4NRL = 20)
    "low_recomb": dict(P0=0.5, T_INV=501, R=R_DEFAULT / 100, A_SS=0.0, B_II=0.0),
    # Illex-like: overdominance holding the inverted frequency near 0.74 for 1.5N
    # generations (fitness 1 - 0.0148, 1, 1 - 0.0052; 2N times the scale = 80)
    "illex_like": dict(P0=0.74, T_INV=3001, R=R_DEFAULT, A_SS=0.0148, B_II=0.0052),
}


def run_slim(slim_bin, params, seed, workdir):
    trees = os.path.join(workdir, f"rep_{seed}.trees")
    traj = os.path.join(workdir, f"rep_{seed}.traj")
    defs = [f"N={N}", f"L={L}", f"INV_START={INV_START}", f"INV_END={INV_END}",
            f"R={params['R']}", f"P0={params['P0']}", f"T_INV={params['T_INV']}",
            f"A_SS={params['A_SS']}", f"B_II={params['B_II']}",
            f"OUT_TREES='{trees}'", f"OUT_TRAJ='{traj}'"]
    cmd = [slim_bin, "-s", str(seed)]
    for d in defs:
        cmd += ["-d", d]
    cmd.append(str(RECIPE))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    status = next((l.split("STATUS=")[1].split()[0] for l in out.splitlines()
                   if "[bench] STATUS=" in l), "UNKNOWN")
    if status != "SEGREGATING":
        return status, None, None
    ts = tskit.load(trees)
    freqs = np.loadtxt(traj, ndmin=1)
    os.remove(trees)
    os.remove(traj)
    return status, ts, freqs


def pair_stats(ts, i_nodes, s_nodes):
    w = [0, INV_START, INV_END + 1, L]
    pi_i = ts.diversity([i_nodes], windows=w, mode="branch")[1][0] / 2
    pi_s = ts.diversity([s_nodes], windows=w, mode="branch")[1][0] / 2
    dxy = np.ravel(ts.divergence([i_nodes, s_nodes], windows=w, mode="branch")[1])[0] / 2
    fl = ts.diversity([list(i_nodes) + list(s_nodes)], windows=[0, INV_START, INV_END + 1, L],
                      mode="branch")
    pi_flank = (fl[0][0] * INV_START + fl[2][0] * (L - INV_END - 1)) / (L - (INV_END + 1 - INV_START)) / 2
    return {"T_II": float(pi_i), "T_SS": float(pi_s), "T_IS": float(dxy),
            "T_flank": float(pi_flank), "pi_I_over_pi_S": float(pi_i / pi_s),
            "dxy_over_pi_I": float(dxy / pi_i)}


def slim_stats(ts, params, seed):
    rts = pyslim.recapitate(ts, ancestral_Ne=N, recombination_rate=params["R"], random_seed=seed)
    alive = pyslim.individuals_alive_at(rts, 0)
    nodes = np.concatenate([rts.individual(i).nodes for i in alive])
    var = next(rts.variants(samples=nodes, left=INV_START, right=INV_START + 1,
                            isolated_as_missing=False))
    carrier = var.genotypes > 0
    rng = np.random.default_rng(seed)
    i_nodes = rng.choice(nodes[carrier], NS, replace=False)
    s_nodes = rng.choice(nodes[~carrier], NS, replace=False)
    return pair_stats(rts, list(i_nodes), list(s_nodes)), float(carrier.mean())


def msinv_stats(freqs, params, seed):
    from msinv.hull.simulator import HullSimulator
    from msinv.hull.inversion import InversionSpec
    t_found = params["T_INV"] - 1                    # generations since the founders
    # freqs[j] is the frequency at tick j + 2, i.e. backward time T_INV - (j + 2).
    tau = [float(params["T_INV"] - (j + 2)) for j in range(len(freqs))][::-1]
    p = [float(x) for x in freqs][::-1]
    tau.append(float(t_found))
    p.append(round(params["P0"] * 2 * N) / (2 * N))
    spec = InversionSpec(
        bp_left=float(INV_START), bp_right=float(INV_END + 1),
        gene_conversion_rate=1e-15, mean_tract_length=1.0, tract_distribution="geometric",
        trajectory={"type": "precomputed", "times": tau, "freqs": [p],
                    "n_e": [float(N)], "t_inv": [float(t_found)]})
    sim = HullSimulator(n_std=NS, n_inv=NS, population_size=N, sequence_length=L,
                        recombination_rate=params["R"], inversions=[spec], seed=seed)
    ts = sim.simulate()
    s_nodes = list(range(NS))
    i_nodes = list(range(NS, 2 * NS))
    return pair_stats(ts, i_nodes, s_nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=500_000_000)
    ap.add_argument("--out", default="results/slim_bench")
    ap.add_argument("--slim", default=os.environ.get("SLIM_BIN", "slim"))
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--max-tries", type=int, default=20)
    a = ap.parse_args()
    params = SCENARIOS[a.scenario]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{a.scenario}_task{a.task_id:05d}.jsonl"
    scen_off = sorted(SCENARIOS).index(a.scenario) * 10_000_000
    with tempfile.TemporaryDirectory(dir=a.scratch) as wd, open(fn, "a") as fh:
        for rep in range(a.reps):
            base = a.seed_base + scen_off + (a.task_id * a.reps + rep) * 100
            t0 = time.time()
            for attempt in range(a.max_tries):
                seed = base + attempt
                status, ts, freqs = run_slim(a.slim, params, seed, wd)
                if status == "SEGREGATING":
                    break
            else:
                fh.write(json.dumps({"scenario": a.scenario, "task": a.task_id, "rep": rep,
                                     "status": "FAILED", "tries": a.max_tries}) + "\n")
                continue
            sl, p_final = slim_stats(ts, params, seed)
            ms = msinv_stats(freqs, params, seed)
            rec = {"scenario": a.scenario, "task": a.task_id, "rep": rep, "seed": seed,
                   "tries": attempt + 1, "p_final": p_final,
                   "p_mean": float(np.mean(freqs)), "slim": sl, "msinv": ms,
                   "wall_s": round(time.time() - t0, 2)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
