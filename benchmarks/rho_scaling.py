#!/usr/bin/env python3
"""Head-to-head rho-scaling benchmark: msprime vs msinv (Rust).

Three configurations at each rho point:
  A. msprime panmictic (ground truth speed)
  B. msinv no-inversion (panmictic limit — should match msprime)
  C. msinv one-inversion (the realistic workload)

Usage:
    .venv/bin/python benchmarks/rho_scaling.py
    .venv/bin/python benchmarks/rho_scaling.py --rho 50,100,500 --reps 10
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import msprime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from msinv import HullSimulator, InversionSpec  # noqa: E402

# --- Fixed parameters ---
Ne = 1000
L = 100_000.0
n_samples = 20
t_inv_factor = 2.0   # t_inv in units of Ne
p_inv = 0.5
gc_rate = 1e-9       # negligible flux; keeps inversion mostly as barrier


def rho_to_r(rho):
    return rho / (4 * Ne * L)


def time_run(fn, reps):
    """Return (mean_seconds, reps_per_sec)."""
    t0 = time.perf_counter()
    for i in range(reps):
        fn(i)
    total = time.perf_counter() - t0
    return total / reps, reps / total


def run_msprime(rho, reps):
    r = rho_to_r(rho)

    def one(i):
        return msprime.sim_ancestry(
            samples=n_samples // 2,       # diploid -> 20 haps
            population_size=Ne,
            sequence_length=L,
            recombination_rate=r,
            random_seed=1000 + i)
    return time_run(one, reps)


def run_msinv_no_inv(rho, reps):
    r = rho_to_r(rho)

    def one(i):
        sim = HullSimulator(
            samples=n_samples,
            population_size=Ne,
            sequence_length=L,
            recombination_rate=r,
            seed=2000 + i)
        return sim.simulate()
    return time_run(one, reps)


def run_msinv_with_inv(rho, reps):
    r = rho_to_r(rho)
    t_inv = t_inv_factor * Ne
    inv = InversionSpec(
        bp_left=float(L * 0.3), bp_right=float(L * 0.7),
        p_inv=p_inv, t_inv=t_inv,
        gene_conversion_rate=gc_rate, inv_id=0)

    def one(i):
        sim = HullSimulator(
            n_std=n_samples // 2, n_inv=n_samples // 2,
            population_size=Ne,
            sequence_length=L,
            recombination_rate=r,
            inversions=[inv],
            seed=3000 + i)
        return sim.simulate()
    return time_run(one, reps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=str, default="50,100,200,500,1000",
                        help="Comma-separated rho values")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", type=str,
                        default="benchmarks/rho_scaling.npz")
    args = parser.parse_args()

    rho_vals = [int(x.strip()) for x in args.rho.split(",")]
    results = {k: [] for k in
               ("msprime", "msinv_no_inv", "msinv_with_inv")}
    results_rps = {k: [] for k in results}

    print(f"n_samples={n_samples} haps, Ne={Ne}, L={L/1000:.0f} kb, "
          f"reps={args.reps}")
    print(f"  rho values: {rho_vals}")
    print(f"{'rho':>6} | {'msprime':>12} | {'msinv-noinv':>12} | "
          f"{'msinv+inv':>12} | {'slowdown':>10}")
    print("-" * 70)

    for rho in rho_vals:
        mp_mean, mp_rps = run_msprime(rho, args.reps)
        ni_mean, ni_rps = run_msinv_no_inv(rho, args.reps)
        wi_mean, wi_rps = run_msinv_with_inv(rho, args.reps)

        results["msprime"].append(mp_mean)
        results["msinv_no_inv"].append(ni_mean)
        results["msinv_with_inv"].append(wi_mean)
        results_rps["msprime"].append(mp_rps)
        results_rps["msinv_no_inv"].append(ni_rps)
        results_rps["msinv_with_inv"].append(wi_rps)

        slowdown = wi_mean / mp_mean
        print(f"{rho:>6} | {mp_mean*1000:>8.1f} ms | "
              f"{ni_mean*1000:>8.1f} ms | {wi_mean*1000:>8.1f} ms | "
              f"{slowdown:>7.1f}x")

    arr = dict(rho_vals=rho_vals, n_samples=n_samples, Ne=Ne, L=L,
               reps=args.reps)
    for k, v in results.items():
        arr[f"{k}_mean_s"] = np.array(v)
    for k, v in results_rps.items():
        arr[f"{k}_rps"] = np.array(v)
    np.savez(args.out, **arr)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
