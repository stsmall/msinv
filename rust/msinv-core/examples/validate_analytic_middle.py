#!/usr/bin/env python3
"""Correctness validation for `analytic_middle` fast path.

Runs N reps of a modestly-scaled Kir/Fol-shaped simulation with and
without `analytic_middle=True` and compares branch-mode π, dxy, and
Fst distributions using `scipy.stats.ks_2samp`. A well-calibrated
prototype should produce statistically indistinguishable distributions
(p > 0.05 — with N=20 reps this is a weak but meaningful bar).

Usage:
    python validate_analytic_middle.py --reps 20 --scale 100
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import ks_2samp

from msinv import HullSimulator, InversionSpec, Demography


# Kir/Fol shape (to be Ne-downscaled by `scale`).
NE_K_CURRENT = 44_000
NE_F_CURRENT = 92_000
NE_F_POSTSPLIT = 3_000_000
NE_ANC_1 = 1_000_000
T_SPLIT = 14_000
T_INV = 330_000
P_INV_3RA_F = 0.73
R = 1.0e-8
GAMMA = 1.0e-9
FLUX_WINDOW = 0.05

N_K = 74
N_F_S = 23
N_F_I = 23


def build_sim(seed, analytic_middle, scale):
    L = 1_000_000
    demo = Demography(pop_sizes=[max(1, NE_K_CURRENT // scale),
                                 max(1, NE_F_CURRENT // scale)])
    demo.add_event(('en', T_SPLIT, 1, max(1, NE_F_POSTSPLIT // scale)))
    demo.add_event(('ej', T_SPLIT, 1, 0))
    demo.add_event(('en', T_SPLIT, 0, max(1, NE_ANC_1 // scale)))
    demo.add_inversion_freq_change(T_SPLIT, 0, inv_id=0, p_inv=0.30)

    inv = InversionSpec(
        bp_left=180_000, bp_right=380_000,
        p_inv={0: 0.0, 1: P_INV_3RA_F},
        t_inv=T_INV, gene_conversion_rate=GAMMA, flux_window=FLUX_WINDOW,
    )

    return HullSimulator(
        sample_config={('S', 0): N_K, ('S', 1): N_F_S, ('I', 1): N_F_I},
        demography=demo,
        sequence_length=L,
        recombination_rate=R,
        inversions=[inv],
        sweeps=[],
        seed=seed,
        analytic_middle=analytic_middle,
    )


def sample_indices():
    k = list(range(N_K))
    fs = list(range(N_K, N_K + N_F_S))
    fi = list(range(N_K + N_F_S, N_K + N_F_S + N_F_I))
    return k, fs, fi


def compute_stats(ts, k, fs, fi):
    # Branch-mode summary stats. tskit returns numpy arrays; we pull the
    # scalar via np.asarray(...).item() which handles both 0-d and 1-d.
    def _scalar(x):
        return float(np.asarray(x).ravel()[0])

    pi_k = _scalar(ts.diversity([k], mode='branch'))
    pi_fs = _scalar(ts.diversity([fs], mode='branch'))
    pi_fi = _scalar(ts.diversity([fi], mode='branch'))

    dxy_kfs = _scalar(ts.divergence([k, fs], mode='branch'))
    dxy_kfi = _scalar(ts.divergence([k, fi], mode='branch'))
    dxy_fsi = _scalar(ts.divergence([fs, fi], mode='branch'))

    fst_kfs = _scalar(ts.Fst([k, fs], mode='branch'))
    fst_kfi = _scalar(ts.Fst([k, fi], mode='branch'))
    fst_fsi = _scalar(ts.Fst([fs, fi], mode='branch'))

    return dict(
        pi_K=pi_k, pi_FS=pi_fs, pi_FI=pi_fi,
        dxy_K_FS=dxy_kfs, dxy_K_FI=dxy_kfi, dxy_FS_FI=dxy_fsi,
        Fst_K_FS=fst_kfs, Fst_K_FI=fst_kfi, Fst_FS_FI=fst_fsi,
    )


def run(reps, scale, verbose=True):
    k, fs, fi = sample_indices()
    keys = ['pi_K', 'pi_FS', 'pi_FI',
            'dxy_K_FS', 'dxy_K_FI', 'dxy_FS_FI',
            'Fst_K_FS', 'Fst_K_FI', 'Fst_FS_FI']
    # distributions: loop_stats[key] and am_stats[key] each a list of size reps
    loop_stats = {ky: [] for ky in keys}
    am_stats = {ky: [] for ky in keys}
    loop_wall = []
    am_wall = []

    for rep in range(reps):
        seed = 1_000 + rep

        # event-loop
        sim = build_sim(seed, analytic_middle=False, scale=scale)
        t0 = time.time()
        ts = sim.simulate()
        loop_wall.append(time.time() - t0)
        stats = compute_stats(ts, k, fs, fi)
        for ky in keys:
            loop_stats[ky].append(stats[ky])

        # analytic-middle
        sim = build_sim(seed, analytic_middle=True, scale=scale)
        t0 = time.time()
        ts = sim.simulate()
        am_wall.append(time.time() - t0)
        stats = compute_stats(ts, k, fs, fi)
        for ky in keys:
            am_stats[ky].append(stats[ky])

        if verbose:
            print(f"  rep {rep+1}/{reps}  loop={loop_wall[-1]:.2f}s  "
                  f"am={am_wall[-1]:.2f}s")

    print(f"\nreps={reps}  scale={scale}")
    print(f"  event-loop   : mean {np.mean(loop_wall):.2f}s  "
          f"total {np.sum(loop_wall):.2f}s")
    print(f"  analytic-mid : mean {np.mean(am_wall):.2f}s  "
          f"total {np.sum(am_wall):.2f}s")

    print("\nKS test (event-loop vs analytic-middle) per statistic:")
    print(f"  {'stat':<12}  {'loop mean':>12}  {'am mean':>12}  "
          f"{'KS':>8}  {'p-value':>10}")
    rows = []
    for ky in keys:
        a = np.asarray(loop_stats[ky])
        b = np.asarray(am_stats[ky])
        res = ks_2samp(a, b)
        print(f"  {ky:<12}  {a.mean():12.4g}  {b.mean():12.4g}  "
              f"{res.statistic:8.4f}  {res.pvalue:10.4f}")
        rows.append(dict(stat=ky, loop_mean=float(a.mean()),
                         am_mean=float(b.mean()),
                         ks=float(res.statistic),
                         pvalue=float(res.pvalue)))

    n_fail = sum(1 for r in rows if r['pvalue'] < 0.05)
    print(f"\n{n_fail} / {len(rows)} statistics with p < 0.05")
    return rows, loop_wall, am_wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--scale', type=int, default=100,
                   help='Ne downscale factor; higher = smaller, faster sim')
    args = p.parse_args()
    run(args.reps, args.scale)


if __name__ == '__main__':
    main()
