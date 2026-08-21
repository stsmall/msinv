#!/usr/bin/env python
"""Fit the chr2 inversion's age under balancing selection, and test neutrality.

    .venv/bin/python -m illex.scripts.fit_balancing --reps 96 --workers 26

WHAT IS FITTED, AND AGAINST WHAT
--------------------------------
Two targets (NOTES sec 5, ``illex.empirical``), both ratios so both
calibration-free:

    pi_I/pi_S = 0.744        dxy/pi_I = 1.846

Free:   t_inv    age in generations (= years for Illex)
        p_start  founding frequency of the inverted arrangement

Derived, not free: ``s_het`` is solved from the condition that the rise
finishes ``--plateau`` generations before the present, so it is a function of
(t_inv, p_start, plateau) rather than an independent parameter. That is a
deliberate reduction, forced by the data: a 2026-08-07 scan over
s_het in [1e-4, 1e-2] found the statistics essentially flat above ~1e-3
(1e-3 and 1e-2 are indistinguishable), because once the rise is fast relative
to t_inv only its *timing* matters, not its speed. Fitting s_het directly
would report spurious precision.

Fixed and not fitted: the moments growth demography, mu = 3e-9,
r = 1.977e-9 (measured chr2 collinear male ReLERNN rate, NOTES sec 8.1),
gamma ~ 0 (flux falsified, NOTES sec 6), p* = 0.626 (asserted, not inferred --
this is the balancing-selection hypothesis).

WHAT THIS RUN ESTABLISHED (2026-08-07)
--------------------------------------
The rising-logistic arm's best point missed both targets in opposite
directions, -9.4% on pi_I/pi_S and +6.5% on dxy/pi_I, which was diagnosed as a
model-shape error rather than a scaling one (NOTES sec 7.2). Replacing the
still-rising trajectory with a rise-to-equilibrium closes it:

    t_inv ~ 7.2e5, p_start ~ 0.025, s_het ~ 3.6e-5  ->  both residuals ~0%

and the age is **robust to the one dimension that stays degenerate**: moving
the plateau from 0 to 100,000 generations shifts the fitted t_inv by 1%
(726,700 -> 719,900) while p_start moves 22% and s_het 18%. So the age is
identified; the founding frequency and the selection strength are only jointly
constrained. The previous rising-logistic age of 750-800 ky was therefore NOT
an artifact of the misspecified trajectory -- it survives the correction, which
is the more important result.

THE NEUTRAL ALTERNATIVE IS NOT CLOSE
------------------------------------
Reported first because it needs no simulation and no mu. For a neutral allele
the expected time to reach 0.626, *conditional on getting there at all*, is
1.650 N_e generations (``illex.balancing.neutral_hitting_time``, exact
diffusion result, verified against Wright-Fisher). Against a fitted age of
~7.2e5 generations that is impossible at every Ne on the growth arm, and the
comparison is generous to neutrality three times over: it conditions on a
~1e-6 event, uses the mean rather than a lower quantile, and evaluating at the
smallest Ne on the arm understates the true timescale.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import optimize

from illex import balancing as B
from illex import empirical, model, stats
from illex.demography import PRESENT_NE_GROWTH
from illex.slim.config import REC_RATE
from illex.theory import N0, N_ANC, N_growth

RHO = 2000.0                   # -> L ~ 37 kb, inside the L-invariance-verified
                               #    band (NOTES sec 7.3)
GAMMA = 1e-15                  # flux off; InversionSpec rejects exactly 0
TARGET_R = empirical.PI_I_OVER_PI_S
TARGET_D = empirical.DXY_OVER_PI_I

# Bracketing grid. Chosen from the coarse scans so the target is INTERIOR in
# both statistics -- the misses change sign across it, which is what makes the
# root a crossing rather than an extrapolation.
T_INV_GRID = [6.5e5, 7.0e5, 7.5e5]
P_START_GRID = [0.022, 0.025, 0.028]
# The degenerate dimension, run as two arms rather than fitted, so the
# trade-off against p_start is visible instead of hidden.
PLATEAU_ARMS = [0.0, 1.0e5]

OUT_CSV = Path("results/illex/fit_balancing.csv")
OUT_JSON = Path("results/illex/fit_balancing.json")


def seq_length(rho: float = RHO) -> int:
    return int(round(rho / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))


def s_het_for(p_start: float, t_inv: float, plateau: float) -> float:
    """s_het such that the rise completes ``plateau`` generations ago.

    ``rise_time`` is monotone decreasing in s_het, so this is a clean 1-D root.
    """
    t_rise = t_inv - plateau
    if t_rise <= 0:
        raise ValueError(
            f"plateau ({plateau!r}) must be < t_inv ({t_inv!r})")
    return float(optimize.brentq(
        lambda s: B.rise_time(p_start, s) - t_rise, 1e-9, 1.0, rtol=1e-12))


def run_one(job) -> dict:
    t_inv, p_start, plateau, rep = job
    s_het = s_het_for(p_start, t_inv, plateau)
    L = seq_length()
    # Seed unique per (cell, rep) and stable across runs.
    seed = (900_000
            + 100_000 * PLATEAU_ARMS.index(plateau)
            + 10_000 * T_INV_GRID.index(t_inv)
            + 1_000 * P_START_GRID.index(p_start)
            + rep)
    t0 = time.time()
    sim = B.build_balancing_sim(seq_length=L, t_inv=t_inv, s_het=s_het,
                                p_start=p_start, gamma=GAMMA, seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    st = stats.arrangement_stats(ts, i_nodes, s_nodes,
                                 interval=model.inversion_interval(sim))
    return {
        "t_inv": t_inv, "p_start": p_start, "plateau": plateau, "rep": rep,
        "s_het": s_het, "s_hom": B.s_homozygote(s_het), "seed": seed,
        "seq_length": L, "num_trees": ts.num_trees,
        "wall_s": round(time.time() - t0, 2),
        **{k: st[k] for k in ("pi_i_over_pi_s", "dxy_over_pi_i",
                              "pi_i", "pi_s", "dxy", "fst")},
    }


def _bilinear(M, t, p):
    """M is indexed [t_index][p_start_index]; interpolate at (t, p)."""
    it = float(np.interp(t, T_INV_GRID, np.arange(len(T_INV_GRID))))
    ip = float(np.interp(p, P_START_GRID, np.arange(len(P_START_GRID))))
    i0 = min(int(np.floor(it)), len(T_INV_GRID) - 2)
    j0 = min(int(np.floor(ip)), len(P_START_GRID) - 2)
    ft, fp = it - i0, ip - j0
    return ((1 - ft) * ((1 - fp) * M[i0][j0] + fp * M[i0][j0 + 1])
            + ft * ((1 - fp) * M[i0 + 1][j0] + fp * M[i0 + 1][j0 + 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=96)
    ap.add_argument("--workers", type=int, default=26)
    args = ap.parse_args()

    # ---- 1. the neutral alternative, no simulation needed ----------------
    print("NEUTRAL ALTERNATIVE (exact diffusion; no mu, no mask)")
    print("  E[generations to reach p=0.626 | it gets there] = 1.650 * Ne")
    for name, ne in (("N_ANC", N_ANC), ("N(t=7.2e5)", float(N_growth(7.2e5))),
                     ("N0 (today)", N0)):
        print(f"    {name:12s} Ne={ne:11,.0f}  ->  "
              f"{B.neutral_hitting_time(n_e=ne):13,.0f} generations "
              f"  P(ever reaching 0.626) = "
              f"{B.neutral_reach_probability(n_e=ne):.2e}")
    print("  against a fitted age of ~7.2e5 generations: neutral drift cannot "
          "deliver\n  the observed frequency in the time the divergence allows.")

    L = seq_length()
    print(f"\nL = {L:,} bp (rho={RHO:g}), r = {REC_RATE:g}, "
          f"reps = {args.reps}/cell, workers = {args.workers}")

    jobs = [(t, p, pl, rep) for pl in PLATEAU_ARMS for t in T_INV_GRID
            for p in P_START_GRID for rep in range(args.reps)]
    print(f"{len(jobs):,} simulations")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f} s")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 2. cell means and the crossing ---------------------------------
    print(f"\n{'plateau':>8s} {'t_inv':>8s} {'p_start':>7s} {'s_het':>9s} "
          f"{'pi_I/pi_S':>17s} {'dxy/pi_I':>17s} {'miss_r':>7s} {'miss_d':>7s} "
          f"{'Fst':>7s}")
    cells = {}
    for pl in PLATEAU_ARMS:
        for t in T_INV_GRID:
            for p in P_START_GRID:
                sub = [x for x in rows if x["t_inv"] == t
                       and x["p_start"] == p and x["plateau"] == pl]
                rv = np.array([x["pi_i_over_pi_s"] for x in sub])
                dv = np.array([x["dxy_over_pi_i"] for x in sub])
                n = len(sub)
                cells[(pl, t, p)] = {
                    "n": n,
                    "r": float(rv.mean()),
                    "r_sem": float(rv.std(ddof=1) / n ** 0.5),
                    "d": float(dv.mean()),
                    "d_sem": float(dv.std(ddof=1) / n ** 0.5),
                    "fst": float(np.mean([x["fst"] for x in sub])),
                    "s_het": sub[0]["s_het"],
                }
                c = cells[(pl, t, p)]
                print(f"{pl:8,.0f} {t:8,.0f} {p:7.3f} {c['s_het']:9.2e} "
                      f"{c['r']:10.4f}+-{c['r_sem']:.4f} "
                      f"{c['d']:10.4f}+-{c['d_sem']:.4f} "
                      f"{100 * (c['r'] - TARGET_R) / TARGET_R:+6.1f}% "
                      f"{100 * (c['d'] - TARGET_D) / TARGET_D:+6.1f}% "
                      f"{c['fst']:7.4f}")
    print(f"{'TARGET':>8s} {'':>8s} {'':>7s} {'':>9s} {TARGET_R:10.4f}"
          f"         {TARGET_D:10.4f}                          "
          f"{empirical.FST:.4f}")

    roots = {}
    print()
    for pl in PLATEAU_ARMS:
        R = [[cells[(pl, t, p)]["r"] for p in P_START_GRID]
             for t in T_INV_GRID]
        D = [[cells[(pl, t, p)]["d"] for p in P_START_GRID]
             for t in T_INV_GRID]

        def resid(v, R=R, D=D):
            t, p = v
            return [(_bilinear(R, t, p) - TARGET_R) / TARGET_R,
                    (_bilinear(D, t, p) - TARGET_D) / TARGET_D]

        sol = optimize.least_squares(
            resid, [7.0e5, 0.025],
            bounds=([T_INV_GRID[0], P_START_GRID[0]],
                    [T_INV_GRID[-1], P_START_GRID[-1]]))
        t_hat, p_hat = float(sol.x[0]), float(sol.x[1])
        s_hat = s_het_for(p_hat, t_hat, pl)
        roots[pl] = {"t_inv": t_hat, "p_start": p_hat, "s_het": s_hat,
                     "s_hom": B.s_homozygote(s_hat),
                     "resid_r": float(resid(sol.x)[0]),
                     "resid_d": float(resid(sol.x)[1])}
        print(f"plateau={pl:9,.0f}:  t_inv = {t_hat:9,.0f}   "
              f"p_start = {p_hat:.4f}   s_het = {s_hat:.2e}   "
              f"(residuals {100 * roots[pl]['resid_r']:+.2f}% / "
              f"{100 * roots[pl]['resid_d']:+.2f}%)")

    ts_ = [r["t_inv"] for r in roots.values()]
    print(f"\nage across the degenerate plateau dimension: "
          f"{min(ts_):,.0f}-{max(ts_):,.0f} generations "
          f"({100 * (max(ts_) - min(ts_)) / np.mean(ts_):.1f}% spread), "
          f"while\np_start spans "
          f"{min(r['p_start'] for r in roots.values()):.4f}-"
          f"{max(r['p_start'] for r in roots.values()):.4f} and s_het "
          f"{min(r['s_het'] for r in roots.values()):.2e}-"
          f"{max(r['s_het'] for r in roots.values()):.2e}.")
    print("So the AGE is identified; the founding frequency and the selection "
          "strength\nare only jointly constrained. Generations = years for "
          "Illex, and every age\nscales inversely with mu = 3e-9.")

    # ---- 3. how tightly the targets pin the age -------------------------
    # dxy/pi_I is what carries the age, so report its local sensitivity: a
    # reader with an error bar on the empirical target can propagate it.
    pl0 = PLATEAU_ARMS[0]
    p_mid = P_START_GRID[1]
    dd = [cells[(pl0, t, p_mid)]["d"] for t in T_INV_GRID]
    slope = (dd[-1] - dd[0]) / (T_INV_GRID[-1] - T_INV_GRID[0])
    print(f"\nsensitivity at p_start={p_mid}: d(dxy/pi_I)/d(t_inv) = "
          f"{slope * 1e5:.3f} per 100,000 generations,")
    print(f"  so a 1% error in the empirical dxy/pi_I moves the age by "
          f"{0.01 * TARGET_D / slope:,.0f} generations")
    print(f"  and a 5% error moves it by "
          f"{0.05 * TARGET_D / slope:,.0f} generations.")

    OUT_JSON.write_text(json.dumps({
        "targets": {"pi_i_over_pi_s": TARGET_R, "dxy_over_pi_i": TARGET_D,
                    "fst_redundant": empirical.FST},
        "fixed": {"rho": RHO, "seq_length": L, "recomb_rate": REC_RATE,
                  "gamma": GAMMA, "p_star": 0.626, "arm": "growth",
                  "mu": 3e-9},
        "neutral": {
            "hitting_time_coefficient_times_ne": 1.650,
            "by_ne": {name: {"ne": ne,
                             "e_time": B.neutral_hitting_time(n_e=ne),
                             "p_reach": B.neutral_reach_probability(n_e=ne)}
                      for name, ne in (("N_ANC", N_ANC),
                                       ("N_at_7.2e5", float(N_growth(7.2e5))),
                                       ("N0", N0))},
        },
        "cells": [{"plateau": pl, "t_inv": t, "p_start": p, **cells[(pl, t, p)]}
                  for (pl, t, p) in cells],
        "roots": {str(k): v for k, v in roots.items()},
        "age_sensitivity_d_per_1e5_gen": slope * 1e5,
        "reps": args.reps,
    }, indent=2))
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
