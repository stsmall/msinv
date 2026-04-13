#!/usr/bin/env python3
"""
Simulate the human MAPT inv17q21 (H1/H2) inversion.

Parameters from literature:
  Length: ~900 kb
  Age: ~3 million years (predates human-Neanderthal split)
  H2 frequency: ~20% in Europeans, ~10% in Africans
  Ne: ~10,000 (human)
  mu: 1.5e-8 per bp per gen
  r: ~1 cM/Mb = 1e-8 per bp per gen (suppressed in heterokaryotypes)
  Generation time: ~29 years
  >2,366 SNVs differentiate H1 and H2

Expected: very high divergence between H1 and H2, extended LD,
essentially no gene flux over 3 My.
"""

import sys
import os
import numpy as np


import importlib.util
spec = importlib.util.spec_from_file_location(
    'msinv', os.path.join(os.path.dirname(__file__), 'msinv.py'))
msinv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msinv)

# ─── MAPT H1/H2 parameters ──────────────────────────────────────────
Ne = 10000
MU = 1.5e-8            # human mutation rate per bp per gen
R = 1.0e-8             # recomb rate per bp per gen
GEN_TIME = 29          # years per generation
P_INV = 0.2            # H2 frequency in Europeans

# Inversion: 900 kb, but simulate scaled
L_SIM = 5000           # simulated region (bases)
INV_LENGTH = 900000    # actual length

# Gene flux: essentially zero for MAPT (complete suppression)
C = 0.0001             # very low — almost no exchange

# Age: 3 million years
T_INV_YEARS = 3000000
T_INV_GEN = T_INV_YEARS / GEN_TIME
T_INV_COAL = T_INV_GEN / (2 * Ne)

# Coalescent-scaled
THETA = 4 * Ne * MU * L_SIM
RHO = 4 * Ne * R * L_SIM

BP_LEFT = 0.2
BP_RIGHT = 0.8

NSAM = 20
N_STD = 16             # H1 (80%)
N_INV = 4              # H2 (20%)
NREPS = 200
NW = 10
SEED = 42


def run_simulation():
    print("=== Human MAPT inv17q21 (H1/H2) Simulation ===")
    print(f"Ne={Ne}, mu={MU}, r={R}")
    print(f"L_sim={L_SIM}, theta={THETA:.1f}, rho={RHO:.1f}")
    print(f"p_inv(H2)={P_INV}, c={C}")
    print(f"Age: {T_INV_YEARS/1e6:.1f} Myr = {T_INV_GEN:.0f} gen "
          f"= {T_INV_COAL:.1f} coalescent units")
    print(f"Samples: {N_STD} H1 + {N_INV} H2")
    print()

    # Deterministic trajectory: strong selection to reach 20% from 1/2N
    # s needed: logistic from 1/20000 to 0.2 in 3My/29yr = 103k gen
    # = 5.17 coalescent units
    traj = msinv.DeterministicTrajectory(P_INV, Ne, s=0.0001)
    print(f"Trajectory: deterministic, t_inv={traj.t_inv:.2f}")

    sim = msinv.MsinvSimulator(
        nsam=NSAM, nreps=NREPS, theta=THETA, rho=RHO, nsites=L_SIM,
        n_std=N_STD, n_inv=N_INV, p_inv=P_INV, c=C,
        seed=SEED, p_inv_func=traj,
        bp_left=BP_LEFT, bp_right=BP_RIGHT)

    wins = np.linspace(0, 1, NW + 1)
    mid = np.linspace(0.5/NW, 1 - 0.5/NW, NW)

    dxy_sum = np.zeros(NW)
    pi_h1_sum = np.zeros(NW)
    pi_h2_sum = np.zeros(NW)
    fst_sum = np.zeros(NW)
    n_ok = 0

    for rep in range(NREPS):
        pos, haps = sim.simulate_one()
        if len(pos) == 0:
            continue
        n_ok += 1

        h1 = haps[:N_STD, :]
        h2 = haps[N_STD:, :]

        for w in range(NW):
            lo, hi = wins[w], wins[w+1]
            idx = [j for j, p in enumerate(pos) if lo <= p < hi]
            if not idx:
                continue

            h1w = h1[:, idx]
            h2w = h2[:, idx]

            # dxy
            diffs = sum(np.sum(h1w[a] != h2w[b])
                       for a in range(N_STD) for b in range(N_INV))
            dxy = diffs / (N_STD * N_INV)
            dxy_sum[w] += dxy

            # pi H1
            d1 = sum(np.sum(h1w[a] != h1w[b])
                    for a in range(N_STD) for b in range(a+1, N_STD))
            pi1 = d1 / (N_STD * (N_STD-1) / 2)
            pi_h1_sum[w] += pi1

            # pi H2
            if N_INV >= 2:
                d2 = sum(np.sum(h2w[a] != h2w[b])
                        for a in range(N_INV) for b in range(a+1, N_INV))
                pi2 = d2 / (N_INV * (N_INV-1) / 2)
                pi_h2_sum[w] += pi2

            # Fst
            pi_w = (pi1 + pi2) / 2 if N_INV >= 2 else pi1
            if dxy > 0:
                fst_sum[w] += 1.0 - pi_w / dxy

    if n_ok == 0:
        print("No successful replicates!")
        return

    dxy = dxy_sum / n_ok
    pi_h1 = pi_h1_sum / n_ok
    pi_h2 = pi_h2_sum / n_ok
    fst = fst_sum / n_ok

    print(f"{n_ok}/{NREPS} replicates\n")
    print(f"{'Win':>4} {'Mid':>6} {'Rgn':>5} {'dxy':>7} {'pi_H1':>7} "
          f"{'pi_H2':>7} {'Fst':>7} {'dxy/pi':>7}")
    print("-" * 55)

    for w in range(NW):
        m = mid[w]
        rgn = "inv" if BP_LEFT < m < BP_RIGHT else "col"
        ratio = dxy[w] / pi_h1[w] if pi_h1[w] > 0 else 0
        print(f"{w:>4d} {m:>6.2f} {rgn:>5} {dxy[w]:>7.2f} "
              f"{pi_h1[w]:>7.2f} {pi_h2[w]:>7.2f} "
              f"{fst[w]:>7.3f} {ratio:>7.1f}")

    out = [w for w in range(NW) if mid[w] < BP_LEFT or mid[w] > BP_RIGHT]
    ins = [w for w in range(NW) if BP_LEFT < mid[w] < BP_RIGHT]

    print(f"\nSummary:")
    for name, vals in [("dxy", dxy), ("pi_H1", pi_h1), ("pi_H2", pi_h2), ("Fst", fst)]:
        o = np.mean([vals[w] for w in out]) if out else 0
        n_ = np.mean([vals[w] for w in ins]) if ins else 0
        print(f"  {name:>6}: collinear={o:.3f}, inversion={n_:.3f}")

    # Scale to per-site values for comparison with empirical
    per_site_dxy_inv = np.mean([dxy[w] for w in ins]) / (L_SIM / NW)
    per_site_pi_h1 = np.mean([pi_h1[w] for w in ins]) / (L_SIM / NW)
    print(f"\nPer-site statistics (inside inversion):")
    print(f"  dxy/site = {per_site_dxy_inv:.6f}")
    print(f"  pi_H1/site = {per_site_pi_h1:.6f}")

    print(f"\nExpected from empirical data:")
    print(f"  >2,366 SNVs in 900kb → ~0.0026 SNVs/site between H1-H2")
    print(f"  Human pi ≈ 0.001")


if __name__ == "__main__":
    run_simulation()
