#!/usr/bin/env python3
"""
Simulate the Anopheles gambiae 2La inversion using msinv.

Parameters from literature:
  Breakpoints: chr2L ~20.5-42.2 Mb (21.6 Mb inversion)
  Ne: ~10,000 (conservative estimate for savanna populations)
  mu: 3.5e-9 per bp per gen (Keightley et al. 2009)
  r: 2.0 cM/Mb = 2e-8 per bp per gen (homokaryotype)
  p_inv: varies 0.1-0.9 by ecology; use 0.5 for polymorphic pop
  c: low (heterokaryotype recombination < 0.5 cM/Mb vs 2.0)
  Empirical: Fst=0.567, Da=1.67%, pi(2La)=1.21%, pi(2L+a)=1.02%

We simulate a 100kb window within and flanking the inversion
to compare with empirical diversity/divergence patterns.
"""

import sys
import os
import numpy as np


import importlib.util
spec = importlib.util.spec_from_file_location(
    'msinv', os.path.join(os.path.dirname(__file__), 'msinv.py'))
msinv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msinv)

# ─── 2La parameters ──────────────────────────────────────────────────
Ne = 10000             # effective population size
MU = 3.5e-9            # per bp per gen
R = 2.0e-8             # per bp per gen (homokaryotype)
P_INV = 0.5            # balanced polymorphism
GENERATIONS_PER_YEAR = 10  # ~10 gen/year for A. gambiae

# Inversion: 21.6 Mb, but we simulate a scaled-down version
# to keep runtime manageable. Scale to L_sim bases.
INV_LENGTH_BP = 21.6e6
L_SIM = 10000          # simulated region length (bases)
SCALE = L_SIM / INV_LENGTH_BP  # scaling factor

# Gene flux: heterokaryotype recomb ~ 0.5 cM/Mb vs 2.0 cM/Mb homo
# c = rate of double crossover relative to single crossover
# Heterokaryotype rate / homokaryotype rate ≈ 0.5/2.0 = 0.25
# But gene flux = double crossover, which is much rarer
# Estimate c ~ 0.001-0.01 (very low)
C = 0.005

# Inversion age: 2La is old, shared across A. gambiae complex
# Estimated >100,000 years = >1,000,000 gen = 50 coalescent units
T_INV_GEN = 1000000    # ~100,000 years
T_INV_COAL = T_INV_GEN / (2 * Ne)

# Coalescent-scaled parameters
THETA = 4 * Ne * MU * L_SIM
RHO = 4 * Ne * R * L_SIM

# Inversion occupies central 60% of simulated region
# (mimics looking at a window that spans the breakpoint)
BP_LEFT = 0.2
BP_RIGHT = 0.8

NSAM = 20              # 10 per arrangement
N_STD = 10
N_INV = 10
NREPS = 200
NW = 10
SEED = 42


def run_simulation():
    """Run msinv with 2La parameters."""
    print("=== Anopheles gambiae 2La Inversion Simulation ===")
    print(f"Ne={Ne}, mu={MU}, r={R}")
    print(f"L_sim={L_SIM} bp, theta={THETA:.1f}, rho={RHO:.1f}")
    print(f"p_inv={P_INV}, c={C}, t_inv={T_INV_COAL:.1f} (2N gen)")
    print(f"Inversion region: [{BP_LEFT:.1f}, {BP_RIGHT:.1f}]")
    print(f"Samples: {N_STD} 2L+a (S) + {N_INV} 2La (I)")
    print()

    # Use stochastic trajectory for realistic inversion history
    rng_traj = np.random.default_rng(SEED)
    traj = msinv.StochasticTrajectory(P_INV, Ne, s=0.001, rng=rng_traj)
    print(f"Stochastic trajectory: t_inv={traj.t_inv:.2f} (2N gen)")

    sim = msinv.MsinvSimulator(
        nsam=NSAM, nreps=NREPS, theta=THETA, rho=RHO, nsites=L_SIM,
        n_std=N_STD, n_inv=N_INV, p_inv=P_INV, c=C,
        seed=SEED, p_inv_func=traj,
        bp_left=BP_LEFT, bp_right=BP_RIGHT)

    # Collect statistics
    wins = np.linspace(0, 1, NW + 1)
    mid = np.linspace(0.5/NW, 1 - 0.5/NW, NW)

    dxy_sum = np.zeros(NW)    # between-arrangement divergence
    pi_std_sum = np.zeros(NW) # within 2L+a diversity
    pi_inv_sum = np.zeros(NW) # within 2La diversity
    fst_sum = np.zeros(NW)
    n_ok = 0

    for rep in range(NREPS):
        pos, haps = sim.simulate_one()
        if len(pos) == 0:
            continue
        n_ok += 1

        s_haps = haps[:N_STD, :]   # 2L+a (Standard)
        i_haps = haps[N_STD:, :]   # 2La (Inverted)

        for w in range(NW):
            lo, hi = wins[w], wins[w+1]
            idx = [j for j, p in enumerate(pos) if lo <= p < hi]
            if not idx:
                continue

            sh = s_haps[:, idx]
            ih = i_haps[:, idx]

            # dxy (between arrangements)
            diffs_si = sum(np.sum(sh[a] != ih[b])
                          for a in range(N_STD) for b in range(N_INV))
            dxy = diffs_si / (N_STD * N_INV)
            dxy_sum[w] += dxy

            # pi within 2L+a
            diffs_ss = sum(np.sum(sh[a] != sh[b])
                          for a in range(N_STD) for b in range(a+1, N_STD))
            pi_s = diffs_ss / (N_STD * (N_STD-1) / 2)
            pi_std_sum[w] += pi_s

            # pi within 2La
            diffs_ii = sum(np.sum(ih[a] != ih[b])
                          for a in range(N_INV) for b in range(a+1, N_INV))
            pi_i = diffs_ii / (N_INV * (N_INV-1) / 2)
            pi_inv_sum[w] += pi_i

            # Fst = 1 - (pi_within / pi_total)
            pi_within = (pi_s + pi_i) / 2
            pi_total = dxy  # approximation: dxy ≈ pi_total for diverged pops
            if pi_total > 0:
                fst_sum[w] += 1.0 - pi_within / pi_total

    if n_ok == 0:
        print("No successful replicates!")
        return

    dxy = dxy_sum / n_ok
    pi_std = pi_std_sum / n_ok
    pi_inv = pi_inv_sum / n_ok
    fst = fst_sum / n_ok

    # Print results
    print(f"\n{n_ok}/{NREPS} replicates")
    print(f"\n{'Win':>4} {'Mid':>6} {'Region':>8} {'dxy':>8} {'pi_S':>8} "
          f"{'pi_I':>8} {'Fst':>8} {'dxy/pi_S':>8}")
    print("-" * 65)

    for w in range(NW):
        m = mid[w]
        rgn = "inv" if BP_LEFT < m < BP_RIGHT else "col"
        ratio = dxy[w] / pi_std[w] if pi_std[w] > 0 else 0
        print(f"{w:>4d} {m:>6.2f} {rgn:>8} {dxy[w]:>8.2f} "
              f"{pi_std[w]:>8.2f} {pi_inv[w]:>8.2f} "
              f"{fst[w]:>8.3f} {ratio:>8.1f}")

    # Summary
    out = [w for w in range(NW) if mid[w] < BP_LEFT or mid[w] > BP_RIGHT]
    ins = [w for w in range(NW) if BP_LEFT < mid[w] < BP_RIGHT]

    print(f"\nSummary:")
    for name, vals in [("dxy", dxy), ("pi_S", pi_std), ("pi_I", pi_inv), ("Fst", fst)]:
        o = np.mean([vals[w] for w in out]) if out else 0
        n_ = np.mean([vals[w] for w in ins]) if ins else 0
        print(f"  {name:>5}: collinear={o:.2f}, inversion={n_:.2f}, "
              f"ratio={n_/o if o > 0 else 0:.2f}")

    # Compare with empirical data
    print(f"\nEmpirical comparison (Cheng et al. 2012):")
    print(f"  Empirical Fst(2La): 0.567")
    print(f"  Simulated Fst(inv): {np.mean([fst[w] for w in ins]):.3f}")
    print(f"  Empirical pi(2La):  1.21%")
    print(f"  Empirical pi(2L+a): 1.02%")


if __name__ == "__main__":
    run_simulation()
