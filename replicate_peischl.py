#!/usr/bin/env python3
"""
Replicate Peischl et al. 2013 Figure 4: coalescence time distributions.

Parameters (from paper):
  N = 500, rho = 4Nr = 0.1, c = 0.01, p_inv = 0.5
  Sample: n=2 (one S, one I)
  Positions: x = 0.01, 0.1, 0.5 within inversion [0, 1]

Expected patterns:
  - T_SI is highest near breakpoints (x=0.01), lowest at center (x=0.5)
  - T_SS ≈ T_II ≈ 2N (standard coalescent)
  - T_SI >> T_SS at breakpoints
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    'msinv_n2', os.path.expanduser('~/msinv.py'))
msinv_n2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msinv_n2)

# Peischl parameters
N = 500
RHO = 0.1        # 4Nr for the inversion
C = 0.01          # gene flux coefficient
P_INV = 0.5
FLUX_W = 0.3     # not specified in paper, use default

# For n=2 msinv, theta doesn't matter for coalescence times
# but we need it for output. Use theta = 4Nmu = 1 (arbitrary)
THETA = 1.0
NSITES = 100      # discrete sites within inversion

NREPS = 5000      # many reps for good distributions

# Positions to measure (in inversion coordinates [0,1])
positions = [0.01, 0.05, 0.1, 0.25, 0.5]


def measure_coalescence_times():
    """Run msinv at n=2 for SI pairs, measure T at different positions."""
    # Use constant frequency (old inversion, effectively infinite age)
    # but with t_inv to prevent infinite T at breakpoints
    t_inv = 20.0  # 20 × 2N = 20000 gen — very old inversion

    p_func = msinv_n2.ConstantFrequency(P_INV, t_inv=t_inv)
    rng = np.random.default_rng(42)

    # For each position, track the coalescence time
    # At n=2, the tree has one T value (the TMRCA)
    # The tree at position x has T that depends on the class composition

    # We'll run the full chromosome simulation and extract T at each position
    # by looking at where mutations fall

    # Actually, for measuring T directly, we use the structured coalescent
    # at a single site. This is what build_initial_tree does in msinv_n2.

    results = {}
    for pair_type, c0, c1, label in [
            ('SI', 0, 1, 'S-I'),
            ('SS', 0, 0, 'S-S'),
            ('II', 1, 1, 'I-I')]:
        for x in positions:
            phi_x = msinv_n2.phi(x, FLUX_W)
            T_vals = []
            for _ in range(NREPS):
                tree = msinv_n2.build_initial_tree(
                    c0, c1, p_func, C, RHO, phi_x, rng)
                T_vals.append(tree.t_coal)

            mean_T = np.mean(T_vals)
            median_T = np.median(T_vals)
            results[(pair_type, x)] = (mean_T, median_T, T_vals)

    return results


def main():
    print("=== Replicating Peischl et al. 2013 Figure 4 ===")
    print(f"N={N}, rho={RHO}, c={C}, p_inv={P_INV}, t_inv=20")
    print(f"{NREPS} replicates per condition")
    print()

    results = measure_coalescence_times()

    # Print table
    print(f"{'Pair':>5} {'x':>6} {'phi(x)':>8} {'E[T]':>8} {'med[T]':>8} "
          f"{'E[T]/E[T_SS]':>12}")
    print("-" * 55)

    for pair_type in ['SS', 'II', 'SI']:
        for x in positions:
            mean_T, median_T, _ = results[(pair_type, x)]
            # Normalize by SS at same position
            ss_mean = results[('SS', x)][0]
            ratio = mean_T / ss_mean if ss_mean > 0 else 0
            phi_x = msinv_n2.phi(x, FLUX_W)
            print(f"{pair_type:>5} {x:>6.2f} {phi_x:>8.3f} "
                  f"{mean_T:>8.2f} {median_T:>8.2f} {ratio:>12.2f}")
        print()

    # Key predictions from Peischl:
    # 1. T_SI >> T_SS at breakpoints (x=0.01)
    # 2. T_SI decreases toward center (x=0.5)
    # 3. T_SS ≈ T_II ≈ 1 (in 2N units) = 2 (in N units)

    print("Key checks:")
    t_si_bp = results[('SI', 0.01)][0]
    t_si_ctr = results[('SI', 0.5)][0]
    t_ss = results[('SS', 0.5)][0]

    print(f"  T_SI at breakpoint (x=0.01): {t_si_bp:.2f}")
    print(f"  T_SI at center (x=0.5):      {t_si_ctr:.2f}")
    print(f"  T_SS at center:              {t_ss:.2f}")
    print(f"  T_SI/T_SS at breakpoint:     {t_si_bp/t_ss if t_ss > 0 else 0:.1f}x")
    print(f"  T_SI/T_SS at center:         {t_si_ctr/t_ss if t_ss > 0 else 0:.1f}x")
    print(f"  Breakpoint/center ratio:     {t_si_bp/t_si_ctr if t_si_ctr > 0 else 0:.1f}x")


if __name__ == "__main__":
    main()
