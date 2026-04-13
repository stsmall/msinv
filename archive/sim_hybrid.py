#!/usr/bin/env python3
"""
Hybrid simulation of a chromosome with a central inversion.

Strategy:
  - Marginals (dxy, pi): site-by-site structured coalescent (EXACT)
  - LD:                   SMC (approximate but captures block structure)

This avoids the SMC class-balance bias for diversity statistics while
still producing the LD heatmap.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from msinv import (Node, get_all_nodes, get_branches, get_leaves_below,
                   find_root, GeneFluxModel, build_structured_tree)
from sim_partial_inv import (simulate_partial_inversion, compute_ld_matrix)


def exact_marginal_stats(n_std, n_inv, theta_per_site, rho, p_inv, c,
                         bp_left, bp_right, flux_window, nreps,
                         n_positions=200, seed=None):
    """
    Compute exact windowed dxy, pi_S, pi_I, pi_total using independent
    structured coalescent trees at each position.

    Outside the inversion: standard coalescent (all samples panmictic).
    Inside the inversion: structured coalescent with gene flux.

    Returns dict with arrays over positions.
    """
    rng = np.random.default_rng(seed)
    fm = GeneFluxModel(w=flux_window)
    inv_len = bp_right - bp_left
    nsam = n_std + n_inv

    S_ids = set(range(n_std))
    I_ids = set(range(n_std, nsam))

    positions = np.linspace(0.005, 0.995, n_positions)

    # Accumulators (per position, averaged over reps)
    acc_pi_s = np.zeros(n_positions)
    acc_pi_i = np.zeros(n_positions)
    acc_pi_t = np.zeros(n_positions)
    acc_dxy = np.zeros(n_positions)

    for rep in range(nreps):
        for pi, pos in enumerate(positions):
            in_inv = bp_left <= pos <= bp_right

            if in_inv:
                inv_pos = (pos - bp_left) / inv_len
                inv_pos = max(0.02, min(0.98, inv_pos))
                phi = fm.phi(inv_pos)
                root, leaves = build_structured_tree(
                    n_std, n_inv, p_inv, c, rho, phi, rng
                )
            else:
                # Standard coalescent (panmictic)
                leaves_list = []
                for i in range(n_std):
                    leaves_list.append(Node(time=0.0, sample_id=i, branch_class='S'))
                for i in range(n_inv):
                    leaves_list.append(Node(time=0.0, sample_id=n_std+i, branch_class='I'))
                active = list(leaves_list)
                t = 0.0
                while len(active) > 1:
                    k = len(active)
                    rate = k * (k - 1) / 2.0
                    t += rng.exponential(1.0 / rate)
                    idx = rng.choice(k, size=2, replace=False)
                    n1, n2 = active[idx[0]], active[idx[1]]
                    coal = Node(time=t, branch_class='S')
                    coal.children = [n1, n2]
                    n1.parent = coal
                    n2.parent = coal
                    for i in sorted(idx, reverse=True):
                        active.pop(i)
                    active.append(coal)
                root = active[0]
                leaves = leaves_list

            # Compute branch-length-based statistics
            branches = get_branches(root)
            pi_s = pi_i = pi_t = dxy_val = 0.0

            for node, bl in branches:
                below = get_leaves_below(node)
                n_below = len(below)
                n_above = nsam - n_below

                s_below = len(below & S_ids)
                s_above = n_std - s_below
                i_below = len(below & I_ids)
                i_above = n_inv - i_below

                # pi_S ∝ s_below * s_above * bl
                pi_s += bl * s_below * s_above
                # pi_I ∝ i_below * i_above * bl
                pi_i += bl * i_below * i_above
                # pi_total ∝ n_below * n_above * bl
                pi_t += bl * n_below * n_above
                # dxy ∝ (s_below * i_above + i_below * s_above) * bl
                dxy_val += bl * (s_below * i_above + i_below * s_above)

            # Normalize by C(n,2)
            c_s = n_std * (n_std - 1) / 2 if n_std >= 2 else 1
            c_i = n_inv * (n_inv - 1) / 2 if n_inv >= 2 else 1
            c_t = nsam * (nsam - 1) / 2
            c_si = n_std * n_inv if n_std > 0 and n_inv > 0 else 1

            acc_pi_s[pi] += pi_s / c_s
            acc_pi_i[pi] += pi_i / c_i
            acc_pi_t[pi] += pi_t / c_t
            acc_dxy[pi] += dxy_val / c_si

    # Average over reps
    acc_pi_s /= nreps
    acc_pi_i /= nreps
    acc_pi_t /= nreps
    acc_dxy /= nreps

    # Scale by theta_per_site / 2 to convert branch lengths to diversity
    # E[pi] = theta/2 * E[T] where theta = 4Nmu per site
    # But we want raw T values for comparison, so return both
    return {
        'positions': positions,
        'T_SS': acc_pi_s,         # expected coalescence time (within S)
        'T_II': acc_pi_i,         # expected coalescence time (within I)
        'T_total': acc_pi_t,      # expected coalescence time (total)
        'T_SI': acc_dxy,          # expected coalescence time (between S,I)
    }


if __name__ == '__main__':
    # Quick test
    result = exact_marginal_stats(
        n_std=5, n_inv=5, theta_per_site=0.02, rho=100,
        p_inv=0.5, c=0.01, bp_left=0.35, bp_right=0.65,
        flux_window=0.3, nreps=500, n_positions=20, seed=42
    )
    print("Position    T_SS    T_II    T_SI   T_SI/T_SS")
    for i in range(len(result['positions'])):
        pos = result['positions'][i]
        tss = result['T_SS'][i]
        tii = result['T_II'][i]
        tsi = result['T_SI'][i]
        ratio = tsi / tss if tss > 0 else 0
        in_inv = '*' if 0.35 <= pos <= 0.65 else ' '
        print(f"  {pos:.3f}   {tss:.2f}   {tii:.2f}   {tsi:.2f}   {ratio:.1f} {in_inv}")
