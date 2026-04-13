#!/usr/bin/env python3
"""
Simulate a chromosome with a central inversion using the SMC.

Three regimes along the chromosome:
  [0, bp_left)       : collinear -- standard coalescent
  [bp_left, bp_right] : inverted  -- structured coalescent + gene flux
  (bp_right, 1]      : collinear -- standard coalescent

Outside the inversion, S/I lineages recombine freely (normal meiotic
recombination in heterokaryotypes).  Inside, heterokaryotypic
recombination is suppressed; only gene flux operates.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from msinv import (Node, get_all_nodes, get_branches, branch_lengths_by_class,
                   get_leaves_below, find_root, GeneFluxModel,
                   build_structured_tree, smc_prune_and_reattach,
                   drop_mutations)


def _standard_prune_and_reattach(root, rho, rng):
    """Standard SMC prune-and-reattach (no class structure)."""
    branches = get_branches(root)
    if not branches:
        return root
    lengths = np.array([bl for _, bl in branches])
    probs = lengths / lengths.sum()
    bi = rng.choice(len(branches), p=probs)
    target, tbl = branches[bi]
    t_cut = target.time + rng.random() * tbl

    # Prune
    current = target
    new_root = root
    pruned = False
    while current.parent is not None:
        p = current.parent
        if len(p.children) == 2:
            siblings = [ch for ch in p.children if ch is not current]
            if not siblings:
                break
            sibling = siblings[0]
            gp = p.parent
            sibling.parent = gp
            if gp is not None:
                gp.children = [sibling if ch is p else ch for ch in gp.children]
            new_root = sibling if new_root is p else new_root
            pruned = True
            break
        elif len(p.children) == 1:
            current = p
        else:
            break

    if not pruned:
        return root

    target.parent = None

    # Reattach: uniform on tree above t_cut (no class restriction)
    above = []
    for n in get_all_nodes(new_root):
        if n.parent is not None and n.parent.time > t_cut:
            lo = max(n.time, t_cut)
            hi = n.parent.time
            if hi > lo:
                above.append((n, lo, hi - lo))

    if above:
        lens = np.array([l for _, _, l in above])
        aprobs = lens / lens.sum()
        ai = rng.choice(len(above), p=aprobs)
        an, lo, _ = above[ai]
        t_a = lo + rng.random() * (an.parent.time - lo)
        coal = Node(time=t_a, branch_class=target.branch_class)
        old_p = an.parent
        coal.parent = old_p
        coal.children = [an, target]
        an.parent = coal
        target.parent = coal
        if old_p is not None:
            old_p.children = [coal if ch is an else ch for ch in old_p.children]
        new_root = find_root(new_root)
    else:
        t_c = max(t_cut, new_root.time) + rng.exponential(1.0)
        coal = Node(time=t_c, branch_class=target.branch_class)
        coal.children = [new_root, target]
        new_root.parent = coal
        target.parent = coal
        new_root = coal

    return new_root


def simulate_partial_inversion(nsam, n_std, n_inv, theta, rho, nsites,
                                p_inv, c, bp_left, bp_right,
                                flux_window=0.3, seed=None):
    """
    Simulate a chromosome with a central inversion.

    Args:
        bp_left, bp_right: inversion breakpoints in [0,1]
        Other args: as in MsinvSimulator

    Returns:
        (positions, haplotypes)
    """
    rng = np.random.default_rng(seed)
    flux_model = GeneFluxModel(w=flux_window)
    inv_len = bp_right - bp_left
    p_std = 1.0 - p_inv

    # Build initial tree at position 0 (collinear region).
    # In collinear regions, S and I lineages can freely recombine,
    # so the tree is effectively a standard coalescent.
    # But we maintain S/I labels for when we enter the inversion.
    leaves = []
    for i in range(n_std):
        leaves.append(Node(time=0.0, sample_id=i, branch_class='S'))
    for i in range(n_inv):
        leaves.append(Node(time=0.0, sample_id=n_std + i, branch_class='I'))

    active = list(leaves)
    t = 0.0
    while len(active) > 1:
        k = len(active)
        rate = k * (k - 1) / 2.0
        t += rng.exponential(1.0 / rate)
        idx = rng.choice(k, size=2, replace=False)
        n1, n2 = active[idx[0]], active[idx[1]]
        coal = Node(time=t, branch_class=n1.branch_class)
        coal.children = [n1, n2]
        n1.parent = coal
        n2.parent = coal
        for i in sorted(idx, reverse=True):
            active.pop(i)
        active.append(coal)
    root = active[0]

    # SMC across chromosome
    trees = []
    pos = 0.0
    prev_in_inv = False

    for _ in range(500000):
        # Determine regime
        in_inversion = bp_left <= pos <= bp_right

        # --- Boundary transition: rebuild tree at breakpoints ---
        if in_inversion and not prev_in_inv:
            # Entering inversion: rebuild with structured coalescent
            # so S/I branch classes reflect the correct marginal distribution
            inv_pos = max(0.02, (pos - bp_left) / inv_len)
            phi_x = flux_model.phi(inv_pos)
            root, _ = build_structured_tree(
                n_std, n_inv, p_inv, c, rho, phi_x, rng
            )
        elif not in_inversion and prev_in_inv:
            # Exiting inversion: rebuild as standard coalescent
            all_leaves_list = []
            for i in range(n_std):
                all_leaves_list.append(
                    Node(time=0.0, sample_id=i, branch_class='S'))
            for i in range(n_inv):
                all_leaves_list.append(
                    Node(time=0.0, sample_id=n_std + i, branch_class='I'))
            active_tmp = list(all_leaves_list)
            t_tmp = 0.0
            while len(active_tmp) > 1:
                k = len(active_tmp)
                rate_tmp = k * (k - 1) / 2.0
                t_tmp += rng.exponential(1.0 / rate_tmp)
                idx = rng.choice(k, size=2, replace=False)
                n1, n2 = active_tmp[idx[0]], active_tmp[idx[1]]
                coal = Node(time=t_tmp, branch_class='S')
                coal.children = [n1, n2]
                n1.parent = coal
                n2.parent = coal
                for i in sorted(idx, reverse=True):
                    active_tmp.pop(i)
                active_tmp.append(coal)
            root = active_tmp[0]

        prev_in_inv = in_inversion

        if in_inversion:
            L_S, L_I = branch_lengths_by_class(root)
            weighted_L = L_S * p_std + L_I * p_inv
            if weighted_L <= 0:
                trees.append((root, pos, 1.0))
                break
            rate = (rho / 2.0) * weighted_L
        else:
            L_total = sum(bl for _, bl in get_branches(root))
            if L_total <= 0:
                trees.append((root, pos, 1.0))
                break
            rate = (rho / 2.0) * L_total

        dx = rng.exponential(1.0 / rate)
        new_pos = pos + dx

        if new_pos >= 1.0:
            trees.append((root, pos, 1.0))
            break

        trees.append((root, pos, new_pos))
        pos = new_pos

        # Determine new regime
        new_in_inv = bp_left <= pos <= bp_right

        if new_in_inv:
            # Inside inversion: class-aware reattachment
            # Map position to inversion [0,1] coordinates
            inv_pos = (pos - bp_left) / inv_len
            # Clamp to avoid exact breakpoints
            inv_pos = max(0.02, min(0.98, inv_pos))
            phi_x = flux_model.phi(inv_pos)

            L_S, L_I = branch_lengths_by_class(root)
            weighted_L = L_S * p_std + L_I * p_inv
            u = rng.random() * weighted_L
            recomb_class = 'S' if u < L_S * p_std else 'I'

            root = smc_prune_and_reattach(
                root, recomb_class, p_inv, c, rho, phi_x, rng
            )
        else:
            # Outside inversion: standard reattachment
            root = _standard_prune_and_reattach(root, rho, rng)

    return drop_mutations(trees, theta, nsam, rng)


def compute_windowed_dxy(positions, haplotypes, n_std, n_inv, n_windows=50):
    """Compute windowed dxy(S/I) and pi."""
    if len(positions) == 0:
        return np.linspace(0, 1, n_windows), np.zeros(n_windows), np.zeros(n_windows), np.zeros(n_windows)

    pos = np.array(positions)
    haps = haplotypes
    haps_S = haps[:n_std]
    haps_I = haps[n_std:n_std + n_inv]

    edges = np.linspace(0, 1, n_windows + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    win_w = 1.0 / n_windows

    dxy = np.zeros(n_windows)
    pi_s = np.zeros(n_windows)
    pi_i = np.zeros(n_windows)

    for w in range(n_windows):
        mask = (pos >= edges[w]) & (pos < edges[w + 1])
        if not np.any(mask):
            continue
        hs = haps_S[:, mask]
        hi = haps_I[:, mask]

        # dxy
        f1 = hs.mean(axis=0)
        f2 = hi.mean(axis=0)
        dxy[w] = np.sum(f1 * (1 - f2) + f2 * (1 - f1)) / win_w

        # pi within S
        n = hs.shape[0]
        if n >= 2:
            fs = hs.mean(axis=0)
            pi_s[w] = np.sum(2 * fs * (1 - fs) * n / (n - 1)) / win_w

        # pi within I
        n = hi.shape[0]
        if n >= 2:
            fi = hi.mean(axis=0)
            pi_i[w] = np.sum(2 * fi * (1 - fi) * n / (n - 1)) / win_w

    return mids, dxy, pi_s, pi_i


def compute_ld_matrix(positions, haplotypes, n_bins=50):
    """
    Compute mean r² between pairs of SNPs binned by position.
    Returns a 2D matrix of mean r² for position bin pairs.
    """
    pos = np.array(positions)
    haps = haplotypes
    nsites = haps.shape[1]

    if nsites < 2 or nsites > 5000:
        # Subsample if too many SNPs
        if nsites > 5000:
            idx = np.sort(np.random.choice(nsites, 5000, replace=False))
            pos = pos[idx]
            haps = haps[:, idx]
            nsites = 5000

    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(pos, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Compute allele frequencies and center
    freqs = haps.mean(axis=0)
    # Filter out monomorphic or near-fixed sites
    keep = (freqs > 0.05) & (freqs < 0.95)
    pos = pos[keep]
    haps = haps[:, keep]
    freqs = freqs[keep]
    bin_idx = bin_idx[keep]
    nsites = haps.shape[1]

    if nsites < 2:
        return np.zeros((n_bins, n_bins))

    # Center haplotypes
    centered = haps - freqs[np.newaxis, :]
    var = freqs * (1 - freqs)

    # Compute r² matrix by bins (subsample for speed)
    ld_matrix = np.zeros((n_bins, n_bins))
    ld_counts = np.zeros((n_bins, n_bins))

    # Subsample pairs for speed
    max_pairs = 50000
    if nsites * (nsites - 1) // 2 > max_pairs:
        pairs = set()
        while len(pairs) < max_pairs:
            i = np.random.randint(nsites)
            j = np.random.randint(nsites)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
        pairs = list(pairs)
    else:
        pairs = [(i, j) for i in range(nsites) for j in range(i + 1, nsites)]

    for i, j in pairs:
        bi, bj = bin_idx[i], bin_idx[j]
        if var[i] > 0 and var[j] > 0:
            D = np.mean(centered[:, i] * centered[:, j])
            r2 = D ** 2 / (var[i] * var[j])
            ld_matrix[bi, bj] += r2
            ld_matrix[bj, bi] += r2
            ld_counts[bi, bj] += 1
            ld_counts[bj, bi] += 1

    mask = ld_counts > 0
    ld_matrix[mask] /= ld_counts[mask]

    return ld_matrix


if __name__ == '__main__':
    # Quick test
    pos, haps = simulate_partial_inversion(
        nsam=10, n_std=5, n_inv=5, theta=20, rho=100, nsites=1000,
        p_inv=0.5, c=0.01, bp_left=0.3, bp_right=0.7, seed=42
    )
    print(f"Seg sites: {len(pos)}")
    if len(pos) > 0:
        print(f"Position range: [{pos[0]:.3f}, {pos[-1]:.3f}]")
