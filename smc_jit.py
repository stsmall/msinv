"""Numba JIT-compiled SMC prune-and-reattach operations."""
import numpy as np
from numba import njit


@njit
def panmictic_prune_reattach(parent, time, klass, left_child, right_sib,
                              sample_id, n_nodes, root):
    """Full panmictic prune-and-reattach in Numba."""
    # Get branches
    count = 0
    indices = np.empty(n_nodes + 100, dtype=np.int32)
    lengths = np.empty(n_nodes + 100, dtype=np.float64)
    for i in range(n_nodes):
        p = parent[i]
        if p >= 0:
            bl = time[p] - time[i]
            if bl > 0:
                indices[count] = i
                lengths[count] = bl
                count += 1
    if count == 0:
        return root, n_nodes

    # Choose branch
    total_L = 0.0
    for i in range(count):
        total_L += lengths[i]
    r = np.random.random() * total_L
    cum = 0.0
    bi = count - 1
    for i in range(count):
        cum += lengths[i]
        if r < cum:
            bi = i
            break
    target = indices[bi]
    t_cut = time[target] + np.random.random() * lengths[bi]

    # Prune
    p = parent[target]
    if p < 0:
        return root, n_nodes

    sib = np.int32(-1)
    c = left_child[p]
    nc = np.int32(0)
    while c >= 0:
        nc += 1
        if c != target:
            sib = c
        c = right_sib[c]
    if nc != 2 or sib < 0:
        return root, n_nodes

    gp = parent[p]
    left_child[p] = -1
    parent[target] = -1
    right_sib[target] = -1
    parent[sib] = -1
    right_sib[sib] = -1

    if gp >= 0:
        prev = np.int32(-1)
        c = left_child[gp]
        while c >= 0:
            if c == p:
                if prev < 0:
                    left_child[gp] = sib
                else:
                    right_sib[prev] = sib
                right_sib[sib] = right_sib[p]
                parent[sib] = gp
                break
            prev = c
            c = right_sib[c]
    right_sib[p] = -1

    new_root = sib if root == p else root

    # Reattach
    above_count = np.int32(0)
    above_idx = np.empty(n_nodes + 100, dtype=np.int32)
    above_len = np.empty(n_nodes + 100, dtype=np.float64)
    for i in range(n_nodes):
        if i == target or i == p:
            continue
        pi = parent[i]
        if pi >= 0 and time[i] <= t_cut < time[pi]:
            above_idx[above_count] = i
            above_len[above_count] = time[pi] - t_cut
            above_count += 1

    if above_count > 0:
        total_a = 0.0
        for i in range(above_count):
            total_a += above_len[i]
        r2 = np.random.random() * total_a
        cum2 = 0.0
        ai = above_count - 1
        for i in range(above_count):
            cum2 += above_len[i]
            if r2 < cum2:
                ai = i
                break
        attach = above_idx[ai]
        ap = parent[attach]
        t_a = t_cut + np.random.random() * (time[ap] - t_cut)

        coal = n_nodes
        time[coal] = t_a
        klass[coal] = klass[attach]
        left_child[coal] = -1
        right_sib[coal] = -1
        parent[coal] = -1
        sample_id[coal] = -1
        n_nodes += 1

        prev2 = np.int32(-1)
        c = left_child[ap]
        while c >= 0:
            if c == attach:
                if prev2 < 0:
                    left_child[ap] = coal
                else:
                    right_sib[prev2] = coal
                right_sib[coal] = right_sib[attach]
                parent[coal] = ap
                break
            prev2 = c
            c = right_sib[c]

        left_child[coal] = attach
        right_sib[attach] = target
        right_sib[target] = -1
        parent[attach] = coal
        parent[target] = coal
    else:
        coal = n_nodes
        t_c = t_cut
        if time[new_root] > t_c:
            t_c = time[new_root]
        t_c += np.random.exponential(1.0)
        time[coal] = t_c
        klass[coal] = np.int8(0)
        left_child[coal] = new_root
        right_sib[coal] = -1
        parent[coal] = -1
        sample_id[coal] = -1
        right_sib[new_root] = target
        parent[new_root] = coal
        parent[target] = coal
        right_sib[target] = -1
        n_nodes += 1
        new_root = coal

    return new_root, n_nodes
