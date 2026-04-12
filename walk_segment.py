"""
Standalone SMC walk segment for bidirectional inversion simulation.

Extracts the walk logic so it can be called multiple times with
different start positions and directions.
"""


def run_walk_segment(sim, root, start_pos, end_pos, rng, n_std, n_inv,
                      bp_l, bp_r, inv_len):
    """
    Run a single left-to-right SMC walk from start_pos to end_pos.

    For leftward walks, the caller should mirror the chromosome:
    swap bp_l/bp_r and un-mirror mutation positions afterward.

    Returns (mutations, root) where mutations is list of (pos, leaf_ids).
    """
    from msinv import (get_all_nodes, get_branches, find_root,
                        _drop_muts_segment, _coalesce_pop, _flux_pop,
                        smc_prune_and_reattach, smc_prune_and_reattach_panmictic,
                        build_structured_tree, ConstantFrequency, Node)
    import numpy as np

    mutations = []
    pos = start_pos

    for _ in range(500000):
        if pos >= end_pos:
            break

        in_inv = bp_l <= pos < bp_r

        # Branch lengths
        L_S = L_I = 0.0
        t_max = 0.0
        stack = [root]
        while stack:
            n = stack.pop()
            if n.time > t_max:
                t_max = n.time
            if n.parent is not None:
                bl = n.parent.time - n.time
                if n.branch_class == 'S':
                    L_S += bl
                else:
                    L_I += bl
            for ch in n.children:
                stack.append(ch)
        L_total = L_S + L_I

        if in_inv and inv_len > 0:
            p_inv_t = sim.p_inv_func(0.5 * t_max)
            p_std_t = 1.0 - p_inv_t
            if p_inv_t > 0:
                weighted_L = L_S * p_std_t + L_I * p_inv_t
            else:
                weighted_L = L_total
            next_boundary = min(bp_r, end_pos)
        else:
            weighted_L = L_total
            if pos < bp_l:
                next_boundary = min(bp_l, end_pos)
            else:
                next_boundary = end_pos

        if weighted_L <= 0:
            _drop_muts_segment(root, pos, next_boundary,
                                sim.theta, rng, mutations)
            pos = next_boundary

            # Boundary: rebuild tree
            if pos < end_pos:
                entering_inv = pos >= bp_l and pos < bp_r and not in_inv
                leaving_inv = pos >= bp_r and in_inv
                if entering_inv and inv_len > 0:
                    _rebuild_structured(sim, root, n_std, n_inv, bp_l, inv_len,
                                         pos, rng)
                elif leaving_inv:
                    root = _rebuild_panmictic(root, rng)
            continue

        rate = (sim.rho / 2.0) * weighted_L
        dx = rng.exponential(1.0 / rate)
        extent = min(dx, next_boundary - pos, end_pos - pos)
        if extent <= 0:
            extent = 1e-10

        new_pos = pos + extent
        _drop_muts_segment(root, pos, new_pos, sim.theta, rng, mutations)

        if dx < (next_boundary - pos) and dx < (end_pos - pos):
            new_pos = pos + dx
            new_in_inv = bp_l <= new_pos < bp_r

            if new_in_inv and inv_len > 0:
                p_inv_t = sim.p_inv_func(0.5 * t_max)
                p_std_t = 1.0 - p_inv_t
                wL = L_S * p_std_t + L_I * p_inv_t
                if wL > 0:
                    u = rng.random() * wL
                    rc = 'S' if u < L_S * p_std_t else 'I'
                else:
                    rc = 'S'
                inv_pos = (new_pos - bp_l) / inv_len
                inv_pos = max(0.02, min(0.98, inv_pos))
                phi_x = sim.flux_model.phi(inv_pos)
                root = smc_prune_and_reattach(
                    root, rc, sim.p_inv, sim.c, sim.rho, phi_x, rng,
                    p_inv_func=sim.p_inv_func)
                root = find_root(root)
            else:
                root = smc_prune_and_reattach_panmictic(root, rng)
                root = find_root(root)
        else:
            # Boundary
            entering = new_pos >= bp_l and pos < bp_l and inv_len > 0
            leaving = new_pos >= bp_r and in_inv

            if entering:
                root = _rebuild_structured_from_leaves(
                    sim, root, n_std, n_inv, bp_l, inv_len, new_pos, rng)
            elif leaving:
                root = _rebuild_panmictic(root, rng)

        pos = new_pos

    return mutations, root


def _rebuild_panmictic(root, rng):
    """Rebuild panmictic tree reusing existing leaves."""
    from msinv import get_all_nodes, Node
    all_leaves = get_all_nodes(root)
    sample_leaves = sorted([n for n in all_leaves if n.is_leaf()],
                            key=lambda n: n.sample_id)
    active = list(sample_leaves)
    for n in active:
        n.parent = None
        n.children = []
    t = 0.0
    while len(active) > 1:
        k = len(active)
        t += rng.exponential(2.0 / (k * (k - 1)))
        idx = rng.choice(k, size=2, replace=False)
        coal = Node(time=t, branch_class='S')
        coal.children = [active[idx[0]], active[idx[1]]]
        active[idx[0]].parent = coal
        active[idx[1]].parent = coal
        for ii in sorted(idx, reverse=True):
            active.pop(ii)
        active.append(coal)
    return active[0]


def _rebuild_structured_from_leaves(sim, root, n_std, n_inv, bp_l, inv_len,
                                      pos, rng):
    """Rebuild structured tree at inversion entry, reusing leaves."""
    from msinv import get_all_nodes, _coalesce_pop, _flux_pop
    inv_pos = max(0.02, (pos - bp_l) / inv_len)
    phi_x = sim.flux_model.phi(inv_pos)

    all_leaves = get_all_nodes(root)
    sample_leaves = sorted([n for n in all_leaves if n.is_leaf()],
                            key=lambda n: n.sample_id)
    active = []
    for leaf in sample_leaves:
        leaf.branch_class = 'S' if leaf.sample_id < n_std else 'I'
        active.append([leaf, leaf.branch_class, leaf.population])
    for leaf in sample_leaves:
        leaf.parent = None
        leaf.children = []

    t = 0.0
    p_inv_func = sim.p_inv_func
    while len(active) > 1:
        p_inv_t = p_inv_func(t)
        if p_inv_t <= 0:
            for e in active:
                e[1] = 'S'
            while len(active) > 1:
                k = len(active)
                dt = rng.exponential(2.0 / (k * (k - 1)))
                t += dt
                _coalesce_pop(active, 'S', active[0][2], t, rng)
            break
        p_std_t = 1.0 - p_inv_t
        k_S = sum(1 for _, c, _ in active if c == 'S')
        k_I = sum(1 for _, c, _ in active if c == 'I')
        rc_S = k_S * (k_S - 1) / 2.0 / p_std_t if k_S >= 2 and p_std_t > 0 else 0
        rc_I = k_I * (k_I - 1) / 2.0 / p_inv_t if k_I >= 2 and p_inv_t > 0 else 0
        rf_SI = k_S * sim.c * (sim.rho / 2) * p_inv_t * phi_x if k_S > 0 else 0
        rf_IS = k_I * sim.c * (sim.rho / 2) * p_std_t * phi_x if k_I > 0 else 0
        total = rc_S + rc_I + rf_SI + rf_IS
        if total <= 0:
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv and t < t_inv:
                t = t_inv
                continue
            break
        dt = rng.exponential(1.0 / total)
        t_inv = getattr(p_inv_func, 't_inv', None)
        if t_inv and t + dt >= t_inv:
            t = t_inv
            continue
        t += dt
        u = rng.random() * total
        cum = rc_S
        if u < cum:
            _coalesce_pop(active, 'S', 0, t, rng)
            continue
        cum += rc_I
        if u < cum:
            _coalesce_pop(active, 'I', 0, t, rng)
            continue
        cum += rf_SI
        if u < cum:
            _flux_pop(active, 'S', 0, t, rng)
            continue
        _flux_pop(active, 'I', 0, t, rng)

    return active[0][0]
