"""
Standalone SMC walk segment for bidirectional inversion simulation.

Extracts the walk logic so it can be called multiple times with
different start positions and directions.
"""


def run_walk_segment(sim, root, start_pos, end_pos, rng, n_std, n_inv,
                      bp_l, bp_r, inv_len, flux_gamma=None):
    """
    Run a single left-to-right SMC walk from start_pos to end_pos.

    For leftward walks, the caller should mirror the chromosome:
    swap bp_l/bp_r and un-mirror mutation positions afterward.

    flux_gamma: absolute gene-flux rate for SMC reattach inside the
    inversion. If None, falls back to sim.gamma or sim.c*sim.rho/2.
    Must match the gamma used to build the tree (otherwise the SMC
    walk injects flux inconsistent with the initial tree).

    Returns (mutations, root) where mutations is list of (pos, leaf_ids).
    """
    import numpy as np
    from .simulator import (get_all_nodes, get_branches, find_root,
                            _drop_muts_segment, _coalesce_pop, _flux_pop,
                            smc_prune_and_reattach, smc_prune_and_reattach_panmictic,
                            build_structured_tree, ConstantFrequency, Node)

    def _full_rebuild_inside_inv(_phi_x, _flux_gamma, _rng):
        """Full structured-coalescent rebuild via build_structured_tree.

        Replaces the in-inv SMC prune-reattach. The single-tree SMC
        prune-reattach cannot reliably preserve the cross-karyotype
        T_MRCA constraint (T_MRCA_KFi >= t_inv) under repeated events:
        pruning a coalescent node above t_inv and reattaching the
        floating S subtree to a residual S branch can silently fire a
        coalescence below t_inv, eroding the karyotype barrier.

        Trade-off: full rebuild loses inversion-internal LD between
        adjacent positions (each in-inv site has an independent tree
        from the structured coalescent). Single-site marginals — which
        dxy, Da, FST, and PCA all depend on — remain correct.

        Uses the simulator's stored sample_config + a fresh demography
        copy so per-(class, pop) rates, demography, and ej events all
        match the initial-tree build at this position.
        """
        _gamma = _flux_gamma if _flux_gamma is not None else getattr(sim, 'gamma', None)
        if _gamma is None:
            _gamma = sim.c * sim.rho / 2.0
        # demo events get consumed by apply_events_at — copy each call.
        _demo = sim.demography.copy() if getattr(sim, 'demography', None) is not None else None
        new_root, _ = build_structured_tree(
            n_std, n_inv, sim.p_inv, _gamma, 2.0, _phi_x, _rng,
            p_inv_func=sim.p_inv_func,
            sample_config=sim.sample_config,
            n_pops=sim.n_pops, mig_rate=sim.mig_rate,
            demo_events=getattr(sim, 'demo_events', None),
            demography=_demo)
        return new_root

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
            # Inside the inversion, per-position recombination is GENE
            # CONVERSION ONLY (Option 3 model). Heterokaryotypes (S/I)
            # do not undergo crossing-over within the inversion; their
            # only sequence exchange is gene conversion at rate
            # gamma * phi(x). Within homokaryotypes (S/S, I/I) standard
            # crossing-over within the karyotype's sub-tree is captured
            # in the sub-population's coalescent marginal, NOT as
            # additional events here (those events are within-class and
            # don't change the SMC tree's class structure).
            #
            # Result: the tree is mostly correlated across the inversion,
            # broken only where flux fires. phi(x) controls where: phi
            # peaks in the middle and drops to 0 at the breakpoints
            # (Peischl 2013), so gene flux is concentrated in the centre
            # and the breakpoints retain the strongest cross-karyotype
            # divergence — matching empirical inversion LD patterns.
            inv_pos_now = (pos - bp_l) / inv_len
            inv_pos_now = max(0.02, min(0.98, inv_pos_now))
            phi_now = sim.flux_model.phi(inv_pos_now)
            # Use the caller-supplied flux_gamma (or sim.gamma).
            _gamma_now = flux_gamma if flux_gamma is not None else getattr(
                sim, 'gamma', None)
            if _gamma_now is None:
                _gamma_now = sim.c * sim.rho / 2.0
            # Per-branch flux weight: probability that this lineage's
            # current chromosome is in a heterokaryote (= p_other for
            # this branch's class in this branch's pop).
            weighted_L_flux = 0.0
            stack2 = [root]
            while stack2:
                n = stack2.pop()
                if n.parent is not None:
                    bl = n.parent.time - n.time
                    pop_i = getattr(n, 'population', 0)
                    p_inv_t_i = sim.p_inv_func(0.5 * t_max, pop_i)
                    if p_inv_t_i > 0:
                        # p_other: prob the homologous chromosome is the
                        # other karyotype, which is when gene conversion
                        # can fire.
                        p_other = (p_inv_t_i if n.branch_class == 'S'
                                   else (1.0 - p_inv_t_i))
                    else:
                        p_other = 0.0
                    weighted_L_flux += bl * p_other
                for ch in n.children:
                    stack2.append(ch)
            # Effective rate per unit position summed across lineages.
            weighted_L = _gamma_now * phi_now * weighted_L_flux
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

        # Outside inv: standard panmictic SMC recombination = rho/2 per
        # unit position summed over branches.
        # Inside inv: weighted_L already includes gamma * phi(x) (Option 3);
        # it's the per-position flux rate, not a recombination rate.
        if in_inv and inv_len > 0:
            rate = weighted_L
        else:
            rate = (sim.rho / 2.0) * weighted_L
        if rate <= 0:
            # No events possible (e.g. gamma=0 inside inv): just walk to
            # the next boundary, drop mutations, and continue. The tree
            # stays unchanged across the entire inv → strong LD inside,
            # which is the biological reality at gamma=0.
            _drop_muts_segment(root, pos, next_boundary,
                                sim.theta, rng, mutations)
            pos = next_boundary
            if pos < end_pos:
                entering_inv = pos >= bp_l and pos < bp_r and not in_inv
                leaving_inv = pos >= bp_r and in_inv
                if entering_inv and inv_len > 0:
                    _rebuild_structured(sim, root, n_std, n_inv, bp_l, inv_len,
                                         pos, rng)
                elif leaving_inv:
                    root = _rebuild_panmictic(root, rng)
            continue
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
                # Inside-inv event = gene-flux (Option 3 model).
                # Pick a branch weighted by p_other (the probability that
                # this lineage's homologous chromosome is the OTHER
                # karyotype, which is when gene conversion can fire).
                # The chosen lineage gets converted: at this position its
                # ancestor is now in the OTHER karyotype's sub-tree.
                # Going forward (in the SMC walk), the lineage's class
                # is flipped — and it will coalesce with branches in the
                # other-class sub-tree until/unless a future flux flips
                # it back. This preserves the tree everywhere except the
                # converted lineage's ancestry, giving the empirical
                # gradient of LD breakdown from breakpoints inward
                # (phi(x) peaks at centre, vanishes at breakpoints).
                from .simulator import _reattach
                wL_branches = []
                stack3 = [root]
                while stack3:
                    n = stack3.pop()
                    if n.parent is not None:
                        bl = n.parent.time - n.time
                        pop_i = getattr(n, 'population', 0)
                        p_inv_t_i = sim.p_inv_func(0.5 * t_max, pop_i)
                        if p_inv_t_i > 0:
                            p_other_b = (p_inv_t_i if n.branch_class == 'S'
                                         else (1.0 - p_inv_t_i))
                        else:
                            p_other_b = 0.0
                        if p_other_b > 0:
                            wL_branches.append((n, bl * p_other_b))
                    for ch in n.children:
                        stack3.append(ch)
                if wL_branches:
                    weights = np.array([w for _, w in wL_branches])
                    probs = weights / weights.sum()
                    bi = rng.choice(len(wL_branches), p=probs)
                    target = wL_branches[bi][0]
                    target_bl = target.parent.time - target.time
                    t_cut = target.time + rng.random() * target_bl
                    inv_pos = (new_pos - bp_l) / inv_len
                    inv_pos = max(0.02, min(0.98, inv_pos))
                    phi_x = sim.flux_model.phi(inv_pos)
                    _gamma = flux_gamma if flux_gamma is not None else getattr(
                        sim, 'gamma', None)
                    if _gamma is None:
                        _gamma = sim.c * sim.rho / 2.0

                    # ---- Manual prune of `target` ----
                    p = target.parent
                    if p is not None and len(p.children) == 2:
                        sibling = [ch for ch in p.children if ch is not target][0]
                        gp = p.parent
                        sibling.parent = gp
                        if gp is not None:
                            gp.children = [sibling if ch is p else ch
                                           for ch in gp.children]
                        # If parent was the root, sibling becomes new root
                        new_root = sibling if root is p else root
                        target.parent = None
                        # ---- Flip class (gene conversion) ----
                        new_cls = 'I' if target.branch_class == 'S' else 'S'
                        target.branch_class = new_cls
                        target_pop = getattr(target, 'population', 0)
                        # ---- Reattach floating into other-class subtree ----
                        root = _reattach(
                            new_root, target, new_cls, t_cut,
                            sim.p_inv, _gamma, 2.0, phi_x, rng,
                            p_inv_func=sim.p_inv_func, pop=target_pop)
                root = find_root(root)
            else:
                root = smc_prune_and_reattach_panmictic(
                    root, rng, demography=getattr(sim, 'demography', None))
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
    from .simulator import get_all_nodes, Node
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
    from .simulator import get_all_nodes, _coalesce_pop, _flux_pop
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
    _n_pops_traj = getattr(p_inv_func, 'n_pops', 1)
    while len(active) > 1:
        p_inv_global = max(
            p_inv_func(t, pp) for pp in range(_n_pops_traj))
        if p_inv_global <= 0:
            for e in active:
                e[1] = 'S'
            while len(active) > 1:
                k = len(active)
                dt = rng.exponential(2.0 / (k * (k - 1)))
                t += dt
                _coalesce_pop(active, 'S', active[0][2], t, rng)
            break
        # Build per-pop rate table
        rates_ws = []
        counts_ws = {}
        for _, cls_w, pop_w in active:
            key = (cls_w, pop_w)
            counts_ws[key] = counts_ws.get(key, 0) + 1
        for (cls_w, pop_w), k_w in counts_ws.items():
            p_inv_t_w = max(p_inv_func(t, pop_w), 0)
            p_std_t_w = 1.0 - p_inv_t_w
            if p_inv_t_w > 0:
                f_w = p_std_t_w if cls_w == 'S' else p_inv_t_w
            else:
                f_w = 1.0
            if k_w >= 2 and f_w > 0:
                rates_ws.append(('coal', cls_w, pop_w, k_w*(k_w-1)/2.0/f_w))
            if k_w > 0 and p_inv_t_w > 0:
                f_other_w = p_inv_t_w if cls_w == 'S' else p_std_t_w
                _g = getattr(sim, 'gamma', None)
                if _g is None:
                    _g = sim.c * sim.rho / 2.0
                rf_w = k_w * _g * f_other_w * phi_x
                if rf_w > 0:
                    rates_ws.append(('flux', cls_w, pop_w, rf_w))
        total = sum(r for _, _, _, r in rates_ws)
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
        cum = 0.0
        for etype_w, cls_w, pop_w, r_w in rates_ws:
            cum += r_w
            if u < cum:
                if etype_w == 'coal':
                    _coalesce_pop(active, cls_w, pop_w, t, rng)
                else:
                    _flux_pop(active, cls_w, pop_w, t, rng)
                break

    return active[0][0]
