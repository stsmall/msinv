#!/usr/bin/env python3
"""
msinv_fast: Array-based + Numba-accelerated coalescent with inversions.

Drop-in replacement for msinv.py with same API but ~10-20x faster.
Uses ArrayTree from tree_core.py for all tree operations.
"""

import numpy as np
import sys
from tree_core import (
    ArrayTree, NULL, CLASS_S, CLASS_I,
    branch_lengths_by_class, get_branches_arr,
    find_branches_above_time, get_leaves_below,
)
try:
    import smc_bridge
    _HAS_C = smc_bridge.is_available()
except ImportError:
    _HAS_C = False

# Import trajectory classes from msinv (unchanged)
from msinv import (
    ConstantFrequency, DeterministicTrajectory, StochasticTrajectory,
    GeneFluxModel, count_lineages_by_class_pop,
)


# ===================================================================
# Structured coalescent (array-based)
# ===================================================================

def build_structured_tree_fast(n_std, n_inv, p_inv, c, rho, phi_x, rng,
                                p_inv_func=None, sample_config=None,
                                n_pops=1, mig_rate=0.0, demo_events=None):
    """Build structured coalescent tree using ArrayTree."""
    if p_inv_func is None:
        p_inv_func = ConstantFrequency(p_inv)

    tree = ArrayTree()

    # Create leaves
    sid = 0
    if sample_config is not None:
        for (cls, pop), count in sorted(sample_config.items()):
            kl = CLASS_S if cls == 'S' else CLASS_I
            for _ in range(count):
                tree.add_node(0.0, kl=kl, pop=pop, sid=sid)
                sid += 1
    else:
        for i in range(n_std):
            tree.add_node(0.0, kl=CLASS_S, pop=0, sid=sid); sid += 1
        for i in range(n_inv):
            tree.add_node(0.0, kl=CLASS_I, pop=0, sid=sid); sid += 1

    nsam = sid
    # active: list of (node_idx, class, pop)
    active = [(i, int(tree.klass[i]), int(tree.population[i]))
              for i in range(nsam)]
    t = 0.0

    if demo_events is None:
        demo_events = []
    demo_events = sorted(demo_events, key=lambda x: x[0])
    demo_idx = 0

    while len(active) > 1:
        while demo_idx < len(demo_events) and demo_events[demo_idx][0] <= t:
            _, etype, eargs = demo_events[demo_idx]
            if etype == 'merge':
                src, dst = eargs
                new_active = []
                for (idx, kl, pop) in active:
                    if pop == src:
                        tree.population[idx] = dst
                        new_active.append((idx, kl, dst))
                    else:
                        new_active.append((idx, kl, pop))
                active = new_active
            demo_idx += 1

        p_inv_t = p_inv_func(t)

        if p_inv_t <= 0:
            # Panmictic
            for i in range(len(active)):
                active[i] = (active[i][0], CLASS_S, active[i][2])
                tree.klass[active[i][0]] = CLASS_S
            pops_present = set(e[2] for e in active)
            if len(pops_present) == 1:
                while len(active) > 1:
                    k = len(active)
                    dt = rng.exponential(2.0 / (k * (k - 1)))
                    t += dt
                    picked = rng.choice(k, size=2, replace=False)
                    i1, i2 = int(picked[0]), int(picked[1])
                    n1, n2 = active[i1][0], active[i2][0]
                    coal = tree.add_node(t, kl=CLASS_S, pop=active[i1][2])
                    tree.add_child_to(coal, n1)
                    tree.add_child_to(coal, n2)
                    pop = active[i1][2]
                    for i in sorted([i1, i2], reverse=True):
                        active.pop(i)
                    active.append((coal, CLASS_S, pop))
                break

        p_std_t = 1.0 - max(p_inv_t, 0)
        p_inv_t = max(p_inv_t, 0)

        # Build rate table
        counts = {}
        for _, kl, pop in active:
            key = (kl, pop)
            counts[key] = counts.get(key, 0) + 1

        rates = []
        for (kl, pop), k in counts.items():
            f = p_std_t if kl == CLASS_S else p_inv_t
            if p_inv_t <= 0:
                f = 1.0
            if k >= 2 and f > 0:
                rates.append(('coal', kl, pop, k * (k - 1) / 2.0 / f))
            if k > 0 and phi_x > 0 and p_inv_t > 0:
                f_other = p_inv_t if kl == CLASS_S else p_std_t
                rf = k * c * (rho / 2.0) * f_other * phi_x
                if rf > 0:
                    rates.append(('flux', kl, pop, rf))
            if k > 0 and mig_rate > 0:
                for other_pop in range(n_pops):
                    if other_pop != pop:
                        rates.append(('mig', kl, pop, k * mig_rate / 2.0))

        total = sum(r for _, _, _, r in rates)
        if total <= 0:
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv is not None and t < t_inv:
                t = t_inv
                continue
            if demo_idx < len(demo_events):
                t = demo_events[demo_idx][0]
                continue
            break

        dt = rng.exponential(1.0 / total)
        t_inv = getattr(p_inv_func, 't_inv', None)
        if t_inv is not None and t + dt >= t_inv:
            t = t_inv
            continue
        if demo_idx < len(demo_events) and t + dt >= demo_events[demo_idx][0]:
            t = demo_events[demo_idx][0]
            continue

        t += dt

        u = rng.random() * total
        cum = 0.0
        event = None
        for etype, kl, pop, r in rates:
            cum += r
            if u < cum:
                event = (etype, kl, pop)
                break
        if event is None:
            event = (rates[-1][0], rates[-1][1], rates[-1][2])

        etype, kl, pop = event

        if etype == 'coal':
            indices = [i for i, (_, k, p) in enumerate(active)
                       if k == kl and p == pop]
            picked = rng.choice(len(indices), size=2, replace=False)
            i1, i2 = indices[int(picked[0])], indices[int(picked[1])]
            n1, n2 = active[i1][0], active[i2][0]
            coal = tree.add_node(t, kl=kl, pop=pop)
            tree.add_child_to(coal, n1)
            tree.add_child_to(coal, n2)
            for i in sorted([i1, i2], reverse=True):
                active.pop(i)
            active.append((coal, kl, pop))

        elif etype == 'flux':
            indices = [i for i, (_, k, p) in enumerate(active)
                       if k == kl and p == pop]
            idx = indices[int(rng.integers(len(indices)))]
            old_node = active[idx][0]
            new_kl = CLASS_I if kl == CLASS_S else CLASS_S
            fn = tree.add_node(t, kl=new_kl, pop=pop)
            tree.add_child_to(fn, old_node)
            active[idx] = (fn, new_kl, pop)

        elif etype == 'mig':
            indices = [i for i, (_, k, p) in enumerate(active)
                       if k == kl and p == pop]
            idx = indices[int(rng.integers(len(indices)))]
            node = active[idx][0]
            other_pops = [p for p in range(n_pops) if p != pop]
            to_pop = other_pops[int(rng.integers(len(other_pops)))]
            tree.population[node] = to_pop
            active[idx] = (node, kl, to_pop)

    tree.find_root_node()
    return tree, nsam


# ===================================================================
# SMC prune-and-reattach (array-based)
# ===================================================================

def smc_prune_reattach_panmictic_fast(tree, rng):
    """Panmictic prune-and-reattach. Uses C core if available."""
    if _HAS_C:
        tree._ensure_space(needed=5)
        smc_bridge.panmictic_pr(tree)
        return
    indices, lengths, count = tree.get_branches(filter_class=-1)
    if count == 0:
        return

    probs = lengths / lengths.sum()
    bi = rng.choice(count, p=probs)
    target = int(indices[bi])
    t_cut = tree.time[target] + rng.random() * lengths[bi]

    # Prune
    p = tree.parent[target]
    if p == NULL:
        return
    nc = tree.num_children(p)
    if nc != 2:
        return

    sib = tree.get_sibling(target)
    if sib == NULL:
        return

    gp = tree.parent[p]
    tree.remove_child_from(p, target)
    tree.remove_child_from(p, sib)
    if gp != NULL:
        tree.remove_child_from(gp, p)
        tree.add_child_to(gp, sib)
    else:
        tree.parent[sib] = NULL

    if tree.root == p:
        tree.root = sib

    tree.parent[target] = NULL

    # Reattach
    above_idx, above_len, above_count = find_branches_above_time(
        tree.parent, tree.time, tree.n, t_cut, -1, tree.klass)

    if above_count > 0:
        aprobs = above_len / above_len.sum()
        ai = rng.choice(above_count, p=aprobs)
        attach = int(above_idx[ai])
        attach_parent = tree.parent[attach]
        t_a = max(tree.time[attach], t_cut) + rng.random() * (
            tree.time[attach_parent] - max(tree.time[attach], t_cut))

        coal = tree.add_node(t_a, kl=tree.klass[attach], pop=tree.population[attach])
        tree.remove_child_from(attach_parent, attach)
        tree.add_child_to(attach_parent, coal)
        tree.add_child_to(coal, attach)
        tree.add_child_to(coal, target)
    else:
        root = tree.root
        t_c = max(t_cut, tree.time[root]) + rng.exponential(1.0)
        coal = tree.add_node(t_c, kl=CLASS_S, pop=0)
        tree.add_child_to(coal, root)
        tree.add_child_to(coal, target)
        tree.root = coal

    tree.find_root_node()


def smc_prune_reattach_structured_fast(tree, recomb_class, p_inv, c, rho,
                                        phi_x, rng, p_inv_func=None):
    """Structured prune-and-reattach on ArrayTree."""
    if p_inv_func is None:
        p_inv_func = ConstantFrequency(p_inv)

    kl_filter = CLASS_S if recomb_class == 'S' else CLASS_I
    indices, lengths, count = tree.get_branches(filter_class=kl_filter)
    if count == 0:
        return

    probs = lengths / lengths.sum()
    bi = rng.choice(count, p=probs)
    target = int(indices[bi])
    t_cut = tree.time[target] + rng.random() * lengths[bi]

    # Prune (same as panmictic)
    p = tree.parent[target]
    if p == NULL:
        return
    nc = tree.num_children(p)
    if nc != 2:
        return

    sib = tree.get_sibling(target)
    if sib == NULL:
        return

    gp = tree.parent[p]
    tree.remove_child_from(p, target)
    tree.remove_child_from(p, sib)
    if gp != NULL:
        tree.remove_child_from(gp, p)
        tree.add_child_to(gp, sib)
    else:
        tree.parent[sib] = NULL

    if tree.root == p:
        tree.root = sib

    tree.parent[target] = NULL

    # Structured reattach
    fclass = kl_filter
    t = t_cut

    for _ in range(50000):
        p_inv_t = p_inv_func(t)

        if p_inv_t <= 0:
            # Panmictic: attach anywhere above t
            above_idx, above_len, above_count = find_branches_above_time(
                tree.parent, tree.time, tree.n, t, -1, tree.klass)
            if above_count > 0:
                aprobs = above_len / above_len.sum()
                ai = rng.choice(above_count, p=aprobs)
                attach = int(above_idx[ai])
                ap = tree.parent[attach]
                t_a = max(tree.time[attach], t) + rng.random() * (
                    tree.time[ap] - max(tree.time[attach], t))
                coal = tree.add_node(t_a, kl=CLASS_S, pop=tree.population[attach])
                tree.remove_child_from(ap, attach)
                tree.add_child_to(ap, coal)
                tree.add_child_to(coal, attach)
                tree.add_child_to(coal, target)
                tree.klass[target] = CLASS_S
            else:
                root = tree.root
                dt = rng.exponential(1.0)
                coal = tree.add_node(t + dt, kl=CLASS_S, pop=0)
                tree.add_child_to(coal, root)
                tree.add_child_to(coal, target)
                tree.root = coal
            break

        p_std_t = 1.0 - p_inv_t

        # Find same-class branches above t
        same_idx, same_len, same_count = find_branches_above_time(
            tree.parent, tree.time, tree.n, t, fclass, tree.klass)

        p_same = p_std_t if fclass == CLASS_S else p_inv_t
        p_other = p_inv_t if fclass == CLASS_S else p_std_t
        rate_coal = same_count / p_same if same_count > 0 and p_same > 0 else 0.0
        rate_flux = c * (rho / 2.0) * p_other * phi_x if phi_x > 0 else 0.0

        total = rate_coal + rate_flux
        if total <= 0:
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv is not None:
                t = t_inv
                continue
            break

        dt = rng.exponential(1.0 / total)
        t_inv = getattr(p_inv_func, 't_inv', None)
        if t_inv is not None and t + dt >= t_inv:
            t = t_inv
            continue

        # Check if we pass above the tree root
        root_time = tree.time[tree.root]
        if t + dt > root_time and t < root_time:
            t = root_time
            # Above root: two-lineage structured coalescent
            rclass = tree.klass[tree.root]
            for _ in range(100000):
                p_inv_t2 = p_inv_func(t)
                if p_inv_t2 <= 0:
                    dt2 = rng.exponential(1.0)
                    coal = tree.add_node(t + dt2, kl=CLASS_S, pop=0)
                    tree.add_child_to(coal, tree.root)
                    tree.add_child_to(coal, target)
                    tree.root = coal
                    break
                p_std_t2 = 1.0 - p_inv_t2
                rc2 = (1.0 / (p_std_t2 if fclass == CLASS_S else p_inv_t2)
                       if fclass == rclass else 0.0)
                p_oth_f = p_inv_t2 if fclass == CLASS_S else p_std_t2
                p_oth_r = p_inv_t2 if rclass == CLASS_S else p_std_t2
                rf_f = c * (rho / 2.0) * p_oth_f * phi_x
                rf_r = c * (rho / 2.0) * p_oth_r * phi_x
                tot2 = rc2 + rf_f + rf_r
                if tot2 <= 0:
                    t_inv2 = getattr(p_inv_func, 't_inv', None)
                    if t_inv2 is not None:
                        t = t_inv2; continue
                    break
                dt2 = rng.exponential(1.0 / tot2)
                t_inv2 = getattr(p_inv_func, 't_inv', None)
                if t_inv2 is not None and t + dt2 >= t_inv2:
                    t = t_inv2; continue
                t += dt2
                u2 = rng.random() * tot2
                if u2 < rc2:
                    coal = tree.add_node(t, kl=fclass, pop=0)
                    tree.add_child_to(coal, tree.root)
                    tree.add_child_to(coal, target)
                    tree.root = coal
                    break
                elif u2 < rc2 + rf_f:
                    fn = tree.add_node(t, kl=CLASS_I if fclass == CLASS_S else CLASS_S, pop=0)
                    tree.add_child_to(fn, target)
                    target = fn
                    fclass = 1 - fclass
                else:
                    new_kl = CLASS_I if rclass == CLASS_S else CLASS_S
                    fn = tree.add_node(t, kl=new_kl, pop=0)
                    old_root = tree.root
                    tree.add_child_to(fn, old_root)
                    tree.root = fn
                    rclass = new_kl
            break

        t += dt

        if rng.random() * total < rate_coal and same_count > 0:
            # Coalesce with a same-class branch
            aprobs = same_len / same_len.sum()
            ai = rng.choice(same_count, p=aprobs)
            attach = int(same_idx[ai])
            ap = tree.parent[attach]
            t_a = t
            coal = tree.add_node(t_a, kl=fclass, pop=tree.population[attach])
            tree.remove_child_from(ap, attach)
            tree.add_child_to(ap, coal)
            tree.add_child_to(coal, attach)
            tree.add_child_to(coal, target)
            break
        else:
            # Gene flux: switch class
            new_kl = CLASS_I if fclass == CLASS_S else CLASS_S
            fn = tree.add_node(t, kl=new_kl, pop=0)
            tree.add_child_to(fn, target)
            target = fn
            fclass = new_kl

    tree.find_root_node()


# ===================================================================
# Mutation model (array-based)
# ===================================================================

def drop_mutations_fast(tree_intervals, theta, nsam, rng):
    """
    Infinite sites mutations on marginal trees (ArrayTree-based).
    tree_intervals: list of (ArrayTree, left, right)
    """
    mutations = []

    for tree, left, right in tree_intervals:
        seg_len = right - left
        if seg_len <= 0:
            continue

        indices, lengths, count = tree.get_branches(filter_class=-1)
        if count == 0:
            continue
        L_total = float(lengths.sum())
        if L_total <= 0:
            continue

        n_muts = rng.poisson(min((theta / 2.0) * L_total * seg_len, 1e6))
        if n_muts == 0:
            continue
        n_muts = min(n_muts, 100000)

        probs = lengths / L_total

        for _ in range(n_muts):
            pos = rng.uniform(left, right)
            bi = rng.choice(count, p=probs)
            node = int(indices[bi])
            leaf_ids = tree.get_leaves(node)
            mutations.append((pos, leaf_ids))

    if not mutations:
        return [], np.zeros((nsam, 0), dtype=int)

    mutations.sort(key=lambda x: x[0])
    positions = [m[0] for m in mutations]
    haplotypes = np.zeros((nsam, len(mutations)), dtype=np.int8)
    for j, (_, ids) in enumerate(mutations):
        for sid in ids:
            haplotypes[sid, j] = 1

    return positions, haplotypes


# ===================================================================
# Fast Simulator (drop-in for MsinvSimulator)
# ===================================================================

class MsinvSimulatorFast:
    """Drop-in replacement for MsinvSimulator using ArrayTree."""

    def __init__(self, nsam, nreps, theta, rho, nsites,
                 n_std=None, n_inv=None,
                 p_inv=0.0, c=0.0,
                 flux_window=0.3, seed=None,
                 p_inv_func=None, t_inv=None,
                 bp_left=0.3, bp_right=0.7,
                 n_pops=1, mig_rate=0.0,
                 sample_config=None, demo_events=None):
        self.nsam = nsam
        self.nreps = nreps
        self.theta = theta
        self.rho = rho
        self.nsites = nsites
        self.p_inv = p_inv
        self.c = c
        self.bp_left = bp_left
        self.bp_right = bp_right
        self.flux_model = GeneFluxModel(w=flux_window)
        self.n_pops = n_pops
        self.mig_rate = mig_rate
        self.sample_config = sample_config
        self.demo_events = demo_events or []

        self.rng = np.random.default_rng(seed)

        if p_inv_func is not None:
            self.p_inv_func = p_inv_func
        elif t_inv is not None:
            self.p_inv_func = ConstantFrequency(p_inv, t_inv=t_inv)
        else:
            self.p_inv_func = ConstantFrequency(p_inv)

        if n_std is not None and n_inv is not None:
            self.n_std = n_std
            self.n_inv = n_inv
        else:
            self.n_std = None
            self.n_inv = None

    def _get_sample(self):
        if self.n_std is not None:
            return self.n_std, self.n_inv
        ni = int(self.rng.binomial(self.nsam, self.p_inv))
        return self.nsam - ni, ni

    def _has_inversion(self):
        return self.c > 0 and 0 < self.p_inv < 1

    def simulate_one(self):
        rng = self.rng

        if not self._has_inversion():
            return self._sim_standard()

        n_std, n_inv = self._get_sample()
        if n_std == 0 or n_inv == 0:
            return self._sim_standard()  # single class = standard

        bp_l = self.bp_left
        bp_r = self.bp_right
        inv_len = bp_r - bp_l

        # Build initial panmictic tree at position 0
        tree = ArrayTree()
        for i in range(n_std):
            tree.add_node(0.0, kl=CLASS_S, pop=0, sid=i)
        for i in range(n_inv):
            tree.add_node(0.0, kl=CLASS_I, pop=0, sid=n_std + i)

        active = list(range(tree.n))
        t = 0.0
        while len(active) > 1:
            k = len(active)
            t += rng.exponential(2.0 / (k * (k - 1)))
            idx = rng.choice(k, size=2, replace=False)
            i1, i2 = int(idx[0]), int(idx[1])
            n1, n2 = active[i1], active[i2]
            coal = tree.add_node(t, kl=CLASS_S, pop=0)
            tree.add_child_to(coal, n1)
            tree.add_child_to(coal, n2)
            for ii in sorted([i1, i2], reverse=True):
                active.pop(ii)
            active.append(coal)
        tree.root = active[0]

        # Seed C RNG if available
        if _HAS_C:
            smc_bridge.seed(int(rng.integers(2**63)))

        trees = []
        pos = 0.0

        for _ in range(500000):
            if pos >= 1.0:
                break

            in_inv = bp_l <= pos < bp_r

            # Branch lengths via C or Numba
            if _HAS_C:
                L_S, L_I, t_max = smc_bridge.branch_lengths(
                    tree.time, tree.parent, tree.klass, tree.n)
            else:
                L_S, L_I, t_max = tree.get_branch_lengths()
            L_total = L_S + L_I

            if in_inv:
                p_inv_t = self.p_inv_func(0.5 * t_max)
                p_std_t = 1.0 - p_inv_t
                weighted_L = (L_S * p_std_t + L_I * p_inv_t
                              if p_inv_t > 0 else L_total)
                next_boundary = bp_r
            else:
                weighted_L = L_total
                next_boundary = bp_l if pos < bp_l else 1.0

            if weighted_L <= 0:
                trees.append((tree, pos, next_boundary))
                pos = next_boundary
                continue

            rate = (self.rho / 2.0) * weighted_L
            dx = rng.exponential(1.0 / rate)
            extent = min(dx, next_boundary - pos, 1.0 - pos)
            if extent <= 0:
                extent = 1e-10

            new_pos = pos + extent
            trees.append((tree, pos, new_pos))

            if dx < (next_boundary - pos) and dx < (1.0 - pos):
                new_pos = pos + dx
                new_in_inv = bp_l <= new_pos < bp_r

                if new_in_inv:
                    p_inv_t = self.p_inv_func(0.5 * t_max)
                    p_std_t = 1.0 - p_inv_t
                    wL = L_S * p_std_t + L_I * p_inv_t
                    if wL > 0:
                        u = rng.random() * wL
                        rc = 'S' if u < L_S * p_std_t else 'I'
                    else:
                        rc = 'S'
                    inv_pos = (new_pos - bp_l) / inv_len
                    inv_pos = max(0.02, min(0.98, inv_pos))
                    phi_x = self.flux_model.phi(inv_pos)
                    smc_prune_reattach_structured_fast(
                        tree, rc, self.p_inv, self.c, self.rho,
                        phi_x, rng, p_inv_func=self.p_inv_func)
                else:
                    smc_prune_reattach_panmictic_fast(tree, rng)
            else:
                # Boundary: rebuild tree
                if new_pos >= bp_l and pos < bp_l:
                    inv_pos = max(0.02, (new_pos - bp_l) / inv_len)
                    phi_x = self.flux_model.phi(inv_pos)
                    tree, nsam = build_structured_tree_fast(
                        n_std, n_inv, self.p_inv, self.c, self.rho,
                        phi_x, rng, p_inv_func=self.p_inv_func,
                        sample_config=self.sample_config,
                        n_pops=self.n_pops, mig_rate=self.mig_rate,
                        demo_events=self.demo_events)
                elif new_pos >= bp_r and in_inv:
                    # Panmictic rebuild
                    new_tree = ArrayTree()
                    leaves = []
                    for i in range(tree.n):
                        if tree.sample_id[i] >= 0:
                            sid = int(tree.sample_id[i])
                            idx = new_tree.add_node(0.0, kl=tree.klass[i],
                                                     pop=tree.population[i], sid=sid)
                            leaves.append(idx)
                    active = list(leaves)
                    t2 = 0.0
                    while len(active) > 1:
                        k = len(active)
                        t2 += rng.exponential(2.0 / (k * (k - 1)))
                        idx = rng.choice(k, size=2, replace=False)
                        i1, i2 = int(idx[0]), int(idx[1])
                        n1, n2 = active[i1], active[i2]
                        coal = new_tree.add_node(t2, kl=CLASS_S, pop=0)
                        new_tree.add_child_to(coal, n1)
                        new_tree.add_child_to(coal, n2)
                        for ii in sorted([i1, i2], reverse=True):
                            active.pop(ii)
                        active.append(coal)
                    new_tree.root = active[0]
                    tree = new_tree

            pos = new_pos

        return drop_mutations_fast(trees, self.theta, self.nsam, rng)

    def _sim_standard(self):
        rng = self.rng
        n = self.nsam
        tree = ArrayTree()
        for i in range(n):
            tree.add_node(0.0, kl=CLASS_S, pop=0, sid=i)
        active = list(range(n))
        t = 0.0
        while len(active) > 1:
            k = len(active)
            t += rng.exponential(2.0 / (k * (k - 1)))
            idx = rng.choice(k, size=2, replace=False)
            i1, i2 = int(idx[0]), int(idx[1])
            n1, n2 = active[i1], active[i2]
            coal = tree.add_node(t, kl=CLASS_S, pop=0)
            tree.add_child_to(coal, n1)
            tree.add_child_to(coal, n2)
            for ii in sorted([i1, i2], reverse=True):
                active.pop(ii)
            active.append(coal)
        tree.root = active[0]

        trees_list = [(tree, 0.0, 1.0)]
        pos = 0.0
        for _ in range(500000):
            L = tree.get_total_bl()
            if L <= 0:
                break
            rate = (self.rho / 2.0) * L
            if rate <= 0:
                break
            dx = rng.exponential(1.0 / rate)
            new_pos = pos + dx
            if new_pos >= 1.0:
                break
            trees_list.append((tree, pos, new_pos))
            pos = new_pos
            smc_prune_reattach_panmictic_fast(tree, rng)

        return drop_mutations_fast(trees_list, self.theta, self.nsam, rng)

    def run(self, outfile=sys.stdout):
        cmd_parts = [f"msinv_fast {self.nsam} {self.nreps}"]
        print(" ".join(cmd_parts), file=outfile)
        seeds = self.rng.integers(0, 2**31, size=3)
        print(f"{seeds[0]} {seeds[1]} {seeds[2]}", file=outfile)
        print(file=outfile)

        for _ in range(self.nreps):
            positions, haplotypes = self.simulate_one()
            nseg = len(positions)
            print("//", file=outfile)
            print(f"segsites: {nseg}", file=outfile)
            if nseg > 0:
                print("positions: " +
                      " ".join(f"{p:.4f}" for p in positions), file=outfile)
                for i in range(self.nsam):
                    print("".join(str(x) for x in haplotypes[i]), file=outfile)
            print(file=outfile)
