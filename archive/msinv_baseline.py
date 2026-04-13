#!/usr/bin/env python3
"""
msinv: Coalescent simulator with chromosomal inversions.

Extends Hudson's ms framework with inversion polymorphism, using the
Sequential Markovian Coalescent (SMC) for recombination along the
chromosome and a structured coalescent for the two karyotype classes
(Standard and Inverted).

References:
  Hudson RR (2002) Bioinformatics 18:337-338.
  Guerrero RF, Rousset F, Kirkpatrick M (2012) Phil Trans R Soc B 367:430-438.
  Peischl S, Koch E, Guerrero RF, Kirkpatrick M (2013) Heredity 111:200-209.

Time is in units of 2N generations (continuous-time coalescent).
Population-scaled parameters:
  theta = 4*N*mu  (per-base mutation rate, scaled by nsites for total)
  rho   = 4*N*r   (recombination rate across the simulated region)

Usage:
  msinv <nsam> <nreps> -t <theta> -r <rho> <nsites> [options]

Options:
  -t <theta>                  Population-scaled mutation rate (4Nmu * L)
  -r <rho> <nsites>           Recombination rate 4Nr and number of sites
  -inv <p_inv> <c>            Inversion: frequency p_inv, gene flux
                              coefficient c (in [0,1])
  -I <n_std> <n_inv>          Sample composition (must sum to nsam)
  -flux_window <w>            Gene flux interval width (default: 0.3)
  -seed <s>                   Random seed
"""

import numpy as np
import sys


# ===================================================================
# Data structures
# ===================================================================

class Node:
    """
    Coalescent tree node.

    branch_class: karyotype class ('S' or 'I') of the branch from
    this node UP to its parent.
    """
    __slots__ = ['time', 'children', 'parent', 'sample_id', 'branch_class']

    def __init__(self, time=0.0, sample_id=None, branch_class='S'):
        self.time = time
        self.children = []
        self.parent = None
        self.sample_id = sample_id
        self.branch_class = branch_class

    def branch_length(self):
        if self.parent is None:
            return 0.0
        return self.parent.time - self.time

    def is_leaf(self):
        return self.sample_id is not None

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf({self.sample_id},{self.branch_class},t={self.time:.3f})"
        nc = len(self.children)
        return f"Node({self.branch_class},t={self.time:.3f},nc={nc})"


def get_all_nodes(root):
    """Iterative DFS to collect all nodes."""
    nodes = []
    stack = [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children)
    return nodes


def branch_lengths_by_class(root):
    """Return (L_S, L_I) total branch lengths."""
    L_S = L_I = 0.0
    for n in get_all_nodes(root):
        if n.parent is not None:
            bl = n.parent.time - n.time
            if n.branch_class == 'S':
                L_S += bl
            else:
                L_I += bl
    return L_S, L_I


def get_branches(root, klass=None):
    """Return [(node, branch_length), ...] for branches of given class."""
    out = []
    for n in get_all_nodes(root):
        if n.parent is not None:
            bl = n.parent.time - n.time
            if bl > 0 and (klass is None or n.branch_class == klass):
                out.append((n, bl))
    return out


def get_leaves_below(node):
    """Collect sample_ids of all leaves below node."""
    ids = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_leaf():
            ids.add(n.sample_id)
        stack.extend(n.children)
    return ids


def find_root(node):
    """Walk up to root from any node."""
    while node.parent is not None:
        node = node.parent
    return node


# ===================================================================
# Gene flux model
# ===================================================================

class GeneFluxModel:
    """
    Spatial model for gene flux within the inversion.

    Positions in [0, 1] where 0 and 1 are breakpoints.
    Gene flux = double crossover with fixed window width w.
    phi(x) = Prob(site x is in a random flux interval).

    Following Peischl et al. (2013):
      phi(x) = min(x, 1-x, w) / (1-w)   for w < 1
    """

    def __init__(self, w=0.3):
        self.w = w

    def phi(self, x):
        """Probability that site x is affected by a random gene flux event."""
        if x <= 0.0 or x >= 1.0:
            return 0.0
        w = self.w
        if w >= 1.0:
            return 1.0
        denom = 1.0 - w
        if denom <= 0:
            return 1.0
        val = min(x, 1.0 - x, w) / denom
        return max(0.0, min(1.0, val))


# ===================================================================
# Structured coalescent tree builder
# ===================================================================

def build_structured_tree(n_std, n_inv, p_inv, c, rho, phi_x, rng):
    """
    Build coalescent tree at a single site under the structured coalescent
    with two karyotype classes.

    Rates from Peischl et al. (2013) equation (2):
      Coal S:     C(k_S,2) / (1 - p_inv)
      Coal I:     C(k_I,2) / p_inv
      Flux S->I:  k_S * c * (rho/2) * p_inv * phi_x
      Flux I->S:  k_I * c * (rho/2) * (1-p_inv) * phi_x

    Returns (root, leaves).
    """
    p_std = 1.0 - p_inv

    # Create leaves
    leaves = []
    for i in range(n_std):
        leaves.append(Node(time=0.0, sample_id=i, branch_class='S'))
    for i in range(n_inv):
        leaves.append(Node(time=0.0, sample_id=n_std + i, branch_class='I'))

    # Active lineages: [node, class]
    active = [[leaf, leaf.branch_class] for leaf in leaves]
    t = 0.0

    while len(active) > 1:
        k_S = sum(1 for _, cls in active if cls == 'S')
        k_I = sum(1 for _, cls in active if cls == 'I')

        # Rates
        rc_S = (k_S * (k_S - 1) / 2.0) / p_std if k_S >= 2 and p_std > 0 else 0.0
        rc_I = (k_I * (k_I - 1) / 2.0) / p_inv if k_I >= 2 and p_inv > 0 else 0.0
        rf_SI = k_S * c * (rho / 2.0) * p_inv * phi_x if k_S > 0 else 0.0
        rf_IS = k_I * c * (rho / 2.0) * p_std * phi_x if k_I > 0 else 0.0

        total = rc_S + rc_I + rf_SI + rf_IS
        if total <= 0:
            raise RuntimeError(
                f"Stuck: k_S={k_S}, k_I={k_I}, phi={phi_x}, "
                f"p_inv={p_inv}, c={c}, rho={rho}"
            )

        dt = rng.exponential(1.0 / total)
        t += dt

        u = rng.random() * total
        cum = 0.0

        cum += rc_S
        if u < cum and rc_S > 0:
            _coalesce(active, 'S', t, rng)
            continue

        cum += rc_I
        if u < cum and rc_I > 0:
            _coalesce(active, 'I', t, rng)
            continue

        cum += rf_SI
        if u < cum and rf_SI > 0:
            _flux(active, 'S', 'I', t, rng)
            continue

        _flux(active, 'I', 'S', t, rng)

    root = active[0][0]
    root.branch_class = active[0][1]
    return root, leaves


def _coalesce(active, klass, t, rng):
    """Coalesce two random lineages of the given class."""
    indices = [i for i, (_, cls) in enumerate(active) if cls == klass]
    picked = rng.choice(len(indices), size=2, replace=False)
    i1, i2 = indices[picked[0]], indices[picked[1]]
    n1, _ = active[i1]
    n2, _ = active[i2]

    coal = Node(time=t, branch_class=klass)
    coal.children = [n1, n2]
    n1.parent = coal
    n2.parent = coal

    for i in sorted([i1, i2], reverse=True):
        active.pop(i)
    active.append([coal, klass])


def _flux(active, from_cls, to_cls, t, rng):
    """Gene flux: one lineage switches class. Creates degree-2 node."""
    indices = [i for i, (_, cls) in enumerate(active) if cls == from_cls]
    idx = indices[rng.integers(len(indices))]
    old_node, _ = active[idx]

    flux_node = Node(time=t, branch_class=to_cls)
    flux_node.children = [old_node]
    old_node.parent = flux_node

    active[idx] = [flux_node, to_cls]


# ===================================================================
# SMC: prune and reattach
# ===================================================================

def smc_prune_and_reattach(root, recomb_class, p_inv, c, rho, phi_x, rng):
    """
    One SMC update: pick a branch of recomb_class, prune above it,
    reattach under structured coalescent.

    Returns new_root.
    """
    branches = get_branches(root, recomb_class)
    if not branches:
        return root

    lengths = np.array([bl for _, bl in branches])
    probs = lengths / lengths.sum()
    bi = rng.choice(len(branches), p=probs)
    target, target_bl = branches[bi]

    t_cut = target.time + rng.random() * target_bl

    # --- Prune ---
    # Walk up from target past degree-2 nodes to nearest coalescent node
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

    # --- Reattach ---
    new_root = _reattach(new_root, target, recomb_class, t_cut,
                         p_inv, c, rho, phi_x, rng)
    return new_root


def _reattach(root, floating, fclass, t_start, p_inv, c, rho, phi_x, rng):
    """
    Reattach floating lineage.  Walk backward in time through tree
    intervals, attempting coalescence or gene flux.
    """
    p_std = 1.0 - p_inv
    t = t_start

    for _safety in range(50000):
        all_nodes = get_all_nodes(root)

        # Sorted unique times above t
        times_above = sorted(set(n.time for n in all_nodes if n.time > t))

        went_above = True
        for t_next in times_above:
            # Find same-class branches alive at time t
            same = [n for n in all_nodes
                    if n.parent is not None
                    and n.time <= t < n.parent.time
                    and n.branch_class == fclass]

            k_same = len(same)
            p_same = p_std if fclass == 'S' else p_inv
            p_other = p_inv if fclass == 'S' else p_std

            rate_coal = k_same / p_same if k_same > 0 and p_same > 0 else 0.0
            rate_flux = c * (rho / 2.0) * p_other * phi_x if phi_x > 0 else 0.0
            total = rate_coal + rate_flux

            if total <= 0:
                t = t_next
                continue

            dt = rng.exponential(1.0 / total)
            t_event = t + dt

            if t_event < t_next:
                if rng.random() * total < rate_coal and k_same > 0:
                    # Coalesce
                    attach_to = same[rng.integers(k_same)]
                    coal = Node(time=t_event, branch_class=fclass)
                    old_p = attach_to.parent
                    coal.parent = old_p
                    coal.children = [attach_to, floating]
                    if old_p is not None:
                        old_p.children = [coal if ch is attach_to else ch
                                          for ch in old_p.children]
                    attach_to.parent = coal
                    floating.parent = coal
                    floating.branch_class = fclass
                    return coal if old_p is None else root
                else:
                    # Gene flux: switch class
                    new_cls = 'I' if fclass == 'S' else 'S'
                    fn = Node(time=t_event, branch_class=new_cls)
                    fn.children = [floating]
                    floating.parent = fn
                    floating.branch_class = fclass
                    floating = fn
                    fclass = new_cls
                    t = t_event
                    went_above = False
                    break  # restart outer loop with updated tree
            else:
                t = t_next

        if went_above:
            # Above all internal nodes -- coalesce above root
            return _coalesce_above_root(root, floating, fclass, t,
                                        p_inv, c, rho, phi_x, rng)

    # Fallback
    return _coalesce_above_root(root, floating, fclass, t,
                                p_inv, c, rho, phi_x, rng)


def _coalesce_above_root(root, floating, fclass, t, p_inv, c, rho, phi_x, rng):
    """
    Coalesce floating lineage with root lineage above the tree.

    Full two-lineage structured coalescent: EITHER lineage can flux
    independently.  Previous implementation only allowed the floating
    lineage to flux, which introduces a systematic bias.
    """
    p_std = 1.0 - p_inv
    rclass = root.branch_class

    for _ in range(100000):
        # Rates for floating lineage
        p_other_f = p_inv if fclass == 'S' else p_std
        rf_floating = c * (rho / 2.0) * p_other_f * phi_x

        # Rates for root lineage
        p_other_r = p_inv if rclass == 'S' else p_std
        rf_root = c * (rho / 2.0) * p_other_r * phi_x

        # Coalescence (only if same class)
        if fclass == rclass:
            p_same = p_std if fclass == 'S' else p_inv
            rc = 1.0 / p_same if p_same > 0 else 0.0
        else:
            rc = 0.0

        total = rc + rf_floating + rf_root
        if total <= 0:
            t += 50.0
            coal = Node(time=t, branch_class=fclass)
            coal.children = [root, floating]
            root.parent = coal
            floating.parent = coal
            floating.branch_class = fclass
            return coal

        dt = rng.exponential(1.0 / total)
        t += dt

        u = rng.random() * total

        if u < rc:
            # Coalesce
            coal = Node(time=t, branch_class=fclass)
            coal.children = [root, floating]
            root.parent = coal
            floating.parent = coal
            floating.branch_class = fclass
            return coal
        elif u < rc + rf_floating:
            # Floating lineage fluxes
            new_cls = 'I' if fclass == 'S' else 'S'
            fn = Node(time=t, branch_class=new_cls)
            fn.children = [floating]
            floating.parent = fn
            floating.branch_class = fclass
            floating = fn
            fclass = new_cls
        else:
            # Root lineage fluxes
            new_cls = 'I' if rclass == 'S' else 'S'
            fn = Node(time=t, branch_class=new_cls)
            fn.children = [root]
            root.parent = fn
            root.branch_class = rclass  # already set
            root = fn
            rclass = new_cls

    # Absolute fallback
    t += 1.0
    coal = Node(time=t, branch_class=fclass)
    coal.children = [root, floating]
    root.parent = coal
    floating.parent = coal
    floating.branch_class = fclass
    return coal


# ===================================================================
# Mutation model (infinite sites)
# ===================================================================

def drop_mutations(trees_intervals, theta, nsam, rng):
    """
    Infinite sites mutations on marginal trees.

    Returns (positions, haplotypes).
    """
    mutations = []

    for root, left, right in trees_intervals:
        seg_len = right - left
        if seg_len <= 0:
            continue
        branches = get_branches(root)
        L_total = sum(bl for _, bl in branches)
        if L_total <= 0:
            continue

        n_muts = rng.poisson(min((theta / 2.0) * L_total * seg_len, 1e6))
        if n_muts == 0:
            continue
        n_muts = min(n_muts, 100000)  # hard cap per segment

        bl_arr = np.array([bl for _, bl in branches])
        bl_probs = bl_arr / bl_arr.sum()

        for _ in range(n_muts):
            pos = rng.uniform(left, right)
            bi = rng.choice(len(branches), p=bl_probs)
            leaf_ids = get_leaves_below(branches[bi][0])
            mutations.append((pos, leaf_ids))

    if not mutations:
        return [], np.zeros((nsam, 0), dtype=int)

    mutations.sort(key=lambda x: x[0])
    positions = [m[0] for m in mutations]
    haplotypes = np.zeros((nsam, len(mutations)), dtype=int)
    for j, (_, ids) in enumerate(mutations):
        for sid in ids:
            haplotypes[sid, j] = 1

    return positions, haplotypes


# ===================================================================
# Main simulator
# ===================================================================

class MsinvSimulator:
    def __init__(self, nsam, nreps, theta, rho, nsites,
                 n_std=None, n_inv=None,
                 p_inv=0.0, c=0.0,
                 flux_window=0.3, seed=None):
        self.nsam = nsam
        self.nreps = nreps
        self.theta = theta
        self.rho = rho
        self.nsites = nsites
        self.p_inv = p_inv
        self.c = c
        self.flux_model = GeneFluxModel(w=flux_window)

        self.rng = np.random.default_rng(seed)

        if n_std is not None and n_inv is not None:
            assert n_std + n_inv == nsam, "n_std + n_inv must equal nsam"
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
        """One replicate. Returns (positions, haplotypes)."""
        rng = self.rng

        if not self._has_inversion():
            return self._sim_standard()

        n_std, n_inv = self._get_sample()
        if n_std == 0 or n_inv == 0:
            return self._sim_single_class(n_std, n_inv)

        # Avoid exact breakpoints where T -> infinity (Guerrero et al. 2012).
        # Peischl et al. (2013): "pick x0 and x1 arbitrarily close to 0 or 1"
        x0 = 0.02
        x1 = 0.98
        phi0 = self.flux_model.phi(x0)

        root, leaves = build_structured_tree(
            n_std, n_inv, self.p_inv, self.c, self.rho, phi0, rng
        )

        # SMC along chromosome [x0, x1]
        trees = []
        pos = x0
        p_std = 1.0 - self.p_inv

        # Use the initial and final trees for the edge regions [0,x0) and (x1,1]
        # This is a constant-tree approximation for the near-breakpoint regions.

        for _ in range(500000):
            L_S, L_I = branch_lengths_by_class(root)
            weighted_L = L_S * p_std + L_I * self.p_inv
            if weighted_L <= 0:
                trees.append((root, pos, x1))
                break

            rate = (self.rho / 2.0) * weighted_L
            dx = rng.exponential(1.0 / rate)
            new_pos = pos + dx

            if new_pos >= x1:
                trees.append((root, pos, x1))
                break

            trees.append((root, pos, new_pos))
            pos = new_pos

            # Determine recombination class
            u = rng.random() * weighted_L
            recomb_class = 'S' if u < L_S * p_std else 'I'

            phi_x = self.flux_model.phi(pos)
            root = smc_prune_and_reattach(
                root, recomb_class,
                self.p_inv, self.c, self.rho, phi_x, rng
            )

        # Add edge regions using edge trees (constant tree approximation)
        if trees:
            first_root = trees[0][0]
            last_root = trees[-1][0]
            trees.insert(0, (first_root, 0.0, x0))
            trees.append((last_root, x1, 1.0))

        return drop_mutations(trees, self.theta, self.nsam, rng)

    def _sim_standard(self):
        """Standard coalescent with SMC (no inversion)."""
        rng = self.rng
        n = self.nsam

        # Build initial tree
        leaves = [Node(time=0.0, sample_id=i) for i in range(n)]
        active = list(leaves)
        t = 0.0
        while len(active) > 1:
            k = len(active)
            rate = k * (k - 1) / 2.0
            t += rng.exponential(1.0 / rate)
            idx = rng.choice(k, size=2, replace=False)
            n1, n2 = active[idx[0]], active[idx[1]]
            coal = Node(time=t)
            coal.children = [n1, n2]
            n1.parent = coal
            n2.parent = coal
            for i in sorted(idx, reverse=True):
                active.pop(i)
            active.append(coal)
        root = active[0]

        # SMC
        trees = []
        pos = 0.0
        for _ in range(500000):
            branches = get_branches(root)
            L = sum(bl for _, bl in branches)
            if L <= 0:
                trees.append((root, pos, 1.0))
                break
            rate = (self.rho / 2.0) * L
            if rate <= 0:
                trees.append((root, pos, 1.0))
                break
            dx = rng.exponential(1.0 / rate)
            new_pos = pos + dx
            if new_pos >= 1.0:
                trees.append((root, pos, 1.0))
                break
            trees.append((root, pos, new_pos))
            pos = new_pos

            # Pick branch
            bl_arr = np.array([bl for _, bl in branches])
            probs = bl_arr / bl_arr.sum()
            bi = rng.choice(len(branches), p=probs)
            target, tbl = branches[bi]
            t_cut = target.time + rng.random() * tbl

            # Prune (standard: no degree-2 nodes in standard tree)
            p = target.parent
            if p is None or len(p.children) != 2:
                continue
            sib = [ch for ch in p.children if ch is not target][0]
            gp = p.parent
            sib.parent = gp
            if gp is not None:
                gp.children = [sib if ch is p else ch for ch in gp.children]
            root = sib if root is p else root

            target.parent = None

            # Reattach (standard SMC)
            above = []
            for n in get_all_nodes(root):
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

                coal = Node(time=t_a)
                old_p = an.parent
                coal.parent = old_p
                coal.children = [an, target]
                an.parent = coal
                target.parent = coal
                if old_p is not None:
                    old_p.children = [coal if ch is an else ch for ch in old_p.children]
                root = find_root(root)
            else:
                t_c = max(t_cut, root.time) + rng.exponential(1.0)
                coal = Node(time=t_c)
                coal.children = [root, target]
                root.parent = coal
                target.parent = coal
                root = coal

        if not trees:
            trees.append((root, 0.0, 1.0))
        return drop_mutations(trees, self.theta, self.nsam, rng)

    def _sim_single_class(self, n_std, n_inv):
        """All samples from one class."""
        rng = self.rng
        n = self.nsam
        p = self.p_inv if n_inv > 0 else (1.0 - self.p_inv)
        klass = 'I' if n_inv > 0 else 'S'

        leaves = [Node(time=0.0, sample_id=i, branch_class=klass)
                  for i in range(n)]
        active = list(leaves)
        t = 0.0
        while len(active) > 1:
            k = len(active)
            rate = (k * (k - 1) / 2.0) / p
            t += rng.exponential(1.0 / rate)
            idx = rng.choice(k, size=2, replace=False)
            n1, n2 = active[idx[0]], active[idx[1]]
            coal = Node(time=t, branch_class=klass)
            coal.children = [n1, n2]
            n1.parent = coal
            n2.parent = coal
            for i in sorted(idx, reverse=True):
                active.pop(i)
            active.append(coal)
        root = active[0]
        trees = [(root, 0.0, 1.0)]
        return drop_mutations(trees, self.theta, self.nsam, rng)

    def run(self, outfile=sys.stdout):
        """Run all replicates, output ms format."""
        cmd_parts = [f"msinv {self.nsam} {self.nreps}"]
        cmd_parts.append(f"-t {self.theta}")
        cmd_parts.append(f"-r {self.rho} {self.nsites}")
        if self._has_inversion():
            cmd_parts.append(f"-inv {self.p_inv} {self.c}")
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


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    nsam = int(argv[0])
    nreps = int(argv[1])
    theta = 0.0; rho = 0.0; nsites = 1000
    p_inv = 0.0; c = 0.0
    n_std = None; n_inv = None
    flux_window = 0.3; seed = None

    i = 2
    while i < len(argv):
        f = argv[i]
        if f == '-t':
            theta = float(argv[i+1]); i += 2
        elif f == '-r':
            rho = float(argv[i+1]); nsites = int(argv[i+2]); i += 3
        elif f == '-inv':
            p_inv = float(argv[i+1]); c = float(argv[i+2]); i += 3
        elif f == '-I':
            n_std = int(argv[i+1]); n_inv = int(argv[i+2]); i += 3
        elif f == '-flux_window':
            flux_window = float(argv[i+1]); i += 2
        elif f in ('-seed', '-seeds'):
            seed = int(argv[i+1]); i += 2
        else:
            print(f"Warning: unknown flag {f}", file=sys.stderr); i += 1

    return dict(nsam=nsam, nreps=nreps, theta=theta, rho=rho,
                nsites=nsites, p_inv=p_inv, c=c,
                n_std=n_std, n_inv=n_inv,
                flux_window=flux_window, seed=seed)


def main():
    p = parse_args()
    sim = MsinvSimulator(**p)
    sim.run()


if __name__ == '__main__':
    main()
