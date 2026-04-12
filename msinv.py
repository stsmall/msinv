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
  -bp <left> <right>          Inversion breakpoints as fractions (default: 0.3 0.7)
  -flux_window <w>            Gene flux interval width (default: 0.3)
  -t_inv <age>                Inversion age in 2N gen (bounds S-I divergence)
  -N <Ne>                     Effective pop size for trajectory (default: 10000)
  -trajectory <type>          'constant', 'deterministic', or 'stochastic'
  -s <sel_coeff>              Selection coefficient for inversion (default: 0)
  -seed <s>                   Random seed

Multi-population options:
  -npops <n>                  Number of populations (default: 1)
  -m <rate>                   Symmetric migration rate 4Nm (default: 0)
  -sample_config <spec>       Sample composition per (class, pop), e.g.
                              "S:0:3,I:0:2,S:1:3,I:1:2"
  -demo_merge <t:src:dst>     Population merge: at time t, pop src → pop dst
                              (can be specified multiple times)

Trajectories:
  constant:       Fixed p_inv, optionally bounded by -t_inv.
                  WARNING: without -t_inv, S-I coalescence time
                  diverges to infinity at breakpoints.
  deterministic:  Logistic sweep from 1/(2N) to p_inv under selection s.
                  t_inv computed automatically from s and N.
  stochastic:     WF diffusion backward with drift + selection.
                  Reflecting boundary at p=0 models recurrent origins
                  at same breakpoints (driven by repeat arrays).
                  t_inv is the time to reach 1/(2N) going backward.

Chromosome structure:
  [0, bp_left):       collinear, panmictic coalescence
  [bp_left, bp_right]: inversion, class-structured coalescence + gene flux
  (bp_right, 1]:      collinear, panmictic coalescence

Examples:
  # Basic: 6 samples (3S+3I), inversion at [0.3,0.7], t_inv=10
  msinv.py 6 100 -t 10 -r 50 1000 -inv 0.5 0.01 -I 3 3 -t_inv 10

  # Stochastic trajectory (neutral, N=10000)
  msinv.py 6 100 -t 10 -r 50 1000 -inv 0.5 0.01 -I 3 3 \
    -trajectory stochastic -N 10000

  # Two populations with migration
  msinv.py 10 100 -t 10 -r 50 1000 -inv 0.5 0.01 -npops 2 -m 1.0 \
    -sample_config "S:0:3,I:0:2,S:1:3,I:1:2" -t_inv 10 \
    -demo_merge "5.0:1:0"
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
    population: population index (0, 1, ...) of this lineage.
    """
    __slots__ = ['time', 'children', 'parent', 'sample_id',
                 'branch_class', 'population']

    def __init__(self, time=0.0, sample_id=None, branch_class='S',
                 population=0):
        self.time = time
        self.children = []
        self.parent = None
        self.sample_id = sample_id
        self.branch_class = branch_class
        self.population = population

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
        for ch in n.children:
            stack.append(ch)
    return nodes


def branch_lengths_by_class(root):
    """Return (L_S, L_I) total branch lengths. Single traversal."""
    L_S = L_I = 0.0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.parent is not None:
            bl = n.parent.time - n.time
            if n.branch_class == 'S':
                L_S += bl
            else:
                L_I += bl
        for ch in n.children:
            stack.append(ch)
    return L_S, L_I


def count_lineages_by_class_pop(active):
    """Count lineages per (class, population) from active list."""
    counts = {}
    for entry in active:
        if len(entry) == 3:
            _, cls, pop = entry
        else:
            _, cls = entry
            pop = 0
        key = (cls, pop)
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_branches(root, klass=None):
    """Return [(node, branch_length), ...] for branches of given class.
    Single traversal without intermediate list."""
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.parent is not None:
            bl = n.parent.time - n.time
            if bl > 0 and (klass is None or n.branch_class == klass):
                out.append((n, bl))
        for ch in n.children:
            stack.append(ch)
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
# Inversion frequency trajectories
# ===================================================================

class ConstantFrequency:
    """Constant inversion frequency with optional finite age."""
    def __init__(self, p_inv, t_inv=None):
        self.p_inv = p_inv
        self.t_inv = t_inv
    def __call__(self, t):
        if self.t_inv is not None and t >= self.t_inv:
            return 0.0
        return self.p_inv


class DeterministicTrajectory:
    """
    Deterministic frequency trajectory for the inversion.

    Going backward from current frequency p_final at t=0 to 1/(2N) at t_inv.
    Uses logistic model: the inversion spread under selection s from 1/(2N).

    Forward: p(t) = p0 * exp(s*t) / (1 - p0 + p0*exp(s*t))
    Backward: p_inv(t_back) decreases as t_back increases.
    """
    def __init__(self, p_final, N, s=0.01):
        self.p_final = p_final
        self.N = N
        self.s_scaled = 2 * N * s
        self.p0 = 1.0 / (2 * N)

        if self.s_scaled > 0 and p_final > self.p0:
            self.t_inv = (np.log(p_final / (1 - p_final))
                          - np.log(self.p0 / (1 - self.p0))) / self.s_scaled
        else:
            self.t_inv = 20.0

    def __call__(self, t):
        if t >= self.t_inv:
            return 0.0
        t_fwd = self.t_inv - t
        if self.s_scaled <= 0:
            return self.p0
        exp_st = np.exp(self.s_scaled * t_fwd)
        p = self.p0 * exp_st / (1 - self.p0 + self.p0 * exp_st)
        return min(p, self.p_final)


class StochasticTrajectory:
    """
    Stochastic frequency trajectory for an inversion.

    Models the recurrent origin framework: inversions appear repeatedly
    at the same breakpoints (driven by repeat arrays / genomic
    architecture) until one escapes drift and establishes.

    Going backward, frequency decreases from p_final toward 1/(2N)
    via Wright-Fisher diffusion:
      dp = -s*p*(1-p)*dt + sqrt(p*(1-p)/(2N)) * dW

    Reflecting boundary at p=0 models recurrent de novo origins
    at the same breakpoints. The inversion "age" t_inv is when p
    first reaches 1/(2N) going backward.
    """
    def __init__(self, p_final, N, s=0.0, rng=None):
        self.p_final = p_final
        self.N = N
        self.s = s
        self.p0 = 1.0 / (2 * N)
        if rng is None:
            rng = np.random.default_rng()
        dt = 1.0 / (2 * N)
        p = p_final
        times, freqs = [0.0], [p]
        t = 0.0
        while p > self.p0 and t < 100.0:
            dp_sel = -s * p * (1.0 - p) * dt
            sd = np.sqrt(max(0, p * (1.0 - p) * dt))
            dp_drift = rng.normal(0, sd) if sd > 0 else 0.0
            p_new = p + dp_sel + dp_drift
            if p_new <= 0:
                p_new = abs(p_new) + self.p0
            if p_new >= 1.0:
                p_new = 2.0 - p_new
            p = np.clip(p_new, self.p0, 1.0 - self.p0)
            t += dt
            times.append(t); freqs.append(p)
        self.t_inv = t
        self._times = np.array(times)
        self._freqs = np.array(freqs)

    def __call__(self, t):
        if t >= self.t_inv:
            return 0.0
        return float(np.interp(t, self._times, self._freqs))


# ===================================================================
# Structured coalescent tree builder
# ===================================================================

def build_structured_tree(n_std, n_inv, p_inv, c, rho, phi_x, rng,
                          p_inv_func=None, sample_config=None,
                          n_pops=1, mig_rate=0.0, demo_events=None):
    """
    Build coalescent tree at a single site under the structured coalescent
    with karyotype classes and (optionally) multiple populations.

    Single population (n_pops=1):
      Coal S:     C(k_S,2) / (1 - p_inv)
      Coal I:     C(k_I,2) / p_inv
      Flux S->I:  k_S * c * (rho/2) * p_inv * phi_x  (within pop)
      Flux I->S:  k_I * c * (rho/2) * (1-p_inv) * phi_x  (within pop)

    Multiple populations (n_pops>1):
      Same rates per (class, pop) combination.
      Migration: k * mig_rate / 2 per (class, pop) → other pop.
      Gene flux: only within the same population.

    sample_config: dict {(class, pop): count} for initial samples.
      If None, uses n_std S in pop 0, n_inv I in pop 0.

    demo_events: list of (time, event_type, args) for population merges etc.
      event_type 'merge': all lineages in pop args[0] move to pop args[1]

    Returns (root, leaves).
    """
    if p_inv_func is None:
        p_inv_func = ConstantFrequency(p_inv)

    # Create leaves
    leaves = []
    sid = 0
    if sample_config is not None:
        for (cls, pop), count in sorted(sample_config.items()):
            for _ in range(count):
                leaves.append(Node(time=0.0, sample_id=sid,
                                   branch_class=cls, population=pop))
                sid += 1
    else:
        for i in range(n_std):
            leaves.append(Node(time=0.0, sample_id=sid, branch_class='S'))
            sid += 1
        for i in range(n_inv):
            leaves.append(Node(time=0.0, sample_id=sid, branch_class='I'))
            sid += 1

    # Active: [node, class, pop]
    active = [[leaf, leaf.branch_class, leaf.population] for leaf in leaves]
    t = 0.0

    # Sort demographic events by time
    if demo_events is None:
        demo_events = []
    demo_events = sorted(demo_events, key=lambda x: x[0])
    demo_idx = 0

    while len(active) > 1:
        # Check demographic events
        while demo_idx < len(demo_events) and demo_events[demo_idx][0] <= t:
            _, etype, eargs = demo_events[demo_idx]
            if etype == 'merge':
                # Move all lineages from pop eargs[0] to pop eargs[1]
                src, dst = eargs
                for entry in active:
                    if entry[2] == src:
                        entry[2] = dst
                        entry[0].population = dst
            demo_idx += 1

        p_inv_t = p_inv_func(t)

        # Beyond inversion age: all become S, panmictic within each pop
        if p_inv_t <= 0:
            for entry in active:
                entry[1] = 'S'
            # If multiple pops still exist, continue with migration
            # Otherwise standard panmictic
            pops_present = set(e[2] for e in active)
            if len(pops_present) == 1:
                while len(active) > 1:
                    k = len(active)
                    dt = rng.exponential(1.0 / (k * (k - 1) / 2.0))
                    t += dt
                    _coalesce_pop(active, 'S', active[0][2], t, rng)
                break
            # Multiple pops: fall through to rate-based loop below
            # with p_inv=0 (panmictic within each pop, migration between)

        p_std_t = 1.0 - max(p_inv_t, 0)
        p_inv_t = max(p_inv_t, 0)

        # Build rate table: (event_type, class, pop, rate)
        rates = []
        counts = count_lineages_by_class_pop(active)

        for (cls, pop), k in counts.items():
            if p_inv_t > 0:
                f = p_std_t if cls == 'S' else p_inv_t
            else:
                f = 1.0  # panmictic

            # Coalescence
            if k >= 2 and f > 0:
                rates.append(('coal', cls, pop, k * (k - 1) / 2.0 / f))

            # Gene flux (within population, only if inversion exists)
            if k > 0 and phi_x > 0 and p_inv_t > 0:
                f_other = p_inv_t if cls == 'S' else p_std_t
                rf = k * c * (rho / 2.0) * f_other * phi_x
                if rf > 0:
                    rates.append(('flux', cls, pop, rf))

            # Migration (to each other population)
            if k > 0 and mig_rate > 0:
                for other_pop in range(n_pops):
                    if other_pop != pop:
                        rates.append(('mig', cls, pop, k * mig_rate / 2.0))

        total = sum(r for _, _, _, r in rates)
        if total <= 0:
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv is not None and t < t_inv:
                t = t_inv
                continue
            # Check next demographic event
            if demo_idx < len(demo_events):
                t = demo_events[demo_idx][0]
                continue
            raise RuntimeError(
                f"Stuck: counts={counts}, phi={phi_x}, "
                f"p_inv={p_inv_t}, c={c}")

        dt = rng.exponential(1.0 / total)

        # Check t_inv
        t_inv = getattr(p_inv_func, 't_inv', None)
        if t_inv is not None and t + dt >= t_inv:
            t = t_inv
            continue

        # Check next demographic event
        if demo_idx < len(demo_events) and t + dt >= demo_events[demo_idx][0]:
            t = demo_events[demo_idx][0]
            continue

        t += dt

        # Choose event
        u = rng.random() * total
        cum = 0.0
        event = None
        for etype, cls, pop, r in rates:
            cum += r
            if u < cum:
                event = (etype, cls, pop)
                break
        if event is None:
            event = (rates[-1][0], rates[-1][1], rates[-1][2])

        etype, cls, pop = event

        if etype == 'coal':
            _coalesce_pop(active, cls, pop, t, rng)
        elif etype == 'flux':
            _flux_pop(active, cls, pop, t, rng)
        elif etype == 'mig':
            _migrate(active, cls, pop, n_pops, t, rng)

    root = active[0][0]
    root.branch_class = active[0][1]
    return root, leaves


def _coalesce(active, klass, t, rng):
    """Coalesce two random lineages of the given class (single-pop compat)."""
    indices = [i for i, entry in enumerate(active)
               if entry[1] == klass]
    picked = rng.choice(len(indices), size=2, replace=False)
    i1, i2 = indices[picked[0]], indices[picked[1]]
    n1 = active[i1][0]
    n2 = active[i2][0]
    pop = active[i1][2] if len(active[i1]) > 2 else 0

    coal = Node(time=t, branch_class=klass, population=pop)
    coal.children = [n1, n2]
    n1.parent = coal
    n2.parent = coal

    for i in sorted([i1, i2], reverse=True):
        active.pop(i)
    active.append([coal, klass, pop])


def _coalesce_pop(active, klass, pop, t, rng):
    """Coalesce two random lineages of given class AND population."""
    indices = [i for i, entry in enumerate(active)
               if entry[1] == klass and entry[2] == pop]
    if len(indices) < 2:
        return
    picked = rng.choice(len(indices), size=2, replace=False)
    i1, i2 = indices[picked[0]], indices[picked[1]]
    n1 = active[i1][0]
    n2 = active[i2][0]

    coal = Node(time=t, branch_class=klass, population=pop)
    coal.children = [n1, n2]
    n1.parent = coal
    n2.parent = coal

    for i in sorted([i1, i2], reverse=True):
        active.pop(i)
    active.append([coal, klass, pop])


def _flux(active, from_cls, to_cls, t, rng):
    """Gene flux: one lineage switches class (single-pop compat)."""
    indices = [i for i, entry in enumerate(active)
               if entry[1] == from_cls]
    idx = indices[rng.integers(len(indices))]
    old_node = active[idx][0]
    pop = active[idx][2] if len(active[idx]) > 2 else 0

    flux_node = Node(time=t, branch_class=to_cls, population=pop)
    flux_node.children = [old_node]
    old_node.parent = flux_node

    active[idx] = [flux_node, to_cls, pop]


def _flux_pop(active, from_cls, pop, t, rng):
    """Gene flux within a specific population."""
    to_cls = 'I' if from_cls == 'S' else 'S'
    indices = [i for i, entry in enumerate(active)
               if entry[1] == from_cls and entry[2] == pop]
    if not indices:
        return
    idx = indices[rng.integers(len(indices))]
    old_node = active[idx][0]

    flux_node = Node(time=t, branch_class=to_cls, population=pop)
    flux_node.children = [old_node]
    old_node.parent = flux_node

    active[idx] = [flux_node, to_cls, pop]


def _migrate(active, klass, from_pop, n_pops, t, rng):
    """Migration: one lineage moves to a random other population."""
    indices = [i for i, entry in enumerate(active)
               if entry[1] == klass and entry[2] == from_pop]
    if not indices:
        return
    idx = indices[rng.integers(len(indices))]

    # Choose destination (uniform among other pops)
    other_pops = [p for p in range(n_pops) if p != from_pop]
    if not other_pops:
        return
    to_pop = other_pops[rng.integers(len(other_pops))]

    # Migration doesn't create a new node, just changes population
    active[idx][0].population = to_pop
    active[idx][2] = to_pop


# ===================================================================
# SMC: prune and reattach
# ===================================================================

def smc_prune_and_reattach_panmictic(root, rng):
    """
    Standard SMC prune-and-reattach for panmictic (collinear) regions.
    No class structure: any lineage can reattach to any branch.
    """
    branches = get_branches(root)
    if not branches:
        return root

    lengths = np.array([bl for _, bl in branches])
    probs = lengths / lengths.sum()
    bi = rng.choice(len(branches), p=probs)
    target, target_bl = branches[bi]

    t_cut = target.time + rng.random() * target_bl

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

    # Reattach: standard (panmictic) — pick any branch above t_cut
    all_nodes = get_all_nodes(new_root)
    above = []
    for n in all_nodes:
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

        coal = Node(time=t_a, branch_class=an.branch_class)
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
        coal = Node(time=t_c, branch_class='S')
        coal.children = [new_root, target]
        new_root.parent = coal
        target.parent = coal
        new_root = coal

    return new_root


def smc_prune_and_reattach(root, recomb_class, p_inv, c, rho, phi_x, rng,
                           p_inv_func=None):
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
                         p_inv, c, rho, phi_x, rng,
                         p_inv_func=p_inv_func)
    return new_root


def _reattach(root, floating, fclass, t_start, p_inv, c, rho, phi_x, rng,
              p_inv_func=None):
    """
    Reattach floating lineage. Walk backward in time through tree
    intervals, attempting coalescence or gene flux.
    Uses p_inv_func(t) for time-varying frequency.
    """
    if p_inv_func is None:
        p_inv_func = ConstantFrequency(p_inv)

    t = t_start

    for _safety in range(50000):
        p_inv_t = p_inv_func(t)

        # Beyond inversion age: panmictic, coalesce with any branch
        if p_inv_t <= 0:
            fclass = 'S'
            return _coalesce_above_root(root, floating, 'S', t,
                                        0.0, c, rho, phi_x, rng,
                                        p_inv_func=ConstantFrequency(0.0))

        p_std_t = 1.0 - p_inv_t
        all_nodes = get_all_nodes(root)
        times_above = sorted(set(n.time for n in all_nodes if n.time > t))

        went_above = True
        for t_next in times_above:
            p_inv_t = p_inv_func(t)
            if p_inv_t <= 0:
                went_above = True
                break

            p_std_t = 1.0 - p_inv_t
            same = [n for n in all_nodes
                    if n.parent is not None
                    and n.time <= t < n.parent.time
                    and n.branch_class == fclass]

            k_same = len(same)
            p_same = p_std_t if fclass == 'S' else p_inv_t
            p_other = p_inv_t if fclass == 'S' else p_std_t

            rate_coal = k_same / p_same if k_same > 0 and p_same > 0 else 0.0
            rate_flux = c * (rho / 2.0) * p_other * phi_x if phi_x > 0 else 0.0
            total = rate_coal + rate_flux

            if total <= 0:
                # Jump to t_inv if available
                t_inv = getattr(p_inv_func, 't_inv', None)
                if t_inv is not None and t_inv < t_next:
                    t = t_inv
                    went_above = False
                    break
                t = t_next
                continue

            dt = rng.exponential(1.0 / total)
            t_event = t + dt

            # Check t_inv
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv is not None and t_event >= t_inv:
                t = t_inv
                went_above = False
                break

            if t_event < t_next:
                if rng.random() * total < rate_coal and k_same > 0:
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
                    new_cls = 'I' if fclass == 'S' else 'S'
                    fn = Node(time=t_event, branch_class=new_cls)
                    fn.children = [floating]
                    floating.parent = fn
                    floating.branch_class = fclass
                    floating = fn
                    fclass = new_cls
                    t = t_event
                    went_above = False
                    break
            else:
                t = t_next

        if went_above:
            return _coalesce_above_root(root, floating, fclass, t,
                                        p_inv_func(t), c, rho, phi_x, rng,
                                        p_inv_func=p_inv_func)

    return _coalesce_above_root(root, floating, fclass, t,
                                p_inv_func(t), c, rho, phi_x, rng,
                                p_inv_func=p_inv_func)


def _coalesce_above_root(root, floating, fclass, t, p_inv, c, rho, phi_x, rng,
                         p_inv_func=None):
    """
    Coalesce floating lineage with root lineage above the tree.
    Uses p_inv_func(t) for time-varying frequency.
    At t >= t_inv, forces panmictic coalescence.
    """
    if p_inv_func is None:
        p_inv_func = ConstantFrequency(p_inv)

    rclass = root.branch_class

    for _ in range(100000):
        p_inv_t = p_inv_func(t)

        # Beyond inversion age: panmictic
        if p_inv_t <= 0:
            fclass = 'S'
            rclass = 'S'
            dt = rng.exponential(1.0)
            t += dt
            coal = Node(time=t, branch_class='S')
            coal.children = [root, floating]
            root.parent = coal
            floating.parent = coal
            floating.branch_class = 'S'
            return coal

        p_std_t = 1.0 - p_inv_t

        p_other_f = p_inv_t if fclass == 'S' else p_std_t
        rf_floating = c * (rho / 2.0) * p_other_f * phi_x

        p_other_r = p_inv_t if rclass == 'S' else p_std_t
        rf_root = c * (rho / 2.0) * p_other_r * phi_x

        if fclass == rclass:
            p_same = p_std_t if fclass == 'S' else p_inv_t
            rc = 1.0 / p_same if p_same > 0 else 0.0
        else:
            rc = 0.0

        total = rc + rf_floating + rf_root
        if total <= 0:
            # Jump to t_inv
            t_inv = getattr(p_inv_func, 't_inv', None)
            if t_inv is not None:
                t = t_inv
                continue
            # True fallback (should not happen with t_inv)
            t += 50.0
            coal = Node(time=t, branch_class=fclass)
            coal.children = [root, floating]
            root.parent = coal
            floating.parent = coal
            floating.branch_class = fclass
            return coal

        dt = rng.exponential(1.0 / total)

        # Check t_inv
        t_inv = getattr(p_inv_func, 't_inv', None)
        if t_inv is not None and t + dt >= t_inv:
            t = t_inv
            continue

        t += dt
        u = rng.random() * total

        if u < rc:
            coal = Node(time=t, branch_class=fclass)
            coal.children = [root, floating]
            root.parent = coal
            floating.parent = coal
            floating.branch_class = fclass
            return coal
        elif u < rc + rf_floating:
            new_cls = 'I' if fclass == 'S' else 'S'
            fn = Node(time=t, branch_class=new_cls)
            fn.children = [floating]
            floating.parent = fn
            floating.branch_class = fclass
            floating = fn
            fclass = new_cls
        else:
            new_cls = 'I' if rclass == 'S' else 'S'
            fn = Node(time=t, branch_class=new_cls)
            fn.children = [root]
            root.parent = fn
            root.branch_class = rclass
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

        # Frequency trajectory
        if p_inv_func is not None:
            self.p_inv_func = p_inv_func
        elif t_inv is not None:
            self.p_inv_func = ConstantFrequency(p_inv, t_inv=t_inv)
        else:
            self.p_inv_func = ConstantFrequency(p_inv)

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
        """
        One replicate: SMC across the full chromosome.

        Three regions:
          [0, bp_left):          collinear, panmictic
          [bp_left, bp_right]:   inversion, class-structured
          (bp_right, 1]:         collinear, panmictic

        Returns (positions, haplotypes).
        """
        rng = self.rng

        if not self._has_inversion():
            return self._sim_standard()

        n_std, n_inv = self._get_sample()
        if n_std == 0 or n_inv == 0:
            return self._sim_single_class(n_std, n_inv)

        bp_l = self.bp_left
        bp_r = self.bp_right
        inv_len = bp_r - bp_l
        p_inv_now = self.p_inv_func(0.0)
        p_std_now = 1.0 - p_inv_now

        # Build initial panmictic tree at position 0 (collinear)
        leaves = [Node(time=0.0, sample_id=i, branch_class='S')
                  for i in range(n_std)]
        leaves += [Node(time=0.0, sample_id=n_std + i, branch_class='I')
                   for i in range(n_inv)]
        active = list(leaves)
        t = 0.0
        # Standard coalescent for initial tree
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
        root = active[0]

        trees = []
        pos = 0.0

        for _ in range(500000):
            if pos >= 1.0:
                break

            in_inv = bp_l <= pos < bp_r

            # Single traversal: get L_S, L_I, L_total, t_max
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

            if in_inv:
                p_inv_t = self.p_inv_func(0.5 * t_max)
                p_std_t = 1.0 - p_inv_t
                if p_inv_t > 0:
                    weighted_L = L_S * p_std_t + L_I * p_inv_t
                else:
                    weighted_L = L_total
                next_boundary = bp_r
            else:
                weighted_L = L_total
                next_boundary = bp_l if pos < bp_l else 1.0

            if weighted_L <= 0:
                trees.append((root, pos, next_boundary))
                pos = next_boundary
                continue

            rate = (self.rho / 2.0) * weighted_L
            dx = rng.exponential(1.0 / rate)

            extent = min(dx, next_boundary - pos, 1.0 - pos)
            if extent <= 0:
                extent = 1e-10

            new_pos = pos + extent
            trees.append((root, pos, new_pos))

            if dx < (next_boundary - pos) and dx < (1.0 - pos):
                # Recombination event
                new_pos = pos + dx

                new_in_inv = bp_l <= new_pos < bp_r

                if new_in_inv:
                    # Structured prune-and-reattach
                    # Reuse L_S, L_I from traversal above (tree unchanged)
                    p_inv_t = self.p_inv_func(0.5 * t_max)
                    p_std_t = 1.0 - p_inv_t
                    weighted_L_now = L_S * p_std_t + L_I * p_inv_t
                    if weighted_L_now > 0:
                        u = rng.random() * weighted_L_now
                        recomb_class = 'S' if u < L_S * p_std_t else 'I'
                    else:
                        recomb_class = 'S'

                    inv_pos = (new_pos - bp_l) / inv_len
                    inv_pos = max(0.02, min(0.98, inv_pos))
                    phi_x = self.flux_model.phi(inv_pos)

                    root = smc_prune_and_reattach(
                        root, recomb_class,
                        self.p_inv, self.c, self.rho, phi_x, rng,
                        p_inv_func=self.p_inv_func
                    )
                else:
                    # Panmictic prune-and-reattach
                    root = smc_prune_and_reattach_panmictic(root, rng)
            else:
                # Hit region boundary. Inversion breakpoints are
                # structural discontinuities — build a new tree
                # from the correct stationary distribution for the
                # new region. This is like forced recombination at
                # the breakpoint, giving independent genealogies
                # across the breakpoint (LD drops to zero there).
                if new_pos >= bp_l and pos < bp_l:
                    # Entering inversion: build structured tree
                    # from equilibrium distribution at bp_left.
                    inv_pos = max(0.02, (new_pos - bp_l) / inv_len)
                    phi_x = self.flux_model.phi(inv_pos)
                    root, _ = build_structured_tree(
                        n_std, n_inv, self.p_inv, self.c, self.rho,
                        phi_x, rng, p_inv_func=self.p_inv_func,
                        sample_config=self.sample_config,
                        n_pops=self.n_pops, mig_rate=self.mig_rate,
                        demo_events=self.demo_events)
                elif new_pos >= bp_r and in_inv:
                    # Leaving inversion: build panmictic tree
                    all_leaves = get_all_nodes(root)
                    sample_leaves = sorted(
                        [n for n in all_leaves if n.is_leaf()],
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
                    root = active[0]

            pos = new_pos

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
    t_inv = None; N_e = 10000
    trajectory = 'constant'; s_coeff = 0.0
    bp_left = 0.3; bp_right = 0.7
    n_pops = 1; mig_rate = 0.0
    sample_config = None; demo_events = []

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
        elif f == '-t_inv':
            t_inv = float(argv[i+1]); i += 2
        elif f == '-N':
            N_e = int(argv[i+1]); i += 2
        elif f == '-trajectory':
            trajectory = argv[i+1]; i += 2
        elif f == '-s':
            s_coeff = float(argv[i+1]); i += 2
        elif f == '-bp':
            bp_left = float(argv[i+1]); bp_right = float(argv[i+2]); i += 3
        elif f == '-npops':
            n_pops = int(argv[i+1]); i += 2
        elif f == '-m':
            mig_rate = float(argv[i+1]); i += 2
        elif f == '-sample_config':
            # Format: "S:0:3,I:0:2,S:1:3,I:1:2" (class:pop:count)
            sample_config = {}
            for part in argv[i+1].split(','):
                cls, pop, cnt = part.split(':')
                sample_config[(cls, int(pop))] = int(cnt)
            i += 2
        elif f == '-demo_merge':
            # Format: time:src_pop:dst_pop
            t_m, src, dst = argv[i+1].split(':')
            demo_events.append((float(t_m), 'merge', (int(src), int(dst))))
            i += 2
        elif f in ('-seed', '-seeds'):
            seed = int(argv[i+1]); i += 2
        else:
            print(f"Warning: unknown flag {f}", file=sys.stderr); i += 1

    return dict(nsam=nsam, nreps=nreps, theta=theta, rho=rho,
                nsites=nsites, p_inv=p_inv, c=c,
                n_std=n_std, n_inv=n_inv,
                flux_window=flux_window, seed=seed,
                t_inv=t_inv, N_e=N_e,
                trajectory=trajectory, s_coeff=s_coeff,
                bp_left=bp_left, bp_right=bp_right,
                n_pops=n_pops, mig_rate=mig_rate,
                sample_config=sample_config, demo_events=demo_events)


def main():
    p = parse_args()

    # Build frequency trajectory
    rng = np.random.default_rng(p['seed'])
    traj_type = p.pop('trajectory')
    N_e = p.pop('N_e')
    s_coeff = p.pop('s_coeff')
    t_inv_arg = p.pop('t_inv')

    if traj_type == 'deterministic':
        p_inv_func = DeterministicTrajectory(p['p_inv'], N_e, s=s_coeff)
        if t_inv_arg is not None:
            p_inv_func.t_inv = t_inv_arg
        print(f"# trajectory=deterministic s={s_coeff} N={N_e} "
              f"t_inv={p_inv_func.t_inv:.4f}", file=sys.stderr)
    elif traj_type == 'stochastic':
        p_inv_func = StochasticTrajectory(p['p_inv'], N_e, s=s_coeff, rng=rng)
        if t_inv_arg is not None:
            p_inv_func.t_inv = t_inv_arg
        print(f"# trajectory=stochastic s={s_coeff} N={N_e} "
              f"t_inv={p_inv_func.t_inv:.4f}", file=sys.stderr)
    else:
        # constant
        if p['p_inv'] > 0 and p['c'] > 0 and t_inv_arg is None:
            print("WARNING: no -t_inv with constant trajectory. "
                  "S-I coalescence may be infinite at breakpoints. "
                  "Use -t_inv or -trajectory stochastic|deterministic.",
                  file=sys.stderr)
        p_inv_func = ConstantFrequency(p['p_inv'], t_inv=t_inv_arg)
        t_str = f"{t_inv_arg:.4f}" if t_inv_arg else "inf"
        print(f"# trajectory=constant t_inv={t_str}", file=sys.stderr)

    p['p_inv_func'] = p_inv_func
    sim = MsinvSimulator(**p)
    sim.run()


if __name__ == '__main__':
    main()
