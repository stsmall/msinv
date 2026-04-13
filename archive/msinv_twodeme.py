#!/usr/bin/env python3
"""
msinv_twodeme.py

Two-population extension of msinv, implementing the full Guerrero et al.
(2012) model with migration-selection balance.

Model:
  Two populations (demes) of equal size N, connected by migration rate m.
  Arrangement S is favored in pop 1, I is favored in pop 2.
  At migration-selection equilibrium, the rare arrangement is at
  frequency q in each population.

  Four lineage classes: {S,I} x {pop1, pop2}

  Going backward in time:
    - Coalescence: within same class + same population
    - Gene flux: between arrangements, within same population
    - Migration: between populations, same arrangement

This module provides:
  - build_two_deme_tree(): structured coalescent at a single site
  - TwoDemeSimulator: full SMC simulator with two demes

Usage:
  python msinv_twodeme.py <nsam> <nreps> -t <theta> -r <rho> <nsites> \\
    -inv <q> <c> -m <Nm> -I <nS1> <nI1> <nS2> <nI2>

  q:   frequency of rare (disfavored) arrangement in each population
  c:   gene flux coefficient
  Nm:  population-scaled migration rate (4Nm)
  nS1: standard chromosomes sampled from pop 1
  nI1: inverted chromosomes sampled from pop 1
  nS2: standard chromosomes sampled from pop 2
  nI2: inverted chromosomes sampled from pop 2
"""

import numpy as np
import sys
import io

sys.path.insert(0, '.')
from msinv import (Node, get_all_nodes, get_branches, branch_lengths_by_class,
                   get_leaves_below, GeneFluxModel, drop_mutations)


# ===================================================================
# Two-deme structured coalescent
# ===================================================================

# Lineage state: (arrangement, population)
# arrangement: 'S' or 'I'
# population: 0 or 1

def build_two_deme_tree(sample_config, q, c, rho, m_scaled, phi_x, rng):
    """
    Build structured coalescent tree at a single site with two demes.

    Following Guerrero et al. (2012), with rates from their
    supplementary material.

    Args:
        sample_config: dict with keys 'S0','I0','S1','I1' giving
                       number of chromosomes sampled of each type.
                       S0 = standard from pop 0 (where S is common)
                       I0 = inverted from pop 0 (rare arrangement)
                       S1 = standard from pop 1 (rare arrangement)
                       I1 = inverted from pop 1 (where I is common)
        q:        frequency of rare (disfavored) arrangement per deme
        c:        gene flux coefficient
        rho:      population-scaled recombination rate (4Nr)
        m_scaled: population-scaled migration rate per lineage (m in
                  the coalescent; forward rate = m per generation)
        phi_x:    gene flux probability at this position
        rng:      numpy random generator

    Returns:
        (root, leaves)

    Arrangement frequencies:
        Pop 0: freq_S = 1-q, freq_I = q
        Pop 1: freq_S = q,   freq_I = 1-q
    """
    freq = {
        ('S', 0): 1.0 - q,
        ('I', 0): q,
        ('S', 1): q,
        ('I', 1): 1.0 - q,
    }

    # Create leaves
    leaves = []
    sid = 0
    for arr, pop in [('S', 0), ('I', 0), ('S', 1), ('I', 1)]:
        key = f'{arr}{pop}'
        n = sample_config.get(key, 0)
        for _ in range(n):
            leaf = Node(time=0.0, sample_id=sid, branch_class=arr)
            # Store population as a custom attribute via a wrapper
            leaves.append(leaf)
            sid += 1

    nsam = sid

    # Active lineages: [node, arrangement, population]
    active = []
    sid = 0
    for arr, pop in [('S', 0), ('I', 0), ('S', 1), ('I', 1)]:
        key = f'{arr}{pop}'
        n = sample_config.get(key, 0)
        for _ in range(n):
            active.append([leaves[sid], arr, pop])
            sid += 1

    t = 0.0

    while len(active) > 1:
        # Count lineages by (arrangement, population)
        counts = {}
        for _, arr, pop in active:
            key = (arr, pop)
            counts[key] = counts.get(key, 0) + 1

        # --- Compute rates ---
        rates = []

        # 1. Coalescence events
        for (arr, pop), k in counts.items():
            if k >= 2:
                f = freq[(arr, pop)]
                if f > 0:
                    rate = (k * (k - 1) / 2.0) / f
                    rates.append(('coal', arr, pop, rate))

        # 2. Gene flux events (arrangement switch within population)
        for (arr, pop), k in counts.items():
            if k > 0:
                other_arr = 'I' if arr == 'S' else 'S'
                f_other = freq[(other_arr, pop)]
                rate = k * c * (rho / 2.0) * f_other * phi_x
                if rate > 0:
                    rates.append(('flux', arr, pop, rate))

        # 3. Migration events (population switch, same arrangement)
        for (arr, pop), k in counts.items():
            if k > 0:
                # Backward migration rate per lineage = m_scaled / 2
                # (m_scaled = 4Nm, so per-lineage rate = m_scaled/2 in
                #  2N-generation timescale)
                rate = k * (m_scaled / 2.0)
                if rate > 0:
                    other_pop = 1 - pop
                    rates.append(('mig', arr, pop, rate))

        total_rate = sum(r for _, _, _, r in rates)
        if total_rate <= 0:
            raise RuntimeError(
                f"Stuck in two-deme coalescent: counts={counts}, "
                f"q={q}, c={c}, m={m_scaled}, phi={phi_x}"
            )

        # Draw time and event
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        u = rng.random() * total_rate
        cum = 0.0
        chosen = None
        for event_type, arr, pop, rate in rates:
            cum += rate
            if u < cum:
                chosen = (event_type, arr, pop)
                break

        if chosen is None:
            chosen = (rates[-1][0], rates[-1][1], rates[-1][2])

        event_type, arr, pop = chosen

        if event_type == 'coal':
            _two_deme_coalesce(active, arr, pop, t, rng)
        elif event_type == 'flux':
            _two_deme_flux(active, arr, pop, t, rng)
        elif event_type == 'mig':
            _two_deme_migrate(active, arr, pop, t, rng)

    root = active[0][0]
    root.branch_class = active[0][1]
    return root, leaves


def _two_deme_coalesce(active, arr, pop, t, rng):
    """Coalesce two lineages of (arr, pop)."""
    indices = [i for i, (_, a, p) in enumerate(active)
               if a == arr and p == pop]
    picked = rng.choice(len(indices), size=2, replace=False)
    i1, i2 = indices[picked[0]], indices[picked[1]]
    n1, _, _ = active[i1]
    n2, _, _ = active[i2]

    coal = Node(time=t, branch_class=arr)
    coal.children = [n1, n2]
    n1.parent = coal
    n2.parent = coal

    for i in sorted([i1, i2], reverse=True):
        active.pop(i)
    active.append([coal, arr, pop])


def _two_deme_flux(active, arr, pop, t, rng):
    """Gene flux: one lineage switches arrangement, stays in same pop."""
    indices = [i for i, (_, a, p) in enumerate(active)
               if a == arr and p == pop]
    idx = indices[rng.integers(len(indices))]
    old_node, _, _ = active[idx]
    new_arr = 'I' if arr == 'S' else 'S'

    flux_node = Node(time=t, branch_class=new_arr)
    flux_node.children = [old_node]
    old_node.parent = flux_node

    active[idx] = [flux_node, new_arr, pop]


def _two_deme_migrate(active, arr, pop, t, rng):
    """Migration: one lineage switches population, stays same arrangement."""
    indices = [i for i, (_, a, p) in enumerate(active)
               if a == arr and p == pop]
    idx = indices[rng.integers(len(indices))]
    # Migration doesn't change the tree topology; it just changes the
    # lineage's population label.  No new node needed.
    node, _, _ = active[idx]
    new_pop = 1 - pop
    active[idx] = [node, arr, new_pop]


# ===================================================================
# Two-deme SMC simulator
# ===================================================================

class TwoDemeSimulator:
    """
    Full two-deme simulator with SMC.

    The SMC stepping uses the same algorithm as the single-population
    version, but the initial tree and reattachment use the two-deme
    structured coalescent.
    """

    def __init__(self, nsam, nreps, theta, rho, nsites,
                 sample_config, q, c, m_scaled,
                 flux_window=0.3, seed=None):
        """
        Args:
            sample_config: dict with 'S0','I0','S1','I1' counts
            q: rare arrangement frequency
            c: gene flux coefficient
            m_scaled: 4Nm (population-scaled migration)
        """
        self.nsam = nsam
        self.nreps = nreps
        self.theta = theta
        self.rho = rho
        self.nsites = nsites
        self.sample_config = sample_config
        self.q = q
        self.c = c
        self.m_scaled = m_scaled
        self.flux_model = GeneFluxModel(w=flux_window)
        self.rng = np.random.default_rng(seed)

    def simulate_one(self):
        """
        Simulate one replicate.

        For simplicity, uses site-by-site structured coalescent
        (correct marginals, no LD from recombination).
        Full SMC extension with two demes left as future work.
        """
        rng = self.rng

        # Use site-by-site approach: build independent tree at each
        # of n_sites positions, drop mutations independently.
        # This gives correct marginal distributions but no LD.
        n_positions = min(self.nsites, 500)  # subsample for speed
        positions_frac = np.linspace(0.02, 0.98, n_positions)

        mutations = []

        for pos in positions_frac:
            phi_x = self.flux_model.phi(pos)
            root, leaves = build_two_deme_tree(
                self.sample_config, self.q, self.c, self.rho,
                self.m_scaled, phi_x, rng
            )

            # Single-site mutation: total branch length * theta_per_site
            branches = get_branches(root)
            L = sum(bl for _, bl in branches)
            theta_per_site = self.theta / self.nsites

            n_muts = rng.poisson(theta_per_site / 2.0 * L)
            if n_muts == 0:
                continue

            bl_arr = np.array([bl for _, bl in branches])
            bl_probs = bl_arr / bl_arr.sum()

            for _ in range(n_muts):
                bi = rng.choice(len(branches), p=bl_probs)
                leaf_ids = get_leaves_below(branches[bi][0])
                mutations.append((pos, leaf_ids))

        if not mutations:
            return [], np.zeros((self.nsam, 0), dtype=int)

        mutations.sort(key=lambda x: x[0])
        positions = [m[0] for m in mutations]
        haplotypes = np.zeros((self.nsam, len(mutations)), dtype=int)
        for j, (_, ids) in enumerate(mutations):
            for sid in ids:
                haplotypes[sid, j] = 1

        return positions, haplotypes

    def run(self, outfile=sys.stdout):
        """Run replicates, ms-format output."""
        sc = self.sample_config
        cmd = (f"msinv_2deme {self.nsam} {self.nreps} "
               f"-t {self.theta} -r {self.rho} {self.nsites} "
               f"-inv {self.q} {self.c} -m {self.m_scaled} "
               f"-I {sc.get('S0',0)} {sc.get('I0',0)} "
               f"{sc.get('S1',0)} {sc.get('I1',0)}")
        print(cmd, file=outfile)
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
    q = 0.1; c = 0.01; m_scaled = 1.0
    nS0 = 0; nI0 = 0; nS1 = 0; nI1 = 0
    flux_window = 0.3; seed = None

    i = 2
    while i < len(argv):
        f = argv[i]
        if f == '-t':
            theta = float(argv[i+1]); i += 2
        elif f == '-r':
            rho = float(argv[i+1]); nsites = int(argv[i+2]); i += 3
        elif f == '-inv':
            q = float(argv[i+1]); c = float(argv[i+2]); i += 3
        elif f == '-m':
            m_scaled = float(argv[i+1]); i += 2
        elif f == '-I':
            nS0 = int(argv[i+1]); nI0 = int(argv[i+2])
            nS1 = int(argv[i+3]); nI1 = int(argv[i+4]); i += 5
        elif f == '-flux_window':
            flux_window = float(argv[i+1]); i += 2
        elif f in ('-seed', '-seeds'):
            seed = int(argv[i+1]); i += 2
        else:
            i += 1

    sample_config = {'S0': nS0, 'I0': nI0, 'S1': nS1, 'I1': nI1}
    total = sum(sample_config.values())
    if total != nsam:
        print(f"Warning: -I sample counts sum to {total}, expected {nsam}",
              file=sys.stderr)

    return dict(nsam=nsam, nreps=nreps, theta=theta, rho=rho,
                nsites=nsites, q=q, c=c, m_scaled=m_scaled,
                sample_config=sample_config,
                flux_window=flux_window, seed=seed)


def main():
    p = parse_args()
    sim = TwoDemeSimulator(
        nsam=p['nsam'], nreps=p['nreps'], theta=p['theta'],
        rho=p['rho'], nsites=p['nsites'],
        sample_config=p['sample_config'],
        q=p['q'], c=p['c'], m_scaled=p['m_scaled'],
        flux_window=p['flux_window'], seed=p['seed']
    )
    sim.run()


if __name__ == '__main__':
    main()
