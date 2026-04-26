#!/usr/bin/env python3
"""
msprime demography for the K - F - Mozambique system, with a hugely
expanding Ghost continental metapopulation donating rare alleles to F.

Base = your KF_msprime_demography.txt (5 pops: K, F, Moz, KF, Anc;
step Ne changes; pulse migrations).  Soften: KF deep bottleneck floor
lifted from 12-27k to 50k across 40-70k g (the closest pi/Fst match).

Overlay: Ghost continental funestus pop, modeled with the same kind of
huge recent expansion that Ag1000g (Miles 2017) and Af1000g (Bodde
et al.) find for natural metapopulations:
  - Ghost(t=0)        = 10 M
  - Ghost(t=87,163 g) =  100 k  (size at the Moz-merge time)
  - Forward growth = ~100x over ~87 k gens
Migration is one-way Ghost -> F at m=1e-5 per gen, donating
expansion-derived rare alleles into Folonzo.  K stays isolated
(incipient speciation), receiving rare alleles only via the existing
ABC pulse migrations from F.

Sources
-------
* KF.mig_prop.model.out      : ABC pulse migrations
* KF.demography.model.out    : ABC + Stairway-plot Ne(t)
* Bodde et al. 2024          : An. funestus continental pi ~1.4-1.7%
* Miles et al. 2017          : Ag1000g BFM/BFS sym_mig: ~30x expansion

Run
---
    python examples/kir_fol_demography.py --debug
    python examples/kir_fol_demography.py --out .tmp/kf.trees
"""

import argparse
import math
import sys

import msprime


# Mutation and recombination rates (Anopheles, paper-rounded values).
MU  = 6.02e-9
RHO = 1.0e-8


def build_demography():
    d = msprime.Demography()

    # ---- Populations (initial = present-day) -------------------------
    # === LOCKED v11 demography ===
    # K: ABC step trajectory with t=200 dip dropped (median values).
    # F: clean exp growth from split-time 158,711 -> present 2,496,632 (~16x).
    # Anc/Moz/KF: original 5-pop structure with KF deep bottleneck floor
    # softened to 50k (otherwise pi crashes).
    # Ghost: continental metapop with huge Ag1000g/Af1000g-style expansion
    # (100k @ 87k g -> 10M today), donating rare alleles via dual-epoch
    # migration (Ghost->F post-split, Ghost->KF pre-split).
    d.add_population(name="K", initial_size=126_772)
    F_NE_PRESENT  = 2_496_632
    F_NE_AT_SPLIT =   158_711
    g_F = math.log(F_NE_PRESENT / F_NE_AT_SPLIT) / 9_194
    d.add_population(name="F",   initial_size=F_NE_PRESENT, growth_rate=g_F)
    d.add_population(name="Moz", initial_size=400_000)
    d.add_population(name="KF",  initial_size= 86_000)
    d.add_population(name="Anc", initial_size=450_000)

    # Ghost: continental funestus metapop with huge Ag1000g/Af1000g-style
    # expansion (100k @ 87k g -> 10M today, ~100x).  No DDT crash on
    # ghost (continental DDT impact diluted in metapop).
    GHOST_NE_PRESENT      = 10_000_000
    GHOST_NE_AT_MOZ_SPLIT =    100_000
    T_GHOST_MERGE         =     87_163
    g_ghost = math.log(GHOST_NE_PRESENT / GHOST_NE_AT_MOZ_SPLIT) / T_GHOST_MERGE
    d.add_population(name="Ghost", initial_size=GHOST_NE_PRESENT, growth_rate=g_ghost)

    # ---- K Ne(t): ABC steps (t=200 dip dropped per v11) -------------
    d.add_population_parameters_change(time=  400, population="K", initial_size=161_546)
    d.add_population_parameters_change(time=  600, population="K", initial_size=152_453)
    d.add_population_parameters_change(time=1_400, population="K", initial_size=174_800)
    d.add_population_parameters_change(time=3_000, population="K", initial_size=182_180)
    d.add_population_parameters_change(time=6_200, population="K", initial_size=159_861)
    # ---- F Ne(t): no step events; F follows pure exp growth ---------
    # (set at population creation with growth_rate g_F)

    # ---- Migration topology by epoch --------------------------------
    # Post-split (0 -> 9194 g): Ghost -> F only (F is connected to
    #     continental metapop), and weak F -> K (post-split gene flow).
    # Pre-split (9194 -> 87163 g): Ghost -> KF (the merged BF ancestor).
    # All rates 1e-5 magnitude per user guidance.
    # msprime backward direction: src -> dst means src lineages migrate
    # backward into dst, equivalent to forward dst -> src.
    M = 1e-5
    # Ghost -> F (forward) = F -> Ghost (backward).  Active 0..9194 g
    # (turned off when F merges into KF).
    d.set_migration_rate(source="F", dest="Ghost", rate=M)
    # F -> K (forward) = K -> F (backward).  Active 0..9194 g.
    d.set_migration_rate(source="K", dest="F", rate=M)
    # Ghost -> KF (forward) = KF -> Ghost (backward).  Active 9194..87163 g.
    # Set initially to 0; switched on at the K-F split.
    # (msprime applies a global rate; we toggle via add_migration_rate_change.)
    d.add_migration_rate_change(time=9_194, source="KF", dest="Ghost", rate=M)
    # Turn off at KF-Moz split (Ghost merges into Anc anyway).
    d.add_migration_rate_change(time=87_163, source="KF", dest="Ghost", rate=0)

    # ---- Pulse migrations from KF.mig_prop.model.out ----------------
    d.add_mass_migration(time=  710, source="K", dest="F", proportion=1.78e-4)
    d.add_mass_migration(time=4_027, source="K", dest="F", proportion=3.80e-5)
    d.add_mass_migration(time=6_081, source="K", dest="F", proportion=7.35e-5)

    # ---- K - F split at 9,194 g -> KF -------------------------------
    d.add_population_split(time=9_194, derived=["K", "F"], ancestral="KF")

    # ---- KF Ne(t)  (ABC trajectory; bottleneck floor lifted to 50k) -
    d.add_population_parameters_change(time=13_000, population="KF", initial_size=81_072)
    d.add_population_parameters_change(time=20_000, population="KF", initial_size=95_546)
    d.add_population_parameters_change(time=30_000, population="KF", initial_size=73_250)
    d.add_population_parameters_change(time=40_000, population="KF", initial_size=50_000)  # softened
    d.add_population_parameters_change(time=50_000, population="KF", initial_size=50_000)  # softened
    d.add_population_parameters_change(time=60_000, population="KF", initial_size=50_000)  # softened
    d.add_population_parameters_change(time=70_000, population="KF", initial_size=50_000)  # softened

    # ---- KF - Moz split at 87,163 g -> Anc; Ghost also joins Anc ----
    d.add_population_split(time=T_GHOST_MERGE, derived=["KF", "Moz"], ancestral="Anc")
    d.add_population_split(time=T_GHOST_MERGE, derived=["Ghost"],     ancestral="Anc")

    d.sort_events()
    return d


# ----------------------------------------------------------------------
# Simulation wrapper
# ----------------------------------------------------------------------
def simulate(
    seed: int = 1,
    n_K: int = 74,
    n_F: int = 92,
    sequence_length: float = 1e6,
    recombination_rate: float = RHO,
    mutation_rate: float = MU,
    ploidy: int = 2,
):
    """Run one msprime replicate and return a mutated TreeSequence."""
    dem = build_demography()
    ts = msprime.sim_ancestry(
        samples={"K": n_K, "F": n_F},
        demography=dem,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        random_seed=seed,
        model="hudson",
    )
    if mutation_rate > 0:
        ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed + 1)
    return ts


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="K-F msprime demography (v2 + expanding Ghost)")
    p.add_argument("--debug",  action="store_true",
                   help="Print DemographyDebugger and exit.")
    p.add_argument("--seed",   type=int, default=1)
    p.add_argument("--length", type=float, default=1e6)
    p.add_argument("--rho",    type=float, default=RHO)
    p.add_argument("--mu",     type=float, default=MU)
    p.add_argument("--n-K",    type=int, default=74)
    p.add_argument("--n-F",    type=int, default=92)
    p.add_argument("--out",    type=str, default=None,
                   help="Optional .trees output path (use .tmp/ for scratch).")
    args = p.parse_args(argv)

    dem = build_demography()
    if args.debug:
        print(dem.debug())
        return 0

    ts = simulate(
        seed=args.seed,
        n_K=args.n_K, n_F=args.n_F,
        sequence_length=args.length,
        recombination_rate=args.rho,
        mutation_rate=args.mu,
    )
    K_nodes = ts.samples(population=dem["K"].id)
    F_nodes = ts.samples(population=dem["F"].id)
    pi_K = ts.diversity(sample_sets=[K_nodes])
    pi_F = ts.diversity(sample_sets=[F_nodes])
    dxy  = ts.divergence(sample_sets=[K_nodes, F_nodes], indexes=[(0, 1)])
    fst  = ts.Fst(sample_sets=[K_nodes, F_nodes], indexes=[(0, 1)])
    print(f"trees={ts.num_trees} sites={ts.num_sites} muts={ts.num_mutations}")
    print(f"pi_K={pi_K:.5f} pi_F={pi_F:.5f} dxy={dxy[0]:.5f} Fst={fst[0]:.4f}")
    if args.out:
        ts.dump(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
