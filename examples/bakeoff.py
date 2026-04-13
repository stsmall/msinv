#!/usr/bin/env python3
"""Three-way bake-off: msprime ground truth vs msinv-SMC (Option-3) vs
the hull simulator (feature/hull-algorithm).

Scenarios:
  1. Panmictic baseline           — all three; hull/SMC must agree with msprime.
  2. Single inv, no gene flux     — hull vs SMC.
  3. Single inv + gene flux       — hull vs SMC; check LD breakdown.
  4. Two non-overlapping inversions — hull vs SMC.
  5. Two-pop with ``ej`` merge    — all three.
  6. Small Kir/Fol mini           — hull vs SMC.

Output: per-scenario table of mean π, dxy, FST and ratios. Each
scenario uses identical parameters across simulators where applicable.
"""
import math
import time
import numpy as np

import msprime
import msinv
from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography as HullDemography


NREPS = 50
SEED_BASE = 1234


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

def per_site_pi_dxy(haps, idx_a, idx_b, region_len_bp):
    """Per-bp pi (within) or dxy (between) for haplotype matrix
    columns spanning ``region_len_bp`` total length."""
    if haps.shape[1] == 0 or region_len_bp <= 0:
        return 0.0
    d = 0
    npairs = 0
    for a in idx_a:
        for b in idx_b:
            if a == b:
                continue
            d += int((haps[a] != haps[b]).sum())
            npairs += 1
    return d / max(npairs, 1) / region_len_bp


def stats_from_treeseq(ts, n_S, n_I, mu, rng,
                        bp_left=None, bp_right=None, L=None):
    """Drop msprime mutations on a tskit TS, then compute per-site
    pi_S, pi_I, dxy_SI inside vs outside the inversion (or whole
    chromosome if no bp args)."""
    seed = int(rng.integers(1, 2**31))
    mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                  discrete_genome=False)
    G = mts.genotype_matrix()
    haps = G.T
    pos_bp = np.array([v.site.position for v in mts.variants()])
    S = list(range(n_S))
    I = list(range(n_S, n_S + n_I))
    L_total = ts.sequence_length if L is None else L

    if bp_left is None:
        return {
            'pi_S': per_site_pi_dxy(haps, S, S, L_total),
            'pi_I': per_site_pi_dxy(haps, I, I, L_total) if n_I else 0.0,
            'dxy_SI': per_site_pi_dxy(haps, S, I, L_total) if n_I else 0.0,
        }
    in_mask = (pos_bp >= bp_left) & (pos_bp < bp_right)
    out_mask = ~in_mask
    in_haps = haps[:, in_mask]
    out_haps = haps[:, out_mask]
    in_len = bp_right - bp_left
    out_len = L_total - in_len
    return {
        'in_pi_S': per_site_pi_dxy(in_haps, S, S, in_len),
        'in_pi_I': per_site_pi_dxy(in_haps, I, I, in_len) if n_I else 0.0,
        'in_dxy_SI': per_site_pi_dxy(in_haps, S, I, in_len) if n_I else 0.0,
        'out_pi_S': per_site_pi_dxy(out_haps, S, S, out_len),
        'out_pi_I': per_site_pi_dxy(out_haps, I, I, out_len) if n_I else 0.0,
        'out_dxy_SI': per_site_pi_dxy(out_haps, S, I, out_len) if n_I else 0.0,
    }


def stats_from_smc(pos_frac, haps, n_S, n_I, L,
                    bp_left_frac=None, bp_right_frac=None):
    """SMC simulator returns positions in [0, 1] and a haps matrix."""
    pos_arr = np.asarray(pos_frac, dtype=float)
    if pos_arr.size == 0:
        return {k: 0.0 for k in ['pi_S', 'pi_I', 'dxy_SI']}
    pos_bp = pos_arr * L if pos_arr.max() <= 1.0 + 1e-9 else pos_arr
    S = list(range(n_S))
    I = list(range(n_S, n_S + n_I))
    if bp_left_frac is None:
        return {
            'pi_S': per_site_pi_dxy(haps, S, S, L),
            'pi_I': per_site_pi_dxy(haps, I, I, L) if n_I else 0.0,
            'dxy_SI': per_site_pi_dxy(haps, S, I, L) if n_I else 0.0,
        }
    in_mask = (pos_bp >= bp_left_frac * L) & (pos_bp < bp_right_frac * L)
    out_mask = ~in_mask
    in_len = (bp_right_frac - bp_left_frac) * L
    out_len = L - in_len
    in_haps = haps[:, in_mask]
    out_haps = haps[:, out_mask]
    return {
        'in_pi_S': per_site_pi_dxy(in_haps, S, S, in_len),
        'in_pi_I': per_site_pi_dxy(in_haps, I, I, in_len) if n_I else 0.0,
        'in_dxy_SI': per_site_pi_dxy(in_haps, S, I, in_len) if n_I else 0.0,
        'out_pi_S': per_site_pi_dxy(out_haps, S, S, out_len),
        'out_pi_I': per_site_pi_dxy(out_haps, I, I, out_len) if n_I else 0.0,
        'out_dxy_SI': per_site_pi_dxy(out_haps, S, I, out_len) if n_I else 0.0,
    }


def avg(stat_list):
    if not stat_list:
        return {}
    keys = stat_list[0].keys()
    return {k: float(np.nanmean([s[k] for s in stat_list])) for k in keys}


def print_table(title, expected, results):
    """results: dict of label → dict of stats."""
    print(f"\n{title}")
    print('-' * 78)
    keys = sorted(set().union(*[r.keys() for r in results.values()]))
    labels = list(results.keys())
    header = f"  {'metric':<14}"
    for lab in labels:
        header += f" {lab:>14}"
    if expected:
        header += f" {'expected':>14}"
    print(header)
    for k in keys:
        line = f"  {k:<14}"
        for lab in labels:
            v = results[lab].get(k, float('nan'))
            line += f" {v:>14.6g}"
        if expected and k in expected:
            line += f" {expected[k]:>14.6g}"
        print(line)


# ---------------------------------------------------------------------------
# Scenario 1: Panmictic baseline (msprime, msinv-SMC, hull)
# ---------------------------------------------------------------------------

def scenario_panmictic_baseline():
    print("\n" + "=" * 78)
    print("SCENARIO 1: Panmictic baseline (no inversion)")
    print("=" * 78)
    Ne = 10_000
    mu = 1e-8
    r = 1e-8
    L = 50_000
    n = 10
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L

    expected = {
        'pi_S': 4 * Ne * mu,   # 4·Ne·µ per site
        'pi_I': 4 * Ne * mu,
        'dxy_SI': 4 * Ne * mu,
    }

    rng = np.random.default_rng(SEED_BASE)
    msp_stats = []; smc_stats = []; hull_stats = []
    for rep in range(NREPS):
        # msprime ground truth: ploidy=1 + population_size=2*Ne so the
        # haplotype coalescent rate matches the diploid Ne convention
        # used by SMC/hull (T_MRCA = 2·Ne for a pair).
        ts = msprime.sim_ancestry(
            samples=n, population_size=2 * Ne, sequence_length=L,
            recombination_rate=r, ploidy=1,
            random_seed=SEED_BASE + rep)
        msp_stats.append(stats_from_treeseq(ts, n, 0, mu, rng, L=L))

        # msinv SMC
        sim = msinv.MsinvSimulator(
            nsam=n, theta=theta, rho=rho, nsites=1000,
            n_std=n, n_inv=0, p_inv=0.0, c=0.0,
            seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n, 0, L))

        # hull
        sim = HullSimulator(
            samples=n, population_size=Ne, sequence_length=L,
            seed=SEED_BASE + rep)
        ts = sim.simulate()
        hull_stats.append(stats_from_treeseq(ts, n, 0, mu, rng, L=L))

    print_table('Per-site stats (mean over %d reps)' % NREPS, expected,
                {'msprime': avg(msp_stats),
                 'smc': avg(smc_stats),
                 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Scenario 2: Single inversion, no gene flux
# ---------------------------------------------------------------------------

def scenario_single_inv_noflux():
    print("\n" + "=" * 78)
    print("SCENARIO 2: Single inversion, γ=0 (hull vs SMC)")
    print("=" * 78)
    Ne = 10_000
    mu = 1e-8
    r = 1e-8
    L = 50_000
    n_S = 5; n_I = 5
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L
    bp_l_frac = 0.30; bp_r_frac = 0.70
    bp_l_g = bp_l_frac * L; bp_r_g = bp_r_frac * L
    p_inv = 0.5
    t_inv_gen = 80_000

    rng = np.random.default_rng(SEED_BASE + 1)
    smc_stats = []; hull_stats = []
    for rep in range(NREPS):
        sim = msinv.MsinvSimulator(
            nsam=n_S + n_I, theta=theta, rho=rho, nsites=1000,
            n_std=n_S, n_inv=n_I,
            p_inv=p_inv, c=0.0, gamma=0.0,
            bp_left=bp_l_frac, bp_right=bp_r_frac,
            t_inv=t_inv_gen / (2 * Ne),
            seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n_S, n_I, L,
                                          bp_l_frac, bp_r_frac))

        sim = HullSimulator(
            n_std=n_S, n_inv=n_I,
            population_size=Ne, sequence_length=L,
            p_inv=p_inv, t_inv=t_inv_gen,
            bp_left=bp_l_g, bp_right=bp_r_g,
            seed=SEED_BASE + rep)
        ts = sim.simulate()
        hull_stats.append(stats_from_treeseq(ts, n_S, n_I, mu, rng,
                                              bp_l_g, bp_r_g, L))

    expected = {
        'in_pi_S': 4 * Ne * (1 - p_inv) * mu,
        'in_pi_I': 4 * Ne * p_inv * mu,
        'in_dxy_SI': 2 * (t_inv_gen + 2 * Ne) * mu,
        'out_pi_S': 4 * Ne * mu,
        'out_pi_I': 4 * Ne * mu,
        'out_dxy_SI': 4 * Ne * mu,
    }
    print_table('Per-site stats (50 reps)', expected,
                {'smc': avg(smc_stats), 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Scenario 3: Single inv + gene flux
# ---------------------------------------------------------------------------

def scenario_single_inv_with_flux():
    print("\n" + "=" * 78)
    print("SCENARIO 3: Single inversion + gene flux (γ > 0)")
    print("=" * 78)
    Ne = 10_000
    mu = 1e-8
    r = 1e-8
    L = 50_000
    g = 5e-8   # higher than mu — strong flux
    n_S = 5; n_I = 5
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L
    gamma = 4 * Ne * g * L
    bp_l_frac = 0.30; bp_r_frac = 0.70
    bp_l_g = bp_l_frac * L; bp_r_g = bp_r_frac * L
    p_inv = 0.5
    t_inv_gen = 80_000

    rng = np.random.default_rng(SEED_BASE + 2)
    smc_stats = []; hull_stats = []
    for rep in range(NREPS):
        sim = msinv.MsinvSimulator(
            nsam=n_S + n_I, theta=theta, rho=rho, nsites=1000,
            n_std=n_S, n_inv=n_I,
            p_inv=p_inv, c=0.0, gamma=gamma,
            bp_left=bp_l_frac, bp_right=bp_r_frac,
            t_inv=t_inv_gen / (2 * Ne),
            seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n_S, n_I, L,
                                          bp_l_frac, bp_r_frac))

        sim = HullSimulator(
            n_std=n_S, n_inv=n_I,
            population_size=Ne, sequence_length=L,
            p_inv=p_inv, t_inv=t_inv_gen,
            bp_left=bp_l_g, bp_right=bp_r_g,
            gene_conversion_rate=g,
            seed=SEED_BASE + rep)
        ts = sim.simulate()
        hull_stats.append(stats_from_treeseq(ts, n_S, n_I, mu, rng,
                                              bp_l_g, bp_r_g, L))

    print_table('Per-site stats (50 reps; γ-driven mixing)', None,
                {'smc': avg(smc_stats), 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Scenario 4: Two inversions
# ---------------------------------------------------------------------------

def scenario_two_inversions():
    print("\n" + "=" * 78)
    print("SCENARIO 4: Two non-overlapping inversions")
    print("=" * 78)
    Ne = 10_000
    mu = 1e-8
    r = 1e-8
    L = 100_000
    n_S = 5; n_I = 5
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L
    inv0 = (0.15, 0.45); inv1 = (0.55, 0.85)
    p_inv = 0.5
    t_inv_gen = 80_000

    inv0_g = (inv0[0] * L, inv0[1] * L)
    inv1_g = (inv1[0] * L, inv1[1] * L)

    rng = np.random.default_rng(SEED_BASE + 3)
    smc_stats = []; hull_stats = []
    for rep in range(NREPS):
        # SMC multi-inv via inversions list
        invs = [
            msinv.InversionSpec(inv0[0], inv0[1], p_inv=p_inv,
                                 c=0.0, gamma=0.0,
                                 t_inv=t_inv_gen / (2 * Ne)),
            msinv.InversionSpec(inv1[0], inv1[1], p_inv=p_inv,
                                 c=0.0, gamma=0.0,
                                 t_inv=t_inv_gen / (2 * Ne)),
        ]
        sim = msinv.MsinvSimulator(
            nsam=n_S + n_I, theta=theta, rho=rho, nsites=1000,
            n_std=n_S, n_inv=n_I,
            inversions=invs, seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n_S, n_I, L,
                                          inv0[0], inv1[1]))

        # Hull multi-inv
        sim = HullSimulator(
            n_std=n_S, n_inv=n_I,
            population_size=Ne, sequence_length=L,
            inversions=[
                InversionSpec(bp_left=inv0_g[0], bp_right=inv0_g[1],
                               p_inv=p_inv, t_inv=t_inv_gen),
                InversionSpec(bp_left=inv1_g[0], bp_right=inv1_g[1],
                               p_inv=p_inv, t_inv=t_inv_gen),
            ],
            seed=SEED_BASE + rep)
        ts = sim.simulate()
        # Stats over inv0 union inv1 (call that "in")
        hull_stats.append(stats_from_treeseq(ts, n_S, n_I, mu, rng,
                                              inv0_g[0], inv1_g[1], L))

    print_table('Two-inv stats (50 reps)', None,
                {'smc': avg(smc_stats), 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Scenario 5: Two-pop with split (msprime, hull, SMC)
# ---------------------------------------------------------------------------

def scenario_two_pop_split():
    print("\n" + "=" * 78)
    print("SCENARIO 5: Two-pop with ej merge (no inversion)")
    print("=" * 78)
    Ne = 10_000
    mu = 1e-8
    r = 1e-8
    L = 50_000
    n_each = 5
    t_split_gen = 4 * Ne   # 40k gen
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L

    rng = np.random.default_rng(SEED_BASE + 4)
    msp_stats = []; smc_stats = []; hull_stats = []
    for rep in range(NREPS):
        # msprime — ploidy=1, pop sizes 2*Ne so haploid coalescent
        # matches the diploid Ne used by SMC/hull.
        demo = msprime.Demography()
        demo.add_population(name='A', initial_size=2 * Ne)
        demo.add_population(name='B', initial_size=2 * Ne)
        demo.add_population(name='Anc', initial_size=2 * Ne)
        demo.add_population_split(time=t_split_gen, derived=['A', 'B'],
                                   ancestral='Anc')
        ts = msprime.sim_ancestry(
            samples={'A': n_each, 'B': n_each},
            demography=demo, sequence_length=L,
            recombination_rate=r, ploidy=1,
            random_seed=SEED_BASE + rep)
        msp_stats.append(stats_from_treeseq(ts, n_each, n_each, mu, rng, L=L))

        # SMC
        smc_demo = msinv.Demography(n_pops=2)
        smc_demo.add_event(('ej', t_split_gen / (2 * Ne), 1, 0))
        sim = msinv.MsinvSimulator(
            nsam=2 * n_each, theta=theta, rho=rho, nsites=1000,
            n_std=2 * n_each, n_inv=0, p_inv=0.0, c=0.0,
            n_pops=2,
            sample_config={('S', 0): n_each, ('S', 1): n_each},
            demography=smc_demo, seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n_each, n_each, L))

        # Hull
        hull_demo = HullDemography(pop_sizes=[Ne, Ne])
        hull_demo.add_event(('ej', t_split_gen, 1, 0))
        sim = HullSimulator(
            sample_config={(None, 0): n_each, (None, 1): n_each},
            demography=hull_demo,
            sequence_length=L, seed=SEED_BASE + rep)
        ts = sim.simulate()
        hull_stats.append(stats_from_treeseq(ts, n_each, n_each, mu, rng, L=L))

    expected = {
        'pi_S': 4 * Ne * mu,
        'pi_I': 4 * Ne * mu,
        'dxy_SI': 2 * (t_split_gen + 2 * Ne) * mu,
    }
    print_table('Per-site stats (msprime is ground truth)', expected,
                {'msprime': avg(msp_stats),
                 'smc': avg(smc_stats),
                 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Scenario 6: Small Kir/Fol mini (hull vs SMC)
# ---------------------------------------------------------------------------

def scenario_kir_fol_mini():
    print("\n" + "=" * 78)
    print("SCENARIO 6: Small Kir/Fol mini (constant Ne both pops)")
    print("=" * 78)
    Ne = 10_000   # constant — avoids the structured-coal-with-huge-Ne artifact
    mu = 1e-8
    r = 1e-8
    L = 50_000
    n_K = 4; n_FS = 2; n_FI = 2
    theta = 4 * Ne * mu * L
    rho = 4 * Ne * r * L
    bp_frac = (0.30, 0.70)
    bp_g = (bp_frac[0] * L, bp_frac[1] * L)
    p_inv = 0.5
    t_split_gen = 4_000     # recent-ish
    t_inv_gen = 40_000      # deep

    rng = np.random.default_rng(SEED_BASE + 5)
    smc_stats = []; hull_stats = []

    class Traj:
        t_inv = t_inv_gen / (2 * Ne)
        def __init__(self): self.n_pops = 2
        def __call__(self, t, pop=0):
            if t >= self.t_inv: return 0.0
            if t >= t_split_gen / (2 * Ne): return p_inv
            return 0.0 if pop == 0 else p_inv

    for rep in range(NREPS):
        traj = Traj()
        smc_demo = msinv.Demography(n_pops=2)
        smc_demo.add_event(('ej', t_split_gen / (2 * Ne), 1, 0))
        sim = msinv.MsinvSimulator(
            nsam=n_K + n_FS + n_FI, theta=theta, rho=rho, nsites=1000,
            n_std=n_K + n_FS, n_inv=n_FI,
            p_inv=p_inv, c=0.0, gamma=0.0,
            bp_left=bp_frac[0], bp_right=bp_frac[1],
            p_inv_func=traj, n_pops=2,
            sample_config={('S', 0): n_K, ('S', 1): n_FS, ('I', 1): n_FI},
            demography=smc_demo,
            t_inv=t_inv_gen / (2 * Ne),
            seed=SEED_BASE + rep)
        pos, haps = sim.simulate_one()
        smc_stats.append(stats_from_smc(pos, haps, n_K + n_FS, n_FI, L,
                                          bp_frac[0], bp_frac[1]))

        # Hull (linked karyotype across the single inv)
        hull_demo = HullDemography(pop_sizes=[Ne, Ne])
        hull_demo.add_event(('ej', t_split_gen, 1, 0))
        sim = HullSimulator(
            sample_config={('S', 0): n_K, ('S', 1): n_FS, ('I', 1): n_FI},
            demography=hull_demo,
            sequence_length=L,
            p_inv=p_inv, t_inv=t_inv_gen,
            bp_left=bp_g[0], bp_right=bp_g[1],
            seed=SEED_BASE + rep)
        ts = sim.simulate()
        hull_stats.append(stats_from_treeseq(ts, n_K + n_FS, n_FI, mu, rng,
                                              bp_g[0], bp_g[1], L))

    print_table('Kir/Fol mini, constant Ne (50 reps)', None,
                {'smc': avg(smc_stats), 'hull': avg(hull_stats)})


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    scenario_panmictic_baseline()
    scenario_single_inv_noflux()
    scenario_single_inv_with_flux()
    scenario_two_inversions()
    scenario_two_pop_split()
    scenario_kir_fol_mini()
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
