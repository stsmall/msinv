#!/usr/bin/env python3
"""Compare the SMC simulator (msinv.simulator.MsinvSimulator) and the
hull simulator (msinv.hull.HullSimulator) on simple shared scenarios.

Both simulators are run with identical parameters. To make outputs
directly comparable, we drop msprime mutations on the hull's
TreeSequence and compute pi/dxy the same way as the SMC simulator.
We also compare T_MRCA directly (hull has it natively; SMC's pi → T
via pi = 2·T·µ).

Intentionally SIMPLE scenarios — no Kir/Fol-scale runs.
Goal: cheap sanity check that hull's headline numbers track the SMC
simulator on plain-vanilla cases, and that hull's class-barrier
property holds (which the SMC simulator's earlier broken-then-Option3
versions struggled with).
"""
import math
import time
import numpy as np

import msinv
from msinv.hull import HullSimulator
from msinv.hull.demography import Demography


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
Ne = 10_000
mu = 1e-8
r = 1e-8
L = 50_000
theta = 4 * Ne * mu * L
rho = 4 * Ne * r * L

n_S = 5
n_I = 5
nsam = n_S + n_I

# Phase 5 fix landed: hull's class is now per-segment, so collinear
# flanks (positions outside the inversion) get panmictic treatment.
# Use a sub-region inversion to verify both simulators agree on both
# inside-inv and outside-inv stats.
bp_left = 0.30
bp_right = 0.70
inv_len_bp = (bp_right - bp_left) * L

p_inv = 0.5
t_inv_gen = 4.0 * 2 * Ne   # = 80,000 generations

NREPS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mrca_time(tree, a, b):
    return tree.time(tree.mrca(a, b))


def stats_from_treeseq(ts, n_S, n_I, bp_left, bp_right, L):
    """Compute T_MRCA-based stats from a tskit TreeSequence.

    Returns dict with mean (within-S, within-I, cross-S-I) T_MRCA both
    INSIDE and OUTSIDE the inversion (in genomic-position weighted
    means).
    """
    samples = list(ts.samples())
    S = samples[:n_S]
    I = samples[n_S:]
    bp_l_g = bp_left * L
    bp_r_g = bp_right * L

    in_ss = []; in_ii = []; in_si = []
    out_ss = []; out_ii = []; out_si = []
    for tree in ts.trees():
        l, rt = tree.interval.left, tree.interval.right
        in_w = max(0.0, min(rt, bp_r_g) - max(l, bp_l_g))
        out_w = (rt - l) - in_w
        if in_w == 0 and out_w == 0:
            continue
        # SS
        ss = []
        for i in range(n_S):
            for j in range(i + 1, n_S):
                ss.append(_mrca_time(tree, S[i], S[j]))
        # II
        ii = []
        for i in range(n_I):
            for j in range(i + 1, n_I):
                ii.append(_mrca_time(tree, I[i], I[j]))
        # SI
        si = []
        for s in S:
            for i in I:
                si.append(_mrca_time(tree, s, i))
        if in_w > 0:
            in_ss.extend([(t, in_w) for t in ss])
            in_ii.extend([(t, in_w) for t in ii])
            in_si.extend([(t, in_w) for t in si])
        if out_w > 0:
            out_ss.extend([(t, out_w) for t in ss])
            out_ii.extend([(t, out_w) for t in ii])
            out_si.extend([(t, out_w) for t in si])

    def _wmean(pairs):
        if not pairs:
            return float('nan')
        ts_, ws = zip(*pairs)
        ws = np.asarray(ws); ts_ = np.asarray(ts_)
        return float(np.sum(ws * ts_) / np.sum(ws))

    return dict(
        in_T_SS=_wmean(in_ss), in_T_II=_wmean(in_ii), in_T_SI=_wmean(in_si),
        out_T_SS=_wmean(out_ss), out_T_II=_wmean(out_ii),
        out_T_SI=_wmean(out_si),
        min_in_T_SI=min((t for t, _ in in_si), default=float('nan')),
    )


def stats_per_site(pos_bp, haps, n_S, n_I, bp_l_g, bp_r_g):
    """Per-bp pi within S, within I, and dxy between S and I, inside vs
    outside the inversion. ``pos_bp`` are SNP positions in genomic
    coordinates; we compute pi as (sum of pairwise diffs at SNPs) /
    (n_pairs * region_length_bp). This is a proper per-bp pi because we
    are dropping a Poisson(mu*L) number of mutations per branch.
    """
    pos_bp = np.asarray(pos_bp)
    in_mask = (pos_bp >= bp_l_g) & (pos_bp < bp_r_g)
    out_mask = ~in_mask
    in_len = bp_r_g - bp_l_g
    S = list(range(n_S))
    I = list(range(n_S, n_S + n_I))

    def _pi(idxA, idxB, mask, region_len):
        if region_len <= 0:
            return float('nan')
        if mask.sum() == 0:
            return 0.0
        d = 0; npairs = 0
        for a in idxA:
            for b in idxB:
                if a == b:
                    continue
                d += (haps[a, mask] != haps[b, mask]).sum()
                npairs += 1
        # Each unordered pair counted twice if idxA==idxB; that's fine
        # since we divide by 2 below for within-class. For dxy we get
        # all ordered pairs.
        return d / max(npairs, 1) / region_len

    return dict(
        in_pi_S=_pi(S, S, in_mask, in_len),
        in_pi_I=_pi(I, I, in_mask, in_len),
        in_dxy_SI=_pi(S, I, in_mask, in_len),
        out_pi_S=_pi(S, S, out_mask, 1.0 - in_len / 1.0),  # not used
        out_pi_I=_pi(I, I, out_mask, 1.0 - in_len / 1.0),
        out_dxy_SI=_pi(S, I, out_mask, 1.0 - in_len / 1.0),
    )


def smc_haps_to_per_site(pos, haps, n_S, n_I,
                          bp_left, bp_right, L_genomic, nsites):
    """SMC returns positions in [0, nsites] (or [0,1] depending on
    version). Convert to genomic bp and use ``stats_per_site``."""
    pos_arr = np.asarray(pos, dtype=float)
    if pos_arr.size == 0:
        return {k: float('nan') for k in ['in_pi_S', 'in_pi_I',
                                            'in_dxy_SI', 'out_pi_S',
                                            'out_pi_I', 'out_dxy_SI']}
    # If positions are fractional [0,1], scale to genomic.
    if pos_arr.max() <= 1.0 + 1e-9:
        pos_bp = pos_arr * L_genomic
    else:
        # SMC defaults to [0, nsites]; rescale.
        pos_bp = pos_arr / nsites * L_genomic
    bp_l_g = bp_left * L_genomic
    bp_r_g = bp_right * L_genomic
    in_mask = (pos_bp >= bp_l_g) & (pos_bp < bp_r_g)
    in_len = bp_r_g - bp_l_g
    out_len = L_genomic - in_len
    S = list(range(n_S)); I = list(range(n_S, n_S + n_I))

    def _pi_region(idxA, idxB, mask, region_len):
        if region_len <= 0 or mask.sum() == 0:
            return 0.0 if region_len > 0 else float('nan')
        d = 0; npairs = 0
        for a in idxA:
            for b in idxB:
                if a == b:
                    continue
                d += (haps[a, mask] != haps[b, mask]).sum()
                npairs += 1
        return d / max(npairs, 1) / region_len

    out_mask = ~in_mask
    return dict(
        in_pi_S=_pi_region(S, S, in_mask, in_len),
        in_pi_I=_pi_region(I, I, in_mask, in_len),
        in_dxy_SI=_pi_region(S, I, in_mask, in_len),
        out_pi_S=_pi_region(S, S, out_mask, out_len),
        out_pi_I=_pi_region(I, I, out_mask, out_len),
        out_dxy_SI=_pi_region(S, I, out_mask, out_len),
    )


def hull_treeseq_to_per_site(ts, mu, n_S, n_I,
                              bp_left, bp_right, L_genomic, rng):
    """Drop msprime mutations on the hull TS, then compute pi/dxy
    in the same units as SMC."""
    import msprime
    seed = int(rng.integers(1, 2**31))
    mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                  discrete_genome=False)
    G = mts.genotype_matrix()  # [n_sites, n_samples]
    haps = G.T  # [n_samples, n_sites]
    pos_bp = np.array([v.site.position for v in mts.variants()])
    bp_l_g = bp_left * L_genomic
    bp_r_g = bp_right * L_genomic
    in_mask = (pos_bp >= bp_l_g) & (pos_bp < bp_r_g)
    out_mask = ~in_mask
    in_len = bp_r_g - bp_l_g
    out_len = L_genomic - in_len
    S = list(range(n_S)); I = list(range(n_S, n_S + n_I))

    def _pi(idxA, idxB, mask, region_len):
        if region_len <= 0 or mask.sum() == 0:
            return 0.0 if region_len > 0 else float('nan')
        d = 0; npairs = 0
        for a in idxA:
            for b in idxB:
                if a == b:
                    continue
                d += int((haps[a, mask] != haps[b, mask]).sum())
                npairs += 1
        return d / max(npairs, 1) / region_len

    return dict(
        in_pi_S=_pi(S, S, in_mask, in_len),
        in_pi_I=_pi(I, I, in_mask, in_len),
        in_dxy_SI=_pi(S, I, in_mask, in_len),
        out_pi_S=_pi(S, S, out_mask, out_len),
        out_pi_I=_pi(I, I, out_mask, out_len),
        out_dxy_SI=_pi(S, I, out_mask, out_len),
    )


# ---------------------------------------------------------------------------
# Run replicates
# ---------------------------------------------------------------------------

print(f"{'='*78}")
print(f"SMC vs HULL — simple comparison (1 pop, n_S={n_S}, n_I={n_I}, "
      f"L={L} bp, NREPS={NREPS})")
print(f"  Ne={Ne}, p_inv={p_inv}, t_inv={t_inv_gen:.0f} gen, "
      f"inv=[{bp_left}, {bp_right})")
print(f"  theta={theta}, rho={rho}, gamma=0")
print(f"{'='*78}")

# ---- HULL ----
print("\n[hull] running...")
t0 = time.time()
hull_per_site = {k: [] for k in [
    'in_pi_S', 'in_pi_I', 'in_dxy_SI',
    'out_pi_S', 'out_pi_I', 'out_dxy_SI']}
hull_T = {k: [] for k in [
    'in_T_SS', 'in_T_II', 'in_T_SI',
    'out_T_SS', 'out_T_II', 'out_T_SI',
    'min_in_T_SI']}
mut_rng = np.random.default_rng(2026)
for rep in range(NREPS):
    sim = HullSimulator(
        n_std=n_S, n_inv=n_I,
        population_size=Ne,
        sequence_length=L,
        p_inv=p_inv, t_inv=t_inv_gen,
        bp_left=bp_left * L, bp_right=bp_right * L,
        gene_conversion_rate=0.0,
        seed=42 + rep,
    )
    ts = sim.simulate()
    s_T = stats_from_treeseq(ts, n_S, n_I, bp_left, bp_right, L)
    for k in hull_T:
        hull_T[k].append(s_T[k])
    s_pi = hull_treeseq_to_per_site(ts, mu, n_S, n_I,
                                      bp_left, bp_right, L, mut_rng)
    for k in hull_per_site:
        hull_per_site[k].append(s_pi[k])
hull_dt = time.time() - t0
print(f"[hull] {NREPS} reps in {hull_dt:.1f}s "
      f"({hull_dt/NREPS*1000:.1f} ms/rep)")

# ---- SMC ----
# msinv.MsinvSimulator on this branch matches main's Option-3
# implementation (in-inv events = gene-flux only).
print("\n[smc ] running...")
t0 = time.time()
smc_per_site = {k: [] for k in [
    'in_pi_S', 'in_pi_I', 'in_dxy_SI',
    'out_pi_S', 'out_pi_I', 'out_dxy_SI']}
nsites_smc = 1000
for rep in range(NREPS):
    sim = msinv.MsinvSimulator(
        nsam=nsam, theta=theta, rho=rho, nsites=nsites_smc,
        n_std=n_S, n_inv=n_I,
        p_inv=p_inv, c=0.0, gamma=0.0,
        bp_left=bp_left, bp_right=bp_right,
        t_inv=t_inv_gen / (2 * Ne),  # SMC takes coal units
        seed=42 + rep,
    )
    pos, haps = sim.simulate_one()
    s = smc_haps_to_per_site(pos, haps, n_S, n_I,
                               bp_left, bp_right, L, nsites_smc)
    for k in smc_per_site:
        smc_per_site[k].append(s[k])
smc_dt = time.time() - t0
print(f"[smc ] {NREPS} reps in {smc_dt:.1f}s "
      f"({smc_dt/NREPS*1000:.1f} ms/rep)")


# ---------------------------------------------------------------------------
# Compare apples-to-apples
# ---------------------------------------------------------------------------

print(f"\n{'='*78}")
print(f"Per-site pi / dxy (mutations dropped on both)")
print(f"{'metric':<10} {'region':<8} {'SMC mean':>14} {'Hull mean':>14} "
      f"{'ratio H/S':>10} {'expected':>14}")
print('-' * 78)
two_mu = 2.0 * mu
# Theoretical expectations
exp = {
    'in_pi_S':  2.0 * 2 * Ne * (1 - p_inv) * mu,   # 2·T·µ with T~2·Ne·p_std
    'in_pi_I':  2.0 * 2 * Ne * p_inv * mu,
    'in_dxy_SI': 2.0 * (t_inv_gen + 2 * Ne) * mu,  # T~t_inv + ancestor coal
    'out_pi_S':  2.0 * 2 * Ne * mu,
    'out_pi_I':  2.0 * 2 * Ne * mu,
    'out_dxy_SI': 2.0 * 2 * Ne * mu,
}
for k in ['in_pi_S', 'in_pi_I', 'in_dxy_SI',
          'out_pi_S', 'out_pi_I', 'out_dxy_SI']:
    region = 'inside' if k.startswith('in_') else 'outside'
    label = k[3:] if k.startswith('in_') else k[4:]
    smc_v = float(np.nanmean(smc_per_site[k]))
    hull_v = float(np.nanmean(hull_per_site[k]))
    ratio = hull_v / smc_v if smc_v else float('nan')
    print(f"{label:<10} {region:<8} {smc_v:>14.6g} {hull_v:>14.6g} "
          f"{ratio:>10.3f} {exp[k]:>14.6g}")
print()

print(f"T_MRCA from hull tree-sequences (gen)")
print(f"  T_SS inside       = {np.nanmean(hull_T['in_T_SS']):.0f}  "
      f"(exp ~ 2·Ne·p_std = {2*Ne*(1-p_inv):.0f})")
print(f"  T_II inside       = {np.nanmean(hull_T['in_T_II']):.0f}  "
      f"(exp ~ 2·Ne·p_inv = {2*Ne*p_inv:.0f})")
print(f"  T_SI inside       = {np.nanmean(hull_T['in_T_SI']):.0f}  "
      f"(exp ≥ t_inv = {t_inv_gen:.0f})")
print(f"  min T_SI inside   = {np.nanmin(hull_T['min_in_T_SI']):.0f}  "
      f"(exp ≥ t_inv = {t_inv_gen:.0f})")
print(f"  T_SS outside      = {np.nanmean(hull_T['out_T_SS']):.0f}  "
      f"(exp ~ 2·Ne = {2*Ne})")

print(f"\nWall time: hull {hull_dt:.1f}s vs smc {smc_dt:.1f}s "
      f"(speedup hull/smc = {smc_dt/hull_dt:.1f}×)")
print(f"{'='*78}")
