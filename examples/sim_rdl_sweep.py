#!/usr/bin/env python3
"""
Simulate the RDL-like sweep-through-inversion scenario.

Full sequence:
  1. Selected allele (RDL) arises on S background, sweeps to fixation
  2. Recombination within S/S breaks up the swept haplotype over time
  3. Gene flux transfers RDL (+ flanking tract) to I background
  4. RDL sweeps on I background
  5. Recombination within I/I breaks up the I-background sweep

We simulate n=2 (one S, one I chromosome) at high resolution to
show the haplotype structure around RDL at different time points.
"""
import sys, os
import msinv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ======================================================
# Parameters
# ======================================================
Ne = 100_000
gen_per_year = 11
p_inv = 0.5
rho = 50         # total recomb rate (4Nr)
theta = 10
nsites = 1000    # positions
bp_l, bp_r = 0.3, 0.7
x_sel = 0.5      # RDL position (center of inversion)

# Selection coefficient for insecticide resistance
s_rdl = 0.1      # strong

# Gene flux
gamma_val = 0.1  # moderate flux rate

# Sweep timing (coalescent units)
s_scaled = 2 * Ne * s_rdl  # 20000
t_sweep = np.log(s_scaled) / s_scaled  # ~0.0005 coal = ~100 gen

# Recombination rate per site per coal unit
rho_per_site = rho  # fractional distance, rho * d for distance d in [0,1]

print("=" * 65)
print("RDL sweep-through-inversion simulation")
print("=" * 65)
print(f"Ne={Ne:,}, s={s_rdl}, p_inv={p_inv}")
print(f"t_sweep = {t_sweep:.5f} coal units = {t_sweep*2*Ne:.0f} gen")
print(f"rho={rho}, theta={theta}")
print(f"gamma={gamma_val}")


# ======================================================
# Direct haplotype simulation
# ======================================================
# Instead of using the full msinv machinery, simulate directly:
# 1. Generate a neutral haplotype matrix (S and I backgrounds)
# 2. Apply the sweep sequence with correct timing

def simulate_neutral_haplotypes(n_S, n_I, rng):
    """Generate neutral haplotype pairs via msinv."""
    sim = msinv.MsinvSimulator(
        nsam=n_S + n_I, nreps=1, theta=theta, rho=rho, nsites=nsites,
        n_std=n_S, n_inv=n_I, p_inv=p_inv, c=0.01, gamma=gamma_val,
        bp_left=bp_l, bp_right=bp_r, t_inv=10.0,
        seed=int(rng.integers(1e8)))
    pos, haps = sim.simulate_one()
    return pos, haps


def apply_rdl_sequence(pos, haps, n_S, n_I, t_flux_gen, rng):
    """
    Apply the full RDL sweep sequence to a neutral haplotype matrix.

    Timeline (going FORWARD, in generations before present):
      t_sweep_start: RDL arises on S, begins sweep
      t_sweep_done = t_sweep_start - sweep_duration: S sweep completes
      t_flux: gene flux transfers RDL + tract to I
      t_I_sweep_done = t_flux - sweep_duration: I sweep completes
      present (t=0): we observe the haplotypes

    The key erosion periods:
      S erosion time = t_sweep_done to present (or to t_flux if we model
        that flux happened while S was still swept)
      I erosion time = t_I_sweep_done to present
    """
    if len(pos) == 0:
        return pos, haps

    pos_arr = np.array(pos)
    h = haps.copy()
    S_idx = list(range(n_S))
    I_idx = list(range(n_S, n_S + n_I))

    sweep_dur_gen = t_sweep * 2 * Ne  # ~100 gen

    # Total time since sweep started on S (generations before present)
    # For RDL: dieldrin use started ~1950, ~800 gen ago at 11 gen/yr
    t_S_sweep_start = 800  # gen before present

    # S sweep completed quickly
    t_S_sweep_done = t_S_sweep_start - sweep_dur_gen  # ~700 gen ago

    # Flux happened at t_flux_gen after S sweep done
    t_flux_happened = t_S_sweep_done - t_flux_gen  # gen before present

    # I sweep started at flux time, completed quickly
    t_I_sweep_done = t_flux_happened - sweep_dur_gen

    # Erosion times
    t_S_erosion = t_S_sweep_done  # gen of recombination on S since sweep
    t_I_erosion = max(0, t_I_sweep_done)  # gen of recombination on I since sweep

    # Convert to coalescent units for haplotype length calculation
    t_S_coal = t_S_erosion / (2 * Ne)
    t_I_coal = t_I_erosion / (2 * Ne)

    # Gene conversion tract length (~500 bp = 0.5% of our 100kb region)
    flux_tract_half = 0.005  # ±0.5% of chromosome around x_sel

    for j, p in enumerate(pos_arr):
        dist = abs(p - x_sel)

        # --- S background ---
        # P(this site still carries swept allele) = exp(-rho*dist * t_erosion)
        # where t_erosion is in coalescent units
        p_S_intact = np.exp(-rho * dist * t_S_coal * p_inv)
        # p_inv factor: recomb only in S/S matings (prob = 1-p_inv... wait,
        # for S chroms, they recombine in S/S matings at rate rho*p_std
        # Actually: effective recomb for S = rho * p_std (only S/S matings)
        p_S_intact = np.exp(-rho * dist * t_S_coal * (1 - p_inv))

        if rng.random() < p_S_intact:
            # This site is still part of the swept block on S
            # All S samples carry the same allele
            ref_allele = h[S_idx[0], j]
            for s in S_idx:
                h[s, j] = ref_allele

        # --- I background ---
        if t_I_erosion > 0:
            # I sweep happened. Was this site in the flux tract?
            in_flux_tract = dist < flux_tract_half

            # P(this site still carries I-swept allele)
            p_I_intact = np.exp(-rho * dist * t_I_coal * p_inv)

            if rng.random() < p_I_intact:
                if in_flux_tract:
                    # This site was in the conversion tract:
                    # It carries the S-origin allele (transferred via flux)
                    # Same allele as the S background at this position
                    for i in I_idx:
                        h[i, j] = h[S_idx[0], j]
                else:
                    # This site is part of the I sweep but NOT the flux tract:
                    # It carries whatever allele was on the I chrom that
                    # received the flux. This is the ORIGINAL I allele.
                    ref_I = h[I_idx[0], j]
                    for i in I_idx:
                        h[i, j] = ref_I

    return pos, h


# ======================================================
# Run simulations for different flux timing scenarios
# ======================================================
n_S = 5; n_I = 5
NR = 200
NW = 50
wins = np.linspace(0, nsites, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2


def compute_stats(pos, haps, S_idx, I_idx):
    """Compute pi_S, pi_I, dxy, and S-I haplotype similarity per window."""
    pos_arr = np.array(pos) * nsites
    pi_S = np.zeros(NW)
    pi_I = np.zeros(NW)
    dxy = np.zeros(NW)
    # Fraction of sites where S and I carry same allele (identity)
    identity = np.zeros(NW)
    n_sites_w = np.zeros(NW)

    nS, nI = len(S_idx), len(I_idx)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0:
            continue
        n_sites_w[w] = mask.sum()
        for j in np.where(mask)[0]:
            # pi_S
            for a in range(nS):
                for b in range(a + 1, nS):
                    if haps[S_idx[a], j] != haps[S_idx[b], j]:
                        pi_S[w] += 1
            # pi_I
            for a in range(nI):
                for b in range(a + 1, nI):
                    if haps[I_idx[a], j] != haps[I_idx[b], j]:
                        pi_I[w] += 1
            # dxy
            for a in S_idx:
                for b in I_idx:
                    if haps[a, j] != haps[b, j]:
                        dxy[w] += 1
            # identity: do S and I share the majority allele?
            s_maj = round(np.mean([haps[s, j] for s in S_idx]))
            i_maj = round(np.mean([haps[i, j] for i in I_idx]))
            if s_maj == i_maj:
                identity[w] += 1

    if nS >= 2:
        pi_S /= (nS * (nS - 1) / 2)
    pi_I /= (nI * (nI - 1) / 2)
    dxy /= (nS * nI)
    identity = np.where(n_sites_w > 0, identity / n_sites_w, 0)

    return pi_S, pi_I, dxy, identity


scenarios = [
    ("No sweep\n(neutral)", None),
    ("S sweep only\n(no flux yet)", 1e9),          # flux hasn't happened
    ("Flux after\n100 gen", 100),
    ("Flux after\n500 gen", 500),
]

S_idx = list(range(n_S))
I_idx = list(range(n_S, n_S + n_I))

results = {}
for label, t_flux in scenarios:
    print(f"\nRunning: {label.replace(chr(10), ' ')}...")
    piS = np.zeros(NW); piI = np.zeros(NW)
    dxy_arr = np.zeros(NW); ident = np.zeros(NW)
    n_ok = 0

    for rep in range(NR):
        rng = np.random.default_rng(42 + rep)
        pos, haps = simulate_neutral_haplotypes(n_S, n_I, rng)
        if len(pos) == 0:
            continue

        if t_flux is not None:
            pos, haps = apply_rdl_sequence(pos, haps, n_S, n_I, t_flux, rng)

        pS, pI, dx, iden = compute_stats(pos, haps, S_idx, I_idx)
        piS += pS; piI += pI; dxy_arr += dx; ident += iden
        n_ok += 1

    if n_ok > 0:
        piS /= n_ok; piI /= n_ok; dxy_arr /= n_ok; ident /= n_ok

    results[label] = dict(piS=piS, piI=piI, dxy=dxy_arr,
                           identity=ident, n=n_ok)
    print(f"  {n_ok}/{NR} reps")


# ======================================================
# Figure
# ======================================================
def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, len(scenarios), hspace=0.35, wspace=0.25)

for col, (label, res) in enumerate(results.items()):
    # Row 0: pi
    ax0 = fig.add_subplot(gs[0, col])
    ax0.plot(mid, smooth(res['piS']), '-', color='#2196F3', lw=2,
             label=r'$\pi_S$')
    ax0.plot(mid, smooth(res['piI']), '-', color='#E91E63', lw=2,
             label=r'$\pi_I$')
    ax0.axvspan(bp_l * nsites, bp_r * nsites, alpha=0.08, color='gray')
    ax0.axvline(x_sel * nsites, color='red', ls=':', alpha=0.6, lw=1.5)
    ax0.set_title(label, fontsize=10, fontweight='bold')
    if col == 0:
        ax0.set_ylabel(r'$\pi$ (within-class)', fontsize=11)
        ax0.legend(fontsize=8)
    ax0.set_xlim(0, nsites)

    # Row 1: dxy
    ax1 = fig.add_subplot(gs[1, col])
    ax1.plot(mid, smooth(res['dxy']), '-', color='#FF9800', lw=2,
             label=r'$d_{XY}$')
    ax1.axvspan(bp_l * nsites, bp_r * nsites, alpha=0.08, color='gray')
    ax1.axvline(x_sel * nsites, color='red', ls=':', alpha=0.6, lw=1.5)
    if col == 0:
        ax1.set_ylabel(r'$d_{XY}$ (S vs I)', fontsize=11)
        ax1.legend(fontsize=8)
    ax1.set_xlim(0, nsites)

    # Row 2: S-I haplotype identity
    ax2 = fig.add_subplot(gs[2, col])
    ax2.plot(mid, smooth(res['identity']), '-', color='#4CAF50', lw=2,
             label='S-I identity')
    ax2.axvspan(bp_l * nsites, bp_r * nsites, alpha=0.08, color='gray')
    ax2.axvline(x_sel * nsites, color='red', ls=':', alpha=0.6, lw=1.5)
    ax2.set_ylim(-0.05, 1.05)
    if col == 0:
        ax2.set_ylabel('S-I haplotype\nidentity', fontsize=11)
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_xlim(0, nsites)

# Annotations
fig.text(0.5, 0.01,
         f'Parameters: Ne={Ne:,}, s={s_rdl}, p_inv={p_inv}, '
         f'rho={rho}, gamma={gamma_val}, '
         f'flux tract ~1 kb, sweep ~{t_sweep*2*Ne:.0f} gen\n'
         f'Red line = RDL selected site. '
         f'S haplotype erodes at rate rho*p_std per coal unit. '
         f'Flux tract length ~{2*0.005*100:.0f} kb.',
         ha='center', fontsize=9, fontstyle='italic', color='#546E7A')

fig.suptitle(
    'RDL-like sweep through inversion: haplotype dynamics\n'
    'Sweep on S background → recombination erodes haplotype → '
    'gene flux transfers RDL to I → sweep on I',
    fontsize=12, fontweight='bold', y=0.98)

fig.savefig('figures/rdl_sweep_haplotypes.pdf', bbox_inches='tight', dpi=150)
print(f"\nSaved: figures/rdl_sweep_haplotypes.pdf")

# Summary
print("\nSummary at RDL site (center windows):")
center_w = [w for w in range(NW) if 450 < mid[w] < 550]
for label, res in results.items():
    piS_c = np.mean([res['piS'][w] for w in center_w])
    piI_c = np.mean([res['piI'][w] for w in center_w])
    dxy_c = np.mean([res['dxy'][w] for w in center_w])
    id_c = np.mean([res['identity'][w] for w in center_w])
    print(f"  {label.replace(chr(10), ' '):30s}: "
          f"piS={piS_c:.3f} piI={piI_c:.3f} dxy={dxy_c:.3f} "
          f"S-I_identity={id_c:.2f}")
