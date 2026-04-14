#!/usr/bin/env python3
"""Gene-flux demo: how γ erodes the karyotype barrier.

Sweeps the gene-conversion rate γ across {0, 1e-8, 1e-7, 1e-6} per
bp per generation in an otherwise identical single-inversion,
single-population scenario (Ne=50k, p_inv=0.5, t_inv=200k gen) and
plots:

  Top row    — within-class pi (averaged over S and I) vs position.
                 At γ=0 it is depressed inside the inv by ~factor p_class
                 (structured-coalescent prediction). As γ grows the
                 within-class sub-pops mix and the depression flattens
                 — first in the centre (where phi(x) peaks) and last
                 at the breakpoints (where phi(x) → 0).

  Bottom row — cross-class dxy (S vs I) vs position. At γ=0 the
                 barrier is rigid and dxy is uniformly elevated inside
                 the inv. As γ grows the centre starts to mix → dxy
                 collapses inward, leaving the breakpoints elevated.

These two effects are the same phi(x) shape acting on two different
quantities: gene flux relaxes the structure that produces both.

Outputs ``figures/gene_flux_demo.pdf``.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import msprime
from msinv import HullSimulator, InversionSpec


# ---- Parameters ----
Ne = 50_000
mu = 1e-8
L = 100_000
bp_l, bp_r = 30_000, 70_000
p_inv = 0.5
t_inv = 200_000
n_S = n_I = 8
NREPS = 40
NW = 25
SEED_BASE = 9999

GAMMAS = [0.0, 1e-8, 5e-8, 1e-7]


# ---- Stat helpers (per-bp, in windows) ----
def per_bp_stats(haps, pos_bp, S, I, wins):
    pi_S = np.zeros(len(wins) - 1)
    pi_I = np.zeros(len(wins) - 1)
    dxy = np.zeros(len(wins) - 1)
    for w in range(len(wins) - 1):
        m = (pos_bp >= wins[w]) & (pos_bp < wins[w + 1])
        if not m.any():
            continue
        # within S
        d = n = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d += int((haps[S[i], m] != haps[S[j], m]).sum()); n += 1
        pi_S[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        # within I
        d = n = 0
        for i in range(len(I)):
            for j in range(i + 1, len(I)):
                d += int((haps[I[i], m] != haps[I[j], m]).sum()); n += 1
        pi_I[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        # cross
        d = n = 0
        for a in S:
            for b in I:
                d += int((haps[a, m] != haps[b, m]).sum()); n += 1
        dxy[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return pi_S, pi_I, dxy


def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')


# ---- Run ----
results = {}   # gamma -> (pi_within_avg, dxy)
wins = np.linspace(0, L, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2
mut_rng = np.random.default_rng(7777)

for gamma in GAMMAS:
    print(f"\nγ = {gamma:.0e} — running {NREPS} reps...")
    t0 = time.time()
    pi_avg = np.zeros(NW); dxy = np.zeros(NW); n_ok = 0
    for rep in range(NREPS):
        sim = HullSimulator(
            n_std=n_S, n_inv=n_I,
            population_size=Ne,
            sequence_length=L,
            inversions=[InversionSpec(
                bp_left=bp_l, bp_right=bp_r,
                p_inv=p_inv, t_inv=t_inv,
                gene_conversion_rate=gamma,
                flux_window=0.05)],
            recombination_rate=1e-8,
            seed=SEED_BASE + rep,
        )
        try:
            ts = sim.simulate()
        except Exception:
            continue
        seed = int(mut_rng.integers(1, 2 ** 31))
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                    discrete_genome=False)
        haps = mts.genotype_matrix().T
        pos = np.array([v.site.position for v in mts.variants()])
        S_idx = list(range(n_S))
        I_idx = list(range(n_S, n_S + n_I))
        pi_S, pi_I, d = per_bp_stats(haps, pos, S_idx, I_idx, wins)
        pi_avg += (pi_S + pi_I) / 2
        dxy += d
        n_ok += 1
    pi_avg /= max(n_ok, 1); dxy /= max(n_ok, 1)
    results[gamma] = (pi_avg, dxy)
    print(f"  done: {n_ok}/{NREPS} reps in {time.time()-t0:.0f}s")


# ---- Plot ----
fig, axes = plt.subplots(2, len(GAMMAS), figsize=(15, 6.5),
                          sharex=True, sharey='row')

# Theoretical baselines
pi_full_theory = 4 * Ne * mu                       # collinear pi expectation
pi_inside_theory = 4 * (Ne * p_inv) * mu           # structured pi (=pi_full * p_class)

for col, gamma in enumerate(GAMMAS):
    pi_avg, dxy = results[gamma]
    ax_top = axes[0, col]; ax_bot = axes[1, col]

    ax_top.plot(mid, smooth(pi_avg), '-', color='#1565C0', lw=2.2,
                 label=r'$\bar\pi$ (within S, I)')
    ax_top.axhline(pi_full_theory, color='#666', ls='--', lw=0.8,
                    label=r'collinear $\pi = 4N_e\mu$' if col == 0 else None)
    ax_top.axhline(pi_inside_theory, color='#C62828', ls=':', lw=1.0,
                    label=r'structured $\pi = 4N_e\mu \cdot p_{class}$' if col == 0 else None)
    ax_top.axvspan(bp_l, bp_r, alpha=0.10, color='gray', zorder=0)
    ax_top.set_title(rf'$\gamma$ = {gamma:.0e}', fontsize=11, fontweight='bold')
    if col == 0:
        ax_top.set_ylabel('Within-class\n pi (per bp)', fontsize=10)
        ax_top.legend(fontsize=8, loc='lower right')

    ax_bot.plot(mid, smooth(dxy), '-', color='#C62828', lw=2.2,
                 label=r'$d_{XY}$ (S vs I)')
    ax_bot.axvspan(bp_l, bp_r, alpha=0.10, color='gray', zorder=0)
    ax_bot.set_xlabel('Position (bp)', fontsize=10)
    if col == 0:
        ax_bot.set_ylabel('Cross-class\n $d_{XY}$ (per bp)', fontsize=10)

fig.suptitle(
    'Gene flux erodes the karyotype barrier',
    fontsize=12, fontweight='bold', y=1.00)

caption = (
    f'Figure. Effect of gene-conversion rate $\\gamma$ on the karyotype barrier. '
    f'Each column shows a different $\\gamma$ value (0, 1e-8, 5e-8, 1e-7 bp$^{{-1}}$ gen$^{{-1}}$). '
    f'(Top) Average within-class diversity $\\bar\\pi = (\\pi_S + \\pi_I)/2$. '
    f'At $\\gamma$=0 (no flux), $\\bar\\pi$ inside the inversion (grey shading) is depressed to '
    f'$4 N_e \\mu \\cdot p_{{class}}$ (red dashed) — the structured-coalescent prediction. '
    f'As $\\gamma$ increases, gene flux relaxes the class barrier and $\\bar\\pi$ recovers '
    f'toward the collinear expectation $4 N_e \\mu$ (grey dashed), '
    f'first at the centre (where $\\phi(x)$ peaks) and last at breakpoints. '
    f'(Bottom) Cross-class $d_{{XY}}$ (S vs I). At $\\gamma$=0, $d_{{XY}}$ is uniformly elevated; '
    f'as $\\gamma$ grows, the centre collapses first, leaving breakpoints as the last barrier. '
    f'Parameters: Ne={Ne:,}, p_inv={p_inv}, t_inv={t_inv:,} gen, '
    f'L={L/1e3:.0f} kb, $\\mu$=1e-8, r=1e-8, n_S=n_I={n_S}, {NREPS} replicates.\n'
    f'Command: pixi run -e all python examples/gene_flux_demo.py'
)
fig.text(0.5, -0.04, caption, ha='center', fontsize=7, wrap=True,
         fontstyle='italic', color='#333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                   edgecolor='#BDBDBD', alpha=0.9))

fig.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures',
                    'gene_flux_demo.pdf')
fig.savefig(OUT, bbox_inches='tight', dpi=150)
print(f'\nSaved: {OUT}')

# ---- Summary table ----
print('\nIn-inv vs collinear ratios (avg within-class pi):')
inv_w = [w for w in range(NW) if bp_l < mid[w] < bp_r]
col_w = [w for w in range(NW) if w not in inv_w]
print(f"  {'gamma':>10} {'pi_inv':>10} {'pi_col':>10} {'pi_ratio':>10}  "
      f"{'dxy_inv':>10} {'dxy_col':>10} {'dxy_ratio':>10}")
for gamma in GAMMAS:
    pi_avg, dxy = results[gamma]
    pi_i = float(np.mean([pi_avg[w] for w in inv_w]))
    pi_c = float(np.mean([pi_avg[w] for w in col_w]))
    d_i = float(np.mean([dxy[w] for w in inv_w]))
    d_c = float(np.mean([dxy[w] for w in col_w]))
    print(f"  {gamma:>10.0e} {pi_i:>10.3e} {pi_c:>10.3e} "
          f"{pi_i/pi_c:>10.2f}  "
          f"{d_i:>10.3e} {d_c:>10.3e} {d_i/d_c:>10.2f}")
