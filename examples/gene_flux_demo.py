#!/usr/bin/env python3
"""Gene-flux demo: how gamma erodes the karyotype barrier.

Sweeps the gene-conversion rate gamma across {0, 1e-8, 5e-8, 1e-7} per
bp per generation. Each gamma value runs in parallel via multiprocessing.

Plots:
  Top row    — within-class pi (averaged over S and I) vs position.
  Bottom row — cross-class dxy (S vs I) vs position.

Outputs ``figures/gene_flux_demo.pdf``.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import msprime
from multiprocessing import Pool

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
        d = n = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d += int((haps[S[i], m] != haps[S[j], m]).sum()); n += 1
        pi_S[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        d = n = 0
        for i in range(len(I)):
            for j in range(i + 1, len(I)):
                d += int((haps[I[i], m] != haps[I[j], m]).sum()); n += 1
        pi_I[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        d = n = 0
        for a in S:
            for b in I:
                d += int((haps[a, m] != haps[b, m]).sum()); n += 1
        dxy[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return pi_S, pi_I, dxy


def run_gamma(args):
    """Run all reps for one gamma value. Returns (gamma, pi_avg, dxy)."""
    gamma, gamma_idx = args
    print(f"  gamma={gamma:.0e}: starting {NREPS} reps...")
    t0 = time.time()
    wins = np.linspace(0, L, NW + 1)
    mut_rng = np.random.default_rng(7777 + gamma_idx)
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
            seed=SEED_BASE + gamma_idx * 1000 + rep,
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
    print(f"  gamma={gamma:.0e}: done {n_ok}/{NREPS} in {time.time()-t0:.0f}s")
    return gamma, pi_avg, dxy


def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')


def main():
    print(f"Gene flux demo: {len(GAMMAS)} gamma values x {NREPS} reps, parallelized")
    t0 = time.time()

    tasks = [(g, i) for i, g in enumerate(GAMMAS)]
    with Pool(len(GAMMAS)) as pool:
        raw = pool.map(run_gamma, tasks)

    results = {gamma: (pi_avg, dxy) for gamma, pi_avg, dxy in raw}
    elapsed = time.time() - t0
    print(f"\nAll gamma values done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save data for later overlay of theoretical predictions
    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2
    save = {'mid': mid, 'Ne': Ne, 'mu': mu, 'L': L,
            'bp_l': bp_l, 'bp_r': bp_r, 'p_inv': p_inv, 't_inv': t_inv}
    for gamma, (pi_avg, dxy) in results.items():
        tag = f'{gamma:.0e}'.replace('+', '')
        save[f'pi_{tag}'] = pi_avg
        save[f'dxy_{tag}'] = dxy
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
    np.savez(os.path.join(OUT_DIR, 'gene_flux_data.npz'), **save)

    # ---- Plot ----

    fig, axes = plt.subplots(2, len(GAMMAS), figsize=(15, 6.5),
                              sharex=True, sharey='row')

    pi_full_theory = 4 * Ne * mu
    pi_inside_theory = 4 * (Ne * p_inv) * mu
    dxy_full_barrier = 2 * mu * (t_inv + 2 * Ne)
    dxy_panmictic = 2 * mu * 2 * Ne
    # Position-dependent theory arrays
    inside = (mid >= bp_l) & (mid <= bp_r)
    pi_th_curve = np.where(inside, pi_inside_theory, pi_full_theory)
    dxy_th_curve = np.where(inside, dxy_full_barrier, dxy_panmictic)

    for col, gamma in enumerate(GAMMAS):
        pi_avg, dxy = results[gamma]
        ax_top = axes[0, col]; ax_bot = axes[1, col]

        # Annotation: 4*Ne*gamma regime
        regime = 4 * Ne * gamma
        regime_label = (rf'$4N_e\gamma$ = {regime:.1f}'
                        + (' (isolated)' if regime < 0.1
                           else ' (mixing)' if regime > 1
                           else ' (transition)'))

        ax_top.plot(mid, smooth(pi_avg), '-', color='#1565C0', lw=2.2,
                     label=r'$\bar\pi$ (within S, I)')
        ax_top.plot(mid, pi_th_curve, '--', color='#C62828', lw=1.0,
                     label=r'$E[\pi_c]$' if col == 0 else None)
        ax_top.axvspan(bp_l, bp_r, alpha=0.10, color='gray', zorder=0)
        ax_top.set_title(rf'$\gamma$ = {gamma:.0e}' + f'\n{regime_label}',
                          fontsize=10, fontweight='bold')
        if col == 0:
            ax_top.set_ylabel('Within-class\n pi (per bp)', fontsize=10)
            ax_top.legend(fontsize=7, loc='lower right')

        ax_bot.plot(mid, smooth(dxy), '-', color='#C62828', lw=2.2,
                     label=r'$d_{XY}$ (S vs I)')
        ax_bot.plot(mid, dxy_th_curve, '--', color='#C62828', lw=1.0,
                     label=r'$E[d_{XY}]$' if col == 0 else None)
        ax_bot.axvspan(bp_l, bp_r, alpha=0.10, color='gray', zorder=0)
        ax_bot.set_xlabel('Position (bp)', fontsize=10)
        if col == 0:
            ax_bot.set_ylabel('Cross-class\n $d_{XY}$ (per bp)', fontsize=10)
            ax_bot.legend(fontsize=7, loc='upper right')

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
    plt.close()
    print(f'Saved: {OUT}')

    # ---- Summary table ----
    inv_w = [w for w in range(NW) if bp_l < mid[w] < bp_r]
    col_w = [w for w in range(NW) if w not in inv_w]
    print(f"\n  {'gamma':>10} {'pi_inv':>10} {'pi_col':>10} {'pi_ratio':>10}  "
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


if __name__ == '__main__':
    main()
