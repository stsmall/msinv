#!/usr/bin/env python3
"""
Generate flux_transfer_probability figure.

Shows the gene-flux transfer probability across the inversion for
different flux-window widths, and empirically measures the realized
S->I transfer rate vs the theoretical Peischl 2013 prediction.

Top panel: theoretical phi(x) = min(x, 1-x, w) / (1-w), Peischl 2013.
Bottom panel: simulated S->I transfer probability at each inversion
position for several gamma values, compared to gamma * phi(x) * t.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import msinv
from msinv.simulator import GeneFluxModel


# ---------- Top panel: theoretical phi(x) ----------
xs = np.linspace(0.001, 0.999, 500)

fig = plt.figure(figsize=(11, 9))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
ax_top = fig.add_subplot(gs[0])

for w, color in [(0.1, '#1976D2'), (0.2, '#388E3C'),
                 (0.3, '#F57C00'), (0.5, '#C2185B')]:
    fm = GeneFluxModel(w=w)
    y = [fm.phi(x) for x in xs]
    ax_top.plot(xs, y, '-', color=color, lw=2,
                label=f'$w={w}$')

ax_top.set_xlabel('Position within inversion $x$', fontsize=11)
ax_top.set_ylabel(r'$\phi(x)$', fontsize=12)
ax_top.set_title('A.  Theoretical flux-transfer probability '
                 r'$\phi(x) = \min(x, 1-x, w) / (1-w)$',
                 fontsize=11, fontweight='bold', loc='left')
ax_top.axvspan(0, 0.001, alpha=0.1, color='gray')
ax_top.axvspan(0.999, 1, alpha=0.1, color='gray')
ax_top.text(0.005, ax_top.get_ylim()[1] * 0.05, 'bp_left',
            fontsize=8, color='gray', rotation=90, va='bottom')
ax_top.text(0.995, ax_top.get_ylim()[1] * 0.05, 'bp_right',
            fontsize=8, color='gray', rotation=90, va='bottom', ha='right')
ax_top.legend(fontsize=9, loc='upper right', title='Flux window $w$')
ax_top.grid(alpha=0.3)
ax_top.set_xlim(0, 1)


# ---------- Bottom panel: empirical S->I transfer ----------
# Simulate with varying gamma, measure per-window (S vs I) identity rate.
# Near breakpoints, phi(x) is small -> less mixing -> more separation.
# Near center, phi(x) is larger -> more mixing -> less separation.
Ne = 10_000
L = 100_000
bp_l = 0.15
bp_r = 0.85
p_inv = 0.5
n_std = 5; n_inv = 5
nsam = n_std + n_inv
NR = 200

# Empirical: measure fraction of S-I pairs IDENTICAL at each position
# (high identity = lots of flux mixing, low = strong class structure)
NW = 30
wins = np.linspace(0, 1, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2

ax_bot = fig.add_subplot(gs[1])

gammas = [0.0, 1.0, 5.0, 20.0]
colors = ['#212121', '#1976D2', '#F57C00', '#C2185B']

for gamma_val, color in zip(gammas, colors):
    identity = np.zeros(NW)
    n_ok = 0
    for rep in range(NR):
        sim = msinv.MsinvSimulator(
            samples=nsam,
            population_size=Ne,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            sequence_length=L,
            n_std=n_std, n_inv=n_inv,
            p_inv=p_inv, gamma=gamma_val,
            bp_left=bp_l, bp_right=bp_r,
            t_inv=5.0,  # ~100k gen
            seed=42 + rep)
        try:
            pos, haps = sim.simulate_one()
        except Exception:
            continue
        if len(pos) == 0:
            continue
        pa = np.array(pos)
        # fraction of segregating sites where majority S allele == majority I allele
        for w in range(NW):
            mask = (pa >= wins[w]) & (pa < wins[w + 1])
            if mask.sum() == 0:
                continue
            idents = 0; total = 0
            for j in np.where(mask)[0]:
                s_maj = int(round(np.mean([haps[s, j] for s in range(n_std)])))
                i_maj = int(round(np.mean([haps[i, j] for i in range(n_std, nsam)])))
                if s_maj == i_maj:
                    idents += 1
                total += 1
            if total > 0:
                identity[w] += idents / total
        n_ok += 1
    if n_ok > 0:
        identity /= n_ok
    ax_bot.plot(mid, identity, '-o', color=color, lw=1.5, ms=4,
                label=rf'$\gamma={gamma_val}$')

# Highlight inversion region
ax_bot.axvspan(bp_l, bp_r, alpha=0.08, color='gray')
ax_bot.axvline(bp_l, color='gray', ls='--', alpha=0.5, lw=0.8)
ax_bot.axvline(bp_r, color='gray', ls='--', alpha=0.5, lw=0.8)
ax_bot.text((bp_l + bp_r) / 2, ax_bot.get_ylim()[1] * 0.95,
            f'inversion [{bp_l}, {bp_r}]',
            ha='center', fontsize=9, color='#546E7A', fontstyle='italic')

ax_bot.set_xlabel('Genomic position', fontsize=11)
ax_bot.set_ylabel('Fraction of segregating sites\nwith S-maj = I-maj',
                  fontsize=11)
ax_bot.set_title(r'B.  Simulated S–I allele-sharing vs gamma '
                 r'(mixing increases with $\gamma \phi(x) t$)',
                 fontsize=11, fontweight='bold', loc='left')
ax_bot.legend(fontsize=9, loc='lower center',
              title=r'Gene-flux rate $\gamma = 4N_e g$')
ax_bot.grid(alpha=0.3)
ax_bot.set_xlim(0, 1)

fig.suptitle('Gene-flux transfer probability in msinv',
             fontsize=13, fontweight='bold', y=0.995)

fig.savefig('figures/flux_transfer_probability.pdf',
            bbox_inches='tight', dpi=150)
print('Saved: figures/flux_transfer_probability.pdf')
