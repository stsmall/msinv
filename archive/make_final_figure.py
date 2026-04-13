#!/usr/bin/env python3
"""
Final figure: exact marginals + SMC LD for a chromosome with central inversion.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from sim_hybrid import exact_marginal_stats
from sim_partial_inv import simulate_partial_inversion, compute_ld_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- Parameters ----
nsam = 10; n_std = 5; n_inv = 5
theta = 20; rho = 100; nsites = 1000
p_inv = 0.5; c = 0.01
bp_left = 0.35; bp_right = 0.65

print("Computing exact marginals (500 reps × 200 positions)...", flush=True)
marginals = exact_marginal_stats(
    n_std=n_std, n_inv=n_inv, theta_per_site=theta/nsites,
    rho=rho, p_inv=p_inv, c=c,
    bp_left=bp_left, bp_right=bp_right,
    flux_window=0.3, nreps=500, n_positions=200, seed=42
)

print("Computing LD via SMC (40 reps)...", flush=True)
n_windows = 50
all_ld = []
for rep in range(40):
    if rep % 10 == 0:
        print(f"  SMC rep {rep}/40...", flush=True)
    pos, haps = simulate_partial_inversion(
        nsam, n_std, n_inv, theta, rho, nsites,
        p_inv, c, bp_left, bp_right, seed=rep * 7 + 100
    )
    all_ld.append(compute_ld_matrix(pos, haps, n_bins=n_windows))

mean_ld = np.mean(all_ld, axis=0)

print("Plotting...", flush=True)

# ---- Figure ----
fig = plt.figure(figsize=(14, 10.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.30, wspace=0.28)
inv_color = '#ede7f6'
pos_m = marginals['positions']

# Scale T to diversity: pi = theta_per_site * T (since theta = 4Nmu per site)
theta_ps = theta / nsites
T_SS = marginals['T_SS']
T_II = marginals['T_II']
T_SI = marginals['T_SI']
T_tot = marginals['T_total']

# -- Panel A: Expected coalescence times --
ax1 = fig.add_subplot(gs[0, 0])
ax1.axvspan(bp_left, bp_right, color=inv_color, zorder=0)
ax1.axvline(bp_left, color='#5e35b1', ls='--', lw=1.2, alpha=0.7)
ax1.axvline(bp_right, color='#5e35b1', ls='--', lw=1.2, alpha=0.7)

ax1.plot(pos_m, T_SI, color='#c62828', lw=2, label='$T_{SI}$ (between arrangements)')
ax1.plot(pos_m, T_SS, color='#1565c0', lw=1.5, label='$T_{SS}$ (within standard)')
ax1.plot(pos_m, T_II, color='#e65100', lw=1.5, ls='--', label='$T_{II}$ (within inverted)')

ax1.set_xlabel('Chromosome position')
ax1.set_ylabel('Expected coalescence time (2N gen)')
ax1.set_title('A.  Coalescence times (exact, 500 reps)',
              loc='left', fontweight='bold', fontsize=10)
ax1.legend(fontsize=8, loc='upper left')
ax1.set_xlim(0, 1)

# -- Panel B: log scale showing the contrast --
ax2 = fig.add_subplot(gs[0, 1])
ax2.axvspan(bp_left, bp_right, color=inv_color, zorder=0)
ax2.axvline(bp_left, color='#5e35b1', ls='--', lw=1.2, alpha=0.7)
ax2.axvline(bp_right, color='#5e35b1', ls='--', lw=1.2, alpha=0.7)

ax2.semilogy(pos_m, T_SI, color='#c62828', lw=2, label='$T_{SI}$')
ax2.semilogy(pos_m, T_SS, color='#1565c0', lw=1.5, label='$T_{SS}$')
ax2.semilogy(pos_m, T_II, color='#e65100', lw=1.5, ls='--', label='$T_{II}$')

# Annotate peaks
inv_mask = (pos_m >= bp_left) & (pos_m <= bp_right)
peak_T = np.max(T_SI[inv_mask])
center_T = T_SI[len(pos_m)//2]
ax2.annotate(f'breakpoint\n$T_{{SI}}$≈{peak_T:.0f}',
             xy=(bp_left + 0.02, peak_T), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='gray'),
             xytext=(0.15, peak_T * 0.8), color='#c62828')
ax2.annotate(f'center\n$T_{{SI}}$≈{center_T:.0f}',
             xy=(0.5, center_T), fontsize=8,
             xytext=(0.5, center_T * 3), ha='center', color='#c62828',
             arrowprops=dict(arrowstyle='->', color='gray'))

ax2.set_xlabel('Chromosome position')
ax2.set_ylabel('Coalescence time (log scale)')
ax2.set_title('B.  Log scale — $T_{SS}$ = $T_{II}$ ≈ 2.0 everywhere',
              loc='left', fontweight='bold', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_xlim(0, 1)

# -- Panel C: LD heatmap (from SMC) --
ax3 = fig.add_subplot(gs[1, :])
ld_plot = mean_ld.copy()
np.fill_diagonal(ld_plot, np.nan)
ld_plot[ld_plot == 0] = np.nan

vmax = np.nanpercentile(ld_plot, 97)
im = ax3.imshow(ld_plot, origin='lower', extent=[0, 1, 0, 1],
                cmap='YlOrRd', aspect='equal', vmin=0, vmax=vmax)

for bp in [bp_left, bp_right]:
    ax3.axvline(bp, color='#5e35b1', ls='--', lw=1.5, alpha=0.8)
    ax3.axhline(bp, color='#5e35b1', ls='--', lw=1.5, alpha=0.8)

mid_inv = (bp_left + bp_right) / 2
ax3.annotate('inversion', xy=(mid_inv, 1.02), xycoords=('data', 'axes fraction'),
             ha='center', fontsize=10, color='#5e35b1', fontweight='bold')

ax3.set_xlabel('Chromosome position (site 1)')
ax3.set_ylabel('Chromosome position (site 2)')
ax3.set_title('C.  Linkage disequilibrium $r^2$ (SMC, 40 reps)',
              loc='left', fontweight='bold', fontsize=10)
cbar = plt.colorbar(im, ax=ax3, fraction=0.025, pad=0.015)
cbar.set_label('Mean $r^2$', fontsize=9)

fig.suptitle(
    f'Coalescent patterns for a chromosome with central inversion [{bp_left}–{bp_right}]\n'
    f'n={nsam} ({n_std}S + {n_inv}I),  θ={theta},  ρ={rho},  '
    f'$p_{{inv}}$={p_inv},  gene flux c={c}',
    fontsize=11, y=0.99)

plt.savefig('chromosome_inversion_final.png', dpi=150, bbox_inches='tight')
print("Saved: chromosome_inversion_final.png")
