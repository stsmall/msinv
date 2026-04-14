#!/usr/bin/env python3
"""Empirical Kir/Fol simulation.

Reproduces the An. funestus 3Ra/3Rb cross-karyotype divergence pattern
(Small et al. 2023 Fig. S13) using msinv's hull simulator.

Outputs:
  figures/empirical_kir_fol.pdf

Constant Ne for both pops (avoids the structured-coalescent dxy
depression at extreme Ne asymmetry — see docs/known_issues.md).
"""
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import msprime  # for mutation dropping on the hull TreeSequence

from msinv import HullSimulator, InversionSpec, Demography


# --- Parameters ---
Ne = 44_000
mu = 3.55e-9
r = 1e-9   # reduced from 4e-8 for computational feasibility (rho≈18)
L = 100_000
t_split_gen = 14_000     # ~1300 yr at 11 gen/yr
t_inv_gen = 385_000      # ~35 kyr — 3Ra age (Small 2023)
p_inv_anc = 0.3

# 3Ra and 3Rb breakpoints, in genomic coords
inv_3Ra = (15_000, 45_000)
inv_3Rb = (55_000, 85_000)

# Sample composition: K=fixed-S, Fol=mixed
n_kir = 10
n_fol_S = 5
n_fol_I = 5

NREPS = 200
NW = 40   # number of windows for the spatial plot
SEED_BASE = 4242

# ---- Sample groups (set during simulation) ----
kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, n_kir + n_fol_S + n_fol_I))


def per_window_stats(haps, pos_bp, group_a, group_b, kind='dxy'):
    """Compute per-window dxy or pi.

    kind='dxy': between groups; 'pi': within group_a (group_b ignored).
    Returns array of length NW.
    """
    wins = np.linspace(0, L, NW + 1)
    vals = np.zeros(NW)
    for w in range(NW):
        mask = (pos_bp >= wins[w]) & (pos_bp < wins[w + 1])
        if mask.sum() == 0:
            continue
        if kind == 'dxy':
            d = 0; n = 0
            for a in group_a:
                for b in group_b:
                    d += (haps[a, mask] != haps[b, mask]).sum()
                    n += 1
            vals[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        else:  # pi
            d = 0; n = 0
            ga = list(group_a)
            for i in range(len(ga)):
                for j in range(i + 1, len(ga)):
                    d += (haps[ga[i], mask] != haps[ga[j], mask]).sum()
                    n += 1
            vals[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return vals


# ---- Run replicates ----
print(f"Running {NREPS} hull simulations of Kir/Fol with both 3Ra+3Rb...")
t0 = time.time()

dxy_kf_same = np.zeros(NW)
dxy_kf_alt = np.zeros(NW)
dxy_fs_fi = np.zeros(NW)
pi_K = np.zeros(NW)
pi_FS = np.zeros(NW)
pi_FI = np.zeros(NW)
n_ok = 0
mut_rng = np.random.default_rng(2026)

for rep in range(NREPS):
    demo = Demography(pop_sizes=[Ne, Ne])
    demo.add_event(('ej', t_split_gen, 1, 0))   # Fol → Kir at t_split

    sim = HullSimulator(
        sample_config={
            ('S', 0): n_kir,            # Kiribina, all S at both invs (linked)
            ('S', 1): n_fol_S,          # Folonzo standard
            ('I', 1): n_fol_I,          # Folonzo inverted
        },
        demography=demo,
        sequence_length=L,
        inversions=[
            InversionSpec(bp_left=inv_3Ra[0], bp_right=inv_3Ra[1],
                              p_inv=p_inv_anc, t_inv=t_inv_gen),
            InversionSpec(bp_left=inv_3Rb[0], bp_right=inv_3Rb[1],
                              p_inv=p_inv_anc, t_inv=t_inv_gen),
        ],
        recombination_rate=r,
        seed=SEED_BASE + rep,
    )
    try:
        ts = sim.simulate()
    except Exception as e:
        continue
    # Drop msprime mutations on the hull TS
    seed = int(mut_rng.integers(1, 2**31))
    mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                  discrete_genome=False)
    G = mts.genotype_matrix()
    haps = G.T
    pos_bp = np.array([v.site.position for v in mts.variants()])

    dxy_kf_same += per_window_stats(haps, pos_bp, kir, fol_same, 'dxy')
    dxy_kf_alt += per_window_stats(haps, pos_bp, kir, fol_alt, 'dxy')
    dxy_fs_fi += per_window_stats(haps, pos_bp, fol_same, fol_alt, 'dxy')
    pi_K += per_window_stats(haps, pos_bp, kir, None, 'pi')
    pi_FS += per_window_stats(haps, pos_bp, fol_same, None, 'pi')
    pi_FI += per_window_stats(haps, pos_bp, fol_alt, None, 'pi')
    n_ok += 1
    if (rep + 1) % 50 == 0:
        print(f"  {rep + 1}/{NREPS}  elapsed={time.time()-t0:.0f}s")

if n_ok > 0:
    for arr in (dxy_kf_same, dxy_kf_alt, dxy_fs_fi, pi_K, pi_FS, pi_FI):
        arr /= n_ok

# Net divergence (Da) — subtracts shared pi
da_kf_same = dxy_kf_same - (pi_K + pi_FS) / 2
da_kf_alt = dxy_kf_alt - (pi_K + pi_FI) / 2
da_fs_fi = dxy_fs_fi - (pi_FS + pi_FI) / 2

print(f"\nDone: {n_ok}/{NREPS} reps in {time.time()-t0:.0f}s")

# ---- Plot ----
def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')


wins = np.linspace(0, L, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, hspace=0.30)

c_kf_same = '#2E7D32'   # K vs F-same (greenish)
c_fs_fi = '#FF8F00'      # Fol within (orange)
c_kf_alt = '#00838F'    # K vs F-alt (teal)


def shade_inv(ax):
    for (l, rt), lbl in [(inv_3Ra, '3Ra'), (inv_3Rb, '3Rb')]:
        ax.axvspan(l, rt, alpha=0.10, color='#90A4AE', zorder=0)
        ax.axvline(l, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.axvline(rt, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.text((l + rt) / 2, ax.get_ylim()[1] * 0.95, lbl,
                ha='center', va='top', fontsize=9, fontstyle='italic',
                color='#546E7A')


# Panel A: dxy
ax_dxy = fig.add_subplot(gs[0])
ax_dxy.plot(mid, smooth(dxy_kf_same), '-', color=c_kf_same, lw=2,
            label=r'K vs F$_S$ (same karyotype)')
ax_dxy.plot(mid, smooth(dxy_fs_fi), '-', color=c_fs_fi, lw=2,
            label=r'F$_S$ vs F$_I$ (within Folonzo)')
ax_dxy.plot(mid, smooth(dxy_kf_alt), '-', color=c_kf_alt, lw=2,
            label=r'K vs F$_I$ (alt karyotype)')
shade_inv(ax_dxy)
ax_dxy.set_ylabel(r'$d_{XY}$ (per bp)', fontsize=11)
ax_dxy.set_xlim(0, L)
ax_dxy.legend(fontsize=9, loc='upper right')
ax_dxy.set_title(
    r'A.  Absolute divergence $d_{XY}$',
    fontsize=11, fontweight='bold', loc='left')
ax_dxy.tick_params(labelbottom=False)

# Panel B: Da on the SAME y-axis as Panel A so the magnitude
# difference (Da is dxy minus avg pi → much smaller absolute value)
# is visually obvious. This avoids the autoscale trap where Da and
# dxy can look "the same magnitude" just because each panel
# auto-fits its own range.
ax_da = fig.add_subplot(gs[1], sharey=ax_dxy)
ax_da.plot(mid, smooth(da_kf_same), '-', color=c_kf_same, lw=2,
           label=r'K vs F$_S$')
ax_da.plot(mid, smooth(da_fs_fi), '-', color=c_fs_fi, lw=2,
           label=r'F$_S$ vs F$_I$')
ax_da.plot(mid, smooth(da_kf_alt), '-', color=c_kf_alt, lw=2,
           label=r'K vs F$_I$')
shade_inv(ax_da)
ax_da.axhline(0, color='#555', ls=':', lw=0.8)
ax_da.set_ylabel(r'$D_a = d_{XY} - (\pi_A + \pi_B)/2$ (per bp)',
                  fontsize=11)
ax_da.set_xlabel('Position (bp)', fontsize=10)
ax_da.set_xlim(0, L)
ax_da.legend(fontsize=9, loc='upper right')
ax_da.set_title(
    r'B.  Net divergence $D_a$ (same y-axis as A → magnitude visible)',
    fontsize=11, fontweight='bold', loc='left')

annot = (
    f'msinv v0.3.0 (hull algorithm).  '
    f'Ne_K = Ne_F = Ne_Anc = {Ne:,}, '
    f'T$_{{split}}$ = {t_split_gen:,} gen, '
    f'T$_{{inv}}$ = {t_inv_gen:,} gen.  '
    f'No gene flux ($\\gamma=0$). '
    f'{n_ok} replicates.'
)
fig.text(0.5, -0.02, annot, ha='center', fontsize=8,
         fontstyle='italic', color='#455A64',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1',
                   edgecolor='#B0BEC5', alpha=0.85))

fig.suptitle(
    'An. funestus Kir/Fol — 3Ra + 3Rb inversion divergence (hull simulator)',
    fontsize=12, fontweight='bold', y=0.99)

fig.savefig('figures/empirical_kir_fol.pdf',
            bbox_inches='tight', dpi=150)
print('Saved: figures/empirical_kir_fol.pdf')

# ---- Summary table ----
print('\nIn-inv vs collinear ratios:')
inv_w = [w for w in range(NW)
         if (inv_3Ra[0] < mid[w] < inv_3Ra[1]) or
            (inv_3Rb[0] < mid[w] < inv_3Rb[1])]
col_w = [w for w in range(NW) if w not in inv_w]
print(f"  {'metric':<14} {'inv':>10} {'col':>10} {'ratio':>10}")
for label, arr in [('dxy K-Fs', dxy_kf_same),
                    ('dxy Fs-Fi', dxy_fs_fi),
                    ('dxy K-Fi', dxy_kf_alt),
                    ('Da K-Fs', da_kf_same),
                    ('Da Fs-Fi', da_fs_fi),
                    ('Da K-Fi', da_kf_alt)]:
    i_m = float(np.mean([arr[w] for w in inv_w]))
    c_m = float(np.mean([arr[w] for w in col_w]))
    ratio = i_m / c_m if c_m != 0 else float('nan')
    print(f"  {label:<14} {i_m:>10.6g} {c_m:>10.6g} {ratio:>10.2f}")
