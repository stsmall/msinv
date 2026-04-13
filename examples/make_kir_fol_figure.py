#!/usr/bin/env python3
"""
Generate presentation figure: msinv simulation of Kir/Fol 3Ra+3Rb
alongside empirical pattern from Small et al. (2023) PNAS Fig S13.

Combines all gamma values (since flux has no effect) into one
pooled simulation, then plots:
  Row 1: Simulated dxy (3 comparisons)
  Row 2: Simulated Fst (2 comparisons)
  Row 3: Empirical Fig S13 image (chromosome 3 only)
"""
import sys, os
import msinv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ======================================================
# Parameters from Small et al. (2023) Table S8
# ======================================================
Ne_K = 70_000
Ne_F = 3_000_000
Ne_Anc = 44_000
N0 = Ne_Anc
t_split = 14_000 / (2 * N0)
t_inv = 385_000 / (2 * N0)
theta = 62.5
rho = 704
nsites = 1000
p_inv_anc = 0.3
size_K = Ne_K / N0
size_F = Ne_F / N0

# Folonzo exponential growth from Ne_Anc to Ne_F over t_split_gen
import math
t_split_gen = 14_000
g_F_coal = math.log(Ne_F / Ne_Anc) / t_split_gen * 2 * N0

# Two inversions on chr 3R
inv_3Ra = (0.15, 0.45)
inv_3Rb = (0.55, 0.85)

# Samples
n_kir = 10; n_fol_S = 5; n_fol_I = 5
nsam = n_kir + n_fol_S + n_fol_I


class KirFolTraj:
    def __init__(self):
        self.n_pops = 2
        self.t_inv = t_inv
        self.t_split = t_split
    def __call__(self, t, pop=0):
        if t >= t_inv: return 0.0
        if t >= t_split: return p_inv_anc
        return 0.0 if pop == 0 else p_inv_anc


kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, nsam))

NW = 40  # finer windows for smoother curves
wins = np.linspace(0, nsites, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2


def compute_dxy(haps, gA, gB, pos_arr):
    dxy = np.zeros(NW)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0: continue
        d = sum(1 for j in np.where(mask)[0]
                for a in gA for b in gB if haps[a, j] != haps[b, j])
        dxy[w] = d / (len(gA) * len(gB))
    return dxy


def compute_fst(haps, gA, gB, pos_arr):
    fst = np.zeros(NW)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0: continue
        pi_a = pi_b = dxy_val = 0
        nA, nB = len(gA), len(gB)
        for j in np.where(mask)[0]:
            cA = sum(haps[a, j] for a in gA)
            cB = sum(haps[b, j] for b in gB)
            pA, pB = cA / nA, cB / nB
            pi_a += 2 * pA * (1 - pA) * nA / (nA - 1) if nA > 1 else 0
            pi_b += 2 * pB * (1 - pB) * nB / (nB - 1) if nB > 1 else 0
            dxy_val += pA * (1 - pB) + pB * (1 - pA)
        pi_w = (pi_a + pi_b) / 2
        if dxy_val > 0:
            fst[w] = max(0, 1 - pi_w / dxy_val)
    return fst


# ======================================================
# Run pooled simulation (gamma=0, since flux doesn't matter)
# ======================================================
NR = 300
print(f"Running {NR} replicates with real Kir/Fol parameters...")

dxy_kf_same = np.zeros(NW)
dxy_f_si = np.zeros(NW)
dxy_kf_alt = np.zeros(NW)
fst_kf_same = np.zeros(NW)
fst_kf_alt = np.zeros(NW)
n_ok = 0

for rep in range(NR):
    traj = KirFolTraj()
    demo = msinv.Demography(n_pops=2, mig_rate=0.0)
    demo.pop_sizes[0] = size_K
    demo.pop_sizes[1] = size_F
    demo.growth_rates[1] = g_F_coal  # exponential growth in Fol
    demo.growth_start[1] = 0.0
    demo.snapshot_initial_state()
    demo.add_event(('eg', t_split, 1, 0.0))
    demo.add_event(('en', t_split, 1, 1.0))
    demo.add_event(('ej', t_split, 0, 1))

    sc = {('S', 0): n_kir, ('S', 1): n_fol_S, ('I', 1): n_fol_I}

    inv1 = msinv.InversionSpec(
        inv_3Ra[0], inv_3Ra[1], p_inv=p_inv_anc, c=0.01, gamma=0.0,
        t_inv=t_inv, trajectory=traj)
    inv2 = msinv.InversionSpec(
        inv_3Rb[0], inv_3Rb[1], p_inv=p_inv_anc, c=0.01, gamma=0.0,
        t_inv=t_inv, trajectory=traj)

    sim = msinv.MsinvSimulator(
        nsam=nsam, nreps=1, theta=theta, rho=rho, nsites=nsites,
        n_std=n_kir + n_fol_S, n_inv=n_fol_I,
        inversions=[inv1, inv2],
        p_inv_func=traj, seed=42 + rep,
        n_pops=2, mig_rate=0.0,
        sample_config=sc, demography=demo)

    try:
        pos, haps = sim.simulate_one()
    except Exception:
        continue
    if len(pos) == 0:
        continue

    pa = np.array(pos) * nsites
    dxy_kf_same += compute_dxy(haps, kir, fol_same, pa)
    dxy_f_si += compute_dxy(haps, fol_same, fol_alt, pa)
    dxy_kf_alt += compute_dxy(haps, kir, fol_alt, pa)
    fst_kf_same += compute_fst(haps, kir, fol_same, pa)
    fst_kf_alt += compute_fst(haps, kir, fol_alt, pa)
    n_ok += 1

    if (rep + 1) % 50 == 0:
        print(f"  {rep + 1}/{NR} done")

if n_ok > 0:
    dxy_kf_same /= n_ok
    dxy_f_si /= n_ok
    dxy_kf_alt /= n_ok
    fst_kf_same /= n_ok
    fst_kf_alt /= n_ok

print(f"Completed: {n_ok}/{NR} replicates")

# Smooth with rolling mean
def smooth(y, k=3):
    return np.convolve(y, np.ones(k)/k, mode='same')

dxy_kf_same_s = smooth(dxy_kf_same)
dxy_f_si_s = smooth(dxy_f_si)
dxy_kf_alt_s = smooth(dxy_kf_alt)
fst_kf_same_s = smooth(fst_kf_same)
fst_kf_alt_s = smooth(fst_kf_alt)


# ======================================================
# Figure
# ======================================================
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)

# Colors matching empirical: K-FS (green/teal), K-FI (darker), FS-FI (orange)
c_kf_same = '#2E7D32'   # K vs F-same (like K-FS in paper)
c_f_si = '#FF8F00'       # F-same vs F-alt (within F)
c_kf_alt = '#00838F'     # K vs F-alt (like K-FI in paper)

def shade_inversions(ax):
    for (l, r), lbl in [(inv_3Ra, '3Ra'), (inv_3Rb, '3Rb')]:
        ax.axvspan(l * nsites, r * nsites, alpha=0.12, color='#90A4AE',
                   zorder=0)
        ax.axvline(l * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.axvline(r * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.text((l + r) / 2 * nsites, ax.get_ylim()[1] * 0.97, lbl,
                ha='center', va='top', fontsize=9, fontstyle='italic',
                color='#546E7A')

# --- Panel A: dxy ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(mid, dxy_kf_same_s, '-', color=c_kf_same, lw=2,
         label=r'K vs F$_S$ (same karyotype)')
ax1.plot(mid, dxy_f_si_s, '-', color=c_f_si, lw=2,
         label=r'F$_S$ vs F$_I$ (within Folonzo)')
ax1.plot(mid, dxy_kf_alt_s, '-', color=c_kf_alt, lw=2,
         label=r'K vs F$_I$ (alt karyotype)')
shade_inversions(ax1)
ax1.set_ylabel('$d_{XY}$', fontsize=12)
ax1.set_xlim(0, nsites)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.set_title('A.  Simulated absolute divergence ($d_{XY}$)', fontsize=11,
              fontweight='bold', loc='left')
ax1.tick_params(labelbottom=False)

# --- Panel B: Fst ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(mid, fst_kf_same_s, '-', color=c_kf_same, lw=2,
         label=r'K vs F$_S$')
ax2.plot(mid, fst_kf_alt_s, '-', color=c_kf_alt, lw=2,
         label=r'K vs F$_I$')
shade_inversions(ax2)
ax2.set_ylabel('$F_{ST}$', fontsize=12)
ax2.set_xlim(0, nsites)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.set_title('B.  Simulated relative divergence ($F_{ST}$)', fontsize=11,
              fontweight='bold', loc='left')
ax2.set_xlabel('Simulated position (sites)', fontsize=10)

# --- Panel C: Empirical from Fig S13 ---
ax3 = fig.add_subplot(gs[2])
try:
    img = Image.open('/tmp/fig_s13.png')
    # Crop to chromosome 3 region (approximate pixel coords from the image)
    w, h = img.size
    # The figure has 3 chromosomes side by side; chr 3 is in the middle
    chr3_left = int(w * 0.34)
    chr3_right = int(w * 0.66)
    chr3_top = int(h * 0.02)
    chr3_bottom = int(h * 0.52)
    img_chr3 = img.crop((chr3_left, chr3_top, chr3_right, chr3_bottom))
    ax3.imshow(img_chr3, aspect='auto')
    ax3.set_title('C.  Empirical divergence, chromosome 3 '
                  '(Small et al. 2023, Fig. S13)',
                  fontsize=11, fontweight='bold', loc='left')
except Exception as e:
    ax3.text(0.5, 0.5, f'Could not load empirical figure:\n{e}',
             transform=ax3.transAxes, ha='center')
    ax3.set_title('C.  Empirical (Fig S13)', fontsize=11,
                  fontweight='bold', loc='left')
ax3.axis('off')

# Parameter annotation
param_text = (
    f'Parameters (Table S8): Ne$_K$={Ne_K:,}, Ne$_F$={Ne_F:,}, '
    f'Ne$_{{Anc}}$={Ne_Anc:,}\n'
    f'T$_{{split}}$=14,000 gen (~1,300 yr), T$_{{inv}}$~385,000 gen (~35 kyr)\n'
    f'$\\theta$={theta:.1f}, $\\rho$={rho:.0f} (100 kb region), '
    f'gene flux decoupled ($\\gamma$=0, no effect on pattern)\n'
    f'Kir: fixed homokaryotype, Fol: polymorphic (p={p_inv_anc}), '
    f'no migration (best-fit IIM), {n_ok} replicates'
)
fig.text(0.5, 0.01, param_text, ha='center', fontsize=8,
         fontstyle='italic', color='#546E7A',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1',
                   edgecolor='#B0BEC5', alpha=0.8))

fig.suptitle(
    'An. funestus: Kiribina vs Folonzo divergence at 3Ra + 3Rb inversions\n'
    'msinv coalescent simulation with real demographic parameters',
    fontsize=13, fontweight='bold', y=0.98)

fig.savefig('figures/kir_fol_presentation.pdf', bbox_inches='tight', dpi=150)
print(f"\nFigure saved: figures/kir_fol_presentation.pdf")

# Print summary stats
inv_w = [w for w in range(NW) if (inv_3Ra[0]*nsites < mid[w] < inv_3Ra[1]*nsites)
         or (inv_3Rb[0]*nsites < mid[w] < inv_3Rb[1]*nsites)]
col_w = [w for w in range(NW) if w not in inv_w]

print("\nSummary statistics:")
for lbl, d in [('dxy K-F_same', dxy_kf_same), ('dxy F_S-F_I', dxy_f_si),
                ('dxy K-F_alt', dxy_kf_alt)]:
    i_m = np.mean([d[w] for w in inv_w])
    c_m = np.mean([d[w] for w in col_w])
    print(f"  {lbl}: inv={i_m:.2f} col={c_m:.2f} ratio={i_m/c_m:.2f}")
for lbl, f in [('Fst K-F_same', fst_kf_same), ('Fst K-F_alt', fst_kf_alt)]:
    i_m = np.mean([f[w] for w in inv_w])
    c_m = np.mean([f[w] for w in col_w])
    print(f"  {lbl}: inv={i_m:.3f} col={c_m:.3f}")
