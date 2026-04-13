#!/usr/bin/env python3
"""
Kir/Fol presentation figure with ALL diagnostics:

  Row A: dxy across chromosome (raw absolute divergence — swamped by
         shared pi_Fol)
  Row B: Da = dxy - (pi_A + pi_B)/2  (net divergence — isolates the
         inversion signal; matches empirical Fig S13 pattern)
  Row C: FST across chromosome (relative divergence)
  Row D: PCA of haplotypes INSIDE 3Ra vs OUTSIDE — demonstrates that
         inside the inversion, samples cluster by karyotype
         (K+Fol_same on one side, Fol_alt on the other) while outside
         they mix.
  Row E: Empirical Fig S13 (cropped chromosome 3)
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.decomposition import PCA
import msinv

# ---- Parameters (Small et al. 2023 Table S8) ----
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
t_split_gen = 14_000
g_F_coal = math.log(Ne_F / Ne_Anc) / t_split_gen * 2 * N0

inv_3Ra = (0.15, 0.45)
inv_3Rb = (0.55, 0.85)

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

NW = 40
wins = np.linspace(0, nsites, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2


def compute_dxy(haps, gA, gB, pa):
    dxy = np.zeros(NW)
    for w in range(NW):
        mask = (pa >= wins[w]) & (pa < wins[w + 1])
        if mask.sum() == 0: continue
        d = sum(1 for j in np.where(mask)[0]
                for a in gA for b in gB if haps[a, j] != haps[b, j])
        dxy[w] = d / (len(gA) * len(gB))
    return dxy


def compute_pi(haps, grp, pa):
    pi = np.zeros(NW)
    n = len(grp)
    if n < 2:
        return pi
    for w in range(NW):
        mask = (pa >= wins[w]) & (pa < wins[w + 1])
        if mask.sum() == 0: continue
        d = 0
        for j in np.where(mask)[0]:
            for a in range(n):
                for b in range(a + 1, n):
                    if haps[grp[a], j] != haps[grp[b], j]:
                        d += 1
        pi[w] = d / (n * (n - 1) / 2)
    return pi


def compute_fst(haps, gA, gB, pa):
    fst = np.zeros(NW)
    for w in range(NW):
        mask = (pa >= wins[w]) & (pa < wins[w + 1])
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


# ---- Run replicates ----
NR = 300
print(f"Running {NR} replicates with real Kir/Fol parameters...")

dxy_kf_same = np.zeros(NW)
dxy_f_si = np.zeros(NW)
dxy_kf_alt = np.zeros(NW)
pi_K = np.zeros(NW)
pi_Fs = np.zeros(NW)
pi_Fi = np.zeros(NW)
fst_kf_same = np.zeros(NW)
fst_kf_alt = np.zeros(NW)
fst_f_si = np.zeros(NW)
# PCA: stack per-rep haplotype matrices inside and outside 3Ra
inside_matrices = []
outside_matrices = []
n_ok = 0

for rep in range(NR):
    traj = KirFolTraj()
    demo = msinv.Demography(n_pops=2, mig_rate=0.0)
    demo.pop_sizes[0] = size_K
    demo.pop_sizes[1] = size_F
    demo.growth_rates[1] = g_F_coal
    demo.growth_start[1] = 0.0
    demo.snapshot_initial_state()
    demo.add_event(('eg', t_split, 1, 0.0))
    demo.add_event(('en', t_split, 1, 1.0))
    demo.add_event(('ej', t_split, 0, 1))

    sc = {('S', 0): n_kir, ('S', 1): n_fol_S, ('I', 1): n_fol_I}

    inv1 = msinv.InversionSpec(
        inv_3Ra[0], inv_3Ra[1], p_inv=p_inv_anc, c=0.0, gamma=0.0,
        t_inv=t_inv, trajectory=traj)
    inv2 = msinv.InversionSpec(
        inv_3Rb[0], inv_3Rb[1], p_inv=p_inv_anc, c=0.0, gamma=0.0,
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
    pi_K += compute_pi(haps, kir, pa)
    pi_Fs += compute_pi(haps, fol_same, pa)
    pi_Fi += compute_pi(haps, fol_alt, pa)
    fst_kf_same += compute_fst(haps, kir, fol_same, pa)
    fst_kf_alt += compute_fst(haps, kir, fol_alt, pa)
    fst_f_si += compute_fst(haps, fol_same, fol_alt, pa)
    # PCA data: sites INSIDE 3Ra and OUTSIDE all inversions
    in_mask = (pa >= inv_3Ra[0] * nsites) & (pa < inv_3Ra[1] * nsites)
    out_mask = ((pa < inv_3Ra[0] * nsites) | (pa >= inv_3Rb[1] * nsites))
    if in_mask.sum() > 0:
        inside_matrices.append(haps[:, in_mask])
    if out_mask.sum() > 0:
        outside_matrices.append(haps[:, out_mask])
    n_ok += 1

    if (rep + 1) % 50 == 0:
        print(f"  {rep + 1}/{NR} done")

if n_ok > 0:
    for arr in (dxy_kf_same, dxy_f_si, dxy_kf_alt,
                pi_K, pi_Fs, pi_Fi,
                fst_kf_same, fst_kf_alt, fst_f_si):
        arr /= n_ok

# Da = dxy - (pi_A + pi_B)/2
da_kf_same = dxy_kf_same - (pi_K + pi_Fs) / 2
da_f_si = dxy_f_si - (pi_Fs + pi_Fi) / 2
da_kf_alt = dxy_kf_alt - (pi_K + pi_Fi) / 2

print(f"Completed: {n_ok}/{NR} replicates")


def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')


# ---- Figure ----
fig = plt.figure(figsize=(14, 18))
gs = GridSpec(5, 2, height_ratios=[1, 1, 1, 1.2, 1],
              width_ratios=[1, 1],
              hspace=0.45, wspace=0.3)

c_kf_same = '#2E7D32'
c_f_si = '#FF8F00'
c_kf_alt = '#00838F'


def shade_inv(ax):
    for (l, r_), lbl in [(inv_3Ra, '3Ra'), (inv_3Rb, '3Rb')]:
        ax.axvspan(l * nsites, r_ * nsites, alpha=0.1, color='#90A4AE',
                   zorder=0)
        ax.axvline(l * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.axvline(r_ * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.text((l + r_) / 2 * nsites, ax.get_ylim()[1] * 0.95, lbl,
                ha='center', va='top', fontsize=9, fontstyle='italic',
                color='#546E7A')


# A. dxy (raw) — full width
ax_dxy = fig.add_subplot(gs[0, :])
ax_dxy.plot(mid, smooth(dxy_kf_same), '-', color=c_kf_same, lw=2,
            label=r'K vs F$_S$ (same kary.)')
ax_dxy.plot(mid, smooth(dxy_f_si), '-', color=c_f_si, lw=2,
            label=r'F$_S$ vs F$_I$ (within Fol)')
ax_dxy.plot(mid, smooth(dxy_kf_alt), '-', color=c_kf_alt, lw=2,
            label=r'K vs F$_I$ (alt kary.)')
shade_inv(ax_dxy)
ax_dxy.set_ylabel(r'$d_{XY}$', fontsize=12)
ax_dxy.set_xlim(0, nsites)
ax_dxy.legend(fontsize=9, loc='lower center', ncol=3)
ax_dxy.set_title(
    r'A.  Raw absolute divergence $d_{XY}$ '
    r'— swamped by $\pi_{Fol}$ with Ne$_F$=3M '
    r'(shared pi dominates)',
    fontsize=11, fontweight='bold', loc='left')
ax_dxy.tick_params(labelbottom=False)

# B. Da = dxy - (pi_A + pi_B)/2 — full width
ax_da = fig.add_subplot(gs[1, :])
ax_da.plot(mid, smooth(da_kf_same), '-', color=c_kf_same, lw=2,
           label=r'K vs F$_S$')
ax_da.plot(mid, smooth(da_f_si), '-', color=c_f_si, lw=2,
           label=r'F$_S$ vs F$_I$')
ax_da.plot(mid, smooth(da_kf_alt), '-', color=c_kf_alt, lw=2,
           label=r'K vs F$_I$')
shade_inv(ax_da)
ax_da.axhline(0, color='#555555', ls=':', lw=0.8)
ax_da.set_ylabel(r'$D_a = d_{XY} - (\pi_A+\pi_B)/2$', fontsize=11)
ax_da.set_xlim(0, nsites)
ax_da.legend(fontsize=9, loc='lower center', ncol=3)
ax_da.set_title(
    r'B.  Net divergence $D_a$ '
    r'(Nei 1987; subtracts shared $\pi$) '
    r'— isolates the inversion signal',
    fontsize=11, fontweight='bold', loc='left')
ax_da.tick_params(labelbottom=False)

# C. FST — full width
ax_fst = fig.add_subplot(gs[2, :])
ax_fst.plot(mid, smooth(fst_kf_same), '-', color=c_kf_same, lw=2,
            label=r'K vs F$_S$')
ax_fst.plot(mid, smooth(fst_f_si), '-', color=c_f_si, lw=2,
            label=r'F$_S$ vs F$_I$')
ax_fst.plot(mid, smooth(fst_kf_alt), '-', color=c_kf_alt, lw=2,
            label=r'K vs F$_I$')
shade_inv(ax_fst)
ax_fst.set_ylabel(r'$F_{ST}$', fontsize=12)
ax_fst.set_xlim(0, nsites)
ax_fst.legend(fontsize=9, loc='upper left')
ax_fst.set_title(
    r'C.  Relative divergence $F_{ST}$',
    fontsize=11, fontweight='bold', loc='left')
ax_fst.set_xlabel('Simulated position (sites)', fontsize=10)

# D. PCA panels — two side by side
# Concatenate haplotypes across reps for a single PCA (large sample)
def pca_panel(ax, matrices, title):
    if not matrices:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        return
    # Each matrix has shape (nsam, n_sites). Per-rep PCA over concatenated.
    # Simpler: horizontal concat.
    X = np.concatenate(matrices, axis=1)  # (nsam, total_sites)
    if X.shape[1] < 2:
        ax.text(0.5, 0.5, 'too few sites', transform=ax.transAxes, ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        return
    # Center
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xc)
    # Plot
    for idx, color, lbl in [(kir, c_kf_same, 'Kir (all S)'),
                             (fol_same, '#1976D2', 'Fol$_S$ (same kary.)'),
                             (fol_alt, c_kf_alt, 'Fol$_I$ (alt kary.)')]:
        ax.scatter(Z[idx, 0], Z[idx, 1], c=color, s=40,
                   edgecolor='black', lw=0.6, label=lbl)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.25)


ax_pca_in = fig.add_subplot(gs[3, 0])
pca_panel(ax_pca_in, inside_matrices,
          'D1.  PCA INSIDE 3Ra inversion — karyotype clustering')

ax_pca_out = fig.add_subplot(gs[3, 1])
pca_panel(ax_pca_out, outside_matrices,
          'D2.  PCA OUTSIDE inversions — no karyotype structure')


# E. Empirical Fig S13
ax_emp = fig.add_subplot(gs[4, :])
try:
    img = Image.open('/tmp/fig_s13.png')
    w, h = img.size
    chr3_left = int(w * 0.34)
    chr3_right = int(w * 0.66)
    chr3_top = int(h * 0.02)
    chr3_bottom = int(h * 0.52)
    img_chr3 = img.crop((chr3_left, chr3_top, chr3_right, chr3_bottom))
    ax_emp.imshow(img_chr3, aspect='auto')
    ax_emp.set_title(
        'E.  Empirical divergence, chromosome 3 (Small et al. 2023, Fig. S13)',
        fontsize=11, fontweight='bold', loc='left')
except Exception as e:
    ax_emp.text(0.5, 0.5, f'Could not load empirical figure:\n{e}',
                transform=ax_emp.transAxes, ha='center')
    ax_emp.set_title('E.  Empirical (Fig S13 not available)',
                     fontsize=11, fontweight='bold', loc='left')
ax_emp.axis('off')

# Parameter annotation
param_text = (
    f'Parameters (Table S8): Ne$_K$={Ne_K:,}, Ne$_F$={Ne_F:,}, '
    f'Ne$_{{Anc}}$={Ne_Anc:,} — EXTREME Ne asymmetry (ratio ~43x)\n'
    f'T$_{{split}}$=14,000 gen (~1,300 yr), T$_{{inv}}$~385,000 gen (~35 kyr). '
    f'$\\theta$={theta:.1f}, $\\rho$={rho:.0f} (100 kb). '
    f'Gene flux $\\gamma$=0 (recombination modifier only). '
    f'{n_ok} replicates.\n'
    f'KEY INSIGHT: $d_{{XY}}$ is swamped by high $\\pi_{{Fol}}$. '
    f'$D_a$ (net divergence) recovers the inversion signal that matches '
    f'empirical $F_{{ST}}$ and PCA patterns.'
)
fig.text(0.5, -0.01, param_text, ha='center', fontsize=8,
         fontstyle='italic', color='#455A64',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECEFF1',
                   edgecolor='#B0BEC5', alpha=0.85))

fig.suptitle(
    'An. funestus: Kiribina vs Folonzo divergence at 3Ra + 3Rb inversions\n'
    'msinv coalescent simulation with real demographic parameters',
    fontsize=13, fontweight='bold', y=0.995)

fig.savefig('figures/kir_fol_presentation.pdf',
            bbox_inches='tight', dpi=150)
print(f"\nFigure saved: figures/kir_fol_presentation.pdf")


# ---- Summary table ----
inv_w = [w for w in range(NW) if
         (inv_3Ra[0]*nsites < mid[w] < inv_3Ra[1]*nsites)
         or (inv_3Rb[0]*nsites < mid[w] < inv_3Rb[1]*nsites)]
col_w = [w for w in range(NW) if w not in inv_w]

print("\nSummary statistics (inv / col ratios):")
print(f"  {'metric':<18} {'comparison':<14} {'inv':>8} {'col':>8} {'ratio':>8}")
for label, d in [('dxy', [('K-F_same', dxy_kf_same),
                          ('F_S-F_I ', dxy_f_si),
                          ('K-F_alt ', dxy_kf_alt)]),
                  ('Da ', [('K-F_same', da_kf_same),
                           ('F_S-F_I ', da_f_si),
                           ('K-F_alt ', da_kf_alt)]),
                  ('Fst', [('K-F_same', fst_kf_same),
                           ('F_S-F_I ', fst_f_si),
                           ('K-F_alt ', fst_kf_alt)])]:
    for cmp, arr in d:
        i_m = np.mean([arr[w] for w in inv_w])
        c_m = np.mean([arr[w] for w in col_w])
        r_val = i_m / c_m if c_m != 0 else 0
        print(f"  {label:<18} {cmp:<14} {i_m:>8.3f} {c_m:>8.3f} {r_val:>8.2f}")
