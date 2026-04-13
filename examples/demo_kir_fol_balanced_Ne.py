#!/usr/bin/env python3
"""
Demonstration: dxy depression inside inversions is caused by extreme
Ne asymmetry under the structured coalescent, NOT a bug in msinv.

Compare Kir/Fol with three levels of Ne_F/Ne_K asymmetry:

  Scenario 1: balanced Ne (Ne_K = Ne_F = 44,000)
              -> dxy K-F_same SHOULD be ~flat across inversion
              -> dxy K-F_alt SHOULD be elevated inside inversion

  Scenario 2: moderate asymmetry (Ne_K=70k, Ne_F=100k)
              -> dxy K-F_same slightly depressed inside inversion
              -> dxy K-F_alt clearly elevated

  Scenario 3: extreme asymmetry (real Kir/Fol: Ne_K=70k, Ne_F=3M)
              -> dxy K-F_same strongly depressed (the documented issue)
              -> dxy K-F_alt elevated

If scenario 1 gives the expected empirical-like pattern and scenario 3
gives the depression, the depression is demonstrably a *model-choice
consequence* of the structured coalescent under extreme Ne_F, not a
bug. The same code produces the correct pattern with balanced Ne.
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import msinv


# ---- Common parameters ----
Ne_Anc = 44_000
N0 = Ne_Anc
t_split_gen = 14_000
t_inv_gen = 385_000
t_split = t_split_gen / (2 * N0)
t_inv = t_inv_gen / (2 * N0)
L_bp = 100_000
mu = 3.55e-9
r = 4.0e-8
theta = 4 * N0 * mu * L_bp
rho = 4 * N0 * r * L_bp
nsites = 1000
p_inv_anc = 0.3
inv_3Ra = (0.15, 0.45)
inv_3Rb = (0.55, 0.85)

n_kir = 10
n_fol_S = 5
n_fol_I = 5
nsam = n_kir + n_fol_S + n_fol_I

NR = 200
NW = 30
wins = np.linspace(0, nsites, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2

kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, nsam))


class KirFolTraj:
    def __init__(self):
        self.n_pops = 2
        self.t_inv = t_inv
        self.t_split = t_split

    def __call__(self, t, pop=0):
        if t >= t_inv:
            return 0.0
        if t >= t_split:
            return p_inv_anc
        return 0.0 if pop == 0 else p_inv_anc


def compute_dxy(haps, gA, gB, pos_arr):
    out = np.zeros(NW)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0:
            continue
        d = 0
        for j in np.where(mask)[0]:
            for a in gA:
                for b in gB:
                    if haps[a, j] != haps[b, j]:
                        d += 1
        out[w] = d / (len(gA) * len(gB))
    return out


def run(label, Ne_K, Ne_F):
    """Run Kir/Fol with given Ne pair."""
    size_K = Ne_K / N0
    size_F = Ne_F / N0
    # Folonzo exponential growth only if Ne_F != Ne_Anc
    if abs(Ne_F - Ne_Anc) > 1:
        g_F_coal = math.log(Ne_F / Ne_Anc) / t_split_gen * 2 * N0
    else:
        g_F_coal = 0.0

    print(f"\n{label}: Ne_K={Ne_K:,}  Ne_F={Ne_F:,}  ratio={Ne_F/Ne_K:.1f}")
    dxy_kfs = np.zeros(NW)
    dxy_fsi = np.zeros(NW)
    dxy_kfi = np.zeros(NW)
    n_ok = 0

    for rep in range(NR):
        traj = KirFolTraj()
        demo = msinv.Demography(n_pops=2, mig_rate=0.0)
        demo.pop_sizes[0] = size_K
        demo.pop_sizes[1] = size_F
        if g_F_coal > 0:
            demo.growth_rates[1] = g_F_coal
            demo.growth_start[1] = 0.0
        demo.snapshot_initial_state()
        if g_F_coal > 0:
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
        dxy_kfs += compute_dxy(haps, kir, fol_same, pa)
        dxy_fsi += compute_dxy(haps, fol_same, fol_alt, pa)
        dxy_kfi += compute_dxy(haps, kir, fol_alt, pa)
        n_ok += 1
        if (rep + 1) % 50 == 0:
            print(f"  {rep+1}/{NR}")
    if n_ok > 0:
        dxy_kfs /= n_ok
        dxy_fsi /= n_ok
        dxy_kfi /= n_ok

    # summary
    inv_w = [w for w in range(NW) if
             (inv_3Ra[0] * nsites < mid[w] < inv_3Ra[1] * nsites)
             or (inv_3Rb[0] * nsites < mid[w] < inv_3Rb[1] * nsites)]
    col_w = [w for w in range(NW) if w not in inv_w]

    def stats(d):
        i_m = np.mean([d[w] for w in inv_w])
        c_m = np.mean([d[w] for w in col_w])
        return i_m, c_m, (i_m / c_m if c_m > 0 else 0)

    ikfs, ckfs, rkfs = stats(dxy_kfs)
    ifsi, cfsi, rfsi = stats(dxy_fsi)
    ikfi, ckfi, rkfi = stats(dxy_kfi)
    print(f"  n={n_ok}  K-Fs: inv={ikfs:.2f} col={ckfs:.2f} ratio={rkfs:.2f}")
    print(f"         Fs-Fi: inv={ifsi:.2f} col={cfsi:.2f} ratio={rfsi:.2f}")
    print(f"         K-Fi: inv={ikfi:.2f} col={ckfi:.2f} ratio={rkfi:.2f}")

    return dict(kfs=dxy_kfs, fsi=dxy_fsi, kfi=dxy_kfi,
                ratio_kfs=rkfs, ratio_fsi=rfsi, ratio_kfi=rkfi,
                n=n_ok)


# ---- Run three scenarios ----
scenarios = [
    ('balanced\nNe_K = Ne_F = 44k', 44_000, 44_000),
    ('moderate asymmetry\nNe_K=70k, Ne_F=100k', 70_000, 100_000),
    ('extreme asymmetry\n(real Kir/Fol)\nNe_K=70k, Ne_F=3M', 70_000, 3_000_000),
]

results = {label: run(label, Ne_K, Ne_F)
           for label, Ne_K, Ne_F in scenarios}


# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey='row')

c_kfs = '#2E7D32'
c_fsi = '#FF8F00'
c_kfi = '#00838F'


def shade(ax):
    for (l, r_) in [inv_3Ra, inv_3Rb]:
        ax.axvspan(l * nsites, r_ * nsites, alpha=0.12, color='#90A4AE')
        ax.axvline(l * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)
        ax.axvline(r_ * nsites, color='#78909C', ls='--', alpha=0.4, lw=0.8)


for i, (label, res) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(mid, res['kfs'], '-', color=c_kfs, lw=1.8,
            label=r'K vs F$_S$ (same kary.)')
    ax.plot(mid, res['fsi'], '-', color=c_fsi, lw=1.8,
            label=r'F$_S$ vs F$_I$ (within Fol)')
    ax.plot(mid, res['kfi'], '-', color=c_kfi, lw=1.8,
            label=r'K vs F$_I$ (alt kary.)')
    shade(ax)
    ax.set_title(label, fontsize=10, fontweight='bold')
    # Annotate ratios
    ax.text(0.02, 0.97,
            f"ratio (inv/col):\n"
            f"  K-F_s: {res['ratio_kfs']:.2f}\n"
            f"  F_s-F_i: {res['ratio_fsi']:.2f}\n"
            f"  K-F_i: {res['ratio_kfi']:.2f}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.85))
    ax.set_xlabel('Position (sites)', fontsize=10)
    if i == 0:
        ax.set_ylabel(r'$d_{XY}$', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')

fig.suptitle(
    'dxy depression inside inversions emerges only under extreme Ne asymmetry\n'
    '(Structured coalescent: S-class coal rate $\\propto 1/(p_{std} \\cdot N_e)$; '
    'asymmetric $N_e$ $\\Rightarrow$ asymmetric T_MRCA inv vs col)',
    fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/demo_kir_fol_Ne_balance.pdf', bbox_inches='tight', dpi=150)
print('\nSaved: figures/demo_kir_fol_Ne_balance.pdf')

# Final ratio summary
print('\n' + '=' * 70)
print('SUMMARY: inv/col dxy ratios (ratio ~1 = flat, <1 = depression)')
print('=' * 70)
print(f"{'Scenario':<40} {'K-F_s':>8} {'F_s-F_i':>8} {'K-F_i':>8}")
for label, res in results.items():
    short = label.split('\n')[0]
    print(f"{short:<40} {res['ratio_kfs']:>8.2f} "
          f"{res['ratio_fsi']:>8.2f} {res['ratio_kfi']:>8.2f}")
print()
print('Expected empirical pattern (Fig S13): K-F_s ~1.0 (flat), '
      'F_s-F_i > 1, K-F_i > 1.')
