#!/usr/bin/env python3
"""
Simulate An. funestus Kiribina/Folonzo 3Ra + 3Rb inversion scenario.

Parameters from Small et al. (2023) PNAS Table S8, Fig S11:
  Ne_K = 70,000   (rice specialist, restricted habitat)
  Ne_F = 3,000,000 (pan-African, habitat generalist, post-human expansion)
  Ne_Anc = 44,000  (ancestral, serial founder effects from S. Africa)
  T_split = 14,000 gen (~1,300 yr at 11 gen/yr)
  mu = 3.55e-9 /bp/gen,  r = 4.0e-8 /bp/gen  (r/mu ≈ 11)
  Inversions 3Ra, 3Rb: ~30,000-40,000 years old (>>330,000 gen)
  Migration: 0 in best-fit IIM model; <1% F2 hybrids observed

Question: Does having BOTH 3Ra and 3Rb help maintain the fixed
karyotype state in Kiribina?  Do the two inversions create a
stronger barrier to introgression than one alone?
"""
import sys, os
import msinv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# ======================================================
# Real parameters from Small et al. (2023)
# ======================================================
Ne_K = 70_000           # Kiribina
Ne_F = 3_000_000        # Folonzo (pan-African)
Ne_Anc = 44_000         # ancestral (serial founder effects)
N0 = Ne_Anc             # reference Ne for coalescent scaling

gen_per_year = 11
t_split_gen = 14_000    # ~1,300 years
t_inv_gen = 385_000     # ~35,000 years (middle of Fig S11 range)

# Coalescent units (2*N0 generations)
t_split = t_split_gen / (2 * N0)       # 0.159
t_inv = t_inv_gen / (2 * N0)           # 4.375

# Relative pop sizes (present-day)
size_K = Ne_K / N0        # 1.59
size_F = Ne_F / N0        # 68.18

# Folonzo exponential growth from Ne_Anc at t_split to Ne_F at present.
# Forward: N_F(t_fwd) = Ne_Anc * exp(g * t_fwd), t_fwd = gen since split
# At t_fwd = t_split_gen: exp(g * t_split_gen) = Ne_F / Ne_Anc
# g = ln(Ne_F/Ne_Anc) / t_split_gen per generation
import math
g_F_per_gen = math.log(Ne_F / Ne_Anc) / t_split_gen
# In coalescent units (scaled by 2*N0):
g_F_coal = g_F_per_gen * 2 * N0
# For Kir, modest founding bottleneck then constant (roughly):
g_K_coal = 0.0  # treat Kir as constant size

# Scaled rates for a 100kb region of chr 3R
L_bp = 100_000
mu = 3.55e-9
r = 4.0e-8
theta = 4 * N0 * mu * L_bp   # ~62.5
rho = 4 * N0 * r * L_bp       # ~704

nsites = 1000   # discretized positions

# Gene flux coefficient — effectively 0 since gamma=0 is set on InversionSpec
# (with the walk_segment fix, gamma=0 is now honored consistently).
c = 0.0

# Inversion positions on chr 3R (approximate, non-overlapping)
# 3Ra spans roughly the middle of 3R, 3Rb is adjacent/overlapping
# For simulation: place them as two distinct inversions
inv_3Ra_left = 0.15
inv_3Ra_right = 0.45
inv_3Rb_left = 0.55
inv_3Rb_right = 0.85

# Ancestral inversion frequency in Folonzo
p_inv_fol = 0.3   # polymorphic
p_inv_anc = 0.3

# Samples
n_kir = 10         # Kiribina: fixed for one homokaryotype
n_fol_S = 5        # Folonzo: same karyotype as Kir
n_fol_I = 5        # Folonzo: alternative arrangement
nsam = n_kir + n_fol_S + n_fol_I

NR = 150
SEED = 42

print("=" * 65)
print("An. funestus Kir/Fol: 3Ra + 3Rb (Small et al. 2023 parameters)")
print("=" * 65)
print(f"Ne_K={Ne_K:,}  Ne_F={Ne_F:,}  Ne_Anc={Ne_Anc:,}")
print(f"Relative sizes: K={size_K:.2f}  F={size_F:.1f}  (ref N0={N0:,})")
print(f"T_split={t_split_gen:,} gen = {t_split:.3f} coal units")
print(f"T_inv={t_inv_gen:,} gen = {t_inv:.2f} coal units (~35 kyr)")
print(f"theta={theta:.1f}  rho={rho:.0f}  (100kb region, r/mu={r/mu:.1f})")
print(f"3Ra: [{inv_3Ra_left:.2f}, {inv_3Ra_right:.2f}]")
print(f"3Rb: [{inv_3Rb_left:.2f}, {inv_3Rb_right:.2f}]")
print(f"Samples: Kir {n_kir}, Fol {n_fol_S}+{n_fol_I}")
print()


# ======================================================
# Per-population trajectory
# ======================================================
class KirFolTrajectory:
    """Kir fixed, Fol polymorphic, ancestral polymorphic."""
    def __init__(self, p_inv_fol, p_inv_anc, t_split, t_inv):
        self.p_inv_fol = p_inv_fol
        self.p_inv_anc = p_inv_anc
        self.t_split = t_split
        self.t_inv = t_inv
        self.n_pops = 2

    def __call__(self, t, pop=0):
        if t >= self.t_inv:
            return 0.0
        if t >= self.t_split:
            return self.p_inv_anc
        if pop == 0:
            return 0.0     # Kiribina: fixed
        return self.p_inv_fol  # Folonzo: polymorphic


# ======================================================
# Helper functions
# ======================================================
kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, nsam))

NW = 20
wins = np.linspace(0, nsites, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2


def compute_dxy_window(haps, grpA, grpB, pos_arr):
    dxy = np.zeros(NW)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0:
            continue
        d = 0
        for j in np.where(mask)[0]:
            for a in grpA:
                for b in grpB:
                    if haps[a, j] != haps[b, j]:
                        d += 1
        dxy[w] = d / (len(grpA) * len(grpB))
    return dxy


def compute_fst_window(haps, grpA, grpB, pos_arr):
    fst = np.zeros(NW)
    for w in range(NW):
        mask = (pos_arr >= wins[w]) & (pos_arr < wins[w + 1])
        if mask.sum() == 0:
            continue
        pi_a = pi_b = dxy_val = 0
        nA, nB = len(grpA), len(grpB)
        for j in np.where(mask)[0]:
            cA = sum(haps[a, j] for a in grpA)
            cB = sum(haps[b, j] for b in grpB)
            pA, pB = cA / nA, cB / nB
            pi_a += 2 * pA * (1 - pA) * nA / (nA - 1) if nA > 1 else 0
            pi_b += 2 * pB * (1 - pB) * nB / (nB - 1) if nB > 1 else 0
            dxy_val += pA * (1 - pB) + pB * (1 - pA)
        pi_w = (pi_a + pi_b) / 2
        if dxy_val > 0:
            fst[w] = max(0, 1 - pi_w / dxy_val)
    return fst


def run_scenario(label, mig_4Nm, use_both_inv=True):
    """Run simulation with given migration and 1 or 2 inversions."""
    print(f"\n--- {label} ---")

    traj = KirFolTrajectory(p_inv_fol, p_inv_anc, t_split, t_inv)

    # Demography: Folonzo exponentially grew from Ne_Anc at t_split
    # to Ne_F = 3M present. Kir constant at Ne_K = 70k.
    demo = msinv.Demography(n_pops=2, mig_rate=mig_4Nm)
    demo.pop_sizes[0] = size_K           # Kir constant
    demo.pop_sizes[1] = size_F           # Fol present-day
    demo.growth_rates[1] = g_F_coal      # Fol exponential growth
    demo.growth_start[1] = 0.0
    demo.snapshot_initial_state()
    demo.add_event(('eg', t_split, 1, 0.0))    # Fol stops growing at split
    demo.add_event(('en', t_split, 1, 1.0))    # Fol size = Ne_Anc at split
    demo.add_event(('ej', t_split, 0, 1))      # merge at split

    sc = {('S', 0): n_kir, ('S', 1): n_fol_S, ('I', 1): n_fol_I}

    if use_both_inv:
        # gamma=0: no gene flux (inversions act purely as recombination
        # barriers, matching the empirical Kir/Fol pattern where S/I
        # classes are cleanly separated)
        inv1 = msinv.InversionSpec(
            inv_3Ra_left, inv_3Ra_right, p_inv=p_inv_fol, c=0.0, gamma=0.0,
            t_inv=t_inv, trajectory=traj)
        inv2 = msinv.InversionSpec(
            inv_3Rb_left, inv_3Rb_right, p_inv=p_inv_fol, c=0.0, gamma=0.0,
            t_inv=t_inv, trajectory=traj)
        inversions = [inv1, inv2]
    else:
        inversions = None  # single inversion mode

    dxy1 = np.zeros(NW)
    dxy2 = np.zeros(NW)
    dxy3 = np.zeros(NW)
    fst1 = np.zeros(NW)
    fst2 = np.zeros(NW)  # F_S vs F_I
    fst3 = np.zeros(NW)
    n_ok = 0

    for rep in range(NR):
        if use_both_inv:
            sim = msinv.MsinvSimulator(
                nsam=nsam, nreps=1, theta=theta, rho=rho, nsites=nsites,
                n_std=n_kir + n_fol_S, n_inv=n_fol_I,
                inversions=inversions,
                p_inv_func=traj, seed=SEED + rep,
                n_pops=2, mig_rate=mig_4Nm,
                sample_config=sc, demography=demo)
        else:
            sim = msinv.MsinvSimulator(
                nsam=nsam, nreps=1, theta=theta, rho=rho, nsites=nsites,
                n_std=n_kir + n_fol_S, n_inv=n_fol_I,
                p_inv=p_inv_fol, c=0.0, gamma=0.0,
                bp_left=inv_3Ra_left, bp_right=inv_3Ra_right,
                p_inv_func=traj, seed=SEED + rep,
                n_pops=2, mig_rate=mig_4Nm,
                sample_config=sc, demography=demo,
                t_inv=t_inv)

        try:
            pos, haps = sim.simulate_one()
        except Exception:
            continue
        if len(pos) == 0:
            continue

        pos_arr = np.array(pos) * nsites
        dxy1 += compute_dxy_window(haps, kir, fol_same, pos_arr)
        dxy2 += compute_dxy_window(haps, fol_same, fol_alt, pos_arr)
        dxy3 += compute_dxy_window(haps, kir, fol_alt, pos_arr)
        fst1 += compute_fst_window(haps, kir, fol_same, pos_arr)
        fst2 += compute_fst_window(haps, fol_same, fol_alt, pos_arr)
        fst3 += compute_fst_window(haps, kir, fol_alt, pos_arr)
        n_ok += 1

        if (rep + 1) % 50 == 0:
            print(f"  {rep + 1}/{NR} done")

    if n_ok > 0:
        dxy1 /= n_ok; dxy2 /= n_ok; dxy3 /= n_ok
        fst1 /= n_ok; fst2 /= n_ok; fst3 /= n_ok

    # Region classification
    def region(x):
        if inv_3Ra_left * nsites < x < inv_3Ra_right * nsites:
            return '3Ra'
        elif inv_3Rb_left * nsites < x < inv_3Rb_right * nsites:
            return '3Rb'
        return 'col'

    inv_w = [w for w in range(NW) if region(mid[w]) in ('3Ra', '3Rb')]
    col_w = [w for w in range(NW) if region(mid[w]) == 'col']

    for lbl, d in [('Kir-Fol_same', dxy1), ('Fol_same-alt', dxy2),
                    ('Kir-Fol_alt', dxy3)]:
        i_m = np.mean([d[w] for w in inv_w]) if inv_w else 0
        c_m = np.mean([d[w] for w in col_w]) if col_w else 0
        print(f"  dxy {lbl}: inv={i_m:.2f} col={c_m:.2f}"
              f" ratio={i_m / c_m:.2f}" if c_m > 0 else "")
    print(f"  ({n_ok}/{NR} reps)")

    return dict(dxy_kir_fol_same=dxy1, dxy_fol_same_alt=dxy2,
                dxy_kir_fol_alt=dxy3, fst_kir_fol_same=fst1,
                fst_fol_same_alt=fst2, fst_kir_fol_alt=fst3, n_ok=n_ok)


# ======================================================
# Run scenarios
# ======================================================
results = {}

# 1. Both 3Ra + 3Rb, no migration
results['3Ra+3Rb\nno migration'] = run_scenario(
    '3Ra + 3Rb, no migration', mig_4Nm=0.0, use_both_inv=True)

# 2. Both 3Ra + 3Rb, low migration (4Nm=1)
results['3Ra+3Rb\n4Nm=1'] = run_scenario(
    '3Ra + 3Rb, 4Nm=1', mig_4Nm=1.0, use_both_inv=True)

# 3. Only 3Ra, no migration
results['3Ra only\nno migration'] = run_scenario(
    '3Ra only, no migration', mig_4Nm=0.0, use_both_inv=False)

# 4. Only 3Ra, low migration (4Nm=1)
results['3Ra only\n4Nm=1'] = run_scenario(
    '3Ra only, 4Nm=1', mig_4Nm=1.0, use_both_inv=False)


# ======================================================
# Figure: 2 rows × 4 columns
# ======================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharex=True)

c_same = '#2196F3'
c_within = '#FF9800'
c_alt = '#E91E63'

for col, (scenario_name, res) in enumerate(results.items()):
    ax_dxy = axes[0, col]
    ax_fst = axes[1, col]
    ax_dxy.set_title(scenario_name, fontsize=9, fontweight='bold')

    # dxy
    ax_dxy.plot(mid, res['dxy_kir_fol_same'], '-o', color=c_same,
                label='Kir vs Fol$_{same}$', ms=3, lw=1.5)
    ax_dxy.plot(mid, res['dxy_fol_same_alt'], '-o', color=c_within,
                label='Fol$_{same}$ vs Fol$_{alt}$', ms=3, lw=1.5)
    ax_dxy.plot(mid, res['dxy_kir_fol_alt'], '-o', color=c_alt,
                label='Kir vs Fol$_{alt}$', ms=3, lw=1.5)

    # Shade inversions
    for (l, r, lbl) in [(inv_3Ra_left, inv_3Ra_right, '3Ra'),
                         (inv_3Rb_left, inv_3Rb_right, '3Rb')]:
        ax_dxy.axvspan(l * nsites, r * nsites, alpha=0.08, color='gray')
        ax_dxy.axvline(l * nsites, color='gray', ls='--', alpha=0.3)
        ax_dxy.axvline(r * nsites, color='gray', ls='--', alpha=0.3)
        ax_fst.axvspan(l * nsites, r * nsites, alpha=0.08, color='gray')
        ax_fst.axvline(l * nsites, color='gray', ls='--', alpha=0.3)
        ax_fst.axvline(r * nsites, color='gray', ls='--', alpha=0.3)

    if col == 0:
        ax_dxy.set_ylabel('$d_{xy}$', fontsize=11)
        ax_dxy.legend(fontsize=7, loc='upper right')

    # Fst
    ax_fst.plot(mid, res['fst_kir_fol_same'], '-o', color=c_same,
                label='Kir vs Fol$_{same}$', ms=3, lw=1.5)
    ax_fst.plot(mid, res['fst_fol_same_alt'], '-o', color=c_within,
                label='Fol$_{same}$ vs Fol$_{alt}$', ms=3, lw=1.5)
    ax_fst.plot(mid, res['fst_kir_fol_alt'], '-o', color=c_alt,
                label='Kir vs Fol$_{alt}$', ms=3, lw=1.5)
    ax_fst.set_xlabel('Position', fontsize=10)
    if col == 0:
        ax_fst.set_ylabel('$F_{ST}$', fontsize=11)
        ax_fst.legend(fontsize=8, loc='upper right')

fig.suptitle(
    'An. funestus 3R: Two inversions (3Ra+3Rb) vs one (3Ra only)\n'
    f'Ne_K={Ne_K:,}, Ne_F={Ne_F:,}, Ne_Anc={Ne_Anc:,}, '
    f'T_split={t_split_gen:,} gen, T_inv~{t_inv_gen:,} gen',
    fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/kir_fol_3Ra_3Rb.pdf', bbox_inches='tight', dpi=150)
print(f"\nFigure saved: figures/kir_fol_3Ra_3Rb.pdf")
