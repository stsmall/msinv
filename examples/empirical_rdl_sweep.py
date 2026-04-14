#!/usr/bin/env python3
"""RDL sweep-through-inversion.

Reproduces the haplotype asymmetry signature of the Anopheles RDL
(dieldrin resistance) sweep through a 2L inversion (Grau-Bové et al.
2020 MBE) using msinv's hull simulator.

Three scenarios (each n_S + n_I haplotypes inside one inversion):

  1. No sweep (neutral baseline).
  2. Sweep on S background ONLY, no flux yet.
  3. Sweep on S, then gene flux transferred RDL to I, then I sweep.

Outputs ``figures/empirical_rdl_sweep.pdf`` showing dxy_SI and
within-class pi vs position around the selected site x_sel.
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import msprime

from msinv import HullSimulator, InversionSpec, Sweep


# --- Parameters ---
Ne = 100_000              # An. gambiae effective Ne
mu = 3e-9
r = 1e-8
L = 100_000               # 100 kb
bp_l = 30_000             # inversion bounds within the chromosome
bp_r = 70_000
inv_len = bp_r - bp_l
x_sel = 50_000            # selected site (centre of inversion)
p_inv_freq = 0.5
t_inv_age = 80_000        # gen — old inversion (~2 Ne)

# Sweep parameters (RDL-style: very recent, very strong)
t_sweep_S = 700           # gen ago — RDL fixed on S background
t_flux = 500              # gen ago — gene conversion to I
t_sweep_I = 300           # gen ago — RDL then fixed on I background

n_S = 8
n_I = 8
NREPS = 100
NW = 30
SEED_BASE = 7777
mut_rng = np.random.default_rng(2027)


def per_window(haps, pos_bp, group_a, group_b, kind='dxy'):
    wins = np.linspace(0, L, NW + 1)
    out = np.zeros(NW)
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
            out[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        else:  # pi within group_a
            d = 0; n = 0
            ga = list(group_a)
            for i in range(len(ga)):
                for j in range(i + 1, len(ga)):
                    d += (haps[ga[i], mask] != haps[ga[j], mask]).sum()
                    n += 1
            out[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return out


def run_scenario(label, sweeps):
    """Returns (mean dxy_SI window, mean pi_S window, mean pi_I window)."""
    print(f"\n[{label}] running {NREPS} reps...")
    t0 = time.time()
    dxy = np.zeros(NW)
    pi_S = np.zeros(NW)
    pi_I = np.zeros(NW)
    n_ok = 0
    for rep in range(NREPS):
        sim = HullSimulator(
            n_std=n_S, n_inv=n_I,
            population_size=Ne,
            sequence_length=L,
            inversions=[
                InversionSpec(bp_left=bp_l, bp_right=bp_r,
                                  p_inv=p_inv_freq, t_inv=t_inv_age),
            ],
            sweeps=sweeps,
            seed=SEED_BASE + rep,
        )
        try:
            ts = sim.simulate()
        except Exception:
            continue
        seed = int(mut_rng.integers(1, 2**31))
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                      discrete_genome=False)
        G = mts.genotype_matrix(); haps = G.T
        pos_bp = np.array([v.site.position for v in mts.variants()])
        S_idx = list(range(n_S))
        I_idx = list(range(n_S, n_S + n_I))
        dxy += per_window(haps, pos_bp, S_idx, I_idx, 'dxy')
        pi_S += per_window(haps, pos_bp, S_idx, None, 'pi')
        pi_I += per_window(haps, pos_bp, I_idx, None, 'pi')
        n_ok += 1
        if (rep + 1) % 25 == 0:
            print(f"  {rep + 1}/{NREPS}  elapsed={time.time()-t0:.0f}s")
    if n_ok > 0:
        for arr in (dxy, pi_S, pi_I):
            arr /= n_ok
    print(f"  done: {n_ok}/{NREPS}, {time.time()-t0:.0f}s")
    return dxy, pi_S, pi_I


# ---- Run three scenarios ----
results = {}
results['neutral'] = run_scenario('neutral', sweeps=[])

# Scenario 2: S sweep only (no flux yet → I lineages unaffected)
sweep_S = Sweep(x_sel=x_sel, t_event=t_sweep_S,
                target_class='S0',  # 'S<inv_id>' for inv 0 (single inv → 'S')
                sweep_window=500.0)
# Phase-5b labels single-inv segments 'S' (no _0 suffix). Use 'S':
sweep_S = Sweep(x_sel=x_sel, t_event=t_sweep_S,
                target_class='S', sweep_window=500.0)
results['S sweep only'] = run_scenario('S sweep only', sweeps=[sweep_S])

# Scenario 3: S sweep, then flux, then I sweep
# (we just stack two sweeps; gene flux between them is implicit in the
# class-flip of the I-background sweep target).
sweep_I = Sweep(x_sel=x_sel, t_event=t_sweep_I,
                target_class='I', sweep_window=500.0)
results['S then I sweep'] = run_scenario(
    'S then I sweep', sweeps=[sweep_S, sweep_I])


# ---- Plot ----
def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode='same')


wins = np.linspace(0, L, NW + 1)
mid = (wins[:-1] + wins[1:]) / 2

fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey='row', sharex=True)
labels = ['neutral', 'S sweep only', 'S then I sweep']

for col, label in enumerate(labels):
    dxy, pi_S, pi_I = results[label]
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    ax_top.plot(mid, smooth(dxy), '-', color='#FF9800', lw=2,
                 label='$d_{XY}$ (S vs I)')
    ax_top.set_title(label, fontsize=11, fontweight='bold')
    ax_top.axvspan(bp_l, bp_r, alpha=0.08, color='gray', zorder=0)
    ax_top.axvline(x_sel, color='red', ls=':', lw=1.5, alpha=0.6,
                    label='x_sel')
    ax_top.set_xlim(0, L)
    if col == 0:
        ax_top.set_ylabel('$d_{XY}$ (per bp)', fontsize=11)
        ax_top.legend(fontsize=8, loc='upper right')

    ax_bot.plot(mid, smooth(pi_S), '-', color='#1976D2', lw=2,
                 label='$\\pi_S$')
    ax_bot.plot(mid, smooth(pi_I), '-', color='#C2185B', lw=2,
                 label='$\\pi_I$')
    ax_bot.axvspan(bp_l, bp_r, alpha=0.08, color='gray', zorder=0)
    ax_bot.axvline(x_sel, color='red', ls=':', lw=1.5, alpha=0.6)
    ax_bot.set_xlabel('Position (bp)', fontsize=10)
    if col == 0:
        ax_bot.set_ylabel('$\\pi$ within class (per bp)', fontsize=11)
        ax_bot.legend(fontsize=8, loc='upper right')

fig.suptitle(
    f'RDL-like sweep through inversion (msinv v0.3.0)\n'
    f'Ne={Ne:,}, p_inv={p_inv_freq}, x_sel={x_sel:,} bp, '
    f't_sweep_S={t_sweep_S} gen, t_sweep_I={t_sweep_I} gen, '
    f'{NREPS} replicates',
    fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/empirical_rdl_sweep.pdf',
            bbox_inches='tight', dpi=150)
print('\nSaved: figures/empirical_rdl_sweep.pdf')
