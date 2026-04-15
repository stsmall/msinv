#!/usr/bin/env python3
"""RDL sweep-through-inversion.

Reproduces the haplotype asymmetry signature of the Anopheles RDL
(dieldrin resistance) sweep through a 2L inversion (Grau-Bové et al.
2020 MBE) using msinv's hull simulator.

Three scenarios run in parallel (each n_S + n_I haplotypes inside one
inversion):

  1. No sweep (neutral baseline).
  2. Sweep on S background ONLY, no flux yet.
  3. Sweep on S, then gene flux transferred RDL to I, then I sweep.

Outputs ``figures/empirical_rdl_sweep.pdf`` showing dxy_SI,
within-class pi, and FST vs position around the selected site x_sel.
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import msprime
from multiprocessing import Pool

from msinv import HullSimulator, InversionSpec, Sweep


# --- Parameters ---
Ne = 50_000               # An. gambiae (reduced from 100k → rho=200)
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
s_coef = 0.05             # strong selection (dieldrin resistance)

n_S = 8
n_I = 8
NREPS = 100
NW = 30
SEED_BASE = 7777


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


def run_scenario(args):
    """Run one scenario — returns (label, dxy, pi_S, pi_I)."""
    label, sweeps, seed_offset = args
    print(f"  [{label}] starting {NREPS} reps...")
    t0 = time.time()
    mut_rng = np.random.default_rng(2027 + seed_offset)
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
                              p_inv=p_inv_freq, t_inv=t_inv_age,
                              gene_conversion_rate=1e-9),
            ],
            sweeps=sweeps,
            recombination_rate=r,
            seed=SEED_BASE + seed_offset * 1000 + rep,
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
    if n_ok > 0:
        dxy /= n_ok; pi_S /= n_ok; pi_I /= n_ok
    print(f"  [{label}] done: {n_ok}/{NREPS} in {time.time()-t0:.0f}s")
    return label, dxy, pi_S, pi_I


def main():
    print(f"RDL sweep: 3 scenarios x {NREPS} reps, parallelized")
    t0 = time.time()

    sweep_S = Sweep(x_sel=x_sel, t_event=t_sweep_S,
                    target_class='S', selection_coefficient=s_coef)
    sweep_I = Sweep(x_sel=x_sel, t_event=t_sweep_I,
                    target_class='I', selection_coefficient=s_coef)

    tasks = [
        ('neutral', [], 0),
        ('S sweep only', [sweep_S], 1),
        ('S then I sweep', [sweep_S, sweep_I], 2),
    ]

    with Pool(3) as pool:
        raw = pool.map(run_scenario, tasks)

    results = {label: (dxy, pi_S, pi_I) for label, dxy, pi_S, pi_I in raw}
    elapsed = time.time() - t0
    print(f"\nAll scenarios done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save data for later overlay of theoretical predictions
    save = {}
    for label, (dxy, pi_S, pi_I) in results.items():
        tag = label.replace(' ', '_')
        save[f'dxy_{tag}'] = dxy
        save[f'piS_{tag}'] = pi_S
        save[f'piI_{tag}'] = pi_I
    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2
    np.savez('figures/rdl_sweep_data.npz', mid=mid,
             Ne=Ne, mu=mu, r=r, L=L, s_coef=s_coef,
             t_sweep_S=t_sweep_S, t_sweep_I=t_sweep_I, **save)

    # ---- Plot ----
    def smooth(y, k=3):
        return np.convolve(y, np.ones(k) / k, mode='same')

    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharey='row', sharex=True)
    labels = ['neutral', 'S sweep only', 'S then I sweep']

    for col, label in enumerate(labels):
        dxy, pi_S, pi_I = results[label]
        fst = 1.0 - (pi_S + pi_I) / 2 / np.maximum(dxy, 1e-20)
        ax_dxy = axes[0, col]
        ax_pi = axes[1, col]
        ax_fst = axes[2, col]

        ax_dxy.plot(mid, smooth(dxy), '-', color='#FF9800', lw=2,
                    label='$d_{XY}$ (S vs I)')
        ax_dxy.set_title(label, fontsize=11, fontweight='bold')
        ax_dxy.axvspan(bp_l, bp_r, alpha=0.08, color='gray', zorder=0)
        ax_dxy.axvline(x_sel, color='red', ls=':', lw=1.5, alpha=0.6,
                       label='x_sel')
        ax_dxy.set_xlim(0, L)
        if col == 0:
            ax_dxy.set_ylabel('$d_{XY}$ (per bp)', fontsize=11)
            ax_dxy.legend(fontsize=8, loc='upper right')

        ax_pi.plot(mid, smooth(pi_S), '-', color='#1976D2', lw=2,
                   label='$\\pi_S$')
        ax_pi.plot(mid, smooth(pi_I), '-', color='#C2185B', lw=2,
                   label='$\\pi_I$')
        ax_pi.axvspan(bp_l, bp_r, alpha=0.08, color='gray', zorder=0)
        ax_pi.axvline(x_sel, color='red', ls=':', lw=1.5, alpha=0.6)
        if col == 0:
            ax_pi.set_ylabel('$\\pi$ within class (per bp)', fontsize=11)
            ax_pi.legend(fontsize=8, loc='upper right')

        ax_fst.plot(mid, smooth(fst), '-', color='#E65100', lw=2,
                    label='$F_{ST}$ (Hudson)')
        ax_fst.axvspan(bp_l, bp_r, alpha=0.08, color='gray', zorder=0)
        ax_fst.axvline(x_sel, color='red', ls=':', lw=1.5, alpha=0.6)
        ax_fst.axhline(0, color='gray', ls=':', lw=0.7)
        ax_fst.set_xlabel('Position (bp)', fontsize=10)
        if col == 0:
            ax_fst.set_ylabel('$F_{ST}$', fontsize=11)
            ax_fst.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        'RDL-like sweep through inversion (msinv hull simulator)',
        fontsize=12, fontweight='bold', y=1.01)

    caption = (
        f'Figure. RDL (dieldrin resistance) sweep-through-inversion, modelled after '
        f'Grau-Bov\u00e9 et al. (2020 MBE). Hitchhiking mode (s={s_coef}) with spatial decay '
        f'P(linked) = exp(-r |x - x_sel| t_dur). Three scenarios for n_S={n_S} + n_I={n_I} haplotypes. '
        f'(Row 1) Cross-class $d_{{XY}}$. (Row 2) Within-class $\\pi_S$, $\\pi_I$. '
        f'(Row 3) Hudson $F_{{ST}}$. '
        f'Left: Neutral baseline. '
        f'Centre: S sweep only (t={t_sweep_S} gen) — $\\pi_S$ valley around x_sel, $\\pi_I$ unaffected '
        f'(haplotype asymmetry). '
        f'Right: S then I sweep (t_S={t_sweep_S}, t_I={t_sweep_I} gen) — both collapse. '
        f'Parameters: Ne={Ne:,}, p_inv={p_inv_freq}, t_inv={t_inv_age:,} gen, '
        f's={s_coef}, $\\gamma$=1e-9, $\\mu$={mu:.0e}, r={r:.0e}, L={L/1e3:.0f} kb, {NREPS} reps.\n'
        f'Command: pixi run -e all python examples/empirical_rdl_sweep.py'
    )
    fig.text(0.5, -0.03, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))

    fig.tight_layout()
    fig.savefig('figures/empirical_rdl_sweep.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: figures/empirical_rdl_sweep.pdf')


if __name__ == '__main__':
    main()
