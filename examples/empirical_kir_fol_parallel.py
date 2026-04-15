#!/usr/bin/env python3
"""Parallel Kir/Fol simulation — 1000 replicates across 10 workers.

Each worker runs 100 reps of the An. funestus 3Ra/3Rb simulation,
accumulates per-window stats, and returns the sums. The main process
combines all workers and plots.

Usage:
    pixi run -e all python examples/empirical_kir_fol_parallel.py
"""
import sys
import time
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import msprime

from msinv import HullSimulator, InversionSpec, Demography


# --- Parameters (same as empirical_kir_fol.py) ---
Ne = 44_000
mu = 3.55e-9
r = 4.0e-8          # An. funestus recombination rate (rho ~ 704)
L = 100_000
t_split_gen = 14_000
t_inv_gen = 385_000
p_inv_anc = 0.3

inv_3Ra = (15_000, 45_000)
inv_3Rb = (55_000, 85_000)

n_kir = 10
n_fol_S = 5
n_fol_I = 5

N_WORKERS = 10
REPS_PER_WORKER = 10
NW = 40

# Sample group indices (set by sample_config order)
kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, n_kir + n_fol_S + n_fol_I))


def per_window_stats(haps, pos_bp, group_a, group_b, kind='dxy'):
    """Compute per-window dxy or pi."""
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


def worker_chunk(args):
    """Run REPS_PER_WORKER reps and return accumulated stats + count."""
    worker_id, seed_base = args
    mut_rng = np.random.default_rng(seed_base)

    acc = {k: np.zeros(NW) for k in
           ['dxy_kf_same', 'dxy_kf_alt', 'dxy_fs_fi',
            'pi_K', 'pi_FS', 'pi_FI']}
    n_ok = 0

    for rep in range(REPS_PER_WORKER):
        seed = seed_base + rep
        demo = Demography(pop_sizes=[Ne, Ne])
        demo.add_event(('ej', t_split_gen, 1, 0))

        sim = HullSimulator(
            sample_config={
                ('S', 0): n_kir,
                ('S', 1): n_fol_S,
                ('I', 1): n_fol_I,
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
            seed=seed,
        )
        try:
            ts = sim.simulate()
        except Exception as e:
            print(f"  Worker {worker_id} rep {rep}: {e}", file=sys.stderr)
            continue

        mseed = int(mut_rng.integers(1, 2**31))
        mts = msprime.sim_mutations(ts, rate=mu, random_seed=mseed,
                                    discrete_genome=False)
        G = mts.genotype_matrix()
        haps = G.T
        pos_bp = np.array([v.site.position for v in mts.variants()])

        acc['dxy_kf_same'] += per_window_stats(haps, pos_bp, kir, fol_same, 'dxy')
        acc['dxy_kf_alt'] += per_window_stats(haps, pos_bp, kir, fol_alt, 'dxy')
        acc['dxy_fs_fi'] += per_window_stats(haps, pos_bp, fol_same, fol_alt, 'dxy')
        acc['pi_K'] += per_window_stats(haps, pos_bp, kir, None, 'pi')
        acc['pi_FS'] += per_window_stats(haps, pos_bp, fol_same, None, 'pi')
        acc['pi_FI'] += per_window_stats(haps, pos_bp, fol_alt, None, 'pi')
        n_ok += 1

        if (rep + 1) % 25 == 0:
            print(f"  Worker {worker_id}: {rep + 1}/{REPS_PER_WORKER}")

    return acc, n_ok


def main():
    print(f"Kir/Fol parallel: {N_WORKERS} workers x {REPS_PER_WORKER} reps "
          f"= {N_WORKERS * REPS_PER_WORKER} total")
    t0 = time.time()

    # Each worker gets a non-overlapping seed range
    tasks = [(w, 10_000 + w * REPS_PER_WORKER) for w in range(N_WORKERS)]

    with Pool(N_WORKERS) as pool:
        results = pool.map(worker_chunk, tasks)

    # Combine
    combined = {k: np.zeros(NW) for k in
                ['dxy_kf_same', 'dxy_kf_alt', 'dxy_fs_fi',
                 'pi_K', 'pi_FS', 'pi_FI']}
    total_ok = 0
    for acc, n_ok in results:
        for k in combined:
            combined[k] += acc[k]
        total_ok += n_ok

    if total_ok > 0:
        for k in combined:
            combined[k] /= total_ok

    elapsed = time.time() - t0
    print(f"\nDone: {total_ok}/{N_WORKERS * REPS_PER_WORKER} reps "
          f"in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Unpack
    dxy_kf_same = combined['dxy_kf_same']
    dxy_kf_alt = combined['dxy_kf_alt']
    dxy_fs_fi = combined['dxy_fs_fi']
    pi_K = combined['pi_K']
    pi_FS = combined['pi_FS']
    pi_FI = combined['pi_FI']

    # Net divergence (Da)
    da_kf_same = dxy_kf_same - (pi_K + pi_FS) / 2
    da_kf_alt = dxy_kf_alt - (pi_K + pi_FI) / 2
    da_fs_fi = dxy_fs_fi - (pi_FS + pi_FI) / 2

    # FST = 1 - pi_within / pi_total  (Hudson estimator)
    # pi_within = (pi_a + pi_b) / 2;  pi_total ~ dxy
    fst_kf_same = 1.0 - (pi_K + pi_FS) / 2 / np.maximum(dxy_kf_same, 1e-20)
    fst_kf_alt = 1.0 - (pi_K + pi_FI) / 2 / np.maximum(dxy_kf_alt, 1e-20)
    fst_fs_fi = 1.0 - (pi_FS + pi_FI) / 2 / np.maximum(dxy_fs_fi, 1e-20)

    # Save raw arrays for reuse
    np.savez('figures/empirical_kir_fol_data.npz',
             dxy_kf_same=dxy_kf_same, dxy_kf_alt=dxy_kf_alt,
             dxy_fs_fi=dxy_fs_fi, pi_K=pi_K, pi_FS=pi_FS, pi_FI=pi_FI,
             da_kf_same=da_kf_same, da_kf_alt=da_kf_alt, da_fs_fi=da_fs_fi,
             fst_kf_same=fst_kf_same, fst_kf_alt=fst_kf_alt,
             fst_fs_fi=fst_fs_fi,
             total_ok=total_ok, elapsed=elapsed)
    print("Saved: figures/empirical_kir_fol_data.npz")

    # ---- Plot (3 panels: dxy, Da, FST) ----
    def smooth(y, k=3):
        return np.convolve(y, np.ones(k) / k, mode='same')

    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    fig = plt.figure(figsize=(12, 11))
    gs = GridSpec(3, 1, hspace=0.30)

    c_kf_same = '#2E7D32'
    c_fs_fi = '#FF8F00'
    c_kf_alt = '#00838F'

    def shade_inv(ax):
        for (lo, hi), lbl in [(inv_3Ra, '3Ra'), (inv_3Rb, '3Rb')]:
            ax.axvspan(lo, hi, alpha=0.10, color='#90A4AE', zorder=0)
            ax.axvline(lo, color='#78909C', ls='--', alpha=0.4, lw=0.8)
            ax.axvline(hi, color='#78909C', ls='--', alpha=0.4, lw=0.8)
            ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95, lbl,
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
    ax_dxy.set_title(r'A.  Absolute divergence $d_{XY}$',
                     fontsize=11, fontweight='bold', loc='left')
    ax_dxy.tick_params(labelbottom=False)

    # Panel B: Da
    ax_da = fig.add_subplot(gs[1])
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
    ax_da.set_xlim(0, L)
    ax_da.legend(fontsize=9, loc='upper right')
    ax_da.set_title(r'B.  Net divergence $D_a$',
                    fontsize=11, fontweight='bold', loc='left')
    ax_da.tick_params(labelbottom=False)

    # Panel C: FST
    ax_fst = fig.add_subplot(gs[2])
    ax_fst.plot(mid, smooth(fst_kf_same), '-', color=c_kf_same, lw=2,
                label=r'K vs F$_S$')
    ax_fst.plot(mid, smooth(fst_fs_fi), '-', color=c_fs_fi, lw=2,
                label=r'F$_S$ vs F$_I$')
    ax_fst.plot(mid, smooth(fst_kf_alt), '-', color=c_kf_alt, lw=2,
                label=r'K vs F$_I$')
    shade_inv(ax_fst)
    ax_fst.axhline(0, color='#555', ls=':', lw=0.8)
    ax_fst.set_ylabel(r'$F_{ST}$ (Hudson)', fontsize=11)
    ax_fst.set_xlabel('Position (bp)', fontsize=10)
    ax_fst.set_xlim(0, L)
    ax_fst.legend(fontsize=9, loc='upper right')
    ax_fst.set_title(r'C.  Hudson $F_{ST}$',
                     fontsize=11, fontweight='bold', loc='left')

    caption = (
        f'Figure. An. funestus Kiribina/Folonzo cross-karyotype divergence for 3Ra and 3Rb '
        f'inversions on chromosome arm 3R, simulated with msinv (hull algorithm, {total_ok} replicates). '
        f'(A) Absolute divergence $d_{{XY}}$. K vs F$_I$ (alt karyotype) shows elevated $d_{{XY}}$ inside '
        f'both inversions due to the recombination barrier. '
        f'(B) Net divergence $D_a = d_{{XY}} - (\\pi_A + \\pi_B)/2$. '
        f'(C) Hudson $F_{{ST}} = 1 - \\pi_W / d_{{XY}}$, showing the relative differentiation signal. '
        f'Parameters: Ne={Ne:,}, T$_{{split}}$={t_split_gen:,} gen, '
        f'T$_{{inv}}$={t_inv_gen:,} gen, $\\mu$={mu:.2e}, r={r:.1e}, '
        f'$\\gamma$=0, {N_WORKERS} workers x {REPS_PER_WORKER} reps.\n'
        f'Command: pixi run -e all python examples/empirical_kir_fol_parallel.py'
    )
    fig.text(0.5, -0.01, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))

    fig.suptitle(
        'An. funestus Kir/Fol — 3Ra + 3Rb inversion divergence (hull simulator)',
        fontsize=12, fontweight='bold', y=0.99)

    fig.savefig('figures/empirical_kir_fol.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: figures/empirical_kir_fol.pdf')

    # Summary table
    inv_w = [w for w in range(NW)
             if (inv_3Ra[0] < mid[w] < inv_3Ra[1]) or
                (inv_3Rb[0] < mid[w] < inv_3Rb[1])]
    col_w = [w for w in range(NW) if w not in inv_w]
    print(f"\n{'metric':<14} {'inv':>10} {'col':>10} {'ratio':>10}")
    for label, arr in [('dxy K-Fs', dxy_kf_same), ('dxy Fs-Fi', dxy_fs_fi),
                        ('dxy K-Fi', dxy_kf_alt), ('Da K-Fs', da_kf_same),
                        ('Da Fs-Fi', da_fs_fi), ('Da K-Fi', da_kf_alt),
                        ('Fst K-Fs', fst_kf_same), ('Fst Fs-Fi', fst_fs_fi),
                        ('Fst K-Fi', fst_kf_alt)]:
        i_m = float(np.mean([arr[w] for w in inv_w]))
        c_m = float(np.mean([arr[w] for w in col_w]))
        ratio = i_m / c_m if c_m != 0 else float('nan')
        print(f"  {label:<14} {i_m:>10.6g} {c_m:>10.6g} {ratio:>10.2f}")


if __name__ == '__main__':
    main()
