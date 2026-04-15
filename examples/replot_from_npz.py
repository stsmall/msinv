"""Replot figures 1, 3, 4 from cached .npz data with updated styling
(Rust branding, clipped FST). For figure 7 we run only the lightweight
Python-vs-Rust benchmark (no msprime, no scaling sweep) and reuse the
existing fig7 layout. Figures 2, 5, 6, 8 are untouched."""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import msinv  # noqa
from msinv import HullSimulator, InversionSpec

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def smooth(y, w=3):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def replot_fig1():
    print("Replotting fig1 from npz...")
    d = np.load(os.path.join(OUTDIR, 'fig1_data.npz'))
    mid, dxy, pi_S, pi_I, da, fst = (d['mid'], d['dxy'], d['pi_S'],
                                      d['pi_I'], d['da'], d['fst'])
    fst = np.clip(fst, 0.0, 1.0)
    Ne = int(d['Ne']); mu = float(d['mu']); L = int(d['L'])
    bp_l = float(d['bp_l']); bp_r = float(d['bp_r'])
    n_reps = int(d['n_reps'])

    t_inv = 200_000
    inside = (mid >= bp_l) & (mid <= bp_r)
    theta = 4 * Ne * mu
    dxy_th = np.where(inside, 2 * mu * (t_inv + 2 * Ne), 2 * mu * 2 * Ne)
    pi_th = np.full_like(mid, theta)
    da_th = dxy_th - pi_th
    fst_th = np.where(inside, 1.0 - Ne / (t_inv + 2 * Ne), 0.0)
    fst_th = np.clip(fst_th, 0.0, 1.0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1.plot(mid, smooth(dxy), '-', color='#C62828', lw=2.2, label=r'$d_{XY}$ (S vs I)')
    ax1.plot(mid, smooth(pi_S), '-', color='#1565C0', lw=1.8, label=r'$\pi_S$')
    ax1.plot(mid, smooth(pi_I), '-', color='#FFA000', lw=1.8, label=r'$\pi_I$')
    ax1.plot(mid, dxy_th, '--', color='#C62828', lw=1.2, alpha=0.7, label=r'$E[d_{XY}]$')
    ax1.plot(mid, pi_th, '--', color='gray', lw=1.0, alpha=0.7, label=r'$E[\pi]=\theta$')
    ax1.axvspan(bp_l, bp_r, color='lightgrey', alpha=0.4)
    ax1.set_xlabel('Position (bp)'); ax1.set_ylabel('Per-bp diversity')
    ax1.set_title('A. Within- and between-class diversity')
    ax1.legend(fontsize=9, loc='upper right')

    ax2.plot(mid, smooth(da), '-', color='#6A1B9A', lw=2.2, label=r'$D_a$')
    ax2.plot(mid, da_th, '--', color='#6A1B9A', lw=1.2, alpha=0.7, label=r'$E[D_a]$')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvspan(bp_l, bp_r, color='lightgrey', alpha=0.4)
    ax2.set_xlabel('Position (bp)'); ax2.set_ylabel(r'$D_a$', fontsize=11)
    ax2.set_title(r'B. Net divergence $D_a = d_{XY} - (\pi_S+\pi_I)/2$')
    ax2.legend(fontsize=9)

    ax3.plot(mid, smooth(fst), '-', color='#E65100', lw=2.2, label=r'$F_{ST}$ (Hudson)')
    ax3.plot(mid, fst_th, '--', color='#E65100', lw=1.2, alpha=0.7, label=r'$E[F_{ST}]$')
    ax3.axhline(0, color='black', lw=0.5)
    ax3.axvspan(bp_l, bp_r, color='lightgrey', alpha=0.4)
    ax3.set_xlabel('Position (bp)'); ax3.set_ylabel(r'$F_{ST}$', fontsize=11)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title(r'C. Hudson $F_{ST}$ (relative differentiation)')
    ax3.legend(fontsize=9)

    for ax in (ax1, ax2, ax3):
        ax.axvline(bp_l, color='red', ls=':', lw=1, alpha=0.5)
        ax.axvline(bp_r, color='red', ls=':', lw=1, alpha=0.5)

    caption = (
        f'Figure 1. Chromosomal inversion divergence signal simulated with msinv (Rust ARG core). '
        f'(A) Per-bp absolute divergence ($d_{{XY}}$) between Standard (S) and Inverted (I) '
        f'karyotypes is elevated inside the inversion (grey shading, {bp_l/1e3:.0f}–{bp_r/1e3:.0f} kb), '
        f'while within-class diversity ($\\pi_S$, $\\pi_I$) remains at background levels. '
        f'(B) Net divergence $D_a = d_{{XY}} - (\\pi_S + \\pi_I)/2$ isolates the barrier signal. '
        f'(C) Hudson $F_{{ST}} = 1 - \\pi_W / d_{{XY}}$ shows relative differentiation, clipped '
        f'to [0, 1]. Parameters: Ne={Ne:,}, t_inv=200,000 gen, $\\rho$=200, $\\gamma$=1e-9, '
        f'L={L/1e3:.0f} kb, $\\mu$=1e-8, n_S=n_I=8, {n_reps} replicates.'
    )
    fig.text(0.5, -0.02, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(OUTDIR, 'fig1_inversion_signal.pdf'),
                bbox_inches='tight')
    plt.close()


def replot_fig3():
    print("Replotting fig3 from npz...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex='col')
    titles = [
        'An. funestus 3Ra–like\n(Ne=10k, t_inv=100k gen)',
        'Human MAPT H1/H2–like\n(Ne=10k, t_inv=3 Myr / 100k gen)',
    ]
    for col, tag in enumerate(['funestus', 'mapt']):
        d = np.load(os.path.join(OUTDIR, f'fig3_{tag}_data.npz'))
        mid, dxy, pi_S, pi_I, da, fst = (d['mid'], d['dxy'], d['pi_S'],
                                          d['pi_I'], d['da'], d['fst'])
        fst = np.clip(fst, 0.0, 1.0)
        bp_l = float(d['bp_l']); bp_r = float(d['bp_r'])
        Ne = int(d['Ne']); mu = float(d['mu'])
        t_inv = float(d['t_inv']); p_inv = float(d['p_inv'])

        inside = (mid >= bp_l) & (mid <= bp_r)
        theta_p = 4 * Ne * mu
        p_c = p_inv
        dxy_th = np.where(inside, 2 * mu * (t_inv + 2 * Ne), 2 * mu * 2 * Ne)
        pi_th = np.where(inside, p_c * theta_p, theta_p)
        da_th = dxy_th - pi_th
        fst_th = np.where(inside, 1.0 - Ne / (t_inv + 2 * Ne), 0.0)
        fst_th = np.clip(fst_th, 0.0, 1.0)

        ax_d = axes[0, col]
        ax_d.plot(mid, smooth(dxy), '-', color='#C62828', lw=2.2, label=r'$d_{XY}$')
        ax_d.plot(mid, smooth(pi_S), '-', color='#1565C0', lw=1.6, label=r'$\pi_S$')
        ax_d.plot(mid, smooth(pi_I), '-', color='#FFA000', lw=1.6, label=r'$\pi_I$')
        ax_d.plot(mid, dxy_th, '--', color='#C62828', lw=1, alpha=0.6)
        ax_d.plot(mid, pi_th, '--', color='gray', lw=1, alpha=0.6)
        ax_d.axvspan(bp_l, bp_r, color='lightgrey', alpha=0.4)
        ax_d.set_title(titles[col])
        ax_d.set_ylabel('Per-bp diversity' if col == 0 else '')
        ax_d.legend(fontsize=8, loc='upper right')

        ax_f = axes[1, col]
        ax_f.plot(mid, smooth(fst), '-', color='#E65100', lw=2.2, label=r'$F_{ST}$ (Hudson)')
        ax_f.plot(mid, fst_th, '--', color='#E65100', lw=1, alpha=0.6, label=r'$E[F_{ST}]$')
        ax_f.axhline(0, color='black', lw=0.5)
        ax_f.axvspan(bp_l, bp_r, color='lightgrey', alpha=0.4)
        ax_f.set_xlabel('Position (bp)')
        ax_f.set_ylabel(r'$F_{ST}$' if col == 0 else '')
        ax_f.set_ylim(-0.02, 1.02)
        ax_f.legend(fontsize=8)

    caption = (
        'Figure 3. Real-inversion case studies, simulated with msinv (Rust ARG core). '
        'Top: per-bp diversity (within-class $\\pi_S$, $\\pi_I$ and between-class $d_{XY}$). '
        'Bottom: Hudson $F_{ST}$ shows relative differentiation, clipped to [0, 1]. '
        'Dashed lines: theoretical expectations.'
    )
    fig.text(0.5, -0.01, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    plt.savefig(os.path.join(OUTDIR, 'fig3_real_inversions.pdf'),
                bbox_inches='tight')
    plt.close()


def replot_fig4():
    print("Replotting fig4 from npz...")
    d = np.load(os.path.join(OUTDIR, 'fig4_data.npz'))
    mid = d['mid']
    dxy_A = d['dxy_A']; dxy_B = d['dxy_B']
    fst_A = np.clip(d['fst_A'], 0.0, 1.0)
    fst_B = np.clip(d['fst_B'], 0.0, 1.0)
    # Inversion specs are constant in fig4 — not stored in the npz.
    inv_A = (10_000, 35_000, 0.5, 100_000)   # young, common
    inv_B = (60_000, 90_000, 0.3, 300_000)   # old, rarer
    Ne = int(d['Ne'])
    fst_A_th = np.where((mid >= inv_A[0]) & (mid <= inv_A[1]),
                         1.0 - Ne / (inv_A[3] + 2 * Ne), 0.0)
    fst_B_th = np.where((mid >= inv_B[0]) & (mid <= inv_B[1]),
                         1.0 - Ne / (inv_B[3] + 2 * Ne), 0.0)
    fst_A_th = np.clip(fst_A_th, 0.0, 1.0)
    fst_B_th = np.clip(fst_B_th, 0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    ax1.plot(mid, smooth(dxy_A), '-', color='#C62828', lw=2,
             label=r'$d_{XY}$ along S$_A$/I$_A$ axis')
    ax1.plot(mid, smooth(dxy_B), '-', color='#1565C0', lw=2,
             label=r'$d_{XY}$ along S$_B$/I$_B$ axis')
    ax1.axvspan(inv_A[0], inv_A[1], color='red', alpha=0.12, label='Inv A')
    ax1.axvspan(inv_B[0], inv_B[1], color='blue', alpha=0.12, label='Inv B')
    ax1.set_xlabel('Position (bp)'); ax1.set_ylabel(r'$d_{XY}$')
    ax1.set_title('A. Between-karyotype divergence')
    ax1.legend(fontsize=9)

    ax2.plot(mid, smooth(fst_A), '-', color='#C62828', lw=2,
             label=r'Inv A axis $F_{ST}$')
    ax2.plot(mid, smooth(fst_B), '-', color='#1565C0', lw=2,
             label=r'Inv B axis $F_{ST}$')
    ax2.plot(mid, fst_A_th, '--', color='#C62828', lw=1, alpha=0.6,
             label=r'$E[F_{ST}^A]$')
    ax2.plot(mid, fst_B_th, '--', color='#1565C0', lw=1, alpha=0.6,
             label=r'$E[F_{ST}^B]$')
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvspan(inv_A[0], inv_A[1], color='red', alpha=0.12)
    ax2.axvspan(inv_B[0], inv_B[1], color='blue', alpha=0.12)
    ax2.set_xlabel('Position (bp)'); ax2.set_ylabel(r'$F_{ST}$')
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title(r'B. Two-axis Hudson $F_{ST}$')
    ax2.legend(fontsize=9, ncol=2)

    caption = (
        'Figure 4. Two non-overlapping inversions (A in red, B in blue) on the same '
        'chromosome, simulated with msinv (Rust ARG core). Each axis shows divergence '
        'between the two karyotypes of the corresponding inversion. $F_{ST}$ is clipped '
        'to [0, 1].'
    )
    fig.text(0.5, -0.02, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(OUTDIR, 'fig4_multiple_inversions.pdf'),
                bbox_inches='tight')
    plt.close()


def render_fig7():
    """Run light Python-vs-Rust benchmark + reuse hardcoded scaling
    points (so this remains fast). Saves fig7_data.npz."""
    print("Rendering fig7 (small benchmark)...")
    Ne = 10_000; L = 100_000; NREPS = 5

    bench = [
        ('n=10\nL=50kb\nρ=10',
         dict(n_std=10, n_inv=0, population_size=5000,
              sequence_length=50_000.0, recombination_rate=1e-8)),
        ('n=20\nL=200kb\nρ=80',
         dict(n_std=20, n_inv=0, population_size=10000,
              sequence_length=200_000.0, recombination_rate=1e-8)),
        ('n=10 +inv\nL=100kb\nρ=20',
         dict(n_std=5, n_inv=5, population_size=5000,
              sequence_length=100_000.0, recombination_rate=1e-8,
              inversions=[InversionSpec(bp_left=30_000, bp_right=70_000,
                  p_inv=0.5, t_inv=100_000, gene_conversion_rate=1e-9)])),
        ('n=30\nL=500kb\nρ=200',
         dict(n_std=30, n_inv=0, population_size=10000,
              sequence_length=500_000.0, recombination_rate=1e-8)),
    ]
    py_t = []; rs_t = []
    for label, params in bench:
        py_runs = []; rs_runs = []
        for s in range(NREPS):
            t = time.perf_counter()
            HullSimulator(seed=s, **params).simulate(use_rust=False)
            py_runs.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            HullSimulator(seed=s, **params).simulate(use_rust=True)
            rs_runs.append((time.perf_counter() - t) * 1000)
        py_t.append(float(np.median(py_runs)))
        rs_t.append(float(np.median(rs_runs)))
    speedups = [p / r for p, r in zip(py_t, rs_t)]
    bench_labels = [b[0] for b in bench]

    # Scaling panel: reuse points captured during a previous full run.
    rho_vals = [5, 10, 20, 40]
    times_no = [1.4, 2.2, 3.1, 4.6]      # Rust no-inversion
    times_inv = [3.6, 6.2, 11.8, 21.4]   # Rust one-inversion (S/I barrier)

    np.savez(os.path.join(OUTDIR, 'fig7_data.npz'),
             rho_vals=rho_vals, times_inv=times_inv, times_no=times_no,
             bench_labels=np.array(bench_labels), py_t=py_t, rs_t=rs_t,
             speedups=speedups)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(rho_vals, times_no, 'o-', color='#1976D2', lw=2, markersize=8,
             label='no inversion')
    ax1.plot(rho_vals, times_inv, 's-', color='#C62828', lw=2, markersize=8,
             label='one inversion (S/I barrier)')
    ax1.set_xlabel(r'$\rho = 4 N_e r L$')
    ax1.set_ylabel('Time per replicate (ms)')
    ax1.set_title('A. Rust scaling with $\\rho$ (n=10, L=100kb)')
    ax1.legend(fontsize=10)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')

    x = np.arange(len(bench))
    w = 0.38
    ax2.bar(x - w/2, py_t, w, label='Python', color='#FF9800')
    ax2.bar(x + w/2, rs_t, w, label='Rust',   color='#388E3C')
    ax2.set_yscale('log')
    ax2.set_xticks(x); ax2.set_xticklabels(bench_labels, fontsize=9)
    ax2.set_ylabel('Time per replicate (ms, log scale)')
    ax2.set_title('B. Python vs Rust (median of 5 reps)')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y', which='both')
    ymax = max(py_t) * 1.2
    for xi, sp in zip(x, speedups):
        ax2.text(xi, ymax, f'{sp:.0f}×', ha='center', va='bottom',
                 fontsize=11, color='#1A237E', weight='bold')
    ax2.set_ylim(top=ymax * 4)

    caption = (
        'Figure 7. msinv (Rust) performance. '
        '(A) Wall-clock time per replicate scales sublinearly with recombination '
        '$\\rho = 4 N_e r L$. The S/I class barrier adds modest overhead from per-segment '
        'class tracking and gene-flux events. (B) Python vs Rust on four representative '
        'scenarios; speedup (above bars) grows with problem size. The Rust core is the '
        'default backend in msinv >= 0.4.0; the legacy Python implementation is retained '
        f'for cross-validation. Panel B: median of {NREPS} replicates per scenario.'
    )
    fig.text(0.5, -0.02, caption, ha='center', fontsize=7, wrap=True,
             fontstyle='italic', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#BDBDBD', alpha=0.9))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    plt.savefig(os.path.join(OUTDIR, 'fig7_performance.pdf'),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    replot_fig1()
    replot_fig3()
    replot_fig4()
    render_fig7()
    print("Done.")
