#!/usr/bin/env python3
"""Kir/Fol baseline pilot at proportional scale.

Uses Small et al. 2023 estimates + Kir/Fol model design notes:
  - Current Ne: K=44k, F=92k
  - Fol expanded to ~3e6 after split (backward: pre-split ancestral contraction)
  - Split ~14k gen ago, gene flow stopped ~1.1k gen ago
  - Ancestral Ne 1e6 → 3e6 at >100k gen
  - 3Ra age 330k gen, p_inv(F)=0.73; 3Rb same age, p_inv(F)=0.40
  - K fixed Standard at both inversions

Proportional scaling preserves real 3R geometry (40% inverted, 3Ra and 3Rb
positioned ~18% and ~72% into the chrom). Two size tiers:

  --scale fast    : L=1 Mb, 200 kb inversions, ~20 ms/rep
  --scale medium  : L=10 Mb, 2 Mb inversions, ~80 ms/rep

Computes branch-mode summary stats (no mutation noise) plus optional
pg_gpu haplotype stats when the module is importable.

Outputs:
  figures/kir_fol_pilot_<scale>_neutral.npz
  figures/kir_fol_pilot_<scale>_neutral.pdf
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from msinv import HullSimulator, InversionSpec, Demography


# ---- Biology (Small et al. 2023 + user-supplied demography 2026-04-24) ----
# All Ne values are EFFECTIVE sizes (from ABC/pg-gan inference in paper).
NE_K_CURRENT = 44_000        # K steady (t=0 onward going back until split)
NE_F_PRESENT = 3_250_000     # F effective size today
NE_F_AT_SPLIT = 44_000       # F Ne at the split (exponential growth from here)
NE_ANC_SPLIT = 64_000        # ancestral Ne at the moment of K-F split
NE_BOTTLE = 10_000           # 18k-25k gen BP bottleneck
NE_MID = 40_000              # 25k-60k gen BP mid-anc
NE_DEEP = 1_000_000          # 60k+ gen BP deep ancestral

T_SPLIT = 13_000             # K-F split (Small 2023)
T_BOTTLE_START = 18_000
T_BOTTLE_END = 25_000
T_DEEP = 60_000
T_INV = 330_000              # 3Ra/3Rb age (both)
P_INV_3RA_F = 0.73
P_INV_3RB_F = 0.40
P_INV_ANC = 0.30

R = 1.0e-8                   # 10× mu (from paper's rates, clean ratio)
GAMMA = 1.0e-7               # gene conversion rate (bumped 100× from 1e-9)
MEAN_TRACT_FRAC = 0.05       # tract length as fraction of inversion length

# Migration: ZERO between 0-13k (strict isolation). F merges into K at split.
# User decision 2026-04-24: rely on F expansion for Tajima's D signal,
# not on migration mixing.

# F expansion: Ne grows FROM 44k AT SPLIT TO 3.25M TODAY. Exponential.
# msinv `eg` event with time=0, pop=1, rate=alpha:
#   N(t') = NE_F_PRESENT * exp(-alpha * t') for t' >= 0 (backward time)
# Want N(T_SPLIT=13000) = 44_000:
#   alpha = ln(NE_F_PRESENT / NE_F_AT_SPLIT) / T_SPLIT
import math
ALPHA_F_GROWTH = math.log(NE_F_PRESENT / NE_F_AT_SPLIT) / T_SPLIT

# sample sizes (Small 2023 observed counts)
N_K = 74          # Kir samples
# F total = 92. 3Ra freq = 0.734 in F → I=round(0.734*92)=68, S=24.
# Matches 3Ra exactly under 1-karyotype simplification (3Rb freq won't
# match because single kary label spans both inversions).
N_F_S = 24        # Fol with standard 3Ra
N_F_I = 68        # Fol with inverted 3Ra


SCALES = {
    'fast':   dict(L=1_000_000,   inv_width=200_000),
    'medium': dict(L=10_000_000, inv_width=2_000_000),
}


def build_sim(scale_cfg, seed):
    L = scale_cfg['L']
    w = scale_cfg['inv_width']

    # Initial pop sizes at t=0. F starts at its PRESENT Ne (3.25M); an
    # `eg` event adds exponential shrinkage going backward so F hits
    # ~44k at T_SPLIT.
    demo = Demography(pop_sizes=[NE_K_CURRENT, NE_F_PRESENT])
    # F expansion (shrinkage going backward).
    demo.add_event(('eg', 0.0, 1, ALPHA_F_GROWTH))
    # Stop F growth at T_SPLIT so pop is fixed at NE_F_AT_SPLIT.
    demo.add_event(('eg', float(T_SPLIT), 1, 0.0))

    # No migration (0-T_SPLIT strict isolation).

    # Split: F merges into K. Ancestral Ne = 64,000.
    demo.add_event(('ej', T_SPLIT, 1, 0))
    demo.add_event(('en', T_SPLIT, 0, NE_ANC_SPLIT))

    # Ancestral Ne trajectory (going back).
    demo.add_event(('en', T_BOTTLE_START, 0, NE_BOTTLE))
    demo.add_event(('en', T_BOTTLE_END, 0, NE_MID))
    demo.add_event(('en', T_DEEP, 0, NE_DEEP))

    # Inversion frequency in ancestral pop after ej (rare barrier-era
    # model assumption — user has P_INV_ANC=0.30 for both inversions).
    demo.add_inversion_freq_change(T_SPLIT, 0, inv_id=0, p_inv=P_INV_ANC)

    # Only 3Ra while using 1-karyotype-per-sample simplification.
    # 3Rb would require per-inversion karyotype tuples to avoid forcing
    # the label to apply to both inversions.
    inv_3ra = InversionSpec(
        bp_left=int(0.18 * L), bp_right=int(0.18 * L) + w,
        p_inv={0: 0.0, 1: P_INV_3RA_F},
        t_inv=T_INV, gene_conversion_rate=GAMMA,
        mean_tract_length=MEAN_TRACT_FRAC * w, tract_distribution='fixed',
    )

    return HullSimulator(
        sample_config={('S', 0): N_K, ('S', 1): N_F_S, ('I', 1): N_F_I},
        demography=demo,
        sequence_length=L,
        recombination_rate=R,
        inversions=[inv_3ra],
        sweeps=[],
        seed=seed,
        iters_max=1_000_000_000,
        # no stop_at — single-sim to MRCA via rate_cache always-on.
    )


def sample_indices():
    """0..N_K = K, then N_F_S = F_S, then N_F_I = F_I."""
    k = list(range(N_K))
    fs = list(range(N_K, N_K + N_F_S))
    fi = list(range(N_K + N_F_S, N_K + N_F_S + N_F_I))
    return k, fs, fi


def branch_stats_per_window(ts, k, fs, fi, nwin):
    L = ts.sequence_length
    edges = np.linspace(0, L, nwin + 1)
    windows = [(float(edges[i]), float(edges[i + 1])) for i in range(nwin)]

    def w(stat_vec):
        return np.asarray(stat_vec, dtype=float)

    pi_k  = w(ts.diversity([k],  windows=edges, mode='branch')).ravel()
    pi_fs = w(ts.diversity([fs], windows=edges, mode='branch')).ravel()
    pi_fi = w(ts.diversity([fi], windows=edges, mode='branch')).ravel()

    dxy_kfs = w(ts.divergence([k, fs], windows=edges, mode='branch')).ravel()
    dxy_kfi = w(ts.divergence([k, fi], windows=edges, mode='branch')).ravel()
    dxy_fsi = w(ts.divergence([fs, fi], windows=edges, mode='branch')).ravel()

    fst_kf = w(ts.Fst([k, fs + fi], windows=edges, mode='branch')).ravel()

    return dict(
        pi_K=pi_k, pi_FS=pi_fs, pi_FI=pi_fi,
        dxy_K_FS=dxy_kfs, dxy_K_FI=dxy_kfi, dxy_FS_FI=dxy_fsi,
        Fst_K_F=fst_kf,
    )


def run(scale_name, nreps, nwin, out_prefix):
    cfg = SCALES[scale_name]
    L = cfg['L']
    k, fs, fi = sample_indices()

    keys = ['pi_K', 'pi_FS', 'pi_FI',
            'dxy_K_FS', 'dxy_K_FI', 'dxy_FS_FI',
            'Fst_K_F']
    acc = {ky: np.zeros(nwin) for ky in keys}
    wall = []
    n_ok = 0
    t0 = time.time()

    for rep in range(nreps):
        seed = 4242 + rep
        sim = build_sim(cfg, seed=seed)
        t_rep = time.time()
        try:
            ts = sim.simulate()
        except Exception as exc:
            print(f"  rep {rep}: failed ({exc})")
            continue
        wall.append(time.time() - t_rep)
        stats = branch_stats_per_window(ts, k, fs, fi, nwin)
        for ky in keys:
            acc[ky] += stats[ky]
        n_ok += 1
        if (rep + 1) % max(1, nreps // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {rep + 1}/{nreps}  elapsed={elapsed:.1f}s "
                  f"mean rep={np.mean(wall):.3f}s")

    if n_ok == 0:
        raise RuntimeError("all reps failed")

    for ky in keys:
        acc[ky] /= n_ok

    edges = np.linspace(0, L, nwin + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    meta = dict(
        scale=scale_name, L=L, inv_width=cfg['inv_width'],
        n_reps=n_ok, mean_rep_s=float(np.mean(wall)),
        NE_K_CURRENT=NE_K_CURRENT, NE_F_CURRENT=NE_F_CURRENT,
        NE_F_POSTSPLIT=NE_F_POSTSPLIT,
        NE_ANC_1=NE_ANC_1, NE_ANC_2=NE_ANC_2,
        T_SPLIT=T_SPLIT, T_INV=T_INV,
        P_INV_3RA_F=P_INV_3RA_F, P_INV_3RB_F=P_INV_3RB_F,
        R=R, GAMMA=GAMMA,
    )
    np.savez(f"{out_prefix}.npz", mids=mids, **acc, **meta)
    print(f"\nwrote {out_prefix}.npz  (n_ok={n_ok}, "
          f"mean rep={np.mean(wall):.3f}s)")

    plot_panels(mids, acc, cfg, f"{out_prefix}.pdf")
    print(f"wrote {out_prefix}.pdf")


def plot_panels(mids, acc, cfg, pdf_path):
    L = cfg['L']
    inv_a = (int(0.18 * L), int(0.18 * L) + cfg['inv_width'])
    inv_b = (int(0.72 * L), int(0.72 * L) + cfg['inv_width'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    def shade(ax):
        for lo, hi, lbl in [(*inv_a, '3Ra'), (*inv_b, '3Rb')]:
            ax.axvspan(lo, hi, alpha=0.12, color='#90A4AE', zorder=0)
            ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92, lbl,
                    ha='center', fontsize=9, color='#455A64')

    ax = axes[0, 0]
    ax.plot(mids, acc['pi_K'], label='K', color='#1976D2')
    ax.plot(mids, acc['pi_FS'], label='F_S', color='#F57C00')
    ax.plot(mids, acc['pi_FI'], label='F_I', color='#D32F2F')
    ax.set_ylabel('pi (branch)')
    ax.legend()
    shade(ax)

    ax = axes[0, 1]
    ax.plot(mids, acc['dxy_K_FS'], label='K-F_S', color='#2E7D32')
    ax.plot(mids, acc['dxy_K_FI'], label='K-F_I', color='#00838F')
    ax.plot(mids, acc['dxy_FS_FI'], label='F_S-F_I', color='#FF8F00')
    ax.set_ylabel('dxy (branch)')
    ax.legend()
    shade(ax)

    ax = axes[1, 0]
    ax.plot(mids, acc['Fst_K_F'], color='#6A1B9A')
    ax.set_ylabel('Fst (K vs F)')
    ax.set_xlabel('position (bp)')
    shade(ax)

    ax = axes[1, 1]
    ratio = np.where(acc['dxy_FS_FI'] > 0,
                     acc['dxy_K_FI'] / acc['dxy_FS_FI'], np.nan)
    ax.plot(mids, ratio, color='#455A64')
    ax.axhline(1.0, color='k', ls=':', lw=0.8)
    ax.set_ylabel('dxy(K-F_I) / dxy(F_S-F_I)')
    ax.set_xlabel('position (bp)')
    shade(ax)

    fig.suptitle(
        f"Kir/Fol neutral baseline, scale={cfg['L']/1e6:g}Mb "
        f"(inversions {cfg['inv_width']/1e6:g}Mb each)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', choices=list(SCALES), default='fast')
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--windows', type=int, default=40)
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prefix = f"{args.outdir}/kir_fol_pilot_{args.scale}_neutral"

    run(args.scale, args.reps, args.windows, prefix)


if __name__ == '__main__':
    main()
