#!/usr/bin/env python3
"""Kir/Fol with 2Ra mating barrier — reduced effective migration.

Tests the hypothesis that 2Ra fixation in K reduces gene flow because
wing morphology (set by 2Ra) causes assortative mating. Only the
triple-S fraction of Folonzo can mate with Kiribina.

Three scenarios:
  1. Neutral baseline (no migration, just split — current model)
  2. Migration with 2Ra barrier (m_eff = m * P(triple-S in F))
  3. Migration without 2Ra barrier (m_eff = m, unrestricted)

Usage:
    .venv/bin/python examples/empirical_kir_fol_2Ra.py
"""
import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import msprime

from msinv import HullSimulator, InversionSpec, Demography


# --- Fixed parameters (Small et al. 2023) ---
Ne = 44_000
mu = 3.55e-9
L = 100_000
RHO = 40
t_split_gen = 14_000
t_inv_gen = 385_000
Ne_mid = 100_000
t_Ne_mid = 60_000
Ne_anc = 1_000_000

p_inv_3Ra = {0: 0.0, 1: 0.73}
p_inv_3Rb = {0: 0.0, 1: 0.40}
p_inv_anc = 0.3

inv_3Ra = (15_000, 45_000)
inv_3Rb = (55_000, 85_000)

n_kir = 10
n_fol_S = 5
n_fol_I = 5
NW = 40
N_REPS = 25

# Karyotype frequencies in Folonzo
p_2Ra_S_fol = 1.0 - 0.45   # 0.55
p_3Ra_S_fol = 1.0 - 0.73   # 0.27
p_3Rb_S_fol = 1.0 - 0.40   # 0.60
p_triple_S = p_2Ra_S_fol * p_3Ra_S_fol * p_3Rb_S_fol  # ~0.089

# Migration: gene flow ceased ~1100 gen ago, was active from 1100-14000
m_base = 1e-4   # baseline per-gen migration rate
t_mig_cease = 1_100

kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, n_kir + n_fol_S + n_fol_I))


def rho_to_r(rho):
    return rho / (4 * Ne * L)


def per_window_stats(haps, pos_bp, group_a, group_b, kind='dxy'):
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
        else:
            d = 0; n = 0
            ga = list(group_a)
            for i in range(len(ga)):
                for j in range(i + 1, len(ga)):
                    d += (haps[ga[i], mask] != haps[ga[j], mask]).sum()
                    n += 1
            vals[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return vals


def run_scenario(label, m_eff, n_reps):
    """Run one migration scenario."""
    r = rho_to_r(RHO)
    tag = f"2Ra_{label.replace(' ', '_')}"
    npz_path = f'figures/kir_fol_{tag}.npz'

    if os.path.exists(npz_path):
        print(f"  Loading cached: {npz_path}")
        sys.stdout.flush()
        data = dict(np.load(npz_path, allow_pickle=True))
        data['label'] = label
        data['m_eff'] = float(data['m_eff'])
        data['total_ok'] = int(data['total_ok'])
        data['elapsed'] = float(data['elapsed'])
        data['fst_ratio'] = float(data['fst_ratio'])
        data['fst_inv'] = float(data['fst_inv'])
        data['fst_col'] = float(data['fst_col'])
        return data

    print(f"\n{'='*60}")
    print(f"{label}  m_eff={m_eff:.2e}  ({n_reps} reps)")
    print(f"{'='*60}")
    sys.stdout.flush()
    t0 = time.time()

    acc = {k: np.zeros(NW) for k in
           ['dxy_kf_same', 'dxy_kf_alt', 'dxy_fs_fi',
            'pi_K', 'pi_FS', 'pi_FI']}
    n_ok = 0
    mut_rng = np.random.default_rng(2026)

    for rep in range(n_reps):
        seed = 10_000 + rep
        demo = Demography(pop_sizes=[Ne, Ne])

        # Migration: active from t_mig_cease to t_split (backwards)
        if m_eff > 0:
            # Set migration from t_mig_cease onwards (going backward)
            demo.add_event(('em', t_mig_cease, 0, 1, m_eff))
            demo.add_event(('em', t_mig_cease, 1, 0, m_eff))

        # At split: merge pops, zero migration
        demo.add_event(('ej', t_split_gen, 1, 0))
        demo.add_inversion_freq_change(t_split_gen, 0, inv_id=0,
                                       p_inv=p_inv_anc)
        demo.add_inversion_freq_change(t_split_gen, 0, inv_id=1,
                                       p_inv=p_inv_anc)
        # Ancestral demography
        demo.add_event(('en', t_split_gen, 0, Ne_mid))
        demo.add_event(('en', t_Ne_mid, 0, Ne_anc))

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
                              p_inv=p_inv_3Ra, t_inv=t_inv_gen),
                InversionSpec(bp_left=inv_3Rb[0], bp_right=inv_3Rb[1],
                              p_inv=p_inv_3Rb, t_inv=t_inv_gen),
            ],
            recombination_rate=r,
            seed=seed,
        )
        try:
            ts = sim.simulate()
        except Exception as e:
            print(f"  rep {rep}: FAILED {e}", file=sys.stderr)
            sys.stderr.flush()
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

        if (rep + 1) % 5 == 0:
            elapsed_so_far = time.time() - t0
            rate = elapsed_so_far / (rep + 1)
            eta = rate * (n_reps - rep - 1)
            print(f"  {rep + 1}/{n_reps}  ok={n_ok}  "
                  f"{elapsed_so_far:.0f}s  ETA {eta:.0f}s")
            sys.stdout.flush()

    if n_ok > 0:
        for k in acc:
            acc[k] /= n_ok

    elapsed = time.time() - t0

    d = acc
    da_kf_same = d['dxy_kf_same'] - (d['pi_K'] + d['pi_FS']) / 2
    da_kf_alt = d['dxy_kf_alt'] - (d['pi_K'] + d['pi_FI']) / 2
    da_fs_fi = d['dxy_fs_fi'] - (d['pi_FS'] + d['pi_FI']) / 2
    fst_kf_same = 1.0 - (d['pi_K'] + d['pi_FS']) / 2 / np.maximum(d['dxy_kf_same'], 1e-20)
    fst_kf_alt = 1.0 - (d['pi_K'] + d['pi_FI']) / 2 / np.maximum(d['dxy_kf_alt'], 1e-20)
    fst_fs_fi = 1.0 - (d['pi_FS'] + d['pi_FI']) / 2 / np.maximum(d['dxy_fs_fi'], 1e-20)

    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2
    inv_w = [w for w in range(NW)
             if (inv_3Ra[0] < mid[w] < inv_3Ra[1]) or
                (inv_3Rb[0] < mid[w] < inv_3Rb[1])]
    col_w = [w for w in range(NW) if w not in inv_w]
    fi = float(np.mean([fst_kf_alt[w] for w in inv_w]))
    fc = float(np.mean([fst_kf_alt[w] for w in col_w]))
    fst_ratio = fi / fc if fc != 0 else float('nan')

    res = {
        'label': label, 'm_eff': m_eff,
        'total_ok': n_ok, 'elapsed': elapsed,
        'fst_ratio': fst_ratio, 'fst_inv': fi, 'fst_col': fc,
        **acc,
        'da_kf_same': da_kf_same, 'da_kf_alt': da_kf_alt,
        'da_fs_fi': da_fs_fi,
        'fst_kf_same': fst_kf_same, 'fst_kf_alt': fst_kf_alt,
        'fst_fs_fi': fst_fs_fi,
    }

    np.savez(npz_path, **{k: v for k, v in res.items()
                          if isinstance(v, (np.ndarray, int, float))})

    print(f"Done: {n_ok}/{n_reps} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  FST(K vs F_I): inv={fi:.4f}  col={fc:.4f}  ratio={fst_ratio:.1f}x")
    print(f"  Saved: {npz_path}")
    sys.stdout.flush()

    return res


def plot_scenarios(results):
    n = len(results)
    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    def smooth(y, k=3):
        return np.convolve(y, np.ones(k) / k, mode='same')

    fig, axes = plt.subplots(3, n, figsize=(5 * n, 10),
                             sharey='row', sharex=True, squeeze=False)

    c_kf_same = '#2E7D32'
    c_fs_fi = '#FF8F00'
    c_kf_alt = '#00838F'

    for col, res in enumerate(results):
        ax_dxy = axes[0, col]
        ax_da = axes[1, col]
        ax_fst = axes[2, col]

        ax_dxy.plot(mid, smooth(res['dxy_kf_same']), '-', color=c_kf_same, lw=2,
                    label=r'K vs F$_S$')
        ax_dxy.plot(mid, smooth(res['dxy_fs_fi']), '-', color=c_fs_fi, lw=2,
                    label=r'F$_S$ vs F$_I$')
        ax_dxy.plot(mid, smooth(res['dxy_kf_alt']), '-', color=c_kf_alt, lw=2,
                    label=r'K vs F$_I$')

        m_str = f"m={res['m_eff']:.1e}" if res['m_eff'] > 0 else "no migration"
        ax_dxy.set_title(f'{res["label"]}\n{m_str}, '
                         f'{res["total_ok"]} reps, {res["elapsed"]:.0f}s',
                         fontsize=9, fontweight='bold')
        for (lo, hi) in [inv_3Ra, inv_3Rb]:
            ax_dxy.axvspan(lo, hi, alpha=0.10, color='#90A4AE', zorder=0)
        ax_dxy.set_xlim(0, L)
        if col == 0:
            ax_dxy.set_ylabel(r'$d_{XY}$ (per bp)', fontsize=11)
            ax_dxy.legend(fontsize=7, loc='upper right')

        ax_da.plot(mid, smooth(res['da_kf_same']), '-', color=c_kf_same, lw=2)
        ax_da.plot(mid, smooth(res['da_fs_fi']), '-', color=c_fs_fi, lw=2)
        ax_da.plot(mid, smooth(res['da_kf_alt']), '-', color=c_kf_alt, lw=2)
        ax_da.axhline(0, color='#555', ls=':', lw=0.8)
        for (lo, hi) in [inv_3Ra, inv_3Rb]:
            ax_da.axvspan(lo, hi, alpha=0.10, color='#90A4AE', zorder=0)
        if col == 0:
            ax_da.set_ylabel(r'$D_a$ (per bp)', fontsize=11)

        ax_fst.plot(mid, smooth(res['fst_kf_same']), '-', color=c_kf_same, lw=2)
        ax_fst.plot(mid, smooth(res['fst_fs_fi']), '-', color=c_fs_fi, lw=2)
        ax_fst.plot(mid, smooth(res['fst_kf_alt']), '-', color=c_kf_alt, lw=2)
        ax_fst.axhline(0, color='#555', ls=':', lw=0.8)
        for (lo, hi) in [inv_3Ra, inv_3Rb]:
            ax_fst.axvspan(lo, hi, alpha=0.10, color='#90A4AE', zorder=0)
        ax_fst.set_xlabel('Position (bp)', fontsize=10)
        if col == 0:
            ax_fst.set_ylabel(r'$F_{ST}$ (Hudson)', fontsize=11)

        ax_fst.text(0.98, 0.95, f'FST ratio: {res["fst_ratio"]:.1f}x',
                    transform=ax_fst.transAxes, ha='right', va='top',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

        inv_w = [w for w in range(NW)
                 if (inv_3Ra[0] < mid[w] < inv_3Ra[1]) or
                    (inv_3Rb[0] < mid[w] < inv_3Rb[1])]
        fst_kfi = float(np.mean([res['fst_kf_alt'][w] for w in inv_w]))
        fst_fsfi = float(np.mean([res['fst_fs_fi'][w] for w in inv_w]))
        if fst_kfi > fst_fsfi:
            ordering = r'K-F$_I$ > F$_S$-F$_I$ $\checkmark$'
            color = '#1B5E20'
        else:
            ordering = r'F$_S$-F$_I$ > K-F$_I$'
            color = '#B71C1C'
        ax_fst.text(0.98, 0.80, ordering,
                    transform=ax_fst.transAxes, ha='right', va='top',
                    fontsize=8, color=color,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8))

    fig.suptitle(
        'An. funestus Kir/Fol: 2Ra mating barrier hypothesis\n'
        f'Migration active {t_mig_cease:,}–{t_split_gen:,} gen ago, '
        f'P(triple-S in F)={p_triple_S:.3f}, m_base={m_base:.0e}',
        fontsize=11, fontweight='bold', y=1.03)

    fig.tight_layout()
    out = 'figures/empirical_kir_fol_2Ra.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Updated plot: {out}')
    sys.stdout.flush()


def main():
    print(f"Kir/Fol 2Ra mating barrier test")
    print(f"  P(triple-S in F) = {p_2Ra_S_fol:.2f} x {p_3Ra_S_fol:.2f} x "
          f"{p_3Rb_S_fol:.2f} = {p_triple_S:.3f}")
    print(f"  m_base = {m_base:.0e}")
    print(f"  m_eff (with 2Ra barrier) = {m_base * p_triple_S:.2e}")
    print(f"  Migration window: {t_mig_cease:,} – {t_split_gen:,} gen ago")
    sys.stdout.flush()

    scenarios = [
        ("Free migration", m_base),
        ("2Ra barrier", m_base * p_triple_S),
        ("No migration", 0.0),
    ]

    results = []
    for label, m_eff in scenarios:
        res = run_scenario(label, m_eff, N_REPS)
        results.append(res)
        plot_scenarios(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'scenario':<20} {'m_eff':>10} {'reps':>6} {'time':>8} "
          f"{'FST_inv':>10} {'FST_col':>10} {'ratio':>8}")
    for res in results:
        print(f"  {res['label']:<18} {res['m_eff']:>10.2e} "
              f"{res['total_ok']:>6} {res['elapsed']:>7.0f}s "
              f"{res['fst_inv']:>10.4f} {res['fst_col']:>10.4f} "
              f"{res['fst_ratio']:>7.2f}x")


if __name__ == '__main__':
    main()
