#!/usr/bin/env python3
"""Kir/Fol neutral baseline — incremental rho sweep with realistic demography.

Runs each (rho, Ne_anc) combination sequentially, saves .npz + updates
plot after each combo completes. Cached results are reused on re-run.

Usage:
    .venv/bin/python examples/empirical_kir_fol_parallel.py
    .venv/bin/python examples/empirical_kir_fol_parallel.py --rho 40,100 --Ne-anc 1000000
    .venv/bin/python examples/empirical_kir_fol_parallel.py --rho 40 --reps 50
"""

import argparse
import os
import sys
import time
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import msprime

from msinv import HullSimulator, InversionSpec, Demography


# --- Fixed parameters (Small et al. 2023) ---
Ne = 44_000
mu = 3.55e-9
L = 100_000
t_split_gen = 14_000
t_inv_gen = 385_000
Ne_mid = 100_000  # 14k-60k gen ago
t_Ne_mid = 60_000  # transition to mid-size

p_inv_3Ra = {0: 0.0, 1: 0.73}
p_inv_3Rb = {0: 0.0, 1: 0.40}
p_inv_anc = 0.3

inv_3Ra = (15_000, 45_000)
inv_3Rb = (55_000, 85_000)

n_kir = 10
n_fol_S = 5
n_fol_I = 5

NW = 40

kir = list(range(n_kir))
fol_same = list(range(n_kir, n_kir + n_fol_S))
fol_alt = list(range(n_kir + n_fol_S, n_kir + n_fol_S + n_fol_I))


def rho_to_r(rho):
    return rho / (4 * Ne * L)


def per_window_stats(haps, pos_bp, group_a, group_b, kind="dxy"):
    wins = np.linspace(0, L, NW + 1)
    vals = np.zeros(NW)
    for w in range(NW):
        mask = (pos_bp >= wins[w]) & (pos_bp < wins[w + 1])
        if mask.sum() == 0:
            continue
        if kind == "dxy":
            d = 0
            n = 0
            for a in group_a:
                for b in group_b:
                    d += (haps[a, mask] != haps[b, mask]).sum()
                    n += 1
            vals[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
        else:
            d = 0
            n = 0
            ga = list(group_a)
            for i in range(len(ga)):
                for j in range(i + 1, len(ga)):
                    d += (haps[ga[i], mask] != haps[ga[j], mask]).sum()
                    n += 1
            vals[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return vals


def run_one(rho, ne_anc, n_reps):
    """Run one (rho, Ne_anc) combo sequentially. Returns result dict."""
    r = rho_to_r(rho)
    tag = f"rho{rho}_Nanc{ne_anc // 1000}k"

    # Check cache
    npz_path = f"figures/kir_fol_{tag}.npz"
    if os.path.exists(npz_path):
        print(f"  Loading cached: {npz_path}")
        sys.stdout.flush()
        data = dict(np.load(npz_path, allow_pickle=True))
        for k in ("rho", "Ne_anc", "total_ok"):
            data[k] = int(data[k])
        for k in ("elapsed", "fst_ratio", "fst_inv", "fst_col", "r"):
            data[k] = float(data[k])
        return data

    print(f"\n{'=' * 60}")
    print(f"{tag}  r={r:.3e}  ({n_reps} reps, sequential)")
    print(f"{'=' * 60}")
    sys.stdout.flush()
    t0 = time.time()

    acc = {
        k: np.zeros(NW)
        for k in ["dxy_kf_same", "dxy_kf_alt", "dxy_fs_fi", "pi_K", "pi_FS", "pi_FI"]
    }
    n_ok = 0
    mut_rng = np.random.default_rng(2026)

    for rep in range(n_reps):
        seed = 10_000 + rep
        demo = Demography(pop_sizes=[Ne, Ne])
        demo.add_event(("ej", t_split_gen, 1, 0))
        demo.add_inversion_freq_change(t_split_gen, 0, inv_id=0, p_inv=p_inv_anc)
        demo.add_inversion_freq_change(t_split_gen, 0, inv_id=1, p_inv=p_inv_anc)
        demo.add_event(("en", t_split_gen, 0, Ne_mid))
        demo.add_event(("en", t_Ne_mid, 0, ne_anc))

        sim = HullSimulator(
            sample_config={
                ("S", 0): n_kir,
                ("S", 1): n_fol_S,
                ("I", 1): n_fol_I,
            },
            demography=demo,
            sequence_length=L,
            inversions=[
                InversionSpec(
                    bp_left=inv_3Ra[0],
                    bp_right=inv_3Ra[1],
                    p_inv=p_inv_3Ra,
                    t_inv=t_inv_gen,
                ),
                InversionSpec(
                    bp_left=inv_3Rb[0],
                    bp_right=inv_3Rb[1],
                    p_inv=p_inv_3Rb,
                    t_inv=t_inv_gen,
                ),
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
        mts = msprime.sim_mutations(
            ts, rate=mu, random_seed=mseed, discrete_genome=False
        )
        G = mts.genotype_matrix()
        haps = G.T
        pos_bp = np.array([v.site.position for v in mts.variants()])

        acc["dxy_kf_same"] += per_window_stats(haps, pos_bp, kir, fol_same, "dxy")
        acc["dxy_kf_alt"] += per_window_stats(haps, pos_bp, kir, fol_alt, "dxy")
        acc["dxy_fs_fi"] += per_window_stats(haps, pos_bp, fol_same, fol_alt, "dxy")
        acc["pi_K"] += per_window_stats(haps, pos_bp, kir, None, "pi")
        acc["pi_FS"] += per_window_stats(haps, pos_bp, fol_same, None, "pi")
        acc["pi_FI"] += per_window_stats(haps, pos_bp, fol_alt, None, "pi")
        n_ok += 1

        if (rep + 1) % 5 == 0:
            elapsed_so_far = time.time() - t0
            rate = elapsed_so_far / (rep + 1)
            eta = rate * (n_reps - rep - 1)
            print(
                f"  {rep + 1}/{n_reps}  ok={n_ok}  "
                f"{elapsed_so_far:.0f}s  ETA {eta:.0f}s"
            )
            sys.stdout.flush()

    if n_ok > 0:
        for k in acc:
            acc[k] /= n_ok

    elapsed = time.time() - t0

    # Derived stats
    d = acc
    da_kf_same = d["dxy_kf_same"] - (d["pi_K"] + d["pi_FS"]) / 2
    da_kf_alt = d["dxy_kf_alt"] - (d["pi_K"] + d["pi_FI"]) / 2
    da_fs_fi = d["dxy_fs_fi"] - (d["pi_FS"] + d["pi_FI"]) / 2
    fst_kf_same = 1.0 - (d["pi_K"] + d["pi_FS"]) / 2 / np.maximum(
        d["dxy_kf_same"], 1e-20
    )
    fst_kf_alt = 1.0 - (d["pi_K"] + d["pi_FI"]) / 2 / np.maximum(d["dxy_kf_alt"], 1e-20)
    fst_fs_fi = 1.0 - (d["pi_FS"] + d["pi_FI"]) / 2 / np.maximum(d["dxy_fs_fi"], 1e-20)

    # FST ratio
    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2
    inv_w = [
        w
        for w in range(NW)
        if (inv_3Ra[0] < mid[w] < inv_3Ra[1]) or (inv_3Rb[0] < mid[w] < inv_3Rb[1])
    ]
    col_w = [w for w in range(NW) if w not in inv_w]
    fi = float(np.mean([fst_kf_alt[w] for w in inv_w]))
    fc = float(np.mean([fst_kf_alt[w] for w in col_w]))
    fst_ratio = fi / fc if fc != 0 else float("nan")

    res = {
        "rho": rho,
        "r": r,
        "Ne_anc": ne_anc,
        "total_ok": n_ok,
        "elapsed": elapsed,
        "fst_ratio": fst_ratio,
        "fst_inv": fi,
        "fst_col": fc,
        **acc,
        "da_kf_same": da_kf_same,
        "da_kf_alt": da_kf_alt,
        "da_fs_fi": da_fs_fi,
        "fst_kf_same": fst_kf_same,
        "fst_kf_alt": fst_kf_alt,
        "fst_fs_fi": fst_fs_fi,
    }

    # Save npz
    np.savez(
        npz_path,
        **{k: v for k, v in res.items() if isinstance(v, (np.ndarray, int, float))},
    )

    print(f"Done: {n_ok}/{n_reps} in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  FST(K vs F_I): inv={fi:.4f}  col={fc:.4f}  ratio={fst_ratio:.1f}x")
    print(f"  Saved: {npz_path}")
    sys.stdout.flush()

    return res


def plot_results(all_results):
    """Plot all completed results."""
    n = len(all_results)
    if n == 0:
        return

    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    def smooth(y, k=3):
        return np.convolve(y, np.ones(k) / k, mode="same")

    fig, axes = plt.subplots(
        3, n, figsize=(5 * n, 10), sharey="row", sharex=True, squeeze=False
    )

    c_kf_same = "#2E7D32"
    c_fs_fi = "#FF8F00"
    c_kf_alt = "#00838F"

    for col, res in enumerate(all_results):
        rho = res["rho"]
        ne_anc = res["Ne_anc"]
        ax_dxy = axes[0, col]
        ax_da = axes[1, col]
        ax_fst = axes[2, col]

        ax_dxy.plot(
            mid,
            smooth(res["dxy_kf_same"]),
            "-",
            color=c_kf_same,
            lw=2,
            label=r"K vs F$_S$",
        )
        ax_dxy.plot(
            mid,
            smooth(res["dxy_fs_fi"]),
            "-",
            color=c_fs_fi,
            lw=2,
            label=r"F$_S$ vs F$_I$",
        )
        ax_dxy.plot(
            mid,
            smooth(res["dxy_kf_alt"]),
            "-",
            color=c_kf_alt,
            lw=2,
            label=r"K vs F$_I$",
        )
        ax_dxy.set_title(
            f"$\\rho$={rho}, N_anc={ne_anc // 1000}k\n"
            f"{res['total_ok']} reps, {res['elapsed']:.0f}s",
            fontsize=9,
            fontweight="bold",
        )
        for lo, hi in [inv_3Ra, inv_3Rb]:
            ax_dxy.axvspan(lo, hi, alpha=0.10, color="#90A4AE", zorder=0)
        ax_dxy.set_xlim(0, L)
        if col == 0:
            ax_dxy.set_ylabel(r"$d_{XY}$ (per bp)", fontsize=11)
            ax_dxy.legend(fontsize=7, loc="upper right")

        ax_da.plot(mid, smooth(res["da_kf_same"]), "-", color=c_kf_same, lw=2)
        ax_da.plot(mid, smooth(res["da_fs_fi"]), "-", color=c_fs_fi, lw=2)
        ax_da.plot(mid, smooth(res["da_kf_alt"]), "-", color=c_kf_alt, lw=2)
        ax_da.axhline(0, color="#555", ls=":", lw=0.8)
        for lo, hi in [inv_3Ra, inv_3Rb]:
            ax_da.axvspan(lo, hi, alpha=0.10, color="#90A4AE", zorder=0)
        if col == 0:
            ax_da.set_ylabel(r"$D_a$ (per bp)", fontsize=11)

        ax_fst.plot(mid, smooth(res["fst_kf_same"]), "-", color=c_kf_same, lw=2)
        ax_fst.plot(mid, smooth(res["fst_fs_fi"]), "-", color=c_fs_fi, lw=2)
        ax_fst.plot(mid, smooth(res["fst_kf_alt"]), "-", color=c_kf_alt, lw=2)
        ax_fst.axhline(0, color="#555", ls=":", lw=0.8)
        for lo, hi in [inv_3Ra, inv_3Rb]:
            ax_fst.axvspan(lo, hi, alpha=0.10, color="#90A4AE", zorder=0)
        ax_fst.set_xlabel("Position (bp)", fontsize=10)
        if col == 0:
            ax_fst.set_ylabel(r"$F_{ST}$ (Hudson)", fontsize=11)

        ax_fst.text(
            0.98,
            0.95,
            f"FST ratio: {res['fst_ratio']:.1f}x",
            transform=ax_fst.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "An. funestus Kir/Fol neutral baseline\n"
        "Ne: 44k → 100k (14k gen) → N_anc (60k gen)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    out = "figures/empirical_kir_fol.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Updated plot: {out}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rho", type=str, default="40", help="Comma-separated rho values (default: 40)"
    )
    parser.add_argument(
        "--Ne-anc",
        type=str,
        default="1000000",
        help="Comma-separated Ne_anc values (default: 1000000)",
    )
    parser.add_argument(
        "--reps", type=int, default=25, help="Number of replicates (default: 25)"
    )
    args = parser.parse_args()

    rho_values = [int(x.strip()) for x in args.rho.split(",")]
    ne_anc_values = [int(x.strip()) for x in args.Ne_anc.split(",")]

    combos = [(rho, na) for rho in rho_values for na in ne_anc_values]
    print(f"Kir/Fol neutral baseline — {len(combos)} combos, {args.reps} reps each:")
    for rho, na in combos:
        print(f"  rho={rho}, Ne_anc={na:,}")
    sys.stdout.flush()

    all_results = []
    for rho, na in combos:
        res = run_one(rho, na, args.reps)
        all_results.append(res)
        all_results.sort(key=lambda x: (x["Ne_anc"], x["rho"]))
        plot_results(all_results)

    # Final summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'rho':>6} {'Ne_anc':>10} {'reps':>6} {'time':>8} "
        f"{'FST_inv':>10} {'FST_col':>10} {'ratio':>8}"
    )
    for res in all_results:
        print(
            f"  {res['rho']:>4} {int(res['Ne_anc']):>10,} "
            f"{res['total_ok']:>6} {res['elapsed']:>7.0f}s "
            f"{res['fst_inv']:>10.4f} {res['fst_col']:>10.4f} "
            f"{res['fst_ratio']:>7.2f}x"
        )


if __name__ == "__main__":
    main()
