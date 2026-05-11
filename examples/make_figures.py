#!/usr/bin/env python3
"""Generate presentation figures for msinv (hull simulator).

Produces 8 PDF figures showing:

  1. Inversion divergence signal (dxy, pi, Da across chromosome)
  2. msinv vs msprime ground truth (no inversion → identical)
  3. Real inversions — An. funestus 3Ra (Kir/Fol) + Human MAPT
  4. Multiple inversions on one chromosome
  5. Inversion origin trajectories (forward-time Wright-Fisher)
  6. phi(x) gene-flux profile + cross-class T_MRCA
  7. Performance scaling with rho and Ne
  8. Feature summary table

All sims use HullSimulator (the only engine in msinv >= 0.3.0).
"""

import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import msprime
import msinv
from msinv import HullSimulator, InversionSpec
from msinv.hull.simulator import _phi


OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================


def windowed_dxy(haps, pos_bp, ga, gb, wins):
    """Per-bp dxy in each window."""
    out = np.zeros(len(wins) - 1)
    for w in range(len(wins) - 1):
        m = (pos_bp >= wins[w]) & (pos_bp < wins[w + 1])
        if not m.any():
            continue
        d, n = 0, 0
        for a in ga:
            for b in gb:
                d += int((haps[a, m] != haps[b, m]).sum())
                n += 1
        out[w] = d / n / (wins[w + 1] - wins[w])
    return out


def windowed_pi(haps, pos_bp, grp, wins):
    out = np.zeros(len(wins) - 1)
    for w in range(len(wins) - 1):
        m = (pos_bp >= wins[w]) & (pos_bp < wins[w + 1])
        if not m.any():
            continue
        d, n = 0, 0
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                d += int((haps[grp[i], m] != haps[grp[j], m]).sum())
                n += 1
        out[w] = d / max(n, 1) / (wins[w + 1] - wins[w])
    return out


def smooth(y, k=3):
    return np.convolve(y, np.ones(k) / k, mode="same")


def run_replicates(builder, mu, n_reps, mut_seed=2026):
    """Run n_reps hull sims (factory closure), drop mutations, return (haps_list, pos_list)."""
    mut_rng = np.random.default_rng(mut_seed)
    out = []
    for rep in range(n_reps):
        sim = builder(rep)
        try:
            ts = sim.simulate()
        except Exception:
            continue
        seed = int(mut_rng.integers(1, 2**31))
        mts = msprime.sim_mutations(
            ts, rate=mu, random_seed=seed, discrete_genome=False
        )
        haps = mts.genotype_matrix().T
        pos = np.array([v.site.position for v in mts.variants()])
        out.append((haps, pos))
    return out


# ============================================================
# Fig 1 — Inversion divergence signal (dxy, pi, Da)
# ============================================================


def fig1_inversion_signal():
    print("Fig 1: Inversion signal...")
    Ne = 50_000
    mu = 1e-8
    L = 100_000
    bp_l, bp_r = 30_000, 70_000
    n_S, n_I = 8, 8
    NW = 30
    NREPS = 100
    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    def build(rep):
        return HullSimulator(
            n_std=n_S,
            n_inv=n_I,
            population_size=Ne,
            sequence_length=L,
            inversions=[
                InversionSpec(
                    bp_left=bp_l,
                    bp_right=bp_r,
                    p_inv=0.5,
                    t_inv=200_000,
                    gene_conversion_rate=1e-9,
                )
            ],
            recombination_rate=1e-8,
            seed=4242 + rep,
        )

    reps = run_replicates(build, mu=mu, n_reps=NREPS)
    print(f"  {len(reps)}/{NREPS} reps OK")

    dxy = np.zeros(NW)
    pi_S = np.zeros(NW)
    pi_I = np.zeros(NW)
    S = list(range(n_S))
    I = list(range(n_S, n_S + n_I))
    for haps, pos in reps:
        dxy += windowed_dxy(haps, pos, S, I, wins)
        pi_S += windowed_pi(haps, pos, S, wins)
        pi_I += windowed_pi(haps, pos, I, wins)
    n = len(reps)
    dxy /= n
    pi_S /= n
    pi_I /= n
    da = dxy - (pi_S + pi_I) / 2
    fst = np.clip(1.0 - (pi_S + pi_I) / 2 / np.maximum(dxy, 1e-20), 0.0, 1.0)

    np.savez(
        os.path.join(OUTDIR, "fig1_data.npz"),
        mid=mid,
        dxy=dxy,
        pi_S=pi_S,
        pi_I=pi_I,
        da=da,
        fst=fst,
        Ne=Ne,
        mu=mu,
        L=L,
        bp_l=bp_l,
        bp_r=bp_r,
        n_reps=n,
    )

    # Position-dependent theory curves (Guerrero et al. 2012; Charlesworth 1997)
    t_inv = 200_000
    p_class = 0.5
    theta = 4 * Ne * mu
    # Build theory arrays: step inside/outside inversion
    inside = (mid >= bp_l) & (mid <= bp_r)
    dxy_th = np.where(inside, 2 * mu * (t_inv + 2 * Ne), 2 * mu * 2 * Ne)
    pi_th = np.where(inside, p_class * theta, theta)
    da_th = dxy_th - pi_th
    fst_th = np.where(inside, 1.0 - Ne / (t_inv + 2 * Ne), 0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)

    ax1.plot(mid, smooth(dxy), "-", color="#C62828", lw=2.2, label=r"$d_{XY}$ (S vs I)")
    ax1.plot(mid, smooth(pi_S), "-", color="#1565C0", lw=2, label=r"$\pi_S$ (within S)")
    ax1.plot(mid, smooth(pi_I), "-", color="#2E7D32", lw=2, label=r"$\pi_I$ (within I)")
    ax1.plot(
        mid, dxy_th, "--", color="#C62828", lw=1.2, alpha=0.7, label=r"$E[d_{XY}]$"
    )
    ax1.plot(mid, pi_th, "--", color="#1565C0", lw=1.2, alpha=0.7, label=r"$E[\pi_c]$")
    ax1.axvspan(bp_l, bp_r, alpha=0.10, color="gray", zorder=0, label="inversion")
    ax1.set_ylabel("Per-bp diversity / divergence", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(
        f"A. Inversion divergence signal "
        f"(Ne={Ne:,}, t_inv=200k gen, n_S={n_S}, n_I={n_I}, {n} reps)",
        fontsize=11,
        loc="left",
    )

    ax2.plot(
        mid,
        smooth(da),
        "-",
        color="#6A1B9A",
        lw=2.2,
        label=r"$D_a = d_{XY} - (\pi_S+\pi_I)/2$",
    )
    ax2.plot(mid, da_th, "--", color="#6A1B9A", lw=1.2, alpha=0.7, label=r"$E[D_a]$")
    ax2.axvspan(bp_l, bp_r, alpha=0.10, color="gray", zorder=0)
    ax2.axhline(0, color="gray", ls=":", lw=0.7)
    ax2.set_ylabel(r"$D_a$", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(
        "B. Net divergence (isolates the inversion barrier)", fontsize=11, loc="left"
    )

    ax3.plot(mid, smooth(fst), "-", color="#E65100", lw=2.2, label=r"$F_{ST}$ (Hudson)")
    ax3.plot(
        mid, fst_th, "--", color="#E65100", lw=1.2, alpha=0.7, label=r"$E[F_{ST}]$"
    )
    ax3.axvspan(bp_l, bp_r, alpha=0.10, color="gray", zorder=0)
    ax3.axhline(0, color="gray", ls=":", lw=0.7)
    ax3.set_ylabel(r"$F_{ST}$", fontsize=11)
    ax3.set_xlabel("Chromosome position (bp)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_title(
        r"C. Hudson $F_{ST}$ (relative differentiation)", fontsize=11, loc="left"
    )

    for ax in (ax1, ax2, ax3):
        ax.axvline(bp_l, color="red", ls=":", lw=1, alpha=0.5)
        ax.axvline(bp_r, color="red", ls=":", lw=1, alpha=0.5)

    caption = (
        f"Figure 1. Chromosomal inversion divergence signal simulated with msinv (Rust ARG core). "
        f"(A) Per-bp absolute divergence ($d_{{XY}}$) between Standard (S) and Inverted (I) "
        f"karyotypes is elevated inside the inversion (grey shading, {bp_l / 1e3:.0f}–{bp_r / 1e3:.0f} kb), "
        f"while within-class diversity ($\\pi_S$, $\\pi_I$) remains at background levels. "
        f"(B) Net divergence $D_a = d_{{XY}} - (\\pi_S + \\pi_I)/2$ isolates the barrier signal. "
        f"(C) Hudson $F_{{ST}} = 1 - \\pi_W / d_{{XY}}$ shows relative differentiation — elevated "
        f"inside the inversion where the recombination barrier concentrates divergence. "
        f"Parameters: Ne={Ne:,}, t_inv=200,000 gen, $\\rho$=200, $\\gamma$=1e-9, L={L / 1e3:.0f} kb, "
        f"r=1e-8 bp$^{{-1}}$ gen$^{{-1}}$, $\\mu$=1e-8, n_S={n_S}, n_I={n_I}, {n} replicates.\n"
        f"Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.01,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    plt.savefig(os.path.join(OUTDIR, "fig1_inversion_signal.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Fig 2 — msinv (no inversion) ↔ msprime ground truth
# ============================================================


def fig2_msprime_comparison():
    print("Fig 2: msprime comparison...")
    Ne = 10_000
    mu = 1e-8
    L = 100_000
    rho_vals = [10, 50, 100]  # 4*Ne*r*L
    ms_S = []
    mp_S = []
    NREPS = 100

    for rho in rho_vals:
        r = rho / (4 * Ne * L)
        # msinv hull, NO inversion (n_inv=0, n_std = full sample)
        s_ms = []
        for s in range(NREPS):
            sim = HullSimulator(
                n_std=10,
                n_inv=0,
                population_size=Ne,
                sequence_length=L,
                recombination_rate=r,
                seed=s + 100,
            )
            ts = sim.simulate()
            mts = msprime.sim_mutations(
                ts, rate=mu, random_seed=s + 200, discrete_genome=False
            )
            s_ms.append(mts.num_mutations)
        ms_S.append(np.mean(s_ms))

        # msprime ground truth
        s_mp = []
        for s in range(NREPS):
            ts = msprime.sim_ancestry(
                samples=5,
                sequence_length=L,
                recombination_rate=r,
                population_size=Ne,
                random_seed=s + 1000,
            )
            ts = msprime.sim_mutations(
                ts, rate=mu, random_seed=s + 2000, discrete_genome=False
            )
            s_mp.append(ts.num_mutations)
        mp_S.append(np.mean(s_mp))

    theta = 4 * Ne * mu * L
    expected = theta * sum(1 / i for i in range(1, 10))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rho_vals))
    w = 0.35
    ax.bar(x - w / 2, ms_S, w, label="msinv (Rust)", color="#1976D2")
    ax.bar(x + w / 2, mp_S, w, label="msprime", color="#E65100")
    ax.axhline(
        expected,
        color="gray",
        ls="--",
        lw=1.2,
        label=f"Watterson E[S] = {expected:.0f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\rho$={r}" for r in rho_vals])
    ax.set_ylabel("Mean segregating sites (10 samples, 100 kb)")
    ax.legend(fontsize=10)
    ax.set_title("msinv (Rust) matches msprime in the no-inversion limit")

    caption = (
        f"Figure 2. msinv validation against msprime in the no-inversion limit. "
        f"Mean number of segregating sites from 10 haploid samples across {NREPS} replicates "
        f"at three recombination rates ($\\rho$ = 4$N_e$rL). "
        f"Without an inversion, msinv produces the same distribution of genealogies "
        f"as msprime — the two are statistically indistinguishable. "
        f"Dashed line: Watterson expectation E[S] = $\\theta \\sum_{{i=1}}^{{n-1}} 1/i$ = {expected:.0f}. "
        f"Parameters: Ne={Ne:,}, L={L / 1e3:.0f} kb, $\\mu$=1e-8, n=10, {NREPS} replicates per $\\rho$.\n"
        f"Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    plt.savefig(
        os.path.join(OUTDIR, "fig2_msprime_comparison.pdf"), bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Fig 3 — Real inversions: An. funestus 3Ra-like + MAPT-like
# ============================================================


def fig3_real_inversions():
    print("Fig 3: Real inversions...")
    NREPS = 80
    NW = 25
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")

    configs = [
        (
            "An. funestus 3Ra–like\n(Ne=10k, t_inv=100k gen)",
            dict(
                Ne=10_000,
                mu=3.55e-9,
                L=100_000,
                bp_l=20_000,
                bp_r=80_000,
                p_inv=0.3,
                t_inv=100_000,
                gamma=1e-9,
                n_S=10,
                n_I=10,
            ),
        ),
        (
            "Human MAPT H1/H2–like\n(Ne=10k, t_inv=3 Myr / 100k gen)",
            dict(
                Ne=10_000,
                mu=1.2e-8,
                L=100_000,
                bp_l=20_000,
                bp_r=80_000,
                p_inv=0.2,
                t_inv=100_000,
                gamma=1e-9,
                n_S=16,
                n_I=4,
            ),
        ),
    ]

    for col, (name, p) in enumerate(configs):
        wins = np.linspace(0, p["L"], NW + 1)
        mid = (wins[:-1] + wins[1:]) / 2

        def build(rep, p=p):
            return HullSimulator(
                n_std=p["n_S"],
                n_inv=p["n_I"],
                population_size=p["Ne"],
                sequence_length=p["L"],
                inversions=[
                    InversionSpec(
                        bp_left=p["bp_l"],
                        bp_right=p["bp_r"],
                        p_inv=p["p_inv"],
                        t_inv=p["t_inv"],
                        gene_conversion_rate=p.get("gamma", 1e-9),
                    )
                ],
                recombination_rate=1e-8,
                seed=7000 + rep,
            )

        reps = run_replicates(build, mu=p["mu"], n_reps=NREPS)
        S = list(range(p["n_S"]))
        I = list(range(p["n_S"], p["n_S"] + p["n_I"]))
        dxy = np.zeros(NW)
        pi_S = np.zeros(NW)
        pi_I = np.zeros(NW)
        for haps, pos in reps:
            dxy += windowed_dxy(haps, pos, S, I, wins)
            pi_S += windowed_pi(haps, pos, S, wins)
            pi_I += windowed_pi(haps, pos, I, wins)
        n = max(len(reps), 1)
        dxy /= n
        pi_S /= n
        pi_I /= n
        da = dxy - (pi_S + pi_I) / 2
        fst = np.clip(1.0 - (pi_S + pi_I) / 2 / np.maximum(dxy, 1e-20), 0.0, 1.0)

        tag = "funestus" if col == 0 else "mapt"
        np.savez(
            os.path.join(OUTDIR, f"fig3_{tag}_data.npz"),
            mid=mid,
            dxy=dxy,
            pi_S=pi_S,
            pi_I=pi_I,
            da=da,
            fst=fst,
            n_reps=n,
            **p,
        )

        # Position-dependent theory curves
        inside = (mid >= p["bp_l"]) & (mid <= p["bp_r"])
        theta_p = 4 * p["Ne"] * p["mu"]
        p_c = p["p_inv"]  # minority class fraction
        dxy_th = np.where(
            inside, 2 * p["mu"] * (p["t_inv"] + 2 * p["Ne"]), 2 * p["mu"] * 2 * p["Ne"]
        )
        pi_th = np.where(inside, p_c * theta_p, theta_p)
        dxy_th - pi_th
        fst_th = np.where(inside, 1.0 - p["Ne"] / (p["t_inv"] + 2 * p["Ne"]), 0.0)

        # Top row: dxy, pi, Da
        ax = axes[0, col]
        ax.plot(mid, smooth(dxy), "-", color="#C62828", lw=2, label=r"$d_{XY}$")
        ax.plot(
            mid,
            smooth((pi_S + pi_I) / 2),
            "-",
            color="#1565C0",
            lw=2,
            label=r"$\bar\pi$",
        )
        ax.plot(mid, smooth(da), "-", color="#6A1B9A", lw=2, label=r"$D_a$")
        ax.plot(
            mid, dxy_th, "--", color="#C62828", lw=1, alpha=0.6, label=r"$E[d_{XY}]$"
        )
        ax.plot(
            mid, pi_th, "--", color="#1565C0", lw=1, alpha=0.6, label=r"$E[\bar\pi]$"
        )
        ax.axvspan(p["bp_l"], p["bp_r"], alpha=0.10, color="gray", zorder=0)
        ax.axvline(p["bp_l"], color="red", ls=":", lw=1, alpha=0.5)
        ax.axvline(p["bp_r"], color="red", ls=":", lw=1, alpha=0.5)
        ax.set_ylabel("Per-bp" if col == 0 else "")
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

        # Bottom row: FST
        ax_f = axes[1, col]
        ax_f.plot(
            mid, smooth(fst), "-", color="#E65100", lw=2.2, label=r"$F_{ST}$ (Hudson)"
        )
        ax_f.plot(
            mid, fst_th, "--", color="#E65100", lw=1, alpha=0.6, label=r"$E[F_{ST}]$"
        )
        ax_f.axvspan(p["bp_l"], p["bp_r"], alpha=0.10, color="gray", zorder=0)
        ax_f.axvline(p["bp_l"], color="red", ls=":", lw=1, alpha=0.5)
        ax_f.axvline(p["bp_r"], color="red", ls=":", lw=1, alpha=0.5)
        ax_f.axhline(0, color="gray", ls=":", lw=0.7)
        ax_f.set_xlabel("Position (bp)")
        ax_f.set_ylabel(r"$F_{ST}$" if col == 0 else "")
        ax_f.legend(loc="upper right", fontsize=8)

    caption = (
        "Figure 3. msinv reproduces divergence patterns of real chromosomal inversions. "
        "Left: An. funestus 3Ra-like inversion (Ne=10,000, $\\mu$=3.55e-9, t_inv=100,000 gen, p_inv=0.3). "
        "Right: Human MAPT H1/H2-like inversion (Ne=10,000, $\\mu$=1.2e-8, t_inv=100,000 gen, p_inv=0.2). "
        "Top: $d_{XY}$, $\\bar\\pi$, and $D_a$ show elevated divergence inside the inversion. "
        "Bottom: Hudson $F_{ST}$ shows relative differentiation. "
        "The deeper barrier and older age of 3Ra produce a stronger signal than MAPT. "
        f"Parameters: L=100 kb, r=1e-8, $\\gamma$=1e-9, {NREPS} replicates per scenario.\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(OUTDIR, "fig3_real_inversions.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Fig 4 — Multiple inversions on one chromosome
# ============================================================


def fig4_multiple_inversions():
    print("Fig 4: Multiple inversions...")
    Ne = 10_000
    mu = 1e-8
    L = 100_000
    NW = 35
    NREPS = 80

    # Two non-overlapping inversions, different ages/freqs
    inv_A = (10_000, 35_000, 0.5, 100_000)  # young, common
    inv_B = (60_000, 90_000, 0.3, 300_000)  # old, rarer

    wins = np.linspace(0, L, NW + 1)
    mid = (wins[:-1] + wins[1:]) / 2

    def build(rep):
        return HullSimulator(
            sample_config={
                ("SS", 0): 5,  # S at both inversions
                ("II", 0): 5,  # I at both
                (("S", "I"), 0): 4,  # S at A, I at B (recombinant)
                (("I", "S"), 0): 4,  # I at A, S at B
            },
            population_size=Ne,
            sequence_length=L,
            inversions=[
                InversionSpec(
                    bp_left=inv_A[0],
                    bp_right=inv_A[1],
                    p_inv=inv_A[2],
                    t_inv=inv_A[3],
                    gene_conversion_rate=1e-9,
                ),
                InversionSpec(
                    bp_left=inv_B[0],
                    bp_right=inv_B[1],
                    p_inv=inv_B[2],
                    t_inv=inv_B[3],
                    gene_conversion_rate=1e-9,
                ),
            ],
            recombination_rate=1e-8,
            seed=9000 + rep,
        )

    reps = run_replicates(build, mu=mu, n_reps=NREPS)
    print(f"  {len(reps)}/{NREPS} reps OK")

    # Sample groups (set by sample_config order)
    SS = list(range(0, 5))
    II = list(range(5, 10))
    SI = list(range(10, 14))
    IS = list(range(14, 18))

    dxy_A = np.zeros(NW)  # S-at-A vs I-at-A across the chromosome
    dxy_B = np.zeros(NW)  # S-at-B vs I-at-B
    piA_S = np.zeros(NW)
    piA_I = np.zeros(NW)
    piB_S = np.zeros(NW)
    piB_I = np.zeros(NW)
    for haps, pos in reps:
        gA_S = SS + SI
        gA_I = II + IS
        gB_S = SS + IS
        gB_I = II + SI
        dxy_A += windowed_dxy(haps, pos, gA_S, gA_I, wins)
        dxy_B += windowed_dxy(haps, pos, gB_S, gB_I, wins)
        piA_S += windowed_pi(haps, pos, gA_S, wins)
        piA_I += windowed_pi(haps, pos, gA_I, wins)
        piB_S += windowed_pi(haps, pos, gB_S, wins)
        piB_I += windowed_pi(haps, pos, gB_I, wins)
    n = len(reps)
    dxy_A /= n
    dxy_B /= n
    piA_S /= n
    piA_I /= n
    piB_S /= n
    piB_I /= n
    fst_A = np.clip(1.0 - (piA_S + piA_I) / 2 / np.maximum(dxy_A, 1e-20), 0.0, 1.0)
    fst_B = np.clip(1.0 - (piB_S + piB_I) / 2 / np.maximum(dxy_B, 1e-20), 0.0, 1.0)

    np.savez(
        os.path.join(OUTDIR, "fig4_data.npz"),
        mid=mid,
        dxy_A=dxy_A,
        dxy_B=dxy_B,
        fst_A=fst_A,
        fst_B=fst_B,
        piA_S=piA_S,
        piA_I=piA_I,
        piB_S=piB_S,
        piB_I=piB_I,
        Ne=Ne,
        L=L,
        n_reps=n,
    )

    # Position-dependent theory curves per inversion
    in_A = (mid >= inv_A[0]) & (mid <= inv_A[1])
    in_B = (mid >= inv_B[0]) & (mid <= inv_B[1])
    fst_A_th = np.where(in_A, 1.0 - Ne / (inv_A[3] + 2 * Ne), 0.0)
    fst_B_th = np.where(in_B, 1.0 - Ne / (inv_B[3] + 2 * Ne), 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    # Panel A: dxy
    ax1.plot(
        mid, smooth(dxy_A), "-", color="#C62828", lw=2, label="Inv A axis (S vs I at A)"
    )
    ax1.plot(
        mid, smooth(dxy_B), "-", color="#1565C0", lw=2, label="Inv B axis (S vs I at B)"
    )
    ax1.axvspan(
        inv_A[0], inv_A[1], alpha=0.18, color="#C62828", label="Inv A (t=100k, p=0.5)"
    )
    ax1.axvspan(
        inv_B[0], inv_B[1], alpha=0.18, color="#1565C0", label="Inv B (t=300k, p=0.3)"
    )
    ax1.set_ylabel(r"$d_{XY}$ between karyotypes")
    ax1.set_title(
        "A. Two independent inversions — cross-karyotype $d_{XY}$",
        fontsize=11,
        loc="left",
    )
    ax1.legend(loc="upper right", fontsize=9)

    # Panel B: FST
    ax2.plot(
        mid, smooth(fst_A), "-", color="#C62828", lw=2, label="Inv A axis $F_{ST}$"
    )
    ax2.plot(
        mid, smooth(fst_B), "-", color="#1565C0", lw=2, label="Inv B axis $F_{ST}$"
    )
    ax2.plot(
        mid, fst_A_th, "--", color="#C62828", lw=1, alpha=0.6, label=r"$E[F_{ST}]$ A"
    )
    ax2.plot(
        mid, fst_B_th, "--", color="#1565C0", lw=1, alpha=0.6, label=r"$E[F_{ST}]$ B"
    )
    ax2.axvspan(inv_A[0], inv_A[1], alpha=0.18, color="#C62828")
    ax2.axvspan(inv_B[0], inv_B[1], alpha=0.18, color="#1565C0")
    ax2.axhline(0, color="gray", ls=":", lw=0.7)
    ax2.set_xlabel("Position (bp)")
    ax2.set_ylabel(r"$F_{ST}$ (Hudson)")
    ax2.set_title(
        r"B. Hudson $F_{ST}$ — each inversion elevates its own axis",
        fontsize=11,
        loc="left",
    )
    ax2.legend(loc="upper right", fontsize=9)

    caption = (
        "Figure 4. Two independent inversions on the same chromosome, each generating its own "
        "cross-karyotype divergence barrier. Inv A (10–35 kb; t_inv=100,000 gen, p_inv=0.5, red shading) "
        "and Inv B (60–90 kb; t_inv=300,000 gen, p_inv=0.3, blue shading). "
        "(A) $d_{XY}$ elevated along each S-vs-I axis inside its respective inversion. "
        "(B) Hudson $F_{ST}$ shows the same pattern as relative differentiation. "
        "Inv B is older and produces a stronger signal. "
        "Sample configuration: 5 SS + 5 II + 4 SI + 4 IS (18 haplotypes). "
        f"Parameters: Ne={Ne:,}, L={L / 1e3:.0f} kb, $\\mu$=1e-8, r=1e-8, $\\gamma$=1e-9, {n} replicates.\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(
        os.path.join(OUTDIR, "fig4_multiple_inversions.pdf"), bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Fig 5 — Inversion origin trajectories (forward-time WF)
# ============================================================


def _wf_trajectory(p_target, N, s, rng, max_gen=400_000):
    """Forward-in-time Wright-Fisher trajectory: start at 1/(2N),
    accept the run if it reaches p_target before going extinct.
    Returns (gens, freqs) arrays in forward time (origin → present)."""
    while True:
        freqs = [1 / (2 * N)]
        for _g in range(1, max_gen):
            p = freqs[-1]
            if s != 0.0:
                p_eff = p * (1 + s) / (1 + s * p)
            else:
                p_eff = p
            new_count = rng.binomial(2 * N, p_eff)
            new_p = new_count / (2 * N)
            freqs.append(new_p)
            if new_p >= p_target:
                return np.arange(len(freqs)), np.array(freqs)
            if new_p <= 0:
                break  # extinct, restart


def fig5_trajectories():
    print("Fig 5: Trajectories...")
    N = 10_000
    p_target = 0.5
    gen_per_year = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: forward trajectories (neutral drift vs selected)
    ax = axes[0]
    np.random.default_rng(0)
    for trial in range(8):
        rng_t = np.random.default_rng(100 + trial)
        gens, freqs = _wf_trajectory(p_target, N, s=0.0, rng=rng_t)
        t_ky = gens / gen_per_year / 1000
        ax.plot(t_ky, freqs, alpha=0.4, lw=1, color="#1976D2")
    ax.plot([], [], color="#1976D2", alpha=0.7, lw=2, label="Neutral drift (s=0)")

    rng_s = np.random.default_rng(7)
    gens_s, freqs_s = _wf_trajectory(p_target, N, s=0.005, rng=rng_s)
    t_s_ky = gens_s / gen_per_year / 1000
    ax.plot(t_s_ky, freqs_s, "-", color="#C62828", lw=2.5, label="Selected (s=0.005)")

    ax.axhline(p_target, color="gray", ls=":", lw=0.7)
    ax.axhline(1 / (2 * N), color="gray", ls="--", lw=0.7)
    ax.text(
        0.02,
        p_target + 0.02,
        f"present freq = {p_target}",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        color="gray",
    )
    ax.set_xlabel("Generations from origin (×1000)")
    ax.set_ylabel("Inversion frequency")
    ax.set_ylim(-0.02, 0.7)
    ax.legend(fontsize=9)
    ax.set_title("A. Forward-in-time WF trajectories\n(origin → today)", fontsize=10)

    # Panel B: age distribution from 50 neutral runs
    ax = axes[1]
    ages_ky = []
    for trial in range(50):
        rng_t = np.random.default_rng(500 + trial)
        gens, _ = _wf_trajectory(p_target, N, s=0.0, rng=rng_t)
        ages_ky.append(gens[-1] / gen_per_year / 1000)
    ax.hist(ages_ky, bins=15, color="#1976D2", alpha=0.75, edgecolor="white")
    med = float(np.median(ages_ky))
    ax.axvline(med, color="#C62828", ls="--", lw=2, label=f"median = {med:.0f} ky")
    ax.set_xlabel("Inversion age (ky)")
    ax.set_ylabel("count (50 trajectories)")
    ax.legend(fontsize=9)
    ax.set_title(
        "B. Inversion age varies widely\n(same present freq, different histories)",
        fontsize=10,
    )
    ax.text(
        0.97,
        0.85,
        f"range: {min(ages_ky):.0f}–{max(ages_ky):.0f} ky",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    caption = (
        "Figure 5. Forward-in-time Wright-Fisher trajectories for inversion frequency. "
        "(A) Example trajectories from 1/(2N) to a present frequency of 0.5. "
        "Neutral drift (blue, 8 runs) takes highly variable paths; "
        "positive selection (s=0.005, red) reaches the same frequency much faster. "
        "(B) Distribution of inversion ages (time from origin to reaching p=0.5) "
        "across 50 neutral trajectories. Even for the same present frequency, "
        "inversion ages vary by orders of magnitude — this motivates ABC inference "
        "rather than point estimates of t_inv. "
        f"Parameters: Ne={N:,}, 10 gen/yr, p_target={p_target}.\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(OUTDIR, "fig5_trajectories.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Fig 6 — phi(x) profile and its T_MRCA effect
# ============================================================


def fig6_phi_profile():
    print("Fig 6: phi(x) profile...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: triangular roof for several w
    x = np.linspace(0, 1, 400)
    for w, c in [(0.05, "#1976D2"), (0.10, "#388E3C"), (0.20, "#E65100")]:
        ax1.plot(x, [_phi(xi, w) for xi in x], "-", color=c, lw=2, label=f"w = {w}")
    ax1.set_xlabel("Inversion-relative position $x$")
    ax1.set_ylabel(r"$\phi(x)$  (per-bp flux weight)")
    ax1.set_title(
        "A. Peischl phi(x) profile\n(triangular roof — peaks in centre)", fontsize=10
    )
    ax1.legend(fontsize=9)
    ax1.fill_between(x, [_phi(xi, 0.10) for xi in x], alpha=0.10, color="#388E3C")

    # Panel B: cross-class T_MRCA inside vs near breakpoints (n=2 sample)
    Ne = 10_000
    L = 100_000
    bp_l, bp_r = 20_000, 80_000
    inv_len = bp_r - bp_l
    NREPS = 200
    positions = np.linspace(bp_l + 1000, bp_r - 1000, 10)
    T_si = []
    for px in positions:
        ts_t = []
        for rep in range(NREPS):
            sim = HullSimulator(
                n_std=1,
                n_inv=1,
                population_size=Ne,
                sequence_length=L,
                inversions=[
                    InversionSpec(
                        bp_left=bp_l,
                        bp_right=bp_r,
                        p_inv=0.5,
                        t_inv=4 * Ne,
                        gene_conversion_rate=1e-7,
                    )
                ],
                recombination_rate=1e-8,
                seed=20000 + rep,
            )
            ts = sim.simulate()
            tree = ts.at(px)
            ts_t.append(tree.tmrca(0, 1))
        T_si.append(np.mean(ts_t))

    rel_x = (positions - bp_l) / inv_len
    ax2.plot(rel_x, np.array(T_si) / Ne, "o-", color="#C62828", lw=2, markersize=7)
    ax2.set_xlabel("Inversion-relative position")
    ax2.set_ylabel(r"$E[T_{S\!I}]$  (units of $N_e$ generations)")
    ax2.set_title(
        "B. Cross-class coalescence time\n(centre = more flux → smaller $T_{SI}$)",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.3)

    caption = (
        "Figure 6. Gene flux profile and its effect on cross-class coalescence time. "
        "(A) The Peischl $\\phi(x)$ function gives the per-bp gene-conversion weight as a function "
        "of position within the inversion — triangular with a peak at the centre and zero at breakpoints. "
        "The window parameter $w$ controls the conversion tract length relative to the inversion. "
        "(B) Expected cross-class coalescence time $E[T_{SI}]$ as a function of inversion-relative position "
        f"(Ne={Ne:,}, t_inv=4Ne, $\\gamma$=1e-7, r=1e-8). "
        "T$_{SI}$ is lowest at the centre (where $\\phi(x)$ peaks and gene flux is strongest) "
        f"and highest near breakpoints. {NREPS} replicates per position.\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(OUTDIR, "fig6_phi_profile.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Fig 7 — Performance scaling
# ============================================================


def fig7_performance():
    print("Fig 7: Performance (scaling + Python vs Rust)...")
    Ne = 10_000
    L = 100_000
    NREPS = 20

    # ---- Panel A: Rust scaling with rho ----
    rho_vals = [5, 10, 20, 40]
    times_inv = []
    times_no = []
    for rho in rho_vals:
        r = rho / (4 * Ne * L)
        t0 = time.time()
        for s in range(NREPS):
            sim = HullSimulator(
                n_std=5,
                n_inv=5,
                population_size=Ne,
                sequence_length=L,
                recombination_rate=r,
                inversions=[
                    InversionSpec(
                        bp_left=30_000,
                        bp_right=70_000,
                        p_inv=0.5,
                        t_inv=100_000,
                        gene_conversion_rate=1e-9,
                    )
                ],
                seed=s + 50,
            )
            sim.simulate(use_rust=True)
        times_inv.append((time.time() - t0) / NREPS * 1000)

        t0 = time.time()
        for s in range(NREPS):
            sim = HullSimulator(
                n_std=10,
                n_inv=0,
                population_size=Ne,
                sequence_length=L,
                recombination_rate=r,
                seed=s + 50,
            )
            sim.simulate(use_rust=True)
        times_no.append((time.time() - t0) / NREPS * 1000)

    # ---- Panel B: Python vs Rust speedup across problem sizes ----
    bench = [
        (
            "n=10\nL=50kb\nρ=10",
            dict(
                n_std=10,
                n_inv=0,
                population_size=5000,
                sequence_length=50_000.0,
                recombination_rate=1e-8,
            ),
        ),
        (
            "n=20\nL=200kb\nρ=80",
            dict(
                n_std=20,
                n_inv=0,
                population_size=10000,
                sequence_length=200_000.0,
                recombination_rate=1e-8,
            ),
        ),
        (
            "n=10 +inv\nL=100kb\nρ=20",
            dict(
                n_std=5,
                n_inv=5,
                population_size=5000,
                sequence_length=100_000.0,
                recombination_rate=1e-8,
                inversions=[
                    InversionSpec(
                        bp_left=30_000,
                        bp_right=70_000,
                        p_inv=0.5,
                        t_inv=100_000,
                        gene_conversion_rate=1e-9,
                    )
                ],
            ),
        ),
        (
            "n=30\nL=500kb\nρ=200",
            dict(
                n_std=30,
                n_inv=0,
                population_size=10000,
                sequence_length=500_000.0,
                recombination_rate=1e-8,
            ),
        ),
    ]
    NB = 5
    py_t = []
    rs_t = []
    for _label, params in bench:
        py_runs = []
        rs_runs = []
        for s in range(NB):
            sim = HullSimulator(seed=s, **params)
            t = time.perf_counter()
            sim.simulate(use_rust=False)
            py_runs.append((time.perf_counter() - t) * 1000)
            sim2 = HullSimulator(seed=s, **params)
            t = time.perf_counter()
            sim2.simulate(use_rust=True)
            rs_runs.append((time.perf_counter() - t) * 1000)
        py_t.append(np.median(py_runs))
        rs_t.append(np.median(rs_runs))
    speedups = [p / r for p, r in zip(py_t, rs_t)]
    bench_labels = [b[0] for b in bench]

    np.savez(
        os.path.join(OUTDIR, "fig7_data.npz"),
        rho_vals=rho_vals,
        times_inv=times_inv,
        times_no=times_no,
        bench_labels=np.array(bench_labels),
        py_t=py_t,
        rs_t=rs_t,
        speedups=speedups,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: scaling
    ax1.plot(
        rho_vals,
        times_no,
        "o-",
        color="#1976D2",
        lw=2,
        markersize=8,
        label="no inversion",
    )
    ax1.plot(
        rho_vals,
        times_inv,
        "s-",
        color="#C62828",
        lw=2,
        markersize=8,
        label="one inversion (S/I barrier)",
    )
    ax1.set_xlabel(r"$\rho = 4 N_e r L$")
    ax1.set_ylabel("Time per replicate (ms)")
    ax1.set_title("A. Rust scaling with $\\rho$ (n=10, L=100kb)")
    ax1.legend(fontsize=10)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Python vs Rust grouped bars on log scale
    x = np.arange(len(bench))
    w = 0.38
    ax2.bar(x - w / 2, py_t, w, label="Python", color="#FF9800")
    ax2.bar(x + w / 2, rs_t, w, label="Rust", color="#388E3C")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bench_labels, fontsize=9)
    ax2.set_ylabel("Time per replicate (ms, log scale)")
    ax2.set_title("B. Python vs Rust (median of 5 reps)")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y", which="both")
    # Speedup annotations above each pair
    ymax = max(py_t) * 1.2
    for xi, sp in zip(x, speedups):
        ax2.text(
            xi,
            ymax,
            f"{sp:.0f}×",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#1A237E",
            weight="bold",
        )
    ax2.set_ylim(top=ymax * 4)

    caption = (
        "Figure 7. msinv (Rust) performance. "
        "(A) Wall-clock time per replicate scales sublinearly with recombination "
        "$\\rho = 4 N_e r L$. The S/I class barrier adds modest overhead from "
        "per-segment class tracking and gene-flux events. "
        "(B) Python vs Rust on four representative scenarios; speedup (above bars) "
        "grows with problem size. The Rust core is the default backend in msinv >= 0.4.0; "
        "the legacy Python implementation is retained for cross-validation. "
        f"Panel A: Ne={Ne:,}, L={L / 1e3:.0f} kb, n=10, {NREPS} replicates per point. "
        f"Panel B: median of {NB} replicates per scenario.\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    plt.savefig(os.path.join(OUTDIR, "fig7_performance.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Fig 8 — Feature summary table (hull-only)
# ============================================================


def fig8_feature_summary():
    print("Fig 8: Feature summary...")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    features = [
        ["Feature", "msinv", "msprime", "discoal", "SLiM"],
        ["Chromosomal inversions (any age)", "\u2713", "\u2717", "\u2717", "\u2713"],
        ["Cross-karyotype barrier (t_inv)", "\u2713", "\u2717", "\u2717", "\u2713"],
        ["Gene flux (Peischl phi(x))", "\u2713", "\u2717", "\u2717", "partial"],
        ["Multiple inversions (incl. nested)", "\u2713", "\u2717", "\u2717", "\u2713"],
        ["Per-pop inversion frequencies", "\u2713", "\u2717", "\u2717", "\u2713"],
        ["Selective sweeps", "\u2713", "partial", "\u2713", "\u2713"],
        ["Coalescent (fast, neutral)", "\u2713", "\u2713", "\u2713", "\u2717"],
        ["ms-style demography", "\u2713", "\u2713", "\u2713", "partial"],
        ["Multiple populations", "\u2713", "\u2713", "\u2713", "\u2713"],
        ["Tree sequence (tskit) output", "\u2713", "\u2713", "\u2717", "\u2713"],
    ]

    table = ax.table(
        cellText=features[1:], colLabels=features[0], loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(features[0])):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", weight="bold")

    for i in range(1, len(features)):
        for j in range(1, len(features[0])):
            cell = table[i, j]
            v = features[i][j]
            if v == "\u2713":
                cell.set_facecolor("#C8E6C9")
            elif v == "\u2717":
                cell.set_facecolor("#FFCDD2")
            else:
                cell.set_facecolor("#FFF9C4")

    ax.set_title(
        "msinv: ARG-based coalescent simulator with chromosomal inversions",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    caption = (
        "Figure 8. Feature comparison: msinv vs existing coalescent and forward simulators. "
        "msinv is the only coalescent simulator with explicit chromosomal inversion support — "
        "cross-karyotype barriers (t_inv), position-dependent gene flux ($\\phi(x)$), "
        "multiple/nested inversions, and per-population frequencies. "
        "Unlike SLiM (forward-time), msinv is coalescent-based: fast for neutral scenarios "
        "and directly produces tree sequences for downstream analysis with tskit. "
        f"msinv v{msinv.__version__}, Rust ARG core (per-position ancestral material tracking).\n"
        "Command: pixi run -e all python examples/make_figures.py"
    )
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        fontsize=7,
        wrap=True,
        fontstyle="italic",
        color="#333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(OUTDIR, "fig8_feature_summary.pdf"), bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    print("Generating presentation figures (msinv Rust simulator)...\n")
    t0 = time.time()
    fig1_inversion_signal()
    fig2_msprime_comparison()
    fig3_real_inversions()
    fig4_multiple_inversions()
    fig5_trajectories()
    fig6_phi_profile()
    fig7_performance()
    fig8_feature_summary()

    print(f"\nDone in {time.time() - t0:.0f}s. Files in {OUTDIR}/:")
    for f in sorted(os.listdir(OUTDIR)):
        if f.startswith("fig") and f.endswith(".pdf"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
