#!/usr/bin/env python3
"""Plot SLIM vs msinv comparison with theory lines.

Reads slim_validation/output/scenario{N}_results.npz and writes
figures/slim_validation_scenario{N}.pdf.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "figures"


def smooth(y, k=3):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


SCENARIO_INVS = {
    1: [(30_000, 70_000)],
    2: [(15_000, 45_000), (55_000, 85_000)],
    3: [(30_000, 70_000)],
}
SCENARIO_XSEL = {3: 50_000}


def plot(scenario, results_path):
    d = np.load(results_path)
    Ne = int(d["Ne"])
    L = int(d["L"])
    mu = float(d["mu_rate"])
    t_inv = int(d["t_inv_factor"]) * Ne
    mid = d["mid"]
    invs = SCENARIO_INVS[scenario]

    # Theory:
    #   pi within class ~ 4*Ne*mu
    #   dxy between karyotypes:
    #       outside inv:  2*mu*(E[T_outside]) = 2*mu*2*Ne = 4*Ne*mu
    #       inside inv:   2*mu*(t_inv + 2*Ne) = 2*mu*(t_inv+2Ne)
    pi_theory = 4 * Ne * mu
    dxy_outside = 4 * Ne * mu
    dxy_inside = 2 * mu * (t_inv + 2 * Ne)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax_pi, ax_dxy, ax_fst = axes

    def shade(ax):
        for lo, hi in invs:
            ax.axvspan(lo, hi, alpha=0.08, color="#90A4AE", zorder=0)
            ax.axvline(lo, color="#78909C", ls="--", alpha=0.4, lw=0.7)
            ax.axvline(hi, color="#78909C", ls="--", alpha=0.4, lw=0.7)
        if scenario in SCENARIO_XSEL:
            ax.axvline(
                SCENARIO_XSEL[scenario],
                color="#D84315",
                ls=":",
                alpha=0.7,
                lw=1.2,
                label="_x_sel_",
            )

    c_slim_s = "#1565C0"
    c_slim_i = "#6A1B9A"
    c_slim_dxy = "#AD1457"
    c_msinv_s = "#2E7D32"
    c_msinv_i = "#00838F"
    c_msinv_dxy = "#EF6C00"
    c_theory = "#212121"

    # pi panel
    ax_pi.plot(
        mid, smooth(d["slim_pi_s"]), "-", color=c_slim_s, lw=1.8, label=r"SLIM $\pi_S$"
    )
    ax_pi.plot(
        mid, smooth(d["slim_pi_i"]), "-", color=c_slim_i, lw=1.8, label=r"SLIM $\pi_I$"
    )
    ax_pi.plot(
        mid,
        smooth(d["msinv_pi_s"]),
        "--",
        color=c_msinv_s,
        lw=1.8,
        label=r"msinv $\pi_S$",
    )
    ax_pi.plot(
        mid,
        smooth(d["msinv_pi_i"]),
        "--",
        color=c_msinv_i,
        lw=1.8,
        label=r"msinv $\pi_I$",
    )
    ax_pi.axhline(
        pi_theory,
        color=c_theory,
        ls=":",
        lw=1.2,
        label=rf"theory $4N_e\mu$={pi_theory:.1e}",
    )
    shade(ax_pi)
    ax_pi.set_ylabel(r"$\pi$ (per bp)")
    ax_pi.set_title(
        f"A. Within-karyotype diversity "
        f"(scenario {scenario}, N_reps={int(d['n_reps'])})",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    ax_pi.legend(fontsize=8, loc="upper right", ncol=2)

    # dxy panel
    ax_dxy.plot(
        mid,
        smooth(d["slim_dxy"]),
        "-",
        color=c_slim_dxy,
        lw=2.0,
        label="SLIM d_XY(S,I)",
    )
    ax_dxy.plot(
        mid,
        smooth(d["msinv_dxy"]),
        "--",
        color=c_msinv_dxy,
        lw=2.0,
        label="msinv d_XY(S,I)",
    )
    ax_dxy.axhline(
        dxy_outside,
        color=c_theory,
        ls=":",
        lw=1.2,
        label=rf"theory outside $4N_e\mu$={dxy_outside:.1e}",
    )
    ax_dxy.axhline(
        dxy_inside,
        color=c_theory,
        ls="--",
        lw=1.2,
        label=rf"theory inside $2\mu(t_{{inv}}+2N_e)$="
        f"{dxy_inside:.1e}",
    )
    shade(ax_dxy)
    ax_dxy.set_ylabel(r"$d_{XY}$ (per bp)")
    ax_dxy.set_title(
        "B. Between-karyotype divergence", loc="left", fontweight="bold", fontsize=10
    )
    ax_dxy.legend(fontsize=8, loc="upper right")

    # Fst panel
    ax_fst.plot(
        mid, smooth(d["slim_fst"]), "-", color=c_slim_dxy, lw=2.0, label="SLIM F_ST"
    )
    ax_fst.plot(
        mid, smooth(d["msinv_fst"]), "--", color=c_msinv_dxy, lw=2.0, label="msinv F_ST"
    )
    fst_inside = 1 - pi_theory / dxy_inside
    ax_fst.axhline(
        fst_inside,
        color=c_theory,
        ls="--",
        lw=1.2,
        label=rf"theory inside={fst_inside:.2f}",
    )
    ax_fst.axhline(0, color="#555", ls=":", lw=0.8)
    shade(ax_fst)
    ax_fst.set_ylabel(r"Hudson $F_{ST}$")
    ax_fst.set_xlabel("Position (bp)")
    ax_fst.set_title(
        "C. Karyotype differentiation", loc="left", fontweight="bold", fontsize=10
    )
    ax_fst.legend(fontsize=8, loc="upper right")

    slim_t = np.array(d["slim_times"])
    msinv_t = np.array(d["msinv_times"])
    caption = (
        f"SLIM vs msinv validation, scenario {scenario}. "
        f"Ne={Ne}, L={L / 1000:.0f} kb, r={float(d['r_rate']):.0e}, "
        f"gamma={float(d['gc_rate']):.0e}, mu={mu:.0e}, "
        f"t_inv={t_inv} gen, burnin={int(d['burnin_factor'])}Ne. "
        f"Runtime per rep: SLIM {slim_t.mean():.1f}s, "
        f"msinv {msinv_t.mean():.3f}s "
        f"(speedup {slim_t.mean() / max(msinv_t.mean(), 1e-9):.0f}x). "
        f"10 S + 10 I haploid samples classified by inv 0 marker. "
        f"Theory lines assume neutral coalescent with instant karyotype "
        f"barrier at t_inv and no flux (upper bound for dxy inside)."
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
    fig.suptitle(
        f"Scenario {scenario}: SLIM vs msinv (hull) "
        f"{'+ sweep on S' if scenario == 3 else ''}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    OUT_FIG.mkdir(exist_ok=True)
    out = OUT_FIG / f"slim_validation_scenario{scenario}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    outdir = HERE / "output"
    if args.all:
        scenarios = [1, 2, 3]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.error("specify --scenario or --all")

    for s in scenarios:
        p = outdir / f"scenario{s}_results.npz"
        if not p.exists():
            print(f"skip scenario {s}: {p} not found")
            continue
        plot(s, p)


if __name__ == "__main__":
    main()
