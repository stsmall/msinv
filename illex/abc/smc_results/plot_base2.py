"""Figures for the SMC rerun on the refitted neutral baseline (base2) and the baseline refit.

Reads results/illex/smc_summary/{runs_base2,ppc_base2,baseline_refit} and results/illex/abc_observed_ctl2.json and writes PNGs to
results/illex/smc_summary/figs/:

  base2_model_probs.png      model probabilities over SMC generations
  base2_posteriors_B.png     B posteriors, flux 0 and flux 1, x range = prior
  base2_trajectory_B.png     2a frequency through time, posterior particles of B re-simulated on the base2 baseline
  base2_ppc.png              per-statistic predictive quantile of the observed value, B / D1 / D2, flux 0 and 1
  baseline_refit.png         baseline posteriors (regression-adjusted, 8 JSFS cells) and accepted-draw checks
  control_jsfs_check.png     control JSFS against the baseline's accepted draws, inversion JSFS against B (flux 1)
  traj_quantiles_f{0,1}.csv  pointwise quantiles of the re-simulated B trajectories (main-text Fig. 4b)

    OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python illex/abc/smc_results/plot_base2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sietch_colab/data_share/illex/popgen_data/illex_manuscript/scripts")
from figstyle import C, apply, despine, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

R = Path("results/illex/smc_summary")
OUT = R / "figs"
MODEL_COL = {"B": "#0072B2", "D1": "#E69F00", "D2": "#009E73", "N": C["muted"]}
FLUX_COL = {0: "#0072B2", 1: "#D55E00"}


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(q, c, x[o])


# ------------------------------------------------------------------------------------------- model probabilities
def fig_model_probs():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, f in zip(axs, (0, 1)):
        tr = pd.read_csv(R / "runs_base2" / f"abc_smc_trace_mc_{f}.csv")
        for m in ("N", "D1", "D2", "B"):
            ax.plot(tr["t"], tr[f"prob_{m}"], color=MODEL_COL[m], label=m)
        ax.set_title(f"flux {f}")
        ax.set_xlabel("SMC generation")
        ax.xaxis.get_major_locator().set_params(integer=True)
        despine(ax)
    axs[0].set_ylabel("model probability")
    axs[0].legend(loc="center right", title="model")
    save(fig, OUT / "base2_model_probs.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- B posteriors
def _post(run, f):
    return pd.read_csv(R / run / f"abc_smc_post_mc_B_{f}.csv")


def fig_posteriors():
    from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR
    sets = {0: _post("runs_base2", 0), 1: _post("runs_base2", 1)}
    panels = [("t_inv", "t_inv (thousand generations)", lambda x: x / 1e3, (PRIOR["t_inv"][0] / 1e3, PRIOR["t_inv"][1] / 1e3)),
              ("t_sel", "t_sel (thousand generations)", lambda x: x / 1e3, (PRIOR["t_inv"][0] / 1e3, PRIOR["t_inv"][1] / 1e3)),
              ("s", "log10 s", np.log10, tuple(np.log10(PRIOR["s"]))),
              ("p0", "p0 (soft start)", lambda x: x, PRIOR["p0"]),
              ("p_star", "p* (equilibrium 2a frequency)", lambda x: x, PRIOR["p_star"]),
              ("gamma_eff", "log10 gene-flux rate", np.log10, (np.log10(GAMMA_MIN), np.log10(GAMMA_MAX)))]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, (col, lab, tf, lim) in zip(axs.ravel(), panels):
        bins = np.linspace(lim[0], lim[1], 45)
        for name, d in sets.items():
            x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x) & (x > 0)] if col in ("s", "gamma_eff") else x[np.isfinite(x)]
            if x.size < 5:
                continue
            ax.hist(tf(x), bins=bins, density=True, histtype="step", lw=1.4, color=FLUX_COL[name],
                    label=f"flux {name}")
        ax.set_xlim(lim)
        ax.set_xlabel(lab)
        ax.set_yticks([])
        despine(ax, keep=("bottom",))
    axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Balancing model (B) posteriors, x range = prior", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / "base2_posteriors_B.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- trajectory
def fig_trajectory(n_target=400, seed=20261110):
    import illex.theory as T
    from illex.abc.trajectory import simulate_paths
    sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
    from ppc_new_baseline import apply_baseline
    p = json.load(open(R / "baseline_refit" / "fit_jsfs8.json"))["params"]
    apply_baseline(**{k: p[k]["median"] for k in ("n_anc", "n0", "t_grow", "r")})
    rng = np.random.default_rng(seed)
    grid = np.arange(0, 1.0e6 + 1, 2_000.0)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, f in zip(axs, (0, 1)):
        d = _post("runs_base2", f)
        got = []
        for _ in range(40):
            take = d.sample(n=1000, replace=True, random_state=int(rng.integers(1 << 31)))
            prm = {"model": "B"}
            for k in ("t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff"):
                prm[k] = pd.to_numeric(take[k], errors="coerce").to_numpy(float)
            prm["hard"] = take["hard"].astype(str).str.lower().eq("true").to_numpy()
            prm["t_sel"] = np.where(prm["hard"], prm["t_inv"], prm["t_sel"])
            paths, _ = simulate_paths(prm, rng, dt=10.0, rec_every=1000.0, n_e_fn=T.N_growth)
            got += paths
            if len(got) >= n_target:
                break
        got = got[:n_target]
        F = np.full((len(got), grid.size), np.nan)
        for i, q in enumerate(got):
            o = np.argsort(q.times)
            t, fr = q.times[o], q.freqs[o]
            m = grid <= t[-1]
            F[i, m] = np.interp(grid[m], t, fr)
            F[i, grid > t[-1]] = 0.0
        x = grid / 1e3
        q = np.nanquantile(F, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
        out = pd.DataFrame({"t_gen": grid, "q025": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q975": q[4]})
        for i in range(min(12, len(got))):
            out[f"path{i}"] = F[i]
        out.to_csv(OUT / f"traj_quantiles_f{f}.csv", index=False, float_format="%.5g")
        col = FLUX_COL[f]
        ax.fill_between(x, q[0], q[4], color=col, alpha=0.15, lw=0, label="95%")
        ax.fill_between(x, q[1], q[3], color=col, alpha=0.30, lw=0, label="50%")
        ax.plot(x, q[2], color=col, lw=1.6, label="median")
        for i in range(min(12, len(got))):
            ax.plot(x, F[i], color=C["ink"], lw=0.4, alpha=0.35)
        ax.axhline(0.626, color=C["muted"], lw=0.8, ls=":")
        ax.set_xlim(x[-1], 0)
        ax.set_xlabel("thousand generations ago")
        ax.set_title(f"flux {f} ({len(got)} re-simulated particles)")
        despine(ax)
        print(f"trajectory flux {f}: {len(got)} paths", flush=True)
    axs[0].set_ylabel("2a frequency")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, OUT / "base2_trajectory_B.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- PPC
STAT_ORDER = (["pi_ratio", "dxy_pi", "f1_ratio", "edge_centre", "tajd_I", "tajd_S"]
              + [f"sfs_{c}_{b}" for c in "IS" for b in ("1", "2", "3", "4", "5", "6p")]
              + [f"ld_{c}_{k}" for c in "IS" for k in range(1, 7)]
              + ["jsfs_01", "jsfs_02", "jsfs_03", "jsfs_04", "jsfs_10", "jsfs_11", "jsfs_12", "jsfs_13", "jsfs_14",
                 "jsfs_20", "jsfs_21", "jsfs_22"])


def fig_ppc():
    d = pd.read_csv(R / "ppc_base2" / "ppc_per_stat.csv")
    j = pd.read_csv(R / "ppc_base2" / "ppc_joint.csv")
    stats = [s for s in STAT_ORDER if s in set(d["stat"])]
    y = {s: i for i, s in enumerate(stats)}
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 10), sharey=True)
    off = {"B": 0.0, "D1": -0.25, "D2": 0.25}
    for ax, f in zip(axs, (0, 1)):
        ax.axvspan(0, 0.025, color=C["faint"], lw=0)
        ax.axvspan(0.975, 1, color=C["faint"], lw=0)
        for m in ("D1", "D2", "B"):
            s = d[(d.flux == f) & (d.model == m)]
            if s.empty:
                continue
            jj = j[(j.flux == f) & (j.model == m)].iloc[0]
            ax.scatter(s["obs_quantile"], [y[k] + off[m] for k in s["stat"]], s=22 if m == "B" else 12,
                       color=MODEL_COL[m], edgecolor="white", lw=0.4, zorder=3,
                       label=f"{m}: joint ppp {jj.joint_ppp:.2f}, {int(jj.n_distinct_particles)} particles")
        for i in range(len(stats)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7", lw=0, zorder=0)
        ax.axvline(0.5, color=C["muted"], lw=0.6, ls=":")
        ax.set_xlim(0, 1)
        ax.set_xlabel("quantile of observed in predictive")
        ax.set_title(f"flux {f}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=8)
        despine(ax, keep=("bottom",))
    axs[0].set_yticks(range(len(stats)))
    axs[0].set_yticklabels(stats, fontsize=8)
    axs[0].set_ylim(len(stats) - 0.5, -0.5)
    fig.suptitle("Posterior predictive checks (shaded = outside the central 95%)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / "base2_ppc.png")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- baseline refit
def _refit_adjusted(feats_jsfs):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
    import baseline_refit as B
    obs = pd.Series(B.observed("results/illex/abc_observed_ctl2.json"))
    feats = B.FEATS + feats_jsfs
    d = pd.read_csv(R / "baseline_refit" / "draws.csv").replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
    X = d[feats].to_numpy(float)
    mad = np.median(np.abs(X - np.median(X, 0)), 0)
    mad = np.where(mad > 0, mad, 1.0)
    dist = np.sqrt((((X - obs[feats].to_numpy(float)) / mad) ** 2).sum(1))
    k = min(max(int(0.01 * len(d)), 50), len(d))
    idx = np.argsort(dist)[:k]
    w = 1 - (dist[idx] / dist[idx].max()) ** 2
    theta = np.log10(d[["n_anc", "n0", "t_grow", "r"]].to_numpy(float)[idx])
    Z = np.column_stack([np.ones(k), (X[idx] - obs[feats].to_numpy(float)) / mad])
    sw = np.sqrt(w)[:, None]
    beta = np.linalg.lstsq(sw * Z, sw * theta, rcond=None)[0]
    return theta - Z[:, 1:] @ beta[1:], w, d.iloc[idx], obs, B


def fig_baseline_refit():
    jsfs8 = ["jsfs_01", "jsfs_02", "jsfs_10", "jsfs_11", "jsfs_12", "jsfs_20", "jsfs_21", "jsfs_22"]
    adj, w, acc, obs, B = _refit_adjusted(jsfs8)
    fit = json.load(open(R / "baseline_refit" / "fit_jsfs8.json"))
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.55, top=0.86, wspace=0.35)
    labs = {"n_anc": "log10 N_anc", "n0": "log10 N0", "t_grow": "log10 T_grow (generations)",
            "r": "log10 r (per bp per gen.)"}
    for j, name in enumerate(("n_anc", "n0", "t_grow", "r")):
        ax = fig.add_subplot(gs[0, j])
        ax.hist(adj[:, j], bins=30, weights=w, density=True, color="#0072B2", alpha=0.55, lw=0)
        med = np.log10(fit["params"][name]["median"])
        ax.axvline(med, color="#0072B2", lw=1.4)
        ax.set_xlabel(labs[name], fontsize=9)
        ax.set_yticks([])
        despine(ax, keep=("bottom",))
    fig.axes[0].plot([], [], color="#0072B2", label="posterior median")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=8)
    chk = fit["check_accepted"]
    groups = [("SFS bin (collinear)", [f"sfs_{b}" for b in B.BINS], [b for b in B.BINS]),
              ("LD by distance bin", [f"ld_{k}" for k in range(1, 7)], [str(k) for k in range(1, 7)]),
              ("control JSFS cell", jsfs8, [c[-2:] for c in jsfs8])]
    for g, (title, keys, ticks) in enumerate(groups):
        ax = fig.add_subplot(gs[1, g if g < 2 else slice(2, 4)])
        for i, k in enumerate(keys):
            c = chk[k]
            ax.plot([i, i], [c["q025"], c["q975"]], color="#0072B2", lw=3, alpha=0.45, solid_capstyle="butt")
            ax.plot(i, c["median"], "_", color="#0072B2", ms=10)
            if c["observed"] is not None:
                outside = not (c["q025"] <= c["observed"] <= c["q975"])
                ax.plot(i, c["observed"], "o", ms=5, color="#D55E00" if outside else C["ink"], zorder=3)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title(title, fontsize=10)
        if g == 1:
            ax.set_yscale("log")
        despine(ax)
    fig.axes[4].set_ylabel("value")
    fig.axes[-1].plot([], [], color="#0072B2", lw=3, alpha=0.45, label="accepted draws, 95%")
    fig.axes[-1].plot([], [], "o", color=C["ink"], ms=5, label="observed")
    fig.axes[-1].plot([], [], "o", color="#D55E00", ms=5, label="observed, outside")
    fig.axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Neutral baseline fitted to collinear chr2 (40-55 + 85-115 Mb)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, OUT / "baseline_refit.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ control JSFS check
def fig_control_jsfs():
    """Control JSFS against the refitted baseline's accepted draws (fit_jsfs8.json; the 8 cells it fits), and the
    inversion JSFS against the balancing model with gene flux (ppc_base2)."""
    obs = json.load(open("results/illex/abc_observed_ctl2.json"))
    ctl = obs["meta"]["jsfs_control"]
    chk = json.load(open(R / "baseline_refit" / "fit_jsfs8.json"))["check_accepted"]
    pp = pd.read_csv(R / "ppc_base2" / "ppc_per_stat.csv")
    pp = pp[(pp.flux == 1) & (pp.model == "B")].set_index("stat")
    cells = [c for c in STAT_ORDER if c.startswith("jsfs_")]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, c in enumerate(cells):
        if c in chk:
            ax.vlines(i - 0.15, chk[c]["q025"], chk[c]["q975"], color=C["muted"], lw=4, alpha=0.6)
            ax.plot(i - 0.15, chk[c]["median"], "_", color=C["muted"], ms=10)
        ax.vlines(i + 0.15, pp.loc[c, "pred_q025"], pp.loc[c, "pred_q975"], color="#0072B2", lw=4, alpha=0.35)
        ax.plot(i + 0.15, pp.loc[c, "pred_median"], "_", color="#0072B2", ms=10)
    ax.plot(x - 0.15, [ctl[c] for c in cells], "o", color=C["ink"], ms=5, zorder=3)
    ax.plot(x + 0.15, [pp.loc[c, "observed"] for c in cells], "o", color="#D55E00", ms=5, zorder=3)
    ax.plot([], [], color=C["muted"], lw=4, alpha=0.6, label="baseline, accepted draws (median, 95%)")
    ax.plot([], [], "o", color=C["ink"], ms=5, label="observed collinear flanks")
    ax.plot([], [], color="#0072B2", lw=4, alpha=0.35, label="balancing model with gene flux (median, 95%)")
    ax.plot([], [], "o", color="#D55E00", ms=5, label="observed inversion")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("jsfs_", "") for c in cells])
    ax.set_xlabel("JSFS cell (2a bin, 2+ bin; bins 0, 1-4, 5-15, 16-19, 20 of 20)")
    ax.set_ylabel("fraction of variable sites")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    despine(ax)
    save(fig, OUT / "control_jsfs_check.png")
    plt.close(fig)


def main():
    apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_model_probs()
    fig_posteriors()
    fig_ppc()
    fig_control_jsfs()
    fig_baseline_refit()
    fig_trajectory()


if __name__ == "__main__":
    main()
