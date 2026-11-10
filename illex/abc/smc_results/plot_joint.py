"""Joint posterior of the balancing model (B) from the final ABC-SMC generation (base2), with rise time.

Each of the 1,000 final-generation particles (runs_base2/abc_smc_post_mc_B_{f}.csv) is re-simulated once on the
base2 neutral baseline (fit_jsfs8.json medians), retrying the same theta until the path survives and ends within
the present-frequency tolerance (at most MAX_TRY tries). From that path:

  rise   forward generations from the onset of selection (t_sel) until 2a first reaches P_RISE
  p_sel  2a frequency at t_sel (drift between t_inv and t_sel moves it away from p0)

Writes to results/illex/smc_summary/figs/:
  joint_B_f{0,1}.csv     particle parameters + rise + p_sel
  joint_B_spearman.csv   Spearman correlations, soft starts, both flux settings
  base2_joint_B.png      corner plot, flux 1, soft starts (98% of the mass), Spearman rho in each panel

    OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python illex/abc/smc_results/plot_joint.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/sietch_colab/data_share/illex/popgen_data/illex_manuscript/scripts")
from figstyle import C, apply, despine, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

R = Path("results/illex/smc_summary")
OUT = R / "figs"
P_RISE = 0.55
MAX_TRY = 20
KEYS = ("t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff")


def resimulate(d, rng):
    import illex.theory as T
    from illex.abc.trajectory import simulate_paths
    n = len(d)
    rise = np.full(n, np.nan)
    p_sel = np.full(n, np.nan)
    tries = np.zeros(n, int)
    todo = np.arange(n)
    hard_all = d["hard"].astype(str).str.lower().eq("true").to_numpy()
    for _ in range(MAX_TRY):
        if todo.size == 0:
            break
        prm = {"model": "B"}
        for k in KEYS:
            prm[k] = pd.to_numeric(d[k], errors="coerce").to_numpy(float)[todo]
        prm["hard"] = hard_all[todo]
        prm["t_sel"] = np.where(prm["hard"], prm["t_inv"], prm["t_sel"])
        tries[todo] += 1
        paths, _ = simulate_paths(prm, rng, dt=10.0, rec_every=100.0, n_e_fn=T.N_growth)
        lookup = {}                             # resampled particles can repeat, so map each t_inv to a queue
        for t, i in zip(prm["t_inv"], todo):
            lookup.setdefault(float(t), []).append(int(i))
        done = []
        for pth in paths:
            i = lookup[float(pth.params["t_inv"])].pop(0)
            o = np.argsort(pth.times)
            t, f = pth.times[o], pth.freqs[o]
            ts = pth.params["t_sel"]
            p_sel[i] = np.interp(ts, t, f)
            m = (t <= ts) & (f >= P_RISE)
            rise[i] = ts - t[m].max() if m.any() else np.nan
            done.append(i)
        todo = np.setdiff1d(todo, done)
    out = d.copy()
    out["hard"] = hard_all
    out["rise"] = rise
    out["p_sel"] = p_sel
    out["tries"] = tries
    out["resim_ok"] = ~np.isin(np.arange(n), todo)
    return out


VARS = [("t_inv", "$t_\\mathrm{inv}$ (kgen)", lambda x: x / 1e3, False),
        ("t_sel", "$t_\\mathrm{sel}$ (kgen)", lambda x: x / 1e3, False),
        ("p0", "$p_0$", lambda x: x, False),
        ("s", "$\\log_{10} s$", np.log10, True),
        ("rise", "$\\log_{10}$ rise to 0.55 (gen)", np.log10, True),
        ("p_star", "$p^*$", lambda x: x, False),
        ("gamma_eff", "$\\log_{10}$ gene flux", np.log10, True)]


def corner(d, path, title):
    apply()
    d = d[~d["hard"] & d["resim_ok"]]
    X = {}
    for k, _, tf, _ in VARS:
        X[k] = tf(pd.to_numeric(d[k], errors="coerce").to_numpy(float))
    k_ = len(VARS)
    fig, axs = plt.subplots(k_, k_, figsize=(10, 10))
    for i, (ki, li, _, _) in enumerate(VARS):
        for j, (kj, lj, _, _) in enumerate(VARS):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            xi, xj = X[ki], X[kj]
            if i == j:
                v = xi[np.isfinite(xi)]
                ax.hist(v, bins=35, color=C["muted"], alpha=0.7)
                ax.set_yticks([])
                despine(ax, keep=("bottom",))
            else:
                m = np.isfinite(xi) & np.isfinite(xj)
                ax.scatter(xj[m], xi[m], s=3, lw=0, color="#CC79A7", alpha=0.45, rasterized=True)
                rho = spearmanr(xj[m], xi[m]).statistic
                ax.text(0.04, 0.96, f"{rho:+.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        fontweight="bold" if abs(rho) >= 0.3 else "normal", color=C["ink"])
                despine(ax)
            if i == k_ - 1:
                ax.set_xlabel(lj, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(li, fontsize=9)
            elif j != i:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle(title, x=0.5, y=0.93, fontsize=11)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    save(fig, path)
    plt.close(fig)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
    from ppc_new_baseline import apply_baseline
    p = json.load(open(R / "baseline_refit" / "fit_jsfs8.json"))["params"]
    apply_baseline(**{k: p[k]["median"] for k in ("n_anc", "n0", "t_grow", "r")})
    rng = np.random.default_rng(20261110)
    rows = []
    for f in (0, 1):
        d = pd.read_csv(R / "runs_base2" / f"abc_smc_post_mc_B_{f}.csv")
        j = resimulate(d, rng)
        j.to_csv(OUT / f"joint_B_f{f}.csv", index=False, float_format="%.6g")
        ok = j["resim_ok"]
        print(f"flux {f}: {ok.sum()}/{len(j)} re-simulated, median tries {j.loc[ok, 'tries'].median():.0f}, "
              f"rise median {np.nanmedian(j['rise']):.0f} gen "
              f"[{np.nanquantile(j['rise'], 0.025):.0f}, {np.nanquantile(j['rise'], 0.975):.0f}]", flush=True)
        s = j[~j["hard"] & ok]
        cols = [k for k, *_ in VARS] + ["p_sel"]
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                x, y = s[cols[a]].astype(float), s[cols[b]].astype(float)
                m = np.isfinite(x) & np.isfinite(y)
                rows.append({"flux": f, "a": cols[a], "b": cols[b], "n": int(m.sum()),
                             "spearman": spearmanr(x[m], y[m]).statistic})
        if f == 1:
            corner(j, OUT / "base2_joint_B.png",
                   "Balancing model, gene flux: joint posterior (final ABC-SMC particles, soft starts)")
    pd.DataFrame(rows).to_csv(OUT / "joint_B_spearman.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
