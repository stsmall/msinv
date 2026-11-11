"""Manuscript figures and exports from the NPE posterior of the balancing model (run on the box, msinv venv).

    OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python illex/abc/npe/plot_npe.py

Reads results/illex/npe/f{0,1}/posterior_samples.csv (train_npe.py) and writes to results/illex/npe/figs/:

  npe_corner_f1.png           joint NPE posterior with gene flux (flux 1): marginals over the prior range on the
                              diagonal, 80/95/99% density contours below it, Spearman rho in each panel
  npe_quantiles.csv           2.5/5/50/95/97.5% quantiles of every parameter, flux 0 and 1, plus the rise time
  traj_quantiles_npe_f1.csv   pointwise quantiles and 200 example paths of 400 trajectories re-simulated from NPE
                              draws on the base2 baseline (Fig. 4b)
  traj_aligned_npe_f1.csv     the same trajectories aligned on the onset of selection (forward generations)
  traj_params_npe_f1.csv      t_inv, t_sel, p0, s and the rise time (generations from t_sel to 0.55) of each path

All NPE draws are soft starts (founding frequency p0), as in the training table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sietch_colab/data_share/illex/popgen_data/illex_manuscript/scripts")
from figstyle import C, OKABE, apply, despine, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

R = Path("results/illex/npe")
OUT = R / "figs"
SMC = Path("results/illex/smc_summary")
RISE_TO = 0.55
MAX_TRY = 50
COL = OKABE[7]
# interior ticks only, so labels of neighbouring panels do not collide
TICKS = {"t_inv": [500, 1000], "t_sel": [500, 1000], "p0": [-2.5, -2, -1.5], "s": [-5, -4, -3, -2],
         "p_star": [0.4, 0.6, 0.8], "gamma": [-8, -7, -6]}


def draws(f):
    d = pd.read_csv(R / f"f{f}" / "posterior_samples.csv")
    return d


def coords(d):
    from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR
    t = (PRIOR["t_inv"][0] / 1e3, PRIOR["t_inv"][1] / 1e3)
    c = [("t_inv", "$t_\\mathrm{inv}$ (kgen)", d["t_inv"] / 1e3, t),
         ("t_sel", "$t_\\mathrm{sel}$ (kgen)", d["t_sel"] / 1e3, (0, t[1])),
         ("p0", "$\\log_{10} p_0$", np.log10(d["p0"]), tuple(np.log10(PRIOR["p0"]))),
         ("s", "$\\log_{10} s$", np.log10(d["s"]), tuple(np.log10(PRIOR["s"]))),
         ("p_star", "$p^{*}$", d["p_star"], PRIOR["p_star"])]
    if "gamma" in d:
        c.append(("gamma", "$\\log_{10}$ gene flux", np.log10(d["gamma"]), (np.log10(GAMMA_MIN), np.log10(GAMMA_MAX))))
    return c


def fig_corner(f=1):
    from scipy.stats import gaussian_kde, spearmanr
    c = coords(draws(f))
    k = len(c)
    fig, axs = plt.subplots(k, k, figsize=(1.45 * k, 1.45 * k))
    sub = np.random.default_rng(0).choice(len(c[0][2]), 6000, replace=False)
    for i in range(k):
        for j in range(k):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            xi, xj = c[i][2].to_numpy(), c[j][2].to_numpy()
            if i == j:
                ax.hist(xi, bins=np.linspace(*c[i][3], 40), density=True, color=COL, alpha=0.55, lw=0)
                ax.axvline(np.median(xi), color=C["ink"], lw=0.8, ls="--")
                ax.set_yticks([])
                despine(ax, keep=("bottom",))
            else:
                kde = gaussian_kde(np.vstack([xj[sub], xi[sub]]))
                gx, gy = np.meshgrid(np.linspace(*c[j][3], 70), np.linspace(*c[i][3], 70))
                zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                # density levels enclosing 99, 95 and 80% of the draws
                dz = np.sort(kde(np.vstack([xj[sub], xi[sub]])))
                lv = np.unique(np.quantile(dz, [0.01, 0.05, 0.20]))
                ax.contourf(gx, gy, zz, levels=np.append(lv, zz.max()), colors=[COL], alpha=0.18)
                ax.contour(gx, gy, zz, levels=lv, colors=[COL], linewidths=0.7)
                rho = spearmanr(xi, xj).statistic
                ax.text(0.04, 0.95, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.5, color=C["ink"], fontweight="bold" if abs(rho) >= 0.3 else "normal",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
                ax.set_ylim(c[i][3])
                ax.set_yticks(TICKS[c[i][0]])
                despine(ax)
            ax.set_xlim(c[j][3])
            ax.set_xticks(TICKS[c[j][0]])
            ax.tick_params(labelsize=6)
            if i == k - 1:
                ax.set_xlabel(c[j][1], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(c[i][1], fontsize=8)
            elif j != i:
                ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.09, right=0.99, bottom=0.08, top=0.99)
    save(fig, OUT / f"npe_corner_f{f}.png")
    plt.close(fig)


def rise_time(pth):
    """Forward generations from the onset of selection until 2a first reaches RISE_TO (nan if never)."""
    o = np.argsort(pth.times)[::-1]                 # oldest first
    t, fr = pth.times[o], pth.freqs[o]
    m = (t <= pth.params["t_sel"]) & (fr >= RISE_TO)
    return float(pth.params["t_sel"] - t[m][0]) if m.any() else np.nan


def trajectories(f=1, n_target=400, n_show=200, seed=20261110):
    import illex.theory as T
    from illex.abc.trajectory import simulate_paths
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "smc_results" / "baseline"))
    from ppc_new_baseline import apply_baseline
    p = json.load(open(SMC / "baseline_refit" / "fit_jsfs8.json"))["params"]
    apply_baseline(**{k: p[k]["median"] for k in ("n_anc", "n0", "t_grow", "r")})
    rng = np.random.default_rng(seed)
    # theta ~ NPE posterior, then a path given theta that passes the present-frequency filter (the conditioning the
    # posterior already carries): failed draws are re-simulated with the same theta, up to MAX_TRY times
    take = draws(f).sample(n=n_target, replace=False, random_state=seed).reset_index(drop=True)
    pending = np.arange(len(take))
    found = {}
    for _ in range(MAX_TRY):
        t = take.iloc[pending]
        n = len(t)
        prm = {"model": "B", "t_inv": t["t_inv"].to_numpy(float), "t_sel": t["t_sel"].to_numpy(float),
               "p0": t["p0"].to_numpy(float), "s": t["s"].to_numpy(float),
               "p_star": t["p_star"].to_numpy(float), "t_stop": np.full(n, np.nan),
               "gamma_eff": t["gamma"].to_numpy(float) if "gamma" in t else np.zeros(n),
               "hard": np.zeros(n, bool)}
        paths, _ = simulate_paths(prm, rng, dt=10.0, rec_every=100.0, n_e_fn=T.N_growth)
        key = {float(v): i for i, v in zip(pending, t["t_inv"].to_numpy(float))}   # continuous draws: unique
        for pth in paths:
            found[key[float(pth.params["t_inv"])]] = pth
        pending = np.array([i for i in pending if i not in found])
        if pending.size == 0:
            break
    print(f"trajectories flux {f}: {len(found)} of {len(take)} draws gave a path within {MAX_TRY} tries", flush=True)
    got = [found[i] for i in sorted(found)]
    grid = np.arange(0, 1.0e6 + 1, 2_000.0)
    F = np.full((len(got), grid.size), np.nan)
    for i, q in enumerate(got):
        o = np.argsort(q.times)
        t, fr = q.times[o], q.freqs[o]
        m = grid <= t[-1]
        F[i, m] = np.interp(grid[m], t, fr)
        F[i, grid > t[-1]] = 0.0
    q = np.nanquantile(F, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    out = pd.DataFrame({"t_gen": grid, "q025": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q975": q[4]})
    out = pd.concat([out, pd.DataFrame({f"path{i}": F[i] for i in range(min(n_show, len(got)))})], axis=1)
    out.to_csv(OUT / f"traj_quantiles_npe_f{f}.csv", index=False, float_format="%.4g")
    onset = np.array([pth.params["t_sel"] for pth in got], float)
    rise = np.array([rise_time(pth) for pth in got])
    pd.DataFrame({"t_inv": [pth.params["t_inv"] for pth in got], "t_sel": onset,
                  "p0": [pth.params["p0"] for pth in got], "s": [pth.params["s"] for pth in got], "rise": rise}
                 ).to_csv(OUT / f"traj_params_npe_f{f}.csv", index=False, float_format="%.6g")
    u = np.arange(-20_000.0, 60_000.0 + 1, 100.0)
    A = np.full((len(got), u.size), np.nan)
    for i, pth in enumerate(got):
        o = np.argsort(pth.times)
        t, fr = pth.times[o], pth.freqs[o]
        tb = onset[i] - u
        m = (tb >= 0) & (tb <= t[-1])
        A[i, m] = np.interp(tb[m], t, fr)
        A[i, tb > t[-1]] = 0.0
    qa = np.nanquantile(A, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    al = pd.DataFrame({"u_gen": u, "q025": qa[0], "q25": qa[1], "q50": qa[2], "q75": qa[3], "q975": qa[4]})
    al = pd.concat([al, pd.DataFrame({f"path{i}": A[i] for i in range(min(n_show, len(got)))})], axis=1)
    al.to_csv(OUT / f"traj_aligned_npe_f{f}.csv", index=False, float_format="%.4g")
    r = rise[np.isfinite(rise)]
    print(f"trajectories flux {f}: {len(got)} paths; rise to {RISE_TO}: {np.isnan(rise).sum()} never, "
          f"2.5/50/97.5% = {np.percentile(r, [2.5, 50, 97.5]).round(0)}; median aligned path reaches {RISE_TO} at "
          f"{u[np.argmax(qa[2] >= RISE_TO)]:.0f} gen", flush=True)
    return rise


def quantiles(rise):
    rows = []
    for f in (0, 1):
        d = draws(f)
        for col in d.columns:
            v = d[col].to_numpy(float)
            rows.append({"flux": f, "parameter": col, **dict(zip(("q025", "q05", "q50", "q95", "q975"),
                                                                 np.quantile(v, [0.025, 0.05, 0.5, 0.95, 0.975])))})
        if f == 1:
            r = rise[np.isfinite(rise)]
            rows.append({"flux": 1, "parameter": f"rise_to_{RISE_TO}",
                         **dict(zip(("q025", "q05", "q50", "q95", "q975"), np.quantile(r, [0.025, 0.05, 0.5, 0.95, 0.975])))})
        print(f"flux {f}: P(s > 1e-4) = {(d['s'] > 1e-4).mean():.3f}", flush=True)
    q = pd.DataFrame(rows)
    q.to_csv(OUT / "npe_quantiles.csv", index=False, float_format="%.4g")
    print(q.to_string(index=False), flush=True)


def main():
    apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_corner(1)
    if "--corner-only" in sys.argv:
        return
    rise = trajectories(1)
    quantiles(rise)


if __name__ == "__main__":
    main()
