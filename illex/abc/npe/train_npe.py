"""Train a neural posterior estimator (NPE, sbi) for the balancing model on the collected table; check it; apply it.

Run on the box with an env that has sbi + torch (balanus-npe has sbi 0.27, torch 2.14; it needs its own libstdc++):

    E=/home/ssmall/miniforge3/envs/balanus-npe
    LD_LIBRARY_PATH=$E/lib CUDA_VISIBLE_DEVICES=2 $E/bin/python illex/abc/npe/train_npe.py \
        --npz results/illex/npe/npe_B_f1.npz --smc results/illex/smc_summary/runs_base2/abc_smc_post_mc_B_1.csv \
        --out results/illex/npe/f1 [--device cuda] [--n-ens 5]

Design (Min et al. 2026, Genetics 233:iyag107; summary-statistic NPE): theta in the coordinates where the prior is a
box (collect_npe.py), x = the 39 statistics the ABC-SMC fitted, z-scored by sbi. Neural spline flow, 5 transforms,
64 hidden units; an ensemble of --n-ens flows trained on the same rows with different seeds (mixture posterior).
Rows are prior draws that passed the present-frequency filter, so the learned posterior is conditional on that
filter, as in the SMC.

Outputs in --out:
  posterior_samples.csv   20,000 draws at the observed x, natural units (t_inv, t_sel, p0, s, p_star, gamma)
  summary.csv             median and 95% interval per parameter, NPE and SMC
  spearman.csv            Spearman correlations of the NPE draws (box coordinates)
  sbc_ranks.png, tarp.png calibration on --n-test held-out rows (rank histograms; expected coverage)
  calibration.csv         per-parameter SBC KS p-value and the coverage at 50/90/95%
  misfit.csv              where x_obs falls among the training x: per-statistic quantile, and the robust
                          Mahalanobis distance of x_obs against the training distribution (its quantile)
  corner_npe_vs_smc.png   NPE draws (contours) over the final SMC particles (points)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABEL = {"t_inv": "$t_\\mathrm{inv}$ (kgen)", "u_sel": "$t_\\mathrm{sel}/t_\\mathrm{inv}$", "log10_p0": "$\\log_{10} p_0$",
         "log10_s": "$\\log_{10} s$", "p_star": "$p^*$", "log10_gamma": "$\\log_{10}$ gene flux"}


def natural(theta, names):
    d = dict(zip(names, theta.T))
    out = {"t_inv": d["t_inv"], "t_sel": d["u_sel"] * d["t_inv"], "p0": 10 ** d["log10_p0"],
           "s": 10 ** d["log10_s"], "p_star": d["p_star"]}
    if "log10_gamma" in d:
        out["gamma"] = 10 ** d["log10_gamma"]
    return pd.DataFrame(out)


def smc_box(path, names):
    s = pd.read_csv(path)
    s = s[~s["hard"].astype(str).str.lower().eq("true")]
    f = lambda k: pd.to_numeric(s[k], errors="coerce").to_numpy(float)  # noqa: E731
    cols = {"t_inv": f("t_inv"), "u_sel": f("t_sel") / f("t_inv"), "log10_p0": np.log10(f("p0")),
            "log10_s": np.log10(f("s")), "p_star": f("p_star")}
    if "log10_gamma" in names:
        cols["log10_gamma"] = np.log10(f("gamma_eff"))
    return np.column_stack([cols[n] for n in names])


def train(theta, x, prior, n_ens, device, seed):
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    posts = []
    for e in range(n_ens):
        torch.manual_seed(seed + e)
        net = posterior_nn(model="nsf", hidden_features=64, num_transforms=5)
        inf = NPE(prior=prior, density_estimator=net, device=device)
        est = inf.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                     torch.as_tensor(x, dtype=torch.float32)).train(
            training_batch_size=256, validation_fraction=0.1, stop_after_epochs=30, show_train_summary=False)
        print(f"  flow {e}: best validation loss {inf._summary['best_validation_loss'][-1]:.4f}, "
              f"{inf._summary['epochs_trained'][-1]} epochs", flush=True)
        posts.append(inf.build_posterior(est))
    if n_ens == 1:
        return posts[0]
    from sbi.inference.posteriors import EnsemblePosterior
    return EnsemblePosterior(posts)


def calibrate(post, prior, theta_t, x_t, names, out, n_post):
    from sbi.diagnostics import check_sbc, run_sbc, run_tarp
    import matplotlib.pyplot as plt
    th = torch.as_tensor(theta_t, dtype=torch.float32)
    xx = torch.as_tensor(x_t, dtype=torch.float32)
    ranks, dap = run_sbc(th, xx, post, num_posterior_samples=n_post, use_batched_sampling=True)
    chk = check_sbc(ranks, th, dap, num_posterior_samples=n_post)
    ks = chk["ks_pvals"].numpy()
    ecp, alpha = run_tarp(th, xx, post, num_posterior_samples=n_post, use_batched_sampling=True)
    ecp, alpha = ecp.numpy(), alpha.numpy()
    r = ranks.numpy()
    # per-parameter central-interval coverage from the ranks: the true value is inside the central c interval when
    # its normalised rank lies in [(1-c)/2, (1+c)/2]
    u = r / n_post
    cov = {c: ((u >= (1 - c) / 2) & (u <= (1 + c) / 2)).mean(axis=0) for c in (0.5, 0.9, 0.95)}
    pd.DataFrame({"parameter": names, "sbc_ks_p": ks, "cov50": cov[0.5], "cov90": cov[0.9], "cov95": cov[0.95]}
                 ).to_csv(out / "calibration.csv", index=False, float_format="%.3f")
    fig, axs = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.2))
    for j, ax in enumerate(axs):
        ax.hist(r[:, j], bins=20, color="#999999")
        ax.axhline(len(r) / 20, color="k", lw=0.8, ls="--")
        ax.set_title(LABEL[names[j]], fontsize=9)
        ax.set_yticks([])
    fig.suptitle(f"SBC ranks, {len(r)} held-out rows (flat = calibrated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sbc_ranks.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(alpha, ecp, color="#CC79A7", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("credibility level")
    ax.set_ylabel("expected coverage")
    ax.set_title("TARP (joint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "tarp.png", dpi=200)
    plt.close(fig)
    print("calibration: SBC KS p", dict(zip(names, np.round(ks, 3))), "| 95% coverage",
          dict(zip(names, np.round(cov[0.95], 3))), flush=True)


def misfit(x, x_obs, names, out):
    q = (x < x_obs).mean(axis=0)
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    z = (x - med) / mad
    zo = (x_obs - med) / mad
    cov = np.cov(z, rowvar=False)
    ic = np.linalg.pinv(cov)
    d = np.einsum("ij,jk,ik->i", z, ic, z)
    do = float(zo @ ic @ zo)
    df = pd.DataFrame({"statistic": names, "quantile_in_training": q})
    df.loc[len(df)] = ["MAHALANOBIS(all)", float((d < do).mean())]
    df.to_csv(out / "misfit.csv", index=False, float_format="%.4f")
    ext = df[(df.quantile_in_training < 0.01) | (df.quantile_in_training > 0.99)]
    print(f"misfit: Mahalanobis quantile of x_obs {float((d < do).mean()):.3f}; statistics outside 1-99%:",
          ext.statistic.tolist(), flush=True)


def corner(npe, smc, names, lo, hi, out):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    k = len(names)
    tf = [(lambda v: v / 1e3) if n == "t_inv" else (lambda v: v) for n in names]
    fig, axs = plt.subplots(k, k, figsize=(2.0 * k, 2.0 * k))
    for i in range(k):
        for j in range(k):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            xi, xj = tf[i](npe[:, i]), tf[j](npe[:, j])
            si, sj = tf[i](smc[:, i]), tf[j](smc[:, j])
            if i == j:
                b = np.linspace(tf[i](lo[i]), tf[i](hi[i]), 40)
                ax.hist(xi, bins=b, density=True, histtype="step", color="#CC79A7", lw=1.5, label="NPE")
                ax.hist(si[np.isfinite(si)], bins=b, density=True, histtype="step", color="#0072B2", lw=1.2,
                        label="ABC-SMC")
                ax.set_yticks([])
            else:
                m = np.isfinite(si) & np.isfinite(sj)
                ax.scatter(sj[m], si[m], s=2, lw=0, color="#0072B2", alpha=0.35, rasterized=True)
                sub = np.random.default_rng(0).choice(len(xi), min(5000, len(xi)), replace=False)
                kde = gaussian_kde(np.vstack([xj[sub], xi[sub]]))
                gx, gy = np.meshgrid(np.linspace(tf[j](lo[j]), tf[j](hi[j]), 80),
                                     np.linspace(tf[i](lo[i]), tf[i](hi[i]), 80))
                zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                lv = np.quantile(zz, [0.80, 0.95, 0.99])
                ax.contour(gx, gy, zz, levels=np.unique(lv), colors="#CC79A7", linewidths=1.0)
            ax.set_xlim(tf[j](lo[j]), tf[j](hi[j]))
            if i != j:
                ax.set_ylim(tf[i](lo[i]), tf[i](hi[i]))
            ax.tick_params(labelsize=7)
            if i == k - 1:
                ax.set_xlabel(LABEL[names[j]], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABEL[names[i]], fontsize=9)
            elif j != i:
                ax.set_yticklabels([])
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Balancing model: NPE posterior (contours) and final ABC-SMC particles (points)", fontsize=11)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.savefig(out / "corner_npe_vs_smc.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--smc", required=True, help="final SMC particles of B for the same flux (comparison only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ens", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=2000, help="held-out rows for SBC / TARP")
    ap.add_argument("--n-post", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from sbi.utils import BoxUniform

    z = np.load(a.npz, allow_pickle=False)
    theta, x, x_obs = z["theta"], z["x"], z["x_obs"]
    names, xnames, lo, hi = list(z["theta_names"]), list(z["x_names"]), z["lo"], z["hi"]
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(theta))
    te, tr = idx[:a.n_test], idx[a.n_test:]
    print(f"{a.npz}: {len(theta)} rows ({len(tr)} train, {len(te)} held out), {len(names)} parameters, "
          f"{x.shape[1]} statistics", flush=True)
    prior = BoxUniform(low=torch.as_tensor(lo, dtype=torch.float32), high=torch.as_tensor(hi, dtype=torch.float32),
                       device=a.device)
    misfit(x[tr], x_obs, xnames, out)
    post = train(theta[tr], x[tr], prior, a.n_ens, a.device, a.seed)
    calibrate(post, prior, theta[te], x[te], names, out, n_post=1000)

    xo = torch.as_tensor(x_obs, dtype=torch.float32).unsqueeze(0)
    draws = post.sample((a.n_post,), x=xo).cpu().numpy()
    nat = natural(draws, names)
    nat.to_csv(out / "posterior_samples.csv", index=False, float_format="%.6g")
    from scipy.stats import spearmanr
    rho = spearmanr(draws).statistic
    pd.DataFrame(rho, index=names, columns=names).to_csv(out / "spearman.csv", float_format="%.3f")
    smc = smc_box(a.smc, names)
    smc_nat = natural(smc[np.all(np.isfinite(smc), axis=1)], names)
    rows = []
    for c in nat.columns:
        for lab, v in (("NPE", nat[c].to_numpy()), ("SMC", smc_nat[c].to_numpy())):
            q = np.quantile(v, [0.025, 0.5, 0.975])
            rows.append({"parameter": c, "method": lab, "q025": q[0], "median": q[1], "q975": q[2]})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    print(summ.to_string(index=False), flush=True)
    corner(draws, smc, names, lo, hi, out)
    json.dump({"npz": a.npz, "n_train": int(len(tr)), "n_test": int(len(te)), "n_ens": a.n_ens, "seed": a.seed,
               "flow": "nsf, 64 hidden, 5 transforms", "names": names}, open(out / "run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
