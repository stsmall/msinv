#!/usr/bin/env python
"""Is the Illex genome-wide folded SFS better explained by a multiple-merger
(Beta) coalescent than by Kingman + the moments exponential-growth history?

Motivation: Illex illecebrosus is an annual, semelparous broadcast spawner with
~1e5 fecundity and boom-bust recruitment -- the profile for sweepstakes
reproductive success, which is classically modelled by a Beta(2-a,a)-coalescent
rather than Kingman. Multiple mergers and population growth both inflate rare
variants, so they are confounded (Freund et al.; and the P. falciparum study,
PMC12871270, reports alpha~1.8 as indistinguishable from moderate growth).

Method: compare NORMALIZED folded SFS shapes, which are scale-free, so no
theta matching is needed. Observed SFS is hypergeometrically projected down to
a tractable sample size. Models are simulated with msprime (Kingman constant,
Kingman + moments growth, and Beta(a) with no growth over an alpha grid), and
each expected shape is scored against the observed counts by multinomial
log-likelihood and by L1 deviation of the normalized shape.

RESULT (2026-08-04): Kingman + moments growth wins. L1 shape deviation is
0.0356 for Kingman+growth, 0.1078 for the best Beta (alpha=1.35 on this grid),
0.5373 for Kingman-constant; Kingman+growth reproduces the observed singleton
fraction (0.4832) to 0.4%. The Beta failure mode is diagnostic -- it overshoots
singletons and undershoots doubletons/tripletons by ~20%, the Lambda-coalescent
singleton spike with a flattened tail, which these data do not show. So
sweepstakes reproductive success is NOT required to explain the Illex SFS, and
the growth history used downstream is supported rather than assumed.

Quote the L1 deviations, NOT the AIC gap: with S = 85 M projected sites the
multinomial log-likelihoods are ~1.7e8 and differences reach 1e6, so the AIC
margin is a sample-size artifact and says nothing about decisiveness.

Caveat: the folded->unfolded step in project_folded() is a shape-level
approximation (neutral 1/i weights), applied only to the observed spectrum.
It cannot manufacture a preference between models -- simulated spectra are
folded natively -- but absolute likelihoods are not exact. Adequate for a
negative control; an exact projection would be warranted had Beta won.

This is an ILLEX analysis. It uses msprime only -- no msinv -- so nothing here
tunes the simulator.

Run from the repo root with the msinv venv (needs msprime + scipy: the
`illex` extra):
  .venv/bin/python illex/scripts/beta_vs_kingman.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import msprime

SFS_FILE = "/sietch_colab/data_share/illex/popgen_data/analysis/steps/08_demography/gw_folded_sfs.txt"
OUT_JSON = Path("results/illex/beta_vs_kingman.json")

# moments SFS fit (08_demography/RESULTS_demography.md)
N_ANC, N0, T_GROW = 547_928.0, 6_808_096.0, 769_519.0
MU = 3e-9
N_PROJ = 40           # haploid sample size to project/simulate at -> 20 folded bins
N_REPS = 100          # msprime replicates per model
SEQ_LEN = 2_000_000
REC = 2.5e-9
ALPHAS = [1.05, 1.2, 1.35, 1.5, 1.65, 1.75, 1.85, 1.9, 1.95, 1.99]


def load_observed():
    v = np.array([int(x) for x in open(SFS_FILE).read().split()], dtype=float)
    return v[1:]                      # drop invariant class


def project_folded(folded, n_from_hap, n_to_hap):
    """Hypergeometric projection of a FOLDED spectrum.

    Unfold ambiguously-folded counts by splitting each folded bin i between the
    unfolded classes i and n-i in proportion to the neutral 1/i weight, project
    the unfolded spectrum, then re-fold. Shape-level approximation, adequate for
    comparing spectra that have all been through the same transform.
    """
    n = n_from_hap
    unf = np.zeros(n, dtype=float)     # index j = derived count j (1..n-1)
    for idx, cnt in enumerate(folded, start=1):
        j, k = idx, n - idx
        if j == k:
            unf[j] += cnt
        else:
            wj, wk = 1.0 / j, 1.0 / k
            unf[j] += cnt * wj / (wj + wk)
            unf[k] += cnt * wk / (wj + wk)
    # project 1..n-1 -> 1..m-1
    m = n_to_hap
    out = np.zeros(m, dtype=float)
    from scipy.stats import hypergeom
    for j in range(1, n):
        if unf[j] <= 0:
            continue
        kk = np.arange(0, m + 1)
        pmf = hypergeom.pmf(kk, n, j, m)
        for k in kk:
            if 1 <= k <= m - 1:
                out[k] += unf[j] * pmf[k]
    # fold
    nb = m // 2
    fol = np.zeros(nb, dtype=float)
    for k in range(1, m):
        b = min(k, m - k)
        if b <= nb:
            fol[b - 1] += out[k]
    return fol


def sim_folded(model, demography, n_hap, reps, seed):
    """Expected folded SFS via BRANCH-mode AFS.

    Mutation-based SFS is unusable here: the Beta(2-a,a) coalescent runs on a
    ~N^(a-1) timescale, so its trees are orders of magnitude shorter than
    Kingman's and a fixed mutation rate yields ~10 sites vs ~2500. Branch mode
    reads the expected spectrum directly off branch lengths -- no mutation
    noise, and insensitive to the timescale difference we are not testing.
    """
    nb = n_hap // 2
    acc = np.zeros(nb, dtype=float)
    for r in range(reps):
        ts = msprime.sim_ancestry(
            samples=n_hap, ploidy=1, sequence_length=SEQ_LEN,
            recombination_rate=REC, model=model, demography=demography,
            random_seed=seed + r,
        )
        af = ts.allele_frequency_spectrum(
            polarised=False, span_normalise=True, mode="branch")
        take = np.asarray(af[1 : nb + 1], dtype=float)
        acc[: len(take)] += take / max(take.sum(), 1e-300)
    return acc


def multinomial_ll(obs, exp_shape):
    p = np.clip(exp_shape / exp_shape.sum(), 1e-300, None)
    return float(np.sum(obs * np.log(p)))


def l1_shape(obs, exp_shape):
    """Total absolute deviation between normalized spectra.

    This, not the AIC gap, is the interpretable model-comparison number here:
    it is invariant to the number of observed sites, whereas the multinomial
    log-likelihood scales with S (=85 M after projection) and inflates every
    difference into the millions.
    """
    o = obs / obs.sum()
    e = exp_shape / exp_shape.sum()
    return float(np.abs(o - e).sum())


def main():
    obs_full = load_observed()
    n_from = 2 * len(obs_full)
    obs = project_folded(obs_full, n_from, N_PROJ)
    print(f"observed: {len(obs_full)} folded bins (n={n_from} hap) "
          f"-> projected to n={N_PROJ} ({len(obs)} bins), S={obs.sum():,.0f}")
    print(f"projected singleton fraction = {obs[0]/obs.sum():.4f}")

    results = {}

    dem_const = msprime.Demography()
    dem_const.add_population(name="A", initial_size=N_ANC)

    dem_growth = msprime.Demography()
    alpha_g = math.log(N0 / N_ANC) / T_GROW
    dem_growth.add_population(name="A", initial_size=N0, growth_rate=alpha_g)
    dem_growth.add_population_parameters_change(
        time=T_GROW, population="A", initial_size=N_ANC, growth_rate=0)

    print("\nsimulating Kingman models...", flush=True)
    for tag, dem in (("kingman_constant", dem_const), ("kingman_moments_growth", dem_growth)):
        e = sim_folded(msprime.StandardCoalescent(), dem, N_PROJ, N_REPS, 1)
        ll = multinomial_ll(obs, e)
        k = 0 if tag.endswith("constant") else 2
        results[tag] = {"ll": ll, "k": k, "aic": 2 * k - 2 * ll,
                        "l1": l1_shape(obs, e),
                        "shape": (e / e.sum()).tolist()}
        print(f"  {tag:24s} lnL = {ll:,.1f}   L1 = {results[tag]['l1']:.4f}",
              flush=True)

    print("\nsimulating Beta-coalescent grid (no growth)...", flush=True)
    for a in ALPHAS:
        e = sim_folded(msprime.BetaCoalescent(alpha=a), dem_const, N_PROJ, N_REPS, 2)
        ll = multinomial_ll(obs, e)
        results[f"beta_alpha_{a}"] = {"ll": ll, "alpha": a, "k": 1,
                                     "aic": 2 * 1 - 2 * ll,
                                     "l1": l1_shape(obs, e),
                                     "shape": (e / e.sum()).tolist()}
        print(f"  alpha={a:<5} lnL = {ll:,.1f}   "
              f"L1 = {results[f'beta_alpha_{a}']['l1']:.4f}", flush=True)

    best_beta = max((v for k, v in results.items() if k.startswith("beta")),
                    key=lambda v: v["ll"])
    king_g = results["kingman_moments_growth"]
    king_c = results["kingman_constant"]

    print("\n" + "=" * 70)
    print("L1 shape deviation from observed (lower = better; THIS is the")
    print("interpretable comparison -- the AIC gap below is inflated by S):")
    print(f'  Kingman + moments growth   L1 = {king_g["l1"]:.4f}')
    print(f'  Beta alpha={best_beta["alpha"]:<5.2f}            L1 = {best_beta["l1"]:.4f}')
    print(f'  Kingman constant           L1 = {king_c["l1"]:.4f}')
    print(f"\nlog-likelihoods / AIC (magnitudes NOT interpretable, S={obs.sum():,.0f}):")
    print(f'  best Beta:  alpha={best_beta["alpha"]:.2f}  '
          f'lnL={best_beta["ll"]:,.1f}  AIC={best_beta["aic"]:,.1f}')
    print(f'  Kingman+growth        lnL={king_g["ll"]:,.1f}  AIC={king_g["aic"]:,.1f}')
    print(f'  Kingman constant      lnL={king_c["ll"]:,.1f}  AIC={king_c["aic"]:,.1f}')
    dlr = 2 * (best_beta["ll"] - king_c["ll"])
    print(f"\nLRT best-Beta vs Kingman-constant: 2dlnL = {dlr:,.1f} (1 df)")
    verdict = ("BETA favoured over Kingman+growth"
               if best_beta["ll"] > king_g["ll"] else
               "KINGMAN+GROWTH favoured over Beta")
    print("VERDICT: " + verdict)
    print("=" * 70)

    results["_meta"] = {"n_proj": N_PROJ, "reps": N_REPS, "seq_len": SEQ_LEN,
                        "obs_projected": obs.tolist(),
                        "obs_singleton_frac": float(obs[0] / obs.sum()),
                        "verdict": verdict, "lrt_beta_vs_kingman_const": dlr}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
