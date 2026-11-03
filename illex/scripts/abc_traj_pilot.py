#!/usr/bin/env python
"""Pilot rejection ABC: trace the 2a frequency from its origin to the present.

    .venv/bin/python -m illex.scripts.abc_traj_pilot --batches 200 --batch 4000 --workers 40 --tag _p1

POLARIZATION. The inverted class I is 2a (the AA cluster, p_now = 0.626) and S is the reference arrangement 2+.

TRAJECTORY, forward in time on the fitted moments growth model (N_ANC 547,928 to N0 6,808,096 over 769,519
generations). The inversion arises at t_inv (generations before present).

  * Single copy: one chromosome, 1/(2N), under selection from its origin. The establishment phase is exact
    Wright-Fisher (binomial each generation) until the arrangement is lost (the attempt is redrawn) or reaches
    ESTABLISHED copies, then the diffusion takes over.
  * Multiple copies: founding frequency p0 at t_inv, neutral drift until selection starts at t_sel, then selection.

Selection is genic on the arrangement, dp = s p (1 - p) per generation. s = 0 is neutral drift throughout. The
diffusion uses steps of DT generations with Wright-Fisher variance p (1 - p) DT / (2 N(t)), and loss at zero is a
rejection. A path is accepted when its present frequency lies within P_TOL of 0.626, and only accepted paths are
passed to msinv, as a precomputed trajectory. Flux is off (gamma 1e-15).

Statistics per accepted sim are pi_I/pi_S, dxy/pi_I and the singleton ratio f1(I)/f1(S), with D against the
observed values using the measurement covariance. Rows go to results/illex/abc_traj{tag}.jsonl.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("results/illex")
P_NOW = 0.626
P_TOL = 0.02
DT = 10.0
ESTABLISHED = 500
REC_EVERY = 1000.0            # trajectory sampling interval passed to msinv, generations
TARGETS = np.array([0.7377, 1.8763, 1.2307])
SE = np.array([0.0263, 0.0503, 0.0115])
CORR_RD = -0.91
PRIOR = dict(t_inv=(500_000.0, 1_200_000.0), p_single=0.5, p0=(1e-3, 0.5), s=(1e-6, 1e-3), p_neutral=0.2)


def draw_prior(rng, n):
    t_inv = rng.uniform(*PRIOR["t_inv"], n)
    single = rng.random(n) < PRIOR["p_single"]
    p0 = np.exp(rng.uniform(np.log(PRIOR["p0"][0]), np.log(PRIOR["p0"][1]), n))
    t_sel = np.where(single, t_inv, rng.uniform(0.0, 1.0, n) * t_inv)
    s = np.exp(rng.uniform(np.log(PRIOR["s"][0]), np.log(PRIOR["s"][1]), n))
    s[rng.random(n) < PRIOR["p_neutral"]] = 0.0
    return t_inv, single, np.where(single, 0.0, p0), t_sel, s


def establish(rng, n_e, s, max_attempts=2_000_000):
    """Exact WF from one copy until ESTABLISHED copies. Returns (generations elapsed, path) or None."""
    two_n = int(2 * n_e)
    for _ in range(max_attempts):
        k, g, path = 1, 0, [(0, 1)]
        while 0 < k < ESTABLISHED:
            p = k / two_n
            k = rng.binomial(two_n, p + s * p * (1.0 - p))
            g += 1
            if g % 100 == 0:
                path.append((g, k))
        if k >= ESTABLISHED:
            path.append((g, k))
            return g, [(gg, kk / two_n) for gg, kk in path]
    return None


def traj_batch(job):
    from illex.theory import N_growth

    seed, n = job
    rng = np.random.default_rng(seed)
    t_inv, single, p0, t_sel, s = draw_prior(rng, n)
    start = t_inv.copy()              # backward time at which the diffusion phase begins
    p = p0.copy()
    est_paths = [None] * n
    ok = np.ones(n, bool)
    for i in np.where(single)[0]:
        r = establish(rng, float(N_growth(t_inv[i])), s[i])
        if r is None or r[0] >= t_inv[i]:
            ok[i] = False
            continue
        g, path = r
        start[i] = t_inv[i] - g
        p[i] = path[-1][1]
        est_paths[i] = [(t_inv[i] - gg, pp) for gg, pp in path]
    t_max = float(np.max(np.where(ok, start, 0.0)))
    grid = np.arange(0.0, t_max + REC_EVERY, REC_EVERY)       # backward record times
    rec = np.full((n, grid.size), np.nan)
    gi = grid.size - 1
    t = t_max
    active = np.zeros(n, bool)
    while t > 0:
        active |= (start >= t) & ok
        a = np.where(active)[0]
        if a.size:
            ne = float(N_growth(t))
            pa = p[a]
            sel = np.where(t <= t_sel[a], s[a], 0.0)
            pa = pa + sel * pa * (1 - pa) * DT + np.sqrt(np.maximum(pa * (1 - pa), 0) * DT / (2 * ne)) \
                * rng.standard_normal(a.size)
            lost = pa <= 0
            ok[a[lost]] = False
            active[a[lost]] = False
            p[a] = np.clip(pa, 0, 1)
        t -= DT
        while gi >= 0 and grid[gi] >= t:
            rec[active, gi] = p[active]
            gi -= 1
    out = []
    for i in np.where(ok & (np.abs(p - P_NOW) <= P_TOL))[0]:
        m = ~np.isnan(rec[i]) & (grid < start[i])
        pts = (est_paths[i] or [(float(t_inv[i]), float(p0[i]))]) + list(zip(grid[m].tolist(), rec[i, m].tolist()))
        pts = sorted(set((float(a_), float(b_)) for a_, b_ in pts))
        times = np.array([x[0] for x in pts])
        freqs = np.array([x[1] for x in pts])
        keep = np.concatenate([[True], np.diff(times) > 1e-6])
        times, freqs = times[keep], freqs[keep]
        if times[0] > 0:
            times = np.concatenate([[0.0], times])
            freqs = np.concatenate([[float(p[i])], freqs])
        out.append(dict(t_inv=float(t_inv[i]), single=bool(single[i]), p0=float(p0[i]), t_sel=float(t_sel[i]),
                        s=float(s[i]), p_now=float(p[i]), times=times.tolist(), freqs=freqs.tolist(),
                        p_max=float(freqs.max()),
                        p_at={str(k): float(np.interp(k, times, freqs)) for k in (100_000, 300_000, 500_000)}))
    return n, int(ok.sum()), out


def run_msinv(job):
    from msinv import HullSimulator, InversionSpec

    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH, growth_demography
    from illex.model import MARGIN_FRACTION, TRACT_FRACTION
    from illex.scripts.sfs_shape import PROJ, project_unfolded
    from illex.slim.config import REC_RATE
    from illex.theory import N_growth

    row, seed = job
    t0 = time.time()
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    margin = L * MARGIN_FRACTION
    spec = InversionSpec(bp_left=margin, bp_right=L - margin, gene_conversion_rate=1e-15,
                         mean_tract_length=max(1.0, (L - 2 * margin) * TRACT_FRACTION),
                         tract_distribution="geometric",
                         trajectory={"type": "precomputed", "times": np.asarray(row["times"]),
                                     "freqs": [np.asarray(row["freqs"])],
                                     "n_e": [float(N_growth(row["t_inv"]))], "t_inv": [row["t_inv"]]})
    sim = HullSimulator(n_std=100, n_inv=100, population_size=PRESENT_NE_GROWTH, demography=growth_demography(),
                        sequence_length=L, recombination_rate=REC_RATE, inversions=[spec], seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    left, right = M.inversion_interval(sim)
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=(left, right))
    f1 = []
    for nodes in (i_nodes, s_nodes):
        n = len(nodes)
        af = ts.allele_frequency_spectrum(sample_sets=[list(nodes)], windows=[0.0, left, right, ts.sequence_length],
                                          mode="branch", polarised=True, span_normalise=True)[1]
        v = np.asarray(af, dtype=float)
        v = project_unfolded(v / max(v[1:n].sum(), 1e-300), n, PROJ)
        f1.append(v[0] / v.sum())
    out = {k: v for k, v in row.items() if k not in ("times", "freqs")}
    out.update(seed=seed, wall_s=round(time.time() - t0, 2), pi_ratio=st["pi_i_over_pi_s"],
               dxy_pi=st["dxy_over_pi_i"], f1_ratio=f1[0] / f1[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tag", default="_p1")
    ap.add_argument("--seed-offset", type=int, default=3_000_000_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    acc, n_tot, n_surv = [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, ns, rows in ex.map(traj_batch, [(a.seed_offset + b, a.batch) for b in range(a.batches)]):
            n_tot += n
            n_surv += ns
            acc.extend(rows)
    print(f"trajectories {n_tot:,}, survived {n_surv:,}, accepted at p_now {len(acc):,} "
          f"({time.time() - t0:.0f} s)", flush=True)
    with gzip.open(OUT / f"abc_traj{a.tag}_paths.jsonl.gz", "wt") as fh:
        for r in acc:
            fh.write(json.dumps(r) + "\n")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_msinv, [(r, a.seed_offset + 10_000_000 + k) for k, r in enumerate(acc)], chunksize=4))
    cov = np.diag(SE ** 2)
    cov[0, 1] = cov[1, 0] = CORR_RD * SE[0] * SE[1]
    W = np.linalg.inv(cov)
    for r in rows:
        res = np.array([r["pi_ratio"], r["dxy_pi"], r["f1_ratio"]]) - TARGETS
        r["D"] = float(res @ W @ res)
    with gzip.open(OUT / f"abc_traj{a.tag}.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"msinv done ({time.time() - t0:.0f} s), {len(rows):,} sims", flush=True)
    rows.sort(key=lambda r: r["D"])
    for q in (0.01, 0.05):
        k = max(1, int(q * len(rows)))
        top = rows[:k]
        print(f"closest {q:.0%} ({k}, D <= {top[-1]['D']:.1f}): single-copy share "
              f"{np.mean([r['single'] for r in top]):.2f}, neutral share {np.mean([r['s'] == 0 for r in top]):.2f}")
        for key in ("t_inv", "p0", "t_sel", "s", "p_max"):
            v = np.array([r[key] for r in top])
            print(f"   {key:6s} median {np.median(v):.4g}  90% {np.quantile(v, 0.05):.4g}-{np.quantile(v, 0.95):.4g}")


if __name__ == "__main__":
    main()
