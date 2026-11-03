#!/usr/bin/env python
"""Refit the inversion with AA as the inverted (derived) arrangement, a selected rise, and gene flux.

    .venv/bin/python -m illex.scripts.refit_rise --reps 16 --workers 40 --tag _rise_pilot

POLARIZATION. Here the inverted class I is AA, at p_now = 0.626, and S is BB. Under a single structural event
the inverted class starts as one chromosome, so it carries less diversity than its frequency sustains until it
equilibrates or gene flux replenishes it. AA fits that: pi_AA/pi_BB = 0.738 against p/(1-p) = 1.67.

TARGETS (I = AA, inversion 60.54-79.50 Mb, 190 windows of 100 kb, delete-one jackknife over 20 1-Mb blocks):

    pi_I/pi_S = 0.7377 +- 0.0263      dxy/pi_I = 1.8763 +- 0.0503      corr -0.91
    singleton ratio f1(I)/f1(S) = 1.2307 +- 0.0115   (ANGSD, 1/0.8256 calibrated by the control 1/1.0161)

TRAJECTORY (``illex.balancing.arrival_curve``, backward time). The inverted class enters at t_inv with a founding
frequency p_start and stays there (dormancy) until the rise begins, rises under overdominance toward
p* = 0.626 over t_rise (s_het solved from t_rise), and has been held at p* since t_arrive. p_start = 0 in the grid
means a single chromosome, 1 / (2 N(t_inv)). A single chromosome cannot persist without rising, so for it the rise
starts at t_inv (t_rise = t_inv - t_arrive) and the t_rise grid value is ignored.

GENE FLUX (gamma, per bp per generation in heterokaryotypes). msinv's Peischl profile with tracts of 1e-4 of the
inversion is flat across the body, so flux exchanges ancestry along the inversion without a breakpoint-to-centre
gradient. gamma = 1e-15 is no flux.

Per-replicate rows go to refit_rise{tag}_reps.jsonl.gz, cell means and scores to refit_rise{tag}.csv.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from illex.scripts.sfs_shape import PROJ, project_unfolded

OUT = Path("results/illex")
P_STAR = 0.626
TARGETS = np.array([0.7377, 1.8763, 1.2307])
SE = np.array([0.0263, 0.0503, 0.0115])
CORR_RD = -0.91
PARS = ("t_inv", "p_start", "t_arrive", "t_rise", "gamma")


def _one(job):
    from illex import balancing as B
    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH
    from illex.slim.config import REC_RATE
    from illex.theory import N_growth

    t_inv, p_start, t_arr, t_rise, gamma, rep, cell, seed_offset = job
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    ps = p_start if p_start > 0 else 1.0 / (2.0 * float(N_growth(t_inv)))
    seed = seed_offset + 1000 * cell + rep
    t0 = time.time()
    sim = B.build_arrival_sim(seq_length=L, t_inv=t_inv, t_arrive=t_arr, t_rise=t_rise, p_start=ps,
                              p_star=P_STAR, gamma=gamma, seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    left, right = M.inversion_interval(sim)
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=(left, right))
    row = {"t_inv": t_inv, "p_start": p_start, "t_arrive": t_arr, "t_rise": t_rise, "gamma": gamma, "rep": rep,
           "seed": seed, "wall_s": round(time.time() - t0, 2),
           "pi_i_over_pi_s": st["pi_i_over_pi_s"], "dxy_over_pi_i": st["dxy_over_pi_i"]}
    for tag, nodes in (("I", i_nodes), ("S", s_nodes)):
        n = len(nodes)
        af = ts.allele_frequency_spectrum(sample_sets=[list(nodes)], windows=[0.0, left, right, ts.sequence_length],
                                          mode="branch", polarised=True, span_normalise=True)[1]
        v = np.asarray(af, dtype=float)
        row[tag] = (v / max(v[1:n].sum(), 1e-300)).tolist()
        row[f"{tag}_n"] = n
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--t-inv", default="850000,1000000,1300000,2000000")
    ap.add_argument("--p-start", default="0,0.025,0.15", help="0 = one chromosome at t_inv")
    ap.add_argument("--t-arrive", default="0,100000")
    ap.add_argument("--t-rise", default="50000,200000")
    ap.add_argument("--gamma", default="1e-15,1e-8,1e-7,1e-6,1e-5")
    ap.add_argument("--tag", default="_rise_pilot")
    ap.add_argument("--seed-offset", type=int, default=2_000_000_000)
    a = ap.parse_args()
    grid = [[float(x) for x in getattr(a, k).split(",")] for k in ("t_inv", "p_start", "t_arrive", "t_rise", "gamma")]
    cells = []
    for c in itertools.product(*grid):
        if c[1] == 0:   # single copy: the rise begins at the origin
            c = (c[0], c[1], c[2], c[0] - c[2], c[4])
        if c[2] + c[3] <= c[0] and c not in cells:
            cells.append(c)
    jobs = [(*c, r, i, a.seed_offset) for i, c in enumerate(cells) for r in range(a.reps)]
    print(f"{len(cells)} cells, {len(jobs):,} sims", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"refit_rise{a.tag}_reps.jsonl.gz", "wt") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")

    cov = np.diag(SE ** 2)
    cov[0, 1] = cov[1, 0] = CORR_RD * SE[0] * SE[1]
    W = np.linalg.inv(cov)
    out = []
    for c in cells:
        sub = [x for x in rows if tuple(x[p] for p in PARS) == c]
        r = float(np.mean([x["pi_i_over_pi_s"] for x in sub]))
        d = float(np.mean([x["dxy_over_pi_i"] for x in sub]))
        spec = {}
        for tag in ("I", "S"):
            n = sub[0][f"{tag}_n"]
            v = project_unfolded(np.mean([np.asarray(x[tag]) for x in sub], axis=0), n, PROJ)
            spec[tag] = v / v.sum()
        f1 = spec["I"][0] / spec["S"][0]
        res = np.array([r, d, f1]) - TARGETS
        out.append(dict(zip(PARS, c), pi_i_over_pi_s=r, dxy_over_pi_i=d, f1_ratio=f1,
                        D_meas=float(res @ W @ res), n=len(sub), wall_s=float(np.mean([x["wall_s"] for x in sub]))))
    out.sort(key=lambda x: x["D_meas"])
    with (OUT / f"refit_rise{a.tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"targets {TARGETS.tolist()}  SE {SE.tolist()}  (D with measurement covariance only)")
    print(f"{'t_inv':>9s} {'p_start':>8s} {'t_arr':>8s} {'t_rise':>8s} {'gamma':>8s} {'piI/piS':>8s} {'dxy/piI':>8s} "
          f"{'f1':>7s} {'D':>8s}")
    for x in out[:25]:
        print(f"{x['t_inv']:9.0f} {x['p_start']:8.3g} {x['t_arrive']:8.0f} {x['t_rise']:8.0f} {x['gamma']:8.0e} "
              f"{x['pi_i_over_pi_s']:8.4f} {x['dxy_over_pi_i']:8.4f} {x['f1_ratio']:7.4f} {x['D_meas']:8.2f}")
    print("best single-copy cells by gamma:")
    for g in grid[4]:
        sc = [x for x in out if x["p_start"] == 0 and x["gamma"] == g]
        if sc:
            x = sc[0]
            print(f"  gamma {g:.0e}: t_inv {x['t_inv']:.0f}, {x['pi_i_over_pi_s']:.3f} / {x['dxy_over_pi_i']:.3f} / "
                  f"{x['f1_ratio']:.3f}, D {x['D_meas']:.1f}")


if __name__ == "__main__":
    main()
