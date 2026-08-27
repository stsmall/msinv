#!/usr/bin/env python
"""Refit the inversion under the CORRECTED polarization.

    .venv/bin/python -m illex.scripts.refit_decline --reps 48 --workers 12

WHY EVERYTHING IS BEING REFITTED
--------------------------------
The polarization was reversed (NOTES sec 8.15): I (inverted, derived) is the BB
cluster at p = 0.374, not the AA cluster at 0.626. The cluster labels are
unchanged; only the interpretation is swapped. Every fit in NOTES sec 7 and
8.6-8.8 was made against the old targets and is invalid.

Corrected targets (differentiated body, 1 Mb block jackknife):

    pi_I/pi_S = 1.3556 +- 0.0481      (was 0.7368)
    dxy/pi_I  = 1.3848 +- 0.0214      (was 1.8794)
    f1(I)/f1(S) = 0.8256              (was 1.211)   ANGSD/GL, NOTES sec 8.5

WHY THE OLD TRAJECTORY FAMILIES CANNOT WORK
-------------------------------------------
Two analytic facts settle the shape before any simulation runs:

1. A long-standing balanced polymorphism at p = 0.374 gives
   pi_I/pi_S -> p/(1-p) = 0.597. The observed 1.356 is **2.27x higher**: the
   inverted class carries far more diversity than its current frequency can
   sustain. A CONSTANT frequency reproducing 1.356 would be p = 0.576. So the
   inverted arrangement was formerly COMMONER and has declined.
2. **[W] A second analytic argument of mine was wrong here, and the pilot
   caught it.** I claimed dxy/pi_I >= 1 + T_anc/t_inv, hence t_inv >= 2.85 My.
   That treats pi_I as growing like 2*mu*t_inv, but pi_I SATURATES at the
   inverted class's own equilibrium, after which extra t_inv only inflates dxy.
   The pilot showed dxy/pi_I RISING with t_inv (3.84 at 3 My, 6.67 at 6 My) --
   the opposite of the predicted direction. The informative range is
   t_inv ~ 1e5-1e6, not >= 2.85 My.

Every family in NOTES sec 8.6-8.8 rises to its equilibrium and stays there, so
none can put the inverted class ABOVE its equilibrium diversity. Hence
``illex.balancing.decline_curve``: held at ``p_hist`` for most of its history,
then declining to 0.374.

WHAT THE FIT FOUND, AND WHAT t_fall IS FOR
------------------------------------------
Arose ~850 ka, held near p_hist ~ 0.70 for most of its history, declining since
~175 ka to 0.374 today -- the mirror image of the pre-correction reading. All
three statistics land inside 3%.

``t_fall``, the duration of the decline, was initially FIXED at 100 ky. It is now
scanned, because the neutrality argument turns on it (NOTES sec 8.17.2): a fall
of 0.326 is 5.7 SD of drift if it took 100 ky but only 1.8 SD if it took 500 ky,
which drift produces ~4% of the time. With four parameters against three targets
t_fall cannot be point-identified, so the script reports a PROFILE -- the best
achievable score at each t_fall with the other three re-optimised. A profile that
rises steeply with t_fall means the data REQUIRE a fast fall and the decline is
genuinely too fast for drift; a flat profile means they merely tolerate one and
the decline test cannot carry weight alone.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from illex import empirical
from illex.scripts.sfs_shape import PROJ, project_unfolded

OUT = Path("results/illex")
GAMMA = 1e-15

# t_fall is now SCANNED, not fixed. With four parameters against three targets
# it cannot be uniquely determined; what the scan delivers is a PROFILE -- the
# best achievable score at each t_fall, with the other three re-optimised. That
# is the quantity the neutrality argument needs (NOTES sec 8.17.2): the decline
# test is decisive only if the data REQUIRE a fast fall, and a flat profile
# would mean they merely tolerate one.
T_FALL = [25_000.0, 50_000.0, 100_000.0, 200_000.0, 400_000.0]

T_INV = [3.0e6, 4.0e6, 6.0e6]
P_HIST = [0.50, 0.58, 0.66]
T_DECLINE = [100_000.0, 300_000.0]

TARGET_R = empirical.PI_I_OVER_PI_S_BODY
TARGET_D = empirical.DXY_OVER_PI_I_BODY
TARGET_F = empirical.SFS_F1_RATIO_BODY


def _one(job):
    from illex import balancing as B
    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH
    from illex.slim.config import REC_RATE

    t_inv, p_hist, t_dec, t_fall, rep = job
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    seed = (9_000_000 + 100_000 * int(t_inv / 1e5) + 1000 * int(p_hist * 100)
            + 10 * int(t_dec / 1e4) + int(t_fall / 1e4) + rep)
    t0 = time.time()
    sim = B.build_decline_sim(seq_length=L, t_inv=t_inv, t_decline=t_dec,
                              t_fall=t_fall, p_hist=p_hist, gamma=GAMMA,
                              seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    left, right = M.inversion_interval(sim)
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=(left, right))
    row = {"t_inv": t_inv, "p_hist": p_hist, "t_decline": t_dec,
           "t_fall": t_fall, "rep": rep,
           "seed": seed, "wall_s": round(time.time() - t0, 2),
           "pi_i_over_pi_s": st["pi_i_over_pi_s"],
           "dxy_over_pi_i": st["dxy_over_pi_i"]}
    for tag, nodes in (("I", i_nodes), ("S", s_nodes)):
        n = len(nodes)
        af = ts.allele_frequency_spectrum(
            sample_sets=[list(nodes)],
            windows=[0.0, left, right, ts.sequence_length],
            mode="branch", polarised=True, span_normalise=True)[1]
        v = np.asarray(af, dtype=float)
        row[tag] = (v / max(v[1:n].sum(), 1e-300)).tolist()
        row[f"{tag}_n"] = n
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=48)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--t-inv")
    ap.add_argument("--p-hist")
    ap.add_argument("--t-decline")
    ap.add_argument("--t-fall")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global T_INV, P_HIST, T_DECLINE, T_FALL
    if a.t_inv:
        T_INV = [float(x) for x in a.t_inv.split(",")]
    if a.p_hist:
        P_HIST = [float(x) for x in a.p_hist.split(",")]
    if a.t_decline:
        T_DECLINE = [float(x) for x in a.t_decline.split(",")]
    if a.t_fall:
        T_FALL = [float(x) for x in a.t_fall.split(",")]

    jobs = [(t, p, d, f, r)
            for t, p, d, f in itertools.product(T_INV, P_HIST, T_DECLINE,
                                                T_FALL)
            if d + f <= t
            for r in range(a.reps)]
    print(f"{len(jobs):,} sims  (p_now={empirical.P_INV}, "
          f"{len(T_FALL)} t_fall values)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f} s")

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("DECLINE family refit, corrected polarization (I = BB, p_now = 0.374)")
    emit(f"targets: pi_I/pi_S={TARGET_R:.4f}  dxy/pi_I={TARGET_D:.4f}  "
         f"f1(I)/f1(S)={TARGET_F:.4f}")
    emit()
    emit(f"{'t_inv':>9s} {'p_hist':>7s} {'t_decl':>9s} {'t_fall':>8s} "
         f"{'pi_I/pi_S':>10s} {'dxy/pi_I':>9s} {'f1ratio':>8s} "
         f"{'miss_r':>7s} {'miss_d':>7s} {'miss_f':>7s} {'score':>7s}")
    best = None
    out_rows = []
    for t, p, d, tf in itertools.product(T_INV, P_HIST, T_DECLINE, T_FALL):
        sub = [x for x in rows if x["t_inv"] == t and x["p_hist"] == p
               and x["t_decline"] == d and x["t_fall"] == tf]
        if not sub:
            continue
        r = float(np.mean([x["pi_i_over_pi_s"] for x in sub]))
        dd = float(np.mean([x["dxy_over_pi_i"] for x in sub]))
        spec = {}
        for tag in ("I", "S"):
            n = sub[0][f"{tag}_n"]
            acc = np.zeros(n + 1, dtype=float)
            for x in sub:
                acc += np.asarray(x[tag], dtype=float)
            acc /= len(sub)
            v = project_unfolded(acc, n, PROJ)
            spec[tag] = v / v.sum()
        f1 = spec["I"][0] / spec["S"][0]
        mr = (r - TARGET_R) / TARGET_R
        md = (dd - TARGET_D) / TARGET_D
        mf = (f1 - TARGET_F) / TARGET_F
        score = mr * mr + md * md + mf * mf
        if best is None or score < best[0]:
            best = (score, t, p, d, r, dd, f1, tf)
        emit(f"{t:9,.0f} {p:7.3f} {d:9,.0f} {tf:8,.0f} {r:10.4f} {dd:9.4f} "
             f"{f1:8.4f} {100 * mr:+6.1f}% {100 * md:+6.1f}% "
             f"{100 * mf:+6.1f}% {score:7.4f}")
        out_rows.append({"t_inv": t, "p_hist": p, "t_decline": d,
                         "t_fall": tf, "pi_i_over_pi_s": r,
                         "dxy_over_pi_i": dd, "f1_ratio": f1,
                         "score": score, "n": len(sub)})
    emit(f"{'TARGET':>9s} {'':>7s} {'':>9s} {'':>8s} {TARGET_R:10.4f} "
         f"{TARGET_D:9.4f} {TARGET_F:8.4f}")
    if best:
        emit()
        emit(f"BEST  t_inv={best[1]:,.0f}  p_hist={best[2]:.3f}  "
             f"t_decline={best[3]:,.0f}  t_fall={best[7]:,.0f}  ->  "
             f"{best[4]:.4f} / {best[5]:.4f} / {best[6]:.4f}  "
             f"(score {best[0]:.5f})")

    # PROFILE over t_fall: best achievable score at each value, the other
    # three parameters re-optimised. This is the output the neutrality
    # argument needs.
    emit()
    emit("=" * 74)
    emit("PROFILE over t_fall (best score at each value, others re-optimised)")
    emit("=" * 74)
    emit(f"  {'t_fall':>9s} {'best score':>11s} {'at t_inv':>10s} "
         f"{'p_hist':>7s} {'t_decl':>9s}   drift SD of the implied fall")
    from illex.theory import N_growth
    for tf in T_FALL:
        cells = [x for x in out_rows if x["t_fall"] == tf]
        if not cells:
            continue
        b = min(cells, key=lambda x: x["score"])
        tt = np.linspace(b["t_decline"], b["t_decline"] + tf, 5000)
        ph = b["p_hist"]
        sd = float(np.sqrt(ph * (1 - ph)
                           * np.trapezoid(1.0 / (2.0 * N_growth(tt)), tt)))
        fall = ph - empirical.P_INV
        emit(f"  {tf:9,.0f} {b['score']:11.5f} {b['t_inv']:10,.0f} "
             f"{ph:7.3f} {b['t_decline']:9,.0f}   "
             f"{fall:.3f} / {sd:.4f} = {fall / sd:5.1f} SD")
    emit()
    emit("  A profile that RISES steeply with t_fall means the data require a")
    emit("  fast fall, and the decline is then too fast for drift. A FLAT")
    emit("  profile means t_fall is unidentified and the decline test cannot")
    emit("  carry weight on its own (NOTES sec 8.17.2).")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"refit_decline{a.tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    (OUT / f"refit_decline{a.tag}.txt").write_text("\n".join(lines) + "\n")
    json.dump({"targets": {"r": TARGET_R, "d": TARGET_D, "f1": TARGET_F},
               "cells": out_rows},
              (OUT / f"refit_decline{a.tag}.json").open("w"), indent=2)
    print(f"\nwrote {OUT}/refit_decline{a.tag}.*")


if __name__ == "__main__":
    main()
