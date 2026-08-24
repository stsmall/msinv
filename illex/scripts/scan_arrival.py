#!/usr/bin/env python
"""Scan the explicit-arrival trajectory family against all THREE statistics.

    .venv/bin/python -m illex.scripts.scan_arrival --reps 48 --workers 26

WHAT THIS IS FOR
----------------
The two-phase balancing family fits pi_I/pi_S and dxy/pi_I exactly but is
rejected by the ANGSD/GL spectrum: it over-predicts the inverted-vs-standard
singleton skew, ratio 1.31 against an observed 1.211, because it keeps the
standard arrangement confined to 1 - p* = 0.374 for too long (NOTES sec 8.5.3).
It cannot be fixed by re-tuning, because 70.3% of its rise is spent above
0.90 p* whatever s_het is.

``illex.balancing.arrival_curve`` adds a third phase and makes the arrival time
explicit. This scan asks whether that buys anything real:

    does t_arrive move the SFS contrast, at fixed pi_I/pi_S and dxy/pi_I?

THE PREDICTION, STATED BEFORE RUNNING
-------------------------------------
Later arrival (smaller t_arrive) means the standard class has been confined for
less time, so its spectrum is less reshaped, so the inverted-vs-standard
singleton ratio should FALL toward the observed 1.211. Meanwhile a long dormancy
at low p_start squeezes the inverted class hard, so pi_I/pi_S should fall too --
the two effects are not independent and p_start will have to rise to compensate.
If the ratio does not move with t_arrive, the third phase is useless and the
tension in sec 8.5.3 is telling us something else.

WHAT IS FIXED
-------------
t_inv is held at the fitted 730,000 because dxy/pi_I pins it and is largely
indifferent to what p does after the origin. t_rise is held at 50,000 -- fast,
so that "arrival" is sharp. Both are scanned separately once the (t_arrive,
p_start) response is known; this run is about whether the new axis does anything.
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
ANGSD_JSON = OUT / "sfs_shape_angsd.json"
OUT_CSV = OUT / "scan_arrival.csv"
OUT_TXT = OUT / "scan_arrival.txt"

T_INV_DEFAULT = [730_000.0]
T_RISE = 50_000.0
T_ARRIVE = [0.0, 100_000.0, 250_000.0, 400_000.0, 550_000.0]
P_START = [0.027, 0.10, 0.25]
T_INV = list(T_INV_DEFAULT)

TARGET_R = empirical.PI_I_OVER_PI_S_BODY
TARGET_D = empirical.DXY_OVER_PI_I_BODY


def _one(job):
    from illex import balancing as B
    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH
    from illex.slim.config import REC_RATE

    t_inv, t_arrive, p_start, rep, drift, p_origin_mode = job
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    seed = (8_000_000 + 100_000 * int(t_inv / 1e4)
            + 1_000 * int(t_arrive / 1e4) + 10 * int(p_start * 100) + rep)
    t0 = time.time()
    if drift:
        # p_origin=None is a genuine single origin, 1/(2N(t_inv)); "hand" starts
        # the drift at the handoff frequency itself, i.e. no net trend, which is
        # the nearest drifting analogue of the fixed-frequency family.
        p_org = None if p_origin_mode == "single" else p_start
        sim = B.build_arrival_drift_sim(
            seq_length=L, t_inv=t_inv, t_arrive=t_arrive, t_rise=T_RISE,
            p_hand=p_start, drift_seed=seed + 777_000, p_origin=p_org,
            seed=seed)
    else:
        sim = B.build_arrival_sim(
            seq_length=L, t_inv=t_inv, t_arrive=t_arrive, t_rise=T_RISE,
            p_start=p_start, seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    left, right = M.inversion_interval(sim)
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=(left, right))
    row = {
        "t_arrive": t_arrive, "p_start": p_start, "rep": rep,
        "t_inv": t_inv, "t_rise": T_RISE, "seed": seed,
        "wall_s": round(time.time() - t0, 2),
        "pi_i_over_pi_s": st["pi_i_over_pi_s"],
        "dxy_over_pi_i": st["dxy_over_pi_i"],
    }
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
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--t-inv", help="comma-separated t_inv grid")
    ap.add_argument("--t-arrive", help="comma-separated t_arrive grid")
    ap.add_argument("--p-start", help="comma-separated p_start grid")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--drift", action="store_true",
                    help="drifting dormancy (guided WF bridge) instead of a "
                         "constant-frequency dormancy")
    ap.add_argument("--p-origin", choices=["single", "hand"], default="single",
                    help="with --drift: 'single' = one chromosome at t_inv "
                         "(genuine single origin); 'hand' = start the drift at "
                         "the handoff frequency (no net trend)")
    args = ap.parse_args()

    global T_INV, T_ARRIVE, P_START
    if args.t_inv:
        T_INV = [float(x) for x in args.t_inv.split(",")]
    if args.t_arrive:
        T_ARRIVE = [float(x) for x in args.t_arrive.split(",")]
    if args.p_start:
        P_START = [float(x) for x in args.p_start.split(",")]
    out_csv = OUT / f"scan_arrival{args.tag}.csv"
    out_txt = OUT / f"scan_arrival{args.tag}.txt"

    ang = json.loads(ANGSD_JSON.read_text())
    oi = np.array(ang["AA_body"], float)
    oi = oi / oi.sum()
    os_ = np.array(ang["BB_body"], float)
    os_ = os_ / os_.sum()
    target_ratio = oi[0] / os_[0]

    jobs = [(ti, ta, p0, r, args.drift, args.p_origin)
            for ti, ta, p0 in itertools.product(T_INV, T_ARRIVE, P_START)
            if ta + T_RISE <= ti
            for r in range(args.reps)]
    print(f"{len(jobs):,} sims  (t_rise={T_RISE:,.0f}, "
          f"{len(T_INV)}x{len(T_ARRIVE)}x{len(P_START)} grid)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=1))
    print(f"done in {time.time() - t0:.0f} s")

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    mode = (f"DRIFTING dormancy (guided WF bridge, p_origin={args.p_origin})"
            if args.drift else "CONSTANT-frequency dormancy")
    emit(f"Explicit-arrival family, {mode}: t_rise={T_RISE:,.0f}, "
         f"{args.reps} reps/cell")
    emit(f"targets: pi_I/pi_S={TARGET_R:.4f}  dxy/pi_I={TARGET_D:.4f}  "
         f"ANGSD f1(I)/f1(S)={target_ratio:.3f}")
    emit()
    emit(f"{'t_inv':>9s} {'t_arrive':>9s} {'p_start':>7s} {'pi_I/pi_S':>10s} "
         f"{'dxy/pi_I':>9s} {'ratio':>7s} {'miss_r':>7s} {'miss_d':>7s} "
         f"{'miss_rat':>8s} {'score':>7s}")
    out_rows = []
    from illex import balancing as B
    best = None
    for ti, ta, p0 in itertools.product(T_INV, T_ARRIVE, P_START):
        sub = [x for x in rows if x["t_inv"] == ti and x["t_arrive"] == ta
               and x["p_start"] == p0]
        if not sub:
            continue
        r = float(np.mean([x["pi_i_over_pi_s"] for x in sub]))
        d = float(np.mean([x["dxy_over_pi_i"] for x in sub]))
        spec = {}
        for tag in ("I", "S"):
            n = sub[0][f"{tag}_n"]
            acc = np.zeros(n + 1, dtype=float)
            for x in sub:
                acc += np.asarray(x[tag], dtype=float)
            acc /= len(sub)
            p = project_unfolded(acc, n, PROJ)
            spec[tag] = p / p.sum()
        ratio = spec["I"][0] / spec["S"][0]
        l1c = float(np.abs((spec["I"] - spec["S"]) - (oi - os_)).sum())
        s_het = B.s_het_for_rise(T_RISE, p0)
        mr = (r - TARGET_R) / TARGET_R
        md = (d - TARGET_D) / TARGET_D
        mt = (ratio - target_ratio) / target_ratio
        score = mr * mr + md * md + mt * mt
        if best is None or score < best[0]:
            best = (score, ti, ta, p0, r, d, ratio)
        emit(f"{ti:9,.0f} {ta:9,.0f} {p0:7.3f} {r:10.4f} {d:9.4f} "
             f"{ratio:7.3f} {100 * mr:+6.1f}% {100 * md:+6.1f}% "
             f"{100 * mt:+7.1f}% {score:7.4f}")
        out_rows.append({
            "t_arrive": ta, "p_start": p0, "t_inv": ti, "t_rise": T_RISE,
            "s_het": s_het, "pi_i_over_pi_s": r, "dxy_over_pi_i": d,
            "f1_I": float(spec["I"][0]), "f1_S": float(spec["S"][0]),
            "ratio": float(ratio), "l1_contrast": l1c,
            "n": len(sub),
        })
    emit(f"{'TARGET':>9s} {'':>9s} {'':>7s} {TARGET_R:10.4f} {TARGET_D:9.4f} "
         f"{target_ratio:7.3f}")
    if best:
        emit()
        emit(f"BEST CELL  t_inv={best[1]:,.0f}  t_arrive={best[2]:,.0f}  "
             f"p_start={best[3]:.3f}  ->  {best[4]:.4f} / {best[5]:.4f} / "
             f"{best[6]:.3f}  (score {best[0]:.5f})")
    emit()
    emit("Reference -- the two-phase family at its fitted point: "
         "ratio 1.312, L1(I-S) 0.101.")
    emit("The question is whether t_arrive moves the ratio at all. If the "
         "column is flat,\nthe third phase buys nothing and sec 8.5.3's "
         "tension is about something else.")

    # Response of each statistic to t_arrive, at each p_start.
    emit()
    emit("response to t_arrive (range across the scanned t_arrive, per "
         "p_start):")
    for p0 in P_START:
        sel = [x for x in out_rows if x["p_start"] == p0]
        if not sel:
            continue
        for key, lab in (("ratio", "f1 ratio"), ("pi_i_over_pi_s", "pi_I/pi_S"),
                         ("dxy_over_pi_i", "dxy/pi_I")):
            v = [x[key] for x in sel]
            emit(f"  p_start={p0:5.3f}  {lab:<10s} "
                 f"{min(v):.4f} -> {max(v):.4f}   (span {max(v) - min(v):.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_csv} and {out_txt}")


if __name__ == "__main__":
    main()
