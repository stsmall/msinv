"""Flux calibration for the trajectory ABC (Task 5): how large c must be before the flux moves dxy / pi_I.

Sweeps c (per-site conversion probability per generation in heterokaryotypes) over 1e-10..1e-5 in half-decade
steps, plus a no-flux baseline on the SAME engine (msinv, c = GAMMA_FLOOR), for three fixed paths:
  (a) a soft D1 pilot-posterior path (t_inv in 7.5-8.5e5, closest to 8.2e5) from results/illex/abc_traj_p1_paths.jsonl.gz
  (b) a hard B path (seeded draw_prior + simulate_paths, first accepted)
  (c) a soft N path with t_inv forced to 1.2e6 (same way)
Only pi_ratio and dxy_pi are used (msinv's within-arrangement LD is biased). c* is the smallest c whose mean
dxy_pi differs from the baseline mean by more than one observed SE; GAMMA_MAX = min(100 c*, 1e-5).

Rerunnable: rows already in the CSV (same path, c, rep) are skipped; --workers sets the pool size.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import resource
import time
from pathlib import Path as FsPath

import numpy as np

GAMMA_CAP = 1e-5
C_GRID = [10 ** (-10 + 0.5 * i) for i in range(11)]            # 1e-10 .. 1e-5
PATH_LABELS = ("a_soft_D1", "b_hard_B", "c_soft_N_1.2Ma")
REPO = FsPath(__file__).resolve().parents[2]
OUT_CSV = REPO / "results/illex/abc_flux_calibration.csv"
OUT_TXT = REPO / "results/illex/abc_flux_calibration.txt"
FIELDS = ["path", "c", "rep", "seed", "pi_ratio", "dxy_pi", "wall_s", "peak_rss_mb"]


def c_star(cs, dxy_by_c, baseline, se):
    """Smallest c (cs ascending) whose mean dxy_pi differs from the baseline mean by more than se, else None."""
    base = float(np.mean(baseline))
    for c, v in zip(cs, dxy_by_c):
        if abs(float(np.mean(v)) - base) > se:
            return float(c)
    return None


def gamma_max(cstar, cap=GAMMA_CAP):
    return cap if cstar is None else min(100.0 * cstar, cap)


def build_paths():
    """The three fixed paths as {label: (Path, provenance dict)}."""
    from illex.abc.trajectory import Path, draw_prior, simulate_paths

    out = {}
    best = None
    with gzip.open(REPO / "results/illex/abc_traj_p1_paths.jsonl.gz", "rt") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            if d["single"] or not 7.5e5 <= d["t_inv"] <= 8.5e5:
                continue
            if best is None or abs(d["t_inv"] - 8.2e5) < abs(best[1]["t_inv"] - 8.2e5):
                best = (i, d)
    i, d = best
    prm = dict(t_inv=d["t_inv"], hard=False, p0=d["p0"], t_sel=d["t_sel"], s=d["s"], t_stop=math.nan,
               p_star=math.nan, gamma_eff=math.nan, model="D1")
    out[PATH_LABELS[0]] = (Path(prm, np.asarray(d["times"], float), np.asarray(d["freqs"], float), d["p_now"]),
                           dict(row=i, **{k: prm[k] for k in ("t_inv", "p0", "t_sel", "s")}, p_now=d["p_now"]))

    def gen(model, cls, seed, t_force=None):
        rng = np.random.default_rng(seed)
        paths = []
        while not paths:                                  # acceptance at p_now is ~0.2%: draw in batches
            n = 20000
            p = draw_prior(model, False, rng, n, cls=cls)
            if t_force is not None:
                p["t_inv"] = np.full(n, t_force)          # N: no t_sel / s, so nothing else depends on t_inv
            paths, _ = simulate_paths(p, rng)
        pa = paths[0]
        pa.params = {k: (float(v) if np.ndim(v) == 0 and k not in ("model", "flux", "hard") else v)
                     for k, v in pa.params.items()}
        return pa

    pb = gen("B", "hard", 20261103)
    pc = gen("N", "soft", 20261104, t_force=1.2e6)
    for lab, pa in ((PATH_LABELS[1], pb), (PATH_LABELS[2], pc)):
        out[lab] = (pa, dict(params={k: (None if isinstance(v, float) and math.isnan(v) else v)
                                     for k, v in pa.params.items() if not isinstance(v, np.ndarray)},
                             p_now=float(pa.p_now)))
    return out


_PATHS = None


def _work(task):
    label, c, rep, seed = task
    import copy

    from illex.abc.simulate import simulate_stats
    path = copy.copy(_PATHS[label][0])
    path.params = dict(path.params, gamma_eff=(float(c) if c > 0 else math.nan))
    t0 = time.time()
    st = simulate_stats(path, seed, engine="msinv")
    return dict(path=label, c=c, rep=rep, seed=seed, pi_ratio=st["pi_ratio"], dxy_pi=st["dxy_pi"],
                wall_s=round(time.time() - t0, 2),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


def _init(paths):
    global _PATHS
    _PATHS = paths


def run(workers, reps, reps_hi):
    import multiprocessing as mp
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    paths = build_paths()
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV) as fh:
            done = {(r["path"], float(r["c"]), int(r["rep"])) for r in csv.DictReader(fh)}
    tasks = []
    for pi, lab in enumerate(PATH_LABELS):
        for c in [0.0] + C_GRID:
            n = reps_hi if c >= 1e-5 else reps
            for r in range(n):
                if (lab, c, r) not in done:
                    tasks.append((lab, c, r, 5_000_000 + pi * 100_000 + r))
    tasks.sort(key=lambda t: -t[1])                           # expensive (large c) first
    new = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init, initargs=(paths,), maxtasksperchild=1) as pool:
            for k, row in enumerate(pool.imap_unordered(_work, tasks)):
                w.writerow(row)
                fh.flush()
                if k % 25 == 0:
                    print(f"{k + 1}/{len(tasks)}", flush=True)
    return paths


def summarise(paths, se, se_src):
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    lines = [f"observed SE of dxy_pi used: {se:.5f} ({se_src})", ""]
    res = {}
    for lab in PATH_LABELS:
        d = df[df.path == lab]
        lines.append(f"== path {lab}: {json.dumps(paths[lab][1], default=str)}")
        lines.append(f"{'c':>10} {'n':>3} {'dxy_pi mean':>12} {'+-SE':>8} {'pi_ratio mean':>14} {'+-SE':>8} "
                     f"{'wall_s':>7} {'rss_mb':>7}")
        by = {}
        for c, g in d.groupby("c"):
            n = len(g)
            lines.append(f"{c:10.3g} {n:3d} {g.dxy_pi.mean():12.4f} {g.dxy_pi.std(ddof=1) / n ** .5:8.4f} "
                         f"{g.pi_ratio.mean():14.4f} {g.pi_ratio.std(ddof=1) / n ** .5:8.4f} "
                         f"{g.wall_s.mean():7.1f} {g.peak_rss_mb.max():7.0f}")
            by[c] = g.dxy_pi.values
        cs = sorted(c for c in by if c > 0)
        cstar = c_star(cs, [by[c] for c in cs], by[0.0], se)
        res[lab] = cstar
        lines += [f"c* = {cstar}", ""]
    gm = gamma_max(res[PATH_LABELS[0]])
    lines += [f"c* per path: {res}", f"GAMMA_MAX = min(100 c*(a), {GAMMA_CAP:g}) = {gm:g}"]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--reps-hi", type=int, default=32, help="reps at c = 1e-5 (the slow point)")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    paths = build_paths() if a.summary_only else run(a.workers, a.reps, a.reps_hi)
    obs = json.load(open(REPO / "results/illex/abc_observed.json"))
    summarise(paths, float(obs["se"]["dxy_pi"]), "results/illex/abc_observed.json se.dxy_pi")


if __name__ == "__main__":
    main()
