#!/usr/bin/env python
"""Does selection with drift reproduce the fitted statistics, or only the deterministic curve?

    .venv/bin/python -m illex.scripts.stochastic_check --reps 96 --workers 24

The grid fit conditions msinv on a deterministic trajectory that sits at p_hist from the moment the barrier
forms, then falls to p_now along dp/dt = (s/p*) p(1-p)(p* - p) with p* = p_now, s solved from the fall
duration. Here the same selection is run as a Wright-Fisher process with the fitted demography N(t), and
each realised trajectory is fed to msinv. Three versions at the best-fitting cell, each with its own
replicates:

  deterministic  the fitted curve (as in the grid)
  drift          Wright-Fisher from p_hist at t_inv, overdominance toward p_hist before the shift and toward
                 p_now after it, both at strength s
  rise           the same, but starting from a single copy at t_inv (conditioned on survival), so the
                 arrangement must first rise to p_hist under selection

Each version reports the mean and SD of the three fitted statistics and the diagonal chi2 of its mean
against the targets. If drift and rise give the same statistics as the deterministic curve, the grid fit's
shortcut is justified and s is a consistent mechanistic reading of the fitted trajectory.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from illex.scripts.param_bootstrap import SE, TARGETS, stats
from illex.scripts.sfs_shape import PROJ, project_unfolded

OUT = Path("results/illex")
STEP = 100            # generations between recorded trajectory points
GAMMA = 1e-15


def wf_trajectories(t_inv, t_decline, t_fall, p_hist, p_now, s, n, start, rng):
    """n realised Wright-Fisher trajectories (backward times, freqs) under the fitted selection and N(t)."""
    from illex.theory import N_growth

    t_shift = t_decline + t_fall
    if start == "rise":
        twoN = 2.0 * float(N_growth(t_inv))
        m = int(4 * n / (2 * s))                    # single-copy starts, survival probability ~2s
        p = np.full(m, 1.0 / twoN)
    else:
        p = np.full(n, p_hist)
    ids = np.arange(p.size)
    rec = []
    for g in range(int(t_inv), -1, -1):             # backward time, oldest first
        ps = p_hist if g > t_shift else p_now
        p = p + (s / ps) * p * (1.0 - p) * (ps - p)
        twoN = int(round(2.0 * float(N_growth(g))))
        p = rng.binomial(twoN, np.clip(p, 0.0, 1.0)) / twoN
        alive = p > 0
        if not alive.all():
            p, ids = p[alive], ids[alive]
        if p.size == 0:
            raise RuntimeError("every trajectory was lost")
        if g % STEP == 0:
            rec.append((g, ids, p.copy()))
    keep = ids[:n]                                  # ids stay sorted, so the first n survivors
    times = np.array([g for g, _, _ in rec][::-1], float)
    out = []
    for i in keep:
        freqs = np.array([v[np.searchsorted(k, i)] for _, k, v in rec][::-1], float)
        out.append((times, np.clip(freqs, 1e-9, 1 - 1e-9)))
    return out


def _sim(job):
    from msinv import HullSimulator, InversionSpec

    from illex import model as M
    from illex import stats as S
    from illex.demography import PRESENT_NE_GROWTH, growth_demography
    from illex.model import MARGIN_FRACTION, TRACT_FRACTION
    from illex.slim.config import REC_RATE
    from illex.theory import N_growth

    times, freqs, t_inv, seed = job
    L = int(round(2000.0 / (4.0 * PRESENT_NE_GROWTH * REC_RATE)))
    margin = L * MARGIN_FRACTION
    spec = InversionSpec(
        bp_left=margin, bp_right=L - margin, gene_conversion_rate=GAMMA,
        mean_tract_length=max(1.0, (L - 2 * margin) * TRACT_FRACTION), tract_distribution="geometric",
        trajectory={"type": "precomputed", "times": list(times), "freqs": [list(freqs)],
                    "n_e": [float(N_growth(t_inv))], "t_inv": [float(t_inv)]})
    sim = HullSimulator(n_std=100, n_inv=100, population_size=PRESENT_NE_GROWTH, demography=growth_demography(),
                        sequence_length=L, recombination_rate=REC_RATE, inversions=[spec], seed=seed)
    ts = sim.simulate()
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    left, right = M.inversion_interval(sim)
    st = S.arrangement_stats(ts, i_nodes, s_nodes, interval=(left, right))
    spec_out = []
    for nodes in (i_nodes, s_nodes):
        n = len(nodes)
        af = ts.allele_frequency_spectrum(sample_sets=[list(nodes)], windows=[0.0, left, right, ts.sequence_length],
                                          mode="branch", polarised=True, span_normalise=True)[1]
        v = np.asarray(af, float)
        spec_out.append(project_unfolded(v / max(v[1:n].sum(), 1e-300), n, PROJ))
    return st["pi_i_over_pi_s"], st["dxy_over_pi_i"], spec_out[0], spec_out[1], float(freqs.min())


def main() -> None:
    from illex.balancing import decline_curve

    ap = argparse.ArgumentParser()
    ap.add_argument("--t-inv", type=float, default=850_000.0)
    ap.add_argument("--p-hist", type=float, default=0.74)
    ap.add_argument("--t-decline", type=float, default=175_000.0)
    ap.add_argument("--t-fall", type=float, default=100_000.0)
    ap.add_argument("--p-now", type=float, default=0.374)
    ap.add_argument("--reps", type=int, default=96)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--versions", default="drift,rise", help="stochastic versions to run besides deterministic")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    dt, df, s = decline_curve(a.t_inv, a.t_decline, a.t_fall, a.p_hist, a.p_now)
    trajs = {"deterministic": [(np.asarray(dt), np.asarray(df))] * a.reps}
    for start in a.versions.split(","):
        t0 = time.time()
        trajs[start] = wf_trajectories(a.t_inv, a.t_decline, a.t_fall, a.p_hist, a.p_now, s, a.reps, start, rng)
        print(f"{start}: {len(trajs[start])} trajectories in {time.time() - t0:.0f} s", flush=True)

    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    res = {"params": vars(a), "s": s, "targets": TARGETS.tolist(), "versions": {}}
    emit(f"Best cell t_inv={a.t_inv:g} p_hist={a.p_hist} t_decline={a.t_decline:g} t_fall={a.t_fall:g}, "
         f"s = {s:.3g}")
    emit(f"targets {TARGETS.tolist()}  SE {SE.tolist()}")
    for k, (name, tr) in enumerate(trajs.items()):
        jobs = [(t, f, a.t_inv, a.seed + 100_000 * (k + 1) + i) for i, (t, f) in enumerate(tr)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_sim, jobs, chunksize=1))
        r = np.array([x[0] for x in rows])
        d = np.array([x[1] for x in rows])
        c = (r, d, np.array([x[2] for x in rows]), np.array([x[3] for x in rows]))
        n = len(r)
        m = stats(np.full(n, 1 / n), c)
        per = np.array([stats(np.eye(n)[i], c) for i in range(n)])
        chi2 = float((((m - TARGETS) / SE) ** 2).sum())
        traj_f = np.array([np.interp([a.t_decline + a.t_fall, a.t_inv / 2], t, f) for t, f in tr])
        emit(f"{name:>13s}: pi_I/pi_S {m[0]:.4f} (sd {per[:, 0].std():.3f})  dxy/pi_I {m[1]:.4f} "
             f"(sd {per[:, 1].std():.3f})  singleton ratio {m[2]:.4f}  chi2 {chi2:.2f}  "
             f"freq at fall start {traj_f[:, 0].mean():.3f} (sd {traj_f[:, 0].std():.3f})")
        res["versions"][name] = {"n": n, "mean": m.tolist(), "sd": per.std(0).tolist(), "chi2": chi2,
                                 "freq_at_fall_start_mean": float(traj_f[:, 0].mean()),
                                 "freq_at_fall_start_sd": float(traj_f[:, 0].std()),
                                 "per_rep": per.tolist()}
    (OUT / f"stochastic_check{a.tag}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"stochastic_check{a.tag}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
