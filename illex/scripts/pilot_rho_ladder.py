#!/usr/bin/env python
"""Stage 0: dual rho ladder x dual origin-path pilot.

Establishes the largest affordable L per (arm, path) before any production
grid runs.

rho/bp = 4*Ne_present*r, so the arms differ 8.8x:
  growth   (Ne 6,808,096): 0.0681   -> needs only 30-75 kb
  constant (Ne   775,000): 7.75e-3  -> needs >=300 kb for the LD panel

Two origin paths are benched at every rung, because ``build_inversion_sim``'s
``p_start`` argument selects between two dynamically different lineage
histories with different cost (see illex/model.py's module docstring):

  path="constant":   legacy p_start=None path. Constant p_inv from 0 to
                      t_inv=T_INV_PILOT, no forced monophyly at t_inv (the
                      "soft" limit). Kept at the brief's original t_inv so
                      these rows stay comparable to earlier pilot numbers.
  path="trajectory":  p_start set. Deterministic-logistic origin from a
                      founding frequency p_start at t_inv up to p_inv today.
                      Uses t_inv=T_INV_PROD / p_start=P_START_PROD, the
                      current production best-fit values -- this is the
                      path all production fitting will actually use, so
                      these rows are the ones that predict production cost.

A ladder that only benched the legacy path would give the wrong budget for
production, since the two paths have different lineage dynamics.

One rep per rung. Sequential, with an RSS watchdog: shared device.

NOTE (C1, flank dilution -- added after this ladder was run; NOT re-run,
see task-final-fixes-report.md): the committed
``results/illex/pilot_rho_ladder.csv``'s ``pi_i_over_pi_s``/
``dxy_over_pi_i`` columns are computed with ``interval=None``, i.e. over
the WHOLE simulated sequence including the collinear
``model.MARGIN_FRACTION`` flank, not restricted to the inversion body. They
are therefore diluted toward the panmictic null relative to the
interval-restricted values used elsewhere (e.g.
``tests/illex/test_floor_harness.py``'s anchor) and must not be compared
directly to those or to the empirical ratios in ``illex.empirical``. The
ladder's cost/RSS numbers (wall time, peak RSS, num_trees) are unaffected
and remain valid -- only the two ratio columns are diluted. Do not re-run
this ladder solely to fix the ratio columns; it costs real GPU-box time and
the cost/RSS numbers are what downstream tasks actually depend on.
"""
from __future__ import annotations

import argparse
import csv
import resource
import time
from pathlib import Path

from illex import model, stats
from illex.demography import PRESENT_NE_CONST, PRESENT_NE_GROWTH

RHO_RUNGS = [200, 500, 1000, 2000, 5000]
RSS_LIMIT_GB = 60.0          # abort a rung above this; well under the 400 GB cap
T_INV_PILOT = 9.5e5          # legacy/"constant" path; near both arms' fitted t_inv
T_INV_PROD = 5.0e5           # "trajectory" path; current production best-fit t_inv
P_START_PROD = 0.15          # "trajectory" path; current production best-fit p_start
PATHS = ["constant", "trajectory"]
OUT = Path("results/illex/pilot_rho_ladder.csv")
FIELDNAMES = [
    "arm", "path", "t_inv", "p_start", "rho_target", "seq_length",
    "wall_s", "peak_rss_gb", "num_trees", "pi_i_over_pi_s", "dxy_over_pi_i",
    "status",
]


def seq_length_for(rho: float, present_ne: float, r: float = 2.5e-9) -> int:
    return int(round(rho / (4.0 * present_ne * r)))


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def run_rung(arm: str, path: str, rho: float, present_ne: float) -> dict:
    L = seq_length_for(rho, present_ne)
    if path == "constant":
        t_inv, p_start = T_INV_PILOT, None
    else:
        t_inv, p_start = T_INV_PROD, P_START_PROD

    row = {
        "arm": arm, "path": path, "t_inv": t_inv, "p_start": p_start,
        "rho_target": rho, "seq_length": L,
        "wall_s": "", "peak_rss_gb": "", "num_trees": "",
        "pi_i_over_pi_s": "", "dxy_over_pi_i": "", "status": "",
    }
    if L < 1000:
        row["status"] = "skipped_too_short"
        return row

    t0 = time.time()
    try:
        sim = model.build_inversion_sim(
            arm=arm, seq_length=L, t_inv=t_inv, gamma=1e-9,
            p_start=p_start, seed=7,
        )
        ts = sim.simulate()
    except Exception as exc:                      # noqa: BLE001 - record and continue
        row["wall_s"] = round(time.time() - t0, 1)
        row["peak_rss_gb"] = round(peak_rss_gb(), 2)
        row["status"] = f"error:{type(exc).__name__}"
        return row

    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    # interval=None preserves this script's original (whole-sequence,
    # flank-diluted) behaviour -- see the module docstring's C1 note. Do
    # not change this to an interval without re-running the ladder.
    st = stats.arrangement_stats(ts, i_nodes, s_nodes, interval=None)
    row.update(
        wall_s=round(time.time() - t0, 1),
        peak_rss_gb=round(peak_rss_gb(), 2),
        num_trees=ts.num_trees,
        pi_i_over_pi_s=round(st["pi_i_over_pi_s"], 4),
        dxy_over_pi_i=round(st["dxy_over_pi_i"], 4),
        status="ok",
    )
    if row["peak_rss_gb"] > RSS_LIMIT_GB:
        row["status"] = "ok_over_rss_limit"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["growth", "constant"])
    ap.add_argument("--paths", nargs="+", default=PATHS,
                     choices=PATHS)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms:
        present_ne = PRESENT_NE_GROWTH if arm == "growth" else PRESENT_NE_CONST
        for path in args.paths:
            for rho in RHO_RUNGS:
                row = run_rung(arm, path, rho, present_ne)
                rows.append(row)
                print(f"[{arm}/{path}] rho={rho:>5} L={row['seq_length']:>8,} "
                      f"wall={row['wall_s']}s rss={row['peak_rss_gb']}GB "
                      f"status={row['status']}", flush=True)
                if row["status"].startswith(("error", "ok_over_rss_limit")):
                    print(f"[{arm}/{path}] stopping ladder at rho={rho}",
                          flush=True)
                    break

    # Append mode: this script is invoked once per (arm, path) combo (see
    # task-5-report.md) so that each combo's peak-RSS watchdog runs in its
    # own fresh process, uncontaminated by an earlier combo's cumulative
    # ru_maxrss. Delete a stale results/illex/pilot_rho_ladder.csv before a
    # fresh full run, or rows will accumulate across invocations.
    write_header = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
