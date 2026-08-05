#!/usr/bin/env python
"""Run N ABC replicates: draw from the prior, invoke SLiM, summarize, append.

One process = one SLURM array task. Each replicate is independent, so a crash
loses only that replicate; failures are recorded with a reason rather than
dropped, because a systematically-failing region of parameter space is itself
information (e.g. p_start so low the inversion is never retained).

Writes one TSV per task; concatenate afterwards. Never appends to a shared file,
which would race across array tasks.

Usage (see submit_talapas.sbatch):
  python -m illex.slim.run_one --task-id 7 --reps 4 --out-dir results/abc \\
      --slim $(which slim) --scratch $TMPDIR
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C
from .summarize import summarize

SLIM_SCRIPT = Path(__file__).resolve().parent / "inversion_abc.slim"


def slim_command(slim_bin: str, params: dict, q: float, trees_path: Path,
                 seed: int) -> list[str]:
    """Build the SLiM 5.2 invocation.

    Follows the 14_sweep_seqmodel harness conventions: the seed goes to `slim -s`
    (NOT a -d define), floats are passed with repr() for full precision, and
    string paths are Eidos-quoted.
    """
    save_path = Path(str(trees_path) + ".restart")
    d = {
        "Q": q,
        "T_INV": params["t_inv"],
        "P_START": params["p_start"],
        "SEL": params["s"],
        "DOM": params["h"],
        "P_FLUX": params["p_flux"],
        "TRACT_FRAC": C.TRACT_FRAC,
        "INV_LEN": C.INV_LEN_SIM,
        "FLANK_LEN": C.FLANK_LEN_SIM,
        "R": C.REC_RATE,
        "MU": C.MU,
        "NREF": C.N_ANC,
        "N0": C.N_NOW,
        "TGROW": C.T_GROW,
    }
    cmd = [slim_bin, "-s", str(seed)]
    for k, v in d.items():
        if isinstance(v, int):
            cmd += ["-d", f"{k}={v}"]
        else:
            cmd += ["-d", f"{k}={v!r}"]          # full-precision float
    cmd += ["-d", f'OUTPATH="{trees_path}"']
    cmd += ["-d", f'SAVEPATH="{save_path}"']
    cmd.append(str(SLIM_SCRIPT))
    return cmd


def parse_slim_result(stdout: str) -> dict:
    """Pull the machine-readable INVERSION_RESULT line the recipe emits."""
    for line in stdout.splitlines():
        if "INVERSION_RESULT" in line:
            out = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
            return out
    return {}


def run_replicate(rep_seed: int, q: float, slim_bin: str, scratch: Path,
                  timeout_s: int) -> dict:
    rng = np.random.default_rng(rep_seed)
    params = C.draw_params(rng)
    trees = scratch / f"sim_{rep_seed}.trees"
    row = {"seed": rep_seed, "Q": q, **params,
           "status": "", "slim_wall_s": "", "n_restarts": "", "n_trees": ""}
    for name in C.STAT_NAMES + C.STAT_NAMES_ABSOLUTE:
        row[name] = ""

    t0 = time.time()
    try:
        proc = subprocess.run(
            slim_command(slim_bin, params, q, trees, rep_seed),
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["slim_wall_s"] = round(time.time() - t0, 1)
        return row
    row["slim_wall_s"] = round(time.time() - t0, 1)

    if proc.returncode != 0:
        row["status"] = "slim_error"
        row["stderr_tail"] = proc.stderr.strip()[-300:]
        return row

    res = parse_slim_result(proc.stdout)
    slim_status = res.get("status", "")
    if "nrestart" in res:
        row["n_restarts"] = res["nrestart"]
    if slim_status == "ABORT_RESTARTS":
        # Not a failure: a measurement. These parameter combinations cannot keep
        # the inversion segregating to the present, which is itself evidence
        # about the neutrality question. collect.sh reports the counts.
        row["status"] = "abort_restarts"
        return row
    if slim_status == "LOST_AT_END" or not trees.exists():
        row["status"] = "lost"
        return row
    if slim_status != "SEGREGATING":
        row["status"] = f"unexpected_slim_status:{slim_status or 'none'}"
        return row

    try:
        stats = summarize(str(trees), q, rep_seed)
    except Exception as exc:                          # noqa: BLE001
        row["status"] = f"summarize_error:{type(exc).__name__}"
        row["stderr_tail"] = str(exc)[:300]
        return row
    finally:
        for p in (trees, Path(str(trees) + ".restart")):
            if p.exists():
                p.unlink()

    # DictWriter is created with extrasaction="ignore", so any extra key in
    # stats is dropped rather than raising.
    row.update(stats)
    row["status"] = "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID; seeds are derived from it so the "
                         "whole sweep is reproducible")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slim", default=shutil.which("slim") or "slim")
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("TMPDIR", "./.tmp")))
    ap.add_argument("--Q", type=float, default=C.Q_DEFAULT)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--seed-base", type=int, default=1_000_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"sims_task{args.task_id:06d}.tsv"

    fields = (["seed", "Q"] + C.PARAM_NAMES + C.STAT_NAMES
              + C.STAT_NAMES_ABSOLUTE
              + ["status", "slim_wall_s", "n_restarts", "n_trees",
                 "stderr_tail"])

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in range(args.reps):
            seed = args.seed_base + args.task_id * 10_000 + r
            row = run_replicate(seed, args.Q, args.slim, args.scratch,
                                args.timeout)
            w.writerow(row)
            fh.flush()
            print(f"[task {args.task_id} rep {r}] status={row['status']} "
                  f"wall={row['slim_wall_s']}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
