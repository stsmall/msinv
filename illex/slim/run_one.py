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

Fixed-parameter mode (Task 9, the SLiM validation of the trajectory-ABC simulator): ``--params`` is a JSON
object (inline, or a file) with the trajectory-ABC keys of one parameter set (model, hard, t_inv, p0, t_sel, s,
t_stop, p_star, gamma_eff, Q), or a sets file {"sets": {name: {...}}} with ``--set name``. Every replicate runs
those parameters with the ABC models of ``inversion_abc.slim`` (BODY_BP body, BND_MARGIN_BP margins) and writes
the ``illex.abc.statvec.STAT_KEYS`` + ``BND_KEYS`` row of ``summarize.summarize_abc``, plus the SLiM wall time
and peak RSS:
  python -m illex.slim.run_one --params results/illex/abc_slim_sets.json --set soft_D1 \\
      --task-id 0 --reps 5 --out-dir results/illex/abc_slim_validation_runs --slim $(which slim)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
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


# ------------------------------------------------------------------ fixed-parameter mode (Task 9) ----
ABC_PARAM_KEYS = ("model", "hard", "t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff", "Q")
Q_MAX = 200.0          # Task 9 design: Q <= 200 and s * Q < 0.1
SQ_MAX = 0.1
SLIM_FALLBACK = Path.home() / "miniforge3/envs/slim_sim/bin/slim"


def default_slim() -> str:
    """$SLIM_BIN, else ``slim`` on PATH, else the box's slim_sim conda env binary."""
    return (os.environ.get("SLIM_BIN") or shutil.which("slim")
            or (str(SLIM_FALLBACK) if SLIM_FALLBACK.exists() else "slim"))


def load_params(spec: str, set_name: str | None = None) -> dict:
    """One parameter set from ``--params`` (inline JSON or a file; a {"sets": ...} file needs ``set_name``)."""
    text = Path(spec).read_text() if Path(spec).exists() else spec
    obj = json.loads(text)
    if "sets" in obj:
        if set_name is None:
            raise ValueError(f"{spec} holds several sets ({', '.join(obj['sets'])}); pass --set")
        from illex.abc.slim_vs_msinv import set_index

        prm = dict(obj["sets"][set_name])
        prm.setdefault("name", set_name)
        prm["set_index"] = set_index(obj["sets"], set_name)
    else:
        prm = dict(obj)
    return check_params(prm)


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def check_params(prm: dict, q_max: float = Q_MAX) -> dict:
    """Normalised parameter set (NaN for unused keys), validated against the model and the Q-scaling limits.

    ``q_max`` exists for the smoke tests only (a tiny Q = 2000 run); the validation design is Q <= Q_MAX.
    """
    from illex.abc.trajectory import MODELS

    missing = [k for k in ("model", "hard", "t_inv", "Q") if k not in prm]
    if missing:
        raise ValueError(f"parameter set lacks {missing}")
    out = dict(prm)
    if out["model"] not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {out['model']!r}")
    out["hard"] = bool(out["hard"])
    for k in ("t_inv", "p0", "t_sel", "s", "t_stop", "p_star", "gamma_eff", "Q"):
        out[k] = _num(out.get(k))
    q, s = out["Q"], out["s"]
    if not (1.0 <= q <= q_max):
        raise ValueError(f"Q = {q} outside [1, {q_max}]")
    if out["model"] != "N" and not (math.isfinite(s) and s * q < SQ_MAX):
        raise ValueError(f"s * Q = {s * q} must be < {SQ_MAX} (Q-scaling validity)")
    if not out["hard"] and not (math.isfinite(out["p0"]) and 0.0 < out["p0"] < 1.0):
        raise ValueError("a soft start needs p0 in (0, 1)")
    if out["model"] == "B" and not (0.0 < out["p_star"] < 1.0):
        raise ValueError("model B needs p_star in (0, 1)")
    return out


def slim_command_abc(slim_bin: str, prm: dict, trees_path: Path, seed: int) -> list[str]:
    """The ``inversion_abc.slim`` invocation of one trajectory-ABC parameter set (rates and sizes of the ABC)."""
    from illex.abc.constants import BND_MARGIN_BP, BODY_BP, MU, REC_RATE, TRACT_BP
    from illex.theory import N0, N_ANC, T_GROW

    hard = prm["hard"]
    t_inv = prm["t_inv"]
    t_sel = t_inv if (hard or not math.isfinite(prm["t_sel"])) else prm["t_sel"]
    sel = prm["s"] if prm["model"] != "N" else 0.0
    c = prm["gamma_eff"] if math.isfinite(prm["gamma_eff"]) else 0.0
    d = {
        "Q": prm["Q"], "T_INV": t_inv, "P_START": prm["p0"] if not hard else 0.0,
        "SEL": sel, "T_SEL": t_sel,
        "T_STOP": prm["t_stop"] if prm["model"] == "D2" else 0.0,
        "P_STAR": prm["p_star"] if prm["model"] == "B" else 0.5,
        "GC_C": c, "GC_TRACT": float(TRACT_BP),
        "INV_LEN": int(BODY_BP), "FLANK_LEN": int(BND_MARGIN_BP),
        "R": REC_RATE, "MU": MU, "NREF": N_ANC, "N0": N0, "TGROW": T_GROW,
    }
    cmd = [slim_bin, "-s", str(seed)]
    for k, v in d.items():
        cmd += ["-d", f"{k}={v}" if isinstance(v, int) else f"{k}={float(v)!r}"]
    cmd += ["-d", f'MODEL="{prm["model"]}"', "-d", f"HARD={'T' if hard else 'F'}"]
    cmd += ["-d", f'OUTPATH="{trees_path}"', "-d", f'SAVEPATH="{trees_path}.restart"']
    cmd.append(str(SLIM_SCRIPT))
    return cmd


def run_measured(cmd: list[str], timeout_s: float):
    """(returncode, stdout, stderr, wall_s, peak_rss_mb, timed_out) of one child process.

    Peak RSS comes from /usr/bin/time (%M), which forks from a small process. A child forked straight from this
    Python process would report max(its own peak, Python's RSS at the fork), since Linux carries the pre-exec
    high-water mark across exec; that is the fallback when /usr/bin/time is missing (an upper bound). The child
    runs in its own session, and a timeout kills the whole group.
    """
    import signal

    t0 = time.time()
    gnu_time = Path("/usr/bin/time")
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe, \
            tempfile.NamedTemporaryFile(suffix=".rss") as fr:
        full = ([str(gnu_time), "-f", "%M", "-o", fr.name, *cmd] if gnu_time.exists() else list(cmd))
        proc = subprocess.Popen(full, stdout=fo, stderr=fe, start_new_session=True)
        hit = threading.Event()

        def _kill():
            hit.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            _, status, ru = os.wait4(proc.pid, 0)
        finally:
            timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
        rss_kb = float(ru.ru_maxrss)
        if gnu_time.exists():
            try:
                rss_kb = float(Path(fr.name).read_text().split()[-1])
            except (ValueError, IndexError):
                pass
        fo.seek(0)
        fe.seek(0)
        out, err = fo.read().decode(errors="replace"), fe.read().decode(errors="replace")
    return proc.returncode, out, err, time.time() - t0, rss_kb / 1024.0, hit.is_set()


def abc_fields() -> list[str]:
    from illex.abc.statvec import BND_KEYS, STAT_KEYS

    return (["set", "rep", "seed", *ABC_PARAM_KEYS, "status", "slim_wall_s", "slim_maxrss_mb", "n_restarts",
             "summarize_wall_s", "py_maxrss_mb", "p_now", "p_obs", "n_nan", "n_i_pop", "n_s_pop"]
            + list(STAT_KEYS) + list(BND_KEYS) + ["stderr_tail"])


def run_replicate_abc(prm: dict, rep: int, seed: int, slim_bin: str, scratch: Path, timeout_s: float,
                      mask="data") -> dict:
    """One fixed-parameter SLiM replicate -> one row (``abc_fields``). Failures are recorded, never raised."""
    from .summarize import abc_seed_words, summarize_abc

    trees = scratch / f"abc_{prm.get('name', 'set')}_{seed}.trees"
    row = {"set": prm.get("name", ""), "rep": rep, "seed": seed, **{k: prm[k] for k in ABC_PARAM_KEYS}}
    cmd = slim_command_abc(slim_bin, prm, trees, abc_seed_words(seed)["slim"])
    try:
        rc, out, err, wall, rss, timed_out = run_measured(cmd, timeout_s)
        row.update(slim_wall_s=round(wall, 1), slim_maxrss_mb=round(rss, 1))
        res = parse_slim_result(out)
        row["n_restarts"] = res.get("nrestart", "")
        if timed_out:
            row["status"] = "timeout"
        elif rc != 0:
            row["status"] = "slim_error"
            row["stderr_tail"] = (err.strip() or out.strip())[-300:].replace("\n", " | ")
        elif res.get("status") == "ABORT_RESTARTS":
            row["status"] = "abort_restarts"
        elif res.get("status") != "SEGREGATING" or not trees.exists():
            row["status"] = f"unexpected_slim_status:{res.get('status', 'none')}"
        else:
            t0 = time.time()
            try:
                row.update(summarize_abc(str(trees), seed, mask=mask))
                row["status"] = "ok"
            except Exception as exc:                          # noqa: BLE001
                row["status"] = f"summarize_error:{type(exc).__name__}"
                row["stderr_tail"] = str(exc)[:300]
            row["summarize_wall_s"] = round(time.time() - t0, 1)
            row["py_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    finally:
        for p in (trees, Path(f"{trees}.restart")):
            if p.exists():
                p.unlink()
    return row


def main_abc(args) -> None:
    prm = load_params(args.params, args.set)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"slim_{prm.get('name', 'set')}_task{args.task_id:06d}.tsv"
    from illex.abc.slim_vs_msinv import SET_SEED_STRIDE

    set_offset = int(prm.get("set_index", 0)) * SET_SEED_STRIDE   # different sets never share seeds
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=abc_fields(), delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in range(args.reps):
            seed = args.seed_base + set_offset + args.task_id * 10_000 + r
            row = run_replicate_abc(prm, r, seed, args.slim, args.scratch, args.timeout,
                                    mask=None if args.no_mask else "data")
            w.writerow(row)
            fh.flush()
            print(f"[{prm.get('name', 'set')} task {args.task_id} rep {r}] status={row['status']} "
                  f"p_now={row.get('p_now', '')} restarts={row.get('n_restarts', '')} "
                  f"slim={row.get('slim_wall_s', '')}s rss={row.get('slim_maxrss_mb', '')}MB", flush=True)
    print(f"wrote {out}")


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
    ap.add_argument("--params", default=None,
                    help="fixed-parameter mode (Task 9): a JSON parameter set, inline or a file")
    ap.add_argument("--set", default=None, help="set name when --params is a {'sets': ...} file")
    ap.add_argument("--no-mask", action="store_true",
                    help="fixed mode: keep every flank site in the boundary keys (default: the data's mask)")
    args = ap.parse_args()
    if args.params is not None:
        if shutil.which(args.slim) is None:
            args.slim = default_slim()
        return main_abc(args)

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
