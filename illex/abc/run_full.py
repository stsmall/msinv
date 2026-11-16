"""Full-length NPE training rows for the balancing model with two flux processes (one task = K accepted rows).

    python -m illex.abc.run_full --task-id 0 --n-accept 16 --bank DIR --out DIR [--workers W] [--save-ts-every N]

Prior draws are soft-start balancing (B) with flux, ``trajectory.draw_prior("B", True, ..., "soft")``, filtered on
the present frequency as in ``run_table``; each accepted path gets the long-flux parameters and its control from
its own row seed (``fulllen.simulate_full``). Seeding is deterministic from the task id (FULL_TAG), so a task can be
rerun exactly.

Rows are appended to DIR/.B_full_{task}.jsonl.gz.partial as each finishes (a killed task keeps its finished rows)
and the file is renamed to B_full_{task}.jsonl.gz when all K rows are done, next to B_full_{task}_counts.json
(prior counts, git HEAD, msinv version and commit). ``--save-ts-every N`` keeps the mutated tree sequence of every
row whose global index (task * K + j) is a multiple of N, under DIR/ts/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path as FsPath

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from illex.abc.statvec import STAT_KEYS  # noqa: E402
from illex.abc.trajectory import KEYS, Counts, draw_prior, simulate_paths  # noqa: E402

FULL_TAG = 20261116
BATCH = 20000

_BANK = None


def _init(bank_dir):
    global _BANK
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    from illex.abc.control_bank import load_bank
    _BANK = load_bank(bank_dir, raw_only=True)


def _work(args):
    path, seed, ts_file = args
    from illex.abc.fulllen import simulate_full
    t0 = time.time()
    out = simulate_full(path, seed, _BANK, save_ts=ts_file)
    import resource
    out["maxrss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return path, seed, out, time.time() - t0


def _row(path, task_id, seed, st, wall):
    row = {"model": "B", "flux": 1, "cls": "soft", "task_id": int(task_id)}
    row.update({k: path.params[k] for k in KEYS})
    row["hard"] = bool(row["hard"])
    row.update(p_max=float(np.max(path.freqs)), p_now=float(path.p_now), path_len=int(len(path.times)))
    row.update({k: st[k] for k in STAT_KEYS})
    for k in ("p_obs", "n_nan", "c_long", "l_long", "ctl_idx", "ctl_seed", "raw_body", "x", "maxrss_gb"):
        row[k] = st[k]
    row.update(seed=int(seed), wall_s=float(wall))
    return row


def _meta():
    import msprime
    import msinv
    here = FsPath(__file__).parent

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=here, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return "unknown"

    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "msinv_version": getattr(msinv, "__version__", "?"), "msinv_file": msinv.__file__,
            "msprime_version": msprime.__version__, "python": sys.version.split()[0], "numpy": np.__version__}


def run(task_id, n_accept, bank_dir, out, workers=1, max_drawn=3_000_000, save_ts_every=0):
    out = FsPath(out)
    out.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence([FULL_TAG, int(task_id)])
    a, b, c = ss.spawn(3)
    prior_rng, path_rng = np.random.default_rng(a), np.random.default_rng(b)
    total = {"hard": Counts(), "soft": Counts()}
    accepted = []
    t0 = time.time()
    while len(accepted) < n_accept and sum(x.drawn for x in total.values()) < max_drawn:
        prm = draw_prior("B", True, prior_rng, BATCH, "soft")
        paths, cnt = simulate_paths(prm, path_rng, condition_on_establishment=True)
        for k in total:
            for f, v in asdict(cnt[k]).items():
                setattr(total[k], f, getattr(total[k], f) + v)
        accepted.extend(paths)
    accepted = accepted[:n_accept]
    print(f"[task {task_id}] {len(accepted)} paths accepted of {sum(x.drawn for x in total.values())} drawn "
          f"in {time.time() - t0:.0f}s", flush=True)
    jobs = []
    for j, p in enumerate(accepted):
        child = np.random.SeedSequence(entropy=c.entropy, spawn_key=c.spawn_key + (j,))
        seed = int(child.generate_state(1)[0] % (2**31 - 1) + 1)
        g = int(task_id) * int(n_accept) + j
        ts_file = None
        if save_ts_every and g % save_ts_every == 0:
            (out / "ts").mkdir(exist_ok=True)
            ts_file = str(out / "ts" / f"B_full_{task_id}_{j}_seed{seed}.trees")
        jobs.append((p, seed, ts_file))
    stem = f"B_full_{task_id}"
    part = out / f".{stem}.jsonl.gz.partial"
    n_done = 0
    import multiprocessing as mp
    with gzip.open(part, "wt") as fh, ProcessPoolExecutor(max(1, workers), initializer=_init, initargs=(bank_dir,),
                                                          mp_context=mp.get_context("spawn")) as ex:
        for p, seed, st, wall in ex.map(_work, jobs, chunksize=1):
            fh.write(json.dumps(_row(p, task_id, seed, st, wall)) + "\n")
            fh.flush()
            n_done += 1
            print(f"[task {task_id}] row {n_done}/{len(jobs)} seed {seed} {wall:.0f}s "
                  f"{st['maxrss_gb']:.1f} GB", flush=True)
    meta = _meta()
    meta.update(task_id=int(task_id), n_accept_requested=int(n_accept), n_rows=n_done, workers=int(workers),
                bank=str(bank_dir), elapsed_s=time.time() - t0)
    (out / f"{stem}_counts.json").write_text(json.dumps({"counts": {k: asdict(v) for k, v in total.items()},
                                                         "meta": meta}, indent=1))
    os.replace(part, out / f"{stem}.jsonl.gz")
    return n_done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n-accept", type=int, required=True)
    ap.add_argument("--bank", required=True, help="control bank directory (control_bank.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-drawn", type=int, default=3_000_000)
    ap.add_argument("--save-ts-every", type=int, default=0)
    a = ap.parse_args(argv)
    n = run(a.task_id, a.n_accept, a.bank, a.out, a.workers, a.max_drawn, a.save_ts_every)
    print(f"done task {a.task_id}: {n} rows")


if __name__ == "__main__":
    main()
