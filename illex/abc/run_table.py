"""Reference-table driver: one task = (model, flux, class, task_id) -> K accepted rows.

    python -m illex.abc.run_table --model D1 --flux 0 --class soft --n-accept 250 --task-id 0 --out DIR [--workers W]

Output ``DIR/{M}_{flux}_{class}_{i}.jsonl.gz`` (one row per accepted path) and ``..._counts.json`` (Counts per
class plus run metadata: git HEAD, msinv / msprime versions). Both are written to temp files and ``os.replace``d only
after every row is done, so a crashed run leaves nothing partial.

Each row: all prior KEYS plus model / flux / cls / task_id, the path summary (p_max, p_now, path_len), STAT_KEYS,
p_obs, n_nan, the engine used, the simulate_stats seed and wall_s.

Seeding is deterministic from (model, flux, class, task_id): SeedSequence([...]) spawns three streams, one each for
the prior draws, the path simulation and the per-row simulate_stats seeds (row j, j = index of the accepted path in
draw order). Batch sizes are a deterministic function of the counts so far, so the table does not depend on W.
Parallelism (``--workers W``): a spawn process pool over accepted paths, thread caps 1 in each worker.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path as FsPath

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is imported (also inherited by spawned workers)

import numpy as np  # noqa: E402

from illex.abc.simulate import simulate_stats  # noqa: E402
from illex.abc.statvec import STAT_KEYS  # noqa: E402
from illex.abc.trajectory import KEYS, MODELS, Counts, draw_prior, simulate_paths  # noqa: E402

CLASSES = ("hard", "soft")
SEED_TAG = 20261103
MAX_BATCH = 20000
MIN_BATCH = 500


def seed_streams(model, flux, cls, task_id):
    """(prior_rng, path_rng, rows_entropy): three distinct streams from (model, flux, class, task_id)."""
    ss = np.random.SeedSequence([SEED_TAG, MODELS.index(model), int(flux), CLASSES.index(cls), int(task_id)])
    a, b, c = ss.spawn(3)
    return np.random.default_rng(a), np.random.default_rng(b), c


def row_seed(rows_ss, j):
    child = np.random.SeedSequence(entropy=rows_ss.entropy, spawn_key=rows_ss.spawn_key + (int(j),))
    return int(child.generate_state(1)[0] % (2**31 - 1) + 1)


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _work(args):
    path, seed = args
    t0 = time.time()
    out = simulate_stats(path, seed, engine="auto")
    return out, time.time() - t0


def _engine_used(path):
    from illex.abc.simulate import engine_for
    return engine_for(path, "auto")


def _meta():
    import msprime
    try:
        import msinv
        mv = getattr(msinv, "__version__", "?")
    except Exception:
        mv = "unavailable"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=FsPath(__file__).parent, capture_output=True,
                              text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    return {"git_head": head, "msinv_version": mv, "msprime_version": msprime.__version__,
            "python": sys.version.split()[0], "numpy": np.__version__}


def _batch_size(k_left, drawn, accepted):
    if accepted == 0:
        return MAX_BATCH if drawn >= 2 * MIN_BATCH else MIN_BATCH * 4
    need = 2.0 * k_left * drawn / accepted
    return int(min(MAX_BATCH, max(MIN_BATCH, math.ceil(need))))


def _flat(path, model, flux, cls, task_id, seed, stats, wall, engine):
    row = {"model": model, "flux": int(flux), "cls": cls, "task_id": int(task_id)}
    row.update({k: path.params[k] for k in KEYS})
    row["hard"] = bool(row["hard"])
    row.update(p_max=float(np.max(path.freqs)), p_now=float(path.p_now), path_len=int(len(path.times)))
    row.update({k: stats[k] for k in STAT_KEYS})
    row.update(p_obs=stats["p_obs"], n_nan=int(stats["n_nan"]), engine=engine, seed=int(seed), wall_s=float(wall))
    return row


def run(model, flux, cls, n_accept, task_id, out, workers=1, max_drawn=20_000_000, verbose=True):
    flux = int(flux)
    out = FsPath(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{flux}_{cls}_{task_id}"
    prior_rng, path_rng, rows_ss = seed_streams(model, flux, cls, task_id)
    total = {"hard": Counts(), "soft": Counts()}
    accepted = []           # paths, in draw order
    t0 = time.time()
    while len(accepted) < n_accept:
        drawn = sum(c.drawn for c in total.values())
        if drawn >= max_drawn:
            break
        nb = min(_batch_size(n_accept - len(accepted), drawn, len(accepted)), max_drawn - drawn)
        prm = draw_prior(model, bool(flux), prior_rng, nb, cls)
        paths, cnt = simulate_paths(prm, path_rng)
        for c in total:
            for f, v in asdict(cnt[c]).items():
                setattr(total[c], f, getattr(total[c], f) + v)
        accepted.extend(paths)
        if verbose:
            print(f"[{stem}] drawn={drawn + nb} accepted={len(accepted)} t={time.time() - t0:.0f}s", flush=True)
    accepted = accepted[:n_accept]
    # Counts.accepted is the number of paths that passed the p_now filter, including any truncated here; keep the
    # drawn/accepted ratio untouched (truncation only matters at the last batch, and is recorded in n_rows).
    jobs = [(p, row_seed(rows_ss, j)) for j, p in enumerate(accepted)]
    engines = [_engine_used(p) for p in accepted]
    tmp = out / f".{stem}.jsonl.gz.tmp{os.getpid()}"
    tmpc = out / f".{stem}_counts.json.tmp{os.getpid()}"
    try:
        with gzip.open(tmp, "wt") as f:
            if workers > 1 and jobs:
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                         mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
                    results = ex.map(_work, jobs, chunksize=1)
                    for (p, sd), eng, (st, w) in zip(jobs, engines, results):
                        f.write(json.dumps(_flat(p, model, flux, cls, task_id, sd, st, w, eng)) + "\n")
            else:
                for (p, sd), eng in zip(jobs, engines):
                    t1 = time.time()
                    st = simulate_stats(p, sd, engine="auto")
                    f.write(json.dumps(_flat(p, model, flux, cls, task_id, sd, st, time.time() - t1, eng)) + "\n")
        meta = _meta()
        meta.update(model=model, flux=flux, cls=cls, task_id=int(task_id), n_accept_requested=int(n_accept),
                    n_rows=len(accepted), max_drawn=int(max_drawn), workers=int(workers),
                    elapsed_s=time.time() - t0)
        tmpc.write_text(json.dumps({"counts": {c: asdict(v) for c, v in total.items()}, "meta": meta}, indent=1))
        os.replace(tmp, out / f"{stem}.jsonl.gz")
        os.replace(tmpc, out / f"{stem}_counts.json")
    except BaseException:
        for t in (tmp, tmpc):
            try:
                t.unlink()
            except FileNotFoundError:
                pass
        raise
    return len(accepted), total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--flux", required=True, type=int, choices=(0, 1))
    ap.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    ap.add_argument("--n-accept", type=int, required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-drawn", type=int, default=20_000_000, help="cap on prior draws; keep rows accepted so far")
    a = ap.parse_args(argv)
    n, tot = run(a.model, a.flux, a.cls, a.n_accept, a.task_id, a.out, a.workers, a.max_drawn)
    print(f"done {a.model} flux={a.flux} {a.cls} task={a.task_id}: {n} rows; counts={ {c: asdict(v) for c, v in tot.items()} }")


if __name__ == "__main__":
    main()
