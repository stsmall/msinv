"""Bank of full-length neutral controls for ``fulllen.simulate_full`` (one task = K controls).

    python -m illex.abc.control_bank --task-id 0 --n 20 --out DIR [--workers W]

Writes DIR/ctl_{task}.jsonl.gz (one ``fulllen.control_entry`` per line: seed, raw statistics, extra material),
atomically. Entry seeds are SeedSequence([BANK_TAG, task, j]); ``load_bank`` reads every task file in seed order.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

BANK_TAG = 20261116


def entry_seed(task_id: int, j: int) -> int:
    ss = np.random.SeedSequence([BANK_TAG, int(task_id), int(j)])
    return int(ss.generate_state(1)[0] % (2**31 - 1) + 1)


def _one(seed):
    from illex.abc.fulllen import control_entry
    t = time.time()
    e = control_entry(seed)
    e["wall_s"] = time.time() - t
    return e


def load_bank(directory, raw_only: bool = False) -> list:
    """Every bank entry under ``directory`` (ctl_*.jsonl.gz), sorted by seed so the order is reproducible.

    ``raw_only`` keeps just the seed and the raw statistics (what ``fulllen.simulate_full`` needs)."""
    out = []
    for f in sorted(glob.glob(str(Path(directory) / "ctl_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                e = json.loads(line)
                out.append({"seed": e["seed"], "raw": e["raw"]} if raw_only else e)
    if not out:
        raise FileNotFoundError(f"no control bank entries in {directory}")
    out.sort(key=lambda e: e["seed"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [entry_seed(a.task_id, j) for j in range(a.n)]
    tmp = out / f".ctl_{a.task_id}.jsonl.gz.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as fh:
        if a.workers > 1:
            import multiprocessing as mp
            with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
                for e in ex.map(_one, seeds):
                    fh.write(json.dumps(e) + "\n")
                    print(f"control seed {e['seed']} {e['wall_s']:.0f}s", flush=True)
        else:
            for s in seeds:
                e = _one(s)
                fh.write(json.dumps(e) + "\n")
                print(f"control seed {e['seed']} {e['wall_s']:.0f}s", flush=True)
    os.replace(tmp, out / f"ctl_{a.task_id}.jsonl.gz")


if __name__ == "__main__":
    main()
