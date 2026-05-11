"""Per-rep .npz persistence + cross-rep aggregation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REP_RE = re.compile(r"^rep_(\d+)$")


def save_rep_stats(path: str | Path, **stats: np.ndarray | float) -> None:
    """Save a dict of per-rep stats to `.npz`. Creates parent directories.

    Keys may be hierarchical (use `__` as separator, e.g. `pi__A`).
    Scalar values are stored as 0-d arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in stats.items()})


def load_rep_stats(path: str | Path) -> dict[str, np.ndarray]:
    """Load a per-rep `.npz` into a dict of numpy arrays."""
    z = np.load(Path(path), allow_pickle=False)
    return {k: z[k] for k in z.files}


def aggregate_track(track_dir: str | Path) -> dict[str, np.ndarray]:
    """Stack per-rep stats from `track_dir/rep_NNN/stats.npz` into
    arrays of shape (n_reps, ...).

    Reps are discovered by scanning subdirs matching `rep_NNN`. Missing
    reps are skipped (their indices recorded under `__rep_indices__`).
    """
    track_dir = Path(track_dir)
    rep_pairs = []
    for sub in sorted(track_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = REP_RE.match(sub.name)
        if not m:
            continue
        npz = sub / "stats.npz"
        if not npz.exists():
            continue
        rep_pairs.append((int(m.group(1)), load_rep_stats(npz)))
    if not rep_pairs:
        return {"__rep_indices__": np.array([], dtype=np.int64)}
    indices = np.array([r for r, _ in rep_pairs], dtype=np.int64)
    keys = set()
    for _, d in rep_pairs:
        keys.update(d.keys())
    out: dict[str, np.ndarray] = {"__rep_indices__": indices}
    for k in keys:
        rows = [d[k] for _, d in rep_pairs if k in d]
        if len(rows) == len(rep_pairs):
            out[k] = np.stack(rows)
    return out
