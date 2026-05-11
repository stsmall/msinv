"""Smoke tests for the pilot bench harness at SCALED-DOWN L.

The full bench (L=10 Mb on v12) is too slow for unit tests; we test
the harness mechanics here and run the real bench manually in Task 3.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.pilot.bench_msinv import run_pilot_rep


def test_smoke_run_creates_outputs(tmp_path):
    """Run the bench at L=50 kb and verify it produces stats + timing."""
    out_dir = tmp_path / "rep_000"
    result = run_pilot_rep(
        out_dir=out_dir,
        rep=0,
        L=50_000,
        seed=12345,
    )
    assert (out_dir / "stats.npz").exists()
    assert (out_dir / "timing.json").exists()
    timing = json.loads((out_dir / "timing.json").read_text())
    assert "wall_seconds" in timing
    assert "peak_rss_bytes" in timing
    assert "iters_consumed" in timing
    assert "num_trees" in timing
    assert "num_sites" in timing
    assert timing["wall_seconds"] > 0
    assert result["wall_seconds"] == timing["wall_seconds"]


def test_smoke_stats_has_expected_keys(tmp_path):
    out_dir = tmp_path / "rep_000"
    run_pilot_rep(
        out_dir=out_dir, rep=0, L=50_000, seed=12345,
    )
    z = np.load(out_dir / "stats.npz", allow_pickle=False)
    keys = set(z.files)
    # Spot-check a few stats from each module are present.
    # F-only sampling: subgroups are F_S and F_I (from p_inv_F=0.73).
    assert "pi__F_S" in keys
    assert "pi__F_I" in keys
    assert "dxy__F_I_F_S" in keys or "dxy__F_S_F_I" in keys
    assert "fst__F_I_F_S" in keys or "fst__F_S_F_I" in keys
    assert any(k.startswith("tajimas_d__") for k in keys)
    assert any(k.startswith("tree_") for k in keys)
    assert any(k.startswith("ld_") for k in keys)
