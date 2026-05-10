"""Smoke tests for the pilot bench harness at SCALED-DOWN params.

The full bench (L=5 Mb, Ne=1e6) is too slow for unit tests; we test
the harness mechanics here and run the real bench manually in Task 7.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from validation.pilot.bench_msinv import run_pilot_rep


def test_smoke_run_creates_outputs(tmp_path):
    """Run the bench at toy params and verify it produces stats + timing."""
    out_dir = tmp_path / "rep_000"
    result = run_pilot_rep(
        out_dir=out_dir,
        rep=0,
        L=10_000,
        Ne=1000,
        n_samples=10,
        inv_bp_left=2_500.0,
        inv_bp_right=7_500.0,
        t_inv=4_000.0,
        mu=1e-7,
        r=1e-7,
        gc_rate=1e-9,
        seed=12345,
    )
    assert (out_dir / "stats.npz").exists()
    assert (out_dir / "timing.json").exists()
    timing = json.loads((out_dir / "timing.json").read_text())
    assert "wall_seconds" in timing
    assert "peak_rss_bytes" in timing
    assert "iters_consumed" in timing
    assert timing["wall_seconds"] > 0
    assert result["wall_seconds"] == timing["wall_seconds"]


def test_smoke_stats_has_expected_keys(tmp_path):
    out_dir = tmp_path / "rep_000"
    run_pilot_rep(
        out_dir=out_dir, rep=0,
        L=10_000, Ne=1000, n_samples=10,
        inv_bp_left=2_500.0, inv_bp_right=7_500.0, t_inv=4_000.0,
        mu=1e-7, r=1e-7, gc_rate=1e-9, seed=12345,
    )
    z = np.load(out_dir / "stats.npz", allow_pickle=False)
    keys = set(z.files)
    # Spot-check a few stats from each module are present
    assert any(k.startswith("pi__") for k in keys)
    assert any(k.startswith("dxy__") for k in keys)
    assert any(k.startswith("fst__") for k in keys)
    assert any(k.startswith("tajimas_d__") for k in keys)
    assert any(k.startswith("tree_") for k in keys)
    assert any(k.startswith("ld_") for k in keys)
