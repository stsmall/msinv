"""Smoke test for Track 3 driver at scaled-down params."""
from pathlib import Path

import numpy as np
import pytest


def test_track3_smoke_3_reps(tmp_path):
    """3 reps × 2 engines at L=50 kb completes and produces aggregator output."""
    from validation.track3_msprime.run import run_track3

    out_root = tmp_path / "track3"
    result = run_track3(
        out_root=out_root,
        n_reps=3,
        L=50_000,
        n_samples=10,
    )
    # Both engine dirs exist with 3 rep subdirs each
    assert (out_root / "msinv").exists()
    assert (out_root / "msprime").exists()
    for engine in ("msinv", "msprime"):
        for rep in range(3):
            assert (out_root / engine / f"rep_{rep:03d}" / "stats.npz").exists()
    # Equivalence table populated
    assert "pi__F" in result["equivalence_table"]
    assert result["equivalence_table"]["pi__F"]["verdict"] in (
        "equivalent", "not_equivalent", "investigate"
    )
