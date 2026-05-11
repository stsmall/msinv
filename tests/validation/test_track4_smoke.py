"""Smoke test for Track 4 driver at scaled-down params."""
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.timeout(300)
def test_track4_smoke_hard_2_reps(tmp_path):
    """2 reps × 2 engines on the hard-sweep subscenario at L=100 kb."""
    from validation.track4_discoal.run import run_track4_subscenario

    out_root = tmp_path / "track4_hard"
    result = run_track4_subscenario(
        out_root=out_root,
        subscenario="hard",
        n_reps=2,
        L=100_000,
        n_samples=10,
    )
    assert (out_root / "msinv").exists()
    assert (out_root / "discoal").exists()
    for engine in ("msinv", "discoal"):
        for rep in range(2):
            assert (out_root / engine / f"rep_{rep:03d}" / "stats.npz").exists()
    assert "equivalence_table" in result
