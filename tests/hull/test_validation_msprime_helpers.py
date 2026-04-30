"""Unit tests for the small math helpers in test_validation_msprime."""

import pytest

from tests.hull.test_validation_msprime import _bonferroni_z


def test_bonferroni_z_matches_spec_table():
    # Spec table values (rounded to 2 dp).
    # K=7 → 3.52; K=8 → 3.55; K=9 → 3.59 at α=0.003.
    assert _bonferroni_z(7) == pytest.approx(3.52, abs=0.01)
    assert _bonferroni_z(8) == pytest.approx(3.55, abs=0.01)
    assert _bonferroni_z(9) == pytest.approx(3.59, abs=0.01)


def test_bonferroni_z_monotone_in_k():
    """More bins → tighter per-bin α → larger z."""
    zs = [_bonferroni_z(k) for k in range(2, 30)]
    assert zs == sorted(zs)
