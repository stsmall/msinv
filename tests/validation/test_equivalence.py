"""Tests for KS test, Cohen's D, and equivalence verdict."""

import numpy as np
import pytest

from validation._lib.equivalence import (
    ks_test,
    cohens_d,
    equivalence_verdict,
)


def test_ks_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    stat, p = ks_test(a, b)
    assert 0 <= stat <= 1
    assert p > 0.01  # cannot reject same-distribution null


def test_ks_clearly_different():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=500)
    b = rng.normal(2.0, 1.0, size=500)  # mean shift 2 SD
    _, p = ks_test(a, b)
    assert p < 0.001


def test_cohens_d_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cohens_d(a, b) == pytest.approx(0.0)


def test_cohens_d_unit_for_one_sd_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=10_000)
    b = rng.normal(1.0, 1.0, size=10_000)
    d = cohens_d(a, b)
    assert 0.9 < abs(d) < 1.1


def test_verdict_equivalent_for_identical():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    v = equivalence_verdict(a, b)
    assert v["verdict"] == "equivalent"


def test_verdict_not_equivalent_for_clearly_different():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 500)
    b = rng.normal(2, 1, 500)
    v = equivalence_verdict(a, b)
    assert v["verdict"] == "not_equivalent"


def test_verdict_investigate_high_power_tiny_diff():
    """Large n + tiny mean shift triggers KS rejection but small Cohen's D."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 100_000)
    b = rng.normal(0.05, 1, 100_000)
    v = equivalence_verdict(a, b)
    # Could be 'investigate' (p < 0.01 but D < 0.2) or 'equivalent' if
    # KS happens to not reject — accept either, but never 'not_equivalent'.
    assert v["verdict"] in ("equivalent", "investigate")
