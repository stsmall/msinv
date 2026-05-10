"""Tests for Garud H1, H12, H2/H1 hand-rolled implementation."""
import numpy as np
import pytest

from validation._lib.stats import hstats_from_haps


def test_h1_all_identical_is_one():
    """If every haplotype is identical, H1 = 1."""
    haps = np.zeros((10, 5), dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(1.0)
    assert out["H12"] == pytest.approx(1.0)


def test_h1_all_distinct():
    """10 distinct haplotypes: each at frequency 0.1, H1 = 10 * 0.1^2 = 0.1."""
    haps = np.eye(10, dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(0.1)


def test_h12_combines_top_two():
    """5 haps: AAAA, AAAA, BBBB, BBBB, CCCC.
    Frequencies: A=0.4, B=0.4, C=0.2.
    H1 = 0.4^2 + 0.4^2 + 0.2^2 = 0.36
    H12 = (0.4 + 0.4)^2 + 0.2^2 = 0.68
    """
    haps = np.array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,1,1],
                     [1,1,1,1],
                     [2,2,2,2]], dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H1"] == pytest.approx(0.36)
    assert out["H12"] == pytest.approx(0.68)


def test_h2_over_h1_known_case():
    """Same setup as test_h12_combines_top_two.
    H2 = H1 - 0.4^2 = 0.36 - 0.16 = 0.20
    H2/H1 = 0.20 / 0.36 ≈ 0.5556.
    """
    haps = np.array([[0,0,0,0],
                     [0,0,0,0],
                     [1,1,1,1],
                     [1,1,1,1],
                     [2,2,2,2]], dtype=np.int8)
    out = hstats_from_haps(haps)
    assert out["H2_over_H1"] == pytest.approx(0.20 / 0.36)
