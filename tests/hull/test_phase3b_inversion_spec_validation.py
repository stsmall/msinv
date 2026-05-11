# tests/hull/test_phase3b_inversion_spec_validation.py
"""Validation rules for the b2-flux InversionSpec fields:
mean_tract_length and tract_distribution."""

import warnings

import pytest

from msinv.hull import InversionSpec


def _kw(**overrides):
    """Minimal valid InversionSpec kwargs; override what each test needs."""
    base = dict(
        bp_left=0.0,
        bp_right=10_000.0,
        p_inv=0.5,
        t_inv=1000.0,
        gene_conversion_rate=1e-9,
    )
    base.update(overrides)
    return base


def test_mean_tract_length_negative_rejected():
    with pytest.raises(ValueError, match="mean_tract_length"):
        InversionSpec(**_kw(mean_tract_length=-1.0))


def test_mean_tract_length_zero_legal():
    # Zero is the canonical "disable flux via zero tract" path.
    inv = InversionSpec(**_kw(mean_tract_length=0.0))
    assert inv.mean_tract_length == 0.0


def test_mean_tract_length_positive_legal():
    inv = InversionSpec(**_kw(mean_tract_length=200.0))
    assert inv.mean_tract_length == 200.0


def test_mean_tract_length_above_half_inv_warns():
    # 7000 bp > inv_length/2 = 5000 bp -> warn (not error).
    with pytest.warns(UserWarning, match="mean_tract_length"):
        inv = InversionSpec(**_kw(mean_tract_length=7000.0))
    # Still constructs successfully.
    assert inv.mean_tract_length == 7000.0


def test_tract_distribution_geometric_legal():
    inv = InversionSpec(**_kw(tract_distribution="geometric"))
    assert inv.tract_distribution == "geometric"


def test_tract_distribution_fixed_legal():
    inv = InversionSpec(**_kw(tract_distribution="fixed"))
    assert inv.tract_distribution == "fixed"


def test_tract_distribution_invalid_rejected():
    with pytest.raises(ValueError, match="tract_distribution"):
        InversionSpec(**_kw(tract_distribution="gamma"))


def test_default_mean_tract_length_is_100():
    inv = InversionSpec(**_kw())
    assert inv.mean_tract_length == 100.0


def test_default_tract_distribution_is_geometric():
    inv = InversionSpec(**_kw())
    assert inv.tract_distribution == "geometric"


def test_flux_window_field_removed():
    """After migration, passing flux_window must raise TypeError
    (Python's default for unexpected kwargs)."""
    with pytest.raises(TypeError):
        InversionSpec(**_kw(flux_window=0.05))
