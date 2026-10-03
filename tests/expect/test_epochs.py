import math

import pytest

from msinv.expect._history import Epochs
from msinv.hull import Demography, InversionSpec


def test_migration_convention_matches_simulator():
    d = Demography([1000.0, 2000.0], migration_matrix=[[0.0, 0.01], [0.02, 0.0]])
    e = Epochs(d)
    # Backward, a lineage in src moves to dst at migration_matrix[dst][src].
    assert e.mig_rate(0, 1, 0.0) == 0.02
    assert e.mig_rate(1, 0, 0.0) == 0.01
    assert e.mig_pairs() == {(0, 1), (1, 0)}


def test_size_change_growth_and_join():
    d = Demography([1000.0, 2000.0], migration_matrix=[[0.0, 0.01], [0.02, 0.0]])
    d.add_population_size_change(100.0, population=0, new_size=500.0)
    d.add_growth_rate_change(200.0, population=1, growth_rate=0.001)
    d.add_population_split(300.0, derived=[1], ancestral=0)
    e = Epochs(d)
    assert e.boundaries() == [100.0, 200.0, 300.0]
    assert e.size(0, 50.0) == 1000.0
    assert e.size(0, 150.0) == 500.0
    assert e.size(1, 250.0) == pytest.approx(2000.0 * math.exp(-0.001 * 50.0))
    assert e.joins_at(300.0) == [(1, 0)]
    assert e.mig_rate(0, 1, 350.0) == 0.0
    assert e.mig_rate(1, 0, 350.0) == 0.0
    assert not e.final_growth_is_zero()


def test_unsupported_event_raises():
    d = Demography([1000.0, 1000.0])
    d.add_inversion_freq_change(10.0, 0, 0, 0.5)
    with pytest.raises(NotImplementedError, match="eig"):
        Epochs(d)


def test_inversion_parent_not_supported():
    with pytest.raises(NotImplementedError, match="nested"):
        InversionSpec(bp_left=0.0, bp_right=10.0, p_inv=0.5, t_inv=100.0, parent=0)
