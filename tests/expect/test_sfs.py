import numpy as np
import pytest

from illex.scripts.sfs_shape import project_unfolded
from msinv.expect import ExpectModel
from msinv.expect.model import fold
from msinv.hull import Demography, InversionSpec

N = 1000.0


def _model(t_inv, p=0.3, r=1e-8):
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv, mean_tract_length=0.0
    )
    return ExpectModel(Demography([N]), inv, r)


def test_kingman_after_t_inv():
    sfs = _model(t_inv=1e-6).expected_sfs(15_000.0, 0, "I", 6)
    b = np.arange(1, 6)
    assert np.allclose(sfs, 4 * N / b, rtol=1e-5)


def test_old_inversion_is_kingman_at_reduced_size():
    sfs = _model(t_inv=1e7).expected_sfs(15_000.0, 0, "I", 6)
    b = np.arange(1, 6)
    assert np.allclose(sfs, 4 * N * 0.3 / b, rtol=1e-6)


def test_pair_branch_length_is_twice_pairwise_time():
    m = _model(t_inv=2000.0, r=1e-6)
    y = 9_500.0
    lb = m.expected_branch_lengths(y, [(0, "I"), (0, "I")])
    assert lb[0] / 2 == pytest.approx(
        m.pairwise_times(y, (0, "I"), (0, "I")).mean, rel=1e-7
    )


def test_fold_convention():
    assert fold(np.array([1.0, 2.0, 3.0, 4.0, 5.0])).tolist() == [6.0, 6.0, 3.0]


def test_f1_equals_illex_projection_of_larger_sample():
    # Projecting the expected SFS of 12 chromosomes to 10 must equal the
    # expected SFS of 10 (spec §3.5), folded as in illex/scripts/sfs_shape.py.
    m = _model(t_inv=4000.0, r=1e-6)
    y = 9_000.0
    big = m.expected_sfs(y, 0, "I", 12)
    folded = project_unfolded(np.concatenate([[0.0], big, [0.0]]), 12, 10)
    assert m.f1(y, 0, "I", n=10) == pytest.approx(folded[0] / folded.sum(), rel=1e-6)


def test_no_pivot_lu_matches_default_bdf(monkeypatch):
    # The diagonal-pivot LU (msinv/expect/_solve.py::_NoPivotBDF) must give
    # the same answer as scipy's default BDF, within the ODE tolerance.
    from msinv.expect import _solve

    m = _model(t_inv=4000.0, r=1e-6)
    fast = m.expected_sfs(9_000.0, 0, "I", 8)
    monkeypatch.setattr(_solve, "_ODE_METHOD", "BDF")
    default = m.expected_sfs(9_000.0, 0, "I", 8)
    assert np.allclose(fast, default, rtol=1e-7, atol=0.0)
