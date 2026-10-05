import time

import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.expect._chain import Chain
from msinv.expect._solve import Solver
from msinv.hull import Demography, InversionSpec

BL, BR = 100_000.0, 200_000.0


def _model(**kw):
    inv = InversionSpec(
        bp_left=BL, bp_right=BR, p_inv=0.3, t_inv=50_000.0, mean_tract_length=0.0
    )
    return ExpectModel(Demography([10_000.0]), inv, 1e-8, **kw)


def test_pairwise_profile_100_positions_under_1s():
    m = _model()
    ys = np.linspace(0.0, BL - 1.0, 100)
    m.profile(ys[:2], (0, "I"), (0, "I"))  # warm-up
    t0 = time.perf_counter()
    out = m.profile(ys, (0, "I"), (0, "I"))
    assert time.perf_counter() - t0 < 1.0
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_f1_n16_regression_guard():
    # Spec §5.F (relaxed 2026-10-03): f1 at n=20 takes minutes per history.
    # Guard the measured scaling at n=16 (24 s on the shared device) with
    # headroom for load, without spending ~8 min on n=20 in the suite.
    m = _model()
    t0 = time.perf_counter()
    val = m.f1(BL - 5_000.0, 0, "I", n=16)
    assert time.perf_counter() - t0 < 90.0
    assert 0.0 < val < 1.0


@pytest.mark.slow
def test_unforced_knots_agree_with_forced():
    times = np.linspace(0.0, 40_000.0, 2001)
    freqs = 0.3 + 0.1 * np.sin(times / 3000.0)
    traj = {
        "type": "precomputed",
        "times": times.tolist(),
        "freqs": [freqs.tolist()],
        "n_e": [10_000.0],
        "t_inv": [40_000.0],
    }
    inv = InversionSpec(bp_left=BL, bp_right=BR, trajectory=traj, mean_tract_length=0.0)
    demo = Demography([10_000.0])
    loose = ExpectModel(demo, inv, 1e-8)
    tight = ExpectModel(demo, inv, 1e-8, force_knots_max=10_000)
    a = loose.pairwise_times(BL - 1_000.0, (0, "I"), (0, "I")).mean
    b = tight.pairwise_times(BL - 1_000.0, (0, "I"), (0, "I")).mean
    assert a == pytest.approx(b, rel=1e-5)


def test_monotone_trajectory_knots_are_not_forced():
    # Restarting BDF at every knot made Illex f1 ~19x slower (400 knots).
    times = np.linspace(0.0, 40_000.0, 400)
    traj = {
        "type": "precomputed",
        "times": times.tolist(),
        "freqs": [np.linspace(0.3, 0.7, 400).tolist()],
        "n_e": [10_000.0],
        "t_inv": [40_000.0],
    }
    inv = InversionSpec(bp_left=BL, bp_right=BR, trajectory=traj, mean_tract_length=0.0)
    m = ExpectModel(Demography([10_000.0]), inv, 1e-8)
    chain = Chain([(0, "I"), (0, "I")], 1, set(), [], False, "barrier", None, None)
    assert Solver(chain, m.epochs, m.freq, [0.0]).breaks == [40_000.0]
