import math

import numpy as np
import pytest
from scipy.integrate import quad

from msinv.expect import ExpectModel, ImpossibleHistoryError
from msinv.hull import Demography, InversionSpec

Y = 15_000.0


def _two_pop(freqs_pop1, mig, origin="barrier"):
    traj = {
        "type": "precomputed",
        "times": [0.0, 250.0, 500.0, 1000.0],
        "freqs": [[0.5] * 4, freqs_pop1],
        "n_e": [1000.0, 1000.0],
        "t_inv": [1000.0, 500.0],
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    return ExpectModel(
        Demography([1000.0, 1000.0], migration_matrix=mig), inv, 1e-8, origin=origin
    )


def test_frequency_jump_to_zero_without_exit_is_impossible():
    m = _two_pop([0.5, 0.5, 0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ImpossibleHistoryError, match="population 1"):
        m.pairwise_times(Y, (1, "I"), (0, "I"))


def test_continuous_decline_with_migration_matches_limit():
    # Lineage A starts in (1, I); backward hazard 1 -> 0 is mig[0][1] = 1e-3
    # times f_0(I) / f_1(I) = 1 / (1 - t/500), so A is still in pop 1 at time
    # t with probability (1 - t/500)^(1/2). With u = sqrt(1 - tau/500) the
    # arrival time tau is uniform in u. After arrival A coalesces with B in
    # (0, I) at 1/(2*1000*0.5) until t_inv = 1000, then at 1/2000.
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]])

    def total(tau):
        rate = 1e-3
        left = 1000.0 - tau
        decay = math.exp(-rate * left)
        return tau + (1 - decay) / rate + decay * 2000.0

    expected, _ = quad(
        lambda u: total(500.0 * (1 - u * u)), 0.0, 1.0, epsabs=0.0, epsrel=1e-10
    )
    got = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert got == pytest.approx(expected, rel=1e-5)


def test_single_origin_with_support_boundary():
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]], origin="single")
    barrier = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]])
    single = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert np.isfinite(single)
    assert single < barrier.pairwise_times(Y, (1, "I"), (0, "I")).mean


def _model(times, freqs, t_inv, mig, sizes=None):
    traj = {
        "type": "precomputed",
        "times": times,
        "freqs": freqs,
        "n_e": [1000.0] * len(freqs),
        "t_inv": t_inv,
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    sizes = sizes or [1000.0] * len(freqs)
    return ExpectModel(Demography(sizes, migration_matrix=mig), inv, 1e-8)


def test_exit_into_cell_vanishing_at_same_time_is_impossible():
    # (0, I) can only leave towards (1, I), which vanishes at the same t_b, so
    # about exp(-0.5) of A's ancestry is stranded in (0, I) at t = 500.
    decline = [0.5, 0.25, 0.0, 0.0]
    mig = [[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]]
    m = _model(
        [0.0, 250.0, 500.0, 1000.0],
        [decline, decline, [0.5] * 4],
        [500.0, 500.0, 1000.0],
        mig,
    )
    with pytest.raises(ImpossibleHistoryError, match="population 0"):
        m.pairwise_times(Y, (0, "I"), (2, "I"))


def _late_boundary(freqs_pop1):
    return _model(
        [0.0, 99_999.0, 100_000.0, 200_000.0],
        [[0.5] * 4, freqs_pop1],
        [200_000.0, 100_000.0],
        [[0.0, 1e-8], [0.0, 0.0]],
    )


def test_steep_continuous_decline_at_large_time_is_not_a_jump():
    m = _late_boundary([0.05, 0.05, 0.0, 0.0])
    assert np.isfinite(m.pairwise_times(Y, (1, "I"), (0, "I")).mean)


def test_jump_to_zero_at_large_time_is_impossible():
    m = _late_boundary([0.05] * 4)
    with pytest.raises(ImpossibleHistoryError, match="jumps to zero"):
        m.pairwise_times(Y, (1, "I"), (0, "I"))


def test_small_migration_decline_matches_limit():
    # As the continuous-decline test with m = 1e-6: P(tau > t) =
    # (1 - t/500)^alpha with alpha = 5e-4, so almost all of A's ancestry is
    # still in (1, I) at t = 500 and leaves through the limit chain.
    # E[g(tau)] = g(0) + int_0^500 g'(t) P(tau > t) dt, where
    # g(tau) = tau + (1 - e^{-r l})/r + 2000 e^{-r l}, l = 1000 - tau, r = 1e-3,
    # so g'(t) = 1 + e^{-r l}.
    alpha, rate = 1e-6 * 500.0, 1e-3
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-6], [0.0, 0.0]])

    def g(tau):
        decay = math.exp(-rate * (1000.0 - tau))
        return tau + (1 - decay) / rate + decay * 2000.0

    integral, _ = quad(
        lambda t: (1 + math.exp(-rate * (1000.0 - t))) * (1 - t / 500.0) ** alpha,
        0.0,
        500.0,
        epsabs=0.0,
        epsrel=1e-12,
    )
    expected = g(0.0) + integral
    got = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert got == pytest.approx(expected, rel=1e-6)


def _linear_decline():
    return _model(
        [0.0, 500.0, 1000.0],
        [[0.5] * 3, [0.5, 0.0, 0.0]],
        [1000.0, 500.0],
        [[0.0, 0.0], [0.0, 0.0]],
    )


def test_coalescence_only_exit_matches_closed_form():
    # Coalescence in (1, I) at 1/(2*1000*f) = 1/(1000 (1 - t/500)) gives
    # P(T > t) = (1 - t/500)^(1/2), so E[T] = 500 * 2/3.
    got = _linear_decline().pairwise_times(Y, (1, "I"), (1, "I")).mean
    assert got == pytest.approx(1000.0 / 3.0, rel=1e-6)


def test_no_exit_after_continuous_decline_is_impossible():
    with pytest.raises(ImpossibleHistoryError, match="population 1"):
        _linear_decline().pairwise_times(Y, (1, "I"), (0, "I"))
