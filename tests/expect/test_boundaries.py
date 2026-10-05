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


def _join_into_empty(origin="barrier"):
    # Pop 1 carries the inversion at 0.5, pop 0 never does (p = 0 throughout);
    # ej at t = 500 < t_inv = 2000 moves pop 1's lineages into pop 0.
    traj = {
        "type": "precomputed",
        "times": [0.0, 1000.0, 2000.0],
        "freqs": [[0.0] * 3, [0.5] * 3],
        "n_e": [1000.0, 1000.0],
        "t_inv": [2000.0, 2000.0],
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    d = Demography([1000.0, 1000.0])
    d.add_event(("ej", 500.0, 1, 0))
    return ExpectModel(d, inv, 1e-8, origin=origin)


def test_join_into_zero_frequency_cell_is_impossible():
    m = _join_into_empty()
    with pytest.raises(ImpossibleHistoryError, match=r"join 1 -> 0.*state I"):
        m.pairwise_times(Y, (1, "I"), (1, "I"))


def test_single_origin_population_joined_away_before_t_inv_is_rejected():
    with pytest.raises(ValueError, match="origin population 1"):
        _join_into_empty(origin="single")


def _decline_at_t_inv(t_inv, f0, n=1000.0):
    traj = {
        "type": "precomputed",
        "times": [0.0, t_inv],
        "freqs": [[f0, 0.0]],
        "n_e": [n],
        "t_inv": [t_inv],
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    return ExpectModel(Demography([n]), inv, 1e-8)


def test_linear_decline_to_zero_at_t_inv_matches_closed_form():
    # f(t) = f0 (1 - t/T) to zero at T = t_inv; coalescence of two I lineages
    # at 1/(2 N f(t)) gives P(T2 > t) = (1 - t/T)^a, a = T / (2 N f0), and the
    # limit chain coalesces any pair still there before t_inv, so
    # E[T2] = T / (a + 1) and E[T2^2] = 2 T^2 / ((a + 1)(a + 2)).
    t_inv, f0, n = 1000.0, 0.4, 1000.0
    a = t_inv / (2.0 * n * f0)
    mean = t_inv / (a + 1.0)
    var = 2.0 * t_inv**2 / ((a + 1.0) * (a + 2.0)) - mean**2
    res = _decline_at_t_inv(t_inv, f0, n).pairwise_times(
        Y, (0, "I"), (0, "I"), times=[0.0, 300.0, 999.0, 1500.0]
    )
    assert res.mean == pytest.approx(mean, rel=1e-6)
    assert res.var == pytest.approx(var, rel=1e-6)
    # Survival is a probability integrated at rtol 1e-8: compare absolutely
    # (the error shrinks with the solver tolerance). Nothing survives t_inv.
    expected = [1.0, 0.7**a, 0.001**a]
    assert res.survival[:3] == pytest.approx(expected, rel=0.0, abs=1e-8)
    assert res.survival[3] == 0.0


def test_coupled_decline_to_zero_at_t_inv_is_finite():
    traj = {
        "type": "coupled",
        "p_final": [0.5, 0.2],
        "n_e": [1000.0, 1000.0],
        "s": [0.02, 0.0],
        "m": 1e-3,
        "seed": 1,
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    d = Demography([1000.0, 1000.0], migration_matrix=[[0, 1e-3], [1e-3, 0]])
    m = ExpectModel(d, inv, 1e-8)
    for other in [(1, "I"), (0, "I")]:
        assert np.isfinite(m.pairwise_times(Y, (1, "I"), other).mean)


def test_survival_requested_inside_clamped_window_is_filled():
    # An epoch boundary 1e-7 before a support boundary at 500 is closer than
    # the eps clamp (5e-7), so no integration runs in [500 - 1e-7, 500).
    traj = {
        "type": "precomputed",
        "times": [0.0, 250.0, 500.0, 1000.0],
        "freqs": [[0.5] * 4, [0.5, 0.25, 0.0, 0.0]],
        "n_e": [1000.0, 1000.0],
        "t_inv": [1000.0, 500.0],
    }
    inv = InversionSpec(
        bp_left=10_000.0, bp_right=20_000.0, trajectory=traj, mean_tract_length=0.0
    )
    d = Demography([1000.0, 1000.0], migration_matrix=[[0.0, 1e-3], [0.0, 0.0]])
    d.add_event(("en", 500.0 - 1e-7, 0, 1000.0))
    m = ExpectModel(d, inv, 1e-8)
    surv = m.pairwise_times(
        Y, (1, "I"), (0, "I"), times=[499.0, 500.0 - 5e-8, 600.0]
    ).survival
    assert np.all(np.isfinite(surv))
    assert surv[0] >= surv[1] >= surv[2]


@pytest.mark.parametrize(
    "times, freqs",
    [
        # flat-bottomed bottleneck of the inverted class
        (
            [0.0, 4999.0, 5000.0, 5010.0, 5011.0, 30000.0],
            [0.3, 0.3, 0.002, 0.002, 0.3, 0.3],
        ),
        # V-shaped bottleneck
        ([0.0, 4995.0, 5000.0, 5005.0, 30000.0], [0.3, 0.3, 0.001, 0.3, 0.3]),
    ],
)
def test_narrow_dip_is_not_stepped_over(times, freqs):
    # Unforced knots let BDF step across a short dip entirely (9.5% error for
    # the flat one); the default forces the knots where p turns.
    traj = {
        "type": "precomputed",
        "times": times,
        "freqs": [freqs],
        "n_e": [10_000.0],
        "t_inv": [40_000.0],
    }
    inv = InversionSpec(
        bp_left=100_000.0, bp_right=200_000.0, trajectory=traj, mean_tract_length=0.0
    )
    demo = Demography([10_000.0])
    default = ExpectModel(demo, inv, 1e-8)
    forced = ExpectModel(demo, inv, 1e-8, force_knots_max=10**6)
    a = default.pairwise_times(99_000.0, (0, "I"), (0, "I")).mean
    b = forced.pairwise_times(99_000.0, (0, "I"), (0, "I")).mean
    assert a == pytest.approx(b, rel=1e-6)
