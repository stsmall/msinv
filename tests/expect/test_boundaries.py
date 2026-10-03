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
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, trajectory=traj,
                        mean_tract_length=0.0)
    return ExpectModel(Demography([1000.0, 1000.0], migration_matrix=mig), inv,
                       1e-8, origin=origin)


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

    expected, _ = quad(lambda u: total(500.0 * (1 - u * u)), 0.0, 1.0,
                       epsabs=0.0, epsrel=1e-10)
    got = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert got == pytest.approx(expected, rel=1e-5)


def test_single_origin_with_support_boundary():
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]], origin="single")
    barrier = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]])
    single = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert np.isfinite(single)
    assert single < barrier.pairwise_times(Y, (1, "I"), (0, "I")).mean
