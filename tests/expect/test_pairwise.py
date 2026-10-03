import math

import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.hull import Demography, InversionSpec

N, P_I, T_INV = 1000.0, 0.3, 2000.0
BL, BR = 10_000.0, 20_000.0
Y_IN = 15_000.0


def _model(origin="barrier", t_inv=T_INV, p=P_I, demo=None, r=1e-8):
    inv = InversionSpec(
        bp_left=BL, bp_right=BR, p_inv=p, t_inv=t_inv, mean_tract_length=0.0
    )
    return ExpectModel(demo or Demography([N]), inv, r, origin=origin)


@pytest.mark.parametrize("state,f", [("I", P_I), ("S", 1 - P_I)])
def test_same_state_barrier_closed_form(state, f):
    a = 2 * N * f
    res = _model().pairwise_times(Y_IN, (0, state), (0, state), times=[1000.0, 3000.0])
    expected = a * (1 - math.exp(-T_INV / a)) + 2 * N * math.exp(-T_INV / a)
    assert res.mean == pytest.approx(expected, rel=1e-6)
    assert res.survival[0] == pytest.approx(math.exp(-1000.0 / a), rel=1e-6)
    assert res.survival[1] == pytest.approx(
        math.exp(-T_INV / a) * math.exp(-1000.0 / (2 * N)), rel=1e-6
    )


def test_between_states_barrier_closed_form():
    res = _model().pairwise_times(Y_IN, (0, "I"), (0, "S"))
    assert res.mean == pytest.approx(T_INV + 2 * N, rel=1e-6)
    assert res.var == pytest.approx((2 * N) ** 2, rel=1e-5)


def test_single_origin_closed_form():
    a = 2 * N * P_I
    res = _model("single").pairwise_times(Y_IN, (0, "I"), (0, "I"))
    assert res.mean == pytest.approx(a * (1 - math.exp(-T_INV / a)), rel=1e-6)


def test_old_inversion_limit():
    res = _model(t_inv=1e7).pairwise_times(Y_IN, (0, "I"), (0, "I"))
    assert res.mean == pytest.approx(2 * N * P_I, rel=1e-6)


def test_flank_site_matches_hand_solved_chain():
    r, delta = 1e-6, 500.0
    m = ExpectModel(
        Demography([N]),
        InversionSpec(
            bp_left=BL, bp_right=BR, p_inv=P_I, t_inv=1e9, mean_tract_length=0.0
        ),
        r,
    )
    c, p = r * delta, P_I
    q = np.array(
        [
            [-(1 / (2 * N * p) + 2 * c * (1 - p)), 2 * c * (1 - p), 0.0],
            [c * p, -c, c * (1 - p)],
            [0.0, 2 * c * p, -(1 / (2 * N * (1 - p)) + 2 * c * p)],
        ]
    )  # transient states II, IS, SS (row generator)
    expected = np.linalg.solve(-q, np.ones(3))
    y = BL - delta
    assert m.pairwise_times(y, (0, "I"), (0, "I")).mean == pytest.approx(
        expected[0], rel=1e-6
    )
    assert m.pairwise_times(y, (0, "I"), (0, "S")).mean == pytest.approx(
        expected[1], rel=1e-6
    )


def test_profile_matches_pointwise():
    m = _model(t_inv=1e5, r=1e-6)
    ys = [Y_IN, BL - 500.0, BR + 2000.0]
    prof = m.profile(ys, (0, "I"), (0, "I"))
    for y, v in zip(ys, prof):
        assert v == pytest.approx(
            m.pairwise_times(y, (0, "I"), (0, "I")).mean, rel=1e-7
        )


def test_equal_frequencies_reduce_to_plain_island_model():
    # Spec §5.B item 4: with f_0(I) = f_1(I) the conditional migration rate is
    # plain m, so I lineages follow a two-deme island model with deme size
    # N * p: E[T_between] = 4 N p + 1 / (2 m).
    m_rate = 1e-3
    demo = Demography([N, N], migration_matrix=[[0.0, m_rate], [m_rate, 0.0]])
    res = _model(t_inv=1e9, demo=demo).pairwise_times(Y_IN, (0, "I"), (1, "I"))
    assert res.mean == pytest.approx(4 * N * P_I + 1 / (2 * m_rate), rel=1e-6)


def test_asymmetric_migration_matches_hand_solved_chain():
    # migration_matrix[1][0] = m: a lineage in pop 0 moves backward to pop 1;
    # nothing moves 1 -> 0. Conditional I hazard 0 -> 1 is m * p1 / p0.
    m_rate, p0, p1 = 1e-3, 0.2, 0.6
    demo = Demography([N, N], migration_matrix=[[0.0, 0.0], [m_rate, 0.0]])
    model = _model(t_inv=1e9, p={0: p0, 1: p1}, demo=demo)
    h = m_rate * p1 / p0
    c0, c1 = 1 / (2 * N * p0), 1 / (2 * N * p1)
    q = np.array(
        [
            [-(c0 + 2 * h), 2 * h, 0.0],
            [0.0, -h, h],
            [0.0, 0.0, -c1],
        ]
    )  # transient states I0I0, I0I1, I1I1 (row generator)
    expected = np.linalg.solve(-q, np.ones(3))
    starts = [((0, "I"), (0, "I")), ((0, "I"), (1, "I")), ((1, "I"), (1, "I"))]
    for (a, b), e in zip(starts, expected):
        assert model.pairwise_times(Y_IN, a, b).mean == pytest.approx(e, rel=1e-6)


# --- Review focus ---------------------------------------------------------


def test_zero_frequency_sample_rejected():
    m = _model(p={0: 0.0, 1: 0.5}, demo=Demography([N, N]))
    with pytest.raises(ValueError, match="zero frequency"):
        m.pairwise_times(Y_IN, (0, "I"), (1, "I"))


def test_disconnected_populations_rejected():
    m = _model(demo=Demography([N, N]))
    with pytest.raises(ValueError, match="never coalesce"):
        m.pairwise_times(Y_IN, (0, "S"), (1, "S"))


def test_breakpoint_sites_are_finite():
    # Gene conversion makes switching active inside; at BL the flux coverage
    # phi -> 0 and BR is a flank site with delta = 0, so both have zero
    # switching and must agree with each other but not with the interior.
    inv = InversionSpec(
        bp_left=BL,
        bp_right=BR,
        p_inv=P_I,
        t_inv=T_INV,
        mean_tract_length=100.0,
        gene_conversion_rate=1e-6,
    )
    m = ExpectModel(Demography([N]), inv, 1e-6)
    assert m.switching_coefficient(Y_IN) > 0.0
    assert m.switching_coefficient(BL) == pytest.approx(0.0, abs=1e-15)
    assert m.switching_coefficient(BR) == 0.0
    inside = m.pairwise_times(Y_IN, (0, "I"), (0, "I")).mean
    at_bl = m.pairwise_times(BL, (0, "I"), (0, "I")).mean
    at_br = m.pairwise_times(BR, (0, "I"), (0, "I")).mean
    assert math.isfinite(at_bl) and math.isfinite(at_br)
    assert at_bl == pytest.approx(at_br, rel=1e-7)
    assert inside != pytest.approx(at_bl, rel=1e-4)


def test_negative_times_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        _model().pairwise_times(Y_IN, (0, "I"), (0, "I"), times=[-1.0, 10.0])


def test_final_growth_rejected():
    demo = Demography([N])
    demo.add_growth_rate_change(100.0, population=0, growth_rate=1e-3)
    with pytest.raises(NotImplementedError, match="growth"):
        _model(demo=demo)


def test_more_than_one_inversion_is_not_supported():
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, p_inv=0.3, t_inv=5000.0)
    with pytest.raises(NotImplementedError, match="exactly one inversion"):
        ExpectModel(Demography([1000.0]), [inv, inv], 1e-8)


def test_precomputed_patterns_match_direct_assembly():
    # qt(t) and the Jacobian are refilled into fixed sparsity patterns; they
    # must equal the direct sums sum_k g_k(t) AT_k and the stacked Jacobian.
    import scipy.sparse as sp

    from msinv.expect._chain import Chain
    from msinv.expect._solve import Solver

    demo = Demography([N, N], migration_matrix=[[0.0, 1e-3], [2e-3, 0.0]])
    inv = InversionSpec(
        bp_left=BL,
        bp_right=BR,
        p_inv={0: 0.3, 1: 0.6},
        t_inv=T_INV,
        mean_tract_length=0.0,
    )
    m = ExpectModel(demo, inv, 1e-6)
    chain = Chain(
        [(0, "I"), (1, "S"), (0, "I")],
        2,
        m.epochs.mig_pairs(),
        [],
        True,
        "barrier",
        None,
    )
    s = Solver(chain, m.epochs, m.freq, [5e-4, 0.0])
    for t in (10.0, T_INV + 500.0):
        direct = sum(s._g(k, t) * at for k, at in s._AT.items())
        assert abs(s.qt(t) - direct).max() <= 1e-12 * abs(direct).max()
        z = np.zeros(s.N + s._n_aux)
        s._memo_t = None
        jac = s._jac(t, z)
        low = sp.vstack([s._rew, s._smat, t * s._smat])
        ref = sp.bmat(
            [
                [direct, sp.csr_matrix((s.N, s._n_aux))],
                [low, sp.csr_matrix((s._n_aux, s._n_aux))],
            ]
        )
        assert abs(jac - ref).max() <= 1e-12 * abs(ref).max()
