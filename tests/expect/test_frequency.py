import numpy as np
import pytest

from msinv.expect._history import FrequencyHistory, flux_coverage
from msinv.hull import InversionSpec

L = 100_000.0


def _inv(**kw):
    return InversionSpec(bp_left=0.0, bp_right=L, mean_tract_length=0.0, **kw)


TWO_POP = {
    "type": "precomputed",
    "times": [0.0, 250.0, 500.0, 1000.0],
    "freqs": [[0.5] * 4, [0.5, 0.25, 0.0, 0.0]],
    "n_e": [1000.0, 1000.0],
    "t_inv": [1000.0, 500.0],
}


def test_constant_frequencies():
    fh = FrequencyHistory(_inv(p_inv=0.3, t_inv=1000.0), 1)
    assert fh.f(0, "I", 10.0) == 0.3
    assert fh.f(0, "S", 10.0) == pytest.approx(0.7)
    assert fh.t_inv == 1000.0
    assert fh.knots == []
    assert fh.support_boundaries() == []
    with pytest.raises(ValueError):
        fh.f(0, "I", 1000.0)


def test_per_population_t_inv_is_a_support_boundary():
    fh = FrequencyHistory(_inv(trajectory=TWO_POP), 2)
    assert fh.t_inv == 1000.0
    assert fh.support_boundaries() == [(500.0, 1, "I")]
    assert fh.f(1, "I", 125.0) == pytest.approx(0.375)
    assert fh.origin_population() == 0


def test_origin_population_must_be_unique():
    fh = FrequencyHistory(_inv(p_inv={0: 0.3, 1: 0.4}, t_inv=1000.0), 2)
    with pytest.raises(ValueError, match="exactly one population"):
        fh.origin_population()


def test_draw_conditional():
    fh = FrequencyHistory(_inv(trajectory=TWO_POP), 2)
    rng = np.random.default_rng(1)
    draws = [fh.draw_conditional(0, 10.0, {}, rng)[0] for _ in range(20000)]
    assert abs(draws.count("I") / 20000 - 0.5) < 0.015
    assert fh.draw_conditional(0, 10.0, {0: "S"}, rng) == {0: "S"}
    with pytest.raises(ValueError, match="zero frequency"):
        fh.draw_conditional(1, 600.0, {0: "I"}, rng)


def test_flux_coverage_flat_region_fixed_tract():
    w = 100.0 / L
    c = flux_coverage(50_000.0, 0.0, L, 100.0, "fixed")
    assert c == pytest.approx(100.0 * w / (1.0 - w), rel=1e-8)


def test_flux_coverage_flat_region_geometric_tract():
    w = 100.0 / L
    c = flux_coverage(50_000.0, 0.0, L, 100.0, "geometric")
    assert c == pytest.approx(100.0 * w / (1.0 - w), rel=1e-6)


def test_flux_coverage_outside_and_at_breakpoint():
    assert flux_coverage(-1.0, 0.0, L, 100.0, "fixed") == 0.0
    assert flux_coverage(L, 0.0, L, 100.0, "fixed") == 0.0
    assert flux_coverage(50_000.0, 0.0, L, 0.0, "fixed") == 0.0
    assert 0.0 <= flux_coverage(0.0, 0.0, L, 100.0, "fixed") < 1e-6


def _precomputed(times, freqs, t_inv=40_000.0):
    return _inv(
        trajectory={
            "type": "precomputed",
            "times": times,
            "freqs": [freqs],
            "n_e": [10_000.0],
            "t_inv": [t_inv],
        }
    )


def test_turning_knots_mark_both_edges_of_a_flat_dip():
    fh = FrequencyHistory(
        _precomputed(
            [0.0, 4999.0, 5000.0, 5010.0, 5011.0, 30000.0],
            [0.3, 0.3, 0.002, 0.002, 0.3, 0.3],
        ),
        1,
    )
    assert fh.turning_knots() == [4999.0, 5000.0, 5010.0, 5011.0]


def test_turning_knots_empty_for_a_monotone_ramp():
    times = np.linspace(0.0, 40_000.0, 400)
    fh = FrequencyHistory(_precomputed(times.tolist(), np.linspace(0.3, 0.7, 400)), 1)
    assert len(fh.knots) == 398  # t = 0 and t = t_inv are not knots
    assert fh.turning_knots() == []
