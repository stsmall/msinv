"""Harness test 1: with gene flux switched off, msinv must reproduce the
analytic dxy/pi_I predicted by theory.py -- under BOTH demographies, whose
predicted floors differ (2.563 growth vs 3.978 constant).

Marked slow. Run with: .venv/bin/python -m pytest tests/illex/ -m slow
"""
import numpy as np
import pytest

from illex import model, stats, theory

N_REPS = 8
SEQ_LEN = 30_000


def _mean_ratio(arm, N_fn, t_inv):
    vals = []
    for rep in range(N_REPS):
        sim = model.build_inversion_sim(
            arm=arm, seq_length=SEQ_LEN, t_inv=t_inv,
            gamma=1e-15,                      # msinv requires gamma > 0
            seed=1000 + rep,
        )
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        vals.append(stats.arrangement_stats(ts, i_nodes, s_nodes)["dxy_over_pi_i"])
    return float(np.mean(vals))


@pytest.mark.slow
@pytest.mark.parametrize("arm,N_fn", [("growth", theory.N_growth),
                                      ("constant", theory.N_const)])
def test_zero_flux_matches_predicted_ratio_at_floor(arm, N_fn):
    """At the floor's t_inv, simulated dxy/pi_I must match the prediction."""
    predicted, t_at_floor = theory.dxy_floor(N_fn)
    observed = _mean_ratio(arm, N_fn, t_at_floor)
    assert observed == pytest.approx(predicted, rel=0.15), (
        f"{arm}: msinv gave {observed:.3f}, theory predicts {predicted:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("arm,N_fn", [("growth", theory.N_growth),
                                      ("constant", theory.N_const)])
def test_zero_flux_never_below_floor(arm, N_fn):
    """The floor is a floor: no t_inv may produce a smaller ratio."""
    floor, _ = theory.dxy_floor(N_fn)
    for t_inv in (4.0e5, 9.0e5, 2.0e6):
        observed = _mean_ratio(arm, N_fn, t_inv)
        assert observed > floor * 0.85, (
            f"{arm} t_inv={t_inv:.0f}: {observed:.3f} below floor {floor:.3f}"
        )


@pytest.mark.slow
def test_growth_floor_is_lower_than_constant_in_simulation():
    """The demography effect must be visible in msinv, not just in theory."""
    g_floor, g_t = theory.dxy_floor(theory.N_growth)
    c_floor, c_t = theory.dxy_floor(theory.N_const)
    assert _mean_ratio("growth", theory.N_growth, g_t) < \
           _mean_ratio("constant", theory.N_const, c_t)
