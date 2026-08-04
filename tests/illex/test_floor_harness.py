"""Harness test 1 (reframed after the Task 4 model-mismatch finding).

The original version of this harness asserted that msinv's plain
``p_inv=``/``t_inv=`` (constant-frequency) path must reproduce
``theory.py``'s analytic dxy/pi_I floor. It doesn't, and it can't: msinv's
``ConstantTrajectory`` barrier does not force monophyly of the I class at
t_inv (it relabels I-class lineages to S and lets them keep coalescing
under ordinary demography -- see ``rust/msinv-core/src/simulator.rs::
cross_barriers_static``), whereas ``theory.py``'s ``within_i`` formula
forces a mass-point coalescence exactly at t_inv (a strict single-origin/
founder model). Both are internally consistent; they encode different
models of the derived class's early history. See task-4-report.md for the
full diagnosis.

msinv's ``deterministic`` trajectory (``build_inversion_sim(...,
p_start=...)``) makes that choice a continuum: ``p_start=1/(2*Ne)`` is the
single-founder limit, ``p_start=None`` is the legacy constant-frequency (no
monophyly) limit, and any value between is a legitimate fitted parameter.
This file tests structural properties of that family instead of asserting
agreement with theory.py's (different) model:

  (a) monotonicity of pi_I/pi_S and dxy/pi_I in p_start;
  (b) the hard/soft limits bracket the empirically observed pi_I/pi_S;
  (c) the semantic distinction this task discovered (constant p_inv lacks
      monophyly at t_inv; the founder limit has it) is pinned as a
      regression guard;
  (d) a fixed numeric regression anchor at a specific (t_inv, p_start).

``theory.py`` remains imported and used only where it's still valid: as
the analytic-limit reference for the qualitative growth-vs-constant
demography effect (which doesn't depend on the disputed within_i
mass-point), not for matching an absolute predicted value.

Marked slow. Run with: .venv/bin/python -m pytest tests/illex/ -m slow
"""
import numpy as np
import pytest

from illex import model, stats, theory

N_REPS = 8
SEQ_LEN = 30_000
MU = 3e-9

ARM = "constant"
P_FINAL = 0.626                       # p_i, matches theory.P_I_DEFAULT
T_INV_ANCHOR = 5.0e5
P_START_HARD = 1.0 / (2.0 * model.trajectory_ne(ARM))   # single-founder limit
P_STARTS = [P_START_HARD, 0.05, 0.15, 0.30]
P_START_ANCHOR = 0.15
TARGET_PI_RATIO_ANCHOR = 0.755
TARGET_DXY_RATIO_ANCHOR = 1.935
EMPIRICAL_PI_RATIO = 0.744             # observed Illex chr2 pi_I/pi_S


def _mean_ratio(arm, N_fn, t_inv):
    """Legacy (soft/constant-p_inv) path only -- used solely for the
    qualitative growth-vs-constant comparison below, which doesn't depend
    on within_i's disputed mass-point assumption."""
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


def _mean_stats(arm, t_inv, p_start, n_reps=N_REPS, seed0=4000):
    """Mean pi_I/pi_S, dxy/pi_I, and E[T]_I (=pi_i/(2*mu)) over n_reps."""
    pi_ratios, dxy_ratios, et_i = [], [], []
    for rep in range(n_reps):
        sim = model.build_inversion_sim(
            arm=arm, seq_length=SEQ_LEN, t_inv=t_inv, p_inv=P_FINAL,
            gamma=1e-15, p_start=p_start, seed=seed0 + rep,
        )
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        r = stats.arrangement_stats(ts, i_nodes, s_nodes, mu=MU)
        pi_ratios.append(r["pi_i_over_pi_s"])
        dxy_ratios.append(r["dxy_over_pi_i"])
        et_i.append(r["pi_i"] / (2.0 * MU))
    return {
        "pi_i_over_pi_s": float(np.mean(pi_ratios)),
        "dxy_over_pi_i": float(np.mean(dxy_ratios)),
        "e_t_i": float(np.mean(et_i)),
    }


@pytest.fixture(scope="module")
def det_stats():
    """Mean stats at T_INV_ANCHOR (constant arm) for each p_start in
    P_STARTS, plus the "soft" legacy constant-p_inv limit -- shared across
    tests (a)-(d) below so each config is only simulated once."""
    results = {p: _mean_stats(ARM, T_INV_ANCHOR, p_start=p) for p in P_STARTS}
    results["soft"] = _mean_stats(ARM, T_INV_ANCHOR, p_start=None)
    return results


@pytest.mark.slow
def test_p_start_monotonicity(det_stats):
    """Raising p_start (a less founder-like origin) must raise pi_I/pi_S
    and lower dxy/pi_I -- msinv's deterministic-trajectory family is
    monotonic in founding frequency."""
    pi_ratios = [det_stats[p]["pi_i_over_pi_s"] for p in P_STARTS]
    dxy_ratios = [det_stats[p]["dxy_over_pi_i"] for p in P_STARTS]
    assert pi_ratios == sorted(pi_ratios), (
        f"pi_I/pi_S not increasing in p_start {P_STARTS}: {pi_ratios}"
    )
    assert dxy_ratios == sorted(dxy_ratios, reverse=True), (
        f"dxy/pi_I not decreasing in p_start {P_STARTS}: {dxy_ratios}"
    )


@pytest.mark.slow
def test_p_start_brackets_observed_pi_ratio(det_stats):
    """The hard (single-founder) and soft (constant p_inv) limits must
    bracket the empirically observed pi_I/pi_S=0.744 -- this is what
    justifies treating p_start as a fitted parameter rather than an
    arbitrary knob."""
    hard = det_stats[P_START_HARD]["pi_i_over_pi_s"]
    soft = det_stats["soft"]["pi_i_over_pi_s"]
    assert hard < 1.0, f"hard limit should give pi_I/pi_S < 1, got {hard:.3f}"
    assert soft > 1.0, f"soft limit should give pi_I/pi_S > 1, got {soft:.3f}"
    assert hard < EMPIRICAL_PI_RATIO < soft, (
        f"model family does not bracket observed {EMPIRICAL_PI_RATIO}: "
        f"hard={hard:.3f}, soft={soft:.3f}"
    )


@pytest.mark.slow
def test_constant_p_inv_lacks_monophyly_founder_limit_has_it(det_stats):
    """Semantic regression: constant p_inv (soft limit) must show msinv's
    barrier-lift behaviour (no forced monophyly -- E[T]_I pushed well past
    t_inv), while the single-founder deterministic path must show a much
    smaller E[T]_I. This pins the Task 4 finding so it cannot silently
    regress if msinv's barrier-crossing code ever changes."""
    soft_et_i = det_stats["soft"]["e_t_i"]
    hard_et_i = det_stats[P_START_HARD]["e_t_i"]
    assert soft_et_i > 1.3 * T_INV_ANCHOR, (
        f"soft (constant p_inv) E[T]_I={soft_et_i:.0f} not clearly past "
        f"t_inv={T_INV_ANCHOR:.0f} -- barrier-lift/no-monophyly signature "
        f"missing"
    )
    assert hard_et_i < T_INV_ANCHOR, (
        f"hard (single-founder) E[T]_I={hard_et_i:.0f} should stay below "
        f"t_inv={T_INV_ANCHOR:.0f}"
    )
    assert hard_et_i < 0.5 * soft_et_i, (
        f"hard E[T]_I={hard_et_i:.0f} not much smaller than soft "
        f"E[T]_I={soft_et_i:.0f}"
    )


@pytest.mark.slow
def test_regression_anchor_p_start_0_15(det_stats):
    """Fixed numeric regression point: t_inv=5e5, p_start=0.15, constant
    arm, gamma~=0 -- pi_I/pi_S~=0.755, dxy/pi_I~=1.935 (empirical targets;
    measured SEMs ~0.005-0.05 at 8 reps, so rel=0.10 is not flaky)."""
    r = det_stats[P_START_ANCHOR]
    assert r["pi_i_over_pi_s"] == pytest.approx(TARGET_PI_RATIO_ANCHOR, rel=0.10), (
        f"pi_I/pi_S={r['pi_i_over_pi_s']:.3f}, target {TARGET_PI_RATIO_ANCHOR}"
    )
    assert r["dxy_over_pi_i"] == pytest.approx(TARGET_DXY_RATIO_ANCHOR, rel=0.10), (
        f"dxy/pi_I={r['dxy_over_pi_i']:.3f}, target {TARGET_DXY_RATIO_ANCHOR}"
    )


@pytest.mark.slow
def test_growth_floor_is_lower_than_constant_in_simulation():
    """The demography effect must be visible in msinv, not just in theory.

    Uses only the legacy constant-p_inv (soft) path on both arms, and only
    compares msinv's own simulated values to each other (never to
    theory.py's absolute prediction) -- so this doesn't depend on the
    within_i mass-point dispute documented above. theory.dxy_floor is used
    solely to pick a representative t_inv per arm.
    """
    _, g_t = theory.dxy_floor(theory.N_growth)
    _, c_t = theory.dxy_floor(theory.N_const)
    assert _mean_ratio("growth", theory.N_growth, g_t) < \
           _mean_ratio("constant", theory.N_const, c_t)
