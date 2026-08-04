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

  (a) monotonicity of pi_I/pi_S and dxy/pi_I in p_start (constant arm, then
      repeated at reduced cost on the growth arm so trajectory_ne("growth")
      is actually exercised);
  (b) the hard/soft limits bracket the empirically observed pi_I/pi_S
      (constant and growth arms);
  (c) the semantic distinction this task discovered (constant p_inv lacks
      monophyly at t_inv; the founder limit has it) is pinned as a
      regression guard;
  (d) a fixed numeric regression anchor at a specific (t_inv, p_start) --
      a simulation snapshot for regression-catching, not an empirical
      value (see ANCHOR_PI_RATIO/ANCHOR_DXY_RATIO below);
  (e) the flux-geometry invariant (mean_tract_length/inv_length ==
      TRACT_FRACTION) holds on the trajectory path specifically.

``theory.py`` remains imported and used only where it's still valid: as
the analytic-limit reference for the qualitative growth-vs-constant
demography effect (which doesn't depend on the disputed within_i
mass-point), not for matching an absolute predicted value.

Marked slow. Run with: .venv/bin/python -m pytest tests/illex/ -m slow
"""
import numpy as np
import pytest

from illex import empirical, model, stats, theory

N_REPS = 8
SEQ_LEN = 30_000
MU = 3e-9

ARM = "constant"
P_FINAL = 0.626                       # p_i, matches theory.P_I_DEFAULT
T_INV_ANCHOR = 5.0e5
P_START_HARD = 1.0 / (2.0 * model.trajectory_ne(ARM))   # single-founder limit
P_STARTS = [P_START_HARD, 0.05, 0.15, 0.30]
P_START_ANCHOR = 0.15

# These two are a SIMULATION SNAPSHOT (one particular msinv run at
# t_inv=5e5, p_start=0.15), NOT empirical observations -- do not confuse
# them with the package's actual empirical values (see illex.empirical and
# tests/illex/test_theory.py). They exist purely to catch a future
# regression in msinv's or model.py's behaviour at this one fixed
# configuration.
#
# INTERVAL-RESTRICTED (C1 fix, task-final-fixes-report.md): measured with
# stats.arrangement_stats(..., interval=model.inversion_interval(sim)), i.e.
# restricted to the inversion body only, NOT the whole simulated sequence
# (the whole-sequence values at this same config were ~0.774/1.888 --
# diluted by the panmictic collinear flank outside the inversion body, see
# model.MARGIN_FRACTION). Regenerated at N_REPS=8, seed0=4000: pi_I/pi_S =
# 0.7015 (SEM 0.0107), dxy/pi_I = 2.2889 (SEM 0.0138).
ANCHOR_PI_RATIO = 0.7015
ANCHOR_DXY_RATIO = 2.2889

EMPIRICAL_PI_RATIO = empirical.PI_I_OVER_PI_S  # observed Illex chr2 pi_I/pi_S

# Growth-arm coverage (Finding 2b): smaller n and fewer reps than the
# constant-arm tests above, since the growth arm costs ~5 s/rep vs.
# ~0.5 s/rep for constant (see task-4-report.md bench numbers).
GROWTH_N_REPS = 4
GROWTH_N_SAMPLES = 30
GROWTH_P_START_HARD = 1.0 / (2.0 * model.trajectory_ne("growth"))
GROWTH_P_STARTS = [GROWTH_P_START_HARD, 0.15, 0.30]


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
        r = stats.arrangement_stats(
            ts, i_nodes, s_nodes, interval=model.inversion_interval(sim)
        )
        vals.append(r["dxy_over_pi_i"])
    return float(np.mean(vals))


def _mean_stats(arm, t_inv, p_start, n_reps=N_REPS, seed0=4000,
                n_i=100, n_s=100):
    """Mean pi_I/pi_S, dxy/pi_I, and E[T]_I (=pi_i/(2*mu)) over n_reps."""
    pi_ratios, dxy_ratios, et_i = [], [], []
    for rep in range(n_reps):
        sim = model.build_inversion_sim(
            arm=arm, seq_length=SEQ_LEN, t_inv=t_inv, p_inv=P_FINAL,
            gamma=1e-15, p_start=p_start, seed=seed0 + rep,
            n_i=n_i, n_s=n_s,
        )
        ts = sim.simulate()
        i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
        r = stats.arrangement_stats(
            ts, i_nodes, s_nodes, mu=MU,
            interval=model.inversion_interval(sim),
        )
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


@pytest.fixture(scope="module")
def det_stats_growth():
    """Same shape as det_stats, but on the growth arm (where
    trajectory_ne("growth") returns theory.N_ANC -- otherwise untested).
    Fewer reps and smaller sample sizes than det_stats to keep runtime sane
    given the growth arm's higher per-rep cost."""
    results = {
        p: _mean_stats("growth", T_INV_ANCHOR, p_start=p,
                       n_reps=GROWTH_N_REPS, seed0=5000,
                       n_i=GROWTH_N_SAMPLES, n_s=GROWTH_N_SAMPLES)
        for p in GROWTH_P_STARTS
    }
    results["soft"] = _mean_stats(
        "growth", T_INV_ANCHOR, p_start=None,
        n_reps=GROWTH_N_REPS, seed0=5000,
        n_i=GROWTH_N_SAMPLES, n_s=GROWTH_N_SAMPLES,
    )
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
    arm, gamma~=0, INTERVAL-RESTRICTED to the inversion body (C1 fix) --
    pi_I/pi_S~=0.7015, dxy/pi_I~=2.2889. These are a simulation snapshot
    (see ANCHOR_PI_RATIO/ANCHOR_DXY_RATIO above), not the package's
    empirical values -- this test only guards against silent regressions
    in msinv's or model.py's behaviour at this one fixed configuration.
    Measured SEMs ~0.011/0.014 at 8 reps, so rel=0.10 is not flaky."""
    r = det_stats[P_START_ANCHOR]
    assert r["pi_i_over_pi_s"] == pytest.approx(ANCHOR_PI_RATIO, rel=0.10), (
        f"pi_I/pi_S={r['pi_i_over_pi_s']:.3f}, anchor {ANCHOR_PI_RATIO}"
    )
    assert r["dxy_over_pi_i"] == pytest.approx(ANCHOR_DXY_RATIO, rel=0.10), (
        f"dxy/pi_I={r['dxy_over_pi_i']:.3f}, anchor {ANCHOR_DXY_RATIO}"
    )


@pytest.mark.slow
def test_p_start_monotonicity_and_bracketing_growth_arm(det_stats_growth):
    """Growth-arm coverage of the trajectory path (Finding 2b): the
    monotonicity and bracketing properties verified on the constant arm
    above must also hold on the growth arm, where trajectory_ne("growth")
    returns theory.N_ANC. Without this, the growth arm's trajectory path
    (and its n_e choice) would be exercised nowhere in the suite."""
    pi_ratios = [det_stats_growth[p]["pi_i_over_pi_s"] for p in GROWTH_P_STARTS]
    dxy_ratios = [det_stats_growth[p]["dxy_over_pi_i"] for p in GROWTH_P_STARTS]
    assert pi_ratios == sorted(pi_ratios), (
        f"growth: pi_I/pi_S not increasing in p_start {GROWTH_P_STARTS}: "
        f"{pi_ratios}"
    )
    assert dxy_ratios == sorted(dxy_ratios, reverse=True), (
        f"growth: dxy/pi_I not decreasing in p_start {GROWTH_P_STARTS}: "
        f"{dxy_ratios}"
    )

    hard = det_stats_growth[GROWTH_P_START_HARD]["pi_i_over_pi_s"]
    soft = det_stats_growth["soft"]["pi_i_over_pi_s"]
    assert hard < 1.0, f"growth hard limit should give pi_I/pi_S < 1, got {hard:.3f}"
    assert soft > 1.0, f"growth soft limit should give pi_I/pi_S > 1, got {soft:.3f}"


def test_trajectory_path_flux_geometry_invariant():
    """The flux-geometry invariant (mean_tract_length / inv_length ==
    TRACT_FRACTION) must hold on the trajectory (p_start set) path
    specifically, not just the legacy path -- both build the InversionSpec
    independently in build_inversion_sim, so this guards against the two
    branches drifting apart. Not marked slow: builds the sim but never
    calls .simulate()."""
    sim = model.build_inversion_sim(
        arm=ARM, seq_length=SEQ_LEN, t_inv=T_INV_ANCHOR, p_inv=P_FINAL,
        gamma=1e-15, p_start=P_START_ANCHOR, seed=1,
    )
    spec = sim.inversions[0]
    inv_len = spec.bp_right - spec.bp_left
    assert spec.mean_tract_length / inv_len == pytest.approx(model.TRACT_FRACTION)


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
