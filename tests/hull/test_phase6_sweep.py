"""Phase 6 — selection sweeps (joint forward WF rewrite).

Tests against the new Sweep API (see docs/superpowers/specs/
2026-04-28-sweep-rewrite-design.md).

Replaces the prior Hudson-Kaplan tests, which targeted
``target_class='P'`` and were rejected by the Rust backend.

T1, T2 are active and exercise the trajectory directly via PyO3.
T3, T4, T5 are skipped pending simulator-side `apply_sweep` dispatch
(deferred per Task 13 of the sweep-rewrite plan).
"""

import math

import pytest

from msinv.hull.sweep import Sweep


def _logistic_pt_discrete(t, s, f0):
    """Discrete-time logistic: f0*(1+s)^t / (1 - f0 + f0*(1+s)^t)."""
    growth = (1.0 + s) ** t
    return f0 * growth / (1.0 - f0 + f0 * growth)


def test_t1_det_logistic_per_gen_within_1e6():
    """T1: DetOnly, panmictic-on-S, no flux. Trajectory matches discrete logistic."""
    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="S", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=500.0, f0=0.01,
        partial_sweep_final_freq=1.0,
    )
    rust_sw = sw.to_rust()
    rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.0], pop_sizes=[10_000.0])
    samples = rust_sw.trajectory_samples()
    # Spot-check at 25%, 50%, 75% along the trajectory
    for frac in [0.25, 0.5, 0.75]:
        i = int(len(samples) * frac)
        sample_t, freq = samples[i]
        forward_t = sw.t_origin - sample_t
        observed = freq[0][1]   # (S, A) class
        expected = _logistic_pt_discrete(forward_t, sw.s, sw.f0)
        assert abs(observed - expected) < 1e-9, (
            f"at frac {frac}, t={sample_t}: obs={observed}, exp={expected}"
        )


def test_t2_stoch_fixation_proportion():
    """T2: Stoch, de novo. Fixation proportion approx 2s/(1+s) within MC error over 200 reps."""
    s = 0.05
    expected = 2 * s / (1 + s)
    n_reps = 200
    fixations = 0
    for r in range(n_reps):
        sw = Sweep(
            x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="S", target_inv=0,
            mode="Stochastic", s=s, t_origin=2_000.0, f0=1.0/(2*5_000),
            partial_sweep_final_freq=0.95, seed=r + 1,
        )
        rust_sw = sw.to_rust()
        rust_sw.build_trajectory(n_pops=1, p_inv_init=[0.0], pop_sizes=[5_000.0])
        if rust_sw.final_a_freq() > 0.5:
            fixations += 1
    observed = fixations / n_reps
    sigma = math.sqrt(expected * (1 - expected) / n_reps)
    # Loose 4-sigma bound; the simulator's discrete WF gives a slightly
    # different fixation rate than the Haldane approximation 2s/(1+s),
    # so widen if needed.
    assert abs(observed - expected) < 4 * sigma, (
        f"obs fix prop = {observed}, expected {expected} +/- {sigma}"
    )


def test_t3_hitchhiking_footprint_kim_stephan():
    """T3: pi reduction at distance d from x_sel matches Kim-Stephan within 25%."""
    import numpy as np
    from msinv.hull import HullSimulator
    from msinv.hull.demography import Demography
    from msinv.hull.sweep import Sweep

    Ne = 10_000.0
    s = 0.05
    L = 100_000.0
    x_sel = L / 2
    t_origin = (2.0 / s) * math.log(2.0 * Ne)   # sojourn time
    n_samples = 30
    n_reps = 15
    r = 1e-7

    # Distance points: near (mostly hitchhiked) and far (mostly escaped).
    distances = [500.0, 5_000.0]

    def run_pair(rep_seed, with_sweep):
        sw_kwargs = dict(
            sample_config={('S', 0): n_samples},
            demography=Demography(pop_sizes=[Ne]),
            sequence_length=L,
            recombination_rate=r,
            seed=rep_seed,
        )
        if with_sweep:
            sw = Sweep(
                x_sel=x_sel, tau=0.0, origin_pop=0, origin_kary="S",
                target_inv=0,
                mode="Deterministic", s=s, t_origin=t_origin,
                f0=1.0/(2*Ne), partial_sweep_final_freq=1.0,
            )
            sim = HullSimulator(sweeps=[sw], **sw_kwargs)
        else:
            sim = HullSimulator(**sw_kwargs)
        return sim.simulate()

    pi_reductions = []
    for d in distances:
        with_pi, no_pi = [], []
        for rep in range(n_reps):
            ts_w = run_pair(rep + 1, True)
            ts_n = run_pair(rep + 1, False)
            w_lo = max(0.0, x_sel + d - 100.0)
            w_hi = min(L, x_sel + d + 100.0)
            # tskit diversity(windows=...) needs breakpoints spanning [0, L]
            wins = [0.0, w_lo, w_hi, L]
            with_pi.append(ts_w.diversity(windows=wins, mode='branch')[1])
            no_pi.append(ts_n.diversity(windows=wins, mode='branch')[1])
        pi_w, pi_n = np.mean(with_pi), np.mean(no_pi)
        if pi_n > 0:
            pi_reductions.append((d, 1.0 - pi_w / pi_n))

    # Kim-Stephan: f_pi(d) = 1 - exp(-2 * alpha * r * d / s), alpha = 2 Ne s
    alpha = 2.0 * Ne * s
    for d, observed in pi_reductions:
        predicted = 1.0 - math.exp(-2.0 * alpha * r * d / s)
        # Tier-1 anchor tolerance: 25% relative or 0.10 absolute, whichever larger.
        # Add extra 0.05 slack for MC noise at n_reps=15.
        tol = max(0.25 * abs(predicted), 0.10) + 0.05
        assert abs(observed - predicted) < tol, (
            f"d={d}: observed reduction {observed:.3f}, "
            f"predicted {predicted:.3f}, tol={tol:.3f}"
        )




def test_t4_soft_sweep_partial_diversity_reduction():
    """T4: f0=0.05, π at x_sel approx (1 - 1/K), K = round(1/f0)."""
    import numpy as np
    from msinv.hull import HullSimulator
    from msinv.hull.demography import Demography
    from msinv.hull.sweep import Sweep

    Ne = 10_000.0
    s = 0.05
    L = 100_000.0
    x_sel = L / 2
    t_origin = (2.0 / s) * math.log(2.0 * Ne)
    f0 = 0.05
    K = round(1.0 / f0)
    n_samples = 50
    n_reps = 12

    def run_pair(rep_seed, with_sweep):
        sw_kwargs = dict(
            sample_config={('S', 0): n_samples},
            demography=Demography(pop_sizes=[Ne]),
            sequence_length=L,
            recombination_rate=1e-12,   # near-zero so soft-sweep signature
                                         # isn't washed out by recomb
            seed=rep_seed,
        )
        if with_sweep:
            sw = Sweep(
                x_sel=x_sel, tau=0.0, origin_pop=0, origin_kary="S",
                target_inv=0,
                mode="Deterministic", s=s, t_origin=t_origin, f0=f0,
                partial_sweep_final_freq=1.0,
            )
            sim = HullSimulator(sweeps=[sw], **sw_kwargs)
        else:
            sim = HullSimulator(**sw_kwargs)
        return sim.simulate()

    with_pi, no_pi = [], []
    for rep in range(n_reps):
        ts_w = run_pair(rep + 1, True)
        ts_n = run_pair(rep + 1, False)
        w_lo = max(0.0, x_sel - 100.0)
        w_hi = min(L, x_sel + 100.0)
        wins = [0.0, w_lo, w_hi, L]
        with_pi.append(ts_w.diversity(windows=wins, mode='branch')[1])
        no_pi.append(ts_n.diversity(windows=wins, mode='branch')[1])

    pi_w, pi_n = np.mean(with_pi), np.mean(no_pi)
    observed_reduction = 1.0 - pi_w / pi_n if pi_n > 0 else 0
    expected_reduction = 1.0 - 1.0 / K
    tol = max(0.25 * expected_reduction, 0.15)
    assert abs(observed_reduction - expected_reduction) < tol, (
        f"observed reduction {observed_reduction:.3f}, "
        f"expected {expected_reduction:.3f} (K={K})"
    )


def test_t5_partial_sweep_final_freq_assignment():
    """T5: c=0.5 → ~50% of lineages assigned to swept fraction."""
    from msinv.hull import HullSimulator
    from msinv.hull.demography import Demography

    sw = Sweep(
        x_sel=50_000.0, tau=0.0, origin_pop=0, origin_kary="S", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=2_000.0, f0=0.001,
        partial_sweep_final_freq=0.5,
    )
    n_samples = 400
    sim = HullSimulator(
        sample_config={('S', 0): n_samples},
        demography=Demography(pop_sizes=[10_000.0]),
        sequence_length=100_000.0,
        recombination_rate=1e-12,
        sweeps=[sw],
        seed=42,
    )
    sim.simulate()
    a_count = sim.sweep_a_count
    observed = a_count / n_samples
    assert abs(observed - 0.5) < 0.10, f"observed A frac = {observed}, expected ~0.5"
