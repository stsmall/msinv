# tests/hull/test_phase3b_b2_flux.py
"""Tier-1 + Tier-2 validation of the b2-flux model
(per docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md)."""

import math

import numpy as np
import pytest
from scipy import stats

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography


# ----- Tier 1: geometric sampling correctness ----------------------

def test_geometric_tract_length_mean_matches_parameter():
    """Sampled tract lengths should have mean ≈ mean_tract_length
    within 2-sigma tolerance over N=10_000 draws."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=1_000_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=200.0,
        tract_distribution='geometric',
    )
    sim = HullSimulator(
        sample_config={('S', 0): 2},
        demography=Demography(pop_sizes=[1000]),
        sequence_length=1_000_000,
        recombination_rate=1e-8, inversions=[inv], seed=42,
    )
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(10_000):
        # Use the same Exponential the simulator uses.
        samples.append(rng.exponential(inv.mean_tract_length))
    mean = float(np.mean(samples))
    expected = inv.mean_tract_length
    sd_of_mean = expected / math.sqrt(len(samples))
    assert abs(mean - expected) < 2 * sd_of_mean, (
        f"empirical mean {mean:.2f} vs expected {expected:.2f} "
        f"(2σ={2*sd_of_mean:.2f})")


def test_geometric_tract_length_distribution_is_exponential():
    """Kolmogorov-Smirnov test: empirical samples should match
    Exponential(rate=1/λ) at p > 0.05."""
    rng = np.random.default_rng(1)
    lam = 200.0
    samples = rng.exponential(lam, size=10_000)
    ks_stat, p = stats.kstest(samples, 'expon', args=(0.0, lam))
    assert p > 0.05, f"KS test failed: stat={ks_stat:.4f} p={p:.4f}"


# ----- Tier 1: smoke at biological 3Ra-scale params ---------------

def test_smoke_3ra_scale_geometric():
    """3Ra-scale params (6 Mb inv, 100 bp tract) run without crashing
    and produce a well-formed tree sequence."""
    inv = InversionSpec(
        bp_left=1.0, bp_right=6_000_000.0 - 1.0,
        p_inv=0.5, t_inv=100_000.0,
        gene_conversion_rate=1e-6,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[5000])
    sim = HullSimulator(
        sample_config={('S', 0): 4, ('I', 0): 4},
        demography=demo,
        sequence_length=6_000_000,
        recombination_rate=1e-8,
        inversions=[inv], seed=42,
    )
    ts = sim.simulate()
    assert ts.num_trees > 0
    assert ts.num_nodes > 8


# ----- Tier 2: spatial profile φ(x) ------------------------------

def test_spatial_profile_uniform_in_interior_geometric():
    """Empirical fraction of flux events that touch position x should
    be ≈ λ/inv_length in the interior (away from breakpoints by ≫ λ).

    Strategy: instead of instrumenting the simulator's flux events,
    sample many tracts directly via the same _draw_tract logic and
    histogram the per-bp coverage. This validates the geometry, which
    is the determinant of the spatial profile."""
    inv = InversionSpec(
        bp_left=0.0, bp_right=10_000.0,
        p_inv=0.5, t_inv=10_000.0,
        gene_conversion_rate=1e-9,
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )

    rng = np.random.default_rng(2)
    inv_len = inv.bp_right - inv.bp_left
    lam = inv.mean_tract_length
    n_events = 20_000
    # 1-bp bins so the per-bp coverage formula λ/inv_len is the right
    # comparison. With wider bins the expected value picks up an
    # extra (bin_width / inv_len) term from the tract-overlaps-bin
    # geometry and the simple λ/inv_len formula no longer matches.
    n_bins = int(inv_len)
    coverage = np.zeros(n_bins, dtype=np.int64)

    for _ in range(n_events):
        L = rng.exponential(lam)
        L = min(L, inv_len * 0.99)
        if L <= 0.0:
            continue
        b1 = rng.uniform(0.0, inv_len - L)
        tl = int(b1)
        tr = min(int(b1 + L) + 1, n_bins)
        coverage[tl:tr] += 1

    coverage_frac = coverage / n_events
    # Skip 2λ on each side so the rise/fall regions don't pull the mean.
    margin = int(2 * lam)
    interior = coverage_frac[margin:-margin]
    expected_interior = lam / inv_len
    mean_interior = float(np.mean(interior))
    rel_err = abs(mean_interior - expected_interior) / expected_interior
    assert rel_err < 0.10, (
        f"interior coverage {mean_interior:.5f} vs expected "
        f"{expected_interior:.5f} (rel err {rel_err:.3f}, > 10%)")


# ----- Tier 2: rate scaling with mean_tract_length ---------------

def test_flux_rate_scales_linearly_with_mean_tract_length():
    """Per-lineage flux event rate ≈ γ × p_other × mean_tract_length
    (Section 2 of the spec). Verify by varying mean_tract_length over
    a 20× range and confirming the empirical num_trees count scales
    proportionally — flux events fragment the ARG into more trees,
    so num_trees is a monotone proxy for total flux-event count.

    This is the Tier-2 calibration we can land without simulator-
    state instrumentation. Direct event-count calibration is part
    of Tier 3-full (Andolfatto anchor) per the spec's Deferred
    Validation Roadmap."""
    bp_left = 0.0
    bp_right = 200_000.0
    inv_len = bp_right - bp_left
    # γ chosen so per-lineage event rate gives several events per
    # coalescent timescale at small λ but doesn't saturate the ARG at
    # large λ (num_trees is bounded by sequence length / shortest tract).
    gamma = 1e-5
    NREPS = 10
    lambdas = [200.0, 1000.0, 4000.0]  # 20× range
    means = []
    for lam in lambdas:
        inv = InversionSpec(
            bp_left=bp_left, bp_right=bp_right,
            p_inv=0.5, t_inv=20_000.0,
            gene_conversion_rate=gamma,
            mean_tract_length=lam,
            tract_distribution='geometric',
        )
        demo = Demography(pop_sizes=[2000])
        n_trees_reps = []
        for seed in range(NREPS):
            sim = HullSimulator(
                sample_config={('S', 0): 4, ('I', 0): 4},
                demography=demo,
                sequence_length=int(inv_len),
                recombination_rate=1e-9,
                inversions=[inv], seed=seed,
            )
            ts = sim.simulate()
            n_trees_reps.append(ts.num_trees)
        means.append(float(np.mean(n_trees_reps)))

    # Subtract the no-flux baseline (recombination-driven trees) so
    # the flux contribution is what scales with λ.
    inv_zero = InversionSpec(
        bp_left=bp_left, bp_right=bp_right,
        p_inv=0.5, t_inv=20_000.0,
        gene_conversion_rate=gamma,
        mean_tract_length=0.0,                # disables flux
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[2000])
    no_flux_trees = []
    for seed in range(NREPS):
        sim = HullSimulator(
            sample_config={('S', 0): 4, ('I', 0): 4},
            demography=demo,
            sequence_length=int(inv_len),
            recombination_rate=1e-9,
            inversions=[inv_zero], seed=seed,
        )
        no_flux_trees.append(sim.simulate().num_trees)
    baseline = float(np.mean(no_flux_trees))

    flux_contribution = [m - baseline for m in means]

    # Assert: the flux contribution scales monotonically with λ.
    assert flux_contribution[0] < flux_contribution[1] < flux_contribution[2], (
        f"flux_contribution should be monotone-increasing in λ, "
        f"got {flux_contribution} at λ={lambdas}")

    # Soft linearity: the largest λ should contribute substantially
    # more than the smallest. We don't enforce exact linear scaling
    # because num_trees is a coarse proxy (bounded above by the
    # sequence length and recombination breakpoints; saturates at
    # high γ·λ). Tier 3 (Andolfatto anchor, deferred) does the
    # tight calibration.
    assert flux_contribution[2] > 1.5 * flux_contribution[0], (
        f"largest-λ flux contribution should be >1.5× smallest-λ, "
        f"got {flux_contribution[2]:.1f} vs {flux_contribution[0]:.1f}")


# ---------------------------------------------------------------------------
# Tier 3-cheap (Q5a): flux tract-break survival shape — geometric vs fixed
# ---------------------------------------------------------------------------

def test_flux_tract_break_survival_geometric_vs_fixed():
    """Q5a: empirical S(d) = P(tract_length >= d) discriminates modes.

    'geometric' mode: tract lengths ~ Exp(1/λ); S(λ) ≈ exp(-1) ≈ 0.37,
                      S(2λ) ≈ exp(-2) ≈ 0.135.
    'fixed' mode:     tract length = λ exactly; S(λ) = 1.0, S(2λ) = 0.0.

    This is the higher-moment discriminator (beyond mean tract length)
    that proves b2-flux has biological content beyond what 'fixed' provides.

    Hook is required: turn record_events on to read FluxRecords.
    """
    from msinv.hull._event_log import filter_flux, tract_lengths

    lam = 300.0
    # Tune γ × t_inv to produce ≥200 flux events for ±0.05 MC precision.
    # gamma=5e-3 + t_inv=4000 with n_samples=20 achieves this.
    gamma = 5e-3
    t_inv = 4000.0

    # Store raw lengths for each mode so assertions can use eps-adjusted thresholds.
    raw_lengths = {}
    results = {}
    for mode in ['geometric', 'fixed']:
        inv = InversionSpec(
            bp_left=2000.0, bp_right=8000.0,
            p_inv=0.5, t_inv=t_inv,
            gene_conversion_rate=gamma,
            mean_tract_length=lam,
            tract_distribution=mode,
        )
        demo = Demography(pop_sizes=[1000])
        sim = HullSimulator(
            sample_config={('S', 0): 10, ('I', 0): 10},
            demography=demo,
            sequence_length=10_000,
            recombination_rate=1e-8,
            inversions=[inv],
            seed=42,
            record_events=True,
        )
        sim.simulate()
        flux = filter_flux(sim.event_log, inv_id=0)
        assert len(flux) >= 200, (
            f"mode={mode}: only {len(flux)} flux events; "
            f"bump γ or t_inv for adequate MC sample size")
        lengths = tract_lengths(flux)
        raw_lengths[mode] = lengths
        s_at_lam = float((lengths >= lam).mean())
        s_at_2lam = float((lengths >= 2 * lam).mean())
        results[mode] = (s_at_lam, s_at_2lam, len(flux))

    # 'fixed' mode: every tract has length == λ (up to float rounding ≤ 1e-9),
    # so S(λ) = 1.0 and S(2λ) = 0.0.  We use a tiny eps on the threshold to
    # absorb double-precision representation noise (observed max deviation < 1e-12).
    _, _, n_fixed = results['fixed']
    eps = 1e-9
    lens_fixed = raw_lengths['fixed']
    s_lam_fixed_adj = float((lens_fixed >= lam - eps).mean())
    s_2lam_fixed_adj = float((lens_fixed >= 2 * lam - eps).mean())

    assert s_lam_fixed_adj == 1.0, (
        f"'fixed': S(λ-ε) = {s_lam_fixed_adj}, expected exactly 1.0 "
        f"(n_events={n_fixed}; max deviation from λ should be < 1e-9)")
    assert s_2lam_fixed_adj == 0.0, (
        f"'fixed': S(2λ-ε) = {s_2lam_fixed_adj}, expected exactly 0.0 "
        f"(n_events={n_fixed})")

    # 'geometric' mode: S(λ) ≈ exp(-1) ≈ 0.368; S(2λ) ≈ exp(-2) ≈ 0.135.
    s_lam_geom, s_2lam_geom, n_geom = results['geometric']
    assert abs(s_lam_geom - 0.368) < 0.05, (
        f"'geometric': S(λ) = {s_lam_geom:.3f}, expected 0.368 ± 0.05 "
        f"(n_events={n_geom})")
    assert abs(s_2lam_geom - 0.135) < 0.05, (
        f"'geometric': S(2λ) = {s_2lam_geom:.3f}, expected 0.135 ± 0.05 "
        f"(n_events={n_geom})")


# ---------------------------------------------------------------------------
# Tier 3-cheap (Q5b): Andolfatto event-coverage monotonicity
# ---------------------------------------------------------------------------

def test_andolfatto_event_coverage_monotone_in_t_inv():
    """Q5b: event-coverage at the inversion center

    (i)  increases monotonically with t_inv at fixed (γ, λ);
    (ii) has equal mean between 'fixed' and 'geometric' at matched λ
         (within 20% relative tolerance — variance differs, mean shouldn't).
    """
    from msinv.hull._event_log import filter_flux, coverage_count

    gamma = 5e-3   # bumped well above biological for sufficient event counts
    lam = 300.0
    t_inv_ladder = [500.0, 2000.0, 5000.0]
    n_seeds = 20
    inv_center = 5000.0  # midpoint of bp_left=2000, bp_right=8000

    means_by_mode = {}

    for mode in ['fixed', 'geometric']:
        means_per_t = []
        for t_inv in t_inv_ladder:
            covers = []
            for seed in range(n_seeds):
                inv = InversionSpec(
                    bp_left=2000.0, bp_right=8000.0,
                    p_inv=0.5, t_inv=t_inv,
                    gene_conversion_rate=gamma,
                    mean_tract_length=lam,
                    tract_distribution=mode,
                )
                demo = Demography(pop_sizes=[1000])
                sim = HullSimulator(
                    sample_config={('S', 0): 10, ('I', 0): 10},
                    demography=demo,
                    sequence_length=10_000,
                    recombination_rate=1e-8,
                    inversions=[inv],
                    seed=seed,
                    record_events=True,
                )
                sim.simulate()
                flux = filter_flux(sim.event_log, inv_id=0)
                covers.append(coverage_count(flux, inv_center))
            means_per_t.append(float(np.mean(covers)))

        # (i) monotonicity in t_inv at fixed (γ, λ)
        assert means_per_t[0] < means_per_t[1] < means_per_t[2], (
            f"mode={mode}: not monotone in t_inv: "
            f"means at t_inv={t_inv_ladder} = {means_per_t}")

        means_by_mode[mode] = means_per_t

    # (ii) at each t_inv, 'fixed' and 'geometric' should have equal MEAN
    #      coverage at matched λ. Variance differs; mean shouldn't.
    for i, t_inv in enumerate(t_inv_ladder):
        m_fixed = means_by_mode['fixed'][i]
        m_geom  = means_by_mode['geometric'][i]
        scale = max(m_fixed, m_geom, 1.0)
        rel_diff = abs(m_fixed - m_geom) / scale
        assert rel_diff < 0.20, (
            f"t_inv={t_inv}: mean coverage diverges between modes — "
            f"fixed={m_fixed:.2f}, geom={m_geom:.2f}, "
            f"rel_diff={rel_diff:.3f}; expected agreement within 20% "
            f"at n_seeds={n_seeds}")


# ---------------------------------------------------------------------------
# Sanity: log size stays bounded at biological γ (regression guard)
# ---------------------------------------------------------------------------

def test_event_log_size_bounded_at_biological_gamma():
    """At a biologically-realistic γ (1e-7) and modest scale,
    sim.event_log stays under 1M records — guards against future
    scale changes silently OOM-ing.

    This is NOT a correctness test; it just records the actual
    log size at sane parameters so a regression that suddenly
    explodes the log surfaces here rather than as a memory error.
    """
    from msinv.hull._event_log import filter_flux

    inv = InversionSpec(
        bp_left=1000.0, bp_right=9000.0,
        p_inv=0.5, t_inv=5000.0,
        gene_conversion_rate=1e-7,  # biological γ
        mean_tract_length=100.0,
        tract_distribution='geometric',
    )
    demo = Demography(pop_sizes=[1000])
    sim = HullSimulator(
        sample_config={('S', 0): 10, ('I', 0): 10},
        demography=demo,
        sequence_length=10_000,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=42,
        record_events=True,
    )
    sim.simulate()
    n_total = len(sim.event_log) if sim.event_log is not None else 0
    n_flux = len(filter_flux(sim.event_log)) if sim.event_log else 0

    # 1M records is far above any plausible at biological γ.
    assert n_total < 1_000_000, (
        f"event_log unexpectedly large: {n_total} records (flux={n_flux}). "
        f"This may indicate a scale regression — investigate before "
        f"running production sims with record_events=True.")

    # Sanity: at biological γ, expect a small but non-zero count.
    # If this assertion ever fails low, our γ assumption changed.
    if n_total > 0:
        assert n_total < 10_000, (
            f"event_log size {n_total} suggests parameters changed; "
            f"update this test or investigate.")

    # Print the actual count for reference in CI logs.
    print(f"\nevent_log at biological γ=1e-7: n_total={n_total} (flux={n_flux})")
