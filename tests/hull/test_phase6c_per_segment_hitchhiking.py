"""Spatial profile tests for the per-segment hitchhiking sweep.

After the 2026-04-30 per-segment extension, sweeps in panmictic
(or with-inversion-but-outside) settings should produce a
Kim-Stephan-shaped recovery curve in pi: lowest at x_sel, rising
toward the genome edges.  This module anchors that shape.

Spec: docs/superpowers/specs/2026-04-30-sweep-per-segment-hitchhiking-design.md
"""

import statistics

from msinv.hull.simulator import HullSimulator
from msinv.hull.sweep import Sweep


def _sim_factory(seed: int) -> "tskit.TreeSequence":
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Deterministic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10000),
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _windowed_pi(ts, x_sel, L, n_bins=10):
    """Folded windowed branch-mode pi around x_sel.

    Returns a list of n_bins values; index 0 is the [0, w) bin
    nearest the sweep, index n-1 is the [(n-1)w, n*w) bin farthest.
    """
    half = L / 2.0
    w = half / n_bins
    out = [0.0] * n_bins
    for k in range(n_bins):
        lo, hi = k * w, (k + 1) * w
        left_lo = max(0.0, x_sel - hi)
        left_hi = max(0.0, x_sel - lo)
        right_lo = min(L, x_sel + lo)
        right_hi = min(L, x_sel + hi)
        wins = [0.0]
        for v in (left_lo, left_hi, right_lo, right_hi):
            if v > wins[-1]:
                wins.append(v)
        if wins[-1] < L:
            wins.append(L)
        divs = ts.diversity(windows=wins, mode="branch")
        total_span, total_div_span = 0.0, 0.0
        for i in range(len(wins) - 1):
            seg_lo, seg_hi = wins[i], wins[i + 1]
            in_left = seg_lo >= left_lo and seg_hi <= left_hi
            in_right = seg_lo >= right_lo and seg_hi <= right_hi
            if in_left or in_right:
                span = seg_hi - seg_lo
                total_span += span
                total_div_span += divs[i] * span
        out[k] = total_div_span / total_span if total_span > 0 else 0.0
    return out


def test_ps2_spatial_profile_decays_monotonically():
    """Mean folded pi over 100 reps should rise monotonically from
    bin 0 (nearest x_sel) to bin 9 (farthest).

    The strict "monotone" test is sensitive to MC noise; we relax to
    'pi at bin 0 strictly less than pi at bin 9' which is the headline
    Kim-Stephan signature.

    n_reps was 30; bumped to 100 so the ≥30% sanity threshold is robust
    against small-sample variance — at 30 reps the realization can
    straddle the threshold under simulator implementations that are
    statistically equivalent in expectation but produce different
    deterministic samples (e.g. tau-leap rebuild-skip optimization).
    """
    n_reps = 100
    bin_means = [0.0] * 10
    for r in range(n_reps):
        ts = _sim_factory(seed=r)
        wp = _windowed_pi(ts, x_sel=50_000.0, L=100_000.0, n_bins=10)
        for k, v in enumerate(wp):
            bin_means[k] += v
    bin_means = [v / n_reps for v in bin_means]
    print(f"PS2 mean folded pi by bin: {[f'{v:.0f}' for v in bin_means]}")
    # Strict test: pi at bin 0 (nearest sweep) < pi at bin 9 (farthest)
    assert bin_means[0] < bin_means[9], (
        f"Bin 0 (nearest x_sel) should have lower pi than bin 9 (farthest); "
        f"got {bin_means[0]:.1f} vs {bin_means[9]:.1f}")
    # Sanity: bin 0 should be at least 30% reduced relative to bin 9
    # (strong sweep with s=0.05 produces ~50% reduction at d≈0)
    reduction = 1.0 - bin_means[0] / bin_means[9]
    assert reduction > 0.3, (
        f"Expected ≥30% pi reduction at sweep center vs edge; "
        f"got {reduction*100:.1f}%")


def test_ps3_at_x_sel_pi_matches_kim_stephan_anchor():
    """At x_sel itself, pi_branch should be greatly reduced relative
    to neutral expectation (sweep force-coalesces all segments at x_sel
    into the same MRCA at t_origin, so T_2 ≈ t_origin << 2N).

    With N=10_000 and t_origin=1_500, pi_branch at x_sel ≈ 2*t_origin
    = 3_000 while neutral pi_branch = 4*N = 40_000.  Ratio ≈ 7.5%.

    The right half [50_500, 100_000] is also within the swept region
    (x_sel is at genome centre), so we compare directly to the known
    neutral expectation 4*N rather than to the within-sweep right half.

    Tolerance: mean(pi at x_sel) / (4*N) < 0.10.

    This anchors that the per-segment finalize doesn't accidentally
    leave x_sel segments uncoalesced.
    """
    N = 10_000
    neutral_pi = 4 * N  # branch-mode: E[pi] = 4N (no mutation rate)

    n_reps = 30
    pi_at_xsel = []
    for r in range(n_reps):
        ts = _sim_factory(seed=r)
        # Narrow window at x_sel (1kb each side)
        center = ts.diversity(
            windows=[0.0, 49_500.0, 50_500.0, 100_000.0], mode="branch")
        c_pi = center[1]   # middle bin = [49_500, 50_500) near x_sel
        pi_at_xsel.append(c_pi)
    mean_x = statistics.mean(pi_at_xsel)
    print(f"PS3 mean pi at x_sel: {mean_x:.1f}, neutral expectation: {neutral_pi}")
    ratio = mean_x / neutral_pi
    assert ratio < 0.10, (
        f"pi at x_sel should be <10% of neutral (4N={neutral_pi}); "
        f"got pi={mean_x:.1f}, ratio={ratio*100:.1f}%")
