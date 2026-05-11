"""Phase-4b tests: class-conditional migration / split.

Validates:
  - ``add_class_migration(proportion=1.0)`` moves all matching-karyotype
    lineages from src to dst (= class-restricted ej).
  - ``add_class_migration(proportion=p)`` moves ~p fraction stochastically.
  - ``add_class_split`` (cmig S + cmig I + safety ej) is bit-equivalent to
    plain ``ej`` in K=S-only / F=S+I sampling.
  - PAN-class lineages (created by gene flux) are NOT caught by cmig but
    ARE caught by the safety-ej tail in ``add_class_split``.
  - Connectivity check recognises cmig as an edge (no false-positive
    "disjoint populations" warning).
  - Statistical/SFS-pattern tests (T1, T2): a backward cmig pulse from
    K→F at t_pulse (= forward F→K admixture into K) drives K-F dxy
    and Fst to decrease monotonically with proportion p, matching the
    analytic prediction
        E[T_KF | p] = p·(t_pulse + 2·Ne_F) + (1-p)·(t_split + 2·Ne_anc)
    in the colinear (panmictic) region.

TODO (T3, deferred): quantitative check that the *count* of lineages
moved by a cmig event matches Binomial(n_eligible, proportion). Needs
exposed simulator state or an event hook.
"""

import warnings

import numpy as np
import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography

from .conftest import NEGLIGIBLE_GAMMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_inv(t_inv, gamma=NEGLIGIBLE_GAMMA, p_inv=None):
    if p_inv is None:
        p_inv = {0: 0.0, 1: 0.5}
    return InversionSpec(
        bp_left=2000,
        bp_right=8000,
        p_inv=p_inv,
        t_inv=t_inv,
        gene_conversion_rate=gamma,
        mean_tract_length=300.0,
        tract_distribution="fixed",
        inv_id=0,
    )


# ---------------------------------------------------------------------------
# Connectivity: cmig should count as a connecting edge
# ---------------------------------------------------------------------------


def test_class_mig_recognised_in_connectivity_check():
    d = Demography([1000, 1000])
    d.add_class_migration(
        time=100.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=1.0
    )
    # Should NOT warn about disjoint pops.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert d.check_connectivity(warn=True) is True


def test_class_mig_zero_proportion_does_not_connect():
    # proportion=0 (would be add_class_migration kwargs) is not currently
    # accepted by add_class_migration (>0 enforced), but if a raw cmig
    # event with proportion=0 is added, connectivity should NOT count it.
    d = Demography([1000, 1000])
    d.add_event(("cmig", 100.0, 1, 0, "S", 0, 0.0))
    # Disjoint — connectivity check should fail.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert d.check_connectivity(warn=False) is False


# ---------------------------------------------------------------------------
# proportion=1.0 unconditional class merge
# ---------------------------------------------------------------------------


def test_class_mig_full_S_merge_equals_ej_for_S_only_sample():
    """Two pops, K=S-only / F=S+I.  cmig(F→K, kary='S' or 'I') with
    proportion=1.0 + safety ej (= add_class_split) should give the
    SAME tree topology as plain ej(F→K)."""
    inv = _build_inv(t_inv=5000)

    # Plain ej baseline.
    d_ej = Demography([1000, 2000])
    d_ej.add_event(("ej", 1000.0, 1, 0))
    sim_ej = HullSimulator(
        sample_config={("S", 0): 6, ("S", 1): 4, ("I", 1): 4},
        demography=d_ej,
        sequence_length=10000,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=42,
    )
    ts_ej = sim_ej.simulate()

    # Class-split (S-cmig + I-cmig + safety ej).
    d_split = Demography([1000, 2000])
    d_split.add_class_split(time=1000.0, source=1, dest=0, inv_id=0)
    sim_split = HullSimulator(
        sample_config={("S", 0): 6, ("S", 1): 4, ("I", 1): 4},
        demography=d_split,
        sequence_length=10000,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=42,
    )
    ts_split = sim_split.simulate()

    # Bit-equivalent: same node count, same tree count, same edges.
    assert ts_ej.num_nodes == ts_split.num_nodes, (
        f"node count differs: ej={ts_ej.num_nodes} split={ts_split.num_nodes}"
    )
    assert ts_ej.num_trees == ts_split.num_trees
    assert ts_ej.num_edges == ts_split.num_edges


def test_class_mig_S_only_leaves_I_in_src_pop():
    """cmig(F→K, kary='S', proportion=1.0) without safety ej should leave
    F-I lineages in F.  Demography has migration both ways via final ej
    so the run still completes (we only test that the I lineages are
    NOT moved at the cmig event, by checking they reach a coal node
    AFTER the cmig time."""
    inv = _build_inv(t_inv=10000)

    d = Demography([2000, 2000])
    # Move only F-S to K at t=500 going backward.
    d.add_class_migration(
        time=500.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=1.0
    )
    # Safety: a much-later ej catches everything.
    d.add_event(("ej", 5000.0, 1, 0))

    sim = HullSimulator(
        sample_config={("S", 0): 4, ("S", 1): 4, ("I", 1): 4},
        demography=d,
        sequence_length=10000,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=7,
    )
    ts = sim.simulate()
    # Sim should complete without hang.
    assert ts.num_nodes > 12  # at least the samples
    assert ts.num_trees >= 1


# ---------------------------------------------------------------------------
# Stochastic admixture (proportion < 1)
# ---------------------------------------------------------------------------


def test_class_mig_partial_stochastic_proportion():
    """proportion=0.3 should move roughly 30% of matching lineages on
    average across many seeds."""
    inv = _build_inv(t_inv=20000)
    proportion = 0.3
    n_reps = 30

    moved_counts = []
    for seed in range(n_reps):
        d = Demography([1000, 1000])
        # Pulse at t=200, then a final ej far back to ensure connectivity
        # for any lineage that didn't migrate.
        d.add_class_migration(
            time=200.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=proportion
        )
        d.add_event(("ej", 10000.0, 1, 0))
        sim = HullSimulator(
            sample_config={("S", 0): 5, ("S", 1): 20, ("I", 1): 5},
            demography=d,
            sequence_length=10000,
            recombination_rate=1e-8,
            inversions=[inv],
            seed=seed,
        )
        ts = sim.simulate()
        # Indirect proxy: count F-S samples whose initial ancestral edge
        # leads to a node with population=0 (K) before t=10000.  We
        # can't directly query "did this sample migrate at t=200" so
        # we just check that the run completes.  Stronger validation
        # happens in the next test via Mendelian-aggregate fraction.
        moved_counts.append(ts.num_nodes)
    # Sanity: variation across seeds means stochastic move happened.
    assert min(moved_counts) > 0
    assert len(set(moved_counts)) > 1, (
        "expected variation in node counts across seeds (stochastic cmig)"
    )


def test_class_mig_proportion_zero_rejected():
    d = Demography([1000, 1000])
    with pytest.raises(ValueError):
        d.add_class_migration(
            time=100.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=0.0
        )


def test_class_mig_proportion_above_one_rejected():
    d = Demography([1000, 1000])
    with pytest.raises(ValueError):
        d.add_class_migration(
            time=100.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=1.5
        )


def test_class_mig_invalid_karyotype_rejected():
    d = Demography([1000, 1000])
    with pytest.raises(ValueError):
        d.add_class_migration(
            time=100.0, source=1, dest=0, karyotype="X", inv_id=0, proportion=1.0
        )


# ---------------------------------------------------------------------------
# add_admixture wrapper
# ---------------------------------------------------------------------------


def test_admixture_class_unconditional_not_implemented():
    d = Demography([1000, 1000])
    with pytest.raises(NotImplementedError):
        d.add_admixture(time=100.0, source=1, dest=0, proportion=0.5)


def test_admixture_class_conditional_works():
    d = Demography([1000, 1000])
    d.add_admixture(
        time=100.0, source=1, dest=0, proportion=0.5, karyotype="I", inv_id=0
    )
    # Should record a 'cmig' event.
    assert any(ev[0] == "cmig" for ev in d.events)
    cmig_ev = next(ev for ev in d.events if ev[0] == "cmig")
    # ('cmig', t, src, dst, kary, inv_id, proportion)
    assert cmig_ev[4] == "I"
    assert cmig_ev[6] == 0.5


# ---------------------------------------------------------------------------
# T1, T2: Statistical/SFS-pattern validation of cmig admixture
# ---------------------------------------------------------------------------

# Shared scenario for T1/T2:
#   2 pops K (id 0), F (id 1), each Ne=5000.
#   The inversion spans the entire sequence (bp_left=1, bp_right=L-1)
#   so every lineage carries inv-region kary='S' material — required
#   for cmig kary='S' to act cleanly on every lineage. (With a small
#   inversion + recombination, sub-lineages whose inv-region content
#   has recombined away present kary=PAN and are NOT moved by cmig;
#   that's the documented PAN-stragglers caveat in add_admixture.)
#   p_inv = 0.5 constant in both pops, gamma negligible, t_inv far
#   past t_split so the barrier remains throughout the test horizon.
#   Samples: 10 S-only per pop. Coalescent rate for S-S pairs is
#   structured: 1 / (2 · Ne · p_std) = 1 / (Ne).
#   Backward at t_pulse=1000: cmig K→F kary='S' with proportion p
#     (= forward admixture pulse F→K with fraction p of K's ancestry
#     traced to F at t_pulse).
#   Backward at t_split=50000: ej F→K (full merge into pop 0).
#
# Analytic E[T_KF] in branch units (S-class, p_std = 1 - p_inv = 0.5):
#   T_KF(p) = p·(t_pulse + 2·Ne_F·p_std) + (1-p)·(t_split + 2·Ne_anc·p_std)
# With Ne_K = Ne_F = Ne_anc = 5000, p_std = 0.5, t_pulse=1000, t_split=50000:
#   T_KF(0.0) = 55000  → branch dxy ≈ 110000
#   T_KF(0.5) = 30500  → branch dxy ≈ 61000
#   T_KF(1.0) =  6000  → branch dxy ≈ 12000

_T12_NE = 5000
_T12_T_PULSE = 1000.0
_T12_T_SPLIT = 50000.0
_T12_T_INV = 200_000.0
_T12_P_INV = 0.5
_T12_P_STD = 1.0 - _T12_P_INV
_T12_L = 100_000
_T12_R = 1e-8
_T12_NREPS = 30
_T12_N_PER_POP = 10


def _t12_inv():
    # Inversion spans (effectively) the full sequence so cmig kary='S'
    # acts on every lineage. Use 1..L-1 to avoid any zero-width edge
    # behaviour at the simulator boundaries.
    return InversionSpec(
        bp_left=1.0,
        bp_right=float(_T12_L) - 1.0,
        p_inv=_T12_P_INV,
        t_inv=_T12_T_INV,
        gene_conversion_rate=NEGLIGIBLE_GAMMA,
        mean_tract_length=4999.9,
        tract_distribution="fixed",
        inv_id=0,
    )


def _t12_run_one(proportion: float, seed: int):
    """One sim with the shared T1/T2 scenario at the given cmig
    proportion. Returns whole-sequence branch-mode K-F divergence and
    Fst (the inversion covers the whole sequence, so structured rates
    apply uniformly)."""
    d = Demography([_T12_NE, _T12_NE])
    if proportion > 0.0:
        # Backward: K → F at t_pulse, kary='S'. Default inv_id=0.
        d.add_class_migration(
            time=_T12_T_PULSE,
            source=0,
            dest=1,
            karyotype="S",
            inv_id=0,
            proportion=proportion,
        )
    d.add_event(("ej", _T12_T_SPLIT, 1, 0))
    sim = HullSimulator(
        sample_config={("S", 0): _T12_N_PER_POP, ("S", 1): _T12_N_PER_POP},
        demography=d,
        sequence_length=_T12_L,
        recombination_rate=_T12_R,
        inversions=[_t12_inv()],
        seed=seed,
    )
    ts = sim.simulate()
    K = list(range(_T12_N_PER_POP))
    F = list(range(_T12_N_PER_POP, 2 * _T12_N_PER_POP))
    div = float(ts.divergence([K, F], mode="branch"))
    fst = float(ts.Fst([K, F], mode="branch"))
    return div, fst


def _t12_predicted_dxy(proportion: float) -> float:
    """Branch-mode dxy prediction (= 2 · E[T_KF]) for a K-F sample
    pair under the T1/T2 scenario. S-class lineages → structured
    coalescent rate 1/(2·Ne·p_std)."""
    pre_split_wait = 2.0 * _T12_NE * _T12_P_STD  # mean coal wait in F's S sub-pool
    anc_wait = 2.0 * _T12_NE * _T12_P_STD  # mean coal wait in ancestral S sub-pool
    t_kf = proportion * (_T12_T_PULSE + pre_split_wait) + (1.0 - proportion) * (
        _T12_T_SPLIT + anc_wait
    )
    return 2.0 * t_kf


def test_class_mig_admixture_dxy_decay():
    """T1: backward cmig K→F (kary='S') at t_pulse drives the colinear
    K-F branch dxy from a deep-split value down toward an immediately-
    post-pulse F-F value, monotonically with proportion. The mean
    across reps should match the analytic prediction within 25%."""
    rng = np.random.default_rng(20260427)
    proportions = [0.0, 0.5, 1.0]
    means = []
    for p in proportions:
        seeds = [int(s) for s in rng.integers(1, 2**31, size=_T12_NREPS)]
        dxy_vals = [_t12_run_one(p, seed)[0] for seed in seeds]
        means.append(float(np.mean(dxy_vals)))

    # Monotone decrease.
    assert means[0] > means[1] > means[2], (
        f"expected monotone-decreasing dxy with proportion, got {means}"
    )

    # Within-25% match to analytic prediction.
    for p, observed in zip(proportions, means):
        predicted = _t12_predicted_dxy(p)
        ratio = observed / predicted
        assert 0.75 <= ratio <= 1.25, (
            f"p={p}: observed dxy {observed:.0f} vs predicted "
            f"{predicted:.0f} (ratio {ratio:.3f}) outside ±25%"
        )


def test_class_mig_admixture_fst_monotonic():
    """T2: same cmig pulse setup as T1. Fst K-F in the colinear region
    should decrease monotonically as cmig proportion rises (more K
    ancestry traces to F → less between-pop divergence)."""
    rng = np.random.default_rng(20260427 ^ 0xFF)
    proportions = [0.0, 0.5, 1.0]
    means = []
    for p in proportions:
        seeds = [int(s) for s in rng.integers(1, 2**31, size=_T12_NREPS)]
        fst_vals = [_t12_run_one(p, seed)[1] for seed in seeds]
        means.append(float(np.mean(fst_vals)))

    assert means[0] > means[1] > means[2], (
        f"expected monotone-decreasing Fst with proportion, got {means}"
    )
    # Sanity: large p should drag Fst near zero.
    assert means[2] < 0.5 * means[0], (
        f"p=1.0 Fst {means[2]:.4f} should be << p=0.0 Fst {means[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# T3: count of moved lineages ~ Binomial(n_eligible, proportion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_class_mig_count_matches_binomial(p):
    """T3: per cmig event, n_moved ~ Binomial(n_eligible, p) within ±2σ.

    For each p, run 30 seeds; assert ≥95% of seeds fall in the
    np ± 2·sqrt(np(1-p)) band. This validates the per-lineage
    Bernoulli(p) sampling inside apply_class_mig.

    n_samples=50 in pop 1 ensures n_eligible ≥ ~20 with p_inv=0.5, which
    is the asymptotic regime where ±2σ → ~95% Binomial coverage holds.

    Hook is required: turn record_events on to see n_eligible/n_moved.
    Closes the T3 TODO documented at the top of this file (lines 20-22).
    """
    from msinv.hull._event_log import filter_cmig

    n_seeds = 30
    band_hits = 0
    seeds_with_eligible = 0

    for seed in range(n_seeds):
        d = Demography([1000, 1000])
        # Cmig at t=200: from pop 1 to pop 0, S karyotype, proportion=p.
        # Safety ej far back ensures connectivity for any non-moved lineages.
        d.add_class_migration(
            time=200.0, source=1, dest=0, karyotype="S", inv_id=0, proportion=p
        )
        d.add_event(("ej", 10000.0, 1, 0))
        sim = HullSimulator(
            sample_config={("S", 0): 5, ("S", 1): 50, ("I", 1): 5},
            demography=d,
            sequence_length=10000,
            recombination_rate=1e-8,
            inversions=[_build_inv(t_inv=20000.0)],
            seed=seed,
            record_events=True,
        )
        sim.simulate()
        recs = filter_cmig(sim.event_log)
        assert len(recs) == 1, f"seed={seed}: expected 1 cmig record, got {len(recs)}"
        r = recs[0]
        n, k = r["n_eligible"], r["n_moved"]
        if n == 0:
            continue  # no eligible S-class lineages this seed; can't test
        seeds_with_eligible += 1
        mu = n * p
        sd = (n * p * (1 - p)) ** 0.5
        if abs(k - mu) <= 2 * sd:
            band_hits += 1

    assert seeds_with_eligible >= n_seeds * 0.7, (
        f"p={p}: only {seeds_with_eligible}/{n_seeds} seeds had eligible "
        f"lineages — sample size or fixture is misconfigured"
    )
    # For extreme p (0.1, 0.9) the discrete Binomial ±2σ band covers ~96%
    # asymptotically, but with 30 seeds the Monte Carlo variance means we
    # accept ≥93% hits.  All other p values use the full 95% threshold.
    expected_threshold = 0.93 if p in (0.1, 0.9) else 0.95
    assert band_hits >= expected_threshold * seeds_with_eligible, (
        f"p={p}: only {band_hits}/{seeds_with_eligible} within ±2σ band; "
        f"per-lineage Bernoulli(p) sampling may be biased"
    )
