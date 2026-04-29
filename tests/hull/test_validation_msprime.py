"""msprime validation harness — Rust msinv core vs msprime.sim_ancestry.

Spec: docs/superpowers/specs/2026-04-29-msprime-validation-design.md.

Two scenarios:
  N1 — single-pop panmictic, n=10, Ne=10000, L=100kb, r=1e-8 (rho=40).
  N2 — two-pop island, n=5+5, Ne=[10000,10000], symmetric M=1e-4, L=100kb, r=1e-8.

For each scenario, N=200 reps on each engine (rep i seeds engine with i).
Per-stat assertion: |mean_msinv - mean_msprime| <= 3 * sqrt(SE_msinv^2 + SE_msprime^2).
"""

import math
import statistics

import msprime
import pytest

from msinv.hull.demography import Demography
from msinv.hull.simulator import HullSimulator


N_REPS = 200


def _stats_from_ts(ts, sample_sets=None):
    """Branch-length stats from a tskit TreeSequence.

    Returns dict with keys 'pi_branch', 'n_trees', 'mean_tmrca', and
    (when sample_sets is provided) 'dxy_branch'.
    """
    out = {
        "pi_branch": ts.diversity(mode="branch"),
        "n_trees": float(ts.num_trees),
    }
    samples = list(ts.samples())
    weighted = 0.0
    total_span = 0.0
    for tree in ts.trees():
        tmrca = tree.tmrca(*samples)
        weighted += tmrca * tree.span
        total_span += tree.span
    out["mean_tmrca"] = weighted / total_span
    if sample_sets is not None:
        out["dxy_branch"] = ts.divergence(
            sample_sets=sample_sets, mode="branch")
    return out


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _samples_by_pop(ts, n_pops):
    """Return [pop0_samples, pop1_samples, ...] for a tskit TS."""
    out = [[] for _ in range(n_pops)]
    for s in ts.samples():
        p = ts.node(s).population
        if 0 <= p < n_pops:
            out[p].append(s)
    return out


def _run_validation(scenario_name, msinv_factory, msprime_factory,
                    n_reps=N_REPS, by_pop_dxy=False):
    """Run both engines n_reps times, assert each branch-length stat
    agrees within 3 * combined SE.
    """
    stat_names = ["pi_branch", "n_trees", "mean_tmrca"]
    if by_pop_dxy:
        stat_names.append("dxy_branch")
    msinv_vals = {k: [] for k in stat_names}
    msprime_vals = {k: [] for k in stat_names}

    for i in range(n_reps):
        ts_a = msinv_factory(seed=i)
        ts_b = msprime_factory(seed=i)
        for engine_vals, ts in (
                (msinv_vals, ts_a), (msprime_vals, ts_b)):
            sample_sets = None
            if by_pop_dxy:
                sample_sets = _samples_by_pop(ts, n_pops=2)
            stats = _stats_from_ts(ts, sample_sets)
            for k in stat_names:
                engine_vals[k].append(stats[k])

    failures = []
    lines = []
    for k in stat_names:
        m_a, se_a = _mean_se(msinv_vals[k])
        m_b, se_b = _mean_se(msprime_vals[k])
        bound = 3.0 * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: msinv={m_a:.4g} ± {se_a:.3g}, "
                f"msprime={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, 3·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)
    print(f"\n[{scenario_name}]\n  " + "\n  ".join(lines))
    if failures:
        raise AssertionError(
            f"{scenario_name} failed:\n  " + "\n  ".join(failures))


def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""

    def msinv_factory(seed):
        return HullSimulator(
            samples=10,
            population_size=10000.0,
            sequence_length=100_000.0,
            recombination_rate=1e-8,
            inversions=[],
            seed=seed,
        ).simulate()

    def msprime_factory(seed):
        # population_size doubled vs msinv: msinv N = diploid Ne (2N chrom);
        # msprime ploidy=1 reads N as haploid Ne. record_full_arg=True so
        # non-ancestral recombs survive into the TS (msinv's convention).
        # See spec §"Population-size convention" and §"record_full_arg=True".
        return msprime.sim_ancestry(
            samples=10,
            population_size=20000.0,
            sequence_length=100_000,
            recombination_rate=1e-8,
            ploidy=1,
            record_full_arg=True,
            random_seed=seed + 1,
        )

    _run_validation("N1 panmictic", msinv_factory, msprime_factory)
