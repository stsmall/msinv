"""Second, long-tract flux process (#30).

An inversion can carry an independent long-tract exchange process
(``long_flux_rate``, ``long_mean_tract_length``) next to short gene
conversion. Each follows the forward flux model with its own ``w``, so a
carried site is covered at the sum of the two forward rates. The Rust
engine is pinned in ``rust/msinv-core/src/flux.rs``; these tests pin the
Python engine, the spec and the end-to-end hit rate.
"""

import dataclasses
import math

import pytest

from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography


def _inv(**kw):
    base = dict(
        bp_left=0.0,
        bp_right=100_000.0,
        p_inv=0.4,
        t_inv=1e9,
        gene_conversion_rate=1e-6,
        mean_tract_length=500.0,
        tract_distribution="geometric",
    )
    base.update(kw)
    return InversionSpec(**base)


def test_long_process_off_by_default():
    inv = _inv()
    assert len(inv.flux_procs()) == 1
    assert len(_inv(long_mean_tract_length=5_000.0).flux_procs()) == 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(long_flux_rate=-1e-9, long_mean_tract_length=1_000.0),
        dict(long_flux_rate=1e-9, long_mean_tract_length=0.0),
        dict(long_flux_rate=1e-9, long_mean_tract_length=1_000.0, long_tract_distribution="poisson"),
    ],
)
def test_long_process_validation(kw):
    with pytest.raises(ValueError):
        _inv(**kw)


@pytest.mark.parametrize("dist", ["fixed", "geometric"])
def test_python_per_site_rate_is_sum_of_processes(dist):
    """Python engine: with both processes a carried site is covered at
    the sum of the forward rates ``gcr * p_other * phi(x) * mean`` of the
    two, for the whole inversion and for a 2 bp sliver around the site."""
    from msinv.hull.lineage import Lineage
    from msinv.hull.segment import make_segment_list
    from msinv.hull.simulator import _phi

    inv = _inv(
        tract_distribution=dist,
        long_flux_rate=2e-8,
        long_mean_tract_length=5_000.0,
        long_tract_distribution=dist,
    )
    sim = HullSimulator(
        sample_config={("S", 0): 2},
        demography=Demography(pop_sizes=[1000]),
        sequence_length=100_000,
        recombination_rate=1e-8,
        inversions=[inv],
        seed=3,
    )
    x = 40_000.0
    expect = 0.4 * (
        1e-6 * _phi(0.4, 500.0 / 100_000.0) * 500.0
        + 2e-8 * _phi(0.4, 5_000.0 / 100_000.0) * 5_000.0
    )
    for intervals in ([(0.0, 100_000.0)], [(x - 1.0, x + 1.0)]):
        head, tail = make_segment_list(intervals, range(len(intervals)))
        lin = Lineage(head, tail, branch_class=inv.class_S())
        rate = sim._flux_lineage_rate(lin, inv)
        # One-process copies of the same (registered) inversion.
        short_only = dataclasses.replace(inv, long_flux_rate=0.0)
        long_only = dataclasses.replace(
            inv, gene_conversion_rate=2e-8, mean_tract_length=5_000.0, long_flux_rate=0.0
        )
        assert rate == pytest.approx(
            sim._flux_lineage_rate(lin, short_only) + sim._flux_lineage_rate(lin, long_only),
            rel=1e-12,
        )
        n = 100_000
        hits = 0
        for _ in range(n):
            drawn = sim._sample_flux_tract(lin, inv)
            if drawn is not None and drawn[1] <= x < drawn[2]:
                hits += 1
        assert rate * hits / n == pytest.approx(expect, rel=0.1), intervals


def test_rust_hits_per_branch_generation_match_forward_rate_two_processes():
    """End to end (Rust engine): at the inversion centre, flux tracts of
    both processes cover a carried site at the summed forward rate, as in
    ``test_flux_hits_per_branch_generation_match_forward_rate``."""
    from msinv.hull._event_log import filter_flux
    from msinv.hull.simulator import _phi

    bp_left, bp_right = 2_000.0, 22_000.0
    L = bp_right - bp_left
    g_s, lam_s = 5e-4, 300.0
    g_l, lam_l = 5e-5, 1_000.0
    p_inv = 0.5
    x = 12_000.0

    def e_len(lam):
        return lam * (1.0 - math.exp(-0.99 * L / lam))

    forward = (1.0 - p_inv) * (
        g_s * _phi((x - bp_left) / L, lam_s / L) * e_len(lam_s)
        + g_l * _phi((x - bp_left) / L, lam_l / L) * e_len(lam_l)
    )
    hits, tree_len = 0, 0.0
    for seed in range(20):
        inv = InversionSpec(
            bp_left=bp_left,
            bp_right=bp_right,
            p_inv=p_inv,
            t_inv=25_000.0,
            gene_conversion_rate=g_s,
            mean_tract_length=lam_s,
            long_flux_rate=g_l,
            long_mean_tract_length=lam_l,
        )
        sim = HullSimulator(
            sample_config={("S", 0): 10, ("I", 0): 10},
            demography=Demography(pop_sizes=[1000]),
            sequence_length=24_000,
            recombination_rate=1e-8,
            inversions=[inv],
            seed=seed,
            record_events=True,
        )
        ts = sim.simulate(use_rust=True)
        tree = ts.at(x)
        tmrca = max(tree.time(r) for r in tree.roots)
        tree_len += tree.total_branch_length
        hits += sum(
            1
            for r in filter_flux(sim.event_log, inv_id=0)
            if r["t"] < tmrca
            and any(s["seg_left"] <= x < s["seg_right"] for s in r["tract_segments"])
        )
    ratio = hits / tree_len / forward
    assert 0.9 < ratio < 1.1, f"{ratio:.3f} of forward {forward:.3e} ({hits} hits)"
