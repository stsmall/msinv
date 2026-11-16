"""Sweeps linked to an inversion (selected site inside or outside it).

In a complete hard sweep every sample carries the beneficial allele at
the selected site, so the tree there must have its root no older than
the founder at ``t_origin``. A lineage left on the arrangement the
founder did not carry would stay unmerged and push the root far past
``t_origin``. The checks cover a site inside the inversion with gene flux
(the allele reaches the other arrangement) and sites outside it near
and far from a breakpoint. A sweep far from the inversion also matches
the same sweep without an inversion. The rates themselves are pinned
by the Rust unit tests in ``simulator.rs::sweep_linkage_tests``.
"""

import numpy as np
import pytest

from msinv.hull.inversion import InversionSpec
from msinv.hull.simulator import HullSimulator
from msinv.hull.sweep import Sweep

N = 10_000.0
T_ORIGIN = 1_000.0


def _hard_sweep(x_sel, seed, final=1.0):
    return Sweep(
        x_sel=x_sel, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.1, t_origin=T_ORIGIN, f0=1 / (2 * N),
        partial_sweep_final_freq=final, seed=seed,
    )


def _root_at(ts, x):
    tree = ts.at(x)
    return max(ts.node(r).time for r in tree.roots)


def test_inside_with_flux_all_lineages_reach_the_founder():
    # Inversion 20 kb, site in the middle, strong flux at the site
    # (exchange rate ~0.01 per generation) so A reaches S.
    L = 20_000.0
    for seed in range(1, 6):
        inv = InversionSpec(bp_left=0.0, bp_right=L, p_inv=0.5, t_inv=1_500.0,
                            gene_conversion_rate=0.02, mean_tract_length=100.0)
        ts = HullSimulator(
            n_std=4, n_inv=4, population_size=N, sequence_length=L,
            recombination_rate=1e-8, inversions=[inv],
            sweeps=[_hard_sweep(L / 2, seed)], seed=seed,
        ).simulate()
        assert _root_at(ts, L / 2) <= T_ORIGIN + 1.0, seed


@pytest.mark.parametrize("gap", [20_000.0, 200_000.0])
def test_outside_all_lineages_reach_the_founder(gap):
    inv_len = 50_000.0
    L = inv_len + gap + 20_000.0
    x_sel = inv_len + gap
    for seed in range(1, 9):
        inv = InversionSpec(bp_left=0.0, bp_right=inv_len, p_inv=0.3, t_inv=4 * N)
        ts = HullSimulator(
            n_std=4, n_inv=4, population_size=N, sequence_length=L,
            recombination_rate=1e-8, inversions=[inv],
            sweeps=[_hard_sweep(x_sel, seed)], seed=seed,
        ).simulate()
        assert _root_at(ts, x_sel) <= T_ORIGIN + 1.0, (gap, seed)


def test_outside_far_matches_no_inversion():
    inv_len, gap = 50_000.0, 200_000.0
    L = inv_len + gap + 20_000.0
    x_sel = inv_len + gap
    win = [0.0, x_sel - 1_000.0, x_sel + 1_000.0, L]

    def pi(with_inv, seed):
        sw = Sweep(x_sel=x_sel, tau=0.0, origin_pop=0, origin_kary="S",
                   target_inv=0, mode="Deterministic", s=0.05, t_origin=2_000.0,
                   f0=1 / (2 * N), seed=seed)
        if with_inv:
            inv = InversionSpec(bp_left=0.0, bp_right=inv_len, p_inv=0.3, t_inv=4 * N)
            sim = HullSimulator(n_std=4, n_inv=4, population_size=N,
                                sequence_length=L, recombination_rate=1e-8,
                                inversions=[inv], sweeps=[sw], seed=seed)
        else:
            sim = HullSimulator(samples=8, population_size=N, sequence_length=L,
                                recombination_rate=1e-8, sweeps=[sw], seed=seed)
        return sim.simulate().simplify().diversity(windows=win, mode="branch")[1]

    seeds = range(1, 31)
    a = np.mean([pi(True, s) for s in seeds])
    b = np.mean([pi(False, s) for s in seeds])
    assert 0.7 < a / b < 1.4, (a, b)


def test_soft_sweep_inside_with_flux_runs():
    L = 20_000.0
    inv = InversionSpec(bp_left=0.0, bp_right=L, p_inv=0.5, t_inv=1_500.0,
                        gene_conversion_rate=0.02, mean_tract_length=100.0)
    sw = Sweep(x_sel=L / 2, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
               mode="Deterministic", s=0.1, t_origin=T_ORIGIN, f0=0.05, seed=3)
    ts = HullSimulator(n_std=4, n_inv=4, population_size=N, sequence_length=L,
                       recombination_rate=1e-8, inversions=[inv], sweeps=[sw],
                       seed=3).simulate().simplify()
    assert np.isfinite(ts.diversity(mode="branch"))


def test_f0_above_origin_arrangement_frequency_is_rejected():
    L = 20_000.0
    inv = InversionSpec(bp_left=0.0, bp_right=L, p_inv=0.1, t_inv=4 * N)
    sw = Sweep(x_sel=L / 2, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
               mode="Deterministic", s=0.1, t_origin=T_ORIGIN, f0=0.2, seed=1)
    with pytest.raises(BaseException, match="cannot exceed"):
        HullSimulator(n_std=4, n_inv=4, population_size=N, sequence_length=L,
                      recombination_rate=1e-8, inversions=[inv], sweeps=[sw],
                      seed=1).simulate()
