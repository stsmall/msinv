#!/usr/bin/env python
"""Validate the parts of the pipeline that do not need SLiM.

SLiM is not installed on the analysis box, so `inversion_abc.slim` cannot be
syntax-checked here. Everything downstream of it can be, and is: the prior
draws, and the statistic machinery in summarize.py exercised against a synthetic
msprime tree sequence carrying a marker mutation at the inversion start.

The synthetic case has NO barrier, so it is also a real check: with no
recombination suppression the two "arrangements" are exchangeable, so
pi_i/pi_s must be ~1 and dxy/pi_i must be ~1. If the karyotype split or the
interval restriction is wrong, those come out visibly off.

  .venv/bin/python -m illex.slim.selftest
"""
from __future__ import annotations

import sys

import numpy as np

from . import config as C


def test_priors(n=40_000) -> None:
    rng = np.random.default_rng(0)
    draws = [C.draw_params(rng) for _ in range(2000)]
    s = np.array([d["s"] for d in draws])
    pf = np.array([d["p_flux"] for d in draws])
    t = np.array([d["t_inv"] for d in draws])
    ps = np.array([d["p_start"] for d in draws])
    h = np.array([d["h"] for d in draws])

    print("priors:")
    print(f"  P(s=0)      = {(s == 0).mean():.3f}  (target "
          f"{C.PRIOR_NEUTRAL_WEIGHT})")
    print(f"  P(p_flux=0) = {(pf == 0).mean():.3f}  (target "
          f"{C.PRIOR_NOFLUX_WEIGHT})")
    print(f"  t_inv    range [{t.min():.3g}, {t.max():.3g}]")
    print(f"  p_start  range [{ps.min():.3g}, {ps.max():.3g}]")
    print(f"  h        range [{h.min():.3f}, {h.max():.3f}]")

    assert abs((s == 0).mean() - C.PRIOR_NEUTRAL_WEIGHT) < 0.05
    assert t.min() >= C.PRIORS["t_inv"][1] and t.max() <= C.PRIORS["t_inv"][2]
    assert ps.min() >= C.PRIORS["p_start"][1]
    # h must be pinned when s == 0 (unidentifiable there).
    assert np.all(h[s == 0] == 0.5), "h not pinned on the neutral atom"
    # Scaling validity guard the .slim script enforces.
    smax = C.PRIORS["s"][2]
    assert smax * C.Q_DEFAULT < 0.1, (
        f"s_max*Q = {smax * C.Q_DEFAULT} >= 0.1; default Q is too large for the "
        "s prior and every strongly-selected simulation would abort")
    print(f"  s_max*Q_default = {smax * C.Q_DEFAULT:.4f} < 0.1 OK")


def test_summarize_machinery() -> None:
    """Exercise the karyotype split + statistics on a synthetic tree sequence."""
    import msprime
    import tskit

    from .summarize import arrangement_stats, folded_sfs_shape

    L = C.FLANK_LEN_SIM * 2 + C.INV_LEN_SIM
    inv_start = float(C.FLANK_LEN_SIM)
    ts = msprime.sim_ancestry(samples=200, ploidy=1, sequence_length=L,
                              recombination_rate=1e-7, population_size=5_000,
                              random_seed=42)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=43)

    # Plant a marker at inv_start carried by a random half of the samples, so the
    # split is known and the two groups are exchangeable by construction.
    rng = np.random.default_rng(7)
    samples = np.asarray(ts.samples())
    carriers = set(rng.choice(samples, size=len(samples) // 2, replace=False)
                   .tolist())

    tables = ts.dump_tables()
    site_id = tables.sites.add_row(position=inv_start, ancestral_state="A")
    tree = ts.at(inv_start)
    for u in sorted(carriers):
        tables.mutations.add_row(site=site_id, node=u, derived_state="T",
                                 time=tree.time(tree.parent(u)) / 2
                                 if tree.parent(u) != tskit.NULL else 1.0)
    tables.sort()
    ts2 = tables.tree_sequence()

    # Re-read the split through the same code path the pipeline uses.
    from .summarize import karyotype_sample_nodes
    i_nodes, s_nodes = karyotype_sample_nodes(ts2, inv_start)
    print("\nsummarize machinery:")
    print(f"  karyotype split: I={len(i_nodes)} S={len(s_nodes)} "
          f"(planted {len(carriers)})")
    assert set(i_nodes.tolist()) == carriers, "karyotype split does not match"

    interval = (inv_start, inv_start + C.INV_LEN_SIM)
    st = arrangement_stats(ts2, i_nodes, s_nodes, interval, 1e-7)
    print(f"  pi_i/pi_s   = {st['pi_i_over_pi_s']:.4f}   (no barrier -> ~1)")
    print(f"  dxy/pi_i    = {st['dxy_over_pi_i']:.4f}   (no barrier -> ~1)")
    assert 0.9 < st["pi_i_over_pi_s"] < 1.1, "exchangeable groups gave pi ratio far from 1"
    assert 0.9 < st["dxy_over_pi_i"] < 1.1, "exchangeable groups gave dxy ratio far from 1"

    sfs = folded_sfs_shape(ts2, i_nodes, interval, C.SFS_PROJ)
    print(f"  folded SFS shape: {len(sfs)} bins, sum={np.nansum(sfs):.4f}, "
          f"singleton frac={sfs[0]:.4f}")
    assert len(sfs) == C.SFS_BINS
    assert abs(np.nansum(sfs) - 1.0) < 1e-6, "SFS shape not normalized"

    # A neutral constant-size folded spectrum should be singleton-dominated.
    assert sfs[0] > sfs[1] > 0, "folded SFS is not monotone at the low end"


def test_stat_vector_contract() -> None:
    print("\nstatistic vector:")
    print(f"  {len(C.STAT_NAMES)} default statistics "
          f"({C.SFS_BINS} SFS bins x 2 arrangements + 3 scalars)")
    assert "fst" not in C.STAT_NAMES, (
        "Fst must stay OUT of the vector: Fst = 1-(r+1)/(2dr) is algebraically "
        "determined by the two ratios (NOTES sec 5.3)")
    assert len(set(C.STAT_NAMES)) == len(C.STAT_NAMES), "duplicate stat names"
    expect = 3 + 2 * C.SFS_BINS
    assert len(C.STAT_NAMES) == expect, f"expected {expect} stats"
    print("  Fst correctly excluded (algebraically redundant)")


def main() -> int:
    test_priors()
    test_stat_vector_contract()
    test_summarize_machinery()
    print("\nALL SELFTESTS PASSED")
    print("NOT covered here: inversion_abc.slim itself (SLiM not installed on "
          "this box). Run illex/slim/smoke_slim.sh on Talapas first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
