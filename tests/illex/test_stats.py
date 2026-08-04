"""Statistics tests. The panmictic calibration test doubles as harness test 2
from the spec (no inversion -> pi ratio 1, Fst 0) and validates the
branch-mode -> pi conversion."""
import pytest

from illex import demography, stats


@pytest.fixture(scope="module")
def neutral_ts():
    """Small neutral no-inversion run at constant Ne."""
    from msinv import HullSimulator
    sim = HullSimulator(
        n_std=30, n_inv=30,
        population_size=demography.PRESENT_NE_CONST,
        sequence_length=20_000,
        recombination_rate=2.5e-9,
        p_inv=0.5, t_inv=1.0e6,
        bp_left=1.0, bp_right=2.0,          # degenerate inversion: no barrier
        gene_conversion_rate=1e-15,
        seed=42,
    )
    return sim, sim.simulate()


def test_pi_matches_4_ne_mu(neutral_ts):
    """Calibrates the branch-mode -> pi conversion against theory: for a
    panmictic population pi = 4*Ne*mu."""
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    out = stats.arrangement_stats(ts, i_nodes, s_nodes)
    expected = 4 * demography.PRESENT_NE_CONST * 3e-9
    assert out["pi_i"] == pytest.approx(expected, rel=0.25)


def test_no_barrier_gives_no_differentiation(neutral_ts):
    """Harness test 2: with a degenerate inversion there is no barrier, so the
    two label sets are exchangeable -- pi ratio ~1, Fst ~0."""
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    out = stats.arrangement_stats(ts, i_nodes, s_nodes)
    assert out["pi_i_over_pi_s"] == pytest.approx(1.0, abs=0.20)
    assert abs(out["fst"]) < 0.05
    assert out["dxy_over_pi_i"] == pytest.approx(1.0, abs=0.20)


def test_node_partition_is_complete(neutral_ts):
    sim, ts = neutral_ts
    i_nodes, s_nodes = stats.sample_nodes_by_karyotype(sim, ts)
    assert len(i_nodes) == 30 and len(s_nodes) == 30
    assert not (set(i_nodes) & set(s_nodes))
    assert set(i_nodes) | set(s_nodes) == set(ts.samples())


@pytest.mark.slow
def test_msinv_matches_msprime_neutral():
    """Harness test 3 from the spec: msinv <-> msprime neutral agreement.

    Repo conventions (CLAUDE.md): msinv `population_size` is DIPLOID Ne with
    per-pair coalescence rate 1/(2N), so msprime must be called with
    `ploidy=1` and `2*N` to match. Compares branch-mode diversity, which is
    mutation-noise-free, over several reps.
    """
    import msprime
    import numpy as np
    from msinv import HullSimulator

    ne, seq_len, r, n_samp, reps = 50_000.0, 100_000, 2.5e-9, 40, 6

    msinv_vals = []
    for rep in range(reps):
        sim = HullSimulator(
            n_std=n_samp // 2, n_inv=n_samp // 2,
            population_size=ne, sequence_length=seq_len,
            recombination_rate=r,
            p_inv=0.5, t_inv=1.0e6,
            bp_left=1.0, bp_right=2.0,       # degenerate: no barrier
            gene_conversion_rate=1e-15, seed=300 + rep,
        )
        ts = sim.simulate()
        msinv_vals.append(ts.diversity(mode="branch"))

    msprime_vals = []
    for rep in range(reps):
        ts = msprime.sim_ancestry(
            samples=n_samp, ploidy=1, population_size=2 * ne,
            sequence_length=seq_len, recombination_rate=r,
            random_seed=400 + rep,
        )
        msprime_vals.append(ts.diversity(mode="branch"))

    a, b = float(np.mean(msinv_vals)), float(np.mean(msprime_vals))
    assert a == pytest.approx(b, rel=0.15), f"msinv {a:.1f} vs msprime {b:.1f}"
