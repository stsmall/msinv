"""Task 4: the observed statistic vector (results/illex/abc_observed.json) and the kernels that build it.

The JSON tests read the file written by ``illex/abc/observed.py`` (run in the pg_gpu env). The kernel tests check
that each data-side transform is the simulation-side transform of ``illex/abc/statvec.py``.
"""
import json
import math
from pathlib import Path

import msprime
import numpy as np
import pytest

from illex.abc import statvec as SV
from illex.abc.constants import P_NOW, TARGETS

OBS = Path("results/illex/abc_observed.json")
RATIOS = ("pi_ratio", "dxy_pi", "f1_ratio")


@pytest.fixture(scope="module")
def obs():
    return json.loads(OBS.read_text())


# ------------------------------------------------------------------------------------- the JSON ----
def test_keys_match_sim(obs):
    assert set(obs) >= {"stats", "se", "raw", "p_obs", "meta"}
    assert list(obs["stats"]) == list(SV.STAT_KEYS)
    assert list(obs["se"]) == list(SV.STAT_KEYS)
    assert set(obs["raw"]) == set(SV.STAT_KEYS)
    assert obs["p_obs"] == P_NOW
    assert all(math.isfinite(obs["stats"][k]) for k in SV.MODEL_KEYS)


def test_ratios_reproduce_targets(obs):
    # TARGETS forms: pi_ratio and dxy_pi are RAW body values (empirical_jackknife.py); f1_ratio is already
    # control-calibrated body/control (sfs_f1_jackknife.py "CALIBRATED TARGET"), so it is checked on stats.
    for k, form in (("pi_ratio", "raw"), ("dxy_pi", "raw"), ("f1_ratio", "stats")):
        v = obs[form][k]
        assert abs(v - TARGETS[k]) < 0.002, (k, form, v, TARGETS[k])


def test_stats_are_raw_over_control(obs):
    ctl = obs["meta"]["control"]
    divided = RATIOS + SV.SFS_KEYS + SV.LD_KEYS + ("tajd_I", "tajd_S")
    for k in divided:
        assert obs["stats"][k] == pytest.approx(obs["raw"][k] / ctl[k], rel=1e-12), k
    for k in SV.JSFS_KEYS + ("edge_centre",):
        assert obs["stats"][k] == obs["raw"][k]
        assert k not in ctl


def test_proportions(obs):
    for c in ("I", "S"):
        assert sum(obs["raw"][f"sfs_{c}_{b}"] for b in SV.SFS_BINS) == pytest.approx(1.0)
    assert sum(obs["stats"][k] for k in SV.JSFS_KEYS) == pytest.approx(1.0)


# ------------------------------------------------------------------------------------ the kernels ----
def _ts(seed=3, n=40, L=40_000.0):
    ts = msprime.sim_ancestry(n // 2, population_size=10_000, sequence_length=L, recombination_rate=1e-8,
                              random_seed=seed)
    return msprime.sim_mutations(ts, rate=2e-8, random_seed=seed, model=msprime.BinaryMutationModel(),
                                 discrete_genome=False)


def test_sfs_bins_from_spectrum_matches_statvec():
    from illex.abc.observed import sfs_bins

    rng = np.random.default_rng(0)
    n = 60
    counts = rng.integers(1, n, size=500)
    unf = np.bincount(counts, minlength=n + 1).astype(float)
    unf[0] += 10_000.0                                     # monomorphic mass must not matter
    assert sfs_bins(unf, n) == pytest.approx(SV._sfs_bins(counts, n))


def test_jsfs_from_matrix_matches_statvec():
    from illex.abc.observed import jsfs_from_2d

    rng = np.random.default_rng(1)
    ni, ns = 40, 30
    ci, cs = rng.integers(0, ni + 1, size=800), rng.integers(0, ns + 1, size=800)
    m = np.zeros((ni + 1, ns + 1))
    np.add.at(m, (ci, cs), 1.0)
    want = SV._jsfs(ci, cs, ni, ns)
    got = jsfs_from_2d(m)
    assert list(got) == list(SV.JSFS_KEYS)
    assert got == pytest.approx(want)


def test_tajd_from_spectrum_matches_tskit():
    from illex.abc.observed import tajd_from_unfolded

    ts = _ts()
    n = ts.num_samples
    counts = ts.genotype_matrix().sum(1)
    unf = np.bincount(counts, minlength=n + 1).astype(float)
    unf[0] += ts.sequence_length - unf.sum()               # sites = bp, as in a realSFS spectrum
    want = float(ts.Tajimas_D(mode="site"))
    assert tajd_from_unfolded(unf, n, n_to=n, length=ts.sequence_length) == pytest.approx(want, rel=1e-9)


def test_ld_kernel_matches_statvec_without_missing():
    from illex.abc.observed import ld_block_sums

    ts = _ts(seed=7, n=100)
    g = np.minimum(ts.genotype_matrix(), 1).astype(np.int8)
    pos = ts.tables.sites.position
    cols = np.arange(ts.num_samples)
    want = SV._ld(g, pos, cols, np.random.default_rng(5))
    perm = np.random.default_rng(5).permutation(cols.size)       # the pairing statvec._ld draws
    m = cols.size // 2
    dos = (g[:, cols[perm[:m]]] + g[:, cols[perm[m:2 * m]]]).astype(np.int8)
    s, c = ld_block_sums(dos, pos, np.zeros(pos.size, int), 1)
    got = s[0] / np.maximum(c[0], 1)
    got[c[0] == 0] = np.nan
    np.testing.assert_allclose(got, want, rtol=1e-9, equal_nan=True)


def test_ld_kernel_pairwise_complete():
    from illex.abc.observed import ld_block_sums

    rng = np.random.default_rng(2)
    n_s, n_i = 60, 30
    pos = np.sort(rng.uniform(0, 40_000, n_s))
    dos = rng.integers(0, 3, size=(n_s, n_i)).astype(np.int8)
    dos[rng.random(dos.shape) < 0.2] = -1
    blk = (pos // 20_000).astype(int)
    s, c = ld_block_sums(dos, pos, blk, 2, maf=0.0)
    nb = len(SV.LD_EDGES) - 1
    ws, wc = np.zeros((2, nb)), np.zeros((2, nb))
    for i in range(n_s):
        for j in range(i + 1, n_s):
            d = pos[j] - pos[i]
            if not (SV.LD_EDGES[0] <= d < SV.LD_EDGES[-1]):
                continue
            ok = (dos[i] >= 0) & (dos[j] >= 0)
            x, y = dos[i, ok].astype(float), dos[j, ok].astype(float)
            if ok.sum() < 2 or x.var() == 0 or y.var() == 0:
                continue
            k = np.searchsorted(SV.LD_EDGES, d, side="right") - 1
            ws[blk[i], k] += np.corrcoef(x, y)[0, 1] ** 2
            wc[blk[i], k] += 1
    np.testing.assert_array_equal(c, wc)
    np.testing.assert_allclose(s, ws, rtol=1e-9, atol=1e-12)
