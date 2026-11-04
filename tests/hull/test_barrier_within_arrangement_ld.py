"""Within-arrangement LD, pi and dxy while an inversion barrier is active.

With no gene flux and a constant inverted frequency p, the inverted body is
exactly a two-deme coalescent: arrangement I is a population of size pN,
arrangement S one of size (1 - p)N, no migration, both founded at t_inv from an
ancestral population of size N. msprime simulates that directly
(``add_population_split``), so within-arrangement LD by distance, pi within each
arrangement and dxy between them must match it.

Regression test for barrier-era partial coalescence. Up to msinv 0.3.5 every
coalescence while a barrier was active merged only the positions where both
lineages carried the event's class; same-arrangement non-overlap and margin
material stayed on remainder lineages under the old nodes. Each coalescence
thus added spurious breakpoints at the overlap edges, and short-range
within-arrangement r^2 fell about 3-fold below the two-deme value, while pi,
dxy and SNP counts stayed correct. The fix merges karyotype-compatible pairs in
full and counts each pair once, at 1/(2N p_K) for a shared karyotype K (or
1/(2N) when one lineage carries only margin material); counting such a pair in
both its body and its margin cell over-coalesces and cuts pi by about 4%, which
the pi/dxy checks guard.

msinv's default is the SMC (coalescence needs shared material), so the LD
reference is msprime's ``model="smc"``. Hudson gives about 3-5% higher r^2 at
these recombination rates, for msinv-independent reasons.
"""

import math

import msprime
import numpy as np
import pytest

from msinv.hull import HullSimulator
from msinv.hull.inversion import InversionSpec

N = 10_000
P = 0.6
T_INV = 4 * N           # long barrier: nearly all within-arrangement
MU = 1e-7               # coalescence happens while it is active
BINS = np.array([100, 300, 1_000, 3_000, 10_000, 20_000])
Z_TOL = 3.5


def _r2_by_distance(mts, nodes, window):
    """Mean haplotype r^2 per distance bin among ``nodes`` (MAF >= 0.05)."""
    g = mts.genotype_matrix()[:, nodes].astype(float)
    pos = mts.tables.sites.position
    keep = (pos >= window[0]) & (pos < window[1])
    g, pos = g[keep], pos[keep]
    f = g.mean(axis=1)
    keep = (f >= 0.05) & (f <= 0.95)
    g, pos = g[keep], pos[keep]
    x = g - g.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).mean(axis=1))
    r2 = ((x @ x.T) / x.shape[1] / np.outer(sd, sd)) ** 2
    i, j = np.triu_indices(len(pos), 1)
    b = np.digitize(pos[j] - pos[i], BINS)
    r2 = r2[i, j]
    return [r2[b == k].mean() if (b == k).any() else np.nan
            for k in range(1, len(BINS))]


def _stats(ts, i_nodes, s_nodes, window, seed):
    body = ts.keep_intervals([window]).trim() if window[0] > 0 else ts
    pi_i = body.diversity([list(i_nodes)], mode="branch")[0] * MU
    pi_s = body.diversity([list(s_nodes)], mode="branch")[0] * MU
    dxy = float(np.ravel(body.divergence([list(i_nodes), list(s_nodes)],
                                         mode="branch"))[0]) * MU
    mts = msprime.sim_mutations(ts, rate=MU, random_seed=seed,
                                model=msprime.BinaryMutationModel(),
                                discrete_genome=False)
    return (_r2_by_distance(mts, i_nodes, window)
            + _r2_by_distance(mts, s_nodes, window)
            + [pi_i, pi_s, dxy])


def _msinv(cfg, seed, use_rust=True):
    left, body = cfg["margin"], cfg["body"]
    spec = InversionSpec(bp_left=left, bp_right=left + body, p_inv=P,
                         t_inv=T_INV, gene_conversion_rate=1e-15,
                         mean_tract_length=1.0)
    n = cfg["n"]
    sim = HullSimulator(n_std=n, n_inv=n, population_size=N,
                        sequence_length=2 * left + body,
                        recombination_rate=cfg["r"], inversions=[spec],
                        seed=seed)
    ts = sim.simulate(use_rust=use_rust)
    s_nodes = np.arange(n)              # n_std samples are node IDs 0..n_std-1
    i_nodes = np.arange(n, 2 * n)
    return _stats(ts, i_nodes, s_nodes, (left, left + body), seed + 7)


def _msprime(cfg, seed):
    d = msprime.Demography()
    d.add_population(name="I", initial_size=P * N)
    d.add_population(name="S", initial_size=(1 - P) * N)
    d.add_population(name="A", initial_size=N)
    d.add_population_split(T_INV, derived=["I", "S"], ancestral="A")
    n = cfg["n"]
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(n, population="I", ploidy=1),
                 msprime.SampleSet(n, population="S", ploidy=1)],
        demography=d, sequence_length=cfg["body"],
        recombination_rate=cfg["r"], ploidy=2, model="smc",
        random_seed=seed)
    return _stats(ts, ts.samples(population=0), ts.samples(population=1),
                  (0, cfg["body"]), seed + 7)


def _z(a, b):
    """Two-sample z per column (NaN-tolerant)."""
    se = np.sqrt(np.nanvar(a, axis=0, ddof=1) / np.sum(~np.isnan(a), axis=0)
                 + np.nanvar(b, axis=0, ddof=1) / np.sum(~np.isnan(b), axis=0))
    return (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / se


def _labels():
    nb = len(BINS) - 1
    return ([f"r2_I[{BINS[k]}-{BINS[k+1]})" for k in range(nb)]
            + [f"r2_S[{BINS[k]}-{BINS[k+1]})" for k in range(nb)]
            + ["pi_I", "pi_S", "dxy"])


def _assert_close(a, b, what):
    z = _z(a, b)
    bad = {lab: (round(float(np.nanmean(a[:, k])), 5),
                 round(float(np.nanmean(b[:, k])), 5), round(float(z[k]), 2))
           for k, lab in enumerate(_labels()) if not abs(z[k]) < Z_TOL}
    assert not bad, f"{what}: (mean_a, mean_b, z) {bad}"


# Rust engine: 50 kb body with 10 kb collinear margins, 30 + 30 haplotypes,
# 4 N r = 2e-3 per bp. As shipped in 0.3.5 the r^2 deficit here is ~14 SE in
# the shortest bin and ~30% at 1-3 kb.
RUST = {"margin": 10_000, "body": 50_000, "n": 30, "r": 5e-8}
RUST_REPS = 200


def _expected_pi_dxy():
    """Exact two-deme expectations (branch mode, times mu): a within-q pair
    coalesces at 1/(2Nq) until t_inv, then at 1/(2N); a between pair waits
    for t_inv, then 1/(2N)."""
    def within(q):
        a = 2 * N * q
        return a * (1 - math.exp(-T_INV / a)) + 2 * N * math.exp(-T_INV / a)
    return np.array([within(P), within(1 - P), T_INV + 2 * N]) * 2 * MU


@pytest.mark.slow
def test_rust_within_arrangement_ld_pi_dxy_match_two_deme_msprime():
    a = np.array([_msinv(RUST, 500_000 + k) for k in range(RUST_REPS)])
    b = np.array([_msprime(RUST, 600_000 + k) for k in range(RUST_REPS)])
    _assert_close(a, b, "msinv (Rust) vs msprime two-deme SMC")
    # Rate counting: one coalescence rate per compatible pair. Counting a
    # pair in both its body and margin cells raises the within-arrangement
    # rate and cuts pi by ~4%, ~5 SE here.
    m = a[:, -3:].mean(axis=0)
    se = a[:, -3:].std(axis=0, ddof=1) / math.sqrt(len(a))
    z = (m - _expected_pi_dxy()) / se
    assert np.all(np.abs(z) < Z_TOL), (m, _expected_pi_dxy(), z)


def _body_pi_dxy(r, seed, use_rust=True, n=10, margin=10_000, body=50_000):
    spec = InversionSpec(bp_left=margin, bp_right=margin + body, p_inv=P,
                         t_inv=T_INV, gene_conversion_rate=1e-15,
                         mean_tract_length=1.0)
    sim = HullSimulator(n_std=n, n_inv=n, population_size=N,
                        sequence_length=2 * margin + body,
                        recombination_rate=r, inversions=[spec], seed=seed)
    ts = sim.simulate(use_rust=use_rust)
    s_nodes, i_nodes = list(range(n)), list(range(n, 2 * n))
    w = [0, margin, margin + body, 2 * margin + body]
    pi_i = ts.diversity([i_nodes], windows=w, mode="branch")[1][0]
    pi_s = ts.diversity([s_nodes], windows=w, mode="branch")[1][0]
    dxy = np.ravel(ts.divergence([i_nodes, s_nodes], windows=w,
                                 mode="branch")[1])[0]
    return np.array([pi_i, pi_s, dxy]) * MU


@pytest.mark.parametrize("r", [1e-12, 1e-9])
def test_body_pi_dxy_count_each_compatible_pair_once(r):
    """Inside the body every marginal tree is a two-deme tree, so pi within
    each arrangement and dxy have the exact expectations whatever r is.
    With little recombination every same-arrangement pair overlaps in both
    the body and the margins; if the pair were counted in both cells, its
    rate would be 1/(2Np) + 1/(2N) and within-arrangement pi would drop by
    ~25-35%."""
    a = np.array([_body_pi_dxy(r, 1_000_000 + k) for k in range(300)])
    m = a.mean(axis=0)
    se = a.std(axis=0, ddof=1) / math.sqrt(len(a))
    z = (m - _expected_pi_dxy()) / se
    assert np.all(np.abs(z) < Z_TOL), (m, _expected_pi_dxy(), z)


# Pure-Python engine (per-pair exact path, 4 N r L <= 100): smaller, so the
# slow O(n^2) loop stays inside the test budget.
PY = {"margin": 5_000, "body": 30_000, "n": 8, "r": 5e-8}
PY_REPS = 40


@pytest.mark.slow
def test_python_backend_matches_rust_and_msprime():
    py = np.array([_msinv(PY, 700_000 + k, use_rust=False)
                   for k in range(PY_REPS)])
    rs = np.array([_msinv(PY, 800_000 + k) for k in range(4 * PY_REPS)])
    ms = np.array([_msprime(PY, 900_000 + k) for k in range(4 * PY_REPS)])
    _assert_close(py, rs, "Python vs Rust backend")
    _assert_close(py, ms, "msinv (Python) vs msprime two-deme SMC")
    _assert_close(rs, ms, "msinv (Rust) vs msprime two-deme SMC, small")


@pytest.mark.parametrize("use_rust", [True, False])
@pytest.mark.parametrize("kary", ["I", "S"])
def test_same_arrangement_pair_coalesces_once(kary, use_rust):
    """Two same-arrangement samples, no recombination, barrier never lifted:
    the pair overlaps in the body (class I or S) AND in both panmictic
    margins, but is one pair of chromosomes in a deme of size qN, so
    E[T] = 2Nq exactly. Counting it once per overlapping class would give
    rate 1/(2Nq) + 1/(2N), i.e. E[T] = 2Nq / (1 + q) (-38% to -40%)."""
    q = P if kary == "I" else 1 - P
    n_std, n_inv = (0, 2) if kary == "I" else (2, 0)
    t = []
    for seed in range(1_000):
        spec = InversionSpec(bp_left=2_000, bp_right=8_000, p_inv=P,
                             t_inv=1e12, gene_conversion_rate=1e-15,
                             mean_tract_length=1.0)
        ts = HullSimulator(n_std=n_std, n_inv=n_inv, population_size=N,
                           sequence_length=10_000, recombination_rate=1e-12,
                           inversions=[spec], seed=2_000_000 + seed
                           ).simulate(use_rust=use_rust).simplify()
        t.append(ts.node(ts.first().root).time)
    t = np.array(t)
    z = (t.mean() - 2 * N * q) / (t.std(ddof=1) / math.sqrt(len(t)))
    assert abs(z) < Z_TOL, (t.mean(), 2 * N * q, z)


def test_compatible_pair_full_merge_unit():
    """Two I-arrangement samples with no recombination coalesce in ONE
    event that carries body and margins together: a single root spans the
    whole sequence (a partial merge leaves the margins on remainder
    lineages, which coalesce separately at other times)."""
    for use_rust in (True, False):
        for seed in range(20):
            spec = InversionSpec(bp_left=2_000, bp_right=8_000, p_inv=0.5,
                                 t_inv=1e9, gene_conversion_rate=1e-15,
                                 mean_tract_length=1.0)
            sim = HullSimulator(n_std=0, n_inv=2, population_size=1_000,
                                sequence_length=10_000,
                                recombination_rate=1e-12, inversions=[spec],
                                seed=seed)
            ts = sim.simulate(use_rust=use_rust).simplify()
            roots = {ts.node(t.root).time for t in ts.trees()}
            assert len(roots) == 1, (use_rust, seed, roots)
            assert math.isfinite(roots.pop())
