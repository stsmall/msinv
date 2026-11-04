"""Task 9c: the breakpoint boundary statistic (PPC-only).

The decay test runs msinv with 100 kb margins at t_inv = 1e6 (about 150 s per replicate) and is marked slow.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import tskit

from illex.abc import statvec as SV
from illex.abc.boundary import (BND_KEYS, SIDES, boundary_stats, build_flank_mask, flank_bins, flank_offsets,
                                 load_flank_mask, mask_accessible_bp, site_sums)
from illex.abc.constants import BND_EDGES_BP, BND_MARGIN_BP, BP_LEFT, BP_RIGHT
from illex.abc.trajectory import Path

NAN = float("nan")
L, BPL, BPR = 400_000.0, 150_000.0, 250_000.0
N = 4                              # chromosomes per class: I = nodes 0..3, S = nodes 4..7


def path(t_inv, p_now=0.626, gamma_eff=NAN):
    prm = dict(model="N", flux=not np.isnan(gamma_eff), t_inv=float(t_inv), hard=False, p0=p_now, t_sel=NAN,
               s=NAN, t_stop=NAN, p_star=NAN, gamma_eff=gamma_eff)
    return Path(params=prm, times=np.array([0.0, float(t_inv)]), freqs=np.array([p_now, p_now]), p_now=p_now)


def star_ts(sites):
    """A star tree of 2N samples under one root over [0, L), with ``sites`` = {position: derived sample nodes}."""
    t = tskit.TableCollection(sequence_length=L)
    for _ in range(2 * N):
        t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = t.nodes.add_row(time=1)
    for u in range(2 * N):
        t.edges.add_row(0, L, root, u)
    for x in sorted(sites):
        s = t.sites.add_row(position=x, ancestral_state="0")
        for u in sites[x]:
            t.mutations.add_row(site=s, node=u, derived_state="1")
    t.sort()
    t.build_index()
    t.compute_mutation_parents()
    return t.tree_sequence()


def pi(a, n=N):
    return 2.0 * a * (n - a) / (n * (n - 1.0))


def dxy(ai, as_, n=N):
    fi, fs = ai / n, as_ / n
    return fi * (1 - fs) + fs * (1 - fi)


# (position, I derived nodes, S derived nodes), binned by hand with left d = BPL - x and right d = x - BPR
SITES = {
    BPL - 500: ((0, 1), (4,)),           # left b0
    BPR + 400: ((0,), ()),               # right b0
    BPR + 3_000: ((0, 1, 2, 3), ()),     # right b1 (a fixed difference)
    BPL - 3_000: ((1,), (5, 6)),         # left b1
    BPR + 10_000: ((2,), (4,)),          # right b2
    BPL - 50_000: ((0, 1, 2), (4,)),     # left b3
}
INSIDE = {200_000.0: ((0,), (4, 5, 6, 7)), BPL: ((1,), (4,)), BPR - 1: ((2,), (5,))}
OUTSIDE = {BPL - 120_000: ((0,), (4, 5)), BPR + 100_000: ((3,), (6,))}


def _ts(extra=()):
    s = {x: tuple(i) + tuple(sn) for x, (i, sn) in SITES.items()}
    for e in extra:
        s.update({x: tuple(i) + tuple(sn) for x, (i, sn) in e.items()})
    return star_ts(s)


def _hand():
    """Hand-computed (pi_I, pi_S, dxy) sums per bin."""
    per = {0: [(2, 1), (1, 0)], 1: [(4, 0), (1, 2)], 2: [(1, 1)], 3: [(3, 1)]}
    out = {}
    for b, rows in per.items():
        pi_i = sum(pi(a) for a, _ in rows)
        pi_s = sum(pi(c) for _, c in rows)
        d = sum(dxy(a, c) for a, c in rows)
        out[f"bnd_piratio_b{b}"] = pi_i / pi_s
        out[f"bnd_dxypi_b{b}"] = d / (0.5 * (pi_i + pi_s))
    return out


I_NODES, S_NODES = np.arange(N), np.arange(N, 2 * N)


def test_flank_bins_outward():
    side, k = flank_bins([BPL - 500, BPR + 3_000, BPL - 1_000, BPR + 999.5, BPL, BPR - 1, BPR, BPL - 100_000,
                          BPR + 100_000, BPL - 99_999.5], BPL, BPR)
    assert side.tolist() == [0, 1, 0, 1, -1, -1, 1, -1, -1, 0]
    assert k.tolist() == [0, 1, 1, 0, -1, -1, 0, -1, -1, 3]


def test_hand_computed_ratios():
    ts = _ts()
    got = boundary_stats(ts, I_NODES, S_NODES, BPL, BPR, err=None)
    want = _hand()
    assert list(got) == list(BND_KEYS)
    for k in BND_KEYS:
        assert got[k] == pytest.approx(want[k], rel=1e-12), k
    # b0 = 2.3333 and 0.9, written out as a guard against a shared error in pi() / dxy()
    assert got["bnd_piratio_b0"] == pytest.approx((4 / 6 + 0.5) / 0.5)
    assert got["bnd_dxypi_b0"] == pytest.approx(0.75 / (0.5 * (4 / 6 + 1.0 / 2 + 0.5)))


def test_inside_and_beyond_never_counted():
    base = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None, per_side=True)
    more = boundary_stats(_ts((INSIDE, OUTSIDE)), I_NODES, S_NODES, BPL, BPR, err=None, per_side=True)
    for k in BND_KEYS:
        assert more[k] == base[k] or (np.isnan(more[k]) and np.isnan(base[k])), k
    assert more["n_seg"] == base["n_seg"] == [2, 2, 1, 1]
    assert more["left"]["n_seg"] == [1, 1, 0, 1] and more["right"]["n_seg"] == [1, 1, 1, 0]


def test_per_side_values():
    got = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None, per_side=True)
    assert got["left"]["bnd_piratio_b0"] == pytest.approx(pi(2) / pi(1))
    assert got["right"]["bnd_dxypi_b2"] == pytest.approx(dxy(1, 1) / pi(1))
    assert np.isnan(got["right"]["bnd_piratio_b1"])           # a fixed difference alone gives pi_I = pi_S = 0


def test_site_sums_is_observed_estimator():
    from illex.abc.observed import site_stats

    rng = np.random.default_rng(1)
    dos = rng.integers(-1, 3, size=(30, 50)).astype(np.int8)       # (individuals, sites), -1 missing
    ii, ss = np.arange(18), np.arange(18, 30)
    a = site_stats(dos, ii, ss)
    b = site_sums(dos[ii].T, dos[ss].T)
    for k in ("pi_I", "pi_S", "dxy"):
        np.testing.assert_allclose(b[k], a[k], rtol=1e-12)


def test_error_model_path_runs():
    import msprime

    ts = msprime.sim_mutations(msprime.sim_ancestry(20, sequence_length=L, recombination_rate=1e-8,
                                                    population_size=1e4, random_seed=3),
                               rate=1e-7, random_seed=3, model=msprime.BinaryMutationModel())
    smp = ts.samples()
    a = boundary_stats(ts, smp[:20], smp[20:], BPL, BPR, rng=np.random.default_rng(4),
                       err_rng=np.random.default_rng(5))
    b = boundary_stats(ts, smp[:20], smp[20:], BPL, BPR, rng=np.random.default_rng(4),
                       err_rng=np.random.default_rng(5))
    assert a == b or all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in BND_KEYS)
    assert all(np.isfinite(a[k]) for k in BND_KEYS)


def test_keys_ppc_only():
    assert set(BND_KEYS) <= set(SV.STAT_KEYS_PPC_ONLY)
    assert not set(BND_KEYS) & set(SV.MODEL_KEYS)
    assert not set(BND_KEYS) & set(SV.STAT_KEYS)        # own run and own data block, not the ABC row
    assert BND_EDGES_BP == (0, 1_000, 5_000, 20_000, 100_000) and BND_MARGIN_BP == 100_000


def test_edges_fixed():
    with pytest.raises(ValueError):
        boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, edges=(0, 10, 100), err=None)


def test_boundary_ppc_refuses_msprime():
    from illex.abc.simulate import boundary_ppc, engine_for

    p = path(1e5)
    assert engine_for(p, "auto") == "msinv"                 # every inversion body runs on msinv (0.4.0)
    with pytest.raises(ValueError):
        boundary_ppc(p, seed=1, engine="msprime")
    with pytest.raises(RuntimeError):                       # auto is msinv, gated by BND_PPC_READY
        boundary_ppc(p, seed=1, engine="auto")


def test_boundary_ppc_refuses_unfixed_msinv():
    from illex.abc import simulate

    assert simulate.BND_PPC_READY is False          # PR #19 is in (0.4.0); flip once the I x S flank-merge fix (0.4.1) lands
    with pytest.raises(RuntimeError):
        simulate.boundary_ppc(path(1e5), seed=1)


# ------------------------------------------------------------------------------ accessibility mask ----
def _full_mask(margin=100_000):
    return {name: np.ones(margin, bool) for name in SIDES}


def test_masked_site_excluded():
    """A site at an inaccessible offset is dropped, so the stats equal those of the ts without that site."""
    j_left_b0 = int(np.ceil(500) - 1)                 # BPL - 500 has offset 499 on the left
    mask = _full_mask()
    mask["left"][j_left_b0] = False
    got = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None, per_side=True, mask=mask)
    drop = {x: tuple(i) + tuple(sn) for x, (i, sn) in SITES.items() if x != BPL - 500}
    want = boundary_stats(star_ts(drop), I_NODES, S_NODES, BPL, BPR, err=None, per_side=True)
    for k in BND_KEYS:
        assert got[k] == want[k] or (np.isnan(got[k]) and np.isnan(want[k])), k
    assert got["left"]["n_seg"] == [0, 1, 0, 1]
    full = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None, mask=_full_mask())
    base = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None)
    assert all(full[k] == base[k] or (np.isnan(full[k]) and np.isnan(base[k])) for k in BND_KEYS)
    # masking the same offset on the right side changes nothing here
    mask_r = _full_mask()
    mask_r["right"][499] = False
    other = boundary_stats(_ts(), I_NODES, S_NODES, BPL, BPR, err=None, mask=mask_r)
    assert all(other[k] == base[k] or (np.isnan(other[k]) and np.isnan(base[k])) for k in BND_KEYS)


def test_offsets_match_data_bases():
    """A data base p0 at its centre p0 + 0.5 has offset BP_LEFT - 1 - p0 (left) or p0 - BP_RIGHT (right), and a
    continuous sim position inside that base gets the same offset."""
    from illex.abc.observed import bin_span

    j = np.arange(100_000)
    for name, p0 in (("left", BP_LEFT - 1 - j), ("right", BP_RIGHT + j)):
        for frac in (0.5, 0.0, 0.999):
            side, jj = flank_offsets(p0 + frac, BP_LEFT, BP_RIGHT)
            assert (side == SIDES.index(name)).all() and np.array_equal(jj, j), (name, frac)
        side, k = flank_bins(p0 + 0.5, BP_LEFT, BP_RIGHT)
        for b in range(4):
            lo, hi = bin_span(name, b)
            assert np.array_equal(np.sort(p0[k == b]), np.arange(lo, hi)), (name, b)
            assert ((j[k == b] >= BND_EDGES_BP[b]) & (j[k == b] < BND_EDGES_BP[b + 1])).all()


def test_sim_mask_accessible_bp_equals_data():
    import json
    from pathlib import Path

    obs = Path(__file__).resolve().parents[2] / "results/illex/abc_observed.json"
    if not obs.exists():
        pytest.skip("abc_observed.json not built")
    counts = json.loads(obs.read_text())["boundary"]["counts"]
    sim = mask_accessible_bp(load_flank_mask())
    for name in SIDES:
        assert sim[name] == counts[name]["accessible_bp"], name
    assert [a + b for a, b in zip(sim["left"], sim["right"])] == counts["pooled"]["accessible_bp"]


def test_stored_mask_equals_bed():
    from illex.abc.observed import ACC_BED

    if not ACC_BED.exists():
        pytest.skip(f"{ACC_BED} not available")
    stored, fresh = load_flank_mask(), build_flank_mask(ACC_BED)
    for name in SIDES:
        assert stored[name].shape == (BND_MARGIN_BP,) and np.array_equal(stored[name], fresh[name]), name


def test_sim_masked_sites_carry_only_accessible_offsets():
    """With the real mask, every simulated site that is counted sits at an accessible data offset."""
    import msprime

    seq = BND_MARGIN_BP * 2 + 40_000.0
    ts = msprime.sim_mutations(msprime.sim_ancestry(10, sequence_length=seq, recombination_rate=1e-8,
                                                    population_size=1e4, random_seed=7),
                               rate=2e-7, random_seed=7, model=msprime.BinaryMutationModel(),
                               discrete_genome=False)
    smp = ts.samples()
    bl, br = BND_MARGIN_BP, BND_MARGIN_BP + 40_000.0
    mask = load_flank_mask()
    got = boundary_stats(ts, smp[:10], smp[10:], bl, br, err=None, per_side=True, mask=mask)
    side, j = flank_offsets(ts.tables.sites.position, bl, br)
    _, k = flank_bins(ts.tables.sites.position, bl, br)
    acc = np.zeros(side.size, bool)
    for s_id, name in enumerate(SIDES):
        on = side == s_id
        acc[on] = mask[name][np.clip(j[on], 0, BND_MARGIN_BP - 1)] & (j[on] < BND_MARGIN_BP)
    g = ts.genotype_matrix()
    seg = (g.sum(1) > 0) & (g.sum(1) < g.shape[1])
    for s_id, name in enumerate(SIDES):
        want = [int(((side == s_id) & (k == b) & acc & seg).sum()) for b in range(4)]
        assert got[name]["n_seg"] == want, name
    assert got["left"]["n_seg"][0] == 0                     # left b0 has no accessible bp in the data


def test_error_free_model_equals_haploid():
    """err = (1, 0) per class (no errors, diploid pairing) gives the err=None haploid result exactly."""
    import msprime

    ts = msprime.sim_mutations(msprime.sim_ancestry(20, sequence_length=L, recombination_rate=1e-8,
                                                    population_size=1e4, random_seed=3),
                               rate=1e-7, random_seed=3, model=msprime.BinaryMutationModel())
    smp = ts.samples()
    a = boundary_stats(ts, smp[:20], smp[20:], BPL, BPR, err={"I": (1.0, 0.0), "S": (1.0, 0.0)},
                       rng=np.random.default_rng(9))
    b = boundary_stats(ts, smp[:20], smp[20:], BPL, BPR, err=None)
    for k in BND_KEYS:
        assert a[k] == pytest.approx(b[k], rel=1e-12), k


def test_boot_se_stratified():
    from illex.abc.observed import _boot_se

    rng = np.random.default_rng(0)
    left = np.array([[1.0, 2.0, 3.0]] * 3)
    right = np.array([[3.0, 1.0, 2.0]] * 2)
    r = _boot_se([left, right], rng, n_boot=200)
    # with identical units within each side, a side-stratified replicate is always the point estimate
    assert r["n_units"] == 5 and r["se"]["piratio"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_finite"]["piratio"] == 200
    one = _boot_se([np.array([[1.0, 1.0, 1.0]]), np.zeros((0, 3))], rng)
    assert one["n_units"] == 1 and np.isnan(one["se"]["piratio"])
    zero = _boot_se([np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])], rng, n_boot=500)
    assert 0 < zero["n_finite"]["piratio"] < 500             # replicates with pi_S = 0 are counted, not hidden


# ------------------------------------------------------------------------------- decay (slow) ----
REPS = 8


def _ppc(seed):
    from illex.abc.simulate import boundary_ppc

    return boundary_ppc(path(1e6), seed=seed, allow_unfixed=True)


def _control(seed):
    """The boundary estimator over the whole msprime neutral control (the flank value of pi ratio and dxy/pi)."""
    from illex.abc.boundary import bin_ratios, diploid_dosages
    from illex.abc.constants import N_INV, N_STD
    from illex.abc.error_model import LD_ERROR
    from illex.abc.simulate import _control_ts, _mutate, _seed_words

    _, s_ctl, _, s_mc, _, s_pc, _, _, s_ec = _seed_words(seed)
    cts = _mutate(_control_ts(s_ctl), s_mc)
    g = np.minimum(cts.genotype_matrix(), 1).astype(np.int8)
    rng, erng = np.random.default_rng(s_pc), np.random.default_rng(s_ec)
    di = diploid_dosages(g, np.arange(N_STD, N_STD + N_INV), rng, LD_ERROR["control"]["I"], erng)
    ds = diploid_dosages(g, np.arange(N_STD), rng, LD_ERROR["control"]["S"], erng)
    st = site_sums(di, ds)
    return bin_ratios({k: st[k].sum() for k in ("pi_I", "pi_S", "dxy")})


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_far_flank_decays_to_control():
    seeds = [900 + r for r in range(REPS)]
    with ProcessPoolExecutor(max_workers=REPS) as ex:
        rows = list(ex.map(_ppc, seeds))
    ctl = [_control(s) for s in seeds]
    for key, ck in (("bnd_piratio_b3", "piratio"), ("bnd_dxypi_b3", "dxypi")):
        m = float(np.nanmean([r[key] for r in rows]))
        c = float(np.nanmean([x[ck] for x in ctl]))
        print(f"{key}: msinv mean {m:.4f}  control mean {c:.4f}  per-rep {[round(r[key], 3) for r in rows]}")
        assert abs(m - c) <= 0.1, (key, m, c)
