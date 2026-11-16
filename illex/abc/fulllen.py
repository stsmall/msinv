"""Full-length simulations of the balancing model with two flux processes (msinv #30), for the NPE training table.

One row simulates the whole 18.96-Mb inversion (FULL_BODY_BP between two MARGIN_BP collinear margins) under an
accepted soft-start balancing path, with short gene conversion (per-site rate c = ``gamma_eff``, mean tract
TRACT_BP) and an independent long-tract exchange process (per-site rate ``c_long``, mean tract ``l_long``), both
geometric. Each process's per-site rate c is converted to msinv's per-bp initiation rate with its own
w = tract / FULL_BODY_BP (``gcr_from_c``), as ``simulate.gamma_from_c`` does for the 40-kb body.

The neutral control is not simulated per row. A bank of full-length controls (FULL_CONTROL_BP, the 45 Mb of chr2
collinear control in the data; ``control_bank.py``) is simulated once, and each row is paired with one bank entry
drawn from its own seed. Body and control are independent simulations in the 40-kb design too, so pairing with an
independent bank entry has the same distribution; at 45 Mb the control's own scatter is small.

The 42 statistics of ``statvec.STAT_KEYS`` are computed with the same functions and random streams as
``statvec.region_stats`` (branch ratios, projected SFS bins, binned joint SFS, LD with the data's genotype error
model, Tajima's D, edge_centre) and divided by the bank control as in ``statvec.assemble``. Every row also keeps
the raw material for later statistics, so that the simulations need not be repeated:

  raw_body, raw_ctl       the statistics before division by the control
  ld_mb_{I,S}             LD sums and pair counts per 1-Mb block x the 6 fitted bins (error model on)
  ld_fine_{I,S}           sums and counts in 12 log bins 100 bp - 20 kb, with the error model and error-free
  ld_long_{I,S}           sums and counts in bins 20 kb - 2 Mb on thinned sites (~1 per LONG_LD_SPACING bp)
  sfs_unf_{I,S}           unfolded derived-count spectra (0..n), site mode
  jsfs_raw                sparse exact joint count spectrum [[i, s, sites], ...] (I count, S count)
  pi_mb_{I,S}, dxy_mb     branch-mode pi and dxy per 1-Mb window
  edge_profile            dxy / mean pi in the 20 edge_centre windows
  site_pi_{I,S}, seg_{I,S}, site_dxy   site-mode pi, segregating sites, dxy over the body
  path_t, path_p          the frequency path, at most PATH_POINTS points (ends kept)
"""
from __future__ import annotations

import math

import numpy as np

from illex.abc import statvec as SV
from illex.abc.constants import MARGIN_BP, MU, N_INV, N_KARYO, N_STD, REC_RATE, TRACT_BP
from illex.abc.error_model import LD_ERROR

FULL_BODY_BP = 18_960_000.0      # 474 x 40 kb, the inversion's length in the data (chr2:60.54-79.50 Mb)
FULL_CONTROL_BP = 45_000_000.0   # chr2 collinear control of the data (40-55 + 85-115 Mb)
MB = 1_000_000.0
LONG_PRIOR = dict(c_long=(1e-11, 1e-6), l_long=(1e4, 5e6))   # log-uniform; c_long near 0 switches the process off
LD_FINE_EDGES = np.logspace(2.0, math.log10(20_000.0), 13)   # 12 log bins, 100 bp - 20 kb
LD_LONG_EDGES = np.array([2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6])
LONG_LD_SPACING = 2_000.0
PATH_POINTS = 200
EXTRA_TAG = 461                  # SeedSequence tag of the row's extra stream (long-flux prior, bank pick, thinning)


def gcr_from_c(c: float, tract: float, body: float | None = None) -> float:
    """msinv per-bp initiation rate for per-site conversion rate ``c`` with mean tract ``tract`` in ``body`` bp
    (default FULL_BODY_BP, read at call time)."""
    w = tract / (FULL_BODY_BP if body is None else body)
    if not 0.0 < w < 1.0:
        raise ValueError(f"tract {tract} must be shorter than the simulated body (w = {w})")
    return float(c) / tract * (1.0 - w) / w


def extra_rng(seed: int) -> np.random.Generator:
    """The row's extra stream, separate from the 9 ``simulate._seed_words`` words."""
    return np.random.default_rng(np.random.SeedSequence([EXTRA_TAG, int(seed)]))


def draw_long(rng: np.random.Generator) -> tuple[float, float]:
    """(c_long, l_long) from their log-uniform priors."""
    lo, hi = LONG_PRIOR["c_long"]
    c = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
    lo, hi = LONG_PRIOR["l_long"]
    return c, float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def _msinv_check():
    import msinv
    from msinv import InversionSpec
    if "long_flux_rate" not in getattr(InversionSpec, "__dataclass_fields__", {}):
        raise RuntimeError(f"fulllen needs msinv with the long-tract flux process (#30); found {msinv.__version__} "
                           f"without InversionSpec.long_flux_rate")


def inversion_ts(path, seed, c_long, l_long):
    """(sim, ts): msinv, the full inversion between two MARGIN_BP margins, two flux processes, crossover suppression."""
    from msinv import HullSimulator, InversionSpec

    from illex.demography import PRESENT_NE_GROWTH, growth_demography
    from illex.theory import N_growth

    _msinv_check()
    c = float(path.params["gamma_eff"])
    t_inv = float(path.params["t_inv"])
    spec = InversionSpec(bp_left=MARGIN_BP, bp_right=MARGIN_BP + FULL_BODY_BP,
                         gene_conversion_rate=gcr_from_c(c, TRACT_BP), mean_tract_length=TRACT_BP,
                         tract_distribution="geometric",
                         long_flux_rate=gcr_from_c(c_long, l_long), long_mean_tract_length=float(l_long),
                         long_tract_distribution="geometric",
                         trajectory={"type": "precomputed", "times": np.asarray(path.times, float),
                                     "freqs": [np.asarray(path.freqs, float)],
                                     "n_e": [float(N_growth(t_inv))], "t_inv": [t_inv]})
    sim = HullSimulator(n_std=N_STD, n_inv=N_INV, population_size=PRESENT_NE_GROWTH,
                        demography=growth_demography(), sequence_length=FULL_BODY_BP + 2 * MARGIN_BP,
                        recombination_rate=REC_RATE, inversions=[spec], seed=int(seed),
                        crossover_suppression=True)
    return sim, sim.simulate()


def control_ts(seed):
    """msprime no-inversion growth control over FULL_CONTROL_BP (as ``simulate._control_ts``)."""
    import msprime

    from illex.theory import ALPHA, N0, N_ANC, T_GROW

    d = msprime.Demography()
    d.add_population(name="A", initial_size=float(N0), growth_rate=float(ALPHA))
    d.add_population_parameters_change(time=float(T_GROW), initial_size=float(N_ANC), growth_rate=0.0,
                                       population="A")
    return msprime.sim_ancestry(samples=[msprime.SampleSet(N_STD + N_INV, population="A", ploidy=1)],
                                demography=d, sequence_length=FULL_CONTROL_BP, recombination_rate=REC_RATE,
                                ploidy=2, model="hudson", random_seed=int(seed))


def _ld_all(geno, pos, cols, win, rng, err, err_rng, xrng):
    """Fitted LD bins (as ``statvec._ld``: same pairing draw, same error draw) plus the extra LD material."""
    from illex.abc.error_model import apply_genotype_errors

    nbin = len(SV.LD_EDGES) - 1
    n_mb = int(math.ceil((win[1] - win[0]) / MB))
    cols = np.asarray(cols)
    perm = rng.permutation(cols.size)
    m = cols.size // 2
    a, b = cols[perm[:m]], cols[perm[m:2 * m]]
    fit = np.full(nbin, np.nan)
    ext = {}
    if m < 2 or geno.shape[0] < 2:
        return fit, ext
    d0 = geno[:, a].astype(np.float64) + geno[:, b]
    d = apply_genotype_errors(d0, err[0], err[1], err_rng) if err is not None else d0
    blk = np.minimum(((pos - win[0]) // MB).astype(int), n_mb - 1)
    s, c = SV.ld_block_sums(d, pos, blk, n_mb)
    tot_s, tot_c = s.sum(0), c.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fit = np.where(tot_c > 0, tot_s / np.maximum(tot_c, 1), np.nan)
    ext["ld_mb"] = [s.tolist(), c.tolist()]
    z = np.zeros(pos.size, int)
    s, c = SV.ld_block_sums(d, pos, z, 1, edges=LD_FINE_EDGES)
    ext["ld_fine"] = [s[0].tolist(), c[0].tolist()]
    s, c = SV.ld_block_sums(d0, pos, z, 1, edges=LD_FINE_EDGES)
    ext["ld_fine_noerr"] = [s[0].tolist(), c[0].tolist()]
    keep, _, _ = SV.ld_site_keep(d)
    idx = np.flatnonzero(keep)
    target = (win[1] - win[0]) / LONG_LD_SPACING
    if idx.size > target:
        idx = np.sort(xrng.choice(idx, int(target), replace=False))
    s, c = SV.ld_block_sums(d[idx], pos[idx], np.zeros(idx.size, int), 1, edges=LD_LONG_EDGES)
    ext["ld_long"] = [s[0].tolist(), c[0].tolist()]
    return fit, ext


def region_full(ts, i_nodes, s_nodes, win, rng, ld_err, err_rng, xrng):
    """(raw, extra): ``statvec.region_stats`` over ``win`` (same values, same streams) and the extra material."""
    i_nodes, s_nodes = np.asarray(i_nodes), np.asarray(s_nodes)
    raw = SV._branch_ratios(ts, i_nodes, s_nodes, win)
    pos = ts.tables.sites.position
    sel = (pos >= win[0]) & (pos < win[1])
    if ts.num_sites:
        g = np.minimum(ts.genotype_matrix()[sel], 1).astype(np.int8)
    else:
        g = np.zeros((0, ts.num_samples), np.int8)
    p = pos[sel]
    col = {int(u): c for c, u in enumerate(ts.samples())}
    ci_cols = np.array([col[int(u)] for u in i_nodes])
    cs_cols = np.array([col[int(u)] for u in s_nodes])
    ci = g[:, ci_cols].sum(1).astype(np.int64)
    cs = g[:, cs_cols].sum(1).astype(np.int64)
    ni, ns = ci_cols.size, cs_cols.size
    for c, cnt, n in (("I", ci, ni), ("S", cs, ns)):
        for b, v in SV._sfs_bins(cnt, n).items():
            raw[f"sfs_{c}_{b}"] = float(v)
    raw.update(SV._jsfs(ci, cs, ni, ns))
    ext = {}
    for c, cols in (("I", ci_cols), ("S", cs_cols)):
        err = None if ld_err is None else ld_err[c]
        fit, e = _ld_all(g, p, cols, win, rng, err, err_rng, xrng)
        for k, v in enumerate(fit):
            raw[f"ld_{c}_{k + 1}"] = float(v)
        for k, v in e.items():
            ext[f"{k}_{c}"] = v
    raw["tajd_I"] = SV._tajd(ts, i_nodes, win)
    raw["tajd_S"] = SV._tajd(ts, s_nodes, win)

    ext["sfs_unf_I"] = np.bincount(ci, minlength=ni + 1).tolist()
    ext["sfs_unf_S"] = np.bincount(cs, minlength=ns + 1).tolist()
    jt = np.zeros((ni + 1, ns + 1), np.int64)
    np.add.at(jt, (ci, cs), 1)
    nz = np.argwhere(jt > 0)
    ext["jsfs_raw"] = [[int(i), int(j), int(jt[i, j])] for i, j in nz]
    ext["seg_I"] = int(((ci > 0) & (ci < ni)).sum())
    ext["seg_S"] = int(((cs > 0) & (cs < ns)).sum())
    windows = SV._windows(ts, win)
    k = 1 if win[0] > 0 else 0
    sp = ts.diversity([list(i_nodes), list(s_nodes)], mode="site", windows=windows)[k]
    ext["site_pi_I"], ext["site_pi_S"] = float(sp[0]), float(sp[1])
    ext["site_dxy"] = float(ts.divergence([list(i_nodes), list(s_nodes)], mode="site", windows=windows)[k])
    edges = list(np.arange(win[0], win[1], MB)) + [win[1]]
    mw = ([0.0] if win[0] > 0 else []) + edges + ([ts.sequence_length] if win[1] < ts.sequence_length else [])
    k0 = 1 if win[0] > 0 else 0
    nmb = len(edges) - 1
    pim = ts.diversity([list(i_nodes), list(s_nodes)], mode="branch", windows=mw)[k0:k0 + nmb]
    dxm = ts.divergence([list(i_nodes), list(s_nodes)], mode="branch", windows=mw)[k0:k0 + nmb]
    ext["pi_mb_I"] = (MU * pim[:, 0]).tolist()
    ext["pi_mb_S"] = (MU * pim[:, 1]).tolist()
    ext["dxy_mb"] = (MU * dxm).tolist()
    return raw, ext


def edge_profile(ts, i_nodes, s_nodes, win):
    """dxy / mean pi in the N_WIN equal windows of ``statvec.edge_centre`` (branch mode)."""
    left, right = win
    w = np.linspace(left, right, SV.N_WIN + 1)
    windows = ([0.0] if left > 0 else []) + list(w) + ([ts.sequence_length] if right < ts.sequence_length else [])
    k0 = 1 if left > 0 else 0
    pi = ts.diversity([list(i_nodes), list(s_nodes)], mode="branch", windows=windows)[k0:k0 + SV.N_WIN]
    dxy = ts.divergence([list(i_nodes), list(s_nodes)], mode="branch", windows=windows)[k0:k0 + SV.N_WIN]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dxy / pi.mean(1)).tolist()


def control_entry(seed: int, ld_error=LD_ERROR) -> dict:
    """One bank control: raw statistics and extra material of a FULL_CONTROL_BP neutral control."""
    from illex.abc.simulate import _mutate

    ss = np.random.SeedSequence([EXTRA_TAG + 1, int(seed)])
    s_anc, s_mut, s_pair, s_err, s_x = (int(v) for v in ss.generate_state(5) % (2**31 - 1) + 1)
    ts = _mutate(control_ts(s_anc), s_mut)
    smp = ts.samples()
    raw, ext = region_full(ts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, FULL_CONTROL_BP),
                           np.random.default_rng(s_pair), ld_error["control"], np.random.default_rng(s_err),
                           np.random.default_rng(s_x))
    return {"seed": int(seed), "raw": raw, "x": ext}


def _path_points(path):
    t = np.asarray(path.times, float)
    f = np.asarray(path.freqs, float)
    if t.size > PATH_POINTS:
        idx = np.unique(np.linspace(0, t.size - 1, PATH_POINTS).round().astype(int))
        t, f = t[idx], f[idx]
    return t.tolist(), f.tolist()


def simulate_full(path, seed: int, bank: list, ld_error=LD_ERROR, save_ts: str | None = None) -> dict:
    """One full-length row: long-flux parameters, STAT_KEYS (bank-divided), p_obs, n_nan, and the extra material.

    ``bank`` is the list of control entries (``control_entry``). ``save_ts`` writes the mutated tree sequence there.
    """
    from illex import stats as S
    from illex.abc.simulate import _mutate, _seed_words

    s_inv, _s_ctl, s_mi, _s_mc, s_pi, _s_pc, s_obs, s_ei, _s_ec = _seed_words(seed)
    xr = extra_rng(seed)
    c_long, l_long = draw_long(xr)
    ctl_idx = int(xr.integers(len(bank)))
    sim, ts = inversion_ts(path, s_inv, c_long, l_long)
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    win = (MARGIN_BP, MARGIN_BP + FULL_BODY_BP)
    mts = _mutate(ts, s_mi)
    if save_ts:
        mts.dump(save_ts)
    raw, ext = region_full(mts, i_nodes, s_nodes, win, np.random.default_rng(s_pi), ld_error["body"],
                           np.random.default_rng(s_ei), xr)
    edge = SV.edge_centre(ts, i_nodes, s_nodes, win)
    ctl = bank[ctl_idx]
    out = SV.assemble(raw, ctl["raw"], edge)
    two_n = 2 * N_KARYO
    out["p_obs"] = float(np.random.default_rng(s_obs).binomial(two_n, float(path.p_now)) / two_n)
    out["n_nan"] = int(sum(math.isnan(out[k]) for k in SV.MODEL_KEYS))
    ext["edge_profile"] = edge_profile(ts, i_nodes, s_nodes, win)
    ext["path_t"], ext["path_p"] = _path_points(path)
    out.update(c_long=c_long, l_long=l_long, ctl_idx=ctl_idx, ctl_seed=int(ctl["seed"]), raw_body=raw, x=ext)
    return out
