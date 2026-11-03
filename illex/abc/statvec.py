"""The ABC statistic vector. ``STAT_KEYS`` is the single contract between simulations, data and the fit.

Every statistic is computed on the inversion body of the inversion simulation and, where it has one, divided by
the same statistic on a separate no-inversion neutral control (``simulate.py``), as the data are divided by the
chr2 collinear control. Classes: I = 2a (inverted), S = 2+ (standard).

  pi_ratio, dxy_pi, f1_ratio   pi_I/pi_S, dxy/pi_I, f1(I)/f1(S); branch mode; body value / control value
  sfs_{I,S}_{1..5,6p}          folded SFS projected to PROJ (hypergeometric, ``project_unfolded``), as
                               proportions of the projected segregating sites; f1..f5 and the pooled >=6 bin;
                               site mode; body / control
  jsfs_{a}{b}                  folded joint SFS, both classes projected to PROJ and binned on a 5 x 5 grid
                               (count bins JSFS_BINS); the 12 folded cells left after dropping the monomorphic
                               cell, as proportions; site mode; NOT divided by the control (its near-fixed cells
                               are ~0 under panmixia, so the ratio is undefined)
  ld_{I,S}_{1..6}              mean genotype r^2 (random within-class diploid pairing, 50 diploids per class at
                               N_INV = N_STD = 100; MAF >= 0.05 within the class) in 6 log-spaced distance bins
                               over LD_EDGES; the diploids pass through the data's genotype error model
                               (``error_model``) and r^2 is the pairwise-complete kernel ``ld_block_sums``
                               shared with the data; body / control
  tajd_{I,S}                   Tajima's D per class, site mode; body / control           (PPC only)
  edge_centre                  mean dxy/pi_mean in edge windows over core windows, the definition of
                               ``empirical_windowed.py`` (|rel| > 0.75 vs < 0.35); branch mode; no control (PPC only)

Nothing here raises on degenerate input: an undefined value is NaN.
"""
from __future__ import annotations

import numpy as np

from illex.abc.constants import MU
from illex.scripts.sfs_shape import PROJ, project_unfolded

SFS_BINS = ("1", "2", "3", "4", "5", "6p")
JSFS_BINS = ((0, 0), (1, 4), (5, 15), (16, 19), (20, 20))      # projected-count bins per class, PROJ = 20
LD_EDGES = np.logspace(np.log10(100.0), np.log10(20_000.0), 7)
LD_MAF = 0.05
CORE_REL, EDGE_REL, N_WIN = 0.35, 0.75, 20


def _jsfs_cells():
    nb = len(JSFS_BINS)
    cells = []
    for a in range(nb):
        for b in range(nb):
            mirror = (nb - 1 - a, nb - 1 - b)
            if (a, b) <= mirror and (a, b) != (0, 0):
                cells.append((a, b))
    return tuple(cells)


JSFS_CELLS = _jsfs_cells()
RATIO_KEYS = ("pi_ratio", "dxy_pi", "f1_ratio")
SFS_KEYS = tuple(f"sfs_{c}_{b}" for c in ("I", "S") for b in SFS_BINS)
JSFS_KEYS = tuple(f"jsfs_{a}{b}" for a, b in JSFS_CELLS)
LD_KEYS = tuple(f"ld_{c}_{k + 1}" for c in ("I", "S") for k in range(len(LD_EDGES) - 1))
STAT_KEYS_PPC_ONLY = ("tajd_I", "tajd_S", "edge_centre")
STAT_KEYS = RATIO_KEYS + SFS_KEYS + JSFS_KEYS + LD_KEYS + STAT_KEYS_PPC_ONLY
MODEL_KEYS = tuple(k for k in STAT_KEYS if k not in STAT_KEYS_PPC_ONLY)  # the statistics the ABC fits

assert PROJ == 20 and JSFS_BINS[-1][1] == PROJ


def _div(a, b):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
        return float("nan")
    return a / b


# ------------------------------------------------------------------ one region's raw statistics ----
def _branch_ratios(ts, i_nodes, s_nodes, win):
    """pi_I/pi_S, dxy/pi_I and f1(I)/f1(S) over window ``win`` = (left, right), branch mode."""
    windows = _windows(ts, win)
    k = 1 if win[0] > 0 else 0
    pi = ts.diversity([i_nodes, s_nodes], mode="branch", windows=windows)[k]
    dxy = ts.divergence([i_nodes, s_nodes], mode="branch", windows=windows)[k]
    f1 = []
    for nodes in (i_nodes, s_nodes):
        n = len(nodes)
        af = np.asarray(ts.allele_frequency_spectrum(sample_sets=[list(nodes)], windows=windows, mode="branch",
                                                     polarised=True, span_normalise=True)[k], float)
        tot = af[1:n].sum()
        if tot <= 0:
            f1.append(float("nan"))
            continue
        v = project_unfolded(af / tot, n, PROJ)
        f1.append(_div(v[0], v.sum()))
    return dict(pi_ratio=_div(pi[0], pi[1]), dxy_pi=_div(dxy, pi[0]), f1_ratio=_div(f1[0], f1[1]))


def _windows(ts, win):
    left, right = win
    w = [left, right]
    if left > 0:
        w = [0.0] + w
    if right < ts.sequence_length:
        w = w + [ts.sequence_length]
    return w


def _sfs_bins(counts, n):
    """Projected folded SFS of derived counts out of n, as f1..f5 + pooled >=6 proportions."""
    if counts.size == 0:
        return dict.fromkeys(SFS_BINS, float("nan"))
    return _sfs_bins_unfolded(np.bincount(counts, minlength=n + 1).astype(float), n)


def _sfs_bins_unfolded(unf, n):
    """f1..f5 + pooled >=6 proportions of an unfolded count spectrum ``unf[0..n]``, projected to PROJ and folded.

    Shared by the simulations (per-site counts, ``_sfs_bins``) and the data (realSFS spectra, ``observed.py``).
    """
    out = dict.fromkeys(SFS_BINS, float("nan"))
    v = project_unfolded(np.asarray(unf, float), n, PROJ)
    tot = v.sum()
    if tot <= 0:
        return out
    v = v / tot
    return dict(zip(SFS_BINS, [*v[:5], v[5:].sum()]))


def _hyper_binned(counts, n):
    """(sites, 5) probability that a site with ``counts`` of n lands in each JSFS_BINS bin after projection."""
    from scipy.stats import hypergeom

    kk = np.arange(PROJ + 1)
    u, inv = np.unique(counts, return_inverse=True)
    pmf = hypergeom.pmf(kk[None, :], n, u[:, None], PROJ)
    binned = np.stack([pmf[:, lo:hi + 1].sum(1) for lo, hi in JSFS_BINS], axis=1)
    return binned[inv]


def _jsfs(ci, cs, ni, ns):
    out = dict.fromkeys(JSFS_KEYS, float("nan"))
    if ci.size == 0:
        return out
    return _jsfs_fold(_hyper_binned(ci, ni).T @ _hyper_binned(cs, ns))


def _jsfs_fold(j):
    """The 12 folded JSFS_CELLS of a (5, 5) binned joint spectrum ``j`` (I bins x S bins), as proportions.

    Shared by the simulations (``_jsfs``) and the data (a projected realSFS 2D spectrum, ``observed.py``).
    """
    out = dict.fromkeys(JSFS_KEYS, float("nan"))
    nb = len(JSFS_BINS)
    fol = {}
    for a, b in JSFS_CELLS:
        ma, mb = nb - 1 - a, nb - 1 - b
        fol[(a, b)] = j[a, b] + (j[ma, mb] if (ma, mb) != (a, b) else 0.0)
    tot = sum(fol.values())
    if tot <= 0:
        return out
    return {f"jsfs_{a}{b}": float(v / tot) for (a, b), v in fol.items()}


LD_KERNEL_VERSION = "pairwise-complete-r2-v1"


def _ld(geno, pos, cols, rng, err=None, err_rng=None):
    """Mean genotype r^2 per LD_EDGES bin. geno: (sites, samples) 0/1; cols: this class's sample columns.

    Chromosomes are paired at random into diploids (``rng``). ``err`` = (call_rate, h) passes the diploids through
    the data's genotype error model (``error_model.apply_genotype_errors``, drawn from ``err_rng``) first. r^2 is
    the pairwise-complete kernel ``ld_block_sums``, the one the data use.
    """
    from illex.abc.error_model import apply_genotype_errors

    nbin = len(LD_EDGES) - 1
    out = np.full(nbin, np.nan)
    cols = np.asarray(cols)
    perm = rng.permutation(cols.size)
    m = cols.size // 2
    a, b = cols[perm[:m]], cols[perm[m:2 * m]]
    if m < 2 or geno.shape[0] < 2:
        return out
    d = geno[:, a].astype(np.float64) + geno[:, b]
    if err is not None:
        d = apply_genotype_errors(d, err[0], err[1], err_rng)
    s, c = ld_block_sums(d, pos, np.zeros(d.shape[0], int), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c[0] > 0, s[0] / np.maximum(c[0], 1), np.nan)


def ld_site_keep(dos, maf=LD_MAF):
    """(keep, called, x) for the LD site filter: within-sample MAF over called genotypes >= ``maf``, >= 2 called,
    and varying called dosages. dos: (sites, individuals), missing as -1 (int) or NaN (float); x is dos with 0 at
    missing genotypes."""
    dos = np.asarray(dos)
    with np.errstate(invalid="ignore"):
        called = (np.isfinite(dos) & (dos >= 0)) if dos.dtype.kind == "f" else dos >= 0
    nc = called.sum(1)
    x = np.where(called, dos, 0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        af = x.sum(1) / (2.0 * nc)
        mean = x.sum(1) / nc
        var = (np.where(called, (x - mean[:, None]) ** 2, 0.0)).sum(1)
    keep = (nc >= 2) & (np.minimum(af, 1.0 - af) >= maf) & (var > 1e-12)
    return keep, called, x


def ld_block_sums(dos, pos, blk, n_blocks, maf=LD_MAF, xp=None, step=1024):
    """Per-block, per-LD_EDGES-bin sums and counts of pairwise-complete genotype r^2 -- the ONE LD kernel, used by
    the simulations (``_ld``) and the data (``observed.py``).

    dos: (sites, individuals) dosage 0/1/2, missing as -1 (int) or NaN (float); pos sorted; blk: block id of each
    site (a pair is credited to its left site's block). Sites pass ``ld_site_keep``; a pair counts when it has >= 2
    jointly called individuals and both variances over those are > 0. With no missing data this is the plain
    Pearson r^2 of the dosages. Returns (sums, counts), each (n_blocks, len(LD_EDGES) - 1).
    """
    xp = xp or np
    nbin = len(LD_EDGES) - 1
    sums = np.zeros((n_blocks, nbin))
    cnt = np.zeros((n_blocks, nbin))
    keep, called, x = ld_site_keep(dos, maf)
    if keep.sum() < 2:
        return sums, cnt
    p = np.asarray(pos, float)[keep]
    b = np.asarray(blk)[keep]
    X = xp.asarray(x[keep])
    M = xp.asarray(called[keep].astype(np.float64))
    X2 = X * X
    edges = xp.asarray(LD_EDGES)
    lo_d, hi_d = LD_EDGES[0], LD_EDGES[-1]
    pd = xp.asarray(p)
    bd = xp.asarray(b)
    S_acc = xp.zeros(n_blocks * nbin)
    C_acc = xp.zeros(n_blocks * nbin)
    ns = p.size
    for i0 in range(0, ns, step):
        i1 = min(i0 + step, ns)
        j0 = i0 + 1
        j1 = int(np.searchsorted(p, p[i1 - 1] + hi_d, side="left"))
        if j1 <= j0:
            continue
        Xi, Mi, X2i = X[i0:i1], M[i0:i1], X2[i0:i1]
        Xj, Mj, X2j = X[j0:j1], M[j0:j1], X2[j0:j1]
        n = Mi @ Mj.T
        sx, sy = Xi @ Mj.T, Mi @ Xj.T
        sxx, syy, sxy = X2i @ Mj.T, Mi @ X2j.T, Xi @ Xj.T
        with np.errstate(invalid="ignore", divide="ignore"):
            nn = xp.maximum(n, 1.0)
            cov = sxy - sx * sy / nn
            vx = sxx - sx * sx / nn
            vy = syy - sy * sy / nn
            dist = pd[j0:j1][None, :] - pd[i0:i1][:, None]
            upper = xp.arange(j0, j1)[None, :] > xp.arange(i0, i1)[:, None]
            ok = upper & (dist >= lo_d) & (dist < hi_d) & (n >= 2) & (vx > 1e-9) & (vy > 1e-9)
            r2 = cov * cov / (vx * vy)
        bi = xp.searchsorted(edges, dist[ok], side="right") - 1
        rows = xp.broadcast_to(bd[i0:i1][:, None], ok.shape)[ok]
        idx = rows * nbin + bi
        S_acc += xp.bincount(idx, weights=r2[ok], minlength=n_blocks * nbin)[: n_blocks * nbin]
        C_acc += xp.bincount(idx, minlength=n_blocks * nbin)[: n_blocks * nbin]
    if xp is not np:
        S_acc, C_acc = S_acc.get(), C_acc.get()
    return np.asarray(S_acc).reshape(n_blocks, nbin), np.asarray(C_acc).reshape(n_blocks, nbin)



def _tajd(ts, nodes, win):
    windows = _windows(ts, win)
    k = 1 if win[0] > 0 else 0
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ts.Tajimas_D(sample_sets=[list(nodes)], windows=windows, mode="site")[k]
    v = float(np.ravel(v)[0])
    return v if np.isfinite(v) else float("nan")


def region_stats(ts, i_nodes, s_nodes, win, rng, ld_err=None, err_rng=None) -> dict:
    """Raw (not yet control-divided) statistics of a mutated tree sequence over ``win`` = (left, right).

    ``ts`` must carry binary infinite-sites mutations. ``rng`` draws the within-class diploid pairing.
    ``ld_err`` = {"I": (call_rate, h), "S": (call_rate, h)} applies the data's genotype error model to the LD
    diploids, drawn from ``err_rng`` (None: error-free).
    """
    i_nodes, s_nodes = np.asarray(i_nodes), np.asarray(s_nodes)
    out = _branch_ratios(ts, i_nodes, s_nodes, win)
    pos = ts.tables.sites.position
    sel = (pos >= win[0]) & (pos < win[1])
    if ts.num_sites:
        g = ts.genotype_matrix()[sel]
        g = np.minimum(g, 1).astype(np.int8)             # binary model; guard against any multi-allelic site
    else:
        g = np.zeros((0, ts.num_samples), np.int8)
    p = pos[sel]
    samples = ts.samples()
    col = {int(u): c for c, u in enumerate(samples)}
    ci_cols = np.array([col[int(u)] for u in i_nodes])
    cs_cols = np.array([col[int(u)] for u in s_nodes])
    ci = g[:, ci_cols].sum(1).astype(np.int64)
    cs = g[:, cs_cols].sum(1).astype(np.int64)
    for c, cnt, n in (("I", ci, ci_cols.size), ("S", cs, cs_cols.size)):
        for b, v in _sfs_bins(cnt, n).items():
            out[f"sfs_{c}_{b}"] = float(v)
    out.update(_jsfs(ci, cs, ci_cols.size, cs_cols.size))
    for c, cols in (("I", ci_cols), ("S", cs_cols)):
        err = None if ld_err is None else ld_err[c]
        for k, v in enumerate(_ld(g, p, cols, rng, err, err_rng)):
            out[f"ld_{c}_{k + 1}"] = float(v)
    out["tajd_I"] = _tajd(ts, i_nodes, win)
    out["tajd_S"] = _tajd(ts, s_nodes, win)
    return out


def edge_centre(ts, i_nodes, s_nodes, win) -> float:
    """Edge-over-core mean dxy/pi_mean over N_WIN equal windows of the body (branch mode)."""
    left, right = win
    w = np.linspace(left, right, N_WIN + 1)
    windows = list(w)
    if left > 0:
        windows = [0.0] + windows
    if right < ts.sequence_length:
        windows = windows + [ts.sequence_length]
    k0 = 1 if left > 0 else 0
    pi = ts.diversity([list(i_nodes), list(s_nodes)], mode="branch", windows=windows)[k0:k0 + N_WIN]
    dxy = ts.divergence([list(i_nodes), list(s_nodes)], mode="branch", windows=windows)[k0:k0 + N_WIN]
    pm = pi.mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = dxy / pm
    mid = 0.5 * (w[:-1] + w[1:])
    rel = np.abs(mid - 0.5 * (left + right)) / (0.5 * (right - left))
    core, edge = r[rel < CORE_REL], r[rel > EDGE_REL]
    if not (np.isfinite(core).all() and np.isfinite(edge).all()) or core.size == 0 or edge.size == 0:
        return float("nan")
    return _div(edge.mean(), core.mean())


def assemble(body: dict, control: dict, edge: float) -> dict:
    """Body statistics divided by the control where a control value exists. Returns exactly STAT_KEYS."""
    out = {}
    for k in RATIO_KEYS + SFS_KEYS + LD_KEYS + ("tajd_I", "tajd_S"):
        out[k] = _div(body[k], control[k])
    for k in JSFS_KEYS:
        out[k] = float(body[k])
    out["edge_centre"] = float(edge)
    return {k: out[k] for k in STAT_KEYS}


__all__ = ["STAT_KEYS", "STAT_KEYS_PPC_ONLY", "MODEL_KEYS", "region_stats", "edge_centre", "assemble", "ld_block_sums",
           "MU"]
