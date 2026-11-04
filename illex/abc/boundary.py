"""Breakpoint boundary statistic (PPC-only, never fitted).

Selection on the inverted arrangement also leaves a footprint in the collinear sequence just outside each
breakpoint, which stays linked to the arrangement over a distance of about 1 / (r t). This module measures it in
distance bins ``BND_EDGES_BP`` counted outward from each breakpoint.

  bnd_piratio_b{k}   pi_I / pi_S in bin k, the left and right flanks pooled
  bnd_dxypi_b{k}     dxy / mean(pi_I, pi_S) in bin k, the left and right flanks pooled

Classes: I = 2a (the AA cluster, inverted), S = 2+ (the BB cluster, reference arrangement).

Both sides use one site-based estimator (``site_sums``, the estimator of ``observed.site_stats``), which takes
per-site pi within each class and dxy between them from called dosages with per-site called n, summed over the sites of a
bin, then divided. On the simulated side the haploid samples are paired at random into diploids and passed
through the data's genotype error model first (``error_model.apply_genotype_errors``) with the CONTROL-region
parameters, since the flanks are collinear.

The distance of a site at position x (continuous, or a 0-based base given at its centre x + 0.5) is
left flank d = bp_left - x for x < bp_left, right flank d = x - bp_right for x >= bp_right, and the site is in
bin k when edges[k] <= d < edges[k + 1]. Sites inside [bp_left, bp_right) are never counted.
"""
from __future__ import annotations

import numpy as np

from illex.abc.constants import BND_EDGES_BP
from illex.abc.error_model import LD_ERROR, apply_genotype_errors
from illex.abc.statvec import BND_KEYS

N_BND = len(BND_EDGES_BP) - 1
assert len(BND_KEYS) == 2 * N_BND
SIDES = ("left", "right")


def _div(a, b):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
        return float("nan")
    return a / b


def flank_bins(pos, bp_left, bp_right, edges=BND_EDGES_BP):
    """(side, bin) per site, with side 0 = left, 1 = right and -1 outside every flank bin (where bin is -1 too)."""
    x = np.asarray(pos, float)
    e = np.asarray(edges, float)
    left = x < bp_left
    right = x >= bp_right
    d = np.where(left, bp_left - x, np.where(right, x - bp_right, np.nan))
    with np.errstate(invalid="ignore"):
        k = np.searchsorted(e, d, side="right") - 1
        ok = (left | right) & np.isfinite(d) & (d >= e[0]) & (d < e[-1])
    side = np.where(ok, np.where(left, 0, 1), -1)
    return side, np.where(ok, k, -1)


def site_sums(dos_i, dos_s):
    """Per-site pi_I, pi_S and dxy (``observed.site_stats``) of dosages (sites, individuals), missing < 0 or NaN.

    Each value uses the per-site called n, and a site where it is undefined contributes 0. Also returns a
    per-site segregating flag (both alleles present among the called I and S genotypes).
    """
    out, fr, alt, tot = {}, {}, 0.0, 0.0
    for c, d in (("I", dos_i), ("S", dos_s)):
        d = np.asarray(d, np.float64)
        ok = np.isfinite(d) & (d >= 0)
        n = 2.0 * ok.sum(1)
        a = np.where(ok, d, 0.0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"pi_{c}"] = np.nan_to_num(2.0 * a * (n - a) / (n * (n - 1.0)))
            fr[c] = a / n
        alt, tot = alt + a, tot + n
    with np.errstate(invalid="ignore"):
        out["dxy"] = np.nan_to_num(fr["I"] * (1 - fr["S"]) + fr["S"] * (1 - fr["I"]))
    out["seg"] = (alt > 0) & (alt < tot)
    return out


def bin_ratios(sums):
    """{piratio, dxypi} of summed {pi_I, pi_S, dxy}."""
    return dict(piratio=_div(sums["pi_I"], sums["pi_S"]),
                dxypi=_div(sums["dxy"], 0.5 * (sums["pi_I"] + sums["pi_S"])))


def bin_sums(st, side, k, keep=None, sides=(0, 1)):
    """Per-bin sums of pi_I, pi_S, dxy and the count of segregating sites over the given sides."""
    sel = np.isin(side, sides) & (k >= 0)
    if keep is not None:
        sel &= keep
    out = {}
    for key in ("pi_I", "pi_S", "dxy"):
        out[key] = np.bincount(k[sel], weights=st[key][sel], minlength=N_BND)[:N_BND]
    out["n_seg"] = np.bincount(k[sel & st["seg"]], minlength=N_BND)[:N_BND]
    return out


def keyed(sums):
    """The BND_KEYS dict from per-bin sums (``bin_sums``)."""
    out = {}
    for b in range(N_BND):
        r = bin_ratios({key: sums[key][b] for key in ("pi_I", "pi_S", "dxy")})
        out[f"bnd_piratio_b{b}"] = r["piratio"]
        out[f"bnd_dxypi_b{b}"] = r["dxypi"]
    return {key: out[key] for key in BND_KEYS}


def diploid_dosages(g, cols, rng, err=None, err_rng=None):
    """(sites, m) float dosages of the haploid columns ``cols`` of ``g`` paired at random into m diploids.

    ``err`` = (call_rate, h) passes them through the data's genotype error model (drawn from ``err_rng``).
    An odd chromosome is dropped.
    """
    cols = np.asarray(cols)
    perm = rng.permutation(cols.size)
    m = cols.size // 2
    d = g[:, cols[perm[:m]]].astype(np.float64) + g[:, cols[perm[m:2 * m]]]
    if err is not None:
        d = apply_genotype_errors(d, err[0], err[1], err_rng)
    return d


def boundary_stats(ts, i_nodes, s_nodes, bp_left, bp_right, edges=BND_EDGES_BP, rng=None,
                   err=LD_ERROR["control"], err_rng=None, per_side=False) -> dict:
    """``BND_KEYS`` of a mutated tree sequence, pi_I/pi_S and dxy/mean(pi_I, pi_S) per flank bin with sides pooled.

    ``rng`` draws the within-class diploid pairing and ``err_rng`` the genotype errors (both default to a fixed
    seed 0). ``err`` = {"I": (call_rate, h), "S": (call_rate, h)}, by default the control-region parameters of
    ``error_model.LD_ERROR``. ``err=None`` skips the diploid pairing and the errors, and uses the haploid
    genotypes directly. ``per_side=True`` adds "left" / "right" dicts of the same keys and "n_seg" site counts.
    """
    if tuple(edges) != tuple(BND_EDGES_BP):
        raise ValueError(f"edges must be BND_EDGES_BP {BND_EDGES_BP} (the key names fix the bins)")
    rng = np.random.default_rng(0) if rng is None else rng
    err_rng = np.random.default_rng(0) if err_rng is None else err_rng
    pos = ts.tables.sites.position
    side, k = flank_bins(pos, bp_left, bp_right, edges)
    sel = side >= 0
    samples = ts.samples()
    col = {int(u): c for c, u in enumerate(samples)}
    ci = np.array([col[int(u)] for u in i_nodes])
    cs = np.array([col[int(u)] for u in s_nodes])
    if ts.num_sites:
        g = np.minimum(ts.genotype_matrix()[sel], 1).astype(np.int8)
    else:
        g = np.zeros((0, ts.num_samples), np.int8)
    if err is None:
        st =_haploid_sums(g[:, ci], g[:, cs])
    else:
        di = diploid_dosages(g, ci, rng, err["I"], err_rng)
        ds = diploid_dosages(g, cs, rng, err["S"], err_rng)
        st = site_sums(di, ds)
    side, k = side[sel], k[sel]
    out = keyed(bin_sums(st, side, k))
    if per_side:
        out["n_seg"] = bin_sums(st, side, k)["n_seg"].tolist()
        for j, name in enumerate(SIDES):
            s = bin_sums(st, side, k, sides=(j,))
            out[name] = keyed(s)
            out[name]["n_seg"] = s["n_seg"].tolist()
    return out


def _haploid_sums(gi, gs):
    """site_sums of haploid 0/1 genotypes (no pairing, no errors), with n the number of chromosomes."""
    out, fr = {}, {}
    for c, g in (("I", gi), ("S", gs)):
        n = float(g.shape[1])
        a = g.sum(1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"pi_{c}"] = np.nan_to_num(2.0 * a * (n - a) / (n * (n - 1.0)))
            fr[c] = a / n
    out["dxy"] = np.nan_to_num(fr["I"] * (1 - fr["S"]) + fr["S"] * (1 - fr["I"]))
    tot = gi.sum(1) + gs.sum(1)
    out["seg"] = (tot > 0) & (tot < gi.shape[1] + gs.shape[1])
    return out
