#!/usr/bin/env python
"""The observed ``STAT_KEYS`` vector for the chr2 trajectory ABC -> results/illex/abc_observed.json.

    bash illex/scripts/run_abc_observed_angsd.sh          # realSFS spectra -> .tmp/abc_observed (once)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD \\
        /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python -m illex.abc.observed

Every statistic is the data-side twin of ``illex/abc/statvec.py`` (the simulation side), with the same bins,
projection, folding and body / control division. Classes: I = 2a = the AA cluster (254 homokaryotypes),
S = 2+ = the BB cluster (95). Heterokaryotypes are not used.

Regions (0-based half-open):
  body     [BP_LEFT, BP_RIGHT) = chr2 60.55-79.52 Mb, the inversion interior
  control  [10, 30) Mb, the collinear REGIONS control of ``empirical_windowed.py`` (far outside a 2 Mb
           breakpoint buffer). Each class is compared against ITSELF on the control, as the simulation's control
           is split into the same two groups.

Sources per statistic:
  pi_ratio, dxy_pi    called genotypes (``.tmp/illex_chr2/{inv,ctl}.vcf.gz``, the depth-filtered callset, every
                      variant of which lies in the accessible mask) as ratios of per-site sums with per-site called
                      n -- the estimator of ``empirical_jackknife.py`` (pg_gpu windowed pi/dxy, missing="include")
  f1_ratio, sfs_*     ANGSD realSFS within-class spectra at full n (steps/04_angsd_chr2 SAFs, accessible sites),
                      projected to PROJ = 20 with ``statvec._sfs_bins_unfolded``
  jsfs_*              ANGSD realSFS 2D AA40 x BB40 (80 chromosomes per class), projected and binned with
                      ``statvec._hyper_binned`` / ``statvec._jsfs_fold``; no control
  ld_*                called genotypes, 20 random subsamples of 50 individuals per class (the simulation's 50
                      diploids), dosage 0/1/2, MAF >= LD_MAF within the subsample, pairwise-complete r^2
                      (``statvec.ld_block_sums``, the kernel the simulations use), LD_EDGES bins
  tajd_*              ANGSD within-class spectra projected to the simulated n (100 per class) and evaluated as
                      Tajima's D of a BODY_BP = CONTROL_BP region with those per-site spectra
  edge_centre         called genotypes, N_WIN equal windows over the body, statvec.edge_centre's definition
  boundary            (a separate "boundary" block, PPC-only, ``boundary_block``) called genotypes in the flanks
                      [BP_LEFT - 100 kb, BP_LEFT) and [BP_RIGHT, BP_RIGHT + 100 kb), read from inv.vcf.gz, as
                      pi_I/pi_S and dxy/mean(pi_I, pi_S) per distance bin, per side and pooled, with a 10-kb-block
                      bootstrap SE within each bin
SEs: delete-one 1-Mb-block jackknife over the body (19 blocks from BP_LEFT, the last 0.97 Mb), each statistic
recomputed with its control held fixed. realSFS cannot drop a hole from a region, so the ANGSD statistics use
per-block spectra summed over the retained blocks (the ``sfs_f1_jackknife.py`` route; its gate is recorded).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from illex.abc import statvec as SV
from illex.abc.error_model import genotype_error_stats, h_from_ho_he
from illex.abc.statvec import LD_KERNEL_VERSION, ld_block_sums  # noqa: F401  one kernel for sim and data
from illex.abc.constants import (BND_EDGES_BP, BND_MARGIN_BP, BODY_BP, BP_LEFT, BP_RIGHT, CONTROL_BP, N_INV, N_STD,
                                 P_NOW)

ANA = Path("/sietch_colab/data_share/illex/popgen_data/analysis/steps")
KARYO = {"I": ANA / "03_karyotype/AA_samples.txt", "S": ANA / "03_karyotype/BB_samples.txt"}
VCF_DIR = Path("/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/illex_chr2")
VCF = {"body": VCF_DIR / "inv.vcf.gz", "control": VCF_DIR / "ctl.vcf.gz"}
ACC_BED = ANA / "03_karyotype/chr2_mask/acc2.bed"
SAF = {"I": ANA / "04_angsd_chr2/saf/AA.2.saf.idx", "S": ANA / "04_angsd_chr2/saf/BB.2.saf.idx"}
SAF40 = Path("/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/sfs2d")
WORK = Path(".tmp/abc_observed")
OUT = Path("results/illex/abc_observed.json")

CTL_LO, CTL_HI = 10_000_000, 30_000_000
# Breakpoint boundary flanks (0-based half-open), read from inv.vcf.gz (it spans 60-80 Mb). They are kept out of VCF
# so that meta["inputs"] of the existing statistics is unchanged.
FLANKS = {"flank_left": (BP_LEFT - BND_MARGIN_BP, BP_LEFT), "flank_right": (BP_RIGHT, BP_RIGHT + BND_MARGIN_BP)}
FLANK_VCF = VCF["body"]
BND_BLOCK = 10_000                 # boundary bootstrap block, bp of distance from the breakpoint
BND_BOOT, BND_SEED = 1000, 20261104
BND_MIN_SEG = 50                   # a bin with fewer accessible segregating sites is uninformative (dropped)
BLOCK = 1_000_000
BLOCK_STARTS = tuple(range(BP_LEFT, BP_RIGHT, BLOCK))
BP_RIGHT_2D = 79_500_000                       # the AA40/BB40 SAFs end here
CLS_FILE = {"I": "AA", "S": "BB"}
N_DIP = {"I": 254, "S": 95}
N_SIM = {"I": N_INV, "S": N_STD}
LD_N, LD_REPS, LD_SEED = N_INV // 2, 20, 20261103
RATIO_KEYS = SV.RATIO_KEYS
DIVIDED = RATIO_KEYS + SV.SFS_KEYS + SV.LD_KEYS + ("tajd_I", "tajd_S")


GENO_VERSION = "stacked-dosage-v1"


def _hash(d) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _span(label):
    """(vcf, lo, hi) of the region label ``body``, ``control``, ``flank_left`` or ``flank_right``."""
    if label in FLANKS:
        return (FLANK_VCF, *FLANKS[label])
    lo, hi = (BP_LEFT, BP_RIGHT) if label == "body" else (CTL_LO, CTL_HI)
    return VCF[label], lo, hi


def geno_params(label):
    vcf, lo, hi = _span(label)
    return {"vcf": str(vcf), "region": [lo, hi], "layout": GENO_VERSION}


def vcf_part_params():
    """Everything the cached called-genotype part depends on; a mismatch forces recomputation."""
    return {"geno": {lab: _hash(geno_params(lab)) for lab in ("body", "control")},
            "karyotypes": {c: str(f) for c, f in KARYO.items()}, "ld_seed": LD_SEED, "ld_n": LD_N,
            "ld_reps": LD_REPS, "maf": SV.LD_MAF, "edges": SV.LD_EDGES.tolist(), "kernel": LD_KERNEL_VERSION,
            "block": BLOCK, "body": [BP_LEFT, BP_RIGHT], "n_win": SV.N_WIN}


def ld_subsamples():
    """The fixed LD subsamples: {class: [LD_REPS sorted index arrays into that class's sample list]}."""
    rng = np.random.default_rng(LD_SEED)
    return {c: [np.sort(rng.choice(N_DIP[c], LD_N, replace=False)) for _ in range(LD_REPS)] for c in ("I", "S")}


# ======================================================================== pure kernels (CPU-testable) ====
def sfs_bins(unf, n):
    """f1..f5 + pooled >=6 proportions of an unfolded spectrum of n chromosomes (statvec's transform)."""
    return {b: float(v) for b, v in SV._sfs_bins_unfolded(unf, n).items()}


def jsfs_from_2d(m):
    """The 12 folded JSFS cells of an unfolded 2D spectrum ``m[i, j]`` (i of I's n_i, j of S's n_s)."""
    m = np.asarray(m, float)
    bi = SV._hyper_binned(np.arange(m.shape[0]), m.shape[0] - 1)
    bs = SV._hyper_binned(np.arange(m.shape[1]), m.shape[1] - 1)
    return SV._jsfs_fold(bi.T @ m @ bs)


def tajd_from_unfolded(unf, n_from, n_to, length):
    """Tajima's D of a ``length``-bp region whose per-site spectrum is ``unf`` / unf.sum(), at n_to chromosomes.

    ``unf[0..n_from]`` counts sites (monomorphic included), so unf.sum() is the number of sites. The spectrum is
    projected to n_to (exact, ``project_unfolded``), and pi and S are rescaled to ``length`` bp, so D carries the
    same S-dependence as a site-mode tskit D over a simulated region of that length (tskit's formula).
    """
    from illex.scripts.sfs_shape import project_unfolded

    unf = np.asarray(unf, float)
    L = unf.sum()
    if L <= 0:
        return float("nan")
    n = int(n_to)
    fol = project_unfolded(unf, n_from, n)                 # folded bins k = 1..n//2
    k = np.arange(1, fol.size + 1)
    scale = float(length) / L
    S = fol.sum() * scale
    pi = (fol * k * (n - k)).sum() / (n * (n - 1) / 2.0) * scale
    i = np.arange(1, n)
    a1, a2 = (1.0 / i).sum(), (1.0 / i**2).sum()
    b1 = (n + 1) / (3.0 * (n - 1))
    b2 = 2.0 * (n * n + n + 3) / (9.0 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2
    e1, e2 = c1 / a1, c2 / (a1**2 + a2)
    den = math.sqrt(e1 * S + e2 * S * (S - 1.0)) if S > 1 else 0.0
    return float((pi - S / a1) / den) if den > 0 else float("nan")


def _xp():
    try:
        import cupy as cp

        cp.zeros(1)
        return cp
    except Exception:  # noqa: BLE001  no GPU / no cupy: numpy
        return np


def jackknife_se(partials):
    """Delete-one jackknife SE from the B leave-one-out values (NaN if any is non-finite)."""
    v = np.asarray(partials, float)
    if v.size < 2 or not np.isfinite(v).all():
        return float("nan")
    b = v.size
    return float(np.sqrt((b - 1) / b * ((v - v.mean()) ** 2).sum()))


# ============================================================================ data loading (pg_gpu) ====
def _read_samples(f):
    return [x.strip() for x in Path(f).read_text().split() if x.strip()]


def load_region(label):
    """(pos 1-based, dosage (individuals, sites) int8 with -1 missing, sample names) from the chr2 VCF, cached.

    pg_gpu's HaplotypeMatrix stacks haplotypes as [all first copies; all second copies] (rows i and i + n_ind are
    individual i: their missingness agrees exactly, and only that pairing reproduces the callset's genotype
    counts). ``GenotypeMatrix.from_haplotype_matrix`` pairs rows (2i, 2i+1) instead, which joins different
    individuals for a VCF-loaded matrix, so dosages are built here from the stacked layout and checked.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / f"{label}.geno.npz"
    side = WORK / f"{label}.geno.json"
    want = _hash(geno_params(label))
    if cache.exists() and side.exists() and json.loads(side.read_text()).get("hash") == want:
        z = np.load(cache, allow_pickle=False)
        return z["pos"], z["dos"], [str(s) for s in z["samples"]]
    from pg_gpu import HaplotypeMatrix

    vcf, lo, hi = _span(label)
    h = HaplotypeMatrix.from_vcf(str(vcf), region=f"2:{lo + 1}-{hi}")
    H = h.haplotypes
    H = np.asarray(H.get() if hasattr(H, "get") else H)
    pos = np.asarray(h.positions.get() if hasattr(h.positions, "get") else h.positions).astype(np.int64)
    n = H.shape[0] // 2
    a, b = H[:n], H[n:]
    if not np.array_equal(a < 0, b < 0):
        raise SystemExit("haplotype layout is not stacked: per-individual missingness disagrees")
    dos = np.where(a < 0, -1, a + b).astype(np.int8)
    samples = [str(s) for s in h.samples]
    sel = (pos - 1 >= lo) & (pos - 1 < hi)
    pos, dos = pos[sel], dos[:, sel]
    np.savez(cache, pos=pos, dos=dos, samples=np.array(samples))
    side.write_text(json.dumps({"hash": want, "params": geno_params(label)}))
    return pos, dos, samples


def accessible(pos):
    bed = np.loadtxt(ACC_BED, usecols=(1, 2), dtype=np.int64)
    p0 = np.asarray(pos) - 1
    k = np.searchsorted(bed[:, 0], p0, side="right") - 1
    return (k >= 0) & (p0 < bed[np.maximum(k, 0), 1])


def site_stats(dos, ii, ss):
    """Per-site pi_I, pi_S (per-site called n) and dxy, undefined sites contributing 0 (pg_gpu 'include')."""
    out = {}
    fr = {}
    for c, idx in (("I", ii), ("S", ss)):
        d = dos[idx]
        ok = d >= 0
        n = 2.0 * ok.sum(0)
        a = np.where(ok, d, 0).sum(0).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"pi_{c}"] = np.nan_to_num(2.0 * a * (n - a) / (n * (n - 1.0)))
            fr[c] = a / n
    with np.errstate(invalid="ignore"):
        out["dxy"] = np.nan_to_num(fr["I"] * (1 - fr["S"]) + fr["S"] * (1 - fr["I"]))
    return out


def _ratios(s):
    return dict(pi_ratio=SV._div(s["pi_I"], s["pi_S"]), dxy_pi=SV._div(s["dxy"], s["pi_I"]))


def _edge(pos, st, keep):
    """statvec.edge_centre on data: N_WIN equal windows over the body, edge/core mean dxy / mean(pi_I, pi_S)."""
    w = np.linspace(BP_LEFT, BP_RIGHT, SV.N_WIN + 1)
    wi = np.clip(np.searchsorted(w, pos - 1, side="right") - 1, 0, SV.N_WIN - 1)
    agg = {k: np.bincount(wi[keep], weights=st[k][keep], minlength=SV.N_WIN) for k in ("pi_I", "pi_S", "dxy")}
    with np.errstate(invalid="ignore", divide="ignore"):
        r = agg["dxy"] / (0.5 * (agg["pi_I"] + agg["pi_S"]))
    mid = 0.5 * (w[:-1] + w[1:])
    rel = np.abs(mid - 0.5 * (BP_LEFT + BP_RIGHT)) / (0.5 * (BP_RIGHT - BP_LEFT))
    core, edge = r[rel < SV.CORE_REL], r[rel > SV.EDGE_REL]
    # A jackknife deletion can empty a ~0.95 Mb window; drop it rather than lose the replicate. On the full body
    # every window is finite, so the point value is statvec.edge_centre's exactly.
    core, edge = core[np.isfinite(core)], edge[np.isfinite(edge)]
    if core.size == 0 or edge.size == 0:
        return float("nan")
    return SV._div(edge.mean(), core.mean())


def _sfs_file(cls, lab):
    f = WORK / f"{CLS_FILE[cls]}.{lab}.sfs"
    v = np.array(f.read_text().split(), float)
    if v.size != 2 * N_DIP[cls] + 1:
        raise SystemExit(f"{f}: {v.size} entries, expected {2 * N_DIP[cls] + 1} (realSFS still running?)")
    return v


def _sfs2d_file(lab):
    f = WORK / f"AA_BB.{lab}.2dSFS"
    v = np.array(f.read_text().split(), float)
    if v.size != 81 * 81:
        raise SystemExit(f"{f}: {v.size} entries, expected 6561 (realSFS still running?)")
    return v.reshape(81, 81)


def _angsd_stats(unf_i, unf_s):
    """f1_ratio, sfs_*, tajd_* from the two within-class spectra of one region."""
    out = {}
    bins = {}
    for c, unf in (("I", unf_i), ("S", unf_s)):
        bins[c] = sfs_bins(unf, 2 * N_DIP[c])
        for b, v in bins[c].items():
            out[f"sfs_{c}_{b}"] = v
        out[f"tajd_{c}"] = tajd_from_unfolded(unf, 2 * N_DIP[c], N_SIM[c], BODY_BP)
    out["f1_ratio"] = SV._div(bins["I"]["1"], bins["S"]["1"])
    return out


def estimate_error_model(names):
    """call_rate and h = 1 - Ho/He per region and class, on the LD SNP set of each of the fixed LD subsamples
    (``statvec.ld_site_keep``), averaged over subsamples. These parameterise ``error_model.LD_ERROR``."""
    subs = ld_subsamples()
    out = {}
    for lab in ("body", "control"):
        pos, dos, samples = load_region(lab)
        col = {s: i for i, s in enumerate(samples)}
        keep = accessible(pos)
        out[lab] = {}
        for c in ("I", "S"):
            idx = np.array([col[s] for s in names[c]])
            cr, hh = [], []
            for sub in subs[c]:
                d = dos[idx[sub]][:, keep].T
                k, _, _ = SV.ld_site_keep(d)
                a, b = genotype_error_stats(d[k])
                cr.append(a), hh.append(b)
            ho_he = float(np.mean(hh))
            out[lab][c] = {"call_rate": float(np.mean(cr)), "ho_he": ho_he, "h": h_from_ho_he(ho_he),
                           "call_rate_sd": float(np.std(cr)), "ho_he_sd": float(np.std(hh))}
    return out


# ================================================================================ breakpoint boundary ====
def accessible_bp(lo, hi):
    """Accessible bases of [lo, hi) (0-based half-open) in the chr2 mask."""
    bed = np.loadtxt(ACC_BED, usecols=(1, 2), dtype=np.int64)
    return int(np.clip(np.minimum(bed[:, 1], hi) - np.maximum(bed[:, 0], lo), 0, None).sum())


def bin_span(side, k):
    """[lo, hi) 0-based bases of boundary bin k on a side ("left" / "right"), with sites binned at their centre."""
    e = BND_EDGES_BP
    if side == "left":
        return BP_LEFT - e[k + 1], BP_LEFT - e[k]
    return BP_RIGHT + e[k], BP_RIGHT + e[k + 1]


def _flank_sites(lab, names):
    """Per-site sums, bins and bootstrap blocks of the accessible called sites of one flank."""
    from illex.abc import boundary as B

    pos, dos, samples = load_region(lab)
    col = {s: i for i, s in enumerate(samples)}
    ii, ss = (np.array([col[s] for s in names[c]]) for c in ("I", "S"))
    keep = accessible(pos) & ((dos[ii] >= 0).any(0) | (dos[ss] >= 0).any(0))
    st = B.site_sums(dos[ii].T, dos[ss].T)          # = site_stats(dos, ii, ss), plus a segregating flag
    x = pos - 1 + 0.5                               # 0-based base, at its centre
    side, k = B.flank_bins(x, BP_LEFT, BP_RIGHT)
    want = 0 if lab == "flank_left" else 1
    if not np.all(side[keep] == want):
        raise SystemExit(f"{lab}: sites outside the flank")
    d = np.where(side == 0, BP_LEFT - x, x - BP_RIGHT)
    lo_edge = np.asarray(BND_EDGES_BP, float)[np.maximum(k, 0)]
    blk = ((d - lo_edge) // BND_BLOCK).astype(int)  # 10-kb blocks counted from each bin's inner edge
    return {"st": {key: st[key][keep] for key in ("pi_I", "pi_S", "dxy", "seg")}, "k": k[keep],
            "blk": blk[keep], "n_vcf": int(pos.size)}


def _boot_se(units, rng):
    """SE of (pi_I/pi_S, dxy/mean pi) from resampling the (n_units, 3) unit sums with replacement."""
    from illex.abc import boundary as B

    u = np.asarray(units, float)
    if u.shape[0] < 2:
        return {"piratio": float("nan"), "dxypi": float("nan")}
    idx = rng.integers(0, u.shape[0], size=(BND_BOOT, u.shape[0]))
    tot = u[idx].sum(1)                              # (BND_BOOT, 3)
    r = [B.bin_ratios(dict(zip(("pi_I", "pi_S", "dxy"), t))) for t in tot]
    out = {}
    for key in ("piratio", "dxypi"):
        v = np.array([x[key] for x in r])
        v = v[np.isfinite(v)]
        out[key] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return out


def boundary_block(names):
    """The abc_observed.json "boundary" block: BND_KEYS per bin pooled over the two flanks and per side.

    The called-genotype estimator of pi_ratio / dxy_pi (``site_stats``, accessible sites with >= 1 called
    genotype in I or S), summed per bin and divided. The SEs come from a bootstrap of 10-kb blocks (counted outward from each
    bin's inner edge, so a block never straddles two bins) within each bin, the two sides' blocks pooled for
    the pooled value. Bins 0-1 and 1-5 kb are narrower than a block, so they have one block per side, and their
    pooled block SE rests on two blocks (their per-side block SE is undefined). A site-level bootstrap SE, which
    ignores linkage, is recorded beside it. A bin with fewer than BND_MIN_SEG accessible segregating sites is
    flagged uninformative and left out of "ppc_keys".
    """
    from illex.abc import boundary as B

    rng = np.random.default_rng(BND_SEED)
    fl = {"left": _flank_sites("flank_left", names), "right": _flank_sites("flank_right", names)}
    keys3 = ("pi_I", "pi_S", "dxy")
    res = {"stats": {}, "se": {}, "se_site_bootstrap": {}, "per_side": {"left": {}, "right": {}},
           "se_per_side": {"left": {}, "right": {}}, "counts": {}, "informative": {}}
    for group, sides in (("pooled", ("left", "right")), ("left", ("left",)), ("right", ("right",))):
        cnt = {"n_sites": [], "n_seg": [], "accessible_bp": [], "n_blocks": []}
        vals, ses, ses_site = {}, {}, {}
        for b in range(B.N_BND):
            units, sites = [], []
            n_sites = n_seg = acc_bp = 0
            for sd in sides:
                f = fl[sd]
                m = f["k"] == b
                st = {key: f["st"][key][m] for key in f["st"]}
                n_sites += int(m.sum())
                n_seg += int(st["seg"].sum())
                acc_bp += accessible_bp(*bin_span(sd, b))
                blk = f["blk"][m]
                for u in np.unique(blk):
                    w = blk == u
                    units.append([st[key][w].sum() for key in keys3])
                sites.append(np.stack([st[key] for key in keys3], 1))
            sites = np.concatenate(sites, 0)
            r = B.bin_ratios(dict(zip(keys3, sites.sum(0))))
            se_b, se_s = _boot_se(units, rng), _boot_se(sites, rng)
            for key in ("piratio", "dxypi"):
                name = f"bnd_{key}_b{b}"
                vals[name], ses[name], ses_site[name] = r[key], se_b[key], se_s[key]
            cnt["n_sites"].append(n_sites), cnt["n_seg"].append(n_seg)
            cnt["accessible_bp"].append(acc_bp), cnt["n_blocks"].append(len(units))
        order = list(B.BND_KEYS)
        vals, ses, ses_site = ({k: d[k] for k in order} for d in (vals, ses, ses_site))
        if group == "pooled":
            res["stats"], res["se"], res["se_site_bootstrap"] = vals, ses, ses_site
        else:
            res["per_side"][group], res["se_per_side"][group] = vals, ses
            res.setdefault("se_site_bootstrap_per_side", {})[group] = ses_site
        res["counts"][group] = cnt
        res["informative"][group] = [n >= BND_MIN_SEG for n in cnt["n_seg"]]
    res["ppc_keys"] = [f"bnd_{key}_b{b}" for key in ("piratio", "dxypi") for b in range(B.N_BND)
                       if res["informative"]["pooled"][b]]
    res["meta"] = {
        "edges_bp": list(BND_EDGES_BP), "margin_bp": BND_MARGIN_BP,
        "flanks": {k: list(v) for k, v in FLANKS.items()}, "vcf": str(FLANK_VCF), "accessible_bed": str(ACC_BED),
        "vcf_variants": {sd: fl[sd]["n_vcf"] for sd in fl},
        "classes": "I = 2a = AA cluster (inverted), S = 2+ = BB cluster (reference arrangement)",
        "estimator": "per-site pi_I, pi_S, dxy with per-site called n (site_stats), summed per bin, then "
                     "bnd_piratio = sum pi_I / sum pi_S, bnd_dxypi = sum dxy / mean(sum pi_I, sum pi_S)",
        "distance": "left d = BP_LEFT - (p0 + 0.5), right d = (p0 + 0.5) - BP_RIGHT, bin k when edges[k] <= d < "
                    "edges[k + 1]",
        "n_seg": "accessible sites with both alleles among the called I and S genotypes",
        "n_sites": "accessible VCF sites with >= 1 called I or S genotype",
        "bootstrap": {"block_bp": BND_BLOCK, "replicates": BND_BOOT, "seed": BND_SEED,
                      "blocks": "10-kb blocks from each bin's inner edge, within the bin, sides pooled for pooled"},
        "min_seg": BND_MIN_SEG, "control": "none (raw values, as the simulated boundary_ppc)",
        "undefined": "null (a bin with no sites, or a bootstrap with fewer than two blocks)"}
    return _json_safe(res)


def _json_safe(x):
    """Non-finite floats to None, so the block is strict JSON (jsonlite reads abc_observed.json)."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ================================================================================================ main ====
def _vcf_part(names, xp, raw, ctl, se_body, meta, t0):
    """pi_ratio, dxy_pi, edge_centre and ld_* (body raw, control, body jackknife partials) from called genotypes."""
    geno = {}
    for lab in ("body", "control"):
        pos, dos, samples = load_region(lab)
        col = {s: i for i, s in enumerate(samples)}
        idx = {c: np.array([col[s] for s in names[c]]) for c in ("I", "S")}
        acc = accessible(pos)
        keep = acc & ((dos[idx["I"]] >= 0).any(0) | (dos[idx["S"]] >= 0).any(0))
        geno[lab] = (pos, dos, idx, keep)
        meta["regions"][lab] = {"vcf_variants": int(pos.size), "vcf_variants_accessible": int(acc.sum()),
                                "vcf_variants_used": int(keep.sum())}
        print(f"{lab}: {pos.size:,} variants, {acc.sum():,} accessible  ({time.time() - t0:.0f}s)", flush=True)
    meta["samples"] = {"I_2aa": int(geno["body"][2]["I"].size), "S_2pp": int(geno["body"][2]["S"].size)}

    pos_b, dos_b, idx_b, keep_b = geno["body"]
    blk_b = ((pos_b - 1 - BP_LEFT) // BLOCK).astype(int)
    nblk = len(BLOCK_STARTS)
    st_b = site_stats(dos_b, idx_b["I"], idx_b["S"])
    blk_sum = {k: np.bincount(blk_b[keep_b], weights=v[keep_b], minlength=nblk) for k, v in st_b.items()}
    raw.update(_ratios({k: v.sum() for k, v in blk_sum.items()}))
    part_ratio = [_ratios({k: v.sum() - v[b] for k, v in blk_sum.items()}) for b in range(nblk)]
    pos_c, dos_c, idx_c, keep_c = geno["control"]
    st_c = site_stats(dos_c, idx_c["I"], idx_c["S"])
    ctl.update(_ratios({k: v[keep_c].sum() for k, v in st_c.items()}))
    for k in RATIO_KEYS[:2]:
        se_body[k] = [p[k] for p in part_ratio]

    raw["edge_centre"] = _edge(pos_b, st_b, keep_b)
    se_body["edge_centre"] = [_edge(pos_b, st_b, keep_b & (blk_b != b)) for b in range(nblk)]
    print(f"ratios/edge done ({time.time() - t0:.0f}s)", flush=True)

    # LD: same 20 subsamples of 50 individuals for body and control, per class
    all_subs = ld_subsamples()
    nbin = len(SV.LD_EDGES) - 1
    ld_meta = {}
    for c in ("I", "S"):
        subs = all_subs[c]
        bs, bc, cs, cc, nsnp = [], [], [], [], {"body": [], "control": []}
        for r, sub in enumerate(subs):
            for lab, (pos, dos, idx, keep) in geno.items():
                d = dos[idx[c][sub]][:, keep].T
                p = pos[keep]
                if lab == "body":
                    s, n = ld_block_sums(d, p, blk_b[keep], nblk, xp=xp)
                    bs.append(s), bc.append(n)
                else:
                    s, n = ld_block_sums(d, p, np.zeros(p.size, int), 1, xp=xp)
                    cs.append(s[0]), cc.append(n[0])
                nsnp[lab].append(int(n.sum()))
            print(f"  LD {c} subsample {r + 1}/{LD_REPS}  ({time.time() - t0:.0f}s)", flush=True)
        bs, bc, cs, cc = map(np.array, (bs, bc, cs, cc))           # (reps, blocks, bins) / (reps, bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            body = np.nanmean(bs.sum(1) / bc.sum(1), axis=0)
            cont = np.nanmean(cs / cc, axis=0)
            parts = np.array([np.nanmean((bs.sum(1) - bs[:, b]) / (bc.sum(1) - bc[:, b]), axis=0)
                              for b in range(nblk)])            # (blocks, bins)
        for k in range(nbin):
            key = f"ld_{c}_{k + 1}"
            raw[key], ctl[key] = float(body[k]), float(cont[k])
            se_body[key] = list(parts[:, k])
        ld_meta[c] = {"subsamples": [[names[c][i] for i in sub] for sub in subs],
                      "pairs_per_subsample_mean": {lab: float(np.mean(v)) for lab, v in nsnp.items()},
                      "pairs_per_bin_body_mean": bc.sum(1).mean(0).tolist(),
                      "pairs_per_bin_control_mean": cc.mean(0).tolist()}
    meta["ld"] = {"seed": LD_SEED, "reps": LD_REPS, "n_individuals": LD_N, "maf": SV.LD_MAF,
                  "edges": SV.LD_EDGES.tolist(), "thinning": "none (all pairs)",
                  "r2": "pairwise-complete genotype r^2 (statvec.ld_block_sums, shared with the simulations)",
                  **ld_meta}



def main() -> None:
    t0 = time.time()
    assert BODY_BP == CONTROL_BP           # the simulated body and control have equal length (tajd scaling)
    xp = _xp()
    print(f"array backend: {xp.__name__}", flush=True)
    raw, ctl, se_body, meta = {}, {}, {}, {"inputs": {}, "regions": {}, "gates": {}}
    meta["inputs"] = {"karyotypes": {c: str(f) for c, f in KARYO.items()},
                      "vcf": {k: str(v) for k, v in VCF.items()}, "accessible_bed": str(ACC_BED),
                      "saf": {c: str(f) for c, f in SAF.items()},
                      "saf40": [str(SAF40 / "AA40.body.saf.idx"), str(SAF40 / "BB40.body.saf.idx")],
                      "realsfs_outputs": str(WORK.resolve())}

    # ---------------------------------------------------------------- called genotypes: ratios, LD, edge
    names = {c: _read_samples(f) for c, f in KARYO.items()}
    vcache = WORK / "vcf_part.json"
    want = _hash(vcf_part_params())
    v = json.loads(vcache.read_text()) if vcache.exists() else {}
    if v.get("param_hash") == want:
        raw.update(v["raw"]), ctl.update(v["ctl"]), se_body.update(v["se_body"]), meta.update(v["meta"])
        print(f"called-genotype part from cache {vcache} (hash {want})", flush=True)
    else:
        if v:
            print(f"cache {vcache} is stale (hash {v.get('param_hash')} != {want}): recomputing", flush=True)
        _vcf_part(names, xp, raw, ctl, se_body, meta, t0)
        vcache.write_text(json.dumps({"raw": raw, "ctl": ctl, "se_body": se_body, "param_hash": want,
                                      "params": vcf_part_params(),
                                      "meta": {k: meta[k] for k in ("regions", "samples", "ld")}}))
    meta["ld"]["param_hash"] = want
    meta["ld_error_model"] = estimate_error_model(names)
    print(f"LD error model: {meta['ld_error_model']}  ({time.time() - t0:.0f}s)", flush=True)
    nblk = len(BLOCK_STARTS)

    # ------------------------------------------------------------------------- ANGSD: sfs, f1, tajd
    a_body = _angsd_stats(_sfs_file("I", "body"), _sfs_file("S", "body"))
    a_ctl = _angsd_stats(_sfs_file("I", "control"), _sfs_file("S", "control"))
    raw.update(a_body)
    for k in ("f1_ratio",) + SV.SFS_KEYS + ("tajd_I", "tajd_S"):
        ctl[k] = a_ctl[k]
    blk1 = {c: np.array([_sfs_file(c, f"b{s}") for s in BLOCK_STARTS]) for c in ("I", "S")}
    tot1 = {c: blk1[c].sum(0) for c in blk1}
    a_sum = _angsd_stats(tot1["I"], tot1["S"])
    parts = [_angsd_stats(tot1["I"] - blk1["I"][b], tot1["S"] - blk1["S"][b]) for b in range(nblk)]
    for k in a_body:
        se_body[k] = [p[k] for p in parts]
    meta["gates"]["sfs_sum_of_blocks_vs_global"] = {k: SV._div(a_sum[k] - a_body[k], a_body[k]) for k in a_body}
    for lab, cls_spec in (("body", ("I", "S")), ("control", ("I", "S"))):
        for c in cls_spec:
            v = _sfs_file(c, lab)
            n = 2 * N_DIP[c]
            meta["regions"][lab][f"angsd_sites_{c}"] = float(v.sum())
            meta["regions"][lab][f"angsd_segregating_{c}"] = float(v[1:n].sum())

    # --------------------------------------------------------------------------------- ANGSD: jsfs
    m_body = _sfs2d_file("body")
    jb = jsfs_from_2d(m_body)
    raw.update(jb)
    blk2 = np.array([_sfs2d_file(f"b{s}") for s in range(BP_LEFT, BP_RIGHT_2D, BLOCK)])
    tot2 = blk2.sum(0)
    js = jsfs_from_2d(tot2)
    meta["gates"]["jsfs_sum_of_blocks_vs_global"] = {k: SV._div(js[k] - jb[k], jb[k]) for k in jb}
    parts2 = [jsfs_from_2d(tot2 - blk2[b]) for b in range(blk2.shape[0])]
    for k in jb:
        se_body[k] = [p[k] for p in parts2]
    meta["regions"]["body"]["angsd2d_sites"] = float(m_body.sum())
    meta["regions"]["body"]["angsd2d_variable"] = float(m_body.sum() - m_body[0, 0] - m_body[-1, -1])
    meta["regions"]["body"]["angsd2d_span"] = [BP_LEFT, BP_RIGHT_2D]
    meta["jsfs"] = {"n_chrom_per_class_native": 80, "proj": SV.PROJ, "bins": [list(b) for b in SV.JSFS_BINS]}

    # ------------------------------------------------------------------------------------ assemble
    stats = {}
    se = {}
    for k in SV.STAT_KEYS:
        if k in DIVIDED:
            stats[k] = SV._div(raw[k], ctl[k])
            se[k] = jackknife_se([SV._div(v, ctl[k]) for v in se_body[k]])
        else:
            stats[k] = float(raw[k])
            se[k] = jackknife_se(se_body[k])
    meta["control"] = {k: ctl[k] for k in SV.STAT_KEYS if k in DIVIDED}
    meta["regions"]["body"]["span"] = [BP_LEFT, BP_RIGHT]
    meta["regions"]["control"]["span"] = [CTL_LO, CTL_HI]
    meta["jackknife"] = {"block_bp": BLOCK, "n_blocks": nblk, "n_blocks_2d": int(blk2.shape[0]),
                         "control": "held fixed", "estimator": "delete-one, SE^2 = (B-1)/B sum (theta_-b - mean)^2"}
    meta["tajd"] = {"n_to": N_SIM, "length_bp": BODY_BP}
    meta["polarization"] = "I = 2a = AA cluster (inverted), S = 2+ = BB cluster (standard)"
    meta["runtime_s"] = round(time.time() - t0, 1)
    out = {"stats": stats, "se": se, "raw": {k: float(raw[k]) for k in SV.STAT_KEYS}, "p_obs": P_NOW,
           "meta": meta}
    out["boundary"] = boundary_block(names)
    print(f"boundary block done ({time.time() - t0:.0f}s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    for k in SV.STAT_KEYS:
        print(f"  {k:<12s} stat {stats[k]:10.5f}  se {se[k]:9.5f}  raw {raw[k]:10.5f}  "
              f"control {ctl.get(k, float('nan')):10.5f}")
    bnd = out["boundary"]
    for k in SV.BND_KEYS:
        b = int(k[-1])
        v = [bnd["stats"][k], bnd["se"][k], bnd["per_side"]["left"][k], bnd["per_side"]["right"][k]]
        v = [float("nan") if x is None else x for x in v]
        print(f"  {k:<16s} pooled {v[0]:8.4f} se {v[1]:7.4f}  left {v[2]:8.4f}  right {v[3]:8.4f}  "
              f"n_seg {bnd['counts']['pooled']['n_seg'][b]}")


if __name__ == "__main__":
    main()
