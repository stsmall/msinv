#!/usr/bin/env python
"""A mutation-rate-free scale for the inversion's age.

    CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
        illex/scripts/mu_free_ratio.py

WHY
---
The age of the chr2 inversion is ~735,000 generations with a +-19,000
statistical error (NOTES sec 7.5.4), but it scales as 1/mu, and mu = 3e-9 is
the weakest external input in the whole chain (NOTES sec 2). A +-30%
uncertainty on mu is +-220,000 generations -- an order of magnitude larger than
everything the simulations control. **No amount of extra simulation can improve
the age.** The only routes out are a better mu, or expressing the age against a
different clock.

This script does the second. Both of these are 2*mu*T for some T:

    dxy(AA, BB)                 between arrangements, within illecebrosus
    div(illecebrosus, coindetii) between species

so their RATIO is free of mu, free of the accessibility mask, and free of the
generation time:

    R = dxy(AA,BB) / div(ill,coin)
      = (t_inv + T_anc_ill) / (T_split + T_anc_spp)

If the illecebrosus-coindetii split ever acquires an independent calibration --
a fossil, a biogeographic vicariance date, a published cephalopod substitution
rate -- the inversion's age follows from R without mu entering anywhere. R is
the durable quantity; the age in years is not.

THE THING THAT MAKES THIS EASY TO GET WRONG
-------------------------------------------
The two divergences must be counted over **the same base pairs**, or the ratio
silently measures the difference between two denominators instead of two times.
A first pass at this divided coindetii substitutions by the nominal 19,954,980
bp of the region and got R = 0.227. That was wrong: AnchorWave aligns only
90.66% of the region, so the denominator was ~10% too large.

Here both quantities are restricted to the intersection

    inversion body  AND  illecebrosus-accessible  AND  coindetii-callable

and then the shared denominator CANCELS: the ratio is computed as a ratio of
counts (summed expected pairwise differences over summed substitutions), so no
per-base rate is ever formed and no accessibility fraction can leak in.

Sources:
  illecebrosus accessible   03_karyotype/chr2_mask/chr2.mask.3state.bed
  coindetii callable        anchorwave_dir/ill_coin_alignment/coindetti_vcf/
                            2.callable.bed   (illecebrosus coordinates)
  coindetii substitutions   same dir, 2.snps.vcf.gz

**`2.callable.bed` does NOT mean "aligned".** It means aligned AND IDENTICAL to
the reference: it is perfectly disjoint from 2.snps.vcf.gz (0 of 216,739 SNP
positions fall inside it), which is the layout est-sfs wants -- matching sites in
the BED, differing sites in the VCF. Taking it as the aligned span silently
excludes every substitution and drives the numerator to zero. The comparable
span is therefore

    coindetii comparable = 2.callable.bed  UNION  2.snps.vcf.gz positions

and the code asserts the disjointness so this cannot be reintroduced quietly.
Indels are deliberately left out of the denominator: they are aligned but are
not opportunities for a substitution.

BIASES, BOTH IN THE SAME DIRECTION
----------------------------------
1. The illecebrosus callset is quality/MAF-filtered, so some real low-frequency
   variants are missing and dxy runs slightly low. AnchorWave fixed differences
   are all retained. This biases R DOWN.
2. Multiple hits. At ~1.2% divergence a Jukes-Cantor correction raises
   div(ill,coin) by ~1%, which also biases the uncorrected R DOWN. Reported
   both ways.

So R should be read as a slight underestimate, i.e. the inversion is if
anything a slightly larger fraction of the split time than stated.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
from pg_gpu import HaplotypeMatrix

T = Path(".tmp/illex_chr2")
OUT = Path("results/illex")
INV_START, INV_STOP = 60_040_617, 79_995_597
# The base axis and window grid deliberately match illex/scripts/
# empirical_jackknife.py (60-80 Mb on a 100 kb grid) so the Fst-defined body
# windows from that run can be reused here. Using the nominal breakpoints as the
# axis origin instead silently misaligns the grids by 40,617 bp, which makes the
# body arm select zero windows and skip without complaint.
AXIS_START, AXIS_STOP = 60_000_000, 80_000_000
REGION = f"2:{AXIS_START}-{AXIS_STOP}"

MASK_3STATE = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
               "03_karyotype/chr2_mask/chr2.mask.3state.bed")
AW = ("/sietch_colab/data_share/illex/alignments/anchorwave_dir/"
      "ill_coin_alignment/coindetti_vcf")
COIN_CALLABLE = f"{AW}/2.callable.bed"
COIN_SNPS = f"{AW}/2.snps.vcf.gz"
BCFTOOLS = "/home/ssmall/bin/bcftools"

BASE_WINDOW = 100_000
BLOCK_SIZES = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
FST_DIFF_CUTOFF = 0.15

# Model quantities needed only to convert R into a statement about t_inv.
# R itself does not depend on them.
T_INV_FIT = 737_003.0
TWO_N_ANC = 2 * 547_928.0


def _intervals_to_bool(path: str, chrom: str, keep=None) -> np.ndarray:
    """Boolean over [INV_START, INV_STOP) from a BED, via awk prefilter."""
    n = AXIS_STOP - AXIS_START
    out = np.zeros(n, dtype=bool)
    cond = f'$1=="{chrom}" && $3>{AXIS_START} && $2<{AXIS_STOP}'
    if keep:
        cond += f' && $4 ~ /{keep}/'
    awk = f'BEGIN{{OFS="\\t"}} {cond} {{print $2, $3}}'
    p = subprocess.run(["awk", awk, path], capture_output=True, text=True,
                       check=True)
    for line in p.stdout.splitlines():
        a, b = line.split()
        lo = max(int(a), AXIS_START) - AXIS_START
        hi = min(int(b), AXIS_STOP) - AXIS_START
        if hi > lo:
            out[lo:hi] = True
    return out


def build_shared(coin_snp_idx: np.ndarray) -> np.ndarray:
    """illecebrosus-accessible AND coindetii-comparable, over the body.

    ``coin_snp_idx`` are 0-based indices of coindetii substitutions; they are
    part of the comparable span (see module docstring) and must be added to
    2.callable.bed rather than assumed to be inside it.
    """
    print("building the shared base set ...", flush=True)
    n = AXIS_STOP - AXIS_START
    acc = _intervals_to_bool(MASK_3STATE, "2", keep="^accessible")
    same = _intervals_to_bool(COIN_CALLABLE, "2")

    # Guard the semantics discovered on 2026-08-22: callable.bed is
    # aligned-AND-IDENTICAL, disjoint from the SNPs.
    overlap = int(same[coin_snp_idx].sum())
    if overlap > 0.01 * len(coin_snp_idx):
        raise AssertionError(
            f"{overlap:,} of {len(coin_snp_idx):,} coindetii SNPs fall inside "
            "2.callable.bed. That contradicts the documented layout "
            "(callable = aligned AND identical, disjoint from the SNP VCF). "
            "Re-derive the comparable span before trusting any ratio.")

    comparable = same.copy()
    comparable[coin_snp_idx] = True
    shared = acc & comparable
    print(f"  inversion body                 {n:12,d} bp")
    print(f"  illecebrosus accessible        {acc.sum():12,d} bp  "
          f"({100 * acc.sum() / n:.2f}%)")
    print(f"  coindetii aligned+identical    {same.sum():12,d} bp  "
          f"({100 * same.sum() / n:.2f}%)")
    print(f"  coindetii substitutions        {len(coin_snp_idx):12,d} bp  "
          f"(disjoint from the above: {overlap} overlap)")
    print(f"  coindetii COMPARABLE           {comparable.sum():12,d} bp  "
          f"({100 * comparable.sum() / n:.2f}%)")
    print(f"  SHARED (accessible & comparable) {shared.sum():10,d} bp  "
          f"({100 * shared.sum() / n:.2f}%)")
    return shared


def dxy_per_site() -> tuple[np.ndarray, np.ndarray]:
    """(positions, expected AA-BB differences per site) over the region."""
    print("\nloading illecebrosus genotypes ...", flush=True)
    h = HaplotypeMatrix.from_vcf(str(T / "inv.vcf.gz"), region=REGION)
    h.load_pop_file(str(T / "pops.tsv"))
    H = np.asarray(h.haplotypes)
    pos = np.asarray(h.positions)
    print(f"  {H.shape[1]:,} variants x {H.shape[0]:,} haplotypes")

    freqs, valid = [], []
    for pop in ("AA", "BB"):
        idx = np.asarray(h.sample_sets[pop])
        sub = H[idx]
        ok = sub >= 0
        n_ok = ok.sum(axis=0)
        alt = np.where(ok, sub, 0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freqs.append(np.where(n_ok > 0, alt / np.maximum(n_ok, 1), np.nan))
        valid.append(n_ok > 0)
    pA, pB = freqs
    keep = valid[0] & valid[1]
    # Expected differences between one random AA and one random BB haplotype.
    dxy = pA * (1.0 - pB) + (1.0 - pA) * pB
    print(f"  {keep.sum():,} sites with data in both arrangements")
    return pos[keep], dxy[keep]


def coin_positions() -> np.ndarray:
    p = subprocess.run(
        [BCFTOOLS, "query", "-r", REGION, "-f", "%POS\n", COIN_SNPS],
        capture_output=True, text=True, check=True)
    return np.fromiter((int(x) for x in p.stdout.split()), dtype=np.int64)


def jackknife(win: pd.DataFrame, block_bp: int):
    """Delete-one-block jackknife on R = sum(dxy) / sum(coin)."""
    blk = (win.window_start // block_bp).to_numpy()
    ids = np.unique(blk)
    b = len(ids)
    full = win.sum_dxy.sum() / win.n_coin.sum()
    partial = np.array([
        win.sum_dxy[blk != i].sum() / win.n_coin[blk != i].sum() for i in ids])
    m = partial.mean()
    est = b * full - (b - 1) * m
    se = float(np.sqrt((b - 1) / b * ((partial - m) ** 2).sum()))
    return b, float(full), float(est), se


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_shared_axis = AXIS_STOP - AXIS_START

    def to_index(p1):
        """1-based VCF position -> 0-based index into the BED-derived array.

        The BEDs are 0-based half-open and the VCF is 1-based, so 1-based P
        occupies BED interval [P-1, P). Positions falling outside the array
        (the single base at each end of the inclusive VCF region) are dropped.
        """
        i = p1 - 1 - AXIS_START
        ok = (i >= 0) & (i < n_shared_axis)
        return i, ok

    cpos_all = coin_positions()
    cidx_all, cok_all = to_index(cpos_all)
    shared = build_shared(cidx_all[cok_all])

    pos, dxy = dxy_per_site()
    idx, ok = to_index(pos)
    keep = ok.copy()
    keep[ok] &= shared[idx[ok]]
    pos, dxy = pos[keep], dxy[keep]
    print(f"  {len(pos):,} retained in the shared base set")

    ckeep = cok_all.copy()
    ckeep[cok_all] &= shared[cidx_all[cok_all]]
    cpos = cpos_all[ckeep]
    print(f"\ncoindetii substitutions in the shared base set: {len(cpos):,}")

    edges = np.arange(AXIS_START, AXIS_STOP + BASE_WINDOW, BASE_WINDOW)
    win = pd.DataFrame({
        "window_start": edges[:-1],
        "window_stop": edges[1:],
        "shared_bp": np.add.reduceat(
            shared, (edges[:-1] - AXIS_START)).astype(np.int64),
        "sum_dxy": np.histogram(pos, bins=edges, weights=dxy)[0],
        "n_dxy_sites": np.histogram(pos, bins=edges)[0],
        "n_coin": np.histogram(cpos, bins=edges)[0].astype(np.int64),
    })
    win = win[(win.shared_bp > 0) & (win.n_coin > 0)].reset_index(drop=True)

    # Differentiated body, from the Fst already computed by the jackknife run.
    body_ok = None
    jk = OUT / "empirical_jackknife_windows.csv"
    if jk.exists():
        w2 = pd.read_csv(jk)
        good = set(w2.loc[w2.fst > FST_DIFF_CUTOFF, "window_start"])
        body_ok = win.window_start.isin(good)

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    arms = [("nominal", "NOMINAL SPAN 60.0-80.0 Mb", win)]
    if body_ok is not None:
        arms.append(("body",
                     f"DIFFERENTIATED BODY (Fst > {FST_DIFF_CUTOFF})",
                     win[body_ok.to_numpy()]))
    else:
        emit("NOTE: results/illex/empirical_jackknife_windows.csv absent, so "
             "the\ndifferentiated-body arm is skipped. Run "
             "illex/scripts/empirical_jackknife.py first.")

    collected = {}
    for key, label, sub in arms:
        if len(sub) < 6:
            emit(f"\nWARNING: arm {key!r} selected only {len(sub)} windows -- "
                 "skipped. If this is the\nbody arm, the window grids have "
                 "misaligned; they must share an origin.")
            continue
        emit(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
        sbp = int(sub.shared_bp.sum())
        d_rate = sub.sum_dxy.sum() / sbp
        c_rate = sub.n_coin.sum() / sbp
        emit(f"windows {len(sub)}   shared bp {sbp:,}")
        emit(f"  dxy(AA,BB)   per shared bp = {d_rate:.6f}   "
             f"({sub.sum_dxy.sum():,.1f} expected differences)")
        emit(f"  div(ill,coin) per shared bp = {c_rate:.6f}   "
             f"({int(sub.n_coin.sum()):,} substitutions)")
        jc = -0.75 * np.log(1.0 - 4.0 / 3.0 * c_rate)
        emit(f"  div, Jukes-Cantor corrected  = {jc:.6f}  "
             f"(+{100 * (jc / c_rate - 1):.2f}%)")
        emit("")
        emit(f"  {'block':>9s} {'n':>5s} {'R = dxy/div':>22s} {'R (JC)':>9s}")
        rows = {}
        for bp in BLOCK_SIZES:
            if len(sub) * BASE_WINDOW < 3 * bp:
                continue
            b, full, est, se = jackknife(sub, bp)
            rows[bp] = {"n_blocks": b, "R": est, "R_se": se,
                        "R_jc": est * c_rate / jc}
            emit(f"  {bp / 1e6:8.2f}M {b:5d} {est:12.4f} +- {se:.4f} "
                 f"({100 * se / est:4.1f}%) {est * c_rate / jc:9.4f}")
        best = 1_000_000 if 1_000_000 in rows else max(rows)
        r, rse = rows[best]["R"], rows[best]["R_se"]
        emit("")
        emit(f"  R = {r:.4f} +- {rse:.4f}  ({best / 1e6:.0f} Mb blocks) "
             "-- mu-free, mask-free")
        emit(f"  i.e. the arrangements' divergence is "
             f"{100 * r:.1f}% +- {100 * rse:.1f}% of the")
        emit("  illecebrosus-coindetii divergence.")
        emit("")
        emit("  Converting to a split time REQUIRES the model's ancestral "
             "coalescent depth,")
        emit("  so this step is no longer assumption-free:")
        emit(f"    t_inv + 2N_ANC = {T_INV_FIT + TWO_N_ANC:,.0f} generations "
             "(fitted, NOTES 7.5)")
        emit(f"    => T_split + T_anc_spp = "
             f"{(T_INV_FIT + TWO_N_ANC) / r:,.0f} generations")
        emit(f"    => T_split ~ {(T_INV_FIT + TWO_N_ANC) / r - TWO_N_ANC:,.0f} "
             "generations if the ancestral Ne matched N_ANC")
        emit("")
        emit("  READ IN THE USEFUL DIRECTION: given an independent calibration "
             "T_cal for the")
        emit("  illecebrosus-coindetii split, the inversion's age follows "
             "WITHOUT mu:")
        emit(f"    t_inv = {r:.4f} * (T_cal + T_anc_spp) - {TWO_N_ANC:,.0f}")
        collected[key] = rows

    win.to_csv(OUT / "mu_free_ratio_windows.csv", index=False)
    (OUT / "mu_free_ratio.txt").write_text("\n".join(lines) + "\n")
    payload = {
        "definition": "R = dxy(AA,BB) / div(illecebrosus, coindetii), "
                      "counted over illecebrosus-accessible AND "
                      "coindetii-callable bases inside the inversion body",
        "mu_free": True,
        "shared_bp": int(win.shared_bp.sum()),
        **{k: {str(bp): v for bp, v in rows.items()}
           for k, rows in collected.items()},
    }
    (OUT / "mu_free_ratio.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}/mu_free_ratio.{{txt,json}} and "
          f"{OUT}/mu_free_ratio_windows.csv")


if __name__ == "__main__":
    main()
