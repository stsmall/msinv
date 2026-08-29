#!/usr/bin/env python
"""Which SNPs are most differentiated between arrangements, and are they coding?

    .venv/bin/python -m illex.scripts.fst_snp_coding

THE QUESTION AND ITS BIG CAVEAT
-------------------------------
"Which high-FST SNPs separate AA from BB, what genes are they in, and are they
nonsynonymous?" is the natural question, but inside an inversion it is NOT the
question it usually is. Recombination between arrangements is suppressed across
the whole 20 Mb, so **every** site in the block is linked to the arrangement and
high differentiation is expected everywhere -- region-wide FST is 0.365. An
individual high-FST SNP inside an inversion is not an independent candidate
causal variant; it is a passenger on a haplotype block.

So the only version of the question with a testable answer is the COMPOSITIONAL
one: among coding sites, are the most-differentiated SNPs enriched for
**0-fold degenerate** (every change nonsynonymous) relative to **4-fold**
(every change synonymous), compared with what the region's own coding SNPs
give? That asks whether arrangement divergence is disproportionately
protein-altering, which is what "functional divergence" would mean here.

THE CONFOUND THAT MUST BE CONTROLLED
------------------------------------
The inverted class has a smaller effective size (~2Np, plus a single-origin
cap), so purifying selection is weaker in it and slightly deleterious
nonsynonymous variants drift up more easily. That produces an excess of
0-fold differentiation with no adaptive story at all. The collinear control --
where AA and BB are exchangeable and no barrier exists -- is carried through
for exactly this reason, as it was for f1 (NOTES sec 8.19).

THE RESULT, UP FRONT: THIS ROUTE FAILS ITS CONTROL
---------------------------------------------------
It does not work, and the control is what proves it. In the collinear region
AA and BB are exchangeable, so there should be essentially no differentiated
sites -- yet 0.25% of control SNPs show dp >= 0.8 and 267 read as FIXED
differences. There is no biological process that does that. The apparent
"fixed differences" are all low-call-rate sites: NONE of them, in either
region, survives a filter of AN >= 75% of maximum. Structured missingness
(documented for this callset) makes a site look fixed when few genotypes are
called.

This is the FOURTH time the same lesson has appeared here -- called-genotype
SFS (sec 8.4), argentinus sufficiency (sec 8.12), the polarization subset, and
now per-site FST. **Do not read per-site allele-frequency differences off this
callset.** The instrument that works is ANGSD genotype likelihoods, and for
this specific question the unfolded AA x BB 2D SFS is the right object: its
corner cells ARE the fixed-difference count, with no calling step to ascertain
against.

What survives from this script: the differentiated tail is overwhelmingly
NON-CODING (43 coding sites out of 11,381 at dp >= 0.8 in the body, 0.4%), and
no 0-fold enrichment is detectable in it (OR 0.89-1.30, p >= 0.57). Both are
consistent with a gene-poor block, and neither depends on the sites being
individually trustworthy.

DATA
----
* per-site AA/BB allele counts from the callset (`.tmp/fstsnp/*.tsv`)
* per-site codon degeneracy from degenotate (`degeneracy-all-sites.bed`,
  column 5; 0/2/3/4-fold), extracted to `.tmp/fstsnp/chr2_degen.tsv`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

T = Path(".tmp/fstsnp")
OUT = Path("results/illex")
GENES = Path("/sietch_colab/data_share/illex/popgen_data/degenotate_illex/"
             "Illex.genes.bed")


MAX_AN = {"AA": 508, "BB": 190}   # 254 and 95 diploids
MIN_AN_FRAC = 0.5                 # see the control-failure note above


def load(region: str, min_an_frac: float = MIN_AN_FRAC) -> pd.DataFrame:
    d = {}
    for pop in ("AA", "BB"):
        x = pd.read_csv(T / f"{region}.{pop}.tsv", sep="\t", header=None,
                        names=["pos", "ref", "alt", "ac", "an"])
        x = x[x.an > 0]
        d[pop] = x.assign(**{f"p_{pop}": x.ac / x.an,
                             f"n_{pop}": x.an})[["pos", f"p_{pop}", f"n_{pop}"]]
    m = d["AA"].merge(d["BB"], on="pos")
    if min_an_frac:
        m = m[(m.n_AA >= min_an_frac * MAX_AN["AA"])
              & (m.n_BB >= min_an_frac * MAX_AN["BB"])]
    m["dp"] = (m.p_AA - m.p_BB).abs()
    # Hudson FST per site, as a ratio of the same numerator/denominator used
    # region-wide -- kept per-site only for ranking, not for interpretation.
    num = ((m.p_AA - m.p_BB) ** 2
           - m.p_AA * (1 - m.p_AA) / (m.n_AA - 1)
           - m.p_BB * (1 - m.p_BB) / (m.n_BB - 1))
    den = m.p_AA * (1 - m.p_BB) + m.p_BB * (1 - m.p_AA)
    m["fst"] = np.where(den > 0, num / den, np.nan)
    return m


def main() -> None:
    deg = pd.read_csv(T / "chr2_degen.tsv", sep="\t", header=None,
                      names=["pos", "degen", "tx"])
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Most-differentiated SNPs between arrangements, and their codon class")
    emit("I = BB (inverted), S = AA. dp = |p_AA - p_BB|.")
    emit()
    emit("READ THE CAVEAT FIRST: recombination between arrangements is")
    emit("suppressed across the whole block, so high dp is expected EVERYWHERE")
    emit("inside the inversion. Individual high-dp SNPs are passengers, not")
    emit("candidate causal variants. Only the 0-fold vs 4-fold COMPOSITION of")
    emit("the differentiated tail is interpretable, and only against the")
    emit("collinear control.")
    emit()

    res = {}
    for region in ("body", "control"):
        m = load(region)
        j = m.merge(deg, on="pos", how="inner")
        res[region] = (m, j)
        emit("=" * 76)
        emit(f"{region.upper()}   {len(m):,} biallelic SNPs, "
             f"{len(j):,} of them in CDS with a degeneracy call")
        emit("=" * 76)
        emit(f"  {'dp class':>14s} {'n SNPs':>10s} {'n coding':>9s} "
             f"{'0-fold':>8s} {'4-fold':>8s} {'0f/(0f+4f)':>11s}")
        for lab, lo, hi in (("all", 0.0, 1.01), ("dp >= 0.5", 0.5, 1.01),
                            ("dp >= 0.8", 0.8, 1.01), ("dp >= 0.9", 0.9, 1.01),
                            ("dp = 1 (fixed)", 0.999, 1.01)):
            sub = m[(m.dp >= lo) & (m.dp < hi)]
            sj = j[(j.dp >= lo) & (j.dp < hi)]
            n0 = int((sj.degen == 0).sum())
            n4 = int((sj.degen == 4).sum())
            frac = f"{n0 / (n0 + n4):11.4f}" if (n0 + n4) else f"{'--':>11s}"
            emit(f"  {lab:>14s} {len(sub):10,} {len(sj):9,} {n0:8,} {n4:8,} "
                 f"{frac}")
        emit()

    # ---- the test ------------------------------------------------------
    emit("=" * 76)
    emit("IS THE DIFFERENTIATED TAIL ENRICHED FOR PROTEIN-ALTERING SITES?")
    emit("=" * 76)
    emit("  Fisher exact, 0-fold vs 4-fold, differentiated tail vs all coding")
    emit("  SNPs in the SAME region (so the region's own codon composition and")
    emit("  gene content are the null, not the genome's).")
    emit()
    from scipy.stats import fisher_exact
    for region in ("body", "control"):
        _, j = res[region]
        emit(f"  {region.upper()}")
        base0 = int((j.degen == 0).sum())
        base4 = int((j.degen == 4).sum())
        for lab, thr in (("dp >= 0.8", 0.8), ("dp >= 0.9", 0.9),
                         ("dp = 1", 0.999)):
            sj = j[j.dp >= thr]
            n0, n4 = int((sj.degen == 0).sum()), int((sj.degen == 4).sum())
            if n0 + n4 < 10:
                emit(f"    {lab:<10s} only {n0 + n4} coding sites -- too few")
                continue
            odds, p = fisher_exact([[n0, n4],
                                    [base0 - n0, base4 - n4]])
            emit(f"    {lab:<10s} {n0:5,} 0-fold / {n4:4,} 4-fold  "
                 f"OR {odds:5.2f}  p = {p:.3g}")
        emit()

    # ---- where are the fixed differences ------------------------------
    mb, jb = res["body"]
    fixed = mb[mb.dp >= 0.999]
    emit("=" * 76)
    emit("FIXED DIFFERENCES IN THE BODY")
    emit("=" * 76)
    emit(f"  {len(fixed):,} sites with dp = 1 out of {len(mb):,} "
         f"({100 * len(fixed) / len(mb):.3f}%)")
    if len(fixed):
        g = pd.read_csv(GENES, sep="\t", header=None,
                        names=["chrom", "start", "stop"])
        g = g[g.chrom.astype(str) == "2"]
        pos = fixed.pos.to_numpy()
        ing = np.zeros(len(pos), dtype=bool)
        for s0, s1 in zip(g.start, g.stop):
            ing |= (pos > s0) & (pos <= s1)
        emit(f"  in an annotated gene: {int(ing.sum()):,} "
             f"({100 * ing.mean():.1f}%)")
        fj = jb[jb.dp >= 0.999]
        emit(f"  in CDS with a degeneracy call: {len(fj):,}")
        if len(fj):
            emit(f"    0-fold {int((fj.degen == 0).sum())}  "
                 f"2-fold {int((fj.degen == 2).sum())}  "
                 f"3-fold {int((fj.degen == 3).sum())}  "
                 f"4-fold {int((fj.degen == 4).sum())}")
            emit()
            emit("  transcripts carrying fixed CDS differences "
                 "(top 15 by count):")
            tx = (fj.tx.str.split(":").str[0].value_counts().head(15))
            for name, cnt in tx.items():
                sub = fj[fj.tx.str.startswith(name)]
                emit(f"    {name:<28s} {cnt:4d} fixed  "
                     f"({int((sub.degen == 0).sum())} 0-fold, "
                     f"{int((sub.degen == 4).sum())} 4-fold)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fst_snp_coding.txt").write_text("\n".join(lines) + "\n")
    mb[mb.dp >= 0.9].to_csv(OUT / "fst_snp_body_top.csv", index=False)
    print(f"\nwrote {OUT}/fst_snp_coding.txt and fst_snp_body_top.csv")


if __name__ == "__main__":
    main()
