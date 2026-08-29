#!/usr/bin/env python
"""Per-site GL FST x codon degeneracy: which genes, which codon positions?

    .venv/bin/python -m illex.scripts.fst_persite_degen

WHY THIS AND NOT THE CALLED VCF
-------------------------------
`fst_snp_coding.py` asked the same question off the callset and FAILED ITS
CONTROL (NOTES sec 8.23): the collinear region -- where AA and BB are
exchangeable -- returned 267 apparent fixed differences and 0.25% of sites at
|dp| >= 0.8, and none of the apparent fixed sites in either region survived
AN >= 75% of maximum. Structured missingness makes a site look differentiated
precisely where few genotypes are called.

This is the genotype-likelihood redo, off the same n=40 SAFs and unfolded 2D
SFS that gave sec 8.24. `realSFS fst print` emits per-site (A, B) with
FST = A/B and the region estimate = sum(A)/sum(B).

TWO QUESTIONS, TWO RESOLUTIONS -- ON PURPOSE
-------------------------------------------
* **Which genes?** Answered PER TRANSCRIPT as a ratio of sums, sum(A)/sum(B)
  over that transcript's coding sites. Per-site FST at 80+80 chromosomes has
  enormous variance; a per-transcript ratio of sums is the same estimator used
  region-wide and is far more stable. This is the answer to "what genes".
* **Which codon positions?** Answered COMPOSITIONALLY, 0-fold vs 4-fold in the
  high-FST tail against the same region's own coding sites as the null. Never
  by naming individual top sites -- see the caveat.

THE RESULT: BOTH ANSWERS ARE ARTIFACTS OF HUDSON'S DENOMINATOR
--------------------------------------------------------------
**Do not read the gene ranking or the codon composition below as biology.**

Hudson FST is sum(A)/sum(B). At a near-monomorphic site both go to ~0, but the
numerator carries the negative sampling corrections -p(1-p)/(n-1), so the ratio
is dragged toward 0 wherever diversity is low. CDS is mostly constrained, so
this bites hard:

  Spearman corr(per-transcript FST, polymorphism per site) = **+0.69**
  quartile of B/site:   Q1 (least polymorphic) median FST 0.020
                        Q4 (most polymorphic)  median FST 0.294

The contradiction that exposed it: transcripts reading FST ~ 0.012 sit inside
100 kb windows at FST 0.37-0.46, which cannot happen in a non-recombining
block, and corr(transcript FST, window FST) is only +0.20.

So the transcript ranking orders genes by HOW POLYMORPHIC THEY ARE, not how
differentiated. The apparent 0-fold DEPLETION in the high-FST tail
(OR 0.18-0.41) is the same artifact seen from the other side: 0-fold sites are
constrained, hence less polymorphic, hence pushed out of the high-FST tail --
and the CONTROL shows the same direction (OR 0.44-0.81), confirming it is
generic rather than inversion-specific.

Conditioning on polymorphism (B >= 0.25) removes the bias but leaves only 220
coding sites in the body and 5 in the control: no usable test, and zero
transcripts with enough sites. The region is gene-poor (95 transcripts), CDS is
mostly constrained, and n = 40/class is a small subsample -- there is simply not
enough well-powered coding variation to answer either question.

WHAT WOULD ACTUALLY WORK: per-transcript **dxy** rather than FST. dxy is an
absolute divergence, not a ratio, so it has no small-denominator pathology. That
is the recommended follow-on and it needs no new ANGSD run.

THE CAVEAT THAT INVALIDATES SITE-LEVEL CLAIMS
---------------------------------------------
Recombination between arrangements is suppressed across the whole 20 Mb, so
EVERY site in the block is linked to the karyotype and high FST is expected
everywhere. An individual high-FST SNP inside an inversion is a passenger on a
haplotype block, NOT a candidate causal variant. Only composition, and only
relative to the control, carries information.

Second caveat: the reference genome is a BB haplotype inside the inversion
(sec 8.24), so ALT-vs-REF asymmetries lean toward AA. FST is polarization-
invariant, so it is much less exposed than the 2D SFS corner counts -- but
mapping bias against AA reads is not removed by that.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

SFS = Path(".tmp/sfs2d")
DEGEN = Path(".tmp/fstsnp/chr2_degen.tsv")
ENTAP = Path("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/"
             "project/functional_entap/entap_outfiles/final_results/annotated.tsv")
OUT = Path("results/illex")
MIN_SITES_PER_TX = 50      # transcripts with fewer coding sites are too noisy


def persite(lab: str) -> pd.DataFrame:
    d = pd.read_csv(SFS / f"AA_BB.{lab}.persite.gz", sep="\t", header=None,
                    names=["chrom", "pos", "A", "B"],
                    usecols=["pos", "A", "B"],
                    dtype={"pos": "int64", "A": "float64", "B": "float64"})
    return d[d.B > 0]


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    deg = pd.read_csv(DEGEN, sep="\t", header=None,
                      names=["pos", "degen", "tx"])
    deg["transcript"] = deg.tx.str.split(":").str[0]

    emit("Per-site GL FST x codon degeneracy (AA vs BB, n=40/class)")
    emit("The called-VCF version of this failed its control (sec 8.23); this is")
    emit("the genotype-likelihood redo.")
    emit()

    store = {}
    for lab in ("body", "control"):
        d = persite(lab)
        j = d.merge(deg, on="pos", how="inner")
        store[lab] = (d, j)
        reg = d.A.sum() / d.B.sum()
        emit("=" * 76)
        emit(f"{lab.upper()}  {len(d):,} sites with FST defined, "
             f"{len(j):,} in CDS   region FST = {reg:.4f}")
        emit("=" * 76)
        d["fst"] = d.A / d.B
        j["fst"] = j.A / j.B
        for q in (0.5, 0.9, 0.99, 0.999):
            emit(f"    per-site FST quantile {q:<6.3f} = {d.fst.quantile(q):8.4f}")
        emit()

    # ---- codon composition in the high-FST tail ------------------------
    emit("=" * 76)
    emit("CODON COMPOSITION OF THE HIGH-FST TAIL")
    emit("=" * 76)
    emit("  0-fold = every change nonsynonymous; 4-fold = every change")
    emit("  synonymous. Null = the SAME region's own coding sites, so gene")
    emit("  content and codon usage are controlled for.")
    emit()
    for lab in ("body", "control"):
        d, j = store[lab]
        j = j.assign(fst=j.A / j.B)
        b0 = int((j.degen == 0).sum())
        b4 = int((j.degen == 4).sum())
        emit(f"  {lab.upper()}   all coding: {b0:,} 0-fold / {b4:,} 4-fold "
             f"(0f frac {b0 / (b0 + b4):.3f})")
        for q in (0.90, 0.99, 0.999):
            thr = d.assign(fst=d.A / d.B).fst.quantile(q)
            s = j[j.fst >= thr]
            n0, n4 = int((s.degen == 0).sum()), int((s.degen == 4).sum())
            if n0 + n4 < 10:
                emit(f"    top {1 - q:<7.3%} (FST>={thr:6.3f}): only {n0 + n4} "
                     "coding sites -- no test")
                continue
            odds, p = fisher_exact([[n0, n4], [b0 - n0, b4 - n4]])
            emit(f"    top {1 - q:<7.3%} (FST>={thr:6.3f}): {n0:5,} 0-fold / "
                 f"{n4:4,} 4-fold  frac {n0 / (n0 + n4):.3f}  OR {odds:5.2f}  "
                 f"p = {p:.3g}")
        emit()

    # ---- which transcripts --------------------------------------------
    emit("=" * 76)
    emit("WHICH GENES -- per-transcript FST  **WITHDRAWN, SEE HEADER**")
    emit("=" * 76)
    d, j = store["body"]
    g = (j.groupby("transcript")
           .agg(A=("A", "sum"), B=("B", "sum"), n=("pos", "size"),
                n0=("degen", lambda x: int((x == 0).sum())),
                n4=("degen", lambda x: int((x == 4).sum())))
           .reset_index())
    g["fst"] = g.A / g.B
    g = g[g.n >= MIN_SITES_PER_TX].sort_values("fst", ascending=False)
    reg = d.A.sum() / d.B.sum()
    emit(f"  region-wide body FST = {reg:.4f}; "
         f"{len(g)} transcripts with >= {MIN_SITES_PER_TX} coding sites")
    emit()
    ann = {}
    if ENTAP.exists():
        a = pd.read_csv(ENTAP, sep="\t", usecols=[0, 12, 13], low_memory=False)
        a.columns = ["q", "desc", "species"]
        ann = dict(zip(a.q, a.desc.fillna("")))
    emit(f"  {'transcript':<24s} {'nCDS':>5s} {'FST':>7s} {'0f':>5s} {'4f':>5s}  "
         "description")
    for _, r in g.head(20).iterrows():
        desc = str(ann.get(r.transcript, ""))[:44] or "(no EnTAP hit)"
        emit(f"  {r.transcript:<24s} {r.n:5d} {r.fst:7.4f} {r.n0:5d} {r.n4:5d}  "
             f"{desc}")
    emit()
    emit("  ** THE TABLE ABOVE IS NOT A RESULT. ** Spearman corr(FST, polymorphism)")
    emit("  = +0.69 across these transcripts; it ranks them by diversity, not by")
    emit("  differentiation. Reproduced only so the artifact is on the record.")
    emit()
    emit("  Compare against the region-wide value, not against zero: inside a")
    emit("  non-recombining block every transcript is differentiated, so a")
    emit("  transcript near the region FST is UNREMARKABLE. Only a transcript")
    emit("  well above the block's own level would mean anything, and even then")
    emit("  linkage means it need not be the target.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fst_persite_degen.txt").write_text("\n".join(lines) + "\n")
    g.to_csv(OUT / "fst_per_transcript_body.csv", index=False)
    print(f"\nwrote {OUT}/fst_persite_degen.txt and fst_per_transcript_body.csv")


if __name__ == "__main__":
    main()
