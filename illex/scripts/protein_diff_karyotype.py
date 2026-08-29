#!/usr/bin/env python
"""Protein-coding differences between arrangements, from genotype likelihoods.

    .venv/bin/python -m illex.scripts.protein_diff_karyotype

THE QUESTION
------------
Is there any noticeable difference between the AA and BB arrangements at the
protein level -- near-fixed amino-acid-changing differences that could carry a
functional story?

WHY THIS INSTRUMENT
-------------------
The called-VCF version failed its control outright (NOTES sec 8.23): structured
missingness made sites look fixed wherever few genotypes were called, and the
collinear region -- where AA and BB are exchangeable and there can be no real
differentiation -- returned 267 apparent fixed differences. Per-site Hudson FST
from GLs then failed for a different reason, a small-denominator bias (sec
8.25).

Allele FREQUENCY difference from genotype likelihoods avoids both. It has no
calling step to ascertain against, and |dp| is not a ratio so it has no
denominator pathology. `-doMajorMinor 4` pins the major allele to the reference
for BOTH groups so the frequencies are comparable site by site.

THE CONTROL IS THE WHOLE TEST
-----------------------------
AA and BB are exchangeable in the collinear region. Whatever |dp| tail appears
there is the false-positive rate, and any claim about the inversion has to clear
it. This is the check the callset failed.

STRAND -- THE BUG THAT ALMOST PRODUCED TWO NONSENSE MUTATIONS
-------------------------------------------------------------
degenotate reports `ref` and the alternative-codon table on the **transcript**
strand; ANGSD reports major/minor on the **genomic** strand. For a minus-strand
gene they are complements. Verified directly: the genome base at 2:68,088,901 is
G, degenotate lists C.

Looking the ANGSD minor allele up in degenotate's table without complementing
called two premature stops (S->* at 68.089 Mb, E->* at 71.895 Mb), both in
minus-strand genes. After complementing they are ordinary missense changes
(S->L and E->K) and **there are ZERO stop-gains among the near-fixed
differences**. Degeneracy class (0-fold/4-fold) is strand-invariant, so the
dN/dS-like result below is unaffected -- only residue identities were wrong.

THE CAVEAT THAT SURVIVES A CLEAN RESULT
---------------------------------------
Recombination between arrangements is suppressed across the whole block, so
every site in it is linked to the karyotype. A near-fixed nonsynonymous site is
a PASSENGER on the haplotype unless something independent implicates it. This
analysis can establish that protein-level differences exist and count them; it
cannot identify a causal one. Five prior analyses (sec 8.10, 8.21, 8.25, 8.26,
8.27) found nothing localised inside the block.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

MAF = Path(".tmp/maf")
DEGEN = Path(".tmp/fstsnp/chr2_degen.tsv")
DEGEN_FULL = Path("/sietch_colab/data_share/illex/popgen_data/degenotate_illex/"
                  "degeneracy-all-sites.bed")
ENTAP = Path("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/"
             "project/functional_entap/entap_outfiles/final_results/annotated.tsv")
OUT = Path("results/illex")
MIN_IND = {"AA": 150, "BB": 60}       # ~60% of 254 / 95
GFF = Path("/sietch_colab/data_share/illex/popgen_data/degenotate_illex/"
           "Illex_F24.gene_lnc_pseudo.func.fix.sq3.FINAL.v2.fixID.gff3")
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def strands() -> dict:
    """Transcript -> strand. Needed to complement the ANGSD minor allele before
    looking it up in degenotate's transcript-strand codon table."""
    import re
    out = {}
    with GFF.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) > 8 and f[2] == "mRNA":
                mm = re.search(r"ID=([^;]+)", f[8])
                if mm:
                    out[mm.group(1)] = f[6]
    return out


def load(region: str) -> pd.DataFrame:
    out = {}
    for c in ("AA", "BB"):
        d = pd.read_csv(MAF / f"{c}.{region}.mafs.gz", sep="\t",
                        usecols=["position", "major", "minor", "knownEM",
                                 "nInd"],
                        dtype={"position": "int64", "knownEM": "float32",
                               "nInd": "int32"})
        d = d[d.nInd >= MIN_IND[c]]
        out[c] = d.rename(columns={"knownEM": f"p_{c}", "nInd": f"n_{c}",
                                   "minor": f"min_{c}"})[
            ["position", f"p_{c}", f"n_{c}", f"min_{c}"]]
    m = out["AA"].merge(out["BB"], on="position")
    # the inferred minor allele must agree, else dp compares different alleles
    m = m[m.min_AA == m.min_BB]
    m["dp"] = (m.p_AA - m.p_BB).abs()
    return m


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    deg = pd.read_csv(DEGEN, sep="\t", header=None,
                      names=["position", "degen", "tx"])
    deg["transcript"] = deg.tx.str.split(":").str[0]

    emit("Protein-coding differences between arrangements, from ANGSD genotype")
    emit("likelihoods. dp = |freq(AA) - freq(BB)| at the same reference-pinned")
    emit("allele. The called-VCF version of this failed its control (sec 8.23).")
    emit()

    store = {}
    for region in ("body", "control"):
        m = load(region)
        store[region] = m
        emit("=" * 76)
        emit(f"{region.upper()}   {len(m):,} sites passing coverage and "
             "major/minor agreement")
        emit("=" * 76)
        emit(f"  {'dp class':>16s} {'n sites':>12s} {'rate':>9s}")
        for lab, thr in (("all", 0.0), ("dp >= 0.5", 0.5), ("dp >= 0.8", 0.8),
                         ("dp >= 0.9", 0.9), ("dp >= 0.95", 0.95),
                         ("dp >= 0.99", 0.99)):
            s = m[m.dp >= thr]
            emit(f"  {lab:>16s} {len(s):12,} {100 * len(s) / len(m):8.4f}%")
        emit()

    b, c = store["body"], store["control"]
    emit("=" * 76)
    emit("THE CONTROL -- does the instrument work this time?")
    emit("=" * 76)
    for thr in (0.8, 0.9, 0.95, 0.99):
        rb = (b.dp >= thr).mean()
        rc = (c.dp >= thr).mean()
        enr = rb / rc if rc > 0 else float("inf")
        emit(f"  dp >= {thr:.2f}:  body {100 * rb:8.4f}%   control "
             f"{100 * rc:8.4f}%   enrichment {enr:8.1f}x")
    emit()
    emit("  The called VCF gave a body/control ratio of ~3x where FST differs")
    emit("  100-fold, which is how we knew it was noise (sec 8.23).")
    emit()

    # ---- coding ---------------------------------------------------------
    emit("=" * 76)
    emit("ARE THE DIFFERENTIATED SITES PROTEIN-ALTERING?")
    emit("=" * 76)
    for region, m in (("body", b), ("control", c)):
        j = m.merge(deg, on="position", how="inner")
        b0 = int((j.degen == 0).sum())
        b4 = int((j.degen == 4).sum())
        emit(f"  {region.upper()}: {len(j):,} coding sites  "
             f"({b0:,} 0-fold / {b4:,} 4-fold, 0f frac {b0 / (b0 + b4):.3f})")
        for thr in (0.5, 0.8, 0.9):
            s = j[j.dp >= thr]
            n0, n4 = int((s.degen == 0).sum()), int((s.degen == 4).sum())
            if n0 + n4 < 10:
                emit(f"    dp >= {thr}: {n0} 0-fold / {n4} 4-fold -- too few "
                     "to test")
                continue
            odds, p = fisher_exact([[n0, n4], [b0 - n0, b4 - n4]])
            emit(f"    dp >= {thr}: {n0:5,} 0-fold / {n4:5,} 4-fold  "
                 f"frac {n0 / (n0 + n4):.3f}  OR {odds:5.2f}  p = {p:.3g}")
        emit()

    # ---- the actual amino-acid changes ---------------------------------
    jb = b.merge(deg, on="position", how="inner")
    hi = jb[(jb.dp >= 0.8) & (jb.degen == 0)]
    emit("=" * 76)
    emit(f"NEAR-FIXED NONSYNONYMOUS SITES IN THE BODY (dp >= 0.8, 0-fold): "
         f"{len(hi):,}")
    emit("=" * 76)
    if len(hi):
        ann = {}
        if ENTAP.exists():
            a = pd.read_csv(ENTAP, sep="\t", usecols=[0, 12], low_memory=False)
            a.columns = ["q", "desc"]
            ann = dict(zip(a.q, a.desc.fillna("")))
        tx = hi.transcript.value_counts()
        emit(f"  across {len(tx)} transcripts")
        emit(f"  {'transcript':<24s}{'n':>4s}  description")
        for name, cnt in tx.head(15).items():
            d_ = str(ann.get(name, ""))[:46] or "(no EnTAP hit)"
            emit(f"  {name:<24s}{cnt:4d}  {d_}")
        hi.to_csv(OUT / "protein_diff_nonsyn_body.csv", index=False)
    emit()
    emit("  CAVEAT: recombination is suppressed across the whole block, so a")
    emit("  near-fixed nonsynonymous site is a PASSENGER on the arrangement's")
    emit("  haplotype unless something independent implicates it. This counts")
    emit("  protein-level differences; it does not identify a causal one.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protein_diff_karyotype.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/protein_diff_karyotype.txt")


if __name__ == "__main__":
    main()
