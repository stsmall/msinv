#!/usr/bin/env python
"""Near-fixed AA/BB sites in ISOSEQ exons that have NO curated gene model.

    .venv/bin/python -m illex.scripts.isoseq_only_diff_sites

WHAT THESE ARE, AND WHAT THEY ARE NOT
-------------------------------------
NOTES sec 8.31 tested whether the high-differentiation sequence outside the
curated annotation is really unannotated genes. Against the raw ISOSEQ
alignments (12.1% genomic coverage, vs the StringTie merge's 6.1% which had lost
most of the signal) the answer was no: near-fixed sites fall in ISOSEQ exons at
**0.98x** the background rate -- exactly proportional, no enrichment.

So this set is NOT a discovery of hidden functional sequence. It is the
proportional expectation. It is pulled out because it is the only CONCRETE
material the annotation gap yields: real transcribed exons, carrying real
arrangement differentiation, with no gene model over them.

And the block-wide caveat still applies with full force: recombination between
arrangements is suppressed across all 20 Mb, so every one of these sites is
linked to the karyotype and is a PASSENGER unless something independent
implicates it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAF = Path(".tmp/maf")
EXONS = Path(".tmp/isoseq/collapse_body_exons.bed")
GENES = Path("/sietch_colab/data_share/illex/popgen_data/degenotate_illex/"
             "Illex.genes.bed")
ISO = ("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/"
       "project/isoseq/isoseq.collapse.bam")
BT = "/home/ssmall/miniforge3/envs/bioinfo-buddy/bin"
OUT = Path("results/illex")
DP_THR = 0.8
MIN_IND = {"AA": 150, "BB": 60}
LO, HI = 60_040_617, 79_995_597


def main() -> None:
    d = {}
    for c in ("AA", "BB"):
        x = pd.read_csv(MAF / f"{c}.body.mafs.gz", sep="\t",
                        usecols=["position", "major", "minor", "knownEM",
                                 "nInd"],
                        dtype={"position": "int64", "knownEM": "float32",
                               "nInd": "int32"})
        d[c] = x[x.nInd >= MIN_IND[c]].rename(
            columns={"knownEM": f"p_{c}", "nInd": f"n_{c}"})
    m = d["AA"].merge(d["BB"][["position", "p_BB", "n_BB"]], on="position")
    m["dp"] = (m.p_AA - m.p_BB).abs()
    hi = m[m.dp >= DP_THR].copy()

    g = pd.read_csv(GENES, sep="\t", header=None, names=["c", "s", "e"])
    g = g[g.c.astype(str) == "2"].sort_values("s")
    pos = hi.position.to_numpy()
    ing = np.zeros(len(pos), bool)
    for s, e in zip(g.s, g.e):
        ing |= (pos > s) & (pos <= e)
    hi = hi[~ing].copy()

    ex = pd.read_csv(EXONS, sep="\t", header=None,
                     names=["c", "s", "e", "tx", "score", "strand"])
    # assign each site to the exon(s) covering it
    rows = []
    p = hi.position.to_numpy()
    for _, r in ex.iterrows():
        sel = (p > r.s) & (p <= r.e)
        if sel.any():
            for q in p[sel]:
                rows.append((q, r.tx, r.strand, int(r.s), int(r.e)))
    a = pd.DataFrame(rows, columns=["position", "iso_tx", "strand",
                                    "exon_start", "exon_end"])
    res = hi.merge(a, on="position", how="inner")

    # collapse the PB.<locus>.<isoform> transcript ids to their locus
    res["iso_locus"] = res.iso_tx.str.rsplit(".", n=1).str[0]
    # distance to the nearest curated gene
    gs, ge = g.s.to_numpy(), g.e.to_numpy()
    res["dist_to_gene"] = [
        int(min(np.min(np.abs(gs - q)), np.min(np.abs(ge - q))))
        for q in res.position]

    uniq = res.drop_duplicates("position").sort_values("position")
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Near-fixed AA/BB sites in ISOSEQ exons with NO curated gene model")
    emit(f"dp >= {DP_THR}, inversion body {LO:,}-{HI:,}")
    emit()
    emit(f"  {len(uniq):,} unique positions in "
         f"{res.iso_locus.nunique()} ISOSEQ loci "
         f"({res.iso_tx.nunique()} isoforms)")
    emit("  Enrichment vs background is 0.98x (sec 8.31) -- this is the")
    emit("  PROPORTIONAL expectation, not a discovery. Every site is linked to")
    emit("  the karyotype across the whole block and is a passenger unless")
    emit("  independently implicated.")
    emit()
    emit("  BY ISOSEQ LOCUS")
    emit(f"  {'locus':<14s}{'n sites':>8s}{'span (Mb)':>22s}{'strand':>7s}"
         f"{'dist to gene':>14s}{'max dp':>8s}")
    for loc, sub in sorted(res.groupby("iso_locus"),
                           key=lambda kv: -kv[1].position.nunique()):
        u = sub.drop_duplicates("position")
        emit(f"  {loc:<14s}{len(u):8d}"
             f"{u.position.min() / 1e6:11.3f}-{u.position.max() / 1e6:<10.3f}"
             f"{sub.strand.iloc[0]:>7s}{int(u.dist_to_gene.min()):14,d}"
             f"{u.dp.max():8.2f}")
    emit()
    emit("  ALL SITES")
    emit(f"  {'position':>12s}{'dp':>6s}{'p_AA':>7s}{'p_BB':>7s}{'maj':>4s}"
         f"{'min':>4s}{'locus':>13s}{'dist_gene':>11s}")
    for _, r in uniq.iterrows():
        emit(f"  {r.position:12,d}{r.dp:6.2f}{r.p_AA:7.3f}{r.p_BB:7.3f}"
             f"{r.major:>4s}{r.minor:>4s}{r.iso_locus:>13s}"
             f"{int(r.dist_to_gene):11,d}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "isoseq_only_diff_sites.txt").write_text("\n".join(lines) + "\n")
    res.to_csv(OUT / "isoseq_only_diff_sites.csv", index=False)
    print(f"\nwrote {OUT}/isoseq_only_diff_sites.{{txt,csv}}")


if __name__ == "__main__":
    main()
