#!/usr/bin/env python
"""Genome-wide Fst(AA, BB): does anything OUTSIDE the inversion track karyotype?

    .venv/bin/python -m illex.scripts.karyotype_fst_scan --workers 10

THE QUESTION
------------
Inside the inversion every locus is perfectly linked to karyotype by
construction — that is what a supergene is, and Fst(AA,BB) = 0.365 is its
summary. The interesting question is whether the co-adapted complex extends
BEYOND the inversion: are there loci elsewhere in the genome whose allele
frequencies covary with the arrangement?

That would be a polygenic architecture — epistatic partners, or co-adapted
alleles the inversion does not physically contain. It is also the only version
of the "coordinated allele frequency change" idea that this dataset can address.
The temporal-covariance approach (variance in allele-frequency change across
consecutive generations, decomposed into drift versus linked selection) needs
multi-generation sampling and fitness proxies; Illex has a single time point, so
that method is unavailable here.

WHAT IS COMPUTED
----------------
Hudson's Fst between AA and BB homokaryotypes, per 100 kb window, genome-wide,
as a RATIO OF SUMS (the correct estimator — averaging per-site Fst is biased):

    num_i = (p1-p2)^2 - p1(1-p1)/(n1-1) - p2(1-p2)/(n2-1)
    den_i = p1(1-p2) + p2(1-p1)
    Fst   = sum_i num_i / sum_i den_i

Allele counts come straight from the callset via ``bcftools +fill-tags``.

    NOTE: this needs BCFTOOLS_PLUGINS set. Without it ``+fill-tags`` fails with
    the unhelpful "Failed to read from standard input: unknown file type",
    which is easy to misdiagnose as a piping problem — it is a missing plugin
    path. The script sets it explicitly.

WHAT THE RESULT WOULD MEAN
--------------------------
* Fst ~ 0 everywhere outside chr2:60-80 Mb → the supergene is self-contained.
  The manuscript's windowed PCA already says this at window scale (eta^2 0.748
  inside versus 0.003 outside); this is the per-window refinement, which can see
  individual loci a PCA summary cannot.
* Isolated elevated windows elsewhere → candidate co-adapted or epistatic
  partners, and the polygenic reading gains real support.

CAVEATS
-------
1. Only 149 AA and 53 BB individuals overlap this callset (the clean-350 set),
   so per-window Fst is noisy. Hudson's estimator is unbiased with unequal and
   small samples, which is why it is used, but the tail of the null distribution
   is wide and single outlier windows should not be trusted without support from
   neighbours.
2. Any real population structure would inflate Fst everywhere. Geographic Fst is
   ~0 in this system, so this is expected to be minor, but the collinear chr2
   windows serve as an internal check: they should look like the rest of the
   genome.
3. Karyotype was called FROM chr2 genotypes, so chr2 is not independent of the
   grouping. That is fine for the inversion itself (it is the thing being
   measured) but means chr2 collinear windows are the closest thing to a
   positive-control-free region and should be read with that in mind.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

VCF_DIR = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
           "14_sweep_seqmodel/results/empirical_scan_fullsfs/vcf")
KARYO = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
         "03_karyotype")
BCFTOOLS = "/home/ssmall/bin/bcftools"
PLUGINS = "/home/ssmall/programs/bcftools-1.21/plugins"
OUT = Path("results/illex")

WINDOW = 100_000
INV = (60_040_617, 79_995_597)
INV_CHR = "2"
MIN_SITES = 50          # windows with fewer informative sites are dropped


def _counts(vcf: str, samples: str) -> pd.DataFrame:
    """POS, AC, AN for one karyotype class from one chromosome VCF."""
    env = dict(os.environ, BCFTOOLS_PLUGINS=PLUGINS)
    p1 = subprocess.Popen(
        [BCFTOOLS, "view", "-S", samples, "--force-samples",
         "-m2", "-M2", "-v", "snps", vcf, "-Ou"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    p2 = subprocess.Popen(
        [BCFTOOLS, "+fill-tags", "-Ou", "--", "-t", "AN,AC"],
        stdin=p1.stdout, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env)
    p3 = subprocess.Popen(
        [BCFTOOLS, "query", "-f", "%POS\t%AC\t%AN\n"],
        stdin=p2.stdout, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env)
    p1.stdout.close()
    p2.stdout.close()
    d = pd.read_csv(p3.stdout, sep="\t", header=None,
                    names=["pos", "ac", "an"],
                    dtype={"pos": np.int64, "ac": np.float64,
                           "an": np.float64})
    p3.wait()
    return d


def scan_chrom(chrom: str) -> pd.DataFrame:
    vcf = f"{VCF_DIR}/chr{chrom}.nomaf.vcf.gz"
    if not Path(vcf).exists():
        return pd.DataFrame()
    a = _counts(vcf, f"{KARYO}/AA_samples.txt")
    b = _counts(vcf, f"{KARYO}/BB_samples.txt")
    m = a.merge(b, on="pos", suffixes=("_a", "_b"))
    # Both classes must be genotyped; Hudson needs n-1 in the denominator.
    m = m[(m.an_a >= 2) & (m.an_b >= 2)]
    if m.empty:
        return pd.DataFrame()
    p1 = (m.ac_a / m.an_a).to_numpy()
    p2 = (m.ac_b / m.an_b).to_numpy()
    n1 = m.an_a.to_numpy()
    n2 = m.an_b.to_numpy()
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    win = (m.pos.to_numpy() // WINDOW) * WINDOW
    d = pd.DataFrame({"win": win, "num": num, "den": den})
    g = d.groupby("win").agg(num=("num", "sum"), den=("den", "sum"),
                             n_sites=("num", "size")).reset_index()
    g = g[(g.n_sites >= MIN_SITES) & (g.den > 0)]
    g["fst"] = (g.num / g.den).clip(lower=0.0)
    g.insert(0, "chrom", chrom)
    return g[["chrom", "win", "n_sites", "fst"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chroms", help="comma-separated subset")
    a = ap.parse_args()

    if a.chroms:
        chroms = a.chroms.split(",")
    else:
        chroms = sorted(
            (f.name[3:-len(".nomaf.vcf.gz")]
             for f in Path(VCF_DIR).glob("chr*.nomaf.vcf.gz")),
            key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
    print(f"{len(chroms)} chromosomes, {a.workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        parts = list(ex.map(scan_chrom, chroms))
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "karyotype_fst_scan.tsv", sep="\t", index=False)

    inv = d[(d.chrom == INV_CHR) & (d.win >= INV[0]) & (d.win < INV[1])]
    rest = d[~((d.chrom == INV_CHR) & (d.win >= INV[0] - 2_000_000)
               & (d.win < INV[1] + 2_000_000))]
    chr2_coll = d[(d.chrom == INV_CHR)
                  & ((d.win < INV[0] - 2_000_000) | (d.win >= INV[1] + 2_000_000))]

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"windows: {len(d):,} total | inversion {len(inv)} | "
         f"chr2 collinear {len(chr2_coll)} | rest of genome "
         f"{len(rest) - len(chr2_coll):,}")
    emit()
    emit(f"{'set':<22s} {'n':>7s} {'median':>8s} {'mean':>8s} {'p99':>8s} "
         f"{'max':>8s}")
    for nm, sub in (("INVERSION body", inv),
                    ("chr2 collinear", chr2_coll),
                    ("all other chromosomes", rest[rest.chrom != INV_CHR])):
        if not len(sub):
            continue
        emit(f"{nm:<22s} {len(sub):7d} {sub.fst.median():8.4f} "
             f"{sub.fst.mean():8.4f} {sub.fst.quantile(.99):8.4f} "
             f"{sub.fst.max():8.4f}")
    emit()

    bg = rest.fst
    thr = bg.quantile(0.9999)
    out = rest[rest.fst > thr].sort_values("fst", ascending=False)
    emit(f"background (everything outside the inversion +/-2 Mb): "
         f"99.99th percentile = {thr:.4f}")
    emit(f"windows above it: {len(out)}")
    if len(out):
        emit(f"  {'chrom':>6s} {'window':>12s} {'n_sites':>8s} {'fst':>8s}  "
             "neighbours>p999")
        for _, r in out.head(25).iterrows():
            nb = rest[(rest.chrom == r.chrom)
                      & (abs(rest.win - r.win) <= 3 * WINDOW)
                      & (rest.fst > bg.quantile(0.999))]
            emit(f"  {r.chrom:>6s} {int(r.win):>12,} {int(r.n_sites):>8,} "
                 f"{r.fst:8.4f}  {len(nb) - 1}")
    emit()
    emit("A real co-adapted locus should show SEVERAL adjacent elevated windows "
         "(linkage),\nnot one isolated spike. The neighbours column is that "
         "check.")

    (OUT / "karyotype_fst_scan.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/karyotype_fst_scan.{{tsv,txt}}")


if __name__ == "__main__":
    main()
