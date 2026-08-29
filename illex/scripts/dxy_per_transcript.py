#!/usr/bin/env python
"""Per-transcript between-arrangement divergence, normalised to its own locale.

    .venv/bin/python -m illex.scripts.dxy_per_transcript

WHY NOT FST
-----------
`fst_persite_degen.py` tried this with Hudson FST and it does not work
(NOTES sec 8.25). FST = sum(A)/sum(B); at a near-monomorphic site both go to ~0
while the numerator carries the negative sampling corrections -p(1-p)/(n-1), so
the ratio is dragged toward 0 wherever diversity is low. CDS is mostly
constrained, so the per-transcript ranking came out ordered by polymorphism
(Spearman +0.69) rather than by differentiation, and transcripts reading
FST ~ 0.012 sat inside 100 kb windows at FST 0.37-0.46.

dxy is an absolute divergence, not a ratio, so it has no small-denominator
pathology. ANGSD's `fst print` already emits it: the B column is the Hudson
denominator p1(1-p2) + p2(1-p1), the chance that an allele drawn from AA and
one from BB differ.

**SCALE CAVEAT.** mean(B) is 3.9x (body) and 3.1x (control) the pg_gpu dxy for
the same regions -- proportional, but NOT on the same scale, and the factor is
not even constant between regions. So B is used ONLY in ratios taken WITHIN a
region, where any constant cancels. No absolute dxy is quoted from it.

THE NORMALISATION, AND WHY IT IS NEEDED
---------------------------------------
Raw per-transcript dxy still confounds two things: how long the barrier has
existed (what we want) and how constrained the gene is (what we do not).
Purifying selection lowers dxy at a constrained gene exactly as it lowers pi.
So each transcript is divided by the mean B of its own 100 kb window, which
carries the same local mutation rate and the same barrier age. What survives is
the gene's divergence relative to its immediate neighbourhood.

A transcript sitting BELOW its window is just constrained -- the expected case,
and uninformative. A transcript sitting ABOVE its window is the candidate: more
diverged between arrangements than the surrounding sequence.

THE CONTROL SETS THE BAR. The collinear region has no barrier, so its
transcript/window ratios measure only constraint plus noise. Its spread is what
"unremarkable" looks like, and a body transcript has to clear it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SFS = Path(".tmp/sfs2d")
DEGEN = Path(".tmp/fstsnp/chr2_degen.tsv")
ENTAP = Path("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/"
             "project/functional_entap/entap_outfiles/final_results/annotated.tsv")
OUT = Path("results/illex")
WIN = 100_000
MIN_CDS = 50


def persite(lab: str) -> pd.DataFrame:
    return pd.read_csv(SFS / f"AA_BB.{lab}.persite.gz", sep="\t", header=None,
                       names=["chrom", "pos", "A", "B"],
                       usecols=["pos", "A", "B"])


def per_transcript(lab: str, deg: pd.DataFrame) -> pd.DataFrame:
    d = persite(lab)
    d["win"] = d.pos // WIN
    win = d.groupby("win").B.mean().rename("win_dxy")
    j = d.merge(deg, on="pos", how="inner")
    g = (j.groupby("transcript")
           .agg(dxy=("B", "mean"), n=("pos", "size"), pos=("pos", "median"),
                n0=("degen", lambda x: int((x == 0).sum())),
                n4=("degen", lambda x: int((x == 4).sum())))
           .reset_index())
    g = g[g.n >= MIN_CDS].copy()
    g["win"] = (g.pos // WIN).astype(int)
    g = g.merge(win, left_on="win", right_index=True, how="left")
    g["ratio"] = g.dxy / g.win_dxy
    g["f0"] = g.n0 / (g.n0 + g.n4).replace(0, np.nan)
    return g


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    deg = pd.read_csv(DEGEN, sep="\t", header=None, names=["pos", "degen", "tx"])
    deg["transcript"] = deg.tx.str.split(":").str[0]

    emit("Per-transcript between-arrangement divergence, normalised to its own")
    emit("100 kb window. Replaces the FST version, which was invalid (sec 8.25).")
    emit("B is proportional to dxy on ANGSD's scale, so only WITHIN-region")
    emit("ratios are used and no absolute dxy is quoted.")
    emit()

    g = {lab: per_transcript(lab, deg) for lab in ("body", "control")}

    emit("=" * 76)
    emit("DOES THE STATISTIC BEHAVE? (this is where FST failed)")
    emit("=" * 76)
    for lab in ("body", "control"):
        x = g[lab]
        emit(f"  {lab:<8s} {len(x):3d} transcripts   "
             f"corr(transcript dxy, window dxy) = {x.dxy.corr(x.win_dxy):+.3f}  "
             f"(Spearman {x.dxy.corr(x.win_dxy, method='spearman'):+.3f})")
    emit("  For comparison, per-transcript FST vs window FST was +0.20 / +0.16.")
    emit()
    emit("  Constraint check -- dxy should FALL as the 0-fold fraction rises:")
    for lab in ("body", "control"):
        x = g[lab].dropna(subset=["f0"])
        emit(f"    {lab:<8s} corr(ratio, 0-fold fraction) = "
             f"{x.ratio.corr(x.f0):+.3f}")
    emit()

    emit("=" * 76)
    emit("THE BAR: what does 'unremarkable' look like in the control?")
    emit("=" * 76)
    c = g["control"].ratio.dropna()
    b = g["body"].ratio.dropna()
    emit(f"  {'':<10s}{'n':>5s}{'median':>9s}{'p90':>9s}{'p95':>9s}{'max':>9s}")
    for lab, x in (("control", c), ("body", b)):
        emit(f"  {lab:<10s}{len(x):5d}{x.median():9.3f}{x.quantile(.9):9.3f}"
             f"{x.quantile(.95):9.3f}{x.max():9.3f}")
    thr = c.quantile(0.95)
    emit()
    emit(f"  Control 95th percentile = {thr:.3f}. Body transcripts above it: "
         f"{int((b > thr).sum())} of {len(b)} "
         f"({100 * (b > thr).mean():.1f}%) -- {5.0:.0f}% expected by chance.")
    emit()

    ann = {}
    if ENTAP.exists():
        a = pd.read_csv(ENTAP, sep="\t", usecols=[0, 12], low_memory=False)
        a.columns = ["q", "desc"]
        ann = dict(zip(a.q, a.desc.fillna("")))

    emit("=" * 76)
    emit("BODY TRANSCRIPTS RANKED BY DIVERGENCE RELATIVE TO THEIR OWN WINDOW")
    emit("=" * 76)
    emit(f"  {'transcript':<22s}{'Mb':>7s}{'nCDS':>6s}{'ratio':>7s}{'0f':>6s}  "
         "description")
    for _, r in g["body"].sort_values("ratio", ascending=False).head(15).iterrows():
        d_ = str(ann.get(r.transcript, ""))[:40] or "(no EnTAP hit)"
        f0 = f"{r.f0:.2f}" if pd.notna(r.f0) else "  --"
        emit(f"  {r.transcript:<22s}{r.pos / 1e6:7.2f}{r.n:6d}{r.ratio:7.3f}"
             f"{f0:>6s}  {d_}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dxy_per_transcript.txt").write_text("\n".join(lines) + "\n")
    for lab in ("body", "control"):
        g[lab].to_csv(OUT / f"dxy_per_transcript_{lab}.csv", index=False)
    print(f"\nwrote {OUT}/dxy_per_transcript.txt and _{{body,control}}.csv")


if __name__ == "__main__":
    main()
