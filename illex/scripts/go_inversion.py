#!/usr/bin/env python
"""GO content of the chr2 inversion, with a null that respects linkage.

    .venv/bin/python -m illex.scripts.go_inversion

WHY REDO THE EXISTING ANALYSIS
------------------------------
`steps/06_gene_content/go_enrichment.py` tests the 102 inversion genes against a
genome-wide gene background with a hypergeometric test. That null assumes the
102 genes are a random SAMPLE OF GENES. They are not: they are one contiguous
20 Mb block inherited as a unit. Two consequences, both visible in its output:

1. **Tandem duplicates are counted as independent observations.** Its single
   most significant term, "fructose-bisphosphate aldolase activity"
   (2 of 3 genome-wide, p = 3.4e-5), is LOC_00005292 and LOC_00005293 — 33 kb
   apart. That is ONE duplication event scored twice. "aldehyde-lyase activity"
   is the same two genes again (FBPA is an aldehyde-lyase).
2. **Gene families cluster in genomes**, so ANY contiguous window is enriched
   for whatever happens to sit in it. The comparison has to be against other
   windows, not against shuffled genes.

It also uses the NOMINAL span. The gene set runs 60,013,280–79,998,501, so it
includes genes outside the breakpoints entirely — ACAD10 starts at 60,013,280,
before the 60,040,617 breakpoint — and the outermost ~500 kb at each end is
collinear flanking sequence with control-like Fst (NOTES sec 4.2). Three of the
top-term genes sit in exactly that flank.

WHAT THIS DOES INSTEAD
----------------------
* Foreground = genes in the **differentiated body** (Fst-defined, 60.5–79.5 Mb),
  with the nominal span reported alongside so the difference is visible.
* **Window null**: the observed count for each GO term is compared against the
  same count in many random contiguous windows of the same physical length
  drawn from elsewhere in the genome. The empirical p is the fraction of windows
  reaching the observed count. This absorbs both gene clustering and gene-density
  variation, which the hypergeometric cannot.
* **Cluster collapsing**: genes with the same GO term within ``CLUSTER_BP`` of
  each other are counted once, so a tandem array contributes one observation.
  Reported next to the raw count so the effect of the correction is explicit.

The point is not to overturn the biology — the lipid/fatty-acid signal may well
survive — but to find out which parts of it are load-bearing.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GFF = ("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/project/"
       "final_gffs/Illex_F24.gene_lnc_pseudo.func.fix.sq3.FINAL.v2.fixID.gff3")
GO_TSV = ("/sietch_colab/data_share/illex/annotations/gene_annotation_dir/"
          "project/functional_entap/entap_outfiles/final_results/"
          "annotated_without_contam_gene_ontology_terms.tsv")
OUT = Path("results/illex")

INV_CHR = "2"
BODY = (60_500_000, 79_500_000)        # Fst-defined differentiated extent
NOMINAL = (60_040_617, 79_995_597)

CLUSTER_BP = 200_000     # genes this close sharing a term = one observation
N_WINDOWS = 2000         # random windows for the empirical null
ROOTS = {"GO:0005575", "GO:0003674", "GO:0008150"}
MIN_K = 2                # ignore terms with <2 genes in the foreground
SEED = 20260824


def load_genes() -> pd.DataFrame:
    """(gene, chrom, start, end) for every gene in the GFF."""
    rows = []
    op = gzip.open if GFF.endswith(".gz") else open
    with op(GFF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = re.search(r"ID=([^;]+)", f[8])
            if not m:
                continue
            rows.append((m.group(1), f[0], int(f[3]), int(f[4])))
    d = pd.DataFrame(rows, columns=["gene", "chrom", "start", "end"])
    d["mid"] = (d.start + d.end) // 2
    return d


def load_go():
    go = pd.read_csv(GO_TSV, sep="\t")
    go["gene"] = (go["query_sequence"].astype(str)
                  .str.split("-mRNA").str[0].str.split("-RA").str[0])
    go = go[~go["go_id"].isin(ROOTS)]
    name = dict(zip(go["go_id"], go["go_term"]))
    cat = dict(zip(go["go_id"], go["category"]))
    gene2go = go.groupby("gene")["go_id"].apply(set).to_dict()
    return gene2go, name, cat


def counts_in(genes: pd.DataFrame, gene2go: dict,
              collapse: bool) -> dict[str, int]:
    """GO-term counts for a gene set, optionally collapsing tandem clusters."""
    per_term = defaultdict(list)
    for g, mid in zip(genes.gene, genes.mid):
        for t in gene2go.get(g, ()):  # noqa: SIM118
            per_term[t].append(mid)
    out = {}
    for t, mids in per_term.items():
        if not collapse:
            out[t] = len(mids)
            continue
        mids = sorted(mids)
        n, last = 1, mids[0]
        for m in mids[1:]:
            if m - last > CLUSTER_BP:
                n += 1
            last = m
        out[t] = n
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, monotone-enforced."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main() -> None:
    genes = load_genes()
    gene2go, term_name, term_cat = load_go()
    annotated = set(gene2go)
    genes = genes[genes.gene.isin(annotated)].reset_index(drop=True)
    print(f"{len(genes):,} annotated genes with >=1 GO term, "
          f"{genes.chrom.nunique()} sequences")

    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    span = BODY[1] - BODY[0]
    fg = genes[(genes.chrom == INV_CHR) & (genes.mid >= BODY[0])
               & (genes.mid < BODY[1])]
    nom = genes[(genes.chrom == INV_CHR) & (genes.mid >= NOMINAL[0])
                & (genes.mid < NOMINAL[1])]
    emit(f"differentiated body {BODY[0]:,}-{BODY[1]:,} ({span / 1e6:.1f} Mb): "
         f"{len(fg)} annotated genes")
    emit(f"nominal span        {NOMINAL[0]:,}-{NOMINAL[1]:,}: {len(nom)} "
         "annotated genes")
    emit("  (the published set of 102 uses the nominal span and includes genes "
         "outside\n   the breakpoints entirely)")
    emit()

    obs_raw = counts_in(fg, gene2go, collapse=False)
    obs_cl = counts_in(fg, gene2go, collapse=True)

    # ---- window null -----------------------------------------------------
    # Windows of the same physical length, from sequences long enough to hold
    # one, excluding chr2 so the inversion cannot sample itself.
    rng = np.random.default_rng(SEED)
    lens = genes.groupby("chrom").mid.max()
    usable = lens[(lens > span) & (lens.index != INV_CHR)]
    emit(f"window null: {N_WINDOWS:,} random {span / 1e6:.1f} Mb windows from "
         f"{len(usable)} sequences (chr2 excluded)")
    by_chrom = {c: g.sort_values("mid").reset_index(drop=True)
                for c, g in genes[genes.chrom.isin(usable.index)].groupby(
                    "chrom")}
    weights = (usable - span).clip(lower=1)
    weights = (weights / weights.sum()).values
    chroms = list(usable.index)

    null_raw = defaultdict(list)
    null_cl = defaultdict(list)
    n_genes_null = []
    for _ in range(N_WINDOWS):
        c = chroms[rng.choice(len(chroms), p=weights)]
        g = by_chrom[c]
        lo = rng.integers(0, max(1, int(lens[c] - span)))
        sub = g[(g.mid >= lo) & (g.mid < lo + span)]
        n_genes_null.append(len(sub))
        cr = counts_in(sub, gene2go, collapse=False)
        cc = counts_in(sub, gene2go, collapse=True)
        for t in set(obs_raw):
            null_raw[t].append(cr.get(t, 0))
            null_cl[t].append(cc.get(t, 0))
    n_genes_null = np.array(n_genes_null)
    emit(f"  genes per null window: median {np.median(n_genes_null):.0f}, "
         f"5-95% {np.percentile(n_genes_null, 5):.0f}-"
         f"{np.percentile(n_genes_null, 95):.0f}   "
         f"(inversion body: {len(fg)})")
    emit()

    rows = []
    for t, k in obs_raw.items():
        if k < MIN_K:
            continue
        nr = np.array(null_raw[t])
        nc = np.array(null_cl[t])
        kc = obs_cl[t]
        rows.append({
            "go_id": t, "cat": term_cat.get(t, ""),
            "term": term_name.get(t, ""),
            "k_raw": k, "k_clustered": kc,
            "null_mean_raw": nr.mean(),
            "p_window_raw": float((nr >= k).mean()),
            "p_window_clustered": float((nc >= kc).mean()),
        })
    res = pd.DataFrame(rows)
    # BH over the clustered window p-values, the strictest of the three.
    # Implemented inline: statsmodels is not in this venv, and pulling it in for
    # four lines would add a dependency to the msinv environment.
    res["fdr_clustered"] = _bh(
        res["p_window_clustered"].clip(lower=1.0 / N_WINDOWS).to_numpy())
    res = res.sort_values(["p_window_clustered", "p_window_raw"])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "go_inversion.tsv", sep="\t", index=False)

    emit(f"{len(res)} terms with >={MIN_K} genes in the body")
    emit(f"  p_window_raw < 0.05:        {(res.p_window_raw < 0.05).sum()}")
    emit(f"  p_window_clustered < 0.05:  "
         f"{(res.p_window_clustered < 0.05).sum()}")
    emit(f"  FDR (clustered) < 0.10:     {(res.fdr_clustered < 0.10).sum()}")
    emit()
    emit("top 30 by the clustered window p (k_raw -> k_clustered shows how much "
         "of the\nsignal is tandem duplication):")
    emit(f"  {'cat':<19s} {'term':<52s} {'k':>3s} {'kc':>3s} {'null':>6s} "
         f"{'p_raw':>7s} {'p_clus':>7s} {'FDR':>7s}")
    for _, r in res.head(30).iterrows():
        # r["cat"], not r.cat -- the latter hits pandas' categorical accessor.
        emit(f"  {r['cat']:<19s} {r['term'][:52]:<52s} {r.k_raw:3d} "
             f"{r.k_clustered:3d} {r.null_mean_raw:6.2f} "
             f"{r.p_window_raw:7.4f} {r.p_window_clustered:7.4f} "
             f"{r.fdr_clustered:7.4f}")

    (OUT / "go_inversion.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/go_inversion.{{tsv,txt}}")


if __name__ == "__main__":
    main()
