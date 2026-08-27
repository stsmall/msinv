#!/usr/bin/env python
"""Which arrangement carries the ANCESTRAL allele at diagnostic sites?

    .venv/bin/python -m illex.scripts.polarization_check

THE QUESTION
------------
The project records **AA = derived (inverted), BB = ancestral** as
**[ESTABLISHED]**. Section 8.13 put that in doubt: the illecebrosus reference is
inverted relative to coindetii across the region, and the reference leans
BB-like inside it, which together would make **BB** derived.

This settles it without needing to infer the reference individual's karyotype:
at arrangement-diagnostic sites, ask which arrangement's majority allele matches
the ancestral state. The derived arrangement should carry derived alleles more
often.

TWO POLARIZATIONS, BECAUSE ONE OF THEM IS CIRCULAR HERE
--------------------------------------------------------
1. **est-sfs** (`polarize/illecebrosus_input/*/estsfs_output/merged/`), which
   models the ingroup SFS together with the outgroup.

   **This is circular for exactly this question.** est-sfs uses ingroup allele
   frequency as evidence — rare alleles are likelier derived. At
   arrangement-diagnostic sites the pooled frequency is set by the arrangement
   frequencies, so the AA-carried allele sits near 0.626 and the BB-carried
   allele near 0.374. est-sfs will therefore tend to call the AA allele
   ancestral *because AA is commoner*, biasing toward "AA is ancestral" — which
   is the very conclusion under test. Restricting to high `anc_prob` reduces but
   does not remove this.

2. **Parsimony from coindetii**, which uses only the outgroup: ancestral = the
   coindetii base, taken from the AnchorWave alignment (a SNP record means
   coindetii differs from the illecebrosus reference; callable-but-no-record
   means it matches). **No ingroup frequency enters**, so it is not circular.
   Its weakness is homoplasy, at roughly the ~1-3% divergence rate.

If the two agree the answer is solid. If they disagree, **parsimony is the one
to trust for this question**, because est-sfs's bias points in a known direction.

The collinear control is carried through as a check: AA and BB are exchangeable
there, so whatever few "diagnostic" sites exist should split ~50/50 under either
method.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/illex")
BCFTOOLS = "/home/ssmall/bin/bcftools"
PLUGINS = "/home/ssmall/programs/bcftools-1.21/plugins"
KARYO = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
         "03_karyotype")
ESTSFS = ("/sietch_colab/data_share/illex/popgen_data/polarize/"
          "illecebrosus_input/gatk_vcfs/estsfs_output/merged/"
          "ancestral_probs.tsv.gz")
_AW = ("/sietch_colab/data_share/illex/alignments/anchorwave_dir/"
       "ill_coin_alignment/coindetti_vcf")
COIN_SNPS = f"{_AW}/2.snps.vcf.gz"
COIN_CALLABLE = f"{_AW}/2.callable.bed"

REGIONS = {
    "inversion body": (".tmp/illex_chr2/inv.vcf.gz", 60_500_000, 79_500_000),
    "collinear control": (".tmp/illex_chr2/ctl.vcf.gz", 10_000_000, 30_000_000),
}
DIAG = [0.3, 0.5, 0.7]        # |p_AA - p_BB| thresholds
MIN_ANC_PROB = 0.95


def vcf_freqs(vcf: str, lo: int, hi: int) -> pd.DataFrame:
    env = dict(os.environ, BCFTOOLS_PLUGINS=PLUGINS)
    out = {}
    for pop in ("AA", "BB"):
        p1 = subprocess.Popen(
            [BCFTOOLS, "view", "-S", f"{KARYO}/{pop}_samples.txt",
             "--force-samples", "-m2", "-M2", "-v", "snps",
             "-r", f"2:{lo}-{hi}", vcf, "-Ou"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        p2 = subprocess.Popen(
            [BCFTOOLS, "+fill-tags", "-Ou", "--", "-t", "AN,AC"],
            stdin=p1.stdout, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env)
        p3 = subprocess.Popen(
            [BCFTOOLS, "query", "-f", "%POS\t%REF\t%ALT\t%AC\t%AN\n"],
            stdin=p2.stdout, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env)
        p1.stdout.close()
        p2.stdout.close()
        d = pd.read_csv(p3.stdout, sep="\t", header=None,
                        names=["pos", "ref", "alt", "ac", "an"])
        p3.wait()
        d = d[d.an > 0]
        out[pop] = d.assign(**{f"p_{pop}": d.ac / d.an})[
            ["pos", "ref", "alt", f"p_{pop}"]]
    return out["AA"].merge(out["BB"], on=["pos", "ref", "alt"])


def estsfs(lo: int, hi: int) -> pd.DataFrame:
    awk = f'$1=="2" && $2>={lo} && $2<{hi} {{print $2"\\t"$6"\\t"$7}}'
    p1 = subprocess.Popen(["zcat", ESTSFS], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["awk", "-F", "\t", awk], stdin=p1.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    d = pd.read_csv(p2.stdout, sep="\t", header=None,
                    names=["pos", "anc", "anc_prob"])
    p2.wait()
    return d


def coin_parsimony(lo: int, hi: int, pos_wanted: np.ndarray) -> pd.DataFrame:
    """Ancestral = the coindetii base, from the AnchorWave alignment."""
    # positions where coindetii differs from the illecebrosus reference
    p = subprocess.run(
        [BCFTOOLS, "query", "-r", f"2:{lo}-{hi}", "-f", "%POS\t%ALT\n",
         COIN_SNPS], capture_output=True, text=True, check=True)
    snp = pd.read_csv(pd.io.common.StringIO(p.stdout), sep="\t", header=None,
                      names=["pos", "coin"]) if p.stdout else \
        pd.DataFrame(columns=["pos", "coin"])
    # callable-but-no-SNP means coindetii matches the reference
    n = hi - lo
    callable_ = np.zeros(n, dtype=bool)
    a = subprocess.run(
        ["awk", f'$1=="2" && $3>{lo} && $2<{hi} {{print $2"\\t"$3}}',
         COIN_CALLABLE], capture_output=True, text=True, check=True)
    for line in a.stdout.splitlines():
        s, e = line.split()
        i0, i1 = max(int(s), lo) - lo, min(int(e), hi) - lo
        if i1 > i0:
            callable_[i0:i1] = True
    return snp, callable_


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Which arrangement carries the ANCESTRAL allele at "
         "arrangement-diagnostic sites?")
    emit("Two polarizations: est-sfs (circular here -- see the module note) and")
    emit("coindetii parsimony (outgroup only, not circular).")
    emit()

    for label, (vcf, lo, hi) in REGIONS.items():
        v = vcf_freqs(vcf, lo, hi)
        e = estsfs(lo, hi)
        snp, callable_ = coin_parsimony(lo, hi, v.pos.to_numpy())
        m = v.merge(e, on="pos", how="left")
        m = m.merge(snp, on="pos", how="left")
        idx = m.pos.to_numpy() - 1 - lo
        inrange = (idx >= 0) & (idx < hi - lo)
        m["coin_callable"] = False
        m.loc[inrange, "coin_callable"] = callable_[idx[inrange]]
        # coindetii base: its ALT where a SNP exists, else the reference base
        m["coin_base"] = np.where(m.coin.notna(), m.coin,
                                  np.where(m.coin_callable, m.ref, None))

        emit(f"{'=' * 76}\n{label}  ({lo:,}-{hi:,})   {len(m):,} sites\n"
             f"{'=' * 76}")
        emit(f"  {'|p_AA-p_BB|':>11s} {'n':>8s}   "
             f"{'est-sfs: AA anc':>16s} {'parsimony: AA anc':>18s}")
        for thr in DIAG:
            d = m[np.abs(m.p_AA - m.p_BB) >= thr].copy()
            if len(d) < 20:
                emit(f"  {'>= ' + str(thr):>11s} {len(d):>8,}   "
                     "(too few sites)")
                continue
            # majority allele of each arrangement
            d["maj_AA"] = np.where(d.p_AA > 0.5, d.alt, d.ref)
            d["maj_BB"] = np.where(d.p_BB > 0.5, d.alt, d.ref)

            ee = d[(d.anc.notna()) & (d.anc_prob >= MIN_ANC_PROB)]
            e_aa = (ee.maj_AA == ee.anc).sum()
            e_bb = (ee.maj_BB == ee.anc).sum()
            e_txt = (f"{100 * e_aa / (e_aa + e_bb):5.1f}%  (n={e_aa + e_bb:,})"
                     if (e_aa + e_bb) else "   n/a")

            pp = d[d.coin_base.notna()]
            p_aa = (pp.maj_AA == pp.coin_base).sum()
            p_bb = (pp.maj_BB == pp.coin_base).sum()
            p_txt = (f"{100 * p_aa / (p_aa + p_bb):5.1f}%  (n={p_aa + p_bb:,})"
                     if (p_aa + p_bb) else "   n/a")
            emit(f"  {'>= ' + str(thr):>11s} {len(d):>8,}   {e_txt:>16s} "
                 f"{p_txt:>18s}")
        emit()

    emit(f"{'=' * 76}\nREADING IT\n{'=' * 76}")
    emit("  'AA anc' is the share of diagnostic sites where the AA majority")
    emit("  allele equals the ancestral base. >50% means AA carries ancestral")
    emit("  states more often, i.e. AA is the ANCESTRAL arrangement and BB is")
    emit("  derived -- the OPPOSITE of the recorded polarization. <50% supports")
    emit("  the recorded 'AA = derived'.")
    emit("  In the collinear control both columns must sit at ~50%; if they do")
    emit("  not, the estimator is biased and neither region can be read.")
    emit("  Where the two methods disagree, trust PARSIMONY: est-sfs's ingroup")
    emit("  frequency term pushes it toward calling the commoner arrangement's")
    emit("  allele ancestral, and AA is the commoner arrangement (0.626).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "polarization_check.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/polarization_check.txt")


if __name__ == "__main__":
    main()
