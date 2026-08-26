#!/usr/bin/env python
"""Does *I. argentinus* sit closer to one chr2 arrangement than the other?

    .venv/bin/python -m illex.scripts.argentinus_equidistance

THE TEST
--------
If the inversion PREDATES the illecebrosus/argentinus split, the two arrangement
classes were already separate lineages when argentinus diverged, so argentinus
should be measurably closer to one of them. If it POSTDATES the split, AA and BB
lineages are exchangeable before the split and argentinus must be exactly
equidistant.

So the quantity is  dxy(AA, arg) - dxy(BB, arg)  across the inversion body, with
the collinear region as a built-in control where it must be zero by construction.

WHY THIS NEEDS NO KARYOTYPES AND NO GOOD ARGENTINUS COVERAGE
------------------------------------------------------------
Writing p for illecebrosus arrangement frequencies and q for argentinus,

    dxy(AA,arg) - dxy(BB,arg)
      = sum [p_AA(1-q) + (1-p_AA)q] - [p_BB(1-q) + (1-p_BB)q]
      = sum (p_AA - p_BB)(1 - 2q)

Three things follow, and together they are why this works where the earlier
"resequence argentinus" recommendation was wrong:

1. **Sites where AA and BB agree cancel exactly.** Only AA/BB-differentiated
   sites can create asymmetry, so the illecebrosus variants-only callset is
   sufficient on that side even though it omits invariant sites.
2. ~~**Reference mapping bias cannot fake a signal.**~~ **THIS CLAIM IS WRONG
   AND IT KILLS THE TEST — see the RESULT section below.** The argument was
   that mapping bias depresses q identically in both terms so a common shift
   cancels. It does not, because the reference genome is not neutral between
   the arrangements: inside the inversion the reference individual carries BB.
3. **No argentinus karyotypes are needed**, only allele frequencies, which is
   what genotype likelihoods estimate well at 0.6x coverage.

DATA
----
* argentinus q: ANGSD ``-doMaf`` over the 10 dedup BAMs (GL-based, no calling),
  with ``-doMajorMinor 3`` and a sites file carrying the VCF's REF/ALT, so q is
  the frequency of the VCF ALT **by construction**.

  A first pass used ``-doMajorMinor 4`` and then kept only sites where ANGSD's
  *inferred* minor happened to equal the VCF ALT — 36% of them. That is a
  selection on argentinus itself: the ALT is likelier to be the inferred minor
  exactly when argentinus carries it, which inflates q in the retained set.
  Forcing the alleles removes the selection.
* illecebrosus p_AA, p_BB: the chr2 callset (254 AA / 95 BB), which is
  well-covered — the low-coverage problem is argentinus's alone.

The earlier judgement that argentinus was "too sparse" came from counting
CALLED genotypes: 102 sites with >=8 of 10 samples across the whole region.
From the BAMs the same region has ~15,600 such sites, and 83% of positions have
at least one sample. The calling thresholds discarded ~150x the usable data.

RESULT: THE TEST DOES NOT WORK, AND THE REASON IS STRUCTURAL
------------------------------------------------------------
Run twice; the collinear control — which must show zero asymmetry — failed both
times (z = +4.4 with inferred alleles, z = +3.1 with forced alleles). The body
statistic also flips sign across allele-frequency classes (-0.34 at MAF
0.02-0.05, +0.19 at MAF > 0.10), so the aggregate is not estimating one thing.

The cause, measured directly: **the reference genome carries the BB
arrangement.** Taking p as the frequency of the NON-reference allele at
arrangement-diagnostic sites (MAF > 0.10):

    inversion body   p_AA = 0.5149   p_BB = 0.2165   difference +0.2984
    collinear ctl    p_AA = 0.3029   p_BB = 0.2874   difference +0.0156

So inside the inversion BB matches the reference and AA does not, while outside
it the two are equivalent. Argentinus reads are mapped to that reference, and
reference bias makes reads matching it map and call more readily — so argentinus
is pulled toward BB **precisely at the diagnostic sites inside the inversion,
and nowhere else.** That is the exact direction, magnitude and location of the
"signal", and it is an artifact.

Deeper argentinus sequencing would NOT fix this: the bias is in the reference,
not the coverage. What would fix it is mapping both species to a THIRD genome —
*I. coindetii* is assembled (GCA_977009265.1) and already aligned to
illecebrosus — so that neither arrangement is privileged. That is a real
project, not a re-run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(".tmp/argeq")
OUT = Path("results/illex")
MAFS = W / "arg.forced.mafs.gz"
BCFTOOLS = "/home/ssmall/bin/bcftools"
PLUGINS = "/home/ssmall/programs/bcftools-1.21/plugins"
KARYO = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
         "03_karyotype")

REGIONS = {
    "inversion body": (".tmp/illex_chr2/inv.vcf.gz", 60_500_000, 79_500_000),
    "collinear control": (".tmp/illex_chr2/ctl.vcf.gz", 10_000_000, 30_000_000),
}
MIN_IND = 3          # argentinus individuals with data at the site
BLOCK = 1_000_000    # jackknife block


def vcf_freqs(vcf: str, lo: int, hi: int) -> pd.DataFrame:
    """POS, REF, ALT, p_AA, p_BB from the illecebrosus callset."""
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
    m = out["AA"].merge(out["BB"], on=["pos", "ref", "alt"])
    return m


def arg_freqs(lo: int, hi: int) -> pd.DataFrame:
    """POS, minor allele, q from the ANGSD MAF table, for one region."""
    awk = (f'NR>1 && $2>={lo} && $2<{hi} '
           '{print $2"\\t"$4"\\t"$6"\\t"$7}')
    p1 = subprocess.Popen(["zcat", str(MAFS)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["awk", "-F", "\t", awk], stdin=p1.stdout,
                          stdout=subprocess.PIPE)
    p1.stdout.close()
    d = pd.read_csv(p2.stdout, sep="\t", header=None,
                    names=["pos", "minor", "q", "n_ind"])
    p2.wait()
    return d


def jackknife(pos, num, den, block=BLOCK):
    """Delete-one-block jackknife on sum(num)/sum(den)."""
    blk = pos // block
    ids = np.unique(blk)
    b = len(ids)
    full = num.sum() / den.sum()
    part = np.array([num[blk != i].sum() / den[blk != i].sum() for i in ids])
    m = part.mean()
    est = b * full - (b - 1) * m
    se = float(np.sqrt((b - 1) / b * ((part - m) ** 2).sum()))
    return b, float(est), se


def main() -> None:
    if not MAFS.exists():
        sys.exit(f"{MAFS} not found -- run .tmp/argeq/run_arg_maf.sh first")
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Argentinus equidistance test: dxy(AA,arg) vs dxy(BB,arg)")
    emit(f"argentinus q from ANGSD -doMaf (GL, 10 BAMs); min {MIN_IND} "
         "individuals per site")
    emit()

    res = {}
    for label, (vcf, lo, hi) in REGIONS.items():
        v = vcf_freqs(vcf, lo, hi)
        a = arg_freqs(lo, hi)
        a = a[a.n_ind >= MIN_IND]
        m = v.merge(a, on="pos")
        # With -doMajorMinor 3 the alleles come from the sites file, so this is
        # an assertion rather than a filter; a mismatch would mean the sites
        # file and the VCF disagree.
        ok = m.minor == m.alt
        if ok.mean() < 0.99:
            emit(f"  WARNING: only {100 * ok.mean():.1f}% allele-matched -- the "
                 "sites file did not take effect")
        m = m[ok]
        p_aa = m.p_AA.to_numpy()
        p_bb = m.p_BB.to_numpy()
        q = m.q.to_numpy()
        pos = m.pos.to_numpy()

        d_aa = p_aa * (1 - q) + (1 - p_aa) * q
        d_bb = p_bb * (1 - q) + (1 - p_bb) * q
        diff = d_aa - d_bb                       # == (p_aa-p_bb)(1-2q)
        wt = np.abs(p_aa - p_bb)

        emit(f"{'=' * 74}\n{label}  ({lo:,}-{hi:,})\n{'=' * 74}")
        emit(f"  VCF sites {len(v):,} | argentinus sites {len(a):,} | "
             f"used {len(m):,} ({100 * ok.mean():.1f}% allele-matched)")
        emit(f"  mean |p_AA - p_BB| = {wt.mean():.4f}   "
             f"mean q(arg) = {q.mean():.4f}")
        emit(f"  dxy(AA,arg) = {d_aa.mean():.6f}   "
             f"dxy(BB,arg) = {d_bb.mean():.6f}")

        # Frequency dependence. AA carries ~366 chromosomes against BB's ~146,
        # so at low frequency a variant is likelier to be SEEN in AA, which can
        # induce an apparent lean. If the statistic is flat across frequency the
        # ascertainment is not driving it.
        comb = (p_aa * 366 + p_bb * 146) / 512
        maf = np.minimum(comb, 1 - comb)
        emit(f"  {'MAF bin':<12s} {'n':>9s} {'standardised':>13s}")
        for lo, hi, lab in ((0, .02, "<0.02"), (.02, .05, "0.02-0.05"),
                            (.05, .10, "0.05-0.10"), (.10, .5, ">0.10")):
            k = (maf >= lo) & (maf < hi)
            if k.sum() > 100 and np.abs(p_aa - p_bb)[k].sum() > 0:
                emit(f"  {lab:<12s} {k.sum():9,} "
                     f"{diff[k].sum() / np.abs(p_aa - p_bb)[k].sum():+13.4f}")
        emit()

        b, est, se = jackknife(pos, diff, np.ones_like(diff))
        # Standardised: asymmetry relative to how differentiated the sites are.
        b2, est2, se2 = jackknife(pos, diff, wt)
        emit(f"  difference   = {est:+.6f} +/- {se:.6f}   "
             f"(z = {est / se if se else float('nan'):+.2f}, {b} blocks)")
        emit(f"  standardised = {est2:+.4f} +/- {se2:.4f}   "
             f"(z = {est2 / se2 if se2 else float('nan'):+.2f})")
        emit("    standardised = sum(d_AA-d_BB)/sum|p_AA-p_BB|, in [-1,1]; "
             "negative = argentinus\n    leans toward the AA (inverted) "
             "allele, positive = toward BB.")
        emit()
        res[label] = {"n": int(len(m)), "d_aa": float(d_aa.mean()),
                      "d_bb": float(d_bb.mean()), "diff": est, "diff_se": se,
                      "std": est2, "std_se": se2}

    emit(f"{'=' * 74}\nVERDICT\n{'=' * 74}")
    c = res["collinear control"]
    i = res["inversion body"]
    emit(f"  control  standardised {c['std']:+.4f} +/- {c['std_se']:.4f}  "
         f"(z {c['std'] / c['std_se']:+.2f})  <- must be ~0")
    emit(f"  BODY     standardised {i['std']:+.4f} +/- {i['std_se']:.4f}  "
         f"(z {i['std'] / i['std_se']:+.2f})")
    dd = i["std"] - c["std"]
    dse = float(np.hypot(i["std_se"], c["std_se"]))
    emit(f"  BODY - CONTROL {dd:+.4f} +/- {dse:.4f}  (z {dd / dse:+.2f})")
    emit()
    emit("  NOTE ON THE ERROR BARS. Inside a non-recombining inversion the 1 Mb")
    emit("  jackknife blocks are NOT independent -- every block shares the same")
    emit("  arrangement genealogy -- so the body's SE is optimistic by an")
    emit("  unknown factor. The control's blocks DO recombine and its SE is")
    emit("  sound. Treat the body's z as an upper bound on the evidence.")
    emit()
    if abs(c["std"] / c["std_se"]) > 3:
        emit("  CONTROL FAILS: the collinear region shows asymmetry it cannot")
        emit("  legitimately have, so something is wrong upstream (allele")
        emit("  matching, mapping bias that differs between arrangements, or")
        emit("  residual structure). Do not interpret the body.")
    elif abs(i["std"] / i["std_se"]) > 3:
        lean = "AA (inverted)" if i["std"] < 0 else "BB (standard)"
        emit(f"  ASYMMETRIC: argentinus leans toward {lean}. That requires the")
        emit("  arrangements to have been distinct lineages when argentinus")
        emit("  diverged -> the inversion PREDATES the split.")
    else:
        emit("  EQUIDISTANT within error: no evidence the arrangements were")
        emit("  distinct lineages at the split -> consistent with the inversion")
        emit("  POSTDATING the illecebrosus/argentinus split. Note this is an")
        emit("  upper bound on t_inv only if the test has the power to detect")
        emit("  the expected asymmetry -- see the SE against the body's own")
        emit("  mean |p_AA-p_BB|.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "argentinus_equidistance.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/argentinus_equidistance.txt")


if __name__ == "__main__":
    main()
