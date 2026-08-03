#!/usr/bin/env python
"""Per-arrangement pi and between-arrangement dxy for the Illex chr2 inversion.

Question: is within-arrangement diversity consistent with a NEUTRAL inversion at
equilibrium? At equilibrium each arrangement behaves as a subpopulation of
~2*Ne*p chromosomes, so

    pi_A / pi_B  ~=  p_A / p_B  =  0.626 / 0.374  =  1.674

i.e. the COMMON arrangement should be the MORE diverse one. The karyotype-log
region heterozygosities (homA 0.037 < homB 0.0605) point the other way. Those
are per-ascertained-SNP though, so this script computes proper pi/dxy instead.

No chr2 accessibility mask exists (acc_aut.bed / genome.bed both skip chr2), so
per-bp values use the full region span as denominator and are LOWER BOUNDS.
The RATIOS are denominator-free and are the actual result.
"""

import os
import sys

# Pin GPU before importing pg_gpu/cupy (shared box; GPU 0 was idle).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd

from pg_gpu import HaplotypeMatrix, diversity, divergence

T = "/sietch_colab/ssmall/projects/msinv_dir/inversion_sims/files/.tmp/illex_chr2"
REGIONS = {
    "inversion": ("2:60000000-80000000", f"{T}/inv.vcf.gz", 60_000_000, 80_000_000),
    "control":   ("2:10000000-30000000", f"{T}/ctl.vcf.gz", 10_000_000, 30_000_000),
}
P_A, P_B = 0.626, 0.374          # A = derived/inverted per polarization
EQ_RATIO = P_A / P_B             # neutral equilibrium expectation for pi_A/pi_B
MU = 3e-9
NE_CONST = 775_000               # pi/(4 mu) from genome-wide pi = 0.0093


def analyse(label, region, vcf, start, end):
    print(f"\n{'='*72}\n{label}  {region}\n{'='*72}", flush=True)
    h = HaplotypeMatrix.from_vcf(vcf, region=region)
    h.load_pop_file(f"{T}/pops.tsv")
    if getattr(h, "device", "GPU") == "CPU":
        try:
            h.transfer_to_gpu()
        except AttributeError:
            pass  # StreamingHaplotypeMatrix manages its own residency

    span = end - start
    nv = h.num_variants
    print(f"  variants={nv:,}  haplotypes={h.num_haplotypes}  span={span:,} bp")
    print(f"  sample_sets={{k: len(v) for k, v in h.sample_sets.items()}}"
          f" -> { {k: len(v) for k, v in h.sample_sets.items()} }")

    out = {"region": label, "n_variants": nv, "span_bp": span}

    # pi per arrangement; span_normalize=True -> per-bp over region span
    for pop in ("AA", "BB"):
        out[f"pi_{pop}"] = float(diversity.pi(h, population=pop))
    out["pi_all"] = float(diversity.pi(h))
    out["dxy_AA_BB"] = float(divergence.dxy(h, "AA", "BB"))
    out["fst_AA_BB"] = float(divergence.fst(h, "AA", "BB"))

    # per-variant-site versions (denominator-free comparison to the
    # per-ascertained-SNP heterozygosities in chr2_karyo.log)
    for k in ("pi_AA", "pi_BB", "pi_all", "dxy_AA_BB"):
        out[f"{k}_pervar"] = out[k] * span / nv if nv else float("nan")

    out["ratio_piAA_piBB"] = out["pi_AA"] / out["pi_BB"] if out["pi_BB"] else float("nan")
    out["ratio_dxy_piAA"] = out["dxy_AA_BB"] / out["pi_AA"] if out["pi_AA"] else float("nan")
    out["ratio_dxy_piBB"] = out["dxy_AA_BB"] / out["pi_BB"] if out["pi_BB"] else float("nan")

    print(f"  pi(AA)      = {out['pi_AA']:.6f}   (per-variant {out['pi_AA_pervar']:.4f})")
    print(f"  pi(BB)      = {out['pi_BB']:.6f}   (per-variant {out['pi_BB_pervar']:.4f})")
    print(f"  pi(all)     = {out['pi_all']:.6f}")
    print(f"  dxy(AA,BB)  = {out['dxy_AA_BB']:.6f}   (per-variant {out['dxy_AA_BB_pervar']:.4f})")
    print(f"  Fst(AA,BB)  = {out['fst_AA_BB']:.4f}")
    print(f"  pi_AA/pi_BB = {out['ratio_piAA_piBB']:.4f}   "
          f"[neutral equilibrium expectation {EQ_RATIO:.3f}]")
    print(f"  dxy/pi_AA   = {out['ratio_dxy_piAA']:.4f}    dxy/pi_BB = {out['ratio_dxy_piBB']:.4f}")
    return h, out


def age_from_ratio(ratio):
    """Invert pi_A/pi_A_eq = 1 - exp(-t/(2*Ne*p_A)) for t_inv.

    Uses the observed pi_A/pi_B ratio relative to the equilibrium p_A/p_B,
    attributing the whole shortfall to the derived class not yet equilibrated.
    """
    frac = ratio / EQ_RATIO
    if not (0 < frac < 1):
        return None, frac
    tau = 2 * NE_CONST * P_A
    return -tau * np.log(1 - frac), frac


def main():
    rows = []
    for label, (region, vcf, s, e) in REGIONS.items():
        if not os.path.exists(vcf):
            sys.exit(f"missing {vcf} -- run extract.sh first")
        _, out = analyse(label, region, vcf, s, e)
        rows.append(out)

    df = pd.DataFrame(rows)
    csv = f"{T}/per_arrangement_stats.csv"
    df.to_csv(csv, index=False)

    print(f"\n{'='*72}\nINTERPRETATION\n{'='*72}")
    inv = df[df.region == "inversion"].iloc[0]
    r = inv["ratio_piAA_piBB"]
    print(f"  observed pi_AA/pi_BB (inversion) = {r:.4f}")
    print(f"  neutral equilibrium expectation  = {EQ_RATIO:.4f}")
    if r < EQ_RATIO:
        t, frac = age_from_ratio(r)
        print(f"  -> derived class is UNDER-diverse by {EQ_RATIO / r:.2f}x")
        if t is not None:
            print(f"  -> non-equilibrium t_inv ~= {t:,.0f} generations "
                  f"({t/1e6:.2f} My at 1 gen/yr)")
        else:
            print(f"  -> shortfall fraction {frac:.3f} outside (0,1); "
                  "cannot invert for t_inv")
        print(f"  -> compare neutral EXPECTED age at p={P_A}: "
              f"{-4*NE_CONST*(P_A/(1-P_A))*np.log(P_A):,.0f} generations")
    else:
        print("  -> consistent with (or exceeding) neutral equilibrium; "
              "the frequency/diversity tension does NOT reproduce")
    print(f"\n  wrote {csv}")


if __name__ == "__main__":
    main()
