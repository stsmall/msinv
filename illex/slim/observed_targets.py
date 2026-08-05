#!/usr/bin/env python
"""Build the observed statistic vector the ABC matches against.

This must produce EXACTLY the keys in config.STAT_NAMES, computed the same way as
summarize.py does on simulations, or the ABC silently compares mismatched
quantities. Two places that is easy to get wrong, both handled here:

1. **Region.** The empirical target is the *empirically differentiated* inversion
   body (38 windows with Fst > 0.15), not the nominal 60-80 Mb span. The nominal
   span's outermost 500 kb windows are collinear flank (Fst 0.003-0.006 against
   0.26-0.51 elsewhere) and including them biases every statistic toward the
   null. See NOTES sec 4.2.
2. **SFS folding and size.** Simulated SFS shapes are folded, branch-mode, over
   SFS_PROJ haplotypes. The observed spectra must be folded and projected to the
   same size, and normalized to sum 1.

The two ratios and p_final come from already-committed measurements; the
per-arrangement SFS shapes require reading genotypes and are computed here with
pg_gpu.

Run with the pg_gpu environment (NOT the msinv venv -- pg_gpu is not installed
there):
  /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \\
      -m illex.slim.observed_targets --out results/illex/abc_observed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C

KARYO_DIR = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
             "03_karyotype")
VCF = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/00_callset/"
       "filtered/2/variants_filt.vcf.gz")
WINDOWED_CSV = "results/illex/empirical_windowed.csv"
FST_DIFF_CUTOFF = 0.15


def differentiated_extent() -> tuple[int, int]:
    """(start, stop) of the empirically differentiated inversion body."""
    import pandas as pd
    df = pd.read_csv(WINDOWED_CSV)
    inv = df[(df.region == "inversion") & (df.fst > FST_DIFF_CUTOFF)]
    if inv.empty:
        raise SystemExit(f"no differentiated windows in {WINDOWED_CSV}")
    return int(inv.window_start.min()), int(inv.window_stop.max())


def project_folded(counts: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
    """Hypergeometric projection of an UNFOLDED count spectrum, then fold.

    Takes unfolded counts (index i = derived allele count) so no 1/i unfolding
    approximation is needed -- unlike the genome-wide folded spectrum used in the
    Beta-vs-Kingman test, here we have genotypes and can count directly.
    """
    from scipy.stats import hypergeom
    out = np.zeros(n_to + 1, dtype=float)
    kk = np.arange(0, n_to + 1)
    for i in range(1, n_from):
        c = counts[i]
        if c <= 0:
            continue
        out += c * hypergeom.pmf(kk, n_from, i, n_to)
    nb = n_to // 2
    fol = np.zeros(nb, dtype=float)
    for k in range(1, n_to):
        b = min(k, n_to - k)
        if 1 <= b <= nb:
            fol[b - 1] += out[k]
    tot = fol.sum()
    return fol / tot if tot > 0 else fol


def arrangement_sfs(start: int, stop: int) -> dict[str, np.ndarray]:
    """Folded, projected, normalized SFS within each arrangement."""
    from pg_gpu import HaplotypeMatrix

    region = f"2:{start}-{stop}"
    shapes = {}
    for tag, sample_file in (("i", "AA_samples.txt"), ("s", "BB_samples.txt")):
        names = [ln.strip() for ln in
                 open(f"{KARYO_DIR}/{sample_file}") if ln.strip()]
        h = HaplotypeMatrix.from_vcf(VCF, region=region, samples=names)
        # Derived-allele counts per site. AA/BB are homokaryotypes, so every
        # haplotype in this matrix is the corresponding arrangement.
        g = h.haplotypes
        g = g.get() if hasattr(g, "get") else np.asarray(g)
        g = np.where(g < 0, 0, g)                     # missing -> ancestral
        dac = g.sum(axis=0).astype(int)
        n_hap = g.shape[0]
        counts = np.bincount(dac, minlength=n_hap + 1).astype(float)
        counts[0] = 0.0
        if n_hap > 0:
            counts[n_hap] = 0.0                       # drop fixed
        shapes[tag] = project_folded(counts, n_hap, C.SFS_PROJ)
        print(f"  arrangement {tag}: {len(names)} samples, {n_hap} haplotypes, "
              f"{int(counts.sum()):,} segregating sites")
    return shapes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/illex/abc_observed.json"))
    ap.add_argument("--skip-sfs", action="store_true",
                    help="write only the ratios and p_final (no pg_gpu needed); "
                         "the resulting file cannot be used with the default "
                         "statistic vector")
    args = ap.parse_args()

    from illex.empirical import DXY_OVER_PI_I, PI_I_OVER_PI_S

    start, stop = differentiated_extent()
    print(f"differentiated inversion body: 2:{start:,}-{stop:,} "
          f"({(stop - start) / 1e6:.2f} Mb)")

    stats = {
        "pi_i_over_pi_s": float(PI_I_OVER_PI_S),
        "dxy_over_pi_i": float(DXY_OVER_PI_I),
        "p_final": float(C.P_INV_OBS),
    }

    if not args.skip_sfs:
        print("computing per-arrangement SFS shapes (pg_gpu)...")
        sh = arrangement_sfs(start, stop)
        for tag in ("i", "s"):
            for k in range(C.SFS_BINS):
                stats[f"sfs_{tag}_{k + 1}"] = float(sh[tag][k])

    # Absolute levels, per ACCESSIBLE bp, for the optional --use-absolute arm.
    # Carries a known ~1.31x calibration offset (NOTES sec 8.3) -- do not use
    # without a nuisance scale parameter.
    absolute = {
        "pi_i_abs": 0.001308 / C.ACC_FRAC_INV,
        "pi_s_abs": 0.001774 / C.ACC_FRAC_INV,
        "dxy_abs": 0.002455 / C.ACC_FRAC_INV,
        "_warning": ("per accessible bp; a ~1.31x calibration offset remains, "
                     "present in the collinear control too, so fit a nuisance "
                     "scale before using these"),
    }

    missing = [s for s in C.STAT_NAMES if s not in stats]
    payload = {
        "region": {"chrom": "2", "start": start, "stop": stop,
                   "definition": f"windows with Fst > {FST_DIFF_CUTOFF}"},
        "stats": stats,
        "absolute_per_accessible_bp": absolute,
        "missing_from_default_vector": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} with {len(stats)} statistics")
    if missing:
        print(f"WARNING: {len(missing)} statistics missing "
              f"({missing[:4]}...); rerun without --skip-sfs before ABC")


if __name__ == "__main__":
    main()
