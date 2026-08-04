#!/usr/bin/env python
"""Windowed per-arrangement pi and dxy across the inversion and control.

FALSIFICATION CHECK: gene flux via double crossover has phi(x) = 0 at the
breakpoints and flat-maximal in the interior, so between-arrangement dxy
should be HIGHEST near the breakpoints and LOWEST mid-inversion. No dip =>
the flux interpretation in the design is wrong.

Run with the pg_gpu env:
  CUDA_VISIBLE_DEVICES=0 /home/ssmall/miniforge3/envs/varbuddy-pggpu/bin/python \
      illex/scripts/empirical_windowed.py

Implementation notes (found while running this):
  - `windowed_analysis` output columns are `start`/`end` (not
    `window_start`/`window_stop` as an earlier draft of this script assumed).
  - When `statistics=['pi', 'dxy', 'fst']` is requested together with
    `populations=['AA', 'BB']`, pg_gpu's "mixed single+two-pop" code path
    (windowed_analysis.py ~line 1281) computes the single-population `pi`
    stat using ONLY `populations[0]`. A single `pi` column therefore silently
    means pi(AA), and pi(BB) is never computed. Confirmed numerically: the
    control-region mean of that column (0.0043237) matches the reference
    pi(AA)=0.004324 exactly, not pi(BB)=0.004374. This script calls `pi`
    separately per population to get both pi_AA and pi_BB.
  - The nominal inversion breakpoints (60,040,617 / 79,995,597) sit inside
    the outermost 500kb windows (60.0-60.5 Mb and 79.5-80.0 Mb), but those
    two windows have Fst ~0.003-0.006 -- indistinguishable from the control
    region's Fst (~0.0035) -- while every other inversion window has
    Fst 0.26-0.51. That is a clean bimodal split: the outermost windows are
    collinear/undifferentiated flanking sequence, not part of the
    differentiated inversion body. Treating them as "near-breakpoint
    inversion" would bias any edge/core comparison, so the empirically
    differentiated extent (Fst > FST_DIFF_CUTOFF) is used to define
    "core" vs "edge" instead of the nominal 60-80 Mb span.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd
from pg_gpu import HaplotypeMatrix, windowed_analysis

T = Path(".tmp/illex_chr2")
OUT = Path("results/illex")
WINDOW = 500_000
REGIONS = {
    "inversion": ("2:60000000-80000000", T / "inv.vcf.gz"),
    "control": ("2:10000000-30000000", T / "ctl.vcf.gz"),
}
# Fst cutoff separating genuinely differentiated inversion windows from
# undifferentiated flanking sequence. The observed distribution is sharply
# bimodal (two windows at 0.003-0.006, all others at 0.26-0.51), so any
# cutoff in roughly [0.02, 0.25] gives an identical partition.
FST_DIFF_CUTOFF = 0.15
# Relative-position thresholds (fraction of the half-width of the
# empirically differentiated extent) used to define "core" vs "edge".
CORE_REL = 0.35
EDGE_REL = 0.75


def load(vcf: Path, region: str) -> HaplotypeMatrix:
    h = HaplotypeMatrix.from_vcf(str(vcf), region=region)
    h.load_pop_file(str(T / "pops.tsv"))
    return h


def windowed_region(label: str, region: str, vcf: Path) -> pd.DataFrame:
    h = load(vcf, region)

    # fst + dxy together (two-population statistics, both require populations=)
    df = windowed_analysis(
        h, window_size=WINDOW, step_size=WINDOW,
        statistics=["fst", "dxy"],
        populations=["AA", "BB"],
        missing_data="include",
    )

    # pi per population, computed separately: passing populations=['AA','BB']
    # together with 'pi' silently collapses to pi(populations[0]) only (see
    # module docstring above), so pi_BB would otherwise never be computed.
    pi_aa = windowed_analysis(
        h, window_size=WINDOW, step_size=WINDOW,
        statistics=["pi"], populations=["AA"], missing_data="include",
    )
    pi_bb = windowed_analysis(
        h, window_size=WINDOW, step_size=WINDOW,
        statistics=["pi"], populations=["BB"], missing_data="include",
    )

    df = df.merge(
        pi_aa[["window_id", "pi"]].rename(columns={"pi": "pi_AA"}),
        on="window_id", how="left",
    )
    df = df.merge(
        pi_bb[["window_id", "pi"]].rename(columns={"pi": "pi_BB"}),
        on="window_id", how="left",
    )
    df.insert(0, "region", label)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, (region, vcf) in REGIONS.items():
        df = windowed_region(label, region, vcf)
        frames.append(df)
        print(f"{label}: {len(df)} windows", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"start": "window_start", "end": "window_stop"})
    csv = OUT / "empirical_windowed.csv"
    cols = ["region", "window_start", "window_stop", "n_variants",
            "pi_AA", "pi_BB", "dxy", "fst"]
    out = out[[c for c in cols if c in out.columns]
              + [c for c in out.columns if c not in cols]]
    out.to_csv(csv, index=False)

    # --- verdict ---
    inv = out[out.region == "inversion"].copy()
    inv["pi_mean"] = 0.5 * (inv.pi_AA + inv.pi_BB)
    inv["dxy_over_pi"] = inv.dxy / inv.pi_mean

    # Empirically differentiated inversion body: exclude undifferentiated
    # flanking windows (Fst below cutoff -- see module docstring).
    diff = inv[inv.fst > FST_DIFF_CUTOFF].copy()
    excluded = inv[inv.fst <= FST_DIFF_CUTOFF]
    lo, hi = diff.window_start.min(), diff.window_stop.max()
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    diff["rel"] = (0.5 * (diff.window_start + diff.window_stop) - mid).abs() / half

    core = diff[diff.rel < CORE_REL]
    edge = diff[diff.rel > EDGE_REL]

    raw_ratio = float(edge.dxy.mean() / core.dxy.mean())
    norm_ratio = float(edge.dxy_over_pi.mean() / core.dxy_over_pi.mean())
    dip_raw = raw_ratio > 1.0
    dip_norm = norm_ratio > 1.0

    whole_region_ratio = 0.002379 / ((0.001288 + 0.001732) / 2)  # reference values

    lines = [
        "FALSIFICATION CHECK: flux predicts dxy higher at breakpoints than mid-inversion",
        "",
        f"Empirically differentiated inversion body: Fst > {FST_DIFF_CUTOFF} "
        f"=> {len(diff)}/{len(inv)} windows ({int(lo):,}-{int(hi):,}); "
        f"{len(excluded)} outermost window(s) excluded as undifferentiated "
        f"flanking sequence (Fst {excluded.fst.min():.4f}-{excluded.fst.max():.4f}, "
        "control-like).",
        "",
        "Raw dxy (confounded by the diversity gradient -- pi also declines",
        "toward the breakpoints, so a raw dxy gradient cannot on its own",
        "distinguish a flux effect from a general diversity/callability",
        "gradient):",
        f"  mean raw dxy, core (|rel pos| < {CORE_REL}):  {core.dxy.mean():.6f}  (n={len(core)})",
        f"  mean raw dxy, edge (|rel pos| > {EDGE_REL}):  {edge.dxy.mean():.6f}  (n={len(edge)})",
        f"  edge/core raw-dxy ratio: {raw_ratio:.3f}  -> central dip {'PRESENT' if dip_raw else 'ABSENT'}",
        "",
        "Diversity-controlled ratio dxy / mean(pi_AA, pi_BB) (isolates the",
        "flux-specific spatial signature from the diversity gradient):",
        f"  mean dxy/pi, core:  {core.dxy_over_pi.mean():.3f}  (n={len(core)})",
        f"  mean dxy/pi, edge:  {edge.dxy_over_pi.mean():.3f}  (n={len(edge)})",
        f"  edge/core dxy/pi ratio: {norm_ratio:.3f}  -> central dip {'PRESENT' if dip_norm else 'ABSENT'}",
        "",
        f"dxy/pi across the full differentiated body: mean={diff.dxy_over_pi.mean():.3f}, "
        f"range=[{diff.dxy_over_pi.min():.3f}, {diff.dxy_over_pi.max():.3f}]; "
        f"whole-region reference dxy/mean(piAA,piBB) = {whole_region_ratio:.3f}.",
        "NOTE ON NORMALISATION: the dxy/pi figures above all use "
        "dxy/mean(pi_AA,pi_BB), the correct baseline for THIS spatial "
        "(core-vs-edge) comparison. That is a DIFFERENT normalisation "
        "from the package's fitted target dxy/pi_I=1.846 "
        "(see illex/empirical.py), which is dxy divided by pi_AA "
        "SPECIFICALLY (the inverted arrangement's own pi), not by the "
        "mean of both arrangements -- do not read the two ratios as the "
        "same quantity.",
        "",
        f"VERDICT: raw-dxy central dip {'PRESENT' if dip_raw else 'ABSENT'}; "
        f"diversity-controlled (dxy/pi) central dip {'PRESENT (nominal direction)' if dip_norm else 'ABSENT'}.",
    ]
    if not dip_raw:
        if dip_norm:
            lines.append(
                "  The diversity-controlled ratio points in the direction flux predicts, "
                "but the raw dxy signal does not, and the magnitude of the "
                "controlled effect is small relative to how flat dxy/pi is across "
                "the whole differentiated body (see range above)."
            )
            lines.append(
                "  Interpretation: dxy/pi is close to spatially uniform across the "
                "inversion and close to the whole-region value -- consistent with a "
                "roughly uniform property of the two arrangements' shared history "
                "(e.g. age/origin), not with a position-dependent double-crossover "
                "flux signature, which would require a clear gradient. The spatial "
                "evidence for flux here is at best marginal and does not by itself "
                "support flux as the explanation for the reduced dxy/pi observed "
                "for the inversion."
            )
        else:
            lines.append(
                "  Neither the raw nor the diversity-controlled statistic shows the "
                "predicted dip."
            )
        lines.append(
            "  ACTION: do not treat this as confirming the flux interpretation. "
            "Phases B-D should not rely solely on the spatial dip as evidence; "
            "revisit/re-examine the design before committing to flux as the "
            "explanation."
        )
    else:
        lines.append("  Both statistics agree: central dip present, flux interpretation supported.")

    txt = "\n".join(lines)
    (OUT / "empirical_windowed_verdict.txt").write_text(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
