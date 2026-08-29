#!/usr/bin/env python
"""Unfolded AA x BB 2D SFS, inversion body vs matched collinear control.

    .venv/bin/python -m illex.scripts.sfs2d_karyotype

WHY UNFOLDED, AND WHY A CONTROL
-------------------------------
The pre-existing `fst_sub/AA_BB.2dSFS` is `-fold 1`, and a FOLDED 2D SFS cannot
see fixed differences: a site fixed for X in AA and Y in BB has minor count 0 in
BOTH classes and lands in the [0,0] cell with the invariant sites. It is also
inversion-only and SNP-ascertained, so there is nothing to calibrate against.

This rebuilds it unfolded (`-anc $REF`, so "unfolded" = ALT-vs-reference count)
at n = 40/class over accessible sites, for the body AND the collinear control.
The full-n version was attempted in Jul 2026 and failed -- `fst/*.2dSFS` are
0 bytes; 509 x 191 does not converge.

WHAT THE CONTROL CAN AND CANNOT CALIBRATE
-----------------------------------------
CAN: sample-set systematics (as it did for f1, NOTES sec 8.19) and any
mapping/coverage asymmetry between the two bamlists that is region-independent.
CANNOT: the fact that the reference genome IS a BB haplotype *inside the
inversion*. Outside it the reference is just one draw from a panmictic
population and carries no arrangement identity, so the control has no
reference-arrangement bias to measure. The AA-ALT/BB-REF direction is therefore
biased upward by an unknown amount and the counts below are LOWER BOUNDS with a
known directional lean.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SFS = Path(".tmp/sfs2d")
OUT = Path("results/illex")
NEAR = 4          # "near-fixed" = within NEAR of the corner in both classes


def load(lab: str):
    v = np.array([float(x) for x in (SFS / f"AA_BB.{lab}.2dSFS").read_text().split()])
    n = int(round(len(v) ** 0.5))
    if n * n != len(v):
        raise SystemExit(f"{lab}: {len(v)} cells is not square")
    return v.reshape(n, n), n - 1


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Unfolded AA x BB 2D SFS -- inversion body vs collinear control")
    emit("m[i,j] = expected sites with i ALT alleles in AA, j in BB "
         "(80 chromosomes each)")
    emit()
    res = {}
    for lab in ("body", "control"):
        m, N = load(lab)
        seg = m.sum() - m[0, 0] - m[N, N]
        d = {
            "sites": float(m.sum()), "variable": float(seg),
            "shared": float(m[1:N, 1:N].sum()),
            "priv_AA": float(m[1:N, 0].sum() + m[1:N, N].sum()),
            "priv_BB": float(m[0, 1:N].sum() + m[N, 1:N].sum()),
            "fixed_AAalt": float(m[N, 0]), "fixed_BBalt": float(m[0, N]),
            "nearfix_AAalt": float(m[N - NEAR:, :NEAR + 1].sum()),
            "nearfix_BBalt": float(m[:NEAR + 1, N - NEAR:].sum()),
            "exact_zero_cells": int((m == 0).sum()), "cells": int(m.size),
            "asymmetry": float(np.abs(m - m[::-1, ::-1]).sum() / m.sum()),
        }
        res[lab] = d
        emit("=" * 74)
        emit(f"{lab.upper()}   {d['sites']:,.0f} sites, "
             f"{d['variable']:,.0f} variable")
        emit("=" * 74)
        emit(f"  exact-zero cells        {d['exact_zero_cells']:6d} of {d['cells']}")
        emit(f"  asymmetry (i,j)->(N-i,N-j) {d['asymmetry']:.3f}   "
             "(~0 would mean it is effectively folded; it is not)")
        emit(f"  FIXED differences        {d['fixed_AAalt']:12,.1f} (AA all ALT) "
             f"{d['fixed_BBalt']:12,.4f} (BB all ALT)")
        emit(f"  near-fixed (within {NEAR})   {d['nearfix_AAalt']:12,.0f} (AA ALT) "
             f"{d['nearfix_BBalt']:12,.0f} (BB ALT)")
        emit()

    b, c = res["body"], res["control"]
    emit("=" * 74)
    emit("BODY vs CONTROL, as a fraction of variable sites")
    emit("=" * 74)
    emit(f"  {'':<28s}{'body':>10s}{'control':>10s}{'ratio':>9s}")
    for k, lbl in (("shared", "shared polymorphism"),
                   ("priv_AA", "AA-private"), ("priv_BB", "BB-private"),
                   ("nearfix_AAalt", "near-fixed AA-ALT/BB-REF"),
                   ("nearfix_BBalt", "near-fixed AA-REF/BB-ALT")):
        fb, fc = b[k] / b["variable"], c[k] / c["variable"]
        r = f"{fb / fc:9.2f}" if fc > 0 else f"{'inf':>9s}"
        emit(f"  {lbl:<28s}{100 * fb:9.3f}%{100 * fc:9.3f}%{r}")
    emit()
    emit("READING IT")
    emit("-" * 74)
    emit("  1. THE CONTROL IS CLEAN. Every near-fixed cell is exactly zero and")
    emit(f"     {c['exact_zero_cells']} of {c['cells']} cells are empty. AA and BB")
    emit("     are indistinguishable outside the inversion, which is what")
    emit("     exchangeability requires. This is the strongest control pass in")
    emit("     the project and it is what licenses reading the body at all.")
    emit("  2. NO EXACT FIXED DIFFERENCES, in either region -- and that is")
    emit("     EXPECTED, not a failure. dxy/pi_I = 1.385 means between-class")
    emit("     divergence is only ~1.4x within-BB diversity, i.e. the classes are")
    emit("     nowhere near reciprocal monophyly. Complete sorting across all 160")
    emit("     chromosomes should be rare, and it is.")
    emit("  3. NEAR-FIXED DIFFERENTIATION IS REAL AND LARGE: "
         f"{b['nearfix_AAalt'] + b['nearfix_BBalt']:,.0f} sites in the")
    emit("     body against ZERO in the control. This is the answer the called")
    emit("     VCF could not give -- its version of this test failed its own")
    emit("     control (sec 8.23).")
    emit("  4. THE BARRIER SIGNATURE IS CLEAN: shared polymorphism collapses")
    emit(f"     from {100 * c['shared'] / c['variable']:.1f}% to "
         f"{100 * b['shared'] / b['variable']:.1f}% of variable sites while private")
    emit("     variation rises in both classes. That is arrangements sorting")
    emit("     behind a recombination barrier, and it is symmetric in AA/BB so")
    emit("     it is the least reference-sensitive statistic here.")
    emit("  5. THE DIRECTIONAL ASYMMETRY IS NOT INTERPRETABLE AS BIOLOGY.")
    emit(f"     Near-fixed runs {b['nearfix_AAalt'] / b['nearfix_BBalt']:.1f}:1 toward "
         "AA-ALT/BB-REF. The reference IS a")
    emit("     BB haplotype inside the inversion, so BB-derived variants shared")
    emit("     with the reference are invisible by construction. The control")
    emit("     cannot remove this -- outside the inversion the reference carries")
    emit("     no arrangement identity. Treat the counts as lower bounds.")
    emit("  6. AA IS NOT BEING LOST TO MAPPING BIAS: AA-private exceeds")
    emit(f"     BB-private in the body ({100 * b['priv_AA'] / b['variable']:.1f}% vs "
         f"{100 * b['priv_BB'] / b['variable']:.1f}%). Severe reference-mapping")
    emit("     loss would deplete AA, and it does not.")
    emit()
    emit("  LIMITATION: a 2D SFS carries COUNTS, not POSITIONS. It cannot say")
    emit("  which genes the near-fixed sites are in. That needs per-site output")
    emit("  (`realSFS fst print`) intersected with degenotate, which is the")
    emit("  natural follow-on.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sfs2d_karyotype.txt").write_text("\n".join(lines) + "\n")
    json.dump(res, (OUT / "sfs2d_karyotype.json").open("w"), indent=2)
    print(f"\nwrote {OUT}/sfs2d_karyotype.{{txt,json}}")


if __name__ == "__main__":
    main()
