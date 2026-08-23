#!/usr/bin/env python
"""Per-karyotype SFS shape from ANGSD genotype likelihoods -- the redo.

    .venv/bin/python -m illex.scripts.sfs_shape_angsd

WHY THIS EXISTS
---------------
``illex/scripts/sfs_shape.py`` computed the within-arrangement spectrum from
CALLED GENOTYPES and it does not work (NOTES sec 8.4). Per-site called
chromosomes run 20-508, at low called-n a rare variant is often not sampled at
all, and the resulting spectrum depends on the called-n floor more strongly than
on anything biological -- the inverted-vs-standard contrast even changes sign
between two defensible floors. Worse, in the COLLINEAR control, where the
panmictic model must be right, the model missed by more than it missed inside
the inversion. Nothing about the inversion could be read out of that.

Genotype likelihoods fix the cause rather than patching the symptom: ANGSD's SAF
integrates over genotype uncertainty per site instead of thresholding it, so
there is no calling step to ascertain against and no floor to choose. This is
already the project's standard for diversity, for exactly this reason -- the VCF
is variants-only with variable depth.

THE INPUT ALREADY EXISTED
-------------------------
``steps/04_angsd_chr2`` built per-karyotype SAFs for all of chr2 on 2026-07-04
(AA 254, AB 284, BB 95 individuals; ``-GL 1``, ``-minQ 20 -minMapQ 20``,
``-remove_bads -only_proper_pairs``, restricted to ``chr2.accessible.sites``).
So no ANGSD rerun was needed -- only ``realSFS -r`` over the intervals of
interest, which ``.tmp/angsd_sfs/run.sh`` does.

WHY ``-fold 0`` AND NOT ``-fold 1``
-----------------------------------
The SAFs were built with ``-anc $REF``, so the unfolded spectrum is really the
reference-polarised ALT-count spectrum, not a derived-count spectrum. That is
fine and is deliberately what we want: hypergeometric projection of an unfolded
count spectrum is EXACT, and folding only at the target size makes
mis-polarisation cancel. It is the identical transform applied to the model
side, which is the only thing that has to hold.

WHAT WOULD MAKE THIS TRUSTWORTHY, AND IT IS CHECKED FIRST
---------------------------------------------------------
The collinear control is the test. AA and BB are exchangeable outside the
inversion, so (a) their two control spectra must agree with each other, and
(b) both must agree with the model's no-inversion baseline. If either fails, the
estimator or the neutral model is still broken and nothing about the inversion
follows -- the same gate the called-genotype version failed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from illex.scripts.sfs_shape import PROJ, l1_shape, project_unfolded

SFS_DIR = Path(".tmp/angsd_sfs")
OUT = Path("results/illex")
MODEL_JSON = OUT / "sfs_shape_model.json"

# Diploid individuals per class in steps/04_angsd_chr2/bamlists.
N_DIP = {"AA": 254, "BB": 95}
ARMS = [("AA", "body"), ("BB", "body"), ("AA", "control"), ("BB", "control")]


def load(cls: str, region: str) -> np.ndarray:
    """realSFS output -> expected counts per ALT-allele class, 0..2N."""
    f = SFS_DIR / f"{cls}.{region}.sfs"
    v = np.array([float(x) for x in f.read_text().split()], dtype=float)
    n_chrom = 2 * N_DIP[cls]
    if v.size != n_chrom + 1:
        raise SystemExit(
            f"{f}: got {v.size} entries, expected {n_chrom + 1} for "
            f"{N_DIP[cls]} diploids. realSFS may still be running, or the "
            "bamlist size changed.")
    return v


def main() -> None:
    spec, tot = {}, {}
    for cls, region in ARMS:
        v = load(cls, region)
        n_chrom = 2 * N_DIP[cls]
        spec[(cls, region)] = project_unfolded(v, n_chrom, PROJ)
        tot[(cls, region)] = float(v[1:n_chrom].sum())

    mod = json.loads(MODEL_JSON.read_text())
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    def norm(v):
        v = np.asarray(v, float)
        return v / v.sum()

    hdr = "  bin              " + " ".join(f"{i + 1:>7d}"
                                          for i in range(PROJ // 2))
    emit("Per-karyotype SFS shape from ANGSD genotype likelihoods, "
         f"projected to n = {PROJ}")
    emit("SAFs from steps/04_angsd_chr2 (AA 254, BB 95 individuals), "
         "realSFS -fold 0 by region")
    emit()

    # ---- gate 1: is the estimator/baseline sound in the collinear region? ---
    emit("=" * 78)
    emit("GATE. COLLINEAR CONTROL 2:10-30 Mb -- must pass before reading "
         "anything else")
    emit("=" * 78)
    base = norm(mod["points"]["baseline_panmictic"]["I"])
    emit(hdr)
    emit("  MODEL no-inv     " + " ".join(f"{x:7.4f}" for x in base))
    for cls in ("AA", "BB"):
        emit(f"  ANGSD {cls:<10s} "
             + " ".join(f"{x:7.4f}" for x in norm(spec[(cls, "control")])))
    ca, cb = norm(spec[("AA", "control")]), norm(spec[("BB", "control")])
    emit()
    emit(f"  (a) AA vs BB in the control  L1 = {l1_shape(ca, cb):.4f}   "
         "(exchangeable outside the inversion, so this must be ~0)")
    emit(f"  (b) AA vs model no-inversion L1 = {l1_shape(ca, base):.4f}")
    emit(f"      BB vs model no-inversion L1 = {l1_shape(cb, base):.4f}")
    emit("  For reference, the CALLED-GENOTYPE version failed (b) at "
         "L1 = 0.61 / 0.59,")
    emit("  which is what made it unusable.")
    emit()

    # ---- the inversion body ------------------------------------------------
    emit("=" * 78)
    emit("INVERSION BODY 2:60.5-79.5 Mb")
    emit("=" * 78)
    emit(hdr)
    for cls in ("AA", "BB"):
        emit(f"  ANGSD {cls:<10s} "
             + " ".join(f"{x:7.4f}" for x in norm(spec[(cls, "body")])))
    for name in mod["points"]:
        if name == "baseline_panmictic":
            continue
        for arr, cls in (("I", "AA"), ("S", "BB")):
            emit(f"  {name[:12]:<12s}{cls:<3s}"
                 + " ".join(f"{x:7.4f}"
                           for x in norm(mod["points"][name][arr])))
    emit()
    emit(f"  {'point':<26s} {'L1 vs AA(I)':>12s} {'L1 vs BB(S)':>12s}")
    for name in mod["points"]:
        if name == "baseline_panmictic":
            continue
        emit(f"  {name:<26s} "
             f"{l1_shape(norm(spec[('AA', 'body')]), norm(mod['points'][name]['I'])):12.4f} "
             f"{l1_shape(norm(spec[('BB', 'body')]), norm(mod['points'][name]['S'])):12.4f}")
    emit()

    # ---- what changes relative to the collinear background -----------------
    emit("=" * 78)
    emit("BODY MINUS CONTROL, per class -- the inversion's own effect")
    emit("=" * 78)
    emit("  A class whose spectrum is unchanged from the collinear background "
         "has not\n  had its genealogy reshaped by the arrangement's frequency "
         "history.")
    for cls in ("AA", "BB"):
        b, c = norm(spec[(cls, "body")]), norm(spec[(cls, "control")])
        emit(f"  {cls}: L1(body, control) = {l1_shape(b, c):.4f}   "
             f"f1 {c[0]:.4f} -> {b[0]:.4f}")
    emit()

    # ---- the contrast ------------------------------------------------------
    emit("=" * 78)
    emit("CONTRAST: inverted vs standard inside the inversion")
    emit("=" * 78)
    oi, os_ = norm(spec[("AA", "body")]), norm(spec[("BB", "body")])
    emit(f"  {'point':<26s} {'f1(I)':>7s} {'f1(S)':>7s} {'ratio':>7s} "
         f"{'L1(I-S profile)':>16s}")
    emit(f"  {'ANGSD EMPIRICAL':<26s} {oi[0]:7.4f} {os_[0]:7.4f} "
         f"{oi[0] / os_[0]:7.3f} {'--':>16s}")
    for name in mod["points"]:
        if name == "baseline_panmictic":
            continue
        vi = norm(mod["points"][name]["I"])
        vs = norm(mod["points"][name]["S"])
        emit(f"  {name:<26s} {vi[0]:7.4f} {vs[0]:7.4f} {vi[0] / vs[0]:7.3f} "
             f"{np.abs((vi - vs) - (oi - os_)).sum():16.4f}")
    emit()
    emit("  sites contributing (realSFS expected polymorphic counts):")
    for k in ARMS:
        emit(f"    {k[0]}/{k[1]:<8s} {tot[k]:14,.0f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sfs_shape_angsd.txt").write_text("\n".join(lines) + "\n")
    json.dump({f"{c}_{r}": spec[(c, r)].tolist() for c, r in ARMS},
              (OUT / "sfs_shape_angsd.json").open("w"), indent=2)
    print(f"\nwrote {OUT}/sfs_shape_angsd.{{txt,json}}")


if __name__ == "__main__":
    main()
