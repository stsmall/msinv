#!/usr/bin/env python
"""Is the mu-free ratio R contaminated by the reference's arrangement?

    .venv/bin/python -m illex.scripts.mu_free_check

THE WORRY
---------
R = dxy(AA,BB) / div(illecebrosus, coindetii) was computed with BOTH terms
measured inside the inversion (NOTES sec 5.5). Section 8.12 then established
that the reference genome carries the **BB arrangement** inside the inversion
(non-reference allele frequency at diagnostic sites: p_AA 0.515 vs p_BB 0.217,
against 0.303 vs 0.287 in collinear sequence). Any comparison to an outgroup
placed on that reference therefore inherits one arrangement.

Which term is actually at risk:

* **Numerator, dxy(AA,BB): SAFE.** It is a within-illecebrosus comparison, so a
  reference that favours BB shifts both classes' allele calls in the same
  direction and the contrast between them is unaffected.
* **Denominator, div(ill,coin): AT RISK IN PRINCIPLE.** It is measured against
  the reference haplotype, which inside the inversion is a BB lineage rather
  than a random illecebrosus one.

Two distinct mechanisms could bite, and they need separating:

1. **Read-mapping bias** — the mechanism that killed the argentinus test
   (sec 8.12). It does *not* apply here: coindetii divergence comes from an
   AnchorWave **assembly-to-assembly** alignment, not from short reads mapped to
   the reference. There is no mapping step to be biased.
2. **Lineage sampling** — the reference haplotype inside the inversion is a BB
   lineage, so div measures that particular lineage's divergence to coindetii
   rather than an average one. Since both arrangements coalesce into the same
   ancestral population at t_inv and follow a shared path to the species split,
   this should shift div by at most an inversion-scale amount, which is small
   against a ~2.5 My split — but it is worth measuring rather than asserting.

THE TEST
--------
Recompute div(ill,coin) per comparable bp in COLLINEAR sequence, where the
reference is arrangement-neutral, and compare it with the value measured inside
the inversion. If they agree, the denominator is a genome-wide quantity and R
stands. If they differ, R must be recomputed with the collinear denominator.

Note R itself cannot be computed "collinear only": its numerator, dxy(AA,BB), is
~0 outside the inversion by construction. The correct repair, if one is needed,
is the inversion numerator over a collinear denominator.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# Paths inlined rather than imported from illex.scripts.mu_free_ratio: that
# module imports pg_gpu at module scope, which lives in a different interpreter.
MASK_3STATE = ("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
               "03_karyotype/chr2_mask/chr2.mask.3state.bed")
_AW = ("/sietch_colab/data_share/illex/alignments/anchorwave_dir/"
       "ill_coin_alignment/coindetti_vcf")
COIN_CALLABLE = f"{_AW}/2.callable.bed"
COIN_SNPS = f"{_AW}/2.snps.vcf.gz"
BCFTOOLS = "/home/ssmall/bin/bcftools"

OUT = Path("results/illex")

REGIONS = {
    "inversion body": (60_500_000, 79_500_000),
    "collinear chr2:10-30 Mb": (10_000_000, 30_000_000),
    "collinear chr2:85-115 Mb": (85_000_000, 115_000_000),
}

# From NOTES sec 5.5 / results/illex/mu_free_ratio.json -- the numerator, which
# sec 8.12 establishes is safe.
DXY_AABB_BODY = 0.005146
R_PUBLISHED = 0.5137


def _bool(path: str, lo: int, hi: int, keep: str | None = None) -> np.ndarray:
    n = hi - lo
    out = np.zeros(n, dtype=bool)
    cond = f'$1=="2" && $3>{lo} && $2<{hi}'
    if keep:
        cond += f' && $4 ~ /{keep}/'
    p = subprocess.run(["awk", f'{cond} {{print $2"\\t"$3}}', path],
                       capture_output=True, text=True, check=True)
    for line in p.stdout.splitlines():
        a, b = line.split()
        i0, i1 = max(int(a), lo) - lo, min(int(b), hi) - lo
        if i1 > i0:
            out[i0:i1] = True
    return out


def snp_positions(lo: int, hi: int) -> np.ndarray:
    p = subprocess.run(
        [BCFTOOLS, "query", "-r", f"2:{lo}-{hi}", "-f", "%POS\n", COIN_SNPS],
        capture_output=True, text=True, check=True)
    return np.fromiter((int(x) for x in p.stdout.split()), dtype=np.int64)


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Is div(illecebrosus, coindetii) the same inside the inversion as in "
         "collinear sequence?")
    emit("Both terms per COMPARABLE bp = illecebrosus-accessible AND "
         "coindetii-comparable\n(callable UNION substitutions), the same "
         "construction as NOTES sec 5.5.")
    emit()
    emit(f"{'region':<26s} {'shared bp':>12s} {'subs':>10s} {'div/bp':>10s} "
         f"{'JC':>10s}")

    res = {}
    for name, (lo, hi) in REGIONS.items():
        acc = _bool(MASK_3STATE, lo, hi, keep="^accessible")
        same = _bool(COIN_CALLABLE, lo, hi)
        pos = snp_positions(lo, hi)
        idx = pos - 1 - lo
        ok = (idx >= 0) & (idx < hi - lo)
        idx = idx[ok]
        comparable = same.copy()
        comparable[idx] = True
        shared = acc & comparable
        n_sub = int(shared[idx].sum())
        n_bp = int(shared.sum())
        div = n_sub / n_bp
        jc = -0.75 * np.log(1.0 - 4.0 / 3.0 * div)
        res[name] = {"shared_bp": n_bp, "subs": n_sub, "div": div, "jc": jc}
        emit(f"{name:<26s} {n_bp:>12,} {n_sub:>10,} {div:>10.6f} {jc:>10.6f}")

    body = res["inversion body"]["div"]
    coll = [v["div"] for k, v in res.items() if k.startswith("collinear")]
    coll_mean = float(np.mean(coll))
    emit()
    emit(f"inversion / collinear = {body / coll_mean:.4f}  "
         f"({100 * (body / coll_mean - 1):+.1f}%)")
    emit()
    emit(f"{'=' * 74}\nEFFECT ON R\n{'=' * 74}")
    emit(f"  numerator dxy(AA,BB) = {DXY_AABB_BODY:.6f} (within-illecebrosus, "
         "unaffected)")
    emit(f"  R with the inversion-internal denominator  = "
         f"{DXY_AABB_BODY / body:.4f}   (published {R_PUBLISHED:.4f})")
    emit(f"  R with the collinear denominator           = "
         f"{DXY_AABB_BODY / coll_mean:.4f}")
    shift = (DXY_AABB_BODY / coll_mean) / (DXY_AABB_BODY / body) - 1
    emit(f"  shift = {100 * shift:+.1f}%")
    emit()
    if abs(shift) < 0.05:
        emit("  R IS ROBUST. The coindetii divergence is the same inside the")
        emit("  inversion as outside it, so the denominator is a genome-wide")
        emit("  quantity and the reference's arrangement does not propagate")
        emit("  into R. The sec 8.12 flag on R is withdrawn.")
        emit("  This is expected on mechanism: the coindetii comparison comes")
        emit("  from an AnchorWave assembly alignment, not from short reads")
        emit("  mapped to the reference, so there is no mapping step to bias.")
    else:
        emit("  R MOVES. Use the collinear denominator: the species divergence")
        emit("  is a genome-wide quantity and should not be estimated from")
        emit("  sequence where the reference carries one arrangement.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mu_free_check.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/mu_free_check.txt")


if __name__ == "__main__":
    main()
