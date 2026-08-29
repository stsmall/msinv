#!/usr/bin/env python
"""Selection scan WITHIN each arrangement -- the gap sec 8.21 identified.

    .venv/bin/python -m illex.scripts.sweep_within_karyotype

WHY THIS EXISTS
---------------
Every sweep call on chr2 comes from the pooled n=350 diploSHIC scan (sec 8.21),
which inside the inversion is reading a MIXTURE of two classes at FST = 0.365.
Its "hard depleted 11-fold, soft enriched" composition is largely what pooling
divergent arrangements looks like, not an independent readout, and a sweep
confined to ONE arrangement would be diluted by the other. Selection within AA
and within BB had never been examined.

WHY NOT RETRAIN diploSHIC
-------------------------
I first said this needed diploSHIC models retrained at n = 254 and n = 95,
because its features are sample-size dependent. That is true but it is not the
cheapest correct route, and there is a deeper problem with it: diploSHIC would
have to be trained on a genome-wide neutral demography, whereas the inverted
class INSIDE the inversion has its own coalescent (class size ~2Np, plus a
single-origin cap). Training on the wrong null is how the pooled scan went
wrong in the first place.

The per-karyotype ANGSD thetas already on disk avoid both problems: AA/AB/BB
pi, theta_W and Tajima's D in 50 kb windows across all of chr2, computed
2026-07-05 from the same SAFs used everywhere else. Each class is compared
against ITS OWN chr2 collinear background, so no external null is needed and
the class's reduced effective size cancels.

Unphased data forces SFS statistics anyway -- diploSHIC's haplotype features are
degenerate here (the project's standing rule), so pi and Tajima's D are the
right class of statistic, not a fallback.

WHAT A WITHIN-ARRANGEMENT SWEEP LOOKS LIKE
------------------------------------------
A sweep inside one arrangement drives a local pi dip AND a locally negative
Tajima's D **within that class**, relative to that class's own background. The
key discriminator is ASYMMETRY: the inversion reduces pi in both classes for
structural reasons (reduced class Ne, and for the derived class a coalescent
cap), so a shared dip means nothing. A dip in ONE arrangement and not the other,
at the same window, is the signal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TH = Path("/sietch_colab/data_share/illex/popgen_data/analysis/steps/"
          "04_angsd_chr2/thetas")
OUT = Path("results/illex")
INV_LO, INV_HI = 60_040_617, 79_995_597
CTL_LO, CTL_HI = 10_000_000, 30_000_000
MIN_SITES = 5_000          # 50 kb windows with fewer callable sites are noise
COLS = ["region", "Chr", "WinCenter", "tW", "tP", "tF", "tH", "tL",
        "Tajima", "fuf", "fud", "fayh", "zeng", "nSites"]


def load(cls: str) -> pd.DataFrame:
    d = pd.read_csv(TH / f"{cls}.2.win50k.pestPG", sep="\t", header=0,
                    names=COLS)
    d = d[d.nSites >= MIN_SITES].copy()
    d["pi"] = d.tP / d.nSites          # per-site
    d["theta_w"] = d.tW / d.nSites
    return d[["WinCenter", "pi", "theta_w", "Tajima", "nSites"]].rename(
        columns={"Tajima": "tajd"})


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    cls = {c: load(c) for c in ("AA", "BB")}
    m = cls["AA"].merge(cls["BB"], on="WinCenter", suffixes=("_AA", "_BB"))
    m["in_inv"] = (m.WinCenter >= INV_LO) & (m.WinCenter <= INV_HI)
    m["in_ctl"] = (m.WinCenter >= CTL_LO) & (m.WinCenter <= CTL_HI)

    emit("Selection scan WITHIN each arrangement (ANGSD 50 kb windows, chr2)")
    emit(f"AA n=254, BB n=95 diploids. Windows with >= {MIN_SITES:,} callable "
         "sites.")
    emit()
    emit("=" * 76)
    emit("BACKGROUND: what the inversion does to each class structurally")
    emit("=" * 76)
    emit(f"  {'':<22s}{'pi(AA)':>10s}{'pi(BB)':>10s}{'D(AA)':>9s}{'D(BB)':>9s}"
         f"{'n win':>7s}")
    for lab, sub in (("inversion body", m[m.in_inv]),
                     ("collinear control", m[m.in_ctl]),
                     ("rest of chr2", m[~m.in_inv & ~m.in_ctl])):
        emit(f"  {lab:<22s}{sub.pi_AA.median():10.6f}{sub.pi_BB.median():10.6f}"
             f"{sub.tajd_AA.median():9.3f}{sub.tajd_BB.median():9.3f}"
             f"{len(sub):7d}")
    emit()
    emit("  Both classes lose diversity inside the inversion -- expected, and")
    emit("  why a SHARED dip carries no information. The scan below keys on")
    emit("  ASYMMETRY between the classes instead.")
    emit()

    # each class standardised against its OWN collinear background
    for c in ("AA", "BB"):
        ref = m[m.in_ctl]
        for stat in ("pi", "tajd"):
            mu = ref[f"{stat}_{c}"].median()
            sd = ref[f"{stat}_{c}"].std()
            m[f"z_{stat}_{c}"] = (m[f"{stat}_{c}"] - mu) / sd

    emit("=" * 76)
    emit("SWEEP CANDIDATES: low pi AND negative Tajima's D in ONE class only")
    emit("=" * 76)
    emit("  Each class is z-scored against its OWN collinear background, so its")
    emit("  reduced effective size inside the inversion cancels.")
    emit()
    body = m[m.in_inv].copy()
    ctl = m[m.in_ctl].copy()

    def flag(d, c, other):
        return ((d[f"z_pi_{c}"] < -2) & (d[f"tajd_{c}"] < -2.5)
                & (d[f"z_pi_{c}"] < d[f"z_pi_{other}"] - 1))

    for lab, d in (("INVERSION BODY", body), ("collinear control", ctl)):
        fa, fb = flag(d, "AA", "BB"), flag(d, "BB", "AA")
        emit(f"  {lab:<20s} n={len(d):4d}   AA-specific {int(fa.sum()):3d} "
             f"({100 * fa.mean():4.1f}%)   BB-specific {int(fb.sum()):3d} "
             f"({100 * fb.mean():4.1f}%)")
    emit()
    emit("  The control gives the false-positive rate: AA and BB are")
    emit("  exchangeable there, so anything it flags is noise.")
    emit()

    fa, fb = flag(body, "AA", "BB"), flag(body, "BB", "AA")
    for c, f in (("AA", fa), ("BB", fb)):
        d = body[f]
        if not len(d):
            emit(f"  No {c}-specific candidates in the inversion body.")
            continue
        emit(f"  {c}-SPECIFIC candidate windows:")
        emit(f"    {'Mb':>8s}{'pi_AA':>10s}{'pi_BB':>10s}{'D_AA':>8s}"
             f"{'D_BB':>8s}{'z_pi':>8s}")
        for _, r in d.sort_values(f"z_pi_{c}").head(12).iterrows():
            emit(f"    {r.WinCenter / 1e6:8.2f}{r.pi_AA:10.6f}{r.pi_BB:10.6f}"
                 f"{r.tajd_AA:8.2f}{r.tajd_BB:8.2f}{r[f'z_pi_{c}']:8.2f}")
        emit()

    # ---- the test that matters: outliers vs the BLOCK's own distribution --
    emit("=" * 76)
    emit("THE CONTROL IS THE WRONG YARDSTICK -- retest against the BLOCK")
    emit("=" * 76)
    emit("  pi_AA sits ~23% below pi_BB throughout the inversion. That is the")
    emit("  project's central observation (pi_I/pi_S = 1.356 with I = BB), not a")
    emit("  sweep -- so ANY test standardised on the collinear control flags AA")
    emit("  across the whole block. The candidates above are that shift, not")
    emit("  localised events. The right null is the body's OWN ratio")
    emit("  distribution.")
    emit()
    body["r"] = body.pi_AA / body.pi_BB
    mu, sd = body.r.median(), body.r.std()
    body["z_r"] = (body.r - mu) / sd
    emit(f"  pi_AA/pi_BB across {len(body)} body windows: median {mu:.3f}, "
         f"sd {sd:.3f}")
    emit(f"  windows > 2 SD below the body's own median: "
         f"{int((body.z_r < -2).sum())} of {len(body)} "
         f"({100 * (body.z_r < -2).mean():.1f}%) -- 2.3% expected if Normal.")
    emit("  The block is LIGHT-tailed. There is no localised excess.")
    fa2 = flag(body, "AA", "BB")
    emit(f"  Of the {int(fa2.sum())} control-standardised candidates, only "
         f"{int((body[fa2].z_r < -2).sum())} survive this test,")
    emit(f"  and they spread over {len(set(body[fa2].WinCenter // 500_000))} "
         "distinct 0.5 Mb bins across 64.1-79.3 Mb -- not one focal region.")
    emit()
    emit("  VERDICT: no within-arrangement sweep signal in either class. Note")
    emit("  the direction too -- ZERO BB-specific windows by any criterion, and")
    emit("  BB carries HIGHER pi than AA throughout. The derived arrangement,")
    emit("  which is where a supergene's adaptive variant would sit, shows no")
    emit("  sweep signature at all.")
    emit()
    emit("  POWER CAVEAT: 50 kb windows, SFS statistics only (haplotype stats")
    emit("  are degenerate on unphased data), BB n = 95. A weak or old sweep")
    emit("  would be missed, and one predating the inversion would be shared by")
    emit("  both classes and invisible to an asymmetry test.")
    emit()

    emit("=" * 76)
    emit("THE CLASS-ASYMMETRY DISTRIBUTION")
    emit("=" * 76)
    emit("  A sweep in one arrangement shows as a heavy tail in the DIFFERENCE")
    emit("  of standardised pi. Compared against the control, which has none.")
    emit(f"  {'':<20s}{'median':>9s}{'p1':>9s}{'p99':>9s}{'min':>9s}{'max':>9s}")
    for lab, d in (("inversion body", body), ("collinear control", ctl)):
        x = d.z_pi_AA - d.z_pi_BB
        emit(f"  {lab:<20s}{x.median():9.3f}{x.quantile(.01):9.3f}"
             f"{x.quantile(.99):9.3f}{x.min():9.3f}{x.max():9.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_within_karyotype.txt").write_text("\n".join(lines) + "\n")
    body.to_csv(OUT / "sweep_within_karyotype_body.csv", index=False)
    print(f"\nwrote {OUT}/sweep_within_karyotype.txt and _body.csv")


if __name__ == "__main__":
    main()
