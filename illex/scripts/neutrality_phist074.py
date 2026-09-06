#!/usr/bin/env python
"""Neutrality tests at the WEIGHTED best-fit history (p_hist = 0.74).

    .venv/bin/python -m illex.scripts.neutrality_phist074

WHY THIS EXISTS
---------------
`neutrality_check.py` documents the FIRST refit epoch (unweighted, p_hist = 0.70;
NOTES sec 8.16-8.17). The decline family was later re-scored as a weighted chi^2
against the jackknife SEs (NOTES sec 8.36 / manuscript sec 3c), and the weighted
optimum moved p_hist 0.70 -> 0.74. Every quantity in this file therefore uses
0.74. This does NOT overwrite `neutrality_check.txt` -- the 0.70 numbers are the
sec 8.17 record and are kept -- exactly as the refit_decline* files keep one file
per epoch.

The two statistics are unchanged (calibration-free: no mu, no mask, no sim):

    attainment  E[t | reaches x] = (4N/x)[x + (1-x)ln(1-x)]
    decline     Var[p(t)-p0] = p0(1-p0) INT dt / (2 N(t))  ->  SD

Note p(1-p) falls past 0.5, so the 0.74 decline SD is SMALLER than the 0.70 one
(ratio sqrt(0.74*0.26 / 0.70*0.30) = 0.957) even though the fall itself is larger.
This is the arithmetic behind manuscript sec 3b's table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from illex.scripts.neutrality_check import drift_var, hitting_time, reach_prob
from illex.theory import N0, N_ANC, N_growth

OUT = Path("results/illex")

P_NOW = 0.374          # present frequency of the inverted (BB) arrangement
P_HIST = 0.74          # weighted-fit historical frequency (NOTES sec 8.36 / MS 3c)
T_DECLINE = 175_000.0  # fall completed this long ago (weighted fit)
T_FALL = 100_000.0     # duration of the fall at the reported best fit
T_INV = 850_000.0      # fitted age
T_FALL_GRID = (25_000.0, 50_000.0, 100_000.0, 200_000.0, 400_000.0)
P_HIST_OLD = 0.70      # the sec 8.17 unweighted value, for the comparison note


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    emit("Neutrality at the WEIGHTED best-fit history (inverted = BB, p_now = 0.374)")
    emit(f"fitted history: arose ~{T_INV / 1e3:.0f} ka, held near "
         f"p_hist = {P_HIST}, fell to {P_NOW} over ~{T_FALL / 1e3:.0f} ky "
         f"ending ~{T_DECLINE / 1e3:.0f} ka")
    emit(f"(supersedes neutrality_check.txt, which is the earlier unweighted "
         f"p_hist = {P_HIST_OLD} epoch, NOTES sec 8.17)")
    emit()

    # ---- 1. attainment -------------------------------------------------
    emit("=" * 74)
    emit("1. ATTAINMENT -- could drift take it to the frequency it reached?")
    emit("=" * 74)
    emit("  The relevant target is the HIGHEST frequency inferred, p_hist, not")
    emit("  the present one: a neutral allele has to get up there before it can")
    emit("  come back down.")
    emit()
    ne_cases = [("N_ANC = 547,928", N_ANC),
                ("N(t=850 ka)", float(N_growth(T_INV))),
                ("N(t=275 ka)", float(N_growth(T_DECLINE + T_FALL))),
                ("N0 = 6,808,096", N0)]
    emit(f"  {'Ne':<26s} {'x=0.374 (now)':>18s} {'x=0.74 (p_hist)':>18s}")
    for name, ne in ne_cases:
        emit(f"  {name:<26s} {hitting_time(P_NOW, ne):>18,.0f} "
             f"{hitting_time(P_HIST, ne):>18,.0f}")
    emit()
    emit(f"  coefficients: E[t|reach x] = {hitting_time(P_NOW, 1.0):.3f}*N at "
         f"x=0.374, {hitting_time(P_HIST, 1.0):.3f}*N at x=0.74")
    emit(f"  (the old, wrong reading used x=0.626: {hitting_time(0.626, 1.0):.3f}*N;"
         f" the unweighted refit used x=0.70: {hitting_time(P_HIST_OLD, 1.0):.3f}*N)")
    emit()
    lo = min(hitting_time(P_HIST, ne) for _, ne in ne_cases)
    hi = max(hitting_time(P_HIST, ne) for _, ne in ne_cases)
    lo_old = min(hitting_time(P_HIST_OLD, ne) for _, ne in ne_cases)
    emit(f"  Reaching p_hist = {P_HIST} by drift takes {lo:,.0f}-{hi:,.0f} "
         "generations,")
    emit(f"  against a fitted total age of {T_INV:,.0f}.")
    emit(f"  BINDING case (smallest Ne, N_ANC): needs {lo / T_INV:.2f}x the "
         "inversion's")
    emit(f"  entire lifetime; widens to {hi / T_INV:.0f}x at N0 as N grows 12.4x.")
    emit(f"  (At the earlier unweighted p_hist = {P_HIST_OLD} the binding margin "
         f"was {lo_old / T_INV:.2f}x, so weighting the fit strengthened this.)")
    emit()
    emit(f"  P(a new neutral inversion ever reaching {P_HIST}) = "
         f"{reach_prob(P_HIST, N_ANC):.2e} to {reach_prob(P_HIST, N0):.2e}")
    emit()

    # ---- 2. the decline ------------------------------------------------
    emit("=" * 74)
    emit("2. THE DECLINE -- how many SD of drift is the fall?")
    emit("=" * 74)
    obs = P_HIST - P_NOW
    sd_best = float(np.sqrt(drift_var(P_HIST, T_DECLINE, T_DECLINE + T_FALL)))
    emit(f"  observed fall: {P_HIST} -> {P_NOW}, i.e. {obs:.3f}")
    emit(f"  N over the best-fit ({T_FALL / 1e3:.0f} ky) window: "
         f"{float(N_growth(T_DECLINE + T_FALL)):,.0f} to "
         f"{float(N_growth(T_DECLINE)):,.0f}")
    emit()
    emit(f"  {'t_fall':>8s}  {'drift SD':>9s}  {'observed fall (SD)':>18s}")
    for tf in T_FALL_GRID:
        s2 = float(np.sqrt(drift_var(P_HIST, T_DECLINE, T_DECLINE + tf)))
        tag = "   <- best fit" if tf == T_FALL else ""
        emit(f"  {tf / 1e3:>6.0f}ky  {s2:>9.4f}  {obs / s2:>18.1f}{tag}")
    emit()
    emit(f"  At the fitted {T_FALL / 1e3:.0f} ky the fall is {obs / sd_best:.1f} SD "
         "of pure drift.")
    emit("  This bounds DRIFT only (diffusion for an unlinked neutral allele); it")
    emit("  takes the fitted p_hist and t_fall as given. The independent test that")
    emit("  t_fall itself is fast -- so a slow drift-out is excluded -- is the")
    emit("  weighted t_fall profile, NOTES sec 8.36 / refit_declinetfall2.txt.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "neutrality_check_phist074.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/neutrality_check_phist074.txt")


if __name__ == "__main__":
    main()
