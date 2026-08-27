#!/usr/bin/env python
"""Is the inversion neutral? Rerun under the CORRECTED polarization.

    .venv/bin/python -m illex.scripts.neutrality_check

WHAT CHANGED
------------
NOTES sec 7.5.3 excluded neutrality using the frequency the inverted arrangement
was believed to sit at: 0.626. The polarization is reversed (sec 8.15), so the
inverted arrangement is at **0.374** today — and the refit (sec 8.16) says it did
not get there by rising, but by **declining from p_hist ~ 0.70 starting ~175 ka**.

That changes the argument's structure, not just its numbers, and it splits into
two independent tests:

1. **Attainment.** Could drift take a new inversion to the highest frequency it
   is inferred to have reached — p_hist ~ 0.70, not the present 0.374?
2. **The decline.** Could drift move it from 0.70 down to 0.374 in the ~100 ky
   the fall is inferred to have taken? This test did not exist before: under the
   old reading the arrangement was rising, and a rise to a *current* frequency
   is the only thing there was to test.

Both are calibration-free: no mutation rate, no accessibility mask, no
simulation.

THE TWO STATISTICS
------------------
Attainment, from the neutral diffusion with absorbing boundaries (exact, and
verified against Wright-Fisher in sec 7.5.3):

    E[t | reaches x] = (4N/x)[x + (1-x)ln(1-x)]      P(reach x) = 1/(2Nx)

Decline, from the neutral variance of an allele frequency over a window, with
the growth demography's time-varying N(t):

    Var[p(t) - p0] = p0(1-p0) * INT dt / (2 N(t))

so an observed change is expressed in standard deviations of pure drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from illex.theory import N0, N_ANC, N_growth

OUT = Path("results/illex")

P_NOW = 0.374          # present frequency of the inverted (BB) arrangement
P_HIST = 0.70          # fitted historical frequency (NOTES sec 8.16.3)
T_DECLINE = 175_000.0  # fall completed this long ago
T_FALL = 100_000.0     # duration of the fall
T_INV = 850_000.0      # fitted age


def hitting_time(x: float, n_e: float) -> float:
    return float((4.0 * n_e / x) * (x + (1.0 - x) * np.log(1.0 - x)))


def reach_prob(x: float, n_e: float) -> float:
    return float(1.0 / (2.0 * n_e * x))


def drift_var(p0: float, t_lo: float, t_hi: float, n_pts: int = 20000) -> float:
    """Var of a neutral frequency change over backward window [t_lo, t_hi]."""
    t = np.linspace(t_lo, t_hi, n_pts)
    return float(p0 * (1.0 - p0) * np.trapezoid(1.0 / (2.0 * N_growth(t)), t))


def main() -> None:
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("Neutrality under the CORRECTED polarization "
         "(inverted = BB, p_now = 0.374)")
    emit(f"fitted history: arose ~{T_INV / 1e3:.0f} ka, held near "
         f"p_hist = {P_HIST}, fell to {P_NOW} over ~{T_FALL / 1e3:.0f} ky "
         f"ending ~{T_DECLINE / 1e3:.0f} ka")
    emit()

    # ---- 1. attainment -------------------------------------------------
    emit("=" * 74)
    emit("1. ATTAINMENT -- could drift take it to the frequency it reached?")
    emit("=" * 74)
    emit("  The relevant target is the HIGHEST frequency inferred, p_hist, not")
    emit("  the present one: a neutral allele has to get up there before it can")
    emit("  come back down.")
    emit()
    emit(f"  {'Ne':<26s} {'x=0.374 (now)':>18s} {'x=0.70 (p_hist)':>18s}")
    ne_cases = [("N_ANC = 547,928", N_ANC),
                ("N(t=850 ka)", float(N_growth(T_INV))),
                ("N(t=275 ka)", float(N_growth(T_DECLINE + T_FALL))),
                ("N0 = 6,808,096", N0)]
    for name, ne in ne_cases:
        emit(f"  {name:<26s} {hitting_time(P_NOW, ne):>18,.0f} "
             f"{hitting_time(P_HIST, ne):>18,.0f}")
    emit()
    emit(f"  coefficients: E[t|reach x] = {hitting_time(P_NOW, 1.0):.3f}*N at "
         f"x=0.374, {hitting_time(P_HIST, 1.0):.3f}*N at x=0.70")
    emit(f"  (the old, wrong reading used x=0.626: {hitting_time(0.626, 1.0):.3f}*N)")
    emit()
    lo = min(hitting_time(P_HIST, ne) for _, ne in ne_cases)
    hi = max(hitting_time(P_HIST, ne) for _, ne in ne_cases)
    emit(f"  Reaching p_hist = 0.70 by drift takes {lo:,.0f}-{hi:,.0f} "
         "generations,")
    emit(f"  against a fitted total age of {T_INV:,.0f}.")
    emit(f"  The BINDING case is the smallest Ne on the arm (N_ANC), which "
         f"needs {lo / T_INV:.2f}x")
    emit("  the entire lifetime of the inversion -- a real but NARROW margin.")
    emit("  It widens fast with Ne: the arrangement arose at 850 ka when N was")
    emit("  N_ANC, but N grows 12.4x from there, so any rise happening later")
    emit(f"  than the origin faces up to {hi / T_INV:.0f}x the available time.")
    emit()
    emit(f"  P(a new neutral inversion ever reaching 0.70) = "
         f"{reach_prob(P_HIST, N_ANC):.2e} to "
         f"{reach_prob(P_HIST, N0):.2e}")
    emit()
    emit("  Per unit Ne the requirement went UP, since the corrected history")
    emit("  has the arrangement reaching a higher frequency than the old reading")
    emit(f"  (0.70 vs 0.626) and the hitting time rises with x: "
         f"{hitting_time(P_HIST, 1.0) / hitting_time(0.626, 1.0):.2f}x.")
    emit("  But the fitted AGE rose too (850 ky vs 720 ky), and the two nearly")
    emit("  cancel: the binding margin was 1.26x before and is 1.25x now. The")
    emit("  attainment test is neither stronger nor weaker than it ever was --")
    emit("  and sec 7.5.3's 'excluded by an order of magnitude' was an")
    emit("  overstatement even then, taken from the largest-Ne case rather than")
    emit("  the binding one.")
    emit()

    # ---- 2. the decline ------------------------------------------------
    emit("=" * 74)
    emit("2. THE DECLINE -- a test that did not exist under the old reading")
    emit("=" * 74)
    v = drift_var(P_HIST, T_DECLINE, T_DECLINE + T_FALL)
    sd = float(np.sqrt(v))
    obs = P_HIST - P_NOW
    emit(f"  observed fall: {P_HIST} -> {P_NOW}, i.e. {obs:.3f}, over "
         f"{T_FALL:,.0f} generations")
    emit(f"  N over that window: {float(N_growth(T_DECLINE + T_FALL)):,.0f} to "
         f"{float(N_growth(T_DECLINE)):,.0f}")
    emit(f"  neutral drift SD over the same window: {sd:.4f}")
    emit(f"  => the observed fall is {obs / sd:.1f} standard deviations of pure "
         "drift")
    emit()
    for tf in (T_FALL, 2 * T_FALL, 5 * T_FALL):
        s2 = float(np.sqrt(drift_var(P_HIST, T_DECLINE, T_DECLINE + tf)))
        emit(f"    if the fall took {tf / 1e3:>5.0f} ky instead: SD {s2:.4f}, "
             f"observed fall = {obs / s2:5.1f} SD")
    emit()
    emit("  **This is where the argument is soft.** t_fall was FIXED at 100 ky")
    emit("  in the refit, not estimated. At 100 ky the fall is 5.7 SD and drift")
    emit("  is excluded comfortably; at 500 ky it is 1.8 SD, which drift can")
    emit("  produce about 4% of the time. So the decline test is decisive only")
    emit("  if the fall was fast, and nothing yet establishes that it was.")
    emit()

    # ---- verdict --------------------------------------------------------
    emit("=" * 74)
    emit("VERDICT")
    emit("=" * 74)
    emit("  Neutrality remains excluded, on two independent grounds -- but the")
    emit("  margins are NARROWER than under the old, wrong polarization, and")
    emit("  that should be stated rather than buried:")
    emit()
    emit(f"    attainment: needs {lo / T_INV:.2f}x the inversion's lifetime at "
         "the most")
    emit("                favourable Ne, up to "
         f"{hi / T_INV:.0f}x at the least favourable")
    emit(f"    decline:    {obs / sd:.1f} SD if the fall took 100 ky, but only "
         f"{obs / float(np.sqrt(drift_var(P_HIST, T_DECLINE, T_DECLINE + 5 * T_FALL))):.1f} SD")
    emit("                if it took 500 ky -- and t_fall was fixed, not fitted")
    emit()
    emit("  Neither argument uses mu, the accessibility mask, or a simulation,")
    emit("  which is their value. Note that sec 7.5.3's 'excluded by an order of")
    emit("  magnitude' was always an overstatement: it quoted the largest-Ne")
    emit(f"  case, whereas the BINDING margin was 1.26x then and is "
         f"{lo / T_INV:.2f}x now.")
    emit()
    emit("  Caveat carried from sec 7.5.3: these are diffusion results for an")
    emit("  unlinked neutral allele. They bound DRIFT, not selection on linked")
    emit("  sites, and they take the fitted p_hist and t_fall as given -- t_fall")
    emit("  in particular was fixed at 100 ky in the refit, not estimated, which")
    emit("  is why the sensitivity to it is tabulated above rather than assumed.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "neutrality_check.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/neutrality_check.txt")


if __name__ == "__main__":
    main()
