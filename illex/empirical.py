"""Canonical home for the Illex chr2 empirical constants (I4).

Before this module existed these values were scattered and incomplete: the
pi ratio lived in ``tests/illex/test_floor_harness.py``, the dxy ratio lived
ONLY inside a single test (see ``tests/illex/test_theory.py``), and Fst
appeared nowhere in the branch's code at all. Both test files import from
here instead of re-typing the numbers.

Source: ``docs/superpowers/specs/2026-08-03-illex-chr2-neutral-sufficiency-
design.md``, "Per-arrangement diversity" table (computed 2026-08-03 from
``variants_filt.vcf.gz``, 349 samples: 254 AA + 95 BB, pg_gpu,
``missing_data='include'``, no MAF filter). AA is the derived/inverted
arrangement (p_inv = 0.626, moderate-confidence polarization), BB is the
ancestral/standard arrangement -- i.e. AA == I, BB == S in this package's
naming.

All region-specific measurements are over the FULL nominal span (inversion
60-80 Mb / control 10-30 Mb on chr2), not the differentiated-body subset
used by ``illex/scripts/empirical_windowed.py``'s core/edge analysis.
"""

from __future__ import annotations

# --- primary fitted statistics, FULL NOMINAL SPAN (chr2:60-80 Mb) ---
# These are the historical targets, measured over the full nominal span. They
# are reproduced to 4 dp by illex/scripts/empirical_jackknife.py (0.7439 /
# 1.8464), which also supplies the standard errors they never had.
#
# For fitting, prefer the *_BODY values below: the simulations are
# interval-restricted to the inversion body, so the body values are the
# like-for-like comparison and the nominal ones are diluted by two collinear
# flanking windows (NOTES sec 4.2).
#
# pi_I / pi_S = pi(AA) / pi(BB).
PI_I_OVER_PI_S = 0.744

# dxy / pi_I = dxy(AA,BB) / pi(AA) -- normalised by the INVERTED class's own
# pi specifically. Unlike the windowed spatial analysis in
# illex/scripts/empirical_windowed.py, which uses mean(pi_AA, pi_BB) as the
# correct baseline (yielding ~1.598), these two normalizations are not
# directly comparable. A third option—pooled pi over combined AA+BB—was
# rejected as wrong: it contains the between-arrangement differences that
# constitute dxy, partly dividing dxy by itself (see progress.md Task 6).
# This is the second fitted target.
DXY_OVER_PI_I = 1.846

# --- like-for-like targets: the empirically DIFFERENTIATED body ---
# Fst > 0.15 selects 189 of 200 100-kb windows, spanning 60.5-79.5 Mb. The
# excluded windows are collinear flanking sequence with control-like Fst
# (NOTES sec 4.2), and including them dilutes dxy/pi_I downward by 1.8%.
# Since the model integrates only the inversion body, these are the values a
# fit should target.
#
# Standard errors are delete-one-block jackknife over 1 Mb blocks (n = 20),
# from illex/scripts/empirical_jackknife.py; see NOTES sec 5.4 for why 1 Mb is
# the right block size and why these SEs are a LOWER bound on the uncertainty
# that should be propagated into an age.
PI_I_OVER_PI_S_BODY = 0.7368
PI_I_OVER_PI_S_BODY_SE = 0.0263
DXY_OVER_PI_I_BODY = 1.8794
DXY_OVER_PI_I_BODY_SE = 0.0503

# Same estimator applied to the full nominal span, for comparability with the
# historical targets above.
PI_I_OVER_PI_S_SE = 0.0262
DXY_OVER_PI_I_SE = 0.0534

# --- mu-free scale: arrangement divergence vs the coindetii species split ---
# R = dxy(AA,BB) / div(illecebrosus, coindetii), counted over
# illecebrosus-accessible AND coindetii-comparable bases (8,849,184 bp in the
# differentiated body). Free of mu, of the accessibility mask and of the
# generation time, because both terms are 2*mu*T. 1 Mb block jackknife.
#
# Its use: given an independent calibration T_cal for the illecebrosus-coindetii
# split, t_inv = R * (T_cal + T_anc_spp) - 2*N_ANC, with mu nowhere in it. This
# is the only identified route to an age that does not inherit mu's uncertainty
# (NOTES sec 5.5).
MU_FREE_R_BODY = 0.5137
MU_FREE_R_BODY_SE = 0.0146
MU_FREE_R_BODY_JC = 0.5102        # Jukes-Cantor corrected for multiple hits
MU_FREE_R_NOMINAL = 0.5019
MU_FREE_R_NOMINAL_SE = 0.0171

# --- Hudson Fst(AA, BB): ALGEBRAICALLY REDUNDANT, not held-out ---
# This was described here as a held-out validation statistic. It is not one,
# and cannot be: with r = PI_I_OVER_PI_S and d = DXY_OVER_PI_I,
#     Fst = 1 - (r + 1) / (2*d*r)
# exactly -- verified to 2.2e-16 over 600 simulations, and it reproduces the
# measured 0.3652 from the two ratios above to 4 dp. Any model matching both
# ratios matches Fst automatically, so Fst can neither validate such a fit nor
# break a parameter degeneracy. See NOTES sec 5.3 / spec amendment A15.
FST = 0.3652

# --- control-region values (chr2:10-30 Mb, collinear -- no inversion
# barrier). These confirm AA/BB are otherwise exchangeable outside the
# inversion (Fst ~0, dxy ~ pi, pi ratio ~1), ruling out a coverage/
# missingness artifact as the driver of the inversion-body pattern above. ---
PI_AA_CONTROL = 0.004324
PI_BB_CONTROL = 0.004374
DXY_CONTROL = 0.004364
FST_CONTROL = 0.0035
