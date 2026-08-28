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

# =====================================================================
# POLARIZATION REVERSED 2026-08-27 -- READ THIS BEFORE USING ANY CONSTANT
# =====================================================================
# I (inverted, DERIVED) == the BB cluster, at frequency 0.374.
# S (standard, ANCESTRAL) == the AA cluster, at 0.626.
#
# For most of this project's history the opposite was recorded. The reference
# genome was assembled from an inverted individual, so the arrangement it
# carries -- BB -- is the derived one; a polarization test that appeared to
# confirm "AA = derived" turned out to use the reference's own base as the
# ancestral state at 69.2% of sites (NOTES sec 8.15). The cluster LABELS are
# unchanged; only the interpretation is swapped.
#
# Consequence: every constant below whose name contains I or S changed meaning,
# and every fit in NOTES sec 7 and 8.6-8.8 was made against the old values and
# is invalid.

P_INV = 0.374                 # frequency of the INVERTED (BB) arrangement

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
PI_I_OVER_PI_S = 1.3429      # was 0.744 under the reversed polarization

# dxy / pi_I = dxy(AA,BB) / pi(AA) -- normalised by the INVERTED class's own
# pi specifically. Unlike the windowed spatial analysis in
# illex/scripts/empirical_windowed.py, which uses mean(pi_AA, pi_BB) as the
# correct baseline (yielding ~1.598), these two normalizations are not
# directly comparable. A third option—pooled pi over combined AA+BB—was
# rejected as wrong: it contains the between-arrangement differences that
# constitute dxy, partly dividing dxy by itself (see progress.md Task 6).
# This is the second fitted target.
DXY_OVER_PI_I = 1.3738       # was 1.846 under the reversed polarization

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
PI_I_OVER_PI_S_BODY = 1.3556
PI_I_OVER_PI_S_BODY_SE = 0.0481
DXY_OVER_PI_I_BODY = 1.3848
DXY_OVER_PI_I_BODY_SE = 0.0214

# Same estimator applied to the full nominal span, for comparability with the
# historical targets above.
PI_I_OVER_PI_S_SE = 0.0471
DXY_OVER_PI_I_SE = 0.0227

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
# WITHDRAWN (NOTES sec 8.13.1): the denominator is not a stable quantity --
# two collinear controls disagree by 70% and R spans 3x across methods.
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


# --- ANGSD/GL within-arrangement SFS contrast (NOTES sec 8.5) ---
# Singleton-fraction ratio f1(I)/f1(S) with I = BB, from
# results/illex/sfs_shape_angsd.json. Was quoted as 1.211 with the labels the
# other way round.
# CALIBRATED against the collinear control (NOTES sec 8.19). The raw body
# ratio is 0.8256 +- 0.0076, but the control ratio -- where AA and BB are
# exchangeable and it must be 1 -- comes out at 1.0161 +- 0.0011, i.e. 15 SE
# from 1. That offset is near-identical in every 1 Mb block, so it is a
# systematic property of the two sample sets (n = 254 vs 95, so realSFS's EM
# estimates the two source spectra with different bias), not of the genome.
# The model side has n_i = n_s = 100 and carries no such asymmetry, so the
# empirical ratio is divided by the control ratio to cancel it.
#
#   raw body 0.8256 / control 1.0161 = 0.8125
#
# Fits scored before 2026-08-28 used the uncalibrated 0.8256.
SFS_F1_RATIO_BODY = 0.8125
SFS_F1_RATIO_BODY_SE = 0.0076
SFS_F1_RATIO_BODY_RAW = 0.8256
SFS_F1_RATIO_CONTROL = 1.0161
