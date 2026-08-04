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

# --- primary fitted statistics, inversion body (chr2:60-80 Mb) ---
# pi_I / pi_S = pi(AA) / pi(BB). This is the ratio the design fits first.
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

# --- held-out validation statistic, inversion body ---
# Hudson Fst(AA, BB). Not used for fitting -- reserved to validate a fit
# obtained from PI_I_OVER_PI_S and DXY_OVER_PI_I alone.
FST = 0.3652

# --- control-region values (chr2:10-30 Mb, collinear -- no inversion
# barrier). These confirm AA/BB are otherwise exchangeable outside the
# inversion (Fst ~0, dxy ~ pi, pi ratio ~1), ruling out a coverage/
# missingness artifact as the driver of the inversion-body pattern above. ---
PI_AA_CONTROL = 0.004324
PI_BB_CONTROL = 0.004374
DXY_CONTROL = 0.004364
FST_CONTROL = 0.0035
