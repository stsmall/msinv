"""Shared constants for the trajectory ABC."""
from __future__ import annotations

from illex.slim.config import REC_RATE  # noqa: F401  chr2 collinear male rate 2.158e-9 per bp per generation

P_NOW = 0.626            # present frequency of the inverted arrangement 2a (provisional; Task 5 replaces it)
P_TOL = 0.02             # acceptance half-width on p_now
# Upper bound of the logU prior on c, the per-site conversion probability per generation in heterokaryotypes.
# Task 5 (illex/abc/calibrate_flux.py, results/illex/abc_flux_calibration.txt): msinv sweep of c over 1e-10..1e-5
# (32 reps x 3 paths, no-flux baseline on msinv, SE(dxy/pi_I) = 0.0455 from abc_observed.json). c* = smallest c whose
# mean dxy/pi_I differs from baseline by > 1 SE: 3.2e-7 (soft D1 path, the rule's path), 1e-7 (hard B, soft N 1.2 Ma).
# 100 c* = 3.2e-5 exceeds the sweep cap, so GAMMA_MAX = min(100 c*, 1e-5) = 1e-5 (at 1e-5 dxy/pi_I is already
# 1.05-1.15, near the no-barrier value, and 1e-4 costs ~40 GB per run).
# Korunes & Noor 2019 Mol Ecol 28:1302 (doi 10.1111/mec.14921): 1e-5 to 2.5e-5 converted sites per bp per generation
# (D. pseudoobscura/persimilis crosses); ML tract length 188-390 bp (TRACT_BP below = 300, within that range).
GAMMA_MAX = 1e-5

PRIOR = dict(t_inv=(3e5, 1.5e6), p_hard=0.5, p0=(1e-3, 0.5), s=(1e-6, 1e-2), p_star=(0.3, 0.95))

MU = 3e-9                # per-bp per-generation mutation rate
TRACT_BP = 300.0         # mean gene-conversion tract length, bp: Korunes & Noor 2019 ML 188-390 bp (doi:10.1111/mec.14921)
BODY_BP = 40_000.0       # simulated inversion body
MARGIN_BP = 10_000.0     # collinear margin on each side of the body (sequence = BODY_BP + 2 * MARGIN_BP)
CONTROL_BP = 40_000.0    # separate no-inversion neutral control (the flank value of every statistic)
N_INV, N_STD = 100, 100  # simulated chromosomes per class (50 diploids each for LD); data has 508 / 190
N_KARYO = 633            # individuals karyotyped; p_obs = Binomial(2 * N_KARYO, p_now) / (2 * N_KARYO)
GAMMA_FLOOR = 1e-15      # msinv flux rate used when c (params['gamma_eff']) is NaN (no-flux model)

BP_LEFT, BP_RIGHT = 60_550_000, 79_520_000   # chr2 inversion breakpoints (reference coordinates)
N_I_CHROM, N_S_CHROM = 508, 190              # observed 2a and 2+ chromosomes (2aa 254, 2++ 95 homokaryotypes)
N_IND = N_KARYO                              # 633 karyotyped individuals
# Observed pi_2a/pi_2+, dxy/pi_2a, f1(2a)/f1(2+). FORMS DIFFER: pi_ratio and dxy_pi are RAW body values
# (empirical_jackknife.py, 60.5-79.5 Mb); f1_ratio is CONTROL-CALIBRATED body/control (sfs_f1_jackknife.py
# ratio of ratios, 1.2112 / 0.9842), i.e. the form of the ABC statistic (abc_observed.json "stats").
TARGETS = dict(pi_ratio=0.7377, dxy_pi=1.8763, f1_ratio=1.2307)
