"""Shared constants for the trajectory ABC."""
from __future__ import annotations

from illex.slim.config import REC_RATE  # noqa: F401  chr2 collinear male rate 2.158e-9 per bp per generation

P_NOW = 0.626            # present frequency of the inverted arrangement 2a (provisional; Task 5 replaces it)
P_TOL = 0.02             # acceptance half-width on p_now
GAMMA_MAX = 1e-5         # upper bound of the logU prior on c, the per-site conversion probability (provisional; Task 5)

PRIOR = dict(t_inv=(3e5, 1.5e6), p_hard=0.5, p0=(1e-3, 0.5), s=(1e-6, 1e-2), p_star=(0.3, 0.95))

MU = 3e-9                # per-bp per-generation mutation rate
TRACT_BP = 1000.0        # mean gene-conversion tract length, bp (to be confirmed from Korunes & Noor in Task 5)
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
