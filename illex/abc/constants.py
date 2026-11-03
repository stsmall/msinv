"""Shared constants for the trajectory ABC."""
from __future__ import annotations

from illex.slim.config import REC_RATE  # noqa: F401  chr2 collinear rate, per bp per generation (Task 1 updates it there)

P_NOW = 0.626            # present frequency of the inverted arrangement 2a (provisional; Task 5 replaces it)
P_TOL = 0.02             # acceptance half-width on p_now
GAMMA_MAX = 1e-4         # upper bound of the logU prior on gamma_eff (provisional)

PRIOR = dict(t_inv=(3e5, 1.5e6), p_hard=0.5, p0=(1e-3, 0.5), s=(1e-6, 1e-2), p_star=(0.3, 0.95))

MU = 3e-9                # per-bp per-generation mutation rate
TRACT_BP = 1000.0        # mean gene-conversion tract length, bp (to be confirmed from Korunes & Noor in Task 5)
BODY_BP = 40_000.0       # simulated inversion body
MARGIN_BP = 10_000.0     # collinear margin on each side of the body (sequence = BODY_BP + 2 * MARGIN_BP)
CONTROL_BP = 40_000.0    # separate no-inversion neutral control (the flank value of every statistic)
N_INV, N_STD = 100, 100  # simulated chromosomes per class (50 diploids each for LD); data has 508 / 190
N_KARYO = 633            # individuals karyotyped; p_obs = Binomial(2 * N_KARYO, p_now) / (2 * N_KARYO)
GAMMA_FLOOR = 1e-15      # msinv flux rate used when c (params['gamma_eff']) is NaN (no-flux model)
