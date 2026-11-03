"""Shared constants for the trajectory ABC (Task 1 adds REC_RATE and the rest here)."""
from __future__ import annotations

P_NOW = 0.626            # present frequency of the inverted arrangement 2a (provisional; Task 5 replaces it)
P_TOL = 0.02             # acceptance half-width on p_now
GAMMA_MAX = 1e-4         # upper bound of the logU prior on gamma_eff (provisional)

PRIOR = dict(t_inv=(3e5, 1.5e6), p_hard=0.5, p0=(1e-3, 0.5), s=(1e-6, 1e-2), p_star=(0.3, 0.95))
