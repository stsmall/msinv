"""Expected pairwise coalescence times for a single-origin inversion.

An inversion arises on ONE chromosome, so backward in time every inverted
lineage must coalesce by t_inv. Within-I times are therefore bounded by t_inv,
which is what makes pi_derived < pi_ancestral the expected direction.

Two independent implementations are provided on purpose:
  * expected_times() -- numerical integration of the hazard, works for any N(t)
  * const_closed_form() -- analytic, constant Ne only
They must agree (see tests). That cross-check exists because this derivation
has produced arithmetic errors before.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

# --- moments exponential-growth model (analysis/steps/08_demography) ---
N_ANC = 547_928.0
N0 = 6_808_096.0
T_GROW = 769_519.0
ALPHA = np.log(N0 / N_ANC) / T_GROW

MU = 3e-9
NE_CONST = 775_000.0          # pi / (4 mu), reproduces observed genome-wide pi
P_I_DEFAULT = 0.626           # derived/inverted arrangement frequency


def N_growth(t):
    """Diploid Ne at backward time t under the moments growth model."""
    t = np.asarray(t, dtype=float)
    return np.where(t <= T_GROW, N0 * np.exp(-ALPHA * np.minimum(t, T_GROW)), N_ANC)


def N_const(t):
    """Diploid Ne, constant."""
    return np.full_like(np.asarray(t, dtype=float), NE_CONST)


def _integrate_ET(hazard, t_max, forced_at=None, dt=200.0):
    """E[T] = int t*h(t)*S(t) dt, plus a mass point at forced_at if given.

    forced_at implements the single-origin cap: lineages that have not
    coalesced by t_inv are forced to coalesce there.
    """
    t = np.arange(0.0, t_max, dt) + dt / 2.0
    h = np.asarray(hazard(t), dtype=float)
    S = np.exp(-np.cumsum(h * dt))
    e = float(np.sum(t * h * S * dt))
    tail = float(S[-1])
    if forced_at is not None:
        e += forced_at * tail
    return e


def expected_times(N_fn, t_inv, p_i=P_I_DEFAULT, dt=200.0, horizon=4.0e7):
    """Mean pairwise coalescence times, in generations."""
    p_s = 1.0 - p_i

    def h_pan(t):
        return 1.0 / (2.0 * N_fn(t))

    def h_i(t):
        return 1.0 / (2.0 * N_fn(t) * p_i)

    def h_s(t):
        t = np.asarray(t, dtype=float)
        return np.where(t < t_inv, 1.0 / (2.0 * N_fn(t) * p_s), 1.0 / (2.0 * N_fn(t)))

    def h_between(t):
        # No coalescence possible before the inversion existed.
        t = np.asarray(t, dtype=float)
        return np.where(t < t_inv, 0.0, 1.0 / (2.0 * N_fn(t)))

    return {
        "panmictic": _integrate_ET(h_pan, horizon, dt=dt),
        "within_i": _integrate_ET(h_i, t_inv, forced_at=t_inv, dt=dt),
        "within_s": _integrate_ET(h_s, horizon, dt=dt),
        "between": _integrate_ET(h_between, horizon, dt=dt),
    }


def const_closed_form(ne, t_inv, p_i=P_I_DEFAULT):
    """Analytic constant-Ne solution. Independent check on expected_times()."""
    p_s = 1.0 - p_i
    tau_i, tau_s, two_ne = 2.0 * ne * p_i, 2.0 * ne * p_s, 2.0 * ne
    e_i = tau_i * (1.0 - np.exp(-t_inv / tau_i))
    # The -t*exp(-t/tau_s) from the integral cancels the +t carried by
    # survivors, leaving 2Ne*exp(-t/tau_s).
    e_s = tau_s * (1.0 - np.exp(-t_inv / tau_s)) + two_ne * np.exp(-t_inv / tau_s)
    return {"panmictic": two_ne, "within_i": e_i, "within_s": e_s,
            "between": t_inv + two_ne}


def ratios(N_fn, t_inv, p_i=P_I_DEFAULT, **kw):
    """The two statistics the design fits."""
    et = expected_times(N_fn, t_inv, p_i=p_i, **kw)
    return {
        "pi_i_over_pi_s": et["within_i"] / et["within_s"],
        "dxy_over_pi_i": et["between"] / et["within_i"],
    }


def dxy_floor(N_fn, p_i=P_I_DEFAULT, bounds=(2.0e5, 5.0e6)):
    """Minimum attainable dxy/pi_I over t_inv. Returns (floor, t_inv_at_floor)."""
    res = minimize_scalar(
        lambda t: ratios(N_fn, t, p_i=p_i)["dxy_over_pi_i"],
        bounds=bounds, method="bounded", options={"xatol": 2000},
    )
    return float(res.fun), float(res.x)


def solve_t_inv(N_fn, target_ratio, p_i=P_I_DEFAULT, bracket=(2.0e5, 5.0e6)):
    """t_inv reproducing an observed pi_I/pi_S."""
    return float(brentq(
        lambda t: ratios(N_fn, t, p_i=p_i)["pi_i_over_pi_s"] - target_ratio,
        bracket[0], bracket[1], xtol=1000,
    ))
