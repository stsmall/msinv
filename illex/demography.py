"""msinv demography for the two Illex arms.

Growth arm: exponential from N0 at the present back to N_ANC at T_GROW,
constant at N_ANC before that. Constant arm: Ne = 775,000 throughout.

msinv's real Demography constructor takes ``pop_sizes: list[float]``
(present-day sizes per population) rather than a scalar
``population_size=``, and ``add_event`` takes a single raw ms-style
tuple such as ``("eg", time, pop, rate)`` rather than keyword args.
Events are kept sorted by time automatically (no ``sort_events()``
method exists) -- see msinv/hull/demography.py.
"""

from __future__ import annotations

import numpy as np
from msinv import Demography

from .theory import ALPHA, N0, N_ANC, NE_CONST, N_growth, T_GROW

PRESENT_NE_GROWTH = N0
PRESENT_NE_CONST = NE_CONST


def growth_demography() -> Demography:
    """Exponential growth backward from N0 to N_ANC, then flat.

    ``eg`` at time 0 sets N(t') = N0 * exp(-ALPHA * t'). The ``en`` at
    T_GROW pins the size to N_ANC; msinv's ``en``/``En`` handling resets
    the population's growth rate to 0 as a side effect (see
    rust/msinv-core/src/demography.rs apply_events_at), so the size stays
    flat at N_ANC for all deeper times -- matching theory.N_growth.
    """
    d = Demography(pop_sizes=[N0])
    d.add_event(("eg", 0.0, 0, ALPHA))
    d.add_event(("en", T_GROW, 0, N_ANC))
    return d


def constant_demography() -> Demography:
    return Demography(pop_sizes=[NE_CONST])


def growth_ne_schedule(t_max: float, n_points: int = 400):
    """(times, Ne) sampled from the same N(t) that theory.py integrates.

    Used for trajectory types that take an explicit n_e schedule rather than
    a scalar.
    """
    t = np.linspace(0.0, float(t_max), int(n_points))
    return t, N_growth(t)
