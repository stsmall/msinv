"""Simulator against Charlesworth (2023) equilibrium coalescence times.

Panmictic population of N diploids, inversion at constant frequency x,
exchange at rate r per site per generation in heterokaryotypes, which in
msinv is ``gene_conversion_rate * flux_coverage(site)``. At equilibrium,
in units of 2N with rho = 4 N r (Equations 5a-5c and 3):

    T12 = 1 + 2/rho
    T11 = (x(1 + 2y) + rho x y) / (1 + rho x y)
    T22 = (y(1 + 2x) + rho x y) / (1 + rho x y)
    x T11 + y T22 = 1

The exact chain (``tests/expect/test_charlesworth2023.py``) matches these to
machine precision; this file checks that the simulator does too, by Monte
Carlo at the inversion centre. The inversion fills the simulated sequence
and recombination is negligible, so lineages are split only by flux.

``t_inv`` is a barrier origin far enough back (at least 10 * 2N and 12
mean T12) that truncation is negligible. Sites within about a base of the
breakpoints, where phi -> 0, keep S and I apart until ``t_inv``, so a
longer ``t_inv`` only costs time.
"""

import itertools

import numpy as np
import pytest

from msinv.expect import flux_coverage
from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography

N = 1000
L, ELL = 2000.0, 500.0
SITE = L / 2
N_S = N_I = 4


def _eq5(x, rho):
    y = 1 - x
    return {
        "II": (x * (1 + 2 * y) + rho * x * y) / (1 + rho * x * y),
        "SS": (y * (1 + 2 * x) + rho * x * y) / (1 + rho * x * y),
        "IS": 1 + 2 / rho,
    }


def _simulate(x, rho, reps):
    r = rho / (4 * N)
    gamma = r / flux_coverage(SITE, 0.0, L, ELL, "geometric")
    t_inv = 2 * N * max(10.0, 12 * (1 + 2 / rho))
    s_nodes, i_nodes = range(N_S), range(N_S, N_S + N_I)
    out = {"II": [], "SS": [], "IS": []}
    for seed in range(1, reps + 1):
        inv = InversionSpec(
            bp_left=0.0,
            bp_right=L,
            p_inv=x,
            t_inv=t_inv,
            gene_conversion_rate=gamma,
            mean_tract_length=ELL,
        )
        sim = HullSimulator(
            sample_config={("S", 0): N_S, ("I", 0): N_I},
            demography=Demography(pop_sizes=[N]),
            sequence_length=int(L),
            recombination_rate=1e-12,
            inversions=[inv],
            seed=seed,
        )
        tree = sim.simulate().at(SITE)
        # sample_config order: S nodes first, then I.
        out["SS"].append(
            np.mean([tree.tmrca(a, b) for a, b in itertools.combinations(s_nodes, 2)])
        )
        out["II"].append(
            np.mean([tree.tmrca(a, b) for a, b in itertools.combinations(i_nodes, 2)])
        )
        out["IS"].append(np.mean([tree.tmrca(a, b) for a in s_nodes for b in i_nodes]))
    return {k: np.asarray(v) / (2 * N) for k, v in out.items()}


# Replicates sized for about 10 s each; coalescence times are heavy
# tailed, so a few hundred replicates under-estimate the standard error.
@pytest.mark.parametrize(
    "x,rho,reps", [(0.1, 0.4, 20_000), (0.3, 4.0, 8_000), (0.5, 1.0, 8_000)]
)
def test_equilibrium_times_match_equation_5(x, rho, reps):
    sims = _simulate(x, rho, reps)
    theory = _eq5(x, rho)
    for key, vals in sims.items():
        mean, se = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
        assert abs(mean - theory[key]) < 4.5 * se, (
            f"T{key} at x={x}, rho={rho}: simulated {mean:.4f} +- {se:.4f}, "
            f"Equation 5 gives {theory[key]:.4f}"
        )
        assert abs(mean / theory[key] - 1) < 0.05
    # Equation 3, per replicate then averaged.
    ts = x * sims["II"] + (1 - x) * sims["SS"]
    se = ts.std(ddof=1) / np.sqrt(len(ts))
    assert abs(ts.mean() - 1.0) < 4.5 * se
