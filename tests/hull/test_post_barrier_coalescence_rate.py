"""Post-barrier coalescence rate inside the inverted region.

Past t_inv there is a single panmictic population at the inverted locus, so a
pair of lineages must coalesce at rate 1/(2N) there, exactly as in the
collinear flanks. Regression test for a double count: if the inversion's
karyotype tag survives barrier removal (I relabelled to S), body segments form a
separate (pop, class) coalescence cell from the panmictic flanks, and a pair
overlapping in both cells coalesces at 2/(2N) (the post-barrier merge is a full
Hudson merge). With a barrier lasting only 0.01N generations and essentially no
recombination, every pair then had E[T] ~ N instead of ~2N.

Exact expectations under constant N, constant inverted frequency p, no flux:
  within class q:  E[T] = 2Nq(1 - exp(-t/2Nq)) + 2N exp(-t/2Nq)
  between classes: E[T] = t + 2N
"""

import math

import numpy as np
import pytest

from msinv.hull import HullSimulator
from msinv.hull.inversion import InversionSpec

N = 2_000
L = 20_000
LEFT, RIGHT = 0.1 * L, 0.9 * L
NS = 10
REPS = 150


def _expect(p, t):
    def within(q):
        a = 2 * N * q
        return a * (1 - math.exp(-t / a)) + 2 * N * math.exp(-t / a)
    return {"I": within(p), "S": within(1 - p), "IS": t + 2 * N}


def _pair_times(p, t, r, seed):
    spec = InversionSpec(bp_left=LEFT, bp_right=RIGHT, p_inv=p, t_inv=t,
                         gene_conversion_rate=1e-15, mean_tract_length=1.0)
    sim = HullSimulator(n_std=NS, n_inv=NS, population_size=N,
                        sequence_length=L, recombination_rate=r,
                        inversions=[spec], seed=seed)
    ts = sim.simulate()
    s_nodes = list(range(NS))          # n_std samples are node IDs 0..n_std-1
    i_nodes = list(range(NS, 2 * NS))
    w = [0, LEFT, RIGHT, L]
    pi_i = ts.diversity([i_nodes], windows=w, mode="branch")[1][0] / 2
    pi_s = ts.diversity([s_nodes], windows=w, mode="branch")[1][0] / 2
    dxy = np.ravel(ts.divergence([i_nodes, s_nodes], windows=w,
                                 mode="branch")[1])[0] / 2
    return float(pi_i), float(pi_s), float(dxy)


@pytest.mark.parametrize("p,t_over_n,r", [
    (0.5, 0.01, 1e-14),     # barrier vanishes almost at once: panmictic 2N
    (0.5, 0.01, 1e-8),
    (0.374, 0.5, 1e-8),
    (0.74, 1.0, 1e-8),
])
def test_pairwise_coalescence_times_match_expectation(p, t_over_n, r):
    t = t_over_n * N
    a = np.array([_pair_times(p, t, r, 4_000_000 + k) for k in range(REPS)])
    e = _expect(p, t)
    for j, key in enumerate(("I", "S", "IS")):
        m = a[:, j].mean()
        se = a[:, j].std(ddof=1) / math.sqrt(len(a))
        z = (m - e[key]) / se
        assert abs(z) < 4.0, (key, m, e[key], z)
