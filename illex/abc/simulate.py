"""One ABC row: an msinv simulation of a precomputed 2a trajectory, plus its neutral control, to ``STAT_KEYS``.

Geometry: a BODY_BP inversion body between two MARGIN_BP collinear margins; body statistics use only the body.
The control is a SEPARATE no-inversion msinv simulation (``inversions=[]``, the engine the msprime-validation
suite checks against msprime) of CONTROL_BP with the same demography, sample sizes, REC_RATE and MU, its samples
split by index into the same two groups (first N_STD as S, the next N_INV as I, the order
``illex.stats.sample_nodes_by_karyotype`` uses). This mirrors the data's control (chr2 collinear sequence far
from the breakpoints); the linked margins are not used as the control.

Flux: the prior variable c (``params["gamma_eff"]``) is the per-site conversion probability per generation in
heterokaryotypes (the Korunes & Noor quantity). msinv's gamma * phi(x) is a per-bp INITIATION rate of tracts of mean
TRACT_BP (event rate gamma * p_other * integral(phi) * L, ``msinv/hull/simulator.py``), with phi ~ w/(1-w) inside
the body (``_phi``) and w = TRACT_BP / BODY_BP. So the initiation rate is c / TRACT_BP and
gamma = c / TRACT_BP * (1 - w) / w. c NaN (no-flux model) uses GAMMA_FLOOR.

One integer seed drives everything (both msinv runs, both mutation drops, the diploid pairings, p_obs).
"""
from __future__ import annotations

import math

import msprime
import numpy as np

from illex.abc import statvec as SV
from illex.abc.constants import (BODY_BP, CONTROL_BP, GAMMA_FLOOR, MARGIN_BP, MU, N_INV, N_KARYO, N_STD,
                                 REC_RATE, TRACT_BP)


def gamma_from_c(c: float) -> float:
    """msinv gene_conversion_rate for per-site conversion probability c (per generation, in heterokaryotypes)."""
    if c is None or not np.isfinite(c):
        return GAMMA_FLOOR
    w = TRACT_BP / BODY_BP
    return float(c) / TRACT_BP * (1.0 - w) / w


def _mutate(ts, seed):
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed, model=msprime.BinaryMutationModel(),
                                 discrete_genome=False)


def _inversion_ts(path, seed):
    from msinv import HullSimulator, InversionSpec

    from illex.demography import PRESENT_NE_GROWTH, growth_demography
    from illex.theory import N_growth

    t_inv = float(path.params["t_inv"])
    spec = InversionSpec(bp_left=MARGIN_BP, bp_right=MARGIN_BP + BODY_BP,
                         gene_conversion_rate=gamma_from_c(path.params.get("gamma_eff", float("nan"))),
                         mean_tract_length=TRACT_BP, tract_distribution="geometric",
                         trajectory={"type": "precomputed", "times": np.asarray(path.times, float),
                                     "freqs": [np.asarray(path.freqs, float)],
                                     "n_e": [float(N_growth(t_inv))], "t_inv": [t_inv]})
    sim = HullSimulator(n_std=N_STD, n_inv=N_INV, population_size=PRESENT_NE_GROWTH,
                        demography=growth_demography(), sequence_length=BODY_BP + 2 * MARGIN_BP,
                        recombination_rate=REC_RATE, inversions=[spec], seed=seed)
    return sim, sim.simulate()


def _control_ts(seed):
    from msinv import HullSimulator

    from illex.demography import PRESENT_NE_GROWTH, growth_demography

    sim = HullSimulator(samples=N_STD + N_INV, population_size=PRESENT_NE_GROWTH,
                        demography=growth_demography(), sequence_length=CONTROL_BP,
                        recombination_rate=REC_RATE, inversions=[], seed=seed)
    return sim.simulate()


def simulate_stats(path, seed: int) -> dict[str, float]:
    """``STAT_KEYS`` for one accepted path, plus ``p_obs`` and ``n_nan`` (the count of NaN statistics).

    Never raises on degenerate output (no segregating sites, zero variance): those keys are NaN.
    """
    from illex import stats as S

    ss = np.random.SeedSequence(int(seed))
    s_inv, s_ctl, s_mi, s_mc, s_pi, s_pc, s_obs = (int(x) for x in ss.generate_state(7) % (2**31 - 1) + 1)

    sim, ts = _inversion_ts(path, s_inv)
    i_nodes, s_nodes = S.sample_nodes_by_karyotype(sim, ts)
    body_win = (MARGIN_BP, MARGIN_BP + BODY_BP)
    mts = _mutate(ts, s_mi)
    body = SV.region_stats(mts, i_nodes, s_nodes, body_win, np.random.default_rng(s_pi))
    edge = SV.edge_centre(ts, i_nodes, s_nodes, body_win)

    cts = _mutate(_control_ts(s_ctl), s_mc)
    smp = cts.samples()
    control = SV.region_stats(cts, smp[N_STD:N_STD + N_INV], smp[:N_STD], (0.0, CONTROL_BP),
                              np.random.default_rng(s_pc))

    out = SV.assemble(body, control, edge)
    two_n = 2 * N_KARYO
    out["p_obs"] = float(np.random.default_rng(s_obs).binomial(two_n, float(path.p_now)) / two_n)
    out["n_nan"] = int(sum(math.isnan(out[k]) for k in SV.STAT_KEYS))
    return out
