"""A sweep on one arrangement inside an active inversion (issue #22).

End-to-end check through the Python API: a partial sweep whose A allele
arose on I (``origin_kary="I"``, no gene flux) runs without the former
crossover-suppression warning, and I x S divergence at the selected site
stays above diversity within either arrangement. The mechanisms (escape
probability at ``r * p_k``, the tag swap drawing against the child's own
arrangement, karyotype-compatible founder merges) are pinned exactly by
the Rust unit tests in ``simulator.rs::sweep_inversion_tests``.

The partial sweep keeps S common: with no gene flux the A allele is
locked to I, so a complete sweep also drives I towards fixation in the
sweep trajectory and leaves almost no S class at the sampling time.
"""

import warnings

import numpy as np

from msinv.hull.inversion import InversionSpec
from msinv.hull.simulator import HullSimulator
from msinv.hull.sweep import Sweep

N = 10_000.0
L = 200_000.0
X_SEL = L / 2


def _sim(seed):
    inv = InversionSpec(bp_left=0.0, bp_right=L, p_inv=0.3, t_inv=4 * N)
    sweep = Sweep(
        x_sel=X_SEL, tau=0.0, origin_pop=0, origin_kary="I", target_inv=0,
        mode="Deterministic", s=0.05, t_origin=1_000.0, f0=1 / (2 * N),
        partial_sweep_final_freq=0.25, seed=seed,
    )
    return HullSimulator(
        n_std=6, n_inv=6, population_size=N, sequence_length=L,
        recombination_rate=1e-8, inversions=[inv], sweeps=[sweep], seed=seed,
    ).simulate().simplify()


def test_sweep_on_one_arrangement_keeps_arrangements_differentiated():
    s_nodes, i_nodes = list(range(6)), list(range(6, 12))
    win = [0.0, X_SEL - 5_000, X_SEL + 5_000, L]
    pi_i, pi_s, dxy = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in range(1, 11):
            ts = _sim(seed)
            pi_i.append(ts.diversity([i_nodes], windows=win, mode="branch")[1, 0])
            pi_s.append(ts.diversity([s_nodes], windows=win, mode="branch")[1, 0])
            dxy.append(ts.divergence([i_nodes, s_nodes], windows=win,
                                     mode="branch")[1])
    pi_i, pi_s, dxy = map(np.mean, (pi_i, pi_s, dxy))
    assert np.isfinite([pi_i, pi_s, dxy]).all()
    assert dxy > 2 * max(pi_i, pi_s), (dxy, pi_i, pi_s)
