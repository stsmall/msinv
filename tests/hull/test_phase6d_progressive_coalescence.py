"""Progressive coalescence amplitude anchor.

After the 2026-04-30 progressive-coalescence extension, the global
mean pi_branch under a hard sweep should be substantially reduced
relative to the neutral expectation 4N.  Spec:
docs/superpowers/specs/2026-04-30-sweep-progressive-coalescence-design.md

PG-B1 emits per-allele coalescence rates inside the sweep window,
producing the Kim-Stephan-shaped temporal collapse of the A
subpopulation; the global pi reduction is the integrated
consequence over the full genome.
"""

import statistics

from msinv.hull.simulator import HullSimulator
from msinv.hull.sweep import Sweep


def _sim_factory(seed: int):
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary="S",
        target_inv=0,
        mode="Deterministic",
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10_000),
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10_000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def test_pg2_global_pi_below_neutral():
    """Mean pi_branch over 30 reps should be ≤ 70% of neutral 4N.

    A hard det sweep with s=0.05, rho=40 across a 100 kb genome drives
    a substantial global diversity reduction.  The exact Kim-Stephan
    expectation is parameter-dependent, so we use a conservative
    threshold (≤70% of neutral) that catches the case where the
    progressive rate fails to engage entirely (neutral pi).
    """
    N = 10_000
    neutral_pi = 4 * N  # branch-mode E[pi] for a single panmictic pop

    n_reps = 30
    pis = []
    for r in range(n_reps):
        ts = _sim_factory(seed=r)
        pis.append(ts.diversity(mode="branch"))
    mean_pi = statistics.mean(pis)
    ratio = mean_pi / neutral_pi
    print(
        f"PG2 mean pi: {mean_pi:.0f}, neutral 4N: {neutral_pi}, "
        f"ratio: {ratio * 100:.1f}%"
    )
    assert ratio < 0.7, f"Expected mean pi ≤70% of neutral 4N; got {ratio * 100:.1f}%"
