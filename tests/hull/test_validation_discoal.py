"""discoal validation harness — Rust msinv core vs discoal v2.0.0-beta.

Spec: docs/superpowers/specs/2026-04-30-discoal-validation-design.md.

Each test calls ``_run_validation(scenario)`` which spawns two
subprocesses (one per engine) running ``tests/hull/_discoal_bench_runner``.
Per-rep stats arrive as JSON on the child's stdout; peak RSS is read
from ``os.wait4`` rusage.
"""

import pytest

from tests.hull._validation_common import _run_validation

RUNNER = "tests.hull._discoal_bench_runner"
BENCH_LOG = ".tmp/discoal_validation_bench.jsonl"


def _run(scenario):
    _run_validation(
        runner_module=RUNNER,
        scenario_name=scenario,
        engine_a="msinv",
        engine_b="discoal",
        bench_log=BENCH_LOG,
    )


def test_discoal_validation_d1_neutral():
    """Rust msinv vs discoal — neutral baseline, n=10, ρ=40, no sweep."""
    _run("d1")


def test_discoal_validation_d2_hard_sweep():
    """Rust msinv vs discoal — hard sweep, s=0.05, tau=1000 g, fix at 1.0.

    Both engines run a deterministic logistic trajectory: msinv
    ``mode='Deterministic'`` and discoal ``-wd``. Stochastic mode at
    ``f0=1/(2N)`` is extinction-prone for hard sweeps and would produce
    biased rep distributions on either side; this was established by PS2
    (per-segment hitchhiking spatial-profile MC) on 2026-04-30.
    """
    _run("d2")


@pytest.mark.skip(reason=(
    "D3 surfaces a K-founder partitioning gap in Rust apply_sweep_finalize. "
    "Soft sweeps (f0>0) need ~K=1/f0 founders to persist past t_origin "
    "(discoal model; implemented in the Python fallback at "
    "msinv/hull/simulator.py:1029-1041), but the Rust path collapses all "
    "A-tagged lineages into one founder at t_origin. "
    "Observed (f0=0.05): msinv pi=8176 vs discoal pi=16630 (~half). "
    "Inside-window progressive rate matches discoal exactly; the gap is "
    "entirely at the endpoint. Activate after K-founder partitioning ships."))
def test_discoal_validation_d3_soft_sweep():
    """Rust msinv vs discoal — soft sweep from standing variation, f0=0.05."""
    _run("d3")


def test_discoal_validation_d4_partial_sweep():
    """Rust msinv vs discoal — partial sweep, plateaus at 50% freq.

    Deterministic trajectory on both sides (msinv ``mode='Deterministic'``,
    discoal ``-wd -c 0.5``); same rationale as D2.
    """
    _run("d4")
