"""discoal validation harness — Rust msinv core vs discoal v2.0.0-beta.

Spec: docs/superpowers/specs/2026-04-30-discoal-validation-design.md.

Each test calls ``_run_validation(scenario)`` which spawns two
subprocesses (one per engine) running ``tests/hull/_discoal_bench_runner``.
Per-rep stats arrive as JSON on the child's stdout; peak RSS is read
from ``os.wait4`` rusage.
"""

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
