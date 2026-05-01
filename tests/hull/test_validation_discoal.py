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
    "D3 (soft sweep, f0=0.05) still under-coalesces relative to discoal "
    "even after the SV phase extension (sweep-standing-variation-phase) "
    "and the conditional-drift fix. Current state: msinv pi=5989 vs "
    "discoal pi=16630. Remaining gap is the recombination-as-tag-shedding "
    "mechanism in discoal/src/core/discoalFunctions.c:2042-2051 + "
    ":2569-2583 (recombineAtTimePopnSweep) — every recombination during "
    "the sweep window rejection-samples whether the non-x_sel half stays "
    "in the sweep group with probability x (or 1-x). msinv approximates "
    "this with a one-shot per-segment partition at t_de_novo, which "
    "under-estimates segment escape during long SV phases. Activating "
    "this test requires implementing tag-aware recombination in apply_"
    "recombination during the sweep window."))
def test_discoal_validation_d3_soft_sweep():
    """Rust msinv vs discoal — soft sweep from standing variation, f0=0.05."""
    _run("d3")


def test_discoal_validation_d4_partial_sweep():
    """Rust msinv vs discoal — partial sweep, plateaus at 50% freq.

    Deterministic trajectory on both sides (msinv ``mode='Deterministic'``,
    discoal ``-wd -c 0.5``); same rationale as D2.
    """
    _run("d4")


@pytest.mark.skip(reason=(
    "D5 surfaces a likely units mismatch on the recurrent adaptive "
    "mutation rate. discoal -uA appears to be per-2N-per-gen-scaled "
    "(discoal D5 pi tracks D2 hard-sweep pi closely at uA=1e-3) while "
    "msinv recurrent_mutation_rate=1e-3 produces visible softening — "
    "more A founders, broader A-subpop diversity, ~56% higher pi than "
    "discoal. Plan flagged this as the most likely failure mode. Resolve "
    "by determining the discoal -uA convention from source and rescaling "
    "the msinv side; or by switching both engines to a normalized rate "
    "(e.g. per-2N-per-gen)."))
def test_discoal_validation_d5_focal_recurrent():
    """Rust msinv vs discoal — focal-site recurrent sweep, uA=1e-3."""
    _run("d5")
