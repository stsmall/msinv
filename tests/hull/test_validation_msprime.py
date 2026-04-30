"""msprime validation harness — Rust msinv core vs msprime.sim_ancestry.

Spec: docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md
(supersedes 2026-04-29-msprime-validation-design.md).

Each test calls ``_run_validation(scenario)`` which spawns two
subprocesses (one per engine) running ``tests/hull/_msprime_bench_runner``.
Per-rep stats arrive as JSON on the child's stdout; peak RSS is read
from ``os.wait4`` rusage. Pass criteria:

- moment stats (``pi_branch``, ``n_trees``, ``mean_tmrca``, ``dxy_branch``):
  ``|Δ| <= 3 * sqrt(SE_a^2 + SE_b^2)``
- AFS bin stats (``afs_*``): Bonferroni-corrected two-sided z, family-
  wise α = 0.003 across all AFS bins in that scenario.
"""

from tests.hull._validation_common import _run_validation

RUNNER = "tests.hull._msprime_bench_runner"
BENCH_LOG = ".tmp/msprime_validation_bench.jsonl"


def _run(scenario):
    _run_validation(
        runner_module=RUNNER,
        scenario_name=scenario,
        engine_a="msinv",
        engine_b="msprime",
        bench_log=BENCH_LOG,
    )


def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""
    _run("n1")


def test_msprime_validation_n2_two_pop_migration():
    """Rust msinv vs msprime — two-pop symmetric migration, M=1e-4."""
    _run("n2")


def test_msprime_validation_n3_two_pop_split():
    """Rust msinv vs msprime — two-pop merge backward at T=2000."""
    _run("n3")


def test_msprime_validation_n4_bottleneck():
    """Rust msinv vs msprime — Ne=10000 → 1000 (1000–2000 gens) → 10000."""
    _run("n4")


def test_msprime_validation_n5_exponential_growth():
    """Rust msinv vs msprime — exponential growth, α=0.0005/gen."""
    _run("n5")


def test_msprime_validation_n6_three_pop_with_split():
    """Rust msinv vs msprime — 3 pops, sym M=5e-5, merge to A at T=3000."""
    _run("n6")
