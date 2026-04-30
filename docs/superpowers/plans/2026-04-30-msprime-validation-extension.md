# msprime validation extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the msprime validation harness with four new scenarios (N3 split, N4 bottleneck, N5 growth, N6 three-pop split) and a per-engine wall-clock + peak-RSS benchmark layer, isolated via subprocess.

**Architecture:** Existing N1/N2 are preserved and migrate to the new dispatch pattern: `_run_validation(scenario_name)` spawns one child subprocess per engine (`tests/hull/_msprime_bench_runner.py`), each running the full 200-rep batch and printing per-rep stats as JSON to stdout. Parent reads JSON, applies pass criteria (3·SE on moments, Bonferroni-z on AFS bins), reads `os.wait4` rusage for clean per-engine peak RSS, and prints a per-scenario benchmark block. Persistent run log goes to `.tmp/msprime_validation_bench.jsonl`.

**Tech Stack:** Python 3.12 (`.venv`), pytest, msprime, msinv (Rust core via PyO3), tskit, stdlib `statistics.NormalDist`, `subprocess`, `os.wait4`, `resource.getrusage`.

**Spec:** `docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md`

**Predecessor (do not regress):** `tests/hull/test_validation_msprime.py` — N1/N2 tests must stay green at every commit.

---

## Phase A — Refactor harness to subprocess pattern

These tasks migrate the existing N1/N2 tests to the new dispatch pattern without changing their pass criteria. Existing tests must pass after every commit in this phase.

### Task A1: Skeleton child runner with no-op scenario

**Files:**
- Create: `tests/hull/_msprime_bench_runner.py`

- [ ] **Step 1: Create the runner skeleton**

```python
"""Child-process runner for the msprime validation harness.

Invoked as ``python -m tests.hull._msprime_bench_runner --scenario NAME
--engine {msinv,msprime} --n-reps N --seed-base K``. Runs the rep batch
and writes a JSON document to stdout::

    {"per_rep_stats": [{stat_name: value, ...}, ...], "per_rep_seconds": [float, ...]}

Importing this module does NOT execute any sim. It is pytest-collectable
but pytest-skipped — pytest sees ``pytestmark = pytest.mark.skip(...)``.
"""

import argparse
import json
import sys
import time

import pytest

pytestmark = pytest.mark.skip("child runner — invoked via subprocess")


# Scenario registry filled in by Phase A2/A3 and Phase C tasks.
# Each entry: name -> {
#   "compute_afs": bool,
#   "n_pops": int,
#   "make_msinv": Callable[[int], (ts, sample_sets_or_None)],
#   "make_msprime": Callable[[int], (ts, sample_sets_or_None)],
# }
SCENARIOS: dict[str, dict] = {}


def _stats_from_ts(ts, sample_sets, compute_afs: bool):
    """Branch-length stats + (optional) AFS bins from a tskit TS.

    Returns a dict suitable for JSON serialization. Single-pop AFS bins
    are keyed ``afs_bin_{k}``; multi-pop marginals are keyed
    ``afs_p{p}_bin_{k}``. Edge bins (0 and n_set) are excluded.
    """
    out: dict[str, float] = {
        "pi_branch": ts.diversity(mode="branch"),
        "n_trees": float(ts.num_trees),
    }
    weighted = sum(tree.time(tree.root) * tree.span for tree in ts.trees())
    out["mean_tmrca"] = weighted / ts.sequence_length
    if sample_sets is not None:
        out["dxy_branch"] = _mean_pairwise_divergence(ts, sample_sets)
    if compute_afs:
        if sample_sets is None:
            afs = ts.allele_frequency_spectrum(
                mode="branch", polarised=True, span_normalise=True)
            for k in range(1, len(afs) - 1):
                out[f"afs_bin_{k}"] = float(afs[k])
        else:
            for p, samples in enumerate(sample_sets):
                afs = ts.allele_frequency_spectrum(
                    sample_sets=[samples], mode="branch",
                    polarised=True, span_normalise=True)
                for k in range(1, len(afs) - 1):
                    out[f"afs_p{p}_bin_{k}"] = float(afs[k])
    return out


def _mean_pairwise_divergence(ts, sample_sets):
    """For 2-pop scenarios, return the single A-B branch divergence.
    For 3-pop, return the mean of the three pairwise divergences.
    """
    n = len(sample_sets)
    if n == 2:
        return float(ts.divergence(sample_sets=sample_sets, mode="branch"))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    vals = [
        float(ts.divergence(
            sample_sets=[sample_sets[i], sample_sets[j]], mode="branch"))
        for i, j in pairs
    ]
    return sum(vals) / len(vals)


def _run_one_engine(scenario_name: str, engine: str,
                    n_reps: int, seed_base: int):
    spec = SCENARIOS[scenario_name]
    factory = (spec["make_msinv"] if engine == "msinv"
               else spec["make_msprime"])
    compute_afs = spec["compute_afs"]
    # warm-up rep, untimed (engine import / .so warm-up)
    factory(seed=seed_base)
    per_rep_stats = []
    per_rep_seconds = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        ts, sample_sets = factory(seed=seed_base + i)
        stats = _stats_from_ts(ts, sample_sets, compute_afs)
        per_rep_seconds.append(time.perf_counter() - t0)
        per_rep_stats.append(stats)
    return per_rep_stats, per_rep_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--engine", required=True, choices=["msinv", "msprime"])
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()
    per_rep_stats, per_rep_seconds = _run_one_engine(
        args.scenario, args.engine, args.n_reps, args.seed_base)
    json.dump(
        {"per_rep_stats": per_rep_stats,
         "per_rep_seconds": per_rep_seconds},
        sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

**Note:** `argparse.ArgumentParser`'s `choices=list(SCENARIOS)` is evaluated at *parse time*, after `SCENARIOS` is populated by other tasks importing the module. The empty registry at this point would make `--scenario X` fail in argparse, which is fine — Task A2 adds entries.

- [ ] **Step 2: Sanity-check that import does not error**

Run: `.venv/bin/python -c "import tests.hull._msprime_bench_runner as r; print(list(r.SCENARIOS))"`
Expected: prints `[]` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py
git commit -m "test: msprime validation runner skeleton

Empty scenario registry, CLI dispatch, _stats_from_ts with optional AFS,
warm-up rep before timed batch. No scenarios yet — added in Phase A2/A3
and Phase C.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Move N1 scenario into runner registry

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N1 to SCENARIOS)

- [ ] **Step 1: Append N1 registration to the runner module**

Below the `SCENARIOS = {}` line in `tests/hull/_msprime_bench_runner.py`, add:

```python
import msprime
from msinv.hull.simulator import HullSimulator


def _make_n1_msinv(seed: int):
    ts = HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n1_msprime(seed: int):
    # population_size doubled vs msinv: msinv N = diploid Ne (2N chrom);
    # msprime ploidy=1 reads N as haploid Ne. record_full_arg=True so
    # non-ancestral recombs survive into the TS (msinv's convention).
    ts = msprime.sim_ancestry(
        samples=10,
        population_size=20000.0,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n1"] = {
    "compute_afs": False,
    "n_pops": 1,
    "make_msinv": _make_n1_msinv,
    "make_msprime": _make_n1_msprime,
}
```

- [ ] **Step 2: Run a tiny end-to-end against the runner CLI**

Run: `.venv/bin/python -m tests.hull._msprime_bench_runner --scenario n1 --engine msinv --n-reps 2 --seed-base 0`
Expected: prints a single JSON line containing `"per_rep_stats"` (list of 2 dicts with `pi_branch`, `n_trees`, `mean_tmrca`) and `"per_rep_seconds"` (list of 2 floats).

Run: `.venv/bin/python -m tests.hull._msprime_bench_runner --scenario n1 --engine msprime --n-reps 2 --seed-base 0`
Expected: same shape, same keys.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py
git commit -m "test: register N1 scenario in msprime validation runner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A3: Move N2 scenario into runner registry

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N2)

- [ ] **Step 1: Add N2 registration below N1**

```python
from msinv.hull.demography import Demography


def _samples_by_pop(ts, n_pops: int):
    populations = ts.tables.nodes.population[ts.samples()]
    return [
        ts.samples()[populations == p].tolist() for p in range(n_pops)]


def _make_n2_msinv(seed: int):
    demo = Demography(
        pop_sizes=[10000.0, 10000.0],
        migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
    )
    ts = HullSimulator(
        sample_config={(None, 0): 5, (None, 1): 5},
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, _samples_by_pop(ts, n_pops=2)


def _make_n2_msprime(seed: int):
    # population sizes doubled; migration rate NOT rescaled
    # (per-lineage per-gen on both sides).
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population(name="B", initial_size=20000.0)
    demo.set_migration_rate(source="A", dest="B", rate=1e-4)
    demo.set_migration_rate(source="B", dest="A", rate=1e-4)
    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, _samples_by_pop(ts, n_pops=2)


SCENARIOS["n2"] = {
    "compute_afs": False,
    "n_pops": 2,
    "make_msinv": _make_n2_msinv,
    "make_msprime": _make_n2_msprime,
}
```

- [ ] **Step 2: Sanity-check that the runner returns N2 stats**

Run: `.venv/bin/python -m tests.hull._msprime_bench_runner --scenario n2 --engine msinv --n-reps 2 --seed-base 0`
Expected: JSON with `pi_branch`, `n_trees`, `mean_tmrca`, **and** `dxy_branch` keys.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py
git commit -m "test: register N2 scenario in msprime validation runner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A4: Replace inline rep loop with subprocess invocation

**Files:**
- Modify: `tests/hull/test_validation_msprime.py`

- [ ] **Step 1: Replace the body of `_run_validation` with subprocess dispatch**

Read the current `tests/hull/test_validation_msprime.py` first (you'll need its imports + the existing N1/N2 test bodies). The new version of `_run_validation(scenario_name, n_reps=200)` does **not** know how to build a TS itself — it shells out to the runner module. The N1/N2 test bodies become tiny (just call `_run_validation("n1")` / `_run_validation("n2")`).

Replace the file contents with:

```python
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

import json
import math
import os
import resource
import statistics
import subprocess
import sys


N_REPS = 200
ALPHA_FAMILY = 0.003  # family-wise α for both moment and AFS families


def _run_one_engine(scenario_name, engine, n_reps):
    """Spawn one child runner process; return (per_rep_stats,
    per_rep_seconds, peak_rss_kb)."""
    cmd = [
        sys.executable, "-m", "tests.hull._msprime_bench_runner",
        "--scenario", scenario_name, "--engine", engine,
        "--n-reps", str(n_reps), "--seed-base", "0",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    pid_unused, status, rusage = os.wait4(proc.pid, os.WNOHANG)  # drained
    # The above wait4 may return (0, 0, ...) if Popen.communicate already
    # reaped the process; in that case use Popen-collected info.
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed for {scenario_name}/{engine} "
            f"(rc={proc.returncode}):\n{stderr.decode()}")
    payload = json.loads(stdout.decode())
    # Resort to RUSAGE_CHILDREN incremental delta because Popen.communicate
    # already wait()ed the child (so os.wait4 above returned 0). Read the
    # cumulative children rusage; the caller is responsible for taking
    # deltas across calls.
    return payload["per_rep_stats"], payload["per_rep_seconds"]


def _peak_rss_after(prev_kb):
    """Return (current_cumulative_max, delta_for_latest_child).

    On Linux, RUSAGE_CHILDREN.ru_maxrss is the maximum RSS of any single
    waited-for child since process start (NOT cumulative). After running
    children sequentially, the latest child's peak is either:
      - exactly current_max (if it exceeded all previous), OR
      - bounded above by current_max (if any previous child was larger).

    Returns a tuple where delta is None when we cannot disambiguate.
    """
    cur = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if cur > prev_kb:
        return cur, cur  # this child set the new max — its peak is `cur`
    return cur, None  # this child's peak is <= cur (we report '<= prev_kb')


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _bonferroni_z(k_bins, alpha=ALPHA_FAMILY):
    """Two-sided z bound for `k_bins` AFS bins at family-wise alpha."""
    per_bin = alpha / k_bins
    return statistics.NormalDist().inv_cdf(1.0 - per_bin / 2.0)


def _agg_engine_vals(per_rep_stats):
    """List of per-rep stat dicts -> dict[stat_name, list[value]]."""
    out: dict[str, list[float]] = {}
    for rep_stats in per_rep_stats:
        for k, v in rep_stats.items():
            out.setdefault(k, []).append(v)
    return out


def _run_validation(scenario_name, n_reps=N_REPS):
    """Run both engines via subprocess; assert per-stat agreement."""
    rss0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    msinv_stats, msinv_secs = _run_one_engine(scenario_name, "msinv", n_reps)
    rss1, msinv_peak = _peak_rss_after(rss0)
    msprime_stats, msprime_secs = _run_one_engine(
        scenario_name, "msprime", n_reps)
    rss2, msprime_peak = _peak_rss_after(rss1)

    a = _agg_engine_vals(msinv_stats)
    b = _agg_engine_vals(msprime_stats)
    keys = list(a.keys())
    assert set(keys) == set(b.keys()), (
        f"stat key mismatch: msinv={set(keys)} vs msprime={set(b.keys())}")

    afs_keys = [k for k in keys if k.startswith("afs_")]
    moment_keys = [k for k in keys if not k.startswith("afs_")]
    z_afs = _bonferroni_z(len(afs_keys)) if afs_keys else None

    failures = []
    lines = []
    for k in moment_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = 3.0 * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: msinv={m_a:.4g} ± {se_a:.3g}, "
                f"msprime={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, 3·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)
    for k in afs_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = z_afs * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: msinv={m_a:.4g} ± {se_a:.3g}, "
                f"msprime={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, {z_afs:.2f}·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)

    print(f"\n[{scenario_name}]")
    for line in lines:
        print(f"  {line}")

    _print_benchmark_block(
        scenario_name, msinv_secs, msprime_secs, msinv_peak, msprime_peak)

    if failures:
        raise AssertionError(
            f"\n[{scenario_name}]\n  " + "\n  ".join(lines))


def _print_benchmark_block(scenario_name, msinv_secs, msprime_secs,
                           msinv_peak_kb, msprime_peak_kb):
    """Print a per-scenario benchmark line. Visible only with pytest -s."""
    m_mean, m_se = _mean_se(msinv_secs)
    p_mean, p_se = _mean_se(msprime_secs)
    m_total = sum(msinv_secs)
    p_total = sum(msprime_secs)
    m_rss_mb = (msinv_peak_kb / 1024.0
                if msinv_peak_kb is not None else float("nan"))
    p_rss_mb = (msprime_peak_kb / 1024.0
                if msprime_peak_kb is not None else float("nan"))
    print(f"[{scenario_name}] benchmarks")
    print(f"  msinv:   per-rep {m_mean*1000:6.1f} ms ± {m_se*1000:.1f}, "
          f"total {m_total:5.1f} s, peak RSS {m_rss_mb:6.1f} MB")
    print(f"  msprime: per-rep {p_mean*1000:6.1f} ms ± {p_se*1000:.1f}, "
          f"total {p_total:5.1f} s, peak RSS {p_rss_mb:6.1f} MB")
    if math.isfinite(p_mean) and p_mean > 0:
        print(f"  ratio:   per-rep msinv/msprime = {m_mean/p_mean:.2f}x;  "
              f"RAM msinv/msprime = "
              f"{m_rss_mb/p_rss_mb:.2f}x" if math.isfinite(m_rss_mb) and
              math.isfinite(p_rss_mb) and p_rss_mb > 0 else "")


def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""
    _run_validation("n1")


def test_msprime_validation_n2_two_pop_migration():
    """Rust msinv vs msprime — two-pop symmetric migration, M=1e-4."""
    _run_validation("n2")
```

- [ ] **Step 2: Run the existing N1/N2 tests, expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s`
Expected: 2 passed. Per-test output shows the per-stat OK lines. Total runtime in the 10–20 s range (was ~5 s before — subprocess overhead).

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_validation_msprime.py
git commit -m "test: refactor msprime harness to subprocess dispatch

Existing N1/N2 tests now call _run_validation(scenario_name) which
shells out to tests.hull._msprime_bench_runner per engine. Per-engine
peak RSS read from RUSAGE_CHILDREN deltas (Linux ru_maxrss).

Pass criteria unchanged: 3·SE on moment stats; AFS family is wired
but no scenario currently sets compute_afs=True.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A5: Add `os.wait4`-based RSS attribution fallback

**Files:**
- Modify: `tests/hull/test_validation_msprime.py:_run_one_engine`

The previous task uses `Popen.communicate()` which calls `wait()`
internally; by the time we'd reach `os.wait4` the child is already
reaped, so `_peak_rss_after` falls back to `RUSAGE_CHILDREN` deltas.
That works on Linux for *increasing* peaks but is bounded-only when
a later child has a smaller peak than an earlier one. This task
swaps to `os.waitpid`-aware spawn so we read the *exact* per-child
rusage.

- [ ] **Step 1: Replace `_run_one_engine` with a manual fork + wait4 path**

In `tests/hull/test_validation_msprime.py`, replace `_run_one_engine` with:

```python
def _run_one_engine(scenario_name, engine, n_reps):
    """Spawn one child runner; return (per_rep_stats, per_rep_seconds,
    peak_rss_kb). Uses Popen + os.wait4 for clean per-child rusage."""
    cmd = [
        sys.executable, "-m", "tests.hull._msprime_bench_runner",
        "--scenario", scenario_name, "--engine", engine,
        "--n-reps", str(n_reps), "--seed-base", "0",
    ]
    # Popen does not auto-wait; we'll do it manually with os.wait4
    # so the per-child rusage is attributable.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Read all stdout / stderr without calling .wait()
    stdout = proc.stdout.read()
    stderr = proc.stderr.read()
    proc.stdout.close()
    proc.stderr.close()
    pid, status, rusage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed for {scenario_name}/{engine} "
            f"(rc={proc.returncode}):\n{stderr.decode()}")
    payload = json.loads(stdout.decode())
    peak_kb = rusage.ru_maxrss  # Linux: KB
    return payload["per_rep_stats"], payload["per_rep_seconds"], peak_kb
```

And update the call sites in `_run_validation`:

```python
    msinv_stats, msinv_secs, msinv_peak = _run_one_engine(
        scenario_name, "msinv", n_reps)
    msprime_stats, msprime_secs, msprime_peak = _run_one_engine(
        scenario_name, "msprime", n_reps)
```

Delete `_peak_rss_after` and the surrounding `rss0/rss1/rss2` plumbing — they're obsolete.

- [ ] **Step 2: Run N1/N2; expect benchmarks to print clean per-engine peak RSS**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s`
Expected: 2 passed. Benchmark block now shows distinct peak RSS per engine
(e.g. `msinv: peak RSS 47.2 MB`, `msprime: peak RSS 142.7 MB`), no NaN.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_validation_msprime.py
git commit -m "test: clean per-child peak RSS via os.wait4

Replace Popen.communicate() (which calls wait() internally) with
manual stream drain + os.wait4, so RUSAGE_CHILDREN reads the exact
peak for that single child rather than the cumulative max.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A6: Append benchmark JSONL persistence

**Files:**
- Modify: `tests/hull/test_validation_msprime.py:_print_benchmark_block`

- [ ] **Step 1: Update `_print_benchmark_block` to also append to `.tmp/msprime_validation_bench.jsonl`**

In `tests/hull/test_validation_msprime.py`, update `_print_benchmark_block` to also append a JSON line. Add these imports at the top (`datetime`, `pathlib`, `subprocess`):

```python
import datetime
import pathlib
```

(`subprocess` is already imported.) Then update `_print_benchmark_block`:

```python
def _git_short_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def _print_benchmark_block(scenario_name, msinv_secs, msprime_secs,
                           msinv_peak_kb, msprime_peak_kb):
    """Print + persist a per-scenario benchmark line."""
    m_mean, m_se = _mean_se(msinv_secs)
    p_mean, p_se = _mean_se(msprime_secs)
    m_total = sum(msinv_secs)
    p_total = sum(msprime_secs)
    m_rss_mb = msinv_peak_kb / 1024.0
    p_rss_mb = msprime_peak_kb / 1024.0
    print(f"[{scenario_name}] benchmarks")
    print(f"  msinv:   per-rep {m_mean*1000:6.1f} ms ± {m_se*1000:.1f}, "
          f"total {m_total:5.1f} s, peak RSS {m_rss_mb:6.1f} MB")
    print(f"  msprime: per-rep {p_mean*1000:6.1f} ms ± {p_se*1000:.1f}, "
          f"total {p_total:5.1f} s, peak RSS {p_rss_mb:6.1f} MB")
    print(f"  ratio:   per-rep msinv/msprime = {m_mean/p_mean:.2f}x;  "
          f"RAM msinv/msprime = {m_rss_mb/p_rss_mb:.2f}x")

    log_dir = pathlib.Path(".tmp")
    log_dir.mkdir(exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "scenario": scenario_name,
        "msinv": {"per_rep_s": m_mean, "per_rep_se": m_se,
                  "total_s": m_total, "peak_rss_mb": m_rss_mb},
        "msprime": {"per_rep_s": p_mean, "per_rep_se": p_se,
                    "total_s": p_total, "peak_rss_mb": p_rss_mb},
    }
    with (log_dir / "msprime_validation_bench.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 2: Run N1/N2 and verify the JSONL appends**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s`
Expected: 2 passed.

Run: `tail -2 .tmp/msprime_validation_bench.jsonl`
Expected: two JSON lines, one per scenario, each with `scenario`, `msinv`, `msprime`, `git_sha`, `ts`.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_validation_msprime.py
git commit -m "test: persist per-scenario benchmarks to .tmp JSONL

Appends one JSON line per scenario per run with timestamp, git SHA,
per-engine wall-clock and peak RSS. Not committed; .tmp/ is local
scratch (per feedback_local_tmp.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — AFS family stat support

`_stats_from_ts` in the runner already computes AFS bins when
`compute_afs=True`. The harness already routes AFS keys to the
Bonferroni bound. This phase adds a small unit test for the
Bonferroni z calculation and confirms AFS bins propagate correctly
via a one-off integration smoke test.

### Task B1: Unit test for `_bonferroni_z`

**Files:**
- Create: `tests/hull/test_validation_msprime_helpers.py`

- [ ] **Step 1: Write the unit test**

```python
"""Unit tests for the small math helpers in test_validation_msprime."""

import pytest

from tests.hull.test_validation_msprime import _bonferroni_z


def test_bonferroni_z_matches_spec_table():
    # Spec table values (rounded to 2 dp).
    # K=7 → 3.52; K=8 → 3.55; K=9 → 3.59 at α=0.003.
    assert _bonferroni_z(7) == pytest.approx(3.52, abs=0.01)
    assert _bonferroni_z(8) == pytest.approx(3.55, abs=0.01)
    assert _bonferroni_z(9) == pytest.approx(3.59, abs=0.01)


def test_bonferroni_z_monotone_in_k():
    """More bins → tighter per-bin α → larger z."""
    zs = [_bonferroni_z(k) for k in range(2, 30)]
    assert zs == sorted(zs)
```

- [ ] **Step 2: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime_helpers.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_validation_msprime_helpers.py
git commit -m "test: unit test Bonferroni z spec values

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Add scenarios N3, N4, N5, N6

Each scenario task: register factories in the runner, add the
pytest function, run it. Pass = msinv and msprime statistics agree
within their bounds (no implementation needed beyond wiring; both
engines already implement the demographic primitives being tested).

### Task C1: N3 — two-pop merge backward (`ej`)

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N3)
- Modify: `tests/hull/test_validation_msprime.py` (add `test_msprime_validation_n3_two_pop_split`)

- [ ] **Step 1: Add N3 factories to the runner**

Append to `tests/hull/_msprime_bench_runner.py` after the N2 block:

```python
def _make_n3_msinv(seed: int):
    demo = Demography(
        pop_sizes=[10000.0, 10000.0],
        migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
    )
    demo.add_population_split(time=2000.0, derived=[1], ancestral=0)
    ts = HullSimulator(
        sample_config={(None, 0): 5, (None, 1): 5},
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, _samples_by_pop(ts, n_pops=2)


def _make_n3_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population(name="B", initial_size=20000.0)
    demo.set_migration_rate(source="A", dest="B", rate=1e-4)
    demo.set_migration_rate(source="B", dest="A", rate=1e-4)
    # mass_migration with proportion=1.0, NOT add_population_split (which
    # auto-derives growth/size resets that don't match msinv ej semantics).
    demo.add_mass_migration(
        time=2000.0, source="B", dest="A", proportion=1.0)
    # Zero all migration at T=2000 to match msinv ej semantics.
    # add_mass_migration leaves the migration matrix active, so without
    # this line msprime would keep M[B][A]=1e-4 sending lineages from
    # populated A back into empty B above T=2000 (~2x Ne inflation).
    # Order matters — must come AFTER add_mass_migration; msprime
    # applies simultaneous events in insertion order.
    demo.add_migration_rate_change(time=2000.0, rate=0.0)
    ts = msprime.sim_ancestry(
        samples={"A": 5, "B": 5},
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, _samples_by_pop(ts, n_pops=2)


SCENARIOS["n3"] = {
    "compute_afs": True,
    "n_pops": 2,
    "make_msinv": _make_n3_msinv,
    "make_msprime": _make_n3_msprime,
}
```

- [ ] **Step 2: Add the pytest function**

Append to the bottom of `tests/hull/test_validation_msprime.py`:

```python
def test_msprime_validation_n3_two_pop_split():
    """Rust msinv vs msprime — two-pop merge backward at T=2000."""
    _run_validation("n3")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n3_two_pop_split -v -s`
Expected: pass. Output shows OK on `pi_branch`, `n_trees`, `mean_tmrca`, `dxy_branch` (3·SE) and 8 AFS bins (`afs_p0_bin_1..4`, `afs_p1_bin_1..4`) at the K=8 bound (`3.55·SE`). Total runtime ~6 s.

If a stat fails, do NOT widen the bound. Triage table is in the spec; investigate whether the failure is a real msinv regression (check `git log --since="last week"` for `simulator.rs`/`demography.rs` changes) or a convention bug (re-check the doubling/`record_full_arg`/`mass_migration` choices vs spec).

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py tests/hull/test_validation_msprime.py
git commit -m "test: msprime validation N3 — two-pop merge backward

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C2: N4 — bottleneck (`en`)

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N4)
- Modify: `tests/hull/test_validation_msprime.py` (add test)

- [ ] **Step 1: Add N4 factories**

Append to `tests/hull/_msprime_bench_runner.py` after N3:

```python
def _make_n4_msinv(seed: int):
    demo = Demography(pop_sizes=[10000.0])
    demo.add_population_size_change(
        time=1000.0, population=0, new_size=1000.0)
    demo.add_population_size_change(
        time=2000.0, population=0, new_size=10000.0)
    ts = HullSimulator(
        samples=10,
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n4_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population_parameters_change(
        time=1000.0, initial_size=2000.0, population="A")
    demo.add_population_parameters_change(
        time=2000.0, initial_size=20000.0, population="A")
    ts = msprime.sim_ancestry(
        samples=10,
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n4"] = {
    "compute_afs": True,
    "n_pops": 1,
    "make_msinv": _make_n4_msinv,
    "make_msprime": _make_n4_msprime,
}
```

- [ ] **Step 2: Add the pytest function**

```python
def test_msprime_validation_n4_bottleneck():
    """Rust msinv vs msprime — Ne=10000 → 1000 (1000–2000 gens) → 10000."""
    _run_validation("n4")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n4_bottleneck -v -s`
Expected: pass. K=9 AFS bins (`afs_bin_1..9`) at `3.59·SE`.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py tests/hull/test_validation_msprime.py
git commit -m "test: msprime validation N4 — bottleneck (en)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C3: N5 — exponential growth (`eg`)

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N5)
- Modify: `tests/hull/test_validation_msprime.py` (add test)

- [ ] **Step 1: Add N5 factories**

Append to the runner after N4:

```python
def _make_n5_msinv(seed: int):
    demo = Demography(pop_sizes=[10000.0])
    demo.add_growth_rate_change(
        time=0.0, population=0, growth_rate=0.0005)
    ts = HullSimulator(
        samples=10,
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, None


def _make_n5_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(
        name="A", initial_size=20000.0, growth_rate=0.0005)
    ts = msprime.sim_ancestry(
        samples=10,
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, None


SCENARIOS["n5"] = {
    "compute_afs": True,
    "n_pops": 1,
    "make_msinv": _make_n5_msinv,
    "make_msprime": _make_n5_msprime,
}
```

- [ ] **Step 2: Add the pytest function**

```python
def test_msprime_validation_n5_exponential_growth():
    """Rust msinv vs msprime — exponential growth, α=0.0005/gen."""
    _run_validation("n5")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n5_exponential_growth -v -s`
Expected: pass.

If `pi_branch` low + `afs_bin_1` high, the `eg` sign convention may differ between engines — inspect the docstrings (msinv `demography.py:259`, msprime help on `add_population` `growth_rate`) before adjusting bounds.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py tests/hull/test_validation_msprime.py
git commit -m "test: msprime validation N5 — exponential growth (eg)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C4: N6 — three-pop with split

**Files:**
- Modify: `tests/hull/_msprime_bench_runner.py` (add N6)
- Modify: `tests/hull/test_validation_msprime.py` (add test)

- [ ] **Step 1: Add N6 factories**

Append to the runner after N5:

```python
def _make_n6_msinv(seed: int):
    demo = Demography(
        pop_sizes=[10000.0, 10000.0, 10000.0],
        migration_matrix=[
            [0.0,  5e-5, 5e-5],
            [5e-5, 0.0,  5e-5],
            [5e-5, 5e-5, 0.0 ],
        ],
    )
    demo.add_population_split(time=3000.0, derived=[1, 2], ancestral=0)
    ts = HullSimulator(
        sample_config={(None, 0): 4, (None, 1): 3, (None, 2): 3},
        demography=demo,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        seed=seed,
    ).simulate()
    return ts, _samples_by_pop(ts, n_pops=3)


def _make_n6_msprime(seed: int):
    demo = msprime.Demography()
    demo.add_population(name="A", initial_size=20000.0)
    demo.add_population(name="B", initial_size=20000.0)
    demo.add_population(name="C", initial_size=20000.0)
    pairs = [("A", "B"), ("B", "A"), ("A", "C"),
             ("C", "A"), ("B", "C"), ("C", "B")]
    for src, dst in pairs:
        demo.set_migration_rate(source=src, dest=dst, rate=5e-5)
    demo.add_mass_migration(
        time=3000.0, source="B", dest="A", proportion=1.0)
    demo.add_mass_migration(
        time=3000.0, source="C", dest="A", proportion=1.0)
    # Match msinv ej semantics: msprime add_mass_migration leaves the
    # migration matrix active, so without this line A↔B and A↔C migrate
    # above T=3000 (populated A → empty B/C), inflating Ne. Order
    # matters — this must come AFTER both add_mass_migration calls,
    # because msprime applies simultaneous events in insertion order.
    # Global form (no source/dest) zeros all off-diagonal rates.
    # See N3 spec/comment for the original derivation.
    demo.add_migration_rate_change(time=3000.0, rate=0.0)
    ts = msprime.sim_ancestry(
        samples={"A": 4, "B": 3, "C": 3},
        demography=demo,
        sequence_length=100_000,
        recombination_rate=1e-8,
        ploidy=1,
        record_full_arg=True,
        random_seed=seed + 1,
    )
    return ts, _samples_by_pop(ts, n_pops=3)


SCENARIOS["n6"] = {
    "compute_afs": True,
    "n_pops": 3,
    "make_msinv": _make_n6_msinv,
    "make_msprime": _make_n6_msprime,
}
```

- [ ] **Step 2: Add the pytest function**

```python
def test_msprime_validation_n6_three_pop_with_split():
    """Rust msinv vs msprime — 3 pops, sym M=5e-5, merge to A at T=3000."""
    _run_validation("n6")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n6_three_pop_with_split -v -s`
Expected: pass. K=7 AFS bins (3 + 2 + 2) at `3.52·SE`. Total runtime ~9 s.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_msprime_bench_runner.py tests/hull/test_validation_msprime.py
git commit -m "test: msprime validation N6 — three-pop with split

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase D — Full validation pass

### Task D1: Full harness run + budget check + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (bump test counts and harness description)

- [ ] **Step 1: Run the entire harness with timing**

Run: `time .venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s 2>&1 | tail -80`
Expected:
- 6 passed (n1..n6).
- All per-scenario benchmark blocks visible: `[scenario] benchmarks` with msinv + msprime lines.
- Total wall-clock under 180 s (the spec budget; expected real time at `N_REPS=200` is ~35 s, leaving ~5× headroom for future drift-sharpening N bumps). If over 180 s, do NOT lower `N_REPS` reflexively — read the per-scenario benchmark blocks to find the slow one and triage (likely an msinv perf regression or a pathological demographic param choice).

- [ ] **Step 2: Run the full Python suite to make sure nothing else regressed**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3`
Expected: 187+ passed, 3 skipped (was 183 passed at start; +4 new tests + 2 helper tests = +6, but minus 2 if helper file was its own test count). Verify the count moved by exactly the expected amount.

- [ ] **Step 3: Sanity-check the JSONL log**

Run: `wc -l .tmp/msprime_validation_bench.jsonl && tail -1 .tmp/msprime_validation_bench.jsonl | .venv/bin/python -m json.tool`
Expected: at least 6 lines from this run; last line is well-formed JSON with `scenario`, `msinv`, `msprime`, `git_sha`, `ts`.

- [ ] **Step 4: Update CLAUDE.md**

Open `CLAUDE.md` and locate the lines:

```
- Python: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py`
  (183 passed, 3 skipped as of 2026-04-29; the 12 sweep-rewrite follow-up skips are now active
  after Phases A-D of `docs/superpowers/plans/2026-04-29-sweep-followups.md`).
- msprime validation: `tests/hull/test_validation_msprime.py` (N1 panmictic + N2 two-pop migration
  vs `msprime.sim_ancestry`; ~7 s; spec `docs/superpowers/specs/2026-04-29-msprime-validation-design.md`).
  Use `-s` to surface the per-stat OK/FAIL summary on failure.
```

Replace with the actual count from Step 2 (substitute `<N>`) and the extended scenario list:

```
- Python: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py`
  (<N> passed, 3 skipped as of 2026-04-30; the 12 sweep-rewrite follow-up skips are now active
  after Phases A-D of `docs/superpowers/plans/2026-04-29-sweep-followups.md`).
- msprime validation: `tests/hull/test_validation_msprime.py` (N1–N6: panmictic, two-pop migration,
  two-pop split, bottleneck, growth, three-pop split — vs `msprime.sim_ancestry`; ~25 s; spec
  `docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md`).
  Each scenario also prints a benchmark block (per-rep wall-clock + peak RSS per engine);
  appended to `.tmp/msprime_validation_bench.jsonl`. Use `-s` to surface OK/FAIL + benchmarks.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md test counts + msprime extension landed

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist (run after writing the plan)

- [x] Spec coverage: every spec section has at least one task.
  - Goal / Scope (N3-N6) → Phase C tasks C1-C4
  - Per-scenario parameters → Phase C, exact code blocks
  - Statistics: Family A (moments) → Task A4 routes via `_run_validation`
  - Statistics: Family B (AFS) → Task A1 emits, Task A4 routes, Task B1 unit-tests z
  - File layout → Tasks A1 (runner), A4 (test file refactor)
  - Test budget → Task D1 enforces under-30s gate
  - Benchmarks → Tasks A4–A6 (subprocess, ru_maxrss via wait4, JSONL persistence)
- [x] Placeholder scan: no TBD/TODO inside steps. Every code step shows the exact code.
- [x] Type consistency: `_run_one_engine` returns `(per_rep_stats, per_rep_seconds, peak_rss_kb)` consistently after Task A5; `SCENARIOS[name]` keys are `compute_afs`, `n_pops`, `make_msinv`, `make_msprime` everywhere.
- [x] No "similar to Task N" — every scenario task repeats its full code block.
