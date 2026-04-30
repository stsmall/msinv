# discoal validation harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-simulator validation of msinv's sweep model against discoal v2.0.0-beta across five scenarios (D1 neutral, D2 hard, D3 soft, D4 partial, D5 focal-site recurrent), reusing the msprime track's harness machinery.

**Architecture:** Extract the parent-side helpers (`_run_validation`, `_run_one_engine`, `_bonferroni_z`, `_print_benchmark_block`) from the msprime test file into `tests/hull/_validation_common.py`. Add a new child-process runner `tests/hull/_discoal_bench_runner.py` that knows how to spawn the discoal binary, glob its `*_repN.trees` output, and compute the same stat dict the msprime runner produces. Add `tests/hull/test_validation_discoal.py` with five tests calling the shared `_run_validation`. Activate windowed-π hitchhiking footprint stats (Class C) once moment stats agree.

**Tech Stack:** Python 3.12 (`.venv`), pytest, tskit, msprime (recapitation only — not used in discoal track), msinv (Rust core via PyO3), discoal v2.0.0-beta at `/home/adkern/discoal/discoal`, stdlib subprocess + os.wait4.

**Spec:** `docs/superpowers/specs/2026-04-30-discoal-validation-design.md`

**Empirical findings already verified:**
- `discoal 10 N L -t T -r R -ts /tmp/X.trees` writes **N separate files** named `/tmp/X_rep1.trees`, `/tmp/X_rep2.trees`, ..., `/tmp/X_repN.trees`. Runner must glob the suffix pattern.
- discoal emits diagnostic DEBUG lines to its own stdout — runner subprocess output should be discarded; only the runner's wrapper script writes JSON to *its* stdout.
- `-d seed1 seed2` sets the RNG seeds (two integers).

**Open empirical questions (resolve in B0):**
- Whether discoal CLI time is in 4N or 2N units. Spec assumes 4N (matches `discoal_msprime_parameter_guide.md`). The CLAUDE.md in the discoal repo claims internal=4N, CLI=2N. Resolve by running a known-tau sweep and inspecting tree heights.

---

## Phase A — Extract shared helpers

### Task A1: Create `tests/hull/_validation_common.py`

**Files:**
- Create: `tests/hull/_validation_common.py`

- [ ] **Step 1: Create the common helper module**

Create `tests/hull/_validation_common.py` with the following content:

```python
"""Shared helpers for cross-simulator validation harnesses.

Used by both the msprime track (`test_validation_msprime.py`) and the
discoal track (`test_validation_discoal.py`).  The parent-side flow
(spawn child runners per engine, read JSON stats, apply pass criteria,
print + persist benchmark) is identical across tracks; only the child
runner module path differs.

Platform note: peak RSS uses ``rusage.ru_maxrss``, which is **kilobytes
on Linux** but **bytes on macOS/BSD**.  This harness assumes Linux; on
other platforms the reported peak RSS will be off by 1024×.  Comparison
stats (the actual pass/fail criteria) are platform-independent.
"""

import datetime
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys


N_REPS = 200
ALPHA_FAMILY = 0.003  # family-wise α for both moment and AFS/window families


def _run_one_engine(runner_module, scenario_name, engine, n_reps):
    """Spawn one child runner; return (per_rep_stats, per_rep_seconds,
    peak_rss_kb).

    `runner_module` is the dotted path passed to ``python -m``, e.g.
    ``"tests.hull._msprime_bench_runner"`` or
    ``"tests.hull._discoal_bench_runner"``.

    Uses Popen + os.wait4 for clean per-child rusage.

    Pipe-buffer note: stdout JSON is ~78 KB at N_REPS=200 (msprime
    track N6, 11 stats x 200 reps), exceeding the 64 KB Linux pipe
    buffer.  This works because proc.stdout.read() blocks and drains
    the pipe as the child writes.  Stderr drain happens AFTER stdout,
    so the child MUST stay quiet on stderr (else the child blocks
    writing to a full stderr pipe while the parent blocks reading
    stdout = deadlock).  Both runners must redirect any third-party
    stdout/stderr (e.g. discoal's DEBUG output) to /dev/null.
    """
    cmd = [
        sys.executable, "-m", runner_module,
        "--scenario", scenario_name, "--engine", engine,
        "--n-reps", str(n_reps), "--seed-base", "0",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    peak_kb = rusage.ru_maxrss
    return payload["per_rep_stats"], payload["per_rep_seconds"], peak_kb


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _bonferroni_z(k_bins, alpha=ALPHA_FAMILY):
    """Two-sided z bound for `k_bins` AFS or window bins at family-wise alpha."""
    per_bin = alpha / k_bins
    return statistics.NormalDist().inv_cdf(1.0 - per_bin / 2.0)


def _agg_engine_vals(per_rep_stats):
    """List of per-rep stat dicts -> dict[stat_name, list[value]]."""
    out: dict[str, list[float]] = {}
    for rep_stats in per_rep_stats:
        for k, v in rep_stats.items():
            out.setdefault(k, []).append(v)
    return out


def _run_validation(runner_module, scenario_name, engine_a, engine_b,
                    bench_log, n_reps=N_REPS):
    """Run two engines via subprocess; assert per-stat agreement.

    `engine_a` and `engine_b` are engine names passed to the runner
    (e.g. "msinv", "msprime", "discoal").  `bench_log` is the path
    (str or Path) to the JSONL benchmark log to append.
    """
    a_stats, a_secs, a_peak = _run_one_engine(
        runner_module, scenario_name, engine_a, n_reps)
    b_stats, b_secs, b_peak = _run_one_engine(
        runner_module, scenario_name, engine_b, n_reps)

    a = _agg_engine_vals(a_stats)
    b = _agg_engine_vals(b_stats)
    keys = list(a.keys())
    assert set(keys) == set(b.keys()), (
        f"stat key mismatch: {engine_a}={set(keys)} vs "
        f"{engine_b}={set(b.keys())}")

    afs_keys = [k for k in keys if k.startswith("afs_")]
    win_keys = [k for k in keys if k.startswith("pi_window_")]
    moment_keys = [
        k for k in keys
        if not k.startswith("afs_") and not k.startswith("pi_window_")]
    z_afs = _bonferroni_z(len(afs_keys)) if afs_keys else None
    z_win = _bonferroni_z(len(win_keys)) if win_keys else None

    failures = []
    lines = []
    for k in moment_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = 3.0 * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: {engine_a}={m_a:.4g} ± {se_a:.3g}, "
                f"{engine_b}={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, 3·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)
    for family_keys, z_family, label in (
            (afs_keys, z_afs, f"{z_afs:.2f}·SE" if z_afs else None),
            (win_keys, z_win, f"{z_win:.2f}·SE" if z_win else None)):
        if not family_keys:
            continue
        for k in family_keys:
            m_a, se_a = _mean_se(a[k])
            m_b, se_b = _mean_se(b[k])
            bound = z_family * math.sqrt(se_a ** 2 + se_b ** 2)
            delta = abs(m_a - m_b)
            ok = delta <= bound
            line = (f"{k}: {engine_a}={m_a:.4g} ± {se_a:.3g}, "
                    f"{engine_b}={m_b:.4g} ± {se_b:.3g}, "
                    f"|Δ|={delta:.4g}, {label}={bound:.4g} "
                    f"→ {'OK' if ok else 'FAIL'}")
            lines.append(line)
            if not ok:
                failures.append(line)

    print(f"\n[{scenario_name}]")
    for line in lines:
        print(f"  {line}")

    _print_benchmark_block(
        scenario_name, engine_a, engine_b, a_secs, b_secs,
        a_peak, b_peak, bench_log)

    if failures:
        raise AssertionError(
            f"\n[{scenario_name}]\n  " + "\n  ".join(lines))


def _git_short_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def _print_benchmark_block(scenario_name, engine_a, engine_b,
                           a_secs, b_secs, a_peak_kb, b_peak_kb,
                           bench_log):
    """Print + persist a per-scenario benchmark line."""
    a_mean, a_se = _mean_se(a_secs)
    b_mean, b_se = _mean_se(b_secs)
    a_total = sum(a_secs)
    b_total = sum(b_secs)
    a_rss_mb = a_peak_kb / 1024.0
    b_rss_mb = b_peak_kb / 1024.0
    print(f"[{scenario_name}] benchmarks")
    print(f"  {engine_a:7s}: per-rep {a_mean*1000:6.1f} ms ± {a_se*1000:.1f}, "
          f"total {a_total:5.1f} s, peak RSS {a_rss_mb:6.1f} MB")
    print(f"  {engine_b:7s}: per-rep {b_mean*1000:6.1f} ms ± {b_se*1000:.1f}, "
          f"total {b_total:5.1f} s, peak RSS {b_rss_mb:6.1f} MB")
    print(f"  ratio:   per-rep {engine_a}/{engine_b} = {a_mean/b_mean:.2f}x;  "
          f"RAM {engine_a}/{engine_b} = {a_rss_mb/b_rss_mb:.2f}x")

    log_dir = pathlib.Path(bench_log).parent
    log_dir.mkdir(exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "scenario": scenario_name,
        engine_a: {"per_rep_s": a_mean, "per_rep_se": a_se,
                   "total_s": a_total, "peak_rss_mb": a_rss_mb},
        engine_b: {"per_rep_s": b_mean, "per_rep_se": b_se,
                   "total_s": b_total, "peak_rss_mb": b_rss_mb},
    }
    with pathlib.Path(bench_log).open("a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 2: Verify import side-effect-free**

Run: `.venv/bin/python -c "import tests.hull._validation_common as c; print(c.N_REPS, c._bonferroni_z(10))"`
Expected: prints `200 3.6063...`

- [ ] **Step 3: Commit**

```bash
git add tests/hull/_validation_common.py
git commit -m "test: extract validation harness common helpers

Shared by both the msprime and discoal validation tracks.
_run_validation now takes runner_module + engine_a/b + bench_log
so different tracks can drive different engine pairs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Migrate msprime track to use common helpers

**Files:**
- Modify: `tests/hull/test_validation_msprime.py`
- Modify: `tests/hull/test_validation_msprime_helpers.py`

- [ ] **Step 1: Replace `tests/hull/test_validation_msprime.py` contents**

Read the current file first. Then replace with:

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

from tests.hull._validation_common import (
    N_REPS, _run_validation, _bonferroni_z,
)

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
```

- [ ] **Step 2: Update the helper unit test imports**

Read `tests/hull/test_validation_msprime_helpers.py` first; it currently imports `_bonferroni_z` from `tests.hull.test_validation_msprime`. Re-point at the common module:

```python
"""Unit tests for the small math helpers in _validation_common."""

import pytest

from tests.hull._validation_common import _bonferroni_z


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

- [ ] **Step 3: Run msprime tests + helpers**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py tests/hull/test_validation_msprime_helpers.py -v -s 2>&1 | tail -30`
Expected: 8 passed (6 msprime + 2 helpers). All scenarios still print their per-stat OK lines and benchmark blocks. JSONL log path unchanged at `.tmp/msprime_validation_bench.jsonl`.

- [ ] **Step 4: Run full Python suite**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3`
Expected: 189 passed, 3 skipped (unchanged from main).

- [ ] **Step 5: Commit**

```bash
git add tests/hull/test_validation_msprime.py tests/hull/test_validation_msprime_helpers.py
git commit -m "test: msprime track imports from _validation_common

Refactor only — N1-N6 + helper tests behavior unchanged. The parent
test file is now a thin scenario-list wrapper around _run_validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — discoal runner skeleton + D1 neutral baseline

### Task B0: Empirically pin discoal CLI time convention

**No file changes; this is research that informs subsequent tasks.**

- [ ] **Step 1: Run a known-tau sweep and read tree heights**

The spec assumes discoal CLI time is in 4N units (per `discoal_msprime_parameter_guide.md`). The discoal repo's CLAUDE.md hints at 2N. Resolve empirically.

Run two probe sims:

```bash
cd /tmp
# Probe A: tau interpreted as 4N units → if CLI time is 4N, sweep at gen 4000 (Ne=10000, tau=0.1)
/home/adkern/discoal/discoal 10 5 100000 -t 40 -r 40 -ws 0.1 -a 1000 -x 0.5 -N 10000 -ts probeA.trees > /dev/null 2>&1

# Probe B: tau interpreted as 2N units → if CLI time is 2N, sweep at gen 2000 (Ne=10000, tau=0.1)
# (Same call; same data — we look at tree heights to disambiguate.)
```

- [ ] **Step 2: Inspect tree heights**

```bash
.venv/bin/python <<'EOF'
import tskit
ts = tskit.load("/tmp/probeA_rep1.trees")
heights = [tree.time(tree.root) for tree in ts.trees()]
print(f"min, mean, max tree height: {min(heights):.0f}, {sum(heights)/len(heights):.0f}, {max(heights):.0f}")
print(f"sweep would be at 4000 gens (4N) or 2000 gens (2N) interpretation")
EOF
```

Expected: tree heights cluster around `4Ne ≈ 40000` generations (from the neutral coalescent at the windows far from the sweep). The sweep punctuates more recent coalescence.

If max tree height ≈ 40000-50000 gens and there's a sharp coalescence cluster around the sweep time, the next question is which time. Use the deterministic-sweep variant for sharper signal:

```bash
/home/adkern/discoal/discoal 10 50 100000 -t 40 -r 40 -wd 0.1 -a 1000 -x 0.5 -N 10000 -ts probeC.trees > /dev/null 2>&1
.venv/bin/python <<'EOF'
import tskit
import statistics
heights = []
for r in range(1, 51):
    ts = tskit.load(f"/tmp/probeC_rep{r}.trees")
    # tree at the sweep position (x=0.5 of a 100kb genome = position 50000)
    t = ts.at(50000.0)
    heights.append(t.time(t.root))
mean_h = statistics.mean(heights)
print(f"mean tree height at sweep position: {mean_h:.0f}")
print(f"4N interpretation predicts ~4000 gens; 2N interpretation predicts ~2000 gens")
EOF
```

- [ ] **Step 3: Document the finding**

Decision: pick whichever convention the empirical data supports. Record the chosen conversion factor. **For the rest of this plan, the spec assumption (`tau_disc = gens / (4·Ne)`) is the default**; if the probe shows otherwise, update Step B2 below to use `gens / (2·Ne)` and call out the deviation in the commit message.

- [ ] **Step 4: Cleanup**

```bash
rm -f /tmp/probeA*.trees /tmp/probeB*.trees /tmp/probeC*.trees
```

This task does not commit. It produces a single fact (the time-conversion factor) used by Task B2.

---

### Task B1: Skeleton discoal child runner

**Files:**
- Create: `tests/hull/_discoal_bench_runner.py`

- [ ] **Step 1: Create the runner skeleton**

```python
"""Child-process runner for the discoal validation harness.

Invoked as ``python -m tests.hull._discoal_bench_runner --scenario NAME
--engine {msinv,discoal} --n-reps N --seed-base K``.  Runs the rep
batch and writes a JSON document to stdout::

    {"per_rep_stats": [{stat_name: value, ...}, ...], "per_rep_seconds": [float, ...]}

Importing this module does NOT execute any sim. It is pytest-collectable
but pytest-skipped.
"""

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest
import tskit

from msinv.hull.simulator import HullSimulator

pytestmark = pytest.mark.skip("child runner — invoked via subprocess")


DISCOAL_BIN = "/home/adkern/discoal/discoal"


# Scenario registry filled in by Task B2 (D1) and Phase C tasks (D2-D5).
# Each entry: name -> {
#   "n_pops": int (always 1 for discoal track v1),
#   "compute_pi_windows": bool,
#   "windows_left_to_right": list[float] | None,  # bin edges within [0, L]
#   "make_msinv": Callable[[int], tskit.TreeSequence],
#   "make_discoal_args": Callable[[int, int], list[str]],  # (seed, n_reps) -> argv
#   "L": float,  # sequence length, needed for window math
#   "x_sel": float | None,  # sweep position, for footprint folding
# }
SCENARIOS: dict[str, dict] = {}


def _gens_to_discoal_time(gens: float, ne_diploid: float) -> float:
    """Convert generations (msinv side) to discoal CLI time (4N units).

    See B0 task in the plan for the empirical pinning of this conversion.
    The 4N convention matches docs/discoal_msprime_parameter_guide.md;
    if B0 found 2N, edit this to `gens / (2.0 * ne_diploid)`.
    """
    return gens / (4.0 * ne_diploid)


def _s_to_alpha(s: float, ne_diploid: float) -> float:
    """Convert per-generation selection coefficient s to discoal -a alpha."""
    return 2.0 * ne_diploid * s


def _stats_from_ts(ts, scenario_spec):
    """Branch-length stats from a tskit TS, optionally with windowed pi.

    Returns dict suitable for JSON serialization.  Single-pop only.
    """
    out: dict[str, float] = {
        "pi_branch": ts.diversity(mode="branch"),
        "n_trees": float(ts.num_trees),
    }
    weighted = sum(tree.time(tree.root) * tree.span for tree in ts.trees())
    out["mean_tmrca"] = weighted / ts.sequence_length

    if scenario_spec.get("compute_pi_windows", False):
        # Folded windowed pi around x_sel.  K bins at distance
        # [0, w], [w, 2w], ..., [(K-1)w, K*w] from the sweep site.
        # Each bin sums the left and right halves of the genome.
        L = scenario_spec["L"]
        x_sel = scenario_spec["x_sel"]
        edges = scenario_spec["windows_left_to_right"]
        K = len(edges) - 1
        # Build windows on the [0, L] coordinate.  Each folded bin k
        # corresponds to two windows: [x_sel - edges[k+1], x_sel - edges[k]]
        # and [x_sel + edges[k], x_sel + edges[k+1]], clamped to [0, L].
        windowed = [0.0] * K
        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            left_lo = max(0.0, x_sel - hi)
            left_hi = max(0.0, x_sel - lo)
            right_lo = min(L, x_sel + lo)
            right_hi = min(L, x_sel + hi)
            wins = []
            if left_hi > left_lo:
                wins.extend([left_lo, left_hi])
            if right_hi > right_lo:
                wins.extend([right_lo, right_hi])
            if not wins:
                continue
            # tskit needs a sorted, contiguous-edge list starting at 0
            # and ending at L for diversity(windows=...).  Build a full
            # list with masking.
            full = [0.0]
            for v in wins:
                if v > full[-1]:
                    full.append(v)
            if full[-1] < L:
                full.append(L)
            # ts.diversity returns one value per [edges[i], edges[i+1]];
            # we need to identify which of those are inside our sub-windows.
            divs = ts.diversity(windows=full, mode="branch")
            total_span = 0.0
            total_pi_span = 0.0
            for i in range(len(full) - 1):
                seg_lo, seg_hi = full[i], full[i + 1]
                # Inside left window?
                in_left = (seg_lo >= left_lo and seg_hi <= left_hi)
                in_right = (seg_lo >= right_lo and seg_hi <= right_hi)
                if in_left or in_right:
                    span = seg_hi - seg_lo
                    total_span += span
                    total_pi_span += divs[i] * span
            windowed[k] = total_pi_span / total_span if total_span > 0 else 0.0
        for k, val in enumerate(windowed):
            out[f"pi_window_{k}"] = float(val)
    return out


def _run_one_engine_batch(scenario_name, engine, n_reps, seed_base):
    """Returns (per_rep_stats, per_rep_seconds)."""
    spec = SCENARIOS[scenario_name]
    if engine == "msinv":
        return _run_msinv_batch(spec, n_reps, seed_base)
    elif engine == "discoal":
        return _run_discoal_batch(spec, n_reps, seed_base)
    raise ValueError(f"unknown engine: {engine}")


def _run_msinv_batch(spec, n_reps, seed_base):
    factory = spec["make_msinv"]
    # warm-up rep, untimed
    factory(seed=seed_base)
    per_rep_stats = []
    per_rep_seconds = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        ts = factory(seed=seed_base + i)
        stats = _stats_from_ts(ts, spec)
        per_rep_seconds.append(time.perf_counter() - t0)
        per_rep_stats.append(stats)
    return per_rep_stats, per_rep_seconds


def _run_discoal_batch(spec, n_reps, seed_base):
    """Run discoal once with numReplicates=n_reps; load each .trees file."""
    make_args = spec["make_discoal_args"]
    # discoal seeds are two integers; derive both from seed_base
    seed1 = seed_base + 1
    seed2 = seed_base + 100001
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = pathlib.Path(tmpdir) / "scenario.trees"
        argv = make_args(out_prefix=str(out_prefix), seed1=seed1, seed2=seed2,
                         n_reps=n_reps)
        # discoal writes DEBUG to stdout; redirect to /dev/null so it
        # doesn't pollute our subprocess pipe budget.
        t0 = time.perf_counter()
        result = subprocess.run(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
        wall = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(
                f"discoal failed (rc={result.returncode}); argv={argv}")
        # discoal writes <prefix>_rep1.trees ... _rep{n_reps}.trees
        # Strip trailing ".trees" so the actual files are
        # <prefix-without-extension>_rep{N}.trees
        prefix_base = str(out_prefix).removesuffix(".trees")
        per_rep_stats = []
        per_rep_seconds = []
        for i in range(n_reps):
            t0 = time.perf_counter()
            tree_path = f"{prefix_base}_rep{i + 1}.trees"
            ts = tskit.load(tree_path)
            stats = _stats_from_ts(ts, spec)
            per_rep_seconds.append(time.perf_counter() - t0)
            per_rep_stats.append(stats)
    # Total wall time was for the full discoal call; we report the
    # parsing time per rep.  The combined "total wall" for a rep is
    # the parse + (run / n_reps) — but for benchmark fidelity we
    # attribute the discoal subprocess wall as a single chunk added
    # to rep 0's time.  Per-rep timing for the engine comparison is
    # parse-only; the total in the benchmark block reflects subprocess
    # wall via os.wait4 rusage at the parent level.
    if per_rep_seconds:
        per_rep_seconds[0] += wall
    return per_rep_stats, per_rep_seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--engine", required=True, choices=["msinv", "discoal"])
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()
    per_rep_stats, per_rep_seconds = _run_one_engine_batch(
        args.scenario, args.engine, args.n_reps, args.seed_base)
    json.dump(
        {"per_rep_stats": per_rep_stats,
         "per_rep_seconds": per_rep_seconds},
        sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check import**

Run: `.venv/bin/python -c "import tests.hull._discoal_bench_runner as r; print(list(r.SCENARIOS), r._gens_to_discoal_time(1000.0, 10000.0))"`
Expected: `[] 0.025` (1000 gens / (4 * 10000) = 0.025 in 4N units).

- [ ] **Step 3: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py
git commit -m "test: discoal validation runner skeleton

Empty SCENARIOS registry, conversion helpers
(_gens_to_discoal_time, _s_to_alpha), CLI dispatch, _stats_from_ts
with optional folded windowed-pi support.  Discoal subprocess wrapper
loads <prefix>_rep1..N.trees files and computes the same stat dict
shape as the msprime track.  No scenarios yet — added in B2 / Phase C.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task B2: Register D1 neutral baseline scenario

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (add D1)
- Create: `tests/hull/test_validation_discoal.py`

- [ ] **Step 1: Add D1 factories to the runner**

Append to `tests/hull/_discoal_bench_runner.py` after the SCENARIOS declaration:

```python
def _make_d1_msinv(seed: int):
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=None,
        seed=seed,
    ).simulate()


def _make_d1_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10",            # sampleSize (haploid)
        str(n_reps),     # numReplicates
        "100000",        # nSites
        "-t", "40",      # theta = 4*Ne*mu*L = 4*10000*1e-8*1e5 = 40
        "-r", "40",      # rho   = 4*Ne*r *L = 4*10000*1e-8*1e5 = 40
        "-N", "10000",   # diploid Ne
        "-ts", out_prefix,
        "-F",            # full ARG mode (matches msinv record_full_arg=True)
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d1"] = {
    "n_pops": 1,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d1_msinv,
    "make_discoal_args": _make_d1_discoal_args,
    "L": 100_000.0,
    "x_sel": None,
}
```

- [ ] **Step 2: Smoke-test the runner CLI for both engines**

Run:

```bash
.venv/bin/python -m tests.hull._discoal_bench_runner --scenario d1 --engine msinv --n-reps 3 --seed-base 0 | head -1
.venv/bin/python -m tests.hull._discoal_bench_runner --scenario d1 --engine discoal --n-reps 3 --seed-base 0 | head -1
```

Expected: each prints a single JSON line with `per_rep_stats` (list of 3 dicts) and `per_rep_seconds` (list of 3 floats). Stat keys per rep: `pi_branch`, `n_trees`, `mean_tmrca` (no `dxy_branch` — single-pop; no `pi_window_*` — `compute_pi_windows=False`).

- [ ] **Step 3: Create `tests/hull/test_validation_discoal.py`**

```python
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
```

- [ ] **Step 4: Run D1**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py::test_discoal_validation_d1_neutral -v -s 2>&1 | tail -25`
Expected: pass.  3 stats OK at 3·SE: `pi_branch`, `n_trees`, `mean_tmrca`.

If a stat fails, STOP and report BLOCKED:
- All three off by similar factors → convention bridge (Ne, theta, rho).
- `n_trees` only off → discoal `-F` flag handling, or msinv `record_full_arg` analogue.
- `pi_branch` off, `mean_tmrca` matches → mutation-rate accidentally being applied somewhere (but branch-mode shouldn't depend on mutations; flag for investigation).

Do NOT widen bounds or remove scenario.  Real failures here indicate convention or flag bugs that the harness is supposed to catch.

- [ ] **Step 5: Run full Python suite**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3`
Expected: 190 passed, 3 skipped (was 189; +1 for D1).

- [ ] **Step 6: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py tests/hull/test_validation_discoal.py
git commit -m "test: discoal validation D1 — neutral baseline

Validates convention bridge (theta=40, rho=40, Ne=10000 diploid)
between msinv and discoal v2.0.0-beta.  No sweep, no inversions,
no demographic events.  Establishes parity before exercising sweep
machinery in D2-D5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Sweep scenarios D2-D5 (moments only)

Sweep scenarios use `compute_pi_windows=False` initially.  Phase D activates windowing across all scenarios at once after moments are confirmed.

### Task C1: D2 hard sweep

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (add D2)
- Modify: `tests/hull/test_validation_discoal.py` (add test)

- [ ] **Step 1: Add D2 factories**

Append to the runner after the D1 block:

```python
from msinv.hull.sweep import Sweep


def _make_d2_msinv(seed: int):
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Stochastic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10000),
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d2_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    # tau in 4N units: 1000 gens / (4 * 10000) = 0.025
    # alpha = 2 * Ne * s = 2 * 10000 * 0.05 = 1000
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-ws", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d2"] = {
    "n_pops": 1,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d2_msinv,
    "make_discoal_args": _make_d2_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}
```

- [ ] **Step 2: Add the test function**

Append to `tests/hull/test_validation_discoal.py`:

```python
def test_discoal_validation_d2_hard_sweep():
    """Rust msinv vs discoal — hard sweep, s=0.05, tau=1000 g, fix at 1.0."""
    _run("d2")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py::test_discoal_validation_d2_hard_sweep -v -s 2>&1 | tail -25`
Expected: pass.  3 stats OK at 3·SE.

If stats fail, triage table from spec:
- Moments OK, footprint slope wrong (not yet measured here): sweep `s` or `tau` conversion.
- Moments fail (low π far from sweep): `Ne` or `θ` mismatch unrelated to sweep.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py tests/hull/test_validation_discoal.py
git commit -m "test: discoal validation D2 — hard complete sweep

s=0.05 (alpha=1000), tau=1000 generations (= 0.025 in 4N units),
sweep at locus midpoint, complete fixation.  msinv side uses
mode=Stochastic with f0=1/(2N) to match discoal -ws stochastic
trajectory starting from one founding adaptive copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C2: D3 soft sweep from standing variation

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (add D3)
- Modify: `tests/hull/test_validation_discoal.py` (add test)

- [ ] **Step 1: Add D3 factories**

Append to the runner after D2:

```python
def _make_d3_msinv(seed: int):
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Stochastic',
        s=0.05,
        t_origin=1500.0,
        f0=0.05,                           # soft sweep: starts at 5% standing variation
        partial_sweep_final_freq=1.0,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d3_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-ws", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-f", "0.05",                      # initial freq for soft sweep
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d3"] = {
    "n_pops": 1,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d3_msinv,
    "make_discoal_args": _make_d3_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}
```

- [ ] **Step 2: Add the test function**

```python
def test_discoal_validation_d3_soft_sweep():
    """Rust msinv vs discoal — soft sweep from standing variation, f0=0.05."""
    _run("d3")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py::test_discoal_validation_d3_soft_sweep -v -s 2>&1 | tail -25`
Expected: pass.  3 stats OK.

If D3 fails but D2 passes, it's almost certainly a soft-sweep `f0` semantics difference — investigate `JointSweepSpec.f0` initialization vs discoal's `-f` interpretation.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py tests/hull/test_validation_discoal.py
git commit -m "test: discoal validation D3 — soft sweep from standing variation

Same as D2 but f0=0.05 (the beneficial allele starts at 5% frequency
across the population, polyphyletic adaptive lineage).  Tests that
msinv's joint trajectory at f0>0 produces the same diversity
distribution as discoal -f.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C3: D4 partial sweep

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (add D4)
- Modify: `tests/hull/test_validation_discoal.py` (add test)

- [ ] **Step 1: Add D4 factories**

Append to the runner after D3:

```python
def _make_d4_msinv(seed: int):
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Stochastic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10000),
        partial_sweep_final_freq=0.5,      # stops at 50%
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d4_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-ws", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-c", "0.5",                       # partial sweep stops at 0.5
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d4"] = {
    "n_pops": 1,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d4_msinv,
    "make_discoal_args": _make_d4_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}
```

- [ ] **Step 2: Add the test function**

```python
def test_discoal_validation_d4_partial_sweep():
    """Rust msinv vs discoal — partial sweep, stops at 50% freq."""
    _run("d4")
```

- [ ] **Step 3: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py::test_discoal_validation_d4_partial_sweep -v -s 2>&1 | tail -25`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py tests/hull/test_validation_discoal.py
git commit -m "test: discoal validation D4 — partial sweep (final freq 0.5)

Sweep stops at 50% derived-allele frequency.  Hitchhiking footprint
(measured later in Phase D) should be shallower than D2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C4: D5 focal-site recurrent (with smoke verification)

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (add D5)
- Modify: `tests/hull/test_validation_discoal.py` (add test)

- [ ] **Step 1: Smoke-verify msinv mode=Stochastic + recurrent_mutation_rate>0 builds a non-empty trajectory**

This combination is unverified end-to-end (J9 only tests the recurrence COUNT under mode=Neutral).  Before wiring into the harness, run a one-off probe:

```bash
.venv/bin/python <<'EOF'
from msinv.hull.sweep import Sweep
sw = Sweep(
    x_sel=50_000.0, tau=1000.0,
    origin_pop=0, origin_kary='S', target_inv=0,
    mode='Stochastic',
    s=0.05, t_origin=1500.0, f0=1.0/(2*10000),
    partial_sweep_final_freq=1.0,
    recurrent_mutation_rate=1e-3,
    seed=1,
)
py = sw.to_rust()
py.build_trajectory(n_pops=1, p_inv_init=[0.0], pop_sizes=[10000.0])
samples = py.trajectory_samples()
print(f"trajectory has {len(samples)} sample points; "
      f"first t={samples[0][0]:.0f}, last t={samples[-1][0]:.0f}, "
      f"final A freq={py.final_a_freq():.3f}")
EOF
```

Expected: prints non-zero number of samples, t spans roughly tau→t_origin, final A freq close to 1.0.  If the trajectory is empty or final A freq is 0, the combination has a latent bug; STOP and report BLOCKED — do not proceed with D5.

- [ ] **Step 2: Add D5 factories (assuming smoke passed)**

Append to the runner after D4:

```python
def _make_d5_msinv(seed: int):
    sweep = Sweep(
        x_sel=50_000.0,
        tau=1000.0,
        origin_pop=0,
        origin_kary='S',
        target_inv=0,
        mode='Stochastic',
        s=0.05,
        t_origin=1500.0,
        f0=1.0 / (2 * 10000),
        partial_sweep_final_freq=1.0,
        recurrent_mutation_rate=1e-3,
        seed=seed,
    )
    return HullSimulator(
        samples=10,
        population_size=10000.0,
        sequence_length=100_000.0,
        recombination_rate=1e-8,
        inversions=[],
        sweeps=[sweep],
        seed=seed,
    ).simulate()


def _make_d5_discoal_args(out_prefix: str, seed1: int, seed2: int,
                          n_reps: int):
    return [
        DISCOAL_BIN,
        "10", str(n_reps), "100000",
        "-t", "40", "-r", "40", "-N", "10000",
        "-ws", "0.025",
        "-a", "1000",
        "-x", "0.5",
        "-uA", "1e-3",                     # recurrent adaptive mutation rate
        "-ts", out_prefix, "-F",
        "-d", str(seed1), str(seed2),
    ]


SCENARIOS["d5"] = {
    "n_pops": 1,
    "compute_pi_windows": False,
    "windows_left_to_right": None,
    "make_msinv": _make_d5_msinv,
    "make_discoal_args": _make_d5_discoal_args,
    "L": 100_000.0,
    "x_sel": 50_000.0,
}
```

- [ ] **Step 3: Add the test function**

```python
def test_discoal_validation_d5_focal_recurrent():
    """Rust msinv vs discoal — focal-site recurrent sweep, uA=1e-3."""
    _run("d5")
```

- [ ] **Step 4: Run; expect pass**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py::test_discoal_validation_d5_focal_recurrent -v -s 2>&1 | tail -25`
Expected: pass.

If D5 fails, the most likely cause is units mismatch on the recurrence rate (per-2N-per-gen vs per-gen).  Investigate before adjusting bounds.

- [ ] **Step 5: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py tests/hull/test_validation_discoal.py
git commit -m "test: discoal validation D5 — focal-site recurrent sweep

mode=Stochastic with recurrent_mutation_rate=1e-3 on the msinv side,
discoal -uA 1e-3.  Smoke-checked the msinv combination produces a
non-empty trajectory before wiring into the cross-engine comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase D — Hitchhiking footprint (Class C windowed-π)

### Task D1: Activate windowed-π on D1-D5

**Files:**
- Modify: `tests/hull/_discoal_bench_runner.py` (flip flag for d1-d5)

The windowed-π helper code is already in `_stats_from_ts` from Task B1.  This task just turns on the flag for each scenario and supplies the bin-edge list.

- [ ] **Step 1: Update each SCENARIOS["dN"] entry**

For each of d1, d2, d3, d4, d5, change in `tests/hull/_discoal_bench_runner.py`:

- `"compute_pi_windows": False,` → `"compute_pi_windows": True,`
- `"windows_left_to_right": None,` → `"windows_left_to_right": [0.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0],`
- For d1 (no sweep), change `"x_sel": None,` → `"x_sel": 50_000.0,` (use the same window center as d2-d5 so D1 acts as a flatness check).

**Edge list explanation:** 11 edges → K=10 bins, each 5kb wide.  `windows_left_to_right` is read by `_stats_from_ts` and folded around `x_sel` into 10 distance-from-sweep bins.

- [ ] **Step 2: Smoke-test the runner emits pi_window_* keys**

Run: `.venv/bin/python -m tests.hull._discoal_bench_runner --scenario d2 --engine msinv --n-reps 3 --seed-base 0 | head -1`
Expected: per-rep stats dict includes `pi_window_0` through `pi_window_9` (10 keys), in addition to `pi_branch`, `n_trees`, `mean_tmrca`.

Same for `--engine discoal`.

- [ ] **Step 3: Run all 5 D-tests**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_discoal.py -v -s 2>&1 | tail -100`
Expected: 5 passed.  Each scenario now reports 13 stats: 3 moments at 3·SE plus 10 windows at the K=10 Bonferroni bound (`_bonferroni_z(10) ≈ 3.61·SE`).

For D1 (no sweep), the 10 windows should all sit near the same π value — if any bin diverges between engines on D1, it indicates a windowing-code bug, not a sweep bug.

For D2-D5, expect the bins nearest the sweep to have lower π in both engines, recovering toward bin 9 (the farthest 45-50kb window).  Both engines should agree per-bin.

- [ ] **Step 4: Run full Python suite**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3`
Expected: 194 passed, 3 skipped (= 189 baseline + 5 D-scenarios).

- [ ] **Step 5: Commit**

```bash
git add tests/hull/_discoal_bench_runner.py
git commit -m "test: activate hitchhiking footprint on D1-D5

K=10 folded distance-from-sweep bins, each 5kb wide, summing left
and right halves of the 100kb genome.  D1 (no sweep) acts as a
flatness check; D2-D5 expect a recovery curve from low-pi near the
sweep to baseline pi at 45-50kb.  Bonferroni z=3.61 per bin
(family-wise alpha=0.003).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase E — Final validation

### Task E1: Full harness pass + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (bump test counts and harness description)

- [ ] **Step 1: Full discoal harness wall-clock**

Run: `time .venv/bin/python -m pytest tests/hull/test_validation_discoal.py -v -s 2>&1 | tail -120`
Expected:
- 5 passed (d1..d5).
- Each scenario benchmark block visible.
- Total wall-clock under 180 s. Estimated real time ~30–50 s for the full discoal track.

- [ ] **Step 2: Full Python suite**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3`
Expected: 194 passed, 3 skipped (was 189; +5 for D1-D5).

- [ ] **Step 3: JSONL log sanity-check**

Run: `wc -l .tmp/discoal_validation_bench.jsonl && tail -1 .tmp/discoal_validation_bench.jsonl | .venv/bin/python -m json.tool`
Expected: at least 5 lines from this run; last line is well-formed JSON with `scenario`, `msinv`, `discoal`, `git_sha`, `ts`.

- [ ] **Step 4: Update CLAUDE.md**

Read CLAUDE.md.  Locate the line currently reading something like:
```
- msprime validation: `tests/hull/test_validation_msprime.py` (N1–N6: panmictic, two-pop migration,
  two-pop split, bottleneck, growth, three-pop split — vs `msprime.sim_ancestry`; ~15 s; spec
  `docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md`).
  Each scenario also prints a benchmark block (per-rep wall-clock + peak RSS per engine);
  appended to `.tmp/msprime_validation_bench.jsonl`. Use `-s` to surface OK/FAIL + benchmarks.
```

Add directly after:
```
- discoal validation: `tests/hull/test_validation_discoal.py` (D1–D5: neutral baseline,
  hard sweep, soft sweep from SV, partial sweep, focal-site recurrent — vs discoal v2.0.0-beta
  at `/home/adkern/discoal/discoal`; ~<actual>s; spec
  `docs/superpowers/specs/2026-04-30-discoal-validation-design.md`).
  Each sweep scenario reports moments + K=10 windowed-pi hitchhiking footprint;
  benchmark block appended to `.tmp/discoal_validation_bench.jsonl`.
```

Bump the Python test count line:
- `(189 passed, 3 skipped as of 2026-04-30; ...)` → `(194 passed, 3 skipped as of 2026-04-30; ...)`

(Substitute `<actual>s` with the actual wall-clock from Step 1 rounded to nearest 5s.)

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md test counts + discoal extension landed

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist

- [x] **Spec coverage:**
  - D1 neutral → Task B2
  - D2 hard sweep → Task C1
  - D3 soft sweep → Task C2
  - D4 partial sweep → Task C3
  - D5 focal recurrent → Task C4 (with smoke verification step)
  - Class B moments → covered in B1's `_stats_from_ts`
  - Class C windowed-π → Task D1
  - Convention bridge (time, alpha, theta, rho) → B0 (empirical) + B1 (helpers) + B2 (D1 wiring)
  - Discoal subprocess output handling → B1 (run_discoal_batch with tempfile + repN.trees globbing + stdout/stderr to /dev/null)
  - Helper extraction → A1 + A2
  - Benchmarks (JSONL) → existing common helper, just routed to a different log path
- [x] **Placeholder scan:** no TBDs.  Each step has full code or a precise command.
- [x] **Type consistency:** `_run_validation` signature matches in common helper and both call sites.  `SCENARIOS` schema same across both runners.  `_stats_from_ts` returns the same shape (dict[str, float]) on both sides.
- [x] **No "similar to Task N":** every C task repeats its full code block.
