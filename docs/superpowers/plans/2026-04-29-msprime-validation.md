# msprime Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pytest-resident harness that validates the Rust msinv core's neutral hot path (coal + recomb + class-aware migration) against `msprime.sim_ancestry` on two scenarios — single-pop panmictic and two-pop symmetric migration.

**Architecture:** One new test file (`tests/hull/test_validation_msprime.py`) with a private `_run_validation` helper plus two pytest test functions (N1 panmictic, N2 two-pop migration). Each scenario draws N=200 reps per engine, computes branch-length stats from each `tskit.TreeSequence` (`pi_branch`, `num_trees`, span-weighted `mean_tmrca`, plus `dxy_branch` for N2), and asserts `|mean_msinv − mean_msprime| ≤ 3·sqrt(SE_msinv² + SE_msprime²)` per stat. No new modules, no fixtures, no plugins.

**Tech Stack:** Python 3.12, pytest, msprime 1.4.1, tskit, msinv hull engine (Rust core via PyO3).

---

## File Structure

Single new file:

- `tests/hull/test_validation_msprime.py` — helper `_run_validation`, helper `_stats_from_ts`, helper `_mean_se`, two test functions.

No edits to `msinv/`, `rust/`, or `_rust_bridge.py`. CLAUDE.md test count gets a one-line bump in the final task.

---

## Task 1: Helper functions and N1 panmictic validation

**Files:**
- Create: `tests/hull/test_validation_msprime.py`

- [ ] **Step 1: Write the file**

```python
"""msprime validation harness — Rust msinv core vs msprime.sim_ancestry.

Spec: docs/superpowers/specs/2026-04-29-msprime-validation-design.md.

Two scenarios:
  N1 — single-pop panmictic, n=10, Ne=10000, L=100kb, r=1e-8 (rho=40).
  N2 — two-pop island, n=5+5, Ne=[10000,10000], symmetric M=1e-4, L=100kb, r=1e-8.

For each scenario, N=200 reps on each engine (rep i seeds engine with i).
Per-stat assertion: |mean_msinv - mean_msprime| <= 3 * sqrt(SE_msinv^2 + SE_msprime^2).
"""

import math
import statistics

import msprime
import pytest

from msinv.hull.demography import Demography
from msinv.hull.simulator import HullSimulator


N_REPS = 200


def _stats_from_ts(ts, sample_sets=None):
    """Branch-length stats from a tskit TreeSequence.

    Returns dict with keys 'pi_branch', 'n_trees', 'mean_tmrca', and
    (when sample_sets is provided) 'dxy_branch'.
    """
    out = {
        "pi_branch": ts.diversity(mode="branch"),
        "n_trees": float(ts.num_trees),
    }
    samples = list(ts.samples())
    weighted = 0.0
    total_span = 0.0
    for tree in ts.trees():
        tmrca = tree.tmrca(*samples)
        weighted += tmrca * tree.span
        total_span += tree.span
    out["mean_tmrca"] = weighted / total_span
    if sample_sets is not None:
        out["dxy_branch"] = ts.divergence(
            sample_sets=sample_sets, mode="branch")
    return out


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _samples_by_pop(ts, n_pops):
    """Return [pop0_samples, pop1_samples, ...] for a tskit TS."""
    out = [[] for _ in range(n_pops)]
    for s in ts.samples():
        p = ts.node(s).population
        if 0 <= p < n_pops:
            out[p].append(s)
    return out


def _run_validation(scenario_name, msinv_factory, msprime_factory,
                    n_reps=N_REPS, by_pop_dxy=False):
    """Run both engines n_reps times, assert each branch-length stat
    agrees within 3 * combined SE.
    """
    stat_names = ["pi_branch", "n_trees", "mean_tmrca"]
    if by_pop_dxy:
        stat_names.append("dxy_branch")
    msinv_vals = {k: [] for k in stat_names}
    msprime_vals = {k: [] for k in stat_names}

    for i in range(n_reps):
        ts_a = msinv_factory(seed=i)
        ts_b = msprime_factory(seed=i)
        for engine_vals, ts in (
                (msinv_vals, ts_a), (msprime_vals, ts_b)):
            sample_sets = None
            if by_pop_dxy:
                sample_sets = _samples_by_pop(ts, n_pops=2)
            stats = _stats_from_ts(ts, sample_sets)
            for k in stat_names:
                engine_vals[k].append(stats[k])

    failures = []
    lines = []
    for k in stat_names:
        m_a, se_a = _mean_se(msinv_vals[k])
        m_b, se_b = _mean_se(msprime_vals[k])
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
    print(f"\n[{scenario_name}]\n  " + "\n  ".join(lines))
    if failures:
        raise AssertionError(
            f"{scenario_name} failed:\n  " + "\n  ".join(failures))


def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""

    def msinv_factory(seed):
        return HullSimulator(
            samples=10,
            population_size=10000.0,
            sequence_length=100_000.0,
            recombination_rate=1e-8,
            inversions=[],
            seed=seed,
        ).simulate()

    def msprime_factory(seed):
        return msprime.sim_ancestry(
            samples=10,
            population_size=10000.0,
            sequence_length=100_000,
            recombination_rate=1e-8,
            ploidy=1,
            random_seed=seed + 1,
        )

    _run_validation("N1 panmictic", msinv_factory, msprime_factory)
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n1_panmictic -v -s`

Expected: PASS in ~2 s. The `-s` flag is required so the per-stat summary line prints (it goes through `print`, not the pytest reporter).

- [ ] **Step 3: Inspect the printed report**

Confirm all three stats end with `→ OK`. The expected line shapes (numbers will vary by run, but order of magnitude should match):

```
[N1 panmictic]
  pi_branch: msinv=~38000 ± ~700, msprime=~38000 ± ~700, |Δ|=<2000, 3·SE=~3000 → OK
  n_trees: msinv=~120 ± ~1.5, msprime=~120 ± ~1.5, |Δ|=<5, 3·SE=~6 → OK
  mean_tmrca: msinv=~22000 ± ~500, msprime=~22000 ± ~500, |Δ|=<2000, 3·SE=~2000 → OK
```

If a stat reports FAIL, do NOT widen the bound — investigate the Rust core. The bound is calibrated so a 3·SE fail means a real ~3% drift, not a flake.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/test_validation_msprime.py
git commit -m "$(cat <<'EOF'
test: msprime validation harness — N1 panmictic

First scenario of the msprime validation harness (spec
docs/superpowers/specs/2026-04-29-msprime-validation-design.md).
Adds tests/hull/test_validation_msprime.py with shared helpers
(_stats_from_ts, _mean_se, _run_validation) and the N1 single-pop
panmictic test (n=10, Ne=10000, L=100kb, ρ=40, 200 reps).
Compares Rust msinv vs msprime.sim_ancestry on pi_branch, num_trees,
and span-weighted mean T_MRCA via the |Δ| ≤ 3·SE bound.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: N2 two-pop migration validation

**Files:**
- Modify: `tests/hull/test_validation_msprime.py` (append second test function only; no edits to existing helpers)

- [ ] **Step 1: Append the second test**

Append to the end of `tests/hull/test_validation_msprime.py`:

```python
def test_msprime_validation_n2_two_pop_migration():
    """Rust msinv vs msprime — two-pop symmetric migration, M=1e-4."""

    def msinv_factory(seed):
        demo = Demography(
            pop_sizes=[10000.0, 10000.0],
            migration_matrix=[[0.0, 1e-4], [1e-4, 0.0]],
        )
        return HullSimulator(
            sample_config={(None, 0): 5, (None, 1): 5},
            demography=demo,
            sequence_length=100_000.0,
            recombination_rate=1e-8,
            inversions=[],
            seed=seed,
        ).simulate()

    def msprime_factory(seed):
        demo = msprime.Demography()
        demo.add_population(name="A", initial_size=10000.0)
        demo.add_population(name="B", initial_size=10000.0)
        demo.set_migration_rate(source="A", dest="B", rate=1e-4)
        demo.set_migration_rate(source="B", dest="A", rate=1e-4)
        return msprime.sim_ancestry(
            samples={"A": 5, "B": 5},
            demography=demo,
            sequence_length=100_000,
            recombination_rate=1e-8,
            ploidy=1,
            random_seed=seed + 1,
        )

    _run_validation(
        "N2 two-pop migration", msinv_factory, msprime_factory,
        by_pop_dxy=True,
    )
```

- [ ] **Step 2: Run the new test**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py::test_msprime_validation_n2_two_pop_migration -v -s`

Expected: PASS in ~4 s. Four stats reported now, including `dxy_branch`.

- [ ] **Step 3: Inspect the report**

Confirm all four stats end with `→ OK`. Order-of-magnitude expected lines:

```
[N2 two-pop migration]
  pi_branch: msinv=~50000 ± ~1000, msprime=~50000 ± ~1000, |Δ|=<3000, 3·SE=~4500 → OK
  n_trees: msinv=~120 ± ~1.5, msprime=~120 ± ~1.5, |Δ|=<5, 3·SE=~6 → OK
  mean_tmrca: msinv=~30000 ± ~700, msprime=~30000 ± ~700, |Δ|=<2500, 3·SE=~3000 → OK
  dxy_branch: msinv=~60000 ± ~2000, msprime=~60000 ± ~2000, |Δ|=<5000, 3·SE=~8000 → OK
```

If `dxy_branch` is the only stat that fails, suspect a migration-matrix-convention bug (`feedback_msprime_api.md`, CLAUDE.md migration convention block). If `pi_branch` and `n_trees` also fail, suspect a regression in the demography path itself.

- [ ] **Step 4: Run the whole new file together**

Run: `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s`

Expected: 2 passed in ~6 s.

- [ ] **Step 5: Commit**

```bash
git add tests/hull/test_validation_msprime.py
git commit -m "$(cat <<'EOF'
test: msprime validation harness — N2 two-pop migration

Second scenario of the msprime validation harness: symmetric two-pop
island model (Ne=[10000,10000], M=1e-4, n=5+5). Adds dxy_branch
(between-pop branch-length divergence) as a fourth comparison stat
for this scenario; the migration-matrix convention check
(spec §"Migration convention check") sits behind it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Bump CLAUDE.md test counts

**Files:**
- Modify: `CLAUDE.md` (Tests section, the per-language test-count lines)

- [ ] **Step 1: Get the current pytest count**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py --collect-only -q 2>&1 | tail -3`

Expected last line: `<N> tests collected in <T>s` where N is the new total. The current spec doc reports "181 passed, 3 skipped" — adding two new functions in this plan should bring it to **183 passed, 3 skipped** (subject to verification on the actual run; record whatever the pytest output says rather than assuming).

- [ ] **Step 2: Update the Python test-count line in CLAUDE.md**

The current line in `CLAUDE.md` (Tests section) reads:

```
- Python: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py`
  (181 passed, 3 skipped as of 2026-04-29; the 12 sweep-rewrite follow-up skips are now active
  after Phases A-D of `docs/superpowers/plans/2026-04-29-sweep-followups.md`).
```

Edit: change `181 passed, 3 skipped` → `<NEW> passed, 3 skipped`, and append a new bullet at the end of the same block (immediately under the existing test-file taxonomy bullet, before the next subheading):

```
- msprime validation: `tests/hull/test_validation_msprime.py` (N1 panmictic + N2 two-pop migration; runs ~6 s; spec `docs/superpowers/specs/2026-04-29-msprime-validation-design.md`)
```

Use the actual N from Step 1, not a guess. If the count is something other than 183, that means another test landed elsewhere — investigate before bumping the number.

- [ ] **Step 3: Verify the full pytest run still passes**

Run: `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -5`

Expected: `<NEW> passed, 3 skipped in <T>s`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: bump CLAUDE.md test count for msprime validation harness

Adds one bullet to the Tests section pointing at
tests/hull/test_validation_msprime.py and updates the per-suite
test count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

Run before handing off:

- [ ] `.venv/bin/python -m pytest tests/hull/test_validation_msprime.py -v -s` → 2 passed, every stat line ends with `→ OK`.
- [ ] `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -3` → previous total + 2 passed, 3 skipped, no new errors.
- [ ] `cd rust && cargo test --release 2>&1 | tail -3` → still 132 lib + 17 integration + 4 sweep-anchor + 2 sweep-trajectory passing (this plan does not touch Rust; the cargo run is a sanity check that nothing was inadvertently rebuilt).
- [ ] `git log --oneline -5` → three new commits with the messages above, all on `main` (or your active branch).
- [ ] CLAUDE.md test count line matches the pytest output, not a guess.

## Notes for the implementer

- **Auto mode is on**, so commits are expected after each task without further confirmation. If a test FAILs, **stop and surface the report** — do not widen the 3·SE bound; the spec calibrates it intentionally to flag real drift.
- **Why no TDD red-green loop here:** The system under test (Rust core) already exists. This is a validation test that should pass on first write because the engine is correct. If your first run fails, that is a real signal, not a normal red-green step.
- **Why `-s` on pytest:** `_run_validation` uses `print(...)` for the per-stat summary so a failure log is interpretable without re-running. `-s` is required for those lines to surface in the terminal during test execution.
- **Seed handling:** `msprime.sim_ancestry(random_seed=...)` requires `>= 1`; we pass `seed + 1`. msinv accepts any 64-bit seed including 0. Rep `i = 0` is a valid msinv seed but msprime would reject `0`, hence the `+1`.
- **No msprime in CI requirements check:** `msprime==1.4.1` is already a venv dep (verified pre-spec). No `pyproject.toml` or `requirements.txt` edit is needed. If a future CI build fails on missing msprime, that is a CI-image concern, not a plan concern.
