# Exact Karyotype Mode, Plan 1: Tier 0 Expectation Calculator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `msinv.expect`, the tier 0 calculator: exact single-site
expected pairwise coalescence times (mean, variance, survival) and expected
SFS / Illex `f1` under an inversion history, sharing the simulators'
frequency trajectories and demography.

**Architecture:** A block-counting continuous-time Markov chain over
lineages labelled by (population, karyotype state) is enumerated once per
sample (`_chain.py`); its generator is a time-weighted sum of constant sparse
matrices, one per rate type, and the forward equations are integrated
segment by segment between forced boundaries with scipy BDF, with an exact
linear solve after the last boundary (`_solve.py`). Frequencies come from the
Rust trajectory objects through a new `InvTrajectory` PyO3 class, so tier 0
reads exactly what tiers 1 and 2 read.

**Tech Stack:** Python ≥ 3.9, numpy, scipy (sparse, `solve_ivp`, `splu`,
`expm_multiply`, `quad`), Rust/PyO3 0.24 (bridge only), pytest, msprime
(reference tests only).

**Spec:** `docs/superpowers/specs/2026-10-02-exact-karyotype-mode-design.md`
(§3.1, §3.2, §3.4 rule 4, §3.5, §4.8 tier 0 part, §5.B). Plans 2 (tier 2
exact mode, Rust) and 3 (release validation campaign) follow separately.

## Global Constraints

- Python target py39: `from __future__ import annotations` in every new
  module; no `match`, no runtime `X | Y` types. Ruff: line length 88, rules
  E/F/W/B/UP (`pyproject.toml [tool.ruff]`).
- Always `.venv/bin/python`; never system python. Use `.venv/bin/python -m pip`.
- Rust build + install: `cd rust && cargo build --release -p msinv-py` then
  `/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so`
  (explicit `/bin/cp`; the hook blocks the copy while any
  `python.*(pytest|msinv)` process runs).
- msinv `population_size=N` is diploid Ne: per-pair coalescence rate 1/(2N).
- Migration convention (verified `simulator.rs:521, 887`): backward in time a
  lineage in population d moves to d' at hazard `migration_matrix[d'][d]`.
  msprime's `migration_matrix[j][k]` (lineage in j → k) is the transpose.
- `t_inv` is global (`t_inv_max()`); per-population trajectory values are
  support boundaries of f_d(I, t).
- `origin` ∈ {`"barrier"` (default), `"single"`}.
- Solver defaults: rtol 1e-8, atol 1e-12. `max_states` default 10⁶, an
  argument of the SFS functions only.
- Per-site flux switching coefficient: γ · C(y) (spec §3.4 rule 4), not γ · φ(y).
- scipy is an optional extra (`msinv[expect]`); core msinv must still import
  without scipy.
- Commits: new commits only, `-c commit.gpgsign=false`, HEREDOC messages.

## Review Focus

1. A sample in a state with zero frequency at t = 0 (e.g. I in a population
   with p = 0) → clear `ValueError`, not a silent zero-rate chain. Test in
   Task 5.
2. Lineages in populations that are never connected (no migration, no join)
   → `ValueError` saying they can never coalesce, not a scipy "singular
   matrix" traceback. Test in Task 5.
3. Sites exactly at a breakpoint (`y == bp_left`, `y == bp_right`) → finite
   results; `bp_right` is a flank site at δ = 0 (no switching). Test in Task 5.
4. Exponential growth still active after the last demographic event → the
   final constant-rate solve is impossible; raise `NotImplementedError` at
   construction. Test in Task 5.
5. Demographic events tier 0 cannot express (`cmig`, `eig`) →
   `NotImplementedError` naming the event, not silently ignored. Test in
   Task 2.

---

### Task 1: Expose trajectories to Python (`InvTrajectory`)

**Files:**
- Modify: `rust/msinv-core/src/trajectory.rs` (trait `Trajectory` at line 35;
  impls at lines 366, 524, 684, 889, 1055, 1193)
- Modify: `rust/msinv-py/src/lib.rs` (trajectory statement at lines 348–495;
  module registration at line 633)
- Modify: `msinv/hull/_rust_bridge.py:58-73`
- Create: `tests/expect/__init__.py` (empty), `tests/expect/test_inv_trajectory.py`

**Interfaces:**
- Produces (Rust): `Trajectory::knots(&self) -> Vec<f64>` (default empty).
- Produces (Python): `msinv._msinv_core.InvTrajectory(inversion_dict)` with
  methods `p_inv(times: list[float], pop: int) -> list[float]`,
  `t_inv(pop: int) -> float`, `t_inv_max() -> float`, `knots() -> list[float]`.
- Produces (Python): `msinv.hull._rust_bridge.inversion_to_dict(inv, n_pops) -> dict`.

- [ ] **Step 1: Write the failing Rust test** — append to
  `rust/msinv-core/src/trajectory.rs`:

```rust
#[cfg(test)]
mod knots_tests {
    use super::*;

    #[test]
    fn closed_form_trajectories_have_no_knots() {
        assert!(ConstantTrajectory::new(vec![0.3], 100.0).knots().is_empty());
        assert!(DeterministicTrajectory::new(0.5, 1000.0, 0.01).knots().is_empty());
    }

    #[test]
    fn precomputed_knots_are_its_time_grid() {
        let tr = PrecomputedTrajectory::with_t_inv(
            vec![0.0, 10.0, 20.0],
            vec![vec![0.5, 0.4, 0.3]],
            vec![1000.0],
            Some(vec![20.0]),
        );
        assert_eq!(tr.knots(), vec![0.0, 10.0, 20.0]);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd rust && cargo test --release -p msinv-core knots_tests`
Expected: compile error `no method named knots`.

- [ ] **Step 3: Add `knots` to the trait and grid-based impls** — in
  `pub trait Trajectory`, after `fn n_pops(&self) -> usize;`:

```rust
    /// Time grid on which the trajectory is linearly interpolated
    /// (backward generations). Empty for closed-form trajectories.
    /// Used by the tier 0 calculator (`msinv.expect`) to place
    /// integration boundaries.
    fn knots(&self) -> Vec<f64> {
        Vec::new()
    }
```

and inside each of `impl Trajectory for StochasticTrajectory`,
`IntegerWFTrajectory`, `StochasticDeterministicTrajectory`,
`BridgeStochasticTrajectory`, `CoupledTrajectory`, `PrecomputedTrajectory`:

```rust
    fn knots(&self) -> Vec<f64> {
        self.times.clone()
    }
```

- [ ] **Step 4: Run Rust tests**

Run: `cd rust && cargo test --release -p msinv-core knots_tests`
Expected: 2 passed.

- [ ] **Step 5: Write the failing Python test** — `tests/expect/test_inv_trajectory.py`:

```python
import pytest

from msinv._msinv_core import InvTrajectory
from msinv.hull import InversionSpec
from msinv.hull._rust_bridge import inversion_to_dict


def test_constant_trajectory_eval():
    inv = InversionSpec(
        bp_left=10.0, bp_right=20.0, p_inv=0.3, t_inv=100.0, mean_tract_length=0.0
    )
    tr = InvTrajectory(inversion_to_dict(inv, 1))
    assert tr.p_inv([0.0, 50.0, 100.0, 150.0], 0) == [0.3, 0.3, 0.0, 0.0]
    assert tr.t_inv_max() == 100.0
    assert tr.knots() == []


def test_precomputed_trajectory_is_linear_and_exposes_knots():
    traj = {
        "type": "precomputed",
        "times": [0.0, 100.0, 200.0],
        "freqs": [[0.6, 0.4, 0.2]],
        "n_e": [1000.0],
        "t_inv": [200.0],
    }
    inv = InversionSpec(
        bp_left=10.0, bp_right=20.0, trajectory=traj, mean_tract_length=0.0
    )
    tr = InvTrajectory(inversion_to_dict(inv, 1))
    assert tr.p_inv([50.0], 0)[0] == pytest.approx(0.5)
    assert tr.knots() == [0.0, 100.0, 200.0]
```

- [ ] **Step 6: Factor the trajectory builder in `lib.rs`.** Add above
  `simulate_raw`:

```rust
/// Build one inversion's frequency trajectory from its inversion dict
/// (`'trajectory'` sub-dict, or legacy `'p_inv'` + `'t_inv'`). Shared by
/// `simulate_raw` and `InvTrajectory` so tier 0 reads the same
/// frequencies as the simulators.
fn build_trajectory_from_inv_dict(
    d: &Bound<'_, PyDict>,
) -> PyResult<Box<dyn Trajectory + Send + Sync>> {
    // Body: the `let trajectory: Box<dyn Trajectory + Send + Sync> = if let
    // Some(traj_obj) = d.get_item("trajectory")? { ... } else { ... };`
    // statement moved here verbatim from simulate_raw (lib.rs:348-495).
    Ok(trajectory)
}
```

Move that statement (the whole `// Build trajectory` statement, every match
arm unchanged) into the function body above `Ok(trajectory)`, and in
`simulate_raw` replace it with:

```rust
            let trajectory = build_trajectory_from_inv_dict(d)?;
```

Then add the class after the function:

```rust
/// Read-only handle on one inversion's frequency trajectory, built from
/// the same inversion dict `simulate_raw` takes (see `msinv.expect`).
#[pyclass(name = "InvTrajectory")]
struct PyInvTrajectory {
    inner: Box<dyn Trajectory + Send + Sync>,
}

#[pymethods]
impl PyInvTrajectory {
    #[new]
    fn new(inversion: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self { inner: build_trajectory_from_inv_dict(inversion)? })
    }

    /// Inverted-arrangement frequency in `pop` at each backward time.
    fn p_inv(&self, times: Vec<f64>, pop: u32) -> Vec<f64> {
        times.iter().map(|&t| self.inner.p_inv_at(t, pop)).collect()
    }

    fn t_inv(&self, pop: u32) -> f64 {
        self.inner.t_inv(pop)
    }

    fn t_inv_max(&self) -> f64 {
        self.inner.t_inv_max()
    }

    fn knots(&self) -> Vec<f64> {
        self.inner.knots()
    }
}
```

and register it in `_msinv_core` after `m.add_class::<PySweep>()?;`:

```rust
    m.add_class::<PyInvTrajectory>()?;
```

- [ ] **Step 7: Factor `inversion_to_dict` in `_rust_bridge.py`.** Add at
  module level:

```python
def inversion_to_dict(inv, n_pops: int) -> dict:
    """Inversion dict in the shape ``simulate_raw`` and ``InvTrajectory`` take."""
    d = {
        "bp_left": float(inv.bp_left),
        "bp_right": float(inv.bp_right),
        "gene_conversion_rate": float(inv.gene_conversion_rate),
        "mean_tract_length": float(inv.mean_tract_length),
        "tract_distribution": str(inv.tract_distribution),
    }
    if getattr(inv, "trajectory", None) is not None:
        d["trajectory"] = dict(inv.trajectory)
    else:
        d["p_inv"] = inv._p_inv_as_list(n_pops)
        d["t_inv"] = float(inv.t_inv)
    return d
```

and replace the body of the `for inv in simulator.inversions:` loop with
`inv_dicts.append(inversion_to_dict(inv, n_pops))`.

- [ ] **Step 8: Build, install, run tests**

```bash
cd rust && cargo build --release -p msinv-py && \
  /bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
cd .. && .venv/bin/python -m pytest tests/expect/test_inv_trajectory.py -v
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py -q
cd rust && cargo test --release
```

Expected: new tests pass; hull suite and Rust suite unchanged (no new
failures vs. HEAD).

- [ ] **Step 9: Commit**

```bash
git add rust/msinv-core/src/trajectory.rs rust/msinv-py/src/lib.rs \
  msinv/hull/_rust_bridge.py tests/expect/__init__.py tests/expect/test_inv_trajectory.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(py): expose inversion trajectories to Python as InvTrajectory

Factor trajectory construction out of simulate_raw so tier 0 reads the same
frequencies; add Trajectory::knots for grid-based trajectories.
EOF
)"
```

---

### Task 2: Package skeleton, `parent` field, demography epochs

**Files:**
- Modify: `msinv/hull/inversion.py` (dataclass fields ~line 85; `__post_init__`)
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Create: `msinv/expect/__init__.py`, `msinv/expect/_history.py`
- Test: `tests/expect/test_epochs.py`

**Interfaces:**
- Produces: `InversionSpec.parent: Optional[int] = None` (raises if set).
- Produces: `msinv.expect._history.Epochs(demography)` with
  `n_pops`, `epochs: list[Epoch]`, `boundaries() -> list[float]`,
  `size(d, t) -> float`, `mig_rate(d_from, d_to, t) -> float`,
  `joins_at(t) -> list[tuple[int, int]]` (src, dst),
  `final_growth_is_zero() -> bool`, `mig_pairs() -> set[tuple[int, int]]`
  ((d_from, d_to) with a positive rate in some epoch).

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_epochs.py`:

```python
import math

import pytest

from msinv.expect._history import Epochs
from msinv.hull import Demography, InversionSpec


def test_migration_convention_matches_simulator():
    d = Demography([1000.0, 2000.0], migration_matrix=[[0.0, 0.01], [0.02, 0.0]])
    e = Epochs(d)
    # Backward, a lineage in src moves to dst at migration_matrix[dst][src].
    assert e.mig_rate(0, 1, 0.0) == 0.02
    assert e.mig_rate(1, 0, 0.0) == 0.01
    assert e.mig_pairs() == {(0, 1), (1, 0)}


def test_size_change_growth_and_join():
    d = Demography([1000.0, 2000.0], migration_matrix=[[0.0, 0.01], [0.02, 0.0]])
    d.add_population_size_change(100.0, population=0, new_size=500.0)
    d.add_growth_rate_change(200.0, population=1, growth_rate=0.001)
    d.add_population_split(300.0, derived=[1], ancestral=0)
    e = Epochs(d)
    assert e.boundaries() == [100.0, 200.0, 300.0]
    assert e.size(0, 50.0) == 1000.0
    assert e.size(0, 150.0) == 500.0
    assert e.size(1, 250.0) == pytest.approx(2000.0 * math.exp(-0.001 * 50.0))
    assert e.joins_at(300.0) == [(1, 0)]
    assert e.mig_rate(0, 1, 350.0) == 0.0
    assert e.mig_rate(1, 0, 350.0) == 0.0
    assert not e.final_growth_is_zero()


def test_unsupported_event_raises():
    d = Demography([1000.0, 1000.0])
    d.add_inversion_freq_change(10.0, 0, 0, 0.5)
    with pytest.raises(NotImplementedError, match="eig"):
        Epochs(d)


def test_inversion_parent_not_supported():
    with pytest.raises(NotImplementedError, match="nested"):
        InversionSpec(bp_left=0.0, bp_right=10.0, p_inv=0.5, t_inv=100.0, parent=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_epochs.py -v`
Expected: `ModuleNotFoundError: No module named 'msinv.expect'`.

- [ ] **Step 3: Add the `parent` field** — in `InversionSpec`, after
  `trajectory: Optional[dict] = None`:

```python
    # Nesting (exact karyotype spec, step 3). Not supported yet.
    parent: Optional[int] = None
```

and as the first statement of `__post_init__`:

```python
        if self.parent is not None:
            raise NotImplementedError(
                "nested inversions (parent=...) are not supported yet; see "
                "docs/superpowers/specs/2026-10-02-exact-karyotype-mode-design.md §6"
            )
```

- [ ] **Step 4: Add the optional extra** — in `pyproject.toml`
  `[project.optional-dependencies]` add `expect = ["scipy>=1.7"]` and change
  `all` to `"msinv[test,plots,illex,expect]"`.

- [ ] **Step 5: Create `msinv/expect/__init__.py`:**

```python
"""Tier 0 expectation calculator (exact karyotype mode spec, §3.5).

Exact single-site expectations under an inversion history: pairwise
coalescence times and the expected site frequency spectrum, from a Markov
chain over lineages labelled by (population, karyotype state). Reads the same
``Demography`` and ``InversionSpec`` objects as ``HullSimulator``. Requires
scipy: ``pip install msinv[expect]``.
"""

from __future__ import annotations

from ._history import Epochs

__all__ = ["Epochs"]
```

- [ ] **Step 6: Create `msinv/expect/_history.py`:**

```python
"""History inputs for tier 0: demography epochs, frequencies, flux coverage."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

_SUPPORTED_EVENTS = ("eN", "en", "eG", "eg", "eM", "em", "ej")


def _size_at(n0: float, growth: float, growth_start: float, t: float) -> float:
    if growth == 0.0:
        return n0
    return n0 * math.exp(-growth * (t - growth_start))


@dataclass
class Epoch:
    """Demographic state from ``start`` until the next epoch."""

    start: float
    sizes: list
    growth: list
    growth_start: list
    mig: list  # msinv convention: mig[dst][src], lineage in src moves to dst
    joins: list = field(default_factory=list)  # (src, dst) applied at start


class Epochs:
    """Piecewise demography read from :class:`msinv.hull.Demography`.

    Events are replayed exactly as ``Demography.apply_event_at`` applies them.
    Backward in time a lineage in population ``d`` moves to ``d2`` at hazard
    ``migration_matrix[d2][d]`` (simulator.rs:887).
    """

    def __init__(self, demography):
        self.n_pops = demography.n_pops
        self.epochs = [
            Epoch(
                0.0,
                [float(x) for x in demography.pop_sizes],
                [float(x) for x in demography.growth_rates],
                [float(x) for x in demography.growth_start],
                [[float(x) for x in row] for row in demography.migration_matrix],
            )
        ]
        events = sorted(demography.events, key=lambda e: e[1])
        for ev in events:
            if ev[0] not in _SUPPORTED_EVENTS:
                raise NotImplementedError(
                    f"tier 0 does not support demographic event {ev[0]!r} "
                    f"(supported: {', '.join(_SUPPORTED_EVENTS)})"
                )
        i = 0
        while i < len(events):
            t = float(events[i][1])
            group = []
            while i < len(events) and abs(events[i][1] - t) <= 1e-9:
                group.append(events[i])
                i += 1
            self.epochs.append(self._apply(self.epochs[-1], t, group))
        self._starts = [e.start for e in self.epochs]

    def _apply(self, prev: Epoch, t: float, group: list) -> Epoch:
        n = self.n_pops
        sizes, growth, g0 = list(prev.sizes), list(prev.growth), list(prev.growth_start)
        mig = [list(row) for row in prev.mig]
        joins = []
        for ev in group:
            kind = ev[0]
            if kind == "eN":
                for p in range(n):
                    sizes[p], growth[p], g0[p] = float(ev[2]), 0.0, t
            elif kind == "en":
                p = ev[2]
                if p < n:
                    sizes[p], growth[p], g0[p] = float(ev[3]), 0.0, t
            elif kind == "eG":
                for p in range(n):
                    sizes[p] = _size_at(sizes[p], growth[p], g0[p], t)
                    growth[p], g0[p] = float(ev[2]), t
            elif kind == "eg":
                p = ev[2]
                if p < n:
                    sizes[p] = _size_at(sizes[p], growth[p], g0[p], t)
                    growth[p], g0[p] = float(ev[3]), t
            elif kind == "eM":
                per = float(ev[2]) / (n - 1) if n > 1 else 0.0
                for a in range(n):
                    for b in range(n):
                        if a != b:
                            mig[a][b] = per
            elif kind == "em":
                _, _, dst, src, rate = ev
                if dst < n and src < n and dst != src:
                    mig[dst][src] = float(rate)
            elif kind == "ej":
                _, _, src, dst = ev
                joins.append((int(src), int(dst)))
                for k in range(n):
                    mig[src][k] = 0.0
                    mig[k][src] = 0.0
        return Epoch(t, sizes, growth, g0, mig, joins)

    def boundaries(self) -> list:
        return self._starts[1:]

    def epoch_at(self, t: float) -> Epoch:
        return self.epochs[bisect.bisect_right(self._starts, t) - 1]

    def size(self, d: int, t: float) -> float:
        e = self.epoch_at(t)
        return _size_at(e.sizes[d], e.growth[d], e.growth_start[d], t)

    def mig_rate(self, d_from: int, d_to: int, t: float) -> float:
        """Backward hazard of a lineage in ``d_from`` moving to ``d_to``."""
        return self.epoch_at(t).mig[d_to][d_from]

    def joins_at(self, t: float) -> list:
        return [j for e in self.epochs if abs(e.start - t) <= 1e-9 for j in e.joins]

    def final_growth_is_zero(self) -> bool:
        return all(g == 0.0 for g in self.epochs[-1].growth)

    def mig_pairs(self) -> set:
        n = self.n_pops
        return {
            (s, d)
            for e in self.epochs
            for d in range(n)
            for s in range(n)
            if s != d and e.mig[d][s] > 0.0
        }
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_epochs.py -v`
Expected: 4 passed. Also `.venv/bin/python -m pytest tests/hull/test_phase3b_inversion_spec_validation.py -q` still passes.

- [ ] **Step 8: Commit**

```bash
git add msinv/hull/inversion.py pyproject.toml msinv/expect tests/expect/test_epochs.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): tier 0 package skeleton and demography epochs

Epochs replays Demography events as apply_event_at does; InversionSpec gains
a parent field that raises until nesting is supported.
EOF
)"
```

---

### Task 3: Frequency history and flux coverage

**Files:**
- Modify: `msinv/expect/_history.py`, `msinv/expect/__init__.py`
- Test: `tests/expect/test_frequency.py`

**Interfaces:**
- Consumes: `InvTrajectory`, `inversion_to_dict` (Task 1).
- Produces: `FrequencyHistory(inversion, n_pops)` with attributes `t_inv`,
  `t_inv_pop`, `knots`; methods `p_all(t) -> list[float]`,
  `f(d, state, t) -> float` (`state` in {"S","I"}; raises at `t >= t_inv`),
  `support_boundaries() -> list[tuple[float, int, str]]`,
  `draw_conditional(d, t, known: dict, rng) -> dict`,
  `origin_population() -> int`.
- Produces: `flux_coverage(y, bp_left, bp_right, mean_tract_length, tract_distribution) -> float`.

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_frequency.py`:

```python
import numpy as np
import pytest

from msinv.expect._history import FrequencyHistory, flux_coverage
from msinv.hull import InversionSpec

L = 100_000.0


def _inv(**kw):
    return InversionSpec(bp_left=0.0, bp_right=L, mean_tract_length=0.0, **kw)


TWO_POP = {
    "type": "precomputed",
    "times": [0.0, 250.0, 500.0, 1000.0],
    "freqs": [[0.5] * 4, [0.5, 0.25, 0.0, 0.0]],
    "n_e": [1000.0, 1000.0],
    "t_inv": [1000.0, 500.0],
}


def test_constant_frequencies():
    fh = FrequencyHistory(_inv(p_inv=0.3, t_inv=1000.0), 1)
    assert fh.f(0, "I", 10.0) == 0.3
    assert fh.f(0, "S", 10.0) == pytest.approx(0.7)
    assert fh.t_inv == 1000.0
    assert fh.knots == []
    assert fh.support_boundaries() == []
    with pytest.raises(ValueError):
        fh.f(0, "I", 1000.0)


def test_per_population_t_inv_is_a_support_boundary():
    fh = FrequencyHistory(_inv(trajectory=TWO_POP), 2)
    assert fh.t_inv == 1000.0
    assert fh.support_boundaries() == [(500.0, 1, "I")]
    assert fh.f(1, "I", 125.0) == pytest.approx(0.375)
    assert fh.origin_population() == 0


def test_origin_population_must_be_unique():
    fh = FrequencyHistory(_inv(p_inv={0: 0.3, 1: 0.4}, t_inv=1000.0), 2)
    with pytest.raises(ValueError, match="exactly one population"):
        fh.origin_population()


def test_draw_conditional():
    fh = FrequencyHistory(_inv(trajectory=TWO_POP), 2)
    rng = np.random.default_rng(1)
    draws = [fh.draw_conditional(0, 10.0, {}, rng)[0] for _ in range(20000)]
    assert abs(draws.count("I") / 20000 - 0.5) < 0.015
    assert fh.draw_conditional(0, 10.0, {0: "S"}, rng) == {0: "S"}
    with pytest.raises(ValueError, match="zero frequency"):
        fh.draw_conditional(1, 600.0, {0: "I"}, rng)


def test_flux_coverage_flat_region_fixed_tract():
    w = 100.0 / L
    c = flux_coverage(50_000.0, 0.0, L, 100.0, "fixed")
    assert c == pytest.approx(100.0 * w / (1.0 - w), rel=1e-8)


def test_flux_coverage_flat_region_geometric_tract():
    w = 100.0 / L
    c = flux_coverage(50_000.0, 0.0, L, 100.0, "geometric")
    assert c == pytest.approx(100.0 * w / (1.0 - w), rel=1e-6)


def test_flux_coverage_outside_and_at_breakpoint():
    assert flux_coverage(-1.0, 0.0, L, 100.0, "fixed") == 0.0
    assert flux_coverage(L, 0.0, L, 100.0, "fixed") == 0.0
    assert flux_coverage(50_000.0, 0.0, L, 0.0, "fixed") == 0.0
    assert 0.0 <= flux_coverage(0.0, 0.0, L, 100.0, "fixed") < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_frequency.py -v`
Expected: `ImportError: cannot import name 'FrequencyHistory'`.

- [ ] **Step 3: Implement** — add to the imports at the top of
  `msinv/expect/_history.py`:

```python
from msinv._msinv_core import InvTrajectory
from msinv.hull._rust_bridge import inversion_to_dict
```

and append:

```python
class FrequencyHistory:
    """Inversion frequencies per population, read from the Rust trajectory.

    ``t_inv`` is global (``t_inv_max``). Per-population values of coupled or
    precomputed trajectories are support boundaries of f_d(I, t), not
    separate inversion ages (spec §3.2).
    """

    def __init__(self, inversion, n_pops: int):
        self.n_pops = n_pops
        self._traj = InvTrajectory(inversion_to_dict(inversion, n_pops))
        self.t_inv = float(self._traj.t_inv_max())
        self.t_inv_pop = [float(self._traj.t_inv(d)) for d in range(n_pops)]
        self.knots = sorted(
            {float(k) for k in self._traj.knots() if 0.0 < k < self.t_inv}
        )
        self._cache_t = None
        self._cache_p = None
        self._check_simplex()

    def p_all(self, t: float) -> list:
        """I frequency in every population at backward time ``t``."""
        if t != self._cache_t:
            self._cache_p = [self._traj.p_inv([t], d)[0] for d in range(self.n_pops)]
            self._cache_t = t
        return self._cache_p

    def f(self, d: int, state: str, t: float) -> float:
        if t >= self.t_inv:
            raise ValueError(f"no karyotype states at or beyond t_inv={self.t_inv}")
        p = self.p_all(t)[d]
        if state == "I":
            return p
        if state == "S":
            return 1.0 - p
        raise ValueError(f"state must be 'S' or 'I', got {state!r}")

    def _check_simplex(self) -> None:
        # One inversion: f_S = 1 - p is derived, so only p needs checking.
        for t in [0.0, *self.knots]:
            for d, p in enumerate(self.p_all(t)):
                if not -1e-12 <= p <= 1.0 + 1e-12:
                    raise ValueError(
                        f"inversion frequency {p} outside [0, 1] in population "
                        f"{d} at t={t}"
                    )

    def support_boundaries(self) -> list:
        """Sorted (t_b, pop, state): f_pop(state) becomes zero at t_b < t_inv."""
        out = set()
        for d in range(self.n_pops):
            if self.t_inv_pop[d] < self.t_inv:
                out.add((self.t_inv_pop[d], d, "I"))
            prev = self.p_all(0.0)[d]
            for k in self.knots:
                p = self.p_all(k)[d]
                if p <= 0.0 < prev:
                    out.add((k, d, "I"))
                if p >= 1.0 > prev:
                    out.add((k, d, "S"))
                prev = p
        return sorted(out)

    def draw_conditional(self, d: int, t: float, known: dict, rng) -> dict:
        """Draw the karyotypes not in ``known`` (inv_id -> 'S'/'I')."""
        if 0 in known:
            if self.f(d, known[0], t) <= 0.0:
                raise ValueError(
                    f"conditioning state {known[0]!r} has zero frequency in "
                    f"population {d} at t={t}"
                )
            return dict(known)
        return {0: "I" if rng.random() < self.f(d, "I", t) else "S"}

    def origin_population(self) -> int:
        eps = 1e-9 * max(1.0, self.t_inv)
        pops = [d for d, p in enumerate(self.p_all(self.t_inv - eps)) if p > 0.0]
        if len(pops) != 1:
            raise ValueError(
                "origin='single' needs exactly one population with f(I) > 0 "
                f"just below t_inv; found {pops}"
            )
        return pops[0]


def _phi(u: float, w: float) -> float:
    """Peischl phi, identical to rust/msinv-core/src/phi.rs::phi."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if w >= 1.0:
        return 1.0
    return min(max(min(u, 1.0 - u, w) / (1.0 - w), 0.0), 1.0)


def _cover_prob(x: float, yr: float, length: float, inv_len: float) -> float:
    """P(tract started from x with this length covers yr); mirrors draw_tract."""
    lo, hi = max(x - length, 0.0), min(x, inv_len - length)
    if hi <= lo:
        b1 = min(max(x - length / 2.0, 0.0), inv_len - length)
        return 1.0 if b1 <= yr <= b1 + length else 0.0
    a, b = max(lo, yr - length), min(hi, yr)
    return max(b - a, 0.0) / (hi - lo)


def flux_coverage(
    y: float,
    bp_left: float,
    bp_right: float,
    mean_tract_length: float,
    tract_distribution: str,
) -> float:
    """C(y) of spec §3.4 rule 4: per-site conversion hazard / (gamma * f_other).

    For a lineage carrying the whole inversion, under the tier 2 event rules
    (``sample_flux_position`` + ``draw_tract``). About phi(y) * mean tract
    length away from the breakpoints.
    """
    from scipy.integrate import quad

    inv_len = bp_right - bp_left
    if not bp_left <= y < bp_right or mean_tract_length <= 0.0:
        return 0.0
    w = mean_tract_length / inv_len
    yr = y - bp_left

    def given_length(length: float) -> float:
        length = min(length, 0.99 * inv_len)
        lo, hi = max(0.0, yr - length), min(inv_len, yr + length)
        if hi <= lo:
            return 0.0
        kinks = (yr, yr - length / 2, yr + length / 2, length, inv_len - length,
                 w * inv_len, (1 - w) * inv_len)
        pts = [p for p in kinks if lo < p < hi]
        val, _ = quad(
            lambda x: _phi(x / inv_len, w) * _cover_prob(x, yr, length, inv_len),
            lo, hi, points=pts or None, limit=200, epsabs=0.0, epsrel=1e-10,
        )
        return val

    if tract_distribution == "fixed":
        return given_length(mean_tract_length)
    cap, m = 0.99 * inv_len, mean_tract_length
    body, _ = quad(
        lambda length: given_length(length) * math.exp(-length / m) / m,
        0.0, cap, limit=200, epsabs=0.0, epsrel=1e-8,
    )
    return body + math.exp(-cap / m) * given_length(cap)
```

Update `msinv/expect/__init__.py` imports/`__all__` to
`from ._history import Epochs, FrequencyHistory, flux_coverage` and
`__all__ = ["Epochs", "FrequencyHistory", "flux_coverage"]`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_frequency.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add msinv/expect tests/expect/test_frequency.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): frequency history and per-site flux coverage C(y)

Frequencies come from the Rust trajectory; per-population t_inv values are
support boundaries. C(y) mirrors sample_flux_position + draw_tract.
EOF
)"
```

---

### Task 4: Block-counting chain enumeration

**Files:**
- Create: `msinv/expect/_chain.py`
- Test: `tests/expect/test_chain.py`

**Interfaces:**
- Produces: constants `S, I, P = 0, 1, 2`, `ABSORBED = 0`;
  `cell(pop, k) -> int`, `pop_of(c)`, `k_of(c)`, `canon(lineages) -> tuple`.
- Produces: `Chain(sample, n_pops, mig_pairs, joins, switching, origin,
  origin_pop, max_states=None)` with attributes `n` (sample size), `size`,
  `states` (index → state tuple; index 0 is `None` = absorbed), `index`
  (state → index), `init` (initial index), `AT` (dict key → csr transpose
  generator; keys `("coal", d, k)`, `("mig", d, d2, k)`, `("sw", d, k)`),
  `BT` (dict boundary name → csr transpose map; names `("join", src, dst)`
  and `"t_inv"`), `rewards` (array (n−1, size): lineages of block b+1 per
  state), `cell_mask` (dict cell → bool array), `transient` (bool array),
  `post` (bool array: all lineages panmictic, i.e. after `t_inv`).

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_chain.py`:

```python
import numpy as np
import pytest

from msinv.expect._chain import I, P, S, Chain, canon, cell


def _chain(sample, n_pops=1, mig=(), joins=(), switching=True, origin="barrier",
           origin_pop=None, max_states=None):
    return Chain(sample, n_pops, set(mig), list(joins), switching, origin,
                 origin_pop, max_states)


def test_pair_chain_states_one_population():
    ch = _chain([(0, "I"), (0, "I")])
    # Before t_inv: II, IS, SS. After: PP. Plus the absorbed state.
    assert ch.size == 5


def test_sfs_chain_size_n20():
    # Bicoloured partitions of 20 with >= 2 parts (24,842 - 2), partitions of
    # 20 with >= 2 parts after t_inv (627 - 1), plus the absorbed state.
    assert _chain([(0, "I")] * 20).size == 24840 + 626 + 1
    assert _chain([(0, "I")] * 20, switching=False).size == 626 + 626 + 1


def test_max_states_guard():
    with pytest.raises(ValueError, match="max_states"):
        _chain([(0, "I")] * 20, max_states=1000)


def test_generators_conserve_probability():
    ch = _chain([(0, "I"), (0, "S"), (1, "I")], n_pops=2, mig=[(0, 1), (1, 0)],
                joins=[(1, 0)])
    for at in ch.AT.values():
        assert np.allclose(np.asarray(at.sum(axis=0)).ravel(), 0.0)
    for bt in ch.BT.values():
        assert np.allclose(np.asarray(bt.sum(axis=0)).ravel(), 1.0)


def test_single_origin_merges_inverted_lineages():
    ch = _chain([(0, "I"), (0, "I"), (0, "S")], switching=False, origin="single",
                origin_pop=0)
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 1), (cell(0, S), 1)])]
    j = ch.BT["t_inv"][:, i].nonzero()[0][0]
    assert ch.states[j] == canon([(cell(0, P), 1), (cell(0, P), 2)])


def test_rewards_count_blocks():
    ch = _chain([(0, "I")] * 3, switching=False)
    i = ch.index[canon([(cell(0, I), 1), (cell(0, I), 2)])]
    assert ch.rewards[:, i].tolist() == [1.0, 1.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_chain.py -v`
Expected: `ModuleNotFoundError: No module named 'msinv.expect._chain'`.

- [ ] **Step 3: Implement `msinv/expect/_chain.py`:**

```python
"""Block-counting chain over lineages labelled by (population, state).

A state is a sorted tuple of (cell, block) pairs, one per lineage, where
cell = 3 * population + k (k: 0 = S, 1 = I, 2 = P, panmictic after t_inv)
and block is the number of sampled chromosomes the lineage subtends
(Hobolth, Siri-Jegousse & Bladt 2019). Every state with a single lineage is
merged into one absorbing state, index 0.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import scipy.sparse as sp

S, I, P = 0, 1, 2
K_OF = {"S": S, "I": I}
ABSORBED = 0


def cell(pop: int, k: int) -> int:
    return 3 * pop + k


def pop_of(c: int) -> int:
    return c // 3


def k_of(c: int) -> int:
    return c % 3


def canon(lineages) -> tuple:
    return tuple(sorted(lineages))


class Chain:
    """Enumerated chain for one sample.

    ``sample``: list of (pop, 'S'|'I'), one per sampled chromosome.
    ``mig_pairs``: (d_from, d_to) with positive migration in some epoch.
    ``joins``: (src, dst) population joins (ej) at any time.
    ``switching``: whether the site's switching coefficient is positive.
    """

    def __init__(self, sample, n_pops, mig_pairs, joins, switching, origin,
                 origin_pop, max_states=None):
        self.n = len(sample)
        if self.n < 2:
            raise ValueError("need at least two sampled chromosomes")
        self.n_pops = n_pops
        self._mig_pairs = sorted(mig_pairs)
        self._joins = sorted(set(joins))
        self._switching = switching
        self._origin = origin
        self._origin_pop = origin_pop
        self._max_states = max_states
        self.states = [None]
        self.index = {}
        self._edges = {}
        self._maps = {}
        start = canon((cell(p, K_OF[s]), 1) for p, s in sample)
        queue = deque()
        self.init = self._target(start, queue)
        while queue:
            s = queue.popleft()
            i = self.index[s]
            for key, t, mult in self._moves(s):
                self._edges.setdefault(key, []).append((i, self._target(t, queue), mult))
            for name, t in self._boundary_images(s):
                self._maps.setdefault(name, []).append((i, self._target(t, queue)))
        self.size = len(self.states)
        self._build_arrays()

    def _target(self, state: tuple, queue) -> int:
        if len(state) == 1:
            return ABSORBED
        if state not in self.index:
            self.index[state] = len(self.states)
            self.states.append(state)
            queue.append(state)
            if self._max_states is not None and len(self.states) > self._max_states:
                raise ValueError(
                    f"tier 0 state space exceeds max_states={self._max_states}; "
                    "use the simulator (tier 1/2) for this sample"
                )
        return self.index[state]

    def _moves(self, s: tuple):
        cnt = Counter(s)
        types = sorted(cnt)
        lin = list(s)
        for ai, a in enumerate(types):
            for b in types[ai:]:
                if a[0] != b[0]:
                    continue
                mult = cnt[a] * (cnt[a] - 1) // 2 if a == b else cnt[a] * cnt[b]
                if mult == 0:
                    continue
                rest = list(lin)
                rest.remove(a)
                rest.remove(b)
                rest.append((a[0], a[1] + b[1]))
                yield ("coal", pop_of(a[0]), k_of(a[0])), canon(rest), mult
        for a in types:
            c, blk = a
            d, k = pop_of(c), k_of(c)
            for src, dst in self._mig_pairs:
                if src != d:
                    continue
                rest = list(lin)
                rest.remove(a)
                rest.append((cell(dst, k), blk))
                yield ("mig", d, dst, k), canon(rest), cnt[a]
            if self._switching and k != P:
                rest = list(lin)
                rest.remove(a)
                rest.append((cell(d, 1 - k), blk))
                yield ("sw", d, k), canon(rest), cnt[a]

    def _boundary_images(self, s: tuple):
        for src, dst in self._joins:
            if any(pop_of(c) == src for c, _ in s):
                yield ("join", src, dst), canon(
                    (cell(dst, k_of(c)), b) if pop_of(c) == src else (c, b)
                    for c, b in s
                )
        if any(k_of(c) != P for c, _ in s):
            yield "t_inv", self._inv_image(s)

    def _inv_image(self, s: tuple) -> tuple:
        if self._origin == "single":
            merged = [b for c, b in s if k_of(c) == I]
            rest = [(cell(pop_of(c), P), b) for c, b in s if k_of(c) != I]
            if merged:
                rest.append((cell(self._origin_pop, P), sum(merged)))
            return canon(rest)
        return canon((cell(pop_of(c), P), b) for c, b in s)

    def _build_arrays(self) -> None:
        n = self.size
        self.AT = {}
        for key, edges in self._edges.items():
            e = np.asarray(edges, dtype=float)
            a = sp.coo_matrix(
                (e[:, 2], (e[:, 0].astype(int), e[:, 1].astype(int))), shape=(n, n)
            ).tocsr()
            a = a - sp.diags(np.asarray(a.sum(axis=1)).ravel())
            self.AT[key] = a.T.tocsr()
        self.BT = {}
        for name, pairs in self._maps.items():
            to = np.arange(n)
            for i, j in pairs:
                to[i] = j
            b = sp.csr_matrix((np.ones(n), (np.arange(n), to)), shape=(n, n))
            self.BT[name] = b.T.tocsr()
        self.rewards = np.zeros((self.n - 1, n))
        self.cell_mask = {}
        for idx in range(1, n):
            for c, b in self.states[idx]:
                self.rewards[b - 1, idx] += 1.0
                if c not in self.cell_mask:
                    self.cell_mask[c] = np.zeros(n, dtype=bool)
                self.cell_mask[c][idx] = True
        self.transient = np.ones(n, dtype=bool)
        self.transient[ABSORBED] = False
        # States after t_inv (all lineages panmictic): the only ones that can
        # hold mass in the final constant-rate phase.
        self.post = np.array(
            [s is not None and all(k_of(c) == P for c, _ in s) for s in self.states]
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_chain.py -v`
Expected: 6 passed (the n = 20 test enumerates ~25k states; allow a few
seconds).

- [ ] **Step 5: Commit**

```bash
git add msinv/expect/_chain.py tests/expect/test_chain.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): block-counting chain over (population, state) cells

Enumerates states once per sample, one constant sparse generator per rate
type, boundary maps for joins and t_inv (barrier or single origin).
EOF
)"
```

---

### Task 5: Solver and pairwise times

**Files:**
- Create: `msinv/expect/_solve.py`, `msinv/expect/model.py`
- Modify: `msinv/expect/__init__.py`
- Test: `tests/expect/test_pairwise.py`

**Interfaces:**
- Consumes: `Epochs`, `FrequencyHistory`, `flux_coverage` (Tasks 2–3),
  `Chain` and constants (Task 4).
- Produces: `ImpossibleHistoryError(ValueError)`;
  `Result(branch_lengths (R, n−1), mean (R,), second_moment (R,), survival (R, T))`;
  `Solver(chain, epochs, freq, coeffs, rtol, atol, force_knots_max).run(times) -> Result`.
- Produces: `ExpectModel(demography, inversion, recombination_rate,
  origin="barrier", rtol=1e-8, atol=1e-12, force_knots_max=1000)` with
  `switching_coefficient(y)`, `pairwise_times(y, a, b, times=None) ->
  PairwiseTimes(mean, var, survival)` (`a`, `b` are `(pop, "S"|"I")`;
  `times` ascending), `profile(ys, a, b) -> np.ndarray` of means.

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_pairwise.py`:

```python
import math

import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.hull import Demography, InversionSpec

N, P_I, T_INV = 1000.0, 0.3, 2000.0
BL, BR = 10_000.0, 20_000.0
Y_IN = 15_000.0


def _model(origin="barrier", t_inv=T_INV, p=P_I, demo=None):
    inv = InversionSpec(bp_left=BL, bp_right=BR, p_inv=p, t_inv=t_inv,
                        mean_tract_length=0.0)
    return ExpectModel(demo or Demography([N]), inv, 1e-8, origin=origin)


@pytest.mark.parametrize("state,f", [("I", P_I), ("S", 1 - P_I)])
def test_same_state_barrier_closed_form(state, f):
    a = 2 * N * f
    res = _model().pairwise_times(Y_IN, (0, state), (0, state), times=[1000.0, 3000.0])
    expected = a * (1 - math.exp(-T_INV / a)) + 2 * N * math.exp(-T_INV / a)
    assert res.mean == pytest.approx(expected, rel=1e-6)
    assert res.survival[0] == pytest.approx(math.exp(-1000.0 / a), rel=1e-6)
    assert res.survival[1] == pytest.approx(
        math.exp(-T_INV / a) * math.exp(-1000.0 / (2 * N)), rel=1e-6
    )


def test_between_states_barrier_closed_form():
    res = _model().pairwise_times(Y_IN, (0, "I"), (0, "S"))
    assert res.mean == pytest.approx(T_INV + 2 * N, rel=1e-6)
    assert res.var == pytest.approx((2 * N) ** 2, rel=1e-5)


def test_single_origin_closed_form():
    a = 2 * N * P_I
    res = _model("single").pairwise_times(Y_IN, (0, "I"), (0, "I"))
    assert res.mean == pytest.approx(a * (1 - math.exp(-T_INV / a)), rel=1e-6)


def test_old_inversion_limit():
    res = _model(t_inv=1e7).pairwise_times(Y_IN, (0, "I"), (0, "I"))
    assert res.mean == pytest.approx(2 * N * P_I, rel=1e-6)


def test_flank_site_matches_hand_solved_chain():
    r, delta = 1e-6, 500.0
    m = ExpectModel(Demography([N]), InversionSpec(
        bp_left=BL, bp_right=BR, p_inv=P_I, t_inv=1e9, mean_tract_length=0.0), r)
    c, p = r * delta, P_I
    q = np.array([
        [-(1 / (2 * N * p) + 2 * c * (1 - p)), 2 * c * (1 - p), 0.0],
        [c * p, -c, c * (1 - p)],
        [0.0, 2 * c * p, -(1 / (2 * N * (1 - p)) + 2 * c * p)],
    ])  # transient states II, IS, SS (row generator)
    expected = np.linalg.solve(-q, np.ones(3))
    y = BL - delta
    assert m.pairwise_times(y, (0, "I"), (0, "I")).mean == pytest.approx(
        expected[0], rel=1e-6)
    assert m.pairwise_times(y, (0, "I"), (0, "S")).mean == pytest.approx(
        expected[1], rel=1e-6)


def test_profile_matches_pointwise():
    m = _model(t_inv=1e5)
    m.r = 1e-6
    ys = [Y_IN, BL - 500.0, BR + 2000.0]
    prof = m.profile(ys, (0, "I"), (0, "I"))
    for y, v in zip(ys, prof):
        assert v == pytest.approx(m.pairwise_times(y, (0, "I"), (0, "I")).mean,
                                  rel=1e-7)


def test_equal_frequencies_reduce_to_plain_island_model():
    # Spec §5.B item 4: with f_0(I) = f_1(I) the conditional migration rate is
    # plain m, so I lineages follow a two-deme island model with deme size
    # N * p: E[T_between] = 4 N p + 1 / (2 m).
    m_rate = 1e-3
    demo = Demography([N, N], migration_matrix=[[0.0, m_rate], [m_rate, 0.0]])
    res = _model(t_inv=1e9, demo=demo).pairwise_times(Y_IN, (0, "I"), (1, "I"))
    assert res.mean == pytest.approx(4 * N * P_I + 1 / (2 * m_rate), rel=1e-6)


# --- Review focus ---------------------------------------------------------

def test_zero_frequency_sample_rejected():
    m = _model(p={0: 0.0, 1: 0.5}, demo=Demography([N, N]))
    with pytest.raises(ValueError, match="zero frequency"):
        m.pairwise_times(Y_IN, (0, "I"), (1, "I"))


def test_disconnected_populations_rejected():
    m = _model(demo=Demography([N, N]))
    with pytest.raises(ValueError, match="never coalesce"):
        m.pairwise_times(Y_IN, (0, "S"), (1, "S"))


def test_breakpoint_sites_are_finite():
    m = _model()
    m.r = 1e-6
    inside = m.pairwise_times(Y_IN, (0, "I"), (0, "I")).mean
    assert m.pairwise_times(BL, (0, "I"), (0, "I")).mean == pytest.approx(inside)
    assert m.pairwise_times(BR, (0, "I"), (0, "I")).mean == pytest.approx(inside)


def test_final_growth_rejected():
    demo = Demography([N])
    demo.add_growth_rate_change(100.0, population=0, growth_rate=1e-3)
    with pytest.raises(NotImplementedError, match="growth"):
        _model(demo=demo)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_pairwise.py -v`
Expected: `ImportError: cannot import name 'ExpectModel'`.

- [ ] **Step 3: Implement `msinv/expect/_solve.py`:**

```python
"""Time integration of the tier 0 chain (spec §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply, splu

from ._chain import I, P, S

_NAME = {S: "S", I: "I"}


class ImpossibleHistoryError(ValueError):
    """Ancestry is left in a (population, state) cell with no support."""


@dataclass
class Result:
    branch_lengths: np.ndarray  # (R, n-1): E[L_b], generations
    mean: np.ndarray  # (R,): E[T_MRCA]
    second_moment: np.ndarray  # (R,): E[T_MRCA^2]
    survival: np.ndarray  # (R, len(times)): P(T_MRCA > t)


class Solver:
    """Integrate R replicas of one chain that differ only in the switching
    coefficient (one per site), as one block-diagonal system."""

    def __init__(self, chain, epochs, freq, coeffs, rtol=1e-8, atol=1e-12,
                 force_knots_max=1000):
        self.ch, self.ep, self.fq = chain, epochs, freq
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.R = len(self.coeffs)
        self.rtol, self.atol = rtol, atol
        eye = sp.identity(self.R, format="csr")
        self._AT = {}
        for key, at in chain.AT.items():
            left = sp.diags(self.coeffs) if key[0] == "sw" else eye
            self._AT[key] = sp.kron(left, at, format="csr")
        self._BT = {k: sp.kron(eye, bt, format="csr") for k, bt in chain.BT.items()}
        self.N = chain.size * self.R
        self._rew = sp.kron(eye, sp.csr_matrix(chain.rewards), format="csr")
        self._smat = sp.kron(
            eye, sp.csr_matrix(chain.transient.astype(float)[None, :]), format="csr"
        )
        self._trans = np.tile(chain.transient, self.R)
        self._post = np.tile(chain.post, self.R)
        self._block = np.repeat(np.arange(self.R), chain.size)
        self._n_aux = self.R * (chain.n - 1) + 2 * self.R
        knots = freq.knots if len(freq.knots) <= force_knots_max else []
        self._supp = freq.support_boundaries()
        pts = set(epochs.boundaries()) | set(knots) | {freq.t_inv}
        pts |= {t for t, _, _ in self._supp}
        self.breaks = sorted(t for t in pts if 0.0 < t < np.inf)
        self._memo_t, self._memo_q = None, None

    # -- rates -------------------------------------------------------------

    def _g(self, key, t: float) -> float:
        kind, d = key[0], key[1]
        pre = t < self.fq.t_inv
        if kind == "coal":
            k = key[2]
            n_d = self.ep.size(d, t)
            if k == P:
                return 1.0 / (2.0 * n_d)
            if not pre:
                return 0.0
            f = self.fq.f(d, _NAME[k], t)
            return 1.0 / (2.0 * n_d * f) if f > 0.0 else 0.0
        if kind == "mig":
            d2, k = key[2], key[3]
            m = self.ep.mig_rate(d, d2, t)
            if m == 0.0:
                return 0.0
            if k == P:
                return m
            if not pre:
                return 0.0
            fd = self.fq.f(d, _NAME[k], t)
            return m * self.fq.f(d2, _NAME[k], t) / fd if fd > 0.0 else 0.0
        if not pre:  # "sw": the per-site coefficient is in the matrix
            return 0.0
        return self.fq.f(d, _NAME[1 - key[2]], t)

    def qt(self, t: float):
        """Transpose generator at t: d(pi)/dt = qt(t) @ pi."""
        out = sp.csr_matrix((self.N, self.N))
        for key, at in self._AT.items():
            g = self._g(key, t)
            if g != 0.0:
                out = out + g * at
        return out

    def _qt_memo(self, t: float):
        if t != self._memo_t:
            self._memo_t, self._memo_q = t, self.qt(t)
        return self._memo_q

    def _rhs(self, t, z):
        q = self._qt_memo(t)
        pi = z[: self.N]
        s = self._smat @ pi
        return np.concatenate([q @ pi, self._rew @ pi, s, t * s])

    def _jac(self, t, z):
        q = self._qt_memo(t)
        m = self._n_aux
        top = sp.hstack([q, sp.csr_matrix((self.N, m))])
        low = sp.vstack([self._rew, self._smat, t * self._smat])
        bottom = sp.hstack([low, sp.csr_matrix((m, m))])
        return sp.vstack([top, bottom], format="csc")

    # -- boundaries --------------------------------------------------------

    def _apply_boundaries(self, tb: float, pi):
        for src, dst in self.ep.joins_at(tb):
            bt = self._BT.get(("join", src, dst))
            if bt is not None:
                pi = bt @ pi
        if tb == self.fq.t_inv and "t_inv" in self._BT:
            pi = self._BT["t_inv"] @ pi
        return pi

    def _flush(self, tb: float, cells: list, pi):
        raise NotImplementedError(
            "trajectories whose frequency reaches zero before t_inv are not "
            "supported yet"
        )

    # -- main loop ---------------------------------------------------------

    def run(self, times=None) -> Result:
        times = np.asarray([] if times is None else times, dtype=float)
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be in ascending order")
        surv = np.full((self.R, len(times)), np.nan)
        z = np.zeros(self.N + self._n_aux)
        pi0 = np.zeros(self.ch.size)
        pi0[self.ch.init] = 1.0
        z[: self.N] = self._apply_boundaries(0.0, np.tile(pi0, self.R))
        t = 0.0
        for tb in self.breaks:
            cells = [(d, st) for tt, d, st in self._supp if tt == tb]
            end = tb - 1e-9 * max(1.0, tb) if cells else tb
            sel = (times >= t) & (times < tb)
            if end > t:
                t_eval = np.append(np.minimum(times[sel], end), end)
                sol = solve_ivp(self._rhs, (t, end), z, method="BDF", jac=self._jac,
                                rtol=self.rtol, atol=self.atol, t_eval=t_eval)
                if not sol.success:
                    raise RuntimeError(f"ODE solver failed on [{t}, {end}]: {sol.message}")
                surv[:, sel] = self._smat @ sol.y[: self.N, :-1]
                z = sol.y[:, -1].copy()
            if cells:
                z[: self.N] = self._flush(tb, cells, z[: self.N])
            z[: self.N] = self._apply_boundaries(tb, z[: self.N])
            t = tb
        return self._finish(t, z, times, surv)

    def _finish(self, t: float, z, times, surv) -> Result:
        n_rw = self.R * (self.ch.n - 1)
        pi = z[: self.N]
        rw = z[self.N : self.N + n_rw].copy()
        m1 = z[self.N + n_rw : self.N + n_rw + self.R].copy()
        m2 = z[self.N + n_rw + self.R :].copy()
        # After the last boundary (always >= t_inv) only post-t_inv states hold
        # mass; pre-t_inv states have zero rates there and would make the
        # system singular.
        idx = np.flatnonzero(self._trans & self._post)
        pi_t = pi[idx]
        late = np.flatnonzero(times >= t)
        if pi_t.sum() > self.atol:
            qtt = self.qt(t)[idx][:, idx].tocsc()
            try:
                lu = splu((-qtt.T).tocsc())
            except RuntimeError as exc:
                raise ValueError(
                    "lineages can never coalesce after the last demographic "
                    "event: no migration or population join connects their "
                    "populations"
                ) from exc
            blk = self._block[idx]
            v1 = lu.solve(np.ones(len(idx)))
            v2 = lu.solve(v1)
            rw += pi_t @ lu.solve(self._rew[:, idx].T.toarray())
            add1 = np.bincount(blk, weights=pi_t * v1, minlength=self.R)
            m1 += add1
            m2 += t * add1 + np.bincount(blk, weights=pi_t * v2, minlength=self.R)
            for j in late:
                v = expm_multiply(qtt * (times[j] - t), pi_t)
                surv[:, j] = np.bincount(blk, weights=v, minlength=self.R)
        else:
            surv[:, late] = 0.0
        return Result(rw.reshape(self.R, self.ch.n - 1), m1, 2.0 * m2, surv)
```

- [ ] **Step 4: Implement `msinv/expect/model.py`:**

```python
"""Public tier 0 API (spec §4.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._chain import Chain
from ._history import Epochs, FrequencyHistory, flux_coverage
from ._solve import Solver


@dataclass
class PairwiseTimes:
    mean: float
    var: float
    survival: Optional[np.ndarray]  # P(T > t) at the requested times


class ExpectModel:
    """Exact single-site expectations under one inversion history."""

    def __init__(self, demography, inversion, recombination_rate,
                 origin="barrier", rtol=1e-8, atol=1e-12, force_knots_max=1000):
        if origin not in ("barrier", "single"):
            raise ValueError(f"origin must be 'barrier' or 'single', got {origin!r}")
        if recombination_rate <= 0.0:
            raise ValueError("recombination_rate must be > 0")
        self.epochs = Epochs(demography)
        if not self.epochs.final_growth_is_zero():
            raise NotImplementedError(
                "tier 0 needs zero growth after the last demographic event "
                "(the final constant-rate phase is solved exactly)"
            )
        self.freq = FrequencyHistory(inversion, demography.n_pops)
        self.inversion = inversion
        self.r = float(recombination_rate)
        self.origin = origin
        self.origin_pop = self.freq.origin_population() if origin == "single" else None
        self._opts = {"rtol": rtol, "atol": atol, "force_knots_max": force_knots_max}

    def switching_coefficient(self, y: float) -> float:
        """Per-lineage switching hazard at site y, before the frequency factor."""
        inv = self.inversion
        if inv.bp_left <= y < inv.bp_right:
            return inv.gene_conversion_rate * flux_coverage(
                y, inv.bp_left, inv.bp_right, inv.mean_tract_length,
                inv.tract_distribution)
        delta = inv.bp_left - y if y < inv.bp_left else y - inv.bp_right
        return self.r * delta

    def _check_sample(self, sample) -> None:
        for pop, st in sample:
            if not 0 <= pop < self.epochs.n_pops:
                raise ValueError(f"population {pop} out of range")
            if self.freq.f(pop, st, 0.0) <= 0.0:
                raise ValueError(
                    f"sampled state {st!r} has zero frequency in population "
                    f"{pop} at t=0"
                )

    def _solve(self, sample, ys, times=None, max_states=None):
        sample = list(sample)
        self._check_sample(sample)
        coeffs = [self.switching_coefficient(y) for y in ys]
        joins = [j for e in self.epochs.epochs for j in e.joins]
        chain = Chain(sample, self.epochs.n_pops, self.epochs.mig_pairs(), joins,
                      any(c > 0.0 for c in coeffs), self.origin, self.origin_pop,
                      max_states)
        return Solver(chain, self.epochs, self.freq, coeffs, **self._opts).run(times)

    def pairwise_times(self, y, a, b, times=None) -> PairwiseTimes:
        """Coalescence time of chromosomes a and b, each (pop, 'S'|'I')."""
        res = self._solve([a, b], [y], times)
        mean = float(res.mean[0])
        surv = None if times is None else res.survival[0].copy()
        return PairwiseTimes(mean, float(res.second_moment[0] - mean**2), surv)

    def profile(self, ys, a, b) -> np.ndarray:
        """Mean pairwise time at each site in ys, in one integration."""
        return self._solve([a, b], list(ys)).mean.copy()
```

Update `msinv/expect/__init__.py`:

```python
from ._history import Epochs, FrequencyHistory, flux_coverage
from ._solve import ImpossibleHistoryError
from .model import ExpectModel, PairwiseTimes

__all__ = [
    "Epochs",
    "ExpectModel",
    "FrequencyHistory",
    "ImpossibleHistoryError",
    "PairwiseTimes",
    "flux_coverage",
]
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_pairwise.py -v`
Expected: 13 passed. If a closed form misses at rel 1e-6, do not loosen the
tolerance: check `rtol`/`atol` reach `solve_ivp` and that `t_inv` is in
`breaks`.

- [ ] **Step 6: Commit**

```bash
git add msinv/expect tests/expect/test_pairwise.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): tier 0 solver and pairwise coalescence times

Segment-wise BDF integration between demographic, trajectory and t_inv
boundaries; exact linear solve after the last boundary. Closed forms for
barrier and single origin, flank chain, review-focus inputs tested.
EOF
)"
```

---

### Task 6: Zero-support boundaries and impossible histories

**Files:**
- Modify: `msinv/expect/_solve.py` (`_flush`)
- Test: `tests/expect/test_boundaries.py`

**Interfaces:**
- Consumes: `Solver`, `ImpossibleHistoryError`, `Chain.cell_mask` (Tasks 4–5).
- Produces: `Solver._flush(tb, cells, pi) -> pi`, the limit-chain rule of
  spec §3.5.

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_boundaries.py`:

```python
import math

import numpy as np
import pytest
from scipy.integrate import quad

from msinv.expect import ExpectModel, ImpossibleHistoryError
from msinv.hull import Demography, InversionSpec

Y = 15_000.0


def _two_pop(freqs_pop1, mig, origin="barrier"):
    traj = {
        "type": "precomputed",
        "times": [0.0, 250.0, 500.0, 1000.0],
        "freqs": [[0.5] * 4, freqs_pop1],
        "n_e": [1000.0, 1000.0],
        "t_inv": [1000.0, 500.0],
    }
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, trajectory=traj,
                        mean_tract_length=0.0)
    return ExpectModel(Demography([1000.0, 1000.0], migration_matrix=mig), inv,
                       1e-8, origin=origin)


def test_frequency_jump_to_zero_without_exit_is_impossible():
    m = _two_pop([0.5, 0.5, 0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ImpossibleHistoryError, match="population 1"):
        m.pairwise_times(Y, (1, "I"), (0, "I"))


def test_continuous_decline_with_migration_matches_limit():
    # Lineage A starts in (1, I); backward hazard 1 -> 0 is mig[0][1] = 1e-3
    # times f_0(I) / f_1(I) = 1 / (1 - t/500), so A is still in pop 1 at time
    # t with probability (1 - t/500)^(1/2). With u = sqrt(1 - tau/500) the
    # arrival time tau is uniform in u. After arrival A coalesces with B in
    # (0, I) at 1/(2*1000*0.5) until t_inv = 1000, then at 1/2000.
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]])

    def total(tau):
        rate = 1e-3
        left = 1000.0 - tau
        return tau + (1 - math.exp(-rate * left)) / rate + math.exp(-rate * left) * 2000.0

    expected, _ = quad(lambda u: total(500.0 * (1 - u * u)), 0.0, 1.0,
                       epsabs=0.0, epsrel=1e-10)
    got = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert got == pytest.approx(expected, rel=1e-5)


def test_single_origin_with_support_boundary():
    m = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]], origin="single")
    barrier = _two_pop([0.5, 0.25, 0.0, 0.0], [[0.0, 1e-3], [0.0, 0.0]])
    single = m.pairwise_times(Y, (1, "I"), (0, "I")).mean
    assert np.isfinite(single)
    assert single < barrier.pairwise_times(Y, (1, "I"), (0, "I")).mean
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_boundaries.py -v`
Expected: 3 failures with `NotImplementedError: trajectories whose frequency
reaches zero before t_inv are not supported yet`.

- [ ] **Step 3: Implement `_flush`** — replace the stub in `_solve.py`:

```python
    def _flush(self, tb: float, cells: list, pi):
        """Resolve ancestry left in cells whose frequency reaches zero at tb.

        All exit hazards from such a cell scale as 1/f, so as f -> 0 the
        remaining mass leaves by the rescaled chain: coalescence within the
        cell at 1/(2N) and migration at m(d -> d') f_d'(K) (spec §3.5). Mass
        this chain cannot move, or mass in a cell whose frequency jumps to
        zero, means the history is impossible.
        """
        tl = tb - 1e-9 * max(1.0, tb)
        for d, st in cells:
            k = 0 if st == "S" else 1
            c = 3 * d + k
            if c not in self.ch.cell_mask:
                continue
            mask = np.tile(self.ch.cell_mask[c], self.R)
            mass = float(pi[mask].sum())
            if mass <= self.atol:
                continue
            msg = (f"history impossible: {mass:.3g} of ancestry remains in "
                   f"population {d}, state {st} at t={tb}, where its frequency "
                   "is zero and no lineage can leave")
            if self.fq.f(d, st, tl) > 1e-6:
                raise ImpossibleHistoryError(msg)
            qts = sp.csr_matrix((self.N, self.N))
            for key, at in self._AT.items():
                if key[0] == "coal" and key[1] == d and key[2] == k:
                    qts = qts + at / (2.0 * self.ep.size(d, tl))
                elif key[0] == "mig" and key[1] == d and key[3] == k:
                    rate = self.ep.mig_rate(d, key[2], tl) * self.fq.f(key[2], st, tl)
                    if rate > 0.0:
                        qts = qts + rate * at
            stuck = mask & (qts.diagonal() >= 0.0)
            tr = np.flatnonzero(mask & ~stuck)
            ab = np.flatnonzero(~mask | stuck)
            new = pi.copy()
            if len(tr):
                y = splu((-qts[tr][:, tr]).tocsc()).solve(pi[tr])
                new[tr] = 0.0
                new[ab] += qts[ab][:, tr] @ y
            if new[stuck].sum() > self.atol:
                raise ImpossibleHistoryError(msg)
            new[stuck] = 0.0
            pi = new
        return pi
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_boundaries.py tests/expect/test_pairwise.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add msinv/expect/_solve.py tests/expect/test_boundaries.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): zero-support boundaries via the rescaled limit chain

Log-divergent exits are resolved exactly at the boundary; frequency jumps
to zero or cells without exit raise ImpossibleHistoryError.
EOF
)"
```

---

### Task 7: Expected SFS and `f1`

**Files:**
- Modify: `msinv/expect/model.py`
- Test: `tests/expect/test_sfs.py`

**Interfaces:**
- Consumes: `ExpectModel._solve` (Task 5).
- Produces: `ExpectModel.expected_branch_lengths(y, sample, max_states=10**6) -> np.ndarray`
  (E[L_b], b = 1..n−1, generations; E[ξ_b] = μ · E[L_b] per site),
  `expected_sfs(y, pop, state, n, max_states=10**6)`,
  `f1(y, pop, state, n=20, max_states=10**6) -> float`, module function
  `fold(unfolded) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** — `tests/expect/test_sfs.py`:

```python
import numpy as np
import pytest

from illex.scripts.sfs_shape import project_unfolded
from msinv.expect import ExpectModel
from msinv.expect.model import fold
from msinv.hull import Demography, InversionSpec

N = 1000.0


def _model(t_inv, p=0.3, r=1e-8):
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv,
                        mean_tract_length=0.0)
    return ExpectModel(Demography([N]), inv, r)


def test_kingman_after_t_inv():
    sfs = _model(t_inv=1e-6).expected_sfs(15_000.0, 0, "I", 6)
    b = np.arange(1, 6)
    assert np.allclose(sfs, 4 * N / b, rtol=1e-5)


def test_old_inversion_is_kingman_at_reduced_size():
    sfs = _model(t_inv=1e7).expected_sfs(15_000.0, 0, "I", 6)
    b = np.arange(1, 6)
    assert np.allclose(sfs, 4 * N * 0.3 / b, rtol=1e-6)


def test_pair_branch_length_is_twice_pairwise_time():
    m = _model(t_inv=2000.0, r=1e-6)
    y = 9_500.0
    lb = m.expected_branch_lengths(y, [(0, "I"), (0, "I")])
    assert lb[0] / 2 == pytest.approx(m.pairwise_times(y, (0, "I"), (0, "I")).mean,
                                      rel=1e-7)


def test_fold_convention():
    assert fold(np.array([1.0, 2.0, 3.0, 4.0, 5.0])).tolist() == [6.0, 6.0, 3.0]


def test_f1_equals_illex_projection_of_larger_sample():
    # Projecting the expected SFS of 12 chromosomes to 10 must equal the
    # expected SFS of 10 (spec §3.5), folded as in illex/scripts/sfs_shape.py.
    m = _model(t_inv=4000.0, r=1e-6)
    y = 9_000.0
    big = m.expected_sfs(y, 0, "I", 12)
    folded = project_unfolded(np.concatenate([[0.0], big, [0.0]]), 12, 10)
    assert m.f1(y, 0, "I", n=10) == pytest.approx(folded[0] / folded.sum(), rel=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/expect/test_sfs.py -v`
Expected: `ImportError: cannot import name 'fold'`.

- [ ] **Step 3: Implement** — add to `msinv/expect/model.py`, module level:

```python
def fold(unfolded) -> np.ndarray:
    """Fold E[L_b], b = 1..n-1, as illex/scripts/sfs_shape.py does."""
    unf = np.asarray(unfolded, dtype=float)
    n = len(unf) + 1
    out = np.zeros(n // 2)
    for b in range(1, n):
        out[min(b, n - b) - 1] += unf[b - 1]
    return out
```

and methods on `ExpectModel`:

```python
    def expected_branch_lengths(self, y, sample, max_states=10**6) -> np.ndarray:
        """E[L_b], b = 1..n-1: expected length (generations) of branches
        subtending b of the sampled chromosomes; E[xi_b] = mu * E[L_b]."""
        res = self._solve(sample, [y], max_states=max_states)
        return res.branch_lengths[0].copy()

    def expected_sfs(self, y, pop, state, n, max_states=10**6) -> np.ndarray:
        """Unfolded expected SFS (per unit mutation rate) of n chromosomes."""
        return self.expected_branch_lengths(y, [(pop, state)] * n, max_states)

    def f1(self, y, pop, state, n=20, max_states=10**6) -> float:
        """Illex f1: folded first class over total, ratio of expectations."""
        folded = fold(self.expected_sfs(y, pop, state, n, max_states))
        return float(folded[0] / folded.sum())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/expect/test_sfs.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add msinv/expect/model.py tests/expect/test_sfs.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(expect): expected SFS and Illex f1 from the block-counting chain

f1 is the folded ratio of expectations; checked against the Illex exact
projection of a larger sample.
EOF
)"
```

---

### Task 8: msprime single-site reference (`origin="barrier"`)

**Files:**
- Test: `tests/expect/test_msprime_reference.py`

**Interfaces:**
- Consumes: `ExpectModel` (Tasks 5, 7).

- [ ] **Step 1: Write the tests.** Each (population, state) is an msprime
  deme with `ploidy=1` and size `2 · N · f`; msprime `set_migration_rate(source, dest)`
  is the backward rate of a lineage moving source → dest.

```python
import msprime
import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.hull import Demography, InversionSpec

pytestmark = pytest.mark.slow
REPS = 4000


def _branch_sfs(dem, samples, n):
    out = []
    for ts in msprime.sim_ancestry(samples=samples, demography=dem, ploidy=1,
                                   sequence_length=1, num_replicates=REPS,
                                   random_seed=7):
        out.append(ts.allele_frequency_spectrum(mode="branch", polarised=True)[1:n])
    a = np.asarray(out)
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(REPS)


def test_one_population_flank_site():
    n_e, p, t_inv, r, delta = 1000.0, 0.3, 3000.0, 1e-6, 300.0
    c = r * delta
    dem = msprime.Demography()
    dem.add_population(name="S", initial_size=2 * n_e * (1 - p))
    dem.add_population(name="I", initial_size=2 * n_e * p)
    dem.add_population(name="P", initial_size=2 * n_e)
    dem.set_migration_rate(source="I", dest="S", rate=c * (1 - p))
    dem.set_migration_rate(source="S", dest="I", rate=c * p)
    dem.add_population_split(time=t_inv, derived=["S", "I"], ancestral="P")
    mean, se = _branch_sfs(dem, {"I": 6}, 6)
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv,
                        mean_tract_length=0.0)
    model = ExpectModel(Demography([n_e]), inv, r)
    got = model.expected_sfs(10_000.0 - delta, 0, "I", 6)
    assert np.all(np.abs(got - mean) < 4 * se)


def test_two_populations_conditional_migration():
    n_e, p0, p1, t_inv, m = 1000.0, 0.2, 0.6, 3000.0, 5e-4
    dem = msprime.Demography()
    for name, size in (("0S", 2 * n_e * (1 - p0)), ("0I", 2 * n_e * p0),
                       ("1S", 2 * n_e * (1 - p1)), ("1I", 2 * n_e * p1),
                       ("0P", 2 * n_e), ("1P", 2 * n_e)):
        dem.add_population(name=name, initial_size=size)
    # Conditional rates m * f_dest(K) / f_src(K) (spec §3.4 rule 3).
    dem.set_migration_rate(source="0I", dest="1I", rate=m * p1 / p0)
    dem.set_migration_rate(source="1I", dest="0I", rate=m * p0 / p1)
    dem.set_migration_rate(source="0S", dest="1S", rate=m * (1 - p1) / (1 - p0))
    dem.set_migration_rate(source="1S", dest="0S", rate=m * (1 - p0) / (1 - p1))
    dem.set_migration_rate(source="0P", dest="1P", rate=m)
    dem.set_migration_rate(source="1P", dest="0P", rate=m)
    dem.add_population_split(time=t_inv, derived=["0S", "0I"], ancestral="0P")
    dem.add_population_split(time=t_inv, derived=["1S", "1I"], ancestral="1P")
    mean, se = _branch_sfs(dem, {"0I": 1, "1I": 1}, 2)
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0,
                        p_inv={0: p0, 1: p1}, t_inv=t_inv, mean_tract_length=0.0)
    model = ExpectModel(Demography([n_e, n_e], migration_matrix=[[0, m], [m, 0]]),
                        inv, 1e-8)
    got = model.pairwise_times(15_000.0, (0, "I"), (1, "I")).mean
    assert abs(got - mean[0] / 2) < 4 * se[0] / 2


def test_size_change_epoch():
    n_e, p, t_inv = 1000.0, 0.4, 3000.0
    dem = msprime.Demography()
    dem.add_population(name="S", initial_size=2 * n_e * (1 - p))
    dem.add_population(name="I", initial_size=2 * n_e * p)
    dem.add_population(name="P", initial_size=n_e)  # 2 * (N / 2) after 800
    dem.add_population_parameters_change(time=800.0, population="I",
                                         initial_size=n_e * p)
    dem.add_population_parameters_change(time=800.0, population="S",
                                         initial_size=n_e * (1 - p))
    dem.add_population_split(time=t_inv, derived=["S", "I"], ancestral="P")
    dem.sort_events()
    mean, se = _branch_sfs(dem, {"I": 4}, 4)
    demo = Demography([n_e])
    demo.add_population_size_change(800.0, population=0, new_size=n_e / 2)
    inv = InversionSpec(bp_left=10_000.0, bp_right=20_000.0, p_inv=p, t_inv=t_inv,
                        mean_tract_length=0.0)
    got = ExpectModel(demo, inv, 1e-8).expected_sfs(15_000.0, 0, "I", 4)
    assert np.all(np.abs(got - mean) < 4 * se)
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/expect/test_msprime_reference.py -v -m slow`
Expected: 3 passed. A failure here is a model disagreement: stop, check the
migration-index convention (Global Constraints) and deme sizes before
touching tolerances.

- [ ] **Step 3: Commit**

```bash
git add tests/expect/test_msprime_reference.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test(expect): msprime single-site reference for origin="barrier"

(population, state) cells as haploid demes of size 2*N*f with conditional
migration; flank switching, two-population migration, size change.
EOF
)"
```

---

### Task 9: Performance targets and knot handling

**Files:**
- Test: `tests/expect/test_perf.py`
- Modify: `CLAUDE.md` ("Tests" section)

**Interfaces:**
- Consumes: `ExpectModel(force_knots_max=...)`, `profile`, `f1`.

- [ ] **Step 1: Write the tests:**

```python
import time

import numpy as np
import pytest

from msinv.expect import ExpectModel
from msinv.hull import Demography, InversionSpec

BL, BR = 100_000.0, 200_000.0


def _model(**kw):
    inv = InversionSpec(bp_left=BL, bp_right=BR, p_inv=0.3, t_inv=50_000.0,
                        mean_tract_length=0.0)
    return ExpectModel(Demography([10_000.0]), inv, 1e-8, **kw)


def test_pairwise_profile_100_positions_under_1s():
    m = _model()
    ys = np.linspace(0.0, BL - 1.0, 100)
    m.profile(ys[:2], (0, "I"), (0, "I"))  # warm-up
    t0 = time.perf_counter()
    out = m.profile(ys, (0, "I"), (0, "I"))
    assert time.perf_counter() - t0 < 1.0
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_f1_n20_one_population_under_5s():
    m = _model()
    t0 = time.perf_counter()
    val = m.f1(BL - 5_000.0, 0, "I", n=20)
    assert time.perf_counter() - t0 < 5.0
    assert 0.0 < val < 1.0


@pytest.mark.slow
def test_unforced_knots_agree_with_forced():
    times = np.linspace(0.0, 40_000.0, 2001)
    freqs = 0.3 + 0.1 * np.sin(times / 3000.0)
    traj = {"type": "precomputed", "times": times.tolist(),
            "freqs": [freqs.tolist()], "n_e": [10_000.0], "t_inv": [40_000.0]}
    inv = InversionSpec(bp_left=BL, bp_right=BR, trajectory=traj,
                        mean_tract_length=0.0)
    demo = Demography([10_000.0])
    loose = ExpectModel(demo, inv, 1e-8, force_knots_max=1000)
    tight = ExpectModel(demo, inv, 1e-8, force_knots_max=10_000)
    a = loose.pairwise_times(BL - 1_000.0, (0, "I"), (0, "I")).mean
    b = tight.pairwise_times(BL - 1_000.0, (0, "I"), (0, "I")).mean
    assert a == pytest.approx(b, rel=1e-5)
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/expect/test_perf.py -v -m "slow or not slow"`
Expected: 3 passed. If a timing target fails, profile
(`.venv/bin/python -m cProfile -s cumtime`) and report the numbers to the
user; do not raise the limits without agreement.

- [ ] **Step 3: Document the suite** — in `CLAUDE.md` under "## Tests", add:

```
- Tier 0 (`msinv.expect`): `.venv/bin/python -m pytest tests/expect/ -m "not slow"`; full incl. msprime reference + perf: `-m "slow or not slow"`
```

- [ ] **Step 4: Full suite + lint**

```bash
.venv/bin/python -m pytest tests/expect/ -m "slow or not slow" -v
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py -q
.venv/bin/python -m ruff format msinv/expect tests/expect
.venv/bin/python -m ruff check msinv/expect tests/expect msinv/hull/_rust_bridge.py msinv/hull/inversion.py
```

Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/expect/test_perf.py CLAUDE.md msinv/expect tests/expect
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
test(expect): tier 0 performance targets and unforced-knot accuracy
EOF
)"
```

---

## Out of scope for this plan (later plans)

- Tier 2 exact mode in Rust (plan 2), including the flux-initiation decision
  (spec §3.4 rule 4).
- Tier 0 vs tier 1/2 comparisons, msprime step-width convergence for
  continuous trajectories, SLiM benchmark (plan 3, release validation).
