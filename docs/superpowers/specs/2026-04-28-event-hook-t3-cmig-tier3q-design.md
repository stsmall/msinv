# Event-log hook for T3 cmig + Tier 3-cheap (Q) — design

**Status:** Approved (brainstorm 2026-04-28)
**Predecessor work:** `docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md` (b2-flux upgrade, shipped at `74101ad`)
**Author:** Claude (Opus 4.7, 1M ctx) with stsmall

## Goal

Add an opt-in, Rust-side simulator-state event log that captures
class-migration (`cmig`) and gene-flux (`flux`) events as they fire,
then expose the log to Python tests via a wrapper attribute. Use the
log to land three deferred validation tests:

1. **T3 cmig binomial-count check** — `n_moved` per scheduled cmig
   event matches `Binomial(n_eligible, proportion)` within ±2σ.
2. **Tier 3-cheap (Q5a) LD-decay shape** — for the b2-flux model,
   the empirical tract-length survival function `S(d)` discriminates
   `tract_distribution='geometric'` (smooth `exp(−d/λ)` decay) from
   `'fixed'` (sharp shoulder at λ).
3. **Tier 3-cheap (Q5b) Andolfatto monotonicity** — fraction-converted,
   estimated from event-coverage counts, increases monotonically with
   `t_inv` at fixed `(γ, λ)`; mean fraction-converted at matched λ
   agrees within 5% between `'fixed'` and `'geometric'`.

## Non-goals

- **Sweep / partial-sweep model rewrite.** Tracked as TODO #3 in
  `project_msinv_todo.md`. 3+ sessions; out of scope.
- **Tier 3-full (R) Andolfatto closed-form anchor.** Adds ~1 day on top
  of Q if the hook generalizes; deferred to a separate session.
- **Cross-engine bit-equivalence harness** (Python `HullSimulator(use_rust=False)`
  vs Rust `use_rust=True`). 1–2 sessions; deferred.
- **ABC pilot.** User-led; do not initiate.
- **Streaming / observer callbacks.** Considered and rejected — the
  per-event PyO3 GIL-acquire overhead at biological γ·λ would be
  prohibitive, and post-hoc aggregation is what the tests actually want.
- **Coalescence / recombination logging.** Out of scope for this design;
  current tests need only cmig + flux.
- **Embedding the log into `ts.metadata`.** Out of scope; the log is a
  parallel structure, not part of the tree sequence.

## Design decisions (locked in brainstorm)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | T3 cmig **+** Tier 3-cheap (Q) | Resume-note-recommended bundle; one hook serves both |
| Hook mechanism | Rust-side `Vec<EventRecord>`, **opt-in** | Cheapest; observer pattern rejected for FFI cost |
| Default state | `record_events=False` | Production sims pay zero overhead |
| Record schema | Rich (cmig: 7 fields; flux: 8 fields) | +luxury fields ≈ +5 MB at 10⁵ events; acceptable |
| API exposure | Wrapper attribute (`sim.event_log`) | Non-breaking; existing tests untouched |
| T3 stat test | Pointwise ±2σ vs Binomial(n_eligible, p) | Simplest readable check; matches resume-note framing |
| Q5a LD-decay | Hook-based tract-break survival `S(d)` | Direct, low-noise; haplotype-based LD considered and rejected as noisier |
| Q5b Andolfatto | Hook-based event-coverage count | Same hook serves both Q sub-tests; tskit edge-walk considered and rejected as fragile |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ rust/msinv-core/src/event_log.rs        (NEW, ~120 LOC)    │
│   pub enum EventRecord { Cmig(CmigRecord), Flux(FluxRecord) }
│   pub struct EventLog { records: Vec<EventRecord> }        │
│     methods: new, push_cmig, push_flux, len, records,      │
│              into_records                                  │
└────────────────────────────────────────────────────────────┘
                            ▲
                            │ Option<&mut EventLog> threaded through
                            │
┌────────────────────────────────────────────────────────────┐
│ rust/msinv-core/src/simulator.rs                           │
│   HullSimulator { …, record_events: bool (default false) } │
│   simulate_with_cache: builds Option<EventLog>             │
│   apply_class_mig: + log: Option<&mut EventLog>            │
│   apply_gene_flux: + log: Option<&mut EventLog>            │
│   pub struct SimResult { tables, event_log: Option<EventLog> }
└────────────────────────────────────────────────────────────┘
                            ▲
                            │ PyO3 conversion
                            │
┌────────────────────────────────────────────────────────────┐
│ rust/msinv-py/src/lib.rs                                   │
│   PyHullSimulator: + record_events kwarg (default False)   │
│   simulate(): returns (ts_table_capsule, py_event_log_or_None)
│     each EventRecord → PyDict with "kind": "cmig"|"flux"   │
└────────────────────────────────────────────────────────────┘
                            ▲
                            │ Python wrapper
                            │
┌────────────────────────────────────────────────────────────┐
│ msinv/hull/_rust_bridge.py                                 │
│   HullSimulator(__init__): + record_events=False           │
│   simulate(): self.event_log = py_event_log_or_None        │
│                                                            │
│ msinv/hull/_event_log.py                (NEW, ~60 LOC)     │
│   filter_cmig(log) -> list[dict]                           │
│   filter_flux(log, inv_id=None) -> list[dict]              │
│   tract_lengths(flux_records) -> np.ndarray                │
│   survival_curve(values, ds) -> np.ndarray                 │
│   coverage_count(flux_records, position) -> int            │
└────────────────────────────────────────────────────────────┘
```

### Component detail

#### `rust/msinv-core/src/event_log.rs` (new file)

```rust
use crate::class_tag::Karyotype;
use crate::lineage::LinUid;

#[derive(Debug, Clone, Copy)]
pub struct CmigRecord {
    pub t: f64,
    pub src: u32,
    pub dst: u32,
    pub kary: Karyotype,
    pub inv_id: u16,
    pub n_eligible: u32,
    pub n_moved: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct FluxRecord {
    pub t: f64,
    pub lineage_uid: LinUid,
    pub position: f64,        // sampled seed point
    pub tract_left: f64,
    pub tract_right: f64,
    pub inv_id: u16,
    pub donor_class: Karyotype,
    pub acceptor_class: Karyotype,
}

#[derive(Debug, Clone, Copy)]
pub enum EventRecord {
    Cmig(CmigRecord),
    Flux(FluxRecord),
}

#[derive(Debug, Default)]
pub struct EventLog {
    records: Vec<EventRecord>,
}

impl EventLog {
    pub fn new() -> Self { Self::default() }
    pub fn push_cmig(&mut self, r: CmigRecord) { self.records.push(EventRecord::Cmig(r)); }
    pub fn push_flux(&mut self, r: FluxRecord) { self.records.push(EventRecord::Flux(r)); }
    pub fn len(&self) -> usize { self.records.len() }
    pub fn records(&self) -> &[EventRecord] { &self.records }
    pub fn into_records(self) -> Vec<EventRecord> { self.records }
}
```

#### `simulator.rs` modifications (touch points)

- `pub struct HullSimulator { …, pub record_events: bool }` — initialize
  to `false` in all constructors (`new`, `panmictic`, `Default`).
- `pub struct SimResult { tables: tables::TableBuilder, pub event_log: Option<EventLog> }`.
- `simulate_with_cache` constructs `let mut event_log: Option<EventLog>
  = if self.record_events { Some(EventLog::new()) } else { None };`
  and threads `event_log.as_mut()` through the dispatch helpers.
- `apply_class_mig` (currently at `simulator.rs:2070`) gains
  `log: Option<&mut EventLog>` and `t: f64`. Implementation:
  ```rust
  fn apply_class_mig(active, arena, spec, rng, inversions,
                      log: Option<&mut EventLog>, t: f64) {
      let mut n_eligible: u32 = 0;
      let mut n_moved: u32 = 0;
      for lin in active.iter_mut() {
          if lin.population != spec.src { continue; }
          let kary = lineage_class_for_inv_id_arena(lin.head, spec.inv_id, arena);
          if kary != Some(spec.kary) { continue; }
          n_eligible += 1;
          if spec.proportion >= 1.0 - 1e-12 || rng.random::<f64>() < spec.proportion {
              lin.population = spec.dst;
              n_moved += 1;
          }
      }
      if let Some(log) = log {
          log.push_cmig(CmigRecord {
              t, src: spec.src, dst: spec.dst,
              kary: spec.kary, inv_id: spec.inv_id,
              n_eligible, n_moved,
          });
      }
  }
  ```
- `apply_gene_flux` (currently at `simulator.rs:1928`) gains
  `log: Option<&mut EventLog>`. The donor/acceptor classes and
  tract endpoints are already locally available at the call site;
  one `log.push_flux(...)` at the end of the function body.
- The two flux dispatch sites (`simulator.rs:441`, `simulator.rs:1193`)
  thread `event_log.as_mut()` through.
- The cmig dispatch site (`simulator.rs:2063`) threads `event_log.as_mut()`
  and the `t` it already has in scope.

#### `rust/msinv-py/src/lib.rs` modifications

- `PyHullSimulator::new` (constructor) gains `record_events: bool`
  with default `False`. Stores on the inner `HullSimulator`.
- `PyHullSimulator::simulate` returns `(PyAny /*tskit table capsule*/,
  Option<PyList /*event records*/>)`. Each `EventRecord` converts to
  a `PyDict` with a `"kind"` discriminator (`"cmig"` or `"flux"`)
  and the variant fields as named keys.

#### `msinv/hull/_rust_bridge.py` modifications

- `HullSimulator.__init__(..., record_events: bool = False)`. Stored.
  Forwarded to PyO3 constructor.
- `HullSimulator.simulate()`: unpack `(ts, event_log)` from the Rust
  call; set `self.event_log = event_log` (`None` if flag was off).

#### `msinv/hull/_event_log.py` (new file, ~60 LOC)

```python
"""Helpers for analyzing simulator event logs.

The event log is a list of dicts (or None), populated when
HullSimulator(record_events=True). Each dict has a 'kind' field of
'cmig' or 'flux' plus the variant's typed fields.
"""

from __future__ import annotations
import numpy as np


def _require_log(log):
    if log is None:
        raise ValueError(
            "event log not recorded; pass record_events=True to HullSimulator")
    return log


def filter_cmig(log):
    return [r for r in _require_log(log) if r["kind"] == "cmig"]


def filter_flux(log, inv_id=None):
    out = [r for r in _require_log(log) if r["kind"] == "flux"]
    if inv_id is not None:
        out = [r for r in out if r["inv_id"] == inv_id]
    return out


def tract_lengths(flux_records):
    return np.array([r["tract_right"] - r["tract_left"] for r in flux_records])


def survival_curve(values, ds):
    """S(d) = fraction of values >= d, evaluated at distances ds."""
    values = np.asarray(values)
    return np.array([np.mean(values >= d) for d in ds])


def coverage_count(flux_records, position):
    """How many flux events have tract_left <= position <= tract_right."""
    return sum(1 for r in flux_records
                 if r["tract_left"] <= position <= r["tract_right"])
```

## Data flow

**With `record_events=True`:**

1. `HullSimulator(..., record_events=True).simulate()` (Python)
2. PyO3 builds Rust `HullSimulator{record_events: true}`, calls `simulate_with_cache`.
3. `simulate_with_cache` initializes `Option<EventLog> = Some(EventLog::new())`.
4. Event loop fires; each `apply_class_mig` / `apply_gene_flux` call receives
   `log.as_mut()` and pushes one record per fire.
5. `SimResult { tables, event_log: Some(log) }` returned.
6. PyO3 converts each `EventRecord` to a `PyDict`; returns `(ts_capsule, py_list)`.
7. Python wrapper sets `self.event_log = py_list` and returns the tskit `TreeSequence`.
8. Test code reads `sim.event_log` via the `_event_log` helpers.

**With `record_events=False` (default):**

1. `Option<EventLog> = None` in `simulate_with_cache`.
2. `apply_class_mig` / `apply_gene_flux` receive `None`. They use
   `if let Some(log) = log { log.push_*(...) }` so the entire push path
   is skipped. **Pre-counting `n_eligible` for cmig must also be gated
   on `log.is_some()`** to avoid the extra pass when off.
3. `SimResult { tables, event_log: None }`.
4. PyO3 returns `(ts_capsule, None)`.
5. Python wrapper sets `self.event_log = None`.

## Cost analysis

**Off (default):** 1 `Option::is_none()` branch per cmig/flux dispatch.
Compiler should hoist; net indistinguishable from current code.

**On (test mode):** at biological γ·λ, ~10⁵ flux events × 64 bytes/record
≈ 6 MB log. PyO3 conversion at end-of-run: ~0.1 s for 10⁵ records (one-shot,
not in hot loop). Acceptable.

**No storage cap.** Runaway sims could OOM if `record_events=True`, but
that's a sim configuration error. Production never hits this (default-off).

## Error handling

| Scenario | Behavior |
|----------|----------|
| Hook on, no events fired | `sim.event_log = []`. Tests that expect events should assert non-empty. |
| Helper called on `None` log | `_require_log` raises `ValueError("event log not recorded; pass record_events=True to HullSimulator")` |
| Sim panics mid-run | Rust unwind drops the log on the stack; PyO3 raises `PyRuntimeError`. No partial-log leakage. Matches existing simulator panic semantics. |
| Multiple `simulate()` calls on one `HullSimulator` | `self.event_log` is overwritten on each call. Tests should snapshot after each `simulate()`. |

**No checksum, no schema versioning, no thread safety.** In-process dict
list of fixed shape; not persisted; simulator is single-threaded.

## Testing

### T3 cmig binomial-count

**File:** extend `tests/hull/test_phase4b_class_migration.py`.
**New test:**

```python
def test_class_mig_count_matches_binomial():
    """T3: n_moved per cmig event ~ Binomial(n_eligible, p) within ±2σ.

    For each p in {0.1, 0.3, 0.5, 0.7, 0.9}, run 30 seeds. Per seed,
    assert n_moved within np ± 2·sqrt(np(1-p)). At least 95% of seeds
    must satisfy this band per p.
    """
    proportions = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_seeds = 30
    for p in proportions:
        within_band = 0
        for seed in range(n_seeds):
            sim = HullSimulator(..., record_events=True, seed=seed)
            sim.add_class_migration(time=..., source=1, dest=0,
                                     karyotype='S', inv_id=0, proportion=p)
            sim.simulate()
            recs = filter_cmig(sim.event_log)
            assert len(recs) == 1
            r = recs[0]
            n, k = r["n_eligible"], r["n_moved"]
            mu, sd = n*p, np.sqrt(n*p*(1-p))
            if abs(k - mu) <= 2*sd:
                within_band += 1
        assert within_band >= 0.95 * n_seeds, \
            f"p={p}: only {within_band}/{n_seeds} within ±2σ band"
```

### Q5a flux tract-break survival

**File:** extend `tests/hull/test_phase3b_b2_flux.py`.
**New test:**

```python
def test_flux_tract_break_survival_geometric_vs_fixed():
    """Q5a: empirical S(d) discriminates 'geometric' (exp decay)
    from 'fixed' (sharp shoulder).

    Expected at d = λ:
      'geometric': S ≈ exp(-1) ≈ 0.37
      'fixed':     S = 1 (all tracts have length exactly λ ≥ λ)
    Expected at d = 2λ:
      'geometric': S ≈ exp(-2) ≈ 0.135
      'fixed':     S = 0
    """
    lam = 300.0
    for mode, expected_at_lam in [('geometric', 0.37), ('fixed', 1.0)]:
        sim = HullSimulator(..., record_events=True)
        # spec with tract_distribution=mode, mean_tract_length=lam
        sim.simulate()
        flux = filter_flux(sim.event_log)
        lengths = tract_lengths(flux)
        s_at_lam = np.mean(lengths >= lam)
        # tolerance ±0.05 for 'geometric' (MC), ±0.0 for 'fixed' (exact)
        if mode == 'geometric':
            assert abs(s_at_lam - 0.37) < 0.05
        else:
            assert s_at_lam == 1.0
```

### Q5b Andolfatto monotonicity + mode-equality

**File:** extend `tests/hull/test_phase3b_b2_flux.py`.
**New test:**

```python
def test_andolfatto_event_coverage_monotone_in_t_inv():
    """Q5b:
    (i) fraction-converted at center of inversion increases monotonically
        with t_inv at fixed (γ, λ).
    (ii) at matched λ, mean fraction-converted is equal between
         'fixed' and 'geometric' within 5%.
    """
    gamma, lam = 5e-7, 300.0
    t_inv_ladder = [500, 2000, 5000]
    n_seeds = 20
    inv_center = 5000  # midpoint of bp_left=2000, bp_right=8000
    for mode in ['fixed', 'geometric']:
        means_per_t = []
        for t_inv in t_inv_ladder:
            covers = []
            for seed in range(n_seeds):
                sim = HullSimulator(..., record_events=True, seed=seed)
                # spec t_inv=t_inv, gene_conversion_rate=gamma, mean_tract_length=lam
                sim.simulate()
                flux = filter_flux(sim.event_log, inv_id=0)
                covers.append(coverage_count(flux, inv_center))
            means_per_t.append(np.mean(covers))
        # (i) monotonicity
        assert means_per_t[0] < means_per_t[1] < means_per_t[2], \
            f"{mode}: not monotone in t_inv: {means_per_t}"
    # (ii) mode-equality at matched λ — recompute or store both modes
    ...
```

### Rust-side unit tests (in `event_log.rs`)

```rust
#[test] fn push_and_retrieve_cmig() { ... }
#[test] fn push_and_retrieve_flux() { ... }
```

### Rust-side integration test (in `simulator.rs`)

```rust
#[test]
fn record_events_round_trip_panmictic_single_cmig() {
    // record_events=true, one scheduled cmig event, assert log.len()==1
}
```

### Verification command

```bash
cd rust && cargo test --release --lib event_log \
  && cargo test --release --lib class_mig \
  && cargo test --release --lib gene_flux
cd .. && /bin/cp -f rust/target/release/lib_msinv_core.so \
  msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/hull/test_phase4b_class_migration.py \
  tests/hull/test_phase3b_b2_flux.py -v
```

**Caveat:** never run the `cp -f` step while a Python sim is live
(SIGBUS on mmap'd .so; see `feedback_so_replacement.md`).

### Test budget

All new tests target ≤ 30 s combined wall-clock under
`cargo test --release` + `pytest -v`. Cell config: small `Ne` (~500),
small `L` (~10 kb), short `t_inv` (~500 gen) so each rep is sub-second.

## Out of scope / deferred

- **Tier 3-full (R) Andolfatto closed-form anchor.** Would build on the
  same hook; defer to a follow-up session if monotonicity (Q5b) reveals
  the need for tighter validation.
- **Sweep / partial-sweep model rewrite** (TODO #3 in `project_msinv_todo.md`).
- **Cross-engine parity harness.**
- **Coalescence / recombination event types in the log.**

## References

- Resume note: `memory/project_b2_flux_session_resume.md`
- Predecessor spec: `docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md`
- Related TODO: `memory/project_msinv_todo.md` (item #3 sweep — separate session)
- Existing test files extended: `tests/hull/test_phase4b_class_migration.py`,
  `tests/hull/test_phase3b_b2_flux.py`
- Build/test workflow: `CLAUDE.md`
