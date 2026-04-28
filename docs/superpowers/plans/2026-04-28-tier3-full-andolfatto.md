# Tier 3-full (R) Andolfatto + coalescent event-count anchors — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two closed-form theoretical anchors for the b2-flux machinery — Andolfatto sample-conversion (C) and coalescent event-count (D) — both consuming the existing `event_log` hook with one new `FluxRecord` field.

**Architecture:** Add `node_id_at_position: i32` to `FluxRecord` (Rust); capture `seg.node_id` of the segment covering `x_event` in `apply_gene_flux`; expose via PyO3 dict; add Python helper `samples_converted_at(flux_records, ts, position)` that unions tskit `tree.samples(node_id)` across records; add two new tests in `tests/hull/test_phase3b_b2_flux.py`.

**Tech Stack:** Rust (`msinv-core`, `msinv-py`), PyO3, Python 3.12 + pytest, tskit, numpy.

**Spec:** `docs/superpowers/specs/2026-04-28-tier3-full-andolfatto-design.md`

---

## File map

**Modify (Rust):**
- `rust/msinv-core/src/event_log.rs` — add field to `FluxRecord` struct, update 2 unit-test literals.
- `rust/msinv-core/src/simulator.rs:1986-1996` (fast path) and `:2027-2037` (split path) — 2 call sites of `log.push_flux(...)`; add segment-walk to find `node_id` covering `x_event`.
- `rust/msinv-py/src/lib.rs:507-515` — add one `dict.set_item("node_id_at_position", ...)` line.

**Modify (Python):**
- `msinv/hull/_event_log.py` — add `samples_converted_at` helper.
- `msinv/hull/__init__.py` — re-export `samples_converted_at`.
- `tests/hull/_event_log_helpers_test.py` — add 3 unit tests.
- `tests/hull/test_phase3b_b2_flux.py` — add `_run_tier3_sim` helper + 2 new tests (C and D).

**Branch:** `feat/tier3-full-andolfatto`. Created off `main` (HEAD `9440a94`).

---

## Pre-flight

- [ ] **Step 0.1: Create feature branch and confirm clean state**

```bash
git checkout -b feat/tier3-full-andolfatto
git status         # expect: "nothing to commit, working tree clean"
git log -1 --oneline   # expect: 9440a94 spec: Tier 3-full Andolfatto ...
```

---

## Task 1: Add `node_id_at_position` field to `FluxRecord`

**Files:**
- Modify: `rust/msinv-core/src/event_log.rs:22-30` (struct), `:79-92` (`push_and_retrieve_flux` test), `:94-109` (`into_records_preserves_order` test)

- [ ] **Step 1.1: Update the `FluxRecord` struct definition**

Edit `rust/msinv-core/src/event_log.rs:22-30`. Replace:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluxRecord {
    pub t: f64,
    pub lineage_uid: LinUid,
    pub position: f64,
    pub tract_left: f64,
    pub tract_right: f64,
    pub inv_id: u16,
}
```

with:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluxRecord {
    pub t: f64,
    pub lineage_uid: LinUid,
    pub position: f64,
    pub tract_left: f64,
    pub tract_right: f64,
    pub inv_id: u16,
    /// tskit node_id of the segment covering `position` at the moment
    /// the flux event fired. Used by Tier 3-full sample-conversion
    /// validation to identify which present-day samples descend from
    /// this lineage at this position. -1 sentinel only if the segment
    /// walk failed (should never happen on a successful flux fire).
    pub node_id_at_position: i32,
}
```

- [ ] **Step 1.2: Update the `push_and_retrieve_flux` Rust test**

Edit `rust/msinv-core/src/event_log.rs:80-92`. Replace:

```rust
    #[test]
    fn push_and_retrieve_flux() {
        let mut log = EventLog::new();
        let r = FluxRecord {
            t: 250.0, lineage_uid: 42, position: 5000.0,
            tract_left: 4850.0, tract_right: 5150.0, inv_id: 0,
        };
        log.push_flux(r);
        assert_eq!(log.len(), 1);
        match log.records()[0] {
            EventRecord::Flux(got) => assert_eq!(got, r),
            _ => panic!("expected Flux variant"),
        }
    }
```

with:

```rust
    #[test]
    fn push_and_retrieve_flux() {
        let mut log = EventLog::new();
        let r = FluxRecord {
            t: 250.0, lineage_uid: 42, position: 5000.0,
            tract_left: 4850.0, tract_right: 5150.0, inv_id: 0,
            node_id_at_position: 17,
        };
        log.push_flux(r);
        assert_eq!(log.len(), 1);
        match log.records()[0] {
            EventRecord::Flux(got) => assert_eq!(got, r),
            _ => panic!("expected Flux variant"),
        }
    }
```

- [ ] **Step 1.3: Update the `into_records_preserves_order` Rust test**

Edit `rust/msinv-core/src/event_log.rs:101-104`. Replace:

```rust
        log.push_flux(FluxRecord {
            t: 20.0, lineage_uid: 1, position: 100.0,
            tract_left: 90.0, tract_right: 110.0, inv_id: 0,
        });
```

with:

```rust
        log.push_flux(FluxRecord {
            t: 20.0, lineage_uid: 1, position: 100.0,
            tract_left: 90.0, tract_right: 110.0, inv_id: 0,
            node_id_at_position: 5,
        });
```

- [ ] **Step 1.4: Run Rust tests; expect compile error in `simulator.rs`**

```bash
cd rust && cargo test --release -p msinv-core --lib 2>&1 | tail -20
```

Expected: compile error citing `simulator.rs:1988` and `:2029` — the existing `FluxRecord { ... }` literals lack `node_id_at_position`. This is fine; Task 2 fixes them.

---

## Task 2: Instrument `apply_gene_flux` to capture `node_id_at_position`

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:1953-2038` (the `apply_gene_flux` function)

- [ ] **Step 2.1: Add a helper to find segment node_id covering a position**

Add this helper just above `apply_gene_flux` (above `rust/msinv-core/src/simulator.rs:1941`):

```rust
/// Find the tskit node_id of the segment that covers `x` in the
/// lineage's segment chain. Returns -1 if no segment covers `x`
/// (caller's invariant violation; flux event should not fire on a
/// position with no covering segment).
fn segment_node_id_at(
    head: SegIdx, x: f64, arena: &SegmentArena,
) -> i32 {
    let mut cur = head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        if seg.left <= x && x < seg.right {
            return seg.node_id;
        }
        if seg.left >= x { break; }
        cur = seg.next;
    }
    -1
}
```

- [ ] **Step 2.2: Capture `node_id_at_position` at start of `apply_gene_flux`**

Edit `rust/msinv-core/src/simulator.rs:1969-1971`. Replace:

```rust
    // Capture uid BEFORE any active.push() that might reallocate.
    let lineage_uid = active[lin_idx].uid;
```

with:

```rust
    // Capture uid + node_id_at_position BEFORE any active.push() that
    // might reallocate or any split_at() that mutates the chain.
    let lineage_uid = active[lin_idx].uid;
    let node_id_at_position =
        segment_node_id_at(active[lin_idx].head, x_event, arena);
    debug_assert!(node_id_at_position >= 0,
        "flux event at x_event={} has no covering segment in lineage uid={}",
        x_event, lineage_uid);
```

- [ ] **Step 2.3: Pass `node_id_at_position` into the fast-path `push_flux`**

Edit `rust/msinv-core/src/simulator.rs:1987-1996`. Replace:

```rust
        if let Some(log) = log {
            log.push_flux(event_log::FluxRecord {
                t,
                lineage_uid,
                position: x_event,
                tract_left,
                tract_right,
                inv_id: inv.inv_id,
            });
        }
        return;
```

with:

```rust
        if let Some(log) = log {
            log.push_flux(event_log::FluxRecord {
                t,
                lineage_uid,
                position: x_event,
                tract_left,
                tract_right,
                inv_id: inv.inv_id,
                node_id_at_position,
            });
        }
        return;
```

- [ ] **Step 2.4: Pass `node_id_at_position` into the split-path `push_flux`**

Edit `rust/msinv-core/src/simulator.rs:2028-2037`. Replace:

```rust
    if let Some(log) = log {
        log.push_flux(event_log::FluxRecord {
            t,
            lineage_uid,
            position: x_event,
            tract_left,
            tract_right,
            inv_id: inv.inv_id,
        });
    }
```

with:

```rust
    if let Some(log) = log {
        log.push_flux(event_log::FluxRecord {
            t,
            lineage_uid,
            position: x_event,
            tract_left,
            tract_right,
            inv_id: inv.inv_id,
            node_id_at_position,
        });
    }
```

- [ ] **Step 2.5: Run full Rust test suite**

```bash
cd rust && cargo test --release 2>&1 | tail -20
```

Expected: PASS — `132 tests passing` (107 lib + 25 integration). The existing `record_events_logs_flux_events_when_gamma_positive` test will still pass; the new field is populated automatically.

- [ ] **Step 2.6: Commit Rust changes**

```bash
cd /home/ssmall/inversion_sims/files
git add rust/msinv-core/src/event_log.rs rust/msinv-core/src/simulator.rs
git commit -m "$(cat <<'EOF'
event-log: add node_id_at_position to FluxRecord

Captures the tskit node_id of the segment covering x_event at the
moment a flux event fires. Required by Tier 3-full sample-conversion
validation to map FluxRecord -> set of present-day samples descended
from this lineage at this position.

apply_gene_flux walks the segment chain via segment_node_id_at()
before either split_at() call, so the captured node_id reflects the
pre-mutation state of the lineage.

EOF
)"
```

---

## Task 3: Audit FluxRecord literals across the workspace

**Files:**
- Read-only check across `rust/msinv-core/src/`, `rust/msinv-core/tests/`, `rust/msinv-core/examples/`, `rust/msinv-py/src/`

- [ ] **Step 3.1: Run rust-struct-field-auditor subagent**

Dispatch the `rust-struct-field-auditor` subagent with the prompt:
> Audit `FluxRecord`. The struct lives in `rust/msinv-core/src/event_log.rs` and gained a new field `node_id_at_position: i32`. Verify every `FluxRecord { ... }` literal across `rust/msinv-core/src/`, `rust/msinv-core/tests/`, `rust/msinv-core/examples/`, and `rust/msinv-py/src/` includes the new field. Report any literal that does not. Read-only.

If the auditor reports unfixed literals, fix each one to set `node_id_at_position: <appropriate value, usually 0 or -1 for tests>`.

- [ ] **Step 3.2: Confirm full Rust suite passes including tests/, examples/, benches/**

```bash
cd rust && cargo test --release 2>&1 | tail -20
```

Expected: all 132 tests pass.

---

## Task 4: Expose `node_id_at_position` in PyO3 bridge

**Files:**
- Modify: `rust/msinv-py/src/lib.rs:507-515`

- [ ] **Step 4.1: Add `node_id_at_position` to the flux dict**

Edit `rust/msinv-py/src/lib.rs:507-515`. Replace:

```rust
            EventRecord::Flux(f) => {
                dict.set_item("kind", "flux")?;
                dict.set_item("t", f.t)?;
                dict.set_item("lineage_uid", f.lineage_uid)?;
                dict.set_item("position", f.position)?;
                dict.set_item("tract_left", f.tract_left)?;
                dict.set_item("tract_right", f.tract_right)?;
                dict.set_item("inv_id", f.inv_id)?;
            }
```

with:

```rust
            EventRecord::Flux(f) => {
                dict.set_item("kind", "flux")?;
                dict.set_item("t", f.t)?;
                dict.set_item("lineage_uid", f.lineage_uid)?;
                dict.set_item("position", f.position)?;
                dict.set_item("tract_left", f.tract_left)?;
                dict.set_item("tract_right", f.tract_right)?;
                dict.set_item("inv_id", f.inv_id)?;
                dict.set_item("node_id_at_position", f.node_id_at_position)?;
            }
```

- [ ] **Step 4.2: Rebuild and install the Rust extension**

Per `CLAUDE.md` build recipe (use `/bin/cp` to bypass shell alias):

```bash
cd /home/ssmall/inversion_sims/files/rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
```

Expected: build succeeds, file copied without prompt.

- [ ] **Step 4.3: Smoke-test the new field from Python**

Run:

```bash
cd /home/ssmall/inversion_sims/files && .venv/bin/python -c "
from msinv.hull import HullSimulator, InversionSpec
from msinv.hull.demography import Demography
inv = InversionSpec(
    bp_left=2000.0, bp_right=8000.0, p_inv=0.5, t_inv=4000.0,
    gene_conversion_rate=5e-3, mean_tract_length=300.0,
    tract_distribution='geometric',
)
sim = HullSimulator(
    sample_config={('S', 0): 4, ('I', 0): 4},
    demography=Demography(pop_sizes=[1000]),
    sequence_length=10_000, recombination_rate=1e-8,
    inversions=[inv], seed=42, record_events=True,
)
sim.simulate()
flux = [r for r in sim.event_log if r['kind'] == 'flux']
assert len(flux) > 0, 'expected at least one flux event'
print(f'n_flux={len(flux)}, first record keys={sorted(flux[0].keys())}')
assert 'node_id_at_position' in flux[0], 'missing node_id_at_position'
assert flux[0]['node_id_at_position'] >= 0, 'node_id_at_position should be a real tskit node id'
print(f'first flux record: {flux[0]}')
print('OK')
"
```

Expected: prints `n_flux=...`, lists keys including `node_id_at_position`, prints the first record dict, ends with `OK`.

- [ ] **Step 4.4: Commit PyO3 bridge change**

```bash
git add rust/msinv-py/src/lib.rs
git commit -m "$(cat <<'EOF'
event-log: expose node_id_at_position in PyO3 flux dict

Surface the new FluxRecord field to Python so samples_converted_at()
(next commit) can call ts.at(position).samples(node_id).

EOF
)"
```

---

## Task 5: Implement `samples_converted_at` helper (TDD)

**Files:**
- Create test: `tests/hull/_event_log_helpers_test.py` (existing file, append)
- Modify helper: `msinv/hull/_event_log.py` (existing file, append)
- Modify re-export: `msinv/hull/__init__.py`

- [ ] **Step 5.1: Write the first failing test (empty log)**

Append to `tests/hull/_event_log_helpers_test.py`:

```python
def test_samples_converted_at_empty_log_returns_zero():
    """No flux records → fraction == 0.0 regardless of ts."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=4, sequence_length=100,
                              recombination_rate=0, random_seed=1)
    assert samples_converted_at([], ts, 50.0) == 0.0
```

- [ ] **Step 5.2: Run test — expect failure (function not defined)**

```bash
cd /home/ssmall/inversion_sims/files
.venv/bin/python -m pytest tests/hull/_event_log_helpers_test.py::test_samples_converted_at_empty_log_returns_zero -v 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name 'samples_converted_at'`.

- [ ] **Step 5.3: Implement `samples_converted_at` in `_event_log.py`**

Append to `msinv/hull/_event_log.py`:

```python
def samples_converted_at(flux_records, ts, position):
    """Fraction of samples whose ancestry at `position` was hit by ≥1
    flux event.

    For each flux record, takes descendants of `node_id_at_position`
    in the marginal tree at `position` and unions them into a
    converted set.

    Parameters
    ----------
    flux_records : iterable of dicts
        Filtered flux records (from `filter_flux`). Each record must
        contain key "node_id_at_position".
    ts : tskit.TreeSequence
    position : float
        Genomic position; should lie inside the inversion under test.

    Returns
    -------
    fraction : float in [0.0, 1.0]
        len(converted) / ts.num_samples; 0.0 if num_samples == 0.
    """
    if ts.num_samples == 0:
        return 0.0
    tree = ts.at(position)
    converted: set[int] = set()
    for rec in flux_records:
        u = int(rec["node_id_at_position"])
        if u < 0:
            continue
        if u >= ts.num_nodes:
            continue
        for s in tree.samples(u):
            converted.add(int(s))
    return len(converted) / ts.num_samples
```

- [ ] **Step 5.4: Re-export from `msinv/hull/__init__.py`**

Find the line in `msinv/hull/__init__.py` that re-exports the existing helpers
(`filter_cmig`, `filter_flux`, `tract_lengths`, `survival_curve`, `coverage_count`)
and add `samples_converted_at` to the same import + `__all__` list. Concretely:

```bash
grep -n "filter_cmig\|coverage_count" msinv/hull/__init__.py
```

Then edit the file to include `samples_converted_at` alongside the others.

- [ ] **Step 5.5: Run the empty-log test — expect pass**

```bash
.venv/bin/python -m pytest tests/hull/_event_log_helpers_test.py::test_samples_converted_at_empty_log_returns_zero -v 2>&1 | tail -10
```

Expected: PASS.

---

## Task 6: Add the remaining 2 helper unit tests

**Files:**
- Modify: `tests/hull/_event_log_helpers_test.py` (append 2 tests)

- [ ] **Step 6.1: Write the root-node test**

Append to `tests/hull/_event_log_helpers_test.py`:

```python
def test_samples_converted_at_root_node_returns_one():
    """A single record pointing at the root → all samples converted."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=4, sequence_length=100,
                              recombination_rate=0, random_seed=2)
    tree = ts.at(50.0)
    root = tree.root
    rec = {"kind": "flux", "node_id_at_position": int(root)}
    assert samples_converted_at([rec], ts, 50.0) == 1.0
```

- [ ] **Step 6.2: Write the specific-descendants test**

Append:

```python
def test_samples_converted_at_specific_descendants_match():
    """A record pointing at a non-root internal node → exactly its
    descendant-leaf set."""
    import msprime
    from msinv.hull._event_log import samples_converted_at
    ts = msprime.sim_ancestry(samples=8, sequence_length=100,
                              recombination_rate=0, random_seed=3)
    tree = ts.at(50.0)
    # Pick an internal node that is not the root and has at least 2
    # leaves below it.
    chosen = None
    for u in tree.nodes():
        if tree.is_internal(u) and u != tree.root:
            leaves = list(tree.samples(u))
            if len(leaves) >= 2:
                chosen = u
                break
    assert chosen is not None, "expected an internal non-root node"
    rec = {"kind": "flux", "node_id_at_position": int(chosen)}
    expected_frac = len(list(tree.samples(chosen))) / ts.num_samples
    assert samples_converted_at([rec], ts, 50.0) == expected_frac
```

- [ ] **Step 6.3: Run all 3 helper tests — expect pass**

```bash
.venv/bin/python -m pytest tests/hull/_event_log_helpers_test.py -v 2>&1 | tail -20
```

Expected: 3 new tests pass; all previously-existing tests in this file also pass (they don't import `samples_converted_at`).

- [ ] **Step 6.4: Commit helper + tests**

```bash
git add msinv/hull/_event_log.py msinv/hull/__init__.py tests/hull/_event_log_helpers_test.py
git commit -m "$(cat <<'EOF'
event-log: add samples_converted_at helper

Wraps tskit tree.samples(u) over the marginal tree at a given
position, unioning across all flux records that hit that position.
Used by Tier 3-full Andolfatto sample-conversion test (next commit).
3 unit tests cover empty log, root-node (all samples), and a
specific internal-node descendant set.

EOF
)"
```

---

## Task 7: Add `_run_tier3_sim` helper + Andolfatto test (C)

**Files:**
- Modify: `tests/hull/test_phase3b_b2_flux.py` (append)

- [ ] **Step 7.1: Add the shared sim helper at module scope**

Append to `tests/hull/test_phase3b_b2_flux.py`:

```python
# ---------------------------------------------------------------------------
# Tier 3-full (R): closed-form Andolfatto + coalescent event-count anchors
# ---------------------------------------------------------------------------

def _run_tier3_sim(t_inv, gamma, seed,
                   bp_left=2000.0, bp_right=8000.0,
                   lam=300.0, p_inv=0.5,
                   Ne=1000, n_S=10, n_I=10,
                   sequence_length=10_000,
                   recombination_rate=1e-8):
    """Run a single Tier 3-full sim and return (ts, event_log).

    Centralizes the parameter set so test C and test D differ only in
    `t_inv` and `gamma`.
    """
    inv = InversionSpec(
        bp_left=bp_left, bp_right=bp_right,
        p_inv=p_inv, t_inv=t_inv,
        gene_conversion_rate=gamma,
        mean_tract_length=lam,
        tract_distribution='geometric',
    )
    sim = HullSimulator(
        sample_config={('S', 0): n_S, ('I', 0): n_I},
        demography=Demography(pop_sizes=[Ne]),
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        inversions=[inv],
        seed=seed,
        record_events=True,
    )
    ts = sim.simulate()
    return ts, sim.event_log
```

- [ ] **Step 7.2: Add the C test (Andolfatto sample-conversion match)**

Append:

```python
def test_andolfatto_sample_fraction_matches_closed_form():
    """Tier 3-full (R), interpretation (b): empirical f̂(t) matches
    1 - exp(-γ·p_other·λ²·t/L) over a 4-point t_inv ladder, within
    ±0.10 abs OR ±20% rel (per-point, n_seeds=30).

    Closed form derivation: see
    docs/superpowers/specs/2026-04-28-tier3-full-andolfatto-design.md.
    The simulator parameterizes per-lineage flux rate as γ·p_other·λ;
    per-event prob of covering interior x is λ/L; product is the
    per-lineage rate of x-flipping events.
    """
    import math
    from msinv.hull._event_log import filter_flux, samples_converted_at

    gamma = 1.5e-5     # γ_C: rate puts f_pred in [0.10, 0.95] over ladder
    lam = 300.0
    bp_left, bp_right = 2000.0, 8000.0
    L = bp_right - bp_left          # 6000.0
    Ne, p_inv = 1000, 0.5
    p_other = 1.0 - p_inv           # 0.5 (symmetric at p_inv=0.5)
    inv_center = 0.5 * (bp_left + bp_right)   # 5000.0
    n_seeds = 30
    t_inv_ladder = [1000.0, 4000.0, 10_000.0, 25_000.0]

    for t_inv in t_inv_ladder:
        f_emp = []
        for seed in range(n_seeds):
            ts, log = _run_tier3_sim(t_inv=t_inv, gamma=gamma, seed=seed,
                                     bp_left=bp_left, bp_right=bp_right,
                                     lam=lam, p_inv=p_inv, Ne=Ne)
            flux = filter_flux(log, inv_id=0)
            f_emp.append(samples_converted_at(flux, ts, inv_center))
        f_hat = float(np.mean(f_emp))
        f_pred = 1.0 - math.exp(-gamma * p_other * (lam ** 2) * t_inv / L)
        tol = max(0.10, 0.20 * f_pred)
        assert abs(f_hat - f_pred) < tol, (
            f"t_inv={t_inv}: f̂={f_hat:.3f} vs predicted {f_pred:.3f} "
            f"(tol={tol:.3f}, n_seeds={n_seeds})")
```

- [ ] **Step 7.3: Run test C — expect pass**

```bash
.venv/bin/python -m pytest tests/hull/test_phase3b_b2_flux.py::test_andolfatto_sample_fraction_matches_closed_form -v 2>&1 | tail -20
```

Expected: PASS in ~3-5 minutes (120 sims; the t_inv=25_000 rung dominates).

If a single ladder point fails by < ~30% relative, recheck γ_C choice and
n_seeds. If failure is by ~2× or more, audit the closed-form factor — the
most likely culprit is a missed factor of λ or boundary correction.

- [ ] **Step 7.4: Commit test C**

```bash
git add tests/hull/test_phase3b_b2_flux.py
git commit -m "$(cat <<'EOF'
test: Tier 3-full Andolfatto sample-fraction match

Asserts empirical f̂(t) matches 1 - exp(-γ·p_other·λ²·t/L) over a
4-point t_inv ladder spanning f ≈ 0.11..0.94, n_seeds=30.

Tightens Q5b (which only checked monotonicity in t_inv) into a
closed-form anchor for the b2-flux machinery. γ chosen so the ladder
spans the informative range of the exp curve, not too saturated.

EOF
)"
```

---

## Task 8: Add the coalescent event-count anchor (test D)

**Files:**
- Modify: `tests/hull/test_phase3b_b2_flux.py` (append)

- [ ] **Step 8.1: Add test D**

Append to `tests/hull/test_phase3b_b2_flux.py`:

```python
def test_event_coverage_matches_coalescent_closed_form():
    """Tier 3-full (D): empirical mean coverage_count(x_center) matches
    γ·p_other·(λ²/L) · 4Ne·H_{n−1} at t_inv ≫ 2Ne, within ±3 SE.

    Why this test: complementary to C. C tests sample-level conversion;
    D tests event-rate scaling against the coalescent closed form for
    expected total branch length under Kingman with constant Ne.
    Different bug class pinpointed: D fails ⇒ flux-rate or coalescent-
    timescale scaling is off; C fails ⇒ sample-vs-event accounting in
    samples_converted_at is wrong.
    """
    import math
    from msinv.hull._event_log import filter_flux, coverage_count

    gamma = 5e-3       # γ_D: matches Q5b; ~500 events/seed at t_inv=25k
    lam = 300.0
    bp_left, bp_right = 2000.0, 8000.0
    L = bp_right - bp_left
    Ne, p_inv = 1000, 0.5
    p_other = 1.0 - p_inv
    n_S, n_I = 10, 10
    n = n_S + n_I                       # 20 lineages at t=0
    t_inv = 25_000.0                    # ≫ 2Ne; truncation corr. ~4e-6
    inv_center = 5000.0
    n_seeds = 30

    H = sum(1.0 / k for k in range(1, n))                  # H_{n−1} ≈ 3.548
    e_total_branch = 4.0 * Ne * H                          # generations
    expected = gamma * p_other * (lam ** 2 / L) * e_total_branch

    counts = []
    for seed in range(n_seeds):
        ts, log = _run_tier3_sim(t_inv=t_inv, gamma=gamma, seed=seed,
                                 bp_left=bp_left, bp_right=bp_right,
                                 lam=lam, p_inv=p_inv, Ne=Ne,
                                 n_S=n_S, n_I=n_I)
        flux = filter_flux(log, inv_id=0)
        counts.append(coverage_count(flux, inv_center))

    mean_emp = float(np.mean(counts))
    se_emp = float(np.std(counts, ddof=1)) / math.sqrt(n_seeds)
    assert abs(mean_emp - expected) < 3 * se_emp, (
        f"empirical {mean_emp:.1f} vs closed form {expected:.1f} "
        f"(3 SE = {3 * se_emp:.1f}, n_seeds={n_seeds})")
```

- [ ] **Step 8.2: Run test D — expect pass**

```bash
.venv/bin/python -m pytest tests/hull/test_phase3b_b2_flux.py::test_event_coverage_matches_coalescent_closed_form -v 2>&1 | tail -20
```

Expected: PASS in ~1-2 minutes (30 sims at t_inv=25_000).

If the empirical mean is consistently ~16% below the closed form: re-examine
truncation (is t_inv really ≫ TMRCA?). If consistently 2× off: re-examine
the ploidy/coalescent-units factor.

- [ ] **Step 8.3: Commit test D**

```bash
git add tests/hull/test_phase3b_b2_flux.py
git commit -m "$(cat <<'EOF'
test: Tier 3-full coalescent event-count anchor

Asserts mean coverage_count(x_center) matches γ·p_other·(λ²/L)·4Ne·H_{n−1}
at t_inv ≫ 2Ne (full coalescence; truncation corr. ~4e-6), within
±3 empirical SE.

Complements the Andolfatto sample-fraction test: D fails ⇒ flux-rate
or coalescent-timescale scaling is off; C fails ⇒ samples_converted_at
mapping is wrong.

EOF
)"
```

---

## Task 9: Bench-off-path verification

- [ ] **Step 9.1: Run the bench-off-path skill**

Invoke the `bench-off-path` skill (per CLAUDE.md, this measures wall-clock
cost of `record_events=False` vs `record_events=True`).

Expected: median + p95 of `record_events=False` is unchanged from baseline
(within MC noise of prior measurements). The new `segment_node_id_at` walk
runs only inside `if let Some(log) = log { ... }` paths and at the start
of `apply_gene_flux` only when log is `Some` — confirm by inspection.

If `record_events=False` regressed: gate `node_id_at_position` capture
behind `log.is_some()` check at the top of the function.

- [ ] **Step 9.2 (only if Step 9.1 reveals a regression): Gate the segment walk**

If bench shows the off-path slowed measurably, tighten the gate:

Edit `rust/msinv-core/src/simulator.rs:1969-1976` so the segment walk is
skipped when `log.is_none()`. Replace:

```rust
    let lineage_uid = active[lin_idx].uid;
    let node_id_at_position =
        segment_node_id_at(active[lin_idx].head, x_event, arena);
    debug_assert!(node_id_at_position >= 0,
        "flux event at x_event={} has no covering segment in lineage uid={}",
        x_event, lineage_uid);
```

with:

```rust
    let lineage_uid = active[lin_idx].uid;
    let node_id_at_position = if log.is_some() {
        let nid = segment_node_id_at(active[lin_idx].head, x_event, arena);
        debug_assert!(nid >= 0,
            "flux event at x_event={} has no covering segment in lineage uid={}",
            x_event, lineage_uid);
        nid
    } else {
        -1
    };
```

Re-run bench. Commit.

---

## Task 10: Final sweep + push

- [ ] **Step 10.1: Run full Python test suite (excluding known failures)**

```bash
.venv/bin/python -m pytest tests/hull/ \
    --ignore=tests/hull/test_stress_corners.py \
    --deselect tests/hull/test_phase6_sweep.py 2>&1 | tail -20
```

Expected: ≥166 passed, 0 failed (was 161 + 5 new = 166; +/- the 17 sweep tests
that error on `target_class='P'` per CLAUDE.md). The "deselect" pattern may
need adjusting for the specific subset listed in CLAUDE.md.

- [ ] **Step 10.2: Run full Rust suite**

```bash
cd rust && cargo test --release 2>&1 | tail -10
```

Expected: 132 tests pass.

- [ ] **Step 10.3: Push the feature branch**

```bash
cd /home/ssmall/inversion_sims/files
git push -u origin feat/tier3-full-andolfatto
```

- [ ] **Step 10.4: Update the resume memo**

The session-resume memory at
`/home/ssmall/.claude/projects/-home-ssmall-inversion-sims-files/memory/project_b2_flux_session_resume.md`
should be updated to reflect that Tier 3-full has shipped. Replace the
"NEXT SESSION" section with a "Shipped this session" summary and propose the
next work item from the deferred list (sweep model rewrite, cross-engine
parity, etc.).

---

## Self-review (already applied inline)

- **Spec coverage:** Every section of the spec maps to a task above:
  - Spec §"Components 1": Task 1 (struct), Task 3 (audit).
  - Spec §"Components 2" (apply_gene_flux): Task 2.
  - Spec §"Components 3" (helper): Task 5 + 6.
  - Spec §"Test C": Task 7.
  - Spec §"Test D": Task 8.
  - Spec §"Helper unit tests": Tasks 5.1, 6.1, 6.2.
  - Spec §"Sim cost / `_run_tier3_sim`": Task 7.1.
  - Spec §"Risks - bench-off-path": Task 9.
  - Spec §"PyO3 bridge": Task 4.
- **Placeholder scan:** No "TBD" / "TODO" / "implement later" remain. Every
  step has actual code or actual commands.
- **Type consistency:** `samples_converted_at(flux_records, ts, position)`
  signature matches across the helper definition, the 3 unit tests, and the
  test-C call site. `node_id_at_position` field name matches across Rust
  struct, Rust literal updates, PyO3 dict key, and Python helper consumer.
- **Pre-existing test exclusions:** Per CLAUDE.md, the stress-corners hang
  and 17 sweep `target_class='P'` failures are pre-existing — Step 10.1
  excludes them rather than chasing.

---
