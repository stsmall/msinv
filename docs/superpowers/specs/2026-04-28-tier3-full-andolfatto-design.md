# Tier 3-full (R): Andolfatto closed-form + coalescent event-count anchors — design

**Status:** Approved (brainstorm 2026-04-28)
**Predecessor work:** `docs/superpowers/specs/2026-04-28-event-hook-t3-cmig-tier3q-design.md` (event log + Q5a/Q5b shipped at `0875d1a`)
**Author:** Claude (Opus 4.7, 1M ctx) with stsmall

## Goal

Tighten the b2-flux validation surface by replacing Q5b's monotonicity
check with two **closed-form theoretical anchors** for the gene-flux
machinery, both consuming the existing `event_log` hook:

1. **(C) Sample-conversion anchor (the cited Andolfatto formula, in
   simulator parameterization).** Empirical fraction `f̂(t_inv)` of
   samples whose ancestry at the inversion center has been hit by ≥1
   flux event matches
   `1 − exp(−γ · p_other · λ² · t_inv / L)` over a 4-point `t_inv`
   ladder, within MC tolerance.
2. **(D) Event-count coalescent anchor.** Empirical mean
   `coverage_count(x_center)` matches the Kingman closed form
   `γ · p_other · (λ²/L) · 2 · (2Ne) · H_{n−1}` at a single
   `t_inv ≫ 2Ne` (so coalescence completes and the truncation
   correction vanishes).

**Bundling:** both tests live in the same file and share helper
machinery, but they require **different `γ` values** to land in
useful regimes (see "Closed-form derivation" below) and therefore
**do NOT share simulations**.

### Closed-form derivation (simulator units)

The simulator parameterizes flux rate as
`r_lineage = γ · p_other · w` with `w = λ/L` (`simulator.rs:355`).
The per-lineage **total** flux event rate (integrating the Peischl
phi(x) profile over a fully-covered inversion) is therefore
`γ · p_other · λ` per generation — this is the validated Tier-2
calibration in `test_flux_rate_scales_linearly_with_mean_tract_length`.

For an interior position `y` (distance ≫ λ from breakpoints), the
per-event probability that the tract `[b1, b1+T]` covers `y` equals
`E[T]/L = λ/L`. Multiplying:

  per-lineage rate of `y`-covering events = `γ · p_other · λ²/L` per gen.

This is the rate of position-`y`-flipping events on a single lineage.
Over an interval of length `t`, the lineage's ancestry at `y` has been
hit by a Poisson(`γ·p_other·λ²·t/L`) number of flux events, so

  P(at least one flip) = `1 − exp(−γ · p_other · λ² · t / L)`

which is the Andolfatto formula in simulator units. (Andolfatto's
paper expresses the same quantity as `γ_init·p_other·λ·t` with
`γ_init = γ·λ/L` the per-bp initiation rate.)

> **Note on the resume note's formula.** The resume memo wrote
> `1 − exp(−γ·p_other·λ·t/L)` (one factor of λ instead of two). Given
> the simulator's `γ` parameterization documented above, the correct
> exponent is `−γ·p_other·λ²·t/L`. This spec uses the corrected form
> throughout; the rate translates back to Andolfatto's published form
> via `γ_init = γ·λ/L`.

## Non-goals

- **Asymmetric `p_inv`.** Restrict to `p_inv = 0.5` so `p_other = 0.5`
  for both classes (S and I); both predictions then apply uniformly to
  all samples without per-class bookkeeping. Variable `p_inv` is a
  follow-up.
- **Variable Ne / multi-pop / selection.** All three add layers to
  the closed form; out of scope. Single panmictic population at
  constant Ne.
- **Truncated coalescent expression for D.** Avoided by picking
  `t_inv ≫ 2Ne` so coalescence completes and the integral collapses to
  the closed-form total branch length. A truncated formula would let
  D run at every `t_inv` rung in the C ladder, but adds derivation
  effort without buying additional discriminative power over the
  currently-passing Q5b monotonicity test.
- **`tract_distribution='fixed'` mode in C/D.** The closed form is the
  same in expectation between `'fixed'` and `'geometric'` at matched
  `λ` (Q5b already validates this within 20%); these tests target only
  `'geometric'` (the biological model).
- **Sweep / partial-sweep model rewrite** (TODO #3 in
  `project_msinv_todo.md`).

## Design decisions (locked in brainstorm)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Two assertions or one? | Both C and D, separate sim runs | C tests the cited formula; D tests rate scaling against coalescent theory. Different bug classes pinpointed. |
| Rust schema change | Add `node_id_at_position: i32` to `FluxRecord` | Cheapest path to interpretation (b); ~5 LOC in `apply_gene_flux`. |
| Sample-conversion algorithm | Marginal-tree descendants of `node_id_at_position`, unioned across events | tskit `tree.samples(u)` already gives the descendant leaves — direct, exact. |
| `γ_C` for test C | `1.5e-5` | Picks rate so `f_pred` spans `[0.10, 0.95]` over the 4-point ladder. |
| `γ_D` for test D | `5e-3` (matches Q5b) | At t_inv=25_000 with full coalescence, expected count ≈ 500 events ⇒ tight 3-SE assertion at n_seeds=30. |
| t_inv ladder for C | `[1000, 4000, 10_000, 25_000]` (4 points, Ne=1000) | Spans `f_pred` from ~0.11 to ~0.94; captures exp curve shape. |
| t_inv for D | `25_000` (≫ 2Ne=2000, ~13× expected TMRCA) | Truncation correction `exp(−t_inv/2Ne) ≈ 4e-6` ⇒ negligible. |
| Sim sharing C↔D | None — different `γ` | Tried bundling at the largest C rung but `γ` requirements conflict (C wants small rate, D wants ~500 events). Each test owns its own n=30 sim batch. |
| Tolerance for C | `max(0.10 abs, 0.20 rel)` per point | At n_seeds=30, MC SE on `f̂` is ≤ √(f(1−f)/30) ≈ 0.09 at f=0.5 (within-seed correlation tightens this); margin is ~1σ-loose, flags rate misalignments. |
| Tolerance for D | `±3 SE` empirical | SE computed from rep variance; 3σ is the conventional MC anchor with no prior on the deviation direction. |
| n_seeds | 30 (each test) | Sweet spot per resume note; same as Q5b. |
| Test mode | `'geometric'` only | `'fixed'` validated by Q5a/Q5b for shape and mean equivalence. |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ rust/msinv-core/src/event_log.rs                             │
│   FluxRecord += node_id_at_position: i32                     │
│   (existing fields untouched)                                │
└──────────────────────────────────────────────────────────────┘
                             │ (used by)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ rust/msinv-core/src/simulator.rs::apply_gene_flux            │
│   On both flux paths, walk segment chain on active[lin_idx]  │
│   to find seg with seg.left ≤ x_event < seg.right; capture   │
│   seg.node_id; pass into FluxRecord.                         │
└──────────────────────────────────────────────────────────────┘
                             │ (consumed by)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ rust/msinv-py/src/lib.rs (PyO3 bridge)                       │
│   FluxRecord → dict gains "node_id_at_position" key          │
└──────────────────────────────────────────────────────────────┘
                             │ (consumed by)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ msinv/hull/_event_log.py                                     │
│   + samples_converted_at(flux_records, ts, position) -> float│
│     uses tskit tree.at(position).samples(node_id)            │
└──────────────────────────────────────────────────────────────┘
                             │ (consumed by)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ tests/hull/test_phase3b_b2_flux.py                           │
│   + test_andolfatto_sample_fraction_matches_closed_form (C)  │
│   + test_event_coverage_matches_coalescent_closed_form (D)   │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. `FluxRecord` schema extension

Add one field:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluxRecord {
    pub t: f64,
    pub lineage_uid: LinUid,
    pub position: f64,
    pub tract_left: f64,
    pub tract_right: f64,
    pub inv_id: u16,
    pub node_id_at_position: i32,   // NEW
}
```

Required updates:
- Every `FluxRecord { ... }` literal in `rust/msinv-core/src/`,
  `rust/msinv-core/tests/`, `rust/msinv-core/examples/`,
  `rust/msinv-core/benches/`, and `rust/msinv-py/src/`. Use the
  `rust-struct-field-auditor` subagent after the field is added.
- Existing Rust unit tests in `event_log.rs` (`push_and_retrieve_flux`,
  `into_records_preserves_order`) gain the field.
- The PyO3 bridge in `rust/msinv-py/src/lib.rs` adds the key
  `"node_id_at_position"` (i64 in Python) to the flux dict.

### 2. `apply_gene_flux` instrumentation

The function already walks the segment chain to verify tract overlap
(`simulator.rs:1958-1966`). Reuse that walk:

```rust
// Find the segment covering x_event (must exist since flux fired here)
let mut node_id_at_position: i32 = -1;
let mut cur = head;
while cur != SEG_NIL {
    let seg = arena.get(cur);
    if seg.left <= x_event && x_event < seg.right {
        node_id_at_position = seg.node_id;
        break;
    }
    if seg.left >= x_event { break; }
    cur = seg.next;
}
debug_assert!(node_id_at_position >= 0,
    "x_event must fall within a covered segment");
```

Pass `node_id_at_position` into both `log.push_flux(...)` call sites
(fast path at `:1986-1996` and split path at `:2027-2037`).

The walk is logically free — `apply_gene_flux` already iterates the
segment chain on every call. We can either fold the lookup into the
existing overlap walk or do a second pass; the latter is clearer and
the segment chain is cache-warm. Implementation will pick the cleaner
of the two; both are O(segments).

### 3. `samples_converted_at` helper

```python
def samples_converted_at(flux_records, ts, position):
    """Fraction of samples whose ancestry at `position` was hit by ≥1
    flux event.

    For each flux record, takes descendants of node_id_at_position in
    the marginal tree at `position` and unions them into a converted
    set.

    Parameters
    ----------
    flux_records : iterable of dicts
        Filtered flux records (from filter_flux). Must contain
        "node_id_at_position".
    ts : tskit.TreeSequence
    position : float
        Genomic position; must lie inside the inversion under test.

    Returns
    -------
    fraction : float in [0.0, 1.0]
        len(converted) / ts.num_samples.
    """
    tree = ts.at(position)
    converted: set[int] = set()
    for rec in flux_records:
        u = int(rec["node_id_at_position"])
        if u < 0:
            continue  # defensive; should not happen with the new schema
        for s in tree.samples(u):
            converted.add(int(s))
    if ts.num_samples == 0:
        return 0.0
    return len(converted) / ts.num_samples
```

Notes:
- `tree.samples(u)` is tskit's built-in descendant-leaf iterator for
  node `u` in this tree. No manual traversal required.
- The set union handles "same sample hit by multiple flux events"
  correctly — that sample is counted once, matching Andolfatto's
  P(≥1 flip).
- Edge case: a flux event whose recorded `node_id_at_position` does
  not appear in the marginal tree at `position` (because the segment
  was split or merged after the event). `tree.samples(u)` raises
  `tskit.LibraryError` for an invalid node. Guard with a presence
  check on `u in tree.nodes()` (or use `try`/`except`); document that
  the helper is robust to such records.

Re-export from `msinv/hull/__init__.py` alongside the existing helpers
(`filter_cmig`, `filter_flux`, `tract_lengths`, `survival_curve`,
`coverage_count`).

### 4. Test C: sample-fraction Andolfatto match

```python
def test_andolfatto_sample_fraction_matches_closed_form():
    """Empirical f̂(t) matches 1 − exp(−γ·p_other·λ²·t/L) over a t_inv
    ladder, within ±0.10 abs OR ±20% rel (per-point, n_seeds=30)."""
    import math
    from msinv.hull._event_log import filter_flux, samples_converted_at

    gamma, lam = 1.5e-5, 300.0      # γ_C: chosen so f_pred spans 0.10-0.95
    bp_left, bp_right = 2000.0, 8000.0
    L = bp_right - bp_left          # 6000.0
    Ne, p_inv = 1000, 0.5
    p_other = 1.0 - p_inv           # 0.5
    n_S, n_I = 10, 10
    inv_center = 0.5 * (bp_left + bp_right)   # 5000.0
    n_seeds = 30
    t_inv_ladder = [1000.0, 4000.0, 10_000.0, 25_000.0]

    for t_inv in t_inv_ladder:
        f_emp = []
        for seed in range(n_seeds):
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
                sequence_length=10_000,
                recombination_rate=1e-8,
                inversions=[inv],
                seed=seed,
                record_events=True,
            )
            ts = sim.simulate()
            flux = filter_flux(sim.event_log, inv_id=0)
            f_emp.append(samples_converted_at(flux, ts, inv_center))
        f_hat = float(np.mean(f_emp))
        f_pred = 1.0 - math.exp(-gamma * p_other * (lam ** 2) * t_inv / L)
        tol = max(0.10, 0.20 * f_pred)
        assert abs(f_hat - f_pred) < tol, (
            f"t_inv={t_inv}: f̂={f_hat:.3f} vs predicted {f_pred:.3f} "
            f"(tol={tol:.3f}, n_seeds={n_seeds})")
```

**Expected `f_pred` at the ladder points** (γ=1.5e-5, p_other=0.5, λ=300, L=6000;
per-gen rate `γ·p_other·λ²/L = 1.125e-4`):
- t_inv=1_000:  f_pred ≈ 0.106
- t_inv=4_000:  f_pred ≈ 0.362
- t_inv=10_000: f_pred ≈ 0.675
- t_inv=25_000: f_pred ≈ 0.940

### 5. Test D: event-count coalescent anchor

```python
def test_event_coverage_matches_coalescent_closed_form():
    """E[coverage_count(x)] ≈ γ·p_other·(λ²/L) · 2·(2Ne)·H_{n−1}
    at t_inv ≫ 2Ne, within ±3 empirical SE."""
    import math
    from msinv.hull._event_log import filter_flux, coverage_count

    gamma, lam = 5e-3, 300.0        # γ_D: matches Q5b; ~500 events/seed
    bp_left, bp_right = 2000.0, 8000.0
    L = bp_right - bp_left
    Ne, p_inv = 1000, 0.5
    p_other = 1.0 - p_inv
    n_S, n_I = 10, 10
    n = n_S + n_I                   # 20
    t_inv = 25_000.0                # ≫ 2Ne; truncation corr. ~4e-6
    inv_center = 5000.0
    n_seeds = 30

    H = sum(1.0 / k for k in range(1, n))                   # H_{n−1}
    e_total_branch = 2.0 * (2 * Ne) * H                     # generations
    expected = gamma * p_other * (lam ** 2 / L) * e_total_branch

    counts = []
    for seed in range(n_seeds):
        inv = InversionSpec(...)        # same as test C
        sim = HullSimulator(..., t_inv=t_inv, seed=seed, record_events=True)
        sim.simulate()
        flux = filter_flux(sim.event_log, inv_id=0)
        counts.append(coverage_count(flux, inv_center))

    mean_emp = float(np.mean(counts))
    se_emp = float(np.std(counts, ddof=1)) / math.sqrt(n_seeds)
    assert abs(mean_emp - expected) < 3 * se_emp, (
        f"empirical {mean_emp:.1f} vs closed form {expected:.1f} "
        f"(3 SE = {3 * se_emp:.1f}, n_seeds={n_seeds})")
```

**Closed-form derivation note:**
- Per-lineage flux event rate = `γ · p_other · λ` per generation
  (Tier-2 calibration: `test_flux_rate_scales_linearly_with_mean_tract_length`).
- Per-event probability of overlapping interior position x ≈ `λ / L`
  (Tier-2 calibration: `test_spatial_profile_uniform_in_interior_geometric`).
- Per-lineage rate of x-touching flux = `γ · p_other · λ²/L` per gen.
- For a Kingman coalescent on n diploid samples with constant Ne,
  expected total branch length = `2 · (2Ne) · H_{n−1}` generations.
- Total expected events at x = (per-lineage rate) × (total branch length)
  = `γ · p_other · (λ²/L) · 2 · (2Ne) · H_{n−1}`.

**Sanity-check expected value** (γ=5e-3, p_other=0.5, λ=300, L=6000,
2Ne=2000, n=20, H_{19} ≈ 3.548):
- e_total_branch ≈ 2 · 2000 · 3.548 ≈ 14_192 gen
- expected ≈ 5e-3 · 0.5 · 15 · 14_192 ≈ **532 events**

Empirical 3-SE for ~532 mean count, n_seeds=30, with ~30% per-seed CV:
`3 · 532 · 0.30 / √30 ≈ 87` (rough). Test passes if empirical is
within ~16% of closed form. Loose enough to absorb boundary effects
(< 1% at this geometry) and finite-n stochasticity, tight enough to
catch a 25%+ rate-scaling bug.

### 6. Sim cost

C and D do **not** share simulations: C uses `γ_C = 1.5e-5` (so f̂
spans the [0.10, 0.95] range over the ladder) and D uses
`γ_D = 5e-3` (so the expected event count is ~500 for tight stats).
A bundling attempt at the largest C rung was considered but the γ
requirements conflict — at γ_C, D's expected count is < 2 events
(noise-dominated); at γ_D, C's f̂ saturates at 1.0 by t_inv=1000.

Sim count:
- **C:** 4 t_inv rungs × 30 seeds = **120 sims**.
- **D:** 1 t_inv × 30 seeds = **30 sims**.
- **Total:** 150 sims. At ~2 sec/sim at this scale (10kb sequence, 20
  samples), expected wall time ≈ 5 minutes for the new tests.

A single helper function `_run_tier3_sim(t_inv, gamma, seed)` is
extracted to avoid duplication across the two tests; it returns
`(ts, sim.event_log)`.

## Validation strategy

### Helper unit tests (3 new in `tests/hull/_event_log_helpers_test.py`)

```python
def test_samples_converted_at_empty_log_returns_zero():
    """No flux records → fraction == 0.0."""

def test_samples_converted_at_root_node_returns_one():
    """Single record with node_id_at_position == root → all samples
    counted; fraction == 1.0."""

def test_samples_converted_at_specific_descendants_match():
    """Synthesize a tree with known structure; a record pointing to a
    specific internal node should yield its known descendant set."""
```

These tests build the helper input directly (no full sim), so they
exercise the descendant-walk logic without depending on the simulator.

### Rust-side parity

`event_log.rs::push_and_retrieve_flux` and
`event_log.rs::into_records_preserves_order` get the new field; they
already round-trip `FluxRecord` so this is a one-line change each.

### Sanity checks

- `γ=0` (zero flux rate) → `coverage_count == 0` and `f̂ == 0` for all
  t_inv. (Existing infrastructure already supports this; not a new
  test, but worth a one-line `pytest.parametrize` if cheap.)
- `f_pred(t_inv → 0) → 0` and `f_pred(t_inv → ∞) → 1` — verified by
  the ladder spanning ~0.11 to ~0.94.

### Pre-existing test suite

Confirm no regressions in:
- `cargo test --release` (full Rust suite incl. tests/, examples/, benches/).
- `.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py` (excluding the 17 pre-existing sweep-target='P' failures and the stress-corners hang per `CLAUDE.md`).

## Risks

- **Closed-form scaling slip.** Ploidy or coalescent-units gotcha
  causes a 2× drift in `expected`. Mitigation: hand-derive in this
  spec (done, §5); cross-check against a γ=0 sanity run and against
  Q5b's already-passing monotonicity in t_inv.
- **`tree.samples(u)` raises on absent node.** Mitigation: guard with
  `if u not in <some presence check>` in the helper; document.
- **Flux event whose `node_id_at_position` lies above an apparent
  re-coalescence at x.** Should still yield the correct descendant
  set in the *marginal tree at x*, because tskit's tree-at-position
  uses the edge state at that position. Edge-walk semantics: verified
  by the `test_samples_converted_at_specific_descendants_match` unit
  test, where we synthesize and assert the expected set.
- **Boundary effects on `λ²/L` approximation.** At `x_center=5000`,
  distance to breakpoints is `3000 = 10·λ`; correction `O(exp(-d/λ))`
  is < 1%. Documented; not a tolerance issue.
- **Test wall time.** 150 sims at this scale (10kb sequence, 20
  samples). The largest-rung sims (`t_inv=25_000`) dominate per-sim
  cost. Estimated total addition: ~5 minutes wall time.

## Implementation sketch (sequence of commits)

1. Rust: extend `FluxRecord` schema; update unit tests in `event_log.rs`.
   Run `rust-struct-field-auditor` subagent. `cargo test --release`
   green.
2. Rust: instrument `apply_gene_flux` to capture
   `node_id_at_position`. `cargo test --release` green.
3. PyO3 bridge: expose new field. Manual smoke-test from Python.
4. Python: add `samples_converted_at` helper + 3 unit tests. Re-export
   from `msinv.hull`.
5. Python: extract `_run_tier3_sim(t_inv, gamma, seed)` helper at
   module scope; add test C. `pytest -v` green.
6. Python: add test D (uses the same helper, different γ).
   `pytest -v` green.
7. Bench-off-path skill: confirm zero regression on the
   `record_events=False` path (no perf cost when the new walk is gated
   behind the `if let Some(log)` guard — but verify).
8. Commit + push to a feature branch `feat/tier3-full-andolfatto`.

## Out of scope / deferred

- **Truncated-coalescent closed form for D at every C rung.** Could
  let D run alongside C at all 4 t_inv values, not just the largest.
  Deferred.
- **Asymmetric `p_inv` test variants.** Deferred — would require
  per-class p_other accounting.
- **Cross-engine bit-equivalence parity harness** — separate roadmap
  item.

## References

- Resume note: `memory/project_b2_flux_session_resume.md`
- b2-flux spec: `docs/superpowers/specs/2026-04-27-peischl-b2-flux-design.md`
- Event-log spec: `docs/superpowers/specs/2026-04-28-event-hook-t3-cmig-tier3q-design.md`
- Andolfatto, P. (2001). Adaptive hitchhiking effects on genome
  variability. *Curr. Opin. Genet. Dev.*, 11(6), 635–641.
- Guerrero, R. F., Rousset, F., & Kirkpatrick, M. (2012). Coalescent
  patterns for chromosomal inversions in divergent populations.
  *Phil. Trans. R. Soc. B*, 367, 430–438.
- Wakeley, J. *Coalescent Theory* (2009), §3.2 (expected total branch
  length under Kingman = `2N · H_{n−1}` haploid scaling, double for
  diploids).
