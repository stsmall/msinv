# Rust Optimization Plan: msprime-Level Performance for msinv

## Goal

Whole-chromosome simulation at r=1e-8 bp/gen (rho=350,000 for Anopheles 3R)
with inversion class tracking, fast enough for ABC (thousands of reps).

## Architecture Decisions

- **No rho threshold**: single exact algorithm at all recombination rates
- **Array index + swap-update** for cache keys (follows msprime)
- **tskit does simplify**: only sort edges in Rust before handoff
- **Fenwick tree** for O(log n) event selection
- **Incremental rate updates**: O(n) per event instead of O(n^2) rebuild

## Existing Rust Foundation

The `feature/rust-pyo3` branch already has:
- `SegmentArena` with free-list recycling (no heap allocation per segment)
- `BranchClass` as compact `u64` bitmask (2 bits per inversion, up to 32)
- Segment linked lists using `u32` arena indices (cache-friendly `Vec<Segment>`)
- `apply_coalescence` frees consumed segments back to arena in-place

---

## Phase A: Cached Lengths + Running Totals (1-2 days)

**Files:** `lineage.rs`, `events.rs`, `simulator.rs`

Add `total_length: f64` to `Lineage`. Maintain through every event:
- Recombination split: `left.len = split - offset; right.len = orig - left.len`
- Coalescence merge: `merged.len = sum of overlapping intervals` (computed during merge)
- Gene flux: subtract tract from outside, add to tract lineage

Add `total_recomb_rate: f64` running scalar to simulator. Update incrementally:
- Recomb: subtract old lineage's `r * len`, add both new lineages' contributions
- Coal: subtract both, add merged
- Flux: subtract old, add outside + tract

**Eliminates:** O(n * segments) recombination rate scan per iteration → O(1)

**Test:** debug assertions that `total_length` invariant holds after every event

---

## Phase B: ClassPopIndex (2-3 days)

**Files:** `simulator.rs` (new struct)

```rust
struct ClassPopIndex {
    buckets: Vec<Vec<usize>>,  // flat array indexed by (class, pop)
    // class_to_idx: maps BranchClass → bucket row
    // On swap_remove of lineage at idx: update the swapped lineage's
    // bucket entry from old_idx → idx
}
```

Maintained incrementally on every lineage add/remove/class-change/migration.
Replaces O(n) filter for finding same-(class, pop) lineages.

For coalescence rate: iterate bucket, compute k*(k-1)/2 / (2*Ne*p_class).
For event dispatch: index directly into bucket to pick random pair.

**Test:** panmictic results bit-identical on same RNG seed

---

## Phase C: Fenwick Tree (3-4 days)

**Files:** new `fenwick.rs`, `rate_index.rs`, `simulator.rs`

```rust
pub struct FenwickTree {
    tree: Vec<f64>,  // 1-indexed prefix sums
    n: usize,
}

impl FenwickTree {
    fn update(&mut self, i: usize, delta: f64);  // O(log n)
    fn total(&self) -> f64;                        // O(1) cached
    fn find(&self, target: f64) -> usize;          // O(log n) binary descent
}
```

Leaf layout (flat array):
```
[coal bucket 0] [coal bucket 1] ... [recomb] [flux 0] [flux 1] ... [mig 0] [mig 1] ...
```

`RateIndex` maps semantic events → leaf indices. Event dispatch:
```rust
let u = rng.random::<f64>() * fenwick.total();
let leaf = fenwick.find(u);
// decode leaf → event kind via RateIndex
```

**Eliminates:** O(n_events) linear scan → O(log n_events) per event selection

---

## Phase D: Incremental Rate Updates — THE BIG WIN (4-5 days)

**Files:** `rate_index.rs`, `simulator.rs`

### Pair rate cache (array-indexed, follows msprime)

```rust
struct PairRateCache {
    // Symmetric matrix stored as flat Vec, indexed by lineage position.
    // overlap[(i, j)] = SmallVec<[(BranchClass, f64); 4]>
    // Only i < j entries stored.
    overlap: Vec<SmallVec<[(BranchClass, f64); 4]>>,
    n: usize,  // current active lineage count
}
```

**Swap-update protocol** (matching msprime):
When lineage at index `idx` is removed via `swap_remove`:
1. Last lineage moves to `idx`
2. Update all cache entries referencing `last` → now reference `idx`
3. Remove all cache entries referencing the consumed lineage

**Per-event cost:**
- Coalescence (i + j → merged): remove O(n) old pairs for i and j, compute O(n) new pairs for merged. Net: O(n * segments)
- Recombination (i → left, right): remove O(n) old pairs for i, compute O(2n) new pairs. Net: O(n * segments)
- Gene flux: same as recombination
- Migration: recompute pairs only if pop changed (rare)

After updating pair cache, push deltas to Fenwick tree via `fenwick.update(leaf, new_rate - old_rate)`.

### Flux rate cache

```rust
struct FluxCache {
    // weights[lineage_idx][inv_idx] = f64
    weights: Vec<SmallVec<[f64; 4]>>,
}
```

Same swap-update protocol. Only recompute for changed lineages.

**Eliminates:** O(n^2 * segments) full rebuild → O(n * segments) incremental per event

---

## Phase E: Optimized Merge + Partial Coalescence (2 days)

**Files:** `events.rs`, `segment.rs`, `lineage.rs`

### split_at returns 4-tuple

```rust
fn split_at(arena: &mut SegmentArena, head: SegIdx, tail: SegIdx, x: f64)
    -> (Option<(SegIdx, SegIdx)>, Option<(SegIdx, SegIdx)>)
    // (left_head, left_tail), (right_head, right_tail)
```

Eliminates all downstream `find_tail` walks.

### apply_coalescence_partial

Rust implementation of `_coalesce_partial` — only merge at positions where
both segments match `allowed_class`. Three output chains (merged, a_remain,
b_remain) built in a single pass with in-place segment reuse.

### In-place segment mutation

When segment `a` extends before the overlap at `l`: mutate `a.right = l`
in place rather than allocating a new segment for `[a.left, l)`.

---

## Phase F: Hull Prescreen + Rust Edge Sort (2 days)

**Files:** `simulator.rs`, `tables.rs`

### Hull overlap prescreen

Before calling `apply_coalescence` in the coal event handler, O(1) check:
```rust
if active[i].hull_right() <= active[j].hull_left()
|| active[j].hull_right() <= active[i].hull_left() {
    return;  // reject — hulls don't overlap
}
```

Where `hull_left() = head.left` and `hull_right() = tail.right` (already
stored on Lineage).

### sort_edges in Rust

```rust
impl TableBuilder {
    fn sort_edges(&mut self) {
        // Sort by (parent_time desc, child, left) — tskit canonical order
        // Avoids Python-side tc.sort() round-trip
    }
}
```

Python bridge sets `edges_sorted=True`, skips `tc.sort()`.

---

## Phase G: Benchmarks (ongoing)

`criterion` benchmarks in `rust/msinv-core/benches/`:

| Benchmark | Parameters | Target |
|-----------|-----------|--------|
| panmictic_low_rho | n=20, rho=40, no inv | < 1ms/rep |
| inversion_moderate | n=20, rho=40, 1 inv | < 5ms/rep |
| inversion_high_rho | n=20, rho=10000, 1 inv | < 100ms/rep |
| full_chromosome | n=20, rho=350000, 1 inv | < 5s/rep |
| abc_batch | n=20, rho=10000, 1000 reps | < 100s total |

Profile with `perf` / `flamegraph` at each phase to verify bottleneck shifts.

---

## Module Layout

```
rust/msinv-core/src/
  class_tag.rs     — BranchClass bitmask (exists)
  segment.rs       — SegmentArena + 4-tuple split_at
  lineage.rs       — Lineage with total_length, hull accessors
  inversion.rs     — InversionSpec (exists)
  tables.rs        — TableBuilder + sort_edges()
  events.rs        — apply_coalescence, apply_coalescence_partial,
                     apply_recombination, apply_gene_flux
  fenwick.rs       — FenwickTree (new)
  rate_index.rs    — RateIndex, PairRateCache, FluxCache (new)
  demography.rs    — Demography (exists)
  phi.rs           — phi/phi_integral (exists)
  sweep.rs         — Sweep (exists)
  simulator.rs     — HullSimulator main loop (major refactor)
  lib.rs           — module re-exports

rust/msinv-py/src/
  lib.rs           — PyO3 bridge (skip tc.sort when edges pre-sorted)
```

---

## Expected Performance

| Metric | Python now | Rust Phase A-B | Rust Phase C-D | Rust all phases |
|--------|-----------|----------------|----------------|-----------------|
| rho=40 per rep | ~0.7s | ~0.01s | ~0.005s | ~0.003s |
| rho=700 per rep | ~180s | ~3s | ~0.5s | ~0.2s |
| rho=10000 per rep | infeasible | ~30s | ~2s | ~0.5s |
| rho=350000 per rep | infeasible | infeasible | ~60s | ~5s |

50-100x over Python total. Within 2-5x of msprime (constant factor
from inversion class tracking overhead).
