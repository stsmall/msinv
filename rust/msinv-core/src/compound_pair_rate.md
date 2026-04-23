# Path 2: compound per-pair rate event loop

## Why

Current main event loop fires ONE event per (pop, class) bucket. A pair
(i, j) whose material spans multiple classes (e.g. PANMICTIC segs
outside an inversion + S-class segs inside) sits in multiple buckets.
When the first bucket fires for cls = X, the partial coalescence
merges X-matching positions and routes non-X positions to
remainder-lineage outputs (a_rem, b_rem). Those remainders themselves
then must coalesce via their own events, producing more remainders.

At realistic anopheles Ne (≥ 10⁶) with long-barrier inversions
(≥ 100 k gen), ~98 % of partial-coal events add exactly one new
lineage. Active-n blows past 10⁵ and each event's bookkeeping cost
(rate-cache peer walks, pair-bucket maintenance, Fenwick updates)
scales with n.

Path 1 attempted a gated "analytic middle" fast path. Failed: the
rate decomposition `Σ C(ks, 2)/(2·Ne·p_class)` double-counted within-
class coalescences by omitting the `1/(2·Ne)` panmictic baseline
subtraction, and at realistic Ne the analytic path was 46-103× SLOWER
than the incremental event loop because its O(n) reclassify step fires
every Hudson-step, not every structural-change event. See
`analytic_middle.md` on `feature/structured-analytic-middle`.

The fix is not a gated fast path. It's a main-path reformulation
where each pair has ONE event at an overlap-weighted compound rate,
no bucket dispatch, no partial-class remainders for shared-panmictic
positions.

## Model

Pair (i, j) with same population `p`. Their material overlap is a set
of intervals, each tagged with a class pair (class_i at position x,
class_j at position x). Cases:

| a_cls | b_cls | coalesces? | rate contribution |
|-------|-------|------------|-------------------|
| PAN   | PAN   | yes        | `L_pan / (2·Ne·L)` |
| S@k   | S@k   | yes        | `L_Sk / (2·Ne·p_std(k)·L)` |
| I@k   | I@k   | yes        | `L_Ik / (2·Ne·p_inv(k)·L)` |
| S@k   | I@k   | NO barrier | 0 |
| PAN   | S@k   | yes*       | `L / (2·Ne·p_std(k)·L)` |
| S@k   | I@k' (k≠k') | depends on inv | `L / (2·Ne·p_std(k)·p_*(k'))` etc. |

(* PAN-vs-S: panmictic at inv k means the lineage has no class
commitment at inv k. It can coalesce with any class at that position.
Rate uses the OTHER lineage's class frequency since the pair's
coalescence rate is constrained by whichever side is class-restricted.)

Total pair rate:

```
r_ij = (1 / (2·Ne·L)) · Σ_overlap_intervals (L_x / p_effective(a_cls, b_cls))
```

where `p_effective` is the product of per-inv class frequencies over
any active barrier (min over the two lineages' classes at each inv).
For uniformly-PAN overlap `p_effective = 1`.

A single event draw:

```
t_next_ij ~ Exp(r_ij)
```

ONE event per pair. When fires, merge at every position where the
pair agrees OR is panmictic on either side. Mismatch positions
(S-vs-I at same inv) stay on the original lineages — that's the real
barrier, not an artifact.

## What changes in the Rust core

### Remove
- `pair_buckets: SmallVec<[(pop, cls, Vec<packed_ij>); 8]>` — per-class
  bucket dispatch.
- `pair_bucket_refs: Vec<PairBucketRefs>` — per-pair back-refs into
  buckets.
- `iter_class_totals`, `emit_coal_events_from_cache` bucket-summary
  functions.
- CoalAggregate + CoalPair event variants → replaced by one
  `CoalPairCompound { i, j }` variant.

### Keep
- Segment arena + linked chains.
- Hull prescreen (`hulls_overlap`).
- Pair positional-bit bitmap for O(n²/64) pairing.
- Peer bitmap (`peer_bits`) — still useful for quick peer walks.
- Recombination, gene flux, migration, sweep, barrier crossing.
- tskit output.

### Add
- `compute_pair_rate(a, b, inversions, barrier_active, pop, Ne, L) -> f64`
  — pure function returning `r_ij`. Walks both chains once with a
  two-pointer merge, accumulates overlap length per effective
  class-frequency denominator, sums.
- `pair_rates: Fenwick` — one rate per nonempty pair, indexed by
  `pair_idx(i, j, cap)`. Event draw = Fenwick proportional pick.
- `apply_coalescence_compound(active, idx_a, idx_b, t, arena, tables,
  next_uid, inversions, barrier_active)` — variant of
  `apply_coalescence_partial` that merges at every pair-compatible
  position in one pass. Mismatch-class positions go to remainder
  lineages (genuine barrier, rare).

### Modify
- `rate_cache.rs` → recompute/clear/update use pair-compound rate.
  Structure stays the same (peer bits + per-lineage segs), but the
  stored value per pair is `f64` rate, not bucket membership.
- `simulator.rs::run_loop` → event dispatch reads from `pair_rates`
  + recomb/flux/migration aggregate leaves.

## Invariants

1. `pair_rates[pair_idx(i, j)] > 0` iff pair has any coalescence-
   eligible overlap (mixed panmictic or matched class on at least one
   position).
2. Total coalescence rate = `pair_rates.sum()`.
3. Firing `CoalPairCompound(i, j)` consumes the pair. Its rate drops
   to 0. New pairs involving the output lineages (merged, optional
   class-mismatch remainders) get fresh rates computed.
4. After any event mutating `active`, the Fenwick and peer_bits reflect
   the new pair set.

## Migration plan

1. **Stage 1 — design doc + isolated rate function** (THIS SESSION).
   Just `compute_pair_rate`, unit tests comparing hand-calculated
   values on synthetic pairs. No event-loop changes yet.
2. **Stage 2 — parallel event loop** (NEXT SESSION).
   Add `HullSimulator::compound_rate: bool` flag. When true, run_loop
   takes the new path with `pair_rates` Fenwick. Old path untouched.
   Parity tests gated by flag: same seed + same params → same tree
   within branch-mode stat tolerance.
3. **Stage 3 — refactor rate_cache**. Reuse peer_bits + lineage_segs,
   drop pair_buckets, store pair_rates.
4. **Stage 4 — Kir/Fol benchmark**. Target: ≤ 5 s/rep for the
   Ne_anc=1M realistic setup.
5. **Stage 5 — flip default** to compound_rate=true, deprecate old
   path, delete bucket code.

## Correctness anchors

- Python reference `_coalesce_partial` in `msinv/hull/simulator.py`
  handles class_ok exactly as Rust today (`==` check, no panmictic
  free-ride). So Python has the SAME ratchet. Path 2 will diverge
  from Python in the class-mismatch handling — we'll need to accept
  that Python is legacy and Path 2 is the canonical semantics.
- Branch-mode π, dxy, Fst distributions must match msprime head-to-
  head (existing parity tests in `tests/hull/test_summary_stats_*.py`).
- Simple scenarios (single-inv, single-pop, short-barrier) where
  ratchet is minimal: Path 2 must match current event loop within
  KS p > 0.1 on 20+ reps.

## Known risks

- `compute_pair_rate` is O(|segs_a| + |segs_b|). Called on every pair
  whose material changes. If current incremental bucket maintenance
  is cheaper than per-pair rate recomputation, Path 2 could regress.
  Mitigation: cache per-pair last-computed rate and diff on updates.
- Fenwick over pair_rates has n² entries; updates on pair mutation
  cost O(log n²) = O(log n). Probably fine.
- Class-mismatch (S-vs-I on both sides at same inv) DOES still
  produce remnants. Rare but possible via gene flux. Keep the
  remainder-lineage output path but it fires only for genuine
  barriers, not for panmictic-in-mixed-class artifacts.

## Non-goals for this branch

- Migration + barrier handling unchanged (per-lineage pop migration,
  cross_barriers_static clears inv bits).
- Sweep handling unchanged.
- Multi-pop structured coalescent unchanged (pair_rates still
  filtered by same-pop).
- pair_bucket_refs layout is going away — any perf notes in
  `project_rho_optimization.md` referencing bucket-specific opts
  will need updating post-merge.
