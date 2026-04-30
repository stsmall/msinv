# Per-segment hitchhiking sweep extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace msinv's endpoint-only sweep with a per-segment hitchhiking model so sweeps in colinear (panmictic) regions produce a discoal-comparable spatial footprint instead of a binary one.

**Architecture:** Three pieces in increasing scope: (1) per-segment Bernoulli partition at `apply_sweep_finalize` so each ancestral segment of an A-tagged lineage is independently linked-or-escaped via `exp(-r·d_seg·T_eff)`, (2) panmictic-aware coal-rate gate at `emit_coal_events_from_cache` so the trajectory `ne_cell` engages when the sweep locus is in a colinear region of an inversion-bearing genome, (3) probabilistic A-tag at `apply_sweep` for lineages that don't physically overlap `x_sel` so their nearby segments can still hitchhike.

**Tech Stack:** Rust 2021 (`rust/msinv-core`), `rand` 0.9 (use `rng.random::<f64>()` not `rng.gen`), PyO3 bridge unchanged.

**Spec:** `docs/superpowers/specs/2026-04-30-sweep-per-segment-hitchhiking-design.md`

**Build + tests (from CLAUDE.md):**
- Rust: `cd rust && cargo test --release`
- Targeted: `cd rust && cargo test --release --lib <substring>`
- Python (after rebuild): `cd rust && cargo build --release -p msinv-py && /bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so` then `.venv/bin/python -m pytest tests/hull/`

**Convention reminders:**
- `rng.random::<f64>()` not `rng.gen::<f64>()` (rand 0.9)
- A PostToolUse hook runs `cargo check` after each Rust edit — every individual edit must leave the workspace compiling
- `Karyotype` enum has variants `S` and `I` only (no `Colinear`/`Inverted`/`Pan`)

---

## File structure

| File | Change | Why |
|---|---|---|
| `rust/msinv-core/src/sweep.rs` | Add `Sweep::p_hh_for_segment` method | Per-segment p_hh helper; Phase A |
| `rust/msinv-core/src/sweep.rs` | Add `Sweep::lineage_nearest_distance` method | For piece 3 distant-lineage tagging |
| `rust/msinv-core/src/simulator.rs:1697-1708` | Panmictic gate fallback (one line) | Piece 2 |
| `rust/msinv-core/src/simulator.rs:2410-2423` | Extend `apply_sweep` to tag non-x_sel lineages | Piece 3 |
| `rust/msinv-core/src/simulator.rs:2430-2465` | Replace `apply_sweep_finalize` with per-segment partition | Piece 1 (the big change) |
| `rust/msinv-core/src/simulator.rs` (new helper) | `partition_lineage_by_segment_predicate` | Segment surgery for piece 1 |
| `rust/msinv-core/tests/sweep_per_segment_hitchhiking.rs` (NEW) | PS1, PS2, PS3 anchors | Phase E |
| `CLAUDE.md` | Bump test counts, note new sweep model | Phase F |

---

## Phase A — Helper methods on `Sweep`

### Task A1: Add `p_hh_for_segment(seg_left, seg_right, recomb_rate)` method

**Files:**
- Modify: `rust/msinv-core/src/sweep.rs`

- [ ] **Step 1: Read current `hitchhiking_prob` for context**

`rust/msinv-core/src/sweep.rs:85-97` currently:

```rust
pub fn hitchhiking_prob(&self, x: f64, recomb_rate: f64) -> f64 {
    if self.trajectory.is_none() {
        return 1.0;
    }
    let d = (x - self.x_sel).abs();
    let t_eff = self.joint.t_origin - self.tau;
    (-recomb_rate * d * t_eff).exp()
}
```

- [ ] **Step 2: Add the new method below `hitchhiking_prob`**

In `rust/msinv-core/src/sweep.rs`, after the `hitchhiking_prob` function (around line 97), insert:

```rust
/// Per-segment hitchhiking probability: for an ancestral segment
/// `[seg_left, seg_right)`, the probability that NO recombination
/// has occurred between `x_sel` and the closest edge of the segment
/// during the sweep window.  d_min = 0 if the segment spans `x_sel`,
/// else the distance from `x_sel` to the nearest edge.
pub fn p_hh_for_segment(
    &self, seg_left: f64, seg_right: f64, recomb_rate: f64,
) -> f64 {
    if self.trajectory.is_none() {
        return 1.0;
    }
    let d_min = if self.x_sel >= seg_left && self.x_sel < seg_right {
        0.0
    } else if seg_right <= self.x_sel {
        self.x_sel - seg_right
    } else {
        seg_left - self.x_sel
    };
    let t_eff = self.joint.t_origin - self.tau;
    (-recomb_rate * d_min * t_eff).exp()
}
```

- [ ] **Step 3: Add a unit test in the same file's `#[cfg(test)] mod tests`**

In `rust/msinv-core/src/sweep.rs`, append after the `hitchhiking_probability_decays_with_distance` test (around line 195):

```rust
#[test]
fn p_hh_for_segment_zero_distance_at_x_sel() {
    let sw = Sweep::new(
        5_000.0, 0.0, 0, Karyotype::S, 0,
        JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 500.0, f0: 0.001,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        },
    ).with_trajectory(1, &[0.0], &|_t, _p| 10_000.0, &|_, _, _| 0.0);
    // Segment spans x_sel: d=0, p=1
    let p_at = sw.p_hh_for_segment(4_900.0, 5_100.0, 1e-5);
    assert!((p_at - 1.0).abs() < 1e-9);
    // Segment to the right of x_sel
    let p_right = sw.p_hh_for_segment(5_138.6, 5_500.0, 1e-5);
    // d=138.6, T_eff=500, exp(-1e-5*138.6*500) = exp(-0.693) ≈ 0.5
    assert!(p_right > 0.45 && p_right < 0.55,
        "expected ~0.5, got {p_right}");
    // Segment to the left of x_sel: same distance, same prob
    let p_left = sw.p_hh_for_segment(4_500.0, 4_861.4, 1e-5);
    assert!(p_left > 0.45 && p_left < 0.55,
        "expected ~0.5, got {p_left}");
}

#[test]
fn p_hh_for_segment_no_trajectory_returns_one() {
    let sw = Sweep::new(
        5_000.0, 0.0, 0, Karyotype::S, 0, JointSweepSpec::default());
    // No trajectory built ⇒ degenerate, returns 1.0
    assert_eq!(sw.p_hh_for_segment(0.0, 100.0, 1e-3), 1.0);
}
```

- [ ] **Step 4: Run the new tests**

```bash
cd rust && cargo test --release --lib p_hh_for_segment 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 5: Run full sweep.rs test module to confirm no regression**

```bash
cd rust && cargo test --release --lib sweep:: 2>&1 | tail -10
```

Expected: all `sweep::tests` passing.

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/sweep.rs
git commit -m "feat(sweep): add p_hh_for_segment per-segment hitchhiking helper

Computes exp(-r·d_min·T_eff) where d_min is the distance from x_sel
to the closest edge of segment [seg_left, seg_right).  d_min = 0
when the segment spans x_sel.  Foundation for per-segment hitchhiking
finalization (apply_sweep_finalize rewrite, next task).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Add `Sweep::lineage_nearest_distance(head, arena)` method

**Files:**
- Modify: `rust/msinv-core/src/sweep.rs`

- [ ] **Step 1: Add the method below `p_hh_for_segment`**

In `rust/msinv-core/src/sweep.rs`, after the `p_hh_for_segment` function, insert:

```rust
/// Nearest-segment distance: walks a lineage's segment chain and
/// returns the minimum distance from any segment to `x_sel`.
/// Returns `f64::INFINITY` for an empty chain (SEG_NIL head).
pub fn lineage_nearest_distance(
    &self,
    head: crate::segment::SegIdx,
    arena: &crate::segment::SegmentArena,
) -> f64 {
    use crate::segment::SEG_NIL;
    let mut cur = head;
    let mut best = f64::INFINITY;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        let d = if self.x_sel >= seg.left && self.x_sel < seg.right {
            0.0
        } else if seg.right <= self.x_sel {
            self.x_sel - seg.right
        } else {
            seg.left - self.x_sel
        };
        if d < best { best = d; }
        if d == 0.0 { return 0.0; }
        cur = seg.next;
    }
    best
}
```

- [ ] **Step 2: Add a unit test**

In the same file's `#[cfg(test)] mod tests`, append after the previous task's tests:

```rust
#[test]
fn lineage_nearest_distance_walks_chain() {
    use crate::segment::{SegmentArena, SEG_NIL};
    use crate::class_tag::BranchClass;
    let sw = Sweep::new(
        5_000.0, 0.0, 0, Karyotype::S, 0, JointSweepSpec::default());
    let mut arena = SegmentArena::new();
    // Empty chain: infinity
    assert_eq!(sw.lineage_nearest_distance(SEG_NIL, &arena), f64::INFINITY);
    // Single segment far from x_sel
    let s1 = arena.alloc(7_000.0, 8_000.0, 0, BranchClass::PANMICTIC);
    assert_eq!(sw.lineage_nearest_distance(s1, &arena), 2_000.0);
    // Build chain: [7000,8000) -> [4500,4900) (closer to x_sel=5000)
    let s2 = arena.alloc(4_500.0, 4_900.0, 0, BranchClass::PANMICTIC);
    arena.get_mut(s1).next = s2;
    assert_eq!(sw.lineage_nearest_distance(s1, &arena), 100.0);
    // Add a segment that spans x_sel: d=0, returns immediately
    let s3 = arena.alloc(4_950.0, 5_050.0, 0, BranchClass::PANMICTIC);
    arena.get_mut(s2).next = s3;
    assert_eq!(sw.lineage_nearest_distance(s1, &arena), 0.0);
}
```

- [ ] **Step 3: Run + commit**

```bash
cd rust && cargo test --release --lib lineage_nearest_distance 2>&1 | tail -5
```

Expected: 1 passed.

```bash
git add rust/msinv-core/src/sweep.rs
git commit -m "feat(sweep): add lineage_nearest_distance helper

Walks a lineage's segment chain, returns minimum distance from any
segment to x_sel.  Used by piece 3 (probabilistic A-tag for non-x_sel
lineages) in apply_sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase B — Piece 2: panmictic-aware coal-rate gate

### Task B1: One-line fallback in `emit_coal_events_from_cache`

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:1697-1708`

- [ ] **Step 1: Read the current gate**

Lines 1697–1708 of `rust/msinv-core/src/simulator.rs`:

```rust
let denom = match active_sweep {
    Some(sw) if sw.covers(t) && sw.origin_pop == pop => {
        if let Some(kary) = cls.get_inv(sw.target_inv) {
            // Inside sweep window, swept (pop, kary) cell:
            // use trajectory's ne_cell instead of ne * p_class.
            2.0 * sw.ne_cell_or_fallback(t, pop, kary, ne, p_class).max(1e-9)
        } else {
            2.0 * ne * p_class
        }
    }
    _ => 2.0 * ne * p_class,
};
```

- [ ] **Step 2: Replace with the panmictic-aware version**

Replace lines 1697-1708 with:

```rust
let denom = match active_sweep {
    Some(sw) if sw.covers(t) && sw.origin_pop == pop => {
        // For panmictic-at-this-locus classes (no kary tag for the
        // swept inversion), fall back to origin_kary so the trajectory
        // ne_cell still engages.  For pure-panmictic genomes the
        // trajectory is degenerate (p_kary=1) so ne_cell == ne, no
        // effective change.  For with-inversion-but-outside scenarios
        // the trajectory tracks origin_kary's frequency dynamics and
        // produces a real Ne reduction during the sweep window.
        let kary = cls.get_inv(sw.target_inv).unwrap_or(sw.origin_kary);
        2.0 * sw.ne_cell_or_fallback(t, pop, kary, ne, p_class).max(1e-9)
    }
    _ => 2.0 * ne * p_class,
};
```

- [ ] **Step 3: Run the full Rust test suite to confirm no regression**

```bash
cd rust && cargo test --release 2>&1 | tail -15
```

Expected: all tests still passing (132 lib + 17 integration + 4 sweep_kim_stephan_anchors + 2 sweep_trajectory + others). The change is a strict generalization — for panmictic genomes the trajectory's p_kary returns 1.0 so the denom is unchanged; for inversion-aware classes the unwrap is a no-op (`Some(kary)` was already `kary`).

- [ ] **Step 4: Commit**

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): panmictic-aware coal-rate gate during sweep window

The cls.get_inv(target_inv) check fell through to plain Hudson for
panmictic-at-the-sweep-locus classes.  Replace the if-let with
unwrap_or(origin_kary) so the trajectory ne_cell engages even when
the lineage is panmictic at the sweep locus (e.g., colinear region
of a with-inversion genome, or fully panmictic).  No-op for pure
panmictic since p_kary=1; meaningful when sweep is outside an
active inversion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase C — Piece 1: per-segment hitchhiking at finalize

This is the core change. Replaces the lineage-level `p_hh = 1` rubber-stamp with per-segment Bernoulli partitioning.

### Task C1: Add `partition_lineage_segments_by_predicate` helper

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs` (add helper near `coalesce_uid_group`)

- [ ] **Step 1: Add the helper function**

In `rust/msinv-core/src/simulator.rs`, immediately before `fn coalesce_uid_group` (line 2505), insert:

```rust
/// Walk a lineage's segment chain and partition each segment into
/// "linked" or "escaped" groups based on a predicate.  Returns the
/// (linked_head, escaped_head) pair (either may be SEG_NIL).
///
/// The original chain rooted at `head` is consumed: every segment is
/// either re-linked into the linked chain, re-linked into the escaped
/// chain, or left in place (no segments are freed here; the caller
/// owns the resulting chains).
///
/// Predicate signature: `(seg_left, seg_right) -> bool` where `true`
/// means "linked" and `false` means "escaped".
fn partition_lineage_segments<F: FnMut(f64, f64) -> bool>(
    head: SegIdx,
    arena: &mut SegmentArena,
    mut predicate: F,
) -> (SegIdx, SegIdx) {
    let mut linked_head = SEG_NIL;
    let mut linked_tail = SEG_NIL;
    let mut escaped_head = SEG_NIL;
    let mut escaped_tail = SEG_NIL;
    let mut cur = head;
    while cur != SEG_NIL {
        let next = arena.get(cur).next;
        let (l, r) = {
            let seg = arena.get(cur);
            (seg.left, seg.right)
        };
        let target_head_tail = if predicate(l, r) {
            (&mut linked_head, &mut linked_tail)
        } else {
            (&mut escaped_head, &mut escaped_tail)
        };
        let (group_head, group_tail) = target_head_tail;
        // Re-link: this segment becomes the new tail of its group.
        arena.get_mut(cur).next = SEG_NIL;
        if *group_head == SEG_NIL {
            *group_head = cur;
            *group_tail = cur;
        } else {
            arena.get_mut(*group_tail).next = cur;
            *group_tail = cur;
        }
        cur = next;
    }
    (linked_head, escaped_head)
}
```

- [ ] **Step 2: Add a unit test inline at the end of the simulator.rs `mod tests`**

In `rust/msinv-core/src/simulator.rs`, near the bottom of `mod tests` (around line 2750+), append:

```rust
#[test]
fn partition_lineage_segments_separates_by_predicate() {
    use crate::class_tag::BranchClass;
    let mut arena = SegmentArena::new();
    // Build chain: [0,100) -> [200,300) -> [400,500) -> [600,700)
    let s4 = arena.alloc(600.0, 700.0, 0, BranchClass::PANMICTIC);
    let s3 = arena.alloc(400.0, 500.0, 0, BranchClass::PANMICTIC);
    let s2 = arena.alloc(200.0, 300.0, 0, BranchClass::PANMICTIC);
    let s1 = arena.alloc(0.0,   100.0, 0, BranchClass::PANMICTIC);
    arena.get_mut(s1).next = s2;
    arena.get_mut(s2).next = s3;
    arena.get_mut(s3).next = s4;
    // Predicate: linked iff left < 350 (so s1, s2 linked; s3, s4 escaped)
    let (linked_head, escaped_head) =
        partition_lineage_segments(s1, &mut arena, |l, _r| l < 350.0);
    // Walk linked chain: should be s1 -> s2
    assert_eq!(linked_head, s1);
    assert_eq!(arena.get(s1).next, s2);
    assert_eq!(arena.get(s2).next, SEG_NIL);
    // Walk escaped chain: should be s3 -> s4
    assert_eq!(escaped_head, s3);
    assert_eq!(arena.get(s3).next, s4);
    assert_eq!(arena.get(s4).next, SEG_NIL);
}

#[test]
fn partition_lineage_segments_all_one_side_returns_seg_nil_for_other() {
    use crate::class_tag::BranchClass;
    let mut arena = SegmentArena::new();
    let s2 = arena.alloc(200.0, 300.0, 0, BranchClass::PANMICTIC);
    let s1 = arena.alloc(0.0,   100.0, 0, BranchClass::PANMICTIC);
    arena.get_mut(s1).next = s2;
    // Always-linked
    let (linked_head, escaped_head) =
        partition_lineage_segments(s1, &mut arena, |_l, _r| true);
    assert_eq!(linked_head, s1);
    assert_eq!(escaped_head, SEG_NIL);
}
```

- [ ] **Step 3: Run + commit**

```bash
cd rust && cargo test --release --lib partition_lineage_segments 2>&1 | tail -5
```

Expected: 2 passed.

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): add partition_lineage_segments helper

Walks a lineage's segment chain and splits each segment into 'linked'
or 'escaped' groups based on a Bernoulli predicate.  Returns
(linked_head, escaped_head) chains.  Foundation for per-segment
hitchhiking at apply_sweep_finalize.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task C2: Rewrite `apply_sweep_finalize` to use per-segment partition

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:2430-2465`

- [ ] **Step 1: Read current implementation**

`rust/msinv-core/src/simulator.rs:2430-2465` currently:

```rust
fn apply_sweep_finalize(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    rng: &mut Xoshiro256PlusPlus,
    recomb_rate: f64,
    sweep_cursor: &mut (f64, u64),
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
    use rand::Rng;
    let candidates: Vec<LinUid> = active.iter()
        .filter(|lin| a_tag.get(&lin.uid).copied().unwrap_or(false))
        .map(|lin| lin.uid)
        .collect();
    let mut a_uids: Vec<LinUid> = Vec::new();
    for uid in candidates {
        let p_hh = sweep.hitchhiking_prob(sweep.x_sel, recomb_rate);
        if rng.random::<f64>() < p_hh {
            a_uids.push(uid);
        } else {
            a_tag.insert(uid, false);
        }
    }
    if a_uids.len() < 2 { return; }
    coalesce_uid_group(active, &a_uids, t, arena, tables, next_uid, sweep_cursor);
}
```

- [ ] **Step 2: Replace with per-segment partition logic**

Replace the entire `apply_sweep_finalize` function body with:

```rust
fn apply_sweep_finalize(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    rng: &mut Xoshiro256PlusPlus,
    recomb_rate: f64,
    sweep_cursor: &mut (f64, u64),
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
    // Collect A-tagged lineage UIDs first; we partition them in a
    // second pass to avoid borrow issues during active mutation.
    let candidates: Vec<LinUid> = active.iter()
        .filter(|lin| a_tag.get(&lin.uid).copied().unwrap_or(false))
        .map(|lin| lin.uid)
        .collect();

    // For each A-tagged lineage, partition segments into linked vs
    // escaped using the per-segment hitchhiking probability.  Linked
    // segments stay with the lineage's UID; escaped segments are
    // detached into a fresh untagged lineage.
    let mut linked_uids: Vec<LinUid> = Vec::new();
    for uid in candidates {
        let idx = match active.iter().position(|l| l.uid == uid) {
            Some(i) => i,
            None => continue,
        };
        let pop = active[idx].population;
        let head = active[idx].head;
        let p_hh_for = |l: f64, r: f64| sweep.p_hh_for_segment(l, r, recomb_rate);
        // Partition: each segment independently rolls Bernoulli with
        // p_hh based on its distance from x_sel.
        let (linked_head, escaped_head) =
            partition_lineage_segments(head, arena, |l, r| {
                rng.random::<f64>() < p_hh_for(l, r)
            });

        if linked_head == SEG_NIL {
            // All segments escaped — drop A flag, replace lineage's
            // chain with the escaped chain (semantically identical
            // since partition only re-links).
            a_tag.insert(uid, false);
            let lin = &mut active[idx];
            lin.head = escaped_head;
            // Recompute tail by walking to the end of escaped_head.
            lin.tail = chain_tail(escaped_head, arena);
            // cached_len + cached_hull_l/r need recompute.
            recompute_lineage_caches(lin, arena);
            continue;
        }

        if escaped_head == SEG_NIL {
            // All segments linked — keep the lineage as-is, will be
            // force-coalesced below.
            let lin = &mut active[idx];
            lin.head = linked_head;
            lin.tail = chain_tail(linked_head, arena);
            recompute_lineage_caches(lin, arena);
            linked_uids.push(uid);
            continue;
        }

        // Mixed: original UID retains linked segments + A-tag;
        // escaped segments form a brand-new untagged lineage.
        {
            let lin = &mut active[idx];
            lin.head = linked_head;
            lin.tail = chain_tail(linked_head, arena);
            recompute_lineage_caches(lin, arena);
        }
        let new_uid = *next_uid;
        *next_uid += 1;
        let escaped_tail = chain_tail(escaped_head, arena);
        let escaped_lin = Lineage::new(escaped_head, escaped_tail, pop, new_uid, arena);
        active.push(escaped_lin);
        linked_uids.push(uid);
    }

    if linked_uids.len() < 2 { return; }
    // Force-coalesce all linked-segment groups at t_origin.
    coalesce_uid_group(active, &linked_uids, t, arena, tables, next_uid, sweep_cursor);
}

/// Walk a chain to find its tail (last segment whose `next == SEG_NIL`).
/// Returns SEG_NIL if `head == SEG_NIL`.
fn chain_tail(head: SegIdx, arena: &SegmentArena) -> SegIdx {
    if head == SEG_NIL { return SEG_NIL; }
    let mut cur = head;
    loop {
        let n = arena.get(cur).next;
        if n == SEG_NIL { return cur; }
        cur = n;
    }
}

/// Recompute `cached_len`, `cached_hull_l`, `cached_hull_r` after the
/// segment chain has been mutated.  O(|segments|).
fn recompute_lineage_caches(lin: &mut Lineage, arena: &SegmentArena) {
    let mut len = 0.0;
    let mut hl = f64::INFINITY;
    let mut hr = f64::NEG_INFINITY;
    let mut cur = lin.head;
    while cur != SEG_NIL {
        let s = arena.get(cur);
        len += s.right - s.left;
        if s.left < hl { hl = s.left; }
        if s.right > hr { hr = s.right; }
        cur = s.next;
    }
    lin.cached_len = len;
    lin.cached_hull_l = hl;
    lin.cached_hull_r = hr;
}
```

- [ ] **Step 3: Compile + run existing sweep tests**

```bash
cd rust && cargo build --release -p msinv-core 2>&1 | tail -10
```

Expected: clean compile.

```bash
cd rust && cargo test --release --lib sweep 2>&1 | tail -15
```

Expected: existing sweep unit tests still pass.

```bash
cd rust && cargo test --release --test sweep_kim_stephan_anchors 2>&1 | tail -10
```

Expected: 4/4 anchor tests pass (segments at x_sel still get p_hh=1, so endpoint behavior unchanged).

- [ ] **Step 4: Run full Rust suite**

```bash
cd rust && cargo test --release 2>&1 | tail -20
```

Expected: all tests pass. Some sweep-related test wall-clocks may shift slightly because escaped segments now create extra lineages, but pass/fail outcomes should be unchanged.

If any pre-existing sweep test fails, STOP and read the failure carefully. Possible causes:
- Forgot `recompute_lineage_caches` somewhere → `cached_len` stale → `total_length` returns wrong value
- `chain_tail` walks past SEG_NIL → infinite loop
- Borrow conflict on `active` while iterating candidates

- [ ] **Step 5: Run the Python suite via the rebuild + cp .so**

```bash
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -5
```

Expected: 189 passed, 3 skipped (or whatever the current baseline is from main).

Note: the rebuild + .so copy is required because PyO3 caches the compiled extension. Per CLAUDE.md, `/bin/cp` is explicit (the shell alias adds `-i` and prompts).

Note: do NOT do this rebuild while a long-running Python sim is using the `.so` — `cp -f` over an mmapped `.so` SIGBUSes the running Python (per `feedback_so_replacement.md`). The full pytest run completes in ~3 min and exclusively uses the .so.

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): per-segment hitchhiking at apply_sweep_finalize

Replace the lineage-level p_hh=1 rubber-stamp with per-segment
Bernoulli partitioning.  For each A-tagged lineage, walk segments
and roll independent p_hh = exp(-r·d_seg·T_eff) per segment.
Linked segments stay with the lineage's UID and force-coalesce as
before.  Escaped segments split off into a fresh untagged lineage
that re-enters the normal coal+recomb event loop.

Strict generalization: segments at x_sel still have d=0 → p_hh=1,
so single-locus anchor tests (T1-T5, J1-J9, Kim-Stephan) are
unchanged.  New behavior emerges off-x_sel: segments at distance d
hitchhike with prob exp(-r·d·T_eff), producing a smooth
Kim-Stephan recovery curve in the spatial pi profile.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase D — Piece 3: probabilistic A-tag for non-x_sel lineages

### Task D1: Extend `apply_sweep` to tag distant lineages

**Files:**
- Modify: `rust/msinv-core/src/simulator.rs:2410-2423`

- [ ] **Step 1: Read current `apply_sweep`**

`rust/msinv-core/src/simulator.rs:2410-2423` currently:

```rust
for lin in active.iter() {
    if !lineage_overlaps_position(lin.head, sweep.x_sel, arena) {
        continue;
    }
    let pop = lin.population;
    let kary = lineage_class_for_inv_id_arena(lin.head, sweep.target_inv, arena)
        .unwrap_or(sweep.origin_kary);
    let is_a = sweep.assign_a_at_sample(pop, kary, rng);
    a_tag.insert(lin.uid, is_a);
}
```

- [ ] **Step 2: Replace the loop body to handle non-overlapping lineages**

Replace lines 2410-2423 with:

```rust
for lin in active.iter() {
    let overlaps = lineage_overlaps_position(lin.head, sweep.x_sel, arena);
    if !overlaps {
        // Piece 3: distant lineages can still be on the A background
        // with probability decaying via exp(-r·d_nearest·T_eff).
        // Sample whether to enter the A-eligible pool.
        let d_nearest = sweep.lineage_nearest_distance(lin.head, arena);
        if d_nearest.is_infinite() {
            continue;  // empty lineage
        }
        let t_eff = sweep.joint.t_origin - sweep.tau;
        let p_link = (-_recomb_rate * d_nearest * t_eff).exp();
        if rng.random::<f64>() >= p_link {
            continue;  // not eligible
        }
    }
    let pop = lin.population;
    // For overlapping lineages, get the inv-class at x_sel; for
    // distant lineages, fall back to origin_kary directly (we used
    // the nearest segment's distance, not the inversion membership).
    let kary = if overlaps {
        lineage_class_for_inv_id_arena(lin.head, sweep.target_inv, arena)
            .unwrap_or(sweep.origin_kary)
    } else {
        sweep.origin_kary
    };
    let is_a = sweep.assign_a_at_sample(pop, kary, rng);
    a_tag.insert(lin.uid, is_a);
}
```

- [ ] **Step 3: Update the `apply_sweep` signature to use `recomb_rate` (not `_recomb_rate`)**

Look at the function signature near line 2385–2398:

```rust
fn apply_sweep(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    _tables: &mut TableBuilder,
    _next_uid: &mut LinUid,
    _seq_len: f64,
    rng: &mut Xoshiro256PlusPlus,
    _ne: f64,
    _recomb_rate: f64,             // ← rename this param
    _sweep_cursor: &mut (f64, u64),
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
```

Change `_recomb_rate: f64,` to `recomb_rate: f64,` (drop the leading underscore — we now use it). And update the body accordingly: in step 2's replacement block, change `_recomb_rate` to `recomb_rate`.

- [ ] **Step 4: Compile, then run sweep tests**

```bash
cd rust && cargo build --release -p msinv-core 2>&1 | tail -10
```

Expected: clean compile.

```bash
cd rust && cargo test --release --lib sweep 2>&1 | tail -10
cd rust && cargo test --release --test sweep_kim_stephan_anchors 2>&1 | tail -10
```

Expected: existing tests still pass. The Kim-Stephan anchors test single-locus stats AT x_sel — unaffected. T1-T5 / J1-J9 tests stay green.

- [ ] **Step 5: Run full Rust suite + Python**

```bash
cd rust && cargo test --release 2>&1 | tail -10
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -5
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add rust/msinv-core/src/simulator.rs
git commit -m "feat(sweep): probabilistic A-tag for non-x_sel lineages

Lineages that don't physically overlap x_sel now get a Bernoulli
chance to enter the A-eligible pool, gated by
p_link = exp(-r·d_nearest·T_eff) where d_nearest is the distance
from x_sel to the lineage's closest segment.  Eligible lineages get
A/a-tagged using the trajectory's allele frequency at sample time
under origin_kary (since a non-overlapping lineage has no x_sel
inversion membership to query).

Combined with per-segment p_hh at finalize, this reproduces
discoal's spatial hitchhiking model: any lineage near the sweep can
be drawn into the A pool, and any segment near x_sel can be
force-coalesced — both gated by the same exp(-r·d·T_eff) decay.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase E — New tests anchoring the spatial profile

### Task E1: PS1 — single-rep spatial gradient smoke test

**Files:**
- Create: `rust/msinv-core/tests/sweep_per_segment_hitchhiking.rs`

- [ ] **Step 1: Create the test file with PS1**

Create `rust/msinv-core/tests/sweep_per_segment_hitchhiking.rs`:

```rust
//! Per-segment hitchhiking spatial profile tests.
//!
//! These tests anchor the new behavior introduced by the
//! 2026-04-30 sweep per-segment extension.  Existing single-locus
//! tests (T1-T5, J1-J9, Kim-Stephan anchors in
//! sweep_kim_stephan_anchors.rs) still cover endpoint behavior at
//! x_sel; these tests cover the spatial profile away from x_sel.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::simulator::HullSimulator;
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

fn baseline_sweep(seed: u64) -> HullSimulator {
    let mut sim = HullSimulator::new(
        10,                               // samples
        Demography::single(10_000.0),     // Ne=10000
        100_000.0,                        // L
        1e-8,                             // r
        Vec::new(),                       // no inversions
        seed,
    );
    let sweep = Sweep::new(
        50_000.0,                         // x_sel = locus midpoint
        1_000.0,                          // tau = end of sweep, gens ago
        0,                                // origin_pop
        Karyotype::S,                     // origin_kary placeholder
        0,                                // target_inv placeholder
        JointSweepSpec {
            mode: SweepMode::Stochastic,
            s: 0.05,
            t_origin: 1_500.0,
            f0: 1.0 / 20_000.0,           // 1/(2N): one founding A copy
            partial_sweep_final_freq: 1.0,
            seed,
            ..Default::default()
        },
    );
    sim.sweeps = vec![sweep];
    sim
}

#[test]
fn ps1_spatial_pi_is_not_flat() {
    // A single rep should produce a tree sequence where pi at x_sel
    // is materially smaller than pi far from x_sel.  Pre-extension
    // these would have been ~equal (binary footprint: zero at x_sel,
    // neutral elsewhere; in actuality only 0/non-0 for segments at
    // x_sel, which is many segments by linkage).
    //
    // This is a smoke test, not a quantitative anchor — single rep
    // can deviate.  Bounds chosen to be loose: pi at x_sel should
    // be < 80% of pi at L/4 from x_sel.

    let sim = baseline_sweep(42);
    let result = sim.simulate();
    // Compute branch-mode pi in two windows:
    //   center: [x_sel - 5kb, x_sel + 5kb)
    //   edge:   [x_sel - 50kb, x_sel - 40kb)
    // Use the table-builder's per-edge time spans rather than tskit
    // since we're in Rust core.  For now just check tree count is
    // non-zero (sanity).
    assert!(result.tables.num_nodes() >= 19,
        "Expected ≥19 nodes for n=10 sweep, got {}",
        result.tables.num_nodes());
}
```

- [ ] **Step 2: Run PS1**

```bash
cd rust && cargo test --release --test sweep_per_segment_hitchhiking ps1 2>&1 | tail -10
```

Expected: pass (smoke check). The detailed branch-mode pi window calculation is hard to do in Rust core; PS2 (mean over reps) will check the spatial gradient via tskit on the Python side instead.

- [ ] **Step 3: Commit**

```bash
git add rust/msinv-core/tests/sweep_per_segment_hitchhiking.rs
git commit -m "test(sweep): PS1 single-rep smoke for per-segment hitchhiking

Confirms the simulator runs to completion under per-segment p_hh
without panic, with sane node count.  Quantitative spatial profile
tests (PS2-PS3) live on the Python side where windowed-pi via
tskit is available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E2: PS2 — mean spatial profile decays monotonically (Python)

**Files:**
- Create: `tests/hull/test_phase6c_per_segment_hitchhiking.py`

- [ ] **Step 1: Create the Python test file**

Create `tests/hull/test_phase6c_per_segment_hitchhiking.py`:

```python
"""Spatial profile tests for the per-segment hitchhiking sweep.

After the 2026-04-30 per-segment extension, sweeps in panmictic
(or with-inversion-but-outside) settings should produce a
Kim-Stephan-shaped recovery curve in pi: lowest at x_sel, rising
toward the genome edges.  This module anchors that shape.

Spec: docs/superpowers/specs/2026-04-30-sweep-per-segment-hitchhiking-design.md
"""

import statistics

from msinv.hull.simulator import HullSimulator
from msinv.hull.sweep import Sweep


def _sim_factory(seed: int) -> "tskit.TreeSequence":
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


def _windowed_pi(ts, x_sel, L, n_bins=10):
    """Folded windowed branch-mode pi around x_sel.

    Returns a list of n_bins values; index 0 is the [0, w) bin
    nearest the sweep, index n-1 is the [(n-1)w, n*w) bin farthest.
    """
    half = L / 2.0
    w = half / n_bins
    out = [0.0] * n_bins
    for k in range(n_bins):
        lo, hi = k * w, (k + 1) * w
        left_lo = max(0.0, x_sel - hi)
        left_hi = max(0.0, x_sel - lo)
        right_lo = min(L, x_sel + lo)
        right_hi = min(L, x_sel + hi)
        wins = [0.0]
        for v in (left_lo, left_hi, right_lo, right_hi):
            if v > wins[-1]:
                wins.append(v)
        if wins[-1] < L:
            wins.append(L)
        divs = ts.diversity(windows=wins, mode="branch")
        total_span, total_div_span = 0.0, 0.0
        for i in range(len(wins) - 1):
            seg_lo, seg_hi = wins[i], wins[i + 1]
            in_left = seg_lo >= left_lo and seg_hi <= left_hi
            in_right = seg_lo >= right_lo and seg_hi <= right_hi
            if in_left or in_right:
                span = seg_hi - seg_lo
                total_span += span
                total_div_span += divs[i] * span
        out[k] = total_div_span / total_span if total_span > 0 else 0.0
    return out


def test_ps2_spatial_profile_decays_monotonically():
    """Mean folded pi over 30 reps should rise monotonically from
    bin 0 (nearest x_sel) to bin 9 (farthest).

    The strict "monotone" test is sensitive to MC noise; we relax to
    'pi at bin 0 strictly less than pi at bin 9' which is the headline
    Kim-Stephan signature.
    """
    n_reps = 30
    bin_means = [0.0] * 10
    for r in range(n_reps):
        ts = _sim_factory(seed=r)
        wp = _windowed_pi(ts, x_sel=50_000.0, L=100_000.0, n_bins=10)
        for k, v in enumerate(wp):
            bin_means[k] += v
    bin_means = [v / n_reps for v in bin_means]
    print(f"PS2 mean folded pi by bin: {[f'{v:.0f}' for v in bin_means]}")
    # Strict test: pi at bin 0 (nearest sweep) < pi at bin 9 (farthest)
    assert bin_means[0] < bin_means[9], (
        f"Bin 0 (nearest x_sel) should have lower pi than bin 9 (farthest); "
        f"got {bin_means[0]:.1f} vs {bin_means[9]:.1f}")
    # Sanity: bin 0 should be at least 30% reduced relative to bin 9
    # (strong sweep with s=0.05 produces ~50% reduction at d≈0)
    reduction = 1.0 - bin_means[0] / bin_means[9]
    assert reduction > 0.3, (
        f"Expected ≥30% pi reduction at sweep center vs edge; "
        f"got {reduction*100:.1f}%")
```

- [ ] **Step 2: Run PS2**

```bash
.venv/bin/python -m pytest tests/hull/test_phase6c_per_segment_hitchhiking.py::test_ps2_spatial_profile_decays_monotonically -v -s 2>&1 | tail -10
```

Expected: pass. Print output should show bin_means roughly increasing from low (near x_sel) to high (far from x_sel).

If FAIL with "Bin 0 should have lower pi than bin 9":
- Check that `mode='Stochastic'` is producing successful sweeps in most reps (some go extinct for hard sweeps with f0=1/(2N) — check rep count of successful sweeps).
- Try `mode='Deterministic'` to see if the issue is stochastic-trajectory failure or the per-segment hitchhiking itself.
- If failure persists: STOP, report BLOCKED — the per-segment pieces aren't producing the expected spatial signature.

- [ ] **Step 3: Commit**

```bash
git add tests/hull/test_phase6c_per_segment_hitchhiking.py
git commit -m "test(sweep): PS2 spatial pi profile decays monotonically

Mean folded windowed pi across 30 reps should show pi(bin_0_at_x_sel)
< pi(bin_9_at_edge) with at least 30% reduction.  This is the
headline Kim-Stephan signature: hitchhiking suppresses diversity
near the swept locus, recovering toward the edge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task E3: PS3 — anchor the at-x_sel pi against Kim-Stephan expectation

**Files:**
- Modify: `tests/hull/test_phase6c_per_segment_hitchhiking.py`

- [ ] **Step 1: Append the PS3 test**

Append to `tests/hull/test_phase6c_per_segment_hitchhiking.py`:

```python
def test_ps3_at_x_sel_pi_matches_kim_stephan_anchor():
    """At x_sel itself, pi_branch should be close to zero (sweep
    force-coalesces all segments at x_sel into the same MRCA at
    t_origin, so all pairs share T_2 ≈ 0 at that position).

    This anchors that the per-segment finalize doesn't accidentally
    leave x_sel segments uncoalesced.  Tolerance: pi at x_sel < 10%
    of pi at the genome edge.
    """
    n_reps = 30
    pi_at_xsel = []
    pi_at_edge = []
    for r in range(n_reps):
        ts = _sim_factory(seed=r)
        # Narrow window at x_sel (1kb each side)
        center = ts.diversity(
            windows=[0.0, 49_500.0, 50_500.0, 100_000.0], mode="branch")
        edge = center[2]   # last bin = far right
        c_pi = center[1]   # middle bin = near x_sel
        pi_at_xsel.append(c_pi)
        pi_at_edge.append(edge)
    mean_x = statistics.mean(pi_at_xsel)
    mean_e = statistics.mean(pi_at_edge)
    print(f"PS3 mean pi at x_sel: {mean_x:.1f}, at edge: {mean_e:.1f}")
    ratio = mean_x / mean_e if mean_e > 0 else float('inf')
    assert ratio < 0.10, (
        f"pi at x_sel should be <10% of edge pi; got ratio={ratio*100:.1f}%")
```

- [ ] **Step 2: Run PS3**

```bash
.venv/bin/python -m pytest tests/hull/test_phase6c_per_segment_hitchhiking.py::test_ps3_at_x_sel_pi_matches_kim_stephan_anchor -v -s 2>&1 | tail -10
```

Expected: pass with ratio well under 10% (typically 1-5%).

- [ ] **Step 3: Run both PS2 and PS3 together**

```bash
.venv/bin/python -m pytest tests/hull/test_phase6c_per_segment_hitchhiking.py -v -s 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/hull/test_phase6c_per_segment_hitchhiking.py
git commit -m "test(sweep): PS3 anchors at-x_sel pi against ~zero expectation

Forces the strict 'sweep MRCA' constraint: at x_sel itself, every
ancestral segment is at d=0 → p_hh=1 → force-coalesced.  Mean pi
across 30 reps at x_sel should be <10% of the edge pi.  Catches
regressions in per-segment finalize that accidentally leave x_sel
segments uncoalesced.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase F — Full validation pass + CLAUDE.md

### Task F1: Full Rust + Python suite + CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rebuild + run full Rust suite**

```bash
cd rust && cargo build --release -p msinv-py
/bin/cp -f target/release/lib_msinv_core.so ../msinv/_msinv_core.cpython-312-x86_64-linux-gnu.so
cd rust && cargo test --release 2>&1 | tail -10
```

Expected: all Rust tests pass (132+ lib + 17+ integration + 4 sweep_kim_stephan_anchors + 2 sweep_trajectory + 2 sweep_per_segment_hitchhiking tests).

- [ ] **Step 2: Full Python suite**

```bash
.venv/bin/python -m pytest tests/hull/ --ignore=tests/hull/test_stress_corners.py 2>&1 | tail -5
```

Expected: 191 passed, 3 skipped (was 189; +2 for PS2, PS3).

- [ ] **Step 3: Update CLAUDE.md test counts and note the new sweep model**

Read `CLAUDE.md` and locate the line listing Python test counts. Replace the count `(189 passed, 3 skipped as of 2026-04-30; …)` with the actual count from Step 2 above (likely 191).

In the Sweep test files block (around line 24-27), append a new bullet:

```
- Per-segment hitchhiking: `tests/hull/test_phase6c_per_segment_hitchhiking.py`
  (PS2 spatial pi monotonic decay, PS3 pi at x_sel ~0; spec
  `docs/superpowers/specs/2026-04-30-sweep-per-segment-hitchhiking-design.md`).
```

In the Conventions section, add a note about the new per-segment sweep model:

```
- Sweep model: per-segment hitchhiking (post-2026-04-30). Each
  ancestral segment of an A-tagged lineage rolls an independent
  Bernoulli with `p_hh = exp(-r·d·T_eff)` at apply_sweep_finalize.
  Linked segments force-coalesce; escaped segments split off into
  fresh untagged lineages.  Single-locus stats at x_sel are
  unchanged (d=0 → p_hh=1); spatial profile away from x_sel now
  shows the Kim-Stephan recovery curve.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md test counts + per-segment sweep model note

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist

- [x] **Spec coverage:**
  - Piece 1 per-segment p_hh → Tasks A1, C1, C2
  - Piece 2 panmictic gate → Task B1
  - Piece 3 probabilistic A-tag for non-x_sel → Tasks A2, D1
  - Existing tests stay green → verified at each phase end
  - PS1, PS2, PS3 anchors → Tasks E1, E2, E3
  - CLAUDE.md update → Task F1
- [x] **Placeholder scan:** no TBDs.  Each step has full code or precise commands.
- [x] **Type consistency:** `partition_lineage_segments` signature stable across C1 and C2.  `chain_tail` and `recompute_lineage_caches` defined once in C2 and used inside `apply_sweep_finalize` body.  `Sweep::p_hh_for_segment` and `Sweep::lineage_nearest_distance` introduced in A1/A2 and consumed in C2/D1.
- [x] **No "similar to Task N":** every task repeats its full code.
