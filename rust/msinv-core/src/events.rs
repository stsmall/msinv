/// Event handlers: coalescence and recombination.
///
/// Each handler mutates the active lineage list, the segment arena,
/// and the table builder.

use crate::class_tag::BranchClass;
use crate::lineage::{ATagMap, LinUid, Lineage};
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};
use crate::sweep_buckets::SweepBuckets;
use crate::tables::TableBuilder;

/// Coalesce two lineages (indices into `active`) at time `t`.
///
/// Walks both segment chains left-to-right. Where both lineages have
/// ancestral material, the overlap interval gets a new parent node
/// (edges recorded). Non-overlapping intervals are passed through.
/// The merged lineage replaces both in `active`.
///
/// Returns the new node id of the coalescence.
/// Coalesce two lineages, optionally restricted to a specific class.
///
/// `allowed_class`: if `Some(cls)`, only merge at positions where both
/// segments' class `can_coalesce` with `cls`. Non-matching overlap stays
/// on the original lineages. If `None`, merge all overlap (panmictic).
pub fn apply_coalescence(
    active: &mut Vec<Lineage>,
    idx_a: usize,
    idx_b: usize,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    a_tag: Option<&mut ATagMap>,
    buckets: Option<&mut SweepBuckets>,
) -> i32 {
    apply_coalescence_partial(active, idx_a, idx_b, t, arena, tables,
                               next_uid, None, a_tag, buckets)
}

/// Coalesce two lineages. `allowed_class = Some(cls)` restricts the
/// merge to overlap positions where both segments carry `cls`;
/// everything else stays on per-input remainder lineages. `None` does a
/// full merge (all material folds into one output).
pub fn apply_coalescence_partial(
    active: &mut Vec<Lineage>,
    idx_a: usize,
    idx_b: usize,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    allowed_class: Option<BranchClass>,
    mut a_tag: Option<&mut ATagMap>,
    mut buckets: Option<&mut SweepBuckets>,
) -> i32 {
    let pop = active[idx_a].population;
    let new_node = tables.add_internal(t, pop as i32);
    let partial = allowed_class.is_some();
    // Capture parent UIDs and flags before any mutation.
    let parent_a_uid = active[idx_a].uid;
    let parent_b_uid = active[idx_b].uid;
    let (fa, fb, pa_present, pb_present) = if let Some(ref map) = a_tag {
        (map.get(&parent_a_uid).copied().unwrap_or(false),
         map.get(&parent_b_uid).copied().unwrap_or(false),
         map.contains_key(&parent_a_uid),
         map.contains_key(&parent_b_uid))
    } else {
        (false, false, false, false)
    };

    let mut sa = active[idx_a].head;
    let mut sb = active[idx_b].head;

    // Three output chains.
    let mut merged_head: SegIdx = SEG_NIL;
    let mut merged_tail: SegIdx = SEG_NIL;
    let mut a_rem_head: SegIdx = SEG_NIL;
    let mut a_rem_tail: SegIdx = SEG_NIL;
    let mut b_rem_head: SegIdx = SEG_NIL;
    let mut b_rem_tail: SegIdx = SEG_NIL;

    macro_rules! chain_append {
        ($head:expr, $tail:expr, $arena:expr, $l:expr, $r:expr, $nid:expr, $bc:expr) => {{
            let idx = $arena.alloc($l, $r, $nid, $bc);
            if $head == SEG_NIL {
                $head = idx;
            } else {
                $arena.get_mut($tail).next = idx;
            }
            $tail = idx;
        }};
    }

    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let (a_left, a_right, a_node, a_bc, a_next) =
            (a.left, a.right, a.node_id, a.branch_class, a.next);
        let b = arena.get(sb);
        let (b_left, b_right, b_node, b_bc, b_next) =
            (b.left, b.right, b.node_id, b.branch_class, b.next);

        if a_right <= b_left {
            if partial {
                chain_append!(a_rem_head, a_rem_tail, arena,
                               a_left, a_right, a_node, a_bc);
            } else {
                // Non-overlap with the other lineage's first remaining
                // segment, but in panmictic mode (allowed_class=None)
                // we still record an edge from a_node to new_node so
                // the merged lineage's tskit ancestry actually flows
                // through new_node for this region (matches discoal /
                // msprime Hudson semantics).
                tables.add_edge(a_left, a_right, new_node, a_node);
                chain_append!(merged_head, merged_tail, arena,
                               a_left, a_right, new_node, a_bc);
            }
            arena.free(sa);
            sa = a_next;
        } else if b_right <= a_left {
            if partial {
                chain_append!(b_rem_head, b_rem_tail, arena,
                               b_left, b_right, b_node, b_bc);
            } else {
                tables.add_edge(b_left, b_right, new_node, b_node);
                chain_append!(merged_head, merged_tail, arena,
                               b_left, b_right, new_node, b_bc);
            }
            arena.free(sb);
            sb = b_next;
        } else {
            let l = a_left.max(b_left);
            let r = a_right.min(b_right);

            if a_left < l {
                if partial {
                    chain_append!(a_rem_head, a_rem_tail, arena,
                                   a_left, l, a_node, a_bc);
                } else {
                    tables.add_edge(a_left, l, new_node, a_node);
                    chain_append!(merged_head, merged_tail, arena,
                                   a_left, l, new_node, a_bc);
                }
            }
            if b_left < l {
                if partial {
                    chain_append!(b_rem_head, b_rem_tail, arena,
                                   b_left, l, b_node, b_bc);
                } else {
                    tables.add_edge(b_left, l, new_node, b_node);
                    chain_append!(merged_head, merged_tail, arena,
                                   b_left, l, new_node, b_bc);
                }
            }

            let class_ok = match allowed_class {
                None => true,
                Some(cls) => a_bc == cls && b_bc == cls,
            };
            if class_ok {
                tables.add_edge(l, r, new_node, a_node);
                tables.add_edge(l, r, new_node, b_node);
                chain_append!(merged_head, merged_tail, arena,
                               l, r, new_node, a_bc);
            } else {
                chain_append!(a_rem_head, a_rem_tail, arena,
                               l, r, a_node, a_bc);
                chain_append!(b_rem_head, b_rem_tail, arena,
                               l, r, b_node, b_bc);
            }

            if a_right == r {
                arena.free(sa);
                sa = a_next;
            } else {
                arena.get_mut(sa).left = r;
            }
            if b_right == r {
                arena.free(sb);
                sb = b_next;
            } else {
                arena.get_mut(sb).left = r;
            }
        }
    }
    while sa != SEG_NIL {
        let a = arena.get(sa);
        let (a_left, a_right, a_node, a_bc, a_next) =
            (a.left, a.right, a.node_id, a.branch_class, a.next);
        if partial {
            chain_append!(a_rem_head, a_rem_tail, arena,
                           a_left, a_right, a_node, a_bc);
        } else {
            tables.add_edge(a_left, a_right, new_node, a_node);
            chain_append!(merged_head, merged_tail, arena,
                           a_left, a_right, new_node, a_bc);
        }
        arena.free(sa);
        sa = a_next;
    }
    while sb != SEG_NIL {
        let b = arena.get(sb);
        let (b_left, b_right, b_node, b_bc, b_next) =
            (b.left, b.right, b.node_id, b.branch_class, b.next);
        if partial {
            chain_append!(b_rem_head, b_rem_tail, arena,
                           b_left, b_right, b_node, b_bc);
        } else {
            tables.add_edge(b_left, b_right, new_node, b_node);
            chain_append!(merged_head, merged_tail, arena,
                           b_left, b_right, new_node, b_bc);
        }
        arena.free(sb);
        sb = b_next;
    }

    // Remove both originals.
    let (lo, hi) = if idx_a < idx_b { (idx_a, idx_b) } else { (idx_b, idx_a) };
    swap_remove_with_buckets(active, hi, buckets.as_deref_mut());
    if lo < active.len() {
        swap_remove_with_buckets(active, lo, buckets.as_deref_mut());
    }

    // Add output lineages, propagating A-flag to each new UID.
    // Merged child: if either parent had a tag, the merged child
    // inherits one too with value `fa || fb` (A dominates). This
    // matches discoal's coalesceAtTimePopnSweep:
    //   AA event: both A → merged A.
    //   aa event: both present with false → merged a (NOT untagged —
    //     untagged would let the merged lineage drop out of
    //     n_a_lower in PG-B1's bucketization and break the rate
    //     model).
    //   Cross-allele (rate 0 in window; defensive): merged A.
    // Remnants: each remnant inherits its parent's flag verbatim,
    // including the present-or-not bit.
    if merged_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(merged_head, merged_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pa_present || pb_present {
                let is_a = fa || fb;
                map.insert(uid, is_a);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, is_a);
                }
            }
        }
    }
    if a_rem_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(a_rem_head, a_rem_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pa_present {
                map.insert(uid, fa);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, fa);
                }
            }
        }
    }
    if b_rem_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(b_rem_head, b_rem_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pb_present {
                map.insert(uid, fb);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, fb);
                }
            }
        }
    }

    new_node
}

/// Helper: `active.swap_remove(idx)` while keeping `SweepBuckets`
/// in sync. Captures the `removed_uid` and `moved_uid` (if a move
/// happens) before the swap so the bucket index can be patched up
/// against the new layout.
pub(crate) fn swap_remove_with_buckets(
    active: &mut Vec<Lineage>,
    idx: usize,
    buckets: Option<&mut SweepBuckets>,
) {
    let removed_uid = active[idx].uid;
    let last = active.len() - 1;
    let moved_uid = if idx < last { Some(active[last].uid) } else { None };
    active.swap_remove(idx);
    if let Some(b) = buckets {
        b.on_active_swap_remove(removed_uid, moved_uid, idx as u32);
    }
}

/// Compound coalescence — Path 2 merge semantics.
///
/// At each overlap position, the pair coalesces iff
/// `a_bc.can_coalesce(b_bc)` (panmictic on either side satisfies this;
/// S-vs-I at any active barrier fails). Remnants ONLY for genuine
/// class mismatch — NOT for panmictic-in-mixed-class positions, which
/// eliminates the ratchet that makes the bucket-dispatch path blow up
/// at realistic Anopheles Ne.
///
/// Solo (non-overlap) segments fold into the merged lineage — the two
/// input lineages are fully consumed by the event, so their non-
/// overlap material continues as merged's ancestry (standard Hudson
/// treatment, same as apply_coalescence_partial's None-case).
pub fn apply_coalescence_compound(
    active: &mut Vec<Lineage>,
    idx_a: usize,
    idx_b: usize,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    mut a_tag: Option<&mut ATagMap>,
    mut buckets: Option<&mut SweepBuckets>,
) -> i32 {
    let pop = active[idx_a].population;
    let new_node = tables.add_internal(t, pop as i32);
    // Capture parent UIDs and flags before any mutation.
    let parent_a_uid = active[idx_a].uid;
    let parent_b_uid = active[idx_b].uid;
    let (fa, fb, pa_present, pb_present) = if let Some(ref map) = a_tag {
        (map.get(&parent_a_uid).copied().unwrap_or(false),
         map.get(&parent_b_uid).copied().unwrap_or(false),
         map.contains_key(&parent_a_uid),
         map.contains_key(&parent_b_uid))
    } else {
        (false, false, false, false)
    };

    let mut sa = active[idx_a].head;
    let mut sb = active[idx_b].head;

    let mut merged_head: SegIdx = SEG_NIL;
    let mut merged_tail: SegIdx = SEG_NIL;
    let mut a_rem_head: SegIdx = SEG_NIL;
    let mut a_rem_tail: SegIdx = SEG_NIL;
    let mut b_rem_head: SegIdx = SEG_NIL;
    let mut b_rem_tail: SegIdx = SEG_NIL;

    macro_rules! chain_append {
        ($head:expr, $tail:expr, $arena:expr, $l:expr, $r:expr, $nid:expr, $bc:expr) => {{
            let idx = $arena.alloc($l, $r, $nid, $bc);
            if $head == SEG_NIL { $head = idx; }
            else { $arena.get_mut($tail).next = idx; }
            $tail = idx;
        }};
    }

    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let (a_left, a_right, a_node, a_bc, a_next) =
            (a.left, a.right, a.node_id, a.branch_class, a.next);
        let b = arena.get(sb);
        let (b_left, b_right, b_node, b_bc, b_next) =
            (b.left, b.right, b.node_id, b.branch_class, b.next);

        if a_right <= b_left {
            // Solo a — fold into merged (both inputs consumed).
            chain_append!(merged_head, merged_tail, arena,
                           a_left, a_right, a_node, a_bc);
            arena.free(sa);
            sa = a_next;
        } else if b_right <= a_left {
            chain_append!(merged_head, merged_tail, arena,
                           b_left, b_right, b_node, b_bc);
            arena.free(sb);
            sb = b_next;
        } else {
            let l = a_left.max(b_left);
            let r = a_right.min(b_right);

            // Pre-overlap solo bits fold into merged.
            if a_left < l {
                chain_append!(merged_head, merged_tail, arena,
                               a_left, l, a_node, a_bc);
            }
            if b_left < l {
                chain_append!(merged_head, merged_tail, arena,
                               b_left, l, b_node, b_bc);
            }

            if a_bc.can_coalesce(b_bc) {
                tables.add_edge(l, r, new_node, a_node);
                tables.add_edge(l, r, new_node, b_node);
                // Merged class: union of commitments. Panmictic OR S
                // = S; S OR S = S. Never forms an inconsistent tag
                // because can_coalesce ensured no S-vs-I at any inv.
                let merged_bc = BranchClass::from_bits_unchecked(
                    a_bc.bits() | b_bc.bits());
                chain_append!(merged_head, merged_tail, arena,
                               l, r, new_node, merged_bc);
            } else {
                // Genuine barrier (S-vs-I at active inv): each side's
                // segment stays on its own remainder lineage.
                chain_append!(a_rem_head, a_rem_tail, arena,
                               l, r, a_node, a_bc);
                chain_append!(b_rem_head, b_rem_tail, arena,
                               l, r, b_node, b_bc);
            }

            if a_right == r { arena.free(sa); sa = a_next; }
            else { arena.get_mut(sa).left = r; }
            if b_right == r { arena.free(sb); sb = b_next; }
            else { arena.get_mut(sb).left = r; }
        }
    }
    // Tail solo material folds into merged.
    while sa != SEG_NIL {
        let a = arena.get(sa);
        let (a_left, a_right, a_node, a_bc, a_next) =
            (a.left, a.right, a.node_id, a.branch_class, a.next);
        chain_append!(merged_head, merged_tail, arena,
                       a_left, a_right, a_node, a_bc);
        arena.free(sa);
        sa = a_next;
    }
    while sb != SEG_NIL {
        let b = arena.get(sb);
        let (b_left, b_right, b_node, b_bc, b_next) =
            (b.left, b.right, b.node_id, b.branch_class, b.next);
        chain_append!(merged_head, merged_tail, arena,
                       b_left, b_right, b_node, b_bc);
        arena.free(sb);
        sb = b_next;
    }

    let (lo, hi) = if idx_a < idx_b { (idx_a, idx_b) } else { (idx_b, idx_a) };
    swap_remove_with_buckets(active, hi, buckets.as_deref_mut());
    if lo < active.len() {
        swap_remove_with_buckets(active, lo, buckets.as_deref_mut());
    }

    // Add output lineages, propagating A-flag to each new UID.
    // Merged child: if either parent had a tag, the merged child
    // inherits one too with value `fa || fb` (A dominates). This
    // matches discoal's coalesceAtTimePopnSweep:
    //   AA event: both A → merged A.
    //   aa event: both present with false → merged a (NOT untagged —
    //     untagged would let the merged lineage drop out of
    //     n_a_lower in PG-B1's bucketization and break the rate
    //     model).
    //   Cross-allele (rate 0 in window; defensive): merged A.
    // Remnants: each remnant inherits its parent's flag verbatim,
    // including the present-or-not bit.
    if merged_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(merged_head, merged_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pa_present || pb_present {
                let is_a = fa || fb;
                map.insert(uid, is_a);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, is_a);
                }
            }
        }
    }
    if a_rem_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(a_rem_head, a_rem_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pa_present {
                map.insert(uid, fa);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, fa);
                }
            }
        }
    }
    if b_rem_head != SEG_NIL {
        let uid = *next_uid; *next_uid += 1;
        let new_idx = active.len() as u32;
        active.push(Lineage::new(b_rem_head, b_rem_tail, pop, uid, arena));
        if let Some(ref mut map) = a_tag {
            if pb_present {
                map.insert(uid, fb);
                if let Some(ref mut b) = buckets {
                    b.set_tag(uid, new_idx, pop, fb);
                }
            }
        }
    }

    new_node
}

/// Split a lineage at position `x` (recombination event).
///
/// The lineage at `active[idx]` is split into two: [head, x) stays,
/// [x, tail] becomes a new lineage. Both are placed back into `active`.
pub fn apply_recombination(
    active: &mut Vec<Lineage>,
    idx: usize,
    x: f64,
    arena: &mut SegmentArena,
    next_uid: &mut LinUid,
    a_tag: Option<&mut ATagMap>,
    buckets: Option<&mut SweepBuckets>,
) {
    let head = active[idx].head;
    if head == SEG_NIL { return; }
    // x at first_seg.left would empty active[idx] — treat as no-op.
    if x <= arena.get(head).left { return; }
    let parent_uid = active[idx].uid;
    let pop = active[idx].population;
    let uid = *next_uid;
    *next_uid += 1;
    let right = active[idx].split_at(x, arena, uid);
    if let Some(right_lin) = right {
        let new_idx = active.len() as u32;
        active.push(right_lin);
        if let Some(map) = a_tag {
            propagate_a_flag_recomb(map, parent_uid, uid, pop, new_idx, buckets);
        }
    }
}

fn propagate_a_flag_recomb(
    a_tag: &mut ATagMap,
    parent: LinUid,
    child: LinUid,
    pop: u32,
    child_idx: u32,
    buckets: Option<&mut SweepBuckets>,
) {
    if let Some(&flag) = a_tag.get(&parent) {
        a_tag.insert(child, flag);
        if let Some(b) = buckets {
            b.set_tag(child, child_idx, pop, flag);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_tag::BranchClass;

    fn build_lineage(arena: &mut SegmentArena, intervals: &[(f64, f64)],
                      uid: LinUid) -> Lineage {
        let bc = BranchClass::PANMICTIC;
        let mut head = SEG_NIL;
        let mut tail = SEG_NIL;
        for (i, &(l, r)) in intervals.iter().enumerate() {
            let idx = arena.alloc(l, r, i as i32, bc);
            if head == SEG_NIL {
                head = idx;
            } else {
                arena.get_mut(tail).next = idx;
            }
            tail = idx;
        }
        Lineage::new(head, tail, 0, uid, arena)
    }

    fn build_lineage_cls(arena: &mut SegmentArena,
                          segs: &[(f64, f64, BranchClass)],
                          uid: LinUid) -> Lineage {
        let mut head = SEG_NIL;
        let mut tail = SEG_NIL;
        for (i, &(l, r, bc)) in segs.iter().enumerate() {
            let idx = arena.alloc(l, r, i as i32, bc);
            if head == SEG_NIL { head = idx; }
            else { arena.get_mut(tail).next = idx; }
            tail = idx;
        }
        Lineage::new(head, tail, 0, uid, arena)
    }

    #[test]
    fn compound_pan_pan_no_remnants() {
        // Two panmictic lineages with full overlap: single merged output.
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;
        let s0 = tables.add_sample(0.0, 0);
        let s1 = tables.add_sample(0.0, 0);
        let lin_a = {
            let idx = arena.alloc(0.0, 100.0, s0, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 0, &arena)
        };
        let lin_b = {
            let idx = arena.alloc(0.0, 100.0, s1, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 1, &arena)
        };
        let mut active = vec![lin_a, lin_b];
        apply_coalescence_compound(
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid, None, None);
        assert_eq!(active.len(), 1, "expected 1 merged output");
        assert_eq!(tables.num_edges(), 2);
    }

    #[test]
    fn compound_pan_plus_S_merges_without_remnants() {
        // Critical ratchet test. Two lineages, each with [0, 50) PAN +
        // [50, 100) S@inv0. Compound-merge must collapse into ONE
        // output (no remnants), unlike apply_coalescence_partial which
        // would produce 3 outputs via the S-bucket event path.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let lin_a = build_lineage_cls(&mut arena,
            &[(0.0, 50.0, pan), (50.0, 100.0, s)], 0);
        let lin_b = build_lineage_cls(&mut arena,
            &[(0.0, 50.0, pan), (50.0, 100.0, s)], 1);
        let mut active = vec![lin_a, lin_b];
        apply_coalescence_compound(
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid, None, None);
        assert_eq!(active.len(), 1,
            "ratchet: expected 1 output (no remnants)");
        // Edges: one pair per overlap interval ([0,50) + [50,100)) ×
        // two edges per pair = 4. All go to new_node.
        assert_eq!(tables.num_edges(), 4);
    }

    #[test]
    fn compound_S_vs_I_produces_remnants() {
        // Genuine barrier: S vs I at active inv. Pair doesn't merge,
        // both stay as remainders.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;
        let s = BranchClass::single(0, Karyotype::S);
        let i = BranchClass::single(0, Karyotype::I);
        let lin_a = build_lineage_cls(&mut arena, &[(0.0, 100.0, s)], 0);
        let lin_b = build_lineage_cls(&mut arena, &[(0.0, 100.0, i)], 1);
        let mut active = vec![lin_a, lin_b];
        apply_coalescence_compound(
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid, None, None);
        assert_eq!(active.len(), 2,
            "barrier: both remain separate");
        assert_eq!(tables.num_edges(), 0);
    }

    #[test]
    fn compound_mixed_overlap_partial_barrier() {
        // Lineage A: [0, 50) PAN + [50, 100) S@inv0.
        // Lineage B: [0, 50) PAN + [50, 100) I@inv0.
        // Pan positions coalesce; S-vs-I positions go to remnants.
        use crate::class_tag::Karyotype;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;
        let pan = BranchClass::PANMICTIC;
        let s = BranchClass::single(0, Karyotype::S);
        let i = BranchClass::single(0, Karyotype::I);
        let lin_a = build_lineage_cls(&mut arena,
            &[(0.0, 50.0, pan), (50.0, 100.0, s)], 0);
        let lin_b = build_lineage_cls(&mut arena,
            &[(0.0, 50.0, pan), (50.0, 100.0, i)], 1);
        let mut active = vec![lin_a, lin_b];
        apply_coalescence_compound(
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid, None, None);
        // Expected: merged (pan part) + a_rem (S at [50,100)) + b_rem (I).
        assert_eq!(active.len(), 3);
        // Edges: only pan overlap [0, 50) × 2 edges = 2.
        assert_eq!(tables.num_edges(), 2);
    }

    #[test]
    fn coalescence_full_overlap() {
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;

        // Two samples covering [0, 100)
        let s0 = tables.add_sample(0.0, 0);
        let s1 = tables.add_sample(0.0, 0);
        let lin_a = {
            let idx = arena.alloc(0.0, 100.0, s0, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 0, &arena)
        };
        let lin_b = {
            let idx = arena.alloc(0.0, 100.0, s1, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 1, &arena)
        };
        let mut active = vec![lin_a, lin_b];

        let new_node = apply_coalescence(
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid, None, None);

        assert_eq!(active.len(), 1);
        assert_eq!(tables.node_time[new_node as usize], 5.0);
        assert_eq!(tables.num_edges(), 2);
        assert_eq!(tables.edge_parent[0], new_node);
        assert_eq!(tables.edge_parent[1], new_node);
        assert!((active[0].total_length(&arena) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn coalescence_partial_overlap() {
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;

        let s0 = tables.add_sample(0.0, 0);
        let s1 = tables.add_sample(0.0, 0);
        let lin_a = {
            let idx = arena.alloc(0.0, 60.0, s0, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 0, &arena)
        };
        let lin_b = {
            let idx = arena.alloc(40.0, 100.0, s1, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 1, &arena)
        };
        let mut active = vec![lin_a, lin_b];

        let new_node = apply_coalescence(
            &mut active, 0, 1, 3.0, &mut arena, &mut tables, &mut next_uid, None, None);

        // Edges (post 2026-05-01 non-overlap-edges fix to match
        // discoal/msprime Hudson semantics):
        //   - [0, 40) non-overlap of a: 1 edge (a -> new_node).
        //   - [40, 60) overlap: 2 edges (a, b -> new_node).
        //   - [60, 100) non-overlap of b: 1 edge (b -> new_node).
        // Total: 4 edges.
        assert_eq!(tables.num_edges(), 4);
        // Merged lineage covers [0, 100) all routed through new_node.
        assert_eq!(active.len(), 1);
        assert!((active[0].total_length(&arena) - 100.0).abs() < 1e-12);
        let _ = new_node;
    }

    #[test]
    fn apply_recombination_skips_zombie_at_first_seg_left() {
        // x == first_seg.left (occurs when rng.random() returns 0.0 so
        // offset == 0 in find_position) must not produce an empty
        // zombie at active[idx].
        let mut arena = SegmentArena::new();
        let mut next_uid = 10u32;
        let lin = build_lineage(&mut arena, &[(100.0, 200.0)], 0);
        let mut active = vec![lin];

        apply_recombination(&mut active, 0, 100.0, &mut arena, &mut next_uid, None, None);

        assert_eq!(active.len(), 1, "no zombie, no spurious split");
        assert_ne!(active[0].head, SEG_NIL);
        assert!((active[0].total_length(&arena) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn recombination_splits_lineage() {
        let mut arena = SegmentArena::new();
        let mut next_uid = 10u32;
        let lin = build_lineage(&mut arena, &[(0.0, 100.0)], 0);
        let mut active = vec![lin];

        apply_recombination(&mut active, 0, 30.0, &mut arena, &mut next_uid, None, None);

        assert_eq!(active.len(), 2);
        let total: f64 = active.iter().map(|l| l.total_length(&arena)).sum();
        assert!((total - 100.0).abs() < 1e-12);
    }

    #[test]
    fn a_flag_propagates_through_recomb() {
        use std::collections::HashMap;
        let mut arena = SegmentArena::new();
        let mut next_uid = 1u32;
        // Single lineage spanning [0, 100]; UID 0 tagged A.
        let lin = build_lineage(&mut arena, &[(0.0, 100.0)], 0u32);
        let mut active = vec![lin];
        let mut a_tag: ATagMap = ATagMap::default();
        a_tag.insert(0u32, true);

        apply_recombination(&mut active, 0, 50.0, &mut arena, &mut next_uid,
                             Some(&mut a_tag), None);

        // Parent UID (0) should still be in the map (its left half kept that UID).
        // The new right-half UID (1) should also be tagged A.
        assert_eq!(active.len(), 2);
        assert_eq!(a_tag.get(&0u32), Some(&true));
        assert_eq!(a_tag.get(&1u32), Some(&true));
    }

    #[test]
    fn a_flag_propagates_through_coal_or() {
        use std::collections::HashMap;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let mut next_uid = 10u32;
        // Two overlapping panmictic lineages, A=tagged, B=not tagged.
        let s0 = tables.add_sample(0.0, 0);
        let s1 = tables.add_sample(0.0, 0);
        let lin_a = {
            let idx = arena.alloc(0.0, 100.0, s0, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 0u32, &arena)
        };
        let lin_b = {
            let idx = arena.alloc(0.0, 100.0, s1, BranchClass::PANMICTIC);
            Lineage::new(idx, idx, 0, 1u32, &arena)
        };
        let mut active = vec![lin_a, lin_b];
        let mut a_tag: ATagMap = ATagMap::default();
        a_tag.insert(0u32, true);
        a_tag.insert(1u32, false);

        apply_coalescence(&mut active, 0, 1, 1.0, &mut arena, &mut tables,
                          &mut next_uid, Some(&mut a_tag), None);

        // Single output at active[0] with new UID. Should be tagged A (OR of inputs).
        assert_eq!(active.len(), 1);
        let new_uid = active[0].uid;
        assert_eq!(a_tag.get(&new_uid), Some(&true), "merged should inherit A via OR");
    }

    #[test]
    fn a_flag_propagates_through_coal_partial_remnants() {
        use std::collections::HashMap;
        use crate::class_tag::Karyotype;
        // A: [0..50] S, [50..100] panmictic; UID 0, tagged A=true
        // B: [0..100] S; UID 1, tagged A=false
        // allowed_class = Some(S):
        //   - S [0..50] on A and S [0..100] on B → merged gets [0..50] (OR → true)
        //   - PAN [50..100] on A: class mismatch with S → a_rem (inherits A's flag = true)
        //   - S [50..100] on B: non-overlapping with A's tail → b_rem (inherits B's flag = false)
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(100.0, 1);
        let next_uid = 10u32;
        let s_class = BranchClass::single(0, Karyotype::S);
        let s02 = tables.add_sample(0.0, 0);
        let s12 = tables.add_sample(0.0, 0);
        let lin_a = {
            let i0 = arena.alloc(0.0, 50.0, s02, s_class);
            let i1 = arena.alloc(50.0, 100.0, s02, BranchClass::PANMICTIC);
            arena.get_mut(i0).next = i1;
            Lineage::new(i0, i1, 0, 0u32, &arena)
        };
        let lin_b = {
            let i0 = arena.alloc(0.0, 100.0, s12, s_class);
            Lineage::new(i0, i0, 0, 1u32, &arena)
        };
        let mut active = vec![lin_a, lin_b];
        let mut a_tag: ATagMap = ATagMap::default();
        a_tag.insert(0u32, true);
        a_tag.insert(1u32, false);
        let mut next_uid2 = next_uid;

        apply_coalescence_partial(&mut active, 0, 1, 1.0, &mut arena, &mut tables,
                                   &mut next_uid2, Some(s_class), Some(&mut a_tag), None);

        // At least one A-tagged output should exist (the merged lineage inherits OR=true).
        let tagged_count = active.iter()
            .filter(|lin| a_tag.get(&lin.uid) == Some(&true))
            .count();
        assert!(tagged_count >= 1, "expected at least one A-tagged output");
    }
}
