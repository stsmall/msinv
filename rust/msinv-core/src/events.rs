/// Event handlers: coalescence and recombination.
///
/// Each handler mutates the active lineage list, the segment arena,
/// and the table builder.

use crate::class_tag::BranchClass;
use crate::lineage::{LinUid, Lineage};
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};
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
) -> i32 {
    apply_coalescence_partial(active, idx_a, idx_b, t, arena, tables,
                               next_uid, None)
}

pub fn apply_coalescence_partial(
    active: &mut Vec<Lineage>,
    idx_a: usize,
    idx_b: usize,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    allowed_class: Option<BranchClass>,
) -> i32 {
    let pop = active[idx_a].population;
    let new_node = tables.add_internal(t, pop as i32);

    let mut sa = active[idx_a].head;
    let mut sb = active[idx_b].head;

    // Build merged chain.
    let mut merged_head: SegIdx = SEG_NIL;
    let mut merged_tail: SegIdx = SEG_NIL;

    let mut append = |arena: &mut SegmentArena, left: f64, right: f64,
                       node_id: i32, bc: BranchClass| {
        let idx = arena.alloc(left, right, node_id, bc);
        if merged_head == SEG_NIL {
            merged_head = idx;
        } else {
            arena.get_mut(merged_tail).next = idx;
        }
        merged_tail = idx;
    };

    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let b = arena.get(sb);
        let a_left = a.left;
        let a_right = a.right;
        let a_node = a.node_id;
        let a_bc = a.branch_class;
        let a_next = a.next;
        let b_left = b.left;
        let b_right = b.right;
        let b_node = b.node_id;
        let b_bc = b.branch_class;
        let b_next = b.next;

        if a_right <= b_left {
            // a entirely before b — pass through a
            append(arena, a_left, a_right, a_node, a_bc);
            arena.free(sa);
            sa = a_next;
        } else if b_right <= a_left {
            // b entirely before a — pass through b
            append(arena, b_left, b_right, b_node, b_bc);
            arena.free(sb);
            sb = b_next;
        } else {
            // Overlap exists
            let l = a_left.max(b_left);
            let r = a_right.min(b_right);

            // Pre-overlap solo bits
            if a_left < l {
                append(arena, a_left, l, a_node, a_bc);
            }
            if b_left < l {
                append(arena, b_left, l, b_node, b_bc);
            }

            // Overlap → coalescence if class matches (or no class filter).
            let class_ok = match allowed_class {
                None => true,
                Some(cls) => a_bc.can_coalesce(cls) && b_bc.can_coalesce(cls),
            };
            if class_ok {
                tables.add_edge(l, r, new_node, a_node);
                tables.add_edge(l, r, new_node, b_node);
                append(arena, l, r, new_node, a_bc);
            } else {
                // Class doesn't match — keep both segments separate.
                append(arena, l, r, a_node, a_bc);
                append(arena, l, r, b_node, b_bc);
            }

            // Advance: consume the side that ended at r, keep tail of the other
            if a_right == r {
                arena.free(sa);
                sa = a_next;
            } else {
                // a extends past r — truncate a's left to r
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
    // Drain remaining segments from whichever side is left
    while sa != SEG_NIL {
        let a = arena.get(sa);
        let a_left = a.left;
        let a_right = a.right;
        let a_node = a.node_id;
        let a_bc = a.branch_class;
        let a_next = a.next;
        append(arena, a_left, a_right, a_node, a_bc);
        arena.free(sa);
        sa = a_next;
    }
    while sb != SEG_NIL {
        let b = arena.get(sb);
        let b_left = b.left;
        let b_right = b.right;
        let b_node = b.node_id;
        let b_bc = b.branch_class;
        let b_next = b.next;
        append(arena, b_left, b_right, b_node, b_bc);
        arena.free(sb);
        sb = b_next;
    }

    // Remove both from active (remove higher index first to avoid shift)
    let (lo, hi) = if idx_a < idx_b { (idx_a, idx_b) } else { (idx_b, idx_a) };
    active.swap_remove(hi);
    if lo < active.len() {
        active.swap_remove(lo);
    }

    // Add merged lineage if it has material
    if merged_head != SEG_NIL {
        let uid = *next_uid;
        *next_uid += 1;
        active.push(Lineage::new(merged_head, merged_tail, pop, uid, arena));
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
) {
    let uid = *next_uid;
    *next_uid += 1;
    let right = active[idx].split_at(x, arena, uid);
    if let Some(right_lin) = right {
        active.push(right_lin);
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
            &mut active, 0, 1, 5.0, &mut arena, &mut tables, &mut next_uid);

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
            &mut active, 0, 1, 3.0, &mut arena, &mut tables, &mut next_uid);

        // Overlap is [40, 60) — 2 edges
        assert_eq!(tables.num_edges(), 2);
        // Merged lineage: [0, 40) from s0, [40, 60) from new_node, [60, 100) from s1
        assert_eq!(active.len(), 1);
        assert!((active[0].total_length(&arena) - 100.0).abs() < 1e-12);
        let _ = new_node;
    }

    #[test]
    fn recombination_splits_lineage() {
        let mut arena = SegmentArena::new();
        let mut next_uid = 10u32;
        let lin = build_lineage(&mut arena, &[(0.0, 100.0)], 0);
        let mut active = vec![lin];

        apply_recombination(&mut active, 0, 30.0, &mut arena, &mut next_uid);

        assert_eq!(active.len(), 2);
        let total: f64 = active.iter().map(|l| l.total_length(&arena)).sum();
        assert!((total - 100.0).abs() < 1e-12);
    }
}
