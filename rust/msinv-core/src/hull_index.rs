//! Hull-overlap index — arena-based AVL tree augmented with `max_end`.
//!
//! Replaces the O(n) lineage scan in `RateCache::recompute_for`
//! (`rate_index.rs:777-792`) with an O(log n + k) overlap query. The
//! v3 spatial-hull rejection (per-pop sorted Vecs) was bottlenecked
//! by O(n) `Vec::insert/remove` on the maintenance side; this module
//! exists to satisfy that retry gate with a true O(log n) structure.
//!
//! ## Layout
//!
//! - Flat `Vec<Node>` arena, `u32` indices for child / parent / max_end
//!   bookkeeping. No `Box<Node>` chase.
//! - `Vec<NodeId>` reverse map keyed on `lineage_idx` for O(1)
//!   "find node for lineage idx" before delete or update.
//! - Free-list `Vec<NodeId>` of vacated slots, reused on insert to
//!   keep the arena dense.
//!
//! ## Invariants
//!
//! - BST ordering by `(hull_l, lineage_idx)`; the lineage idx
//!   tiebreaker lets multiple lineages share a hull.
//! - AVL height invariant: `|h(L) - h(R)| ≤ 1` at every node.
//! - Augmentation: `max_end = max(self.hull_r, left.max_end, right.max_end)`.
//! - Reverse map invariant: `idx_to_node[idx]` is `NULL` iff `idx` is
//!   not in the tree, else points at the node carrying that idx.
//!
//! ## Single-population scope
//!
//! This struct stores ONE population's worth of lineages. The
//! simulator owns a `Vec<HullIndex>` indexed by population so each
//! tree stays small (typical demography is 2-3 pops).
//!
//! ## Stage 1: data structure + tests, no simulator wiring.
//! Stages 2-3 will thread maintenance through the main loop + flip
//! the `recompute_for` query call site. See
//! `project_validation_tracks_resume.md` for the full sequence.

use std::cmp::Ordering;

/// Arena slot id. `NULL` is the sentinel for "no node here".
pub type NodeId = u32;
pub const NULL: NodeId = u32::MAX;

#[derive(Clone, Debug)]
struct Node {
    hull_l: f64,
    hull_r: f64,
    lineage_idx: u32,
    left: NodeId,
    right: NodeId,
    parent: NodeId,
    height: u32,
    max_end: f64,
}

impl Node {
    fn fresh(idx: u32, hull_l: f64, hull_r: f64) -> Self {
        Self {
            hull_l,
            hull_r,
            lineage_idx: idx,
            left: NULL,
            right: NULL,
            parent: NULL,
            height: 1,
            max_end: hull_r,
        }
    }
}

/// Per-population hull-overlap index.
#[derive(Default, Debug, Clone)]
pub struct HullIndex {
    nodes: Vec<Node>,
    free_slots: Vec<NodeId>,
    root: NodeId,
    /// `idx_to_node[idx] == NULL` iff `idx` not in this index.
    idx_to_node: Vec<NodeId>,
    len: u32,
}

impl HullIndex {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_slots: Vec::new(),
            root: NULL,
            idx_to_node: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            free_slots: Vec::new(),
            root: NULL,
            idx_to_node: Vec::with_capacity(cap),
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn contains(&self, idx: u32) -> bool {
        let i = idx as usize;
        i < self.idx_to_node.len() && self.idx_to_node[i] != NULL
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_slots.clear();
        self.root = NULL;
        for slot in self.idx_to_node.iter_mut() {
            *slot = NULL;
        }
        self.len = 0;
    }

    /// Insert a lineage with bounds `[hull_l, hull_r]`. Panics in debug
    /// if the lineage is already present.
    pub fn insert(&mut self, idx: u32, hull_l: f64, hull_r: f64) {
        debug_assert!(hull_l <= hull_r, "insert: hull_l > hull_r");
        debug_assert!(!self.contains(idx), "insert: idx {idx} already present");

        let new_id = self.alloc_slot(Node::fresh(idx, hull_l, hull_r));
        if (idx as usize) >= self.idx_to_node.len() {
            self.idx_to_node.resize(idx as usize + 1, NULL);
        }
        self.idx_to_node[idx as usize] = new_id;
        self.len += 1;

        if self.root == NULL {
            self.root = new_id;
            return;
        }

        // BST descent + attach as leaf, recording parent.
        let mut cur = self.root;
        loop {
            let cmp = Self::key_cmp(
                hull_l, idx,
                self.nodes[cur as usize].hull_l,
                self.nodes[cur as usize].lineage_idx,
            );
            match cmp {
                Ordering::Less | Ordering::Equal => {
                    let l = self.nodes[cur as usize].left;
                    if l == NULL {
                        self.nodes[cur as usize].left = new_id;
                        self.nodes[new_id as usize].parent = cur;
                        break;
                    }
                    cur = l;
                }
                Ordering::Greater => {
                    let r = self.nodes[cur as usize].right;
                    if r == NULL {
                        self.nodes[cur as usize].right = new_id;
                        self.nodes[new_id as usize].parent = cur;
                        break;
                    }
                    cur = r;
                }
            }
        }
        // Walk up: refresh height + max_end + rebalance.
        self.rebalance_up(self.nodes[new_id as usize].parent);
    }

    /// Remove the lineage at `idx`. Panics in debug if not present.
    pub fn remove(&mut self, idx: u32) {
        debug_assert!(self.contains(idx), "remove: idx {idx} not present");
        let node_id = self.idx_to_node[idx as usize];
        self.idx_to_node[idx as usize] = NULL;
        self.len -= 1;

        // Standard BST delete; if 2 children, swap with in-order
        // successor (leftmost of right subtree) THEN delete the
        // successor's slot (which now has 0 or 1 child).
        let (l, r) = (
            self.nodes[node_id as usize].left,
            self.nodes[node_id as usize].right,
        );
        let removal_target;
        let rebalance_from;
        if l == NULL || r == NULL {
            // 0 or 1 child: replace node by its non-null child.
            let child = if l == NULL { r } else { l };
            let parent = self.nodes[node_id as usize].parent;
            self.replace_in_parent(node_id, child);
            if child != NULL {
                self.nodes[child as usize].parent = parent;
            }
            removal_target = node_id;
            rebalance_from = parent;
        } else {
            // 2 children: pull in-order successor's payload, delete
            // the successor's old slot in its place.
            let succ = self.min_node(r);
            // Copy hull/idx from succ into node_id; refresh reverse
            // map for the relocated lineage idx.
            let succ_idx = self.nodes[succ as usize].lineage_idx;
            self.nodes[node_id as usize].hull_l = self.nodes[succ as usize].hull_l;
            self.nodes[node_id as usize].hull_r = self.nodes[succ as usize].hull_r;
            self.nodes[node_id as usize].lineage_idx = succ_idx;
            self.idx_to_node[succ_idx as usize] = node_id;

            // Now detach succ. Successor has no left child by defn.
            let succ_right = self.nodes[succ as usize].right;
            let succ_parent = self.nodes[succ as usize].parent;
            self.replace_in_parent(succ, succ_right);
            if succ_right != NULL {
                self.nodes[succ_right as usize].parent = succ_parent;
            }
            removal_target = succ;
            rebalance_from = succ_parent;
        }
        self.free_slot(removal_target);
        self.rebalance_up(rebalance_from);
    }

    /// Update an existing lineage's hull. Implementation: remove +
    /// insert. Bounds-only changes still pay full O(log n) but avoid
    /// the BST-resort complexity.
    pub fn update(&mut self, idx: u32, new_l: f64, new_r: f64) {
        debug_assert!(self.contains(idx), "update: idx {idx} not present");
        self.remove(idx);
        self.insert(idx, new_l, new_r);
    }

    /// Append every lineage idx whose hull overlaps `(q_l, q_r)` (open
    /// intervals: `a.r > b.l && b.r > a.l`) into `out`. O(log n + k).
    pub fn iter_overlaps(&self, q_l: f64, q_r: f64, out: &mut Vec<u32>) {
        if self.root == NULL {
            return;
        }
        self.walk_overlaps(self.root, q_l, q_r, out);
    }

    fn walk_overlaps(&self, n: NodeId, q_l: f64, q_r: f64, out: &mut Vec<u32>) {
        if n == NULL {
            return;
        }
        let node = &self.nodes[n as usize];
        // Subtree pruning: every interval in this subtree has
        // `r ≤ max_end`, so if max_end ≤ q_l, none overlaps.
        if node.max_end <= q_l {
            return;
        }
        self.walk_overlaps(node.left, q_l, q_r, out);
        // Right-subtree pruning: by BST invariant every right entry
        // has `hull_l ≥ node.hull_l`, so if node.hull_l ≥ q_r the
        // node and its right subtree all start past q_r.
        if node.hull_l >= q_r {
            return;
        }
        if node.hull_r > q_l && q_r > node.hull_l {
            out.push(node.lineage_idx);
        }
        self.walk_overlaps(node.right, q_l, q_r, out);
    }

    // ---------- internals ---------------------------------------

    #[inline]
    fn key_cmp(la: f64, ia: u32, lb: f64, ib: u32) -> Ordering {
        match la.partial_cmp(&lb).unwrap_or(Ordering::Equal) {
            Ordering::Equal => ia.cmp(&ib),
            o => o,
        }
    }

    fn alloc_slot(&mut self, n: Node) -> NodeId {
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot as usize] = n;
            slot
        } else {
            self.nodes.push(n);
            (self.nodes.len() - 1) as NodeId
        }
    }

    fn free_slot(&mut self, id: NodeId) {
        // Clear references so accidental dangling reads are loud.
        let n = &mut self.nodes[id as usize];
        n.left = NULL;
        n.right = NULL;
        n.parent = NULL;
        n.height = 0;
        n.max_end = f64::NEG_INFINITY;
        self.free_slots.push(id);
    }

    /// Replace `child` with `replacement` in `child.parent`'s child
    /// slot, OR set the root if child is the root. Does NOT update
    /// `replacement.parent` (callers do it explicitly to handle the
    /// NULL case).
    fn replace_in_parent(&mut self, child: NodeId, replacement: NodeId) {
        let parent = self.nodes[child as usize].parent;
        if parent == NULL {
            self.root = replacement;
            return;
        }
        if self.nodes[parent as usize].left == child {
            self.nodes[parent as usize].left = replacement;
        } else {
            debug_assert_eq!(self.nodes[parent as usize].right, child);
            self.nodes[parent as usize].right = replacement;
        }
    }

    fn min_node(&self, mut n: NodeId) -> NodeId {
        while self.nodes[n as usize].left != NULL {
            n = self.nodes[n as usize].left;
        }
        n
    }

    #[inline]
    fn height(&self, n: NodeId) -> u32 {
        if n == NULL { 0 } else { self.nodes[n as usize].height }
    }

    #[inline]
    fn max_end_of(&self, n: NodeId) -> f64 {
        if n == NULL { f64::NEG_INFINITY } else { self.nodes[n as usize].max_end }
    }

    fn refresh(&mut self, n: NodeId) {
        if n == NULL {
            return;
        }
        let (l, r, hr) = (
            self.nodes[n as usize].left,
            self.nodes[n as usize].right,
            self.nodes[n as usize].hull_r,
        );
        let h = 1 + self.height(l).max(self.height(r));
        let me = hr.max(self.max_end_of(l)).max(self.max_end_of(r));
        let node = &mut self.nodes[n as usize];
        node.height = h;
        node.max_end = me;
    }

    #[inline]
    fn balance_factor(&self, n: NodeId) -> i32 {
        if n == NULL {
            0
        } else {
            self.height(self.nodes[n as usize].left) as i32
                - self.height(self.nodes[n as usize].right) as i32
        }
    }

    /// Walk up from `start`, refreshing height + max_end at each
    /// ancestor; rebalance whenever the AVL invariant breaks.
    fn rebalance_up(&mut self, start: NodeId) {
        let mut cur = start;
        while cur != NULL {
            self.refresh(cur);
            let bf = self.balance_factor(cur);
            let parent = self.nodes[cur as usize].parent;
            if bf > 1 {
                let l = self.nodes[cur as usize].left;
                if self.balance_factor(l) < 0 {
                    self.rotate_left(l);
                }
                self.rotate_right(cur);
            } else if bf < -1 {
                let r = self.nodes[cur as usize].right;
                if self.balance_factor(r) > 0 {
                    self.rotate_right(r);
                }
                self.rotate_left(cur);
            }
            cur = parent;
        }
    }

    /// Left rotation pivoting on `x`:
    ///
    /// ```text
    ///       x                 y
    ///      / \               / \
    ///     α   y     ===>    x   γ
    ///        / \           / \
    ///       β   γ         α   β
    /// ```
    fn rotate_left(&mut self, x: NodeId) {
        let y = self.nodes[x as usize].right;
        debug_assert!(y != NULL, "rotate_left: missing right child");
        let beta = self.nodes[y as usize].left;
        let xp = self.nodes[x as usize].parent;

        // y takes x's parent slot.
        self.nodes[y as usize].parent = xp;
        if xp == NULL {
            self.root = y;
        } else if self.nodes[xp as usize].left == x {
            self.nodes[xp as usize].left = y;
        } else {
            self.nodes[xp as usize].right = y;
        }
        // x becomes y.left, β becomes x.right.
        self.nodes[y as usize].left = x;
        self.nodes[x as usize].parent = y;
        self.nodes[x as usize].right = beta;
        if beta != NULL {
            self.nodes[beta as usize].parent = x;
        }
        // Refresh order matters: x first (now child of y), then y.
        self.refresh(x);
        self.refresh(y);
    }

    /// Right rotation pivoting on `y`. Mirror of `rotate_left`.
    fn rotate_right(&mut self, y: NodeId) {
        let x = self.nodes[y as usize].left;
        debug_assert!(x != NULL, "rotate_right: missing left child");
        let beta = self.nodes[x as usize].right;
        let yp = self.nodes[y as usize].parent;

        self.nodes[x as usize].parent = yp;
        if yp == NULL {
            self.root = x;
        } else if self.nodes[yp as usize].left == y {
            self.nodes[yp as usize].left = x;
        } else {
            self.nodes[yp as usize].right = x;
        }
        self.nodes[x as usize].right = y;
        self.nodes[y as usize].parent = x;
        self.nodes[y as usize].left = beta;
        if beta != NULL {
            self.nodes[beta as usize].parent = y;
        }
        self.refresh(y);
        self.refresh(x);
    }

    // ---------- debug invariant checks (used in tests) -----------

    /// Walk the whole tree validating AVL height + max_end + BST
    /// ordering invariants. Returns `(height, max_end)` of the
    /// subtree at `n`. Panics on the first invariant break.
    #[cfg(test)]
    fn validate_subtree(
        &self,
        n: NodeId,
        lo_key: Option<(f64, u32)>,
        hi_key: Option<(f64, u32)>,
    ) -> (u32, f64) {
        if n == NULL {
            return (0, f64::NEG_INFINITY);
        }
        let node = &self.nodes[n as usize];
        let key = (node.hull_l, node.lineage_idx);
        if let Some(lo) = lo_key {
            assert!(
                Self::key_cmp(key.0, key.1, lo.0, lo.1) == Ordering::Greater,
                "BST ordering: node {key:?} ≤ ancestor {lo:?}",
            );
        }
        if let Some(hi) = hi_key {
            assert!(
                Self::key_cmp(key.0, key.1, hi.0, hi.1) != Ordering::Greater,
                "BST ordering: node {key:?} > ancestor {hi:?}",
            );
        }
        if node.left != NULL {
            assert_eq!(self.nodes[node.left as usize].parent, n);
        }
        if node.right != NULL {
            assert_eq!(self.nodes[node.right as usize].parent, n);
        }
        let (lh, lm) = self.validate_subtree(node.left, lo_key, Some(key));
        let (rh, rm) = self.validate_subtree(node.right, Some(key), hi_key);
        let bf = lh as i32 - rh as i32;
        assert!(
            bf.abs() <= 1,
            "AVL balance broken at idx={} bf={}", node.lineage_idx, bf,
        );
        let h = 1 + lh.max(rh);
        assert_eq!(node.height, h, "stale height at idx={}", node.lineage_idx);
        let me = node.hull_r.max(lm).max(rm);
        assert_eq!(
            node.max_end, me,
            "stale max_end at idx={}", node.lineage_idx,
        );
        (h, me)
    }

    #[cfg(test)]
    fn validate(&self) {
        if self.root != NULL {
            assert_eq!(self.nodes[self.root as usize].parent, NULL);
        }
        self.validate_subtree(self.root, None, None);
        // Reverse-map invariant: every non-NULL slot should point at a
        // node carrying that idx, AND every node's idx should map back
        // to its slot.
        for (idx, &node_id) in self.idx_to_node.iter().enumerate() {
            if node_id != NULL {
                assert_eq!(
                    self.nodes[node_id as usize].lineage_idx, idx as u32,
                    "reverse map mismatch at idx={idx}",
                );
            }
        }
        // Live count.
        let live = self.idx_to_node.iter().filter(|&&n| n != NULL).count();
        assert_eq!(live as u32, self.len, "len mismatch");
    }

    #[cfg(test)]
    fn linear_overlaps(&self, q_l: f64, q_r: f64) -> Vec<u32> {
        // Brute-force oracle: walk every live slot.
        let mut out = Vec::new();
        for (idx, &node_id) in self.idx_to_node.iter().enumerate() {
            if node_id == NULL {
                continue;
            }
            let n = &self.nodes[node_id as usize];
            if n.hull_r > q_l && q_r > n.hull_l {
                out.push(idx as u32);
            }
        }
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256Plus;

    fn sorted(mut v: Vec<u32>) -> Vec<u32> {
        v.sort_unstable();
        v
    }

    #[test]
    fn empty() {
        let h = HullIndex::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        let mut out = Vec::new();
        h.iter_overlaps(0.0, 1.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn single_insert_query() {
        let mut h = HullIndex::new();
        h.insert(7, 0.0, 1.0);
        h.validate();
        assert_eq!(h.len(), 1);
        assert!(h.contains(7));
        let mut out = Vec::new();
        h.iter_overlaps(0.5, 0.6, &mut out);
        assert_eq!(out, vec![7]);
        // Just-touching boundary: open intervals → no overlap.
        out.clear();
        h.iter_overlaps(1.0, 2.0, &mut out);
        assert!(out.is_empty());
        out.clear();
        h.iter_overlaps(-1.0, 0.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn duplicate_hulls_distinguished_by_idx() {
        // Two lineages with the same hull both insert + query.
        let mut h = HullIndex::new();
        h.insert(1, 10.0, 20.0);
        h.insert(2, 10.0, 20.0);
        h.insert(3, 10.0, 20.0);
        h.validate();
        let mut out = Vec::new();
        h.iter_overlaps(15.0, 16.0, &mut out);
        assert_eq!(sorted(out), vec![1, 2, 3]);
        h.remove(2);
        h.validate();
        assert!(!h.contains(2));
        assert!(h.contains(1) && h.contains(3));
        let mut out = Vec::new();
        h.iter_overlaps(15.0, 16.0, &mut out);
        assert_eq!(sorted(out), vec![1, 3]);
    }

    #[test]
    fn ascending_insert_stays_balanced() {
        let mut h = HullIndex::new();
        for i in 0u32..32 {
            h.insert(i, i as f64, i as f64 + 0.5);
            h.validate();
        }
        // Worst-case AVL height = 1.44·log₂(n+2) - 0.328 ≈ 7.2 at n=32.
        let height = h.nodes[h.root as usize].height;
        assert!(height <= 8, "height {height} exceeds AVL bound for n=32");
    }

    #[test]
    fn descending_insert_stays_balanced() {
        let mut h = HullIndex::new();
        for i in (0u32..32).rev() {
            h.insert(i, i as f64, i as f64 + 0.5);
            h.validate();
        }
        let height = h.nodes[h.root as usize].height;
        assert!(height <= 8, "height {height} exceeds AVL bound for n=32");
    }

    #[test]
    fn random_insert_remove_query() {
        let mut rng = Xoshiro256Plus::seed_from_u64(0xc0ffee);
        let mut h = HullIndex::new();
        let n = 500;
        // Insert random hulls.
        let mut hulls: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                let a = rng.random::<f64>() * 100.0;
                let b = a + rng.random::<f64>() * 10.0;
                (a, b)
            })
            .collect();
        for i in 0..n {
            h.insert(i as u32, hulls[i].0, hulls[i].1);
        }
        h.validate();

        // Random queries against linear-scan oracle.
        for _ in 0..200 {
            let a = rng.random::<f64>() * 100.0;
            let b = a + rng.random::<f64>() * 5.0;
            let mut got = Vec::new();
            h.iter_overlaps(a, b, &mut got);
            got.sort_unstable();
            assert_eq!(got, h.linear_overlaps(a, b), "query [{a}, {b}]");
        }

        // Random removes interleaved with queries.
        let mut alive: Vec<u32> = (0..n as u32).collect();
        alive.shuffle(&mut rng);
        for &victim in alive.iter().take(n / 2) {
            h.remove(victim);
            h.validate();
        }
        for _ in 0..200 {
            let a = rng.random::<f64>() * 100.0;
            let b = a + rng.random::<f64>() * 5.0;
            let mut got = Vec::new();
            h.iter_overlaps(a, b, &mut got);
            got.sort_unstable();
            assert_eq!(got, h.linear_overlaps(a, b), "post-remove query");
        }

        // Updates: change hulls on remaining alive lineages.
        for &alive_idx in alive.iter().skip(n / 2) {
            let new_l = rng.random::<f64>() * 100.0;
            let new_r = new_l + rng.random::<f64>() * 10.0;
            hulls[alive_idx as usize] = (new_l, new_r);
            h.update(alive_idx, new_l, new_r);
            h.validate();
        }
        for _ in 0..200 {
            let a = rng.random::<f64>() * 100.0;
            let b = a + rng.random::<f64>() * 5.0;
            let mut got = Vec::new();
            h.iter_overlaps(a, b, &mut got);
            got.sort_unstable();
            assert_eq!(got, h.linear_overlaps(a, b), "post-update query");
        }
    }

    #[test]
    fn full_drain_then_refill() {
        // Exercise the free-list slot reuse path.
        let mut h = HullIndex::new();
        for i in 0u32..50 {
            h.insert(i, i as f64, i as f64 + 1.0);
        }
        h.validate();
        for i in 0u32..50 {
            h.remove(i);
        }
        h.validate();
        assert!(h.is_empty());
        // Reinsert the same idx range; reverse map should be reusable.
        for i in 0u32..50 {
            h.insert(i, (50.0 - i as f64), (50.0 - i as f64) + 1.0);
        }
        h.validate();
        let mut out = Vec::new();
        h.iter_overlaps(25.0, 26.0, &mut out);
        assert_eq!(sorted(out).len(), 1);
    }

    #[test]
    fn pruning_correctness_max_end() {
        // A query far to the right of all intervals' max_end must
        // hit the `max_end <= q_l` prune at the root.
        let mut h = HullIndex::new();
        for i in 0u32..16 {
            h.insert(i, i as f64, i as f64 + 0.5);
        }
        let mut out = Vec::new();
        h.iter_overlaps(100.0, 101.0, &mut out);
        assert!(out.is_empty());
        // A query spanning everything must return all 16.
        h.iter_overlaps(-1.0, 1000.0, &mut out);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn update_changes_max_end() {
        let mut h = HullIndex::new();
        h.insert(0, 0.0, 1.0);
        h.insert(1, 5.0, 6.0);
        h.insert(2, 10.0, 11.0);
        // Query (8, 9) does not overlap any hull at first.
        let mut out = Vec::new();
        h.iter_overlaps(8.0, 9.0, &mut out);
        assert!(out.is_empty());
        // Stretch idx=0's right end to cover the query window. The
        // ancestor's max_end must propagate up so the new overlap is
        // visible to subsequent queries.
        h.update(0, 0.0, 12.0);
        h.validate();
        out.clear();
        h.iter_overlaps(8.0, 9.0, &mut out);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn delete_two_child_swap_preserves_reverse_map() {
        // Delete a 2-child node; the in-order successor's payload
        // (including lineage_idx) gets relocated. Verify reverse map
        // tracks the relocation.
        let mut h = HullIndex::new();
        // Build a small tree where the root will have two children.
        // 5 / 3 / 7 / 6 / 8 / 2 (varied to force balancing).
        for &(idx, l) in &[(5u32, 5.0), (3, 3.0), (7, 7.0), (6, 6.0), (8, 8.0), (2, 2.0)] {
            h.insert(idx, l, l + 0.1);
        }
        h.validate();
        // Delete a 2-child internal node.
        h.remove(5);
        h.validate();
        for &remaining in &[2u32, 3, 6, 7, 8] {
            assert!(h.contains(remaining), "{} should still be present", remaining);
        }
        assert!(!h.contains(5));
    }

    #[test]
    fn idx_independent_of_arena_index() {
        // Insert with sparse idx values; reverse map grows correctly.
        let mut h = HullIndex::new();
        h.insert(100, 0.0, 1.0);
        h.insert(50, 5.0, 6.0);
        h.insert(200, 10.0, 11.0);
        h.validate();
        assert!(h.contains(100));
        assert!(h.contains(50));
        assert!(h.contains(200));
        assert!(!h.contains(0));
        assert!(!h.contains(150));
        let mut out = Vec::new();
        h.iter_overlaps(5.5, 5.6, &mut out);
        assert_eq!(out, vec![50]);
    }

    #[test]
    fn stress_height_bound() {
        // n=10000 worst-case AVL height ≤ 1.44 log₂(10002) ≈ 19.1 ≤ 20.
        let mut rng = Xoshiro256Plus::seed_from_u64(0x12345);
        let mut h = HullIndex::new();
        let n = 10000u32;
        for i in 0..n {
            let a = rng.random::<f64>() * 1000.0;
            let b = a + rng.random::<f64>();
            h.insert(i, a, b);
        }
        let height = h.nodes[h.root as usize].height;
        assert!(height <= 20, "height {height} > 20 at n=10000");
    }
}
