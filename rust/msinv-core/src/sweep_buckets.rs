//! Sweep-allele indexed buckets: O(n_A) replacement for the SV-phase
//! any-pair candidate scan at `simulator.rs:944-1041`.
//!
//! Maintained alongside the canonical `a_tag: HashMap<LinUid, bool>`
//! and `active: Vec<Lineage>`. For each `(pop, allele)` it stores the
//! `(uid, active_idx)` of every tagged + live lineage. The picker
//! reads bucket entries directly instead of walking `active`.
//!
//! Stage 1: data structure + tests, no simulator wiring.
//! Stage 2 will thread maintenance hooks through the main loop and
//! sweep helpers; Stage 3 flips the picker read site to bucket-iter.

use std::collections::HashMap;

use crate::lineage::{LinUid, Lineage};

/// Allele subgroup. Mirrors the picker's `want_a` axis: `A` for tag=true,
/// `ALower` for tag=false. Untagged lineages are not bucketed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Allele {
    A,
    ALower,
}

impl Allele {
    #[inline]
    fn idx(self) -> usize {
        match self {
            Allele::A => 0,
            Allele::ALower => 1,
        }
    }

    #[inline]
    fn from_is_a(is_a: bool) -> Self {
        if is_a { Allele::A } else { Allele::ALower }
    }
}

/// Reverse pointer: where is `uid` stored?
#[derive(Copy, Clone, Debug)]
struct Pos {
    pop: u32,
    allele: Allele,
    pos: u32,
}

/// `(uid, active_idx)` pair stored in a bucket. Carries `uid` so swap-
/// remove inside the bucket can fix up the moved entry's reverse pointer.
type Entry = (LinUid, u32);

/// Per-(pop, allele) buckets of tagged active lineages.
#[derive(Default, Debug)]
pub struct SweepBuckets {
    /// `buckets[pop][allele.idx()]` is the bucket for that (pop, allele).
    buckets: Vec<[Vec<Entry>; 2]>,
    /// Reverse: uid → (pop, allele, position-in-bucket). Only contains
    /// uids currently in some bucket.
    pos_of: HashMap<LinUid, Pos>,
}

impl SweepBuckets {
    pub fn new(n_pops: u32) -> Self {
        Self {
            buckets: (0..n_pops).map(|_| Default::default()).collect(),
            pos_of: HashMap::new(),
        }
    }

    /// Ensure `buckets` has at least `n_pops` slots. Grows with empty
    /// buckets if needed.
    pub fn ensure_pops(&mut self, n_pops: u32) {
        while (self.buckets.len() as u32) < n_pops {
            self.buckets.push(Default::default());
        }
    }

    /// Number of `(uid, _)` pairs in the (pop, allele) bucket.
    #[inline]
    pub fn len(&self, pop: u32, allele: Allele) -> usize {
        self.buckets
            .get(pop as usize)
            .map(|b| b[allele.idx()].len())
            .unwrap_or(0)
    }

    /// Read-only slice of entries for (pop, allele). Returns `&[]` when
    /// `pop` is out of range.
    #[inline]
    pub fn entries(&self, pop: u32, allele: Allele) -> &[Entry] {
        match self.buckets.get(pop as usize) {
            Some(b) => &b[allele.idx()],
            None => &[],
        }
    }

    /// Set or update the tag for `uid`. If `uid` was previously tagged
    /// (in any bucket), it is moved. Use this for both initial tagging
    /// and tag flips. `idx` is the lineage's current position in `active`.
    pub fn set_tag(&mut self, uid: LinUid, idx: u32, pop: u32, is_a: bool) {
        self.ensure_pops(pop + 1);
        let allele = Allele::from_is_a(is_a);
        if let Some(old) = self.pos_of.get(&uid).copied() {
            if old.pop == pop && old.allele == allele {
                // Same (pop, allele). Only the active idx might have
                // shifted — update in place.
                self.buckets[pop as usize][allele.idx()][old.pos as usize].1 = idx;
                return;
            }
            self.remove_at(old);
        }
        let bucket = &mut self.buckets[pop as usize][allele.idx()];
        let pos = bucket.len() as u32;
        bucket.push((uid, idx));
        self.pos_of.insert(uid, Pos { pop, allele, pos });
    }

    /// Remove `uid` from its bucket if present. No-op if untagged.
    pub fn clear_tag(&mut self, uid: LinUid) {
        if let Some(p) = self.pos_of.remove(&uid) {
            self.remove_at(p);
        }
    }

    /// `active.swap_remove(removed_idx)` happened. Caller must pass:
    /// - `removed_uid`: the uid that was at `removed_idx` and is now gone
    /// - `moved_uid`:   `Some(uid)` of the lineage that was at `len-1`
    ///                   and is now at `removed_idx`. `None` iff
    ///                   `removed_idx == old_len - 1` (no move happened).
    /// - `new_idx`:     the slot the moved lineage now occupies (i.e.
    ///                   `removed_idx`). Ignored when `moved_uid` is `None`.
    ///
    /// This handles both bucket removal of the gone uid and active-idx
    /// fixup for the moved uid.
    pub fn on_active_swap_remove(
        &mut self,
        removed_uid: LinUid,
        moved_uid: Option<LinUid>,
        new_idx: u32,
    ) {
        if let Some(p) = self.pos_of.remove(&removed_uid) {
            self.remove_at(p);
        }
        if let Some(mu) = moved_uid {
            if let Some(p) = self.pos_of.get(&mu).copied() {
                self.buckets[p.pop as usize][p.allele.idx()][p.pos as usize].1 =
                    new_idx;
            }
        }
    }

    /// `active.push(Lineage { uid, pop, .. })` happened at index `new_idx`.
    /// If the lineage already has an `a_tag` entry (e.g. inherited at
    /// recomb split or coalescence), call `set_tag(uid, new_idx, pop, is_a)`.
    /// If it's untagged, this method is a no-op (we don't track untagged).
    /// Provided as a documentation anchor; callers should invoke
    /// `set_tag` directly when they have the tag value available.
    #[inline]
    pub fn on_active_push(&mut self) {
        // Intentionally empty — see set_tag.
    }

    /// `lineage.population` changed from `old_pop` to `new_pop`. If
    /// tagged, move bucket entry. `idx` is the lineage's current active
    /// index (typically unchanged by migration, but pass it explicitly
    /// for safety).
    pub fn on_pop_change(&mut self, uid: LinUid, new_pop: u32, idx: u32) {
        let Some(old) = self.pos_of.get(&uid).copied() else {
            return;
        };
        if old.pop == new_pop {
            // No-op; idx may have shifted, refresh it.
            self.buckets[new_pop as usize][old.allele.idx()][old.pos as usize].1 =
                idx;
            return;
        }
        self.remove_at(old);
        self.ensure_pops(new_pop + 1);
        let bucket = &mut self.buckets[new_pop as usize][old.allele.idx()];
        let pos = bucket.len() as u32;
        bucket.push((uid, idx));
        self.pos_of.insert(
            uid,
            Pos { pop: new_pop, allele: old.allele, pos },
        );
    }

    /// Drop all entries. Use when starting a fresh simulation; cheaper
    /// than reconstructing the struct.
    pub fn clear(&mut self) {
        for slot in self.buckets.iter_mut() {
            slot[0].clear();
            slot[1].clear();
        }
        self.pos_of.clear();
    }

    // ---- internals ----

    /// Remove the entry at `p` from its bucket via swap_remove, fixing
    /// up the moved-into-position entry's reverse pointer.
    fn remove_at(&mut self, p: Pos) {
        let bucket = &mut self.buckets[p.pop as usize][p.allele.idx()];
        let last = bucket.len() - 1;
        bucket.swap_remove(p.pos as usize);
        // pos_of for the removed uid was already pulled out by the caller
        // (or not, in the two callers — set_tag, on_active_swap_remove).
        // Either way, fix the moved-into-position entry.
        if (p.pos as usize) != last {
            let moved_uid = bucket[p.pos as usize].0;
            if let Some(entry) = self.pos_of.get_mut(&moved_uid) {
                entry.pos = p.pos;
            }
        }
    }

    // ---- diagnostics: invariant check used by Stage 2's debug-assert ----

    /// Assert: every `(uid, idx)` in any bucket has `pos_of[uid]`
    /// pointing back to that bucket position. Every uid in `pos_of` is
    /// in exactly one bucket. Used by tests + Stage 2's
    /// `#[cfg(debug_assertions)]` invariant at the picker site.
    /// Stronger cross-check: walks every bucket entry and verifies the
    /// stored `(uid, idx)` pair still points to a live lineage in
    /// `active` whose `(uid, pop)` matches the bucket key, and whose
    /// `a_tag` value matches the bucket's allele. Catches stale-idx
    /// drift that `assert_invariants` (bucket ↔ pos_of only) cannot
    /// see — required by the Stage 3 picker which reads bucket idxs
    /// directly without re-checking against `active`.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_consistent_with(
        &self,
        active: &[Lineage],
        a_tag: &HashMap<LinUid, bool>,
    ) {
        self.assert_invariants();
        for (pop_idx, slot) in self.buckets.iter().enumerate() {
            for (a_idx, bucket) in slot.iter().enumerate() {
                let allele = if a_idx == 0 { Allele::A } else { Allele::ALower };
                let expected_a = matches!(allele, Allele::A);
                for &(uid, idx) in bucket.iter() {
                    let i = idx as usize;
                    assert!(
                        i < active.len(),
                        "bucket entry uid={} idx={} out of bounds for active.len()={}",
                        uid, idx, active.len()
                    );
                    assert_eq!(
                        active[i].uid, uid,
                        "bucket idx {} points to active uid {} but bucket carries uid {}",
                        idx, active[i].uid, uid
                    );
                    assert_eq!(
                        active[i].population as usize, pop_idx,
                        "bucket pop {} but active[{}].population={}",
                        pop_idx, idx, active[i].population
                    );
                    let actual_a = a_tag.get(&uid).copied();
                    assert_eq!(
                        actual_a, Some(expected_a),
                        "bucket allele {:?} for uid {} but a_tag[{}]={:?}",
                        allele, uid, uid, actual_a
                    );
                }
            }
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub fn assert_invariants(&self) {
        let mut seen: HashMap<LinUid, Pos> = HashMap::new();
        for (pop_idx, slot) in self.buckets.iter().enumerate() {
            for (a_idx, bucket) in slot.iter().enumerate() {
                for (pos, &(uid, _)) in bucket.iter().enumerate() {
                    let p = Pos {
                        pop: pop_idx as u32,
                        allele: if a_idx == 0 { Allele::A } else { Allele::ALower },
                        pos: pos as u32,
                    };
                    let prior = seen.insert(uid, p);
                    assert!(
                        prior.is_none(),
                        "uid {} in two buckets: {:?} and {:?}",
                        uid, prior, p
                    );
                    let recorded = self
                        .pos_of
                        .get(&uid)
                        .copied()
                        .unwrap_or_else(|| panic!("uid {} in bucket but not in pos_of", uid));
                    assert_eq!(
                        recorded.pop, p.pop,
                        "pos_of[{}].pop != bucket pop ({} vs {})",
                        uid, recorded.pop, p.pop
                    );
                    assert_eq!(
                        recorded.allele, p.allele,
                        "pos_of[{}].allele mismatch", uid
                    );
                    assert_eq!(
                        recorded.pos, p.pos,
                        "pos_of[{}].pos != bucket pos ({} vs {})",
                        uid, recorded.pos, p.pos
                    );
                }
            }
        }
        assert_eq!(
            seen.len(), self.pos_of.len(),
            "pos_of has {} entries but buckets have {}",
            self.pos_of.len(), seen.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(b: &SweepBuckets) { b.assert_invariants(); }

    #[test]
    fn empty_buckets_have_zero_len() {
        let b = SweepBuckets::new(3);
        assert_eq!(b.len(0, Allele::A), 0);
        assert_eq!(b.len(2, Allele::ALower), 0);
        check(&b);
    }

    #[test]
    fn set_tag_inserts() {
        let mut b = SweepBuckets::new(2);
        b.set_tag(7, 3, 1, true);
        assert_eq!(b.len(1, Allele::A), 1);
        assert_eq!(b.len(1, Allele::ALower), 0);
        assert_eq!(b.entries(1, Allele::A)[0], (7, 3));
        check(&b);
    }

    #[test]
    fn set_tag_flip_moves_bucket() {
        let mut b = SweepBuckets::new(2);
        b.set_tag(7, 3, 0, true);
        assert_eq!(b.len(0, Allele::A), 1);
        b.set_tag(7, 3, 0, false); // flip
        assert_eq!(b.len(0, Allele::A), 0);
        assert_eq!(b.len(0, Allele::ALower), 1);
        assert_eq!(b.entries(0, Allele::ALower)[0], (7, 3));
        check(&b);
    }

    #[test]
    fn set_tag_same_allele_updates_idx() {
        let mut b = SweepBuckets::new(2);
        b.set_tag(7, 3, 0, true);
        b.set_tag(7, 99, 0, true); // same allele, new idx
        assert_eq!(b.len(0, Allele::A), 1);
        assert_eq!(b.entries(0, Allele::A)[0], (7, 99));
        check(&b);
    }

    #[test]
    fn clear_tag_removes() {
        let mut b = SweepBuckets::new(2);
        b.set_tag(7, 3, 0, true);
        b.set_tag(8, 4, 0, true);
        b.clear_tag(7);
        assert_eq!(b.len(0, Allele::A), 1);
        assert_eq!(b.entries(0, Allele::A)[0], (8, 4));
        check(&b);
    }

    #[test]
    fn clear_tag_idempotent() {
        let mut b = SweepBuckets::new(2);
        b.clear_tag(99); // never tagged
        check(&b);
    }

    #[test]
    fn swap_remove_no_move() {
        // removed_idx was the last slot (no moved_uid).
        let mut b = SweepBuckets::new(1);
        b.set_tag(7, 0, 0, true);
        b.set_tag(8, 1, 0, true);
        // active.swap_remove(1) — uid 8 was at idx 1, no other slot to move
        b.on_active_swap_remove(8, None, 1);
        assert_eq!(b.len(0, Allele::A), 1);
        assert_eq!(b.entries(0, Allele::A)[0], (7, 0));
        check(&b);
    }

    #[test]
    fn swap_remove_with_move_updates_idx() {
        // removed_idx in middle; last slot moves into removed_idx.
        let mut b = SweepBuckets::new(1);
        b.set_tag(7, 0, 0, true);
        b.set_tag(8, 1, 0, true);
        b.set_tag(9, 2, 0, true);
        // active.swap_remove(0): uid 7 gone, uid 9 moves from idx 2 to idx 0.
        b.on_active_swap_remove(7, Some(9), 0);
        assert_eq!(b.len(0, Allele::A), 2);
        let entries = b.entries(0, Allele::A);
        assert!(entries.iter().any(|&(u, i)| u == 8 && i == 1));
        assert!(entries.iter().any(|&(u, i)| u == 9 && i == 0));
        check(&b);
    }

    #[test]
    fn swap_remove_untagged_moved() {
        // Removed lineage is tagged; moved lineage is untagged → only
        // remove the gone uid, no idx fixup.
        let mut b = SweepBuckets::new(1);
        b.set_tag(7, 0, 0, true);
        b.on_active_swap_remove(7, Some(99), 0);
        assert_eq!(b.len(0, Allele::A), 0);
        check(&b);
    }

    #[test]
    fn swap_remove_tagged_moved_only() {
        // Removed lineage is untagged; moved lineage is tagged → only
        // update the moved uid's idx.
        let mut b = SweepBuckets::new(1);
        b.set_tag(9, 2, 0, true);
        // Pretend active.swap_remove(0) — uid 0 was untagged, moved uid 9
        // from idx 2 to idx 0.
        b.on_active_swap_remove(0, Some(9), 0);
        assert_eq!(b.len(0, Allele::A), 1);
        assert_eq!(b.entries(0, Allele::A)[0], (9, 0));
        check(&b);
    }

    #[test]
    fn pop_change_moves_bucket() {
        let mut b = SweepBuckets::new(3);
        b.set_tag(7, 0, 0, true);
        b.on_pop_change(7, 2, 0);
        assert_eq!(b.len(0, Allele::A), 0);
        assert_eq!(b.len(2, Allele::A), 1);
        assert_eq!(b.entries(2, Allele::A)[0], (7, 0));
        check(&b);
    }

    #[test]
    fn pop_change_untagged_noop() {
        let mut b = SweepBuckets::new(3);
        b.on_pop_change(99, 1, 0);
        check(&b);
    }

    #[test]
    fn ensure_pops_grows_lazily() {
        let mut b = SweepBuckets::new(1);
        b.set_tag(7, 0, 5, true); // pop=5, beyond initial size
        assert_eq!(b.len(5, Allele::A), 1);
        check(&b);
    }

    #[test]
    fn many_uids_same_bucket() {
        // Stress: insert a chain, then remove the head, then verify
        // pos_of fixups for the moved tail entry.
        let mut b = SweepBuckets::new(1);
        for i in 0..100u32 {
            b.set_tag(i, i, 0, true);
        }
        check(&b);
        for i in 0..50u32 {
            // Pretend active.swap_remove(i): uid i gone, uid (99-i) moves
            // from old slot to slot i.
            b.on_active_swap_remove(i, Some(99 - i), i);
            check(&b);
        }
        assert_eq!(b.len(0, Allele::A), 50);
    }

    #[test]
    fn flip_after_swap_keeps_consistency() {
        let mut b = SweepBuckets::new(1);
        b.set_tag(7, 0, 0, true);
        b.set_tag(8, 1, 0, true);
        b.on_active_swap_remove(7, Some(8), 0);
        // uid 8 is now at active idx 0, in A bucket.
        b.set_tag(8, 0, 0, false); // flip to ALower; same idx
        assert_eq!(b.len(0, Allele::A), 0);
        assert_eq!(b.len(0, Allele::ALower), 1);
        assert_eq!(b.entries(0, Allele::ALower)[0], (8, 0));
        check(&b);
    }

    #[test]
    fn clear_drops_all() {
        let mut b = SweepBuckets::new(2);
        b.set_tag(7, 0, 0, true);
        b.set_tag(8, 1, 1, false);
        b.clear();
        assert_eq!(b.len(0, Allele::A), 0);
        assert_eq!(b.len(1, Allele::ALower), 0);
        assert!(b.pos_of.is_empty());
        check(&b);
    }

    #[test]
    fn detects_duplicate_uid_in_invariants() {
        // Direct construction to violate invariant — confirms the
        // checker actually catches divergence (used by Stage 2 to
        // catch missed maintenance points).
        let mut b = SweepBuckets::new(1);
        b.buckets[0][0].push((7, 0));
        b.buckets[0][0].push((7, 1)); // same uid twice!
        b.pos_of.insert(7, Pos { pop: 0, allele: Allele::A, pos: 0 });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.assert_invariants();
        }));
        assert!(result.is_err(), "invariant check should panic on duplicate uid");
    }

    #[test]
    fn detects_orphan_pos_of() {
        // pos_of has entries that aren't in any bucket.
        let mut b = SweepBuckets::new(1);
        b.pos_of.insert(99, Pos { pop: 0, allele: Allele::A, pos: 0 });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.assert_invariants();
        }));
        assert!(result.is_err(), "invariant check should panic on orphan pos_of");
    }
}
