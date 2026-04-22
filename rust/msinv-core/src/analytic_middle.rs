/// Analytic "structured Hudson" middle phase.
///
/// PROTOTYPE — feature/structured-analytic-middle. See
/// `analytic_middle.md` for context and scope.
///
/// Between the last demographic event and the oldest `t_inv`, a
/// realistic anopheles Kir/Fol simulation enters a regime with
/// uniform-in-time demography:
///   - one population (effectively — all post-Ej)
///   - zero migration
///   - constant ancestral Ne
///   - one or more inversion barriers still active with fixed p_inv
///
/// The main event loop burns hundreds of thousands of partial-
/// coalescence events here because 98% of class-mismatched coals grow
/// the active-lineage count by 1 (the "remnant ratchet"). Per-event
/// cost scales with `n`, and `n` spikes to ~135k at `t_inv`.
///
/// This module replaces the per-event rebuild of the pair-rate cache
/// with a Kingman-style exponential-clock loop: per-class Hudson rates
/// are summed into three scalars (r_S, r_I, r_P) and the next event is
/// drawn from their total. No recombination, no gene flux, no
/// migration during the analytic middle phase — this is the
/// "option (a)" recombination-free prototype.
///
/// Scope (minimum-viable):
///   - single inversion
///   - single population (or all lineages already in pop 0)
///   - no migration
///   - no pending demographic events or sweeps before horizon
///   - barrier active, `t_inv` > current t
///   - `analytic_middle: bool` flag on `HullSimulator` opt-in
///
/// Out of scope for MVP:
///   - recombination (it's disabled during the middle phase)
///   - gene flux (ditto — set `gene_conversion_rate` equal to what the
///     caller supplies, but we don't fire flux events)
///   - multi-inversion (would need per-inv class tags)
///   - multi-pop (would need per-pop pools + migration)

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::{BranchClass, Karyotype};
use crate::demography::Demography;
use crate::events::apply_coalescence_partial;
use crate::inversion::InversionSpec;
use crate::lineage::{LinUid, Lineage};
use crate::segment::{SegmentArena, SEG_NIL};
use crate::sweep::Sweep;
use crate::tables::TableBuilder;

/// Return true if the analytic middle phase is applicable at current
/// state. If this returns false, fall back to the event-loop path.
pub fn can_use_analytic_middle(
    t_now: f64,
    active: &[Lineage],
    demo: &Demography,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    pending_sweeps: &[Sweep],
    horizon_out: &mut f64,
) -> bool {
    // Single inversion only (MVP).
    if inversions.len() != 1 { return false; }
    if !barrier_active[0] { return false; }
    let inv = &inversions[0];
    if inv.t_inv <= t_now + 1e-9 { return false; }
    // Gene-flux during the middle phase isn't modelled by the analytic
    // path. Skip if the inversion has a non-negligible gamma so the
    // main event loop handles flux correctly. Note: the Rust validator
    // requires gamma > 0 globally, so "gamma tiny" is the best we get.
    if inv.gene_conversion_rate > 1e-8 {
        return false;
    }

    // Single effective population: either n_pops==1, or every active
    // lineage is already in the same pop.
    let effective_pop = if demo.n_pops == 1 {
        0u32
    } else {
        if active.is_empty() { return false; }
        let p0 = active[0].population;
        for lin in active.iter() {
            if lin.population != p0 { return false; }
        }
        p0
    };

    // No migration out of or into the effective pop.
    if demo.n_pops > 1 {
        for dst in 0..demo.n_pops as usize {
            for src in 0..demo.n_pops as usize {
                if demo.migration_matrix[dst][src] != 0.0 {
                    return false;
                }
            }
        }
    }

    // No pending demographic event strictly between t_now and t_inv.
    // Events at t_now are already applied; events at t_inv are the
    // barrier-crossing boundary which terminates the phase cleanly.
    for ev in &demo.events {
        let tev = ev.time();
        if tev > t_now + 1e-9 && tev < inv.t_inv - 1e-9 {
            return false;
        }
    }

    // No pending sweep strictly between t_now and t_inv.
    for sw in pending_sweeps {
        if sw.t_event > t_now + 1e-9 && sw.t_event < inv.t_inv - 1e-9 {
            return false;
        }
    }

    // Ne should be effectively constant through the horizon. Cheapest
    // filter: no active growth rate for the effective pop.
    if (effective_pop as usize) < demo.growth_rates.len()
        && demo.growth_rates[effective_pop as usize] != 0.0
    {
        return false;
    }

    // Must have ≥ 2 lineages to have any coal work to do.
    if active.len() < 2 { return false; }

    *horizon_out = inv.t_inv;
    true
}

/// Classify a lineage's karyotype relative to a single inversion.
/// Returns Some(Karyotype::S|I) if the lineage has any material inside
/// `[bp_left, bp_right)` with the corresponding inv tag set, and None
/// if the lineage has no material inside the inversion (or its inv
/// material is mixed S+I, which can happen only after a flux event —
/// we conservatively treat that as "skip").
fn classify_for_inv(
    lin: &Lineage,
    inv: &InversionSpec,
    arena: &SegmentArena,
) -> Option<Karyotype> {
    let mut seen_s = false;
    let mut seen_i = false;
    let mut cur = lin.head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        let l = seg.left.max(inv.bp_left);
        let r = seg.right.min(inv.bp_right);
        if r > l {
            match seg.branch_class.get_inv(inv.inv_id) {
                Some(Karyotype::S) => seen_s = true,
                Some(Karyotype::I) => seen_i = true,
                None => {}
            }
        }
        cur = seg.next;
    }
    match (seen_s, seen_i) {
        (true, false) => Some(Karyotype::S),
        (false, true) => Some(Karyotype::I),
        _ => None,
    }
}

/// Run the analytic middle phase until `horizon_t` (or lineage count
/// drops to 1). Updates `t` to `horizon_t` on return; mutates
/// `active`, `arena`, `tables`, `next_uid`. Returns nothing — the
/// caller is expected to call `can_use_analytic_middle` first.
#[allow(clippy::too_many_arguments)]
pub fn run_analytic_middle(
    t: &mut f64,
    horizon_t: f64,
    active: &mut Vec<Lineage>,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    rng: &mut Xoshiro256PlusPlus,
    demo: &Demography,
    inversions: &[InversionSpec],
) -> AnalyticMiddleStats {
    debug_assert_eq!(inversions.len(), 1);
    let inv = &inversions[0];
    let pop_for_ne = active.first().map(|l| l.population).unwrap_or(0);
    let ne = demo.size_at(pop_for_ne, *t).max(1e-9);
    let p_inv_val = inv.p_inv_for(pop_for_ne).clamp(1e-12, 1.0 - 1e-12);
    let p_std_val = 1.0 - p_inv_val;
    let s_tag = BranchClass::PANMICTIC.with_inv(inv.inv_id, Karyotype::S);
    let i_tag = BranchClass::PANMICTIC.with_inv(inv.inv_id, Karyotype::I);

    // Per-event rate constants.
    let inv_2ne_pstd = 1.0 / (2.0 * ne * p_std_val);
    let inv_2ne_pinv = 1.0 / (2.0 * ne * p_inv_val);
    let inv_2ne = 1.0 / (2.0 * ne);

    // Build initial class pools by scanning `active`.
    let mut s_pool: Vec<usize> = Vec::with_capacity(active.len());
    let mut i_pool: Vec<usize> = Vec::with_capacity(active.len());
    for (i, lin) in active.iter().enumerate() {
        match classify_for_inv(lin, inv, arena) {
            Some(Karyotype::S) => s_pool.push(i),
            Some(Karyotype::I) => i_pool.push(i),
            None => {}
        }
    }

    let mut stats = AnalyticMiddleStats::default();

    // Safety cap. With the remnant ratchet, the event count can
    // balloon; keep the absolute ceiling generous but finite.
    const MAX_ITERS: u64 = 100_000_000;

    for _iter in 0..MAX_ITERS {
        if *t >= horizon_t { break; }
        if active.len() < 2 { break; }

        let n = active.len() as f64;
        let ks = s_pool.len() as f64;
        let ki = i_pool.len() as f64;

        let rate_s = if ks >= 2.0 { ks * (ks - 1.0) * 0.5 * inv_2ne_pstd } else { 0.0 };
        let rate_i = if ki >= 2.0 { ki * (ki - 1.0) * 0.5 * inv_2ne_pinv } else { 0.0 };
        let rate_p = if n >= 2.0 { n * (n - 1.0) * 0.5 * inv_2ne } else { 0.0 };

        let total = rate_s + rate_i + rate_p;
        if total <= 0.0 {
            *t = horizon_t;
            break;
        }

        let u: f64 = rng.random();
        let dt = -u.ln() / total;
        let t_new = *t + dt;
        if t_new >= horizon_t {
            *t = horizon_t;
            break;
        }
        *t = t_new;

        // Pick which of the three rates fires.
        let pick: f64 = rng.random::<f64>() * total;
        let pre_len = active.len();

        if pick < rate_s {
            // S-class pair.
            if s_pool.len() < 2 { continue; }
            let ii = rng.random_range(0..s_pool.len());
            let mut jj = rng.random_range(0..(s_pool.len() - 1));
            if jj >= ii { jj += 1; }
            let a = s_pool[ii];
            let b = s_pool[jj];
            apply_coalescence_partial(active, a, b, *t, arena, tables,
                                       next_uid, Some(s_tag));
            reclassify_after_coal(
                active, arena, inv, a, b, pre_len,
                &mut s_pool, &mut i_pool);
            stats.s_events += 1;
        } else if pick < rate_s + rate_i {
            if i_pool.len() < 2 { continue; }
            let ii = rng.random_range(0..i_pool.len());
            let mut jj = rng.random_range(0..(i_pool.len() - 1));
            if jj >= ii { jj += 1; }
            let a = i_pool[ii];
            let b = i_pool[jj];
            apply_coalescence_partial(active, a, b, *t, arena, tables,
                                       next_uid, Some(i_tag));
            reclassify_after_coal(
                active, arena, inv, a, b, pre_len,
                &mut s_pool, &mut i_pool);
            stats.i_events += 1;
        } else {
            // Panmictic any-pair.
            if active.len() < 2 { continue; }
            let n_act = active.len();
            let ii = rng.random_range(0..n_act);
            let mut jj = rng.random_range(0..(n_act - 1));
            if jj >= ii { jj += 1; }
            apply_coalescence_partial(active, ii, jj, *t, arena, tables,
                                       next_uid, Some(BranchClass::PANMICTIC));
            reclassify_after_coal(
                active, arena, inv, ii, jj, pre_len,
                &mut s_pool, &mut i_pool);
            stats.p_events += 1;
        }

        stats.peak_n = stats.peak_n.max(active.len());
    }

    // Advance t to the horizon if the loop exited because rates
    // vanished or active collapsed.
    if *t < horizon_t && active.len() < 2 {
        *t = horizon_t;
    }

    stats
}

/// Rebuild the S / I pools after `apply_coalescence_partial` mutated
/// `active`. The event calls `active.swap_remove(hi); active.swap_remove(lo);`
/// and then pushes up to three new lineages (merged + a_rem + b_rem).
///
/// Simplest correct approach for the MVP: drop the `a` and `b` entries
/// from each pool, patch swap_remove index moves for the surviving
/// slots (lineages that were at the very end before swap_remove may
/// have moved into `hi` or `lo`), and then classify + push newly
/// appended lineages from `old_len - 2 ..= new_len - 1`.
fn reclassify_after_coal(
    active: &[Lineage],
    arena: &SegmentArena,
    inv: &InversionSpec,
    a: usize,
    b: usize,
    pre_len: usize,
    s_pool: &mut Vec<usize>,
    i_pool: &mut Vec<usize>,
) {
    // The full rebuild is O(n) which is fine for MVP; the point of the
    // analytic middle is skipping event-list rebuild (O(n²/64) bitmap
    // walks etc), not rebuilding a flat Vec of classified indices.
    s_pool.clear();
    i_pool.clear();
    let _ = (a, b, pre_len); // unused in rebuild variant
    for (i, lin) in active.iter().enumerate() {
        match classify_for_inv(lin, inv, arena) {
            Some(Karyotype::S) => s_pool.push(i),
            Some(Karyotype::I) => i_pool.push(i),
            None => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AnalyticMiddleStats {
    pub s_events: u64,
    pub i_events: u64,
    pub p_events: u64,
    pub peak_n: usize,
}

impl AnalyticMiddleStats {
    pub fn total_events(&self) -> u64 {
        self.s_events + self.i_events + self.p_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::SegmentArena;
    use rand::SeedableRng;

    fn make_pair_lineages(
        arena: &mut SegmentArena,
        tables: &mut TableBuilder,
        inv: &InversionSpec,
        n_s: u32,
        n_i: u32,
    ) -> Vec<Lineage> {
        let mut active = Vec::new();
        let mut uid: LinUid = 0;
        let s_tag = BranchClass::PANMICTIC.with_inv(inv.inv_id, Karyotype::S);
        let i_tag = BranchClass::PANMICTIC.with_inv(inv.inv_id, Karyotype::I);
        // Lay each lineage out with one panmictic segment outside inv on
        // each side plus the inversion-interior tagged segment.
        let seq_len = 1000.0;
        for _ in 0..n_s {
            let node = tables.add_sample(0.0, 0);
            let s0 = arena.alloc(0.0, inv.bp_left, node, BranchClass::PANMICTIC);
            let s1 = arena.alloc(inv.bp_left, inv.bp_right, node, s_tag);
            let s2 = arena.alloc(inv.bp_right, seq_len, node, BranchClass::PANMICTIC);
            arena.get_mut(s0).next = s1;
            arena.get_mut(s1).next = s2;
            active.push(Lineage::new(s0, s2, 0, uid, arena));
            uid += 1;
        }
        for _ in 0..n_i {
            let node = tables.add_sample(0.0, 0);
            let s0 = arena.alloc(0.0, inv.bp_left, node, BranchClass::PANMICTIC);
            let s1 = arena.alloc(inv.bp_left, inv.bp_right, node, i_tag);
            let s2 = arena.alloc(inv.bp_right, seq_len, node, BranchClass::PANMICTIC);
            arena.get_mut(s0).next = s1;
            arena.get_mut(s1).next = s2;
            active.push(Lineage::new(s0, s2, 0, uid, arena));
            uid += 1;
        }
        active
    }

    #[test]
    fn gate_rejects_multi_inv() {
        let inv_a = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let inv_b = InversionSpec {
            bp_left: 300.0, bp_right: 400.0,
            p_inv: vec![0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 1,
        };
        let demo = Demography::single_pop(1000.0);
        let mut horizon = 0.0;
        let rejected = !can_use_analytic_middle(
            0.0, &[], &demo, &[inv_a, inv_b],
            &[true, true], &[], &mut horizon);
        assert!(rejected);
    }

    #[test]
    fn gate_rejects_migration() {
        let inv = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5, 0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let mut demo = Demography::new(vec![1000.0, 1000.0]);
        demo.migration_matrix[0][1] = 1e-4;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 2);
        let active = make_pair_lineages(&mut arena, &mut tables, &inv, 2, 2);
        let mut horizon = 0.0;
        assert!(!can_use_analytic_middle(
            0.0, &active, &demo, &[inv],
            &[true], &[], &mut horizon));
    }

    #[test]
    fn gate_rejects_demo_event_before_horizon() {
        let inv = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let mut demo = Demography::single_pop(1000.0);
        demo.add_event(crate::demography::DemoEvent::EN {
            t: 5_000.0, n: 5000.0,
        });
        let mut horizon = 0.0;
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 1);
        let active = make_pair_lineages(&mut arena, &mut tables, &inv, 2, 2);
        assert!(!can_use_analytic_middle(
            0.0, &active, &demo, &[inv],
            &[true], &[], &mut horizon));
    }

    #[test]
    fn gate_accepts_clean_middle_phase() {
        let inv = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let demo = Demography::single_pop(1000.0);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 1);
        let active = make_pair_lineages(&mut arena, &mut tables, &inv, 4, 4);
        let mut horizon = 0.0;
        assert!(can_use_analytic_middle(
            0.0, &active, &demo, &[inv],
            &[true], &[], &mut horizon));
        assert_eq!(horizon, 10_000.0);
    }

    #[test]
    fn classifier_picks_s_vs_i() {
        let inv = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5], t_inv: 10_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 1);
        let active = make_pair_lineages(&mut arena, &mut tables, &inv, 3, 2);
        let mut n_s = 0;
        let mut n_i = 0;
        for lin in &active {
            match classify_for_inv(lin, &inv, &arena) {
                Some(Karyotype::S) => n_s += 1,
                Some(Karyotype::I) => n_i += 1,
                None => {}
            }
        }
        assert_eq!(n_s, 3);
        assert_eq!(n_i, 2);
    }

    #[test]
    fn run_terminates_and_produces_edges() {
        // Tiny sim: 4 S + 4 I lineages, t_inv = 500, Ne=100.
        // Analytic middle should drain most within-class pairs.
        let inv = InversionSpec {
            bp_left: 100.0, bp_right: 200.0,
            p_inv: vec![0.5], t_inv: 500.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let demo = Demography::single_pop(100.0);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 1);
        let mut active = make_pair_lineages(
            &mut arena, &mut tables, &inv, 4, 4);
        let mut next_uid: LinUid = 100;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut t = 0.0;
        let pre_edges = tables.num_edges();
        let stats = run_analytic_middle(
            &mut t, 500.0,
            &mut active, &mut arena, &mut tables, &mut next_uid,
            &mut rng, &demo, &[inv]);
        assert!(t <= 500.0 + 1e-9);
        assert!(stats.total_events() >= 1,
            "expected at least one coalescence: {stats:?}");
        assert!(tables.num_edges() > pre_edges);
    }
}
