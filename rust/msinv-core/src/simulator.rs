/// HullSimulator: the main event loop.
///
/// Phase C: inversions (class barriers, per-pair coal rates, gene flux,
/// barrier crossing) on top of the Phase B panmictic loop.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::{BranchClass, Karyotype};
use crate::demography::{DemoEvent, Demography};
use crate::events::{apply_coalescence, apply_coalescence_partial, apply_recombination};
use crate::inversion::InversionSpec;
use crate::lineage::{LinUid, Lineage};
use crate::phi::{phi, phi_integral};
use crate::rate_index::RateCache;
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};
use crate::sweep::Sweep;
use crate::tables::TableBuilder;

// ---------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------
pub struct SimResult {
    pub tables: TableBuilder,
}

// ---------------------------------------------------------------
// Sample configuration entry
// ---------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct SampleEntry {
    /// Per-inversion karyotype: None = panmictic, Some(S/I) per inv.
    pub karyotypes: Vec<Option<Karyotype>>,
    pub population: u32,
    pub count: u32,
}

// ---------------------------------------------------------------
// Event tag for the competing-rates dispatcher
// ---------------------------------------------------------------
enum Event {
    CoalPair { i: usize, j: usize, class: BranchClass },
    CoalPanmicticPop { pop: u32 },
    Recombination,
    Flux { lineage_idx: usize, inv_idx: usize },
    Migration { lineage_idx: usize, dst_pop: u32 },
}

// ---------------------------------------------------------------
// HullSimulator
// ---------------------------------------------------------------
pub struct HullSimulator {
    pub samples: Vec<SampleEntry>,
    pub demography: Demography,
    pub sequence_length: f64,
    pub recombination_rate: f64,
    pub inversions: Vec<InversionSpec>,
    pub sweeps: Vec<Sweep>,
    pub seed: u64,
}

impl HullSimulator {
    /// Convenience constructor for the simple n_std/n_inv case.
    pub fn simple(
        n_std: u32, n_inv: u32,
        population_size: f64,
        sequence_length: f64,
        recombination_rate: f64,
        inversions: Vec<InversionSpec>,
        seed: u64,
    ) -> Self {
        let n_inv_specs = inversions.len();
        let mut samples = Vec::new();
        if n_std > 0 {
            samples.push(SampleEntry {
                karyotypes: vec![Some(Karyotype::S); n_inv_specs],
                population: 0,
                count: n_std,
            });
        }
        if n_inv > 0 {
            samples.push(SampleEntry {
                karyotypes: vec![Some(Karyotype::I); n_inv_specs],
                population: 0,
                count: n_inv,
            });
        }
        Self {
            samples,
            demography: Demography::single_pop(population_size),
            sequence_length,
            recombination_rate,
            inversions,
            sweeps: vec![],
            seed,
        }
    }

    /// Panmictic-only constructor (back-compat with Phase B).
    pub fn panmictic(
        n_samples: u32,
        population_size: f64,
        sequence_length: f64,
        recombination_rate: f64,
        seed: u64,
    ) -> Self {
        Self {
            samples: vec![SampleEntry {
                karyotypes: vec![],
                population: 0,
                count: n_samples,
            }],
            demography: Demography::single_pop(population_size),
            sequence_length,
            recombination_rate,
            inversions: vec![],
            sweeps: vec![],
            seed,
        }
    }

    pub fn simulate(&self) -> SimResult {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(
            self.sequence_length, self.demography.n_pops);
        let mut next_uid: LinUid = 0;
        let mut demo = self.demography.clone();

        let mut active = self.make_initial_lineages(
            &mut arena, &mut tables, &mut next_uid);

        self.run_loop(&mut active, &mut arena, &mut tables,
                       &mut next_uid, &mut rng, &mut demo);

        SimResult { tables }
    }

    // ---------------------------------------------------------------
    // Initial lineages
    // ---------------------------------------------------------------
    fn make_initial_lineages(
        &self,
        arena: &mut SegmentArena,
        tables: &mut TableBuilder,
        next_uid: &mut LinUid,
    ) -> Vec<Lineage> {
        let mut active = Vec::new();
        let sorted_invs = &self.inversions;

        for entry in &self.samples {
            for _ in 0..entry.count {
                let node_id = tables.add_sample(0.0, entry.population as i32);
                let (head, tail) = make_initial_segments(
                    self.sequence_length, node_id, sorted_invs,
                    &entry.karyotypes, arena);
                let uid = *next_uid;
                *next_uid += 1;
                active.push(Lineage::new(head, tail, entry.population, uid, arena));
            }
        }
        active
    }

    // ---------------------------------------------------------------
    // Main event loop
    // ---------------------------------------------------------------
    fn run_loop(
        &self,
        active: &mut Vec<Lineage>,
        arena: &mut SegmentArena,
        tables: &mut TableBuilder,
        next_uid: &mut LinUid,
        rng: &mut Xoshiro256PlusPlus,
        demo: &mut Demography,
    ) {
        let mut t: f64 = 0.0;

        // Track which inversions' barriers are still active.
        let mut barrier_active: Vec<bool> = self.inversions.iter()
            .map(|_| true).collect();

        // Pending sweeps, sorted by t_event (earliest first).
        let mut pending_sweeps: Vec<Sweep> = self.sweeps.clone();
        pending_sweeps.sort_by(|a, b| a.t_event.partial_cmp(&b.t_event).unwrap());

        // Running totals for O(1) recombination rate (Phase A).
        let mut total_material: f64 = active.iter()
            .map(|l| l.cached_len).sum();
        let mut total_recomb_rate: f64 = total_material * self.recombination_rate;

        // Phase D: incremental pair rate cache.
        let max_lins = (active.len() * 4).max(64);  // generous headroom
        let mut rate_cache = RateCache::new(max_lins);
        rate_cache.rebuild(&active, arena);
        let mut cache_dirty = false;  // force full rebuild when true

        for _ in 0..10_000_000u64 {
            let n = active.len();
            if n <= 1 {
                if n == 0 || active[0].total_length(arena)
                    >= self.sequence_length - 1e-9
                {
                    return;
                }
                return;
            }

            // Check for barrier crossings.
            let mut any_barrier = false;
            let mut earliest_barrier = f64::INFINITY;
            for (k, inv) in self.inversions.iter().enumerate() {
                if barrier_active[k] {
                    any_barrier = true;
                    earliest_barrier = earliest_barrier.min(inv.t_inv);
                }
            }

            // Next demographic event boundary.
            let t_demo = demo.next_event_time(t);

            // --- Compute all event rates ---
            let mut all_events: Vec<(f64, Event)> = Vec::new();

            // Coalescence: use cached pair rates when inversions active.
            if cache_dirty {
                rate_cache.rebuild(active, arena);
                cache_dirty = false;
            }
            if any_barrier {
                // Structured: emit events from the pair rate cache.
                emit_coal_events_from_cache(
                    &rate_cache, active, &*demo, t,
                    &self.inversions, &barrier_active,
                    &mut all_events);
            } else {
                // Post-barrier panmictic: Hudson per-pop buckets.
                compute_coal_events(
                    active, arena, demo, t, &self.inversions, &barrier_active,
                    &mut all_events);
            }

            // Recombination (O(1) from running total).
            if total_recomb_rate > 0.0 {
                all_events.push((total_recomb_rate, Event::Recombination));
            }

            // Gene flux.
            if any_barrier {
                self.compute_flux_rates(
                    active, arena, &barrier_active, &mut all_events);
            }
            // Migration.
            for (rate, lin_idx, dst) in demo.migration_rates(active) {
                all_events.push((rate, Event::Migration {
                    lineage_idx: lin_idx, dst_pop: dst,
                }));
            }

            let total_rate: f64 = all_events.iter().map(|(r, _)| *r).sum();

            // Next sweep boundary.
            let t_sweep = pending_sweeps.first()
                .map(|s| s.t_event).unwrap_or(f64::INFINITY);

            // Next deterministic boundary.
            let next_boundary = earliest_barrier.min(t_demo).min(t_sweep);

            if total_rate <= 0.0 {
                if next_boundary < f64::INFINITY {
                    t = next_boundary;
                    self.cross_barriers(active, arena, &mut barrier_active, t);
                    demo.apply_events_at(t, active);
                    if !pending_sweeps.is_empty()
                        && (pending_sweeps[0].t_event - t).abs() < 1e-9
                    {
                        let sweep = pending_sweeps.remove(0);
                        apply_sweep(active, &sweep, t, arena, tables,
                                     next_uid, self.sequence_length);
                    }
                    // Sweeps/barriers may change lineage structure.
                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                    total_recomb_rate = total_material * self.recombination_rate;
                    cache_dirty = true;
                    continue;
                }
                return;
            }

            // Draw waiting time.
            let u: f64 = rng.random();
            let dt = -u.ln() / total_rate;
            let t_event = t + dt;

            // Check if a deterministic boundary happens first.
            if next_boundary <= t_event {
                t = next_boundary;
                self.cross_barriers(active, arena, &mut barrier_active, t);
                demo.apply_events_at(t, active);
                if !pending_sweeps.is_empty()
                    && (pending_sweeps[0].t_event - t).abs() < 1e-9
                {
                    let sweep = pending_sweeps.remove(0);
                    apply_sweep(active, &sweep, t, arena, tables,
                                 next_uid, self.sequence_length);
                }
                // Sweeps/barriers may change lineage structure.
                total_material = active.iter()
                    .map(|l| l.cached_len).sum();
                total_recomb_rate = total_material * self.recombination_rate;
                cache_dirty = true;
                continue;
            }
            t = t_event;

            // Pick which event fires.
            let u2: f64 = rng.random::<f64>() * total_rate;
            let mut cum = 0.0;
            let mut chosen_event = None;
            for (rate, event) in &all_events {
                cum += rate;
                if u2 < cum {
                    chosen_event = Some(event);
                    break;
                }
            }
            let chosen_event = match chosen_event {
                Some(e) => e,
                None => continue,
            };

            match chosen_event {
                Event::CoalPair { i, j, class } => {
                    let (i, j) = (*i, *j);
                    let cls = *class;
                    let n_before = active.len();

                    // Remove cache entries for consumed lineages.
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    rate_cache.remove_lineage(hi);
                    rate_cache.remove_lineage(lo);

                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        Some(cls));

                    // Handle swap_remove index changes in cache.
                    // swap_remove(hi): last element moved to hi
                    let old_last_hi = n_before - 1;
                    if hi != old_last_hi {
                        rate_cache.swap_update(hi, old_last_hi);
                    } else {
                        rate_cache.swap_update(hi, hi);
                    }
                    // swap_remove(lo): (n_before-2)th element moved to lo
                    let old_last_lo = n_before - 2;
                    if lo < old_last_lo {
                        rate_cache.swap_update(lo, old_last_lo);
                    } else {
                        rate_cache.swap_update(lo, lo);
                    }

                    // Recompute pairs for newly pushed lineages.
                    let n_after = active.len();
                    for new_idx in (n_before - 2)..n_after {
                        if new_idx < n_after {
                            rate_cache.recompute_for(new_idx, active, arena);
                        }
                    }

                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                }
                Event::CoalPanmicticPop { pop } => {
                    let pop = *pop;
                    let pool: Vec<usize> = active.iter().enumerate()
                        .filter(|(_, l)| l.population == pop)
                        .map(|(i, _)| i).collect();
                    if pool.len() >= 2 {
                        let ii = rng.random_range(0..pool.len());
                        let mut jj = rng.random_range(0..pool.len() - 1);
                        if jj >= ii { jj += 1; }
                        apply_coalescence(
                            active, pool[ii], pool[jj], t, arena,
                            tables, next_uid);
                        // Recompute after merge.
                        total_material = active.iter()
                            .map(|l| l.cached_len).sum();
                    }
                }
                Event::Recombination => {
                    let u_lin: f64 = rng.random::<f64>();
                    let target = u_lin * total_material;
                    let mut cum_len = 0.0;
                    let mut chosen_idx = 0;
                    for (idx, lin) in active.iter().enumerate() {
                        cum_len += lin.cached_len;
                        if cum_len > target {
                            chosen_idx = idx;
                            break;
                        }
                    }
                    let lin_len = active[chosen_idx].cached_len;
                    let x_offset: f64 = rng.random::<f64>() * lin_len;
                    let x = find_position(active, chosen_idx, x_offset,
                                           arena, self.sequence_length);
                    let n_before = active.len();
                    apply_recombination(active, chosen_idx, x, arena,
                                         next_uid);
                    // Recomb: lineage at chosen_idx is replaced (swap_remove
                    // + push of left and right). Recompute pairs for the
                    // modified indices.
                    if any_barrier {
                        // The original lineage was swap_removed; last moved
                        // to chosen_idx. Two new lineages pushed at end.
                        let old_last = n_before - 1;
                        rate_cache.remove_lineage(chosen_idx);
                        if chosen_idx != old_last {
                            rate_cache.swap_update(chosen_idx, old_last);
                        } else {
                            rate_cache.swap_update(chosen_idx, chosen_idx);
                        }
                        // Recompute for the two new lineages.
                        let n_after = active.len();
                        for new_idx in (n_before - 1)..n_after {
                            rate_cache.recompute_for(new_idx, active, arena);
                        }
                    }
                    // Recombination preserves total material.
                }
                Event::Flux { lineage_idx, inv_idx } => {
                    let (li, ii) = (*lineage_idx, *inv_idx);
                    let inv = &self.inversions[ii];
                    if let Some(x_event) = self.sample_flux_position(
                        active, li, inv, arena, rng)
                    {
                        let (tl, tr) = self.draw_tract(x_event, inv, rng);
                        if tr > tl {
                            let n_before = active.len();
                            apply_gene_flux(active, li, tl, tr, inv,
                                             arena, next_uid);
                            // Flux: same structure as recomb (swap_remove + push).
                            if any_barrier {
                                let old_last = n_before - 1;
                                rate_cache.remove_lineage(li);
                                if li != old_last {
                                    rate_cache.swap_update(li, old_last);
                                } else {
                                    rate_cache.swap_update(li, li);
                                }
                                let n_after = active.len();
                                for new_idx in (n_before - 1)..n_after {
                                    rate_cache.recompute_for(
                                        new_idx, active, arena);
                                }
                            }
                        }
                    }
                }
                Event::Migration { lineage_idx, dst_pop } => {
                    let idx = *lineage_idx;
                    if idx < active.len() {
                        active[idx].population = *dst_pop;
                    }
                    // Migration doesn't change material.
                }
            }

            // GC: remove lineages that are the sole carrier at every
            // position they cover. These can never produce more edges
            // (no other lineage to coalesce with at those positions).
            let n_before_gc = active.len();
            gc_sole_lineages(active, arena);
            if active.len() != n_before_gc {
                // GC removed lineages — recompute total and cache.
                total_material = active.iter()
                    .map(|l| l.cached_len).sum();
                cache_dirty = true;
            }

            // Keep recomb rate in sync.
            total_recomb_rate = total_material * self.recombination_rate;
        }
    }

    // ---------------------------------------------------------------
    // Per-pair, per-class coalescence rates
    // ---------------------------------------------------------------
    fn compute_coal_rates_structured(
        &self,
        active: &[Lineage],
        arena: &SegmentArena,
        demo: &Demography,
        t: f64,
        barrier_active: &[bool],
        events: &mut Vec<(f64, Event)>,
    ) {
        let n = active.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if active[i].population != active[j].population {
                    continue;
                }
                let ne_pop = demo.size_at(active[i].population, t).max(1e-9);
                let overlaps = overlap_by_class(
                    active[i].head, active[j].head, arena);
                for (cls, ov_len) in &overlaps {
                    if *ov_len <= 0.0 { continue; }
                    let p_class = self.p_class_for(*cls, t, barrier_active);
                    if p_class <= 0.0 { continue; }
                    let rate = 1.0 / (2.0 * ne_pop * p_class);
                    events.push((rate, Event::CoalPair {
                        i, j, class: *cls,
                    }));
                }
            }
        }
    }

    fn compute_coal_rates_panmictic(
        &self,
        active: &[Lineage],
        _arena: &SegmentArena,
        demo: &Demography,
        t: f64,
        events: &mut Vec<(f64, Event)>,
    ) {
        // Bucket lineages by population.
        let mut buckets: Vec<(u32, usize)> = Vec::new(); // (pop, count)
        for lin in active.iter() {
            if let Some(entry) = buckets.iter_mut().find(|(p, _)| *p == lin.population) {
                entry.1 += 1;
            } else {
                buckets.push((lin.population, 1));
            }
        }
        for (pop, k) in &buckets {
            if *k < 2 { continue; }
            let ne = demo.size_at(*pop, t).max(1e-9);
            let kf = *k as f64;
            let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
            if rate > 0.0 {
                events.push((rate, Event::CoalPanmicticPop { pop: *pop }));
            }
        }
    }

    /// Effective sub-population frequency for a given BranchClass tag.
    fn p_class_for(&self, cls: BranchClass, t: f64, barrier_active: &[bool]) -> f64 {
        if cls.is_panmictic() {
            return 1.0;
        }
        let mut p = 1.0;
        for (k, inv) in self.inversions.iter().enumerate() {
            if !barrier_active[k] || t >= inv.t_inv {
                continue;
            }
            match cls.get_inv(inv.inv_id) {
                Some(Karyotype::S) => p *= inv.p_std(),
                Some(Karyotype::I) => p *= inv.p_inv,
                None => {}
            }
        }
        p
    }

    // ---------------------------------------------------------------
    // Gene flux rates
    // ---------------------------------------------------------------
    fn compute_flux_rates(
        &self,
        active: &[Lineage],
        arena: &SegmentArena,
        barrier_active: &[bool],
        events: &mut Vec<(f64, Event)>,
    ) {
        for (inv_idx, inv) in self.inversions.iter().enumerate() {
            if !barrier_active[inv_idx] { continue; }
            if inv.gene_conversion_rate <= 0.0 { continue; }
            let p_std = inv.p_std();
            for (lin_idx, lin) in active.iter().enumerate() {
                // Determine lineage's class for this inversion.
                let kary = lineage_class_for_inv(lin, inv, arena);
                let p_other = match kary {
                    Some(Karyotype::S) => inv.p_inv,
                    Some(Karyotype::I) => p_std,
                    None => continue,
                };
                if p_other <= 0.0 { continue; }
                let weight = flux_lineage_weight(lin, inv, arena);
                if weight <= 0.0 { continue; }
                let rate = inv.gene_conversion_rate * p_other * weight;
                if rate > 0.0 {
                    events.push((rate, Event::Flux {
                        lineage_idx: lin_idx,
                        inv_idx,
                    }));
                }
            }
        }
    }

    fn sample_flux_position(
        &self,
        active: &[Lineage],
        lin_idx: usize,
        inv: &InversionSpec,
        arena: &SegmentArena,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Option<f64> {
        let inv_len = inv.length();
        let w = inv.flux_window;
        let mut intervals: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
        let mut cum = 0.0;
        let mut cur = active[lin_idx].head;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            let l = seg.left.max(inv.bp_left);
            let r = seg.right.min(inv.bp_right);
            if r > l {
                let a = (l - inv.bp_left) / inv_len;
                let b = (r - inv.bp_left) / inv_len;
                let weight = phi_integral(a, b, w) * inv_len;
                intervals.push((l, r, a, b, weight));
                cum += weight;
            }
            cur = seg.next;
        }
        if cum <= 0.0 { return None; }

        // Pick interval by weight.
        let u = rng.random::<f64>() * cum;
        let mut running = 0.0;
        let mut chosen = intervals.last().unwrap();
        for entry in &intervals {
            running += entry.4;
            if u < running {
                chosen = entry;
                break;
            }
        }
        let (_l, _r, a, b, _w) = *chosen;

        // Rejection sample within interval using phi density.
        let phi_max = if w < 1.0 { w / (1.0 - w) } else { 1.0 };
        for _ in 0..1000 {
            let xx: f64 = rng.random::<f64>() * (b - a) + a;
            if rng.random::<f64>() * phi_max < phi(xx, w) {
                return Some(inv.bp_left + xx * inv_len);
            }
        }
        // Fallback: uniform in chosen segment.
        Some(rng.random::<f64>() * (_r - _l) + _l)
    }

    fn draw_tract(
        &self,
        x_event: f64,
        inv: &InversionSpec,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (f64, f64) {
        let inv_len = inv.length();
        let w_g = inv.flux_window * inv_len;
        let x_rel = x_event - inv.bp_left;
        let b1_lo = (x_rel - w_g).max(0.0);
        let b1_hi = (x_rel).min(inv_len - w_g);
        let b1 = if b1_hi <= b1_lo {
            (x_rel - w_g / 2.0).clamp(0.0, inv_len - w_g)
        } else {
            rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
        };
        let tl = (inv.bp_left + b1).max(inv.bp_left);
        let tr = (tl + w_g).min(inv.bp_right);
        (tl, tr)
    }

    // ---------------------------------------------------------------
    // Barrier crossing
    // ---------------------------------------------------------------
    fn cross_barriers(
        &self,
        active: &mut [Lineage],
        arena: &mut SegmentArena,
        barrier_active: &mut [bool],
        t: f64,
    ) {
        for (k, inv) in self.inversions.iter().enumerate() {
            if barrier_active[k] && t >= inv.t_inv {
                barrier_active[k] = false;
                // Flip all segments' class tags for this inversion to panmictic.
                for lin in active.iter() {
                    let mut cur = lin.head;
                    while cur != SEG_NIL {
                        let seg = arena.get_mut(cur);
                        seg.branch_class = seg.branch_class.clear_inv(inv.inv_id);
                        cur = seg.next;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------

/// Remove lineages that are the sole carrier at every position they
/// cover — these can't produce more edges under SMC'.
fn gc_sole_lineages(active: &mut Vec<Lineage>, arena: &SegmentArena) {
    if active.len() <= 1 {
        return;
    }
    // For each lineage, check if any other lineage overlaps it at
    // any position. If not, remove it.
    let mut to_remove: Vec<usize> = Vec::new();
    'outer: for (i, lin_i) in active.iter().enumerate() {
        let mut cur = lin_i.head;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            // Check if any other lineage has material overlapping [seg.left, seg.right).
            for (j, lin_j) in active.iter().enumerate() {
                if j == i { continue; }
                // Quick check: does lin_j have any segment overlapping seg?
                let mut cur_j = lin_j.head;
                while cur_j != SEG_NIL {
                    let sj = arena.get(cur_j);
                    if sj.right > seg.left && sj.left < seg.right {
                        // Overlap found — this lineage still matters.
                        continue 'outer;
                    }
                    if sj.left >= seg.right {
                        break; // segments sorted, no more overlap possible
                    }
                    cur_j = sj.next;
                }
            }
            cur = arena.get(cur).next;
        }
        // No other lineage overlaps lin_i at any position.
        to_remove.push(i);
    }
    // Remove in reverse order to preserve indices.
    for &idx in to_remove.iter().rev() {
        active.swap_remove(idx);
    }
}

/// Emit coalescence events from the RateCache. O(n^2) iteration of
/// cached pairs, but the CACHE itself is maintained incrementally —
/// only O(n) pairs are recomputed after each event.
fn emit_coal_events_from_cache(
    cache: &RateCache,
    active: &[Lineage],
    demo: &Demography,
    t: f64,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    events: &mut Vec<(f64, Event)>,
) {
    for (i, j, overlaps) in cache.iter_pairs() {
        let ne = demo.size_at(active[i].population, t).max(1e-9);
        for (cls, _ov_len) in overlaps {
            let p_class = p_class_for_tag(*cls, inversions, barrier_active, t);
            if p_class <= 0.0 { continue; }
            let rate = 1.0 / (2.0 * ne * p_class);
            events.push((rate, Event::CoalPair { i, j, class: *cls }));
        }
    }
}

/// Compute coal events list. Post-t_inv: Hudson per-pop buckets, O(n).
/// Active inversions: per-pair overlap-by-class, O(n^2).
fn compute_coal_events(
    active: &[Lineage],
    arena: &SegmentArena,
    demo: &Demography,
    t: f64,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    events: &mut Vec<(f64, Event)>,
) {
    let any_inv_active = barrier_active.iter().any(|&b| b);

    if !any_inv_active {
        // Hudson per-pop buckets.
        let mut buckets: Vec<(u32, Vec<usize>)> = Vec::new();
        for (i, lin) in active.iter().enumerate() {
            if let Some(e) = buckets.iter_mut().find(|(p, _)| *p == lin.population) {
                e.1.push(i);
            } else {
                buckets.push((lin.population, vec![i]));
            }
        }
        for (pop, indices) in &buckets {
            let k = indices.len();
            if k < 2 { continue; }
            let ne = demo.size_at(*pop, t).max(1e-9);
            let kf = k as f64;
            let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
            events.push((rate, Event::CoalPanmicticPop { pop: *pop }));
        }
        return;
    }

    // Structured: per-pair overlap-by-class.
    let n = active.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if active[i].population != active[j].population { continue; }
            let ne = demo.size_at(active[i].population, t).max(1e-9);
            let overlaps = overlap_by_class(active[i].head, active[j].head, arena);
            for (cls, _ov_len) in &overlaps {
                let p_class = p_class_for_tag(*cls, inversions, barrier_active, t);
                if p_class <= 0.0 { continue; }
                let rate = 1.0 / (2.0 * ne * p_class);
                events.push((rate, Event::CoalPair { i, j, class: *cls }));
            }
        }
    }
}

/// Effective sub-population frequency for a BranchClass tag.
fn p_class_for_tag(cls: BranchClass, inversions: &[InversionSpec],
                    barrier_active: &[bool], t: f64) -> f64 {
    if cls.is_panmictic() {
        return 1.0;
    }
    let mut p = 1.0;
    for (k, inv) in inversions.iter().enumerate() {
        if !barrier_active[k] || t >= inv.t_inv { continue; }
        match cls.get_inv(inv.inv_id) {
            Some(Karyotype::S) => p *= inv.p_std(),
            Some(Karyotype::I) => p *= inv.p_inv,
            None => {}
        }
    }
    p
}

/// When a coalescence event fires, pick a (class, pop) bucket
/// weighted by rate, then pick two lineages from that bucket.
fn sample_and_coalesce(
    active: &mut Vec<Lineage>,
    arena: &mut SegmentArena,
    demo: &Demography,
    t: f64,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    rng: &mut Xoshiro256PlusPlus,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
) {
    let any_inv_active = barrier_active.iter().any(|&b| b);

    // Build (class, pop, rate, indices) buckets.
    struct Bucket {
        rate: f64,
        indices: Vec<usize>,
        allowed_class: Option<BranchClass>, // None = panmictic
    }
    let mut buckets: Vec<Bucket> = Vec::new();

    if !any_inv_active {
        // Panmictic: one bucket per pop.
        let mut pop_map: Vec<(u32, Vec<usize>)> = Vec::new();
        for (i, lin) in active.iter().enumerate() {
            if let Some(e) = pop_map.iter_mut().find(|(p, _)| *p == lin.population) {
                e.1.push(i);
            } else {
                pop_map.push((lin.population, vec![i]));
            }
        }
        for (pop, indices) in pop_map {
            if indices.len() < 2 { continue; }
            let ne = demo.size_at(pop, t).max(1e-9);
            let kf = indices.len() as f64;
            let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
            buckets.push(Bucket { rate, indices, allowed_class: None });
        }
    } else {
        // Structured: per-pair overlap-by-class. Each pair with
        // overlap at a matching class gets its own bucket entry.
        let n = active.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if active[i].population != active[j].population { continue; }
                let ne = demo.size_at(active[i].population, t).max(1e-9);
                let overlaps = overlap_by_class(
                    active[i].head, active[j].head, arena);
                for (cls, _ov_len) in &overlaps {
                    let p_class = p_class_for_tag(
                        *cls, inversions, barrier_active, t);
                    if p_class <= 0.0 { continue; }
                    let rate = 1.0 / (2.0 * ne * p_class);
                    buckets.push(Bucket {
                        rate, indices: vec![i, j],
                        allowed_class: Some(*cls),
                    });
                }
            }
        }
    }

    if buckets.is_empty() { return; }
    let total: f64 = buckets.iter().map(|b| b.rate).sum();
    if total <= 0.0 { return; }

    let u = rng.random::<f64>() * total;
    let mut cum = 0.0;
    let mut chosen = 0;
    for (i, b) in buckets.iter().enumerate() {
        cum += b.rate;
        if u < cum { chosen = i; break; }
    }

    let bucket = &buckets[chosen];
    let indices = &bucket.indices;
    let allowed = bucket.allowed_class;
    if indices.len() == 2 {
        apply_coalescence_partial(active, indices[0], indices[1], t,
                                   arena, tables, next_uid, allowed);
    } else {
        let ii = rng.random_range(0..indices.len());
        let mut jj = rng.random_range(0..indices.len() - 1);
        if jj >= ii { jj += 1; }
        apply_coalescence_partial(active, indices[ii], indices[jj], t,
                                   arena, tables, next_uid, allowed);
    }
}

/// Build initial segment chain for one sample lineage.
fn make_initial_segments(
    seq_len: f64,
    node_id: i32,
    inversions: &[InversionSpec],
    karyotypes: &[Option<Karyotype>],
    arena: &mut SegmentArena,
) -> (SegIdx, SegIdx) {
    if inversions.is_empty() {
        let idx = arena.alloc(0.0, seq_len, node_id, BranchClass::PANMICTIC);
        return (idx, idx);
    }

    // Collect all breakpoints.
    let mut bps = vec![0.0, seq_len];
    for inv in inversions {
        bps.push(inv.bp_left);
        bps.push(inv.bp_right);
    }
    bps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bps.dedup();

    let mut head = SEG_NIL;
    let mut tail = SEG_NIL;
    for window in bps.windows(2) {
        let (a, b) = (window[0], window[1]);
        if b <= a || a >= seq_len || b <= 0.0 { continue; }
        let a = a.max(0.0);
        let b = b.min(seq_len);

        // Determine class at the midpoint of this interval.
        let mut cls = BranchClass::PANMICTIC;
        for (k, inv) in inversions.iter().enumerate() {
            if inv.bp_left <= a && b <= inv.bp_right {
                let kary = karyotypes.get(k).copied().flatten();
                if let Some(kary) = kary {
                    cls = cls.with_inv(inv.inv_id, kary);
                }
            }
        }

        let idx = arena.alloc(a, b, node_id, cls);
        if head == SEG_NIL {
            head = idx;
        } else {
            arena.get_mut(tail).next = idx;
        }
        tail = idx;
    }
    (head, tail)
}

/// Compute overlap between two segment chains, bucketed by matching
/// BranchClass. Only counts positions where BOTH have material AND
/// their classes agree.
fn overlap_by_class(
    head_a: SegIdx, head_b: SegIdx, arena: &SegmentArena,
) -> Vec<(BranchClass, f64)> {
    let mut result: Vec<(BranchClass, f64)> = Vec::new();
    let mut sa = head_a;
    let mut sb = head_b;
    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let b = arena.get(sb);
        if a.right <= b.left {
            sa = a.next;
            continue;
        }
        if b.right <= a.left {
            sb = b.next;
            continue;
        }
        let l = a.left.max(b.left);
        let r = a.right.min(b.right);
        if r > l && a.branch_class == b.branch_class {
            let cls = a.branch_class;
            // Accumulate into result.
            if let Some(entry) = result.iter_mut().find(|(c, _)| *c == cls) {
                entry.1 += r - l;
            } else {
                result.push((cls, r - l));
            }
        }
        if a.right < b.right {
            sa = a.next;
        } else {
            sb = b.next;
        }
    }
    result
}

/// Determine a lineage's karyotype for one inversion.
fn lineage_class_for_inv(
    lin: &Lineage, inv: &InversionSpec, arena: &SegmentArena,
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
    if seen_s && !seen_i { Some(Karyotype::S) }
    else if seen_i && !seen_s { Some(Karyotype::I) }
    else { None }
}

/// Per-lineage flux weight: integral of phi(x) over in-inv material.
fn flux_lineage_weight(
    lin: &Lineage, inv: &InversionSpec, arena: &SegmentArena,
) -> f64 {
    let inv_len = inv.length();
    if inv_len <= 0.0 { return 0.0; }
    let w = inv.flux_window;
    let mut weight = 0.0;
    let mut cur = lin.head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        let l = seg.left.max(inv.bp_left);
        let r = seg.right.min(inv.bp_right);
        if r > l {
            let a = (l - inv.bp_left) / inv_len;
            let b = (r - inv.bp_left) / inv_len;
            weight += phi_integral(a, b, w) * inv_len;
        }
        cur = seg.next;
    }
    weight
}

/// Apply a gene-flux event: split tract out of lineage, flip class
/// for the specified inversion.
fn apply_gene_flux(
    active: &mut Vec<Lineage>,
    lin_idx: usize,
    tract_left: f64,
    tract_right: f64,
    inv: &InversionSpec,
    arena: &mut SegmentArena,
    next_uid: &mut LinUid,
) {
    let uid = *next_uid;
    *next_uid += 1;
    // Split at tract_left → (outside_left, rest)
    let rest = active[lin_idx].split_at(tract_left, arena, uid);
    if rest.is_none() {
        return; // no material at or after tract_left
    }
    let mut rest = rest.unwrap();

    let uid2 = *next_uid;
    *next_uid += 1;
    // Split rest at tract_right → (tract, outside_right)
    let outside_right = rest.split_at(tract_right, arena, uid2);

    // Flip class tags on the tract (rest is now the tract).
    let mut cur = rest.head;
    while cur != SEG_NIL {
        let seg = arena.get_mut(cur);
        seg.branch_class = seg.branch_class.flip_inv(inv.inv_id);
        cur = seg.next;
    }

    // Add flipped tract back to active.
    active.push(rest);
    // Add outside_right if non-empty.
    if let Some(right_lin) = outside_right {
        active.push(right_lin);
    }
}

/// Convert an offset within a lineage's ancestral material to a
/// genomic position.
fn find_position(
    active: &[Lineage], idx: usize, offset: f64, arena: &SegmentArena,
    seq_len: f64,
) -> f64 {
    let mut remaining = offset;
    let mut cur = active[idx].head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        let seg_len = seg.right - seg.left;
        if remaining < seg_len {
            return seg.left + remaining;
        }
        remaining -= seg_len;
        cur = seg.next;
    }
    seq_len
}

/// Force-coalesce all qualifying lineages at a sweep event.
///
/// Qualifying: has material at x_sel, class matches sweep.target,
/// population matches sweep.population. The sweep window
/// [x_sel - w, x_sel + w] is split out of each qualifying lineage,
/// then all windows are sequentially coalesced.
fn apply_sweep(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    seq_len: f64,
) {
    let x_lo = if sweep.sweep_window > 0.0 {
        sweep.x_sel - sweep.sweep_window
    } else {
        sweep.x_sel
    };
    let x_hi = if sweep.sweep_window > 0.0 {
        sweep.x_sel + sweep.sweep_window
    } else {
        sweep.x_sel + (seq_len * 1e-12).max(1e-9)
    };

    // Identify qualifying lineage indices.
    let mut qualifying: Vec<usize> = Vec::new();
    for (i, lin) in active.iter().enumerate() {
        if let Some(pop) = sweep.population {
            if lin.population != pop { continue; }
        }
        if let Some(cls) = lin.class_at(sweep.x_sel, arena) {
            if sweep.class_matches(cls) {
                qualifying.push(i);
            }
        }
    }
    if qualifying.len() < 2 { return; }

    // Split each qualifying lineage at x_lo and x_hi so the sweep
    // window is isolated. Collect window lineage UIDs (not indices —
    // indices go stale after apply_coalescence's swap_remove).
    let mut window_uids: Vec<LinUid> = Vec::new();
    qualifying.sort_unstable();
    for &orig_idx in qualifying.iter().rev() {
        let uid1 = *next_uid; *next_uid += 1;
        let rest = active[orig_idx].split_at(x_lo, arena, uid1);
        if rest.is_none() { continue; }
        let mut rest = rest.unwrap();
        let uid2 = *next_uid; *next_uid += 1;
        let right_of_hi = rest.split_at(x_hi, arena, uid2);

        if active[orig_idx].head == SEG_NIL {
            active.swap_remove(orig_idx);
        }
        if let Some(right) = right_of_hi {
            if right.head != SEG_NIL {
                active.push(right);
            }
        }
        if rest.head != SEG_NIL {
            let rest_uid = rest.uid;
            active.push(rest);
            window_uids.push(rest_uid);
        }
    }

    if window_uids.len() < 2 { return; }

    // Sequentially coalesce all window lineages by UID.
    let eps = (t * 1e-12).max(1e-9);
    let mut merged_uid = window_uids[0];
    for (k, &other_uid) in window_uids[1..].iter().enumerate() {
        let t_merge = t + (k as f64 + 1.0) * eps;
        let mi = active.iter().position(|l| l.uid == merged_uid);
        let oi = active.iter().position(|l| l.uid == other_uid);
        if let (Some(mi), Some(oi)) = (mi, oi) {
            apply_coalescence(active, mi, oi, t_merge, arena, tables, next_uid);
            // Merged lineage is the last one pushed by apply_coalescence.
            merged_uid = active.last().unwrap().uid;
        }
    }
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panmictic_no_recomb_gives_single_tree() {
        let sim = HullSimulator::panmictic(10, 1000.0, 100.0, 0.0, 42);
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 19);
        assert_eq!(result.tables.num_edges(), 18);
    }

    #[test]
    fn panmictic_with_recomb_gives_multiple_trees() {
        // Low rho (4*500*1e-4*50 = 10) to keep O(n^2) pair
        // enumeration tractable. High-rho performance needs Fenwick
        // tree rate computation (future optimization).
        let sim = HullSimulator::panmictic(4, 500.0, 50.0, 1e-4, 42);
        let result = sim.simulate();
        // 4 samples no-recomb → 7 nodes. With rho=10, expect more.
        assert!(result.tables.num_nodes() >= 7,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    fn two_samples_no_recomb() {
        let sim = HullSimulator::panmictic(2, 100.0, 50.0, 0.0, 7);
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 3);
        assert_eq!(result.tables.num_edges(), 2);
    }

    #[test]
    fn coal_times_positive() {
        let sim = HullSimulator::panmictic(5, 1000.0, 100.0, 0.0, 123);
        let result = sim.simulate();
        for i in 5..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0);
        }
    }

    #[test]
    fn single_inv_more_nodes_than_panmictic() {
        // With an inversion barrier, S/I pairs can't coalesce until
        // t_inv, producing more nodes (longer genealogy).
        let inv = InversionSpec {
            bp_left: 30.0, bp_right: 70.0,
            p_inv: 0.5, t_inv: 5000.0,
            gene_conversion_rate: 0.0, flux_window: 0.05, inv_id: 0,
        };
        let sim = HullSimulator::simple(
            5, 5, 1000.0, 100.0, 0.0, vec![inv], 42);
        let result = sim.simulate();
        // 10 samples + at least 9 internal = 19 nodes, but with the
        // barrier more recombination-like events (from partial overlap
        // at different classes) typically produce extra nodes.
        assert!(result.tables.num_nodes() >= 19,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    fn barrier_crossing_reduces_active_classes() {
        // Very old inversion → barrier crossed early, should
        // behave like panmictic after t_inv.
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: 100.0,
            p_inv: 0.5, t_inv: 1.0, // crossed almost immediately
            gene_conversion_rate: 0.0, flux_window: 0.05, inv_id: 0,
        };
        let sim = HullSimulator::simple(
            3, 3, 1000.0, 100.0, 0.0, vec![inv], 42);
        let result = sim.simulate();
        // Should still produce a valid tree with 6 samples.
        assert!(result.tables.num_nodes() >= 11);
    }

    #[test]
    fn gene_flux_produces_extra_nodes() {
        // With gene flux, flux events split lineages → more nodes.
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: 200.0,
            p_inv: 0.5, t_inv: 20_000.0,
            gene_conversion_rate: 5e-5, flux_window: 0.05, inv_id: 0,
        };
        let no_flux = HullSimulator::simple(
            4, 4, 1000.0, 200.0, 0.0,
            vec![InversionSpec { gene_conversion_rate: 0.0, ..inv.clone() }],
            42);
        let with_flux = HullSimulator::simple(
            4, 4, 1000.0, 200.0, 0.0, vec![inv], 42);
        let r_no = no_flux.simulate();
        let r_yes = with_flux.simulate();
        // Gene flux creates additional lineages → more coalescence
        // events → more nodes/edges.
        assert!(r_yes.tables.num_nodes() >= r_no.tables.num_nodes(),
            "flux={} vs no_flux={}", r_yes.tables.num_nodes(),
            r_no.tables.num_nodes());
    }

    #[test]
    fn two_pop_with_merge() {
        use crate::demography::{Demography, DemoEvent};
        let mut demo = Demography::new(vec![1000.0, 1000.0]);
        demo.add_event(DemoEvent::Ej { t: 500.0, src: 1, dst: 0 });

        let sim = HullSimulator {
            samples: vec![
                SampleEntry {
                    karyotypes: vec![], population: 0, count: 5,
                },
                SampleEntry {
                    karyotypes: vec![], population: 1, count: 5,
                },
            ],
            demography: demo,
            sequence_length: 100.0,
            recombination_rate: 0.0,
            inversions: vec![],
            sweeps: vec![],
            seed: 42,
        };
        let result = sim.simulate();
        // 10 samples + at least 9 internal = 19 nodes.
        // With pop split, T_MRCA >= 500 for cross-pop pairs.
        assert!(result.tables.num_nodes() >= 19,
            "Got {} nodes", result.tables.num_nodes());
        // All internal node times should be positive.
        for i in 10..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0);
        }
    }

    #[test]
    fn two_pop_with_migration() {
        use crate::demography::Demography;
        let mut demo = Demography::new(vec![1000.0, 1000.0]);
        // Symmetric migration at 0.001 per gen.
        demo.migration_matrix[0][1] = 0.001;
        demo.migration_matrix[1][0] = 0.001;

        let sim = HullSimulator {
            samples: vec![
                SampleEntry {
                    karyotypes: vec![], population: 0, count: 3,
                },
                SampleEntry {
                    karyotypes: vec![], population: 1, count: 3,
                },
            ],
            demography: demo,
            sequence_length: 100.0,
            recombination_rate: 0.0,
            inversions: vec![],
            sweeps: vec![],
            seed: 42,
        };
        let result = sim.simulate();
        // Should produce a valid tree with migration allowing
        // cross-pop coalescence.
        assert!(result.tables.num_nodes() >= 11,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    fn two_pop_inversion_with_merge() {
        use crate::demography::{Demography, DemoEvent};
        let mut demo = Demography::new(vec![1000.0, 1000.0]);
        demo.add_event(DemoEvent::Ej { t: 500.0, src: 1, dst: 0 });

        let inv = InversionSpec {
            bp_left: 30.0, bp_right: 70.0,
            p_inv: 0.5, t_inv: 5000.0,
            gene_conversion_rate: 0.0, flux_window: 0.05, inv_id: 0,
        };
        let sim = HullSimulator {
            samples: vec![
                SampleEntry {
                    karyotypes: vec![Some(Karyotype::S)],
                    population: 0, count: 3,
                },
                SampleEntry {
                    karyotypes: vec![Some(Karyotype::I)],
                    population: 1, count: 3,
                },
            ],
            demography: demo,
            sequence_length: 100.0,
            recombination_rate: 0.0,
            inversions: vec![inv],
            sweeps: vec![],
            seed: 42,
        };
        let result = sim.simulate();
        // Should complete. Cross-pop + cross-karyotype TMRCA >=
        // max(t_split=500, t_inv=5000) = 5000.
        assert!(result.tables.num_nodes() >= 11);
    }

    #[test]
    fn sweep_reduces_diversity_in_window() {
        use crate::sweep::Sweep;
        // Sweep at centre of [0, 100) at t=100 gen — all lineages
        // carrying material at x=50 coalesce to a single ancestor.
        let mut sim = HullSimulator::panmictic(
            6, 10_000.0, 100.0, 0.0, 42);
        sim.sweeps.push(Sweep {
            x_sel: 50.0,
            t_event: 100.0,
            target: None,       // all classes
            population: None,
            sweep_window: 10.0, // [40, 60)
        });
        let result = sim.simulate();
        // 6 samples should still all end up in a tree. The sweep
        // forces a coalescence at t=100 for the [40,60] window.
        assert!(result.tables.num_nodes() >= 11,
            "Got {} nodes", result.tables.num_nodes());
        // There should be at least one node at t ≈ 100 (the sweep).
        let near_100 = result.tables.node_time.iter()
            .filter(|&&t| (t - 100.0).abs() < 1.0)
            .count();
        assert!(near_100 >= 1,
            "Expected node(s) at t~100 from sweep, found {}", near_100);
    }

    #[test]
    fn sweep_on_s_class_only() {
        use crate::sweep::Sweep;
        let inv = InversionSpec {
            bp_left: 20.0, bp_right: 80.0,
            p_inv: 0.5, t_inv: 50_000.0,
            gene_conversion_rate: 0.0, flux_window: 0.05, inv_id: 0,
        };
        let mut sim = HullSimulator::simple(
            4, 4, 5_000.0, 100.0, 0.0, vec![inv], 42);
        sim.sweeps.push(Sweep {
            x_sel: 50.0,
            t_event: 200.0,
            target: Some((0, Karyotype::S)), // only S lineages
            population: None,
            sweep_window: 5.0,
        });
        let result = sim.simulate();
        // Should complete without panic. S lineages coalesce at t=200
        // near x=50; I lineages are unaffected by the sweep.
        assert!(result.tables.num_nodes() >= 15,
            "Got {} nodes", result.tables.num_nodes());
    }
}
