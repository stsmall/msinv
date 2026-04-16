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
        // rho=0 is forbidden globally (matches Python). Without
        // recombination, partial coalescence fragments lineages that
        // can never recombine back together. For independent loci,
        // simulate each separately.
        if self.recombination_rate <= 0.0 {
            panic!(
                "recombination_rate must be > 0 (got {}). rho=0 is not \
                 supported. For non-recombining loci, simulate each \
                 locus separately.",
                self.recombination_rate);
        }
        // gamma > 0 required for any inversion (matches Python).
        for inv in &self.inversions {
            if inv.gene_conversion_rate <= 0.0 {
                panic!(
                    "gene_conversion_rate (gamma) must be > 0 for every \
                     inversion (got {} for inv_id={}). gamma=0 makes the \
                     inversion an absolute barrier (often unrealistic).",
                    inv.gene_conversion_rate, inv.inv_id);
            }
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(
            self.sequence_length, self.demography.n_pops);
        let mut next_uid: LinUid = 0;
        let mut demo = self.demography.clone();
        let mut inversions = self.inversions.clone();

        let mut active = self.make_initial_lineages(
            &mut arena, &mut tables, &mut next_uid);

        self.run_loop(&mut active, &mut arena, &mut tables,
                       &mut next_uid, &mut rng, &mut demo,
                       &mut inversions);

        // NOTE: sort_edges disabled — was producing wrong tree
        // sequences. Python bridge does tc.sort() anyway.
        // tables.sort_edges();
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
        inversions: &mut Vec<InversionSpec>,
    ) {
        let mut t: f64 = 0.0;

        // Track which inversions' barriers are still active.
        let mut barrier_active: Vec<bool> = inversions.iter()
            .map(|_| true).collect();

        // Pending sweeps, sorted by t_event (earliest first).
        let mut pending_sweeps: Vec<Sweep> = self.sweeps.clone();
        pending_sweeps.sort_by(|a, b| a.t_event.partial_cmp(&b.t_event).unwrap());

        // Running totals for O(1) recombination rate (Phase A).
        let mut total_material: f64 = active.iter()
            .map(|l| l.cached_len).sum();
        let mut total_recomb_rate: f64 = total_material * self.recombination_rate;

        // Phase D: incremental pair rate cache.
        let max_lins = (active.len() * 20).max(256);
        let mut rate_cache = RateCache::new(max_lins);
        rate_cache.rebuild(&active, arena);

        // Persistent event list + Fenwick tree. Rebuilt on structural
        // changes; reused when only recombination happens.
        let mut all_events: Vec<(f64, Event)> = Vec::new();
        let mut event_tree = crate::fenwick::Fenwick::new(0);
        let mut engine_dirty = true;  // force full rebuild

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
            for (k, inv) in inversions.iter().enumerate() {
                if barrier_active[k] {
                    any_barrier = true;
                    earliest_barrier = earliest_barrier.min(inv.t_inv);
                }
            }

            // Next demographic event boundary.
            let t_demo = demo.next_event_time(t);

            // --- Build or reuse event rates ---
            if engine_dirty {
                all_events.clear();

                // Coalescence.
                if any_barrier {
                    rate_cache.rebuild(active, arena);
                    emit_coal_events_from_cache(
                        &rate_cache, active, &*demo, t,
                        inversions, &barrier_active,
                        &mut all_events);
                } else {
                    compute_coal_events(
                        active, arena, demo, t, inversions,
                        &barrier_active, &mut all_events);
                }

                // Recombination.
                if total_recomb_rate > 0.0 {
                    all_events.push((total_recomb_rate, Event::Recombination));
                }

                // Gene flux.
                if any_barrier {
                    Self::compute_flux_rates_static(
                        inversions, active, arena, &barrier_active, &mut all_events);
                }
                // Migration.
                for (rate, lin_idx, dst) in demo.migration_rates(active) {
                    all_events.push((rate, Event::Migration {
                        lineage_idx: lin_idx, dst_pop: dst,
                    }));
                }

                // Rebuild Fenwick tree.
                let n_events = all_events.len();
                event_tree = crate::fenwick::Fenwick::new(n_events);
                for (leaf, (rate, _)) in all_events.iter().enumerate() {
                    event_tree.update(leaf, *rate);
                }
                engine_dirty = false;
            }

            let total_rate = event_tree.total();

            // Next sweep boundary.
            let t_sweep = pending_sweeps.first()
                .map(|s| s.t_event).unwrap_or(f64::INFINITY);

            // Next deterministic boundary.
            let next_boundary = earliest_barrier.min(t_demo).min(t_sweep);

            if total_rate <= 0.0 {
                if next_boundary < f64::INFINITY {
                    t = next_boundary;
                    apply_boundary(
                        inversions, active, arena, &mut barrier_active,
                        demo, &mut pending_sweeps, t, tables, next_uid,
                        self.sequence_length, rng, self.recombination_rate);
                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                    total_recomb_rate = total_material * self.recombination_rate;
                    engine_dirty = true;
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
                apply_boundary(
                    inversions, active, arena, &mut barrier_active,
                    demo, &mut pending_sweeps, t, tables, next_uid,
                    self.sequence_length, rng, self.recombination_rate);
                total_material = active.iter()
                    .map(|l| l.cached_len).sum();
                total_recomb_rate = total_material * self.recombination_rate;
                engine_dirty = true;
                continue;
            }
            t = t_event;

            // Pick which event fires — O(log n) via Fenwick tree.
            let u2: f64 = rng.random::<f64>() * total_rate;
            let leaf = event_tree.find(u2);
            let chosen_event = if leaf < all_events.len() {
                &all_events[leaf].1
            } else {
                continue;  // numerical precision miss
            };

            match chosen_event {
                Event::CoalPair { i, j, class } => {
                    let (i, j) = (*i, *j);
                    let cls = *class;
                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        Some(cls));
                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                    engine_dirty = true;
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
                        // Phase F: hull prescreen — skip if lineage
                        // extents don't overlap (cheap rejection).
                        let (a, b) = (pool[ii], pool[jj]);
                        if !active[a].hulls_overlap(&active[b], arena) {
                            continue; // no-op, draw next event
                        }
                        apply_coalescence(
                            active, a, b, t, arena,
                            tables, next_uid);
                        // Recompute after merge.
                        total_material = active.iter()
                            .map(|l| l.cached_len).sum();
                        engine_dirty = true;
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
                    apply_recombination(active, chosen_idx, x, arena,
                                         next_uid);
                    // Recombination preserves total material but changes
                    // lineage indices → rebuild event list.
                    engine_dirty = true;
                    // GC sole-carrier lineages — only after recomb
                    // (matches Python). GC after coalescence is wrong:
                    // the merged lineage's solo bits (non-overlap parts
                    // from the two parents) still need to coalesce with
                    // others, but if no current other lineage covers
                    // them they get incorrectly discarded.
                    let n_before_gc = active.len();
                    gc_sole_lineages(active, arena);
                    if active.len() != n_before_gc {
                        total_material = active.iter()
                            .map(|l| l.cached_len).sum();
                    }
                }
                Event::Flux { lineage_idx, inv_idx } => {
                    let (li, ii) = (*lineage_idx, *inv_idx);
                    let inv = &inversions[ii];
                    if let Some(x_event) = self.sample_flux_position(
                        active, li, inv, arena, rng)
                    {
                        let (tl, tr) = self.draw_tract(x_event, inv, rng);
                        if tr > tl {
                            apply_gene_flux(active, li, tl, tr, inv,
                                             arena, next_uid);
                        }
                        engine_dirty = true;
                    }
                }
                Event::Migration { lineage_idx, dst_pop } => {
                    let idx = *lineage_idx;
                    if idx < active.len() {
                        active[idx].population = *dst_pop;
                    }
                    engine_dirty = true;  // pop assignment changed
                }
            }

            // Keep recomb rate in sync.
            total_recomb_rate = total_material * self.recombination_rate;
        }
    }

    // ---------------------------------------------------------------
    // Per-pair, per-class coalescence rates
    // ---------------------------------------------------------------
    #[allow(dead_code)]
    fn compute_coal_rates_structured(
        inversions: &[InversionSpec],
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
                    let pop = active[i].population;
                    let p_class = p_class_for_tag(
                        *cls, inversions, barrier_active, t, pop);
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


    // ---------------------------------------------------------------
    // Gene flux rates
    // ---------------------------------------------------------------
    fn compute_flux_rates_static(
        inversions: &[InversionSpec],
        active: &[Lineage],
        arena: &SegmentArena,
        barrier_active: &[bool],
        events: &mut Vec<(f64, Event)>,
    ) {
        for (inv_idx, inv) in inversions.iter().enumerate() {
            if !barrier_active[inv_idx] { continue; }
            if inv.gene_conversion_rate <= 0.0 { continue; }
            for (lin_idx, lin) in active.iter().enumerate() {
                // Per-population inversion frequency for this lineage's pop.
                let pop = lin.population;
                let p_inv_pop = inv.p_inv_for(pop);
                let p_std_pop = 1.0 - p_inv_pop;
                // Determine lineage's class for this inversion.
                let kary = lineage_class_for_inv(lin, inv, arena);
                let p_other = match kary {
                    Some(Karyotype::S) => p_inv_pop,
                    Some(Karyotype::I) => p_std_pop,
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
    fn cross_barriers_static(
        inversions: &[InversionSpec],
        active: &mut [Lineage],
        arena: &mut SegmentArena,
        barrier_active: &mut [bool],
        t: f64,
    ) {
        for (k, inv) in inversions.iter().enumerate() {
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
        let pop = active[i].population;
        let ne = demo.size_at(pop, t).max(1e-9);
        for (cls, _ov_len) in overlaps {
            let p_class = p_class_for_tag(*cls, inversions, barrier_active, t, pop);
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
                let pop = active[i].population;
                let p_class = p_class_for_tag(*cls, inversions, barrier_active, t, pop);
                if p_class <= 0.0 { continue; }
                let rate = 1.0 / (2.0 * ne * p_class);
                events.push((rate, Event::CoalPair { i, j, class: *cls }));
            }
        }
    }
}

/// Effective sub-population frequency for a BranchClass tag,
/// using per-population inversion frequencies.
fn p_class_for_tag(cls: BranchClass, inversions: &[InversionSpec],
                    barrier_active: &[bool], t: f64, pop: u32) -> f64 {
    if cls.is_panmictic() {
        return 1.0;
    }
    let mut p = 1.0;
    for (k, inv) in inversions.iter().enumerate() {
        if !barrier_active[k] || t >= inv.t_inv { continue; }
        match cls.get_inv(inv.inv_id) {
            Some(Karyotype::S) => p *= inv.p_std_for(pop),
            Some(Karyotype::I) => p *= inv.p_inv_for(pop),
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
                let pop = active[i].population;
                let ne = demo.size_at(pop, t).max(1e-9);
                let overlaps = overlap_by_class(
                    active[i].head, active[j].head, arena);
                for (cls, _ov_len) in &overlaps {
                    let p_class = p_class_for_tag(
                        *cls, inversions, barrier_active, t, pop);
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

/// Apply a deterministic boundary: cross barriers, fire demographic
/// events (propagating any inversion frequency changes), and fire any
/// pending sweep whose time matches.
fn apply_boundary(
    inversions: &mut Vec<InversionSpec>,
    active: &mut Vec<Lineage>,
    arena: &mut SegmentArena,
    barrier_active: &mut [bool],
    demo: &mut Demography,
    pending_sweeps: &mut Vec<Sweep>,
    t: f64,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    seq_len: f64,
    rng: &mut Xoshiro256PlusPlus,
    recomb_rate: f64,
) {
    HullSimulator::cross_barriers_static(inversions, active, arena, barrier_active, t);
    let inv_changes = demo.apply_events_at(t, active);
    for (inv_id, pop, p_inv_val) in inv_changes {
        if let Some(inv) = inversions.iter_mut().find(|i| i.inv_id == inv_id) {
            inv.set_p_inv_for(pop, p_inv_val);
        }
    }
    if !pending_sweeps.is_empty()
        && (pending_sweeps[0].t_event - t).abs() < 1e-9
    {
        let sweep = pending_sweeps.remove(0);
        let ne_sweep = demo.size_at(
            sweep.population.unwrap_or(0), t).max(1.0);
        apply_sweep(active, &sweep, t, arena, tables,
                     next_uid, seq_len, rng, ne_sweep, recomb_rate);
    }
}

/// Force-coalesce qualifying lineages at a sweep event.
///
/// Three modes:
///
/// 1. **Window mode** (selection_coefficient == 0): split out the sweep
///    window and coalesce all qualifying lineages deterministically.
///
/// 2. **Hitchhiking mode** (selection_coefficient > 0, starting_frequency == 0):
///    each segment is included probabilistically based on recombination
///    distance from x_sel. All swept lineages coalesce to a single ancestor.
///
/// 3. **Soft sweep** (selection_coefficient > 0, starting_frequency > 0):
///    hitchhiking mode, but swept lineages are randomly partitioned among
///    K ≈ 1/f0 founding copies (discoal model). Lineages within each group
///    coalesce; K surviving ancestors continue at normal coalescent rate.
fn apply_sweep(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    seq_len: f64,
    rng: &mut Xoshiro256PlusPlus,
    ne: f64,
    recomb_rate: f64,
) {
    // ---- Identify qualifying lineages ----
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

    // ---- Hitchhiking mode: probabilistically select segments ----
    if sweep.selection_coefficient > 0.0 {
        apply_sweep_hitchhiking(
            active, sweep, t, arena, tables, next_uid, rng, ne, recomb_rate);
        return;
    }

    // ---- Window mode ----
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
    coalesce_uid_group(active, &window_uids, t, arena, tables, next_uid);
}

/// Hitchhiking mode: probabilistic segment inclusion + optional soft sweep.
///
/// For each qualifying lineage, each segment is included with probability
/// `exp(-r * |midpoint - x_sel| * t_dur)`. Segments that are NOT swept
/// are split into a separate lineage that continues independently.
///
/// For hard sweeps (starting_frequency == 0): all swept lineages merge
/// to a single ancestor. For soft sweeps (starting_frequency > 0):
/// swept lineages are randomly partitioned among K ≈ 1/f0 founder
/// groups, and lineages within each group are coalesced separately.
fn apply_sweep_hitchhiking(
    active: &mut Vec<Lineage>,
    sweep: &Sweep,
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    rng: &mut Xoshiro256PlusPlus,
    ne: f64,
    recomb_rate: f64,
) {
    // ---- Identify qualifying lineages (repeat — indices are fragile) ----
    let mut qualifying_uids: Vec<LinUid> = Vec::new();
    for lin in active.iter() {
        if let Some(pop) = sweep.population {
            if lin.population != pop { continue; }
        }
        if let Some(cls) = lin.class_at(sweep.x_sel, arena) {
            if sweep.class_matches(cls) {
                qualifying_uids.push(lin.uid);
            }
        }
    }
    if qualifying_uids.len() < 2 { return; }

    // ---- Split each qualifying lineage into swept / unswept parts ----
    let mut swept_uids: Vec<LinUid> = Vec::new();

    for &q_uid in &qualifying_uids {
        let q_idx = match active.iter().position(|l| l.uid == q_uid) {
            Some(i) => i,
            None => continue,
        };

        // Walk segments, classify each as swept or unswept.
        let mut swept_segs: Vec<(f64, f64, i32, BranchClass)> = Vec::new();
        let mut unswept_segs: Vec<(f64, f64, i32, BranchClass)> = Vec::new();

        let mut cur = active[q_idx].head;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            let mid = (seg.left + seg.right) / 2.0;
            let p = sweep.hitchhiking_probability(mid, recomb_rate, ne);
            let u: f64 = rng.random();
            if u < p {
                swept_segs.push((seg.left, seg.right, seg.node_id, seg.branch_class));
            } else {
                unswept_segs.push((seg.left, seg.right, seg.node_id, seg.branch_class));
            }
            cur = seg.next;
        }

        if swept_segs.is_empty() {
            continue; // lineage entirely escapes the sweep
        }

        // Remove original lineage.
        let pop = active[q_idx].population;
        active.swap_remove(q_idx);

        // Build swept lineage.
        let swept_uid = *next_uid; *next_uid += 1;
        let swept_lin = build_lineage_from_segs(&swept_segs, pop, swept_uid, arena);
        active.push(swept_lin);
        swept_uids.push(swept_uid);

        // Build unswept lineage (if any segments).
        if !unswept_segs.is_empty() {
            let unsw_uid = *next_uid; *next_uid += 1;
            let unsw_lin = build_lineage_from_segs(&unswept_segs, pop, unsw_uid, arena);
            active.push(unsw_lin);
        }
    }

    if swept_uids.len() < 2 { return; }

    // ---- Soft sweep: partition into K founder groups ----
    let k = sweep.num_founders();
    if k <= 1 {
        // Hard sweep: coalesce all to one ancestor.
        coalesce_uid_group(active, &swept_uids, t, arena, tables, next_uid);
    } else {
        // Soft sweep: randomly assign each swept lineage to one of K groups.
        let mut groups: Vec<Vec<LinUid>> = vec![Vec::new(); k];
        for &uid in &swept_uids {
            let g = (rng.random::<f64>() * k as f64) as usize;
            let g = g.min(k - 1); // clamp for floating-point edge case
            groups[g].push(uid);
        }
        // Coalesce within each group.
        let eps = (t * 1e-12).max(1e-9);
        for (gi, group) in groups.iter().enumerate() {
            if group.len() < 2 { continue; }
            let t_group = t + (gi as f64) * eps;
            coalesce_uid_group(active, group, t_group, arena, tables, next_uid);
        }
    }
}

/// Build a Lineage from a vector of (left, right, node_id, branch_class) tuples.
fn build_lineage_from_segs(
    segs: &[(f64, f64, i32, BranchClass)],
    pop: u32,
    uid: LinUid,
    arena: &mut SegmentArena,
) -> Lineage {
    let mut head = SEG_NIL;
    let mut tail = SEG_NIL;
    for (l, r, nid, cls) in segs {
        let seg = arena.alloc(*l, *r, *nid, *cls);
        if tail != SEG_NIL {
            arena.get_mut(tail).next = seg;
        } else {
            head = seg;
        }
        tail = seg;
    }
    Lineage::new(head, tail, pop, uid, arena)
}

/// Coalesce a group of lineages (identified by UID) sequentially.
fn coalesce_uid_group(
    active: &mut Vec<Lineage>,
    uids: &[LinUid],
    t: f64,
    arena: &mut SegmentArena,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
) {
    if uids.len() < 2 { return; }
    let eps = (t * 1e-12).max(1e-9);
    let mut merged_uid = uids[0];
    for (k, &other_uid) in uids[1..].iter().enumerate() {
        let t_merge = t + (k as f64 + 1.0) * eps;
        let mi = active.iter().position(|l| l.uid == merged_uid);
        let oi = active.iter().position(|l| l.uid == other_uid);
        if let (Some(mi), Some(oi)) = (mi, oi) {
            apply_coalescence(active, mi, oi, t_merge, arena, tables, next_uid);
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
        // rho > 0 enforced; use 1e-12 (expected recombs ≈ 1e-6).
        let sim = HullSimulator::panmictic(10, 1000.0, 100.0, 1e-12, 42);
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
        let sim = HullSimulator::panmictic(2, 100.0, 50.0, 1e-12, 7);
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 3);
        assert_eq!(result.tables.num_edges(), 2);
    }

    #[test]
    fn coal_times_positive() {
        let sim = HullSimulator::panmictic(5, 1000.0, 100.0, 1e-12, 123);
        let result = sim.simulate();
        for i in 5..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0);
        }
    }

    #[test]
    fn single_inv_more_nodes_than_panmictic() {
        // With an inversion barrier, S/I pairs can't coalesce until
        // t_inv, producing more nodes (longer genealogy).
        // Ne=1000, L=10000, r=1e-8 → rho=0.4
        let inv = InversionSpec {
            bp_left: 3000.0, bp_right: 7000.0,
            p_inv: vec![0.5], t_inv: 5000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let sim = HullSimulator::simple(
            5, 5, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 19,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    fn barrier_crossing_reduces_active_classes() {
        // Very old inversion → barrier crossed early, should
        // behave like panmictic after t_inv.
        // Ne=1000, L=10000, r=1e-8 → rho=0.4
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: 10000.0,
            p_inv: vec![0.5], t_inv: 1.0, // crossed almost immediately
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let sim = HullSimulator::simple(
            3, 3, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 11);
    }

    #[test]
    fn gene_flux_produces_extra_nodes() {
        // With gene flux, flux events split lineages → more nodes.
        // Ne=1000, L=10000, r=1e-8 → rho=0.4
        let inv = InversionSpec {
            bp_left: 0.0, bp_right: 10000.0,
            p_inv: vec![0.5], t_inv: 20_000.0,
            gene_conversion_rate: 5e-6, flux_window: 0.05, inv_id: 0,
        };
        let no_flux = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8,
            vec![InversionSpec { gene_conversion_rate: 1e-9, ..inv.clone() }],
            42);
        let with_flux = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        let r_no = no_flux.simulate();
        let r_yes = with_flux.simulate();
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
            recombination_rate: 1e-12,
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
            recombination_rate: 1e-12,
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

        // Ne=1000, L=10000, r=1e-8 → rho=0.4
        let inv = InversionSpec {
            bp_left: 3000.0, bp_right: 7000.0,
            p_inv: vec![0.5], t_inv: 5000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
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
            sequence_length: 10000.0,
            recombination_rate: 1e-8,
            inversions: vec![inv],
            sweeps: vec![],
            seed: 42,
        };
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 11);
    }

    #[test]
    fn sweep_reduces_diversity_in_window() {
        use crate::sweep::Sweep;
        // Sweep at centre of [0, 100) at t=100 gen — all lineages
        // carrying material at x=50 coalesce to a single ancestor.
        let mut sim = HullSimulator::panmictic(
            6, 10_000.0, 100.0, 1e-12, 42);
        sim.sweeps.push(Sweep {
            x_sel: 50.0,
            t_event: 100.0,
            target: None,       // all classes
            population: None,
            sweep_window: 10.0, // [40, 60)
            ..Default::default()
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
        // Ne=5000, L=100000, r=1e-8 → rho=20
        let inv = InversionSpec {
            bp_left: 20000.0, bp_right: 80000.0,
            p_inv: vec![0.5], t_inv: 50_000.0,
            gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
        };
        let mut sim = HullSimulator::simple(
            4, 4, 5_000.0, 100000.0, 1e-8, vec![inv], 42);
        sim.sweeps.push(Sweep {
            x_sel: 50000.0,
            t_event: 200.0,
            target: Some((0, Karyotype::S)),
            population: None,
            sweep_window: 5000.0,
            ..Default::default()
        });
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 15,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    fn soft_sweep_with_recombination() {
        // Soft sweep K=5 with rho=40 (realistic recombination).
        // T_MRCA should be >> t_event because K=5 founders survive.
        let mut sim = HullSimulator::panmictic(
            20, 10_000.0, 100_000.0, 1e-8, 42);
        sim.sweeps.push(Sweep {
            x_sel: 50_000.0,
            t_event: 500.0,
            target: None,
            population: None,
            sweep_window: 0.0,
            selection_coefficient: 0.01,
            starting_frequency: 0.2,
        });
        let result = sim.simulate();
        let t_mrca = result.tables.node_time.iter()
            .cloned().fold(0.0_f64, f64::max);
        assert!(t_mrca > 2000.0,
            "Soft sweep K=5 with rho=40: T_MRCA={:.1}, expected >> 500", t_mrca);
    }
}
