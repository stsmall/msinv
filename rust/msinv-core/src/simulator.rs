/// HullSimulator: the main event loop.
///
/// Phase C: inversions (class barriers, per-pair coal rates, gene flux,
/// barrier crossing) on top of the Phase B panmictic loop.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use smallvec::SmallVec;

use crate::class_tag::{BranchClass, Karyotype};
use crate::demography::Demography;
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
    /// Per-pair coalescence (used by the non-cache structured fallback
    /// `compute_coal_rates_structured`). Hot path uses CoalAggregate.
    CoalPair { i: usize, j: usize, class: BranchClass },
    /// Aggregate coalescence rate for all pairs in `pop` whose overlap
    /// lies in `class`. Firing samples the specific (i, j) pair from
    /// RateCache proportional to overlap length in that class — avoids
    /// the O(n^2) per-pair event-list entries that dominated rho≥500.
    CoalAggregate { pop: u32, class: BranchClass },
    CoalPanmicticPop { pop: u32 },
    Recombination,
    /// Aggregate gene-flux rate for all lineages interacting with
    /// `inv_idx`. Firing samples a lineage proportional to its cached
    /// per-lineage flux rate — avoids the O(n * segs) full flux scan
    /// on every event-list rebuild.
    FluxAggregate { inv_idx: usize },
    /// Aggregate migration: all lineages in `src_pop` migrating to
    /// `dst_pop` with the same per-lineage rate. Firing picks one
    /// lineage uniformly from `pop_buckets[src_pop]` and migrates it.
    /// Replaces the O(n · n_pops) per-lineage Migration events that
    /// dominated multi-pop event-list builds at ~9.5% of run_loop.
    MigrationAggregate { src_pop: u32, dst_pop: u32 },
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
        let mut rate_cache = RateCache::new(0, self.sequence_length);
        self.simulate_with_cache(&mut rate_cache)
    }

    /// Simulate, reusing the caller-owned `rate_cache`. The cache is
    /// `reset()` before this rep and its heap allocations survive for
    /// the next call. Use when driving many reps from one thread (e.g.
    /// bench binaries, single-process ABC loops) to amortise the
    /// triangular overlap array allocation across reps.
    pub fn simulate_with_cache(
        &self, rate_cache: &mut RateCache,
    ) -> SimResult {
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
                       &mut inversions, rate_cache);

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
        rate_cache: &mut RateCache,
    ) {
        let mut t: f64 = 0.0;

        // Track which inversions' barriers are still active.
        let mut barrier_active: Vec<bool> = inversions.iter()
            .map(|_| true).collect();

        // Pending sweeps, sorted by t_event (earliest first).
        let mut pending_sweeps: Vec<Sweep> = self.sweeps.clone();
        pending_sweeps.sort_by(|a, b| a.t_event.partial_cmp(&b.t_event).unwrap());

        // Monotone sweep-merge cursor shared across all sweeps at the
        // same base t (prevents TSK_ERR_BAD_NODE_TIME_ORDERING when two
        // sweeps fire simultaneously).
        let mut sweep_cursor: (f64, u64) = (f64::NAN, 0);

        // Running totals for O(1) recombination rate (Phase A).
        let mut total_material: f64 = active.iter()
            .map(|l| l.cached_len).sum();
        let mut total_recomb_rate: f64 = total_material * self.recombination_rate;

        // Phase D: incremental pair rate cache. Pre-size generously:
        // `pair_idx` is capacity-dependent, so every `ensure_capacity`
        // growth must reindex (O(n²)) to preserve correctness of
        // class_totals. Oversizing up front avoids most mid-run grows
        // for rho ≤ 8000 without wasting meaningful memory (triangular
        // array stays sparse).
        let max_lins = (active.len() * 40).max(2048);
        rate_cache.reset(max_lins, self.sequence_length);
        rate_cache.rebuild(&active, arena);

        // Persistent event list + Fenwick tree. Rebuilt on structural
        // changes; reused when only recombination happens.
        let mut all_events: Vec<(f64, Event)> = Vec::with_capacity(1024);
        let mut rate_buf: Vec<f64> = Vec::with_capacity(1024);
        // Per-pop lineage index buckets — refreshed inside the
        // `engine_dirty` rebuild block. Gives O(1) pair picks in the
        // multi-pop CoalPanmicticPop handler and feeds aggregate
        // migration. Buckets stay valid between rebuilds because
        // events mutating `active` set engine_dirty=true, forcing a
        // rebuild on the next iteration before the next aggregate
        // fire.
        let mut pop_buckets: Vec<Vec<u32>> =
            (0..demo.n_pops).map(|_| Vec::new()).collect();
        let mut event_tree = crate::fenwick::Fenwick::new(0);
        // Fenwick over lineage cached_lens. Enables O(log n) proportional
        // selection for recombination, replacing the O(n) linear scan
        // that was super-linear at rho ≥ 1000. Maintained in lockstep
        // with `active` by mirroring swap_remove / push on tree slots.
        let mut lin_len_tree = crate::fenwick::Fenwick::new(0);
        let mut lin_tree_dirty = true;
        // Per-lineage flux rate cache (parallel to `active`) + per-inv
        // totals. Emitted as one aggregate Event::FluxAggregate per
        // inversion so rebuild cost drops from O(n · segs) to O(1).
        let mut flux_per_lin: Vec<FluxPerLin> = Vec::new();
        let mut flux_total: Vec<f64> = vec![0.0; inversions.len()];
        let mut flux_dirty = true;
        let mut engine_dirty = true;  // force full rebuild of event list
        let mut cache_dirty = true;   // force full rebuild of rate_cache
        // Counter throttling gc_sole_lineages — run every GC_STRIDE
        // recombs. Sole-carrier lineages contribute no coalescence rate
        // so a few rounds of delay has no correctness impact.
        const GC_STRIDE: u32 = 64;
        let mut gc_counter: u32 = 0;

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

            if lin_tree_dirty || lin_len_tree.len() < active.len() {
                rate_buf.clear();
                rate_buf.extend(active.iter().map(|l| l.cached_len));
                lin_len_tree.build_from(&rate_buf);
                lin_tree_dirty = false;
            }
            // Safety net: flux cache must stay parallel to `active`.
            // Any size drift means some mutation path missed its flux
            // hook — rebuild now so recomb/coal updates see a valid
            // shape even before engine_dirty gets its turn.
            if flux_dirty
                || flux_per_lin.len() != active.len()
                || flux_total.len() != inversions.len()
            {
                if flux_total.len() != inversions.len() {
                    flux_total = vec![0.0; inversions.len()];
                }
                flux_rebuild_full(
                    &mut flux_per_lin, &mut flux_total,
                    active, inversions, arena, &barrier_active);
                flux_dirty = false;
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

                // Rebuild per-pop index buckets so coal and migration
                // emission can consume them. Only built when n_pops
                // >= 2. Pre-reserve capacities to avoid Vec::push
                // capacity-growth branches in the hot fill loop.
                if demo.n_pops >= 2 {
                    while pop_buckets.len() < demo.n_pops as usize {
                        pop_buckets.push(Vec::new());
                    }
                    for b in pop_buckets.iter_mut() { b.clear(); }
                    for (i, l) in active.iter().enumerate() {
                        pop_buckets[l.population as usize].push(i as u32);
                    }
                }

                // Coalescence.
                if any_barrier {
                    if cache_dirty {
                        rate_cache.rebuild(active, arena);
                        cache_dirty = false;
                    }
                    emit_coal_events_from_cache(
                        &rate_cache, active, &*demo, t,
                        inversions, &barrier_active,
                        &mut all_events);
                } else {
                    compute_coal_events(
                        active, arena, demo, t, inversions,
                        &barrier_active, &pop_buckets,
                        &mut all_events);
                }

                // Recombination.
                if total_recomb_rate > 0.0 {
                    all_events.push((total_recomb_rate, Event::Recombination));
                }

                // Gene flux — aggregate per-inversion events sourced
                // from the incrementally maintained per-lineage cache.
                // Rebuild already handled at the top of the loop.
                if any_barrier {
                    for (ii, total) in flux_total.iter().enumerate() {
                        if *total > 0.0 {
                            all_events.push((*total, Event::FluxAggregate {
                                inv_idx: ii,
                            }));
                        }
                    }
                }

                // Migration — aggregate one event per (src, dst) pair.
                // rate = |pop_buckets[src]| * m[dst][src]. Firing picks
                // a lineage uniformly from pop_buckets[src]. Replaces
                // O(n · n_pops) per-lineage entries with O(n_pops²).
                if demo.n_pops >= 2 {
                    for src in 0..demo.n_pops as usize {
                        let count = pop_buckets[src].len() as f64;
                        if count == 0.0 { continue; }
                        for dst in 0..demo.n_pops as usize {
                            if dst == src { continue; }
                            let m = demo.migration_matrix[dst][src];
                            if m > 0.0 {
                                all_events.push((count * m,
                                    Event::MigrationAggregate {
                                        src_pop: src as u32,
                                        dst_pop: dst as u32,
                                    }));
                            }
                        }
                    }
                }

                // Rebuild Fenwick tree. O(n) batch build via build_from.
                rate_buf.clear();
                rate_buf.extend(all_events.iter().map(|(r, _)| *r));
                event_tree.build_from(&rate_buf);
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
                        self.sequence_length, rng, self.recombination_rate,
                        &mut sweep_cursor);
                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                    total_recomb_rate = total_material * self.recombination_rate;
                    engine_dirty = true;
                    cache_dirty = true;
                    lin_tree_dirty = true;
                    flux_dirty = true;
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
                    self.sequence_length, rng, self.recombination_rate,
                    &mut sweep_cursor);
                total_material = active.iter()
                    .map(|l| l.cached_len).sum();
                total_recomb_rate = total_material * self.recombination_rate;
                engine_dirty = true;
                cache_dirty = true;
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
                Event::CoalAggregate { pop, class } => {
                    let pop = *pop;
                    let cls = *class;
                    // class_totals tracks count of matching pairs for
                    // (pop, cls). Pre-pick target k, walk iter_pairs
                    // once with early exit at kth match. Avoids the
                    // Vec::push per candidate that dominated the
                    // rho=2000 flamegraph (~22% of wall).
                    let mut count: usize = 0;
                    for (p, c, n) in rate_cache.iter_class_totals() {
                        if p == pop && c == cls {
                            count = n as usize;
                            break;
                        }
                    }
                    if count == 0 { continue; }
                    let target = rng.random_range(0..count);
                    let mut seen: usize = 0;
                    let mut chosen: Option<(usize, usize)> = None;
                    for (ii, jj, overlaps) in rate_cache.iter_pairs() {
                        if active[ii].population != pop { continue; }
                        let mut has = false;
                        for (c, _) in overlaps {
                            if *c == cls { has = true; break; }
                        }
                        if !has { continue; }
                        if seen == target {
                            chosen = Some((ii, jj));
                            break;
                        }
                        seen += 1;
                    }
                    let (i, j) = match chosen {
                        Some(p) => p,
                        None => continue,
                    };
                    let pre_len = active.len();
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    let old_i_len = active[i].cached_len;
                    let old_j_len = active[j].cached_len;
                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        Some(cls));
                    let post_len = active.len();
                    // Incremental total_material: remove the two merged
                    // lineages' contributions, add the new lineages'.
                    let mut delta = -old_i_len - old_j_len;
                    for new_idx in (pre_len - 2)..post_len {
                        delta += active[new_idx].cached_len;
                    }
                    total_material += delta;
                    // Mirror active's swap_remove(hi); swap_remove(lo); push*
                    // on the length Fenwick so subsequent recomb picks stay
                    // O(log n). Do swap-pattern in the same order as
                    // apply_coalescence_partial.
                    tree_swap_remove(&mut lin_len_tree, hi, pre_len - 1);
                    tree_swap_remove(&mut lin_len_tree, lo, pre_len - 2);
                    for new_idx in (pre_len - 2)..post_len {
                        lin_len_tree.grow(new_idx + 1);
                        lin_len_tree.set(new_idx, active[new_idx].cached_len);
                    }
                    engine_dirty = true;

                    if any_barrier && !cache_dirty {
                        rate_cache.remove_lineage(hi);
                        rate_cache.swap_update(hi, pre_len - 1);
                        rate_cache.remove_lineage(lo);
                        rate_cache.swap_update(lo, pre_len - 2);
                        for new_idx in (pre_len - 2)..post_len {
                            rate_cache.recompute_for(new_idx, active, arena);
                        }
                    }
                    if any_barrier && !flux_dirty {
                        flux_swap_remove(hi, &mut flux_per_lin, &mut flux_total);
                        flux_swap_remove(lo, &mut flux_per_lin, &mut flux_total);
                        for new_idx in (pre_len - 2)..post_len {
                            flux_push(new_idx, &mut flux_per_lin,
                                      &mut flux_total, active, inversions,
                                      arena, &barrier_active);
                        }
                    }
                }
                Event::CoalPair { i, j, class } => {
                    let (i, j) = (*i, *j);
                    let cls = *class;
                    let pre_len = active.len();
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    let old_i_len = active[i].cached_len;
                    let old_j_len = active[j].cached_len;
                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        Some(cls));
                    let post_len = active.len();
                    let mut delta = -old_i_len - old_j_len;
                    for new_idx in (pre_len - 2)..post_len {
                        delta += active[new_idx].cached_len;
                    }
                    total_material += delta;
                    tree_swap_remove(&mut lin_len_tree, hi, pre_len - 1);
                    tree_swap_remove(&mut lin_len_tree, lo, pre_len - 2);
                    for new_idx in (pre_len - 2)..post_len {
                        lin_len_tree.grow(new_idx + 1);
                        lin_len_tree.set(new_idx, active[new_idx].cached_len);
                    }
                    engine_dirty = true;

                    if any_barrier && !cache_dirty {
                        rate_cache.remove_lineage(hi);
                        rate_cache.swap_update(hi, pre_len - 1);
                        rate_cache.remove_lineage(lo);
                        rate_cache.swap_update(lo, pre_len - 2);
                        for new_idx in (pre_len - 2)..post_len {
                            rate_cache.recompute_for(new_idx, active, arena);
                        }
                    }
                    if any_barrier && !flux_dirty {
                        flux_swap_remove(hi, &mut flux_per_lin, &mut flux_total);
                        flux_swap_remove(lo, &mut flux_per_lin, &mut flux_total);
                        for new_idx in (pre_len - 2)..post_len {
                            flux_push(new_idx, &mut flux_per_lin,
                                      &mut flux_total, active, inversions,
                                      arena, &barrier_active);
                        }
                    }
                }
                Event::CoalPanmicticPop { pop } => {
                    let pop = *pop;
                    // Single-pop fast path: every lineage matches, so
                    // skip the filter walk over active entirely.
                    // Multi-pop path: read the count from `pop_counts`
                    // (refreshed during engine rebuild), pre-pick two
                    // distinct ranks, then walk active once with early
                    // exit at the higher rank. Avoids the pool_buf
                    // build (Vec::push per match) — only the filter
                    // walk remains.
                    let (a, b) = if demo.n_pops == 1 {
                        let n_act = active.len();
                        if n_act < 2 { continue; }
                        let ii = rng.random_range(0..n_act);
                        let mut jj = rng.random_range(0..n_act - 1);
                        if jj >= ii { jj += 1; }
                        (ii, jj)
                    } else {
                        let bucket = &pop_buckets[pop as usize];
                        let count = bucket.len();
                        if count < 2 { continue; }
                        let ii = rng.random_range(0..count);
                        let mut jj = rng.random_range(0..count - 1);
                        if jj >= ii { jj += 1; }
                        (bucket[ii] as usize, bucket[jj] as usize)
                    };
                    {
                        // Phase F: hull prescreen — skip if lineage
                        // extents don't overlap (cheap rejection).
                        if !active[a].hulls_overlap(&active[b], arena) {
                            continue; // no-op, draw next event
                        }
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        let old_a_len = active[a].cached_len;
                        let old_b_len = active[b].cached_len;
                        let pre_len = active.len();
                        apply_coalescence(
                            active, a, b, t, arena,
                            tables, next_uid);
                        let post_len = active.len();
                        // Incremental total_material + lin_len_tree.
                        let mut delta = -old_a_len - old_b_len;
                        for new_idx in (pre_len - 2)..post_len {
                            delta += active[new_idx].cached_len;
                        }
                        total_material += delta;
                        tree_swap_remove(&mut lin_len_tree, hi, pre_len - 1);
                        tree_swap_remove(&mut lin_len_tree, lo, pre_len - 2);
                        for new_idx in (pre_len - 2)..post_len {
                            lin_len_tree.grow(new_idx + 1);
                            lin_len_tree.set(new_idx, active[new_idx].cached_len);
                        }
                        if any_barrier && !flux_dirty {
                            flux_swap_remove(hi, &mut flux_per_lin, &mut flux_total);
                            flux_swap_remove(lo, &mut flux_per_lin, &mut flux_total);
                            for new_idx in (pre_len - 2)..post_len {
                                flux_push(new_idx, &mut flux_per_lin,
                                          &mut flux_total, active, inversions,
                                          arena, &barrier_active);
                            }
                        }
                        engine_dirty = true;
                        cache_dirty = true;
                    }
                }
                Event::Recombination => {
                    let u_lin: f64 = rng.random::<f64>();
                    let target = u_lin * total_material;
                    // O(log n) proportional selection via the length
                    // Fenwick. Clamp to last valid index in case `target`
                    // floats just past total (FP rounding).
                    let chosen_idx = {
                        let raw = lin_len_tree.find(target);
                        if raw >= active.len() { active.len() - 1 } else { raw }
                    };
                    let lin_len = active[chosen_idx].cached_len;
                    if lin_len <= 0.0 { continue; }
                    let x_offset: f64 = rng.random::<f64>() * lin_len;
                    let x = find_position(active, chosen_idx, x_offset,
                                           arena, self.sequence_length);
                    let len_before_split = active.len();
                    apply_recombination(active, chosen_idx, x, arena,
                                         next_uid);
                    let len_after_split = active.len();
                    // Recombination preserves total material.
                    engine_dirty = true;
                    // Update lin_len_tree: chosen_idx's cached_len shrank;
                    // new lineage (if any) was pushed at the end.
                    lin_len_tree.set(chosen_idx, active[chosen_idx].cached_len);
                    if len_after_split > len_before_split {
                        let new_idx = len_after_split - 1;
                        lin_len_tree.grow(new_idx + 1);
                        lin_len_tree.set(new_idx, active[new_idx].cached_len);
                    }
                    // Incremental cache update.
                    if any_barrier && !cache_dirty {
                        if len_after_split > len_before_split {
                            // Specialised split path: skip recompute for
                            // pairs whose "other" lineage lies entirely
                            // on one side of the split point; move slot
                            // data rather than rerun compute_overlap.
                            rate_cache.apply_recomb_split(
                                chosen_idx, len_after_split - 1, x,
                                active, arena);
                        } else {
                            // No split happened (edge case): row idx
                            // still needs refresh.
                            rate_cache.recompute_for(chosen_idx, active, arena);
                        }
                    }
                    if any_barrier && !flux_dirty {
                        flux_update_for(chosen_idx, &mut flux_per_lin,
                                         &mut flux_total, active, inversions,
                                         arena, &barrier_active);
                        if len_after_split > len_before_split {
                            flux_push(len_after_split - 1, &mut flux_per_lin,
                                      &mut flux_total, active, inversions,
                                      arena, &barrier_active);
                        }
                    }
                    // GC sole-carrier lineages — only after recomb
                    // (matches Python). GC after coalescence is wrong:
                    // the merged lineage's solo bits (non-overlap parts
                    // from the two parents) still need to coalesce with
                    // others, but if no current other lineage covers
                    // them they get incorrectly discarded.
                    // Throttled: run every GC_STRIDE recombs. Sole
                    // carriers have zero coalescence rate, so delaying
                    // removal a few events is correctness-preserving.
                    gc_counter += 1;
                    if gc_counter >= GC_STRIDE {
                        gc_counter = 0;
                        let n_before_gc = active.len();
                        let removed = gc_sole_lineages_with_removed(active, arena);
                        if !removed.is_empty() {
                            total_material = active.iter()
                                .map(|l| l.cached_len).sum();
                            // Mirror each swap_remove on the auxiliary
                            // caches. `removed` is in descending order
                            // so bookkeeping stays monotone.
                            let mut len_snapshot = n_before_gc;
                            for &idx in &removed {
                                let last_idx = len_snapshot - 1;
                                if any_barrier && !cache_dirty {
                                    rate_cache.remove_lineage(idx);
                                    rate_cache.swap_update(idx, last_idx);
                                }
                                if any_barrier && !flux_dirty {
                                    flux_swap_remove(idx, &mut flux_per_lin,
                                                      &mut flux_total);
                                }
                                tree_swap_remove(&mut lin_len_tree,
                                                  idx, last_idx);
                                len_snapshot -= 1;
                            }
                        }
                    }
                }
                Event::FluxAggregate { inv_idx } => {
                    let ii = *inv_idx;
                    if ii >= flux_total.len() { continue; }
                    let total = flux_total[ii];
                    if total <= 0.0 { continue; }
                    // Weighted lineage pick from per-lineage flux cache.
                    let u: f64 = rng.random::<f64>() * total;
                    let mut running = 0.0;
                    let mut chose_li: Option<usize> = None;
                    for (li_idx, entries) in flux_per_lin.iter().enumerate() {
                        for (iii, rate) in entries.iter() {
                            if *iii == ii {
                                running += *rate;
                                if running >= u {
                                    chose_li = Some(li_idx);
                                    break;
                                }
                            }
                        }
                        if chose_li.is_some() { break; }
                    }
                    let li = match chose_li { Some(l) => l, None => continue };
                    let inv = &inversions[ii];
                    let pre_len_flux = active.len();
                    if let Some(x_event) = self.sample_flux_position(
                        active, li, inv, arena, rng)
                    {
                        let (tl, tr) = self.draw_tract(x_event, inv, rng);
                        if tr > tl {
                            apply_gene_flux(active, li, tl, tr, inv,
                                             arena, next_uid);
                        }
                        engine_dirty = true;
                        total_material = active.iter()
                            .map(|l| l.cached_len).sum();
                        lin_tree_dirty = true;
                        // Incremental rate_cache update: apply_gene_flux
                        // only mutates `li` plus any appended lineages.
                        // Recomputing those rows instead of the whole
                        // O(n² × segs) rebuild is a large win at rho ≥
                        // 1000 where flux events fire thousands of times.
                        let post_len = active.len();
                        if any_barrier && !cache_dirty {
                            rate_cache.recompute_for(li, active, arena);
                            for new_idx in pre_len_flux..post_len {
                                rate_cache.recompute_for(new_idx, active, arena);
                            }
                        }
                        if !flux_dirty {
                            flux_update_for(li, &mut flux_per_lin,
                                             &mut flux_total, active,
                                             inversions, arena, &barrier_active);
                            for new_idx in pre_len_flux..post_len {
                                flux_push(new_idx, &mut flux_per_lin,
                                          &mut flux_total, active,
                                          inversions, arena, &barrier_active);
                            }
                        }
                    }
                }
                Event::MigrationAggregate { src_pop, dst_pop } => {
                    let src = *src_pop;
                    let dst = *dst_pop;
                    let bucket = &pop_buckets[src as usize];
                    if bucket.is_empty() { continue; }
                    let pick = rng.random_range(0..bucket.len());
                    let idx = bucket[pick] as usize;
                    active[idx].population = dst;
                    engine_dirty = true;
                    if any_barrier && !flux_dirty && idx < flux_per_lin.len() {
                        flux_update_for(idx, &mut flux_per_lin,
                                         &mut flux_total, active, inversions,
                                         arena, &barrier_active);
                    }
                    if any_barrier && !cache_dirty {
                        rate_cache.recompute_for(idx, active, arena);
                    }
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

/// Per-lineage cached gene-flux rates: list of (inv_idx, rate) entries
/// for inversions where the lineage contributes non-zero hazard.
type FluxPerLin = SmallVec<[(usize, f64); 2]>;

/// Fill `out` (cleared first) with the lineage's flux entries across
/// all active inversions. In-place to avoid the ~48-byte SmallVec
/// return+push copy that dominated `flux_rebuild_full` pre-rewrite.
fn compute_lin_flux_into(
    out: &mut FluxPerLin,
    lin: &Lineage,
    inversions: &[InversionSpec],
    arena: &SegmentArena,
    barrier_active: &[bool],
) {
    out.clear();
    for (ii, inv) in inversions.iter().enumerate() {
        if !barrier_active[ii] { continue; }
        if inv.gene_conversion_rate <= 0.0 { continue; }
        let pop = lin.population;
        let p_inv_pop = inv.p_inv_for(pop);
        let p_std_pop = 1.0 - p_inv_pop;
        let kary = lineage_class_for_inv(lin, inv, arena);
        let p_other = match kary {
            Some(Karyotype::S) => p_inv_pop,
            Some(Karyotype::I) => p_std_pop,
            None => continue,
        };
        if p_other <= 0.0 { continue; }
        let w = flux_lineage_weight(lin, inv, arena);
        if w <= 0.0 { continue; }
        let rate = inv.gene_conversion_rate * p_other * w;
        if rate > 0.0 {
            out.push((ii, rate));
        }
    }
}

/// Rebuild the full flux cache from scratch — call on boundaries,
/// sweeps, GC, or any event that invalidates many lineages.
fn flux_rebuild_full(
    flux_per_lin: &mut Vec<FluxPerLin>,
    flux_total: &mut [f64],
    active: &[Lineage],
    inversions: &[InversionSpec],
    arena: &SegmentArena,
    barrier_active: &[bool],
) {
    // Fill entries in place to avoid the per-lineage SmallVec return +
    // push copy (~48 bytes) that showed up as ~20% of run_loop self-time
    // in the multi-pop flamegraph at rho=2000.
    flux_per_lin.resize_with(active.len(), FluxPerLin::new);
    flux_per_lin.truncate(active.len());
    for t in flux_total.iter_mut() { *t = 0.0; }
    for (i, lin) in active.iter().enumerate() {
        compute_lin_flux_into(&mut flux_per_lin[i], lin, inversions,
                               arena, barrier_active);
        for (ii, rate) in flux_per_lin[i].iter() {
            flux_total[*ii] += *rate;
        }
    }
}

/// Recompute flux for one lineage at `li` and update totals by diff.
fn flux_update_for(
    li: usize,
    flux_per_lin: &mut [FluxPerLin],
    flux_total: &mut [f64],
    active: &[Lineage],
    inversions: &[InversionSpec],
    arena: &SegmentArena,
    barrier_active: &[bool],
) {
    for (ii, rate) in flux_per_lin[li].iter() {
        flux_total[*ii] -= *rate;
    }
    compute_lin_flux_into(&mut flux_per_lin[li],
        &active[li], inversions, arena, barrier_active);
    for (ii, rate) in flux_per_lin[li].iter() {
        flux_total[*ii] += *rate;
    }
}

/// Mirror `active.swap_remove(idx)` on the flux cache: subtract removed
/// entries from totals and let Vec::swap_remove relocate the last slot.
fn flux_swap_remove(
    idx: usize,
    flux_per_lin: &mut Vec<FluxPerLin>,
    flux_total: &mut [f64],
) {
    if idx >= flux_per_lin.len() { return; }
    for (ii, rate) in flux_per_lin[idx].iter() {
        flux_total[*ii] -= *rate;
    }
    flux_per_lin.swap_remove(idx);
}

/// Append a new lineage's flux entries and credit its totals.
fn flux_push(
    li: usize,
    flux_per_lin: &mut Vec<FluxPerLin>,
    flux_total: &mut [f64],
    active: &[Lineage],
    inversions: &[InversionSpec],
    arena: &SegmentArena,
    barrier_active: &[bool],
) {
    // Grow then fill in place (vs push of a returned SmallVec).
    flux_per_lin.push(FluxPerLin::new());
    let last = flux_per_lin.len() - 1;
    compute_lin_flux_into(&mut flux_per_lin[last],
        &active[li], inversions, arena, barrier_active);
    for (ii, rate) in flux_per_lin[last].iter() {
        flux_total[*ii] += *rate;
    }
}

/// Mirror `active.swap_remove(idx)` on a length Fenwick by moving the
/// last slot's value into `idx` and zeroing the last slot. The tree's
/// logical size shrinks by 1 (trailing zero is inert for `find`).
#[inline]
fn tree_swap_remove(
    tree: &mut crate::fenwick::Fenwick,
    idx: usize,
    last_idx: usize,
) {
    if tree.len() == 0 { return; }
    if idx == last_idx {
        tree.set(idx, 0.0);
        return;
    }
    let val_last = tree.range_sum(last_idx, last_idx + 1);
    tree.set(idx, val_last);
    tree.set(last_idx, 0.0);
}

/// Remove lineages that are the sole carrier at every position they
/// cover — these can't produce more edges under SMC'.
///
/// Sweepline implementation: collect all segments tagged with owner,
/// Sweepline GC for sole-carrier lineages. Sorts all segments by left,
/// walks left-to-right maintaining an "open" set of segments whose
/// right > current left. A lineage has external overlap iff at some
/// point its open segment coexists with an open segment from a
/// different owner. Lineages never marked are sole-carriers and get
/// swap_removed. Returns the removed indices in descending order so
/// the caller can replay swap_removes on auxiliary caches.
fn gc_sole_lineages_with_removed(
    active: &mut Vec<Lineage>,
    arena: &SegmentArena,
) -> Vec<usize> {
    let n = active.len();
    if n <= 1 { return Vec::new(); }

    let mut segs: Vec<(f64, f64, u32)> = Vec::with_capacity(n * 2);
    for (i, lin) in active.iter().enumerate() {
        let mut cur = lin.head;
        while cur != SEG_NIL {
            let s = arena.get(cur);
            segs.push((s.left, s.right, i as u32));
            cur = s.next;
        }
    }
    if segs.is_empty() { return Vec::new(); }
    segs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut has_overlap = vec![false; n];
    let mut open: Vec<(f64, u32)> = Vec::with_capacity(64);
    for &(l, r, owner) in &segs {
        open.retain(|&(rr, _)| rr > l);
        let owner_idx = owner as usize;
        for &(_, o2) in &open {
            if o2 != owner {
                has_overlap[owner_idx] = true;
                has_overlap[o2 as usize] = true;
            }
        }
        open.push((r, owner));
    }

    let mut removed = Vec::new();
    for i in (0..n).rev() {
        if !has_overlap[i] {
            active.swap_remove(i);
            removed.push(i);
        }
    }
    removed
}

/// Emit aggregate coalescence events from the RateCache. Walks the
/// incrementally-maintained (pop, class, total_overlap) table — O(k)
/// where k = number of distinct (pop, class) combinations, typically
/// ≤ pops × 2^|inversions|. Dispatch samples a specific pair from
/// iter_pairs when the aggregate fires.
fn emit_coal_events_from_cache(
    cache: &RateCache,
    _active: &[Lineage],
    demo: &Demography,
    t: f64,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    events: &mut Vec<(f64, Event)>,
) {
    // Read counts from the incrementally-maintained class_totals table.
    // O(pops × classes) per emit — the main scaling win at rho ≥ 500.
    for (pop, cls, count) in cache.iter_class_totals() {
        if count <= 0.0 { continue; }
        let p_class = p_class_for_tag(cls, inversions, barrier_active, t, pop);
        if p_class <= 0.0 { continue; }
        let ne = demo.size_at(pop, t).max(1e-9);
        let rate = count / (2.0 * ne * p_class);
        events.push((rate, Event::CoalAggregate { pop, class: cls }));
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
    pop_buckets: &[Vec<u32>],
    events: &mut Vec<(f64, Event)>,
) {
    let any_inv_active = barrier_active.iter().any(|&b| b);

    if !any_inv_active {
        // Hudson per-pop coalescence rate — read counts directly from
        // the precomputed `pop_buckets` instead of rebuilding a local
        // association list with linear-scan finds (~6% of multi-pop
        // run_loop self-time).
        if demo.n_pops >= 2 {
            for (pop, bucket) in pop_buckets.iter().enumerate() {
                let k = bucket.len();
                if k < 2 { continue; }
                let ne = demo.size_at(pop as u32, t).max(1e-9);
                let kf = k as f64;
                let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
                events.push((rate,
                    Event::CoalPanmicticPop { pop: pop as u32 }));
            }
        } else {
            // Single-pop: no buckets built — use active.len() directly.
            let k = active.len();
            if k >= 2 {
                let ne = demo.size_at(0, t).max(1e-9);
                let kf = k as f64;
                let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
                events.push((rate, Event::CoalPanmicticPop { pop: 0 }));
            }
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
    sweep_cursor: &mut (f64, u64),
) {
    HullSimulator::cross_barriers_static(inversions, active, arena, barrier_active, t);
    let inv_changes = demo.apply_events_at(t, active);
    for (inv_id, pop, p_inv_val) in inv_changes {
        if let Some(inv) = inversions.iter_mut().find(|i| i.inv_id == inv_id) {
            inv.set_p_inv_for(pop, p_inv_val);
        }
    }
    // Drain all sweeps scheduled at this t (simultaneous sweeps).
    while !pending_sweeps.is_empty()
        && (pending_sweeps[0].t_event - t).abs() < 1e-9
    {
        let sweep = pending_sweeps.remove(0);
        let ne_sweep = demo.size_at(
            sweep.population.unwrap_or(0), t).max(1.0);
        apply_sweep(active, &sweep, t, arena, tables,
                     next_uid, seq_len, rng, ne_sweep, recomb_rate,
                     sweep_cursor);
    }
}

/// Monotonically increasing merge time, shared across all sweep merges
/// at the same base `t`. Resets when `t` changes.
fn next_sweep_merge_t(cursor: &mut (f64, u64), t: f64) -> f64 {
    if cursor.0 != t {
        *cursor = (t, 0);
    }
    cursor.1 += 1;
    let eps = (t * 1e-12).max(1e-9);
    t + (cursor.1 as f64) * eps
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
    sweep_cursor: &mut (f64, u64),
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
            active, sweep, t, arena, tables, next_uid, rng, ne, recomb_rate,
            sweep_cursor);
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
    coalesce_uid_group(active, &window_uids, t, arena, tables, next_uid,
                       sweep_cursor);
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
    sweep_cursor: &mut (f64, u64),
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
        coalesce_uid_group(active, &swept_uids, t, arena, tables, next_uid,
                           sweep_cursor);
    } else {
        // Soft sweep: randomly assign each swept lineage to one of K groups.
        let mut groups: Vec<Vec<LinUid>> = vec![Vec::new(); k];
        for &uid in &swept_uids {
            let g = (rng.random::<f64>() * k as f64) as usize;
            let g = g.min(k - 1); // clamp for floating-point edge case
            groups[g].push(uid);
        }
        // Coalesce within each group; shared cursor keeps merge times
        // strictly increasing across groups.
        for group in groups.iter() {
            if group.len() < 2 { continue; }
            coalesce_uid_group(active, group, t, arena, tables, next_uid,
                               sweep_cursor);
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
    sweep_cursor: &mut (f64, u64),
) {
    if uids.len() < 2 { return; }
    let mut merged_uid = uids[0];
    for &other_uid in uids[1..].iter() {
        let t_merge = next_sweep_merge_t(sweep_cursor, t);
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
