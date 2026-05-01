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
use crate::rate_index::{FlatSeg, RateCache};
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};
use crate::sweep::Sweep;
use crate::tables::TableBuilder;
use crate::event_log;

// ---------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------
pub struct SimResult {
    pub tables: TableBuilder,
    pub event_log: Option<event_log::EventLog>,
    /// Number of sample lineages assigned to the A (swept) haplotype
    /// at τ. Populated by `apply_sweep` tags; 0 when no sweeps fire.
    pub sweep_a_count: u64,
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

/// Allele subgroup for sweep-aware coalescence events. Used by
/// `CoalAggregate` to differentiate the progressive-coalescence
/// rates (A-tagged vs a-tagged vs untagged-involved) emitted during
/// the sweep window. Outside the sweep window every event is `Mixed`
/// and the consumer falls back to the standard "any pair" sampler.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AlleleTag {
    /// No sweep-window distinction; pair selection is the standard
    /// Hudson rule. Inside the sweep window represents the
    /// "untagged-involved" subgroup (UU, UA, Ua pairs).
    Mixed,
    /// Both lineages of the picked pair must be A-tagged.
    A,
    /// Both lineages of the picked pair must be a-tagged.
    ALower,
}

enum Event {
    /// Per-pair coalescence (used by the non-cache structured fallback
    /// `compute_coal_rates_structured`). Hot path uses CoalAggregate.
    CoalPair { i: usize, j: usize, class: BranchClass },
    /// Aggregate coalescence rate for all pairs in `pop` whose overlap
    /// lies in `class`. Firing samples the specific (i, j) pair from
    /// RateCache proportional to overlap length in that class — avoids
    /// the O(n^2) per-pair event-list entries that dominated rho≥500.
    CoalAggregate { pop: u32, class: BranchClass, allele: AlleleTag },
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
    /// Early-stop time (generations backward). Default f64::INFINITY
    /// — run to MRCA. Set to a finite value to return a partial
    /// tskit TreeSequence whose still-active lineages become the
    /// starting state for msprime recapitation via
    /// `sim_ancestry(initial_state=ts)`.
    pub stop_at: f64,
    /// Path 2 compound per-pair rate path. When true, dispatch
    /// coalescence via PairRateCache + apply_coalescence_compound,
    /// eliminating the remnant ratchet. MVP supports single-pop
    /// coal + recomb only; panics on flux, migration, sweep, or
    /// barrier crossings. Default false → production bucket path.
    pub compound_rate: bool,
    /// Event-loop iteration cap. On hit the loop returns a partial
    /// TreeSequence — recapitate with msprime. Default 10_000_000;
    /// raise (up to ~1e9) for runs expected to need more events
    /// before barrier era completes. If the cap hits while
    /// `t < max(inv.t_inv_max())` a warning is printed to stderr because
    /// the barrier era is incomplete and recap won't rescue it.
    pub iters_max: u64,
    /// Number of recomb events between `gc_sole_lineages` passes
    /// (msinv's analog of msprime mid-sim simplify). Default 160.
    /// Lower values shrink peak active-n during the barrier-era
    /// ratchet — each pass drops lineages whose material no longer
    /// overlaps any other lineage. Stride 16 roughly 10x more
    /// aggressive; stride 1 runs gc_sole on every recomb (highest
    /// overhead, most pruning). Downstream: fewer active lineages at
    /// `stop_at` means tractable Hudson recap in msprime.
    pub gc_stride: u32,
    /// If true, simulate_with_cache populates SimResult::event_log.
    /// Default false; production sims should leave this off.
    pub record_events: bool,
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
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
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
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
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

        // v1 sweep API supports a single sweep per simulation. Multiple
        // sweeps with overlapping windows would silently mis-apply
        // ne_cell scaling because emit_coal_events_from_cache uses
        // .find(|s| s.covers(t)). Out of v1 scope per
        // docs/superpowers/specs/2026-04-28-sweep-rewrite-design.md §"In scope (v1)".
        debug_assert!(self.sweeps.len() <= 1,
            "v1 sweep API supports a single sweep; got {}", self.sweeps.len());

        if self.compound_rate {
            panic!("compound_rate=True is experimental and disabled on main; \
                    the compound event loop lacks incremental flux + lineage-length \
                    caches and is slower than the bucket path at biological rho. \
                    Work continues on the feature/compound-caches branch.");
        }
        let mut event_log: Option<event_log::EventLog> =
            if self.record_events { Some(event_log::EventLog::new()) } else { None };
        let sweep_a_count = self.run_loop(
            &mut active, &mut arena, &mut tables,
            &mut next_uid, &mut rng, &mut demo,
            &mut inversions, rate_cache, event_log.as_mut());

        // Edge sort is left to the caller: the PyO3 bridge calls
        // `tables.sort_edges()` before handing columns to tskit so
        // `tc.sort()` can be skipped. Bench / test paths that just
        // read `SimResult::tables` skip the sort cost.
        SimResult { tables, event_log, sweep_a_count }
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
    /// Path 2 event loop. Coal via PairRateCache + compound merge,
    /// recomb via standard split, barrier crossings + demographic
    /// events via apply_boundary. Still panics on migration, flux,
    /// and sweeps — those are Stage 3c.2+ work.
    #[allow(dead_code)]
    fn run_loop_compound(
        &self,
        active: &mut Vec<crate::lineage::Lineage>,
        arena: &mut SegmentArena,
        tables: &mut TableBuilder,
        next_uid: &mut LinUid,
        rng: &mut Xoshiro256PlusPlus,
        demo: &mut crate::demography::Demography,
        inversions: &mut Vec<InversionSpec>,
    ) {
        use crate::events::{apply_coalescence_compound, apply_recombination};
        use crate::pair_rate_cache::PairRateCache;
        use crate::class_tag::Karyotype;

        let mut barrier_active: Vec<bool> = inversions.iter().map(|_| true).collect();
        let mut pending_sweeps: Vec<Sweep> = self.sweeps.clone();
        pending_sweeps.sort_by(|a, b| a.tau.partial_cmp(&b.tau).unwrap());
        // Snapshot demography state into each sweep's trajectory closures
        // BEFORE any apply_events_at calls fire — otherwise the snapshot
        // would capture mutated pop_sizes rather than t=0 values.
        populate_sweep_trajectories(&mut pending_sweeps, demo, inversions);
        let mut finalized_sweeps: Vec<Sweep> = Vec::new();
        let mut sweep_cursor: (f64, u64) = (f64::NAN, 0);
        let mut a_tag: std::collections::HashMap<LinUid, bool> = std::collections::HashMap::new();

        let mut t: f64 = 0.0;
        let max_lins = (active.len() * 40).max(2048);
        let mut pair_rates = PairRateCache::new(max_lins);
        pair_rates.rebuild(active, arena, inversions, &barrier_active,
                            demo, t, self.sequence_length);

        let mut total_material: f64 = active.iter().map(|l| l.cached_len).sum();
        let gc_stride = self.gc_stride.max(1);
        let mut gc_counter: u32 = 0;

        // Per-pop counts refreshed each iter for migration aggregation.
        // Cheap to recompute (O(n)); migration is multi-pop-only and
        // typically n_pops is small, so no incremental upkeep needed.
        let n_pops = demo.n_pops as usize;
        let mut pop_counts: Vec<u32> = vec![0; n_pops];

        for _ in 0..self.iters_max {
            let n = active.len();
            if n <= 1 { return; }
            if t >= self.stop_at { return; }

            // Earliest deterministic boundary.
            let mut earliest_barrier = f64::INFINITY;
            for (k, inv) in inversions.iter().enumerate() {
                if barrier_active[k] {
                    earliest_barrier = earliest_barrier.min(inv.t_inv_max());
                }
            }
            let t_demo = demo.next_event_time(t);
            let t_sweep = pending_sweeps.first()
                .map(|s| s.tau).unwrap_or(f64::INFINITY);
            let t_sweep_origin = finalized_sweeps.first()
                .map(|s| s.t_de_novo()).unwrap_or(f64::INFINITY);
            let next_boundary = earliest_barrier.min(t_demo).min(t_sweep).min(t_sweep_origin);

            let coal_rate = pair_rates.total();
            let recomb_rate = total_material * self.recombination_rate;
            // Per-(lineage, inv) flux rate. Flattened as (li, ii, rate).
            // Linear re-computation each iter — acceptable at realistic
            // rates where flux is a minor fraction of events.
            let mut flux_entries: Vec<(usize, usize, f64)> = Vec::new();
            let mut flux_rate: f64 = 0.0;
            for (ii, inv) in inversions.iter().enumerate() {
                if !barrier_active[ii] { continue; }
                if inv.gene_conversion_rate <= 0.0 { continue; }
                for li in 0..active.len() {
                    let r = flux_lineage_rate_arena(
                        active[li].head, inv,
                        active[li].population, t, arena);
                    if r > 0.0 {
                        flux_entries.push((li, ii, r));
                        flux_rate += r;
                    }
                }
            }
            // Migration aggregate rate across all (src, dst).
            let mig_rate = if n_pops >= 2 {
                for c in pop_counts.iter_mut() { *c = 0; }
                for lin in active.iter() {
                    pop_counts[lin.population as usize] += 1;
                }
                let mut r = 0.0;
                for src in 0..n_pops {
                    let cnt = pop_counts[src] as f64;
                    if cnt == 0.0 { continue; }
                    for dst in 0..n_pops {
                        if dst == src { continue; }
                        r += cnt * demo.migration_matrix[dst][src];
                    }
                }
                r
            } else { 0.0 };
            let total_rate = coal_rate + recomb_rate + mig_rate + flux_rate;

            // No coalescence possible + nothing but recomb firing →
            // sweep out sole-carrier lineages (no overlap with anyone).
            // Without this, recomb-on-fragments loop unboundedly at
            // boundaries where all remaining material is non-shared.
            if coal_rate <= 0.0 && recomb_rate > 0.0 && active.len() > 1 {
                let removed = gc_sole_lineages_with_removed(active, arena);
                if !removed.is_empty() {
                    pair_rates.rebuild(active, arena, inversions, &barrier_active,
                                        demo, t, self.sequence_length);
                    total_material = active.iter().map(|l| l.cached_len).sum();
                    continue;
                }
            }

            // Zero-rate case: jump straight to next boundary if any.
            if total_rate <= 0.0 {
                if next_boundary < f64::INFINITY {
                    t = next_boundary;
                    apply_boundary(
                        inversions, active, arena, &mut barrier_active,
                        demo, &mut pending_sweeps, &mut finalized_sweeps, t, tables, next_uid,
                        self.sequence_length, rng, self.recombination_rate,
                        &mut sweep_cursor, None, &mut a_tag);
                    pair_rates.rebuild(active, arena, inversions, &barrier_active,
                                        demo, t, self.sequence_length);
                    total_material = active.iter().map(|l| l.cached_len).sum();
                    continue;
                }
                return;
            }

            let dt = -rng.random::<f64>().ln() / total_rate;
            let t_event = t + dt;

            // Deterministic boundary fires before stochastic event.
            if next_boundary <= t_event {
                t = next_boundary;
                apply_boundary(
                    inversions, active, arena, &mut barrier_active,
                    demo, &mut pending_sweeps, &mut finalized_sweeps, t, tables, next_uid,
                    self.sequence_length, rng, self.recombination_rate,
                    &mut sweep_cursor, None, &mut a_tag);
                pair_rates.rebuild(active, arena, inversions, &barrier_active,
                                    demo, t, self.sequence_length);
                total_material = active.iter().map(|l| l.cached_len).sum();
                continue;
            }
            t = t_event;

            let u: f64 = rng.random::<f64>() * total_rate;
            if u >= coal_rate + recomb_rate + mig_rate {
                // Gene flux: sample (li, ii) proportional.
                let mut u_flux = u - coal_rate - recomb_rate - mig_rate;
                let mut picked: Option<(usize, usize)> = None;
                for &(li, ii, r) in &flux_entries {
                    if u_flux < r { picked = Some((li, ii)); break; }
                    u_flux -= r;
                }
                let (li, ii) = match picked {
                    Some(p) => p,
                    None => continue,
                };
                let inv = &inversions[ii];
                let pre_len = active.len();
                if let Some(x_event) = self.sample_flux_position(
                    active, li, inv, arena, rng)
                {
                    let (tl, tr) = self.draw_tract(x_event, inv, rng);
                    if tr > tl {
                        apply_gene_flux(active, li, tl, tr, inv,
                                         arena, next_uid, None, t, x_event);
                    }
                }
                let post_len = active.len();
                total_material = active.iter().map(|l| l.cached_len).sum();
                pair_rates.recompute_for(
                    li, active, arena, inversions,
                    &barrier_active, demo, t, self.sequence_length);
                for new_idx in pre_len..post_len {
                    pair_rates.recompute_for(
                        new_idx, active, arena, inversions,
                        &barrier_active, demo, t, self.sequence_length);
                }
                continue;
            }
            if u >= coal_rate + recomb_rate {
                // Migration: sample (src, dst) then a lineage in src.
                let mut u_mig = u - coal_rate - recomb_rate;
                let mut picked: Option<(u32, u32)> = None;
                'outer: for src in 0..n_pops {
                    let cnt = pop_counts[src] as f64;
                    if cnt == 0.0 { continue; }
                    for dst in 0..n_pops {
                        if dst == src { continue; }
                        let r = cnt * demo.migration_matrix[dst][src];
                        if u_mig < r {
                            picked = Some((src as u32, dst as u32));
                            break 'outer;
                        }
                        u_mig -= r;
                    }
                }
                let (src, dst) = match picked {
                    Some(p) => p,
                    None => continue,
                };
                // Uniformly pick one lineage in src.
                let target = rng.random_range(0..pop_counts[src as usize] as usize);
                let mut seen = 0usize;
                let mut chosen_idx = 0usize;
                for (i, lin) in active.iter().enumerate() {
                    if lin.population == src {
                        if seen == target { chosen_idx = i; break; }
                        seen += 1;
                    }
                }
                active[chosen_idx].population = dst;
                pair_rates.recompute_for(
                    chosen_idx, active, arena, inversions,
                    &barrier_active, demo, t, self.sequence_length);
                continue;
            }
            if u < coal_rate {
                // Coalescence.
                let (i, j) = match pair_rates.sample_pair(u) {
                    Some(p) => p,
                    None => continue,
                };
                let pre_len = active.len();
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                let old_i_len = active[i].cached_len;
                let old_j_len = active[j].cached_len;
                apply_coalescence_compound(
                    active, i, j, t, arena, tables, next_uid,
                    Some(&mut a_tag));
                let post_len = active.len();
                let mut delta = -old_i_len - old_j_len;
                for new_idx in (pre_len - 2)..post_len {
                    delta += active[new_idx].cached_len;
                }
                total_material += delta;

                pair_rates.remove_lineage(hi, pre_len);
                pair_rates.swap_update(hi, pre_len - 1);
                pair_rates.remove_lineage(lo, pre_len - 1);
                pair_rates.swap_update(lo, pre_len - 2);
                for new_idx in (pre_len - 2)..post_len {
                    pair_rates.recompute_for(
                        new_idx, active, arena, inversions,
                        &barrier_active, demo, t, self.sequence_length);
                }
            } else {
                // Recombination.
                let u_lin: f64 = rng.random::<f64>() * total_material;
                let mut acc = 0.0;
                let mut chosen_idx = active.len() - 1;
                for (k, lin) in active.iter().enumerate() {
                    acc += lin.cached_len;
                    if acc > u_lin { chosen_idx = k; break; }
                }
                let lin_len = active[chosen_idx].cached_len;
                if lin_len <= 0.0 { continue; }
                let x_offset: f64 = rng.random::<f64>() * lin_len;
                let x = find_position(
                    active, chosen_idx, x_offset, arena,
                    self.sequence_length);
                let len_before = active.len();
                apply_recombination(
                    active, chosen_idx, x, arena, next_uid,
                    Some(&mut a_tag));
                let len_after = active.len();
                {
                    let mut swap_indices: SmallVec<[usize; 2]> = SmallVec::new();
                    swap_indices.push(chosen_idx);
                    if len_after > len_before {
                        swap_indices.push(len_after - 1);
                    }
                    apply_sweep_recomb_tag_swap(
                        active, &swap_indices, &finalized_sweeps,
                        t, arena, rng, &mut a_tag);
                }
                pair_rates.recompute_for(
                    chosen_idx, active, arena, inversions,
                    &barrier_active, demo, t, self.sequence_length);
                if len_after > len_before {
                    pair_rates.recompute_for(
                        len_after - 1, active, arena, inversions,
                        &barrier_active, demo, t, self.sequence_length);
                }
                gc_counter += 1;
                if gc_counter >= gc_stride {
                    gc_counter = 0;
                    let removed = gc_sole_lineages_with_removed(active, arena);
                    if !removed.is_empty() {
                        pair_rates.rebuild(active, arena, inversions, &barrier_active,
                                            demo, t, self.sequence_length);
                        total_material = active.iter().map(|l| l.cached_len).sum();
                    }
                }
            }
        }
        self.warn_cap_hit(t, inversions, "compound");
    }

    /// Emit a warning when the event-loop iter cap fires. Most
    /// dangerous case: `t` is still below `max(inv.t_inv_max())`, meaning
    /// msinv never finished the barrier era — downstream recapitation
    /// can't rescue a truncated barrier phase (msprime has no
    /// inversion concept). Print to stderr so pilots / tests notice.
    fn warn_cap_hit(&self, t: f64, inversions: &[InversionSpec], path: &'static str) {
        let max_t_inv = inversions.iter()
            .map(|inv| inv.t_inv_max())
            .fold(0.0_f64, f64::max);
        if t < max_t_inv {
            eprintln!(
                "[msinv WARN] {path} loop hit iters_max={} at t={t:.0} \
                 but max(t_inv)={max_t_inv:.0} — barrier era INCOMPLETE. \
                 Raise HullSimulator.iters_max; do not trust recap.",
                self.iters_max);
        } else if t < self.stop_at && t < f64::INFINITY {
            eprintln!(
                "[msinv WARN] {path} loop hit iters_max={} at t={t:.0} \
                 (barrier era complete). Returned partial ARG — recap \
                 with msprime to finish.",
                self.iters_max);
        }
    }

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
        mut event_log: Option<&mut event_log::EventLog>,
    ) -> u64 {
        let mut t: f64 = 0.0;

        // Track which inversions' barriers are still active.
        let mut barrier_active: Vec<bool> = inversions.iter()
            .map(|_| true).collect();

        // Pending sweeps, sorted by tau (earliest first).
        let mut pending_sweeps: Vec<Sweep> = self.sweeps.clone();
        pending_sweeps.sort_by(|a, b| a.tau.partial_cmp(&b.tau).unwrap());
        populate_sweep_trajectories(&mut pending_sweeps, demo, inversions);
        let mut finalized_sweeps: Vec<Sweep> = Vec::new();

        // Monotone sweep-merge cursor shared across all sweeps at the
        // same base t (prevents TSK_ERR_BAD_NODE_TIME_ORDERING when two
        // sweeps fire simultaneously).
        let mut sweep_cursor: (f64, u64) = (f64::NAN, 0);
        let mut a_tag: std::collections::HashMap<LinUid, bool> = std::collections::HashMap::new();
        // Record sample UIDs before any recombination/coalescence so we can
        // count only the *initial sample lineages* that were tagged A at τ.
        // Using a_tag.values().count() would also count recombination children
        // (which inherit the flag via propagate_a_flag_recomb), inflating the
        // count above n_samples.
        let sample_uids: Vec<LinUid> = active.iter().map(|l| l.uid).collect();
        let count_a_samples = |map: &std::collections::HashMap<LinUid, bool>| -> u64 {
            sample_uids.iter()
                .filter(|uid| map.get(uid).copied().unwrap_or(false))
                .count() as u64
        };

        // Running totals for O(1) recombination rate (Phase A).
        let mut total_material: f64 = active.iter()
            .map(|l| l.cached_len).sum();
        let mut total_recomb_rate: f64 = total_material * self.recombination_rate;

        // Phase D: incremental pair rate cache. Pre-size generously:
        // `pair_idx` is capacity-dependent, so every `ensure_capacity`
        // growth must reindex (O(n²)) to preserve correctness of the
        // pair buckets. Oversizing up front avoids most mid-run grows
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
        let gc_stride = self.gc_stride.max(1);
        let mut gc_counter: u32 = 0;

        for _ in 0..self.iters_max {
            // Optional early stop (used by msprime-recapitation wrapper):
            // simulate up to stop_at time, then return partial TS.
            if t >= self.stop_at { break; }
            let n = active.len();
            if n <= 1 {
                if n == 0 || active[0].total_length(arena)
                    >= self.sequence_length - 1e-9
                {
                    return count_a_samples(&a_tag);
                }
                return count_a_samples(&a_tag);
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
                    rate_cache,
                    active, inversions, arena, &barrier_active, t);
                flux_dirty = false;
            }

            // Check for barrier crossings.
            let mut any_barrier = false;
            let mut earliest_barrier = f64::INFINITY;
            for (k, inv) in inversions.iter().enumerate() {
                if barrier_active[k] {
                    any_barrier = true;
                    earliest_barrier = earliest_barrier.min(inv.t_inv_max());
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

                // Coalescence. Always drive through the rate_cache so
                // post-barrier Hudson rate uses actual overlap counts
                // (not k(k-1)/2 with rejection sampling which burns
                // ~1e9 no-op iters at Hudson equilibrium n~120k).
                if cache_dirty {
                    rate_cache.rebuild(active, arena);
                    cache_dirty = false;
                }
                let active_sweep: Option<&Sweep> = finalized_sweeps.iter()
                    .find(|s| s.covers(t));
                emit_coal_events_from_cache(
                    &rate_cache, active, arena, &*demo, t,
                    inversions, &barrier_active,
                    &mut all_events, active_sweep, &a_tag);

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
                .map(|s| s.tau).unwrap_or(f64::INFINITY);
            let t_sweep_origin = finalized_sweeps.first()
                .map(|s| s.t_de_novo()).unwrap_or(f64::INFINITY);

            // Next deterministic boundary.
            let next_boundary = earliest_barrier.min(t_demo).min(t_sweep).min(t_sweep_origin);

            if total_rate <= 0.0 {
                if next_boundary < f64::INFINITY {
                    t = next_boundary;
                    apply_boundary(
                        inversions, active, arena, &mut barrier_active,
                        demo, &mut pending_sweeps, &mut finalized_sweeps, t, tables, next_uid,
                        self.sequence_length, rng, self.recombination_rate,
                        &mut sweep_cursor,
                        event_log.as_deref_mut(), &mut a_tag);
                    total_material = active.iter()
                        .map(|l| l.cached_len).sum();
                    total_recomb_rate = total_material * self.recombination_rate;
                    engine_dirty = true;
                    cache_dirty = true;
                    lin_tree_dirty = true;
                    flux_dirty = true;
                    continue;
                }
                return count_a_samples(&a_tag);
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
                    demo, &mut pending_sweeps, &mut finalized_sweeps, t, tables, next_uid,
                    self.sequence_length, rng, self.recombination_rate,
                    &mut sweep_cursor,
                    event_log.as_deref_mut(), &mut a_tag);
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
                Event::CoalAggregate { pop, class, allele } => {
                    let pop = *pop;
                    let cls = *class;
                    let allele = *allele;
                    // Direct O(1) pick from the (pop, cls) pair bucket:
                    // maintained by every overlap mutation, indexed by
                    // packed (i, j). Replaces the iter_pairs walk that
                    // was ~15% of wall at rho=2000 (bitmap advance +
                    // class scan per match). Bucket length doubles as
                    // the (pop, cls) pair count feeding CoalAggregate.
                    let bucket = rate_cache.pair_bucket_for(pop, cls);
                    if bucket.is_empty() { continue; }
                    // Fast path: outside any active sweep window, every
                    // event the emitter produced was Mixed with the
                    // standard "any pair" rate. `a_tag` may still hold
                    // entries from a finished sweep (used by
                    // count_a_samples for sweep_a_count accounting), so
                    // gate on the live sweep state instead of
                    // a_tag.is_empty().
                    let in_any_sweep = finalized_sweeps.iter().any(|s| s.covers(t));
                    let (i, j) = if !in_any_sweep && matches!(allele, AlleleTag::Mixed) {
                        let target = rng.random_range(0..bucket.len());
                        crate::rate_index::unpack_ij(bucket[target])
                    } else {
                        // Inside the sweep window: filter pairs to match
                        // the event's allele subgroup. PG-B1 emits three
                        // events per swept cell with rates derived from
                        // (n_A choose 2), (n_a choose 2), and the count
                        // of untagged-involved pairs — the consumer must
                        // honor those subgroups when sampling. Cost is
                        // O(|bucket|) per fired event — bounded by O(n^2).
                        let matches_pair = |packed: u64| -> bool {
                            let (i, j) = crate::rate_index::unpack_ij(packed);
                            let i_tag = a_tag.get(&active[i].uid).copied();
                            let j_tag = a_tag.get(&active[j].uid).copied();
                            match allele {
                                AlleleTag::A => {
                                    i_tag == Some(true) && j_tag == Some(true)
                                }
                                AlleleTag::ALower => {
                                    i_tag == Some(false) && j_tag == Some(false)
                                }
                                AlleleTag::Mixed => {
                                    // Untagged-involved: at least one of
                                    // the two lineages is untagged.
                                    i_tag.is_none() || j_tag.is_none()
                                }
                            }
                        };
                        let matching: SmallVec<[u64; 32]> = bucket.iter()
                            .copied().filter(|&p| matches_pair(p)).collect();
                        if matching.is_empty() { continue; }
                        let target = rng.random_range(0..matching.len());
                        crate::rate_index::unpack_ij(matching[target])
                    };
                    let pre_len = active.len();
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    let old_i_len = active[i].cached_len;
                    let old_j_len = active[j].cached_len;
                    // Post-barrier (all inversions inactive) everything
                    // is PAN, so fall into Hudson's full merge: non-
                    // overlap segments fold into merged, lineage count
                    // drops by 1 per event. Barrier-era Some(cls) keeps
                    // class-mismatched regions on remainder lineages so
                    // S/I can't coalesce via a PAN-class event.
                    let allowed = if any_barrier {
                        Some(cls)
                    } else {
                        None
                    };
                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        allowed, Some(&mut a_tag));
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

                    if !cache_dirty {
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
                            if cache_dirty {
                                rate_cache.rebuild_segs_for(new_idx, active, arena);
                            }
                            let pop = active[new_idx].population;
                            let segs = rate_cache.lineage_segs(new_idx);
                            flux_push(&mut flux_per_lin,
                                      &mut flux_total, segs, pop,
                                      inversions, &barrier_active, t);
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
                    // See CoalAggregate above — post-barrier wants
                    // Hudson full merge.
                    let allowed = if any_barrier {
                        Some(cls)
                    } else {
                        None
                    };
                    apply_coalescence_partial(
                        active, i, j, t, arena, tables, next_uid,
                        allowed, Some(&mut a_tag));
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

                    if !cache_dirty {
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
                            if cache_dirty {
                                rate_cache.rebuild_segs_for(new_idx, active, arena);
                            }
                            let pop = active[new_idx].population;
                            let segs = rate_cache.lineage_segs(new_idx);
                            flux_push(&mut flux_per_lin,
                                      &mut flux_total, segs, pop,
                                      inversions, &barrier_active, t);
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
                        // Hull overlap is necessary but not sufficient
                        // for fragmented chains with gaps. Skip pairs
                        // with no real segment overlap — else we'd
                        // create an orphan node and a fictitious merge.
                        if !segments_overlap(
                            active[a].head, active[b].head, arena) {
                            continue;
                        }
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        let old_a_len = active[a].cached_len;
                        let old_b_len = active[b].cached_len;
                        let pre_len = active.len();
                        apply_coalescence(
                            active, a, b, t, arena,
                            tables, next_uid, Some(&mut a_tag));
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
                                rate_cache.rebuild_segs_for(new_idx, active, arena);
                                let pop = active[new_idx].population;
                                let segs = rate_cache.lineage_segs(new_idx);
                                flux_push(&mut flux_per_lin,
                                          &mut flux_total, segs, pop,
                                          inversions, &barrier_active, t);
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
                                         next_uid, Some(&mut a_tag));
                    let len_after_split = active.len();
                    // Discoal-style tag rejection-sampling for in-window
                    // recombs. `chosen_idx` is the [head, x) child; if a
                    // split actually happened, `len_after_split - 1` is
                    // the [x, tail) child.
                    {
                        let mut swap_indices: SmallVec<[usize; 2]> = SmallVec::new();
                        swap_indices.push(chosen_idx);
                        if len_after_split > len_before_split {
                            swap_indices.push(len_after_split - 1);
                        }
                        apply_sweep_recomb_tag_swap(
                            active, &swap_indices, &finalized_sweeps,
                            t, arena, rng, &mut a_tag);
                    }
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
                    if !cache_dirty {
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
                        if cache_dirty {
                            rate_cache.rebuild_segs_for(chosen_idx, active, arena);
                            if len_after_split > len_before_split {
                                rate_cache.rebuild_segs_for(
                                    len_after_split - 1, active, arena);
                            }
                        }
                        let pop_c = active[chosen_idx].population;
                        let segs_c = rate_cache.lineage_segs(chosen_idx);
                        flux_update_for(chosen_idx, &mut flux_per_lin,
                                         &mut flux_total, segs_c, pop_c,
                                         inversions, &barrier_active, t);
                        if len_after_split > len_before_split {
                            let new_idx = len_after_split - 1;
                            let pop_n = active[new_idx].population;
                            let segs_n = rate_cache.lineage_segs(new_idx);
                            flux_push(&mut flux_per_lin,
                                      &mut flux_total, segs_n, pop_n,
                                      inversions, &barrier_active, t);
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
                    if gc_counter >= gc_stride {
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
                                if !cache_dirty {
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
                                             arena, next_uid,
                                             event_log.as_deref_mut(), t, x_event);
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
                        if !cache_dirty {
                            rate_cache.recompute_for(li, active, arena);
                            for new_idx in pre_len_flux..post_len {
                                rate_cache.recompute_for(new_idx, active, arena);
                            }
                        }
                        if !flux_dirty {
                            if cache_dirty {
                                rate_cache.rebuild_segs_for(li, active, arena);
                                for new_idx in pre_len_flux..post_len {
                                    rate_cache.rebuild_segs_for(new_idx, active, arena);
                                }
                            }
                            let pop_li = active[li].population;
                            let segs_li = rate_cache.lineage_segs(li);
                            flux_update_for(li, &mut flux_per_lin,
                                             &mut flux_total, segs_li, pop_li,
                                             inversions, &barrier_active, t);
                            for new_idx in pre_len_flux..post_len {
                                let pop_n = active[new_idx].population;
                                let segs_n = rate_cache.lineage_segs(new_idx);
                                flux_push(&mut flux_per_lin,
                                          &mut flux_total, segs_n, pop_n,
                                          inversions, &barrier_active, t);
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
                        if cache_dirty {
                            rate_cache.rebuild_segs_for(idx, active, arena);
                        }
                        let pop = active[idx].population;
                        let segs = rate_cache.lineage_segs(idx);
                        flux_update_for(idx, &mut flux_per_lin,
                                         &mut flux_total, segs, pop,
                                         inversions, &barrier_active, t);
                    }
                    if !cache_dirty {
                        rate_cache.recompute_for(idx, active, arena);
                    }
                }
            }

            // Keep recomb rate in sync.
            total_recomb_rate = total_material * self.recombination_rate;
        }
        self.warn_cap_hit(t, inversions, "bucket");
        count_a_samples(&a_tag)
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
        let w = inv.mean_tract_length / inv.length();
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
        use rand_distr::{Distribution, Exp};
        let inv_len = inv.length();

        if inv.mean_tract_length <= 0.0 {
            return (x_event, x_event);
        }

        let l = match inv.tract_distribution {
            crate::inversion::TractDistribution::Fixed => inv.mean_tract_length,
            crate::inversion::TractDistribution::Geometric => {
                let exp = Exp::new(1.0 / inv.mean_tract_length).expect("mean > 0 by guard above");
                exp.sample(rng)
            }
        };
        let l = l.min(inv_len * 0.99);

        let x_rel = x_event - inv.bp_left;
        let b1_lo = (x_rel - l).max(0.0);
        let b1_hi = (x_rel).min(inv_len - l);
        let b1 = if b1_hi <= b1_lo {
            (x_rel - l / 2.0).clamp(0.0, inv_len - l)
        } else {
            rng.random::<f64>() * (b1_hi - b1_lo) + b1_lo
        };
        let tl = (inv.bp_left + b1).max(inv.bp_left);
        let tr = (tl + l).min(inv.bp_right);
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
            if barrier_active[k] && t >= inv.t_inv_max() {
                barrier_active[k] = false;
                // At t_inv: every I-class segment morphs into S (the inversion
                // mutation arose from an S precursor; going past t_inv we're
                // pre-inversion era, lineage lives in the S subpopulation).
                // S-class stays S.  Class label is preserved (not cleared to
                // PAN) so it stays meaningful for any class-conditional
                // events that fire later.
                for lin in active.iter() {
                    let mut cur = lin.head;
                    while cur != SEG_NIL {
                        let seg = arena.get_mut(cur);
                        if let Some(Karyotype::I) = seg.branch_class.get_inv(inv.inv_id) {
                            seg.branch_class = seg.branch_class
                                .with_inv(inv.inv_id, Karyotype::S);
                        }
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
/// Reads the lineage's segment view as a flat `&[FlatSeg]` slice
/// (provided by `RateCache::lineage_segs`) so the per-inversion walk
/// hits sequential memory instead of chasing arena indices.
fn compute_lin_flux_into(
    out: &mut FluxPerLin,
    segs: &[FlatSeg],
    t: f64,
    pop: u32,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
) {
    out.clear();
    for (ii, inv) in inversions.iter().enumerate() {
        if !barrier_active[ii] { continue; }
        if inv.gene_conversion_rate <= 0.0 { continue; }
        let rate = flux_lineage_rate_segs(segs, inv, pop, t);
        if rate > 0.0 {
            out.push((ii, rate));
        }
    }
}

/// Rebuild the full flux cache from scratch — call on boundaries,
/// sweeps, GC, or any event that invalidates many lineages. Refreshes
/// `rate_cache.lineage_segs` first so the per-lineage inner loop reads
/// flat slices.
fn flux_rebuild_full(
    flux_per_lin: &mut Vec<FluxPerLin>,
    flux_total: &mut [f64],
    rate_cache: &mut RateCache,
    active: &[Lineage],
    inversions: &[InversionSpec],
    arena: &SegmentArena,
    barrier_active: &[bool],
    t: f64,
) {
    flux_per_lin.resize_with(active.len(), FluxPerLin::new);
    flux_per_lin.truncate(active.len());
    for entry in flux_total.iter_mut() { *entry = 0.0; }
    // Pre-barrier epochs have no flux; skip the O(n · segs) refresh and
    // leave every entry empty. Matches prior behaviour where the inner
    // `barrier_active` guard short-circuited each lineage's inv loop.
    let any_barrier_active = barrier_active.iter().any(|&b| b);
    if !any_barrier_active {
        for entry in flux_per_lin.iter_mut() { entry.clear(); }
        return;
    }
    rate_cache.refresh_lineage_segs(active, arena);
    for (i, lin) in active.iter().enumerate() {
        let segs = rate_cache.lineage_segs(i);
        compute_lin_flux_into(&mut flux_per_lin[i], segs, t, lin.population,
                               inversions, barrier_active);
        for (ii, rate) in flux_per_lin[i].iter() {
            flux_total[*ii] += *rate;
        }
    }
}

/// Recompute flux for one lineage at `li` and update totals by diff.
/// Caller must have refreshed `rate_cache.lineage_segs(li)` if it
/// could be stale (see `RateCache::rebuild_segs_for`).
fn flux_update_for(
    li: usize,
    flux_per_lin: &mut [FluxPerLin],
    flux_total: &mut [f64],
    segs: &[FlatSeg],
    pop: u32,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    t: f64,
) {
    for (ii, rate) in flux_per_lin[li].iter() {
        flux_total[*ii] -= *rate;
    }
    compute_lin_flux_into(&mut flux_per_lin[li], segs, t, pop,
                          inversions, barrier_active);
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

/// Append a new lineage's flux entries and credit its totals. Caller
/// supplies the fresh `segs` slice (typically `rate_cache.lineage_segs`
/// after an appropriate rebuild) and the lineage's population.
fn flux_push(
    flux_per_lin: &mut Vec<FluxPerLin>,
    flux_total: &mut [f64],
    segs: &[FlatSeg],
    pop: u32,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    t: f64,
) {
    flux_per_lin.push(FluxPerLin::new());
    let last = flux_per_lin.len() - 1;
    compute_lin_flux_into(&mut flux_per_lin[last], segs, t, pop,
                          inversions, barrier_active);
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
///
/// When `active_sweep` is Some and `t` is inside the sweep window for
/// the swept (pop, kary) cell, the denominator switches to
/// `2 * ne_cell(t, pop, kary)` from the trajectory instead of
/// `2 * ne * p_class`. This drives the Kim-Stephan coalescence footprint.
fn emit_coal_events_from_cache(
    cache: &RateCache,
    active: &[Lineage],
    arena: &SegmentArena,
    demo: &Demography,
    t: f64,
    inversions: &[InversionSpec],
    barrier_active: &[bool],
    events: &mut Vec<(f64, Event)>,
    active_sweep: Option<&Sweep>,
    a_tag: &std::collections::HashMap<LinUid, bool>,
) {
    // Read pair counts directly from the pair_buckets — each bucket's
    // length is the (pop, cls) pair count. O(pops × classes) per emit.
    for (pop, cls, count) in cache.iter_class_totals() {
        if count <= 0.0 { continue; }
        let p_class = p_class_for_tag(cls, inversions, barrier_active, t, pop);
        if p_class <= 0.0 { continue; }
        let ne = demo.size_at(pop, t).max(1e-9);

        // Outside sweep window OR sweep is in a different population:
        // emit a single Mixed event with the standard rate.
        let in_sweep_cell = match active_sweep {
            Some(sw) => sw.covers(t) && sw.origin_pop == pop,
            None => false,
        };
        if !in_sweep_cell {
            events.push((count / (2.0 * ne * p_class),
                Event::CoalAggregate { pop, class: cls, allele: AlleleTag::Mixed }));
            continue;
        }

        // Inside the sweep window in the swept population. Bucketize
        // active lineages in this (pop, cls) cell by allele tag.
        // Cost: O(|active|) per emit while a sweep is active.
        let mut n_a_upper: usize = 0;
        let mut n_a_lower: usize = 0;
        let mut n_untagged: usize = 0;
        for lin in active.iter() {
            if lin.population != pop { continue; }
            if !lineage_has_class(lin.head, cls, arena) { continue; }
            match a_tag.get(&lin.uid).copied() {
                Some(true)  => n_a_upper += 1,
                Some(false) => n_a_lower += 1,
                None        => n_untagged += 1,
            }
        }

        // Determine kary for the swept inversion at this class. For
        // panmictic-at-this-locus (no kary tag on target_inv), fall
        // back to origin_kary so the trajectory queries still engage.
        let sw = active_sweep.expect("in_sweep_cell implies Some(sw)");
        let kary = cls.get_inv(sw.target_inv).unwrap_or(sw.origin_kary);
        let p_kary = sw.trajectory.as_ref()
            .map_or(p_class, |traj| traj.p_kary(t, pop, kary));
        let p_a = sw.trajectory.as_ref()
            .map_or(0.0, |traj| traj.p_allele_given_kary(t, pop, kary));
        let p_kary_safe = p_kary.max(1e-9);

        // AA: pairs of A-tagged lineages — rate denom 2 N p_kary p_A.
        // Going backward, p_A drops from ~1 at τ to f0 at t_origin so
        // this rate climbs and drives the A pool to its single founder.
        if n_a_upper >= 2 && p_a > 1e-9 {
            let pairs = (n_a_upper * (n_a_upper - 1)) as f64 * 0.5;
            let denom = 2.0 * ne * p_kary_safe * p_a;
            events.push((pairs / denom,
                Event::CoalAggregate { pop, class: cls, allele: AlleleTag::A }));
        }
        // aa: pairs of a-tagged lineages — rate denom 2 N p_kary (1-p_A).
        if n_a_lower >= 2 && (1.0 - p_a) > 1e-9 {
            let pairs = (n_a_lower * (n_a_lower - 1)) as f64 * 0.5;
            let denom = 2.0 * ne * p_kary_safe * (1.0 - p_a);
            events.push((pairs / denom,
                Event::CoalAggregate { pop, class: cls, allele: AlleleTag::ALower }));
        }
        // Mixed: untagged-involved pairs only (UU + UA + Ua). Cross-
        // allele A × a pairs have rate zero during the sweep window
        // and are excluded — the consumer must filter the bucket to
        // honor that exclusion (handled in PG-C1).
        let n_normal_pairs =
            n_untagged * n_untagged.saturating_sub(1) / 2
            + n_untagged * n_a_upper
            + n_untagged * n_a_lower;
        if n_normal_pairs > 0 {
            let denom = 2.0 * ne * p_kary_safe;
            events.push((n_normal_pairs as f64 / denom,
                Event::CoalAggregate { pop, class: cls, allele: AlleleTag::Mixed }));
        }
    }
}

/// Helper: true if any segment in the lineage's chain has
/// `branch_class == cls`. Used by the per-allele rate emitter to
/// decide which (pop, cls) cells a lineage participates in.
fn lineage_has_class(head: SegIdx, cls: BranchClass, arena: &SegmentArena) -> bool {
    let mut cur = head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        if seg.branch_class == cls { return true; }
        cur = seg.next;
    }
    false
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
        if !barrier_active[k] || t >= inv.t_inv_max() { continue; }
        match cls.get_inv(inv.inv_id) {
            Some(Karyotype::S) => p *= inv.p_std_at(t, pop),
            Some(Karyotype::I) => p *= inv.p_inv_at(t, pop),
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

/// Arena-walking counterpart to `lineage_class_for_inv_segs`. Used by
/// the compound event loop, which doesn't maintain the RateCache's
/// flat seg mirror.
fn lineage_class_for_inv_arena(
    head: SegIdx, inv: &InversionSpec, arena: &SegmentArena,
) -> Option<Karyotype> {
    let mut seen_s = false;
    let mut seen_i = false;
    let mut cur = head;
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

/// Per-lineage flux rate with class resolved per segment.
///
/// Sums over each in-inv segment of `head`'s chain:
///   γ · p_other(seg.class) · phi_integral(seg-normalized-bounds, w) · inv_len
/// where p_other = p_inv(t, pop) for class S segments, 1 - p_inv for class I.
/// Panmictic segments (`get_inv → None`) contribute 0.
///
/// Replaces the prior "one karyotype per lineage" model: mixed-class lineages
/// (which b2-flux's partial-tract events create regularly) now contribute the
/// correct non-zero rate from BOTH their S and I segments, instead of being
/// zero-blocked by `lineage_class_for_inv_arena` returning None.
fn flux_lineage_rate_arena(
    head: SegIdx,
    inv: &InversionSpec,
    pop: u32,
    t: f64,
    arena: &SegmentArena,
) -> f64 {
    let inv_len = inv.length();
    if inv_len <= 0.0 { return 0.0; }
    let w_phi = inv.mean_tract_length / inv_len;
    let p_inv_v = inv.p_inv_at(t, pop);
    let mut rate = 0.0;
    let mut cur = head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        let next = seg.next;
        let l = seg.left.max(inv.bp_left);
        let r = seg.right.min(inv.bp_right);
        if r > l {
            if let Some(kary) = seg.branch_class.get_inv(inv.inv_id) {
                let p_other = match kary {
                    Karyotype::S => p_inv_v,
                    Karyotype::I => 1.0 - p_inv_v,
                };
                if p_other > 0.0 {
                    let a = (l - inv.bp_left) / inv_len;
                    let b = (r - inv.bp_left) / inv_len;
                    rate += inv.gene_conversion_rate * p_other
                          * phi_integral(a, b, w_phi) * inv_len;
                }
            }
            // Panmictic (None) → no flux contribution.
        }
        cur = next;
    }
    rate
}

/// Determine a lineage's karyotype for one inversion, reading flat segs.
fn lineage_class_for_inv_segs(
    segs: &[FlatSeg], inv: &InversionSpec,
) -> Option<Karyotype> {
    let mut seen_s = false;
    let mut seen_i = false;
    for &(sl, sr, cls) in segs {
        let l = sl.max(inv.bp_left);
        let r = sr.min(inv.bp_right);
        if r > l {
            match cls.get_inv(inv.inv_id) {
                Some(Karyotype::S) => seen_s = true,
                Some(Karyotype::I) => seen_i = true,
                None => {}
            }
        }
    }
    if seen_s && !seen_i { Some(Karyotype::S) }
    else if seen_i && !seen_s { Some(Karyotype::I) }
    else { None }
}

/// Flat-segs counterpart of `flux_lineage_rate_arena`. Same semantics; used
/// by the cache-rebuild path that operates on the RateCache flat-seg mirror.
fn flux_lineage_rate_segs(
    segs: &[FlatSeg],
    inv: &InversionSpec,
    pop: u32,
    t: f64,
) -> f64 {
    let inv_len = inv.length();
    if inv_len <= 0.0 { return 0.0; }
    let w_phi = inv.mean_tract_length / inv_len;
    let p_inv_v = inv.p_inv_at(t, pop);
    let mut rate = 0.0;
    for &(sl, sr, cls) in segs {
        let l = sl.max(inv.bp_left);
        let r = sr.min(inv.bp_right);
        if r > l {
            if let Some(kary) = cls.get_inv(inv.inv_id) {
                let p_other = match kary {
                    Karyotype::S => p_inv_v,
                    Karyotype::I => 1.0 - p_inv_v,
                };
                if p_other > 0.0 {
                    let a = (l - inv.bp_left) / inv_len;
                    let b = (r - inv.bp_left) / inv_len;
                    rate += inv.gene_conversion_rate * p_other
                          * phi_integral(a, b, w_phi) * inv_len;
                }
            }
        }
    }
    rate
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
    log: Option<&mut event_log::EventLog>,
    t: f64,
    x_event: f64,
) {
    let head = active[lin_idx].head;
    if head == SEG_NIL { return; }

    // Combined scan: capture first_left for the zombie-guard path and
    // confirm the tract touches at least one seg. No overlap ⇒ no-op.
    let first_left = arena.get(head).left;
    let mut tract_hits_material = false;
    let mut cur = head;
    while cur != SEG_NIL {
        let seg = arena.get(cur);
        if seg.left >= tract_right { break; }
        if seg.right > tract_left { tract_hits_material = true; break; }
        cur = seg.next;
    }
    if !tract_hits_material { return; }

    // Capture uid BEFORE any active.push() that might reallocate or
    // any split_at() that mutates the chain.
    let lineage_uid = active[lin_idx].uid;

    // Build per-segment (left, right, node_id) list spanning the tract,
    // but ONLY when an event log is attached. Off-path (`log is None`)
    // pays zero overhead: no segment walk, no allocation.
    let tract_segments: Option<Vec<(f64, f64, i32)>> = if log.is_some() {
        let mut segs: Vec<(f64, f64, i32)> = Vec::new();
        let mut cur = active[lin_idx].head;
        while cur != SEG_NIL {
            let seg = arena.get(cur);
            if seg.left >= tract_right { break; }
            if seg.right > tract_left {
                let l = seg.left.max(tract_left);
                let r = seg.right.min(tract_right);
                if r > l {
                    segs.push((l, r, seg.node_id));
                }
            }
            cur = seg.next;
        }
        debug_assert!(!segs.is_empty(),
            "flux event at x_event={} has no covering segments in lineage uid={}",
            x_event, lineage_uid);
        Some(segs)
    } else {
        None
    };

    if tract_left <= first_left {
        // Fast path: no material precedes the tract, so split at
        // tract_right only — active[lin_idx] becomes the tract.
        let uid = *next_uid; *next_uid += 1;
        let outside_right = active[lin_idx].split_at(tract_right, arena, uid);
        let mut cur = active[lin_idx].head;
        while cur != SEG_NIL {
            let seg = arena.get_mut(cur);
            seg.branch_class = seg.branch_class.flip_inv(inv.inv_id);
            cur = seg.next;
        }
        if let Some(right_lin) = outside_right {
            active.push(right_lin);
        }
        // Successful flip — log and return.
        if let Some(log) = log {
            log.push_flux(event_log::FluxRecord {
                t,
                lineage_uid,
                position: x_event,
                tract_left,
                tract_right,
                inv_id: inv.inv_id,
                tract_segments: tract_segments.expect("tract_segments built when log is Some"),
            });
        }
        return;
    }

    let uid = *next_uid;
    *next_uid += 1;
    let rest = active[lin_idx].split_at(tract_left, arena, uid);
    if rest.is_none() {
        return;  // no-op: split produced nothing
    }
    let mut rest = rest.unwrap();

    let uid2 = *next_uid;
    *next_uid += 1;
    let outside_right = rest.split_at(tract_right, arena, uid2);

    let mut cur = rest.head;
    while cur != SEG_NIL {
        let seg = arena.get_mut(cur);
        seg.branch_class = seg.branch_class.flip_inv(inv.inv_id);
        cur = seg.next;
    }

    // Outside material was one chromosome — splice A + outside_right
    // into a single lineage with a "hole" for the tract.
    if let Some(right_lin) = outside_right {
        active[lin_idx].append_chain(right_lin, arena);
    }

    active.push(rest);

    // Successful flip — log.
    if let Some(log) = log {
        log.push_flux(event_log::FluxRecord {
            t,
            lineage_uid,
            position: x_event,
            tract_left,
            tract_right,
            inv_id: inv.inv_id,
            tract_segments: tract_segments.expect("tract_segments built when log is Some"),
        });
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
    finalized_sweeps: &mut Vec<Sweep>,
    t: f64,
    tables: &mut TableBuilder,
    next_uid: &mut LinUid,
    seq_len: f64,
    rng: &mut Xoshiro256PlusPlus,
    recomb_rate: f64,
    sweep_cursor: &mut (f64, u64),
    event_log: Option<&mut event_log::EventLog>,
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
    // Order matches Python (msinv/hull/simulator.py ~1250-1263):
    // class barriers, then sweeps, then demographic events. A sweep
    // scheduled at the same t as an Ej must still see the pre-merge
    // populations — firing demo first would silently zero the sweep's
    // target pool when Ej moves all pop-N lineages into pop-M.
    HullSimulator::cross_barriers_static(inversions, active, arena, barrier_active, t);
    // Drain all τ-boundary sweeps: tag lineages (Phase B), then move to
    // finalized_sweeps queue for the t_origin forced-coalescence (Phase C1).
    while !pending_sweeps.is_empty()
        && (pending_sweeps[0].tau - t).abs() < 1e-9
    {
        let sweep = pending_sweeps.remove(0);
        let ne_sweep = demo.size_at(sweep.origin_pop, t).max(1.0);
        apply_sweep(active, &sweep, t, arena, tables,
                     next_uid, seq_len, rng, ne_sweep, recomb_rate,
                     sweep_cursor, a_tag);
        finalized_sweeps.push(sweep);
    }
    // Keep finalized_sweeps sorted by t_de_novo so [0] is the next to fire.
    finalized_sweeps.sort_by(|a, b|
        a.t_de_novo().partial_cmp(&b.t_de_novo()).unwrap());
    // Drain finalized sweeps whose t_de_novo matches the current boundary.
    while !finalized_sweeps.is_empty()
        && (finalized_sweeps[0].t_de_novo() - t).abs() < 1e-9
    {
        let sweep = finalized_sweeps.remove(0);
        apply_sweep_finalize(active, &sweep, t, arena, tables,
                              next_uid, rng, recomb_rate, sweep_cursor, a_tag);
    }
    let (inv_changes, class_mig) = demo.apply_events_at(t, active);
    for (inv_id, pop, p_inv_val) in inv_changes {
        if let Some(inv) = inversions.iter_mut().find(|i| i.inv_id == inv_id) {
            inv.set_p_inv_for(pop, p_inv_val);
        }
    }
    // Class-conditional migration: needs arena (read class) + rng
    // (proportion < 1 sampling).  Applied after inv_changes so the
    // class queries reflect the post-event inversion frequencies.
    let mut log_ref = event_log;
    for spec in class_mig {
        apply_class_mig(active, arena, &spec, rng, inversions,
                         log_ref.as_deref_mut(), t);
    }
}

/// Apply a class-conditional migration spec (DemoEvent::ClassMig).
/// For each lineage in `src` whose karyotype at `inv_id` matches
/// `kary`, move it to `dst` with probability `proportion`.
fn apply_class_mig(
    active: &mut [Lineage],
    arena: &SegmentArena,
    spec: &crate::demography::ClassMigSpec,
    rng: &mut Xoshiro256PlusPlus,
    inversions: &[InversionSpec],
    log: Option<&mut event_log::EventLog>,
    t: f64,
) {
    use rand::Rng;
    // Locate the InversionSpec for the requested inv_id (just for any
    // future class-aware logic — currently only inv_id is used to
    // look up segment class via inversion bp range).
    let _ = inversions;
    let mut n_eligible: u32 = 0;
    let mut n_moved: u32 = 0;
    for lin in active.iter_mut() {
        if lin.population != spec.src { continue; }
        let kary = lineage_class_for_inv_id_arena(lin.head, spec.inv_id, arena);
        if kary != Some(spec.kary) { continue; }
        n_eligible += 1;
        if spec.proportion >= 1.0 - 1e-12 || rng.random::<f64>() < spec.proportion {
            lin.population = spec.dst;
            n_moved += 1;
        }
    }
    if let Some(log) = log {
        log.push_cmig(event_log::CmigRecord {
            t,
            src: spec.src,
            dst: spec.dst,
            kary: spec.kary,
            inv_id: spec.inv_id,
            n_eligible,
            n_moved,
        });
    }
}

/// Find a lineage's karyotype at a given inv_id by scanning its
/// segment chain.  Returns the karyotype of the first segment with a
/// non-PAN class for that inv, or None if all PAN.
fn lineage_class_for_inv_id_arena(
    head: SegIdx,
    inv_id: u16,
    arena: &SegmentArena,
) -> Option<Karyotype> {
    let mut s = head;
    while s != crate::segment::SEG_NIL {
        let seg = arena.get(s);
        if let Some(k) = seg.branch_class.get_inv(inv_id) {
            return Some(k);
        }
        s = seg.next;
    }
    None
}

/// Monotonically increasing merge time, shared across all sweep merges
/// at the same base `t`. Resets when `t` changes.
///
fn next_sweep_merge_t(cursor: &mut (f64, u64), t: f64) -> f64 {
    if cursor.0 != t {
        *cursor = (t, 0);
    }
    cursor.1 += 1;
    let eps = (t * 1e-12).max(1e-9);
    t + (cursor.1 as f64) * eps
}

/// Populate `Sweep::trajectory` for any sweep that doesn't already carry one,
/// using closures that snapshot the current `Demography` arrays. Called once
/// at sim entry from each run-loop. Sweeps with a caller-supplied trajectory
/// are respected (skip-if-some).
fn populate_sweep_trajectories(
    pending_sweeps: &mut [Sweep],
    demo: &crate::demography::Demography,
    inversions: &[InversionSpec],
) {
    let n_pops = demo.n_pops;
    let pop_sizes_snap = demo.pop_sizes.clone();
    let growth_rates_snap = demo.growth_rates.clone();
    let growth_start_snap = demo.growth_start.clone();
    let events_snap = demo.events.clone();
    let mig_snap = demo.migration_matrix.clone();

    for sw in pending_sweeps.iter_mut() {
        if sw.trajectory.is_some() { continue; }
        let p_inv_init: Vec<f64> = (0..n_pops as usize).map(|p| {
            inversions.iter().find(|i| i.inv_id == sw.target_inv)
                .map(|i| i.p_inv_at(0.0, p as u32))
                .unwrap_or(0.0)
        }).collect();
        // Per-sweep clones so the closures own their data and don't hold a
        // borrow on the outer snapshot vectors (which a future multi-sweep
        // loop iteration could otherwise contend with).
        let ps = pop_sizes_snap.clone();
        let gr = growth_rates_snap.clone();
        let gs = growth_start_snap.clone();
        let ev = events_snap.clone();
        let mg = mig_snap.clone();
        let pop_size_at = move |t: f64, p: u32| -> f64 {
            let pp = p as usize;
            if pp >= ps.len() { return 1.0; }
            let mut size = ps[pp];
            let mut growth = gr[pp];
            let mut growth_start = gs[pp];
            for e in &ev {
                if e.time() > t { break; }
                match e {
                    crate::demography::DemoEvent::EN { n, .. } =>
                        { size = *n; growth = 0.0; growth_start = e.time(); }
                    crate::demography::DemoEvent::En { pop: p2, n, .. } if *p2 as usize == pp =>
                        { size = *n; growth = 0.0; growth_start = e.time(); }
                    crate::demography::DemoEvent::EG { alpha, .. } =>
                        { growth = *alpha; growth_start = e.time(); }
                    crate::demography::DemoEvent::Eg { pop: p2, alpha, .. } if *p2 as usize == pp =>
                        { growth = *alpha; growth_start = e.time(); }
                    _ => {}
                }
            }
            if growth == 0.0 { size } else { size * (-growth * (t - growth_start)).exp() }
        };
        let mig_at = move |_t: f64, i: u32, j: u32| -> f64 {
            if (i as usize) >= mg.len() { return 0.0; }
            *mg[i as usize].get(j as usize).unwrap_or(&0.0)
        };
        let placeholder = Sweep::new(
            sw.x_sel, sw.tau, sw.origin_pop, sw.origin_kary,
            sw.target_inv, sw.joint.clone());
        let real = std::mem::replace(sw, placeholder);
        *sw = real.with_trajectory(n_pops, &p_inv_init, &pop_size_at, &mig_at);
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
    _tables: &mut TableBuilder,
    _next_uid: &mut LinUid,
    _seq_len: f64,
    rng: &mut Xoshiro256PlusPlus,
    _ne: f64,
    recomb_rate: f64,
    _sweep_cursor: &mut (f64, u64),
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
    // Phase B: at τ (entry into the sweep window going backward), tag every
    // lineage overlapping x_sel in the sweep's (origin_pop, origin_kary) cell
    // as A or a using the trajectory's per-(pop, kary) A frequency. Tagging
    // is a one-shot at τ; later events inside the window (Phase C/D) consume
    // the tag.
    if (t - sweep.tau).abs() > 1e-9 {
        return;
    }
    if sweep.trajectory.is_none() {
        return;
    }
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
            let t_eff = sweep.t_de_novo() - sweep.tau;
            let p_link = (-recomb_rate * d_nearest * t_eff).exp();
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
}

/// At `t == sweep.joint.t_origin`: for each A-bearing lineage,
/// partition segments into linked vs escaped using per-segment
/// hitchhiking probabilities.  Linked segments stay with the
/// lineage's UID and are force-coalesced at t_origin.  Escaped
/// segments split off into a fresh untagged lineage that re-enters
/// the normal coal+recomb event loop.
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

    // De novo merge gate: the per-lineage merge only fires when the
    // trajectory actually has a standing-variation phase (t_de_novo
    // strictly past t_origin). For hard sweeps with f0=1/(2N), there
    // is no SV phase — we collapse A-tagged into a single founder at
    // t_origin as before, preserving the per-segment hitchhiking
    // signature (PS2/PS3) and the prior single-MRCA semantics.
    let has_sv_phase = sweep.t_de_novo() > sweep.joint.t_origin + 1e-9;

    if has_sv_phase {
        // At t_de_novo (the trajectory's extinction time), the A
        // allele arose by mutation on a single chromosome that is
        // otherwise indistinguishable from the rest of the
        // population. Each surviving A-tagged lineage merges with a
        // random non-A target in its own population — that's the de
        // novo origin.
        //
        // NOTE: candidates exclude not just A-tagged but anything
        // currently tagged in a_tag (i.e. just-escaped lineages
        // tagged false by the per-segment partition). The escaped
        // lineages came from the same A-tagged founders and re-
        // merging with them would undo the per-segment hitchhiking.
        for &a_uid in linked_uids.iter() {
            let a_idx = match active.iter().position(|l| l.uid == a_uid) {
                Some(i) => i,
                None => continue,
            };
            let a_pop = active[a_idx].population;
            let candidates: Vec<usize> = active.iter().enumerate()
                .filter(|(j, lin)| {
                    *j != a_idx
                        && lin.population == a_pop
                        && !a_tag.contains_key(&lin.uid)
                })
                .map(|(j, _)| j)
                .collect();
            if candidates.is_empty() { continue; }
            let pick = rng.random_range(0..candidates.len());
            let target_idx = candidates[pick];
            if !segments_overlap(active[a_idx].head, active[target_idx].head, arena) {
                continue;
            }
            let (lo, hi) = if a_idx < target_idx {
                (a_idx, target_idx)
            } else {
                (target_idx, a_idx)
            };
            let t_merge = next_sweep_merge_t(sweep_cursor, t);
            apply_coalescence_partial(
                active, lo, hi, t_merge, arena, tables, next_uid,
                None, Some(a_tag));
        }
    }

    // Edge case (always runs): any A-tagged lineages still in `active`
    // get collapsed among themselves. For hard sweeps without an SV
    // phase, this is the original endpoint behavior. With an SV
    // phase, this catches stragglers when no eligible non-A target
    // existed in the lineage's pop.
    let still_a: Vec<LinUid> = active.iter()
        .filter(|lin| a_tag.get(&lin.uid).copied().unwrap_or(false))
        .map(|lin| lin.uid)
        .collect();
    if still_a.len() >= 2 {
        coalesce_uid_group(active, &still_a, t, arena, tables, next_uid, sweep_cursor);
    }
    // Keep `a_tag` populated past `t_de_novo`. The map doubles as the
    // post-sweep accounting input for `count_a_samples` (T5 / partial
    // sweep `sweep_a_count`). The progressive-coal logic is gated on
    // an active sweep covering `t` at emit time (PG-B1's
    // `in_sweep_cell`) and at consume time (PG-C1 filter), so stale
    // entries don't leak into the post-window neutral coalescent.
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

fn lineage_overlaps_position(head: SegIdx, x: f64, arena: &SegmentArena) -> bool {
    let mut s = head;
    while s != SEG_NIL {
        let seg = arena.get(s);
        if seg.left <= x && x < seg.right {
            return true;
        }
        s = seg.next;
    }
    false
}

/// Discoal-style per-recombination tag rejection-sampling. Called
/// from each Event::Recombination consumer after `apply_recombination`
/// returns. For each new child lineage that does NOT contain `x_sel`,
/// rejection-samples its sweep-group tag against the trajectory's
/// current `p_A(t)`:
///
/// - A-tagged child stays A with prob `p_A(t)`, else becomes a-tagged.
/// - a-tagged (or untagged, treated as a) child stays a with prob
///   `1 - p_A(t)`, else becomes A-tagged.
///
/// Mirrors discoal `recombineAtTimePopnSweep`
/// (`/home/adkern/discoal/src/core/discoalFunctions.c:2569-2583`):
/// the parent containing `x_sel` keeps its sweep-group, the other
/// parent rejection-samples against the group's bgkd freq.
fn apply_sweep_recomb_tag_swap(
    active: &[Lineage],
    new_indices: &[usize],
    sweeps: &[Sweep],
    t: f64,
    arena: &SegmentArena,
    rng: &mut Xoshiro256PlusPlus,
    a_tag: &mut std::collections::HashMap<LinUid, bool>,
) {
    let sweep = match sweeps.iter().find(|s| s.covers(t)) {
        Some(s) => s,
        None => return,
    };
    let traj = match sweep.trajectory.as_ref() {
        Some(t) => t,
        None => return,
    };
    // Pragmatic gate: only fire the swap when an SV phase is active
    // (f0 > 1/(2N)). For hard sweeps (f0 = 1/(2N)) firing the swap
    // during the selection phase overshoots discoal even with the
    // merged-tag inheritance fix in events.rs (msinv pi 3918 → 5826
    // vs discoal target 4616). The mechanism mismatch in this regime
    // is not yet identified — apply_recombination's split semantics,
    // recomb-event firing rate, and trajectory-time query all match
    // discoal at the surface level. Documented as a known divergence;
    // soft sweeps still get the swap during the SV phase as designed.
    if sweep.t_de_novo() <= sweep.joint.t_origin + 1e-9 {
        return;
    }
    let p_a = traj.p_allele_given_kary(t, sweep.origin_pop, sweep.origin_kary);
    for &idx in new_indices {
        if idx >= active.len() { continue; }
        if active[idx].population != sweep.origin_pop { continue; }
        if active[idx].head == SEG_NIL { continue; }
        // Children that contain x_sel keep their tag (the sweep
        // mutation rides with them).
        if lineage_overlaps_position(active[idx].head, sweep.x_sel, arena) {
            continue;
        }
        let uid = active[idx].uid;
        let was_a = a_tag.get(&uid).copied().unwrap_or(false);
        // discoal: stays in current group with prob popnFreq, where
        // popnFreq is x for the A-group and (1 - x) for the a-group.
        let stay_prob = if was_a { p_a } else { 1.0 - p_a };
        if rng.random::<f64>() < stay_prob {
            // Keep current tag. Ensure entry exists (untagged-but-
            // considered-a is represented as a_tag = false) so future
            // swaps can flip it correctly.
            a_tag.entry(uid).or_insert(was_a);
        } else {
            // Switch.
            a_tag.insert(uid, !was_a);
        }
    }
}

/// Build a Lineage from a vector of (left, right, node_id, branch_class) tuples.
///
/// Currently unused; kept for future use by per-A-lineage finalize ops
/// that may need to build a synthetic lineage during sweep dispatch.
#[allow(dead_code)]
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
            // Skip disjoint pairs — forcing a merge would manufacture
            // ancestry correlation (orphan node + fictitious lineage).
            if !segments_overlap(active[mi].head, active[oi].head, arena) {
                continue;
            }
            apply_coalescence(active, mi, oi, t_merge, arena, tables, next_uid, None);
            merged_uid = active.last().unwrap().uid;
        }
    }
}

/// True if the two segment chains share any genomic position. Both
/// chains are assumed sorted by left boundary. O(|a| + |b|) two-pointer
/// walk; returns on the first overlap found.
fn segments_overlap(
    a_head: SegIdx, b_head: SegIdx, arena: &SegmentArena,
) -> bool {
    let mut sa = a_head;
    let mut sb = b_head;
    while sa != SEG_NIL && sb != SEG_NIL {
        let a = arena.get(sa);
        let b = arena.get(sb);
        if a.right <= b.left { sa = a.next; continue; }
        if b.right <= a.left { sb = b.next; continue; }
        return true;
    }
    false
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
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_panmictic_no_recomb() {
        // Path 2 MVP smoke: same expected topology as the bucket
        // path for a plain panmictic, no-inversion, no-recomb run.
        let mut sim = HullSimulator::panmictic(10, 1000.0, 100.0, 1e-12, 42);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 19);
        assert_eq!(result.tables.num_edges(), 18);
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_two_samples() {
        let mut sim = HullSimulator::panmictic(2, 100.0, 50.0, 1e-12, 7);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 3);
        assert_eq!(result.tables.num_edges(), 2);
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_with_low_recomb_runs() {
        // Very low rho — mostly coalescent. Just proves the recomb
        // plumbing doesn't break the compound path.
        // rho = 4·500·1e-6·50 = 0.1.
        let mut sim = HullSimulator::panmictic(4, 500.0, 50.0, 1e-6, 42);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 7,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_old_inversion_barrier_crosses() {
        // Very old inversion: barrier crosses almost immediately,
        // compound path should drive through cross_barriers_static
        // and finish as panmictic.
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 1.0);
  s.gene_conversion_rate = 1e-30;
  s.inv_id = 0;
  s };
        let mut sim = HullSimulator::simple(
            3, 3, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 11,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_inversion_barrier_forces_extra_nodes() {
        // Long-standing barrier (t_inv = 5·Ne) — S/I pairs can't
        // coalesce until t_inv, so nodes >= panmictic case.
        let inv = { let mut s = InversionSpec::with_p_inv(3000.0, 7000.0, vec![0.5], 5000.0);
  s.gene_conversion_rate = 1e-30;
  s.inv_id = 0;
  s };
        let mut sim = HullSimulator::simple(
            5, 5, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 19,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_with_gene_flux() {
        // Active barrier + nontrivial gene conversion. Flux should
        // fire and not crash.
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 20_000.0);
  s.gene_conversion_rate = 5e-6;
  s.inv_id = 0;
  s };
        let mut sim = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        sim.compound_rate = true;
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 15,
            "Got {} nodes", result.tables.num_nodes());
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_with_migration() {
        // Two-pop symmetric migration, no inversion. Finite m ensures
        // all pairs eventually collapse into one pop and coalesce.
        use crate::demography::Demography;
        let mut demo = Demography::new(vec![500.0, 500.0]);
        demo.migration_matrix[0][1] = 1e-3;
        demo.migration_matrix[1][0] = 1e-3;
        let sim = HullSimulator {
            samples: vec![
                SampleEntry { karyotypes: vec![], population: 0, count: 4 },
                SampleEntry { karyotypes: vec![], population: 1, count: 4 },
            ],
            demography: demo,
            sequence_length: 1000.0,
            recombination_rate: 1e-12,
            inversions: vec![],
            sweeps: vec![],
            seed: 42,
            stop_at: f64::INFINITY,
            compound_rate: true,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
        };
        let result = sim.simulate();
        assert_eq!(result.tables.num_nodes(), 15);
    }

    #[test]
    #[ignore = "compound_rate disabled on main"]
    fn compound_rate_with_en_demo_event() {
        // Ancestral pop size change at t=2000 (Ne 1000 → 5000).
        // Should fire via apply_boundary + trigger pair_rates rebuild.
        use crate::demography::{Demography, DemoEvent};
        let mut demo = Demography::single_pop(1000.0);
        demo.events.push(DemoEvent::EN { t: 2000.0, n: 5000.0 });
        let sim = HullSimulator {
            samples: vec![SampleEntry {
                karyotypes: vec![], population: 0, count: 6,
            }],
            demography: demo,
            sequence_length: 1000.0,
            recombination_rate: 1e-12,
            inversions: vec![],
            sweeps: vec![],
            seed: 42,
            stop_at: f64::INFINITY,
            compound_rate: true,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
        };
        let result = sim.simulate();
        // 6 samples, no recomb → 11 nodes.
        assert_eq!(result.tables.num_nodes(), 11);
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
        let inv = { let mut s = InversionSpec::with_p_inv(3000.0, 7000.0, vec![0.5], 5000.0);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
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
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 1.0);
  s.inv_id = 0;
  s };
        let sim = HullSimulator::simple(
            3, 3, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 11);
    }

    #[test]
    fn gene_flux_produces_extra_nodes() {
        // With gene flux, flux events split lineages → more nodes.
        // Ne=1000, L=10000, r=1e-8 → rho=0.4
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 20_000.0);
  s.gene_conversion_rate = 5e-6;
  s.inv_id = 0;
  s };
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
    fn coalesce_uid_group_skips_non_overlapping_pairs() {
        use crate::class_tag::BranchClass;

        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(1000.0, 1);
        let node_a = tables.add_sample(0.0, 0);
        let node_b = tables.add_sample(0.0, 0);

        let bc = BranchClass::PANMICTIC;
        let s_a = arena.alloc(0.0, 400.0, node_a, bc);
        let lin_a = Lineage::new(s_a, s_a, 0, 0, &arena);
        let s_b = arena.alloc(600.0, 1000.0, node_b, bc);
        let lin_b = Lineage::new(s_b, s_b, 0, 1, &arena);

        let mut active = vec![lin_a, lin_b];
        let mut next_uid: LinUid = 10;
        let mut cursor = (f64::NAN, 0u64);

        let n_edges_before = tables.num_edges();
        coalesce_uid_group(&mut active, &[0, 1], 100.0,
                            &mut arena, &mut tables, &mut next_uid,
                            &mut cursor);

        assert_eq!(tables.num_edges(), n_edges_before,
            "added edges for non-overlapping pair");
        assert_eq!(active.len(), 2,
            "non-overlapping pair was merged (active.len() = {})",
            active.len());
    }

    #[test]
    fn apply_gene_flux_outside_material_stays_one_lineage() {
        // Gene conversion transfers one tract from a donor to a receiver.
        // Going backward, the receiver's ancestry OUTSIDE the tract is
        // still one chromosome's worth — it must stay on one lineage
        // (with a "hole" where the tract was). Python's apply_gene_flux
        // explicitly re-merges A + C into the outside lineage. Rust was
        // splitting into 3 (A, tract, C) — extra fragmentation creates
        // spurious independent coalescences on either side of the tract.
        use crate::class_tag::{BranchClass, Karyotype};
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 1000.0, vec![0.5], 10_000.0);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };

        let mut arena = SegmentArena::new();
        let bc = BranchClass::single(0, Karyotype::I);
        // Single segment [100, 900). Tract [400, 500) carves a hole.
        let s = arena.alloc(100.0, 900.0, 0, bc);
        let lin = Lineage::new(s, s, 0, 0, &arena);
        let mut active = vec![lin];

        let mut next_uid: LinUid = 10;
        apply_gene_flux(&mut active, 0, 400.0, 500.0,
                        &inv, &mut arena, &mut next_uid, None, 0.0, 450.0);

        // Expected: 2 lineages — outside (= [100, 400) + [500, 900), one
        // lineage with a hole) and the flipped tract [400, 500).
        // Current Rust bug produces 3: A=[100, 400), tract=[400, 500),
        // C=[500, 900) — the outside material gets spuriously fragmented.
        let total: f64 = active.iter().map(|l| l.cached_len).sum();
        assert!((total - 800.0).abs() < 1e-12,
            "total = {} (expected 800)", total);
        assert_eq!(active.len(), 2,
            "BUG: gene flux fragmented outside material. active.len() = {} \
             (expected 2). Segments per lineage: {:?}",
            active.len(),
            active.iter().map(|l| l.cached_len).collect::<Vec<_>>());
    }

    #[test]
    fn apply_gene_flux_empty_tract_pushes_zombie() {
        // Tract falls entirely in a gap between the lineage's segments.
        // rest.split_at(tract_right) makes `rest` empty (all rest material
        // was past tract_right) and `active.push(rest)` pushes a zombie.
        use crate::class_tag::{BranchClass, Karyotype};
        let inv = { let mut s = InversionSpec::with_p_inv(0.0, 1000.0, vec![0.5], 10_000.0);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };

        let mut arena = SegmentArena::new();
        let bc = BranchClass::single(0, Karyotype::I);
        // Segments [100, 200) + [500, 600). Tract [250, 400) falls in the
        // gap — no material overlaps it.
        let s1 = arena.alloc(100.0, 200.0, 0, bc);
        let s2 = arena.alloc(500.0, 600.0, 1, bc);
        arena.get_mut(s1).next = s2;
        let lin = Lineage::new(s1, s2, 0, 0, &arena);
        let mut active = vec![lin];

        let mut next_uid: LinUid = 10;
        apply_gene_flux(&mut active, 0, 250.0, 400.0,
                        &inv, &mut arena, &mut next_uid, None, 0.0, 325.0);

        let total: f64 = active.iter().map(|l| l.cached_len).sum();
        assert!((total - 200.0).abs() < 1e-12,
            "total material = {} (expected 200.0)", total);

        let zombies: Vec<usize> = active.iter().enumerate()
            .filter(|(_, l)| l.head == SEG_NIL || l.cached_len == 0.0)
            .map(|(i, _)| i).collect();
        assert!(zombies.is_empty(),
            "BUG: {} zombie(s) at {:?} — active={:?}",
            zombies.len(), zombies,
            active.iter().map(|l| (l.head, l.cached_len)).collect::<Vec<_>>());
    }

    #[test]
    fn apply_gene_flux_leaves_empty_zombie() {
        // Repro: if the lineage's first segment starts at inv.bp_left
        // and the drawn tract_left == inv.bp_left, Lineage::split_at
        // makes the lineage's head = SEG_NIL (cached_len = 0) and
        // apply_gene_flux never removes it — so active[li] is a zombie.
        use crate::class_tag::{BranchClass, Karyotype};
        let inv = { let mut s = InversionSpec::with_p_inv(100.0, 200.0, vec![0.5], 10_000.0);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };

        let mut arena = SegmentArena::new();
        let bc = BranchClass::single(0, Karyotype::I);
        // Lineage with a single segment starting exactly at inv.bp_left.
        let idx = arena.alloc(inv.bp_left, inv.bp_right, 0, bc);
        let lin = Lineage::new(idx, idx, 0, 0, &arena);
        let mut active = vec![lin];

        let mut next_uid: LinUid = 10;
        // Tract starting exactly at inv.bp_left — drawable by draw_tract
        // (b1 = 0 when the event lands near the left edge).
        apply_gene_flux(&mut active, 0,
                        inv.bp_left, inv.bp_left + 5.0,
                        &inv, &mut arena, &mut next_uid, None, 0.0, inv.bp_left);

        // Expected: no zombies. Total ancestral material unchanged (100.0).
        let total: f64 = active.iter().map(|l| l.cached_len).sum();
        assert!((total - 100.0).abs() < 1e-12,
            "total material = {} (expected 100.0)", total);

        let zombies: Vec<usize> = active.iter().enumerate()
            .filter(|(_, l)| l.head == SEG_NIL || l.cached_len == 0.0)
            .map(|(i, _)| i).collect();
        assert!(zombies.is_empty(),
            "BUG: {} zombie lineage(s) at indices {:?} — active={:?}",
            zombies.len(), zombies,
            active.iter().map(|l| (l.head, l.cached_len)).collect::<Vec<_>>());
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
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
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
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
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
        let inv = { let mut s = InversionSpec::with_p_inv(3000.0, 7000.0, vec![0.5], 5000.0);
  s.gene_conversion_rate = 1e-9;
  s.inv_id = 0;
  s };
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
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
        };
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 11);
    }

    // -----------------------------------------------------------------
    // Class-conditional migration (cmig / ClassMig) unit tests
    // -----------------------------------------------------------------

    use crate::demography::ClassMigSpec;
    use crate::lineage::Lineage;
    use crate::class_tag::{BranchClass, Karyotype};

    fn _mk_lineage(arena: &mut SegmentArena, pop: u32, cls: BranchClass,
                   bp_left: f64, bp_right: f64, uid: u32) -> Lineage {
        let seg_idx = arena.alloc(bp_left, bp_right, 0, cls);
        Lineage::new(seg_idx, seg_idx, pop, uid, arena)
    }

    #[test]
    fn class_mig_full_S_moves_only_S_lineages() {
        let mut arena = SegmentArena::new();
        let s_cls = BranchClass::single(0, Karyotype::S);
        let i_cls = BranchClass::single(0, Karyotype::I);
        // Two S lineages and two I lineages, all in pop 1.
        let mut active = vec![
            _mk_lineage(&mut arena, 1, s_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, s_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, i_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, i_cls, 0.0, 1000.0, 0),
        ];
        // K-only-S sample reproduction: cmig S, F=1 → K=0.
        let spec = ClassMigSpec {
            src: 1, dst: 0, kary: Karyotype::S, inv_id: 0, proportion: 1.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        apply_class_mig(&mut active, &arena, &spec, &mut rng, &[], None, 0.0);

        // Both S-class lineages should now be in pop 0.
        // Both I-class lineages should still be in pop 1.
        assert_eq!(active[0].population, 0, "S lineage 0 not moved");
        assert_eq!(active[1].population, 0, "S lineage 1 not moved");
        assert_eq!(active[2].population, 1, "I lineage 2 wrongly moved");
        assert_eq!(active[3].population, 1, "I lineage 3 wrongly moved");
    }

    #[test]
    fn class_mig_full_I_moves_only_I_lineages() {
        let mut arena = SegmentArena::new();
        let s_cls = BranchClass::single(0, Karyotype::S);
        let i_cls = BranchClass::single(0, Karyotype::I);
        let mut active = vec![
            _mk_lineage(&mut arena, 1, s_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, i_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, i_cls, 0.0, 1000.0, 0),
        ];
        let spec = ClassMigSpec {
            src: 1, dst: 0, kary: Karyotype::I, inv_id: 0, proportion: 1.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        apply_class_mig(&mut active, &arena, &spec, &mut rng, &[], None, 0.0);

        assert_eq!(active[0].population, 1, "S lineage wrongly moved");
        assert_eq!(active[1].population, 0, "I lineage 1 not moved");
        assert_eq!(active[2].population, 0, "I lineage 2 not moved");
    }

    #[test]
    fn class_mig_does_not_touch_lineages_not_in_src() {
        let mut arena = SegmentArena::new();
        let s_cls = BranchClass::single(0, Karyotype::S);
        // Lineage in pop 0 (= dst), should stay put even though kary matches.
        let mut active = vec![
            _mk_lineage(&mut arena, 0, s_cls, 0.0, 1000.0, 0),
            _mk_lineage(&mut arena, 1, s_cls, 0.0, 1000.0, 0),
        ];
        let spec = ClassMigSpec {
            src: 1, dst: 0, kary: Karyotype::S, inv_id: 0, proportion: 1.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        apply_class_mig(&mut active, &arena, &spec, &mut rng, &[], None, 0.0);

        assert_eq!(active[0].population, 0, "pop-0 S lineage moved (shouldn't)");
        assert_eq!(active[1].population, 0, "pop-1 S lineage not moved");
    }

    #[test]
    fn class_mig_skips_pan_only_lineages() {
        // Lineage with no class tag for inv_id=0 (PAN) should not be
        // caught by cmig with kary='S' or 'I'.
        let mut arena = SegmentArena::new();
        let pan_cls = BranchClass::PANMICTIC;
        let mut active = vec![
            _mk_lineage(&mut arena, 1, pan_cls, 0.0, 1000.0, 0),
        ];
        let spec = ClassMigSpec {
            src: 1, dst: 0, kary: Karyotype::S, inv_id: 0, proportion: 1.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        apply_class_mig(&mut active, &arena, &spec, &mut rng, &[], None, 0.0);
        // PAN lineage has no class tag for inv 0 → skipped, stays in pop 1.
        assert_eq!(active[0].population, 1,
            "PAN lineage incorrectly migrated by cmig");
    }

    #[test]
    fn class_mig_partial_proportion_stochastic() {
        // proportion=0.5: with 200 lineages and seed 42, expect roughly
        // half (within ~3 SD = ±20).
        let mut arena = SegmentArena::new();
        let s_cls = BranchClass::single(0, Karyotype::S);
        let n_lin = 200;
        let mut active: Vec<Lineage> = (0..n_lin)
            .map(|_| _mk_lineage(&mut arena, 1, s_cls, 0.0, 1000.0, 0))
            .collect();
        let spec = ClassMigSpec {
            src: 1, dst: 0, kary: Karyotype::S, inv_id: 0, proportion: 0.5,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        apply_class_mig(&mut active, &arena, &spec, &mut rng, &[], None, 0.0);
        let moved = active.iter().filter(|l| l.population == 0).count();
        // 200 * 0.5 = 100, ±3*sqrt(200*0.5*0.5) ≈ ±21
        assert!((moved as i64 - 100).abs() < 25,
            "stochastic proportion=0.5 moved {} of 200; expected ~100", moved);
    }

    #[test]
    #[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
    fn soft_sweep_with_recombination() {
        // Body deleted in sweep-rewrite Task 11. Will be rewritten in
        // Task 16 once the new JointSweepTrajectory operator lands.
    }

    #[test]
    fn record_events_logs_one_cmig_event() {
        use crate::event_log::EventRecord;
        use crate::demography::{Demography, DemoEvent};

        // 2-population demography with a single cmig event at t=50:
        // all S-class lineages in pop 1 move to pop 0.
        let mut demo = Demography::new(vec![1000.0, 1000.0]);
        demo.add_event(DemoEvent::ClassMig {
            t: 50.0,
            src: 1,
            dst: 0,
            kary: Karyotype::S,
            inv_id: 0,
            proportion: 1.0,
        });
        // Merge remaining lineages at t=500 so the simulation terminates.
        demo.add_event(DemoEvent::Ej { t: 500.0, src: 1, dst: 0 });

        // Inversion spanning the whole sequence so S/I class tags are set.
        let inv = {
            let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 5000.0);
            s.gene_conversion_rate = 1e-9;
            s.inv_id = 0;
            s
        };

        let mut sim = HullSimulator {
            samples: vec![
                SampleEntry {
                    karyotypes: vec![Some(Karyotype::S)],
                    population: 0,
                    count: 3,
                },
                SampleEntry {
                    karyotypes: vec![Some(Karyotype::S)],
                    population: 1,
                    count: 3,
                },
            ],
            demography: demo,
            sequence_length: 10000.0,
            recombination_rate: 1e-8,
            inversions: vec![inv],
            sweeps: vec![],
            seed: 42,
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
        };
        sim.record_events = true;

        let result = sim.simulate();
        let log = result.event_log.expect("event_log should be Some");
        let cmig_recs: Vec<_> = log.records().iter()
            .filter_map(|r| if let EventRecord::Cmig(c) = r { Some(*c) } else { None })
            .collect();
        assert_eq!(cmig_recs.len(), 1,
            "expected exactly one cmig record, got {}", cmig_recs.len());
        assert_eq!(cmig_recs[0].src, 1);
        assert_eq!(cmig_recs[0].dst, 0);
        assert_eq!(cmig_recs[0].inv_id, 0);
        assert!(cmig_recs[0].n_eligible > 0,
            "test setup defective: no eligible lineages — kary check or inv_id wrong");
    }

    #[test]
    fn record_events_logs_flux_events_when_gamma_positive() {
        use crate::event_log::EventRecord;

        // Inversion spanning the whole sequence, high gene_conversion_rate so
        // flux fires frequently enough to guarantee ≥1 event.
        let inv = {
            let mut s = InversionSpec::with_p_inv(0.0, 10000.0, vec![0.5], 20_000.0);
            s.gene_conversion_rate = 5e-6;
            s.inv_id = 0;
            s
        };

        let mut sim = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8, vec![inv], 42);
        sim.record_events = true;

        let result = sim.simulate();
        let log = result.event_log.expect("event_log should be Some");
        let flux_recs: Vec<_> = log.records().iter()
            .filter_map(|r| if let EventRecord::Flux(f) = r { Some(f.clone()) } else { None })
            .collect();
        assert!(!flux_recs.is_empty(),
            "expected at least one FluxRecord; tune gene_conversion_rate/t_inv if zero");
        for r in &flux_recs {
            assert!(r.tract_right > r.tract_left,
                "tract bounds inverted: left={}, right={}", r.tract_left, r.tract_right);
            assert_eq!(r.inv_id, 0);
            assert!(r.tract_left >= 0.0 && r.tract_right <= 10000.0,
                "tract [{}, {}) out of sequence bounds [0, 10000)",
                r.tract_left, r.tract_right);
        }
    }

    #[test]
    fn apply_sweep_tags_lineages_with_assigned_a() {
        use crate::class_tag::{BranchClass, Karyotype};
        use crate::sweep_trajectory::{JointSweepSpec, SweepMode};
        use std::collections::HashMap;

        let mut arena = SegmentArena::new();
        let head_a = arena.alloc(0.0, 10_000.0, 0, BranchClass::PANMICTIC);
        let head_b = arena.alloc(0.0, 10_000.0, 1, BranchClass::PANMICTIC);
        let mut active = vec![
            Lineage::new(head_a, head_a, 0, 0u32, &arena),
            Lineage::new(head_b, head_b, 0, 1u32, &arena),
        ];

        let sweep = Sweep::new(5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 200.0, f0: 0.99,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            }).with_trajectory(1, &[0.0],
            &|_t, _p| 10_000.0, &|_, _, _| 0.0);

        let mut tables = TableBuilder::new(10_000.0, 1);
        let mut next_uid: LinUid = 2;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut a_tag: HashMap<LinUid, bool> = HashMap::new();
        let mut sweep_cursor = (0.0, 0u64);

        apply_sweep(&mut active, &sweep, 0.0, &mut arena, &mut tables,
                    &mut next_uid, 10_000.0, &mut rng, 10_000.0, 1e-12,
                    &mut sweep_cursor, &mut a_tag);

        // f0=0.99 with high A frequency on origin_kary=S → expect both lineages tagged.
        assert_eq!(a_tag.len(), 2, "expected both lineages tagged");
        let n_a = a_tag.values().filter(|&&v| v).count();
        assert!(n_a >= 1, "expected at least 1 A-tagged with f0=0.99, got {n_a}");
    }

    #[test]
    fn apply_sweep_skips_lineages_outside_x_sel_window() {
        use crate::class_tag::{BranchClass, Karyotype};
        use crate::sweep_trajectory::{JointSweepSpec, SweepMode};
        use std::collections::HashMap;

        let mut arena = SegmentArena::new();
        // Lineage A spans x_sel=5000; lineage B is to the right of x_sel and doesn't overlap.
        let head_a = arena.alloc(0.0, 6_000.0, 0, BranchClass::PANMICTIC);
        let head_b = arena.alloc(7_000.0, 10_000.0, 1, BranchClass::PANMICTIC);
        let mut active = vec![
            Lineage::new(head_a, head_a, 0, 0u32, &arena),
            Lineage::new(head_b, head_b, 0, 1u32, &arena),
        ];

        let sweep = Sweep::new(5_000.0, 0.0, 0, Karyotype::S, 0,
            JointSweepSpec {
                mode: SweepMode::Deterministic,
                s: 0.05, t_origin: 200.0, f0: 0.99,
                partial_sweep_final_freq: 0.99,
                ..Default::default()
            }).with_trajectory(1, &[0.0],
            &|_t, _p| 10_000.0, &|_, _, _| 0.0);

        let mut tables = TableBuilder::new(10_000.0, 1);
        let mut next_uid: LinUid = 2;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut a_tag: HashMap<LinUid, bool> = HashMap::new();
        let mut sweep_cursor = (0.0, 0u64);

        // Use a high recomb_rate so p_link = exp(-r*d*T_eff) ≈ 0 for lineage B
        // (d_nearest ≈ 2000, T_eff = 200 → r=1.0 → p_link ≈ 0).
        // With D1, non-overlapping lineages are gated by p_link; at r=1.0 they
        // are excluded with overwhelming probability. Lineage A still overlaps
        // x_sel and is tagged deterministically.
        apply_sweep(&mut active, &sweep, 0.0, &mut arena, &mut tables,
                    &mut next_uid, 10_000.0, &mut rng, 10_000.0, 1.0,
                    &mut sweep_cursor, &mut a_tag);

        assert!(a_tag.contains_key(&0u32), "lineage A overlaps x_sel; should be tagged");
        assert!(!a_tag.contains_key(&1u32), "lineage B: p_link≈0 at r=1.0; should NOT be tagged");
    }

    /// 4 sample lineages, all with A=true via f0=1.0 (probability 1 of A
    /// assignment at τ). With f0=1.0 the SV phase runs from t_origin
    /// backward until the drift hits 1/(2N); A-tagged lineages coalesce
    /// progressively inside that window. MRCA must therefore land at or
    /// past t_origin (and at or before t_de_novo).
    #[test]
    fn forced_coal_collapses_a_lineages_in_sweep_window() {
        use crate::class_tag::Karyotype;
        use crate::sweep_trajectory::{JointSweepSpec, SweepMode};
        use crate::demography::Demography;
        use crate::sweep::Sweep;

        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: 0.05, t_origin: 500.0, f0: 1.0,
            partial_sweep_final_freq: 0.99,
            ..Default::default()
        };
        let sweep = Sweep::new(5_000.0, 0.0, 0, Karyotype::S, 0, spec)
            .with_trajectory(1, &[0.0],
                &|_t, _p| 10_000.0,
                &|_t, _i, _j| 0.0);
        let t_de_novo = sweep.t_de_novo();

        let sim = HullSimulator {
            samples: vec![SampleEntry { karyotypes: vec![], population: 0, count: 4 }],
            demography: Demography::single_pop(10_000.0),
            sequence_length: 10_000.0, recombination_rate: 1e-12,
            inversions: vec![], sweeps: vec![sweep],
            seed: 7, stop_at: f64::INFINITY,
            compound_rate: false, iters_max: 1_000_000,
            gc_stride: 160, record_events: false,
        };
        let result = sim.simulate();
        // The deepest internal node should fall inside the sweep window
        // [t_origin, t_de_novo]. Loose tolerance on the upper bound to
        // account for the de novo merge happening at exactly t_de_novo.
        let max_node_t = result.tables.node_time.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_node_t >= 500.0 - 1e-3 && max_node_t <= t_de_novo + 1.0,
            "MRCA at {}, expected in [t_origin=500, t_de_novo={}]",
            max_node_t, t_de_novo);
    }

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
}
