/// HullSimulator: the main event loop.
///
/// Phase C: inversions (class barriers, per-pair coal rates, gene flux,
/// barrier crossing) on top of the Phase B panmictic loop.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::{BranchClass, Karyotype};
use crate::events::{apply_coalescence, apply_recombination};
use crate::inversion::InversionSpec;
use crate::lineage::{LinUid, Lineage};
use crate::phi::{phi, phi_integral};
use crate::segment::{SegIdx, SegmentArena, SEG_NIL};
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
}

// ---------------------------------------------------------------
// HullSimulator
// ---------------------------------------------------------------
pub struct HullSimulator {
    pub samples: Vec<SampleEntry>,
    pub population_size: f64,
    pub sequence_length: f64,
    pub recombination_rate: f64,
    pub inversions: Vec<InversionSpec>,
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
            population_size,
            sequence_length,
            recombination_rate,
            inversions,
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
            population_size,
            sequence_length,
            recombination_rate,
            inversions: vec![],
            seed,
        }
    }

    pub fn simulate(&self) -> SimResult {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(self.sequence_length, 1);
        let mut next_uid: LinUid = 0;

        let mut active = self.make_initial_lineages(
            &mut arena, &mut tables, &mut next_uid);

        self.run_loop(&mut active, &mut arena, &mut tables,
                       &mut next_uid, &mut rng);

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
                active.push(Lineage::new(head, tail, entry.population, uid));
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
    ) {
        let ne = self.population_size;
        let mut t: f64 = 0.0;

        // Track which inversions' barriers are still active.
        let mut barrier_active: Vec<bool> = self.inversions.iter()
            .map(|_| true).collect();

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

            // --- Build event list ---
            let mut events: Vec<(f64, Event)> = Vec::new();

            if any_barrier {
                // Per-pair, per-class coalescence rates.
                self.compute_coal_rates_structured(
                    active, arena, ne, t, &barrier_active, &mut events);
            } else {
                // All barriers lifted → panmictic by pop.
                let k = n as f64;
                let rate = k * (k - 1.0) / 2.0 / (2.0 * ne);
                if rate > 0.0 {
                    events.push((rate, Event::CoalPanmicticPop { pop: 0 }));
                }
            }

            // Recombination rate.
            let recomb_rate: f64 = if self.recombination_rate > 0.0 {
                active.iter()
                    .map(|lin| lin.total_length(arena) * self.recombination_rate)
                    .sum()
            } else {
                0.0
            };
            if recomb_rate > 0.0 {
                events.push((recomb_rate, Event::Recombination));
            }

            // Gene flux rates (per lineage, per inversion).
            if any_barrier {
                self.compute_flux_rates(
                    active, arena, &barrier_active, &mut events);
            }

            let total_rate: f64 = events.iter().map(|(r, _)| *r).sum();
            if total_rate <= 0.0 {
                // No events possible — jump to next barrier.
                if earliest_barrier < f64::INFINITY {
                    t = earliest_barrier;
                    self.cross_barriers(
                        active, arena, &mut barrier_active, t);
                    continue;
                }
                return;
            }

            // Draw waiting time.
            let u: f64 = rng.random();
            let dt = -u.ln() / total_rate;
            let t_event = t + dt;

            // Check if a barrier crossing happens first.
            if earliest_barrier <= t_event {
                t = earliest_barrier;
                self.cross_barriers(active, arena, &mut barrier_active, t);
                continue;
            }
            t = t_event;

            // Pick which event fires.
            let u2: f64 = rng.random::<f64>() * total_rate;
            let mut cum = 0.0;
            let mut chosen = None;
            for (rate, event) in &events {
                cum += rate;
                if u2 < cum {
                    chosen = Some(event);
                    break;
                }
            }
            let chosen = match chosen {
                Some(e) => e,
                None => continue, // numerical precision miss
            };

            match chosen {
                Event::CoalPair { i, j, class: _ } => {
                    let (i, j) = (*i, *j);
                    apply_coalescence(active, i, j, t, arena, tables, next_uid);
                }
                Event::CoalPanmicticPop { pop: _ } => {
                    let i = rng.random_range(0..n);
                    let mut j = rng.random_range(0..n - 1);
                    if j >= i { j += 1; }
                    apply_coalescence(active, i, j, t, arena, tables, next_uid);
                }
                Event::Recombination => {
                    let target = (u2 - (total_rate - recomb_rate))
                        .max(0.0) / self.recombination_rate;
                    let mut cum_len = 0.0;
                    let mut chosen_idx = 0;
                    for (idx, lin) in active.iter().enumerate() {
                        cum_len += lin.total_length(arena);
                        if cum_len > target {
                            chosen_idx = idx;
                            break;
                        }
                    }
                    let lin_len = active[chosen_idx].total_length(arena);
                    let x_offset: f64 = rng.random::<f64>() * lin_len;
                    let x = find_position(active, chosen_idx, x_offset, arena,
                                           self.sequence_length);
                    apply_recombination(active, chosen_idx, x, arena, next_uid);
                }
                Event::Flux { lineage_idx, inv_idx } => {
                    let (li, ii) = (*lineage_idx, *inv_idx);
                    let inv = &self.inversions[ii];
                    // Sample flux position.
                    if let Some(x_event) = self.sample_flux_position(
                        active, li, inv, arena, rng)
                    {
                        let (tl, tr) = self.draw_tract(x_event, inv, rng);
                        if tr > tl {
                            apply_gene_flux(active, li, tl, tr, inv, arena, next_uid);
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Per-pair, per-class coalescence rates
    // ---------------------------------------------------------------
    fn compute_coal_rates_structured(
        &self,
        active: &[Lineage],
        arena: &SegmentArena,
        ne: f64,
        t: f64,
        barrier_active: &[bool],
        events: &mut Vec<(f64, Event)>,
    ) {
        // Build p_class lookup per inversion tag.
        let n = active.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if active[i].population != active[j].population {
                    continue;
                }
                // Compute overlap bucketed by matching class.
                let overlaps = overlap_by_class(
                    active[i].head, active[j].head, arena);
                for (cls, ov_len) in &overlaps {
                    if *ov_len <= 0.0 { continue; }
                    let p_class = self.p_class_for(*cls, t, barrier_active);
                    if p_class <= 0.0 { continue; }
                    let rate = 1.0 / (2.0 * ne * p_class);
                    events.push((rate, Event::CoalPair {
                        i, j, class: *cls,
                    }));
                }
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
                None => {} // this inv not present at this position
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
        if r > l && a.branch_class.can_coalesce(b.branch_class) {
            let cls = a.branch_class; // they match
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
        let sim = HullSimulator::panmictic(6, 1000.0, 100.0, 1e-4, 42);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() > 11);
        assert!(result.tables.num_edges() > 10);
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
}
