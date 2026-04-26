/// RateEngine: persistent Fenwick-backed event rate tracker.
///
/// Maintains coalescence, recombination, flux, and migration rates
/// incrementally. After each event, only the affected rates are
/// updated — O(n * log K) per event instead of O(n^2) full rebuild,
/// where K is the number of Fenwick leaves.
///
/// Layout of Fenwick leaves:
///   [0]              = total recombination rate
///   [1..n_flux]      = per-(lineage, inv) flux rates
///   [n_flux+1..]     = coal pair rates (structured) OR pop bucket rates (panmictic)
///
/// For now, we use a simpler approach: maintain a Vec of (rate, EventTag)
/// and a Fenwick tree over them. The Vec is rebuilt only when the event
/// structure changes (barrier crossing, GC), and point-updated after
/// normal events (coal, recomb, flux).

use crate::class_tag::BranchClass;
use crate::demography::Demography;
use crate::fenwick::Fenwick;
use crate::inversion::InversionSpec;
use crate::lineage::Lineage;
use crate::rate_index::RateCache;
use crate::segment::SegmentArena;

/// Tag identifying what a Fenwick leaf represents.
#[derive(Clone, Debug)]
pub enum RateTag {
    Recombination,
    CoalPair { i: usize, j: usize, class: BranchClass },
    CoalPop { pop: u32 },
    Flux { lineage_idx: usize, inv_idx: usize },
    Migration { lineage_idx: usize, dst_pop: u32 },
}

pub struct RateEngine {
    pub fenwick: Fenwick,
    pub tags: Vec<RateTag>,
    n_leaves: usize,
}

impl RateEngine {
    /// Build from scratch — used at init and after structural changes.
    pub fn build_structured(
        cache: &RateCache,
        active: &[Lineage],
        _arena: &SegmentArena,
        demo: &Demography,
        t: f64,
        inversions: &[InversionSpec],
        barrier_active: &[bool],
        recomb_rate_total: f64,
        flux_rates: &[(f64, usize, usize)],  // (rate, lin_idx, inv_idx)
        migration_rates: &[(f64, usize, u32)], // (rate, lin_idx, dst_pop)
    ) -> Self {
        let mut tags = Vec::new();
        let mut rates = Vec::new();

        // Recombination (1 leaf).
        tags.push(RateTag::Recombination);
        rates.push(recomb_rate_total);

        // Coal pairs from cache. Classes derived from the bucket
        // back-references — no per-pair class array stored anymore.
        for (i, j, refs) in cache.iter_pairs() {
            let pop = active[i].population;
            let ne = demo.size_at(pop, t).max(1e-9);
            for (slot, _pos) in refs.iter().copied() {
                let cls = cache.class_for_ref(slot);
                let p_class = p_class_for(cls, inversions, barrier_active, t, pop);
                if p_class <= 0.0 { continue; }
                let rate = 1.0 / (2.0 * ne * p_class);
                tags.push(RateTag::CoalPair { i, j, class: cls });
                rates.push(rate);
            }
        }

        // Flux rates.
        for &(rate, li, ii) in flux_rates {
            tags.push(RateTag::Flux { lineage_idx: li, inv_idx: ii });
            rates.push(rate);
        }

        // Migration rates.
        for &(rate, li, dst) in migration_rates {
            tags.push(RateTag::Migration { lineage_idx: li, dst_pop: dst });
            rates.push(rate);
        }

        let n = rates.len();
        let mut fenwick = Fenwick::new(n);
        for (i, &r) in rates.iter().enumerate() {
            fenwick.update(i, r);
        }

        Self { fenwick, tags, n_leaves: n }
    }

    /// Build for panmictic (post-barrier) regime.
    pub fn build_panmictic(
        active: &[Lineage],
        demo: &Demography,
        t: f64,
        recomb_rate_total: f64,
        migration_rates: &[(f64, usize, u32)],
    ) -> Self {
        let mut tags = Vec::new();
        let mut rates = Vec::new();

        // Recombination.
        tags.push(RateTag::Recombination);
        rates.push(recomb_rate_total);

        // Per-pop coal buckets.
        let mut buckets: Vec<(u32, usize)> = Vec::new();
        for lin in active.iter() {
            if let Some(b) = buckets.iter_mut().find(|(p, _)| *p == lin.population) {
                b.1 += 1;
            } else {
                buckets.push((lin.population, 1));
            }
        }
        for (pop, k) in &buckets {
            if *k < 2 { continue; }
            let ne = demo.size_at(*pop, t).max(1e-9);
            let kf = *k as f64;
            let rate = kf * (kf - 1.0) / 2.0 / (2.0 * ne);
            tags.push(RateTag::CoalPop { pop: *pop });
            rates.push(rate);
        }

        // Migration.
        for &(rate, li, dst) in migration_rates {
            tags.push(RateTag::Migration { lineage_idx: li, dst_pop: dst });
            rates.push(rate);
        }

        let n = rates.len();
        let mut fenwick = Fenwick::new(n);
        for (i, &r) in rates.iter().enumerate() {
            fenwick.update(i, r);
        }

        Self { fenwick, tags, n_leaves: n }
    }

    pub fn total_rate(&self) -> f64 {
        self.fenwick.total()
    }

    /// Select an event proportional to rate. Returns the tag.
    pub fn select(&self, u: f64) -> Option<&RateTag> {
        let leaf = self.fenwick.find(u);
        if leaf < self.n_leaves {
            Some(&self.tags[leaf])
        } else {
            None
        }
    }

    /// Update the recombination rate (leaf 0).
    pub fn update_recomb(&mut self, new_rate: f64) {
        let old = if self.n_leaves > 0 {
            self.fenwick.prefix_sum(0)
        } else {
            0.0
        };
        self.fenwick.update(0, new_rate - old);
    }
}

/// Compute p_class for a BranchClass tag, using per-population
/// inversion frequencies queried at the current backward time `t`.
fn p_class_for(cls: BranchClass, inversions: &[InversionSpec],
               barrier_active: &[bool], t: f64, pop: u32) -> f64 {
    use crate::class_tag::Karyotype;
    if cls.is_panmictic() { return 1.0; }
    let mut p = 1.0;
    for (k, inv) in inversions.iter().enumerate() {
        if !barrier_active[k] { continue; }
        match cls.get_inv(inv.inv_id) {
            Some(Karyotype::S) => p *= inv.p_std_at(t, pop),
            Some(Karyotype::I) => p *= inv.p_inv_at(t, pop),
            None => {}
        }
    }
    p
}
