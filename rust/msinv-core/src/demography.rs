/// Multi-population demographic model in generations.
///
/// Supports ms-style events: population size changes, exponential
/// growth, migration rate changes, and population mergers.

use crate::lineage::Lineage;

/// Demographic event types.
#[derive(Clone, Debug)]
pub enum DemoEvent {
    /// Set ALL pops' sizes to N.
    EN { t: f64, n: f64 },
    /// Set one pop's size to N.
    En { t: f64, pop: u32, n: f64 },
    /// Set ALL pops' exp growth rate.
    EG { t: f64, alpha: f64 },
    /// Set one pop's exp growth rate.
    Eg { t: f64, pop: u32, alpha: f64 },
    /// Set ALL off-diagonal migration rates to M/(n_pops-1).
    EM { t: f64, m: f64 },
    /// Set migration rate from src→dst.
    Em { t: f64, dst: u32, src: u32, m: f64 },
    /// Merge: move all lineages in src into dst (going backward).
    Ej { t: f64, src: u32, dst: u32 },
}

impl DemoEvent {
    pub fn time(&self) -> f64 {
        match self {
            DemoEvent::EN { t, .. } => *t,
            DemoEvent::En { t, .. } => *t,
            DemoEvent::EG { t, .. } => *t,
            DemoEvent::Eg { t, .. } => *t,
            DemoEvent::EM { t, .. } => *t,
            DemoEvent::Em { t, .. } => *t,
            DemoEvent::Ej { t, .. } => *t,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Demography {
    pub n_pops: u32,
    pub pop_sizes: Vec<f64>,
    pub growth_rates: Vec<f64>,
    pub growth_start: Vec<f64>,
    /// Migration matrix: mig[dst][src] = rate from src → dst.
    pub migration_matrix: Vec<Vec<f64>>,
    pub events: Vec<DemoEvent>,
}

impl Demography {
    pub fn new(pop_sizes: Vec<f64>) -> Self {
        let n = pop_sizes.len() as u32;
        Self {
            n_pops: n,
            pop_sizes: pop_sizes.clone(),
            growth_rates: vec![0.0; n as usize],
            growth_start: vec![0.0; n as usize],
            migration_matrix: vec![vec![0.0; n as usize]; n as usize],
            events: Vec::new(),
        }
    }

    /// Single-population shorthand.
    pub fn single_pop(ne: f64) -> Self {
        Self::new(vec![ne])
    }

    pub fn add_event(&mut self, event: DemoEvent) {
        self.events.push(event);
        self.events.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
    }

    /// Effective size of `pop` at time `t` (backward), accounting for
    /// current growth rate.
    pub fn size_at(&self, pop: u32, t: f64) -> f64 {
        let p = pop as usize;
        if p >= self.pop_sizes.len() {
            return 1.0;
        }
        let n = self.pop_sizes[p];
        let g = self.growth_rates[p];
        if g == 0.0 {
            return n;
        }
        n * (-g * (t - self.growth_start[p])).exp()
    }

    /// Time of the next event at or after `t_now`, or +inf.
    pub fn next_event_time(&self, t_now: f64) -> f64 {
        for ev in &self.events {
            if ev.time() >= t_now - 1e-9 {
                return ev.time();
            }
        }
        f64::INFINITY
    }

    /// Apply all events scheduled at time `t`, mutating pop sizes /
    /// growth / migration and moving lineages for merge events.
    pub fn apply_events_at(&mut self, t: f64, active: &mut [Lineage]) {
        let mut remaining = Vec::new();
        let events = std::mem::take(&mut self.events);
        for ev in events {
            if (ev.time() - t).abs() > 1e-9 {
                remaining.push(ev);
                continue;
            }
            match ev {
                DemoEvent::EN { n, .. } => {
                    for p in 0..self.n_pops as usize {
                        self.pop_sizes[p] = n;
                        self.growth_rates[p] = 0.0;
                        self.growth_start[p] = t;
                    }
                }
                DemoEvent::En { pop, n, .. } => {
                    let p = pop as usize;
                    if p < self.pop_sizes.len() {
                        self.pop_sizes[p] = n;
                        self.growth_rates[p] = 0.0;
                        self.growth_start[p] = t;
                    }
                }
                DemoEvent::EG { alpha, .. } => {
                    for p in 0..self.n_pops as usize {
                        self.pop_sizes[p] = self.size_at(p as u32, t);
                        self.growth_rates[p] = alpha;
                        self.growth_start[p] = t;
                    }
                }
                DemoEvent::Eg { pop, alpha, .. } => {
                    let p = pop as usize;
                    if p < self.pop_sizes.len() {
                        self.pop_sizes[p] = self.size_at(pop, t);
                        self.growth_rates[p] = alpha;
                        self.growth_start[p] = t;
                    }
                }
                DemoEvent::EM { m, .. } => {
                    let per = if self.n_pops > 1 {
                        m / (self.n_pops - 1) as f64
                    } else {
                        0.0
                    };
                    for i in 0..self.n_pops as usize {
                        for j in 0..self.n_pops as usize {
                            if i != j {
                                self.migration_matrix[i][j] = per;
                            }
                        }
                    }
                }
                DemoEvent::Em { dst, src, m, .. } => {
                    let (d, s) = (dst as usize, src as usize);
                    if d < self.n_pops as usize && s < self.n_pops as usize
                        && d != s
                    {
                        self.migration_matrix[d][s] = m;
                    }
                }
                DemoEvent::Ej { src, dst, .. } => {
                    for lin in active.iter_mut() {
                        if lin.population == src {
                            lin.population = dst;
                        }
                    }
                    // Zero migration to/from src.
                    for k in 0..self.n_pops as usize {
                        self.migration_matrix[src as usize][k] = 0.0;
                        self.migration_matrix[k][src as usize] = 0.0;
                    }
                }
            }
        }
        self.events = remaining;
    }

    /// Compute per-lineage migration rates. Returns (rate, lineage_idx, dst_pop).
    pub fn migration_rates(&self, active: &[Lineage]) -> Vec<(f64, usize, u32)> {
        if self.n_pops < 2 {
            return Vec::new();
        }
        let mut rates = Vec::new();
        for (i, lin) in active.iter().enumerate() {
            let src = lin.population as usize;
            for dst in 0..self.n_pops as usize {
                if dst == src { continue; }
                let m = self.migration_matrix[dst][src];
                if m > 0.0 {
                    rates.push((m, i, dst as u32));
                }
            }
        }
        rates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pop_size() {
        let d = Demography::single_pop(10_000.0);
        assert_eq!(d.n_pops, 1);
        assert_eq!(d.size_at(0, 0.0), 10_000.0);
        assert_eq!(d.size_at(0, 100.0), 10_000.0); // no growth
    }

    #[test]
    fn exp_growth() {
        let mut d = Demography::single_pop(10_000.0);
        d.growth_rates[0] = 0.001;
        d.growth_start[0] = 0.0;
        // At t=100, N = 10000 * exp(-0.001 * 100) = 10000 * exp(-0.1)
        let expected = 10_000.0 * (-0.1f64).exp();
        assert!((d.size_at(0, 100.0) - expected).abs() < 1.0);
    }

    #[test]
    fn event_ordering() {
        let mut d = Demography::new(vec![1000.0, 1000.0]);
        d.add_event(DemoEvent::Ej { t: 500.0, src: 1, dst: 0 });
        d.add_event(DemoEvent::En { t: 200.0, pop: 0, n: 5000.0 });
        // Should be sorted by time.
        assert_eq!(d.events[0].time(), 200.0);
        assert_eq!(d.events[1].time(), 500.0);
    }

    #[test]
    fn next_event_time() {
        let mut d = Demography::new(vec![1000.0, 1000.0]);
        d.add_event(DemoEvent::Ej { t: 500.0, src: 1, dst: 0 });
        assert_eq!(d.next_event_time(0.0), 500.0);
        assert_eq!(d.next_event_time(600.0), f64::INFINITY);
    }

    #[test]
    fn merge_event_moves_lineages() {
        use crate::segment::{SegmentArena, SEG_NIL};
        let mut d = Demography::new(vec![1000.0, 1000.0]);
        d.add_event(DemoEvent::Ej { t: 100.0, src: 1, dst: 0 });

        let mut arena = SegmentArena::new();
        let seg0 = arena.alloc(0.0, 100.0, 0,
            crate::class_tag::BranchClass::PANMICTIC);
        let seg1 = arena.alloc(0.0, 100.0, 1,
            crate::class_tag::BranchClass::PANMICTIC);
        let mut active = vec![
            Lineage::new(seg0, seg0, 0, 0),
            Lineage::new(seg1, seg1, 1, 1),
        ];

        d.apply_events_at(100.0, &mut active);

        // Both lineages should now be in pop 0.
        assert_eq!(active[0].population, 0);
        assert_eq!(active[1].population, 0);
        // Migration from/to pop 1 should be zeroed.
        assert_eq!(d.migration_matrix[0][1], 0.0);
        assert_eq!(d.migration_matrix[1][0], 0.0);
    }
}
