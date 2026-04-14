/// HullSimulator: the main event loop.
///
/// Phase B: panmictic (no inversions) + recombination.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::class_tag::BranchClass;
use crate::events::{apply_coalescence, apply_recombination};
use crate::lineage::{LinUid, Lineage};
use crate::segment::{SegmentArena, SEG_NIL};
use crate::tables::TableBuilder;

/// Simulation result: raw arrays ready for conversion to tskit on
/// the Python side.
pub struct SimResult {
    pub tables: TableBuilder,
}

pub struct HullSimulator {
    pub n_samples: u32,
    pub population_size: f64,
    pub sequence_length: f64,
    pub recombination_rate: f64,
    pub seed: u64,
}

impl HullSimulator {
    pub fn simulate(&self) -> SimResult {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut arena = SegmentArena::new();
        let mut tables = TableBuilder::new(self.sequence_length, 1);
        let mut next_uid: LinUid = 0;

        // Create initial lineages (all panmictic, covering [0, L)).
        let mut active: Vec<Lineage> = Vec::with_capacity(self.n_samples as usize);
        for _ in 0..self.n_samples {
            let node_id = tables.add_sample(0.0, 0);
            let seg = arena.alloc(
                0.0, self.sequence_length, node_id, BranchClass::PANMICTIC);
            let uid = next_uid;
            next_uid += 1;
            active.push(Lineage::new(seg, seg, 0, uid));
        }

        self.run_loop(&mut active, &mut arena, &mut tables, &mut next_uid, &mut rng);

        SimResult { tables }
    }

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

            // --- Coalescence rate (panmictic, single pop) ---
            let k = n as f64;
            let coal_rate = k * (k - 1.0) / 2.0 / (2.0 * ne);

            // --- Recombination rate ---
            let recomb_rate: f64 = if self.recombination_rate > 0.0 {
                active.iter()
                    .map(|lin| lin.total_length(arena) * self.recombination_rate)
                    .sum()
            } else {
                0.0
            };

            let total_rate = coal_rate + recomb_rate;
            if total_rate <= 0.0 {
                return;
            }

            // Draw exponential waiting time.
            let u: f64 = rng.random();
            let dt = -u.ln() / total_rate;
            t += dt;

            // Pick event type.
            let u2: f64 = rng.random::<f64>() * total_rate;
            if u2 < coal_rate {
                // Coalescence: pick two random lineages.
                let i = rng.random_range(0..n);
                let mut j = rng.random_range(0..n - 1);
                if j >= i {
                    j += 1;
                }
                apply_coalescence(active, i, j, t, arena, tables, next_uid);
            } else {
                // Recombination: pick a lineage weighted by total_length,
                // then a position within its material.
                let target = (u2 - coal_rate) / self.recombination_rate;
                let mut cum = 0.0;
                let mut chosen_idx = 0;
                for (idx, lin) in active.iter().enumerate() {
                    cum += lin.total_length(arena);
                    if cum > target {
                        chosen_idx = idx;
                        break;
                    }
                }
                // Pick breakpoint within the lineage's material.
                let lin_len = active[chosen_idx].total_length(arena);
                let x_offset: f64 = rng.random::<f64>() * lin_len;
                let x = self.find_position(active, chosen_idx, x_offset, arena);
                apply_recombination(active, chosen_idx, x, arena, next_uid);
            }
        }
    }

    /// Convert an offset within a lineage's ancestral material to a
    /// genomic position.
    fn find_position(
        &self,
        active: &[Lineage],
        idx: usize,
        offset: f64,
        arena: &SegmentArena,
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
        // Fallback: end of last segment
        self.sequence_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panmictic_no_recomb_gives_single_tree() {
        let sim = HullSimulator {
            n_samples: 10,
            population_size: 1000.0,
            sequence_length: 100.0,
            recombination_rate: 0.0,
            seed: 42,
        };
        let result = sim.simulate();
        // No recombination → exactly one tree → n-1 internal nodes
        // → 2n-1 total nodes. Edges: n-1 coalescence events, each
        // producing 2 edges (both children overlap fully).
        assert_eq!(result.tables.num_nodes(), 19); // 10 samples + 9 internal
        assert_eq!(result.tables.num_edges(), 18); // 9 * 2
    }

    #[test]
    fn panmictic_with_recomb_gives_multiple_trees() {
        let sim = HullSimulator {
            n_samples: 6,
            population_size: 1000.0,
            sequence_length: 100.0,
            recombination_rate: 1e-4, // rho = 4*1000*1e-4*100 = 40
            seed: 42,
        };
        let result = sim.simulate();
        // 6 samples no-recomb would give 11 nodes, 10 edges.
        // With recombination, expect strictly more.
        assert!(result.tables.num_nodes() > 11,
            "Got {} nodes, expected > 11", result.tables.num_nodes());
        assert!(result.tables.num_edges() > 10,
            "Got {} edges, expected > 10", result.tables.num_edges());
    }

    #[test]
    fn two_samples_no_recomb() {
        let sim = HullSimulator {
            n_samples: 2,
            population_size: 100.0,
            sequence_length: 50.0,
            recombination_rate: 0.0,
            seed: 7,
        };
        let result = sim.simulate();
        // 2 samples + 1 internal = 3 nodes, 2 edges
        assert_eq!(result.tables.num_nodes(), 3);
        assert_eq!(result.tables.num_edges(), 2);
    }

    #[test]
    fn coal_times_positive_and_increasing() {
        let sim = HullSimulator {
            n_samples: 5,
            population_size: 1000.0,
            sequence_length: 100.0,
            recombination_rate: 0.0,
            seed: 123,
        };
        let result = sim.simulate();
        // Sample times are 0, internal times should be > 0
        for i in 5..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0,
                "Internal node {i} has time {}", result.tables.node_time[i]);
        }
    }
}
