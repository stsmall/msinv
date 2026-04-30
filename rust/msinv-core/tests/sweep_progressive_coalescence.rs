//! Progressive coalescence sweep extension smoke test.
//!
//! Verifies the simulator runs to completion under the new per-allele
//! rate model emitted by `emit_coal_events_from_cache` (PG-B1) and
//! filtered by the CoalAggregate consumer (PG-C1).  Spec:
//! `docs/superpowers/specs/2026-04-30-sweep-progressive-coalescence-design.md`.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

fn progressive_sweep_sim(seed: u64) -> HullSimulator {
    let spec = JointSweepSpec {
        mode: SweepMode::Deterministic,
        s: 0.05,
        t_origin: 1_500.0,
        f0: 1.0 / 20_000.0,
        partial_sweep_final_freq: 1.0,
        seed,
        ..Default::default()
    };
    let sweep = Sweep::new(
        50_000.0,        // x_sel
        1_000.0,         // tau
        0,               // origin_pop
        Karyotype::S,
        0,               // target_inv
        spec,
    );
    HullSimulator {
        samples: vec![SampleEntry {
            karyotypes: vec![],
            population: 0,
            count: 10,
        }],
        demography: Demography::single_pop(10_000.0),
        sequence_length: 100_000.0,
        recombination_rate: 1e-8,
        inversions: vec![],
        sweeps: vec![sweep],
        seed,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    }
}

#[test]
fn pg1_progressive_sweep_completes() {
    // Deterministic hard sweep with progressive coalescence active in
    // the (origin_pop, origin_kary) cell. Confirms the simulator runs
    // to completion without panic and produces a non-trivial tree
    // sequence. n=10 → ≥19 nodes (n samples + n-1 internal MRCAs).
    let sim = progressive_sweep_sim(42);
    let result = sim.simulate();
    assert!(
        result.tables.num_nodes() >= 19,
        "Expected ≥19 nodes for n=10 sweep, got {}",
        result.tables.num_nodes()
    );
}
