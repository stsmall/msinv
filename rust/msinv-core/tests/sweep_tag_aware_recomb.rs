//! Tests for the discoal-style tag rejection-sampling on recombination
//! during the sweep window. Spec:
//! `docs/superpowers/specs/2026-04-30-sweep-tag-aware-recombination-design.md`.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

fn build_sweep(f0: f64, mode: SweepMode, seed: u64, t_origin: f64) -> Sweep {
    let spec = JointSweepSpec {
        mode,
        s: 0.05,
        t_origin,
        f0,
        partial_sweep_final_freq: 1.0,
        seed,
        ..Default::default()
    };
    Sweep::new(50_000.0, 1_000.0, 0, Karyotype::S, 0, spec)
        .with_trajectory(
            1, &[0.0],
            &|_t, _p| 10_000.0,
            &|_t, _i, _j| 0.0,
        )
}

#[test]
fn tr1_simulator_completes_with_in_window_recombination() {
    // Soft sweep with f0=0.05 + non-trivial recombination rate. The
    // in-window recombs should fire the tag-swap helper many times;
    // the simulator must reach MRCA without panic.
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42, 1500.0);
    let sim = HullSimulator {
        samples: vec![SampleEntry {
            karyotypes: vec![],
            population: 0,
            count: 10,
        }],
        demography: Demography::single_pop(10_000.0),
        sequence_length: 100_000.0,
        recombination_rate: 1e-8,
        inversions: vec![],
        sweeps: vec![sw],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 100_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(
        result.tables.num_nodes() >= 19,
        "Expected ≥19 nodes for n=10 sweep; got {}",
        result.tables.num_nodes()
    );
}

#[test]
fn tr1_hard_sweep_unchanged() {
    // f0 = 1/(2N) → no SV phase → only selection-phase recombs. Even
    // those fire the swap, but at p_A near 1 (early sweep, going
    // backward) the swap is mostly a no-op. The simulation should
    // complete and produce the expected node count.
    let sw = build_sweep(1.0 / 20_000.0, SweepMode::Deterministic, 42, 1500.0);
    let sim = HullSimulator {
        samples: vec![SampleEntry {
            karyotypes: vec![],
            population: 0,
            count: 10,
        }],
        demography: Demography::single_pop(10_000.0),
        sequence_length: 100_000.0,
        recombination_rate: 1e-8,
        inversions: vec![],
        sweeps: vec![sw],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 100_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(
        result.tables.num_nodes() >= 19,
        "Expected ≥19 nodes for n=10 hard sweep; got {}",
        result.tables.num_nodes()
    );
}
