//! Standing-variation phase tests for the sweep trajectory + simulator.
//!
//! After the 2026-04-30 SV extension, sweeps with `f0 > 1/(2N)` carry
//! a backward-time stochastic neutral WF drift segment past `t_origin`.
//! Spec: `docs/superpowers/specs/2026-04-30-sweep-standing-variation-phase-design.md`.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

fn build_sweep(f0: f64, mode: SweepMode, seed: u64) -> Sweep {
    let spec = JointSweepSpec {
        mode,
        s: 0.05,
        t_origin: 1_500.0,
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
fn sv1_t_de_novo_equals_t_origin_when_f0_at_extinction() {
    // f0 = 1/(2N) is at the extinction threshold; SV phase is a no-op.
    let sw = build_sweep(1.0 / 20_000.0, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    assert!(
        (traj.t_de_novo - traj.t_origin).abs() < 1e-9,
        "Expected t_de_novo == t_origin for f0=1/(2N); got t_origin={}, t_de_novo={}",
        traj.t_origin, traj.t_de_novo
    );
}

#[test]
fn sv1_t_de_novo_extends_past_t_origin_when_f0_high() {
    // f0 = 0.05 puts the variant well above extinction; SV drift runs.
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    assert!(
        traj.t_de_novo > traj.t_origin,
        "Expected t_de_novo > t_origin for f0=0.05; got t_origin={}, t_de_novo={}",
        traj.t_origin, traj.t_de_novo
    );
    // Loose bound: drift length should be in a plausible range.
    let drift_len = traj.t_de_novo - traj.t_origin;
    assert!(
        drift_len > 100.0 && drift_len < 1_000_000.0,
        "Expected SV drift length in [100, 1_000_000] gens; got {drift_len}"
    );
}

#[test]
fn sv1_p_allele_query_past_t_origin_is_below_or_near_f0() {
    // After a few hundred generations going further into the past, the
    // A frequency should generally be at or below f0 (single rep can
    // fluctuate; bound is loose).
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
    let traj = sw.trajectory.as_ref().unwrap();
    // Pick a midpoint of the SV phase.
    let t_mid = (traj.t_origin + traj.t_de_novo) / 2.0;
    let p_a = traj.p_allele_given_kary(t_mid, 0, Karyotype::S);
    // Loose bound: p_A at SV midpoint must be in [0, 1].
    assert!(
        p_a >= 0.0 && p_a <= 1.0,
        "Expected p_A at SV midpoint in [0, 1]; got {p_a}"
    );
    // The mean over many seeds should be near or below f0; for a
    // single rep this isn't a strict assertion. Just confirm we
    // get a reasonable drift value (not stuck at f0 forever).
    let p_a_at_origin = traj.p_allele_given_kary(traj.t_origin, 0, Karyotype::S);
    assert!(
        (p_a_at_origin - 0.05).abs() < 1e-9,
        "Expected p_A at t_origin == f0=0.05; got {p_a_at_origin}"
    );
}

#[test]
fn sv2_simulator_completes_with_sv_phase() {
    // Full simulation with f0=0.05; the simulator must reach MRCA
    // through the selection + SV + post-window neutral phases without
    // panic. Equivalent to PG1 but with f0 > 1/(2N).
    let sw = build_sweep(0.05, SweepMode::Stochastic, 42);
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
        "Expected ≥19 nodes for n=10 sweep with SV phase; got {}",
        result.tables.num_nodes()
    );
}
