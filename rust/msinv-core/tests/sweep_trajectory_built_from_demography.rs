//! Phase A acceptance test: simulator must build the joint sweep
//! trajectory using the live `Demography` accessors at run time, so
//! a sweep window that crosses an `En` event sees the correct pop
//! size in its trajectory.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{DemoEvent, Demography};
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{build_joint_trajectory, JointSweepSpec, SweepMode};

#[test]
fn simulator_builds_trajectory_from_demography_smoke() {
    // Smoke test: a sim with a sweep but no pre-built trajectory
    // runs without panicking. (Currently the run-loop tolerates a
    // None trajectory by no-op'ing apply_sweep — once A2 lands, the
    // trajectory is built and apply_sweep can do real work in B+.)
    let mut demo = Demography::new(vec![10_000.0]);
    demo.add_event(DemoEvent::En { t: 300.0, pop: 0, n: 100.0 });
    let spec = JointSweepSpec {
        mode: SweepMode::Deterministic,
        s: 0.05, t_origin: 600.0, f0: 0.001,
        partial_sweep_final_freq: 0.99,
        ..Default::default()
    };
    let sweep = Sweep::new(5_000.0, 0.0, 0, Karyotype::S, 0, spec);
    let sim = HullSimulator {
        samples: vec![SampleEntry {
            karyotypes: vec![],
            population: 0,
            count: 4,
        }],
        demography: demo,
        sequence_length: 10_000.0,
        recombination_rate: 1e-12,
        inversions: vec![],
        sweeps: vec![sweep],
        seed: 7,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 1_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 4);
}

/// Bottleneck Ne=100 inflates drift variance vs. constant Ne=10_000
/// when both are routed through the live demography. Verifies that
/// pop_size_at is what the trajectory build is consuming.
#[test]
fn trajectory_bottleneck_increases_drift_variance() {
    let mk_traj = |with_bottleneck: bool, seed: u64| {
        let mut demo = Demography::new(vec![10_000.0]);
        if with_bottleneck {
            demo.add_event(DemoEvent::En { t: 300.0, pop: 0, n: 100.0 });
        }
        let spec = JointSweepSpec {
            mode: SweepMode::Stochastic,
            s: 0.05, t_origin: 600.0, f0: 0.01,
            partial_sweep_final_freq: 1.0, seed,
            ..Default::default()
        };
        build_joint_trajectory(
            &spec, 1, 0, Karyotype::S, &[0.0],
            &|t, p| demo.pop_size_at(p, t),
            &|_t, _i, _j| 0.0, 0.0,
        )
    };
    let var_for = |bottleneck: bool| -> f64 {
        let finals: Vec<f64> = (0..40)
            .map(|r| mk_traj(bottleneck, r + 1).samples.last()
                .map(|s| s.freq[0][1]).unwrap_or(0.0))
            .collect();
        let mean = finals.iter().sum::<f64>() / finals.len() as f64;
        finals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finals.len() as f64
    };
    let var_with = var_for(true);
    let var_without = var_for(false);
    assert!(var_with > 2.0 * var_without,
        "expected bottleneck to inflate drift variance; with={var_with}, without={var_without}");
}
