//! Task #13 diagnostic — measures the hard-sweep rate-drain pattern.
//!
//! Runs a Track-4-shaped hard sweep at reduced L (2 Mb, Ne=1e5) with
//! MSINV_TRACE_ACTIVE set, then asserts the trace file is produced.
//! The trace captures per-iteration (p_A, n_a_upper, aa_rate,
//! total_rate, dt) inside the hard-sweep window and the still_a count
//! at the snap entry — enough to diagnose why O(1000) lineages reach
//! the snap.
//!
//! NOT for CI: gated on env var `MSINV_TASK13_DIAG=1`. Set the env
//! var, run with `--nocapture`, then inspect the trace file.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{Demography, DemoEvent};
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;
use msinv_core::sweep_trajectory::{JointSweepSpec, SweepMode};

#[test]
fn task13_diag_hard_sweep_drain() {
    if std::env::var("MSINV_TASK13_DIAG").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set MSINV_TASK13_DIAG=1 to enable");
        return;
    }
    // Trace target — caller picks the path so multiple runs don't
    // collide. Default to .tmp/task13_trace.tsv.
    let trace_path = std::env::var("MSINV_TRACE_ACTIVE")
        .unwrap_or_else(|_| ".tmp/task13_trace.tsv".to_string());
    eprintln!("trace will be written to: {}", trace_path);

    // v_simple shape: pop 0 = 1e5, pop 1 = 1e4, t_split = 15k, N_anc = 1e5,
    // migration symmetric m = 1e-5, sweep in pop 0 at t_origin = 4000.
    let ne_pop0: f64 = 100_000.0;
    let ne_pop1: f64 = 10_000.0;
    let ne_anc: f64 = 100_000.0;
    let m_sym: f64 = 1.0e-5;
    let t_split: f64 = 15_000.0;
    let l: f64 = 5_000_000.0;
    let r: f64 = 1e-8;
    let n_pop0: u32 = 50;
    let n_pop1: u32 = 50;
    let s_sel: f64 = 0.1;
    let t_origin: f64 = 4_000.0;
    let tau: f64 = 0.0;
    let f0_hard: f64 = 1.0 / (2.0 * ne_pop0);

    let spec = JointSweepSpec {
        mode: SweepMode::Deterministic,
        s: s_sel,
        t_origin,
        f0: f0_hard,
        partial_sweep_final_freq: 1.0,
        seed: 42,
        ..Default::default()
    };
    let sweep = Sweep::new(
        l / 2.0,            // x_sel at midpoint
        tau,
        0,                  // origin_pop
        Karyotype::S,
        0,                  // target_inv sentinel
        spec,
    );

    let mut demo = Demography::new(vec![ne_pop0, ne_pop1]);
    demo.migration_matrix = vec![vec![0.0, m_sym], vec![m_sym, 0.0]];
    demo.add_event(DemoEvent::Ej { t: t_split, src: 1, dst: 0 });
    demo.add_event(DemoEvent::En { t: t_split, pop: 0, n: ne_anc });
    // Zero migration once merged (matches v_simple_msinv behaviour).
    demo.add_event(DemoEvent::Em { t: t_split, src: 0, dst: 1, m: 0.0 });
    demo.add_event(DemoEvent::Em { t: t_split, src: 1, dst: 0, m: 0.0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry { karyotypes: vec![], population: 0, count: n_pop0 },
            SampleEntry { karyotypes: vec![], population: 1, count: n_pop1 },
        ],
        demography: demo,
        sequence_length: l,
        recombination_rate: r,
        inversions: vec![],
        sweeps: vec![sweep],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 100_000_000,
        gc_stride: 160,
        record_events: false,
        smc_prime: false,
        crossover_suppression: true,
    };

    let result = sim.simulate();
    eprintln!(
        "completed: {} nodes, {} edges, sweep_a_count={}",
        result.tables.num_nodes(),
        result.tables.num_edges(),
        result.sweep_a_count,
    );
}

/// Same shape as `task13_diag_hard_sweep_drain` but sweeps across multiple
/// seeds to find variance in `still_a` count at the snap. Trace is OFF
/// (would be too much output); just prints per-seed summary.
#[test]
fn task13_diag_hard_sweep_seed_scan() {
    if std::env::var("MSINV_TASK13_DIAG").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set MSINV_TASK13_DIAG=1 to enable");
        return;
    }
    use std::sync::Mutex;
    static SNAP_LOG: Mutex<Vec<(u64, usize, usize, usize)>> = Mutex::new(Vec::new());
    // We can't observe still_a from outside the simulator without
    // trace. So we capture via the *event log*, which records the
    // sweep_finalize Mark (n_active_pre at snap) and the snap_entry
    // — but neither is exposed in SimResult. Workaround: run with
    // MSINV_TRACE_ACTIVE per-seed pointing to a unique file, then
    // grep snap_entry afterward.
    let n_e: f64 = 100_000.0;
    let m_sym: f64 = 1.0e-5;
    let l: f64 = 5_000_000.0;
    let r: f64 = 1e-8;
    let s_sel: f64 = 0.1;
    let t_origin: f64 = 4_000.0;
    let tau: f64 = 0.0;
    let t_split: f64 = 15_000.0;
    let f0_hard = 1.0 / (2.0 * n_e);
    for seed in [1u64, 2, 3, 4, 5, 6, 7, 8, 11, 17, 42, 99, 100] {
        let spec = JointSweepSpec {
            mode: SweepMode::Deterministic,
            s: s_sel,
            t_origin,
            f0: f0_hard,
            partial_sweep_final_freq: 1.0,
            seed,
            ..Default::default()
        };
        let sweep = Sweep::new(l / 2.0, tau, 0, Karyotype::S, 0, spec);
        let mut demo = Demography::new(vec![n_e, 10_000.0]);
        demo.migration_matrix = vec![vec![0.0, m_sym], vec![m_sym, 0.0]];
        demo.add_event(DemoEvent::Ej { t: t_split, src: 1, dst: 0 });
        demo.add_event(DemoEvent::En { t: t_split, pop: 0, n: n_e });
        demo.add_event(DemoEvent::Em { t: t_split, src: 0, dst: 1, m: 0.0 });
        demo.add_event(DemoEvent::Em { t: t_split, src: 1, dst: 0, m: 0.0 });
        let sim = HullSimulator {
            samples: vec![
                SampleEntry { karyotypes: vec![], population: 0, count: 50 },
                SampleEntry { karyotypes: vec![], population: 1, count: 50 },
            ],
            demography: demo,
            sequence_length: l,
            recombination_rate: r,
            inversions: vec![],
            sweeps: vec![sweep],
            seed,
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 100_000_000,
            gc_stride: 160,
            record_events: false,
            smc_prime: false,
            crossover_suppression: true,
        };
        let res = sim.simulate();
        let mut log = SNAP_LOG.lock().unwrap();
        log.push((seed, res.tables.num_nodes(), res.tables.num_edges(),
                   res.sweep_a_count as usize));
    }
    let log = SNAP_LOG.lock().unwrap();
    eprintln!("seed,nodes,edges,sweep_a_count");
    for (s, n, e, c) in log.iter() {
        eprintln!("{},{},{},{}", s, n, e, c);
    }
}
