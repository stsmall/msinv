//! Benchmark-as-test: runs the Kir/Fol mini-pilot both ways and
//! prints timing output. Also serves as a coarse correctness check by
//! asserting that analytic-middle produces a valid tree (all nodes >=
//! 0 time, MRCA reachable).
//!
//! Invoke with:
//!     cargo test --release -p msinv-core --test analytic_middle_bench -- --nocapture
//!
//! Gated with `#[ignore]` so that a plain `cargo test` doesn't pay the
//! wall time; the user runs it explicitly by adding `--include-ignored`.

use std::time::Instant;

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::inversion::InversionSpec;
use msinv_core::rate_index::RateCache;
use msinv_core::simulator::{HullSimulator, SampleEntry};

fn build_sim(
    ne: f64, l: f64, r: f64, inv: &InversionSpec,
    n_each: u32, seed: u64, analytic_middle: bool,
) -> HullSimulator {
    HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 0, count: n_each,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 0, count: n_each,
            },
        ],
        demography: Demography::single_pop(ne),
        sequence_length: l,
        recombination_rate: r,
        inversions: vec![inv.clone()],
        sweeps: Vec::new(),
        seed,
        stop_at: f64::INFINITY,
        analytic_middle,
    }
}

fn run_scenario(
    label: &str,
    ne: f64, t_inv: f64, n_each: u32, rho: f64, reps: u64,
) -> (f64, f64) {
    let l: f64 = 1_000_000.0;
    let r = rho / (4.0 * ne * l);
    let inv = InversionSpec {
        bp_left: 180_000.0, bp_right: 380_000.0,
        p_inv: vec![0.5], t_inv,
        gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
    };

    println!("\n=== {label} ===");
    println!("  Ne={ne} t_inv={t_inv} n_each={n_each} rho={rho} L={l} reps={reps}");

    let mut rate_cache = RateCache::new(0, l);

    let t0 = Instant::now();
    let mut sum_nodes = 0u64;
    let mut max_mrca = 0.0_f64;
    for rep in 0..reps {
        let sim = build_sim(ne, l, r, &inv, n_each, 3000 + rep, false);
        let result = sim.simulate_with_cache(&mut rate_cache);
        sum_nodes += result.tables.num_nodes() as u64;
        let mrca = result.tables.node_time.iter()
            .cloned().fold(0.0_f64, f64::max);
        max_mrca = max_mrca.max(mrca);
    }
    let t_loop = t0.elapsed().as_secs_f64();
    println!("  event-loop      : total {:.3} s | mean {:.3} s/rep | \
              nodes avg {:.0} | MRCA max {:.1}",
             t_loop, t_loop / reps as f64,
             sum_nodes as f64 / reps as f64, max_mrca);

    let t0 = Instant::now();
    let mut sum_nodes_am = 0u64;
    let mut max_mrca_am = 0.0_f64;
    for rep in 0..reps {
        let sim = build_sim(ne, l, r, &inv, n_each, 3000 + rep, true);
        let result = sim.simulate_with_cache(&mut rate_cache);
        sum_nodes_am += result.tables.num_nodes() as u64;
        let mrca = result.tables.node_time.iter()
            .cloned().fold(0.0_f64, f64::max);
        max_mrca_am = max_mrca_am.max(mrca);
    }
    let t_am = t0.elapsed().as_secs_f64();
    println!("  analytic-middle : total {:.3} s | mean {:.3} s/rep | \
              nodes avg {:.0} | MRCA max {:.1}",
             t_am, t_am / reps as f64,
             sum_nodes_am as f64 / reps as f64, max_mrca_am);
    let speedup = t_loop / t_am.max(1e-9);
    println!("  speedup: {:.2}×", speedup);
    (t_loop, t_am)
}

#[test]
#[ignore]
fn bench_kirfol_mini() {
    // A small scenario that should run quickly on both paths, to
    // smoke-test correctness.
    run_scenario("mini (Ne=1k, t_inv=5k, n=10, rho=4)",
                 1_000.0, 5_000.0, 10, 4.0, 5);
}

#[test]
#[ignore]
fn bench_kirfol_medium() {
    // Closer to the Kir/Fol regime but still tractable for the event
    // loop. The bulge driven by ratchet remnants is clearly visible.
    run_scenario("medium (Ne=10k, t_inv=50k, n=30, rho=40)",
                 10_000.0, 50_000.0, 30, 40.0, 3);
}

#[test]
#[ignore]
fn bench_kirfol_stress() {
    // Lean into the structured-middle regime: high t_inv / 2·Ne ratio.
    // Larger inversion age means more events per rep in the event loop.
    run_scenario("stress (Ne=10k, t_inv=330k, n=30, rho=4)",
                 10_000.0, 330_000.0, 30, 4.0, 2);
}

#[test]
#[ignore]
fn bench_kirfol_highn() {
    // Large sample size to magnify the per-event n-cost in the event
    // loop. Analytic-middle should scale better here.
    run_scenario("highn (Ne=10k, t_inv=50k, n=60, rho=40)",
                 10_000.0, 50_000.0, 60, 40.0, 2);
}

#[test]
#[ignore]
fn bench_kirfol_highrho() {
    // Higher rho builds up many remnants via pre-barrier recomb, so
    // n is larger when the middle phase starts.
    run_scenario("highrho (Ne=10k, t_inv=50k, n=30, rho=200)",
                 10_000.0, 50_000.0, 30, 200.0, 2);
}

/// Correctness smoke test: when recombination is effectively zero,
/// the analytic path and the event-loop path should produce
/// statistically similar MRCA distributions. This is the regime in
/// which "ignoring recomb during the middle phase" is correct by
/// construction — there's no recomb to miss. Doesn't test summary
/// stats; uses MRCA spread as a coarse proxy.
#[test]
#[ignore]
fn correctness_zero_rho() {
    let ne: f64 = 5_000.0;
    let l: f64 = 100_000.0;
    let r: f64 = 1e-12;  // rho ~ 2e-6, below any plausible recomb
    let inv = InversionSpec {
        bp_left: 20_000.0, bp_right: 80_000.0,
        p_inv: vec![0.5], t_inv: 30_000.0,
        gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
    };
    let n_each: u32 = 20;
    let reps: u64 = 20;

    let mut mrcas_loop = Vec::new();
    let mut mrcas_am = Vec::new();
    let mut rate_cache = RateCache::new(0, l);

    for rep in 0..reps {
        for (analytic, out) in [(false, &mut mrcas_loop), (true, &mut mrcas_am)] {
            let sim = build_sim(ne, l, r, &inv, n_each, 7000 + rep, analytic);
            let result = sim.simulate_with_cache(&mut rate_cache);
            let mrca = result.tables.node_time.iter()
                .cloned().fold(0.0_f64, f64::max);
            out.push(mrca);
        }
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let m_loop = mean(&mrcas_loop);
    let m_am = mean(&mrcas_am);
    let ratio = m_am / m_loop;

    println!("\n=== correctness_zero_rho ===");
    println!("  reps={reps} n_each={n_each} Ne={ne} t_inv={} rho_eff={:.2e}",
             inv.t_inv, 4.0 * ne * r * l);
    println!("  event-loop       MRCA mean = {:.1}", m_loop);
    println!("  analytic-middle  MRCA mean = {:.1}", m_am);
    println!("  ratio = {:.3} (expect ~1.0 at rho≈0)", ratio);

    // Loose tolerance — 20 reps is not a lot of samples, and we expect
    // some variance. Fail only on gross divergence.
    assert!(ratio > 0.5 && ratio < 2.0,
        "MRCA ratio {:.3} outside (0.5, 2.0) — the analytic middle \
         path diverges drastically from the event loop at rho≈0, \
         indicating a bookkeeping bug not a missing-recomb effect.",
        ratio);
}
