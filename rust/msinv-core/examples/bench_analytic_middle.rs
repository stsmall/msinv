//! Head-to-head benchmark for feature/structured-analytic-middle.
//!
//! Runs an anopheles Kir/Fol-style workload — single population,
//! single inversion, barrier from present to old t_inv — both with the
//! standard event loop (`analytic_middle=false`) and with the analytic
//! fast path (`analytic_middle=true`). Prints per-rep wall time, total
//! edges, and MRCA time per path.
//!
//! Usage:
//!     cargo build --release --example bench_analytic_middle
//!     ./target/release/examples/bench_analytic_middle [reps] [ne] [t_inv] [n_samples] [rho]
//!
//! Default params mimic the kirfol pilot but scaled down so the stock
//! event-loop path finishes in reasonable time. The pre-existing
//! `rate_index.rs` panic at Ne_anc=1e6 + multi-pop still blocks the
//! full-scale Kir/Fol comparison; see project_panic_kirfol_en.md.

use std::env;
use std::time::Instant;

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::inversion::InversionSpec;
use msinv_core::rate_index::RateCache;
use msinv_core::simulator::{HullSimulator, SampleEntry};

fn main() {
    let args: Vec<String> = env::args().collect();
    let reps: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let ne: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000.0);
    let t_inv: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50_000.0);
    let n_each: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);
    let rho: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(40.0);

    let l: f64 = 1_000_000.0;
    let r = rho / (4.0 * ne * l);
    let inv = InversionSpec {
        bp_left: 180_000.0, bp_right: 380_000.0,
        p_inv: vec![0.5], t_inv,
        gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
    };

    println!("Kir/Fol mini-pilot: Ne={ne} t_inv={t_inv} n_each={n_each} \
              rho={rho} L={l} reps={reps}");
    println!("rho_eff = 4*Ne*r*L = {:.2}", 4.0 * ne * r * l);

    let mut rate_cache = RateCache::new(0, l);

    println!("\n-- standard event loop (analytic_middle=false) --");
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
    println!("total {:.3} s | mean {:.3} s/rep | nodes avg {:.0} | MRCA max {:.1}",
             t_loop, t_loop / reps as f64,
             sum_nodes as f64 / reps as f64, max_mrca);

    println!("\n-- analytic middle (analytic_middle=true) --");
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
    println!("total {:.3} s | mean {:.3} s/rep | nodes avg {:.0} | MRCA max {:.1}",
             t_am, t_am / reps as f64,
             sum_nodes_am as f64 / reps as f64, max_mrca_am);

    println!("\nSpeedup: {:.2}× (analytic vs event-loop)",
             t_loop / t_am.max(1e-9));
}

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
