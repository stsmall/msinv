//! Standalone bench binary for profiling.
//!
//! Runs a single-inversion workload at configurable rho and prints
//! wallclock per rep. Intended for `perf record` / flamegraph.
//!
//! Usage:
//!     cargo build --release --example bench_rho
//!     perf record -g --call-graph=dwarf ./target/release/examples/bench_rho 500 10
//!     perf report --stdio | head -40

use std::env;
use std::time::Instant;

use msinv_core::inversion::InversionSpec;
use msinv_core::simulator::HullSimulator;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rho: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500.0);
    let reps: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let ne: f64 = 1000.0;
    let l: f64 = 100_000.0;
    let r = rho / (4.0 * ne * l);
    let t_inv = 2.0 * ne;
    let inv = InversionSpec {
        bp_left: 30_000.0, bp_right: 70_000.0,
        p_inv: vec![0.5], t_inv,
        gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
    };

    println!("rho={rho} reps={reps} n=20 Ne={ne} L={l}");
    let t0 = Instant::now();
    for rep in 0..reps {
        let sim = HullSimulator::simple(
            10, 10, ne, l, r, vec![inv.clone()], 3000 + rep);
        let _result = sim.simulate();
    }
    let total = t0.elapsed().as_secs_f64();
    println!("total {:.3} s  mean {:.3} s/rep  {:.1} reps/s",
             total, total / reps as f64, reps as f64 / total);
}
