//! Standalone bench binary for profiling.
//!
//! Runs a single-inversion workload at configurable rho and prints
//! wallclock per rep. Intended for `perf record` / flamegraph.
//!
//! Usage:
//!     cargo build --release --example bench_rho
//!     ./target/release/examples/bench_rho <rho> <reps> [n_pops] [compound] [Ne] [t_inv_mult]
//!
//! When n_pops > 1 each pop has 20/n_pops/2 std + inv lineages, with
//! symmetric migration at rate 1/(4 Ne) per lineage per unit time.
//! Pass "compound" as the 4th arg to run with the Path 2 compound
//! per-pair rate event loop instead of the bucket path.
//! `Ne` overrides effective size (default 1000). `t_inv_mult` scales
//! t_inv = t_inv_mult * Ne (default 2.0).

use std::env;
use std::time::Instant;

use msinv_core::demography::Demography;
use msinv_core::inversion::InversionSpec;
use msinv_core::rate_index::RateCache;
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::class_tag::Karyotype;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rho: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500.0);
    let reps: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n_pops: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let compound: bool = args.get(4).map(|s| s == "compound").unwrap_or(false);
    let ne: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1000.0);
    let t_inv_mult: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let n_samples: u32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(20);

    let l: f64 = 100_000.0;
    let r = rho / (4.0 * ne * l);
    let t_inv = t_inv_mult * ne;
    let inv = InversionSpec {
        bp_left: 30_000.0, bp_right: 70_000.0,
        p_inv: vec![0.5], t_inv,
        gene_conversion_rate: 1e-9, flux_window: 0.05, inv_id: 0,
    };

    let per_class = n_samples / 2;
    println!("rho={rho} reps={reps} n_pops={n_pops} Ne={ne} L={l} \
              n_samples={n_samples} t_inv={t_inv} compound={compound}");
    let mut rate_cache = RateCache::new(0, l);
    let t0 = Instant::now();
    for rep in 0..reps {
        let mut sim = if n_pops == 1 {
            HullSimulator::simple(
                per_class, per_class, ne, l, r, vec![inv.clone()], 3000 + rep)
        } else {
            build_multi_pop(n_pops, ne, l, r, vec![inv.clone()],
                             n_samples, 3000 + rep)
        };
        sim.compound_rate = compound;
        let result = sim.simulate_with_cache(&mut rate_cache);
        let nodes = result.tables.num_nodes();
        let edges = result.tables.num_edges();
        let stats = rate_cache.mem_stats();
        let total_bytes: usize = stats.iter().map(|(_, b)| *b).sum();
        eprintln!("rep {rep}: n={} cap={} nodes={nodes} edges={edges} \
                   cache_total={} MB",
                  rate_cache.n_active(), rate_cache.cap(),
                  total_bytes / (1024 * 1024));
        for (k, v) in &stats {
            if *v >= 1024 * 1024 {
                eprintln!("    {:20}  {:6} MB", k, v / (1024 * 1024));
            }
        }
    }
    let total = t0.elapsed().as_secs_f64();
    println!("total {:.3} s  mean {:.3} s/rep  {:.1} reps/s",
             total, total / reps as f64, reps as f64 / total);
}

fn build_multi_pop(
    n_pops: u32, ne: f64, l: f64, r: f64,
    invs: Vec<InversionSpec>, n_samples: u32, seed: u64,
) -> HullSimulator {
    let per_pop_std = (n_samples / 2).max(1) / n_pops;
    let per_pop_inv = (n_samples / 2).max(1) / n_pops;
    let n_inv_specs = invs.len();
    let mut samples = Vec::new();
    for p in 0..n_pops {
        if per_pop_std > 0 {
            samples.push(SampleEntry {
                karyotypes: vec![Some(Karyotype::S); n_inv_specs],
                population: p, count: per_pop_std,
            });
        }
        if per_pop_inv > 0 {
            samples.push(SampleEntry {
                karyotypes: vec![Some(Karyotype::I); n_inv_specs],
                population: p, count: per_pop_inv,
            });
        }
    }
    let mig_rate = 1.0 / (4.0 * ne);
    let mut demo = Demography::new(vec![ne; n_pops as usize]);
    for i in 0..n_pops as usize {
        for j in 0..n_pops as usize {
            if i != j { demo.migration_matrix[i][j] = mig_rate; }
        }
    }
    HullSimulator {
        samples,
        demography: demo,
        sequence_length: l,
        recombination_rate: r,
        inversions: invs,
        sweeps: Vec::new(),
        seed,
        stop_at: f64::INFINITY,
        compound_rate: false,
    }
}
