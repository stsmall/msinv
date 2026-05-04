//! Microbench: HullIndex (arena AVL) vs the current Vec n-scan
//! pattern from `RateCache::recompute_for`.
//!
//! Synthetic workload: N lineages, each with a hull `[l, l + width]`
//! sampled from a uniform distribution; `Q` random query intervals
//! of similar shape. We time:
//!
//! * insert-N (build cost)
//! * Q overlap queries
//! * N updates (delete + reinsert)
//! * N removes
//!
//! For Vec n-scan we time the equivalent loop pattern from
//! `rate_index.rs:777-792`.
//!
//! Usage:
//!     cargo run --release --example bench_hull_index -- <N> <Q>
//! Defaults: N=20000 Q=20000.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use rand::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

use msinv_core::hull_index::HullIndex;

fn gen_hulls(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let l = rng.random::<f64>() * 1.0;
            let w = (rng.random::<f64>() * rng.random::<f64>()) * 0.05;
            (l, l + w)
        })
        .collect()
}

fn gen_queries(q: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    (0..q)
        .map(|_| {
            let l = rng.random::<f64>();
            let w = rng.random::<f64>() * 0.005;
            (l, l + w)
        })
        .collect()
}

fn vec_overlap_query(
    hulls: &[(f64, f64)], q_l: f64, q_r: f64, out: &mut Vec<u32>,
) {
    // Mirror of the rate_index.rs:777-792 inner pattern (population
    // filter + hull-interval test). We omit the bitmap test since
    // it's an additional shared filter unrelated to the spatial
    // structure being benched.
    for (idx, &(other_l, other_r)) in hulls.iter().enumerate() {
        if !(other_r > q_l && q_r > other_l) {
            continue;
        }
        out.push(idx as u32);
    }
}

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{:.2} ms", ns / 1_000_000.0)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let q: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    let hulls = gen_hulls(n, 0xc0ffee);
    let queries = gen_queries(q, 0xdeadbeef);

    println!("=== bench_hull_index N={} Q={} ===", n, q);

    // ---- HullIndex build ----
    let t0 = Instant::now();
    let mut hi = HullIndex::with_capacity(n);
    for (idx, &(l, r)) in hulls.iter().enumerate() {
        hi.insert(idx as u32, l, r);
    }
    let dt_build = t0.elapsed();
    println!(
        "HullIndex insert × {n}: {dt_build:?} ({} per op)",
        fmt_ns(dt_build.as_nanos() as f64 / n as f64),
    );

    // ---- HullIndex queries ----
    let mut out = Vec::with_capacity(64);
    let mut total_results_idx: u64 = 0;
    let t0 = Instant::now();
    for &(a, b) in &queries {
        out.clear();
        hi.iter_overlaps(a, b, &mut out);
        total_results_idx += out.len() as u64;
        black_box(&out);
    }
    let dt_query_idx = t0.elapsed();
    println!(
        "HullIndex iter_overlaps × {q}: {dt_query_idx:?} ({} per query, {:.1} avg results)",
        fmt_ns(dt_query_idx.as_nanos() as f64 / q as f64),
        total_results_idx as f64 / q as f64,
    );

    // ---- Vec n-scan queries ----
    let mut total_results_vec: u64 = 0;
    let t0 = Instant::now();
    for &(a, b) in &queries {
        out.clear();
        vec_overlap_query(&hulls, a, b, &mut out);
        total_results_vec += out.len() as u64;
        black_box(&out);
    }
    let dt_query_vec = t0.elapsed();
    println!(
        "Vec n-scan × {q}: {dt_query_vec:?} ({} per query, {:.1} avg results)",
        fmt_ns(dt_query_vec.as_nanos() as f64 / q as f64),
        total_results_vec as f64 / q as f64,
    );
    assert_eq!(
        total_results_idx, total_results_vec,
        "result-count mismatch: HullIndex={} Vec={}",
        total_results_idx, total_results_vec,
    );
    let speedup = dt_query_vec.as_nanos() as f64 / dt_query_idx.as_nanos() as f64;
    println!("query speedup vs Vec: {speedup:.2}×");

    // ---- HullIndex update (every lineage flicks its hull once) ----
    let new_hulls = gen_hulls(n, 0xfacefeed);
    let t0 = Instant::now();
    for (idx, &(l, r)) in new_hulls.iter().enumerate() {
        hi.update(idx as u32, l, r);
    }
    let dt_update = t0.elapsed();
    println!(
        "HullIndex update × {n}: {dt_update:?} ({} per op)",
        fmt_ns(dt_update.as_nanos() as f64 / n as f64),
    );

    // ---- HullIndex remove ----
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = Xoshiro256Plus::seed_from_u64(0x12345);
    order.shuffle(&mut rng);
    let t0 = Instant::now();
    for &idx in &order {
        hi.remove(idx);
    }
    let dt_remove = t0.elapsed();
    println!(
        "HullIndex remove × {n}: {dt_remove:?} ({} per op)",
        fmt_ns(dt_remove.as_nanos() as f64 / n as f64),
    );
    assert!(hi.is_empty());

    // ---- Vec maintenance reference: Vec::insert at random position ----
    // Mirrors the v3 rejection's failure mode — Vec::insert/remove
    // is O(n) shift. Only run if N <= 5000 to keep total bench time
    // sane (this scales O(N^2)).
    if n <= 5_000 {
        let mut sorted: Vec<(f64, f64, u32)> = Vec::with_capacity(n);
        let t0 = Instant::now();
        for (idx, &(l, r)) in hulls.iter().enumerate() {
            // Find insertion position (binary search on hull_l).
            let pos = sorted.binary_search_by(|p| {
                p.0.partial_cmp(&l).unwrap_or(std::cmp::Ordering::Equal)
            }).unwrap_or_else(|p| p);
            sorted.insert(pos, (l, r, idx as u32));
        }
        let dt_vec_insert = t0.elapsed();
        println!(
            "[reference] Vec::insert sorted × {n}: {dt_vec_insert:?} ({} per op)",
            fmt_ns(dt_vec_insert.as_nanos() as f64 / n as f64),
        );
        black_box(sorted);
    }
}
