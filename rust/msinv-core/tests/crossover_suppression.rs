//! Heterokaryotype crossover suppression (`crossover_suppression`).
//!
//! With the whole sequence inverted, a constant frequency p, a barrier
//! older than any TMRCA and only arrangement-I samples, the model is one
//! population of size pN with recombination rate r p. msinv's panmictic
//! mode simulates that directly, so every statistic of the raw tables must
//! have the same distribution (here: node count and distinct edge
//! boundaries). With suppression off the rate is r (msinv <= 0.3.5).

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::Demography;
use msinv_core::inversion::InversionSpec;
use msinv_core::simulator::{HullSimulator, SampleEntry};

const N: f64 = 1_000.0;
const L: f64 = 10_000.0;
const R: f64 = 1e-6;
const P: f64 = 0.6;
const N_SAMPLES: u32 = 4;
const REPS: u64 = 300;

fn sim(inverted: bool, pop_size: f64, r: f64, suppress: bool, seed: u64)
    -> HullSimulator {
    let (inversions, karyotypes) = if inverted {
        let mut inv = InversionSpec::with_p_inv(0.0, L, vec![P], 1e12);
        inv.gene_conversion_rate = 1e-15;
        inv.mean_tract_length = 1.0;
        (vec![inv], vec![Some(Karyotype::I)])
    } else {
        (vec![], vec![])
    };
    HullSimulator {
        samples: vec![SampleEntry { karyotypes, population: 0, count: N_SAMPLES }],
        demography: Demography::single_pop(pop_size),
        sequence_length: L,
        recombination_rate: r,
        inversions,
        sweeps: vec![],
        seed,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
        smc_prime: false,
        crossover_suppression: suppress,
    }
}

/// (node count, distinct interior edge boundaries) for one replicate.
fn stats(s: &HullSimulator) -> (f64, f64) {
    let t = s.simulate().tables;
    let mut b: Vec<f64> = t.edge_left.iter().chain(t.edge_right.iter())
        .copied().filter(|&x| x > 0.0 && x < L).collect();
    b.sort_by(|a, c| a.partial_cmp(c).unwrap());
    b.dedup();
    (t.num_nodes() as f64, b.len() as f64)
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, v)
}

fn z(a: &[f64], b: &[f64]) -> f64 {
    let (ma, va) = mean_var(a);
    let (mb, vb) = mean_var(b);
    (ma - mb) / (va / a.len() as f64 + vb / b.len() as f64).sqrt()
}

fn compare(inv_r: f64, suppress: bool, pan_n: f64, pan_r: f64) -> [(f64, f64, f64); 2] {
    let a: Vec<(f64, f64)> = (0..REPS)
        .map(|s| stats(&sim(true, N, inv_r, suppress, 1_000 + s))).collect();
    let b: Vec<(f64, f64)> = (0..REPS)
        .map(|s| stats(&sim(false, pan_n, pan_r, true, 9_000 + s))).collect();
    let mut out = [(0.0, 0.0, 0.0); 2];
    for k in 0..2 {
        let xa: Vec<f64> = a.iter().map(|v| if k == 0 { v.0 } else { v.1 }).collect();
        let xb: Vec<f64> = b.iter().map(|v| if k == 0 { v.0 } else { v.1 }).collect();
        out[k] = (mean_var(&xa).0, mean_var(&xb).0, z(&xa, &xb));
    }
    out
}

#[test]
fn inverted_class_matches_rescaled_panmictic() {
    for (k, (ma, mb, zz)) in compare(R, true, P * N, R * P).iter().enumerate() {
        assert!(zz.abs() < 4.0, "stat {k}: inverted {ma} vs panmictic {mb}, z={zz}");
    }
}

#[test]
fn suppression_off_matches_full_rate_panmictic() {
    for (k, (ma, mb, zz)) in compare(R, false, P * N, R).iter().enumerate() {
        assert!(zz.abs() < 4.0, "stat {k}: inverted {ma} vs panmictic {mb}, z={zz}");
    }
}

#[test]
fn suppression_reduces_breakpoints() {
    // Same seeds, same model except the flag: the suppressed run has
    // clearly fewer boundaries on average (expected ratio ~p).
    let on: Vec<f64> = (0..REPS).map(|s| stats(&sim(true, N, R, true, 1_000 + s)).1).collect();
    let off: Vec<f64> = (0..REPS).map(|s| stats(&sim(true, N, R, false, 1_000 + s)).1).collect();
    assert!(z(&off, &on) > 10.0, "off {} on {}", mean_var(&off).0, mean_var(&on).0);
}
