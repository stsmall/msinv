//! Tier-1 (25% relative) anchor tests: trajectory output vs Kim-Stephan
//! closed forms.

use msinv_core::class_tag::Karyotype;
use msinv_core::sweep_kim_stephan as ks;
use msinv_core::sweep_trajectory::*;

const TOLERANCE: f64 = 0.25;

fn rel_err(observed: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-12 { return observed.abs(); }
    (observed - expected).abs() / expected.abs()
}

#[test]
fn a1_sojourn_time_matches_simulation() {
    // Sojourn time as defined by Kim-Stephan: time from f0 = 1/(2Ne)
    // up to near-fixation 1 - 1/(2Ne). Use that threshold for the
    // measurement to keep apples-to-apples with the closed form.
    let s = 0.01;
    let ne = 10_000.0;
    let expected = ks::sojourn_time(s, ne);
    let near_fix = 1.0 - 1.0 / (2.0 * ne);
    let spec = JointSweepSpec {
        mode: SweepMode::Deterministic,
        s,
        t_origin: 5.0 * expected,
        f0: 1.0 / (2.0 * ne),
        partial_sweep_final_freq: near_fix,
        ..Default::default()
    };
    let traj = build_joint_trajectory(
        &spec, 1, 0, Karyotype::S, &[0.0],
        &|_t, _p| ne, &|_, _, _| 0.0, 0.0,
    );
    let t_cross = traj.samples
        .iter()
        .find(|s_| s_.freq[0][CLASS_S_A_BENEF] >= near_fix)
        .map(|s_| spec.t_origin - s_.t)
        .unwrap_or(spec.t_origin);
    let err = rel_err(t_cross, expected);
    assert!(err < TOLERANCE, "sojourn observed={t_cross}, expected={expected}, rel_err={err}");
}

#[test]
fn a2_fixation_probability_over_reps() {
    let s = 0.05;
    let ne = 5_000.0;
    let expected = ks::fixation_probability(s);
    let n_reps = 1_000;
    let mut fixations = 0;
    for r in 0..n_reps {
        let spec = JointSweepSpec {
            mode: SweepMode::Stochastic,
            s,
            t_origin: 5_000.0,
            f0: 1.0 / (2.0 * ne),
            partial_sweep_final_freq: 0.95,
            seed: r as u64 + 1,
            ..Default::default()
        };
        let traj = build_joint_trajectory(
            &spec, 1, 0, Karyotype::S, &[0.0],
            &|_t, _p| ne, &|_, _, _| 0.0, 0.0,
        );
        if traj.samples.last().unwrap().freq[0][CLASS_S_A_BENEF] > 0.5 {
            fixations += 1;
        }
    }
    let observed = fixations as f64 / n_reps as f64;
    let sigma = (expected * (1.0 - expected) / n_reps as f64).sqrt();
    let err = rel_err(observed, expected);
    assert!(err < TOLERANCE || (observed - expected).abs() < 3.0 * sigma,
        "fix prob observed={observed}, expected={expected}, rel_err={err}");
}

#[test]
fn a3_pi_reduction_footprint_direction() {
    // Sketch: requires running a full coalescent sim with the trajectory
    // attached; full check is in tests/hull/test_phase6_sweep.py T3.
    // Here, just sanity-check the formula direction.
    let s = 0.01; let ne = 10_000.0; let recomb = 1e-8;
    let near = ks::pi_reduction_factor(s, ne, recomb, 100.0);
    let far  = ks::pi_reduction_factor(s, ne, recomb, 1e6);
    assert!(near < far, "near={near} should be < far={far}");
}

#[test]
fn a4_flux_mixing_time_inverse_relation() {
    let t_low_gamma  = ks::flux_mixing_time(1e-6, 1000.0);
    let t_high_gamma = ks::flux_mixing_time(1e-3, 1000.0);
    assert!(t_low_gamma > t_high_gamma);
    let ratio = t_low_gamma / t_high_gamma;
    assert!((ratio - 1000.0).abs() / 1000.0 < TOLERANCE);
}
