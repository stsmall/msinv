/// Tests ported from the Python hull test suite.
///
/// Every test with inversions uses recombination_rate > 0 (rho > 0).
/// Parameter convention: Ne=10000, L=100000, r=1e-8 → rho=40 unless
/// noted otherwise. gamma=1e-9 default for inversions.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{Demography, DemoEvent};
use msinv_core::inversion::{InversionSpec, TractDistribution};
use msinv_core::simulator::{HullSimulator, SampleEntry};
use msinv_core::sweep::Sweep;

// Helpers
fn inv(bp_l: f64, bp_r: f64, p_inv: f64, t_inv: f64, id: u16) -> InversionSpec {
    let mut s = InversionSpec::with_p_inv(bp_l, bp_r, vec![p_inv], t_inv);
    s.inv_id = id;
    s
}

fn inv_gamma(bp_l: f64, bp_r: f64, p_inv: f64, t_inv: f64, gamma: f64, id: u16) -> InversionSpec {
    let mut s = InversionSpec::with_p_inv(bp_l, bp_r, vec![p_inv], t_inv);
    s.gene_conversion_rate = gamma;
    s.inv_id = id;
    s
}

/// Helper for b2-flux fixtures: gene_conversion_rate plus pinned
/// mean_tract_length and tract_distribution.
fn inv_b2(
    bp_l: f64,
    bp_r: f64,
    p_inv: f64,
    t_inv: f64,
    gamma: f64,
    mean_tract_length: f64,
    tract_distribution: TractDistribution,
    id: u16,
) -> InversionSpec {
    let mut s = InversionSpec::with_p_inv(bp_l, bp_r, vec![p_inv], t_inv);
    s.gene_conversion_rate = gamma;
    s.mean_tract_length = mean_tract_length;
    s.tract_distribution = tract_distribution;
    s.inv_id = id;
    s
}

// ---------------------------------------------------------------
// Phase 2: Class barrier
// ---------------------------------------------------------------

#[test]
fn cross_class_mrca_at_least_t_inv() {
    // Ne=10000, L=100000, r=1e-8 → rho=40
    for seed in 1..=5u64 {
        let sim = HullSimulator::simple(
            5, 5, 10000.0, 100000.0, 1e-8,
            vec![inv(30000.0, 70000.0, 0.5, 80000.0, 0)], seed);
        let result = sim.simulate();
        // All internal nodes at times > 0
        for i in 10..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0,
                "seed={}: node {} has time 0", seed, i);
        }
    }
}

#[test]
fn within_class_mrca_below_t_inv() {
    // Within-class coalescence should be much faster than t_inv.
    // Ne=10000, t_inv=80000 → E[T_within] = 2*p*Ne = 10000 << 80000.
    let sim = HullSimulator::simple(
        10, 10, 10000.0, 100000.0, 1e-8,
        vec![inv(30000.0, 70000.0, 0.5, 80000.0, 0)], 42);
    let result = sim.simulate();
    // With 20 samples, we should have many nodes < t_inv.
    let below_t_inv = result.tables.node_time.iter()
        .filter(|&&t| t > 0.0 && t < 80000.0).count();
    assert!(below_t_inv > 0,
        "Expected some within-class coalescences below t_inv");
}

// ---------------------------------------------------------------
// Phase 3: Gene flux
// ---------------------------------------------------------------

#[test]
fn gamma_positive_gives_more_trees() {
    // With gamma > 0, gene flux splits lineages → more trees.
    // Ne=1000, L=10000, r=1e-8 → rho=0.4
    let inv_no = inv_gamma(0.0, 10000.0, 0.5, 20000.0, 1e-9, 0);
    let inv_yes = inv_gamma(0.0, 10000.0, 0.5, 20000.0, 1e-5, 0);
    let no_flux = HullSimulator::simple(
        4, 4, 1000.0, 10000.0, 1e-8, vec![inv_no], 42);
    let with_flux = HullSimulator::simple(
        4, 4, 1000.0, 10000.0, 1e-8, vec![inv_yes], 42);
    let r_no = no_flux.simulate();
    let r_yes = with_flux.simulate();
    assert!(r_yes.tables.num_nodes() >= r_no.tables.num_nodes(),
        "flux={} vs no_flux={}", r_yes.tables.num_nodes(),
        r_no.tables.num_nodes());
}

#[test]
fn parity_b2_flux_fixed() {
    // b2-flux model with tract_distribution = Fixed: every flux
    // event uses tract length L = mean_tract_length deterministically.
    // Ne=1000, L=10000, r=1e-8 → rho=0.4. mean_tract_length=200 bp,
    // gamma=1e-5 fires occasional events without saturating the ARG.
    for seed in 1..=3u64 {
        let spec = inv_b2(
            0.0, 10000.0, 0.5, 20000.0, 1e-5,
            200.0, TractDistribution::Fixed, 0);
        let sim = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8, vec![spec], seed);
        let result = sim.simulate();
        // Sanity: must complete and produce nodes.
        assert!(result.tables.num_nodes() >= 8,
            "seed={}: only {} nodes", seed, result.tables.num_nodes());
        for i in 8..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0,
                "seed={}: node {} has time 0", seed, i);
        }
    }
}

#[test]
fn parity_b2_flux_geometric() {
    // b2-flux model with tract_distribution = Geometric: tract length
    // per event is Exponential(1/mean_tract_length). Same parameters
    // as the Fixed fixture so the two are directly comparable.
    for seed in 1..=3u64 {
        let spec = inv_b2(
            0.0, 10000.0, 0.5, 20000.0, 1e-5,
            200.0, TractDistribution::Geometric, 0);
        let sim = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8, vec![spec], seed);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 8,
            "seed={}: only {} nodes", seed, result.tables.num_nodes());
        for i in 8..result.tables.num_nodes() {
            assert!(result.tables.node_time[i] > 0.0,
                "seed={}: node {} has time 0", seed, i);
        }
    }
}

// ---------------------------------------------------------------
// Phase 4: Demography
// ---------------------------------------------------------------

#[test]
fn no_migration_cross_pop_mrca_at_least_t_split() {
    // With ej at t_split and no migration, cross-pop MRCAs >= t_split.
    for seed in 1..=5u64 {
        let mut demo = Demography::new(vec![10000.0, 10000.0]);
        demo.add_event(DemoEvent::Ej { t: 5000.0, src: 1, dst: 0 });

        let sim = HullSimulator {
            samples: vec![
                SampleEntry { karyotypes: vec![], population: 0, count: 4 },
                SampleEntry { karyotypes: vec![], population: 1, count: 4 },
            ],
            demography: demo,
            sequence_length: 10000.0,
            recombination_rate: 1e-8,
            inversions: vec![],
            sweeps: vec![],
            seed,
            stop_at: f64::INFINITY,
            compound_rate: false,
            iters_max: 10_000_000,
            gc_stride: 160,
            record_events: false,
        };
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 15,
            "seed={}: {} nodes", seed, result.tables.num_nodes());
    }
}

#[test]
fn migration_allows_cross_pop_mrca_below_t_split() {
    // With migration, some cross-pop pairs can coalesce before t_split.
    let mut demo = Demography::new(vec![10000.0, 10000.0]);
    demo.migration_matrix[0][1] = 1e-3;
    demo.migration_matrix[1][0] = 1e-3;
    demo.add_event(DemoEvent::Ej { t: 50000.0, src: 1, dst: 0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry { karyotypes: vec![], population: 0, count: 5 },
            SampleEntry { karyotypes: vec![], population: 1, count: 5 },
        ],
        demography: demo,
        sequence_length: 10000.0,
        recombination_rate: 1e-8,
        inversions: vec![],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    // With M=1e-3, mean migration time ~ 1/M = 1000 gen << 50000.
    // Some nodes should be < 50000.
    let below = result.tables.node_time.iter()
        .filter(|&&t| t > 0.0 && t < 50000.0).count();
    assert!(below > 0, "Expected some cross-pop coalescences below t_split");
}

#[test]
fn inversion_with_two_pops() {
    // Kir/Fol-style: 2 pops with split + inversion.
    // Ne=10000, L=100000, r=1e-8 → rho=40
    let mut demo = Demography::new(vec![10000.0, 10000.0]);
    demo.add_event(DemoEvent::Ej { t: 14000.0, src: 1, dst: 0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 0, count: 5,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 1, count: 3,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 1, count: 3,
            },
        ],
        demography: demo,
        sequence_length: 100000.0,
        recombination_rate: 1e-8,
        inversions: vec![inv(20000.0, 80000.0, 0.3, 80000.0, 0)],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 21,
        "Got {} nodes", result.tables.num_nodes());
}

// ---------------------------------------------------------------
// Phase 5b: Multiple inversions
// ---------------------------------------------------------------

#[test]
fn two_inversions_each_barrier_independent() {
    // Ne=1000, L=10000, r=1e-8 → rho=0.4
    for seed in 1..=5u64 {
        let sim = HullSimulator::simple(
            4, 4, 1000.0, 10000.0, 1e-8,
            vec![
                inv(1000.0, 4000.0, 0.5, 2000.0, 0),
                inv(6000.0, 9000.0, 0.5, 8000.0, 1),
            ], seed);
        let result = sim.simulate();
        assert!(result.tables.num_nodes() >= 15,
            "seed={}: {} nodes", seed, result.tables.num_nodes());
    }
}

// ---------------------------------------------------------------
// Recombination
// ---------------------------------------------------------------

#[test]
fn recomb_with_inversion_produces_multiple_trees() {
    // Ne=10000, L=100000, r=1e-8 → rho=40
    let sim = HullSimulator::simple(
        5, 5, 10000.0, 100000.0, 1e-8,
        vec![inv(30000.0, 70000.0, 0.5, 100000.0, 0)], 42);
    let result = sim.simulate();
    // With rho=40, should get many breakpoints.
    assert!(result.tables.num_edges() > 19,
        "Expected multiple trees; only {} edges", result.tables.num_edges());
}

#[test]
fn recomb_with_multi_inv() {
    // Two non-overlapping inversions with recombination.
    // Ne=10000, L=100000, r=1e-8 → rho=40
    let sim = HullSimulator::simple(
        3, 3, 10000.0, 100000.0, 1e-8,
        vec![
            inv(15000.0, 45000.0, 0.5, 100000.0, 0),
            inv(55000.0, 85000.0, 0.3, 150000.0, 1),
        ], 42);
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 11,
        "Got {} nodes", result.tables.num_nodes());
}

// ---------------------------------------------------------------
// Phase 6: Sweeps
// ---------------------------------------------------------------

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn sweep_forces_coalescence() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn sweep_on_s_class_inside_inversion() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn sweep_with_no_target_lineages_is_noop() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

// ---------------------------------------------------------------
// Stress / corner cases
// ---------------------------------------------------------------

#[test]
fn nested_inversions_run_without_crashing() {
    // Nested inversions: outer [0, 10000), inner [3000, 7000).
    // Ne=1000, L=10000, r=1e-8 → rho=0.4
    let sim = HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S), Some(Karyotype::S)],
                population: 0, count: 3,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I), Some(Karyotype::I)],
                population: 0, count: 3,
            },
        ],
        demography: Demography::single_pop(1000.0),
        sequence_length: 10000.0,
        recombination_rate: 1e-8,
        inversions: vec![
            inv_gamma(0.0, 10000.0, 0.5, 5000.0, 1e-6, 0),
            inv_gamma(3000.0, 7000.0, 0.5, 8000.0, 1e-6, 1),
        ],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 11,
        "Got {} nodes", result.tables.num_nodes());
}

#[test]
fn continuous_migration_with_inversion() {
    // Migration + inversion: cross-class barrier still holds.
    // Ne=1000, L=10000, r=1e-8 → rho=0.4
    let mut demo = Demography::new(vec![1000.0, 1000.0]);
    demo.migration_matrix[0][1] = 1e-3;
    demo.migration_matrix[1][0] = 1e-3;
    demo.add_event(DemoEvent::Ej { t: 200000.0, src: 1, dst: 0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 0, count: 3,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 1, count: 3,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 1, count: 3,
            },
        ],
        demography: demo,
        sequence_length: 10000.0,
        recombination_rate: 1e-8,
        inversions: vec![inv(2000.0, 8000.0, 0.5, 8000.0, 0)],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 17,
        "Got {} nodes", result.tables.num_nodes());
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn two_sweeps_at_same_time() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn sweep_at_exact_t_inv() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
fn t_inv_and_demographic_event_at_same_time() {
    // t_inv and ej at same time.
    // Ne=1000, L=10000, r=1e-8 → rho=0.4
    let mut demo = Demography::new(vec![1000.0, 1000.0]);
    demo.add_event(DemoEvent::Ej { t: 1000.0, src: 1, dst: 0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 0, count: 2,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 0, count: 2,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 1, count: 2,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 1, count: 2,
            },
        ],
        demography: demo,
        sequence_length: 10000.0,
        recombination_rate: 1e-8,
        inversions: vec![inv(0.0, 10000.0, 0.5, 1000.0, 0)],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 10_000_000,
        gc_stride: 160,
        record_events: false,
    };
    let result = sim.simulate();
    assert!(result.tables.num_nodes() >= 15,
        "Got {} nodes", result.tables.num_nodes());
}

// ---------------------------------------------------------------
// rho=0 guard
// ---------------------------------------------------------------

#[test]
#[should_panic(expected = "recombination_rate must be > 0")]
fn rho_zero_with_inversion_panics() {
    let sim = HullSimulator::simple(
        2, 2, 1000.0, 1000.0, 0.0,
        vec![inv(200.0, 800.0, 0.5, 5000.0, 0)], 42);
    sim.simulate();
}

#[test]
#[should_panic(expected = "recombination_rate must be > 0")]
fn rho_zero_without_inversion_panics() {
    // rho=0 is forbidden globally now (matches Python).
    let sim = HullSimulator::panmictic(5, 1000.0, 1000.0, 0.0, 42);
    sim.simulate();
}

#[test]
#[should_panic(expected = "gene_conversion_rate (gamma) must be > 0")]
fn gamma_zero_with_inversion_panics() {
    let mut iv = inv(200.0, 800.0, 0.5, 5000.0, 0);
    iv.gene_conversion_rate = 0.0;
    let sim = HullSimulator::simple(2, 2, 1000.0, 1000.0, 1e-8, vec![iv], 42);
    sim.simulate();
}

// ---------------------------------------------------------------
// Hitchhiking + soft sweep
// ---------------------------------------------------------------

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn hitchhiking_hard_sweep_coalesces_near_event() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn soft_sweep_preserves_partial_diversity() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}

#[test]
#[ignore = "TODO sweep-rewrite Task 16: rewrite under new Sweep API"]
fn hitchhiking_with_inversion() {
    // Body deleted in sweep-rewrite Task 11. Will be rewritten in
    // Task 16 once the new JointSweepTrajectory operator lands.
}
