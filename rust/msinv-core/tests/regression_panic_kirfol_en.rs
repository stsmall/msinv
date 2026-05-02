//! Regression for the rate_index OOB panic on multi-pop + inversion +
//! `En` event raising ancestral Ne to 1M+.
//!
//! Originally reported in `project_panic_kirfol_en.md` (2026-04-23):
//! `RateCache::move_pair` indexed `pair_bucket_refs` past its length
//! (e.g. `len=857887, idx=857892`) when a Kir/Fol-style demography
//! merged pop 1 → pop 0 at t=14k and simultaneously raised pop 0's
//! Ne to 1M. Fixed by the rate_index session of 2026-04-22/23
//! (commits d441aaa, 9a30959, 2839022, 0f18a11): widened pack/unpack
//! to u64, sparse pair_bucket_refs, 1.5× grow factor + reset shrink.
//! This test pins the exact panic-trigger config so the fix can't
//! silently regress.
//!
//! Verified pre-fix: `git checkout ed54ba6 && cargo build --release`
//! → this scenario panics. Post-fix at HEAD: completes cleanly.

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{DemoEvent, Demography};
use msinv_core::inversion::InversionSpec;
use msinv_core::simulator::{HullSimulator, SampleEntry};

#[test]
fn en_to_1m_after_ej_does_not_panic() {
    let inv = InversionSpec::with_p_inv(
        180_000.0,
        380_000.0,
        vec![0.0, 0.73],
        330_000.0,
    );

    let mut demo = Demography::new(vec![44_000.0, 92_000.0]);
    demo.add_event(DemoEvent::Ej { t: 14_000.0, src: 1, dst: 0 });
    demo.add_event(DemoEvent::En { t: 14_000.0, pop: 0, n: 1_000_000.0 });

    let sim = HullSimulator {
        samples: vec![
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 0,
                count: 74,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::S)],
                population: 1,
                count: 23,
            },
            SampleEntry {
                karyotypes: vec![Some(Karyotype::I)],
                population: 1,
                count: 23,
            },
        ],
        demography: demo,
        sequence_length: 1_000_000.0,
        recombination_rate: 1e-8,
        inversions: vec![inv],
        sweeps: vec![],
        seed: 42,
        stop_at: f64::INFINITY,
        compound_rate: false,
        iters_max: 1_000_000,
        gc_stride: 160,
        record_events: false,
    };

    let result = sim.simulate();
    assert!(
        result.tables.num_nodes() > 120,
        "sim produced fewer nodes than the 120 input samples — \
         scenario degenerate, regression test no longer guarding panic"
    );
}
