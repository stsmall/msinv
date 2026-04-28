//! Optional event log capturing cmig and flux events as they fire.
//!
//! Off by default; enabled via `HullSimulator::record_events`. When on,
//! the simulator pushes a [`CmigRecord`] per scheduled cmig event and a
//! [`FluxRecord`] per flux fire. Used by validation tests T3 (cmig
//! Binomial) and Tier 3-cheap Q5a/Q5b (flux survival + Andolfatto).

use crate::class_tag::Karyotype;
use crate::lineage::LinUid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmigRecord {
    pub t: f64,
    pub src: u32,
    pub dst: u32,
    pub kary: Karyotype,
    pub inv_id: u16,
    pub n_eligible: u32,
    pub n_moved: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FluxRecord {
    pub t: f64,
    pub lineage_uid: LinUid,
    pub position: f64,
    pub tract_left: f64,
    pub tract_right: f64,
    pub inv_id: u16,
    /// Per-segment node IDs spanning the tract at the moment the flux
    /// event fired. Each entry is `(seg_left, seg_right, node_id)`
    /// where `seg_left/seg_right` are the segment's bounds (clamped to
    /// the tract; `seg_left ≥ tract_left`, `seg_right ≤ tract_right`)
    /// and `node_id` is the tskit node ID for that segment at fire time.
    ///
    /// Used by Tier 3-full sample-conversion validation: for a query
    /// position p ∈ [tract_left, tract_right], the helper finds the
    /// entry with `seg_left ≤ p < seg_right` and queries
    /// `tree.samples(node_id)` against the marginal tree at p.
    /// Replaces the prior single `node_id_at_position`, which was
    /// captured at `x_event` only and missed the right node when the
    /// lineage's segment chain was fragmented within the tract.
    pub tract_segments: Vec<(f64, f64, i32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventRecord {
    Cmig(CmigRecord),
    Flux(FluxRecord),
}

#[derive(Debug, Default)]
pub struct EventLog {
    records: Vec<EventRecord>,
}

impl EventLog {
    pub fn new() -> Self { Self::default() }

    pub fn push_cmig(&mut self, r: CmigRecord) {
        self.records.push(EventRecord::Cmig(r));
    }

    pub fn push_flux(&mut self, r: FluxRecord) {
        self.records.push(EventRecord::Flux(r));
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    pub fn records(&self) -> &[EventRecord] { &self.records }
    pub fn into_records(self) -> Vec<EventRecord> { self.records }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve_cmig() {
        let mut log = EventLog::new();
        let r = CmigRecord {
            t: 100.0, src: 1, dst: 0, kary: Karyotype::S,
            inv_id: 0, n_eligible: 50, n_moved: 22,
        };
        log.push_cmig(r);
        assert_eq!(log.len(), 1);
        match log.records()[0] {
            EventRecord::Cmig(got) => assert_eq!(got, r),
            _ => panic!("expected Cmig variant"),
        }
    }

    #[test]
    fn push_and_retrieve_flux() {
        let mut log = EventLog::new();
        let r = FluxRecord {
            t: 250.0, lineage_uid: 42, position: 5000.0,
            tract_left: 4850.0, tract_right: 5150.0, inv_id: 0,
            tract_segments: vec![(4850.0, 5150.0, 17)],
        };
        log.push_flux(r.clone());
        assert_eq!(log.len(), 1);
        match &log.records()[0] {
            EventRecord::Flux(got) => assert_eq!(got, &r),
            _ => panic!("expected Flux variant"),
        }
    }

    #[test]
    fn into_records_preserves_order() {
        let mut log = EventLog::new();
        log.push_cmig(CmigRecord {
            t: 10.0, src: 0, dst: 1, kary: Karyotype::S,
            inv_id: 0, n_eligible: 1, n_moved: 1,
        });
        log.push_flux(FluxRecord {
            t: 20.0, lineage_uid: 1, position: 100.0,
            tract_left: 90.0, tract_right: 110.0, inv_id: 0,
            tract_segments: vec![(90.0, 110.0, 5)],
        });
        let recs = log.into_records();
        assert_eq!(recs.len(), 2);
        assert!(matches!(recs[0], EventRecord::Cmig(_)));
        assert!(matches!(recs[1], EventRecord::Flux(_)));
    }
}
