use std::cell::RefCell;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{DemoEvent, Demography};
use msinv_core::inversion::{InversionSpec, TractDistribution};
use msinv_core::trajectory::{
    BridgeStochasticTrajectory, ConstantTrajectory, CoupledTrajectory,
    DeterministicTrajectory, IntegerWFTrajectory, PrecomputedTrajectory,
    StochasticDeterministicTrajectory, StochasticTrajectory, Trajectory,
};
use msinv_core::rate_index::RateCache;
use msinv_core::simulator::{HullSimulator, SampleEntry, SimResult};
use msinv_core::sweep::Sweep;

thread_local! {
    /// Reusable pair-rate cache. Amortises the first-call allocation
    /// of the triangular overlap / pair-bucket arrays (~500ms for a
    /// 4096-lineage capacity) across every subsequent `simulate_raw`
    /// call on the same Python thread. Cache state is reset at the
    /// top of `HullSimulator::simulate_with_cache` so each call
    /// starts clean; only the heap allocations persist.
    ///
    /// Thread-local is safe here because PyO3 holds the GIL during
    /// `simulate_raw`, so there is exactly one active caller per
    /// Python thread. Different threads get separate caches.
    static CACHE: RefCell<Option<RateCache>> = const { RefCell::new(None) };
}

/// Convert a SimResult's TableBuilder into a Python dict of numpy arrays.
fn tables_to_pydict(py: Python<'_>, result: SimResult) -> PyResult<Py<PyDict>> {
    let t = result.tables;
    let dict = PyDict::new(py);
    dict.set_item("sequence_length", t.sequence_length)?;
    dict.set_item("num_populations", t.num_populations)?;
    dict.set_item("node_flags", PyArray1::from_vec(py, t.node_flags))?;
    dict.set_item("node_time", PyArray1::from_vec(py, t.node_time))?;
    dict.set_item("node_population", PyArray1::from_vec(py, t.node_population))?;
    dict.set_item("edge_left", PyArray1::from_vec(py, t.edge_left))?;
    dict.set_item("edge_right", PyArray1::from_vec(py, t.edge_right))?;
    dict.set_item("edge_parent", PyArray1::from_vec(py, t.edge_parent))?;
    dict.set_item("edge_child", PyArray1::from_vec(py, t.edge_child))?;
    Ok(dict.into())
}

/// Parse a karyotype character ('S' or 'I') into Karyotype enum.
fn parse_kary(c: char) -> Option<Karyotype> {
    match c {
        'S' => Some(Karyotype::S),
        'I' => Some(Karyotype::I),
        _ => None,
    }
}

/// Parse a sweep `target_class` string — None or "any" → no filter,
/// "S"/"I" → inv 0, "S<id>"/"I<id>" → that inv. "P" (panmictic-only)
/// is rejected: Rust Sweep can't express it.
fn parse_sweep_target(tc: Option<&str>) -> PyResult<Option<(u16, Karyotype)>> {
    let s = match tc {
        None | Some("any") => return Ok(None),
        Some("P") => return Err(pyo3::exceptions::PyValueError::new_err(
            "target_class='P' (panmictic-only sweep) is not supported by the \
             Rust backend. Use 'any', or an 'S'/'I'/'S<id>'/'I<id>' karyotype.")),
        Some(s) => s,
    };
    let mut chars = s.chars();
    let kary = parse_kary(chars.next().unwrap())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("unrecognised target_class {s:?}; first char must be 'S' or 'I'")))?;
    let inv_id: u16 = if s.len() == 1 {
        0
    } else {
        s[1..].parse().map_err(|_| pyo3::exceptions::PyValueError::new_err(
            format!("invalid inv_id in target_class {s:?}")))?
    };
    Ok(Some((inv_id, kary)))
}

// ---------------------------------------------------------------
// simulate_raw: the main entry point from Python
// ---------------------------------------------------------------

/// Run a hull simulation and return raw table arrays as a dict.
///
/// This is the Rust-accelerated equivalent of
/// ``HullSimulator(...).simulate()`` in the Python version. The
/// Python wrapper in ``msinv/hull/_rust_bridge.py`` calls this
/// function and converts the dict into a tskit TreeSequence.
#[pyfunction]
#[pyo3(signature = (
    sample_config,
    pop_sizes,
    sequence_length,
    recombination_rate = 0.0,
    inversions = None,
    sweeps = None,
    demo_events = None,
    migration_matrix = None,
    seed = 42,
    stop_at = f64::INFINITY,
    compound_rate = false,
    iters_max = 10_000_000u64,
    gc_stride = 160u32,
    record_events = false
))]
#[allow(clippy::too_many_arguments)]
fn simulate_raw(
    py: Python<'_>,
    sample_config: &Bound<'_, PyList>,   // list of (karyotype_str, pop, count)
    pop_sizes: Vec<f64>,
    sequence_length: f64,
    recombination_rate: f64,
    inversions: Option<&Bound<'_, PyList>>,
    sweeps: Option<&Bound<'_, PyList>>,
    demo_events: Option<&Bound<'_, PyList>>,
    migration_matrix: Option<Vec<Vec<f64>>>,
    seed: u64,
    stop_at: f64,
    compound_rate: bool,
    iters_max: u64,
    gc_stride: u32,
    record_events: bool,
) -> PyResult<(Py<PyDict>, PyObject)> {
    // --- Demography ---
    let mut demo = Demography::new(pop_sizes);
    if let Some(mig) = migration_matrix {
        demo.migration_matrix = mig;
    }
    if let Some(events) = demo_events {
        for item in events.iter() {
            let tup: &Bound<'_, PyTuple> = item.downcast()?;
            let etype: String = tup.get_item(0)?.extract()?;
            match etype.as_str() {
                "ej" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let src: u32 = tup.get_item(2)?.extract()?;
                    let dst: u32 = tup.get_item(3)?.extract()?;
                    demo.add_event(DemoEvent::Ej { t, src, dst });
                }
                "en" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let pop: u32 = tup.get_item(2)?.extract()?;
                    let n: f64 = tup.get_item(3)?.extract()?;
                    demo.add_event(DemoEvent::En { t, pop, n });
                }
                "eN" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let n: f64 = tup.get_item(2)?.extract()?;
                    demo.add_event(DemoEvent::EN { t, n });
                }
                "eg" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let pop: u32 = tup.get_item(2)?.extract()?;
                    let alpha: f64 = tup.get_item(3)?.extract()?;
                    demo.add_event(DemoEvent::Eg { t, pop, alpha });
                }
                "eG" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let alpha: f64 = tup.get_item(2)?.extract()?;
                    demo.add_event(DemoEvent::EG { t, alpha });
                }
                "em" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let dst: u32 = tup.get_item(2)?.extract()?;
                    let src: u32 = tup.get_item(3)?.extract()?;
                    let m: f64 = tup.get_item(4)?.extract()?;
                    demo.add_event(DemoEvent::Em { t, dst, src, m });
                }
                "eM" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let m: f64 = tup.get_item(2)?.extract()?;
                    demo.add_event(DemoEvent::EM { t, m });
                }
                "eig" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let pop: u32 = tup.get_item(2)?.extract()?;
                    let inv_id: u16 = tup.get_item(3)?.extract()?;
                    let p_inv: f64 = tup.get_item(4)?.extract()?;
                    demo.add_event(DemoEvent::Eig { t, pop, inv_id, p_inv });
                }
                // Class-conditional migration / admixture / class-ej.
                // ('cmig', t, src, dst, kary_str, inv_id, proportion)
                // proportion = 1.0 → unconditional class merge (= class ej)
                // proportion < 1.0 → stochastic Bernoulli admixture
                "cmig" | "ejk" => {
                    let t: f64 = tup.get_item(1)?.extract()?;
                    let src: u32 = tup.get_item(2)?.extract()?;
                    let dst: u32 = tup.get_item(3)?.extract()?;
                    let kary_str: String = tup.get_item(4)?.extract()?;
                    let inv_id: u16 = tup.get_item(5)?.extract()?;
                    let proportion: f64 = if tup.len() > 6 {
                        tup.get_item(6)?.extract()?
                    } else { 1.0 };
                    let kary = parse_kary(kary_str.chars().next().unwrap())
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                            format!("ClassMig kary must be 'S' or 'I', got {:?}", kary_str)))?;
                    demo.add_event(DemoEvent::ClassMig {
                        t, src, dst, kary, inv_id, proportion,
                    });
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown demographic event type: {etype}")));
                }
            }
        }
    }

    // --- Inversions ---
    // Each inversion dict can carry EITHER:
    //   (a) Legacy: 'p_inv' (float or list[float]) + 't_inv' (float).
    //       Wraps in a ConstantTrajectory (back-compat, identical to
    //       pre-trajectory-port behaviour).
    //   (b) New:    'trajectory' = {'type': 'constant'|'deterministic'|
    //               'stochastic'|'coupled', ...args}.  Builds the
    //       corresponding Rust Trajectory.
    let mut inv_specs: Vec<InversionSpec> = Vec::new();
    if let Some(inv_list) = inversions {
        for (i, item) in inv_list.iter().enumerate() {
            let d: &Bound<'_, PyDict> = item.downcast()?;
            let bp_left: f64 = d.get_item("bp_left")?.unwrap().extract()?;
            let bp_right: f64 = d.get_item("bp_right")?.unwrap().extract()?;
            let gcr: f64 = d.get_item("gene_conversion_rate")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.0);
            let mtl: f64 = d.get_item("mean_tract_length")?
                .map_or_else(|| Ok(100.0_f64), |v| v.extract())?;
            let td_str: String = d.get_item("tract_distribution")?
                .map_or_else(|| Ok("geometric".to_string()), |v| v.extract())?;
            let td = match td_str.as_str() {
                "fixed" => TractDistribution::Fixed,
                "geometric" => TractDistribution::Geometric,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("tract_distribution must be 'fixed' or 'geometric', got {td_str:?}"))),
            };

            // Build trajectory
            let trajectory: Box<dyn Trajectory + Send + Sync> =
                if let Some(traj_obj) = d.get_item("trajectory")? {
                    let td: &Bound<'_, PyDict> = traj_obj.downcast()?;
                    let ttype: String = td.get_item("type")?
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                            "trajectory dict requires 'type' key"))?
                        .extract()?;
                    match ttype.as_str() {
                        "constant" => {
                            let p = td.get_item("p_inv")?.unwrap();
                            let p_vec: Vec<f64> = if let Ok(v) = p.extract::<f64>() {
                                vec![v]
                            } else { p.extract()? };
                            let t_inv: f64 = td.get_item("t_inv")?.unwrap().extract()?;
                            Box::new(ConstantTrajectory::new(p_vec, t_inv))
                        }
                        "deterministic" => {
                            let p_final: f64 = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: f64    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: f64      = td.get_item("s")?.unwrap().extract()?;
                            // Optional p_start (founding frequency).
                            // Defaults to 1/(2N) (hard sweep) when omitted.
                            // Set to e.g. 0.05 for partial-SHIC-style soft sweep
                            // on standing variation.
                            let p_start: Option<f64> = td.get_item("p_start")?
                                .and_then(|v| v.extract().ok());
                            match p_start {
                                Some(p0) => Box::new(
                                    DeterministicTrajectory::new_with_p_start(
                                        p_final, p0, n_e, s)),
                                None => Box::new(
                                    DeterministicTrajectory::new(p_final, n_e, s)),
                            }
                        }
                        "stochastic" => {
                            let p_final: f64 = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: f64    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: f64      = td.get_item("s")?.unwrap().extract()?;
                            let seed: u64   = td.get_item("seed")?
                                .and_then(|v| v.extract().ok()).unwrap_or(42);
                            Box::new(StochasticTrajectory::new(p_final, n_e, s, seed))
                        }
                        // Discoal-style stochastic-then-deterministic.
                        // Phase 1 (small p, drift-dominated): integer-WF
                        //   from p_start to det_threshold (default
                        //   5/(2N), configurable).
                        // Phase 2 (large p, selection-dominated): closed-
                        //   form logistic from det_threshold to p_final.
                        "stoch_det" | "stochastic_deterministic" => {
                            let p_final: f64 = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: f64    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: f64      = td.get_item("s")?.unwrap().extract()?;
                            let p_start: f64 = td.get_item("p_start")?
                                .and_then(|v| v.extract().ok())
                                .unwrap_or(1.0 / (2.0 * n_e));
                            let det_threshold: Option<f64> = td.get_item("det_threshold")?
                                .and_then(|v| v.extract().ok());
                            let seed: u64 = td.get_item("seed")?
                                .and_then(|v| v.extract().ok()).unwrap_or(42);
                            let max_attempts: u32 = td.get_item("max_attempts")?
                                .and_then(|v| v.extract().ok()).unwrap_or(100);
                            Box::new(StochasticDeterministicTrajectory::new(
                                p_final, n_e, s, p_start, det_threshold,
                                seed, max_attempts
                            ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?)
                        }
                        // Integer-copy WF trajectory: forward-simulate
                        // discrete WF with selection from p_start to
                        // p_final, rejection-resampling lost paths.
                        // The robust large-N replacement for 'stochastic'
                        // (which uses continuous-diffusion approx).
                        "integer_wf" => {
                            let p_final: f64 = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: f64    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: f64      = td.get_item("s")?.unwrap().extract()?;
                            let p_start: f64 = td.get_item("p_start")?
                                .and_then(|v| v.extract().ok())
                                .unwrap_or(1.0 / (2.0 * n_e));
                            let seed: u64   = td.get_item("seed")?
                                .and_then(|v| v.extract().ok()).unwrap_or(42);
                            let max_attempts: u32 = td.get_item("max_attempts")?
                                .and_then(|v| v.extract().ok()).unwrap_or(100);
                            Box::new(IntegerWFTrajectory::new(
                                p_final, n_e, s, p_start, seed, max_attempts
                            ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?)
                        }
                        // Bridge stochastic: conditioned on BOTH t_inv
                        // and p_final.  partialdiscoal-style incomplete
                        // sweep.  s>0 is recommended for tractable
                        // acceptance rates.
                        "bridge" => {
                            let p_final: f64 = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: f64    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: f64      = td.get_item("s")?.unwrap().extract()?;
                            let t_inv: f64  = td.get_item("t_inv")?.unwrap().extract()?;
                            let seed: u64   = td.get_item("seed")?
                                .and_then(|v| v.extract().ok()).unwrap_or(42);
                            let tolerance: f64 = td.get_item("tolerance")?
                                .and_then(|v| v.extract().ok()).unwrap_or(0.02);
                            let max_attempts: u64 = td.get_item("max_attempts")?
                                .and_then(|v| v.extract().ok()).unwrap_or(10_000);
                            Box::new(BridgeStochasticTrajectory::new(
                                p_final, n_e, s, t_inv, seed, tolerance, max_attempts
                            ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?)
                        }
                        // Precomputed: user supplies (times, freqs, n_e).
                        // freqs is a list of per-pop arrays.
                        // Optional 't_inv': per-pop barrier dissolution
                        // times.  If omitted, inferred from when freq
                        // reaches 1/(2N).
                        "precomputed" => {
                            let times: Vec<f64> = td.get_item("times")?.unwrap().extract()?;
                            let freqs: Vec<Vec<f64>> = td.get_item("freqs")?.unwrap().extract()?;
                            let n_e: Vec<f64> = td.get_item("n_e")?.unwrap().extract()?;
                            let t_inv_explicit: Option<Vec<f64>> =
                                td.get_item("t_inv")?.and_then(|v| v.extract().ok());
                            Box::new(PrecomputedTrajectory::with_t_inv(
                                times, freqs, n_e, t_inv_explicit))
                        }
                        "coupled" => {
                            let p_final: Vec<f64> = td.get_item("p_final")?.unwrap().extract()?;
                            let n_e: Vec<f64>    = td.get_item("n_e")?.unwrap().extract()?;
                            let s: Vec<f64>      = td.get_item("s")?.unwrap().extract()?;
                            let m: f64           = td.get_item("m")?.unwrap().extract()?;
                            let seed: u64        = td.get_item("seed")?
                                .and_then(|v| v.extract().ok()).unwrap_or(42);
                            Box::new(CoupledTrajectory::new(p_final, n_e, s, m, seed))
                        }
                        other => return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("unknown trajectory type: {:?}", other))),
                    }
                } else {
                    // Legacy back-compat: p_inv + t_inv
                    let p_inv_obj = d.get_item("p_inv")?.ok_or_else(||
                        pyo3::exceptions::PyValueError::new_err(
                            "inversion dict requires 'p_inv' or 'trajectory'"))?;
                    let p_inv: Vec<f64> = if let Ok(v) = p_inv_obj.extract::<f64>() {
                        vec![v]
                    } else if let Ok(v) = p_inv_obj.extract::<Vec<f64>>() {
                        v
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "p_inv must be a float or list of floats"));
                    };
                    let t_inv: f64 = d.get_item("t_inv")?.unwrap().extract()?;
                    Box::new(ConstantTrajectory::new(p_inv, t_inv))
                };

            let mut spec = InversionSpec::new(bp_left, bp_right, trajectory);
            spec.gene_conversion_rate = gcr;
            spec.mean_tract_length = mtl;
            spec.tract_distribution = td;
            spec.inv_id = i as u16;
            inv_specs.push(spec);
        }
    }

    // --- Sweeps ---
    let mut sweep_specs: Vec<Sweep> = Vec::new();
    if let Some(sw_list) = sweeps {
        for item in sw_list.iter() {
            let d: &Bound<'_, PyDict> = item.downcast()?;
            let x_sel: f64 = d.get_item("x_sel")?.unwrap().extract()?;
            let t_event: f64 = d.get_item("t_event")?.unwrap().extract()?;
            let sw_win: f64 = d.get_item("sweep_window")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.0);
            let pop: Option<u32> = d.get_item("population")?
                .and_then(|v| v.extract().ok());
            let target: Option<(u16, Karyotype)> =
                parse_sweep_target(d.get_item("target_class")?
                    .and_then(|v| v.extract::<String>().ok()).as_deref())?;
            let sel_coeff: f64 = d.get_item("selection_coefficient")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.0);
            let start_freq: f64 = d.get_item("starting_frequency")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.0);
            sweep_specs.push(Sweep {
                x_sel, t_event, target, population: pop,
                sweep_window: sw_win,
                selection_coefficient: sel_coeff,
                starting_frequency: start_freq,
            });
        }
    }

    // --- Sample config ---
    let n_inv = inv_specs.len();
    let mut samples: Vec<SampleEntry> = Vec::new();
    for item in sample_config.iter() {
        let tup: &Bound<'_, PyTuple> = item.downcast()?;
        let kary_str: String = tup.get_item(0)?.extract()?;
        let pop: u32 = tup.get_item(1)?.extract()?;
        let count: u32 = tup.get_item(2)?.extract()?;

        let karyotypes: Vec<Option<Karyotype>> = if kary_str == "P" || n_inv == 0 {
            vec![None; n_inv]
        } else if kary_str.len() == 1 {
            // Linked karyotype: same for all inversions.
            let k = parse_kary(kary_str.chars().next().unwrap());
            vec![k; n_inv]
        } else {
            // Per-inversion karyotype string (e.g. "SI").
            kary_str.chars().map(|c| parse_kary(c)).collect()
        };

        samples.push(SampleEntry { karyotypes, population: pop, count });
    }

    // --- Build and run ---
    let sim = HullSimulator {
        samples,
        demography: demo,
        sequence_length,
        recombination_rate,
        inversions: inv_specs,
        sweeps: sweep_specs,
        seed,
        stop_at,
        compound_rate,
        iters_max,
        gc_stride,
        record_events,
    };
    let mut result = CACHE.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() {
            *slot = Some(RateCache::new(0, sim.sequence_length));
        }
        sim.simulate_with_cache(slot.as_mut().unwrap())
    });
    // Sort edges into tskit canonical order so the Python bridge can
    // skip `tc.sort()` (previously ~3.6% of single-pop wall at rho=2000).
    result.tables.sort_edges();
    let event_log = result.event_log.take();
    let py_tables = tables_to_pydict(py, result)?;
    let py_log = event_log_to_pylist(py, event_log)?;
    Ok((py_tables, py_log))
}

/// Convert an optional `EventLog` into a Python list of dicts, or `None`.
///
/// Each record is a dict with a `kind` key (`"cmig"` or `"flux"`) plus the
/// variant-specific fields. Returns `py.None()` when the log is absent (i.e.
/// `record_events=False`).
fn event_log_to_pylist(
    py: Python<'_>,
    log: Option<msinv_core::event_log::EventLog>,
) -> PyResult<PyObject> {
    use msinv_core::event_log::EventRecord;

    let log = match log {
        None => return Ok(py.None()),
        Some(l) => l,
    };
    let py_list = PyList::empty(py);
    for rec in log.into_records() {
        let dict = PyDict::new(py);
        match rec {
            EventRecord::Cmig(c) => {
                dict.set_item("kind", "cmig")?;
                dict.set_item("t", c.t)?;
                dict.set_item("src", c.src)?;
                dict.set_item("dst", c.dst)?;
                dict.set_item("kary", format!("{:?}", c.kary))?;
                dict.set_item("inv_id", c.inv_id)?;
                dict.set_item("n_eligible", c.n_eligible)?;
                dict.set_item("n_moved", c.n_moved)?;
            }
            EventRecord::Flux(f) => {
                dict.set_item("kind", "flux")?;
                dict.set_item("t", f.t)?;
                dict.set_item("lineage_uid", f.lineage_uid)?;
                dict.set_item("position", f.position)?;
                dict.set_item("tract_left", f.tract_left)?;
                dict.set_item("tract_right", f.tract_right)?;
                dict.set_item("inv_id", f.inv_id)?;
            }
        }
        py_list.append(dict)?;
    }
    Ok(py_list.into())
}

/// msinv Rust core — PyO3 extension module.
#[pymodule]
fn _msinv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simulate_raw, m)?)?;
    Ok(())
}
