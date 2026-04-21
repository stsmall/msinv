use std::cell::RefCell;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use msinv_core::class_tag::Karyotype;
use msinv_core::demography::{DemoEvent, Demography};
use msinv_core::inversion::InversionSpec;
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
    seed = 42
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
) -> PyResult<Py<PyDict>> {
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
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown demographic event type: {etype}")));
                }
            }
        }
    }

    // --- Inversions ---
    let mut inv_specs: Vec<InversionSpec> = Vec::new();
    if let Some(inv_list) = inversions {
        for (i, item) in inv_list.iter().enumerate() {
            let d: &Bound<'_, PyDict> = item.downcast()?;
            let bp_left: f64 = d.get_item("bp_left")?.unwrap().extract()?;
            let bp_right: f64 = d.get_item("bp_right")?.unwrap().extract()?;
            // p_inv: accept float (scalar → all pops same) or list (per-pop).
            let p_inv_obj = d.get_item("p_inv")?.unwrap();
            let p_inv: Vec<f64> = if let Ok(v) = p_inv_obj.extract::<f64>() {
                vec![v]
            } else if let Ok(v) = p_inv_obj.extract::<Vec<f64>>() {
                v
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "p_inv must be a float or list of floats"));
            };
            let t_inv: f64 = d.get_item("t_inv")?.unwrap().extract()?;
            let gcr: f64 = d.get_item("gene_conversion_rate")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.0);
            let fw: f64 = d.get_item("flux_window")?
                .and_then(|v| v.extract().ok()).unwrap_or(0.05);
            inv_specs.push(InversionSpec {
                bp_left, bp_right, p_inv, t_inv,
                gene_conversion_rate: gcr, flux_window: fw,
                inv_id: i as u16,
            });
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
            let target: Option<(u16, Karyotype)> = {
                let tc: Option<String> = d.get_item("target_class")?
                    .and_then(|v| v.extract().ok());
                match tc.as_deref() {
                    None | Some("any") => None,
                    Some(s) if s.len() == 1 => {
                        parse_kary(s.chars().next().unwrap()).map(|k| (0, k))
                    }
                    Some(s) if s.len() >= 2 => {
                        let kary = parse_kary(s.chars().next().unwrap());
                        let inv_id: u16 = s[1..].parse().unwrap_or(0);
                        kary.map(|k| (inv_id, k))
                    }
                    _ => None,
                }
            };
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
    tables_to_pydict(py, result)
}

/// msinv Rust core — PyO3 extension module.
#[pymodule]
fn _msinv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simulate_raw, m)?)?;
    Ok(())
}
