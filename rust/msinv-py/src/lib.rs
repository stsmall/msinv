use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use msinv_core::simulator::HullSimulator;

/// Run a panmictic simulation and return raw table arrays as a dict.
///
/// The Python caller converts these arrays into a tskit TreeSequence.
#[pyfunction]
#[pyo3(signature = (n_samples, population_size, sequence_length, recombination_rate=0.0, seed=42))]
fn simulate_panmictic(
    py: Python<'_>,
    n_samples: u32,
    population_size: f64,
    sequence_length: f64,
    recombination_rate: f64,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let sim = HullSimulator {
        n_samples,
        population_size,
        sequence_length,
        recombination_rate,
        seed,
    };
    let result = sim.simulate();
    let t = result.tables;

    let dict = PyDict::new(py);
    dict.set_item("sequence_length", t.sequence_length)?;
    dict.set_item("num_populations", t.num_populations)?;

    // Convert Vecs to numpy arrays (zero-copy where possible).
    dict.set_item("node_flags",
        PyArray1::from_vec(py, t.node_flags))?;
    dict.set_item("node_time",
        PyArray1::from_vec(py, t.node_time))?;
    dict.set_item("node_population",
        PyArray1::from_vec(py, t.node_population))?;
    dict.set_item("edge_left",
        PyArray1::from_vec(py, t.edge_left))?;
    dict.set_item("edge_right",
        PyArray1::from_vec(py, t.edge_right))?;
    dict.set_item("edge_parent",
        PyArray1::from_vec(py, t.edge_parent))?;
    dict.set_item("edge_child",
        PyArray1::from_vec(py, t.edge_child))?;

    Ok(dict.into())
}

/// msinv Rust core — PyO3 extension module.
#[pymodule]
fn _msinv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(simulate_panmictic, m)?)?;
    Ok(())
}
