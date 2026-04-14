use pyo3::prelude::*;

/// msinv Rust core — PyO3 extension module.
#[pymodule]
fn _msinv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
