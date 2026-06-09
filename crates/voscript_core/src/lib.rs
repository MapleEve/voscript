#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyModule};

pub const CORE_SMOKE_CAPABILITY: &str = "core_smoke";
pub const RUST_EXTENSION_CAPABILITY: &str = "rust_extension";
pub const CRATE_CAPABILITY: &str = "crate";
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn core_smoke(py: Python<'_>, payload: Py<PyAny>) -> PyResult<Py<PyDict>> {
    let capabilities = PyDict::new(py);
    capabilities.set_item(CORE_SMOKE_CAPABILITY, true)?;
    capabilities.set_item(RUST_EXTENSION_CAPABILITY, true)?;
    capabilities.set_item(CRATE_CAPABILITY, PACKAGE_NAME)?;

    let response = PyDict::new(py);
    response.set_item("ok", true)?;
    response.set_item("echoed", payload.bind(py))?;
    response.set_item("version", PACKAGE_VERSION)?;
    response.set_item("capabilities", capabilities)?;
    Ok(response.unbind())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn voscript_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", PACKAGE_VERSION)?;
    module.add_function(wrap_pyfunction!(core_smoke, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn package_version_is_set() {
        assert_eq!(super::PACKAGE_VERSION, "0.8.0");
    }

    #[test]
    fn core_smoke_capability_name_is_stable() {
        assert_eq!(super::CORE_SMOKE_CAPABILITY, "core_smoke");
    }
}
