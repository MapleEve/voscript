#[cfg(feature = "python-bindings")]
use pyo3::exceptions::{PyKeyError, PyValueError};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyList, PyModule};

pub mod voiceprint;

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
fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing required key: {key}")))
}

#[cfg(feature = "python-bindings")]
fn optional_f64(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<f64>(),
        _ => Ok(default),
    }
}

#[cfg(feature = "python-bindings")]
fn optional_usize(dict: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<usize>(),
        _ => Ok(default),
    }
}

#[cfg(feature = "python-bindings")]
fn parse_voiceprint_candidate(
    item: Bound<'_, PyAny>,
) -> PyResult<voiceprint::VoiceprintScoreCandidate> {
    let dict = item.cast_into::<PyDict>()?;
    let speaker_id = required_item(&dict, "speaker_id")?.extract::<String>()?;
    let name = required_item(&dict, "name")?.extract::<String>()?;
    let embedding = required_item(&dict, "embedding")?.extract::<Vec<f64>>()?;
    let sample_count = required_item(&dict, "sample_count")?.extract::<usize>()?;
    let sample_spread = match dict.get_item("sample_spread")? {
        Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
        _ => None,
    };
    Ok(voiceprint::VoiceprintScoreCandidate {
        speaker_id,
        name,
        embedding,
        sample_count,
        sample_spread,
    })
}

#[cfg(feature = "python-bindings")]
fn parse_voiceprint_request(
    payload: &Bound<'_, PyDict>,
) -> PyResult<voiceprint::VoiceprintScoreRequest> {
    let query_embedding = required_item(payload, "query_embedding")?.extract::<Vec<f64>>()?;
    let candidates_any = required_item(payload, "candidates")?;
    let candidates_list = candidates_any.cast_into::<PyList>()?;
    let mut candidates = Vec::with_capacity(candidates_list.len());
    for item in candidates_list.iter() {
        candidates.push(parse_voiceprint_candidate(item)?);
    }

    let cohort = match payload.get_item("cohort")? {
        Some(value) if !value.is_none() => Some(value.extract::<Vec<Vec<f64>>>()?),
        _ => None,
    };

    Ok(voiceprint::VoiceprintScoreRequest {
        query_embedding,
        candidates,
        threshold: optional_f64(payload, "threshold", 0.75)?,
        asnorm_threshold: optional_f64(payload, "asnorm_threshold", 0.5)?,
        cohort,
        asnorm_top_n: optional_usize(payload, "asnorm_top_n", 200)?,
        asnorm_min_margin: optional_f64(payload, "asnorm_min_margin", 0.05)?,
    })
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn voiceprint_score(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let request = parse_voiceprint_request(payload)?;
    let decision =
        voiceprint::score_voiceprint_candidates(request).map_err(PyValueError::new_err)?;

    let response = PyDict::new(py);
    response.set_item("matched_id", decision.matched_id)?;
    response.set_item("matched_name", decision.matched_name)?;
    response.set_item("similarity", decision.similarity)?;
    response.set_item("reason", decision.reason)?;
    response.set_item("asnorm_active", decision.asnorm_active)?;
    response.set_item("asnorm_reason", decision.asnorm_reason)?;

    let candidates = PyList::empty(py);
    for candidate in decision.candidates {
        let item = PyDict::new(py);
        item.set_item("speaker_id", candidate.speaker_id)?;
        item.set_item("name", candidate.name)?;
        item.set_item("raw_similarity", candidate.raw_similarity)?;
        item.set_item("similarity", candidate.similarity)?;
        item.set_item("effective_threshold", candidate.effective_threshold)?;
        item.set_item("score_method", candidate.score_method)?;
        item.set_item("sample_count", candidate.sample_count)?;
        item.set_item("sample_spread", candidate.sample_spread)?;
        candidates.append(item)?;
    }
    response.set_item("candidates", candidates)?;
    Ok(response.unbind())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn voscript_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", PACKAGE_VERSION)?;
    module.add_function(wrap_pyfunction!(core_smoke, module)?)?;
    module.add_function(wrap_pyfunction!(voiceprint_score, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn package_version_is_set() {
        assert_eq!(super::PACKAGE_VERSION, "0.8.1");
    }

    #[test]
    fn core_smoke_capability_name_is_stable() {
        assert_eq!(super::CORE_SMOKE_CAPABILITY, "core_smoke");
    }
}
