#[cfg(feature = "python-bindings")]
use pyo3::exceptions::{PyKeyError, PyValueError};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyList, PyModule};

pub mod contracts;
pub mod postprocess;
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
fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.str()?.to_string())),
        _ => Ok(None),
    }
}

#[cfg(feature = "python-bindings")]
fn optional_bool(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => value.extract::<bool>(),
        _ => Ok(default),
    }
}

#[cfg(feature = "python-bindings")]
fn item_f64_or_default(item: Option<Bound<'_, PyAny>>, default: f64) -> PyResult<f64> {
    match item {
        Some(value) if !value.is_none() => {
            if let Ok(parsed) = value.extract::<f64>() {
                return Ok(parsed);
            }
            let text = value.str()?.to_string();
            Ok(text.parse::<f64>().unwrap_or(default))
        }
        _ => Ok(default),
    }
}

#[cfg(feature = "python-bindings")]
fn optional_f64_or_default(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    item_f64_or_default(dict.get_item(key)?, default)
}

#[cfg(feature = "python-bindings")]
fn parse_manifest_entry(
    item: Bound<'_, PyAny>,
) -> PyResult<Option<contracts::ArtifactManifestEntry>> {
    let dict = match item.cast_into::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => return Ok(None),
    };
    let name = match optional_string(&dict, "name")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let filename = match optional_string(&dict, "filename")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let role = match optional_string(&dict, "role")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let media_type = match optional_string(&dict, "media_type")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let entry = contracts::ArtifactManifestEntry {
        name: match contracts::public_safe_text(&name, "name") {
            Ok(value) => value,
            Err(_) => return Ok(None),
        },
        filename: match contracts::public_safe_manifest_filename(&filename) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        },
        role: match contracts::public_safe_text(&role, "role") {
            Ok(value) => value,
            Err(_) => return Ok(None),
        },
        media_type: match contracts::public_safe_text(&media_type, "media_type") {
            Ok(value) => value,
            Err(_) => return Ok(None),
        },
        required_for_result: optional_bool(&dict, "required_for_result", false)?,
        speaker_label: match optional_string(&dict, "speaker_label")? {
            Some(value) => contracts::public_safe_text(&value, "speaker_label").ok(),
            None => None,
        },
    };
    Ok(Some(entry))
}

#[cfg(feature = "python-bindings")]
fn append_manifest_entries(
    py: Python<'_>,
    target: &Bound<'_, PyList>,
    payload: &Bound<'_, PyDict>,
    category: &str,
) -> PyResult<()> {
    let Some(raw_entries) = payload.get_item(category)? else {
        return Ok(());
    };
    if raw_entries.is_none() {
        return Ok(());
    }
    let Ok(entries) = raw_entries.cast_into::<PyList>() else {
        return Ok(());
    };
    for item in entries.iter() {
        let Some(entry) = parse_manifest_entry(item)? else {
            continue;
        };
        let entry_dict = PyDict::new(py);
        entry_dict.set_item("name", entry.name)?;
        entry_dict.set_item("filename", entry.filename)?;
        entry_dict.set_item("role", entry.role)?;
        entry_dict.set_item("media_type", entry.media_type)?;
        entry_dict.set_item("required_for_result", entry.required_for_result)?;
        if let Some(speaker_label) = entry.speaker_label {
            entry_dict.set_item("speaker_label", speaker_label)?;
        }
        target.append(entry_dict)?;
    }
    Ok(())
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
fn parse_postprocess_word(item: Bound<'_, PyAny>) -> PyResult<postprocess::Word> {
    let dict = match item.cast_into::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => {
            return Ok(postprocess::Word {
                word: String::new(),
                start: 0.0,
                end: 0.0,
                score: 0.0,
            });
        }
    };
    let word = match dict.get_item("word")? {
        Some(value) if !value.is_none() => value.str()?.to_string(),
        _ => String::new(),
    };
    Ok(postprocess::Word {
        word,
        start: optional_f64_or_default(&dict, "start", 0.0)?,
        end: optional_f64_or_default(&dict, "end", 0.0)?,
        score: optional_f64_or_default(&dict, "score", 0.0)?,
    })
}

#[cfg(feature = "python-bindings")]
fn parse_postprocess_words(dict: &Bound<'_, PyDict>) -> PyResult<Vec<postprocess::Word>> {
    let words_any = match dict.get_item("words")? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(Vec::new()),
    };
    let words_list = words_any.cast_into::<PyList>()?;
    let mut words = Vec::with_capacity(words_list.len());
    for item in words_list.iter() {
        words.push(parse_postprocess_word(item)?);
    }
    Ok(words)
}

#[cfg(feature = "python-bindings")]
fn parse_aligned_segment(item: Bound<'_, PyAny>) -> PyResult<postprocess::AlignedSegment> {
    let dict = item.cast_into::<PyDict>()?;
    let text = match dict.get_item("text")? {
        Some(value) if !value.is_none() => value.str()?.to_string(),
        _ => String::new(),
    };
    let speaker = match dict.get_item("speaker")? {
        Some(value) if !value.is_none() => value.str()?.to_string(),
        _ => "UNKNOWN".to_string(),
    };
    Ok(postprocess::AlignedSegment {
        start: optional_f64_or_default(&dict, "start", 0.0)?,
        end: optional_f64_or_default(&dict, "end", 0.0)?,
        text,
        speaker,
        words: parse_postprocess_words(&dict)?,
    })
}

#[cfg(feature = "python-bindings")]
fn parse_speaker_match(item: Bound<'_, PyAny>) -> PyResult<postprocess::SpeakerMatch> {
    let dict = match item.cast_into::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => return Ok(postprocess::SpeakerMatch::default()),
    };
    Ok(postprocess::SpeakerMatch {
        matched_id: optional_string(&dict, "matched_id")?,
        matched_name: optional_string(&dict, "matched_name")?,
        similarity: Some(optional_f64_or_default(&dict, "similarity", 0.0)?),
    })
}

#[cfg(feature = "python-bindings")]
fn parse_postprocess_request(
    payload: &Bound<'_, PyDict>,
) -> PyResult<(
    Vec<postprocess::AlignedSegment>,
    std::collections::HashMap<String, postprocess::SpeakerMatch>,
)> {
    let segments_any = required_item(payload, "aligned_segments")?;
    let segments_list = segments_any.cast_into::<PyList>()?;
    let mut aligned_segments = Vec::with_capacity(segments_list.len());
    for item in segments_list.iter() {
        aligned_segments.push(parse_aligned_segment(item)?);
    }

    let speaker_map_any = required_item(payload, "speaker_map")?;
    let speaker_map_dict = speaker_map_any.cast_into::<PyDict>()?;
    let mut speaker_map = std::collections::HashMap::new();
    for (key, value) in speaker_map_dict.iter() {
        speaker_map.insert(key.str()?.to_string(), parse_speaker_match(value)?);
    }

    Ok((aligned_segments, speaker_map))
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn artifact_manifest_contract(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let response = PyDict::new(py);
    let manifest_version = match optional_string(payload, "manifest_version")? {
        Some(value) if !value.trim().is_empty() => {
            contracts::public_safe_text(&value, "manifest_version")
                .map_err(PyValueError::new_err)?
        }
        _ => contracts::ARTIFACT_MANIFEST_VERSION.to_string(),
    };
    response.set_item("manifest_version", manifest_version)?;

    for category in contracts::ARTIFACT_MANIFEST_CATEGORIES {
        let entries = PyList::empty(py);
        append_manifest_entries(py, &entries, payload, category)?;
        response.set_item(category, entries)?;
    }
    Ok(response.unbind())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn status_payload_contract(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let status = optional_string(payload, "status")?.unwrap_or_default();
    let updated_at = optional_string(payload, "updated_at")?
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string());

    let response = PyDict::new(py);
    response.set_item("status", contracts::normalize_job_status(&status))?;
    response.set_item("updated_at", updated_at)?;
    match optional_string(payload, "error")? {
        Some(error) => response.set_item("error", error)?,
        None => response.set_item("error", py.None())?,
    }
    if let Some(filename) = optional_string(payload, "filename")?
        .and_then(|value| contracts::public_safe_status_filename(&value))
    {
        response.set_item("filename", filename)?;
    }
    Ok(response.unbind())
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
#[pyfunction]
fn postprocess_segments(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let (aligned_segments, speaker_map) = parse_postprocess_request(payload)?;
    let result = postprocess::build_result_segments(aligned_segments, speaker_map);

    let response = PyDict::new(py);
    let segments = PyList::empty(py);
    for segment in result.segments {
        let item = PyDict::new(py);
        item.set_item("id", segment.id)?;
        item.set_item("start", segment.start)?;
        item.set_item("end", segment.end)?;
        item.set_item("text", segment.text)?;
        item.set_item("speaker_label", segment.speaker_label)?;
        item.set_item("speaker_id", segment.speaker_id)?;
        item.set_item("speaker_name", segment.speaker_name)?;
        item.set_item("similarity", segment.similarity)?;
        if !segment.words.is_empty() {
            let words = PyList::empty(py);
            for word in segment.words {
                let word_item = PyDict::new(py);
                word_item.set_item("word", word.word)?;
                word_item.set_item("start", word.start)?;
                word_item.set_item("end", word.end)?;
                word_item.set_item("score", word.score)?;
                words.append(word_item)?;
            }
            item.set_item("words", words)?;
        }
        segments.append(item)?;
    }
    response.set_item("segments", segments)?;
    response.set_item("unique_speakers", result.unique_speakers)?;
    Ok(response.unbind())
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn voscript_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", PACKAGE_VERSION)?;
    module.add_function(wrap_pyfunction!(artifact_manifest_contract, module)?)?;
    module.add_function(wrap_pyfunction!(core_smoke, module)?)?;
    module.add_function(wrap_pyfunction!(status_payload_contract, module)?)?;
    module.add_function(wrap_pyfunction!(voiceprint_score, module)?)?;
    module.add_function(wrap_pyfunction!(postprocess_segments, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn package_version_is_set() {
        assert_eq!(super::PACKAGE_VERSION, "0.8.4");
    }

    #[test]
    fn core_smoke_capability_name_is_stable() {
        assert_eq!(super::CORE_SMOKE_CAPABILITY, "core_smoke");
    }
}
