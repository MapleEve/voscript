pub const ARTIFACT_MANIFEST_VERSION: &str = "artifact_manifest.v1";
pub const ARTIFACT_MANIFEST_CATEGORIES: [&str; 3] = ["stable", "optional", "experimental"];

pub const JOB_STATUS_QUEUED: &str = "queued";
pub const JOB_STATUS_CONVERTING: &str = "converting";
pub const JOB_STATUS_DENOISING: &str = "denoising";
pub const JOB_STATUS_TRANSCRIBING: &str = "transcribing";
pub const JOB_STATUS_IDENTIFYING: &str = "identifying";
pub const JOB_STATUS_COMPLETED: &str = "completed";
pub const JOB_STATUS_FAILED: &str = "failed";

#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactManifestEntry {
    pub name: String,
    pub filename: String,
    pub role: String,
    pub media_type: String,
    pub required_for_result: bool,
    pub speaker_label: Option<String>,
}

fn strip_control_chars(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect::<String>()
        .trim()
        .to_string()
}

pub fn public_safe_text(value: &str, field_name: &str) -> Result<String, String> {
    let text = strip_control_chars(value);
    if text.is_empty() {
        return Err(format!("artifact manifest {field_name} must not be empty"));
    }
    Ok(text)
}

pub fn public_safe_manifest_filename(value: &str) -> Result<String, String> {
    let filename = public_safe_text(value, "filename")?;
    if filename.contains('/')
        || filename.contains('\\')
        || filename.contains("://")
        || filename == "."
        || filename == ".."
    {
        return Err("artifact manifest filename must not include a path or URL".to_string());
    }
    Ok(filename)
}

pub fn public_safe_status_filename(value: &str) -> Option<String> {
    let normalized = strip_control_chars(&value.replace('\\', "/"));
    let filename = normalized.rsplit('/').next().unwrap_or("").trim();
    if filename.is_empty() {
        None
    } else {
        Some(filename.to_string())
    }
}

pub fn is_known_job_status(status: &str) -> bool {
    matches!(
        status,
        JOB_STATUS_QUEUED
            | JOB_STATUS_CONVERTING
            | JOB_STATUS_DENOISING
            | JOB_STATUS_TRANSCRIBING
            | JOB_STATUS_IDENTIFYING
            | JOB_STATUS_COMPLETED
            | JOB_STATUS_FAILED
    )
}

pub fn normalize_job_status(status: &str) -> String {
    let value = strip_control_chars(status).to_ascii_lowercase();
    if is_known_job_status(value.as_str()) {
        value
    } else {
        JOB_STATUS_FAILED.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_filename_rejects_paths_and_urls() {
        assert!(public_safe_manifest_filename("result.json").is_ok());
        assert!(public_safe_manifest_filename("../result.json").is_err());
        assert!(public_safe_manifest_filename("http://host/result.json").is_err());
    }

    #[test]
    fn status_filename_keeps_only_basename() {
        assert_eq!(
            public_safe_status_filename("private/audio.wav").as_deref(),
            Some("audio.wav")
        );
        assert_eq!(
            public_safe_status_filename("C:\\tmp\\audio.wav").as_deref(),
            Some("audio.wav")
        );
    }

    #[test]
    fn job_status_normalizes_unknown_to_failed() {
        assert_eq!(
            normalize_job_status("TRANSCribing"),
            JOB_STATUS_TRANSCRIBING
        );
        assert_eq!(normalize_job_status("mystery"), JOB_STATUS_FAILED);
    }
}
