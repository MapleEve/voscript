use voscript_core::voiceprint::{
    score_voiceprint_candidates, VoiceprintScoreCandidate, VoiceprintScoreRequest,
};

fn vec_at(angle: f64) -> Vec<f64> {
    vec![angle.cos(), angle.sin()]
}

fn candidate(
    speaker_id: &str,
    name: &str,
    angle: f64,
    sample_count: usize,
    sample_spread: Option<f64>,
) -> VoiceprintScoreCandidate {
    VoiceprintScoreCandidate {
        speaker_id: speaker_id.to_string(),
        name: name.to_string(),
        embedding: vec_at(angle),
        sample_count,
        sample_spread,
    }
}

fn cohort(angles: &[f64]) -> Vec<Vec<f64>> {
    angles.iter().map(|angle| vec_at(*angle)).collect()
}

#[test]
fn raw_scoring_matches_top_candidate_with_adaptive_threshold() {
    let result = score_voiceprint_candidates(VoiceprintScoreRequest {
        query_embedding: vec_at(0.0),
        candidates: vec![
            candidate("spk_alice", "Alice", 0.72_f64.acos(), 1, None),
            candidate("spk_bob", "Bob", 0.69_f64.acos(), 3, Some(0.0)),
        ],
        threshold: 0.75,
        asnorm_threshold: 0.5,
        cohort: None,
        asnorm_top_n: 200,
        asnorm_min_margin: 0.05,
    })
    .expect("voiceprint score should succeed");

    assert_eq!(result.matched_id.as_deref(), Some("spk_alice"));
    assert_eq!(result.matched_name.as_deref(), Some("Alice"));
    assert_eq!(result.reason, "matched");
    assert!(!result.asnorm_active);
    assert_eq!(result.asnorm_reason, "not_requested");
    assert!((result.similarity - 0.72).abs() < 1e-9);
    assert_eq!(result.candidates[0].speaker_id, "spk_alice");
    assert_eq!(result.candidates[0].score_method, "raw_cosine");
    assert!((result.candidates[0].effective_threshold - 0.70).abs() < 1e-9);
}

#[test]
fn small_asnorm_cohort_falls_back_to_raw_scoring() {
    let result = score_voiceprint_candidates(VoiceprintScoreRequest {
        query_embedding: vec_at(0.0),
        candidates: vec![candidate("spk_alice", "Alice", 0.0, 1, None)],
        threshold: 0.75,
        asnorm_threshold: 0.5,
        cohort: Some(cohort(&[0.0, 0.1, -0.1, 0.2, -0.2])),
        asnorm_top_n: 200,
        asnorm_min_margin: 0.05,
    })
    .expect("voiceprint score should succeed");

    assert_eq!(result.matched_id.as_deref(), Some("spk_alice"));
    assert!(!result.asnorm_active);
    assert_eq!(result.asnorm_reason, "cohort_too_small");
    assert_eq!(result.candidates[0].score_method, "raw_cosine");
    assert!((result.similarity - 1.0).abs() < 1e-8);
}

#[test]
fn asnorm_margin_rejects_ambiguous_top_two() {
    let result = score_voiceprint_candidates(VoiceprintScoreRequest {
        query_embedding: vec_at(0.0),
        candidates: vec![
            candidate("spk_first", "First", 0.0, 3, Some(0.0)),
            candidate("spk_second", "Second", 0.005, 3, Some(0.0)),
        ],
        threshold: 0.75,
        asnorm_threshold: 0.5,
        cohort: Some(cohort(&[
            1.0, 1.1, 1.2, 1.3, 1.4, -1.0, -1.1, -1.2, -1.3, -1.4,
        ])),
        asnorm_top_n: 200,
        asnorm_min_margin: 0.05,
    })
    .expect("voiceprint score should succeed");

    assert_eq!(result.matched_id, None);
    assert_eq!(result.matched_name, None);
    assert_eq!(result.reason, "ambiguous_margin");
    assert!(result.asnorm_active);
    assert_eq!(result.asnorm_reason, "active");
    assert!((result.similarity - 4.89135345).abs() < 1e-6);
    assert_eq!(result.candidates[0].score_method, "asnorm");
    assert!((result.candidates[1].similarity - 4.88978820).abs() < 1e-6);
}

#[test]
fn non_finite_embeddings_are_rejected() {
    let error = score_voiceprint_candidates(VoiceprintScoreRequest {
        query_embedding: vec![1.0, f64::NAN],
        candidates: vec![],
        threshold: 0.75,
        asnorm_threshold: 0.5,
        cohort: None,
        asnorm_top_n: 200,
        asnorm_min_margin: 0.05,
    })
    .expect_err("non-finite embeddings must fail closed");

    assert!(error.contains("finite"));
}
