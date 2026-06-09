const SINGLE_SAMPLE_RELAXATION: f64 = 0.05;
const SPREAD_RELAXATION_K: f64 = 3.0;
const SPREAD_RELAXATION_CAP: f64 = 0.10;
const ABSOLUTE_FLOOR: f64 = 0.60;
const ASNORM_MIN_COHORT_SIZE: usize = 10;
const ASNORM_SINGLE_SAMPLE_PENALTY: f64 = 0.10;
const ASNORM_LEGACY_SPREAD_UNKNOWN_PENALTY: f64 = 0.05;
const ASNORM_LOW_SAMPLE_PENALTY: f64 = 0.025;
const ASNORM_SPREAD_PENALTY_K: f64 = 0.50;
const ASNORM_SPREAD_PENALTY_CAP: f64 = 0.10;
const ASNORM_STABLE_RELAXATION: f64 = 0.02;

#[derive(Debug, Clone)]
pub struct VoiceprintScoreCandidate {
    pub speaker_id: String,
    pub name: String,
    pub embedding: Vec<f64>,
    pub sample_count: usize,
    pub sample_spread: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct VoiceprintScoreRequest {
    pub query_embedding: Vec<f64>,
    pub candidates: Vec<VoiceprintScoreCandidate>,
    pub threshold: f64,
    pub asnorm_threshold: f64,
    pub cohort: Option<Vec<Vec<f64>>>,
    pub asnorm_top_n: usize,
    pub asnorm_min_margin: f64,
}

#[derive(Debug, Clone)]
pub struct VoiceprintScoredCandidate {
    pub speaker_id: String,
    pub name: String,
    pub raw_similarity: f64,
    pub similarity: f64,
    pub effective_threshold: f64,
    pub score_method: String,
    pub sample_count: usize,
    pub sample_spread: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct VoiceprintScoreDecision {
    pub matched_id: Option<String>,
    pub matched_name: Option<String>,
    pub similarity: f64,
    pub reason: String,
    pub asnorm_active: bool,
    pub asnorm_reason: String,
    pub candidates: Vec<VoiceprintScoredCandidate>,
}

pub fn effective_threshold(base: f64, sample_count: usize, sample_spread: Option<f64>) -> f64 {
    let dynamic = if sample_count <= 1 || sample_spread.is_none() {
        if sample_count <= 1 {
            base - SINGLE_SAMPLE_RELAXATION
        } else {
            base
        }
    } else {
        let spread = sample_spread.unwrap_or(0.0).max(0.0);
        base - (SPREAD_RELAXATION_K * spread).min(SPREAD_RELAXATION_CAP)
    };
    ABSOLUTE_FLOOR.max(base.min(dynamic))
}

pub fn effective_asnorm_threshold(
    base: f64,
    sample_count: usize,
    sample_spread: Option<f64>,
) -> f64 {
    if sample_count <= 1 {
        return base + ASNORM_SINGLE_SAMPLE_PENALTY;
    }
    let Some(spread) = sample_spread else {
        return base + ASNORM_LEGACY_SPREAD_UNKNOWN_PENALTY;
    };

    let low_sample_penalty =
        (3usize.saturating_sub(sample_count)) as f64 * ASNORM_LOW_SAMPLE_PENALTY;
    let spread_penalty = (spread.max(0.0) * ASNORM_SPREAD_PENALTY_K).min(ASNORM_SPREAD_PENALTY_CAP);
    let mut threshold = base + low_sample_penalty + spread_penalty;
    if sample_count >= 3 && spread <= 0.03 {
        threshold -= ASNORM_STABLE_RELAXATION;
    }
    threshold.max(0.0)
}

pub fn score_voiceprint_candidates(
    request: VoiceprintScoreRequest,
) -> Result<VoiceprintScoreDecision, String> {
    validate_embedding("query_embedding", &request.query_embedding)?;
    if !request.threshold.is_finite()
        || !request.asnorm_threshold.is_finite()
        || !request.asnorm_min_margin.is_finite()
    {
        return Err("voiceprint thresholds must be finite".to_string());
    }
    if request.candidates.is_empty() {
        return Ok(no_match(
            "no_candidates",
            "not_requested",
            false,
            Vec::new(),
            0.0,
        ));
    }

    let query = match normalize(&request.query_embedding)? {
        Some(value) => value,
        None => {
            return Ok(no_match(
                "invalid_query",
                "not_requested",
                false,
                Vec::new(),
                0.0,
            ))
        }
    };

    for candidate in &request.candidates {
        validate_embedding("candidate embedding", &candidate.embedding)?;
        if candidate.embedding.len() != request.query_embedding.len() {
            return Err("voiceprint embeddings must share dimension".to_string());
        }
        if !candidate.sample_spread.unwrap_or(0.0).is_finite() {
            return Err("voiceprint sample_spread values must be finite".to_string());
        }
    }

    let (asnorm_active, asnorm_reason, normalized_cohort) = match request.cohort {
        None => (false, "not_requested".to_string(), None),
        Some(cohort) if cohort.len() < ASNORM_MIN_COHORT_SIZE => {
            validate_cohort(&cohort, request.query_embedding.len())?;
            (false, "cohort_too_small".to_string(), None)
        }
        Some(cohort) => {
            let normalized = normalize_cohort(&cohort, request.query_embedding.len())?;
            (true, "active".to_string(), Some(normalized))
        }
    };

    let mut scored = Vec::with_capacity(request.candidates.len());
    for candidate in request.candidates {
        let raw_similarity = cosine_from_normalized(&candidate.embedding, &query)?;
        let (similarity, effective, score_method) = if let Some(cohort) = &normalized_cohort {
            let normalized_score = asnorm_score(
                &candidate.embedding,
                &request.query_embedding,
                raw_similarity,
                cohort,
                request.asnorm_top_n,
            )?;
            (
                normalized_score,
                effective_asnorm_threshold(
                    request.asnorm_threshold,
                    candidate.sample_count,
                    candidate.sample_spread,
                ),
                "asnorm",
            )
        } else {
            (
                raw_similarity,
                effective_threshold(
                    request.threshold,
                    candidate.sample_count,
                    candidate.sample_spread,
                ),
                "raw_cosine",
            )
        };
        scored.push(VoiceprintScoredCandidate {
            speaker_id: candidate.speaker_id,
            name: candidate.name,
            raw_similarity,
            similarity,
            effective_threshold: effective,
            score_method: score_method.to_string(),
            sample_count: candidate.sample_count,
            sample_spread: candidate.sample_spread,
        });
    }

    scored.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if scored.is_empty() {
        return Ok(no_match(
            "no_candidates",
            &asnorm_reason,
            asnorm_active,
            scored,
            0.0,
        ));
    }

    let best_similarity = scored[0].similarity;
    let best_effective_threshold = scored[0].effective_threshold;

    if asnorm_active {
        let second_score = scored.get(1).map(|candidate| candidate.similarity);
        if let Some(second) = second_score {
            if best_similarity - second < request.asnorm_min_margin {
                return Ok(no_match(
                    "ambiguous_margin",
                    &asnorm_reason,
                    asnorm_active,
                    scored,
                    best_similarity,
                ));
            }
        }
    }

    if best_similarity >= best_effective_threshold {
        let matched_id = scored[0].speaker_id.clone();
        let matched_name = scored[0].name.clone();
        Ok(VoiceprintScoreDecision {
            matched_id: Some(matched_id),
            matched_name: Some(matched_name),
            similarity: best_similarity,
            reason: "matched".to_string(),
            asnorm_active,
            asnorm_reason,
            candidates: scored,
        })
    } else {
        Ok(no_match(
            "below_threshold",
            &asnorm_reason,
            asnorm_active,
            scored,
            best_similarity,
        ))
    }
}

fn no_match(
    reason: &str,
    asnorm_reason: &str,
    asnorm_active: bool,
    candidates: Vec<VoiceprintScoredCandidate>,
    similarity: f64,
) -> VoiceprintScoreDecision {
    VoiceprintScoreDecision {
        matched_id: None,
        matched_name: None,
        similarity,
        reason: reason.to_string(),
        asnorm_active,
        asnorm_reason: asnorm_reason.to_string(),
        candidates,
    }
}

fn validate_embedding(name: &str, embedding: &[f64]) -> Result<(), String> {
    if embedding.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} values must be finite"));
    }
    Ok(())
}

fn validate_cohort(cohort: &[Vec<f64>], dim: usize) -> Result<(), String> {
    for embedding in cohort {
        validate_embedding("cohort embedding", embedding)?;
        if embedding.len() != dim {
            return Err("voiceprint cohort embeddings must share dimension".to_string());
        }
    }
    Ok(())
}

fn normalize_cohort(cohort: &[Vec<f64>], dim: usize) -> Result<Vec<Vec<f64>>, String> {
    validate_cohort(cohort, dim)?;
    let mut normalized = Vec::with_capacity(cohort.len());
    for embedding in cohort {
        let Some(vector) = normalize(embedding)? else {
            return Err("voiceprint cohort embeddings must not be zero vectors".to_string());
        };
        normalized.push(vector);
    }
    Ok(normalized)
}

fn normalize(embedding: &[f64]) -> Result<Option<Vec<f64>>, String> {
    validate_embedding("embedding", embedding)?;
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm < 1e-12 {
        return Ok(None);
    }
    Ok(Some(embedding.iter().map(|value| value / norm).collect()))
}

fn cosine_from_normalized(embedding: &[f64], normalized_query: &[f64]) -> Result<f64, String> {
    let Some(normalized_embedding) = normalize(embedding)? else {
        return Ok(0.0);
    };
    Ok(dot(&normalized_embedding, normalized_query))
}

fn asnorm_score(
    enroll_emb: &[f64],
    test_emb: &[f64],
    raw_similarity: f64,
    normalized_cohort: &[Vec<f64>],
    top_n: usize,
) -> Result<f64, String> {
    let (mean_e, std_e) = cohort_stats(enroll_emb, normalized_cohort, top_n)?;
    let (mean_t, std_t) = cohort_stats(test_emb, normalized_cohort, top_n)?;
    Ok(0.5 * ((raw_similarity - mean_e) / std_e + (raw_similarity - mean_t) / std_t))
}

fn cohort_stats(
    embedding: &[f64],
    normalized_cohort: &[Vec<f64>],
    top_n: usize,
) -> Result<(f64, f64), String> {
    let Some(normalized_embedding) = normalize(embedding)? else {
        return Err("voiceprint AS-norm embedding must not be zero vector".to_string());
    };
    let mut scores = normalized_cohort
        .iter()
        .map(|cohort_embedding| dot(cohort_embedding, &normalized_embedding))
        .collect::<Vec<_>>();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let count = top_n.max(1).min(scores.len());
    let top = &scores[..count];
    let mean = top.iter().sum::<f64>() / count as f64;
    let variance = top
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / count as f64;
    Ok((mean, variance.sqrt() + 1e-8))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}
