use std::collections::{HashMap, HashSet};

pub const MERGE_GAP_SECONDS: f64 = 0.05;

#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub speaker: String,
    pub words: Vec<Word>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeakerMatch {
    pub matched_id: Option<String>,
    pub matched_name: Option<String>,
    pub similarity: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub speaker_label: String,
    pub speaker_id: Option<String>,
    pub speaker_name: String,
    pub similarity: f64,
    pub words: Vec<Word>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PostprocessResult {
    pub segments: Vec<ResultSegment>,
    pub unique_speakers: Vec<String>,
}

fn safe_number(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn round_to(value: f64, scale: f64) -> f64 {
    (value * scale).round() / scale
}

fn round_time(value: f64) -> f64 {
    round_to(safe_number(value).max(0.0), 1000.0)
}

fn round_score(value: f64) -> f64 {
    round_to(safe_number(value), 10000.0)
}

pub fn normalize_word(word: Word) -> Word {
    Word {
        word: word.word,
        start: round_time(word.start),
        end: round_time(word.end),
        score: round_score(word.score),
    }
}

pub fn normalize_segment(segment: AlignedSegment) -> AlignedSegment {
    AlignedSegment {
        start: round_time(segment.start),
        end: round_time(segment.end),
        text: segment.text.trim().to_string(),
        speaker: if segment.speaker.is_empty() {
            "UNKNOWN".to_string()
        } else {
            segment.speaker
        },
        words: segment.words.into_iter().map(normalize_word).collect(),
    }
}

fn can_merge_segments(previous: &AlignedSegment, current: &AlignedSegment) -> bool {
    previous.speaker == current.speaker
        && previous.words.is_empty()
        && current.words.is_empty()
        && current.start <= previous.end + MERGE_GAP_SECONDS
}

pub fn merge_aligned_segments(segments: Vec<AlignedSegment>) -> Vec<AlignedSegment> {
    let mut merged: Vec<AlignedSegment> = Vec::new();
    for raw_segment in segments {
        let segment = normalize_segment(raw_segment);
        if let Some(previous) = merged.last_mut() {
            if can_merge_segments(previous, &segment) {
                previous.end = previous.end.max(segment.end);
                let previous_text = previous.text.trim();
                let current_text = segment.text.trim();
                previous.text = match (previous_text.is_empty(), current_text.is_empty()) {
                    (true, true) => String::new(),
                    (true, false) => current_text.to_string(),
                    (false, true) => previous_text.to_string(),
                    (false, false) => format!("{previous_text} {current_text}"),
                };
                continue;
            }
        }
        merged.push(segment);
    }
    merged
}

pub fn build_display_names(
    speaker_labels: &[String],
    speaker_map: &HashMap<String, SpeakerMatch>,
) -> HashMap<String, String> {
    let mut labels_by_name: Vec<(String, Vec<String>)> = Vec::new();

    for speaker_label in speaker_labels {
        let speaker_name = speaker_map
            .get(speaker_label)
            .and_then(|entry| entry.matched_name.as_ref())
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| speaker_label.clone());

        if let Some((_, labels)) = labels_by_name
            .iter_mut()
            .find(|(known_name, _)| known_name == &speaker_name)
        {
            labels.push(speaker_label.clone());
        } else {
            labels_by_name.push((speaker_name, vec![speaker_label.clone()]));
        }
    }

    let mut display_names = HashMap::new();
    for (speaker_name, labels) in labels_by_name {
        for (index, speaker_label) in labels.into_iter().enumerate() {
            let display_name = if index == 0 {
                speaker_name.clone()
            } else {
                format!("{} ({})", speaker_name, index + 1)
            };
            display_names.insert(speaker_label, display_name);
        }
    }
    display_names
}

pub fn build_result_segments(
    aligned_segments: Vec<AlignedSegment>,
    speaker_map: HashMap<String, SpeakerMatch>,
) -> PostprocessResult {
    let merged_segments = merge_aligned_segments(aligned_segments);
    let mut seen_labels = HashSet::new();
    let mut speaker_labels = Vec::new();
    for segment in &merged_segments {
        if seen_labels.insert(segment.speaker.clone()) {
            speaker_labels.push(segment.speaker.clone());
        }
    }

    let display_names = build_display_names(&speaker_labels, &speaker_map);
    let mut seen_speakers = HashSet::new();
    let mut unique_speakers = Vec::new();
    let mut segments = Vec::with_capacity(merged_segments.len());

    for (index, segment) in merged_segments.into_iter().enumerate() {
        let speaker_label = segment.speaker;
        let speaker_match = speaker_map.get(&speaker_label);
        let speaker_name = display_names
            .get(&speaker_label)
            .cloned()
            .unwrap_or_else(|| speaker_label.clone());
        let similarity = speaker_match
            .and_then(|entry| entry.similarity)
            .map(safe_number)
            .unwrap_or(0.0);
        if seen_speakers.insert(speaker_name.clone()) {
            unique_speakers.push(speaker_name.clone());
        }
        segments.push(ResultSegment {
            id: index,
            start: segment.start,
            end: segment.end,
            text: segment.text,
            speaker_label,
            speaker_id: speaker_match.and_then(|entry| entry.matched_id.clone()),
            speaker_name,
            similarity,
            words: segment.words,
        });
    }

    PostprocessResult {
        segments,
        unique_speakers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_preserves_speaker_label_and_skips_word_segments() {
        let segments = vec![
            AlignedSegment {
                start: 0.0,
                end: 1.0,
                text: " first ".to_string(),
                speaker: "SPEAKER_00".to_string(),
                words: vec![],
            },
            AlignedSegment {
                start: 1.02,
                end: 2.0,
                text: "second".to_string(),
                speaker: "SPEAKER_00".to_string(),
                words: vec![],
            },
            AlignedSegment {
                start: 2.0,
                end: 3.0,
                text: "worded".to_string(),
                speaker: "SPEAKER_00".to_string(),
                words: vec![Word {
                    word: "worded".to_string(),
                    start: 2.0,
                    end: 2.5,
                    score: 0.5,
                }],
            },
        ];

        let merged = merge_aligned_segments(segments);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].speaker, "SPEAKER_00");
        assert_eq!(merged[0].text, "first second");
        assert_eq!(merged[1].text, "worded");
    }

    #[test]
    fn display_names_disambiguate_without_merging_speakers() {
        let labels = vec!["SPEAKER_00".to_string(), "SPEAKER_01".to_string()];
        let mut speaker_map = HashMap::new();
        speaker_map.insert(
            "SPEAKER_00".to_string(),
            SpeakerMatch {
                matched_id: Some("spk_same".to_string()),
                matched_name: Some("Maple".to_string()),
                similarity: Some(2.0),
            },
        );
        speaker_map.insert(
            "SPEAKER_01".to_string(),
            SpeakerMatch {
                matched_id: Some("spk_same".to_string()),
                matched_name: Some("Maple".to_string()),
                similarity: Some(1.0),
            },
        );

        let display_names = build_display_names(&labels, &speaker_map);

        assert_eq!(display_names["SPEAKER_00"], "Maple");
        assert_eq!(display_names["SPEAKER_01"], "Maple (2)");
    }

    #[test]
    fn word_normalization_is_json_safe() {
        let word = normalize_word(Word {
            word: "hello".to_string(),
            start: -1.0,
            end: f64::NAN,
            score: f64::INFINITY,
        });

        assert_eq!(
            word,
            Word {
                word: "hello".to_string(),
                start: 0.0,
                end: 0.0,
                score: 0.0,
            }
        );
    }

    #[test]
    fn result_segments_preserve_labels_and_disambiguate_names() {
        let aligned_segments = vec![
            AlignedSegment {
                start: 0.0,
                end: 1.0,
                text: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                words: vec![],
            },
            AlignedSegment {
                start: 1.0,
                end: 2.0,
                text: "world".to_string(),
                speaker: "SPEAKER_01".to_string(),
                words: vec![Word {
                    word: "world".to_string(),
                    start: -1.0,
                    end: 2.0,
                    score: 0.77777,
                }],
            },
        ];
        let mut speaker_map = HashMap::new();
        speaker_map.insert(
            "SPEAKER_00".to_string(),
            SpeakerMatch {
                matched_id: Some("spk_same".to_string()),
                matched_name: Some("Maple".to_string()),
                similarity: Some(2.0),
            },
        );
        speaker_map.insert(
            "SPEAKER_01".to_string(),
            SpeakerMatch {
                matched_id: Some("spk_same".to_string()),
                matched_name: Some("Maple".to_string()),
                similarity: Some(1.0),
            },
        );

        let result = build_result_segments(aligned_segments, speaker_map);

        assert_eq!(result.unique_speakers, vec!["Maple", "Maple (2)"]);
        assert_eq!(result.segments[0].speaker_label, "SPEAKER_00");
        assert_eq!(result.segments[1].speaker_label, "SPEAKER_01");
        assert_eq!(result.segments[0].speaker_name, "Maple");
        assert_eq!(result.segments[1].speaker_name, "Maple (2)");
        assert_eq!(result.segments[1].words[0].start, 0.0);
        assert_eq!(result.segments[1].words[0].score, 0.7778);
    }
}
