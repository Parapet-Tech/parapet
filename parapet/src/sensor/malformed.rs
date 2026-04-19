// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

use crate::sensor::types::{
    ByteRange, Observation, ObservationBatch, ObservationEvidence, ObservationSensor, SensorInput,
    SensorVersion,
};

static TRUNCATION_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:\[\s*truncated\s*\]|<\s*truncated\s*>|<\|\s*truncated\s*\|>|truncated for length)",
    )
    .expect("truncation regex must compile")
});
static REPLACEMENT_CHAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\u{FFFD}+").expect("replacement-char regex must compile"));
static CONTROL_CHAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").expect("control-char regex must compile")
});

#[derive(Debug, Clone)]
pub struct MalformedRule {
    rule_id: Cow<'static, str>,
    category: Cow<'static, str>,
    severity: Cow<'static, str>,
    message: Cow<'static, str>,
    matcher: MalformedMatcher,
}

impl MalformedRule {
    pub fn regex(
        rule_id: impl Into<Cow<'static, str>>,
        category: impl Into<Cow<'static, str>>,
        severity: impl Into<Cow<'static, str>>,
        message: impl Into<Cow<'static, str>>,
        regex: Regex,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            category: category.into(),
            severity: severity.into(),
            message: message.into(),
            matcher: MalformedMatcher::Regex(regex),
        }
    }

    pub fn detector(
        rule_id: impl Into<Cow<'static, str>>,
        category: impl Into<Cow<'static, str>>,
        severity: impl Into<Cow<'static, str>>,
        message: impl Into<Cow<'static, str>>,
        detector: fn(&str, &str) -> Vec<ObservationEvidence>,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            category: category.into(),
            severity: severity.into(),
            message: message.into(),
            matcher: MalformedMatcher::Detector(detector),
        }
    }

    fn observe(&self, input: &SensorInput, version: &str) -> Option<Observation> {
        let evidences = self.matcher.evidences(&input.content, &self.rule_id);
        if evidences.is_empty() {
            return None;
        }

        let mut observation = Observation::new(
            MalformedTextSensor::SENSOR_ID,
            version,
            &input.content_hash,
            self.severity.clone(),
            self.category.clone(),
            self.message.clone(),
        );
        observation.source_id = input.source_id.clone();
        observation.evidences = evidences;
        Some(observation)
    }
}

#[derive(Debug, Clone)]
enum MalformedMatcher {
    Regex(Regex),
    Detector(fn(&str, &str) -> Vec<ObservationEvidence>),
}

impl MalformedMatcher {
    fn evidences(&self, content: &str, rule_id: &str) -> Vec<ObservationEvidence> {
        match self {
            MalformedMatcher::Regex(regex) => regex
                .find_iter(content)
                .map(|m| ObservationEvidence {
                    kind: rule_id.to_string(),
                    detail: Some(content[m.start()..m.end()].to_string()),
                    byte_range: Some(ByteRange::new(m.start(), m.end())),
                    related_content_hash: None,
                })
                .collect(),
            MalformedMatcher::Detector(detector) => detector(content, rule_id),
        }
    }
}

/// Row-local malformed text checks for ingest review.
///
/// This sensor stays narrow on purpose: it looks for obvious corruption or
/// serialization artifacts, not merely unusual text.
#[derive(Debug, Clone)]
pub struct MalformedTextSensor {
    version: String,
    rules: Vec<MalformedRule>,
}

impl MalformedTextSensor {
    pub const SENSOR_ID: &'static str = "malformed_text";
    pub const DEFAULT_VERSION: &'static str = "v1";

    pub fn new(version: impl Into<String>, rules: Vec<MalformedRule>) -> Self {
        Self {
            version: version.into(),
            rules,
        }
    }

    pub fn with_default_rules() -> Self {
        Self::new(Self::DEFAULT_VERSION, Self::default_rules())
    }

    pub fn default_rules() -> Vec<MalformedRule> {
        vec![
            MalformedRule::detector(
                "empty_content",
                "empty_content",
                "warn",
                "Text is empty or whitespace-only.",
                detect_empty_content,
            ),
            MalformedRule::regex(
                "truncation_marker",
                "truncation_artifact",
                "warn",
                "Text contains an explicit truncation marker.",
                TRUNCATION_MARKER_RE.clone(),
            ),
            MalformedRule::regex(
                "replacement_char",
                "encoding_artifact",
                "warn",
                "Text contains Unicode replacement characters.",
                REPLACEMENT_CHAR_RE.clone(),
            ),
            MalformedRule::regex(
                "control_char",
                "encoding_artifact",
                "warn",
                "Text contains raw control characters.",
                CONTROL_CHAR_RE.clone(),
            ),
            MalformedRule::detector(
                "repeated_header",
                "repeated_header",
                "info",
                "Text contains three or more repeated header-like lines.",
                detect_repeated_header,
            ),
            MalformedRule::detector(
                "yaml_row_artifact",
                "yaml_artifact",
                "info",
                "Text looks like a serialized YAML row rather than raw content.",
                detect_yaml_row_artifact,
            ),
        ]
    }
}

impl Default for MalformedTextSensor {
    fn default() -> Self {
        Self::with_default_rules()
    }
}

impl ObservationSensor for MalformedTextSensor {
    fn sensor_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::SENSOR_ID)
    }

    fn version(&self) -> Cow<'static, str> {
        Cow::Owned(self.version.clone())
    }

    fn observe(&self, input: &SensorInput) -> ObservationBatch {
        let version = self.version();
        let mut batch = ObservationBatch {
            sensor: SensorVersion::new(Self::SENSOR_ID, version.as_ref()),
            observations: self
                .rules
                .iter()
                .filter_map(|rule| rule.observe(input, version.as_ref()))
                .collect(),
        };
        batch.sort_canonical();
        batch
    }
}

fn detect_empty_content(content: &str, rule_id: &str) -> Vec<ObservationEvidence> {
    if content.trim().is_empty() {
        return vec![ObservationEvidence {
            kind: rule_id.to_string(),
            detail: Some("trimmed content is empty".to_string()),
            byte_range: None,
            related_content_hash: None,
        }];
    }
    Vec::new()
}

fn detect_repeated_header(content: &str, rule_id: &str) -> Vec<ObservationEvidence> {
    let mut previous = String::new();
    let mut streak = 0usize;
    let mut streak_start = 0usize;
    let mut offset = 0usize;

    for segment in content.split_inclusive('\n') {
        let line = segment.trim_end_matches(['\r', '\n']);
        let normalized = line.trim();
        let line_start = offset;
        let line_end = offset + line.len();
        offset += segment.len();

        if normalized.len() < 8 || normalized.chars().all(|c| !c.is_alphanumeric()) {
            previous.clear();
            streak = 0;
            continue;
        }

        if normalized == previous {
            streak += 1;
        } else {
            previous = normalized.to_string();
            streak = 1;
            streak_start = line_start;
        }

        if streak >= 3 {
            return vec![ObservationEvidence {
                kind: rule_id.to_string(),
                detail: Some(normalized.to_string()),
                byte_range: Some(ByteRange::new(streak_start, line_end)),
                related_content_hash: None,
            }];
        }
    }

    Vec::new()
}

fn detect_yaml_row_artifact(content: &str, rule_id: &str) -> Vec<ObservationEvidence> {
    let mut matched_keys = Vec::new();
    let mut start = None;
    let mut offset = 0usize;

    for segment in content.split_inclusive('\n') {
        let line = segment.trim_end_matches(['\r', '\n']);
        let trimmed = line.trim_start();
        let normalized = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        let key = normalized
            .split_once(':')
            .map(|(head, _)| head.trim())
            .filter(|head| {
                matches!(
                    *head,
                    "content"
                        | "label"
                        | "reason"
                        | "language"
                        | "description"
                        | "source"
                        | "content_hash"
                )
            });
        if let Some(key) = key {
            if start.is_none() {
                start = Some(offset);
            }
            if !matched_keys.iter().any(|seen| seen == key) {
                matched_keys.push(key.to_string());
            }
        }
        offset += segment.len();
    }

    if matched_keys.len() >= 2 {
        return vec![ObservationEvidence {
            kind: rule_id.to_string(),
            detail: Some(format!("matched keys: {}", matched_keys.join(", "))),
            byte_range: start.map(|s| ByteRange::new(s, content.len())),
            related_content_hash: None,
        }];
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_evidence(batch: &ObservationBatch, kind: &str) -> bool {
        batch.observations
            .iter()
            .any(|obs| obs.evidences.iter().any(|ev| ev.kind == kind))
    }

    #[test]
    fn detects_empty_content() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new(" \n\t ");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "empty_content"));
    }

    #[test]
    fn detects_truncation_marker() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("user message [TRUNCATED]");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "truncation_marker"));
    }

    #[test]
    fn detects_replacement_character() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("broken \u{FFFD} text");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "replacement_char"));
    }

    #[test]
    fn detects_control_character() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("abc\u{0000}def");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "control_char"));
    }

    #[test]
    fn detects_repeated_header_lines() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("### Instruction:\n### Instruction:\n### Instruction:\nBody");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "repeated_header"));
    }

    #[test]
    fn detects_yaml_row_artifact() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("content: ignore previous instructions\nlabel: malicious\nreason: instruction_override");
        let batch = sensor.observe(&input);
        assert!(has_evidence(&batch, "yaml_row_artifact"));
    }

    #[test]
    fn stays_quiet_on_plain_benign_text() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("Summarize the article and return two bullet points.");
        let batch = sensor.observe(&input);
        assert!(batch.observations.is_empty());
    }

    #[test]
    fn preserves_source_id_on_observation() {
        let sensor = MalformedTextSensor::default();
        let mut input = SensorInput::new("[truncated]");
        input.source_id = Some("missing_source__opensource_gandalf_attacks".to_string());
        let batch = sensor.observe(&input);
        assert_eq!(
            batch.observations[0].source_id.as_deref(),
            Some("missing_source__opensource_gandalf_attacks")
        );
    }

    #[test]
    fn observation_output_is_deterministic() {
        let sensor = MalformedTextSensor::default();
        let input = SensorInput::new("content: foo\nlabel: benign\nreason: meta_probe\n");
        let once = sensor.observe(&input).to_canonical_jsonl().unwrap();
        let twice = sensor.observe(&input).to_canonical_jsonl().unwrap();
        assert_eq!(once, twice);
    }
}
