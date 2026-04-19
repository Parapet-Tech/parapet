// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;

use sha2::{Digest, Sha256};

/// Canonical sensor input for one row under observation.
///
/// `content_hash` follows the existing parapet-data contract:
/// SHA256 of UTF-8 bytes after trimming leading/trailing whitespace only.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SensorInput {
    pub content: String,
    pub content_hash: String,
    pub source_id: Option<String>,
    pub label_at_time: Option<String>,
    pub reason_at_time: Option<String>,
    pub layer_at_time: Option<String>,
}

impl SensorInput {
    pub fn new(content: impl Into<String>) -> Self {
        let content = content.into();
        let content_hash = content_hash(&content);
        Self {
            content,
            content_hash,
            source_id: None,
            label_at_time: None,
            reason_at_time: None,
            layer_at_time: None,
        }
    }
}

/// Metadata-only input for corpus-scope sensors.
///
/// This avoids pulling full text into sensors that only need stable row
/// metadata such as `content_hash`, origin, and labels.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CorpusSensorInput {
    pub content_hash: String,
    pub source_id: Option<String>,
    pub label_at_time: Option<String>,
    pub reason_at_time: Option<String>,
    pub layer_at_time: Option<String>,
    pub origin_id: Option<String>,
    pub row_locator: Option<String>,
}

impl CorpusSensorInput {
    pub fn new(content_hash: impl Into<String>) -> Self {
        Self {
            content_hash: content_hash.into(),
            source_id: None,
            label_at_time: None,
            reason_at_time: None,
            layer_at_time: None,
            origin_id: None,
            row_locator: None,
        }
    }
}

/// Stable semantic version string for an observation sensor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SensorVersion {
    pub sensor_id: String,
    pub version: String,
}

impl SensorVersion {
    pub fn new(sensor_id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            version: version.into(),
        }
    }
}

/// Named byte range for JSONL-friendly evidence serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}

impl ByteRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Structured evidence explaining why a sensor emitted an observation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ObservationEvidence {
    pub kind: String,
    pub detail: Option<String>,
    pub byte_range: Option<ByteRange>,
    pub related_content_hash: Option<String>,
}

impl ObservationEvidence {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            detail: None,
            byte_range: None,
            related_content_hash: None,
        }
    }
}

/// One normalized sensor observation.
///
/// `run_id` is intentionally not duplicated per row. Observation artifacts are
/// scoped by a run-specific path, and rows stay focused on stable content-level
/// evidence so identical findings can be compared across runs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct Observation {
    pub sensor_id: String,
    pub sensor_version: String,
    pub content_hash: String,
    pub source_id: Option<String>,
    pub severity: String,
    pub category: String,
    pub message: String,
    pub evidences: Vec<ObservationEvidence>,
}

impl Observation {
    pub fn new(
        sensor_id: impl Into<String>,
        sensor_version: impl Into<String>,
        content_hash: impl Into<String>,
        severity: impl Into<String>,
        category: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            sensor_version: sensor_version.into(),
            content_hash: content_hash.into(),
            source_id: None,
            severity: severity.into(),
            category: category.into(),
            message: message.into(),
            evidences: Vec::new(),
        }
    }
}

/// A deterministic batch of observations emitted by one sensor for one input.
///
/// `sensor` is batch-level metadata for manifest-style artifacts. Canonical
/// JSONL stays row-oriented and self-describing, so `to_canonical_jsonl()`
/// serializes only the observations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ObservationBatch {
    pub sensor: SensorVersion,
    pub observations: Vec<Observation>,
}

impl ObservationBatch {
    pub fn empty(sensor_id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            sensor: SensorVersion::new(sensor_id, version),
            observations: Vec::new(),
        }
    }

    pub fn sort_canonical(&mut self) {
        self.observations.sort_by(|a, b| {
            (
                &a.content_hash,
                &a.category,
                &a.severity,
                &a.message,
                &a.source_id,
            )
                .cmp(&(
                    &b.content_hash,
                    &b.category,
                    &b.severity,
                    &b.message,
                    &b.source_id,
                ))
        });
    }

    pub fn to_canonical_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut cloned = self.clone();
        cloned.sort_canonical();

        let mut out = String::new();
        for observation in &cloned.observations {
            out.push_str(&serde_json::to_string(observation)?);
            out.push('\n');
        }
        Ok(out)
    }
}

/// A reusable, deterministic observation sensor.
pub trait ObservationSensor: Send + Sync {
    fn sensor_id(&self) -> Cow<'static, str>;
    fn version(&self) -> Cow<'static, str>;
    fn observe(&self, input: &SensorInput) -> ObservationBatch;
}

/// A reusable, deterministic corpus-scope observation sensor.
pub trait CorpusObservationSensor: Send + Sync {
    fn sensor_id(&self) -> Cow<'static, str>;
    fn version(&self) -> Cow<'static, str>;
    fn observe_corpus(&self, inputs: &[CorpusSensorInput]) -> ObservationBatch;
}

/// SHA256 of UTF-8 bytes after trimming leading/trailing whitespace only.
///
/// This intentionally preserves internal whitespace and does not apply
/// normalization before hashing.
pub fn content_hash(text: &str) -> String {
    let trimmed = text.trim();
    let mut hasher = Sha256::new();
    hasher.update(trimmed.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_stable_for_identical_input() {
        assert_eq!(content_hash("test"), content_hash("test"));
    }

    #[test]
    fn content_hash_trims_leading_and_trailing_whitespace_only() {
        let base = "hello world";
        assert_eq!(content_hash(base), content_hash("  hello world\t\n"));
        assert_ne!(content_hash("a  b"), content_hash("a b"));
    }

    #[test]
    fn sensor_input_new_populates_content_hash() {
        let input = SensorInput::new("keep me");
        assert_eq!(input.content_hash, content_hash("keep me"));
    }

    #[test]
    fn corpus_sensor_input_new_sets_content_hash_only() {
        let input = CorpusSensorInput::new("abc123");
        assert_eq!(input.content_hash, "abc123");
        assert!(input.source_id.is_none());
        assert!(input.origin_id.is_none());
    }

    #[test]
    fn observation_evidence_new_has_no_location_or_related_hash() {
        let evidence = ObservationEvidence::new("test_kind");
        assert_eq!(evidence.kind, "test_kind");
        assert!(evidence.byte_range.is_none());
        assert!(evidence.related_content_hash.is_none());
    }

    #[test]
    fn observation_batch_sort_canonical_orders_stably() {
        let mut batch = ObservationBatch {
            sensor: SensorVersion::new("test_sensor", "1"),
            observations: vec![
                Observation::new("test_sensor", "1", "b", "warn", "beta", "second"),
                Observation::new("test_sensor", "1", "a", "warn", "beta", "first"),
            ],
        };

        batch.sort_canonical();
        assert_eq!(batch.observations[0].content_hash, "a");
        assert_eq!(batch.observations[1].content_hash, "b");
    }

    #[test]
    fn observation_batch_canonical_jsonl_is_deterministic() {
        let batch = ObservationBatch {
            sensor: SensorVersion::new("test_sensor", "1"),
            observations: vec![
                Observation::new("test_sensor", "1", "b", "warn", "beta", "second"),
                Observation::new("test_sensor", "1", "a", "warn", "alpha", "first"),
            ],
        };

        let once = batch.to_canonical_jsonl().unwrap();
        let twice = batch.to_canonical_jsonl().unwrap();
        assert_eq!(once, twice);
        assert!(once.starts_with("{\"sensor_id\":\"test_sensor\",\"sensor_version\":\"1\",\"content_hash\":\"a\""));
    }
}
