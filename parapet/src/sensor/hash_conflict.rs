// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::sensor::types::{
    CorpusObservationSensor, CorpusSensorInput, Observation, ObservationBatch, ObservationEvidence,
    SensorVersion,
};

const DEFAULT_EVIDENCE_CAP: usize = 20;

#[derive(Debug, Clone)]
pub struct HashConflictSensor {
    version: String,
    evidence_cap: usize,
}

impl HashConflictSensor {
    pub const SENSOR_ID: &'static str = "hash_conflict";
    pub const DEFAULT_VERSION: &'static str = "v1";

    pub fn new(version: impl Into<String>, evidence_cap: usize) -> Self {
        Self {
            version: version.into(),
            evidence_cap,
        }
    }

    fn duplicate_severity(origins: &BTreeSet<String>) -> &'static str {
        let has_trainish = origins
            .iter()
            .any(|origin| matches!(origin.as_str(), "staging" | "training_mix" | "curated"));
        let has_evalish = origins
            .iter()
            .any(|origin| matches!(origin.as_str(), "challenge" | "holdout"));

        if has_trainish && has_evalish {
            "warn"
        } else {
            "info"
        }
    }

    fn make_observation(
        &self,
        content_hash: &str,
        severity: &str,
        category: &str,
        message: String,
        rows: &[ClusterRow<'_>],
    ) -> Observation {
        let mut observation = Observation::new(
            Self::SENSOR_ID,
            &self.version,
            content_hash,
            severity,
            category,
            message,
        );
        observation.evidences = capped_evidences(rows, self.evidence_cap);
        observation
    }
}

impl Default for HashConflictSensor {
    fn default() -> Self {
        Self::new(Self::DEFAULT_VERSION, DEFAULT_EVIDENCE_CAP)
    }
}

impl CorpusObservationSensor for HashConflictSensor {
    fn sensor_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::SENSOR_ID)
    }

    fn version(&self) -> Cow<'static, str> {
        Cow::Owned(self.version.clone())
    }

    fn observe_corpus(&self, inputs: &[CorpusSensorInput]) -> ObservationBatch {
        let version = self.version();
        let mut groups: BTreeMap<&str, Vec<ClusterRow<'_>>> = BTreeMap::new();

        for input in inputs {
            groups
                .entry(input.content_hash.as_str())
                .or_default()
                .push(ClusterRow::new(input));
        }

        let mut observations = Vec::new();

        for (content_hash, rows) in groups {
            if rows.len() < 2 {
                continue;
            }

            let distinct_labels: BTreeSet<&str> =
                rows.iter().filter_map(|row| row.label.as_deref()).collect();
            if distinct_labels.len() >= 2 {
                observations.push(self.make_observation(
                    content_hash,
                    "warn",
                    "label_conflict",
                    "Same content hash appears with conflicting labels.".to_string(),
                    &rows,
                ));
            }

            let distinct_origins: BTreeSet<String> = rows
                .iter()
                .filter_map(|row| row.origin.as_deref())
                .map(ToOwned::to_owned)
                .collect();
            if distinct_origins.len() >= 2 {
                let severity = Self::duplicate_severity(&distinct_origins);
                let message = if severity == "warn" {
                    "Same content hash crosses a train/eval origin boundary."
                } else {
                    "Same content hash appears in multiple origins."
                };
                observations.push(self.make_observation(
                    content_hash,
                    severity,
                    "cross_corpus_duplicate",
                    message.to_string(),
                    &rows,
                ));
            }

            let mut subgroup_counts: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            for row in &rows {
                let Some(origin) = row.origin.as_deref() else {
                    continue;
                };
                let Some(source) = row.source.as_deref() else {
                    continue;
                };
                *subgroup_counts.entry((origin, source)).or_default() += 1;
            }

            for ((origin, source), count) in subgroup_counts {
                if count < 2 {
                    continue;
                }

                let subgroup_rows: Vec<ClusterRow<'_>> = rows
                    .iter()
                    .filter(|row| {
                        row.origin.as_deref() == Some(origin) && row.source.as_deref() == Some(source)
                    })
                    .cloned()
                    .collect();
                observations.push(self.make_observation(
                    content_hash,
                    "info",
                    "same_source_duplicate",
                    format!(
                        "Same content hash appears {} times in origin={} source={}.",
                        count, origin, source
                    ),
                    &subgroup_rows,
                ));
            }
        }

        let mut batch = ObservationBatch {
            sensor: SensorVersion::new(Self::SENSOR_ID, version.as_ref()),
            observations,
        };
        batch.sort_canonical();
        batch
    }
}

#[derive(Debug, Clone)]
struct ClusterRow<'a> {
    content_hash: &'a str,
    source: Option<String>,
    label: Option<String>,
    origin: Option<String>,
    row_locator: Option<String>,
}

impl<'a> ClusterRow<'a> {
    fn new(input: &'a CorpusSensorInput) -> Self {
        Self {
            content_hash: input.content_hash.as_str(),
            source: normalize_field(input.source_id.as_deref()),
            label: normalize_field(input.label_at_time.as_deref()),
            origin: normalize_field(input.origin_id.as_deref()),
            row_locator: normalize_nonempty(input.row_locator.as_deref()),
        }
    }
}

fn normalize_field(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
}

fn normalize_nonempty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn option_sort_key(value: &Option<String>) -> (u8, &str) {
    match value {
        Some(value) => (0, value.as_str()),
        None => (1, ""),
    }
}

fn evidence_detail(row: &ClusterRow<'_>) -> String {
    let origin = row.origin.as_deref().unwrap_or("-");
    let source = row.source.as_deref().unwrap_or("-");
    let label = row.label.as_deref().unwrap_or("-");
    let locator = row.row_locator.as_deref().unwrap_or("-");
    format!(
        "origin={} source={} label={} locator={}",
        origin, source, label, locator
    )
}

fn capped_evidences(rows: &[ClusterRow<'_>], cap: usize) -> Vec<ObservationEvidence> {
    let mut ordered = rows.to_vec();
    ordered.sort_by(|a, b| {
        (
            option_sort_key(&a.origin),
            option_sort_key(&a.source),
            option_sort_key(&a.label),
            option_sort_key(&a.row_locator),
        )
            .cmp(&(
                option_sort_key(&b.origin),
                option_sort_key(&b.source),
                option_sort_key(&b.label),
                option_sort_key(&b.row_locator),
            ))
    });

    let limit = ordered.len().min(cap);
    let mut evidences = ordered
        .iter()
        .take(limit)
        .map(|row| ObservationEvidence {
            kind: "cluster_row".to_string(),
            detail: Some(evidence_detail(row)),
            byte_range: None,
            related_content_hash: Some(row.content_hash.to_string()),
        })
        .collect::<Vec<_>>();

    if ordered.len() > cap {
        evidences.push(ObservationEvidence {
            kind: "and_more".to_string(),
            detail: Some(format!("{} additional rows", ordered.len() - cap)),
            byte_range: None,
            related_content_hash: None,
        });
    }

    evidences
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(hash: &str, origin: &str, source: &str, label: &str, locator: &str) -> CorpusSensorInput {
        let mut row = CorpusSensorInput::new(hash);
        row.origin_id = Some(origin.to_string());
        row.source_id = Some(source.to_string());
        row.label_at_time = Some(label.to_string());
        row.row_locator = Some(locator.to_string());
        row
    }

    fn find_category<'a>(batch: &'a ObservationBatch, category: &str) -> &'a Observation {
        batch
            .observations
            .iter()
            .find(|obs| obs.category == category)
            .expect("expected category to be present")
    }

    #[test]
    fn detects_same_source_duplicate() {
        let sensor = HashConflictSensor::default();
        let rows = vec![
            input("hash-a", "staging", "alpha", "benign", "a#1"),
            input("hash-a", "staging", "alpha", "benign", "a#2"),
        ];

        let batch = sensor.observe_corpus(&rows);
        let observation = find_category(&batch, "same_source_duplicate");
        assert_eq!(observation.severity, "info");
        assert!(observation.message.contains("origin=staging source=alpha"));
        assert!(observation.source_id.is_none());
    }

    #[test]
    fn detects_label_conflict_with_normalized_labels() {
        let sensor = HashConflictSensor::default();
        let rows = vec![
            input("hash-b", "staging", "alpha", "Malicious ", "a#1"),
            input("hash-b", "staging", "alpha", " benign", "a#2"),
        ];

        let batch = sensor.observe_corpus(&rows);
        let observation = find_category(&batch, "label_conflict");
        assert_eq!(observation.severity, "warn");
        let details = observation
            .evidences
            .iter()
            .filter_map(|evidence| evidence.detail.as_deref())
            .collect::<Vec<_>>();
        assert!(details.iter().any(|detail| detail.contains("label=malicious")));
        assert!(details.iter().any(|detail| detail.contains("label=benign")));
    }

    #[test]
    fn warns_for_train_eval_boundary_crossing() {
        let sensor = HashConflictSensor::default();
        let rows = vec![
            input("hash-c", "staging", "alpha", "malicious", "a#1"),
            input("hash-c", "challenge", "beta", "malicious", "b#1"),
        ];

        let batch = sensor.observe_corpus(&rows);
        let observation = find_category(&batch, "cross_corpus_duplicate");
        assert_eq!(observation.severity, "warn");
        assert!(observation.message.contains("train/eval origin boundary"));
    }

    #[test]
    fn emits_info_for_non_critical_origin_reuse() {
        let sensor = HashConflictSensor::default();
        let rows = vec![
            input("hash-d", "staging", "alpha", "malicious", "a#1"),
            input("hash-d", "verified", "beta", "malicious", "b#1"),
        ];

        let batch = sensor.observe_corpus(&rows);
        let observation = find_category(&batch, "cross_corpus_duplicate");
        assert_eq!(observation.severity, "info");
    }

    #[test]
    fn output_is_deterministic_under_shuffled_input() {
        let sensor = HashConflictSensor::default();
        let rows_a = vec![
            input("hash-e", "challenge", "beta", "benign", "b#1"),
            input("hash-e", "staging", "alpha", "malicious", "a#1"),
            input("hash-e", "staging", "alpha", "malicious", "a#2"),
        ];
        let rows_b = vec![
            input("hash-e", "staging", "alpha", "malicious", "a#2"),
            input("hash-e", "challenge", "beta", "benign", "b#1"),
            input("hash-e", "staging", "alpha", "malicious", "a#1"),
        ];

        let jsonl_a = sensor.observe_corpus(&rows_a).to_canonical_jsonl().unwrap();
        let jsonl_b = sensor.observe_corpus(&rows_b).to_canonical_jsonl().unwrap();
        assert_eq!(jsonl_a, jsonl_b);
    }

    #[test]
    fn caps_large_clusters_with_and_more_evidence() {
        let sensor = HashConflictSensor::new("v1", 2);
        let rows = vec![
            input("hash-f", "staging", "alpha", "benign", "a#1"),
            input("hash-f", "staging", "alpha", "benign", "a#2"),
            input("hash-f", "challenge", "beta", "benign", "b#1"),
        ];

        let batch = sensor.observe_corpus(&rows);
        let observation = find_category(&batch, "cross_corpus_duplicate");
        assert_eq!(observation.evidences.len(), 3);
        assert_eq!(observation.evidences[2].kind, "and_more");
        assert_eq!(
            observation.evidences[2].detail.as_deref(),
            Some("1 additional rows")
        );
    }
}
