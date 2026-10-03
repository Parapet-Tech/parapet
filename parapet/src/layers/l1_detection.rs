// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Replayable lexical evidence. This module makes no policy decision.
use super::l1::{l1_weights, SvmModel};
use super::l1_harness::{
    squash, strip_quotes, threshold_outcome, AttributedFeature, CalibrationParams, L1Harness,
    L1Model, L1Signal, OutcomeView, MENTION_RAW_DELTA_THRESHOLD,
};
use crate::config::{L1Config, L1Mode};
use crate::message::{Message, Role, TrustLevel};
use crate::normalize::{neutralize_role_markers, L0Normalizer, Normalizer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: &str = "l1-detection-record/v0";
pub const NORMALIZATION: &str = "l0:nfkc,html_strip,invisible_strip,confusable_fix,role_markers";
pub const TOP_K: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordMode {
    Shadow,
    Block,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionOutcome {
    ThresholdBreached,
    ThresholdNotBreached,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionStatus {
    Present,
    AbsentByContract,
    Errored,
}

/// Field declaration order is part of the v0 byte contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1DetectionRecord {
    pub schema_version: String,
    pub model_id: String,
    pub model_digest: String,
    pub code_version: String,
    pub mode: RecordMode,
    pub threshold: f64,
    pub mention_delta_threshold: f64,
    pub source_digest: String,
    pub source_len_bytes: usize,
    pub normalization: String,
    pub input_digest: String,
    pub input_len_bytes: usize,
    pub signal: L1Signal,
    pub outcome_view: OutcomeView,
    pub effective_raw: f64,
    pub detection_outcome: DetectionOutcome,
    pub view_text_digest: String,
    pub view_text_len_scalars: usize,
    pub view_text_lower_len_scalars: usize,
    pub attribution_status: AttributionStatus,
    pub top_k: usize,
    pub top_features: Vec<AttributedFeature>,
    pub feature_count_matched: Option<usize>,
    pub bias: f64,
    pub matched_weight_sum_positive: Option<f64>,
    pub matched_weight_sum_negative: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelIdentity {
    pub schema_version: String,
    pub model_id: String,
    pub model_digest: String,
    pub code_version: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScanError {
    #[error("input is empty after normalization")]
    EmptyInput,
    #[error("non-finite score, attribution, or threshold")]
    NonFinite,
}
#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("non-finite record field")]
    NonFinite,
    #[error("record serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn order_features(features: &mut [AttributedFeature]) {
    features.sort_by(|a, b| {
        b.weight
            .abs()
            .total_cmp(&a.weight.abs())
            .then_with(|| a.feature.as_bytes().cmp(b.feature.as_bytes()))
    });
}

fn digest_model(
    analyzer: &str,
    range: (u32, u32),
    bias: f64,
    calibration: CalibrationParams,
    entries: &[(&str, f64)],
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"parapet-l1-model/v1\n");
    hash.update(analyzer.as_bytes());
    hash.update(b"\n");
    hash.update(range.0.to_be_bytes());
    hash.update(range.1.to_be_bytes());
    hash.update(b"binary=1;lowercase=1\n");
    for number in [bias, calibration.a, calibration.b] {
        hash.update(number.to_bits().to_be_bytes());
    }
    hash.update((entries.len() as u32).to_be_bytes());
    let mut ordered = entries.to_vec();
    ordered.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for (feature, weight) in ordered {
        hash.update((feature.len() as u32).to_be_bytes());
        hash.update(feature.as_bytes());
        hash.update(weight.to_bits().to_be_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub fn model_identity() -> ModelIdentity {
    let entries: Vec<_> = l1_weights::WEIGHTS
        .entries()
        .map(|(&f, &w)| (f, w))
        .collect();
    ModelIdentity {
        schema_version: SCHEMA_VERSION.into(),
        model_id: "l1-generalist-2026-04-17".into(),
        model_digest: digest_model(
            "char_wb",
            (3, 5),
            l1_weights::BIAS,
            SvmModel.calibration(),
            &entries,
        ),
        code_version: env!("CARGO_PKG_VERSION").into(),
    }
}

/// Scan one untrusted user message with the compiled generalist.
pub fn scan_text(
    text: &str,
    cfg: &L1Config,
    normalize: bool,
) -> Result<L1DetectionRecord, ScanError> {
    scan_with(
        text,
        cfg,
        normalize,
        &SvmModel,
        &L0Normalizer,
        model_identity(),
        l1_weights::BIAS,
    )
}

// Explicit dependencies keep unsupported/error attribution paths testable without
// changing the public compiled-model entry point.
fn scan_with(
    text: &str,
    cfg: &L1Config,
    normalize: bool,
    model: &dyn L1Model,
    normalizer: &dyn Normalizer,
    identity: ModelIdentity,
    bias: f64,
) -> Result<L1DetectionRecord, ScanError> {
    if !cfg.threshold.is_finite() {
        return Err(ScanError::NonFinite);
    }
    let mut message = Message::new(Role::User, text);
    message.trust = TrustLevel::Untrusted;
    if normalize {
        message.content = normalizer.normalize(text);
        neutralize_role_markers(std::slice::from_mut(&mut message));
    }
    if message.content.is_empty() {
        return Err(ScanError::EmptyInput);
    }
    let signal = L1Harness::scan(std::slice::from_ref(&message), model).remove(0);
    if !signal_finite(&signal) {
        return Err(ScanError::NonFinite);
    }
    let outcome = threshold_outcome(&signal, cfg.threshold);
    let view = match outcome.outcome_view {
        OutcomeView::Raw => message.content.clone(),
        OutcomeView::Unquoted => {
            strip_quotes(&message.content).unwrap_or_else(|| message.content.clone())
        }
        OutcomeView::Squashed => squash(&message.content),
    };
    let attributed = model.score_attributed(&view);
    if !attributed.margin.is_finite()
        || attributed.features.as_ref().is_some_and(|features| {
            features
                .iter()
                .any(|f| !f.weight.is_finite() || !f.contribution.is_finite())
        })
    {
        return Err(ScanError::NonFinite);
    }
    let mut status = AttributionStatus::AbsentByContract;
    let mut features = Vec::new();
    let (mut count, mut positive, mut negative) = (None, None, None);
    if attributed.margin.to_bits() != outcome.effective_raw.to_bits() {
        status = AttributionStatus::Errored;
    } else if let Some(mut all) = attributed.features {
        status = AttributionStatus::Present;
        order_features(&mut all);
        let (mut pos, mut neg) = (0.0_f64, 0.0_f64);
        for feature in &all {
            if feature.weight > 0.0 {
                pos += feature.weight;
            }
            if feature.weight < 0.0 {
                neg += feature.weight;
            }
        }
        count = Some(all.len());
        positive = Some(pos);
        negative = Some(neg);
        all.truncate(TOP_K);
        features = all;
    }
    let record = L1DetectionRecord {
        schema_version: identity.schema_version,
        model_id: identity.model_id,
        model_digest: identity.model_digest,
        code_version: identity.code_version,
        mode: match cfg.mode {
            L1Mode::Shadow => RecordMode::Shadow,
            L1Mode::Block => RecordMode::Block,
        },
        threshold: cfg.threshold,
        mention_delta_threshold: MENTION_RAW_DELTA_THRESHOLD,
        source_digest: sha256(text.as_bytes()),
        source_len_bytes: text.len(),
        normalization: if normalize { NORMALIZATION } else { "none" }.into(),
        input_digest: sha256(message.content.as_bytes()),
        input_len_bytes: message.content.len(),
        signal,
        outcome_view: outcome.outcome_view,
        effective_raw: outcome.effective_raw,
        detection_outcome: if outcome.threshold_breached {
            DetectionOutcome::ThresholdBreached
        } else {
            DetectionOutcome::ThresholdNotBreached
        },
        view_text_digest: sha256(view.as_bytes()),
        view_text_len_scalars: view.chars().count(),
        view_text_lower_len_scalars: view.to_lowercase().chars().count(),
        attribution_status: status,
        top_k: TOP_K,
        top_features: features,
        feature_count_matched: count,
        bias,
        matched_weight_sum_positive: positive,
        matched_weight_sum_negative: negative,
    };
    validate_finite(&record).map_err(|_| ScanError::NonFinite)?;
    Ok(record)
}

fn signal_finite(s: &L1Signal) -> bool {
    [
        s.raw_score,
        s.raw_unquoted_score,
        s.raw_squash_score,
        s.raw_score_delta,
    ]
    .iter()
    .all(|x| x.is_finite())
        && [s.score, s.unquoted_score, s.squash_score]
            .iter()
            .all(|x| x.is_finite())
}
fn validate_finite(r: &L1DetectionRecord) -> Result<(), RecordError> {
    let finite = [
        r.threshold,
        r.mention_delta_threshold,
        r.effective_raw,
        r.bias,
    ]
    .iter()
    .all(|x| x.is_finite())
        && [r.matched_weight_sum_positive, r.matched_weight_sum_negative]
            .iter()
            .flatten()
            .all(|x| x.is_finite())
        && signal_finite(&r.signal)
        && r.top_features
            .iter()
            .all(|f| f.weight.is_finite() && f.contribution.is_finite());
    if finite {
        Ok(())
    } else {
        Err(RecordError::NonFinite)
    }
}

/// The sole byte serializer for digesting records. This guards finiteness,
/// not semantic conformance; callers may construct deliberately synthetic records.
pub fn record_bytes(record: &L1DetectionRecord) -> Result<Vec<u8>, RecordError> {
    validate_finite(record)?;
    Ok(serde_json::to_vec(record)?)
}

/// Hash the tagged library bytes, without including a JSONL newline.
pub fn record_digest(record: &L1DetectionRecord) -> Result<String, RecordError> {
    let bytes = record_bytes(record)?;
    let mut hash = Sha256::new();
    hash.update(b"parapet-l1-detection-record-digest/v0\n");
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::super::l1_harness::Attributed;
    use super::*;
    use std::collections::HashMap;
    fn config() -> L1Config {
        L1Config {
            mode: L1Mode::Shadow,
            threshold: 0.0,
            min_agree: 1,
            generalist_solo_threshold: None,
            specialists: HashMap::new(),
        }
    }
    struct Unsupported;
    impl L1Model for Unsupported {
        fn score(&self, _: &str) -> f64 {
            -0.5
        }
        fn calibration(&self) -> CalibrationParams {
            CalibrationParams { a: 0.6, b: 0.0 }
        }
    }
    struct Measured {
        score: f64,
        attributed: f64,
        weight: Option<f64>,
    }
    impl L1Model for Measured {
        fn score(&self, _: &str) -> f64 {
            self.score
        }
        fn calibration(&self) -> CalibrationParams {
            Unsupported.calibration()
        }
        fn score_attributed(&self, _: &str) -> Attributed {
            Attributed {
                margin: self.attributed,
                features: Some(
                    self.weight
                        .into_iter()
                        .map(|weight| AttributedFeature {
                            feature: "x".into(),
                            weight,
                            contribution: weight,
                            occurrences: 1,
                            spans: vec![[0, 1]],
                        })
                        .collect(),
                ),
            }
        }
    }
    fn scan(model: &dyn L1Model, config: &L1Config) -> Result<L1DetectionRecord, ScanError> {
        scan_with(
            "hello",
            config,
            false,
            model,
            &L0Normalizer,
            model_identity(),
            -0.5,
        )
    }
    #[test]
    fn attribution_status_three_way() {
        let absent = scan(&Unsupported, &config()).unwrap();
        let empty = scan(
            &Measured {
                score: -0.5,
                attributed: -0.5,
                weight: None,
            },
            &config(),
        )
        .unwrap();
        let error = scan(
            &Measured {
                score: -0.5,
                attributed: 0.5,
                weight: None,
            },
            &config(),
        )
        .unwrap();
        assert_eq!(
            absent.attribution_status,
            AttributionStatus::AbsentByContract
        );
        assert_eq!(empty.attribution_status, AttributionStatus::Present);
        assert_eq!(empty.feature_count_matched, Some(0));
        assert_eq!(
            empty.matched_weight_sum_positive.unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            empty.matched_weight_sum_negative.unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(error.attribution_status, AttributionStatus::Errored);
        for record in [absent, error] {
            assert!(record.top_features.is_empty());
            assert_eq!(record.feature_count_matched, None);
            assert_eq!(record.matched_weight_sum_positive, None);
            assert_eq!(record.matched_weight_sum_negative, None);
            assert_eq!(
                record.detection_outcome,
                DetectionOutcome::ThresholdNotBreached
            );
            let value = serde_json::to_value(&record).unwrap();
            for field in [
                "feature_count_matched",
                "matched_weight_sum_positive",
                "matched_weight_sum_negative",
            ] {
                assert!(value[field].is_null());
            }
        }
    }
    #[test]
    fn non_finite_is_scan_error() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut cfg = config();
            cfg.threshold = bad;
            assert!(matches!(
                scan_text("hello", &cfg, true),
                Err(ScanError::NonFinite)
            ));
            for model in [
                Measured {
                    score: bad,
                    attributed: 0.0,
                    weight: None,
                },
                Measured {
                    score: 0.0,
                    attributed: bad,
                    weight: None,
                },
                Measured {
                    score: 0.0,
                    attributed: 0.0,
                    weight: Some(bad),
                },
            ] {
                assert!(matches!(scan(&model, &config()), Err(ScanError::NonFinite)));
            }
        }
    }
    #[test]
    fn top_feature_order_rule() {
        let make = |name: &str, weight| AttributedFeature {
            feature: name.into(),
            weight,
            contribution: weight,
            occurrences: 0,
            spans: vec![],
        };
        let mut a = vec![
            make("é", -1.0),
            make("a", 1.0),
            make("aa", -1.0),
            make("z", 2.0),
            make("b", 0.0),
        ];
        let mut b = a.clone();
        b.reverse();
        order_features(&mut a);
        order_features(&mut b);
        assert_eq!(
            a.iter().map(|f| f.feature.as_str()).collect::<Vec<_>>(),
            vec!["z", "a", "aa", "é", "b"]
        );
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );

        struct TiedModel {
            features: Vec<AttributedFeature>,
        }
        impl L1Model for TiedModel {
            fn score(&self, _: &str) -> f64 {
                1.5
            }
            fn calibration(&self) -> CalibrationParams {
                CalibrationParams { a: 0.6, b: 0.0 }
            }
            fn score_attributed(&self, _: &str) -> Attributed {
                Attributed {
                    margin: 1.5,
                    features: Some(self.features.clone()),
                }
            }
        }
        let matched = |name: &str, weight, span| AttributedFeature {
            feature: name.into(),
            weight,
            contribution: weight,
            occurrences: 1,
            spans: vec![span],
        };
        let features = vec![
            matched("é", -1.0, [12, 13]),
            matched("b", 0.0, [5, 6]),
            matched("aa", -1.0, [2, 4]),
            matched("z", 2.0, [7, 8]),
            matched("zz", 1.0, [9, 11]),
            matched("a", 1.0, [0, 1]),
        ];
        // Hand-written R4 oracle: magnitude first, then unsigned UTF-8 bytes,
        // including a proper prefix and an ASCII/non-ASCII tie.
        let expected = vec![
            matched("z", 2.0, [7, 8]),
            matched("a", 1.0, [0, 1]),
            matched("aa", -1.0, [2, 4]),
            matched("zz", 1.0, [9, 11]),
            matched("é", -1.0, [12, 13]),
            matched("b", 0.0, [5, 6]),
        ];
        let mut reversed = features.clone();
        reversed.reverse();
        for input in [features, reversed] {
            let record = scan_with(
                "a aa b z zz é",
                &config(),
                false,
                &TiedModel { features: input },
                &L0Normalizer,
                model_identity(),
                -0.5,
            )
            .unwrap();
            assert_eq!(record.attribution_status, AttributionStatus::Present);
            assert_eq!(record.feature_count_matched, Some(6));
            assert_eq!(
                serde_json::to_value(&record.top_features).unwrap(),
                serde_json::to_value(&expected).unwrap()
            );
        }
    }
    #[test]
    fn model_digest_preimage_vector() {
        // Constant calculated independently using Python struct.pack('>II'),
        // struct.pack('>d'), UTF-8 byte lengths and hashlib.sha256.
        let entries = [("é", -0.5), (" a", 0.25)];
        let cal = CalibrationParams { a: 0.6, b: 0.0 };
        let expected = "d6d5b60899a012ecb956ecc5e9e47049086be7a3922a361384de299ee7779026";
        assert_eq!(
            digest_model("char_wb", (3, 5), -0.75, cal, &entries),
            expected
        );
        assert_eq!(
            digest_model("char_wb", (3, 5), -0.75, cal, &[entries[1], entries[0]]),
            expected
        );
        for changed in [
            digest_model("char", (3, 5), -0.75, cal, &entries),
            digest_model("char_wb", (2, 5), -0.75, cal, &entries),
            digest_model("char_wb", (3, 6), -0.75, cal, &entries),
            digest_model("char_wb", (3, 5), -0.5, cal, &entries),
            digest_model(
                "char_wb",
                (3, 5),
                -0.75,
                CalibrationParams { a: 0.7, b: 0.0 },
                &entries,
            ),
            digest_model(
                "char_wb",
                (3, 5),
                -0.75,
                CalibrationParams { a: 0.6, b: 0.1 },
                &entries,
            ),
            digest_model(
                "char_wb",
                (3, 5),
                -0.75,
                cal,
                &[("é", f64::from_bits((-0.5_f64).to_bits() + 1)), entries[1]],
            ),
        ] {
            assert_ne!(changed, expected);
        }
    }
}
