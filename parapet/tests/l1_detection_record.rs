// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0
use parapet::layers::l1_detection::*;
use parapet::layers::l1_harness::{AttributedFeature, L1Signal, OutcomeView};
use parapet::message::Role;

fn golden() -> L1DetectionRecord {
    let text_digest = "11fb682be0a0233d5fb899721ecfc1827d20f0d2ff2e093310efa61efca8af1c";
    L1DetectionRecord {
        schema_version: "l1-detection-record/v0".into(),
        model_id: "l1-generalist-2026-04-17".into(),
        model_digest: "fb01f1688ca0559c15f8c9aa3fc22837fe77379613568b2c077ad92defcab5f8".into(),
        code_version: "0.1.0".into(),
        mode: RecordMode::Shadow,
        threshold: 0.0,
        mention_delta_threshold: 1.0,
        source_digest: text_digest.into(),
        source_len_bytes: 2,
        normalization: "l0:nfkc,html_strip,invisible_strip,confusable_fix,role_markers".into(),
        input_digest: text_digest.into(),
        input_len_bytes: 2,
        signal: L1Signal {
            message_index: 0,
            role: Role::User,
            raw_score: -0.09356798999999999,
            raw_unquoted_score: -0.09356798999999999,
            raw_squash_score: -0.48029451,
            score: 0.1_f32,
            unquoted_score: 0.1_f32,
            squash_score: 0.25_f32,
            quote_detected: false,
            raw_score_delta: 0.0,
        },
        outcome_view: OutcomeView::Raw,
        effective_raw: -0.09356798999999999,
        detection_outcome: DetectionOutcome::ThresholdNotBreached,
        view_text_digest: text_digest.into(),
        view_text_len_scalars: 2,
        view_text_lower_len_scalars: 2,
        attribution_status: AttributionStatus::Present,
        top_k: 10,
        top_features: vec![
            AttributedFeature {
                feature: " ai".into(),
                weight: 0.21518782,
                contribution: 0.21518782,
                occurrences: 1,
                spans: vec![[0, 2]],
            },
            AttributedFeature {
                feature: " бе".into(),
                weight: -0.125,
                contribution: -0.125,
                occurrences: 2,
                spans: vec![[0, 2], [3, 5]],
            },
        ],
        feature_count_matched: Some(2),
        bias: -0.48029451,
        matched_weight_sum_positive: Some(0.21518782),
        matched_weight_sum_negative: Some(-0.125),
    }
}

#[test]
fn record_digest_vector() {
    let record = golden();
    let actual = record_bytes(&record).unwrap();
    let expected = include_bytes!("fixtures/a1/A1_golden_record_v0.json");
    assert_eq!(
        String::from_utf8_lossy(&actual),
        String::from_utf8_lossy(expected)
    );
    assert_eq!(
        record_digest(&record).unwrap(),
        "8a79fc20cda6eee55c719881cdad69cb55ad53c83246201f42350a20be70b244"
    );
    for field in 0..26 {
        let mut changed = record.clone();
        mutate(&mut changed, field);
        assert_ne!(
            record_digest(&changed).unwrap(),
            record_digest(&record).unwrap(),
            "field {field}"
        );
    }
    assert_eq!(actual.len(), 1403);
    assert_ne!(actual.last(), Some(&b'\n'));
}

use parapet::config::{L1Config, L1Mode};
use parapet::layers::l1::SvmModel;
use parapet::layers::l1_harness::{
    strip_quotes, threshold_outcome, L1Harness, L1Model, MENTION_RAW_DELTA_THRESHOLD,
};
use parapet::message::{Message, TrustLevel};
use parapet::normalize::{neutralize_role_markers, L0Normalizer, Normalizer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
#[path = "../src/layers/l1_weights.rs"]
mod weights;
fn cfg() -> L1Config {
    L1Config {
        mode: L1Mode::Shadow,
        threshold: 0.0,
        min_agree: 1,
        generalist_solo_threshold: None,
        specialists: HashMap::new(),
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn fixtures() -> Vec<Value> {
    include_str!("fixtures/a1/demo_cases.jsonl")
        .lines()
        .chain(include_str!("fixtures/a1/workflow_twin.jsonl").lines())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn anchors(name: &str) -> Vec<Value> {
    let bytes = match name {
        "baseline" => include_str!("fixtures/a1/a1_baseline_main_88fd3e8.jsonl"),
        _ => include_str!("fixtures/a1/0B_L1_measurements.jsonl"),
    };
    // Keep float lexemes as strings, then use Rust's correctly rounded parse.
    // serde_json's default fast float parser is not a bit-preserving oracle.
    let re = regex::Regex::new(r#"("(?:bias|matched_weight_sum_positive|matched_weight_sum_negative|weight|contribution|listed_positive_inside_fraction)"\s*:\s*)(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"#).unwrap();
    bytes
        .lines()
        .map(|line| serde_json::from_str(&re.replace_all(line, "$1\"$2\"")).unwrap())
        .collect()
}
fn number(v: &Value) -> f64 {
    v.as_str()
        .map(|s| s.parse().unwrap())
        .unwrap_or_else(|| v.as_f64().unwrap())
}
fn normalized(text: &str, normalize: bool) -> Message {
    let mut msg = Message::new(Role::User, text);
    msg.trust = TrustLevel::Untrusted;
    if normalize {
        msg.content = L0Normalizer.normalize(text);
        neutralize_role_markers(std::slice::from_mut(&mut msg));
    }
    msg
}
fn view(text: &str, normalize: bool) -> String {
    let msg = normalized(text, normalize);
    let s = L1Harness::scan(std::slice::from_ref(&msg), &SvmModel).remove(0);
    if s.quote_detected && s.raw_score_delta > 1.0 {
        strip_quotes(&msg.content).unwrap()
    } else {
        msg.content
    }
}
fn rows() -> Vec<(Value, L1DetectionRecord)> {
    fixtures()
        .into_iter()
        .map(|f| {
            let r = scan_text(f["text"].as_str().unwrap(), &cfg(), true).unwrap();
            (f, r)
        })
        .collect()
}
fn bits(a: f64, b: f64) {
    assert_eq!(a.to_bits(), b.to_bits(), "{a:?} != {b:?}");
}

// Independent feature-against-text oracle. It never creates extractor windows:
// locate the unpadded feature substring and check the word-boundary predicates.
fn positions(text: &str, feature: &str) -> Vec<[usize; 2]> {
    let chars: Vec<_> = text.to_lowercase().chars().collect();
    let needle: Vec<_> = feature.trim_matches(' ').chars().collect();
    if needle.is_empty() || needle.len() > chars.len() {
        return vec![];
    }
    let mut spans = Vec::new();
    for start in 0..=chars.len() - needle.len() {
        let end = start + needle.len();
        if chars[start..end] == needle
            && !chars[start..end].iter().any(|c| c.is_whitespace())
            && (!feature.starts_with(' ') || start == 0 || chars[start - 1].is_whitespace())
            && (!feature.ends_with(' ') || end == chars.len() || chars[end].is_whitespace())
        {
            spans.push([start, end]);
        }
    }
    spans
}
fn all_matches(text: &str) -> Vec<AttributedFeature> {
    let mut all: Vec<_> = weights::WEIGHTS
        .entries()
        .filter_map(|(&feature, &weight)| {
            let spans = positions(text, feature);
            if spans.is_empty() {
                None
            } else {
                Some(AttributedFeature {
                    feature: feature.into(),
                    weight,
                    contribution: weight,
                    occurrences: spans.len(),
                    spans,
                })
            }
        })
        .collect();
    all.sort_by(|a, b| {
        b.weight
            .abs()
            .partial_cmp(&a.weight.abs())
            .unwrap()
            .then(a.feature.as_bytes().cmp(b.feature.as_bytes()))
    });
    all
}
fn sums(all: &[AttributedFeature]) -> (f64, f64) {
    let (mut pos, mut neg) = (0.0_f64, 0.0_f64);
    for f in all {
        if f.weight > 0.0 {
            pos += f.weight;
        }
        if f.weight < 0.0 {
            neg += f.weight;
        }
    }
    (pos, neg)
}
fn same_features(a: &[AttributedFeature], b: &[AttributedFeature]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.feature == b.feature
                && a.weight.to_bits() == b.weight.to_bits()
                && a.contribution.to_bits() == b.contribution.to_bits()
                && a.occurrences == b.occurrences
                && a.spans == b.spans
        })
}
fn truthful(text: &str, features: &[AttributedFeature]) -> bool {
    features
        .iter()
        .all(|f| f.spans == positions(text, &f.feature) && f.occurrences == f.spans.len())
}

#[test]
fn baseline_reproduction() {
    for ((f, r), b) in rows().iter().zip(anchors("baseline")) {
        assert_eq!(f["id"], b["id"]);
        for (key, value) in [
            ("raw_score", r.signal.raw_score),
            ("raw_unquoted_score", r.signal.raw_unquoted_score),
            ("raw_squash_score", r.signal.raw_squash_score),
            ("raw_score_delta", r.signal.raw_score_delta),
            ("effective_raw", r.effective_raw),
        ] {
            assert_eq!(
                format!("{:016x}", value.to_bits()),
                b[key]["bits"].as_str().unwrap(),
                "{} {key}",
                f["id"]
            );
        }
        assert_eq!(json!(r.signal.quote_detected), b["quote_detected"]);
        assert_eq!(json!(r.outcome_view), b["outcome_view"]);
        assert_eq!(json!(r.detection_outcome), b["detection_outcome"]);
    }
}
#[test]
fn top_features_exact_prefix() {
    for (f, r) in rows() {
        let all = all_matches(&view(f["text"].as_str().unwrap(), true));
        assert!(
            same_features(&r.top_features, &all[..10.min(all.len())]),
            "{}",
            f["id"]
        );
        let measured = anchors("measurement")
            .into_iter()
            .find(|m| m["id"] == f["id"] && m["normalize"] == true)
            .unwrap();
        let expected: Vec<_> = measured["top_features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| AttributedFeature {
                feature: v["feature"].as_str().unwrap().into(),
                weight: number(&v["weight"]),
                contribution: number(&v["contribution"]),
                occurrences: v["occurrences"].as_u64().unwrap() as usize,
                spans: serde_json::from_value(v["spans"].clone()).unwrap(),
            })
            .collect();
        assert!(
            same_features(&r.top_features, &expected),
            "measurement {}",
            f["id"]
        );
    }
}
#[test]
fn count_and_sums_all_rows() {
    for (f, r) in rows() {
        let all = all_matches(&view(f["text"].as_str().unwrap(), true));
        let (pos, neg) = sums(&all);
        assert_eq!(r.feature_count_matched, Some(all.len()));
        bits(r.matched_weight_sum_positive.unwrap(), pos);
        bits(r.matched_weight_sum_negative.unwrap(), neg);
        let m = anchors("measurement")
            .into_iter()
            .find(|m| m["id"] == f["id"] && m["normalize"] == true)
            .unwrap();
        assert_eq!(json!(r.feature_count_matched), m["feature_count_matched"]);
        bits(pos, number(&m["matched_weight_sum_positive"]));
        bits(neg, number(&m["matched_weight_sum_negative"]));
    }
}
#[test]
fn span_truth_all_rows() {
    for (f, r) in rows() {
        assert!(truthful(
            &view(f["text"].as_str().unwrap(), true),
            &r.top_features
        ));
    }
}
#[test]
fn span_truth_rejects_shifted() {
    let mut r = scan_text("AI AI", &cfg(), false).unwrap();
    r.top_features[0].spans[0] = [1, 3];
    assert!(!truthful("AI AI", &r.top_features));
}
#[test]
fn span_truth_rejects_missing() {
    let mut r = scan_text("AI AI", &cfg(), false).unwrap();
    r.top_features[0].spans.pop();
    r.top_features[0].occurrences -= 1;
    assert!(!truthful("AI AI", &r.top_features));
}
#[test]
fn selection_rejects_omitted_feature() {
    let (f, mut r) = rows().remove(0);
    r.top_features.remove(0);
    let all = all_matches(&view(f["text"].as_str().unwrap(), true));
    assert!(!same_features(&r.top_features, &all[..10.min(all.len())]));
}
#[test]
fn attribution_present_on_fixture() {
    for (_, r) in rows() {
        assert_eq!(r.attribution_status, AttributionStatus::Present);
    }
}
#[test]
fn short_word_single_occurrence() {
    let r = scan_text("AI", &cfg(), false).unwrap();
    assert_eq!(
        r.top_features
            .iter()
            .map(|f| f.feature.as_str())
            .collect::<Vec<_>>(),
        vec![" ai", " ai "]
    );
    for f in r.top_features {
        assert_eq!(f.spans, vec![[0, 2]]);
        assert_eq!(f.occurrences, 1);
    }
}
#[test]
fn repeated_word_spans() {
    let a = scan_text("AI", &cfg(), false).unwrap();
    let b = scan_text("AI AI", &cfg(), false).unwrap();
    bits(a.effective_raw, b.effective_raw);
    assert_eq!(a.top_features.len(), b.top_features.len());
    for (a, b) in a.top_features.iter().zip(&b.top_features) {
        assert_eq!(a.feature, b.feature);
        assert_eq!(b.spans, vec![[0, 2], [3, 5]]);
        assert_eq!(b.occurrences, 2);
    }
}
#[test]
fn matched_sums_cover_all_matches() {
    let (f, r) = rows().remove(0);
    let all = all_matches(&view(f["text"].as_str().unwrap(), true));
    assert!(all.len() > 10);
    assert_ne!(sums(&r.top_features), sums(&all));
    assert_eq!(
        (
            r.matched_weight_sum_positive.unwrap(),
            r.matched_weight_sum_negative.unwrap()
        ),
        sums(&all)
    );
}

fn expected_record(f: &Value, c: &L1Config, normalize: bool) -> L1DetectionRecord {
    let source = f["text"].as_str().unwrap();
    let msg = normalized(source, normalize);
    let signal = L1Harness::scan(std::slice::from_ref(&msg), &SvmModel).remove(0);
    let mention = signal.quote_detected && signal.raw_score_delta > MENTION_RAW_DELTA_THRESHOLD;
    let effective_raw = if mention {
        signal.raw_unquoted_score
    } else {
        signal.raw_score
    };
    let text = if mention {
        strip_quotes(&msg.content).unwrap()
    } else {
        msg.content.clone()
    };
    let mut all = all_matches(&text);
    let count = all.len();
    let (pos, neg) = sums(&all);
    all.truncate(10);
    let id = model_identity();
    L1DetectionRecord {
        schema_version: id.schema_version,
        model_id: id.model_id,
        model_digest: id.model_digest,
        code_version: env!("CARGO_PKG_VERSION").into(),
        mode: if c.mode == L1Mode::Shadow {
            RecordMode::Shadow
        } else {
            RecordMode::Block
        },
        threshold: c.threshold,
        mention_delta_threshold: MENTION_RAW_DELTA_THRESHOLD,
        source_digest: f["raw_sha256"].as_str().unwrap().into(),
        source_len_bytes: source.len(),
        normalization: if normalize {
            "l0:nfkc,html_strip,invisible_strip,confusable_fix,role_markers"
        } else {
            "none"
        }
        .into(),
        input_digest: hash(msg.content.as_bytes()),
        input_len_bytes: msg.content.len(),
        signal,
        outcome_view: if mention {
            OutcomeView::Unquoted
        } else {
            OutcomeView::Raw
        },
        effective_raw,
        detection_outcome: if effective_raw >= c.threshold {
            DetectionOutcome::ThresholdBreached
        } else {
            DetectionOutcome::ThresholdNotBreached
        },
        view_text_digest: hash(text.as_bytes()),
        view_text_len_scalars: text.chars().count(),
        view_text_lower_len_scalars: text.to_lowercase().chars().count(),
        attribution_status: AttributionStatus::Present,
        top_k: 10,
        top_features: all,
        feature_count_matched: Some(count),
        bias: weights::BIAS,
        matched_weight_sum_positive: Some(pos),
        matched_weight_sum_negative: Some(neg),
    }
}
fn float_bits(r: &L1DetectionRecord) -> Vec<u64> {
    let s = &r.signal;
    let mut b = vec![
        r.threshold.to_bits(),
        r.mention_delta_threshold.to_bits(),
        r.effective_raw.to_bits(),
        r.bias.to_bits(),
        s.raw_score.to_bits(),
        s.raw_unquoted_score.to_bits(),
        s.raw_squash_score.to_bits(),
        s.raw_score_delta.to_bits(),
        s.score.to_bits() as u64,
        s.unquoted_score.to_bits() as u64,
        s.squash_score.to_bits() as u64,
    ];
    b.extend(
        [r.matched_weight_sum_positive, r.matched_weight_sum_negative]
            .iter()
            .flatten()
            .map(|x| x.to_bits()),
    );
    for f in &r.top_features {
        b.extend([f.weight.to_bits(), f.contribution.to_bits()]);
    }
    b
}
fn oracle_accepts(r: &L1DetectionRecord, expected: &L1DetectionRecord) -> bool {
    serde_json::to_value(r).unwrap() == serde_json::to_value(expected).unwrap()
        && float_bits(r) == float_bits(expected)
}
fn mutate(r: &mut L1DetectionRecord, field: usize) {
    match field {
        0 => r.schema_version.push('x'),
        1 => r.model_id.push('x'),
        2 => r.model_digest.push('x'),
        3 => r.code_version.push('x'),
        4 => r.mode = RecordMode::Block,
        5 => r.threshold += 0.25,
        6 => r.mention_delta_threshold += 0.25,
        7 => r.source_digest.push('x'),
        8 => r.source_len_bytes += 1,
        9 => r.normalization.push('x'),
        10 => r.input_digest.push('x'),
        11 => r.input_len_bytes += 1,
        12 => r.signal.message_index += 1,
        13 => r.outcome_view = OutcomeView::Squashed,
        14 => r.effective_raw += 0.25,
        15 => r.detection_outcome = DetectionOutcome::ThresholdBreached,
        16 => r.view_text_digest.push('x'),
        17 => r.view_text_len_scalars += 1,
        18 => r.view_text_lower_len_scalars += 1,
        19 => r.attribution_status = AttributionStatus::Errored,
        20 => r.top_k += 1,
        21 => r.top_features[0].feature.push('x'),
        22 => r.feature_count_matched = None,
        23 => r.bias += 0.25,
        24 => r.matched_weight_sum_positive = None,
        25 => r.matched_weight_sum_negative = None,
        _ => panic!("field"),
    }
}
#[test]
fn record_field_oracle() {
    for f in fixtures() {
        for normalize in [true, false] {
            if !normalize && !f["id"].as_str().unwrap().starts_with("wf_") {
                continue;
            }
            let r = scan_text(f["text"].as_str().unwrap(), &cfg(), normalize).unwrap();
            let expected = expected_record(&f, &cfg(), normalize);
            assert!(
                oracle_accepts(&r, &expected),
                "{} normalize={normalize}",
                f["id"]
            );
            assert_eq!(r.code_version, model_identity().code_version);
            let signal = serde_json::to_value(&r.signal).unwrap();
            let mut keys: Vec<_> = signal
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort();
            assert_eq!(
                keys,
                vec![
                    "message_index",
                    "quote_detected",
                    "raw_score",
                    "raw_score_delta",
                    "raw_squash_score",
                    "raw_unquoted_score",
                    "role",
                    "score",
                    "squash_score",
                    "unquoted_score"
                ]
            );
            let m = anchors("measurement")
                .into_iter()
                .find(|m| m["id"] == f["id"] && m["normalize"] == normalize)
                .unwrap();
            assert_eq!(
                view(f["text"].as_str().unwrap(), normalize),
                m["view_text"].as_str().unwrap()
            );
            assert_eq!(json!(r.view_text_len_scalars), m["view_len"]);
            assert_eq!(json!(r.view_text_lower_len_scalars), m["lower_len"]);
            for (key, value) in [
                ("raw_score", r.signal.raw_score),
                ("raw_unquoted_score", r.signal.raw_unquoted_score),
                ("raw_squash_score", r.signal.raw_squash_score),
                ("raw_score_delta", r.signal.raw_score_delta),
            ] {
                assert_eq!(
                    format!("{:016x}", value.to_bits()),
                    m[key]["bits"].as_str().unwrap(),
                    "{} {key}",
                    f["id"]
                );
            }
            assert_eq!(json!(r.signal.quote_detected), m["quote_detected"]);
            assert_eq!(json!(r.outcome_view), m["outcome_view"]);
            assert_eq!(
                r.top_features.len(),
                m["top_features"].as_array().unwrap().len()
            );
            for (feature, measured) in r
                .top_features
                .iter()
                .zip(m["top_features"].as_array().unwrap())
            {
                assert_eq!(feature.feature, measured["feature"].as_str().unwrap());
                bits(feature.weight, number(&measured["weight"]));
                bits(feature.contribution, number(&measured["contribution"]));
                assert_eq!(json!(feature.spans), measured["spans"]);
                assert_eq!(json!(feature.occurrences), measured["occurrences"]);
            }
            // Reproduce the normalization-off measurement rows too.
            assert_eq!(
                format!("{:016x}", r.effective_raw.to_bits()),
                m["effective_raw"]["bits"].as_str().unwrap()
            );
            assert_eq!(json!(r.feature_count_matched), m["feature_count_matched"]);
            bits(
                r.matched_weight_sum_positive.unwrap(),
                number(&m["matched_weight_sum_positive"]),
            );
            bits(
                r.matched_weight_sum_negative.unwrap(),
                number(&m["matched_weight_sum_negative"]),
            );
        }
    }
}
#[test]
fn record_echoes_config() {
    let mut c = cfg();
    c.mode = L1Mode::Block;
    c.threshold = 0.25;
    for f in fixtures() {
        let r = scan_text(f["text"].as_str().unwrap(), &c, true).unwrap();
        assert!(oracle_accepts(&r, &expected_record(&f, &c, true)));
    }
}
#[test]
fn oracle_rejects_single_field_mutation() {
    let f = fixtures().remove(0);
    let r = scan_text(f["text"].as_str().unwrap(), &cfg(), true).unwrap();
    let e = expected_record(&f, &cfg(), true);
    assert!(oracle_accepts(&r, &e));
    for field in 0..26 {
        let mut bad = r.clone();
        mutate(&mut bad, field);
        assert!(!oracle_accepts(&bad, &e), "field {field}");
    }
}
#[test]
fn record_schema_conformance() {
    let names = [
        "schema_version",
        "model_id",
        "model_digest",
        "code_version",
        "mode",
        "threshold",
        "mention_delta_threshold",
        "source_digest",
        "source_len_bytes",
        "normalization",
        "input_digest",
        "input_len_bytes",
        "signal",
        "outcome_view",
        "effective_raw",
        "detection_outcome",
        "view_text_digest",
        "view_text_len_scalars",
        "view_text_lower_len_scalars",
        "attribution_status",
        "top_k",
        "top_features",
        "feature_count_matched",
        "bias",
        "matched_weight_sum_positive",
        "matched_weight_sum_negative",
    ];
    for (_, r) in rows() {
        let value = serde_json::to_value(&r).unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.len(), 26);
        for name in names {
            assert!(obj.contains_key(name));
        }
        assert_eq!(r.schema_version, "l1-detection-record/v0");
        assert!(["shadow", "block"].contains(&value["mode"].as_str().unwrap()));
        assert!(["raw", "unquoted", "squashed"].contains(&value["outcome_view"].as_str().unwrap()));
        assert!(["threshold_breached", "threshold_not_breached"]
            .contains(&value["detection_outcome"].as_str().unwrap()));
        assert_eq!(r.attribution_status, AttributionStatus::Present);
        for name in [
            "feature_count_matched",
            "matched_weight_sum_positive",
            "matched_weight_sum_negative",
        ] {
            assert!(!value[name].is_null());
        }
    }
}
#[test]
fn record_matches_engine_rule() {
    for (_, r) in rows() {
        let o = threshold_outcome(&r.signal, r.threshold);
        assert_eq!(o.outcome_view, r.outcome_view);
        bits(o.effective_raw, r.effective_raw);
        assert_eq!(
            o.threshold_breached,
            r.detection_outcome == DetectionOutcome::ThresholdBreached
        );
    }
}
#[test]
fn mention_dampening_recorded() {
    let (_, r) = rows().into_iter().find(|(f, _)| f["id"] == "b05").unwrap();
    assert_eq!(r.outcome_view, OutcomeView::Unquoted);
    assert!(r.signal.quote_detected);
    assert!(r.signal.raw_score_delta > r.mention_delta_threshold);
    assert!(r.signal.raw_score > r.threshold);
    assert_eq!(r.detection_outcome, DetectionOutcome::ThresholdNotBreached);
}
#[test]
fn workflow_twin_shape() {
    let rs = rows();
    let a = &rs.iter().find(|(f, _)| f["id"] == "wf_ben").unwrap().1;
    let b = &rs.iter().find(|(f, _)| f["id"] == "wf_inj").unwrap().1;
    let av = serde_json::to_value(a).unwrap();
    let bv = serde_json::to_value(b).unwrap();
    assert_eq!(
        av.as_object().unwrap().keys().collect::<Vec<_>>(),
        bv.as_object().unwrap().keys().collect::<Vec<_>>()
    );
    assert_ne!(av, bv);
    assert_eq!(a.detection_outcome, DetectionOutcome::ThresholdNotBreached);
    assert_eq!(b.detection_outcome, DetectionOutcome::ThresholdBreached);
}
#[test]
fn view_rule_boundaries() {
    let mut s = golden().signal;
    s.quote_detected = true;
    s.raw_score = 2.0;
    s.raw_unquoted_score = -2.0;
    for (delta, view) in [
        (f64::from_bits(1.0_f64.to_bits() - 1), OutcomeView::Raw),
        (1.0, OutcomeView::Raw),
        (f64::from_bits(1.0_f64.to_bits() + 1), OutcomeView::Unquoted),
    ] {
        s.raw_score_delta = delta;
        assert_eq!(threshold_outcome(&s, 0.25).outcome_view, view);
    }
    s.quote_detected = false;
    s.raw_score_delta = 2.0;
    assert_eq!(threshold_outcome(&s, 0.25).outcome_view, OutcomeView::Raw);
}
#[test]
fn threshold_rule_boundaries() {
    let mut s = golden().signal;
    for (margin, breached) in [
        (f64::from_bits(0.25_f64.to_bits() - 1), false),
        (0.25, true),
        (f64::from_bits(0.25_f64.to_bits() + 1), true),
    ] {
        s.raw_score = margin;
        assert_eq!(threshold_outcome(&s, 0.25).threshold_breached, breached);
    }
}
fn assert_empty_present(r: &L1DetectionRecord) {
    assert_eq!(r.attribution_status, AttributionStatus::Present);
    assert!(r.top_features.is_empty());
    assert_eq!(r.feature_count_matched, Some(0));
    bits(r.matched_weight_sum_positive.unwrap(), 0.0);
    bits(r.matched_weight_sum_negative.unwrap(), 0.0);
}
#[test]
fn lowercase_length_change_recorded() {
    let r = scan_text("İ", &cfg(), true).unwrap();
    assert_eq!(r.view_text_len_scalars, 1);
    assert_eq!(r.view_text_lower_len_scalars, 2);
    assert_empty_present(&r);
}
#[test]
fn whitespace_only_is_present_empty() {
    let r = scan_text("   \t\n", &cfg(), true).unwrap();
    assert_empty_present(&r);
    assert_eq!(r.detection_outcome, DetectionOutcome::ThresholdNotBreached);
    let bytes = String::from_utf8(record_bytes(&r).unwrap()).unwrap();
    assert!(bytes.contains("\"matched_weight_sum_positive\":0.0"));
    assert!(bytes.contains("\"matched_weight_sum_negative\":0.0"));
}
#[test]
fn empty_after_normalization_errors() {
    for text in ["", "<!--x-->"] {
        assert!(matches!(
            scan_text(text, &cfg(), true),
            Err(ScanError::EmptyInput)
        ));
    }
}
#[test]
fn zero_match_breach_below_bias() {
    let mut c = cfg();
    let a = scan_text("☃", &c, true).unwrap();
    assert_empty_present(&a);
    bits(a.effective_raw, weights::BIAS);
    assert_eq!(a.detection_outcome, DetectionOutcome::ThresholdNotBreached);
    c.threshold = weights::BIAS - 1.0;
    let b = scan_text("☃", &c, true).unwrap();
    assert_empty_present(&b);
    assert_eq!(b.detection_outcome, DetectionOutcome::ThresholdBreached);
}
#[test]
fn l0_boundary_recorded() {
    let fs = fixtures();
    let ben = fs.iter().find(|f| f["id"] == "wf_ben").unwrap()["text"]
        .as_str()
        .unwrap();
    let comment = fs.iter().find(|f| f["id"] == "wf_inj_comment").unwrap()["text"]
        .as_str()
        .unwrap();
    let a = scan_text(ben, &cfg(), true).unwrap();
    let b = scan_text(comment, &cfg(), true).unwrap();
    let av: Vec<_> = normalized(ben, true).content.chars().collect();
    let bv: Vec<_> = normalized(comment, true).content.chars().collect();
    let start = av.iter().zip(&bv).position(|(x, y)| x != y).unwrap();
    let len = bv.len() - av.len();
    assert_eq!((start, len), (146, 2));
    assert!(bv[start..start + len].iter().all(|c| c.is_whitespace()));
    let mut stripped = bv.clone();
    stripped.drain(start..start + len);
    assert_eq!(av, stripped);
    assert_eq!(
        serde_json::to_value(&a.signal).unwrap(),
        serde_json::to_value(&b.signal).unwrap()
    );
    bits(a.effective_raw, b.effective_raw);
    assert_eq!(a.outcome_view, b.outcome_view);
    assert_eq!(a.detection_outcome, b.detection_outcome);
    assert_eq!(a.feature_count_matched, b.feature_count_matched);
    bits(
        a.matched_weight_sum_positive.unwrap(),
        b.matched_weight_sum_positive.unwrap(),
    );
    bits(
        a.matched_weight_sum_negative.unwrap(),
        b.matched_weight_sum_negative.unwrap(),
    );
    assert_eq!(a.top_features.len(), b.top_features.len());
    for (a, b) in a.top_features.iter().zip(&b.top_features) {
        assert_eq!(a.feature, b.feature);
        bits(a.weight, b.weight);
        bits(a.contribution, b.contribution);
        assert_eq!(a.occurrences, b.occurrences);
        assert_eq!(
            a.spans
                .iter()
                .map(|&[s, e]| if s >= start {
                    [s + len, e + len]
                } else {
                    [s, e]
                })
                .collect::<Vec<_>>(),
            b.spans
        );
    }
    assert_ne!(a.source_digest, b.source_digest);
    assert_ne!(a.source_len_bytes, b.source_len_bytes);
    assert_ne!(a.input_digest, b.input_digest);
    assert_eq!((a.input_len_bytes, b.input_len_bytes), (162, 164));
    assert!(a.normalization.contains("html_strip"));
    assert!(truthful(&view(ben, true), &a.top_features));
    assert!(truthful(&view(comment, true), &b.top_features));
    for (text, bits_expected, outcome) in [
        (
            ben,
            0xbfe9aa5cd03a957e,
            DetectionOutcome::ThresholdNotBreached,
        ),
        (
            comment,
            0x3fdf076c85e531c6,
            DetectionOutcome::ThresholdBreached,
        ),
    ] {
        let r = scan_text(text, &cfg(), false).unwrap();
        assert_eq!(r.effective_raw.to_bits(), bits_expected);
        assert_eq!(r.detection_outcome, outcome);
        assert_eq!(r.normalization, "none");
        assert_eq!(r.input_digest, r.source_digest);
    }
}

#[test]
fn model_digest_recomputes() {
    let mut bytes = b"parapet-l1-model/v1\nchar_wb\n".to_vec();
    bytes.extend(3_u32.to_be_bytes());
    bytes.extend(5_u32.to_be_bytes());
    bytes.extend(b"binary=1;lowercase=1\n");
    let cal = SvmModel.calibration();
    for x in [weights::BIAS, cal.a, cal.b] {
        bytes.extend(x.to_bits().to_be_bytes());
    }
    let mut entries: Vec<_> = weights::WEIGHTS.entries().collect();
    entries.sort_by_key(|(f, _)| f.as_bytes());
    bytes.extend((entries.len() as u32).to_be_bytes());
    for (feature, weight) in entries {
        bytes.extend((feature.len() as u32).to_be_bytes());
        bytes.extend(feature.as_bytes());
        bytes.extend(weight.to_bits().to_be_bytes());
    }
    let id = model_identity();
    assert_eq!(id.model_digest, hash(&bytes));
    assert_eq!(
        id.model_digest,
        "fb01f1688ca0559c15f8c9aa3fc22837fe77379613568b2c077ad92defcab5f8"
    );
    assert_eq!(id.model_id, "l1-generalist-2026-04-17");
}
fn ac7_fraction(f: &Value, r: &L1DetectionRecord) -> Option<f64> {
    if f["injected_span_chars"].is_null()
        || r.detection_outcome != DetectionOutcome::ThresholdBreached
        || r.outcome_view != OutcomeView::Raw
        || r.view_text_len_scalars != r.view_text_lower_len_scalars
    {
        return None;
    }
    let chars: Vec<_> = f["text"].as_str().unwrap().chars().collect();
    let span = &f["injected_span_chars"];
    let start = span[0].as_u64().unwrap() as usize;
    let end = span[1].as_u64().unwrap() as usize;
    let prefix = normalized(&chars[..start].iter().collect::<String>(), true).content;
    let interval = normalized(&chars[start..end].iter().collect::<String>(), true).content;
    let suffix = normalized(&chars[end..].iter().collect::<String>(), true).content;
    assert_eq!(
        format!("{prefix}{interval}{suffix}"),
        normalized(f["text"].as_str().unwrap(), true).content
    );
    let mapped_start = prefix.chars().count();
    let mapped_end = mapped_start + interval.chars().count();
    let measured = anchors("measurement")
        .into_iter()
        .find(|m| m["id"] == f["id"] && m["normalize"] == true)
        .unwrap();
    assert_eq!(json!([mapped_start, mapped_end]), measured["mapped_span"]);
    let (mut inside, mut total) = (0.0, 0.0);
    for feature in &r.top_features {
        if feature.weight > 0.0 {
            total += feature.weight;
            if feature
                .spans
                .iter()
                .any(|&[s, e]| s >= mapped_start && e <= mapped_end)
            {
                inside += feature.weight;
            }
        }
    }
    assert!(total > 0.0);
    Some(inside / total)
}
#[test]
fn known_signal_points_into_injection() {
    let mut subjects = Vec::new();
    for (f, r) in rows() {
        if let Some(fraction) = ac7_fraction(&f, &r) {
            subjects.push(f["id"].as_str().unwrap().to_owned());
            if f["id"] != "i04" {
                assert!(fraction >= 0.5);
            }
            let m = anchors("measurement")
                .into_iter()
                .find(|m| m["id"] == f["id"] && m["normalize"] == true)
                .unwrap();
            assert!((fraction - number(&m["listed_positive_inside_fraction"])).abs() < 1e-9);
        }
        if f["id"] == "b07" {
            assert_eq!(r.detection_outcome, DetectionOutcome::ThresholdBreached);
            assert!(ac7_fraction(&f, &r).is_none());
        }
    }
    assert_eq!(
        subjects,
        vec!["i01", "i02", "i04", "i07", "i08", "i09", "wf_inj"]
    );
}
#[test]
fn fixture_records() {
    let mut records = Vec::new();
    let mut outcomes = Vec::new();
    for (f, r) in rows() {
        records.extend(record_bytes(&r).unwrap());
        records.push(b'\n');
        outcomes.push(json!({"id":f["id"],"detection_outcome":r.detection_outcome,"outcome_view":r.outcome_view,
            "effective_raw_bits":format!("{:016x}",r.effective_raw.to_bits()),"ac7_fraction":ac7_fraction(&f,&r),
            "benign_false_alarm":f["id"]=="b07","record_digest":record_digest(&r).unwrap()}));
    }
    assert_eq!(outcomes.len(), 23);
    // Artifact export is opt-in; the caller supplies an existing evidence dir.
    if let Ok(dir) = std::env::var("PARAPET_A1_EVIDENCE_DIR") {
        std::fs::write(std::path::Path::new(&dir).join("records.jsonl"), records).unwrap();
        std::fs::write(
            std::path::Path::new(&dir).join("first_run_outcomes.json"),
            serde_json::to_vec_pretty(&outcomes).unwrap(),
        )
        .unwrap();
    }
}
/// A second serializer is not covered by the digest contract. Consumers must
/// retain the library bytes and never reconstruct a digest from a JSON map.
#[test]
fn digest_requires_library_bytes() {
    let r = golden();
    let bytes = record_bytes(&r).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let _not_required_to_match = serde_json::to_vec(&value).unwrap();
    let mut preimage = b"parapet-l1-detection-record-digest/v0\n".to_vec();
    preimage.extend(&bytes);
    assert_eq!(hash(&preimage), record_digest(&r).unwrap());
    let mut spliced = value;
    spliced["id"] = json!("caller-id");
    let mut other = b"parapet-l1-detection-record-digest/v0\n".to_vec();
    other.extend(serde_json::to_vec(&spliced).unwrap());
    assert_ne!(hash(&other), record_digest(&r).unwrap());
    for field in 0..26 {
        let mut changed = r.clone();
        mutate(&mut changed, field);
        assert_ne!(
            record_digest(&changed).unwrap(),
            record_digest(&r).unwrap(),
            "field {field}"
        );
    }
}
#[test]
fn non_finite_leaf_refuses_bytes() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for leaf in 0..17 {
            let mut r = golden();
            match leaf {
                0 => r.threshold = bad,
                1 => r.mention_delta_threshold = bad,
                2 => r.effective_raw = bad,
                3 => r.bias = bad,
                4 => r.matched_weight_sum_positive = Some(bad),
                5 => r.matched_weight_sum_negative = Some(bad),
                6 => r.signal.raw_score = bad,
                7 => r.signal.raw_unquoted_score = bad,
                8 => r.signal.raw_squash_score = bad,
                9 => r.signal.raw_score_delta = bad,
                10 => r.signal.score = bad as f32,
                11 => r.signal.unquoted_score = bad as f32,
                12 => r.signal.squash_score = bad as f32,
                13 => r.top_features[0].weight = bad,
                14 => r.top_features[0].contribution = bad,
                15 => r.top_features[1].weight = bad,
                16 => r.top_features[1].contribution = bad,
                _ => unreachable!(),
            }
            assert!(
                matches!(record_bytes(&r), Err(RecordError::NonFinite)),
                "leaf {leaf}"
            );
            assert!(
                matches!(record_digest(&r), Err(RecordError::NonFinite)),
                "leaf {leaf}"
            );
        }
    }
}
// Extract ordinary Rust string literals in the existing test modules, including
// escapes, so AC1 follows new literal test inputs without maintaining a copy.
fn existing_literals(source: &str) -> Vec<String> {
    let source = source.split_once("#[cfg(test)]").unwrap().1;
    let mut literals = Vec::new();
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' && chars.peek() == Some(&'/') {
            for x in chars.by_ref() {
                if x == '\n' {
                    break;
                }
            }
            continue;
        }
        if c != '"' {
            continue;
        }
        let mut text = String::new();
        while let Some(c) = chars.next() {
            if c == '"' {
                break;
            }
            if c != '\\' {
                text.push(c);
                continue;
            }
            match chars.next().unwrap() {
                'n' => text.push('\n'),
                'r' => text.push('\r'),
                't' => text.push('\t'),
                '0' => text.push('\0'),
                '\\' => text.push('\\'),
                '"' => text.push('"'),
                '\'' => text.push('\''),
                'u' => {
                    assert_eq!(chars.next(), Some('{'));
                    let mut hex = String::new();
                    for c in chars.by_ref() {
                        if c == '}' {
                            break;
                        }
                        hex.push(c);
                    }
                    text.push(
                        char::from_u32(u32::from_str_radix(&hex.replace('_', ""), 16).unwrap())
                            .unwrap(),
                    );
                }
                '\n' => {
                    while chars.peek().is_some_and(|c| c.is_whitespace()) {
                        chars.next();
                    }
                }
                c => panic!("unsupported Rust string escape {c}"),
            }
        }
        literals.push(text);
    }
    literals
}
#[test]
fn attribution_margin_equivalence() {
    let mut inputs: Vec<_> = fixtures()
        .iter()
        .map(|f| view(f["text"].as_str().unwrap(), true))
        .collect();
    let mut literals = existing_literals(include_str!("../src/layers/l1.rs"));
    literals.extend(existing_literals(include_str!(
        "../src/layers/l1_harness.rs"
    )));
    assert!(literals.len() > 100);
    inputs.extend(literals);
    for text in inputs {
        let attributed = SvmModel.score_attributed(&text);
        bits(attributed.margin, SvmModel.score(&text));
        let features = attributed.features.unwrap();
        assert!(same_features(&features, &all_matches(&text)), "{text:?}");
        let (pos, neg) = sums(&features);
        assert!(
            (weights::BIAS + pos + neg - attributed.margin).abs() < 1e-9,
            "{text:?}"
        );
    }
}
