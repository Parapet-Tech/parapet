// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

use parapet::receipt::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn fixtures() -> Vec<(&'static str, &'static [u8])> {
    vec![
        (
            "policy.json",
            include_bytes!("fixtures/receipt_v0/policy.json"),
        ),
        (
            "base_proposal.json",
            include_bytes!("fixtures/receipt_v0/base_proposal.json"),
        ),
        (
            "base_proposal_context.json",
            include_bytes!("fixtures/receipt_v0/base_proposal_context.json"),
        ),
        (
            "base_decision.json",
            include_bytes!("fixtures/receipt_v0/base_decision.json"),
        ),
        (
            "base_attempt.json",
            include_bytes!("fixtures/receipt_v0/base_attempt.json"),
        ),
        (
            "base_observation.json",
            include_bytes!("fixtures/receipt_v0/base_observation.json"),
        ),
        (
            "base_verification.json",
            include_bytes!("fixtures/receipt_v0/base_verification.json"),
        ),
        (
            "base_row.json",
            include_bytes!("fixtures/receipt_v0/base_row.json"),
        ),
        (
            "arguments.json",
            include_bytes!("fixtures/receipt_v0/arguments.json"),
        ),
        (
            "F7_wire.json",
            include_bytes!("fixtures/receipt_v0/F7_wire.json"),
        ),
        (
            "F7_anthropic.json",
            include_bytes!("fixtures/receipt_v0/F7_anthropic.json"),
        ),
        (
            "F13_row.json",
            include_bytes!("fixtures/receipt_v0/F13_row.json"),
        ),
        (
            "F41_row.json",
            include_bytes!("fixtures/receipt_v0/F41_row.json"),
        ),
        (
            "F42_row.json",
            include_bytes!("fixtures/receipt_v0/F42_row.json"),
        ),
        (
            "F44_row.json",
            include_bytes!("fixtures/receipt_v0/F44_row.json"),
        ),
        (
            "F43_stream.json",
            include_bytes!("fixtures/receipt_v0/F43_stream.json"),
        ),
        (
            "F43_nonstream.json",
            include_bytes!("fixtures/receipt_v0/F43_nonstream.json"),
        ),
        (
            "F47_negative_zero.json",
            include_bytes!("fixtures/receipt_v0/F47_negative_zero.json"),
        ),
        (
            "F47_float_one.json",
            include_bytes!("fixtures/receipt_v0/F47_float_one.json"),
        ),
        (
            "F47_integer_one.json",
            include_bytes!("fixtures/receipt_v0/F47_integer_one.json"),
        ),
        (
            "F47_wire.json",
            include_bytes!("fixtures/receipt_v0/F47_wire.json"),
        ),
        (
            "F1_row.json",
            include_bytes!("fixtures/receipt_v0/F1_row.json"),
        ),
        (
            "F10_row.json",
            include_bytes!("fixtures/receipt_v0/F10_row.json"),
        ),
        (
            "F5_proposal.json",
            include_bytes!("fixtures/receipt_v0/F5_proposal.json"),
        ),
        (
            "F9_proposal.json",
            include_bytes!("fixtures/receipt_v0/F9_proposal.json"),
        ),
        (
            "F2_row.json",
            include_bytes!("fixtures/receipt_v0/F2_row.json"),
        ),
        (
            "F3_row.json",
            include_bytes!("fixtures/receipt_v0/F3_row.json"),
        ),
        (
            "F14_row.json",
            include_bytes!("fixtures/receipt_v0/F14_row.json"),
        ),
        (
            "F29_space.json",
            include_bytes!("fixtures/receipt_v0/F29_space.json"),
        ),
        (
            "F29_order.json",
            include_bytes!("fixtures/receipt_v0/F29_order.json"),
        ),
        (
            "F31_proposal_id.json",
            include_bytes!("fixtures/receipt_v0/F31_proposal_id.json"),
        ),
        (
            "F31_action_id.json",
            include_bytes!("fixtures/receipt_v0/F31_action_id.json"),
        ),
        (
            "F38_wire.json",
            include_bytes!("fixtures/receipt_v0/F38_wire.json"),
        ),
        (
            "F38_form_0.json",
            include_bytes!("fixtures/receipt_v0/F38_form_0.json"),
        ),
        (
            "F38_form_1.json",
            include_bytes!("fixtures/receipt_v0/F38_form_1.json"),
        ),
        (
            "F38_form_2.json",
            include_bytes!("fixtures/receipt_v0/F38_form_2.json"),
        ),
        (
            "F38_form_3.json",
            include_bytes!("fixtures/receipt_v0/F38_form_3.json"),
        ),
        (
            "F38_form_4.json",
            include_bytes!("fixtures/receipt_v0/F38_form_4.json"),
        ),
        (
            "F38_mismatch_row.json",
            include_bytes!("fixtures/receipt_v0/F38_mismatch_row.json"),
        ),
        (
            "F57_observation_ref.json",
            include_bytes!("fixtures/receipt_v0/F57_observation_ref.json"),
        ),
        (
            "F57_proposed_path.json",
            include_bytes!("fixtures/receipt_v0/F57_proposed_path.json"),
        ),
        (
            "F57_target_ref.json",
            include_bytes!("fixtures/receipt_v0/F57_target_ref.json"),
        ),
        (
            "F57_resolved_only.json",
            include_bytes!("fixtures/receipt_v0/F57_resolved_only.json"),
        ),
        (
            "F57_reference_only.json",
            include_bytes!("fixtures/receipt_v0/F57_reference_only.json"),
        ),
        (
            "F57_identity.json",
            include_bytes!("fixtures/receipt_v0/F57_identity.json"),
        ),
        (
            "F58_row.json",
            include_bytes!("fixtures/receipt_v0/F58_row.json"),
        ),
        (
            "F58_tag_mismatch_row.json",
            include_bytes!("fixtures/receipt_v0/F58_tag_mismatch_row.json"),
        ),
        (
            "F74_detection.json",
            include_bytes!("fixtures/receipt_v0/F74_detection.json"),
        ),
        (
            "F74_row.json",
            include_bytes!("fixtures/receipt_v0/F74_row.json"),
        ),
    ]
}

fn bytes(name: &str) -> &'static [u8] {
    fixtures().into_iter().find(|(n, _)| *n == name).unwrap().1
}
fn value(name: &str) -> Value {
    serde_json::from_slice(bytes(name)).unwrap()
}
fn oracle_hash(tag: Option<&[u8]>, bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    if let Some(tag) = tag {
        hash.update(tag);
        hash.update([10]);
    }
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}
fn roundtrip(kind: &str, bytes: &[u8]) -> Result<Vec<u8>, serde_json::Error> {
    // Only types-and-bytes parsing: this is not a row validator.
    match kind {
        "proposal" => record_bytes(ReceiptRecord::Proposal(&serde_json::from_slice(bytes)?)),
        "proposal_context" => record_bytes(ReceiptRecord::ProposalContext(
            &serde_json::from_slice(bytes)?,
        )),
        "decision" => record_bytes(ReceiptRecord::Decision(&serde_json::from_slice(bytes)?)),
        "attempt" => record_bytes(ReceiptRecord::Attempt(&serde_json::from_slice(bytes)?)),
        "observation" => record_bytes(ReceiptRecord::Observation(&serde_json::from_slice(bytes)?)),
        "verification" => {
            record_bytes(ReceiptRecord::Verification(&serde_json::from_slice(bytes)?))
        }
        "row" => record_bytes(ReceiptRecord::Row(&serde_json::from_slice(bytes)?)),
        "arguments" => canonical_argument_bytes(&serde_json::from_slice(bytes)?),
        "policy" => Ok(canonical_policy_bytes(&serde_json::from_slice(bytes)?).unwrap()),
        _ => Ok(bytes.to_vec()), // Wire instruments, identity vectors and DR2 carried bytes.
    }
}

#[test]
fn receipt_fixture_roundtrip() {
    let oracle: Vec<Value> =
        serde_json::from_slice(include_bytes!("fixtures/receipt_v0/oracle.json")).unwrap();
    assert_eq!(oracle.len(), fixtures().len());
    for entry in oracle {
        let name = entry["file"].as_str().unwrap();
        let input = bytes(name);
        let result = roundtrip(entry["kind"].as_str().unwrap(), input);
        if !entry["parse"].as_bool().unwrap() {
            assert!(result.is_err(), "foreign shape parsed: {name}");
        } else if entry["canonical"].as_bool().unwrap() {
            assert_eq!(result.unwrap(), input, "{name}");
        } else {
            assert_ne!(result.unwrap(), input, "{name}");
        }
        assert_eq!(input.len() as u64, entry["len_bytes"].as_u64().unwrap());
        assert_eq!(
            payload_digest(input),
            entry["raw_digest"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            record_digest(input, RecordDomain::Stage),
            entry["stage_digest"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            record_digest(input, RecordDomain::Row),
            entry["row_digest"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            arguments_digest(input),
            entry["arguments_digest"].as_str().unwrap(),
            "{name}"
        );
        println!(
            "DIGEST {name} raw={} stage={} row={} arguments={}",
            payload_digest(input),
            record_digest(input, RecordDomain::Stage),
            record_digest(input, RecordDomain::Row),
            arguments_digest(input)
        );
    }
}

#[test]
fn receipt_er2_oracle() {
    for (_, input) in fixtures() {
        assert_eq!(
            record_digest(input, RecordDomain::Stage),
            oracle_hash(Some(b"parapet-effect-receipt-stage-digest/v0"), input)
        );
        assert_eq!(
            record_digest(input, RecordDomain::Row),
            oracle_hash(Some(b"parapet-effect-receipt-digest/v0"), input)
        );
        assert_ne!(
            record_digest(input, RecordDomain::Stage),
            record_digest(input, RecordDomain::Row)
        );
    }
    // Digests use carried bytes, including non-canonical whitespace.
    assert_ne!(
        record_digest(bytes("base_proposal.json"), RecordDomain::Stage),
        record_digest(bytes("F29_space.json"), RecordDomain::Stage)
    );
}

fn ordered(input: &[u8], keys: &[&str]) {
    let text = std::str::from_utf8(input).unwrap();
    let mut offset = 0;
    for key in keys {
        let needle = format!("\"{key}\":");
        let position = text[offset..].find(&needle).unwrap();
        offset += position + needle.len();
    }
    assert!(!input.starts_with(&[0xef, 0xbb, 0xbf]));
    assert_ne!(input.last(), Some(&b'\n'));
    assert_eq!(
        serde_json::from_slice::<Value>(input)
            .unwrap()
            .as_object()
            .unwrap()
            .len(),
        keys.len()
    );
}

#[test]
fn receipt_er1_table_order() {
    let tables: &[(&str, &str, &[&str])] = &[
        (
            "base_proposal.json",
            "proposal",
            &[
                "schema_version",
                "stage",
                "provider",
                "index",
                "tool_name",
                "parse_status",
                "parse_error_code",
                "arguments_digest",
                "arguments_len_bytes",
                "effect_kind",
                "expected_effect",
            ],
        ),
        (
            "base_proposal_context.json",
            "proposal_context",
            &[
                "schema_version",
                "stage",
                "proposal_ref",
                "request_ref",
                "choice_index",
                "model",
                "tool_call_id",
                "arguments_wire_form",
                "native_arguments",
                "arguments_payload",
                "detection_payload",
            ],
        ),
        (
            "base_decision.json",
            "decision",
            &[
                "schema_version",
                "stage",
                "proposal_ref",
                "policy",
                "code",
                "detection",
                "detection_binding_rule",
                "enforcement",
                "decision",
                "evaluated_policy",
                "reason_code",
                "failed_constraints",
                "inputs_consumed",
            ],
        ),
        (
            "base_attempt.json",
            "attempt",
            &[
                "schema_version",
                "stage",
                "decision_ref",
                "action_id",
                "executor",
                "root",
                "target",
                "operation",
                "sink",
                "policy_check",
                "before_state",
                "outcome",
                "error_code",
                "content_payload",
            ],
        ),
        (
            "base_observation.json",
            "observation",
            &[
                "schema_version",
                "stage",
                "attempt_ref",
                "observer",
                "target_ref",
                "observed",
                "footprint",
            ],
        ),
        (
            "base_verification.json",
            "verification",
            &[
                "schema_version",
                "stage",
                "observation_ref",
                "method",
                "outcome",
                "comparison",
                "reason_code",
            ],
        ),
        (
            "base_row.json",
            "row",
            &[
                "schema_version",
                "proposal_id",
                "action_id",
                "stages",
                "proposal_context",
                "summary",
                "float_policy",
                "crosswalk",
            ],
        ),
    ];
    for (name, kind, keys) in tables {
        let output = roundtrip(kind, bytes(name)).unwrap();
        ordered(&output, keys);
        assert_eq!(output, bytes(name));
    }
    // Nested object order is also covered byte-for-byte by the hand-authored fixtures.
    let output = roundtrip("attempt", bytes("base_attempt.json")).unwrap();
    assert!(std::str::from_utf8(&output)
        .unwrap()
        .contains("\"target\":{\"proposed_path_digest\":"));
    assert!(std::str::from_utf8(&output)
        .unwrap()
        .contains("\"target_ref\":{\"availability\":\"present\",\"value\":"));
}

#[test]
fn receipt_availability_encoding() {
    let cases = [
        (
            PresentUnavailable::<String, ToolNameError>::Present("λ".into()),
            "{\"availability\":\"present\",\"value\":\"λ\",\"error_code\":null}",
        ),
        (
            PresentUnavailable::Unavailable(ToolNameError::MissingName),
            "{\"availability\":\"unavailable\",\"value\":null,\"error_code\":\"missing_name\"}",
        ),
    ];
    for (field, expected) in cases {
        assert_eq!(serde_json::to_string(&field).unwrap(), expected);
        assert_eq!(
            serde_json::from_str::<PresentUnavailable<String, ToolNameError>>(expected).unwrap(),
            field
        );
    }
    assert_eq!(
        serde_json::to_string(&PresentErrored::<u64, ArgumentsError>::Errored(
            ArgumentsError::ParseError
        ))
        .unwrap(),
        r#"{"availability":"errored","value":null,"error_code":"parse_error"}"#
    );
    assert_eq!(
        serde_json::to_string(&PresentAbsent::<String>::AbsentByContract).unwrap(),
        r#"{"availability":"absent_by_contract","value":null,"error_code":null}"#
    );
    for invalid in [
        r#"{"availability":"redacted","value":null,"error_code":null}"#,
        r#"{"availability":"errored","value":null,"error_code":"missing_name"}"#,
        r#"{"availability":"present","value":"x","error_code":"missing_name"}"#,
        r#"{"availability":"unavailable","value":null,"error_code":"not_a_code"}"#,
    ] {
        assert!(
            serde_json::from_str::<PresentUnavailable<String, ToolNameError>>(invalid).is_err()
        );
    }
    // Catches m17: this value has the exact present shape except for vocabulary.
    assert!(serde_json::from_str::<PresentUnavailable<String, ToolNameError>>(
        r#"{"availability":"redacted","value":"x","error_code":null}"#
    )
    .is_err());
    let mut redacted_slot = json!({
        "availability": "present",
        "reason": null,
        "digest": "0".repeat(64),
        "record": value("base_attempt.json"),
    });
    assert!(serde_json::from_value::<Slot<Attempt, AttemptAbsentReason>>(
        redacted_slot.clone()
    )
    .is_ok());
    redacted_slot["availability"] = json!("redacted");
    // Catches m17: a valid present Slot must reject the redacted vocabulary word.
    assert!(serde_json::from_value::<Slot<Attempt, AttemptAbsentReason>>(
        redacted_slot
    )
    .is_err());
    let mut redacted_present_slot = json!({
        "availability": "present",
        "reason": null,
        "digest": "0".repeat(64),
        "record": value("base_proposal.json"),
    });
    assert!(serde_json::from_value::<PresentSlot<Proposal>>(
        redacted_present_slot.clone()
    )
    .is_ok());
    redacted_present_slot["availability"] = json!("redacted");
    // Catches m17: a valid PresentSlot must reject the redacted vocabulary word.
    assert!(serde_json::from_value::<PresentSlot<Proposal>>(redacted_present_slot).is_err());
    for reason in [
        AttemptAbsentReason::Blocked,
        AttemptAbsentReason::Unsupported,
        AttemptAbsentReason::NotEnforced,
        AttemptAbsentReason::NoExecutor,
    ] {
        let slot = Slot::<Attempt, _>::AbsentByContract(reason);
        let output = serde_json::to_vec(&slot).unwrap();
        ordered(&output, &["availability", "reason", "digest", "record"]);
        assert_eq!(
            serde_json::from_slice::<Slot<Attempt, AttemptAbsentReason>>(&output).unwrap(),
            slot
        );
    }
    for reason in [
        PostAttemptAbsentReason::AttemptAbsent,
        PostAttemptAbsentReason::AttemptRejected,
    ] {
        let slot = Slot::<Observation, _>::AbsentByContract(reason);
        assert_eq!(
            serde_json::from_slice::<Slot<Observation, PostAttemptAbsentReason>>(
                &serde_json::to_vec(&slot).unwrap()
            )
            .unwrap(),
            slot
        );
    }
    assert!(serde_json::from_value::<PresentSlot<Proposal>>(
        json!({"availability":"absent_by_contract","reason":"blocked","digest":null,"record":null})
    )
    .is_err());
}

#[test]
fn receipt_closed_fields_and_required_nulls() {
    for (name, kind) in [
        ("base_proposal_context.json", "proposal_context"),
        ("base_decision.json", "decision"),
        ("base_attempt.json", "attempt"),
        ("base_observation.json", "observation"),
        ("base_verification.json", "verification"),
        ("base_row.json", "row"),
        ("policy.json", "policy"),
    ] {
        let complete = value(name);
        for key in complete.as_object().unwrap().keys() {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(key);
            // Policy's ordinary schema parse is used here so a failure is returned,
            // rather than unwrapped by the valid-fixture roundtrip helper.
            if kind == "policy" {
                assert!(serde_json::from_value::<ResolvedActionPolicyV0>(missing).is_err());
            } else {
                assert!(
                    roundtrip(kind, &serde_json::to_vec(&missing).unwrap()).is_err(),
                    "{name}: missing {key}"
                );
            }
        }
    }
    let base = value("base_proposal.json");
    for key in base.as_object().unwrap().keys() {
        let mut missing = base.clone();
        missing.as_object_mut().unwrap().remove(key);
        assert!(
            serde_json::from_value::<Proposal>(missing).is_err(),
            "missing {key}"
        );
    }
    for invalid in [json!(null), json!(0.0), json!(-1)] {
        let mut altered = base.clone();
        altered["index"] = invalid;
        assert!(serde_json::from_value::<Proposal>(altered).is_err());
    }
    let mut policy = value("policy.json");
    policy["tools"]["write_file"]["constraints"]["content"]
        .as_object_mut()
        .unwrap()
        .remove("min");
    assert!(serde_json::from_value::<ResolvedActionPolicyV0>(policy).is_err());
    let mut row = value("base_row.json");
    row["stages"]["proposal"]["record"]["extra"] = json!(true);
    assert!(serde_json::from_value::<Row>(row).is_err());
}

fn wire_bytes(case: &Value) -> Vec<u8> {
    let parsed = match &case["wire"] {
        Value::String(s) if !s.is_empty() => serde_json::from_str(s).unwrap(),
        Value::Object(_) => case["wire"].clone(),
        _ => json!({}),
    };
    canonical_argument_bytes(&parsed).unwrap()
}

#[test]
fn receipt_c1_cross_provider() {
    let cases = value("F7_wire.json");
    let results: Vec<_> = cases.as_array().unwrap().iter().map(wire_bytes).collect();
    for (case, actual) in cases.as_array().unwrap().iter().zip(&results) {
        assert_eq!(actual, case["expected"].as_str().unwrap().as_bytes());
    }
    assert_eq!(arguments_digest(&results[0]), arguments_digest(&results[1]));
    assert_ne!(arguments_digest(&results[0]), arguments_digest(&results[2]));
}
#[test]
fn receipt_c1_empty_wire_forms() {
    let cases = value("F38_wire.json");
    assert_eq!(cases.as_array().unwrap().len(), 5);
    for case in cases.as_array().unwrap() {
        let actual = wire_bytes(case);
        assert_eq!(actual, b"{}");
        assert_eq!(
            arguments_digest(&actual),
            oracle_hash(Some(b"parapet-effect-proposal-arguments/v0"), b"{}")
        );
    }
    let mut proposal_ids = Vec::new();
    for i in 0..5 {
        let row: Row = serde_json::from_slice(bytes(&format!("F38_form_{i}.json"))).unwrap();
        let context = row.proposal_context.record;
        let native_present = matches!(context.native_arguments, PresentUnavailable::Present(_));
        assert_eq!(native_present, i == 0 || i == 3);
        assert_eq!(
            row.stages.proposal.record.arguments_digest,
            PresentErrored::Present(arguments_digest(b"{}"))
        );
        assert_eq!(
            row.stages.proposal.record.arguments_len_bytes,
            PresentErrored::Present(2)
        );
        proposal_ids.push(row.proposal_id);
    }
    assert!(proposal_ids.iter().all(|id| id == &proposal_ids[0]));
    // The API consumes parsed values; these five wire forms are caller-owned inputs.
    let mismatched = value("F38_mismatch_row.json");
    assert_eq!(
        mismatched["proposal_context"]["record"]["arguments_wire_form"],
        "absent"
    );
    assert_eq!(
        mismatched["proposal_context"]["record"]["native_arguments"]["availability"],
        "present"
    );
}
#[test]
fn receipt_c1_numeric_bytes() {
    let cases = value("F47_wire.json");
    let mut digests = Vec::new();
    for case in cases.as_array().unwrap() {
        let actual = wire_bytes(case);
        assert_eq!(actual, case["expected"].as_str().unwrap().as_bytes());
        digests.push(arguments_digest(&actual));
    }
    digests.sort();
    digests.dedup();
    assert_eq!(digests.len(), 3);
}
#[test]
fn receipt_c2_vs_raw() {
    let args = bytes("arguments.json");
    let proposal: Proposal = serde_json::from_slice(bytes("base_proposal.json")).unwrap();
    let context: ProposalContext =
        serde_json::from_slice(bytes("base_proposal_context.json")).unwrap();
    assert_eq!(
        proposal.arguments_digest,
        PresentErrored::Present(arguments_digest(args))
    );
    let PresentUnavailable::Present(reference) = context.arguments_payload else {
        panic!()
    };
    assert_eq!(reference.digest, oracle_hash(None, args));
    assert_eq!(reference.len_bytes, args.len() as u64);
    assert_ne!(reference.digest, arguments_digest(args));
}
#[test]
fn receipt_proposal_identity() {
    let baseline: Row = serde_json::from_slice(bytes("base_row.json")).unwrap();
    let identity = &baseline.proposal_id;
    for name in [
        "F13_row.json",
        "F41_row.json",
        "F42_row.json",
        "F44_row.json",
    ] {
        let row: Row = serde_json::from_slice(bytes(name)).unwrap();
        assert_eq!(&row.proposal_id, identity);
        assert_eq!(row.stages.proposal, baseline.stages.proposal);
        assert_ne!(
            row.proposal_context.digest,
            baseline.proposal_context.digest
        );
        let output =
            record_bytes(ReceiptRecord::ProposalContext(&row.proposal_context.record)).unwrap();
        assert_eq!(
            row.proposal_context.digest,
            record_digest(&output, RecordDomain::Stage)
        );
    }
    assert_eq!(bytes("F43_stream.json"), bytes("F43_nonstream.json"));
    let other_choice: Row = serde_json::from_slice(bytes("F44_row.json")).unwrap();
    assert_eq!(
        other_choice.proposal_context.record.request_ref,
        baseline.proposal_context.record.request_ref
    );
    assert_ne!(
        other_choice.proposal_context.record.choice_index,
        baseline.proposal_context.record.choice_index
    );
    let mut proposal = baseline.stages.proposal.record.clone();
    proposal.index += 1;
    assert_ne!(
        record_digest(
            &record_bytes(ReceiptRecord::Proposal(&proposal)).unwrap(),
            RecordDomain::Stage
        ),
        *identity
    );
}

#[test]
fn receipt_target_ref_recompute() {
    let vector = value("F57_identity.json");
    let nonce: [u8; 32] = std::array::from_fn(|i| i as u8);
    let root = root_digest(&nonce, vector["root_path"].as_str().unwrap());
    assert_eq!(root, vector["root_digest"]);
    let root_raw: [u8; 32] =
        std::array::from_fn(|i| u8::from_str_radix(&root[i * 2..i * 2 + 2], 16).unwrap());
    assert_eq!(target_ref(&root_raw, "a.txt"), vector["target_ref"]);
    assert_eq!(
        proposed_path_digest("a.txt"),
        vector["proposed_path_digest"]
    );
    assert_ne!(target_ref(&root_raw, "b.txt"), vector["target_ref"]);
    let mut other_root = root_raw;
    other_root[0] ^= 1;
    assert_ne!(target_ref(&other_root, "a.txt"), vector["target_ref"]);
    assert_ne!(root_digest(&[0; 32], "/home/user/root"), root);
    assert_ne!(
        proposed_path_digest("./a.txt"),
        proposed_path_digest("a.txt")
    );
    let mut preimage = b"parapet-effect-target/v0\n".to_vec();
    preimage.extend_from_slice(&root_raw);
    preimage.push(0);
    preimage.extend_from_slice(b"a.txt");
    assert_eq!(target_ref(&root_raw, "a.txt"), oracle_hash(None, &preimage));
}

#[test]
fn receipt_policy_and_code_digest() {
    let input = bytes("policy.json");
    let policy: ResolvedActionPolicyV0 = serde_json::from_slice(input).unwrap();
    assert_eq!(canonical_policy_bytes(&policy).unwrap(), input);
    assert_eq!(
        policy_digest(input),
        oracle_hash(Some(b"parapet-resolved-policy/v0"), input)
    );
    assert_eq!(
        code_digest(b"fixture executable"),
        oracle_hash(Some(b"parapet-engine-code/v0"), b"fixture executable")
    );
    assert_ne!(
        code_digest(b"fixture executable"),
        payload_digest(b"fixture executable")
    );
    let mut bad = policy.clone();
    bad.tools
        .get_mut("write_file")
        .unwrap()
        .constraints
        .get_mut("content")
        .unwrap()
        .min = Some(f64::NAN);
    assert!(matches!(
        canonical_policy_bytes(&bad),
        Err(PolicyBytesError::NonFinite)
    ));
    // Catches m20: max must reject non-finite bounds independently of min.
    for bound in [f64::NAN, f64::INFINITY] {
        let mut bad_max = policy.clone();
        bad_max
            .tools
            .get_mut("write_file")
            .unwrap()
            .constraints
            .get_mut("content")
            .unwrap()
            .max = Some(bound);
        assert!(matches!(
            canonical_policy_bytes(&bad_max),
            Err(PolicyBytesError::NonFinite)
        ));
    }

}

#[test]
fn receipt_dr2_two_digests() {
    let input = bytes("F74_detection.json");
    let row: Row = serde_json::from_slice(bytes("F74_row.json")).unwrap();
    let PresentUnavailableAbsent::Present(detection) = row.stages.decision.record.detection else {
        panic!()
    };
    let PresentUnavailableAbsent::Present(payload) = row.proposal_context.record.detection_payload
    else {
        panic!()
    };
    assert_eq!(payload.digest, oracle_hash(None, input));
    assert_eq!(
        detection.record_digest,
        oracle_hash(Some(b"parapet-l1-detection-record-digest/v0"), input)
    );
    assert_eq!(detection_record_digest(input), detection.record_digest);
    assert_ne!(payload.digest, detection.record_digest);
    assert_eq!(payload.len_bytes, input.len() as u64);
    assert_eq!(detection.record_len_bytes, payload.len_bytes);
    let record: parapet::layers::l1_detection::L1DetectionRecord =
        serde_json::from_slice(input).unwrap();
    assert_eq!(detection.input_digest, record.input_digest);
    assert_eq!(
        serde_json::to_value(detection.mode).unwrap(),
        serde_json::to_value(record.mode).unwrap()
    );
    assert_eq!(
        serde_json::to_value(detection.detection_outcome).unwrap(),
        serde_json::to_value(record.detection_outcome).unwrap()
    );
}

#[test]
fn receipt_dr2_recompute_matches_l1() {
    use parapet::layers::{
        l1_detection as l1,
        l1_harness::{AttributedFeature, L1Signal, OutcomeView},
    };
    let record = l1::L1DetectionRecord {
        schema_version: l1::SCHEMA_VERSION.into(),
        model_id: "fixture-model".into(),
        model_digest: payload_digest(b"fixture-model"),
        code_version: "0.1.0".into(),
        mode: l1::RecordMode::Shadow,
        threshold: 0.5,
        mention_delta_threshold: 1.0,
        source_digest: payload_digest(b"hello"),
        source_len_bytes: 5,
        normalization: l1::NORMALIZATION.into(),
        input_digest: payload_digest(b"hello"),
        input_len_bytes: 5,
        signal: L1Signal {
            message_index: 0,
            role: parapet::message::Role::User,
            raw_score: 0.25,
            raw_unquoted_score: 0.25,
            raw_squash_score: 0.25,
            score: 0.25,
            unquoted_score: 0.25,
            squash_score: 0.25,
            quote_detected: false,
            raw_score_delta: 0.0,
        },
        outcome_view: OutcomeView::Raw,
        effective_raw: 0.25,
        detection_outcome: l1::DetectionOutcome::ThresholdNotBreached,
        view_text_digest: payload_digest(b"hello"),
        view_text_len_scalars: 5,
        view_text_lower_len_scalars: 5,
        attribution_status: l1::AttributionStatus::Present,
        top_k: 10,
        top_features: vec![AttributedFeature {
            feature: "he".into(),
            weight: 0.25,
            contribution: 0.25,
            occurrences: 1,
            spans: vec![[0, 2]],
        }],
        feature_count_matched: Some(1),
        bias: 0.0,
        matched_weight_sum_positive: Some(0.25),
        matched_weight_sum_negative: Some(0.0),
    };
    assert_eq!(l1::SCHEMA_VERSION, "l1-detection-record/v0");
    let carried = l1::record_bytes(&record).unwrap();
    assert_eq!(carried, bytes("F74_detection.json"));
    assert_eq!(
        detection_record_digest(&carried),
        l1::record_digest(&record).unwrap()
    );
}

#[test]
fn receipt_rejected_attempt_field_shape() {
    let mut attempt: Attempt = serde_json::from_slice(bytes("base_attempt.json")).unwrap();
    attempt.outcome = AttemptOutcome::Rejected;
    attempt.error_code = Some(AttemptError::SinkRejected);
    attempt.target.target_ref = PresentUnavailable::Unavailable(TargetError::NotResolved);
    attempt.target.resolved_path = PresentUnavailable::Unavailable(TargetError::NotResolved);
    attempt.before_state = PresentUnavailableErroredAbsent::AbsentByContract;
    attempt.content_payload = PresentUnavailableAbsent::AbsentByContract;
    let output = record_bytes(ReceiptRecord::Attempt(&attempt)).unwrap();
    let parsed: Attempt = serde_json::from_slice(&output).unwrap();
    assert_eq!(parsed, attempt);
    let text = std::str::from_utf8(&output).unwrap();
    assert!(text.contains(r#""target_ref":{"availability":"unavailable","value":null,"error_code":"not_resolved"},"resolved_path":{"availability":"unavailable","value":null,"error_code":"not_resolved"}"#));
    assert!(text.contains(
        r#""before_state":{"availability":"absent_by_contract","value":null,"error_code":null}"#
    ));
    assert!(text.contains(
        r#""content_payload":{"availability":"absent_by_contract","value":null,"error_code":null}"#
    ));
}

#[test]
fn receipt_decision_enum_vocabulary() {
    // Catches m21: section 9 enforcement strings must be exact in both directions.
    for (variant, literal) in [
        (Enforcement::Enforced, r#""enforced""#),
        (Enforcement::Shadow, r#""shadow""#),
        (Enforcement::NotEvaluated, r#""not_evaluated""#),
        (Enforcement::SuppressedByResponse, r#""suppressed_by_response""#),
    ] {
        assert_eq!(serde_json::to_string(&variant).unwrap(), literal);
        assert_eq!(serde_json::from_str::<Enforcement>(literal).unwrap(), variant);
    }
    for (variant, literal) in [
        (DecisionOutcome::Allow, r#""allow""#),
        (DecisionOutcome::Block, r#""block""#),
    ] {
        assert_eq!(serde_json::to_string(&variant).unwrap(), literal);
        assert_eq!(serde_json::from_str::<DecisionOutcome>(literal).unwrap(), variant);
    }
    for (variant, literal) in [
        (DecisionReason::Allowed, r#""allowed""#),
        (DecisionReason::ToolNotAllowed, r#""tool_not_allowed""#),
        (DecisionReason::ArgumentMissing, r#""argument_missing""#),
        (DecisionReason::TypeCheck, r#""type_check""#),
        (DecisionReason::StartsWith, r#""starts_with""#),
        (DecisionReason::NotContains, r#""not_contains""#),
        (DecisionReason::Matches, r#""matches""#),
        (DecisionReason::OneOf, r#""one_of""#),
        (DecisionReason::MaxLength, r#""max_length""#),
        (DecisionReason::Min, r#""min""#),
        (DecisionReason::Max, r#""max""#),
        (DecisionReason::UrlHost, r#""url_host""#),
        (DecisionReason::Multiple, r#""multiple""#),
        (DecisionReason::ProposalUnparsable, r#""proposal_unparsable""#),
        (DecisionReason::ProposalShapeInvalid, r#""proposal_shape_invalid""#),
        (DecisionReason::PolicyLayerOff, r#""policy_layer_off""#),
        (DecisionReason::DetectionGate, r#""detection_gate""#),
    ] {
        assert_eq!(serde_json::to_string(&variant).unwrap(), literal);
        assert_eq!(serde_json::from_str::<DecisionReason>(literal).unwrap(), variant);
    }
}

// Pure function fixture expectations are supplied independently of production code.
fn a2_cases() -> Value {
    serde_json::from_slice(include_bytes!("fixtures/receipt_v0/a2_function_cases.json")).unwrap()
}

fn a2_method_case(case: &Value) {
    let proposal: Proposal = serde_json::from_value(case["proposal"].clone()).unwrap();
    let attempt: Attempt = serde_json::from_value(case["attempt"].clone()).unwrap();
    let observation: Observation = serde_json::from_value(case["observation"].clone()).unwrap();
    let result = verify_expected_effect(&proposal, &attempt, &observation);
    if case["expected"].is_null() {
        assert_eq!(result, None, "{}", case["id"]);
    } else {
        let expected = &case["expected"];
        let result = result.unwrap();
        assert_eq!(
            serde_json::to_value(result.outcome).unwrap(),
            expected["outcome"],
            "{}",
            case["id"]
        );
        assert_eq!(
            serde_json::to_value(result.reason_code).unwrap(),
            expected["reason_code"],
            "{}",
            case["id"]
        );
        assert_eq!(
            serde_json::to_value(result.comparison).unwrap(),
            expected["comparison"],
            "{}",
            case["id"]
        );
    }
}

macro_rules! a2_method_line_test {
    ($name:ident, $line:literal) => {
        #[test]
        fn $name() {
            let cases = a2_cases();
            for case in cases["methods"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["line"] == $line)
            {
                a2_method_case(case);
            }
        }
    };
}
a2_method_line_test!(receipt_method_line_1, 1);
a2_method_line_test!(receipt_method_line_2, 2);
a2_method_line_test!(receipt_method_line_3, 3);
a2_method_line_test!(receipt_method_line_4, 4);
a2_method_line_test!(receipt_method_line_5, 5);
a2_method_line_test!(receipt_method_line_6, 6);
a2_method_line_test!(receipt_method_line_7, 7);
a2_method_line_test!(receipt_method_line_8, 8);
a2_method_line_test!(receipt_method_line_9, 9);
a2_method_line_test!(receipt_method_line_10, 10);

macro_rules! a2_domain_test {
    ($name:ident, $index:literal) => {
        #[test]
        fn $name() {
            a2_method_case(&a2_cases()["out_of_domain"][$index]);
        }
    };
}
a2_domain_test!(receipt_method_out_of_domain_absent, 0);
a2_domain_test!(receipt_method_out_of_domain_errored, 1);
a2_domain_test!(receipt_method_out_of_domain_rejected, 2);

#[test]
fn receipt_delta_all_pairs_and_missing_measurements() {
    for c in a2_cases()["deltas"].as_array().unwrap() {
        let before = serde_json::from_value(c["before"].clone()).unwrap();
        let observed = serde_json::from_value(c["observed"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(receipt_delta(&before, &observed)).unwrap(),
            c["expected"],
            "{}",
            c["id"]
        );
    }
}

macro_rules! a2_summary_line_test {
    ($name:ident, $line:literal) => {
        #[test]
        fn $name() {
            for c in a2_cases()["summaries"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["line"] == $line)
            {
                let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
                assert_eq!(
                    serde_json::to_value(receipt_summary(&row)).unwrap(),
                    c["expected"],
                    "{}",
                    c["id"]
                );
            }
        }
    };
}
a2_summary_line_test!(receipt_summary_line_1, 1);
a2_summary_line_test!(receipt_summary_line_2, 2);
a2_summary_line_test!(receipt_summary_line_3, 3);
a2_summary_line_test!(receipt_summary_line_4, 4);
a2_summary_line_test!(receipt_summary_sink_all_three_codes, 5);
a2_summary_line_test!(receipt_summary_line_6, 6);
a2_summary_line_test!(receipt_summary_line_7, 7);
a2_summary_line_test!(receipt_summary_line_8, 8);
a2_summary_line_test!(receipt_summary_line_9, 9);
a2_summary_line_test!(receipt_summary_line_10, 10);

#[test]
fn receipt_summary_f54_wrong_labels_and_f8() {
    for c in a2_cases()["wrong_summaries"].as_array().unwrap() {
        let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
        let result = receipt_summary(&row);
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            c["expected"],
            "{}",
            c["id"]
        );
        assert_ne!(result, row.summary, "{}", c["id"]);
    }
}

#[test]
fn receipt_summary_precedence_and_total_invalid_slots() {
    let mut row: Row =
        serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    row.stages.proposal.record.effect_kind = EffectKind::Unsupported;
    row.stages.decision.record.enforcement = Enforcement::Shadow;
    row.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::NoExecutor);
    assert_eq!(receipt_summary(&row), Summary::NotAttemptedUnsupported);
    row.stages.proposal.record.effect_kind = EffectKind::WriteFile;
    assert_eq!(receipt_summary(&row), Summary::NotAttemptedNotEnforced);
    row.stages.decision.record.enforcement = Enforcement::Enforced;
    assert_eq!(receipt_summary(&row), Summary::NotAttemptedNoExecutor);
    // Slot-table invalid but representable: classification remains deterministic.
    row.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::Blocked);
    assert_eq!(receipt_summary(&row), Summary::AllowedEffectVerified);
}

fn a2_projection_cases(prefix: &str) {
    for c in a2_cases()["projections"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["id"].as_str().unwrap().starts_with(prefix))
    {
        let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
        let p = receipt_projection(&row);
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            c["expected"],
            "{}",
            c["id"]
        );
        assert_eq!(
            projection_bytes(&p),
            c["expected_bytes"].as_str().unwrap().as_bytes(),
            "{}",
            c["id"]
        );
        let bytes = record_bytes(ReceiptRecord::Row(&row)).unwrap();
        // Independent digest oracle, deliberately does not call record_digest.
        let mut h = Sha256::new();
        h.update(b"parapet-effect-receipt-digest/v0\n");
        h.update(&bytes);
        assert_eq!(p.projection_of, format!("{:x}", h.finalize()));
    }
}

#[test]
fn receipt_projection_all_thirteen_keys_and_nested_values() {
    a2_projection_cases("");
}
#[test]
fn receipt_projection_f69_order_and_distinct_missingness() {
    a2_projection_cases("F69");
}
#[test]
fn receipt_projection_f70_byte_limits_utf8_and_resolved_source() {
    a2_projection_cases("F70");
}
#[test]
fn receipt_projection_f71_json_escapes_control_bytes() {
    a2_projection_cases("F71");
    for c in a2_cases()["projections"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["id"].as_str().unwrap().starts_with("F71"))
    {
        let bytes = c["expected_bytes"].as_str().unwrap();
        assert!(bytes.contains("\\u001b"));
        assert!(!bytes.contains('\x1b'));
    }
}
#[test]
fn receipt_projection_f72_all_ten_lines() {
    a2_projection_cases("F72");
}
#[test]
fn receipt_projection_f12_detection_display() {
    a2_projection_cases("F12");
    assert_eq!(
        detection_display_text(DetectionAvailability::AbsentByContract),
        Some("not produced by this deployment")
    );
    assert_eq!(
        detection_display_text(DetectionAvailability::Unavailable),
        Some("could not be produced")
    );
    assert_eq!(detection_display_text(DetectionAvailability::Present), None);
}

#[test]
fn receipt_projection_instruments_pass_fail_and_exemptions() {
    for c in a2_cases()["instruments"].as_array().unwrap() {
        let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
        let forbidden: Vec<_> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().as_bytes())
            .collect();
        let p = &c["presented"];
        assert_eq!(
            serde_json::to_value(projection_field_not_allowed(p)).unwrap(),
            c["expected_keys"],
            "{}",
            c["id"]
        );
        assert_eq!(
            serde_json::to_value(redaction_failure(&row, p, &forbidden)).unwrap(),
            c["expected_redaction"],
            "{}",
            c["id"]
        );
        assert_eq!(
            serde_json::to_value(projection_mismatch(&row, p)).unwrap(),
            c["expected_mismatch"],
            "{}",
            c["id"]
        );
    }
}

#[test]
fn receipt_projection_f52_reasons_do_not_carry_paths() {
    for c in a2_cases()["projections"].as_array().unwrap() {
        for entry in c["expected"]["not_present"].as_array().unwrap() {
            if let Some(code) = entry["reason_code"].as_str() {
                assert!(!code.contains('/'));
            }
        }
    }
}

#[test]
fn receipt_projection_determinism_fixture_output() {
    let cases = a2_cases();
    // One output stream can be compared across fresh processes with one test thread.
    for group in ["summaries", "wrong_summaries", "projections"] {
        for c in cases[group].as_array().unwrap() {
            let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
            let p = receipt_projection(&row);
            println!(
                "A2_PROJECTION {} {} {}",
                group,
                c["id"],
                String::from_utf8(projection_bytes(&p)).unwrap()
            );
            let presented = serde_json::to_value(&p).unwrap();
            println!(
                "A2_INSTRUMENT {} {} {:?} {:?} {:?}",
                group,
                c["id"],
                projection_field_not_allowed(&presented),
                redaction_failure(&row, &presented, &[]),
                projection_mismatch(&row, &presented)
            );
        }
    }
    for c in cases["instruments"].as_array().unwrap() {
        let row: Row = serde_json::from_value(c["row"].clone()).unwrap();
        let forbidden: Vec<_> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().as_bytes())
            .collect();
        let p = &c["presented"];
        println!(
            "A2_PROJECTION instruments {} {}",
            c["id"],
            String::from_utf8(projection_bytes(&receipt_projection(&row))).unwrap()
        );
        println!(
            "A2_INSTRUMENT {} {:?} {:?} {:?}",
            c["id"],
            projection_field_not_allowed(p),
            redaction_failure(&row, p, &forbidden),
            projection_mismatch(&row, p)
        );
    }
    for group in ["methods", "out_of_domain"] {
        for c in cases[group].as_array().unwrap() {
            let proposal: Proposal = serde_json::from_value(c["proposal"].clone()).unwrap();
            let attempt: Attempt = serde_json::from_value(c["attempt"].clone()).unwrap();
            let observation: Observation =
                serde_json::from_value(c["observation"].clone()).unwrap();
            println!(
                "A2_METHOD {} {:?}",
                c["id"],
                verify_expected_effect(&proposal, &attempt, &observation)
            );
        }
    }
    for c in cases["deltas"].as_array().unwrap() {
        let before = serde_json::from_value(c["before"].clone()).unwrap();
        let observed = serde_json::from_value(c["observed"].clone()).unwrap();
        println!(
            "A2_DELTA {} {:?}",
            c["id"],
            receipt_delta(&before, &observed)
        );
    }
}

// Catch mutation m26: BQ-5 precision 1 excludes keys from redaction matching.
#[test]
fn receipt_projection_redaction_excludes_closed_and_extra_keys() {
    fn assert_marker_absent_from_strings(value: &serde_json::Value, marker: &str) {
        match value {
            serde_json::Value::String(text) => assert!(!text.contains(marker)),
            serde_json::Value::Array(values) => {
                for value in values {
                    assert_marker_absent_from_strings(value, marker);
                }
            }
            serde_json::Value::Object(values) => {
                for value in values.values() {
                    assert_marker_absent_from_strings(value, marker);
                }
            }
            _ => {}
        }
    }

    let cases = a2_cases();
    let row: Row = serde_json::from_value(cases["projections"][0]["row"].clone()).unwrap();
    let projection = serde_json::to_value(receipt_projection(&row)).unwrap();
    let closed_key = "projection_of";
    assert!(projection.as_object().unwrap().contains_key(closed_key));
    assert_marker_absent_from_strings(&projection, closed_key);
    assert_eq!(
        redaction_failure(&row, &projection, &[closed_key.as_bytes()]),
        ProjectionInstrument::Pass
    );

    let marker = "KEY_ONLY_FIXTURE_7b91c2";
    assert_marker_absent_from_strings(&projection, marker);
    let mut extra_key = projection.clone();
    extra_key["policy"].as_object_mut().unwrap().insert(
        format!("extra_{marker}_key"),
        serde_json::Value::Null,
    );
    assert_marker_absent_from_strings(&extra_key, marker);
    assert_eq!(
        redaction_failure(&row, &extra_key, &[marker.as_bytes()]),
        ProjectionInstrument::Pass
    );
    assert_eq!(
        projection_field_not_allowed(&extra_key),
        ProjectionInstrument::ProjectionFieldNotAllowed
    );

    let mut positive_twin = projection;
    assert!(positive_twin["proposal_id"].is_string());
    positive_twin["proposal_id"] = serde_json::Value::String(marker.to_owned());
    assert_eq!(
        projection_field_not_allowed(&positive_twin),
        ProjectionInstrument::Pass
    );
    assert_eq!(
        redaction_failure(&row, &positive_twin, &[marker.as_bytes()]),
        ProjectionInstrument::RedactionFailure
    );
}

// Configurable resolver double shared by later validator witnesses.
struct ReceiptPayloadResolverDouble {
    results: Vec<(PayloadStore, String, PayloadResolution)>,
    calls: std::cell::RefCell<Vec<(PayloadStore, String)>>,
}

impl PayloadResolver for ReceiptPayloadResolverDouble {
    fn resolve(&self, store: PayloadStore, digest: &str) -> PayloadResolution {
        self.calls.borrow_mut().push((store, digest.to_owned()));
        self.results
            .iter()
            .find(|(s, d, _)| *s == store && d == digest)
            .map(|(_, _, result)| result.clone())
            .expect("unexpected payload lookup")
    }
}

#[test]
fn receipt_validator_skeleton_resolver_partition_and_calls() {
    let resolver = ReceiptPayloadResolverDouble {
        results: vec![
            (
                PayloadStore::ReceiptEvidence,
                "a".into(),
                PayloadResolution::Found(vec![0, 255]),
            ),
            (
                PayloadStore::ReceiptEvidence,
                "b".into(),
                PayloadResolution::NotFound,
            ),
            (
                PayloadStore::RootEvidence,
                "a".into(),
                PayloadResolution::NotConfigured,
            ),
        ],
        calls: Default::default(),
    };
    assert_eq!(
        resolver.resolve(PayloadStore::ReceiptEvidence, "a"),
        PayloadResolution::Found(vec![0, 255])
    );
    assert_eq!(
        resolver.resolve(PayloadStore::ReceiptEvidence, "b"),
        PayloadResolution::NotFound
    );
    assert_eq!(
        resolver.resolve(PayloadStore::RootEvidence, "a"),
        PayloadResolution::NotConfigured
    );
    assert_eq!(
        *resolver.calls.borrow(),
        vec![
            (PayloadStore::ReceiptEvidence, "a".into()),
            (PayloadStore::ReceiptEvidence, "b".into()),
            (PayloadStore::RootEvidence, "a".into())
        ]
    );
    assert_ne!(ValidationResult::ValidUnresolved, ValidationResult::Valid);
    assert_ne!(
        ValidationResult::ValidUnresolved,
        ValidationResult::Rejected(RejectionCode::SchemaError)
    );
}

#[test]
fn receipt_validator_skeleton_closed_rejection_vocabulary() {
    let cases = [
        (RejectionCode::MissingStageSlot, "missing_stage_slot"),
        (
            RejectionCode::SlotAvailabilityInvalid,
            "slot_availability_invalid",
        ),
        (
            RejectionCode::FieldAvailabilityInvalid,
            "field_availability_invalid",
        ),
        (RejectionCode::StageOrderViolation, "stage_order_violation"),
        (RejectionCode::BrokenStageRef, "broken_stage_ref"),
        (RejectionCode::AttemptWithoutAllow, "attempt_without_allow"),
        (
            RejectionCode::AttemptForUnsupported,
            "attempt_for_unsupported",
        ),
        (
            RejectionCode::AttemptWithoutEnforcement,
            "attempt_without_enforcement",
        ),
        (RejectionCode::AttemptWithoutSink, "attempt_without_sink"),
        (
            RejectionCode::ObservationAfterRejectedAttempt,
            "observation_after_rejected_attempt",
        ),
        (
            RejectionCode::ObservationMissingAfterAttempt,
            "observation_missing_after_attempt",
        ),
        (
            RejectionCode::VerificationMissingAfterAttempt,
            "verification_missing_after_attempt",
        ),
        (
            RejectionCode::VerifiedWithoutCompletedAttempt,
            "verified_without_completed_attempt",
        ),
        (
            RejectionCode::VerificationInconsistent,
            "verification_inconsistent",
        ),
        (RejectionCode::NoSilentUpgrade, "no_silent_upgrade"),
        (RejectionCode::RowIdentityMismatch, "row_identity_mismatch"),
        (RejectionCode::SummaryMismatch, "summary_mismatch"),
        (RejectionCode::NonIntegerNumber, "non_integer_number"),
        (RejectionCode::IndexOutOfRange, "index_out_of_range"),
        (RejectionCode::NonCanonicalBytes, "non_canonical_bytes"),
        (RejectionCode::ErrorCodeRequired, "error_code_required"),
        (RejectionCode::ErrorCodeForbidden, "error_code_forbidden"),
        (RejectionCode::ReasonCodeRequired, "reason_code_required"),
        (RejectionCode::ReasonCodeForbidden, "reason_code_forbidden"),
        (
            RejectionCode::ErrorCodeNotInClosedList,
            "error_code_not_in_closed_list",
        ),
        (
            RejectionCode::ToolNameRequiredWhenParsed,
            "tool_name_required_when_parsed",
        ),
        (
            RejectionCode::NativeArgumentsAvailabilityMismatch,
            "native_arguments_availability_mismatch",
        ),
        (
            RejectionCode::ArgumentsPayloadMismatch,
            "arguments_payload_mismatch",
        ),
        (
            RejectionCode::ErroredProposalAllowed,
            "errored_proposal_allowed",
        ),
        (
            RejectionCode::EnforcementInconsistent,
            "enforcement_inconsistent",
        ),
        (
            RejectionCode::PolicyDriftUnrejected,
            "policy_drift_unrejected",
        ),
        (
            RejectionCode::AttemptContentMismatch,
            "attempt_content_mismatch",
        ),
        (
            RejectionCode::TargetBindingMismatch,
            "target_binding_mismatch",
        ),
        (
            RejectionCode::RejectedAttemptShapeInvalid,
            "rejected_attempt_shape_invalid",
        ),
        (RejectionCode::SinkCodeMismatch, "sink_code_mismatch"),
        (
            RejectionCode::DetectionPayloadMismatch,
            "detection_payload_mismatch",
        ),
        (
            RejectionCode::DetectionCopyMismatch,
            "detection_copy_mismatch",
        ),
        (RejectionCode::RedactedInCanonical, "redacted_in_canonical"),
        (RejectionCode::SchemaError, "schema_error"),
    ];
    for (code, name) in cases {
        assert_eq!(serde_json::to_value(code).unwrap(), serde_json::json!(name));
        assert_eq!(
            serde_json::from_value::<RejectionCode>(serde_json::json!(name)).unwrap(),
            code
        );
    }
    for instrument in [
        "foreign_stage_field",
        "projection_field_not_allowed",
        "redaction_failure",
        "projection_mismatch",
        "proposal_count_mismatch",
        "proposal_coordinate_duplicate",
        "row_coordinate_unformable",
        "detection_input_mismatch",
        "valid_unresolved",
        "unknown",
    ] {
        assert!(serde_json::from_value::<RejectionCode>(serde_json::json!(instrument)).is_err());
    }
}

fn s2_validate(bytes: &[u8]) -> ValidationResult {
    let resolver = ReceiptPayloadResolverDouble {
        results: vec![(
            PayloadStore::ReceiptEvidence,
            "a0843e9c3b03ba2286b62cdbbacec9fa556bcc72046a965b24abfe3a1941852d".into(),
            PayloadResolution::Found(include_bytes!("fixtures/receipt_v0/arguments.json").to_vec()),
        )],
        calls: Default::default(),
    };
    let result = validator::validate_receipt(bytes, &resolver);
    result
}

#[test]
fn s2_canonical_row_and_a18_boundaries() {
    let row: Row =
        serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    let canonical = record_bytes(ReceiptRecord::Row(&row)).unwrap();
    assert_eq!(s2_validate(&canonical), ValidationResult::Valid);
    let rejection = ValidationResult::Rejected(RejectionCode::NonCanonicalBytes);
    for suffix in [b" ".as_slice(), b"\n", b"\t"] {
        let mut changed = canonical.clone();
        changed.extend_from_slice(suffix);
        assert_eq!(s2_validate(&changed), rejection);
    }
    let mut bom = vec![0xef, 0xbb, 0xbf];
    bom.extend_from_slice(&canonical);
    assert_eq!(s2_validate(&bom), rejection);
    let mut leading = vec![b' '];
    leading.extend_from_slice(&canonical);
    assert_eq!(s2_validate(&leading), rejection);
}

#[test]
fn s2_f29_whitespace_and_key_order() {
    let row: Row =
        serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    let canonical = record_bytes(ReceiptRecord::Row(&row)).unwrap();
    let text = String::from_utf8(canonical).unwrap();
    let rejection = ValidationResult::Rejected(RejectionCode::NonCanonicalBytes);
    assert_eq!(
        s2_validate(text.replacen(':', ": ", 1).as_bytes()),
        rejection
    );
    // Swap complete member byte spans, preserving values, digest refs and types.
    for (first, second) in [
        (
            "\"schema_version\":\"parapet-effect-receipt/v0\"",
            format!("\"proposal_id\":\"{}\"", row.proposal_id),
        ),
        ("\"availability\":\"present\"", "\"reason\":null".into()),
    ] {
        let pair = format!("{first},{second}");
        assert!(text.contains(&pair), "key pair must be exercised: {pair}");
        let changed = text.replacen(&pair, &format!("{second},{first}"), 1);
        assert_eq!(s2_validate(changed.as_bytes()), rejection);
    }
}

#[test]
fn s2_f29_carried_record_bytes_with_matching_refs() {
    for raw in [
        include_bytes!("fixtures/receipt_v0/F29_space.json").as_slice(),
        include_bytes!("fixtures/receipt_v0/F29_order.json").as_slice(),
    ] {
        let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
        let proposal: Proposal = serde_json::from_slice(raw).unwrap();
        let mut row: Row =
            serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
        row.stages.proposal.record = proposal;
        let proposal_bytes =
            record_bytes(ReceiptRecord::Proposal(&row.stages.proposal.record)).unwrap();
        let proposal_digest = record_digest(raw, RecordDomain::Stage);
        row.proposal_id = proposal_digest.clone();
        row.stages.proposal.digest = proposal_digest.clone();
        row.proposal_context.record.proposal_ref = proposal_digest.clone();
        row.proposal_context.digest = record_digest(
            &record_bytes(ReceiptRecord::ProposalContext(&row.proposal_context.record)).unwrap(),
            RecordDomain::Stage,
        );
        row.stages.decision.record.proposal_ref = proposal_digest.clone();
        row.stages.decision.digest = record_digest(
            &record_bytes(ReceiptRecord::Decision(&row.stages.decision.record)).unwrap(),
            RecordDomain::Stage,
        );
        if let Slot::Present { digest, record } = &mut row.stages.attempt {
            record.decision_ref = row.stages.decision.digest.clone();
            record.sink.proposal.r#ref = Some(proposal_digest);
            record.sink.context.r#ref = Some(row.proposal_context.digest.clone());
            record.sink.decision.r#ref = Some(row.stages.decision.digest.clone());
            *digest = record_digest(
                &record_bytes(ReceiptRecord::Attempt(record)).unwrap(),
                RecordDomain::Stage,
            );
        }
        let attempt_digest = match &row.stages.attempt {
            Slot::Present { digest, .. } => digest.clone(),
            _ => panic!("base attempt"),
        };
        if let Slot::Present { digest, record } = &mut row.stages.observation {
            record.attempt_ref = attempt_digest;
            *digest = record_digest(
                &record_bytes(ReceiptRecord::Observation(record)).unwrap(),
                RecordDomain::Stage,
            );
        }
        let observation_digest = match &row.stages.observation {
            Slot::Present { digest, .. } => digest.clone(),
            _ => panic!("base observation"),
        };
        if let Slot::Present { digest, record } = &mut row.stages.verification {
            record.observation_ref = observation_digest;
            *digest = record_digest(
                &record_bytes(ReceiptRecord::Verification(record)).unwrap(),
                RecordDomain::Stage,
            );
        }
        let canonical = String::from_utf8(record_bytes(ReceiptRecord::Row(&row)).unwrap()).unwrap();
        let proposal_text = std::str::from_utf8(&proposal_bytes).unwrap();
        assert_eq!(canonical.matches(proposal_text).count(), 1);
        let carried = canonical.replacen(proposal_text, std::str::from_utf8(raw).unwrap(), 1);
        assert_eq!(
            s2_validate(carried.as_bytes()),
            ValidationResult::Rejected(RejectionCode::NonCanonicalBytes)
        );
    }
}

#[test]
fn s2_raw_availability_defects_remain_representable_and_strings_use_default_escaping() {
    let mut row: Row =
        serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    row.proposal_context.record.model = "模型 🦀 \"quoted\" \\ newline\n".into();
    let canonical = String::from_utf8(record_bytes(ReceiptRecord::Row(&row)).unwrap()).unwrap();
    assert_ne!(s2_validate(canonical.as_bytes()), ValidationResult::Rejected(RejectionCode::NonCanonicalBytes));
    assert_eq!(
        s2_validate(canonical.replacen("模", "\\u6a21", 1).as_bytes()),
        ValidationResult::Rejected(RejectionCode::NonCanonicalBytes)
    );
    // Correctly encoded raw triples with invalid required-when correspondence
    // must survive step 2. Their checks belong to subsequent subtasks.
    let malformed = canonical.replacen(
        "\"availability\":\"present\",\"reason\":null",
        "\"availability\":\"absent_by_contract\",\"reason\":null",
        1,
    );
    assert!(serde_json::from_str::<Row>(&malformed).is_err());
    assert_ne!(s2_validate(malformed.as_bytes()), ValidationResult::Rejected(RejectionCode::NonCanonicalBytes));
    let malformed = canonical.replacen(
        "\"availability\":\"present\",\"value\":\"write_file\",\"error_code\":null",
        "\"availability\":\"present\",\"value\":\"write_file\",\"error_code\":\"missing_name\"",
        1,
    );
    assert!(serde_json::from_str::<Row>(&malformed).is_err());
    assert_ne!(s2_validate(malformed.as_bytes()), ValidationResult::Rejected(RejectionCode::NonCanonicalBytes));
}

fn s3_row_text() -> String {
    let row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    String::from_utf8(record_bytes(ReceiptRecord::Row(&row)).unwrap()).unwrap()
}

fn s3_reject(text: &[u8], code: RejectionCode) {
    assert_eq!(s2_validate(text), ValidationResult::Rejected(code));
}

#[test]
fn s3_a16_number_tokens_anywhere_and_integer_boundaries_o1_f9() {
    let base = s3_row_text();
    assert_eq!(s2_validate(base.as_bytes()), ValidationResult::Valid);
    for field in ["index", "choice_index"] {
        let needle = format!("\"{field}\":0");
        assert!(base.contains(&needle));
        for token in ["0.0", "1e0", "1E+0", "-0", "-0.0", "1e9999"] {
            s3_reject(base.replacen(&needle, &format!("\"{field}\":{token}"), 1).as_bytes(), RejectionCode::NonIntegerNumber);
        }
        for token in ["-1", "2147483648", "4294967296", "18446744073709551616", "999999999999999999999999999999999999"] {
            s3_reject(base.replacen(&needle, &format!("\"{field}\":{token}"), 1).as_bytes(), RejectionCode::IndexOutOfRange);
        }
        for token in ["0", "2147483647"] {
            let result = s2_validate(base.replacen(&needle, &format!("\"{field}\":{token}"), 1).as_bytes());
            assert_ne!(result, ValidationResult::Rejected(RejectionCode::NonIntegerNumber));
            assert_ne!(result, ValidationResult::Rejected(RejectionCode::IndexOutOfRange));
        }
    }
    for token in ["0.0", "1e0", "-0"] {
        // ER6 applies even at a location whose declared type is string.
        let changed = base.replacen("\"summary\":\"allowed_effect_verified\"", &format!("\"summary\":{token}"), 1);
        s3_reject(changed.as_bytes(), RejectionCode::NonIntegerNumber);
    }
    for token in ["-1", "18446744073709551616"] {
        let changed = base.replacen("\"summary\":\"allowed_effect_verified\"", &format!("\"summary\":{token}"), 1);
        s3_reject(changed.as_bytes(), RejectionCode::SchemaError);
    }
}

#[test]
fn s3_length_number_positions_and_passing_controls() {
    let base = s3_row_text();
    for field in ["content_len_bytes", "len_bytes"] {
        let needle = format!("\"{field}\":");
        let start = base.match_indices(&needle).map(|(offset, _)| offset + needle.len())
            .find(|&offset| base.as_bytes()[offset].is_ascii_digit()).unwrap();
        let end = start + base[start..].bytes().take_while(u8::is_ascii_digit).count();
        assert!(end > start, "{field}");
        let changed = format!("{}0.0{}", &base[..start], &base[end..]);
        s3_reject(changed.as_bytes(), RejectionCode::NonIntegerNumber);
    }
    let mut detection_row: serde_json::Value = serde_json::from_slice(
        include_bytes!("fixtures/receipt_v0/F74_row.json"),
    ).unwrap();
    // All step-1 fields are valid except this row-162 number token.
    // Carried digests and payload correspondence are checked at later steps.
    detection_row["stages"]["decision"]["record"]["detection"]["value"]["record_len_bytes"] = serde_json::json!(0.0);
    s3_reject(&serde_json::to_vec(&detection_row).unwrap(), RejectionCode::NonIntegerNumber);
    let needle = "\"arguments_len_bytes\":{\"availability\":\"present\",\"value\":";
    let start = base.find(needle).unwrap() + needle.len();
    let end = start + base[start..].bytes().take_while(u8::is_ascii_digit).count();
    s3_reject(format!("{}0.0{}", &base[..start], &base[end..]).as_bytes(), RejectionCode::NonIntegerNumber);
    assert_eq!(s2_validate(base.as_bytes()), ValidationResult::Valid);
}

#[test]
fn s3_a17_duplicates_at_nested_objects_and_escaped_keys() {
    let base = s3_row_text();
    for needle in ["\"schema_version\":", "\"availability\":", "\"content_len_bytes\":"] {
        let changed = base.replacen(needle, &format!("{needle}null,{needle}"), 1);
        s3_reject(changed.as_bytes(), RejectionCode::SchemaError);
    }
    s3_reject(base.replacen("\"index\":0", "\"index\":0,\"\\u0069ndex\":0", 1).as_bytes(), RejectionCode::SchemaError);
    // Duplicate keys inside an array element must survive the token parser.
    s3_reject(base.replacen("\"inputs_consumed\":[", "\"inputs_consumed\":[{\"x\":1,\"x\":2},", 1).as_bytes(), RejectionCode::SchemaError);
}

#[test]
fn s3_a19_invalid_json_utf8_surrogates_and_valid_unicode() {
    for bytes in [b"{".as_slice(), b"{}{}", b"{\"x\":01}", b"{\"x\":1.}", b"{\"x\":1e}", b"{\"x\":true,}", b"[1,]", b"{\"x\":\"\\ud800\"}", b"{\"x\":\"\\udc00\"}", b"{\"x\":\"\xff\"}"] {
        s3_reject(bytes, RejectionCode::SchemaError);
    }
    let mut row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap();
    row.proposal_context.record.model = "redacted 模型 🦀".into();
    assert_ne!(s2_validate(&record_bytes(ReceiptRecord::Row(&row)).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s3_availability_vocabulary_and_redaction_only_at_structural_positions() {
    let base = s3_row_text();
    // Every availability position, fields as well as lifecycle/context slots.
    let needle = "\"availability\":\"present\"";
    let mut count = 0;
    for (offset, _) in base.match_indices(needle) {
        for (value, code) in [("redacted", RejectionCode::RedactedInCanonical), ("unknown", RejectionCode::SchemaError)] {
            let changed = format!("{}\"availability\":\"{value}\"{}", &base[..offset], &base[offset + needle.len()..]);
            s3_reject(changed.as_bytes(), code);
        }
        count += 1;
    }
    assert!(count > 15);
    for value in ["absent_by_contract", "unavailable", "errored"] {
        // Canonical vocabulary passes step 1; later correspondence may reject.
        let result = s2_validate(base.replacen(needle, &format!("\"availability\":\"{value}\""), 1).as_bytes());
        assert_ne!(result, ValidationResult::Rejected(RejectionCode::SchemaError));
        assert_ne!(result, ValidationResult::Rejected(RejectionCode::RedactedInCanonical));
    }
}

#[test]
fn s3_bq19_missing_lifecycle_and_context_slots_f10() {
    let mut value: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for key in ["proposal", "decision", "attempt", "observation", "verification"] {
        let mut changed = value.clone();
        changed["stages"].as_object_mut().unwrap().remove(key);
        s3_reject(&serde_json::to_vec(&changed).unwrap(), RejectionCode::MissingStageSlot);
    }
    value.as_object_mut().unwrap().remove("proposal_context");
    s3_reject(&serde_json::to_vec(&value).unwrap(), RejectionCode::MissingStageSlot);
}

#[test]
fn s3_bq11_location_major_depth_first_despite_wire_order() {
    let base = s3_row_text();
    let changed = base.replacen("\"index\":0", "\"index\":2147483648", 1)
        .replacen("\"choice_index\":0", "\"choice_index\":0.0", 1);
    s3_reject(changed.as_bytes(), RejectionCode::IndexOutOfRange);
    let mut value: serde_json::Value = serde_json::from_str(&changed).unwrap();
    // serde's lexical key order puts context before stages and availability
    // before index, but the contract table/location order must still win.
    value["stages"]["proposal"]["record"]["tool_name"]["availability"] = "redacted".into();
    s3_reject(&serde_json::to_vec(&value).unwrap(), RejectionCode::IndexOutOfRange);
    value["stages"]["proposal"]["record"]["index"] = 0.into();
    s3_reject(&serde_json::to_vec(&value).unwrap(), RejectionCode::RedactedInCanonical);
    value["stages"].as_object_mut().unwrap().remove("decision");
    s3_reject(&serde_json::to_vec(&value).unwrap(), RejectionCode::RedactedInCanonical);
    value["stages"]["proposal"]["record"]["tool_name"]["availability"] = "present".into();
    s3_reject(&serde_json::to_vec(&value).unwrap(), RejectionCode::MissingStageSlot);
}

// S4a: step-1 witnesses deliberately use Value serialization. Any canonical
// byte or digest defect this introduces is strictly later than step 1 (R2).
fn s4a_reject(value: &serde_json::Value, code: RejectionCode) {
    s3_reject(&serde_json::to_vec(value).unwrap(), code);
}

#[test]
fn s4a_closed_objects_missing_fields_foreign_fields_and_t1_f5() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    let mut objects = Vec::new();
    fn gather(v: &serde_json::Value, path: String, out: &mut Vec<String>) {
        if let Some(o) = v.as_object() {
            out.push(path.clone());
            for (k, child) in o { gather(child, format!("{path}/{k}"), out); }
        }
    }
    gather(&base, String::new(), &mut objects);
    for path in objects {
        for key in ["extra", "success", "timestamp", "own_digest"] {
            let mut changed = base.clone();
            changed.pointer_mut(&path).unwrap().as_object_mut().unwrap().insert(key.into(), true.into());
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
        for key in base.pointer(&path).unwrap().as_object().unwrap().keys() {
            let mut changed = base.clone();
            changed.pointer_mut(&path).unwrap().as_object_mut().unwrap().remove(key);
            let code = if path == "/stages" || (path.is_empty() && key == "proposal_context") { RejectionCode::MissingStageSlot } else { RejectionCode::SchemaError };
            s4a_reject(&changed, code);
        }
    }
    let mut changed = base.clone();
    changed["stages"]["proposal"]["record"]["outcome"] = "verified".into(); // F5
    changed["proposal_context"]["record"]["choice_index"] = serde_json::json!(0.0); // T1
    s4a_reject(&changed, RejectionCode::SchemaError);
}

#[test]
fn s4a_row_literals_summary_action_ids_and_request_identity() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for path in ["/schema_version", "/summary", "/float_policy", "/crosswalk"] {
        for replacement in [serde_json::json!("unknown"), serde_json::json!(true), serde_json::json!(null)] {
            let mut changed = base.clone();
            *changed.pointer_mut(path).unwrap() = replacement;
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
    }
    for path in ["/action_id", "/stages/attempt/record/action_id"] {
        for bad in ["", "A", "-a", "a/b", "a b", &"a".repeat(129)] {
            let mut changed = base.clone(); *changed.pointer_mut(path).unwrap() = bad.into();
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
    }
    for bad in ["", &"é".repeat(129), &"a".repeat(5000)] {
        let mut changed = base.clone(); changed["proposal_context"]["record"]["request_ref"]["value"]["request_id"] = bad.into();
        s4a_reject(&changed, RejectionCode::SchemaError);
    }
    let mut changed = base.clone(); changed["proposal_context"]["record"]["request_ref"]["value"]["request_id_source"] = "unknown".into();
    s4a_reject(&changed, RejectionCode::SchemaError);
    // F48: unavailable is schema-valid. R3 isolates step 1 without asserting Valid
    // over unrepaired context bytes and dependent digests.
    changed = base.clone();
    changed["proposal_context"]["record"]["request_ref"] = serde_json::json!({"availability":"unavailable","value":null,"error_code":"request_id_oversize"});
    assert_ne!(s2_validate(&serde_json::to_vec(&changed).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4a_slot_shapes_nonpresent_records_and_f30() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for slot in ["proposal", "decision", "attempt", "observation", "verification"] {
        for (field, bad) in [("digest", serde_json::json!("bad")), ("reason", serde_json::json!("unknown")), ("record", serde_json::json!(false))] {
            let mut changed = base.clone(); changed["stages"][slot][field] = bad;
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
        let mut changed = base.clone(); changed["stages"][slot]["availability"] = "absent_by_contract".into();
        changed["stages"][slot]["record"]["stage"] = "wrong_kind".into();
        s4a_reject(&changed, RejectionCode::SchemaError);
    }
    let mut changed = base.clone(); changed["stages"]["verification"]["record"]["observation_ref"] = serde_json::Value::Null;
    s4a_reject(&changed, RejectionCode::SchemaError); // F30
    changed = base.clone(); changed["stages"]["verification"]["record"]["outcome"] = "completed".into();
    s4a_reject(&changed, RejectionCode::SchemaError);
}

#[test]
fn s4a_digest_formats_payload_stores_and_nonpresent_payload_types() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    let mut paths = Vec::new();
    fn gather(v: &serde_json::Value, path: String, out: &mut Vec<String>) {
        if let Some(o) = v.as_object() {
            for (k, child) in o {
                let next = format!("{path}/{k}");
                if child.as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())) { out.push(next.clone()); }
                gather(child, next, out);
            }
        }
    }
    gather(&base, String::new(), &mut paths);
    assert!(paths.len() > 20);
    for path in paths {
        for bad in ["a".repeat(63), "a".repeat(65), "A".repeat(64), "g".repeat(64)] {
            let mut changed = base.clone(); *changed.pointer_mut(&path).unwrap() = bad.into();
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
    }
    for path in ["/proposal_context/record/arguments_payload", "/stages/attempt/record/content_payload"] {
        let good_store = base.pointer(path).unwrap()["value"]["store"].as_str().unwrap();
        for bad in ["unknown", if good_store == "receipt-evidence" { "root-evidence" } else { "receipt-evidence" }] {
            let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["value"]["store"] = bad.into();
            s4a_reject(&changed, RejectionCode::SchemaError);
        }
        let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["value"]["media_type"] = true.into();
        s4a_reject(&changed, RejectionCode::SchemaError);
        let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["availability"] = "unavailable".into();
        changed.pointer_mut(path).unwrap()["value"]["digest"] = "bad".into();
        s4a_reject(&changed, RejectionCode::SchemaError); // A2: type even when non-present
    }
    // contract_hash is an arbitrary string; R3 checks only its schema format.
    let mut changed = base.clone(); changed["stages"]["decision"]["record"]["policy"]["contract_hash"] = "arbitrary runtime form".into();
    assert_ne!(s2_validate(&serde_json::to_vec(&changed).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4a_closed_error_lists_a4_a5_a22_and_f37() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for path in [
        "/stages/proposal/record/tool_name", "/stages/proposal/record/arguments_digest", "/stages/proposal/record/arguments_len_bytes", "/stages/proposal/record/expected_effect",
        "/proposal_context/record/request_ref", "/proposal_context/record/tool_call_id", "/proposal_context/record/native_arguments", "/proposal_context/record/arguments_payload", "/proposal_context/record/detection_payload",
        "/stages/decision/record/code", "/stages/decision/record/detection", "/stages/decision/record/detection_binding_rule",
        "/stages/attempt/record/before_state", "/stages/attempt/record/target/target_ref", "/stages/attempt/record/target/resolved_path", "/stages/attempt/record/content_payload",
        "/stages/observation/record/target_ref", "/stages/observation/record/observed", "/stages/observation/record/footprint", "/stages/verification/record/comparison",
    ] {
        for bad in ["unknown", "", "A", &"a".repeat(65)] {
            let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["error_code"] = bad.into();
            s4a_reject(&changed, RejectionCode::ErrorCodeNotInClosedList);
        }
        let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["error_code"] = false.into();
        s4a_reject(&changed, RejectionCode::SchemaError);
    }
    for path in ["/stages/proposal/record/parse_error_code", "/stages/attempt/record/error_code", "/stages/verification/record/reason_code"] {
        let mut changed = base.clone(); *changed.pointer_mut(path).unwrap() = "unknown".into();
        s4a_reject(&changed, RejectionCode::ErrorCodeNotInClosedList);
    }
    for (path, code, availability) in [
        ("/stages/attempt/record/before_state", "no_read_access", "errored"),
        ("/stages/attempt/record/before_state", "io_error", "unavailable"),
        ("/stages/observation/record/observed", "no_read_access", "errored"),
        ("/stages/observation/record/observed", "io_error", "unavailable"),
        ("/stages/observation/record/observed", "observer_failed", "unavailable"),
    ] {
        let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["availability"] = availability.into(); changed.pointer_mut(path).unwrap()["error_code"] = code.into();
        s4a_reject(&changed, RejectionCode::ErrorCodeNotInClosedList);
    }
    let mut changed = base.clone();
    changed["stages"]["observation"]["record"]["observed"] = serde_json::json!({"availability":"errored","value":null,"error_code":"Permission denied (os error 13): /home/user/root/a.txt"});
    s4a_reject(&changed, RejectionCode::ErrorCodeNotInClosedList); // F37
}

#[test]
fn s4a_bare_types_wrapped_value_types_and_closed_list_controls() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for path in [
        "/stages/proposal/record/provider", "/stages/proposal/record/tool_name/value",
        "/stages/proposal/record/arguments_len_bytes/value", "/stages/proposal/record/expected_effect/value/target_path",
        "/proposal_context/record/model", "/proposal_context/record/arguments_payload/value/len_bytes",
        "/stages/decision/record/policy/contract_hash", "/stages/decision/record/evaluated_policy",
        "/stages/attempt/record/before_state/value/exists", "/stages/attempt/record/before_state/value/len_bytes",
        "/stages/observation/record/observed/value/resolved_within_root",
        "/stages/verification/record/comparison/value/expected_len_bytes",
    ] {
        let mut changed = base.clone();
        *changed.pointer_mut(path).unwrap() = serde_json::json!({"availability":"present","value":null,"error_code":null});
        s4a_reject(&changed, RejectionCode::SchemaError);
    }
    for field in ["native_arguments", "detection_payload"] {
        let mut changed = base.clone();
        changed["proposal_context"]["record"][field]["value"] = base["proposal_context"]["record"]["arguments_payload"]["value"].clone();
        changed["proposal_context"]["record"][field]["value"]["store"] = "root-evidence".into();
        s4a_reject(&changed, RejectionCode::SchemaError);
    }
    let mut detection: serde_json::Value = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F74_row.json")).unwrap();
    for field in ["record_digest", "input_digest"] {
        let saved = detection["stages"]["decision"]["record"]["detection"]["value"][field].clone();
        detection["stages"]["decision"]["record"]["detection"]["value"][field] = "bad".into();
        s4a_reject(&detection, RejectionCode::SchemaError);
        detection["stages"]["decision"]["record"]["detection"]["value"][field] = saved;
    }
    for code in ["attempt_ambiguous", "observation_missing", "attempt_errored", "outside_root", "target_missing", "length_mismatch", "digest_mismatch", "before_state_missing", "no_delta_observed"] {
        let mut changed = base.clone(); changed["stages"]["verification"]["record"]["reason_code"] = code.into();
        // R3: check the closed list only; canonical bytes and nullness/outcome
        // correspondence remain for later steps.
        assert_ne!(s2_validate(&serde_json::to_vec(&changed).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
    for (path, code, availability) in [
        ("/stages/attempt/record/before_state", "no_read_access", "unavailable"),
        ("/stages/attempt/record/before_state", "io_error", "errored"),
        ("/stages/observation/record/observed", "no_read_access", "unavailable"),
        ("/stages/observation/record/observed", "io_error", "errored"),
        ("/stages/observation/record/observed", "observer_failed", "errored"),
    ] {
        let mut changed = base.clone(); changed.pointer_mut(path).unwrap()["availability"] = availability.into(); changed.pointer_mut(path).unwrap()["error_code"] = code.into();
        assert_ne!(s2_validate(&serde_json::to_vec(&changed).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

// S4b R2: Value serialization adds only later step-2/3 defects.
#[test]
fn s4b_proposal_vocabulary_and_order() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for (field, bad, allowed) in [
        ("schema_version", "v1", vec!["parapet-effect-receipt/v0"]),
        ("stage", "decision", vec!["proposal"]),
        ("provider", "other", vec!["openai", "anthropic"]),
        ("parse_status", "unknown", vec!["parsed", "errored"]),
        ("effect_kind", "read_file", vec!["write_file", "unsupported"]),
    ] {
        let mut row = base.clone();
        row["stages"]["proposal"]["record"][field] = bad.into();
        s4a_reject(&row, RejectionCode::SchemaError);
        row["proposal_context"]["record"]["choice_index"] = serde_json::json!(0.5);
        s4a_reject(&row, RejectionCode::SchemaError);
        for token in allowed {
            let mut row = base.clone(); row["stages"]["proposal"]["record"][field] = token.into();
            // R3 vocabulary only; derivation remains step 6.
            assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
        }
    }
}

#[test]
fn s4b_proposal_payload_shapes() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for path in [
        "/stages/proposal/record/tool_name/value", "/stages/proposal/record/parse_error_code",
        "/stages/proposal/record/arguments_digest/value", "/stages/proposal/record/arguments_len_bytes/value",
        "/stages/proposal/record/expected_effect/value/target_path", "/stages/proposal/record/expected_effect/value/content_digest",
        "/stages/proposal/record/expected_effect/value/content_len_bytes", "/stages/proposal/record/expected_effect/value/content_media_type",
    ] {
        let mut row = base.clone(); *row.pointer_mut(path).unwrap() = false.into();
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for path in ["/stages/proposal/record/arguments_digest/value", "/stages/proposal/record/expected_effect/value/content_digest"] {
        for bad in ["a".repeat(63), "a".repeat(65), "A".repeat(64), "g".repeat(64)] {
            let mut row = base.clone(); *row.pointer_mut(path).unwrap() = bad.into();
            s4a_reject(&row, RejectionCode::SchemaError);
        }
    }
    for field in ["target_path", "content_digest", "content_len_bytes", "content_media_type"] {
        let mut row = base.clone(); row["stages"]["proposal"]["record"]["expected_effect"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["stages"]["proposal"]["record"]["expected_effect"]["value"]["extra"] = true.into();
    s4a_reject(&row, RejectionCode::SchemaError);
    for bad in ["text/plain", "application/json", "text/plain; charset=UTF-8", ""] {
        let mut row = base.clone(); row["stages"]["proposal"]["record"]["expected_effect"]["value"]["content_media_type"] = bad.into();
        s4a_reject(&row, RejectionCode::SchemaError);
        row["stages"]["proposal"]["record"]["expected_effect"]["availability"] = "absent_by_contract".into();
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4b_proposal_closed_codes() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for (field, allowed) in [
        ("tool_name", vec!["missing_name"]), ("arguments_digest", vec!["parse_error"]), ("arguments_len_bytes", vec!["parse_error"]),
        ("expected_effect", vec!["not_an_object", "missing_path", "missing_content", "wrong_type", "unsupported_argument"]),
        ("parse_error_code", vec!["parse_error", "missing_name", "missing_id", "not_an_object", "buffer_overflow", "delta_timeout", "stream_aborted"]),
    ] {
        for bad in ["unknown".to_string(), "".to_string(), "A".to_string(), "a".repeat(65)] {
            let mut row = base.clone(); let record = &mut row["stages"]["proposal"]["record"];
            if field == "parse_error_code" { record[field] = bad.into(); } else { record[field]["error_code"] = bad.into(); }
            s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
        }
        for code in allowed {
            let mut row = base.clone(); let record = &mut row["stages"]["proposal"]["record"];
            if field == "parse_error_code" { record[field] = code.into(); } else { record[field]["error_code"] = code.into(); }
            // R3 closed-list acceptance only, required/forbidden later.
            assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
        }
    }
}

// Section 7 step-1 witnesses; serialization defects occur only at later steps.
#[test]
fn s4c_context_shape_and_literals() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["schema_version", "stage", "proposal_ref", "model", "tool_call_id", "arguments_wire_form"] {
        let mut row = base.clone();
        row["proposal_context"]["record"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for (field, bad) in [
        ("schema_version", serde_json::json!("v1")),
        ("stage", serde_json::json!("proposal")),
        ("proposal_ref", serde_json::Value::Null),
        ("model", serde_json::json!(false)),
        ("arguments_wire_form", serde_json::json!(false)),
        ("arguments_wire_form", serde_json::json!("unknown")),
    ] {
        let mut row = base.clone(); row["proposal_context"]["record"][field] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for bad in ["a".repeat(63), "a".repeat(65), "A".repeat(64), "g".repeat(64)] {
        let mut row = base.clone(); row["proposal_context"]["record"]["proposal_ref"] = bad.into();
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["proposal_context"]["record"]["extra"] = true.into();
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["proposal_context"]["record"]["tool_call_id"]["value"] = false.into();
    s4a_reject(&row, RejectionCode::SchemaError);
    row["proposal_context"]["record"]["detection_payload"]["error_code"] = "unknown".into();
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4c_context_model_byte_bound_and_wire_vocabulary() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for model in ["a".repeat(257), "é".repeat(129), "界".repeat(86)] {
        let mut row = base.clone(); row["proposal_context"]["record"]["model"] = model.into();
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for model in [String::new(), "a".repeat(256), "é".repeat(128), "界".repeat(85)] {
        let mut row = base.clone(); row["proposal_context"]["record"]["model"] = model.into();
        // R3: only the step-1 bound, not overall validity.
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
    for form in ["string", "object", "null_value", "absent"] {
        let mut row = base.clone(); row["proposal_context"]["record"]["arguments_wire_form"] = form.into();
        // R3: availability correspondence remains a later step.
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

#[test]
fn s4c_context_closed_codes() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for (field, allowed) in [
        ("tool_call_id", vec!["missing_id"]),
        ("native_arguments", vec!["no_string_form", "null_arguments", "absent_arguments"]),
        ("arguments_payload", vec!["store_not_configured", "parse_error"]),
        ("detection_payload", vec!["store_not_configured"]),
    ] {
        for bad in ["unknown".to_string(), String::new(), "A".to_string(), "a".repeat(65)] {
            let mut row = base.clone(); row["proposal_context"]["record"][field]["error_code"] = bad.into();
            s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
        }
        for code in allowed {
            let mut row = base.clone(); row["proposal_context"]["record"][field]["error_code"] = code.into();
            // R3: list membership only; pairing remains step 6.
            assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
        }
        let mut row = base.clone(); row["proposal_context"]["record"][field]["error_code"] = false.into();
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

// S4d rows 155, 193, 194: not a check (policy bytes / declaration truth / policy shape).
// All rejection witnesses introduce decision step-1 defects; Value serialization
// can introduce only later step-2 canonical or step-3 digest defects (R2).
fn s4d_detection_value() -> serde_json::Value {
    serde_json::json!({"record_digest": "a".repeat(64), "record_len_bytes": 1,
        "schema_version": "l1-detection-record/v0", "detection_outcome": "threshold_breached",
        "mode": "block", "input_digest": "b".repeat(64)})
}

#[test]
fn s4d_row_151() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("v1"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["schema_version"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for bad in [serde_json::json!("proposal"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["stage"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_173() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["enforcement"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_178() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["decision"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_179() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("true"), serde_json::Value::Null] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["evaluated_policy"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_182() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("outside_root"), serde_json::json!("size_limit"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["reason_code"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_153() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["contract_hash", "policy_digest", "policy_version"] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["policy"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["policy"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
    row["stages"]["decision"]["record"]["policy"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4d_row_157() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["code_version", "code_digest"] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["code"]["value"] = serde_json::json!({"code_version":"v0", "code_digest":"a".repeat(64)});
        row["stages"]["decision"]["record"]["code"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["code"]["value"] = serde_json::json!({"code_version":"v0", "code_digest":"a".repeat(64)});
        row["stages"]["decision"]["record"]["code"]["value"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
        row["stages"]["decision"]["record"]["code"]["value"] = serde_json::json!({"code_version":"v0", "code_digest":"a".repeat(64)});
    row["stages"]["decision"]["record"]["code"]["value"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    for bad in [serde_json::json!("A".repeat(64)), serde_json::json!("a".repeat(63))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["code"]["value"] = serde_json::json!({"code_version":"v0", "code_digest":"a".repeat(64)});
        row["stages"]["decision"]["record"]["code"]["value"]["code_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_160() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["record_digest", "record_len_bytes", "schema_version", "detection_outcome", "mode", "input_digest"] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
    row["stages"]["decision"]["record"]["detection"]["value"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4d_row_154() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("a".repeat(63)), serde_json::json!("a".repeat(65)), serde_json::json!("A".repeat(64)), serde_json::json!("g".repeat(64))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["policy"]["policy_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_161() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("a".repeat(63)), serde_json::json!("a".repeat(65)), serde_json::json!("A".repeat(64)), serde_json::json!("g".repeat(64))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"]["record_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_166() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("a".repeat(63)), serde_json::json!("a".repeat(65)), serde_json::json!("A".repeat(64)), serde_json::json!("g".repeat(64))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"]["input_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_163() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"]["schema_version"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_164() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"]["detection_outcome"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_165() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["value"] = s4d_detection_value();
        row["stages"]["decision"]["record"]["detection"]["value"]["mode"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_158() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["code"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    let mut row = base.clone();
    row["stages"]["decision"]["record"]["code"]["error_code"] = serde_json::json!("no_build_digest");
    assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
}

#[test]
fn s4d_row_167() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    let mut row = base.clone();
    row["stages"]["decision"]["record"]["detection"]["error_code"] = serde_json::json!("scan_error");
    assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
}

#[test]
fn s4d_row_172() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection_binding_rule"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
}

#[test]
fn s4d_row_170() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["detection_binding_rule"]["value"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_185() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!(false), serde_json::json!([{"argument":false,"constraint":"min"}]), serde_json::json!([{"argument":null,"constraint":"other"}]), serde_json::json!([{"argument":null}]), serde_json::json!([{"argument":null,"constraint":"min","extra":true}])] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["failed_constraints"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_186() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    let sorted = serde_json::json!([{"argument":"content","constraint":"argument_present"}, {"argument":"path","constraint":"max_length"}, {"argument":"path","constraint":"starts_with"}]);
    let mut row = base.clone();
    row["stages"]["decision"]["record"]["failed_constraints"] = sorted.clone();
    row["stages"]["decision"]["record"]["reason_code"] = serde_json::json!("multiple");
    // F66 validator half: set order only; reason derivation is later step 6.
    assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    for bad in [serde_json::json!([sorted[1], sorted[0], sorted[2]]), serde_json::json!([sorted[0], sorted[0]]),
        serde_json::json!([{"argument":"path","constraint":"starts_with"},{"argument":"path","constraint":"max_length"}]),
        serde_json::json!([{"argument":"a","constraint":"min"},{"argument":null,"constraint":"tool_allowed"}]),
        serde_json::json!([{"argument":"é","constraint":"min"},{"argument":"z","constraint":"min"}])] {
        let mut row = base.clone(); row["stages"]["decision"]["record"]["failed_constraints"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for good in [serde_json::json!([]), serde_json::json!([{"argument":null,"constraint":"tool_allowed"}]),
        serde_json::json!([{"argument":null,"constraint":"tool_allowed"},{"argument":"a","constraint":"min"}]),
        serde_json::json!([{"argument":"z","constraint":"min"},{"argument":"é","constraint":"min"}])] {
        let mut row = base.clone(); row["stages"]["decision"]["record"]["failed_constraints"] = good;
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

#[test]
fn s4d_row_191() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!(false), serde_json::json!([false]), serde_json::json!(["other"])] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["inputs_consumed"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4d_row_192() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!(["proposal","policy"]), serde_json::json!(["policy","policy"]), serde_json::json!(["detection","proposal"])] {
        let mut row = base.clone();
        row["stages"]["decision"]["record"]["inputs_consumed"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for good in [serde_json::json!([]), serde_json::json!(["policy"]), serde_json::json!(["proposal"]), serde_json::json!(["detection"]), serde_json::json!(["policy","proposal"]), serde_json::json!(["policy","detection"]), serde_json::json!(["proposal","detection"]), serde_json::json!(["policy","proposal","detection"])] {
        let mut row = base.clone(); row["stages"]["decision"]["record"]["inputs_consumed"] = good;
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

// S4e: section-10 step-1 witnesses; serialization defects are later steps.

#[test]
fn s4e_row_197() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["schema_version", "stage"] {
        for bad in [serde_json::json!("other"), serde_json::json!(false)] {
            let mut row = base.clone(); row["stages"]["attempt"]["record"][field] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
    }
}

#[test]
fn s4e_row_199() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::Value::Null, serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(129)), serde_json::json!("é"), serde_json::json!("a/")] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["action_id"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for good in ["a", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "0._:-z"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["action_id"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

#[test]
fn s4e_row_200() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["name", "version"] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["executor"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["executor"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
    row["stages"]["attempt"]["record"]["executor"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4e_row_201() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["executor"]["name"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_202() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["root_id", "root_digest"] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["root"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["root"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
    row["stages"]["attempt"]["record"]["root"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    for bad in [serde_json::json!("A".repeat(64)), serde_json::json!("a".repeat(63))] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["root"]["root_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_203() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["proposed_path_digest", "target_ref", "resolved_path"] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["target"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["target"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
    row["stages"]["attempt"]["record"]["target"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    for bad in [serde_json::json!("A".repeat(64)), serde_json::json!("a".repeat(63))] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["target"]["proposed_path_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_209() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["operation"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_210() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for slot in ["proposal", "context", "decision"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"].as_object_mut().unwrap().remove(slot);
        s4a_reject(&row, RejectionCode::SchemaError);
        for field in ["status", "ref"] {
            let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"][slot].as_object_mut().unwrap().remove(field);
            s4a_reject(&row, RejectionCode::SchemaError);
        }
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"][slot]["extra"] = serde_json::json!(true);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4e_row_211() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for slot in ["proposal", "context", "decision"] {
        for bad in [serde_json::json!("other"), serde_json::json!(false)] {
            let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"][slot]["status"] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
        for bad in [serde_json::json!("A".repeat(64)), serde_json::json!("a".repeat(63)), serde_json::json!(false)] {
            let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"][slot]["ref"] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
        for good in ["accepted", "rejected", "unavailable"] {
            let mut row = base.clone(); row["stages"]["attempt"]["record"]["sink"][slot]["status"] = serde_json::json!(good);
            row["stages"]["attempt"]["record"]["sink"][slot]["ref"] = serde_json::Value::Null;
            assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
        }
    }
}

#[test]
fn s4e_row_217() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["contract_hash", "policy_digest"] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["policy_check"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["policy_check"][field] = serde_json::json!(false);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone();
    row["stages"]["attempt"]["record"]["policy_check"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    for bad in [serde_json::json!("A".repeat(64)), serde_json::json!("a".repeat(63))] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["policy_check"]["policy_digest"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_220() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["exists", "digest", "len_bytes"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["before_state"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for (field, bad) in [("exists", serde_json::json!("true")), ("digest", serde_json::json!("A".repeat(64))), ("len_bytes", serde_json::json!(false))] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["before_state"]["value"][field] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["stages"]["attempt"]["record"]["before_state"]["value"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["attempt"]["record"]["before_state"]["value"]["len_bytes"] = serde_json::json!(1.5);
    s4a_reject(&row, RejectionCode::NonIntegerNumber);
}

#[test]
fn s4e_row_222() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["before_state"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
}

#[test]
fn s4e_row_223() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for (availability, bad, good) in [("unavailable", "io_error", "no_read_access"), ("errored", "no_read_access", "io_error")] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["before_state"] = serde_json::json!({"availability":availability,"value":null,"error_code":bad});
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
        row["stages"]["attempt"]["record"]["before_state"]["error_code"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

#[test]
fn s4e_row_225() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(false)] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["outcome"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for good in ["completed", "errored", "ambiguous", "rejected"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["outcome"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

#[test]
fn s4e_row_226() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!(false), serde_json::json!({})] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
}

#[test]
fn s4e_row_228() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    for good in ["action_id_collision", "outside_root", "symlink_component", "traversal", "policy_drift", "size_limit", "sink_rejected", "sink_unavailable", "sink_ref_mismatch", "root_unavailable", "io_error", "interrupted", "rename_unconfirmed"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["error_code"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

#[test]
fn s4e_row_231() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("unknown"), serde_json::json!(""), serde_json::json!("A"), serde_json::json!("a".repeat(65))] {
        let mut row = base.clone();
        row["stages"]["attempt"]["record"]["content_payload"]["error_code"] = bad;
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    for good in ["store_not_configured"] {
        let mut row = base.clone(); row["stages"]["attempt"]["record"]["content_payload"]["error_code"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

// S4f: Draft 12 section 11 unconditional observation-record constraints.
#[test]
fn s4f_row_235() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["schema_version", "stage"] {
        for bad in [serde_json::json!("other"), serde_json::json!(false), serde_json::Value::Null] {
            let mut row = base.clone(); row["stages"]["observation"]["record"][field] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
    }
}

#[test]
fn s4f_row_237() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["name", "version", "source"] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observer"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
        for bad in [serde_json::json!(false), serde_json::json!({}), serde_json::Value::Null] {
            let mut row = base.clone(); row["stages"]["observation"]["record"]["observer"][field] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
    }
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observer"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observer"] = serde_json::json!([]);
    s4a_reject(&row, RejectionCode::SchemaError);
}

#[test]
fn s4f_row_238() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in ["other", "", "Filesystem_read"] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observer"]["source"] = serde_json::json!(bad);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    assert_ne!(s2_validate(&serde_json::to_vec(&base).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4f_row_241() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["exists", "digest", "len_bytes", "resolved_within_root"] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for (field, bad) in [("exists", serde_json::json!("true")), ("resolved_within_root", serde_json::json!(0)), ("digest", serde_json::json!("A".repeat(64))), ("digest", serde_json::json!(false)), ("len_bytes", serde_json::json!(false)), ("len_bytes", serde_json::json!(-1))] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"][field] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"] = serde_json::json!([]);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"]["len_bytes"] = serde_json::json!(1.5);
    s4a_reject(&row, RejectionCode::NonIntegerNumber);
    // Null digest/length pairing is owned by step 6, not this shape check.
    let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"]["value"]["digest"] = serde_json::Value::Null;
    row["stages"]["observation"]["record"]["observed"]["value"]["len_bytes"] = serde_json::Value::Null;
    row["stages"]["observation"]["record"]["observed"]["value"]["exists"] = serde_json::json!(false);
    assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4f_row_242() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in ["unknown", "", "A", &"a".repeat(65)] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"] = serde_json::json!({"availability":"unavailable","value":null,"error_code":bad});
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    for (availability, code) in [("unavailable", "no_read_access"), ("errored", "io_error"), ("errored", "observer_failed")] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"] = serde_json::json!({"availability":availability,"value":null,"error_code":code});
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

#[test]
fn s4f_row_243() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for (availability, bad, good) in [("unavailable", "io_error", "no_read_access"), ("unavailable", "observer_failed", "no_read_access"), ("errored", "no_read_access", "io_error")] {
        let mut row = base.clone(); row["stages"]["observation"]["record"]["observed"] = serde_json::json!({"availability":availability,"value":null,"error_code":bad});
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
        row["stages"]["observation"]["record"]["observed"]["error_code"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
}

// S4g: Draft 12 section 12 unconditional verification-record constraints.
#[test]
fn s4g_row_246() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["schema_version", "stage"] {
        for bad in [serde_json::json!("other"), serde_json::json!(false), serde_json::Value::Null] {
            let mut row = base.clone(); row["stages"]["verification"]["record"][field] = bad;
            s4a_reject(&row, RejectionCode::SchemaError);
        }
    }
}

#[test]
fn s4g_row_248() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!(""), serde_json::json!(false), serde_json::Value::Null] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["method"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    assert_ne!(s2_validate(&serde_json::to_vec(&base).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4g_row_249() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!("other"), serde_json::json!("Verified"), serde_json::json!(false), serde_json::Value::Null] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["outcome"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for good in ["verified", "failed", "unknown"] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["outcome"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
}

#[test]
fn s4g_row_252() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for field in ["expected_digest", "observed_digest", "expected_len_bytes", "observed_len_bytes", "within_root", "before_digest", "delta"] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"].as_object_mut().unwrap().remove(field);
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for (field, bad) in [("expected_digest", serde_json::json!(false)), ("expected_digest", serde_json::json!("A".repeat(64))), ("observed_digest", serde_json::json!(false)), ("before_digest", serde_json::json!(false)), ("expected_len_bytes", serde_json::json!(false)), ("expected_len_bytes", serde_json::json!(-1)), ("observed_len_bytes", serde_json::json!(false)), ("within_root", serde_json::json!(0)), ("delta", serde_json::json!(false)), ("delta", serde_json::Value::Null), ("delta", serde_json::json!("other"))] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"][field] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"]["extra"] = serde_json::json!(true);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"] = serde_json::json!([]);
    s4a_reject(&row, RejectionCode::SchemaError);
    let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"]["expected_len_bytes"] = serde_json::json!(1.5);
    s4a_reject(&row, RejectionCode::NonIntegerNumber);
    for good in ["created", "changed", "unchanged", "removed", "unknown"] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["comparison"]["value"]["delta"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
    }
    // Inner nullness consistency is compared by the later step-9 re-run.
    let mut row = base.clone();
    for field in ["observed_digest", "observed_len_bytes", "before_digest"] {
        row["stages"]["verification"]["record"]["comparison"]["value"][field] = serde_json::Value::Null;
    }
    assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s4g_row_254() {
    let base: serde_json::Value = serde_json::from_str(&s3_row_text()).unwrap();
    for bad in [serde_json::json!(false), serde_json::json!(0), serde_json::json!({}), serde_json::json!([])] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["reason_code"] = bad;
        s4a_reject(&row, RejectionCode::SchemaError);
    }
    for bad in ["other", "", "Attempt_ambiguous", &"a".repeat(65)] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["reason_code"] = serde_json::json!(bad);
        s4a_reject(&row, RejectionCode::ErrorCodeNotInClosedList);
    }
    for good in ["attempt_ambiguous", "observation_missing", "attempt_errored", "outside_root", "target_missing", "length_mismatch", "digest_mismatch", "before_state_missing", "no_delta_observed"] {
        let mut row = base.clone(); row["stages"]["verification"]["record"]["reason_code"] = serde_json::json!(good);
        assert_ne!(s2_validate(&serde_json::to_vec(&row).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
    }
    assert_ne!(s2_validate(&serde_json::to_vec(&base).unwrap()), ValidationResult::Rejected(RejectionCode::ErrorCodeNotInClosedList));
}


// Steps 3-5 witnesses start with typed rows and the library serializer. Raw
// slot pairing defects are applied to those bytes, never a Value serialization.
fn s5_base() -> Row {
    serde_json::from_slice(include_bytes!("fixtures/receipt_v0/base_row.json")).unwrap()
}

fn s5_change(row: &mut Row, change: impl FnOnce(&mut Value)) {
    let mut value = serde_json::to_value(&*row).unwrap();
    change(&mut value);
    *row = serde_json::from_value(value).unwrap();
}

fn s5_rechain(row: &mut Row, preserve_ref: &str) {
    row.stages.proposal.digest = record_digest(
        &record_bytes(ReceiptRecord::Proposal(&row.stages.proposal.record)).unwrap(), RecordDomain::Stage);
    row.proposal_id = row.stages.proposal.digest.clone();
    if preserve_ref != "proposal_context" {
        row.proposal_context.record.proposal_ref = row.stages.proposal.digest.clone();
    }
    row.proposal_context.digest = record_digest(
        &record_bytes(ReceiptRecord::ProposalContext(&row.proposal_context.record)).unwrap(), RecordDomain::Stage);
    if preserve_ref != "decision" {
        row.stages.decision.record.proposal_ref = row.stages.proposal.digest.clone();
    }
    row.stages.decision.digest = record_digest(
        &record_bytes(ReceiptRecord::Decision(&row.stages.decision.record)).unwrap(), RecordDomain::Stage);
    if let Slot::Present { digest, record } = &mut row.stages.attempt {
        if preserve_ref != "attempt" { record.decision_ref = row.stages.decision.digest.clone(); }
        record.sink.proposal.r#ref = Some(row.stages.proposal.digest.clone());
        record.sink.context.r#ref = Some(row.proposal_context.digest.clone());
        record.sink.decision.r#ref = Some(row.stages.decision.digest.clone());
        row.action_id = Some(record.action_id.clone());
        *digest = record_digest(&record_bytes(ReceiptRecord::Attempt(record)).unwrap(), RecordDomain::Stage);
    } else { row.action_id = None; }
    if let Slot::Present { digest, record } = &mut row.stages.observation {
        if preserve_ref != "observation" {
            if let Slot::Present { digest, .. } = &row.stages.attempt { record.attempt_ref = digest.clone(); }
        }
        *digest = record_digest(&record_bytes(ReceiptRecord::Observation(record)).unwrap(), RecordDomain::Stage);
    }
    if let Slot::Present { digest, record } = &mut row.stages.verification {
        if preserve_ref != "verification" {
            if let Slot::Present { digest, .. } = &row.stages.observation { record.observation_ref = digest.clone(); }
        }
        *digest = record_digest(&record_bytes(ReceiptRecord::Verification(record)).unwrap(), RecordDomain::Stage);
    }
    row.summary = receipt_summary(row);
}

fn s5_bytes(row: &Row) -> Vec<u8> { record_bytes(ReceiptRecord::Row(row)).unwrap() }

fn s5_absent(row: &mut Row) {
    row.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::NoExecutor);
    row.stages.observation = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
    row.stages.verification = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
    s5_rechain(row, "");
}

fn s5_slot_bytes(row: &Row, name: &str, change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let bytes = String::from_utf8(s5_bytes(row)).unwrap();
    let value: Value = serde_json::from_str(&bytes).unwrap();
    let slot = if name == "proposal_context" { &value[name] } else { &value["stages"][name] };
    let mut edited = slot.clone();
    change(&mut edited);
    // The record remains the exact typed serialization, even when the slot
    // wrapper is not representable by the typed availability sums.
    let typed_record = if slot["record"].is_null() { "null".to_owned() } else {
        let marker = format!("\"digest\":{},\"record\":", slot["digest"]);
        let start = bytes.find(&marker).unwrap() + marker.len();
        let mut depth = 0; let mut string = false; let mut escape = false; let mut end = start;
        for (i, c) in bytes.as_bytes()[start..].iter().enumerate() {
            if string { if escape { escape = false; } else if *c == b'\\' { escape = true; } else if *c == b'"' { string = false; } }
            else if *c == b'"' { string = true; }
            else if *c == b'{' { depth += 1; }
            else if *c == b'}' { depth -= 1; if depth == 0 { end = start + i + 1; break; } }
        }
        bytes[start..end].to_owned()
    };
    let original = format!("{{\"availability\":{},\"reason\":{},\"digest\":{},\"record\":{}}}", slot["availability"], slot["reason"], slot["digest"], typed_record);
    let record = if edited["record"].is_null() { "null" } else { &typed_record };
    let replacement = format!("{{\"availability\":{},\"reason\":{},\"digest\":{},\"record\":{}}}", edited["availability"], edited["reason"], edited["digest"], record);
    let named_original = format!("\"{name}\":{original}");
    let named_replacement = format!("\"{name}\":{replacement}");
    assert!(bytes.contains(&named_original));
    bytes.replacen(&named_original, &named_replacement, 1).into_bytes()
}

fn s5_reject(bytes: &[u8], code: RejectionCode) {
    assert_eq!(s2_validate(bytes), ValidationResult::Rejected(code));
}

#[test]
fn s5_row_32_refs_and_rows_35_125_152_198_236_247_f2() {
    for (name, field) in [("decision", "proposal_ref"), ("attempt", "decision_ref"),
        ("observation", "attempt_ref"), ("verification", "observation_ref"), ("proposal_context", "proposal_ref")] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            let slot = if name == "proposal_context" { &mut v[name] } else { &mut v["stages"][name] };
            slot["record"][field] = json!("0".repeat(64));
        });
        s5_rechain(&mut row, name);
        s5_reject(&s5_bytes(&row), RejectionCode::BrokenStageRef);
    }
    assert_eq!(s2_validate(&s5_bytes(&s5_base())), ValidationResult::Valid);
}

#[test]
fn s5_row_33_and_262_slot_digest_all_six_carried_spans() {
    let row = s5_base();
    for name in ["proposal", "decision", "attempt", "observation", "verification", "proposal_context"] {
        let bytes = s5_slot_bytes(&row, name, |s| s["digest"] = json!("0".repeat(64)));
        s5_reject(&bytes, RejectionCode::BrokenStageRef);
    }
    // A decision digest is checked even when there is no attempt reference.
    let mut no_executor = row;
    s5_absent(&mut no_executor);
    s5_reject(&s5_slot_bytes(&no_executor, "decision", |s| s["digest"] = json!("0".repeat(64))), RejectionCode::BrokenStageRef);
    assert_eq!(s2_validate(&s5_bytes(&no_executor)), ValidationResult::Valid);
}

#[test]
fn s5_row_7_26_275_f1_f33_required_slot_subsets() {
    let mut row = s5_base(); s5_absent(&mut row);
    for name in ["proposal", "decision", "proposal_context"] {
        for availability in ["absent_by_contract", "unavailable", "errored"] {
            s5_reject(&s5_slot_bytes(&row, name, |s| {
                s["availability"] = json!(availability); s["reason"] = Value::Null;
                s["digest"] = Value::Null; s["record"] = Value::Null;
            }), RejectionCode::SlotAvailabilityInvalid);
        }
    }
    // F1 has a present decision above a missing proposal; step 4 wins over L1.
    let row = s5_base();
    s5_reject(&s5_slot_bytes(&row, "proposal", |s| {
        s["availability"] = json!("unavailable"); s["digest"] = Value::Null; s["record"] = Value::Null;
    }), RejectionCode::SlotAvailabilityInvalid);
}

#[test]
fn s5_row_23_nullness_and_bq21_skips_incomplete_and_nonpresent_slots() {
    let row = s5_base();
    for name in ["proposal", "decision", "attempt", "observation", "verification", "proposal_context"] {
        for field in ["digest", "record"] {
            s5_reject(&s5_slot_bytes(&row, name, |s| s[field] = Value::Null), RejectionCode::SlotAvailabilityInvalid);
        }
    }
    // Well-formed carried records on non-present slots are not ER7a locations.
    for field in ["digest", "record"] {
        s5_reject(&s5_slot_bytes(&row, "attempt", |s| {
            s["availability"] = json!("absent_by_contract"); s["reason"] = json!("no_executor");
            if field == "digest" { s["record"] = Value::Null; s["digest"] = json!("0".repeat(64)); }
            else { s["digest"] = Value::Null; }
        }), RejectionCode::SlotAvailabilityInvalid);
    }
}

#[test]
fn s5_row_24_25_reason_pairing_and_slot_sets_f34_null_reason() {
    let row = s5_base();
    for name in ["proposal", "decision", "attempt", "observation", "verification", "proposal_context"] {
        s5_reject(&s5_slot_bytes(&row, name, |s| s["reason"] = json!("blocked")), RejectionCode::SlotAvailabilityInvalid);
    }
    let mut row = row; s5_absent(&mut row);
    for name in ["attempt", "observation", "verification"] {
        s5_reject(&s5_slot_bytes(&row, name, |s| s["reason"] = Value::Null), RejectionCode::SlotAvailabilityInvalid);
        s5_reject(&s5_slot_bytes(&row, name, |s| s["reason"] = json!(if name == "attempt" { "attempt_absent" } else { "no_executor" })), RejectionCode::SlotAvailabilityInvalid);
    }
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
}

#[test]
fn s5_row_27_optional_slot_subsets() {
    let row = s5_base();
    for name in ["attempt", "observation", "verification"] {
        for availability in ["unavailable", "errored"] {
            s5_reject(&s5_slot_bytes(&row, name, |s| {
                s["availability"] = json!(availability); s["reason"] = json!(if name == "attempt" { "no_executor" } else { "attempt_absent" });
                s["digest"] = Value::Null; s["record"] = Value::Null;
            }), RejectionCode::SlotAvailabilityInvalid);
        }
    }
}

#[test]
fn s5_row_31_42_absent_referents_l1_before_l5() {
    let mut row = s5_base();
    row.stages.observation = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::StageOrderViolation);
    let mut row = s5_base();
    row.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::NoExecutor);
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::StageOrderViolation);
    // All three L3 defects coexist with L1; L1 remains first.
    s5_change(&mut row, |v| { v["stages"]["decision"]["record"]["decision"] = json!("block"); });
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::StageOrderViolation);
}

#[test]
fn s5_row_36_f3_l3_allow_first() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["decision"] = json!("block"));
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptWithoutAllow);
    s5_change(&mut row, |v| { v["stages"]["proposal"]["record"]["effect_kind"] = json!("unsupported");
        v["stages"]["decision"]["record"]["enforcement"] = json!("shadow"); });
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptWithoutAllow);
}

#[test]
fn s5_row_37_f16_l3_unsupported_before_enforcement() {
    let mut row = s5_base();
    s5_change(&mut row, |v| { v["stages"]["proposal"]["record"]["effect_kind"] = json!("unsupported");
        v["stages"]["proposal"]["record"]["expected_effect"] = json!({"availability":"absent_by_contract","value":null,"error_code":null}); });
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptForUnsupported);
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["enforcement"] = json!("shadow"));
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptForUnsupported);
}

#[test]
fn s5_row_38_f53_attempt_without_enforcement() {
    let mut row = s5_base();
    s5_change(&mut row, |v| { let d = &mut v["stages"]["decision"]["record"];
        d["enforcement"] = json!("not_evaluated"); d["evaluated_policy"] = json!(false); d["reason_code"] = json!("policy_layer_off"); });
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptWithoutEnforcement);
}

#[test]
fn s5_row_39_evaluated_policy_has_no_l3_code() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["evaluated_policy"] = json!(false));
    s5_rechain(&mut row, "");
    for code in [RejectionCode::AttemptWithoutAllow, RejectionCode::AttemptForUnsupported, RejectionCode::AttemptWithoutEnforcement] {
        assert_ne!(s2_validate(&s5_bytes(&row)), ValidationResult::Rejected(code));
    }
}

#[test]
fn s5_row_41_f22_rejected_attempt_with_observation() {
    let mut row = s5_base();
    s5_change(&mut row, |v| { let a = &mut v["stages"]["attempt"]["record"];
        a["outcome"] = json!("rejected"); a["error_code"] = json!("action_id_collision");
        for f in ["before_state", "content_payload"] { a[f] = json!({"availability":"absent_by_contract","value":null,"error_code":null}); }
    });
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::ObservationAfterRejectedAttempt);
}

#[test]
fn s5_row_43_44_52_280_f32_missing_post_attempt_stages() {
    for outcome in ["completed", "errored", "ambiguous"] {
        let mut row = s5_base();
        s5_change(&mut row, |v| { let a = &mut v["stages"]["attempt"]["record"];
            a["outcome"] = json!(outcome); a["error_code"] = if outcome == "completed" { Value::Null } else { json!("io_error") }; });
        row.stages.observation = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
        row.stages.verification = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
        s5_rechain(&mut row, "");
        s5_reject(&s5_bytes(&row), RejectionCode::ObservationMissingAfterAttempt);
        let base = s5_base(); row.stages.observation = base.stages.observation;
        s5_rechain(&mut row, "");
        s5_reject(&s5_bytes(&row), RejectionCode::VerificationMissingAfterAttempt);
    }
}

#[test]
fn s5_row_48_f23_f24_verified_requires_completed_and_pass_pairs() {
    for (outcome, error) in [("ambiguous", "rename_unconfirmed"), ("errored", "io_error")] {
        let mut row = s5_base();
        s5_change(&mut row, |v| { let a = &mut v["stages"]["attempt"]["record"];
            a["outcome"] = json!(outcome); a["error_code"] = json!(error);
            if outcome == "errored" { a["before_state"]["value"] = json!({"exists":true,
                "digest":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824","len_bytes":5}); }
        });
        s5_rechain(&mut row, "");
        s5_reject(&s5_bytes(&row), RejectionCode::VerifiedWithoutCompletedAttempt);
        if let (Slot::Present { record: a, .. }, Slot::Present { record: o, .. }, Slot::Present { record: v, .. }) =
            (&row.stages.attempt, &row.stages.observation, &mut row.stages.verification) {
            let method = verify_expected_effect(&row.stages.proposal.record, a, o).unwrap();
            v.outcome = method.outcome; v.reason_code = method.reason_code; v.comparison = method.comparison;
        }
        s5_rechain(&mut row, "");
        assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
    }
}

#[test]
fn s5_row_53_54_f31_row_identity() {
    let mut row = s5_base(); row.proposal_id = "0".repeat(64);
    s5_reject(&s5_bytes(&row), RejectionCode::RowIdentityMismatch);
    let mut row = s5_base(); row.action_id = Some("other".into());
    s5_reject(&s5_bytes(&row), RejectionCode::RowIdentityMismatch);
    let mut row = s5_base(); s5_absent(&mut row); row.action_id = Some("other".into());
    s5_reject(&s5_bytes(&row), RejectionCode::RowIdentityMismatch);
}

#[test]
fn s5_f34_no_executor_pass() {
    let mut row = s5_base(); s5_absent(&mut row);
    assert_eq!(row.summary, Summary::NotAttemptedNoExecutor);
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
}

#[test]
fn s5_f53_pass_and_later_enforcement_defect() {
    let mut row = s5_base(); s5_absent(&mut row);
    s5_change(&mut row, |v| { let d = &mut v["stages"]["decision"]["record"];
        d["enforcement"] = json!("not_evaluated"); d["evaluated_policy"] = json!(false); d["reason_code"] = json!("policy_layer_off"); });
    row.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::NotEnforced);
    s5_rechain(&mut row, "");
    assert_eq!(row.summary, Summary::NotAttemptedNotEnforced);
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
    // The fixture's enforced-only variant belongs to step 6, still unbuilt.
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["enforcement"] = json!("enforced"));
    s5_rechain(&mut row, "");
    assert_ne!(s2_validate(&s5_bytes(&row)), ValidationResult::Rejected(RejectionCode::SlotAvailabilityInvalid));
}

#[test]
fn s5_o2_noncanonical_before_digest() {
    let row = s5_base();
    let broken = s5_slot_bytes(&row, "proposal", |s| s["digest"] = json!("0".repeat(64)));
    s5_reject(&broken, RejectionCode::BrokenStageRef);
    for bytes in [s5_bytes(&row), broken] {
        let text = String::from_utf8(bytes).unwrap().replacen("\"stage\":\"proposal\",", "\"stage\":\"proposal\", ", 1);
        s5_reject(text.as_bytes(), RejectionCode::NonCanonicalBytes);
    }
}

#[test]
fn s5_o3_digest_before_context_subset() {
    let row = s5_base();
    let absent = s5_slot_bytes(&row, "proposal_context", |s| {
        s["availability"] = json!("absent_by_contract"); s["reason"] = Value::Null; s["digest"] = Value::Null; s["record"] = Value::Null;
    });
    s5_reject(&absent, RejectionCode::SlotAvailabilityInvalid);
    let broken = s5_slot_bytes(&row, "proposal", |s| s["digest"] = json!("0".repeat(64)));
    s5_reject(&broken, RejectionCode::BrokenStageRef);
    let text = String::from_utf8(absent).unwrap().replacen(&format!("\"digest\":\"{}\"", row.stages.proposal.digest), &format!("\"digest\":\"{}\"", "0".repeat(64)), 1);
    s5_reject(text.as_bytes(), RejectionCode::BrokenStageRef);
}

#[test]
fn s5_o3b_late_context_digest_before_early_proposal_reason() {
    let row = s5_base();
    let reason = s5_slot_bytes(&row, "proposal", |s| s["reason"] = json!("blocked"));
    s5_reject(&reason, RejectionCode::SlotAvailabilityInvalid);
    let broken = s5_slot_bytes(&row, "proposal_context", |s| s["digest"] = json!("0".repeat(64)));
    s5_reject(&broken, RejectionCode::BrokenStageRef);
    let text = String::from_utf8(broken).unwrap().replacen("\"reason\":null", "\"reason\":\"blocked\"", 1);
    s5_reject(text.as_bytes(), RejectionCode::BrokenStageRef);
}

#[test]
fn s5_o4_slot_shape_before_lifecycle() {
    let mut row = s5_base();
    let reason = s5_slot_bytes(&row, "proposal", |s| s["reason"] = json!("blocked"));
    s5_reject(&reason, RejectionCode::SlotAvailabilityInvalid);
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["decision"] = json!("block"));
    s5_rechain(&mut row, "");
    s5_reject(&s5_bytes(&row), RejectionCode::AttemptWithoutAllow);
    s5_reject(&s5_slot_bytes(&row, "proposal", |s| s["reason"] = json!("blocked")), RejectionCode::SlotAvailabilityInvalid);
}

// S6 raw witnesses preserve table-order serialization and repair the complete
// carried-byte digest chain. This also permits defects the typed sums exclude.
fn s6_end(text: &str, start: usize) -> usize {
    let mut depth = 0usize;
    let mut string = false;
    let mut escape = false;
    for (offset, b) in text.as_bytes()[start..].iter().copied().enumerate() {
        if string {
            if escape { escape = false; }
            else if b == b'\\' { escape = true; }
            else if b == b'"' { string = false; if depth == 0 { return start + offset + 1; } }
        } else {
            match b {
                b'"' => string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' if depth > 0 => { depth -= 1; if depth == 0 { return start + offset + 1; } },
                b',' | b'}' | b']' if depth == 0 => return start + offset,
                _ => (),
            }
        }
    }
    text.len()
}

fn s6_span(text: &str, path: &str) -> (usize, usize) {
    let mut start = 0;
    let mut end = text.len();
    for name in path.split('.') {
        assert_eq!(&text[start..start+1], "{");
        let mut cursor = start + 1;
        loop {
            let key_end = s6_end(text, cursor);
            let key: String = serde_json::from_str(&text[cursor..key_end]).unwrap();
            let value_start = key_end + 1;
            let value_end = s6_end(text, value_start);
            if key == name { start = value_start; end = value_end; break; }
            cursor = value_end + 1;
            assert!(cursor < end, "missing {path}");
        }
    }
    (start, end)
}

fn s6_set(text: &mut String, path: &str, value: Value) {
    let (start, end) = s6_span(text, path);
    text.replace_range(start..end, &serde_json::to_string(&value).unwrap());
}

fn s6_record_path(name: &str) -> String {
    if name == "proposal_context" { format!("{name}.record") }
    else { format!("stages.{name}.record") }
}

fn s6_field(text: &mut String, name: &str, field: &str, availability: &str, code: Value) {
    let path = format!("{}.{}", s6_record_path(name), field);
    s6_set(text, &format!("{path}.availability"), json!(availability));
    if availability != "present" { s6_set(text, &format!("{path}.value"), Value::Null); }
    s6_set(text, &format!("{path}.error_code"), code);
}

fn s6_repair(text: &mut String) {
    let mut digests = std::collections::BTreeMap::<&str, String>::new();
    for name in ["proposal", "proposal_context", "decision", "attempt", "observation", "verification"] {
        let path = s6_record_path(name);
        let (start, end) = s6_span(text, &path);
        if &text[start..end] == "null" { continue; }
        let referent = match name {
            "proposal_context" | "decision" => Some(("proposal_ref", "proposal")),
            "attempt" => Some(("decision_ref", "decision")),
            "observation" => Some(("attempt_ref", "attempt")),
            "verification" => Some(("observation_ref", "observation")),
            _ => None,
        };
        if let Some((field, referent)) = referent {
            if let Some(digest) = digests.get(referent) { s6_set(text, &format!("{path}.{field}"), json!(digest)); }
        }
        if name == "attempt" {
            for (field, referent) in [("proposal", "proposal"), ("context", "proposal_context"), ("decision", "decision")] {
                s6_set(text, &format!("{path}.sink.{field}.ref"), json!(digests[referent]));
            }
        }
        let (start, end) = s6_span(text, &path);
        let digest = record_digest(text[start..end].as_bytes(), RecordDomain::Stage);
        let slot = path.strip_suffix(".record").unwrap();
        s6_set(text, &format!("{slot}.digest"), json!(digest));
        if name == "proposal" { s6_set(text, "proposal_id", json!(digest)); }
        digests.insert(name, digest);
    }
}

fn s6_reject(mut text: String, code: RejectionCode) {
    s6_repair(&mut text);
    assert_eq!(s2_validate(text.as_bytes()), ValidationResult::Rejected(code));
}

fn s6_text() -> String { String::from_utf8(s5_bytes(&s5_base())).unwrap() }

fn s6_errored() -> String {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        v["stages"]["attempt"]["record"]["outcome"] = json!("errored");
        v["stages"]["attempt"]["record"]["error_code"] = json!("io_error");
        v["stages"]["verification"]["record"]["outcome"] = json!("failed");
        v["stages"]["verification"]["record"]["reason_code"] = json!("attempt_errored");
    });
    s5_rechain(&mut row, "");
    String::from_utf8(s5_bytes(&row)).unwrap()
}

#[test]
fn s6_row_57() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "request_ref", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_92() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "tool_name", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_101() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "arguments_digest", "unavailable", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_106() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "arguments_len_bytes", "unavailable", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_113() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "expected_effect", "unavailable", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_129() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "tool_call_id", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_132() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "native_arguments", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_135() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "arguments_payload", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_142() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "detection_payload", "errored", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_156() {
    let mut text = s6_text();
    s6_field(&mut text, "decision", "code", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_159() {
    let mut text = s6_text();
    s6_field(&mut text, "decision", "detection", "errored", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_169() {
    let mut text = s6_text();
    s6_field(&mut text, "decision", "detection_binding_rule", "unavailable", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_204() {
    let mut text = s6_text();
    s6_field(&mut text, "attempt", "target.target_ref", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_206() {
    let mut text = s6_text();
    s6_field(&mut text, "attempt", "target.resolved_path", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_230() {
    let mut text = s6_text();
    s6_field(&mut text, "attempt", "content_payload", "errored", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_239() {
    let mut text = s6_text();
    s6_field(&mut text, "observation", "target_ref", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_240() {
    let mut text = s6_text();
    s6_field(&mut text, "observation", "observed", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_244() {
    let mut text = s6_text();
    s6_field(&mut text, "observation", "footprint", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_250() {
    let mut text = s6_text();
    s6_field(&mut text, "verification", "comparison", "unavailable", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_6_all_field_subsets() {
    for (name, field, availability) in [
        ("proposal", "tool_name", "errored"),
        ("proposal", "arguments_digest", "unavailable"),
        ("proposal", "arguments_len_bytes", "unavailable"),
        ("proposal", "expected_effect", "unavailable"),
        ("decision", "code", "errored"),
        ("decision", "detection", "errored"),
        ("decision", "detection_binding_rule", "unavailable"),
        ("attempt", "target.target_ref", "errored"),
        ("attempt", "target.resolved_path", "errored"),
        ("attempt", "content_payload", "errored"),
        ("observation", "target_ref", "errored"),
        ("observation", "footprint", "errored"),
        ("verification", "comparison", "unavailable"),
        ("proposal_context", "request_ref", "errored"),
        ("proposal_context", "tool_call_id", "errored"),
        ("proposal_context", "native_arguments", "errored"),
        ("proposal_context", "arguments_payload", "errored"),
        ("proposal_context", "detection_payload", "errored")
    ] {
        let mut text = s6_text();
        s6_field(&mut text, name, field, availability, Value::Null);
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
}

#[test]
fn s6_row_12_value_both_directions_all_locations() {
    for (name, field) in [
        ("proposal", "tool_name"),
        ("proposal", "arguments_digest"),
        ("proposal", "arguments_len_bytes"),
        ("proposal", "expected_effect"),
        ("decision", "code"),
        ("decision", "detection"),
        ("decision", "detection_binding_rule"),
        ("attempt", "target.target_ref"),
        ("attempt", "target.resolved_path"),
        ("attempt", "before_state"),
        ("attempt", "content_payload"),
        ("observation", "target_ref"),
        ("observation", "observed"),
        ("observation", "footprint"),
        ("verification", "comparison"),
        ("proposal_context", "request_ref"),
        ("proposal_context", "tool_call_id"),
        ("proposal_context", "native_arguments"),
        ("proposal_context", "arguments_payload"),
        ("proposal_context", "detection_payload")
    ] {
        let mut text = s6_text();
        s6_field(&mut text, name, field, "present", Value::Null);
        s6_set(&mut text, &format!("{}.{}.value", s6_record_path(name), field), Value::Null);
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
    let mut text = s6_text();
    s6_set(&mut text, "stages.proposal.record.tool_name.availability", json!("unavailable"));
    s6_set(&mut text, "stages.proposal.record.tool_name.error_code", json!("missing_name"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_13_present_code_forbidden() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.proposal.record.tool_name.error_code", json!("missing_name"));
    s6_reject(text, RejectionCode::ErrorCodeForbidden);
}

#[test]
fn s6_row_14_missing_code() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.observation.record.footprint.error_code", Value::Null);
    s6_reject(text, RejectionCode::ErrorCodeRequired);
}

#[test]
fn s6_row_15_absent_code_forbidden() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.decision.record.detection.error_code", json!("scan_error"));
    s6_reject(text, RejectionCode::ErrorCodeForbidden);
}

#[test]
fn s6_row_95_parsed_tool_name() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "tool_name", "unavailable", json!("missing_name"));
    s6_reject(text, RejectionCode::ToolNameRequiredWhenParsed);
}

#[test]
fn s6_row_100_parse_code_both_directions() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.proposal.record.parse_error_code", json!("parse_error"));
    s6_reject(text, RejectionCode::ErrorCodeForbidden);
    let mut row = s5_base();
    s5_absent(&mut row);
    let mut text = String::from_utf8(s5_bytes(&row)).unwrap();
    s6_set(&mut text, "stages.proposal.record.parse_status", json!("errored"));
    s6_field(&mut text, "proposal", "arguments_digest", "errored", json!("parse_error"));
    s6_field(&mut text, "proposal", "arguments_len_bytes", "errored", json!("parse_error"));
    s6_reject(text, RejectionCode::ErrorCodeRequired);
}

#[test]
fn s6_row_103_parse_pair_both_directions() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.proposal.record.parse_status", json!("errored"));
    s6_set(&mut text, "stages.proposal.record.parse_error_code", json!("parse_error"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "arguments_digest", "errored", json!("parse_error"));
    s6_field(&mut text, "proposal", "arguments_len_bytes", "errored", json!("parse_error"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_108_length_digest_pair() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "arguments_len_bytes", "errored", json!("parse_error"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    let mut text = s6_text();
    s6_set(&mut text, "stages.proposal.record.arguments_digest.error_code", json!("parse_error"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_114_effect_pair_both_directions() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal", "expected_effect", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    let mut row = s5_base(); s5_absent(&mut row);
    let mut text = String::from_utf8(s5_bytes(&row)).unwrap();
    s6_set(&mut text, "stages.proposal.record.effect_kind", json!("unsupported"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_133_native_pair_both_directions() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "native_arguments", "unavailable", json!("no_string_form"));
    s6_reject(text, RejectionCode::NativeArgumentsAvailabilityMismatch);
    let mut text = s6_text();
    s6_set(&mut text, "proposal_context.record.arguments_wire_form", json!("absent"));
    s6_reject(text, RejectionCode::NativeArgumentsAvailabilityMismatch);
}

#[test]
fn s6_row_143_detection_payload_pair() {
    let mut text = s6_text();
    s6_field(&mut text, "proposal_context", "detection_payload", "unavailable", json!("store_not_configured"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_171_detection_binding_pair() {
    let mut text = s6_text();
    s6_field(&mut text, "decision", "detection", "unavailable", json!("scan_error"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_207_target_pair_both_directions() {
    for field in ["target.target_ref", "target.resolved_path"] {
        let mut text = s6_text();
        s6_field(&mut text, "attempt", field, "unavailable", json!("not_resolved"));
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
    let mut text = s6_text();
    s6_set(&mut text, "stages.attempt.record.target.resolved_path.error_code", json!("not_resolved"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_219_full_subset_redacted_step_1() {
    let mut text = s6_text();
    s6_field(&mut text, "attempt", "before_state", "redacted", Value::Null);
    s6_reject(text, RejectionCode::RedactedInCanonical);
}

#[test]
fn s6_row_227_attempt_code_both_directions() {
    let mut text = s6_errored();
    s6_set(&mut text, "stages.attempt.record.error_code", Value::Null);
    s6_reject(text, RejectionCode::ErrorCodeRequired);
    let mut text = s6_text();
    s6_set(&mut text, "stages.attempt.record.error_code", json!("io_error"));
    s6_reject(text, RejectionCode::ErrorCodeForbidden);
}

#[test]
fn s6_row_251_comparison_pair() {
    let mut text = s6_text();
    s6_field(&mut text, "verification", "comparison", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    let mut text = s6_text();
    s6_field(&mut text, "observation", "observed", "unavailable", json!("no_read_access"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
}

#[test]
fn s6_row_255_reason_both_directions() {
    let mut text = s6_errored();
    s6_set(&mut text, "stages.verification.record.reason_code", Value::Null);
    s6_reject(text, RejectionCode::ReasonCodeRequired);
    let mut text = s6_text();
    s6_set(&mut text, "stages.verification.record.reason_code", json!("digest_mismatch"));
    s6_reject(text, RejectionCode::ReasonCodeForbidden);
}

#[test]
fn s6_o5_step_5_precedes_step_6() {
    let mut early = s6_text();
    s6_set(&mut early, "stages.decision.record.decision", json!("block"));
    s6_reject(early, RejectionCode::AttemptWithoutAllow);
    let mut late = s6_text();
    s6_field(&mut late, "proposal", "tool_name", "absent_by_contract", Value::Null);
    s6_reject(late, RejectionCode::FieldAvailabilityInvalid);
    let mut both = s6_text();
    s6_set(&mut both, "stages.decision.record.decision", json!("block"));
    s6_field(&mut both, "proposal", "tool_name", "absent_by_contract", Value::Null);
    s6_reject(both, RejectionCode::AttemptWithoutAllow);
}

#[test]
fn s6_t2_rule_1_sweeps_before_rule_2() {
    let mut early = s6_errored();
    s6_field(&mut early, "verification", "comparison", "absent_by_contract", Value::Null);
    s6_reject(early, RejectionCode::FieldAvailabilityInvalid);
    let mut late = s6_errored();
    s6_set(&mut late, "stages.attempt.record.error_code", Value::Null);
    s6_reject(late, RejectionCode::ErrorCodeRequired);
    let mut both = s6_errored();
    s6_field(&mut both, "verification", "comparison", "absent_by_contract", Value::Null);
    s6_set(&mut both, "stages.attempt.record.error_code", Value::Null);
    s6_reject(both, RejectionCode::FieldAvailabilityInvalid);
}

fn s6_method(row: &mut Row) {
    if let (Slot::Present { record: a, .. }, Slot::Present { record: o, .. }, Slot::Present { record: v, .. }) =
        (&row.stages.attempt, &row.stages.observation, &mut row.stages.verification) {
        let method = verify_expected_effect(&row.stages.proposal.record, a, o).unwrap();
        v.outcome = method.outcome; v.reason_code = method.reason_code; v.comparison = method.comparison;
    }
    s5_rechain(row, "");
}

fn s6_detection() -> Row {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        v["stages"]["decision"]["record"]["detection"] = json!({"availability":"present", "value": {
            "record_digest":"0".repeat(64), "record_len_bytes":0, "schema_version":"l1-detection-record/v0",
            "detection_outcome":"threshold_not_breached", "mode":"shadow", "input_digest":"0".repeat(64)
        }, "error_code":null});
        v["stages"]["decision"]["record"]["detection_binding_rule"] = json!({"availability":"present",
            "value":"last_untrusted_user_message_before_l0/v0", "error_code":null});
        v["proposal_context"]["record"]["detection_payload"] = json!({"availability":"unavailable", "value":null,
            "error_code":"store_not_configured"});
    });
    s5_rechain(&mut row, ""); row
}

#[test]
fn s6_f11_observation_unavailable() {
    let mut row = s5_base();
    s5_change(&mut row, |v| { v["stages"]["observation"]["record"]["observed"] =
        json!({"availability":"unavailable", "value":null, "error_code":"no_read_access"}); });
    s6_method(&mut row);
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
    // The verified variant's verification_inconsistent outcome belongs to S9.
}

#[test]
fn s6_f14_code_unavailable() {
    let row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F14_row.json")).unwrap();
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
    // ER7b indeterminate is outside byte validation.
}

#[test]
fn s6_f35_attempt_codes() { s6_row_227_attempt_code_both_directions(); }

#[test]
fn s6_f36_reason_required() {
    let mut text = s6_errored();
    s6_set(&mut text, "stages.verification.record.reason_code", Value::Null);
    s6_reject(text, RejectionCode::ReasonCodeRequired);
}

#[test]
fn s6_f38_wire_forms() {
    s6_row_133_native_pair_both_directions();
    let forms: [&[u8]; 5] = [include_bytes!("fixtures/receipt_v0/F38_form_0.json"),
        include_bytes!("fixtures/receipt_v0/F38_form_1.json"), include_bytes!("fixtures/receipt_v0/F38_form_2.json"),
        include_bytes!("fixtures/receipt_v0/F38_form_3.json"), include_bytes!("fixtures/receipt_v0/F38_form_4.json")];
    for (i, bytes) in forms.into_iter().enumerate() {
        let row: Row = serde_json::from_slice(bytes).unwrap();
        assert_eq!(row.stages.proposal.record.arguments_digest, PresentErrored::Present(arguments_digest(b"{}")));
        assert_eq!(matches!(row.proposal_context.record.native_arguments, PresentUnavailable::Present(_)), i == 0 || i == 3);
    }
    let wire: Value = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F38_wire.json")).unwrap();
    for case in wire.as_array().unwrap() { assert_eq!(arguments_digest(&wire_bytes(case)), arguments_digest(b"{}")); }
    let row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F38_mismatch_row.json")).unwrap();
    s6_reject(String::from_utf8(s5_bytes(&row)).unwrap(), RejectionCode::NativeArgumentsAvailabilityMismatch);
}

#[test]
fn s6_f49_detection_binding_both_directions() {
    let mut text = s6_text();
    s6_set(&mut text, "stages.decision.record.detection_binding_rule.availability", json!("present"));
    s6_set(&mut text, "stages.decision.record.detection_binding_rule.value", json!("last_untrusted_user_message_before_l0/v0"));
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    for availability in ["present", "unavailable"] {
        let mut text = String::from_utf8(s5_bytes(&s6_detection())).unwrap();
        if availability == "unavailable" { s6_field(&mut text, "decision", "detection", "unavailable", json!("scan_error")); }
        s6_field(&mut text, "decision", "detection_binding_rule", "absent_by_contract", Value::Null);
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
}

#[test]
fn s6_f57_target_availability() {
    s6_row_207_target_pair_both_directions();
    // Three target-binding outcomes are step 8, owned by S8.
}

#[test]
fn s6_f63_rejected_resolution_paths() {
    for (path, code) in [("../x", "traversal"), ("link/x", "symlink_component"), ("/outside/x", "outside_root")] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            v["stages"]["proposal"]["record"]["expected_effect"]["value"]["target_path"] = json!(path);
            let a = &mut v["stages"]["attempt"]["record"];
            a["target"]["proposed_path_digest"] = json!(proposed_path_digest(path));
            for field in ["target_ref", "resolved_path"] { a["target"][field] =
                json!({"availability":"unavailable", "value":null, "error_code":"not_resolved"}); }
            for field in ["before_state", "content_payload"] { a[field] =
                json!({"availability":"absent_by_contract", "value":null, "error_code":null}); }
            a["outcome"] = json!("rejected"); a["error_code"] = json!(code);
        });
        row.stages.observation = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected);
        row.stages.verification = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected);
        s5_rechain(&mut row, "");
        assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
        let mut text = String::from_utf8(s5_bytes(&row)).unwrap();
        s6_set(&mut text, "stages.attempt.record.target.resolved_path.availability", json!("present"));
        s6_set(&mut text, "stages.attempt.record.target.resolved_path.value", json!("x"));
        s6_set(&mut text, "stages.attempt.record.target.resolved_path.error_code", Value::Null);
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
    // Present target_ref in pre-resolution rejection: S7 rule 11, not asserted here.
}

#[test]
fn s6_f76_detection_payload_pairings() {
    assert_eq!(s2_validate(&s5_bytes(&s5_base())), ValidationResult::Valid);
    assert_eq!(s2_validate(&s5_bytes(&s6_detection())), ValidationResult::Valid);
    let payload: Value = serde_json::to_value(s5_base().proposal_context.record.arguments_payload).unwrap();
    for detection in ["absent_by_contract", "unavailable"] {
        let mut text = s6_text();
        if detection == "unavailable" {
            s6_field(&mut text, "decision", "detection", "unavailable", json!("scan_error"));
            s6_set(&mut text, "stages.decision.record.detection_binding_rule.availability", json!("present"));
            s6_set(&mut text, "stages.decision.record.detection_binding_rule.value", json!("last_untrusted_user_message_before_l0/v0"));
        }
        // Edit components separately to retain canonical payload-ref key order.
        let value = &payload["value"];
        let (start, end) = s6_span(&text, "proposal_context.record.arguments_payload.value");
        let carried = text[start..end].to_owned();
        s6_set(&mut text, "proposal_context.record.detection_payload.availability", json!("present"));
        let (start, end) = s6_span(&text, "proposal_context.record.detection_payload.value");
        text.replace_range(start..end, &carried);
        assert!(value.is_object());
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
    let mut text = String::from_utf8(s5_bytes(&s6_detection())).unwrap();
    s6_field(&mut text, "proposal_context", "detection_payload", "absent_by_contract", Value::Null);
    s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    // Replayer sentence belongs to 1A.4; unavailable resolver continuation to S8.
}

#[test]
fn s6_f77_both_absent_comparison() {
    for errored in [false, true] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            let o = &mut v["stages"]["observation"]["record"]["observed"]["value"];
            o["exists"] = json!(false); o["digest"] = Value::Null; o["len_bytes"] = Value::Null;
            if errored {
                v["stages"]["attempt"]["record"]["outcome"] = json!("errored");
                v["stages"]["attempt"]["record"]["error_code"] = json!("io_error");
            }
        });
        s6_method(&mut row);
        assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Valid);
        let mut text = String::from_utf8(s5_bytes(&row)).unwrap();
        s6_field(&mut text, "verification", "comparison", "absent_by_contract", Value::Null);
        s6_reject(text, RejectionCode::FieldAvailabilityInvalid);
    }
    // Unknown/created/removed delta forgery: step 9, owned by S9.
}

// S7 witnesses use the unchanged S5/S6 builders and raw canonical repair.
fn s7_set(text: &mut String, stage: &str, field: &str, value: Value) {
    s6_set(text, &format!("{}.{}", s6_record_path(stage), field), value);
}
fn s7_check(mut text: String, code: RejectionCode) {
    s6_repair(&mut text);
    assert_eq!(s2_validate(text.as_bytes()), ValidationResult::Rejected(code));
}
// R3: an intermediate step-6 acceptance; no assertion of whole-row validity.
fn s7_accept(mut text: String) {
    s6_repair(&mut text);
    let result = s2_validate(text.as_bytes());
    for code in [RejectionCode::EnforcementInconsistent, RejectionCode::ErroredProposalAllowed,
        RejectionCode::SchemaError, RejectionCode::RejectedAttemptShapeInvalid,
        RejectionCode::SlotAvailabilityInvalid] {
        assert_ne!(result, ValidationResult::Rejected(code));
    }
}
fn s7_absent(enforcement: &str, evaluated: bool, reason: &str) -> String {
    let mut row = s5_base(); s5_absent(&mut row);
    s5_change(&mut row, |v| {
        let d = &mut v["stages"]["decision"]["record"];
        d["enforcement"] = json!(enforcement); d["evaluated_policy"] = json!(evaluated);
        d["reason_code"] = json!(reason);
    });
    row.stages.attempt = Slot::AbsentByContract(if enforcement == "enforced" {
        AttemptAbsentReason::NoExecutor
    } else { AttemptAbsentReason::NotEnforced });
    s5_rechain(&mut row, ""); String::from_utf8(s5_bytes(&row)).unwrap()
}
fn s7_block() -> String {
    let mut text = s7_absent("enforced", true, "max_length");
    s7_set(&mut text, "decision", "decision", json!("block"));
    s7_set(&mut text, "decision", "failed_constraints", json!([{ "argument":"path", "constraint":"max_length" }]));
    s6_set(&mut text, "stages.attempt.reason", json!("blocked")); text
}
fn s7_proposal_error(parse: bool, allow: bool) -> String {
    let mut text = s7_block();
    s7_set(&mut text, "decision", "failed_constraints", json!([]));
    s7_set(&mut text, "decision", "evaluated_policy", json!(false));
    s7_set(&mut text, "decision", "reason_code", json!(if parse { "proposal_unparsable" } else { "proposal_shape_invalid" }));
    if parse {
        s7_set(&mut text, "proposal", "parse_status", json!("errored"));
        s7_set(&mut text, "proposal", "parse_error_code", json!("missing_name"));
        s6_field(&mut text, "proposal", "tool_name", "unavailable", json!("missing_name"));
        for f in ["arguments_digest", "arguments_len_bytes"] { s6_field(&mut text, "proposal", f, "errored", json!("parse_error")); }
        s7_set(&mut text, "proposal", "effect_kind", json!("unsupported"));
        s6_field(&mut text, "proposal", "expected_effect", "absent_by_contract", Value::Null);
    } else { s6_field(&mut text, "proposal", "expected_effect", "errored", json!("wrong_type")); }
    if allow {
        s7_set(&mut text, "decision", "decision", json!("allow"));
        s6_set(&mut text, "stages.attempt.reason", json!(if parse { "unsupported" } else { "no_executor" }));
    }
    text
}
fn s7_rejected(code: &str) -> String {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        let a = &mut v["stages"]["attempt"]["record"];
        a["outcome"] = json!("rejected"); a["error_code"] = json!(code);
        if code != "size_limit" {
            for f in ["target_ref", "resolved_path"] { a["target"][f] =
                json!({"availability":"unavailable","value":null,"error_code":"not_resolved"}); }
        }
        for f in ["before_state", "content_payload"] { a[f] =
            json!({"availability":"absent_by_contract","value":null,"error_code":null}); }
        if code == "sink_rejected" { a["sink"]["context"]["status"] = json!("rejected"); a["sink"]["context"]["ref"] = Value::Null; }
    });
    row.stages.observation = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected);
    row.stages.verification = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected);
    s5_rechain(&mut row, "");
    let mut text = String::from_utf8(s5_bytes(&row)).unwrap();
    text = text.replace("\"allowed_effect_verified\"", if code == "sink_rejected" {
        "\"not_attempted_sink\""
    } else { "\"not_attempted_rejected\"" }); text
}

#[test]
fn s7_row_28() {
    for (text, wrong) in [(s7_absent("enforced", true, "allowed"), "blocked"),
        (s7_absent("shadow", true, "allowed"), "no_executor"),
        (s7_block(), "unsupported")] {
        let mut bad = text.clone(); s6_set(&mut bad, "stages.attempt.reason", json!(wrong));
        s7_accept(text); s7_check(bad, RejectionCode::SlotAvailabilityInvalid);
    }
    let mut unsupported = s7_absent("enforced", true, "allowed");
    s7_set(&mut unsupported, "proposal", "tool_name.value", json!("other"));
    s7_set(&mut unsupported, "proposal", "effect_kind", json!("unsupported"));
    s6_field(&mut unsupported, "proposal", "expected_effect", "absent_by_contract", Value::Null);
    s6_set(&mut unsupported, "stages.attempt.reason", json!("unsupported"));
    s7_accept(unsupported.clone());
    s6_set(&mut unsupported, "stages.attempt.reason", json!("no_executor"));
    s7_check(unsupported, RejectionCode::SlotAvailabilityInvalid);
}
#[test]
fn s7_row_29() {
    for text in [s7_absent("enforced", true, "allowed"), s7_rejected("traversal")] {
        s7_accept(text.clone());
        for slot in ["observation", "verification"] {
            let mut bad = text.clone();
            let wrong = if text.contains("\"outcome\":\"rejected\"") { "attempt_absent" } else { "attempt_rejected" };
            s6_set(&mut bad, &format!("stages.{slot}.reason"), json!(wrong));
            s7_check(bad, RejectionCode::SlotAvailabilityInvalid);
        }
    }
}
#[test]
fn s7_row_112() {
    let mut text = s6_text(); s7_set(&mut text, "proposal", "tool_name.value", json!("other"));
    s7_check(text, RejectionCode::SchemaError);
    let mut text = s7_absent("enforced", true, "allowed");
    s7_set(&mut text, "proposal", "effect_kind", json!("unsupported"));
    s6_field(&mut text, "proposal", "expected_effect", "absent_by_contract", Value::Null);
    s7_check(text, RejectionCode::SchemaError);
}
#[test]
fn s7_row_120() { s7_check(s7_proposal_error(false, true), RejectionCode::ErroredProposalAllowed); }
#[test]
fn s7_row_121() {
    for (field, value) in [("evaluated_policy", json!(true)), ("reason_code", json!("allowed"))] {
        let mut text = s7_proposal_error(false, false); s7_set(&mut text, "decision", field, value);
        s7_check(text, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_174() {
    let pass = s7_absent("not_evaluated", false, "policy_layer_off"); s7_accept(pass.clone());
    let mut bad = pass; s7_set(&mut bad, "decision", "enforcement", json!("enforced"));
    s7_check(bad, RejectionCode::EnforcementInconsistent);
    for (field, value) in [("evaluated_policy", json!(true)), ("decision", json!("block")), ("reason_code", json!("allowed"))] {
        let mut bad = s7_absent("not_evaluated", false, "policy_layer_off");
        s7_set(&mut bad, "decision", field, value); s7_check(bad, RejectionCode::EnforcementInconsistent);
    }
}
#[test]
fn s7_row_175() {
    s7_accept(s7_absent("suppressed_by_response", true, "allowed"));
    for (field, value) in [("decision", json!("block")), ("evaluated_policy", json!(false))] {
        let mut bad = s7_absent("suppressed_by_response", true, "allowed");
        s7_set(&mut bad, "decision", field, value); s7_check(bad, RejectionCode::EnforcementInconsistent);
    }
}
#[test]
fn s7_row_176() { s7_accept(s7_absent("shadow", true, "allowed")); }
#[test]
fn s7_row_180() {
    for parse in [true, false] { s7_check(s7_proposal_error(parse, true), RejectionCode::ErroredProposalAllowed); }
}
#[test]
fn s7_row_181() {
    let mut bad = s7_absent("enforced", true, "allowed");
    s7_set(&mut bad, "decision", "evaluated_policy", json!(false)); s7_check(bad, RejectionCode::SchemaError);
    let mut bad = s7_proposal_error(true, false);
    s7_set(&mut bad, "decision", "evaluated_policy", json!(true)); s7_check(bad, RejectionCode::SchemaError);
}
#[test]
fn s7_row_183() {
    for mut text in [s7_absent("enforced", true, "allowed"), s7_block(), s7_proposal_error(true, false)] {
        s7_set(&mut text, "decision", "reason_code", json!("multiple")); s7_check(text, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_184() {
    let mut bad = s7_absent("enforced", true, "allowed");
    s7_set(&mut bad, "decision", "reason_code", json!("detection_gate")); s7_check(bad, RejectionCode::SchemaError);
}
#[test]
fn s7_row_187() {
    let mut bad = s7_absent("enforced", true, "max_length");
    s7_set(&mut bad, "decision", "failed_constraints", json!([{ "argument":"path", "constraint":"max_length" }]));
    s7_check(bad, RejectionCode::SchemaError);
    let mut bad = s7_proposal_error(true, false);
    s7_set(&mut bad, "decision", "failed_constraints", json!([{ "argument":"path", "constraint":"max_length" }]));
    s7_check(bad, RejectionCode::SchemaError);
}
#[test]
fn s7_row_188() {
    for failures in [json!([{ "argument":"path", "constraint":"tool_allowed" }]),
        json!([{ "argument":null, "constraint":"tool_allowed" }, { "argument":"path", "constraint":"max_length" }])] {
        let mut bad = s7_block();
        s7_set(&mut bad, "decision", "reason_code", json!(if failures.as_array().unwrap().len() == 1 { "tool_not_allowed" } else { "multiple" }));
        s7_set(&mut bad, "decision", "failed_constraints", failures); s7_check(bad, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_189() {
    let mut bad = s7_block(); s7_set(&mut bad, "decision", "reason_code", json!("multiple"));
    s7_set(&mut bad, "decision", "failed_constraints", json!([
        {"argument":"path","constraint":"argument_present"}, {"argument":"path","constraint":"max_length"}]));
    s7_check(bad, RejectionCode::SchemaError);
}
#[test]
fn s7_row_189a() {
    let mut bad = s7_block(); s7_set(&mut bad, "decision", "failed_constraints", json!([]));
    s7_set(&mut bad, "decision", "reason_code", json!("allowed")); s7_check(bad, RejectionCode::SchemaError);
}
#[test]
fn s7_row_195() {
    let mut row = s6_detection();
    s5_change(&mut row, |v| v["stages"]["decision"]["record"]["detection"]["value"]["detection_outcome"] = json!("threshold_breached"));
    s5_rechain(&mut row, ""); s7_accept(String::from_utf8(s5_bytes(&row)).unwrap());
}
#[test]
fn s7_row_205() {
    for code in ["traversal", "size_limit"] {
        let mut bad = s7_rejected(code); s7_accept(bad.clone());
        let base = s6_text();
        if code == "size_limit" {
            for f in ["target_ref", "resolved_path"] { s6_field(&mut bad, "attempt", &format!("target.{f}"), "unavailable", json!("not_resolved")); }
        } else {
            for f in ["target_ref", "resolved_path"] {
                let path = format!("stages.attempt.record.target.{f}"); let (a,b) = s6_span(&base, &path);
                let (x,y) = s6_span(&bad, &path); bad.replace_range(x..y, &base[a..b]);
            }
        }
        s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
    }
    let mut bad = s6_text();
    for f in ["target_ref", "resolved_path"] { s6_field(&mut bad, "attempt", &format!("target.{f}"), "unavailable", json!("not_resolved")); }
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
}
#[test]
fn s7_row_221() {
    for (exists, digest, len) in [(false, Value::Null, json!(1)), (false, json!("0".repeat(64)), Value::Null),
        (true, Value::Null, json!(1)), (true, json!("0".repeat(64)), Value::Null)] {
        let mut bad = s6_text();
        s7_set(&mut bad, "attempt", "before_state.value.exists", json!(exists));
        s7_set(&mut bad, "attempt", "before_state.value.digest", digest);
        s7_set(&mut bad, "attempt", "before_state.value.len_bytes", len); s7_check(bad, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_224() {
    let mut bad = s7_rejected("traversal"); s6_field(&mut bad, "attempt", "before_state", "unavailable", json!("no_read_access"));
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
    let mut bad = s6_text(); s6_field(&mut bad, "attempt", "before_state", "absent_by_contract", Value::Null);
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
}
#[test]
fn s7_row_229() {
    let mut bad = s7_rejected("traversal"); s7_set(&mut bad, "attempt", "error_code", json!("io_error"));
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
    for outcome in ["errored", "ambiguous"] {
        let mut bad = s6_errored(); s7_set(&mut bad, "attempt", "outcome", json!(outcome));
        s7_set(&mut bad, "attempt", "error_code", json!("size_limit")); s7_check(bad, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_232() {
    let mut bad = s7_rejected("traversal"); s6_field(&mut bad, "attempt", "content_payload", "unavailable", json!("store_not_configured"));
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
    let mut bad = s6_text(); s6_field(&mut bad, "attempt", "content_payload", "absent_by_contract", Value::Null);
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
}
#[test]
fn s7_row_241a() {
    for (exists, digest, len) in [(false, Value::Null, json!(1)), (false, json!("0".repeat(64)), Value::Null),
        (true, Value::Null, json!(1)), (true, json!("0".repeat(64)), Value::Null)] {
        let mut bad = s6_text(); s7_set(&mut bad, "observation", "observed.value.exists", json!(exists));
        s7_set(&mut bad, "observation", "observed.value.digest", digest);
        s7_set(&mut bad, "observation", "observed.value.len_bytes", len); s7_check(bad, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_o10_step3_before_rule10() {
    let mut defect = s6_text(); s7_set(&mut defect, "attempt", "before_state.value.len_bytes", json!(1));
    s7_check(defect.clone(), RejectionCode::SchemaError);
    for mut text in [s6_text(), defect] {
        s6_repair(&mut text); s6_set(&mut text, "stages.proposal.digest", json!("0".repeat(64)));
        assert_eq!(s2_validate(text.as_bytes()), ValidationResult::Rejected(RejectionCode::BrokenStageRef));
    }
}
#[test]
fn s7_o11_rule10_before_rule11() {
    let mut shape = s7_rejected("traversal"); let base = s6_text();
    let path = "stages.attempt.record.before_state";
    let (a,b) = s6_span(&base, path); let (x,y) = s6_span(&shape, path); shape.replace_range(x..y, &base[a..b]);
    s7_check(shape.clone(), RejectionCode::RejectedAttemptShapeInvalid);
    let mut nullness = s6_text(); s7_set(&mut nullness, "attempt", "before_state.value.len_bytes", json!(1));
    s7_check(nullness, RejectionCode::SchemaError);
    s7_set(&mut shape, "attempt", "before_state.value.len_bytes", json!(1));
    s7_check(shape, RejectionCode::SchemaError);
}
#[test]
fn s7_row_287_rule7_before_rule8() {
    let mut first = s7_absent("enforced", true, "allowed");
    s7_set(&mut first, "proposal_context", "arguments_wire_form", json!("object"));
    s7_check(first.clone(), RejectionCode::NativeArgumentsAvailabilityMismatch);
    let mut second = s7_absent("not_evaluated", false, "policy_layer_off");
    s7_set(&mut second, "decision", "enforcement", json!("enforced"));
    s7_check(second.clone(), RejectionCode::EnforcementInconsistent);
    s7_set(&mut second, "proposal_context", "arguments_wire_form", json!("object"));
    s7_check(second, RejectionCode::NativeArgumentsAvailabilityMismatch);
}
#[test]
fn s7_row_287_rule8_before_rule9() {
    let first = s7_absent("not_evaluated", true, "allowed"); s7_check(first, RejectionCode::EnforcementInconsistent);
    let second = s7_proposal_error(false, true); s7_check(second.clone(), RejectionCode::ErroredProposalAllowed);
    let mut both = second; s7_set(&mut both, "decision", "enforcement", json!("suppressed_by_response"));
    s7_check(both, RejectionCode::EnforcementInconsistent);
}
#[test]
fn s7_row_287_rule9_before_rule10() {
    let first = s7_proposal_error(false, true); s7_check(first.clone(), RejectionCode::ErroredProposalAllowed);
    let mut second = s7_proposal_error(false, false); s7_set(&mut second, "decision", "reason_code", json!("allowed"));
    s7_check(second, RejectionCode::SchemaError);
    let mut both = first; s7_set(&mut both, "decision", "reason_code", json!("allowed"));
    s7_check(both, RejectionCode::ErroredProposalAllowed);
}
#[test]
fn s7_row_287_rule11_before_rule12() {
    let mut first = s7_rejected("traversal"); s6_field(&mut first, "attempt", "before_state", "unavailable", json!("no_read_access"));
    s7_check(first.clone(), RejectionCode::RejectedAttemptShapeInvalid);
    let mut second = s7_rejected("traversal"); s6_set(&mut second, "stages.observation.reason", json!("attempt_absent"));
    s7_check(second, RejectionCode::SlotAvailabilityInvalid);
    s6_set(&mut first, "stages.observation.reason", json!("attempt_absent"));
    s7_check(first, RejectionCode::RejectedAttemptShapeInvalid);
}
#[test]
fn s7_f15() { s7_row_180(); }
#[test]
fn s7_f17() { s7_accept(s7_rejected("action_id_collision")); }
#[test]
fn s7_f39() { s7_accept(s7_proposal_error(true, false)); }
#[test]
fn s7_f40() { s7_row_120(); }
#[test]
fn s7_f53() { s7_row_174(); }
#[test]
fn s7_f59() {
    s7_accept(s7_rejected("sink_rejected"));
    let mut bad = s7_rejected("sink_rejected"); s6_field(&mut bad, "attempt", "before_state", "unavailable", json!("no_read_access"));
    s7_check(bad, RejectionCode::RejectedAttemptShapeInvalid);
}
#[test]
fn s7_f62() { s7_accept(s7_rejected("action_id_collision")); }
#[test]
fn s7_f63() {
    for code in ["traversal", "symlink_component", "outside_root", "sink_rejected", "sink_unavailable", "sink_ref_mismatch", "policy_drift", "action_id_collision", "root_unavailable"] {
        let mut text = s7_rejected(code); s7_accept(text.clone());
        let base = s6_text();
        for f in ["target_ref", "resolved_path"] {
            let path = format!("stages.attempt.record.target.{f}"); let (a,b) = s6_span(&base, &path);
            let (x,y) = s6_span(&text, &path); text.replace_range(x..y, &base[a..b]);
        }
        s7_check(text, RejectionCode::RejectedAttemptShapeInvalid);
    }
}
#[test]
fn s7_f64() { s7_accept(s7_rejected("size_limit")); }
#[test]
fn s7_f65() { s7_row_175(); }
#[test]
fn s7_f66() {
    for (reason, failures) in [("multiple", json!([
        {"argument":"content","constraint":"argument_present"}, {"argument":"path","constraint":"max_length"}, {"argument":"path","constraint":"starts_with"}])),
        ("tool_not_allowed", json!([{ "argument":null, "constraint":"tool_allowed" }]))] {
        let mut text = s7_block(); s7_set(&mut text, "decision", "reason_code", json!(reason));
        s7_set(&mut text, "decision", "failed_constraints", failures); s7_accept(text);
    }
    s7_accept(s7_absent("enforced", true, "allowed"));
    s7_row_183(); s7_row_187(); s7_row_188(); s7_row_189(); s7_row_189a();
}

#[test]
fn s7_f66_single_constraint_mapping_and_derivation() {
    for constraint in ["argument_present", "type_check", "starts_with", "not_contains", "matches", "one_of", "max_length", "min", "max", "url_host"] {
        let mut text = s7_block();
        s7_set(&mut text, "decision", "failed_constraints", json!([{ "argument":"path", "constraint":constraint }]));
        s7_set(&mut text, "decision", "reason_code", json!(if constraint == "argument_present" { "argument_missing" } else { constraint }));
        s7_accept(text.clone());
        s7_set(&mut text, "decision", "reason_code", json!("multiple"));
        s7_check(text, RejectionCode::SchemaError);
    }
}
#[test]
fn s7_row_28_precedence_overlapping_conditions() {
    let mut text = s7_absent("shadow", true, "allowed");
    s7_set(&mut text, "proposal", "tool_name.value", json!("other"));
    s7_set(&mut text, "proposal", "effect_kind", json!("unsupported"));
    s6_field(&mut text, "proposal", "expected_effect", "absent_by_contract", Value::Null);
    s6_set(&mut text, "stages.attempt.reason", json!("unsupported"));
    s7_accept(text.clone());
    s6_set(&mut text, "stages.attempt.reason", json!("not_enforced"));
    s7_check(text, RejectionCode::SlotAvailabilityInvalid);
    let mut text = s7_proposal_error(true, false);
    s7_set(&mut text, "decision", "enforcement", json!("shadow"));
    s7_accept(text.clone());
    s6_set(&mut text, "stages.attempt.reason", json!("unsupported"));
    s7_check(text, RejectionCode::SlotAvailabilityInvalid);
}


// S8: resolver behavior stays in integration tests; canonical repair precedes defects.
fn s8_two_payload() -> Row {
    serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F74_row.json")).unwrap()
}
fn s8_resolver(row: &Row, arguments: PayloadResolution, detection: Option<PayloadResolution>) -> ReceiptPayloadResolverDouble {
    let mut results = Vec::new();
    if let PresentUnavailable::Present(p) = &row.proposal_context.record.arguments_payload {
        results.push((PayloadStore::ReceiptEvidence, p.digest.clone(), arguments));
    }
    if let Some(result) = detection {
        if let PresentUnavailableAbsent::Present(p) = &row.proposal_context.record.detection_payload {
            results.push((PayloadStore::ReceiptEvidence, p.digest.clone(), result));
        }
    }
    ReceiptPayloadResolverDouble { results, calls: std::cell::RefCell::new(Vec::new()) }
}
fn s8_args() -> PayloadResolution { PayloadResolution::Found(include_bytes!("fixtures/receipt_v0/arguments.json").to_vec()) }
fn s8_detection() -> PayloadResolution { PayloadResolution::Found(include_bytes!("fixtures/receipt_v0/F74_detection.json").to_vec()) }
fn s8_validate(row: &Row, resolver: &ReceiptPayloadResolverDouble) -> ValidationResult {
    validator::validate_receipt(&s5_bytes(row), resolver)
}
fn s8_reject(mut row: Row, code: RejectionCode) {
    s5_rechain(&mut row, "");
    assert_eq!(s8_validate(&row, &s8_resolver(&row, s8_args(), Some(s8_detection()))), ValidationResult::Rejected(code));
}
fn s8_control(row: &Row) {
    // R1 controls must remain accepted when the step-9 and step-10 stubs land.
    assert_eq!(row.summary, receipt_summary(row));
    if let (Slot::Present { record: a, .. }, Slot::Present { record: o, .. }, Slot::Present { record: v, .. }) =
        (&row.stages.attempt, &row.stages.observation, &row.stages.verification) {
        let method = verify_expected_effect(&row.stages.proposal.record, a, o).unwrap();
        assert_eq!(v.outcome, method.outcome);
        assert_eq!(v.reason_code, method.reason_code);
        assert_eq!(v.comparison, method.comparison);
    }
    assert_eq!(s8_validate(row, &s8_resolver(row, s8_args(), Some(s8_detection()))), ValidationResult::Valid);
}
fn s8_no_byte_rejection(result: ValidationResult) {
    for code in [RejectionCode::AttemptWithoutSink, RejectionCode::SinkCodeMismatch,
        RejectionCode::PolicyDriftUnrejected, RejectionCode::AttemptContentMismatch,
        RejectionCode::ArgumentsPayloadMismatch, RejectionCode::DetectionPayloadMismatch,
        RejectionCode::DetectionCopyMismatch, RejectionCode::TargetBindingMismatch] {
        assert_ne!(result, ValidationResult::Rejected(code));
    }
}
// Rechain only from attempt onward: preserve the sink vector intentionally under test.
fn s8_tail(row: &mut Row) {
    if let Slot::Present { digest, record } = &mut row.stages.attempt {
        *digest = record_digest(&record_bytes(ReceiptRecord::Attempt(record)).unwrap(), RecordDomain::Stage);
    }
    if let Slot::Present { digest, record } = &mut row.stages.observation {
        if let Slot::Present { digest: a, .. } = &row.stages.attempt { record.attempt_ref = a.clone(); }
        *digest = record_digest(&record_bytes(ReceiptRecord::Observation(record)).unwrap(), RecordDomain::Stage);
    }
    if let Slot::Present { digest, record } = &mut row.stages.verification {
        if let Slot::Present { digest: o, .. } = &row.stages.observation { record.observation_ref = o.clone(); }
        *digest = record_digest(&record_bytes(ReceiptRecord::Verification(record)).unwrap(), RecordDomain::Stage);
    }
    row.summary = receipt_summary(row);
}
fn s8_sink_row(code: &str, vector: [(&str, bool); 3]) -> Row {
    let mut row: Row = serde_json::from_str(&s7_rejected(code)).unwrap();
    s5_rechain(&mut row, "");
    s5_change(&mut row, |v| {
        for (key, (status, wrong)) in ["proposal", "context", "decision"].into_iter().zip(vector) {
            let slot = &mut v["stages"]["attempt"]["record"]["sink"][key];
            slot["status"] = json!(status);
            if status != "accepted" { slot["ref"] = Value::Null; }
            else if wrong { slot["ref"] = json!("0".repeat(64)); }
        }
    });
    s8_tail(&mut row); row
}
fn s8_sink_result(row: &Row) -> ValidationResult {
    s8_validate(row, &s8_resolver(row, s8_args(), None))
}

#[test]
fn s8_r1_row_141_arguments_not_configured() {
    let row = s5_base(); let resolver = s8_resolver(&row, PayloadResolution::NotConfigured, None);
    let result = s8_validate(&row, &resolver);
    assert_eq!(result, ValidationResult::ValidUnresolved);
    assert_ne!(result, ValidationResult::Valid);
    assert!(!matches!(result, ValidationResult::Rejected(_)));
    assert_eq!(resolver.calls.borrow().len(), 1);
}
#[test]
fn s8_r2_wrong_summary_continues_after_arguments() {
    let mut row = s5_base(); row.summary = Summary::AllowedEffectUnknown;
    let resolver = s8_resolver(&row, PayloadResolution::NotConfigured, None);
    let result = s8_validate(&row, &resolver);
    s8_no_byte_rejection(result); assert_ne!(result, ValidationResult::Valid);
    assert_eq!(*resolver.calls.borrow(), vec![(PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json")))]);
    // The exact summary_mismatch result is S9-owned.
}
#[test]
fn s8_r3_wrong_summary_not_found() {
    let mut row = s5_base(); row.summary = Summary::AllowedEffectUnknown;
    assert_eq!(s8_validate(&row, &s8_resolver(&row, PayloadResolution::NotFound, None)), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
}
#[test]
fn s8_r4_f58_exact_arguments_control() { s8_control(&s5_base()); }
#[test]
fn s8_r5_not_configured_does_not_hide_detection_not_found() {
    let row = s8_two_payload(); let resolver = s8_resolver(&row, PayloadResolution::NotConfigured, Some(PayloadResolution::NotFound));
    assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::DetectionPayloadMismatch));
    assert_eq!(resolver.calls.borrow().len(), 2);
}
#[test]
fn s8_r6_row_141_detection_and_both_not_configured() {
    for args in [s8_args(), PayloadResolution::NotConfigured] {
        let row = s8_two_payload(); let resolver = s8_resolver(&row, args, Some(PayloadResolution::NotConfigured));
        assert_eq!(s8_validate(&row, &resolver), ValidationResult::ValidUnresolved);
        assert_eq!(resolver.calls.borrow().len(), 2);
    }
}
#[test]
fn s8_r7_row_149_t3_arguments_before_detection() {
    let row = s8_two_payload(); let resolver = s8_resolver(&row, PayloadResolution::NotFound, Some(PayloadResolution::NotFound));
    assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
    assert_eq!(*resolver.calls.borrow(), vec![(PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json")))]);
}
#[test]
fn s8_r8_f76_unavailable_detection_is_not_resolved() {
    let row = s6_detection(); let resolver = s8_resolver(&row, s8_args(), None);
    assert_eq!(s8_validate(&row, &resolver), ValidationResult::Valid);
    assert_eq!(*resolver.calls.borrow(), vec![(PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json")))]);
}
#[test]
fn s8_r9_wrong_summary_continues_after_detection() {
    let mut row = s8_two_payload(); row.summary = Summary::AllowedEffectUnknown;
    let resolver = s8_resolver(&row, s8_args(), Some(PayloadResolution::NotConfigured));
    let result = s8_validate(&row, &resolver);
    s8_no_byte_rejection(result); assert_ne!(result, ValidationResult::Valid);
    assert_eq!(*resolver.calls.borrow(), vec![(PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json"))),
        (PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/F74_detection.json")))]);
    // The exact summary_mismatch result is S9-owned.
}

#[test]
fn s8_row_137_arguments_not_found() {
    let row = s5_base();
    assert_eq!(s8_validate(&row, &s8_resolver(&row, PayloadResolution::NotFound, None)), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
}
#[test]
fn s8_row_138_arguments_raw_digest() {
    let row = s5_base();
    assert_eq!(s8_validate(&row, &s8_resolver(&row, PayloadResolution::Found(b"wrong".to_vec()), None)), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
}
#[test]
fn s8_row_139_105_arguments_tagged_digest() {
    let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["proposal"]["record"]["arguments_digest"]["value"] = json!("0".repeat(64)));
    s8_reject(row, RejectionCode::ArgumentsPayloadMismatch);
}
#[test]
fn s8_row_140a_arguments_row_length_before_lookup() {
    let mut row = s5_base(); s5_change(&mut row, |v| v["proposal_context"]["record"]["arguments_payload"]["value"]["len_bytes"] = json!(35));
    s5_rechain(&mut row, "");
    for resolution in [PayloadResolution::NotConfigured, PayloadResolution::NotFound, s8_args()] {
        let resolver = s8_resolver(&row, resolution, None);
        assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
        assert!(resolver.calls.borrow().is_empty());
    }
}
#[test]
fn s8_row_140b_110_arguments_resolved_length() {
    let mut row = s5_base(); s5_change(&mut row, |v| {
        v["proposal_context"]["record"]["arguments_payload"]["value"]["len_bytes"] = json!(35);
        v["stages"]["proposal"]["record"]["arguments_len_bytes"]["value"] = json!(35);
    }); s8_reject(row, RejectionCode::ArgumentsPayloadMismatch);
}
#[test]
fn s8_row_145_detection_not_found() {
    let row = s8_two_payload();
    assert_eq!(s8_validate(&row, &s8_resolver(&row, s8_args(), Some(PayloadResolution::NotFound))), ValidationResult::Rejected(RejectionCode::DetectionPayloadMismatch));
}
#[test]
fn s8_row_146_detection_raw_digest() {
    let row = s8_two_payload();
    assert_eq!(s8_validate(&row, &s8_resolver(&row, s8_args(), Some(PayloadResolution::Found(b"wrong".to_vec())))), ValidationResult::Rejected(RejectionCode::DetectionPayloadMismatch));
}
#[test]
fn s8_row_147a_detection_row_length_before_lookup() {
    let mut row = s8_two_payload(); s5_change(&mut row, |v| v["proposal_context"]["record"]["detection_payload"]["value"]["len_bytes"] = json!(1214));
    s5_rechain(&mut row, "");
    for resolution in [PayloadResolution::NotConfigured, PayloadResolution::NotFound, s8_detection()] {
        let resolver = s8_resolver(&row, s8_args(), Some(resolution));
        assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::DetectionPayloadMismatch));
        assert_eq!(resolver.calls.borrow().len(), 1);
    }
}
#[test]
fn s8_row_147b_detection_resolved_length() {
    let mut row = s8_two_payload(); s5_change(&mut row, |v| {
        v["proposal_context"]["record"]["detection_payload"]["value"]["len_bytes"] = json!(1214);
        v["stages"]["decision"]["record"]["detection"]["value"]["record_len_bytes"] = json!(1214);
    }); s8_reject(row, RejectionCode::DetectionPayloadMismatch);
}
#[test]
fn s8_row_148_detection_tagged_digest() {
    for digest in ["0".repeat(64), payload_digest(include_bytes!("fixtures/receipt_v0/F74_detection.json"))] {
        let mut row = s8_two_payload(); s5_change(&mut row, |v| v["stages"]["decision"]["record"]["detection"]["value"]["record_digest"] = json!(digest));
        s8_reject(row, RejectionCode::DetectionPayloadMismatch);
    }
}
#[test]
fn s8_row_168_detection_copy_each_field() {
    for (field, value) in [("mode", json!("block")), ("input_digest", json!("0".repeat(64))), ("detection_outcome", json!("threshold_breached"))] {
        let mut row = s8_two_payload(); s5_change(&mut row, |v| v["stages"]["decision"]["record"]["detection"]["value"][field] = value);
        s8_reject(row, RejectionCode::DetectionCopyMismatch);
    }
}
#[test]
fn s8_row_83_f57_attempt_target_ref() {
    let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["target"]["target_ref"]["value"] = json!("0".repeat(64)));
    s8_reject(row, RejectionCode::TargetBindingMismatch);
}
#[test]
fn s8_row_84_f57_observation_target_ref() {
    let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["observation"]["record"]["target_ref"]["value"] = json!("0".repeat(64)));
    s8_reject(row, RejectionCode::TargetBindingMismatch);
}
#[test]
fn s8_row_85_f57_proposed_path_digest() {
    let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["target"]["proposed_path_digest"] = json!("0".repeat(64)));
    s8_reject(row, RejectionCode::TargetBindingMismatch);
}
#[test]
fn s8_row_213_f4_f28_f60_nonrejected_sink() {
    for (status, reference) in [("rejected", Value::Null), ("unavailable", Value::Null), ("accepted", json!("0".repeat(64))), ("accepted", Value::Null)] {
        let mut row = s5_base();
        s5_change(&mut row, |v| { v["stages"]["attempt"]["record"]["sink"]["decision"]["status"] = json!(status); v["stages"]["attempt"]["record"]["sink"]["decision"]["ref"] = reference; });
        s8_tail(&mut row);
        assert_eq!(s8_sink_result(&row), ValidationResult::Rejected(RejectionCode::AttemptWithoutSink));
    }
}
#[test]
fn s8_row_214_f73_sink_precedence_and_wrong_codes() {
    for (vector, code) in [
        ([("accepted",false),("unavailable",false),("rejected",false)],"sink_unavailable"),
        ([("accepted",false),("rejected",false),("accepted",true)],"sink_rejected"),
        ([("accepted",true),("accepted",false),("accepted",false)],"sink_ref_mismatch"),
        ([("unavailable",false);3],"sink_unavailable")] {
        let row = s8_sink_row(code, vector); s8_control(&row);
        for wrong in ["sink_unavailable", "sink_rejected", "sink_ref_mismatch"] {
            if wrong == code { continue; }
            let mut bad = row.clone(); s5_change(&mut bad, |v| v["stages"]["attempt"]["record"]["error_code"] = json!(wrong)); s8_tail(&mut bad);
            assert_eq!(s8_sink_result(&bad), ValidationResult::Rejected(RejectionCode::SinkCodeMismatch));
        }
    }
}
#[test]
fn s8_row_215_passing_sink_with_sink_code() {
    for code in ["sink_unavailable", "sink_rejected", "sink_ref_mismatch"] {
        let row = s8_sink_row(code, [("accepted",false);3]);
        assert_eq!(s8_sink_result(&row), ValidationResult::Rejected(RejectionCode::SinkCodeMismatch));
    }
}
#[test]
fn s8_row_218_f26_policy_drift_both_fields() {
    for field in ["contract_hash", "policy_digest"] {
        let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["policy_check"][field] = json!("0".repeat(64)));
        s8_reject(row, RejectionCode::PolicyDriftUnrejected);
    }
}
#[test]
fn s8_row_233_f27_content_digest_and_length() {
    for (field, value) in [("digest", json!("0".repeat(64))), ("len_bytes", json!(999))] {
        let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["content_payload"]["value"][field] = value);
        s8_reject(row, RejectionCode::AttemptContentMismatch);
    }
    let mut row = s5_base(); s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["content_payload"] = json!({"availability":"unavailable","value":null,"error_code":"store_not_configured"}));
    s6_method(&mut row); s8_control(&row);
}
#[test]
fn s8_f4_f28_rejected_decision_sink_pass() {
    let row = s8_sink_row("sink_rejected", [("accepted",false),("accepted",false),("rejected",false)]); s8_control(&row);
}
#[test]
fn s8_f59_rejected_context_sink_pass() {
    let row = s8_sink_row("sink_rejected", [("accepted",false),("rejected",false),("accepted",false)]); s8_control(&row);
}
#[test]
fn s8_f60_rejected_ref_mismatch_pass() {
    let row = s8_sink_row("sink_ref_mismatch", [("accepted",false),("accepted",false),("accepted",true)]); s8_control(&row);
}
#[test]
fn s8_f21_f61_policy_drift_rejected_pass_and_a7() {
    for drift in [false, true] {
        let mut row: Row = serde_json::from_str(&s7_rejected("policy_drift")).unwrap();
        if drift { s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["policy_check"]["policy_digest"] = json!("0".repeat(64))); }
        s5_rechain(&mut row, ""); s8_control(&row);
    }
    let mut row = s8_sink_row("sink_rejected", [("accepted",false),("rejected",false),("accepted",false)]);
    s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["policy_check"]["policy_digest"] = json!("0".repeat(64)));
    s8_tail(&mut row); s8_control(&row);
}
#[test]
fn s8_f58_existing_fixture_control_and_tag_mismatch() {
    let row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F58_row.json")).unwrap(); s8_control(&row);
    let row: Row = serde_json::from_slice(include_bytes!("fixtures/receipt_v0/F58_tag_mismatch_row.json")).unwrap();
    assert_eq!(s8_sink_result(&row), ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
    let bytes = include_bytes!("fixtures/receipt_v0/arguments.json"); assert_ne!(payload_digest(bytes), arguments_digest(bytes));
}
#[test]
fn s8_f74_two_payload_control_raw_and_tagged_differ() {
    let row = s8_two_payload(); s8_control(&row);
    let bytes = include_bytes!("fixtures/receipt_v0/F74_detection.json"); assert_ne!(payload_digest(bytes), detection_record_digest(bytes));
}
#[test]
fn s8_f76_absent_detection_pass() { s8_control(&s5_base()); }
#[test]
fn s8_row_210a_nonnull_rejected_ref_a21() {
    let mut row = s8_sink_row("sink_rejected", [("accepted",false),("rejected",false),("accepted",false)]);
    s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["sink"]["context"]["ref"] = json!("0".repeat(64)));
    s8_tail(&mut row); s8_control(&row);
}
#[test]
fn s8_o6_required_fields_before_attempt_bytes() {
    for (early, late, code) in [(true,false,RejectionCode::FieldAvailabilityInvalid), (false,true,RejectionCode::AttemptContentMismatch), (true,true,RejectionCode::FieldAvailabilityInvalid)] {
        let mut row = s5_base(); s5_change(&mut row, |v| {
            if early { v["stages"]["verification"]["record"]["comparison"] = json!({"availability":"absent_by_contract","value":null,"error_code":null}); }
            if late { v["stages"]["attempt"]["record"]["content_payload"]["value"]["digest"] = json!("0".repeat(64)); }
        }); s8_reject(row, code);
    }
}
#[test]
fn s8_o7_attempt_bytes_before_target_binding() {
    for (early, late, code) in [(true,false,RejectionCode::AttemptContentMismatch), (false,true,RejectionCode::TargetBindingMismatch), (true,true,RejectionCode::AttemptContentMismatch)] {
        let mut row = s5_base(); s5_change(&mut row, |v| {
            if early { v["stages"]["attempt"]["record"]["content_payload"]["value"]["digest"] = json!("0".repeat(64)); }
            if late { v["stages"]["observation"]["record"]["target_ref"]["value"] = json!("0".repeat(64)); }
        }); s8_reject(row, code);
    }
}

// S9: repair carried identities before testing method and summary defects.
fn s9_check(row: &Row, code: RejectionCode) {
    assert_eq!(s2_validate(&s5_bytes(row)), ValidationResult::Rejected(code));
}
fn s9_control(row: &Row) {
    assert_eq!(row.summary, receipt_summary(row));
    if let (Slot::Present { record: a, .. }, Slot::Present { record: o, .. }, Slot::Present { record: v, .. }) =
        (&row.stages.attempt, &row.stages.observation, &row.stages.verification) {
        let method = verify_expected_effect(&row.stages.proposal.record, a, o).unwrap();
        assert_eq!(v.outcome, method.outcome);
        assert_eq!(v.reason_code, method.reason_code);
        assert_eq!(v.comparison, method.comparison);
    }
    assert_eq!(s2_validate(&s5_bytes(row)), ValidationResult::Valid);
}
fn s9_forge_verified(row: &mut Row) {
    s5_change(row, |v| {
        v["stages"]["verification"]["record"]["outcome"] = json!("verified");
        v["stages"]["verification"]["record"]["reason_code"] = Value::Null;
    });
    s5_rechain(row, "");
}
fn s9_wrong_delta(row: &mut Row, delta: &str) {
    s5_change(row, |v| v["stages"]["verification"]["record"]["comparison"]["value"]["delta"] = json!(delta));
    s5_rechain(row, "");
}
fn s9_missing_before() -> Row {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["attempt"]["record"]["before_state"] =
        json!({"availability":"unavailable","value":null,"error_code":"no_read_access"}));
    s6_method(&mut row); row
}

#[test]
fn s9_row_49_f6_verified_on_failed_digest() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["observation"]["record"]["observed"]["value"]["digest"] = json!("0".repeat(64)));
    s6_method(&mut row); s9_control(&row);
    s9_forge_verified(&mut row);
    s9_check(&row, RejectionCode::VerificationInconsistent);
}
#[test]
fn s9_row_253_comparison_each_copied_field() {
    let row = s5_base(); s9_control(&row);
    for (field, value) in [
        ("expected_digest", json!("0".repeat(64))),
        ("observed_digest", json!("0".repeat(64))),
        ("expected_len_bytes", json!(6)),
        ("observed_len_bytes", json!(6)),
        ("within_root", json!(false)),
        ("before_digest", json!("0".repeat(64))),
        ("delta", json!("changed")),
    ] {
        let mut bad = row.clone();
        s5_change(&mut bad, |v| v["stages"]["verification"]["record"]["comparison"]["value"][field] = value);
        s5_rechain(&mut bad, "");
        s9_check(&bad, RejectionCode::VerificationInconsistent);
    }
}
#[test]
fn s9_row_256_delta_exists_pairs() {
    for (before_exists, observed_exists, expected_delta) in [
        (false, true, Delta::Created), (true, true, Delta::Changed),
        (false, false, Delta::Unchanged), (true, false, Delta::Removed),
    ] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            let b = &mut v["stages"]["attempt"]["record"]["before_state"]["value"];
            b["exists"] = json!(before_exists);
            b["digest"] = if before_exists { json!("0".repeat(64)) } else { Value::Null };
            b["len_bytes"] = if before_exists { json!(5) } else { Value::Null };
            let o = &mut v["stages"]["observation"]["record"]["observed"]["value"];
            o["exists"] = json!(observed_exists);
            if !observed_exists { o["digest"] = Value::Null; o["len_bytes"] = Value::Null; }
        });
        s6_method(&mut row); s9_control(&row);
        if let Slot::Present { record: v, .. } = &row.stages.verification {
            if let PresentAbsent::Present(c) = &v.comparison { assert_eq!(c.delta, expected_delta); }
            else { panic!("comparison must be present"); }
        }
        s9_wrong_delta(&mut row, "unknown");
        s9_check(&row, RejectionCode::VerificationInconsistent);
    }
}
#[test]
fn s9_row_257_ordered_method_and_forged_reason() {
    // Each line's later predicates also fail where possible, proving first match.
    for (line, outcome, reason) in [
        (1, VerificationOutcome::Unknown, Some(VerificationReason::AttemptAmbiguous)),
        (2, VerificationOutcome::Unknown, Some(VerificationReason::ObservationMissing)),
        (3, VerificationOutcome::Failed, Some(VerificationReason::AttemptErrored)),
        (4, VerificationOutcome::Failed, Some(VerificationReason::OutsideRoot)),
        (5, VerificationOutcome::Failed, Some(VerificationReason::TargetMissing)),
        (6, VerificationOutcome::Failed, Some(VerificationReason::LengthMismatch)),
        (7, VerificationOutcome::Failed, Some(VerificationReason::DigestMismatch)),
        (8, VerificationOutcome::Unknown, Some(VerificationReason::BeforeStateMissing)),
        (9, VerificationOutcome::Unknown, Some(VerificationReason::NoDeltaObserved)),
        (10, VerificationOutcome::Verified, None),
    ] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            if line == 1 || line == 2 || line == 3 {
                let a = &mut v["stages"]["attempt"]["record"];
                a["outcome"] = json!(if line == 1 { "ambiguous" } else { "errored" });
                a["error_code"] = json!(if line == 1 { "rename_unconfirmed" } else { "io_error" });
            }
            if line <= 2 {
                v["stages"]["observation"]["record"]["observed"] =
                    json!({"availability":"unavailable","value":null,"error_code":"no_read_access"});
            } else if line <= 7 {
                let o = &mut v["stages"]["observation"]["record"]["observed"]["value"];
                o["resolved_within_root"] = json!(line > 4);
                if line <= 5 { o["exists"] = json!(false); o["digest"] = Value::Null; o["len_bytes"] = Value::Null; }
                else { o["digest"] = json!("0".repeat(64)); if line == 6 { o["len_bytes"] = json!(6); } }
            }
            if line == 8 {
                v["stages"]["attempt"]["record"]["before_state"] =
                    json!({"availability":"unavailable","value":null,"error_code":"no_read_access"});
            }
            if line == 9 {
                let expected = v["stages"]["proposal"]["record"]["expected_effect"]["value"].clone();
                v["stages"]["attempt"]["record"]["before_state"]["value"] =
                    json!({"exists":true,"digest":expected["content_digest"],"len_bytes":expected["content_len_bytes"]});
            }
        });
        s6_method(&mut row);
        if let Slot::Present { record: v, .. } = &row.stages.verification {
            assert_eq!(v.outcome, outcome); assert_eq!(v.reason_code, reason);
        }
        s9_control(&row);
        if line == 10 { s9_wrong_delta(&mut row, "changed"); }
        else {
            s5_change(&mut row, |v| v["stages"]["verification"]["record"]["reason_code"] =
                json!(if line == 7 { "length_mismatch" } else { "digest_mismatch" }));
            s5_rechain(&mut row, "");
        }
        s9_check(&row, RejectionCode::VerificationInconsistent);
    }
}
#[test]
fn s9_row_277_f8_unknown_with_verified_summary() {
    let mut row = s9_missing_before(); s9_control(&row);
    row.summary = Summary::AllowedEffectVerified;
    s9_check(&row, RejectionCode::SummaryMismatch);
}
#[test]
fn s9_r2_wrong_summary_continues_after_arguments() {
    let mut row = s5_base(); row.summary = Summary::AllowedEffectUnknown;
    let resolver = s8_resolver(&row, PayloadResolution::NotConfigured, None);
    assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::SummaryMismatch));
    assert_eq!(*resolver.calls.borrow(), vec![(PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json")))]);
}
#[test]
fn s9_r9_wrong_summary_continues_after_detection() {
    let mut row = s8_two_payload(); row.summary = Summary::AllowedEffectUnknown;
    let resolver = s8_resolver(&row, s8_args(), Some(PayloadResolution::NotConfigured));
    assert_eq!(s8_validate(&row, &resolver), ValidationResult::Rejected(RejectionCode::SummaryMismatch));
    assert_eq!(*resolver.calls.borrow(), vec![
        (PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/arguments.json"))),
        (PayloadStore::ReceiptEvidence, payload_digest(include_bytes!("fixtures/receipt_v0/F74_detection.json"))),
    ]);
}
#[test]
fn s9_o8_target_binding_before_verification() {
    for (early, late, code) in [
        (true, false, RejectionCode::TargetBindingMismatch),
        (false, true, RejectionCode::VerificationInconsistent),
        (true, true, RejectionCode::TargetBindingMismatch),
    ] {
        let mut row = s5_base();
        if early { s5_change(&mut row, |v| v["stages"]["observation"]["record"]["target_ref"]["value"] = json!("0".repeat(64))); }
        if late { s9_wrong_delta(&mut row, "changed"); }
        s5_rechain(&mut row, ""); s9_check(&row, code);
    }
}
#[test]
fn s9_o9_verification_before_summary() {
    for (early, late, code) in [
        (true, false, RejectionCode::VerificationInconsistent),
        (false, true, RejectionCode::SummaryMismatch),
        (true, true, RejectionCode::VerificationInconsistent),
    ] {
        let mut row = s5_base();
        if early { s9_wrong_delta(&mut row, "changed"); }
        if late { row.summary = Summary::AllowedEffectUnknown; }
        s9_check(&row, code);
    }
}
#[test]
fn s9_f11_observation_unavailable() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["observation"]["record"]["observed"] =
        json!({"availability":"unavailable","value":null,"error_code":"no_read_access"}));
    s6_method(&mut row); s9_control(&row);
    s9_forge_verified(&mut row);
    s9_check(&row, RejectionCode::VerificationInconsistent);
}
#[test]
fn s9_f23_ambiguous_and_f24_errored() {
    for (outcome, code) in [("ambiguous", "rename_unconfirmed"), ("errored", "io_error")] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            let expected = v["stages"]["proposal"]["record"]["expected_effect"]["value"].clone();
            let a = &mut v["stages"]["attempt"]["record"];
            a["outcome"] = json!(outcome); a["error_code"] = json!(code);
            if outcome == "errored" { a["before_state"]["value"] =
                json!({"exists":true,"digest":expected["content_digest"],"len_bytes":expected["content_len_bytes"]}); }
        });
        s6_method(&mut row); s9_control(&row);
        s9_forge_verified(&mut row);
        s9_check(&row, RejectionCode::VerifiedWithoutCompletedAttempt);
    }
}
#[test]
fn s9_f25_forged_observed_digest_and_outcome() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["observation"]["record"]["observed"]["value"]["digest"] = json!("0".repeat(64)));
    s6_method(&mut row); s9_control(&row);
    s9_forge_verified(&mut row);
    s5_change(&mut row, |v| {
        let expected = v["stages"]["proposal"]["record"]["expected_effect"]["value"]["content_digest"].clone();
        v["stages"]["verification"]["record"]["comparison"]["value"]["observed_digest"] = expected;
    });
    s5_rechain(&mut row, "");
    s9_check(&row, RejectionCode::VerificationInconsistent);
}
#[test]
fn s9_f54_summary_lines_and_wrong_labels() {
    let mut rows = Vec::new();
    let mut blocked: Row = serde_json::from_str(&s7_block()).unwrap();
    s5_rechain(&mut blocked, ""); rows.push((1, blocked, Summary::BlockedBeforeExecution));
    let mut unsupported = s5_base(); s5_absent(&mut unsupported);
    s5_change(&mut unsupported, |v| {
        let p = &mut v["stages"]["proposal"]["record"];
        p["tool_name"]["value"] = json!("other"); p["effect_kind"] = json!("unsupported");
        p["expected_effect"] = json!({"availability":"absent_by_contract","value":null,"error_code":null});
    });
    unsupported.stages.attempt = Slot::AbsentByContract(AttemptAbsentReason::Unsupported);
    s5_rechain(&mut unsupported, ""); rows.push((2, unsupported, Summary::NotAttemptedUnsupported));
    let shadow: Row = serde_json::from_str(&s7_absent("shadow", true, "allowed")).unwrap();
    rows.push((3, shadow, Summary::NotAttemptedNotEnforced));
    let mut no_executor = s5_base(); s5_absent(&mut no_executor);
    rows.push((4, no_executor, Summary::NotAttemptedNoExecutor));
    for (code, vector) in [
        ("sink_rejected", [("accepted",false),("rejected",false),("accepted",false)]),
        ("sink_unavailable", [("accepted",false),("unavailable",false),("accepted",false)]),
        ("sink_ref_mismatch", [("accepted",false),("accepted",true),("accepted",false)]),
    ] { rows.push((5, s8_sink_row(code, vector), Summary::NotAttemptedSink)); }
    rows.push((6, serde_json::from_str(&s7_rejected("size_limit")).unwrap(), Summary::NotAttemptedRejected));
    rows.push((8, s5_base(), Summary::AllowedEffectVerified));
    let mut failed = s5_base();
    s5_change(&mut failed, |v| v["stages"]["observation"]["record"]["observed"]["value"]["digest"] = json!("0".repeat(64)));
    s6_method(&mut failed); rows.push((9, failed, Summary::AllowedEffectFailed));
    rows.push((10, s9_missing_before(), Summary::AllowedEffectUnknown));
    assert_eq!(rows.len(), 11);
    for (line, mut row, expected) in rows {
        assert_eq!(row.summary, expected, "line {line}"); s9_control(&row);
        row.summary = if expected == Summary::AllowedEffectUnknown { Summary::AllowedEffectVerified } else { Summary::AllowedEffectUnknown };
        s9_check(&row, RejectionCode::SummaryMismatch);
    }
    let mut incomplete = s5_base();
    incomplete.stages.verification = Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent);
    s5_rechain(&mut incomplete, "");
    assert_eq!(incomplete.summary, Summary::Incomplete);
    s9_check(&incomplete, RejectionCode::VerificationMissingAfterAttempt);
    incomplete.summary = Summary::AllowedEffectVerified;
    s9_check(&incomplete, RejectionCode::VerificationMissingAfterAttempt);
}
#[test]
fn s9_f55_unchanged_before_state() {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        let expected = v["stages"]["proposal"]["record"]["expected_effect"]["value"].clone();
        v["stages"]["attempt"]["record"]["before_state"]["value"] =
            json!({"exists":true,"digest":expected["content_digest"],"len_bytes":expected["content_len_bytes"]});
    });
    s6_method(&mut row); s9_control(&row);
    s9_forge_verified(&mut row);
    s9_check(&row, RejectionCode::VerificationInconsistent);
}
#[test]
fn s9_f56_missing_before_state() {
    let mut row = s9_missing_before(); s9_control(&row);
    s9_forge_verified(&mut row);
    s9_check(&row, RejectionCode::VerificationInconsistent);
}
#[test]
fn s9_f77_both_absent_delta_and_errored_precedence() {
    for errored in [false, true] {
        let mut row = s5_base();
        s5_change(&mut row, |v| {
            let o = &mut v["stages"]["observation"]["record"]["observed"]["value"];
            o["exists"] = json!(false); o["digest"] = Value::Null; o["len_bytes"] = Value::Null;
            if errored {
                v["stages"]["attempt"]["record"]["outcome"] = json!("errored");
                v["stages"]["attempt"]["record"]["error_code"] = json!("io_error");
            }
        });
        s6_method(&mut row); s9_control(&row);
        if let Slot::Present { record: v, .. } = &row.stages.verification {
            assert_eq!(v.outcome, VerificationOutcome::Failed);
            assert_eq!(v.reason_code, Some(if errored { VerificationReason::AttemptErrored } else { VerificationReason::TargetMissing }));
            if let PresentAbsent::Present(c) = &v.comparison {
                assert_eq!(c.delta, Delta::Unchanged); assert_eq!(c.before_digest, None); assert_eq!(c.observed_digest, None);
            } else { panic!("comparison must be present"); }
        }
        for delta in ["unknown", "created", "removed"] {
            let mut bad = row.clone(); s9_wrong_delta(&mut bad, delta);
            s9_check(&bad, RejectionCode::VerificationInconsistent);
        }
        let mut bad = row.clone();
        s5_change(&mut bad, |v| v["stages"]["verification"]["record"]["comparison"] =
            json!({"availability":"absent_by_contract","value":null,"error_code":null}));
        s5_rechain(&mut bad, "");
        s9_check(&bad, RejectionCode::FieldAvailabilityInvalid);
    }
}

// S10 measured gaps: explicit controls for steps 1/2 and location-major T1.
#[test]
fn s10_o1_schema_before_noncanonical_with_single_controls() {
    let base = s3_row_text();
    let mut unknown = base.replacen("\"stage\":\"proposal\",", "\"stage\":\"proposal\",\"extra\":true,", 1);
    assert_ne!(unknown, base);
    s6_repair(&mut unknown);
    assert_eq!(s2_validate(unknown.as_bytes()), ValidationResult::Rejected(RejectionCode::SchemaError));
    let space = format!(" {base}");
    assert_eq!(s2_validate(space.as_bytes()), ValidationResult::Rejected(RejectionCode::NonCanonicalBytes));
    let both = format!(" {unknown}");
    assert_eq!(s2_validate(both.as_bytes()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn s10_t1_proposal_schema_before_context_number_with_single_controls() {
    let base = s3_row_text();
    let mut unknown = base.replacen("\"stage\":\"proposal\",", "\"stage\":\"proposal\",\"extra\":true,", 1);
    assert_ne!(unknown, base);
    s6_repair(&mut unknown);
    assert_eq!(s2_validate(unknown.as_bytes()), ValidationResult::Rejected(RejectionCode::SchemaError));
    let mut number = base.clone();
    s6_set(&mut number, "proposal_context.record.choice_index", json!(0.0));
    s6_repair(&mut number);
    assert_eq!(s2_validate(number.as_bytes()), ValidationResult::Rejected(RejectionCode::NonIntegerNumber));
    let mut both = unknown;
    s6_set(&mut both, "proposal_context.record.choice_index", json!(0.0));
    s6_repair(&mut both);
    assert_eq!(s2_validate(both.as_bytes()), ValidationResult::Rejected(RejectionCode::SchemaError));
}

#[test]
fn fix1_f57_common_target_ref_recompute() {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        v["stages"]["attempt"]["record"]["target"]["target_ref"]["value"] = json!("0".repeat(64));
        v["stages"]["observation"]["record"]["target_ref"]["value"] = json!("0".repeat(64));
    });
    s5_rechain(&mut row, "");
    assert_eq!(s8_validate(&row, &s8_resolver(&row, s8_args(), Some(s8_detection()))),
        ValidationResult::Rejected(RejectionCode::TargetBindingMismatch));
}

#[test]
fn fix1_arguments_raw_reference_only() {
    let mut row = s5_base();
    s5_change(&mut row, |v| {
        v["proposal_context"]["record"]["arguments_payload"]["value"]["digest"] = json!("0".repeat(64));
    });
    s5_rechain(&mut row, "");
    assert_eq!(s8_validate(&row, &s8_resolver(&row, s8_args(), Some(s8_detection()))),
        ValidationResult::Rejected(RejectionCode::ArgumentsPayloadMismatch));
}

#[test]
fn fix1_verification_outcome_only() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["stages"]["observation"]["record"]["observed"]["value"]["digest"] = json!("0".repeat(64)));
    s6_method(&mut row);
    s9_control(&row);
    s5_change(&mut row, |v| v["stages"]["verification"]["record"]["outcome"] = json!("unknown"));
    s5_rechain(&mut row, "");
    assert_eq!(s2_validate(&s5_bytes(&row)), ValidationResult::Rejected(RejectionCode::VerificationInconsistent));
}

#[test]
fn fix1_f48_unavailable_request_ref_valid() {
    let mut row = s5_base();
    s5_change(&mut row, |v| v["proposal_context"]["record"]["request_ref"] =
        json!({"availability":"unavailable","value":null,"error_code":"request_id_oversize"}));
    s5_rechain(&mut row, "");
    s8_control(&row);
}

#[test]
fn fix1_f17_f62_action_id_collision_valid() {
    let mut text = s7_rejected("action_id_collision");
    s6_repair(&mut text);
    assert_eq!(s2_validate(text.as_bytes()), ValidationResult::Valid);
}
