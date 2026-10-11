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
