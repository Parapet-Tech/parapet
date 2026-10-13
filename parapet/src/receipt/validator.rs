// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Contract e7b70be3, section 16.2: validation pipeline.

use super::PayloadResolver;
use serde::{Deserialize, Serialize};

/// Contract e7b70be3, section 16.4: closed validator rejection vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionCode {
    /// Contract e7b70be3, section 16.4: `missing_stage_slot`.
    MissingStageSlot,
    /// Contract e7b70be3, section 16.4: `slot_availability_invalid`.
    SlotAvailabilityInvalid,
    /// Contract e7b70be3, section 16.4: `field_availability_invalid`.
    FieldAvailabilityInvalid,
    /// Contract e7b70be3, section 16.4: `stage_order_violation`.
    StageOrderViolation,
    /// Contract e7b70be3, section 16.4: `broken_stage_ref`.
    BrokenStageRef,
    /// Contract e7b70be3, section 16.4: `attempt_without_allow`.
    AttemptWithoutAllow,
    /// Contract e7b70be3, section 16.4: `attempt_for_unsupported`.
    AttemptForUnsupported,
    /// Contract e7b70be3, section 16.4: `attempt_without_enforcement`.
    AttemptWithoutEnforcement,
    /// Contract e7b70be3, section 16.4: `attempt_without_sink`.
    AttemptWithoutSink,
    /// Contract e7b70be3, section 16.4: `observation_after_rejected_attempt`.
    ObservationAfterRejectedAttempt,
    /// Contract e7b70be3, section 16.4: `observation_missing_after_attempt`.
    ObservationMissingAfterAttempt,
    /// Contract e7b70be3, section 16.4: `verification_missing_after_attempt`.
    VerificationMissingAfterAttempt,
    /// Contract e7b70be3, section 16.4: `verified_without_completed_attempt`.
    VerifiedWithoutCompletedAttempt,
    /// Contract e7b70be3, section 16.4: `verification_inconsistent`.
    VerificationInconsistent,
    /// Contract e7b70be3, section 16.4: `no_silent_upgrade`.
    NoSilentUpgrade,
    /// Contract e7b70be3, section 16.4: `row_identity_mismatch`.
    RowIdentityMismatch,
    /// Contract e7b70be3, section 16.4: `summary_mismatch`.
    SummaryMismatch,
    /// Contract e7b70be3, section 16.4: `non_integer_number`.
    NonIntegerNumber,
    /// Contract e7b70be3, section 16.4: `index_out_of_range`.
    IndexOutOfRange,
    /// Contract e7b70be3, section 16.4: `non_canonical_bytes`.
    NonCanonicalBytes,
    /// Contract e7b70be3, section 16.4: `error_code_required`.
    ErrorCodeRequired,
    /// Contract e7b70be3, section 16.4: `error_code_forbidden`.
    ErrorCodeForbidden,
    /// Contract e7b70be3, section 16.4: `reason_code_required`.
    ReasonCodeRequired,
    /// Contract e7b70be3, section 16.4: `reason_code_forbidden`.
    ReasonCodeForbidden,
    /// Contract e7b70be3, section 16.4: `error_code_not_in_closed_list`.
    ErrorCodeNotInClosedList,
    /// Contract e7b70be3, section 16.4: `tool_name_required_when_parsed`.
    ToolNameRequiredWhenParsed,
    /// Contract e7b70be3, section 16.4: `native_arguments_availability_mismatch`.
    NativeArgumentsAvailabilityMismatch,
    /// Contract e7b70be3, section 16.4: `arguments_payload_mismatch`.
    ArgumentsPayloadMismatch,
    /// Contract e7b70be3, section 16.4: `errored_proposal_allowed`.
    ErroredProposalAllowed,
    /// Contract e7b70be3, section 16.4: `enforcement_inconsistent`.
    EnforcementInconsistent,
    /// Contract e7b70be3, section 16.4: `policy_drift_unrejected`.
    PolicyDriftUnrejected,
    /// Contract e7b70be3, section 16.4: `attempt_content_mismatch`.
    AttemptContentMismatch,
    /// Contract e7b70be3, section 16.4: `target_binding_mismatch`.
    TargetBindingMismatch,
    /// Contract e7b70be3, section 16.4: `rejected_attempt_shape_invalid`.
    RejectedAttemptShapeInvalid,
    /// Contract e7b70be3, section 16.4: `sink_code_mismatch`.
    SinkCodeMismatch,
    /// Contract e7b70be3, section 16.4: `detection_payload_mismatch`.
    DetectionPayloadMismatch,
    /// Contract e7b70be3, section 16.4: `detection_copy_mismatch`.
    DetectionCopyMismatch,
    /// Contract e7b70be3, section 16.4: `redacted_in_canonical`.
    RedactedInCanonical,
    /// Contract e7b70be3, section 16.4: `schema_error`.
    SchemaError,
}

/// Contract e7b70be3, section 16.2: validation outcome, including unresolved payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationResult {
    /// Contract e7b70be3, section 16.2: all checks passed.
    Valid,
    /// Contract e7b70be3, section 16.4: the first rejection.
    Rejected(RejectionCode),
    /// Contract e7b70be3, sections 7.10 and 7.11: required byte checks unresolved.
    ValidUnresolved,
}

/// Contract e7b70be3, section 16.2: validate row bytes with an injected resolver.
pub fn validate_receipt(bytes: &[u8], resolver: &dyn PayloadResolver) -> ValidationResult {
    let steps: [fn(&[u8], &dyn PayloadResolver) -> Result<(), RejectionCode>; 6] = [
        step_json_shape,
        step_canonical_bytes,
        step_stage_references,
        step_slot_table,
        step_lifecycle,
        step_required_fields,
    ];
    for step in steps {
        if let Err(code) = step(bytes, resolver) {
            return ValidationResult::Rejected(code);
        }
    }
    // BQ-10(c3): parse once after permissive steps 1-6, and fail closed.
    let row = match serde_json::from_slice::<super::Row>(bytes) {
        Ok(row) => row,
        Err(_) => return ValidationResult::Rejected(RejectionCode::SchemaError),
    };
    let mut unresolved = false;
    if let Err(code) = step_attempt_bytes(&row, resolver, &mut unresolved) {
        return ValidationResult::Rejected(code);
    }
    for step in [step_target_binding, step_verification, step_summary] {
        if let Err(code) = step(&row, resolver) {
            return ValidationResult::Rejected(code);
        }
    }
    if unresolved { ValidationResult::ValidUnresolved } else { ValidationResult::Valid }
}

// BQ-22 A16-A19: retain number lexemes and duplicate object members before
// constructing a serde Value. String decoding uses serde's UTF-8/surrogate rules.
#[derive(Debug)]
enum JsonToken {
    Object(Vec<(String, JsonToken)>),
    Array(Vec<JsonToken>),
    String(String),
    Number(String),
    Null,
    Bool,
}

struct JsonTokens<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl JsonTokens<'_> {
    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.cursor)
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.cursor += 1;
        }
    }

    fn string(&mut self) -> Result<String, RejectionCode> {
        let start = self.cursor;
        if self.bytes.get(start) != Some(&b'"') {
            return Err(RejectionCode::SchemaError);
        }
        self.cursor += 1;
        let mut escaped = false;
        while let Some(&byte) = self.bytes.get(self.cursor) {
            self.cursor += 1;
            if !escaped && byte == b'"' {
                return serde_json::from_slice(&self.bytes[start..self.cursor])
                    .map_err(|_| RejectionCode::SchemaError);
            }
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            }
        }
        Err(RejectionCode::SchemaError)
    }

    fn value(&mut self, depth: usize) -> Result<JsonToken, RejectionCode> {
        if depth > 128 {
            return Err(RejectionCode::SchemaError);
        }
        self.whitespace();
        match self.bytes.get(self.cursor) {
            Some(b'"') => Ok(JsonToken::String(self.string()?)),
            Some(b'{') => {
                self.cursor += 1;
                self.whitespace();
                let mut members = Vec::new();
                if self.bytes.get(self.cursor) == Some(&b'}') {
                    self.cursor += 1;
                    return Ok(JsonToken::Object(members));
                }
                loop {
                    self.whitespace();
                    let key = self.string()?;
                    self.whitespace();
                    if self.bytes.get(self.cursor) != Some(&b':') {
                        return Err(RejectionCode::SchemaError);
                    }
                    self.cursor += 1;
                    members.push((key, self.value(depth + 1)?));
                    self.whitespace();
                    match self.bytes.get(self.cursor) {
                        Some(b'}') => {
                            self.cursor += 1;
                            break;
                        }
                        Some(b',') => self.cursor += 1,
                        _ => return Err(RejectionCode::SchemaError),
                    }
                }
                Ok(JsonToken::Object(members))
            }
            Some(b'[') => {
                self.cursor += 1;
                self.whitespace();
                let mut items = Vec::new();
                if self.bytes.get(self.cursor) == Some(&b']') {
                    self.cursor += 1;
                    return Ok(JsonToken::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.whitespace();
                    match self.bytes.get(self.cursor) {
                        Some(b']') => {
                            self.cursor += 1;
                            break;
                        }
                        Some(b',') => self.cursor += 1,
                        _ => return Err(RejectionCode::SchemaError),
                    }
                }
                Ok(JsonToken::Array(items))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.cursor;
                if self.bytes.get(self.cursor) == Some(&b'-') {
                    self.cursor += 1;
                }
                match self.bytes.get(self.cursor) {
                    Some(b'0') => self.cursor += 1,
                    Some(b'1'..=b'9') => {
                        while self.bytes.get(self.cursor).is_some_and(u8::is_ascii_digit) {
                            self.cursor += 1;
                        }
                    }
                    _ => return Err(RejectionCode::SchemaError),
                }
                if self.bytes.get(self.cursor) == Some(&b'.') {
                    self.cursor += 1;
                    let begin = self.cursor;
                    while self.bytes.get(self.cursor).is_some_and(u8::is_ascii_digit) {
                        self.cursor += 1;
                    }
                    if begin == self.cursor {
                        return Err(RejectionCode::SchemaError);
                    }
                }
                if self
                    .bytes
                    .get(self.cursor)
                    .is_some_and(|b| matches!(b, b'e' | b'E'))
                {
                    self.cursor += 1;
                    if self
                        .bytes
                        .get(self.cursor)
                        .is_some_and(|b| matches!(b, b'+' | b'-'))
                    {
                        self.cursor += 1;
                    }
                    let begin = self.cursor;
                    while self.bytes.get(self.cursor).is_some_and(u8::is_ascii_digit) {
                        self.cursor += 1;
                    }
                    if begin == self.cursor {
                        return Err(RejectionCode::SchemaError);
                    }
                }
                Ok(JsonToken::Number(
                    std::str::from_utf8(&self.bytes[start..self.cursor])
                        .map_err(|_| RejectionCode::SchemaError)?
                        .into(),
                ))
            }
            _ => {
                for literal in [b"true".as_slice(), b"false", b"null"] {
                    if self.bytes[self.cursor..].starts_with(literal) {
                        self.cursor += literal.len();
                        return Ok(if literal == b"null" {
                            JsonToken::Null
                        } else {
                            JsonToken::Bool
                        });
                    }
                }
                Err(RejectionCode::SchemaError)
            }
        }
    }
}

// BQ-11: descriptors follow the contract's table order, depth first. Row
// lifecycle slots precede context, which precedes ordinary row fields.
fn json_fields(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "RawField" => &[
            ("availability", "any"),
            ("value", "any"),
            ("error_code", "any"),
        ],
        "RawSlot" => &[
            ("availability", "any"),
            ("reason", "any"),
            ("digest", "any"),
            ("record", "any"),
        ],
        "MirrorPayloadRef" => &[
            ("digest", "any"),
            ("len_bytes", "any"),
            ("media_type", "any"),
            ("store", "any"),
        ],
        "MirrorRequestRef" => &[("request_id", "any"), ("request_id_source", "any")],
        "MirrorExpectedEffect" => &[
            ("target_path", "any"),
            ("content_digest", "any"),
            ("content_len_bytes", "any"),
            ("content_media_type", "any"),
        ],
        "MirrorProposal" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("provider", "any"),
            ("index", "any"),
            ("tool_name", "field:string"),
            ("parse_status", "any"),
            ("parse_error_code", "any"),
            ("arguments_digest", "field:hex"),
            ("arguments_len_bytes", "field:integer"),
            ("effect_kind", "any"),
            ("expected_effect", "field:MirrorExpectedEffect"),
        ],
        "MirrorProposalContext" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("proposal_ref", "any"),
            ("request_ref", "field:MirrorRequestRef"),
            ("choice_index", "any"),
            ("model", "any"),
            ("tool_call_id", "field:string"),
            ("arguments_wire_form", "any"),
            ("native_arguments", "field:MirrorPayloadRef"),
            ("arguments_payload", "field:MirrorPayloadRef"),
            ("detection_payload", "field:MirrorPayloadRef"),
        ],
        "MirrorPolicyIdentity" => &[
            ("contract_hash", "any"),
            ("policy_digest", "any"),
            ("policy_version", "any"),
        ],
        "MirrorCodeIdentity" => &[("code_version", "any"), ("code_digest", "any")],
        "MirrorDetection" => &[
            ("record_digest", "any"),
            ("record_len_bytes", "any"),
            ("schema_version", "any"),
            ("detection_outcome", "any"),
            ("mode", "any"),
            ("input_digest", "any"),
        ],
        "MirrorFailedConstraint" => &[("argument", "any"), ("constraint", "any")],
        "MirrorDecision" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("proposal_ref", "any"),
            ("policy", "MirrorPolicyIdentity"),
            ("code", "field:MirrorCodeIdentity"),
            ("detection", "field:MirrorDetection"),
            ("detection_binding_rule", "field:string"),
            ("enforcement", "any"),
            ("decision", "any"),
            ("evaluated_policy", "any"),
            ("reason_code", "any"),
            ("failed_constraints", "array:MirrorFailedConstraint"),
            ("inputs_consumed", "any"),
        ],
        "MirrorExecutor" => &[("name", "any"), ("version", "any")],
        "MirrorRoot" => &[("root_id", "any"), ("root_digest", "any")],
        "MirrorTarget" => &[
            ("proposed_path_digest", "any"),
            ("target_ref", "field:hex"),
            ("resolved_path", "field:string"),
        ],
        "MirrorSinkResult" => &[("status", "any"), ("ref", "any")],
        "MirrorSink" => &[
            ("proposal", "MirrorSinkResult"),
            ("context", "MirrorSinkResult"),
            ("decision", "MirrorSinkResult"),
        ],
        "MirrorPolicyCheck" => &[("contract_hash", "any"), ("policy_digest", "any")],
        "MirrorBeforeState" => &[("exists", "any"), ("digest", "any"), ("len_bytes", "any")],
        "MirrorAttempt" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("decision_ref", "any"),
            ("action_id", "any"),
            ("executor", "MirrorExecutor"),
            ("root", "MirrorRoot"),
            ("target", "MirrorTarget"),
            ("operation", "any"),
            ("sink", "MirrorSink"),
            ("policy_check", "MirrorPolicyCheck"),
            ("before_state", "field:MirrorBeforeState"),
            ("outcome", "any"),
            ("error_code", "any"),
            ("content_payload", "field:MirrorPayloadRef"),
        ],
        "MirrorObserver" => &[("name", "any"), ("version", "any"), ("source", "any")],
        "MirrorObserved" => &[
            ("exists", "any"),
            ("digest", "any"),
            ("len_bytes", "any"),
            ("resolved_within_root", "any"),
        ],
        "MirrorFootprint" => &[("paths_checked", "any"), ("undeclared_residue", "any")],
        "MirrorObservation" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("attempt_ref", "any"),
            ("observer", "MirrorObserver"),
            ("target_ref", "field:hex"),
            ("observed", "field:MirrorObserved"),
            ("footprint", "field:MirrorFootprint"),
        ],
        "MirrorComparison" => &[
            ("expected_digest", "any"),
            ("observed_digest", "any"),
            ("expected_len_bytes", "any"),
            ("observed_len_bytes", "any"),
            ("within_root", "any"),
            ("before_digest", "any"),
            ("delta", "any"),
        ],
        "MirrorVerification" => &[
            ("schema_version", "any"),
            ("stage", "any"),
            ("observation_ref", "any"),
            ("method", "any"),
            ("outcome", "any"),
            ("comparison", "field:MirrorComparison"),
            ("reason_code", "any"),
        ],
        "MirrorStages" => &[
            ("proposal", "slot:MirrorProposal"),
            ("decision", "slot:MirrorDecision"),
            ("attempt", "slot:MirrorAttempt"),
            ("observation", "slot:MirrorObservation"),
            ("verification", "slot:MirrorVerification"),
        ],
        "MirrorRow" => &[
            ("stages", "MirrorStages"),
            ("proposal_context", "slot:MirrorProposalContext"),
            ("schema_version", "any"),
            ("proposal_id", "any"),
            ("action_id", "any"),
            ("summary", "any"),
            ("float_policy", "any"),
            ("crosswalk", "any"),
        ],
        _ => &[],
    }
}

fn check_json_token(node: &JsonToken, kind: &str, key: &str) -> Result<(), RejectionCode> {
    if let JsonToken::Number(token) = node {
        if token == "-0" || token.contains(['.', 'e', 'E']) {
            return Err(RejectionCode::NonIntegerNumber);
        }
        let integer = token.parse::<u64>();
        if key == "index" || key == "choice_index" {
            if integer.is_err() || integer.is_ok_and(|n| n >= 1 << 31) {
                return Err(RejectionCode::IndexOutOfRange);
            }
        } else if integer.is_err() {
            return Err(RejectionCode::SchemaError);
        }
    }
    if let Some(inner) = kind.strip_prefix("array:") {
        if let JsonToken::Array(items) = node {
            for item in items {
                check_json_token(item, inner, key)?;
            }
            return Ok(());
        }
        return Err(RejectionCode::SchemaError);
    }
    let (shape, inner) = if let Some(inner) = kind.strip_prefix("field:") {
        ("RawField", inner)
    } else if let Some(inner) = kind.strip_prefix("slot:") {
        ("RawSlot", inner)
    } else {
        (kind, "any")
    };
    if kind != "any" && kind != "serde_json::Value" {
        if matches!(node, JsonToken::Null)
            && (kind.starts_with("field:") || kind.starts_with("slot:"))
        {
            return Err(RejectionCode::SchemaError);
        }
        if !json_fields(shape).is_empty() && !matches!(node, JsonToken::Object(_)) {
            check_json_token(node, "any", key)?;
            return Err(RejectionCode::SchemaError);
        }
        if matches!(kind, "string" | "hex" | "integer") {
            check_json_token(node, "any", key)?;
            return scalar_shape(node, kind);
        }
    }
    match node {
        JsonToken::Object(members) => {
            let mut seen = std::collections::HashSet::new();
            if members.iter().any(|(key, _)| !seen.insert(key)) {
                return Err(RejectionCode::SchemaError);
            }
            let fields = json_fields(shape);
            for &(field, child) in fields {
                let member = members.iter().find(|(name, _)| name == field);
                if member.is_none()
                    && (shape == "MirrorStages"
                        || (shape == "MirrorRow" && field == "proposal_context"))
                {
                    return Err(RejectionCode::MissingStageSlot);
                }
                if member.is_none() {
                    return Err(RejectionCode::SchemaError);
                }
                if let Some((_, value)) = member {
                    if field == "availability" && matches!(shape, "RawField" | "RawSlot") {
                        match value {
                            JsonToken::String(s) if s == "redacted" => {
                                return Err(RejectionCode::RedactedInCanonical)
                            }
                            JsonToken::String(s)
                                if matches!(
                                    s.as_str(),
                                    "present" | "absent_by_contract" | "unavailable" | "errored"
                                ) => {}
                            _ => {
                                check_json_token(value, "any", field)?;
                                return Err(RejectionCode::SchemaError);
                            }
                        }
                    }
                    let child = if (shape == "RawField" && field == "value")
                        || (shape == "RawSlot" && field == "record")
                    {
                        inner
                    } else {
                        child
                    };
                    // Null payloads are retained for later availability checks.
                    if !matches!(value, JsonToken::Null)
                        || !matches!(
                            (shape, field),
                            ("RawField", "value") | ("RawSlot", "record")
                        )
                    {
                        check_json_token(
                            value,
                            child,
                            if shape == "RawField" && field == "value" {
                                key
                            } else if shape == "MirrorObservation" && field == "target_ref" {
                                "observation_target_ref"
                            } else {
                                field
                            },
                        )?;
                    }
                    shared_field_shape(value, shape, field, key, members)?;
                }
            }
            // Unknown positions still obey ER6 and duplicate-key rules before
            // the generic closed-field-set error at this location.
            for (field, value) in members {
                if !fields.iter().any(|(known, _)| known == field) {
                    check_json_token(value, "any", field)?;
                    if !fields.is_empty() {
                        return Err(RejectionCode::SchemaError);
                    }
                }
            }
        }
        JsonToken::Array(items) => {
            for item in items {
                check_json_token(item, kind, key)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// Draft 12 sections 3.2, 5 and 16.3: shared schema rules. These checks
// run at the corresponding field in the existing BQ-11 traversal.
fn scalar_shape(node: &JsonToken, kind: &str) -> Result<(), RejectionCode> {
    let valid = match (kind, node) {
        ("string", JsonToken::String(_)) => true,
        ("hex", JsonToken::String(s)) => {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        ("integer", JsonToken::Number(s)) => s.parse::<u64>().is_ok(),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(RejectionCode::SchemaError)
    }
}

fn shared_field_shape(
    value: &JsonToken,
    shape: &str,
    field: &str,
    owner: &str,
    members: &[(String, JsonToken)],
) -> Result<(), RejectionCode> {
    let null = matches!(value, JsonToken::Null);
    let string = match value {
        JsonToken::String(s) => Some(s.as_str()),
        _ => None,
    };
    if (shape == "RawField" && field == "error_code")
        || (field == "parse_error_code" && shape == "MirrorProposal")
        || (field == "reason_code" && shape == "MirrorVerification")
        || (field == "error_code" && shape == "MirrorAttempt")
    {
        if null {
            return Ok(());
        }
        let Some(code) = string else {
            return Err(RejectionCode::SchemaError);
        };
        let list: &[&str] = match if shape == "RawField" { owner } else { field } {
            "tool_name" => &["missing_name"],
            "parse_error_code" => &[
                "parse_error",
                "missing_name",
                "missing_id",
                "not_an_object",
                "buffer_overflow",
                "delta_timeout",
                "stream_aborted",
            ],
            "arguments_digest" | "arguments_len_bytes" => &["parse_error"],
            "expected_effect" => &[
                "not_an_object",
                "missing_path",
                "missing_content",
                "wrong_type",
                "unsupported_argument",
            ],
            "request_ref" => &["request_id_oversize", "request_id_invalid"],
            "tool_call_id" => &["missing_id"],
            "native_arguments" => &["no_string_form", "null_arguments", "absent_arguments"],
            "arguments_payload" => &["store_not_configured", "parse_error"],
            "detection_payload" | "content_payload" => &["store_not_configured"],
            "code" => &["no_build_digest"],
            "detection" => &["scan_error"],
            "before_state" => &["no_read_access", "io_error"],
            "target_ref" => &["not_resolved"],
            "observation_target_ref" => &["path_unresolvable"],
            "resolved_path" => &["not_resolved"],
            "observed" => &["no_read_access", "io_error", "observer_failed"],
            "footprint" => &["not_configured"],
            "error_code" => &[
                "action_id_collision",
                "outside_root",
                "symlink_component",
                "traversal",
                "policy_drift",
                "size_limit",
                "sink_rejected",
                "sink_unavailable",
                "sink_ref_mismatch",
                "root_unavailable",
                "io_error",
                "interrupted",
                "rename_unconfirmed",
            ],
            "reason_code" => &[
                "attempt_ambiguous",
                "observation_missing",
                "attempt_errored",
                "outside_root",
                "target_missing",
                "length_mismatch",
                "digest_mismatch",
                "before_state_missing",
                "no_delta_observed",
            ],
            _ => &[],
        };
        let availability = members.iter().find_map(|(k, v)| {
            if k == "availability" {
                if let JsonToken::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            } else {
                None
            }
        });
        if shape == "RawField" && matches!(owner, "before_state" | "observed") {
            let paired = match code {
                "no_read_access" => availability == Some("unavailable"),
                "io_error" | "observer_failed" => availability == Some("errored"),
                _ => false,
            };
            if !paired {
                return Err(RejectionCode::ErrorCodeNotInClosedList);
            }
        }
        if !(1..=64).contains(&code.len())
            || !code
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            || !list.contains(&code)
        {
            return Err(RejectionCode::ErrorCodeNotInClosedList);
        }
    }
    if shape == "RawSlot" {
        match field {
            "reason" if !null => {
                if !string.is_some_and(|s| {
                    [
                        "blocked",
                        "unsupported",
                        "not_enforced",
                        "no_executor",
                        "attempt_absent",
                        "attempt_rejected",
                    ]
                    .contains(&s)
                }) {
                    return Err(RejectionCode::SchemaError);
                }
            }
            "digest" if !null => scalar_shape(value, "hex")?,
            "record" if !null && !matches!(value, JsonToken::Object(_)) => {
                return Err(RejectionCode::SchemaError)
            }
            _ => {}
        }
    }
    if (field.ends_with("_digest")
        || field.ends_with("_ref")
        || field == "proposal_id"
        || field == "digest")
        && !matches!(shape, "RawField" | "RawSlot")
        && !json_fields(shape)
            .iter()
            .any(|(k, t)| *k == field && t.starts_with("field:"))
        && !matches!(field, "request_ref" | "target_ref")
    {
        if null
            && matches!(
                (shape, field),
                ("MirrorBeforeState", "digest")
                    | ("MirrorObserved", "digest")
                    | ("MirrorComparison", "observed_digest" | "before_digest")
            )
        {
            return Ok(());
        }
        scalar_shape(value, "hex")?;
    }
    if shape == "MirrorSinkResult" && field == "ref" && !null {
        scalar_shape(value, "hex")?;
    }
    if field == "contract_hash" {
        scalar_shape(value, "string")?;
    }
    if field == "action_id" && !(shape == "MirrorRow" && null) {
        let valid = string.is_some_and(|s| {
            (1..=128).contains(&s.len())
                && (s.as_bytes()[0].is_ascii_lowercase() || s.as_bytes()[0].is_ascii_digit())
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._:-".contains(&b))
        });
        if !valid {
            return Err(RejectionCode::SchemaError);
        }
    }
    if shape == "MirrorRequestRef" {
        if field == "request_id" && !string.is_some_and(|s| (1..=256).contains(&s.len())) {
            return Err(RejectionCode::SchemaError);
        }
        if field == "request_id_source"
            && !string.is_some_and(|s| ["generated", "supplied"].contains(&s))
        {
            return Err(RejectionCode::SchemaError);
        }
    }
    if shape == "MirrorPayloadRef" {
        if field == "media_type" {
            scalar_shape(value, "string")?;
        }
        if field == "store" {
            let expected = if owner == "content_payload" {
                "root-evidence"
            } else {
                "receipt-evidence"
            };
            if string != Some(expected) {
                return Err(RejectionCode::SchemaError);
            }
        }
    }
    // A bare table field cannot be substituted with a wrapped field triple.
    let bare_type = match (shape, field) {
        ("MirrorProposal", "provider" | "parse_status" | "effect_kind")
        | ("MirrorProposalContext", "model" | "arguments_wire_form")
        | ("MirrorDecision", "enforcement" | "decision" | "reason_code")
        | ("MirrorExecutor", "name" | "version")
        | ("MirrorRoot", "root_id")
        | ("MirrorSinkResult", "status")
        | ("MirrorAttempt", "operation" | "outcome")
        | ("MirrorObserver", "name" | "version" | "source")
        | ("MirrorVerification", "method")
        | ("MirrorFailedConstraint", "constraint") => Some("string"),
        ("MirrorFailedConstraint", "argument") if !null => Some("string"),
        ("MirrorProposal", "index") | ("MirrorProposalContext", "choice_index") => Some("integer"),
        _ => None,
    };
    if let Some(kind) = bare_type {
        scalar_shape(value, kind)?;
    }
    if shape == "MirrorDecision" && field == "evaluated_policy" && !matches!(value, JsonToken::Bool)
    {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorDecision"
        && field == "inputs_consumed"
        && !matches!(value, JsonToken::Array(items) if items.iter().all(|i| matches!(i, JsonToken::String(_))))
    {
        return Err(RejectionCode::SchemaError);
    }
    // W2/W3: every carried record is its slot's kind, including non-present slots.
    let stage = match shape {
        "MirrorProposal" => Some("proposal"),
        "MirrorProposalContext" => Some("proposal_context"),
        "MirrorDecision" => Some("decision"),
        "MirrorAttempt" => Some("attempt"),
        "MirrorObservation" => Some("observation"),
        "MirrorVerification" => Some("verification"),
        _ => None,
    };
    if stage.is_some() && field == "stage" && string != stage {
        return Err(RejectionCode::SchemaError);
    }
    if stage.is_some() && field == "schema_version" && string != Some("parapet-effect-receipt/v0") {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorVerification"
        && field == "outcome"
        && !string.is_some_and(|s| ["verified", "failed", "unknown"].contains(&s))
    {
        return Err(RejectionCode::SchemaError);
    }
    // Draft 12 section 6: unconditional proposal vocabulary is step 1;
    // derivation from tool_name/parse_status remains a step-6 rule (A6).
    if shape == "MirrorProposal" {
        let allowed: &[&str] = match field {
            "provider" => &["openai", "anthropic"],
            "parse_status" => &["parsed", "errored"],
            "effect_kind" => &["write_file", "unsupported"],
            _ => &[],
        };
        if !allowed.is_empty() && !string.is_some_and(|s| allowed.contains(&s)) {
            return Err(RejectionCode::SchemaError);
        }
    }
    // Draft 12 section 9: unconditional decision vocabulary and list order.
    // Derivation from other fields is deliberately deferred to step 6.
    let allowed: &[&str] = match (shape, field) {
        ("MirrorDecision", "enforcement") => &["enforced", "shadow", "not_evaluated", "suppressed_by_response"],
        ("MirrorDecision", "decision") => &["allow", "block"],
        ("MirrorDecision", "reason_code") => &[
            "allowed", "tool_not_allowed", "argument_missing", "type_check", "starts_with",
            "not_contains", "matches", "one_of", "max_length", "min", "max", "url_host",
            "multiple", "proposal_unparsable", "proposal_shape_invalid", "policy_layer_off", "detection_gate",
        ],
        ("MirrorDetection", "schema_version") => &["l1-detection-record/v0"],
        ("MirrorDetection", "detection_outcome") => &["threshold_breached", "threshold_not_breached"],
        ("MirrorDetection", "mode") => &["shadow", "block"],
        ("MirrorFailedConstraint", "constraint") => &[
            "tool_allowed", "argument_present", "type_check", "starts_with", "not_contains",
            "matches", "one_of", "max_length", "min", "max", "url_host",
        ],
        _ => &[],
    };
    if !allowed.is_empty() && !string.is_some_and(|s| allowed.contains(&s)) {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "RawField" && owner == "detection_binding_rule" && field == "value"
        && !null && string != Some("last_untrusted_user_message_before_l0/v0")
    {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorDecision" && field == "failed_constraints" {
        let JsonToken::Array(items) = value else { return Err(RejectionCode::SchemaError); };
        let mut previous = None;
        for item in items {
            // Element shape and enums have already passed depth-first checks.
            let JsonToken::Object(fields) = item else { return Err(RejectionCode::SchemaError); };
            let argument = fields.iter().find_map(|(k, v)| match (k.as_str(), v) {
                ("argument", JsonToken::String(s)) => Some(Some(s.as_str())),
                ("argument", JsonToken::Null) => Some(None),
                _ => None,
            }).ok_or(RejectionCode::SchemaError)?;
            let constraint = fields.iter().find_map(|(k, v)| match (k.as_str(), v) {
                ("constraint", JsonToken::String(s)) => Some(s.as_str()),
                _ => None,
            }).ok_or(RejectionCode::SchemaError)?;
            let current = (argument, constraint);
            if previous.is_some_and(|p| p >= current) {
                return Err(RejectionCode::SchemaError);
            }
            previous = Some(current);
        }
    }
    if shape == "MirrorDecision" && field == "inputs_consumed" {
        let JsonToken::Array(items) = value else { return Err(RejectionCode::SchemaError); };
        let mut previous = None;
        for item in items {
            let rank = match item {
                JsonToken::String(s) if s == "policy" => 0,
                JsonToken::String(s) if s == "proposal" => 1,
                JsonToken::String(s) if s == "detection" => 2,
                _ => return Err(RejectionCode::SchemaError),
            };
            if previous.is_some_and(|p| p >= rank) {
                return Err(RejectionCode::SchemaError);
            }
            previous = Some(rank);
        }
    }
    // Draft 12 section 10: unconditional attempt vocabulary at step 1.
    if shape == "MirrorExecutor" && field == "name" && string != Some("parapet-file-controller") {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorAttempt" {
        if field == "operation" && string != Some("atomic_write_no_follow/v0") {
            return Err(RejectionCode::SchemaError);
        }
        if field == "outcome"
            && !string.is_some_and(|s| ["completed", "errored", "ambiguous", "rejected"].contains(&s))
        {
            return Err(RejectionCode::SchemaError);
        }
    }
    if shape == "MirrorSinkResult" && field == "status"
        && !string.is_some_and(|s| ["accepted", "rejected", "unavailable"].contains(&s))
    {
        return Err(RejectionCode::SchemaError);
    }
    // Draft 12 section 11: the v0 observer source is an unconditional literal.
    if shape == "MirrorObserver" && field == "source" && string != Some("filesystem_read") {
        return Err(RejectionCode::SchemaError);
    }
    // Draft 12 section 12: unconditional verification vocabulary at step 1.
    if shape == "MirrorVerification" && field == "method"
        && string != Some("expected-vs-observed/v0")
    {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorComparison" && field == "delta"
        && !string.is_some_and(|s| ["created", "changed", "unchanged", "removed", "unknown"].contains(&s))
    {
        return Err(RejectionCode::SchemaError);
    }
    // Draft 12 section 7: unconditional context constraints at step 1.
    if shape == "MirrorProposalContext" {
        if field == "model" && !string.is_some_and(|s| s.len() <= 256) {
            return Err(RejectionCode::SchemaError);
        }
        if field == "arguments_wire_form"
            && !string.is_some_and(|s| ["string", "object", "null_value", "absent"].contains(&s))
        {
            return Err(RejectionCode::SchemaError);
        }
    }
    // Section 6.1's v0 literal applies to every non-null carried payload,
    // including one on a non-present field (BQ-22 A2).
    if shape == "MirrorExpectedEffect"
        && field == "content_media_type"
        && string != Some("text/plain; charset=utf-8")
    {
        return Err(RejectionCode::SchemaError);
    }
    // Shared non-null payload shape (A2), independent of availability pairing.
    let scalar = match (shape, field) {
        ("MirrorPayloadRef", "len_bytes")
        | ("MirrorExpectedEffect", "content_len_bytes")
        | ("MirrorDetection", "record_len_bytes") => Some("integer"),
        ("MirrorExpectedEffect", "target_path" | "content_media_type")
        | ("MirrorPolicyIdentity", "policy_version")
        | ("MirrorCodeIdentity", "code_version")
        | ("MirrorDetection", "schema_version" | "detection_outcome" | "mode") => Some("string"),
        ("MirrorBeforeState" | "MirrorObserved", "len_bytes")
        | ("MirrorComparison", "observed_len_bytes")
            if !null =>
        {
            Some("integer")
        }
        ("MirrorComparison", "expected_len_bytes") => Some("integer"),
        ("MirrorComparison", "delta") => Some("string"),
        _ => None,
    };
    if let Some(kind) = scalar {
        scalar_shape(value, kind)?;
    }
    if matches!(
        (shape, field),
        ("MirrorBeforeState" | "MirrorObserved", "exists")
            | ("MirrorObserved", "resolved_within_root")
            | ("MirrorFootprint", "undeclared_residue")
            | ("MirrorComparison", "within_root")
    ) && !matches!(value, JsonToken::Bool)
    {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorFootprint"
        && field == "paths_checked"
        && !matches!(value, JsonToken::Array(items) if items.iter().all(|i| matches!(i, JsonToken::String(_))))
    {
        return Err(RejectionCode::SchemaError);
    }
    if shape == "MirrorRow" {
        let valid = match field {
            "schema_version" => string == Some("parapet-effect-receipt/v0"),
            "float_policy" => string == Some("bit-exact"),
            "crosswalk" => string == Some("parapet-effect-receipt-crosswalk/v0"),
            "summary" => string.is_some_and(|s| {
                [
                    "blocked_before_execution",
                    "not_attempted_unsupported",
                    "not_attempted_not_enforced",
                    "not_attempted_no_executor",
                    "not_attempted_sink",
                    "not_attempted_rejected",
                    "incomplete",
                    "allowed_effect_verified",
                    "allowed_effect_failed",
                    "allowed_effect_unknown",
                ]
                .contains(&s)
            }),
            _ => true,
        };
        if !valid {
            return Err(RejectionCode::SchemaError);
        }
    }
    Ok(())
}

// Section 16.2 step 1: JSON hazards and S4a shared schema rules.
// Record-specific enum/format rules are allocated to subsequent S4 children.
fn step_json_shape(bytes: &[u8], _resolver: &dyn PayloadResolver) -> Result<(), RejectionCode> {
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    std::str::from_utf8(bytes).map_err(|_| RejectionCode::SchemaError)?;
    let mut parser = JsonTokens { bytes, cursor: 0 };
    let node = parser.value(0)?;
    parser.whitespace();
    if parser.cursor != bytes.len() {
        return Err(RejectionCode::SchemaError);
    }
    check_json_token(&node, "MirrorRow", "")
}

// Section 16.2 step 2, ER3 and BQ-10(a2). This serializer stays private.
fn step_canonical_bytes(
    bytes: &[u8],
    _resolver: &dyn PayloadResolver,
) -> Result<(), RejectionCode> {
    canonical_bytes(bytes)
}

fn canonical_bytes(bytes: &[u8]) -> Result<(), RejectionCode> {
    // A18: tolerate a BOM for parsing, then reject it by byte equality.
    let parsed_bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    let model: MirrorRow =
        serde_json::from_slice(parsed_bytes).map_err(|_| RejectionCode::SchemaError)?;
    let canonical = serde_json::to_vec(&model).map_err(|_| RejectionCode::SchemaError)?;
    if canonical != bytes {
        return Err(RejectionCode::NonCanonicalBytes);
    }
    // Each present record is checked at its structural slot position. Later
    // ER2 checks can borrow these slices of the input, never serializer output.
    macro_rules! check_span {
        ($record:expr, $path:expr) => {
            if let Some(record) = &$record {
                let expected =
                    serde_json::to_vec(record).map_err(|_| RejectionCode::SchemaError)?;
                if carried_value(bytes, $path) != Some(expected.as_slice()) {
                    return Err(RejectionCode::NonCanonicalBytes);
                }
            }
        };
    }
    check_span!(
        model.stages.proposal.record,
        &["stages", "proposal", "record"]
    );
    check_span!(
        model.stages.decision.record,
        &["stages", "decision", "record"]
    );
    check_span!(
        model.stages.attempt.record,
        &["stages", "attempt", "record"]
    );
    check_span!(
        model.stages.observation.record,
        &["stages", "observation", "record"]
    );
    check_span!(
        model.stages.verification.record,
        &["stages", "verification", "record"]
    );
    check_span!(
        model.proposal_context.record,
        &["proposal_context", "record"]
    );
    if let Ok(row) = serde_json::from_slice::<super::Row>(parsed_bytes) {
        let typed = super::record_bytes(super::ReceiptRecord::Row(&row))
            .map_err(|_| RejectionCode::SchemaError)?;
        if typed != bytes {
            return Err(RejectionCode::NonCanonicalBytes);
        }
    }
    Ok(())
}

// Section 16.2 step 3; implemented by a later subtask.
fn step_stage_references(
    bytes: &[u8],
    _resolver: &dyn PayloadResolver,
) -> Result<(), RejectionCode> {
    let row: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| RejectionCode::SchemaError)?;
    // BQ-20/BQ-21: only a present slot with both carried components is an
    // ER7a location. Missing components belong to step 4.
    for name in RECEIPT_SLOT_ORDER {
        let slot = receipt_slot(&row, name);
        if !digestible_slot(slot) {
            continue;
        }
        let path: &[&str] = if name == "proposal_context" {
            &["proposal_context", "record"]
        } else {
            &["stages", name, "record"]
        };
        let carried = carried_value(bytes, path).ok_or(RejectionCode::SchemaError)?;
        let digest = super::record_digest(carried, super::RecordDomain::Stage);
        if slot["digest"].as_str() != Some(digest.as_str()) {
            return Err(RejectionCode::BrokenStageRef);
        }
        let reference = match name {
            "decision" | "proposal_context" => Some(("proposal", "proposal_ref")),
            "attempt" => Some(("decision", "decision_ref")),
            "observation" => Some(("attempt", "attempt_ref")),
            "verification" => Some(("observation", "observation_ref")),
            _ => None,
        };
        if let Some((referent_name, field)) = reference {
            let referent = receipt_slot(&row, referent_name);
            if digestible_slot(referent) && slot["record"][field] != referent["digest"] {
                return Err(RejectionCode::BrokenStageRef);
            }
        }
    }
    Ok(())
}

// Section 16.2 step 4; implemented by a later subtask.
fn step_slot_table(bytes: &[u8], _resolver: &dyn PayloadResolver) -> Result<(), RejectionCode> {
    let row: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| RejectionCode::SchemaError)?;
    for name in RECEIPT_SLOT_ORDER {
        let slot = receipt_slot(&row, name);
        let present = slot_present(slot);
        let required = matches!(name, "proposal" | "decision" | "proposal_context");
        let allowed = present || (!required && slot["availability"] == "absent_by_contract");
        let reason_allowed = if present {
            slot["reason"].is_null()
        } else {
            match name {
                "attempt" => matches!(slot["reason"].as_str(),
                    Some("blocked" | "unsupported" | "not_enforced" | "no_executor")),
                "observation" | "verification" => matches!(slot["reason"].as_str(),
                    Some("attempt_absent" | "attempt_rejected")),
                _ => false,
            }
        };
        if !allowed || !reason_allowed
            || present == slot["digest"].is_null()
            || present == slot["record"].is_null()
        {
            return Err(RejectionCode::SlotAvailabilityInvalid);
        }
    }
    Ok(())
}

// Section 16.2 step 5; implemented by a later subtask.
fn step_lifecycle(bytes: &[u8], _resolver: &dyn PayloadResolver) -> Result<(), RejectionCode> {
    let row: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| RejectionCode::SchemaError)?;
    // L1 is rule-major and precedes every L3-L9 rule. Context has no position.
    let mut earlier_present = true;
    for name in &RECEIPT_SLOT_ORDER[..5] {
        let present = slot_present(receipt_slot(&row, name));
        if present && !earlier_present {
            return Err(RejectionCode::StageOrderViolation);
        }
        earlier_present &= present;
    }
    // L2 has already passed step 3, so it adds no reachable rejection.
    let proposal = &row["stages"]["proposal"]["record"];
    let decision = &row["stages"]["decision"]["record"];
    let attempt_slot = receipt_slot(&row, "attempt");
    let attempt_present = slot_present(attempt_slot);
    let attempt = &attempt_slot["record"];
    let observation_present = slot_present(receipt_slot(&row, "observation"));
    let verification_slot = receipt_slot(&row, "verification");
    let verification_present = slot_present(verification_slot);
    // BQ-22 A14: allow, effect kind, enforcement, with no fourth L3 code.
    if attempt_present && decision["decision"] != "allow" {
        return Err(RejectionCode::AttemptWithoutAllow);
    }
    if attempt_present && proposal["effect_kind"] != "write_file" {
        return Err(RejectionCode::AttemptForUnsupported);
    }
    if attempt_present && decision["enforcement"] != "enforced" {
        return Err(RejectionCode::AttemptWithoutEnforcement);
    }
    if attempt_present && attempt["outcome"] == "rejected" && observation_present {
        return Err(RejectionCode::ObservationAfterRejectedAttempt);
    }
    if attempt_present && attempt["outcome"] != "rejected" {
        if !observation_present {
            return Err(RejectionCode::ObservationMissingAfterAttempt);
        }
        if !verification_present {
            return Err(RejectionCode::VerificationMissingAfterAttempt);
        }
    }
    // L6/L10 are step-1 closed-field checks; L7's full method is step 9.
    if verification_present && verification_slot["record"]["outcome"] == "verified"
        && attempt["outcome"] != "completed"
    {
        return Err(RejectionCode::VerifiedWithoutCompletedAttempt);
    }
    // L8 incomplete rows have already failed L5 (BQ-14).
    if row["proposal_id"] != row["stages"]["proposal"]["digest"]
        || (attempt_present && row["action_id"] != attempt["action_id"])
        || (!attempt_present && !row["action_id"].is_null())
    {
        return Err(RejectionCode::RowIdentityMismatch);
    }
    Ok(())
}

// Section 16.2 step 6; implemented by a later subtask.
fn step_required_fields(
    _bytes: &[u8],
    _resolver: &dyn PayloadResolver,
) -> Result<(), RejectionCode> {
    let row: MirrorRow = serde_json::from_slice(_bytes)
        .map_err(|_| RejectionCode::SchemaError)?;
    // A23 rule 1: finish the entire BQ-11 location sweep before code nullness.
    let p = row.stages.proposal.record.as_ref();
    let d = row.stages.decision.record.as_ref();
    let a = row.stages.attempt.record.as_ref();
    let o = row.stages.observation.record.as_ref();
    let v = row.stages.verification.record.as_ref();
    let c = row.proposal_context.record.as_ref();
    if let Some(p) = p {
        required_field_shape(&p.tool_name, &["present", "unavailable"])?;
        required_field_shape(&p.arguments_digest, &["present", "errored"])?;
        if (p.arguments_digest.availability == "errored") != (p.parse_status == "errored") {
            return Err(RejectionCode::FieldAvailabilityInvalid);
        }
        required_field_shape(&p.arguments_len_bytes, &["present", "errored"])?;
        required_field_pair(&p.arguments_len_bytes, &p.arguments_digest)?;
        required_field_shape(&p.expected_effect, &["present", "absent_by_contract", "errored"])?;
        if (p.expected_effect.availability == "absent_by_contract") != (p.effect_kind == "unsupported") {
            return Err(RejectionCode::FieldAvailabilityInvalid);
        }
    }
    if let Some(d) = d {
        required_field_shape(&d.code, &["present", "unavailable"])?;
        required_field_shape(&d.detection, &["present", "absent_by_contract", "unavailable"])?;
        required_field_shape(&d.detection_binding_rule, &["present", "absent_by_contract"])?;
        if (d.detection_binding_rule.availability == "absent_by_contract") !=
            (d.detection.availability == "absent_by_contract") {
            return Err(RejectionCode::FieldAvailabilityInvalid);
        }
    }
    if let Some(a) = a {
        required_field_shape(&a.target.target_ref, &["present", "unavailable"])?;
        required_field_shape(&a.target.resolved_path, &["present", "unavailable"])?;
        required_field_pair(&a.target.resolved_path, &a.target.target_ref)?;
        required_field_shape(&a.before_state, &["present", "unavailable", "errored", "absent_by_contract"])?;
        required_field_shape(&a.content_payload, &["present", "unavailable", "absent_by_contract"])?;
    }
    if let Some(o) = o {
        required_field_shape(&o.target_ref, &["present", "unavailable"])?;
        required_field_shape(&o.observed, &["present", "unavailable", "errored"])?;
        required_field_shape(&o.footprint, &["present", "unavailable"])?;
    }
    if let Some(v) = v {
        required_field_shape(&v.comparison, &["present", "absent_by_contract"])?;
        if (v.comparison.availability == "present") !=
            o.is_some_and(|o| o.observed.availability == "present") {
            return Err(RejectionCode::FieldAvailabilityInvalid);
        }
    }
    if let Some(c) = c {
        required_field_shape(&c.request_ref, &["present", "unavailable"])?;
        required_field_shape(&c.tool_call_id, &["present", "unavailable"])?;
        required_field_shape(&c.native_arguments, &["present", "unavailable"])?;
        required_field_shape(&c.arguments_payload, &["present", "unavailable"])?;
        required_field_shape(&c.detection_payload, &["present", "unavailable", "absent_by_contract"])?;
        if (c.detection_payload.availability == "absent_by_contract") !=
            !d.is_some_and(|d| d.detection.availability == "present") {
            return Err(RejectionCode::FieldAvailabilityInvalid);
        }
    }
    // A23 rules 2 and 3 each sweep all fields before the next rule runs.
    for required in [true, false] {
        if let Some(p) = p {
            required_field_code(&p.tool_name, required)?;
            required_code(p.parse_status == "errored", p.parse_error_code.as_ref(), required)?;
            required_field_code(&p.arguments_digest, required)?;
            required_field_code(&p.arguments_len_bytes, required)?;
            required_field_code(&p.expected_effect, required)?;
        }
        if let Some(d) = d {
            required_field_code(&d.code, required)?;
            required_field_code(&d.detection, required)?;
            required_field_code(&d.detection_binding_rule, required)?;
        }
        if let Some(a) = a {
            required_field_code(&a.target.target_ref, required)?;
            required_field_code(&a.target.resolved_path, required)?;
            required_field_code(&a.before_state, required)?;
            required_code(a.outcome != "completed", a.error_code.as_ref(), required)?;
            required_field_code(&a.content_payload, required)?;
        }
        if let Some(o) = o {
            required_field_code(&o.target_ref, required)?;
            required_field_code(&o.observed, required)?;
            required_field_code(&o.footprint, required)?;
        }
        if let Some(v) = v { required_field_code(&v.comparison, required)?; }
        if let Some(c) = c {
            required_field_code(&c.request_ref, required)?;
            required_field_code(&c.tool_call_id, required)?;
            required_field_code(&c.native_arguments, required)?;
            required_field_code(&c.arguments_payload, required)?;
            required_field_code(&c.detection_payload, required)?;
        }
    }
    // A23 rules 4 and 5: verification reason nullness (decision reasons differ).
    if let Some(v) = v {
        let non_null = v.reason_code.as_ref().is_some_and(|code| !code.is_null());
        if v.outcome != "verified" && !non_null { return Err(RejectionCode::ReasonCodeRequired); }
        if v.outcome == "verified" && non_null { return Err(RejectionCode::ReasonCodeForbidden); }
    }
    // A23 rules 6 and 7. S7 appends rules 8-12 after these checks.
    if p.is_some_and(|p| p.parse_status == "parsed" && p.tool_name.availability != "present") {
        return Err(RejectionCode::ToolNameRequiredWhenParsed);
    }
    if c.is_some_and(|c| (c.native_arguments.availability == "present") !=
        (c.arguments_wire_form == "string")) {
        return Err(RejectionCode::NativeArgumentsAvailabilityMismatch);
    }
    // A23 rule 8: BQ-17 enforcement consistency, rows 174 and 175.
    if let Some(d) = d {
        let off = d.evaluated_policy == false && d.decision == "allow" &&
            d.reason_code == "policy_layer_off";
        if (d.enforcement == "not_evaluated") != off ||
            (d.enforcement == "suppressed_by_response" &&
                (d.decision != "allow" || d.evaluated_policy != true)) {
            return Err(RejectionCode::EnforcementInconsistent);
        }
    }
    // A23 rule 9: proposal parse/shape errors require block (120, 180).
    if p.is_some_and(|p| p.parse_status == "errored" ||
        p.expected_effect.availability == "errored") &&
        d.is_some_and(|d| d.decision != "block") {
        return Err(RejectionCode::ErroredProposalAllowed);
    }
    // A23 rule 10: schema derivations in placement-table row order.
    if let Some(p) = p {
        let write = p.parse_status == "parsed" && p.tool_name.availability == "present" &&
            p.tool_name.value.as_ref().is_some_and(|name| name == "write_file");
        if (p.effect_kind == "write_file") != write { return Err(RejectionCode::SchemaError); }
        if p.expected_effect.availability == "errored" && d.is_some_and(|d|
            d.evaluated_policy != false || d.reason_code != "proposal_shape_invalid") &&
            p.parse_status != "errored" {
            return Err(RejectionCode::SchemaError);
        }
    }
    if let (Some(p), Some(d)) = (p, d) {
        let cause = if p.parse_status == "errored" { Some("proposal_unparsable") }
            else if p.expected_effect.availability == "errored" { Some("proposal_shape_invalid") }
            else if d.enforcement == "not_evaluated" { Some("policy_layer_off") }
            else { None };
        if (d.evaluated_policy == false) != cause.is_some() {
            return Err(RejectionCode::SchemaError);
        }
        let reason = match cause {
            Some(cause) => cause,
            None => match d.failed_constraints.as_slice() {
                [] => "allowed",
                [failure] => match failure.constraint.as_str() {
                    Some("tool_allowed") => "tool_not_allowed",
                    Some("argument_present") => "argument_missing",
                    Some(predicate) => predicate,
                    None => return Err(RejectionCode::SchemaError),
                },
                _ => "multiple",
            },
        };
        if d.reason_code != reason { return Err(RejectionCode::SchemaError); }
        if d.reason_code == "detection_gate" { return Err(RejectionCode::SchemaError); }
        if (d.decision == "allow" || d.evaluated_policy == false) &&
            !d.failed_constraints.is_empty() { return Err(RejectionCode::SchemaError); }
        if d.failed_constraints.iter().any(|f| f.constraint == "tool_allowed" &&
            (d.failed_constraints.len() != 1 || f.argument.as_ref().is_some_and(|a| !a.is_null()))) {
            return Err(RejectionCode::SchemaError);
        }
        if d.failed_constraints.iter().any(|f| f.constraint == "argument_present" &&
            d.failed_constraints.iter().any(|other|
                other.argument == f.argument && other.constraint != "argument_present")) {
            return Err(RejectionCode::SchemaError);
        }
        if d.decision == "block" && d.evaluated_policy == true &&
            d.failed_constraints.is_empty() { return Err(RejectionCode::SchemaError); }
    }
    if let Some(a) = a {
        if a.before_state.value.as_ref().is_some_and(|b|
            !required_state_nullness(&b.exists, b.digest.as_ref(), b.len_bytes.as_ref())) {
            return Err(RejectionCode::SchemaError);
        }
        if (a.outcome == "errored" || a.outcome == "ambiguous") &&
            required_preflight_code(a.error_code.as_ref()) {
            return Err(RejectionCode::SchemaError);
        }
    }
    if o.is_some_and(|o| o.observed.value.as_ref().is_some_and(|b|
        !required_state_nullness(&b.exists, b.digest.as_ref(), b.len_bytes.as_ref()))) {
        return Err(RejectionCode::SchemaError);
    }
    // A23 rule 11: rejected-attempt shape, rows 205, 224, 229, 232.
    if let Some(a) = a {
        let rejected = a.outcome == "rejected";
        let unresolved = rejected && required_preflight_code(a.error_code.as_ref()) &&
            !a.error_code.as_ref().is_some_and(|code| code == "size_limit");
        if (a.target.target_ref.availability == "unavailable") != unresolved ||
            (a.before_state.availability == "absent_by_contract") != rejected ||
            (rejected && !required_preflight_code(a.error_code.as_ref())) ||
            (a.content_payload.availability == "absent_by_contract") != rejected {
            return Err(RejectionCode::RejectedAttemptShapeInvalid);
        }
    }
    // A23 rule 12: BQ-18 correspondence, after all preceding rules.
    if row.stages.attempt.availability != "present" {
        let reason = if d.is_some_and(|d| d.decision == "block") { "blocked" }
            else if p.is_some_and(|p| p.effect_kind == "unsupported") { "unsupported" }
            else if d.is_some_and(|d| d.enforcement != "enforced") { "not_enforced" }
            else { "no_executor" };
        if row.stages.attempt.reason != reason { return Err(RejectionCode::SlotAvailabilityInvalid); }
    }
    let reason = if a.is_none() { Some("attempt_absent") }
        else if a.is_some_and(|a| a.outcome == "rejected") { Some("attempt_rejected") }
        else { None };
    for slot_reason in [&row.stages.observation.reason, &row.stages.verification.reason] {
        if reason.is_some_and(|reason| slot_reason != reason) {
            return Err(RejectionCode::SlotAvailabilityInvalid);
        }
    }
    Ok(())
}

// Section 16.2 step 7; implemented by a later subtask.
fn step_attempt_bytes(
    row: &super::Row,
    resolver: &dyn PayloadResolver,
    unresolved: &mut bool,
) -> Result<(), RejectionCode> {
    use super::{AttemptError, AttemptOutcome, PayloadResolution, PayloadStore,
        PresentAbsentErrored, PresentErrored, PresentUnavailable, PresentUnavailableAbsent, Slot};
    let proposal = &row.stages.proposal.record;
    let decision = &row.stages.decision.record;
    if let Slot::Present { record: attempt, .. } = &row.stages.attempt {
        let sink_error = s8_sink_error(row, attempt);
        // Rule-major: sink acceptance, sink code, drift, then content.
        if attempt.outcome != AttemptOutcome::Rejected && sink_error.is_some() {
            return Err(RejectionCode::AttemptWithoutSink);
        }
        let has_sink_code = matches!(attempt.error_code, Some(AttemptError::SinkUnavailable |
            AttemptError::SinkRejected | AttemptError::SinkRefMismatch));
        if sink_error.is_some_and(|code| attempt.error_code != Some(code)) ||
            (sink_error.is_none() && has_sink_code) {
            return Err(RejectionCode::SinkCodeMismatch);
        }
        if attempt.outcome != AttemptOutcome::Rejected &&
            (attempt.policy_check.contract_hash != decision.policy.contract_hash ||
             attempt.policy_check.policy_digest != decision.policy.policy_digest) {
            return Err(RejectionCode::PolicyDriftUnrejected);
        }
        if let (AttemptOutcome::Completed, PresentUnavailableAbsent::Present(payload),
            PresentAbsentErrored::Present(expected)) =
            (attempt.outcome, &attempt.content_payload, &proposal.expected_effect) {
            if payload.digest != expected.content_digest || payload.len_bytes != expected.content_len_bytes {
                return Err(RejectionCode::AttemptContentMismatch);
            }
        }
    }
    // Arguments precede detection; row-only length checks precede each lookup.
    if let PresentUnavailable::Present(payload) = &row.proposal_context.record.arguments_payload {
        if !matches!(proposal.arguments_len_bytes, PresentErrored::Present(n) if n == payload.len_bytes) {
            return Err(RejectionCode::ArgumentsPayloadMismatch);
        }
        match resolver.resolve(PayloadStore::ReceiptEvidence, &payload.digest) {
            PayloadResolution::NotFound => return Err(RejectionCode::ArgumentsPayloadMismatch),
            PayloadResolution::NotConfigured => *unresolved = true,
            PayloadResolution::Found(bytes) => {
                if super::payload_digest(&bytes) != payload.digest || bytes.len() as u64 != payload.len_bytes ||
                    !matches!(&proposal.arguments_digest, PresentErrored::Present(d) if *d == super::arguments_digest(&bytes)) {
                    return Err(RejectionCode::ArgumentsPayloadMismatch);
                }
            }
        }
    }
    if let PresentUnavailableAbsent::Present(payload) = &row.proposal_context.record.detection_payload {
        let PresentUnavailableAbsent::Present(detection) = &decision.detection else {
            return Err(RejectionCode::SchemaError);
        };
        if payload.len_bytes != detection.record_len_bytes {
            return Err(RejectionCode::DetectionPayloadMismatch);
        }
        match resolver.resolve(PayloadStore::ReceiptEvidence, &payload.digest) {
            PayloadResolution::NotFound => return Err(RejectionCode::DetectionPayloadMismatch),
            PayloadResolution::NotConfigured => *unresolved = true,
            PayloadResolution::Found(bytes) => {
                if super::payload_digest(&bytes) != payload.digest || bytes.len() as u64 != payload.len_bytes ||
                    super::detection_record_digest(&bytes) != detection.record_digest {
                    return Err(RejectionCode::DetectionPayloadMismatch);
                }
                let copy: S8DetectionCopy = serde_json::from_slice(&bytes)
                    .map_err(|_| RejectionCode::DetectionCopyMismatch)?;
                if copy.detection_outcome != detection.detection_outcome || copy.mode != detection.mode ||
                    copy.input_digest != detection.input_digest {
                    return Err(RejectionCode::DetectionCopyMismatch);
                }
            }
        }
    }
    Ok(())
}

// Section 16.2 step 8; implemented by a later subtask.
fn step_target_binding(
    row: &super::Row,
    _resolver: &dyn PayloadResolver,
) -> Result<(), RejectionCode> {
    use super::{PresentAbsentErrored, PresentUnavailable, Slot};
    if let Slot::Present { record: attempt, .. } = &row.stages.attempt {
        // BQ-11 location order: attempt record (83,85), then observation (84).
        if let (PresentUnavailable::Present(reference), PresentUnavailable::Present(path)) =
            (&attempt.target.target_ref, &attempt.target.resolved_path) {
            let root = s8_digest_bytes(&attempt.root.root_digest)?;
            if *reference != super::target_ref(&root, path) {
                return Err(RejectionCode::TargetBindingMismatch);
            }
        }
        if let PresentAbsentErrored::Present(expected) = &row.stages.proposal.record.expected_effect {
            if attempt.target.proposed_path_digest != super::proposed_path_digest(&expected.target_path) {
                return Err(RejectionCode::TargetBindingMismatch);
            }
        }
        if let Slot::Present { record: observation, .. } = &row.stages.observation {
            if let (PresentUnavailable::Present(observed), PresentUnavailable::Present(attempted)) =
                (&observation.target_ref, &attempt.target.target_ref) {
                if observed != attempted { return Err(RejectionCode::TargetBindingMismatch); }
            }
        }
    }
    Ok(())
}

// Section 16.2 step 9: compare the section 12.2 re-run.
fn step_verification(row: &super::Row, _resolver: &dyn PayloadResolver) -> Result<(), RejectionCode> {
    if let super::Slot::Present { record: verification, .. } = &row.stages.verification {
        let (attempt, observation) = match (&row.stages.attempt, &row.stages.observation) {
            (super::Slot::Present { record: attempt, .. }, super::Slot::Present { record: observation, .. }) => {
                (attempt, observation)
            }
            _ => return Err(RejectionCode::VerificationInconsistent),
        };
        let method = match super::verify_expected_effect(&row.stages.proposal.record, attempt, observation) {
            Some(method) => method,
            None => return Err(RejectionCode::VerificationInconsistent),
        };
        if verification.outcome != method.outcome
            || verification.reason_code != method.reason_code
            || verification.comparison != method.comparison
        {
            return Err(RejectionCode::VerificationInconsistent);
        }
    }
    Ok(())
}

// Section 16.2 step 10: compare the section 14.2 recompute.
fn step_summary(row: &super::Row, _resolver: &dyn PayloadResolver) -> Result<(), RejectionCode> {
    if row.summary != super::receipt_summary(row) {
        return Err(RejectionCode::SummaryMismatch);
    }
    Ok(())
}

// BQ-10(a): validator-private table-order mirrors. Vocabulary and availability
// correspondence are deliberately left to steps 1, 4 and 6.
#[derive(Debug, Serialize, Deserialize)]
struct RawField<T> {
    availability: serde_json::Value,
    value: Option<T>,
    error_code: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct RawSlot<T> {
    availability: serde_json::Value,
    reason: serde_json::Value,
    digest: serde_json::Value,
    record: Option<T>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorPayloadRef {
    digest: serde_json::Value,
    len_bytes: serde_json::Value,
    media_type: serde_json::Value,
    store: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorRequestRef {
    request_id: serde_json::Value,
    request_id_source: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorExpectedEffect {
    target_path: serde_json::Value,
    content_digest: serde_json::Value,
    content_len_bytes: serde_json::Value,
    content_media_type: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorProposal {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    provider: serde_json::Value,
    index: serde_json::Value,
    tool_name: RawField<serde_json::Value>,
    parse_status: serde_json::Value,
    parse_error_code: Option<serde_json::Value>,
    arguments_digest: RawField<serde_json::Value>,
    arguments_len_bytes: RawField<serde_json::Value>,
    effect_kind: serde_json::Value,
    expected_effect: RawField<MirrorExpectedEffect>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorProposalContext {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    proposal_ref: serde_json::Value,
    request_ref: RawField<MirrorRequestRef>,
    choice_index: serde_json::Value,
    model: serde_json::Value,
    tool_call_id: RawField<serde_json::Value>,
    arguments_wire_form: serde_json::Value,
    native_arguments: RawField<MirrorPayloadRef>,
    arguments_payload: RawField<MirrorPayloadRef>,
    detection_payload: RawField<MirrorPayloadRef>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorPolicyIdentity {
    contract_hash: serde_json::Value,
    policy_digest: serde_json::Value,
    policy_version: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorCodeIdentity {
    code_version: serde_json::Value,
    code_digest: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorDetection {
    record_digest: serde_json::Value,
    record_len_bytes: serde_json::Value,
    schema_version: serde_json::Value,
    detection_outcome: serde_json::Value,
    mode: serde_json::Value,
    input_digest: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorFailedConstraint {
    argument: Option<serde_json::Value>,
    constraint: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorDecision {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    proposal_ref: serde_json::Value,
    policy: MirrorPolicyIdentity,
    code: RawField<MirrorCodeIdentity>,
    detection: RawField<MirrorDetection>,
    detection_binding_rule: RawField<serde_json::Value>,
    enforcement: serde_json::Value,
    decision: serde_json::Value,
    evaluated_policy: serde_json::Value,
    reason_code: serde_json::Value,
    failed_constraints: Vec<MirrorFailedConstraint>,
    inputs_consumed: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorExecutor {
    name: serde_json::Value,
    version: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorRoot {
    root_id: serde_json::Value,
    root_digest: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorTarget {
    proposed_path_digest: serde_json::Value,
    target_ref: RawField<serde_json::Value>,
    resolved_path: RawField<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorSinkResult {
    status: serde_json::Value,
    r#ref: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorSink {
    proposal: MirrorSinkResult,
    context: MirrorSinkResult,
    decision: MirrorSinkResult,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorPolicyCheck {
    contract_hash: serde_json::Value,
    policy_digest: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorBeforeState {
    exists: serde_json::Value,
    digest: Option<serde_json::Value>,
    len_bytes: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorAttempt {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    decision_ref: serde_json::Value,
    action_id: serde_json::Value,
    executor: MirrorExecutor,
    root: MirrorRoot,
    target: MirrorTarget,
    operation: serde_json::Value,
    sink: MirrorSink,
    policy_check: MirrorPolicyCheck,
    before_state: RawField<MirrorBeforeState>,
    outcome: serde_json::Value,
    error_code: Option<serde_json::Value>,
    content_payload: RawField<MirrorPayloadRef>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorObserver {
    name: serde_json::Value,
    version: serde_json::Value,
    source: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorObserved {
    exists: serde_json::Value,
    digest: Option<serde_json::Value>,
    len_bytes: Option<serde_json::Value>,
    resolved_within_root: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorFootprint {
    paths_checked: Vec<serde_json::Value>,
    undeclared_residue: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorObservation {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    attempt_ref: serde_json::Value,
    observer: MirrorObserver,
    target_ref: RawField<serde_json::Value>,
    observed: RawField<MirrorObserved>,
    footprint: RawField<MirrorFootprint>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorComparison {
    expected_digest: serde_json::Value,
    observed_digest: Option<serde_json::Value>,
    expected_len_bytes: serde_json::Value,
    observed_len_bytes: Option<serde_json::Value>,
    within_root: serde_json::Value,
    before_digest: Option<serde_json::Value>,
    delta: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorVerification {
    schema_version: serde_json::Value,
    stage: serde_json::Value,
    observation_ref: serde_json::Value,
    method: serde_json::Value,
    outcome: serde_json::Value,
    comparison: RawField<MirrorComparison>,
    reason_code: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorStages {
    proposal: RawSlot<MirrorProposal>,
    decision: RawSlot<MirrorDecision>,
    attempt: RawSlot<MirrorAttempt>,
    observation: RawSlot<MirrorObservation>,
    verification: RawSlot<MirrorVerification>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MirrorRow {
    schema_version: serde_json::Value,
    proposal_id: serde_json::Value,
    action_id: Option<serde_json::Value>,
    stages: MirrorStages,
    proposal_context: RawSlot<MirrorProposalContext>,
    summary: serde_json::Value,
    float_policy: serde_json::Value,
    crosswalk: serde_json::Value,
}

// Step-6 helpers operate on the permissive mirrors; no typed handoff or resolver.
fn required_field_shape<T>(field: &RawField<T>, allowed: &[&str]) -> Result<(), RejectionCode> {
    if !allowed.iter().any(|availability| field.availability == *availability) ||
        (field.availability == "present") != field.value.is_some() {
        return Err(RejectionCode::FieldAvailabilityInvalid);
    }
    Ok(())
}

fn required_field_pair<T, U>(left: &RawField<T>, right: &RawField<U>) -> Result<(), RejectionCode> {
    if left.availability != right.availability || left.error_code != right.error_code {
        return Err(RejectionCode::FieldAvailabilityInvalid);
    }
    Ok(())
}

fn required_code(needed: bool, code: Option<&serde_json::Value>, required: bool) -> Result<(), RejectionCode> {
    let non_null = code.is_some_and(|code| !code.is_null());
    if required && needed && !non_null { return Err(RejectionCode::ErrorCodeRequired); }
    if !required && !needed && non_null { return Err(RejectionCode::ErrorCodeForbidden); }
    Ok(())
}

fn required_field_code<T>(field: &RawField<T>, required: bool) -> Result<(), RejectionCode> {
    required_code(field.availability == "unavailable" || field.availability == "errored",
        Some(&field.error_code), required)
}

// BQ-15: locate values in the carried canonical bytes, rather than returning
// re-serialized bytes for later digest checks. Called only after ER3 equality.
fn carried_value<'a>(bytes: &'a [u8], path: &[&str]) -> Option<&'a [u8]> {
    if path.is_empty() {
        return Some(bytes);
    }
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut cursor = 1;
    while cursor < bytes.len() && bytes[cursor] != b'}' {
        let key_end = value_end(bytes, cursor)?;
        let key: String = serde_json::from_slice(&bytes[cursor..key_end]).ok()?;
        if bytes.get(key_end) != Some(&b':') {
            return None;
        }
        let start = key_end + 1;
        let end = value_end(bytes, start)?;
        if key == path[0] {
            return carried_value(&bytes[start..end], &path[1..]);
        }
        cursor = end;
        if bytes.get(cursor) == Some(&b',') {
            cursor += 1;
        }
    }
    None
}

// Scan a value boundary in already canonical JSON. Escaped quotes and brackets
// inside strings do not affect the boundary.
fn value_end(bytes: &[u8], start: usize) -> Option<usize> {
    let first = *bytes.get(start)?;
    let mut string = first == b'"';
    let mut escaped = false;
    let mut depth = usize::from(first == b'{' || first == b'[');
    for (i, &byte) in bytes
        .iter()
        .enumerate()
        .skip(start + usize::from(string || depth > 0))
    {
        if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
        } else {
            match byte {
                b'"' => string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' if depth > 0 => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                b',' | b'}' | b']' if depth == 0 => return Some(i),
                _ => {}
            }
        }
    }
    if !string && depth == 0 {
        Some(bytes.len())
    } else {
        None
    }
}

// BQ-11 location order shared by steps 3-5; context is not a lifecycle stage.
const RECEIPT_SLOT_ORDER: [&str; 6] = [
    "proposal", "decision", "attempt", "observation", "verification", "proposal_context",
];

fn receipt_slot<'a>(row: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    if name == "proposal_context" { &row[name] } else { &row["stages"][name] }
}

fn slot_present(slot: &serde_json::Value) -> bool {
    slot["availability"] == "present"
}

fn digestible_slot(slot: &serde_json::Value) -> bool {
    slot_present(slot) && !slot["digest"].is_null() && !slot["record"].is_null()
}

// BQ-22 nullness is relational, so it belongs to step 6 rather than shape.
fn required_state_nullness(exists: &serde_json::Value,
    digest: Option<&serde_json::Value>, len: Option<&serde_json::Value>) -> bool {
    let digest_present = digest.is_some_and(|v| !v.is_null());
    let len_present = len.is_some_and(|v| !v.is_null());
    digest_present == (exists == true) && len_present == (exists == true)
}

fn required_preflight_code(code: Option<&serde_json::Value>) -> bool {
    code.and_then(serde_json::Value::as_str).is_some_and(|code| matches!(code,
        "sink_rejected" | "sink_unavailable" | "sink_ref_mismatch" | "policy_drift" |
        "action_id_collision" | "root_unavailable" | "traversal" | "symlink_component" |
        "outside_root" | "size_limit"))
}

// Step-7 sink precedence is over the complete three-result vector.
fn s8_sink_error(row: &super::Row, attempt: &super::Attempt) -> Option<super::AttemptError> {
    use super::{AttemptError, SinkStatus};
    let vector = [(&attempt.sink.proposal, row.stages.proposal.digest.as_str()),
        (&attempt.sink.context, row.proposal_context.digest.as_str()),
        (&attempt.sink.decision, attempt.decision_ref.as_str())];
    if vector.iter().any(|(result, _)| result.status == SinkStatus::Unavailable) {
        Some(AttemptError::SinkUnavailable)
    } else if vector.iter().any(|(result, _)| result.status == SinkStatus::Rejected) {
        Some(AttemptError::SinkRejected)
    } else if vector.iter().any(|(result, expected)| result.r#ref.as_deref() != Some(*expected)) {
        Some(AttemptError::SinkRefMismatch)
    } else { None }
}

// Only the three copied fields are interpreted; the detection schema is owned by L1.
#[derive(Deserialize)]
struct S8DetectionCopy {
    detection_outcome: super::DetectionOutcome,
    mode: super::DetectionMode,
    input_digest: String,
}

fn s8_digest_bytes(digest: &str) -> Result<[u8; 32], RejectionCode> {
    let mut raw = [0; 32];
    if digest.len() != 64 { return Err(RejectionCode::SchemaError); }
    for (i, pair) in digest.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).map_err(|_| RejectionCode::SchemaError)?;
        raw[i] = u8::from_str_radix(text, 16).map_err(|_| RejectionCode::SchemaError)?;
    }
    Ok(raw)
}

#[cfg(test)]
mod canonical_tests {
    use super::super::{record_bytes, ReceiptRecord, Row};
    use super::*;

    fn check_row(value: &serde_json::Value, count: &mut usize) {
        macro_rules! check_record {
            ($typed:ty, $mirror:ty, $variant:ident) => {
                if let Ok(record) = serde_json::from_value::<$typed>(value.clone()) {
                    let typed = record_bytes(ReceiptRecord::$variant(&record)).unwrap();
                    let mirror: $mirror = serde_json::from_slice(&typed).unwrap();
                    assert_eq!(serde_json::to_vec(&mirror).unwrap(), typed);
                }
            };
        }
        check_record!(super::super::Proposal, MirrorProposal, Proposal);
        check_record!(
            super::super::ProposalContext,
            MirrorProposalContext,
            ProposalContext
        );
        check_record!(super::super::Decision, MirrorDecision, Decision);
        check_record!(super::super::Attempt, MirrorAttempt, Attempt);
        check_record!(super::super::Observation, MirrorObservation, Observation);
        check_record!(super::super::Verification, MirrorVerification, Verification);

        if let Ok(row) = serde_json::from_value::<Row>(value.clone()) {
            let typed = record_bytes(ReceiptRecord::Row(&row)).unwrap();
            let mirror: MirrorRow = serde_json::from_slice(&typed).unwrap();
            assert_eq!(serde_json::to_vec(&mirror).unwrap(), typed);
            // BQ-15: the bytes at each slot's record position are verbatim.
            for (path, record) in [
                (
                    vec!["stages", "proposal", "record"],
                    Some(ReceiptRecord::Proposal(&row.stages.proposal.record)),
                ),
                (
                    vec!["stages", "decision", "record"],
                    Some(ReceiptRecord::Decision(&row.stages.decision.record)),
                ),
                (
                    vec!["stages", "attempt", "record"],
                    match &row.stages.attempt {
                        super::super::Slot::Present { record, .. } => {
                            Some(ReceiptRecord::Attempt(record))
                        }
                        _ => None,
                    },
                ),
                (
                    vec!["stages", "observation", "record"],
                    match &row.stages.observation {
                        super::super::Slot::Present { record, .. } => {
                            Some(ReceiptRecord::Observation(record))
                        }
                        _ => None,
                    },
                ),
                (
                    vec!["stages", "verification", "record"],
                    match &row.stages.verification {
                        super::super::Slot::Present { record, .. } => {
                            Some(ReceiptRecord::Verification(record))
                        }
                        _ => None,
                    },
                ),
                (
                    vec!["proposal_context", "record"],
                    Some(ReceiptRecord::ProposalContext(&row.proposal_context.record)),
                ),
            ] {
                if let Some(record) = record {
                    assert_eq!(
                        carried_value(&typed, &path).unwrap(),
                        record_bytes(record).unwrap()
                    );
                }
            }
            *count += 1;
        }
        match value {
            serde_json::Value::Object(fields) => {
                for nested in fields.values() {
                    check_row(nested, count);
                }
            }
            serde_json::Value::Array(items) => {
                for nested in items {
                    check_row(nested, count);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn s2_all_typed_fixture_rows_match_both_serializers_and_carried_spans() {
        let mut count = 0;
        for entry in std::fs::read_dir(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/receipt_v0"
        ))
        .unwrap()
        {
            let path = entry.unwrap().path();
            if let Ok(value) = serde_json::from_slice(&std::fs::read(path).unwrap()) {
                check_row(&value, &mut count);
            }
        }
        assert!(
            count > 10,
            "fixture sweep must exercise typed rows, including invalid states"
        );
    }
    #[test]
    fn s2_table_order_is_checked_at_every_object_position() {
        let row: Row = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/receipt_v0/base_row.json"
        ))
        .unwrap();
        let bytes = record_bytes(ReceiptRecord::Row(&row)).unwrap();
        fn visit(
            bytes: &[u8],
            value: &serde_json::Value,
            path: &mut Vec<String>,
            count: &mut usize,
        ) {
            if let serde_json::Value::Object(fields) = value {
                let refs: Vec<_> = path.iter().map(String::as_str).collect();
                let object = carried_value(bytes, &refs).unwrap();
                if fields.len() > 1 {
                    let key_end = value_end(object, 1).unwrap();
                    let first_end = value_end(object, key_end + 1).unwrap();
                    let second_start = first_end + 1;
                    let second_key_end = value_end(object, second_start).unwrap();
                    let second_end = value_end(object, second_key_end + 1).unwrap();
                    let offset = object.as_ptr() as usize - bytes.as_ptr() as usize;
                    let mut swapped = Vec::new();
                    swapped.extend_from_slice(&bytes[..offset + 1]);
                    swapped.extend_from_slice(&object[second_start..second_end]);
                    swapped.push(b',');
                    swapped.extend_from_slice(&object[1..first_end]);
                    swapped.extend_from_slice(&bytes[offset + second_end..]);
                    assert_eq!(
                        canonical_bytes(&swapped),
                        Err(RejectionCode::NonCanonicalBytes),
                        "{path:?}"
                    );
                    *count += 1;
                }
                for (key, nested) in fields {
                    path.push(key.clone());
                    visit(bytes, nested, path, count);
                    path.pop();
                }
            }
        }
        let mut count = 0;
        visit(
            &bytes,
            &serde_json::from_slice(&bytes).unwrap(),
            &mut vec![],
            &mut count,
        );
        assert!(count > 30);
    }
    /// Inert witness double for the BQ-10 item 3 typed-handoff witness (contract e7b70be3, section 16.2).
    struct S8TypedHandoffResolver;

    impl PayloadResolver for S8TypedHandoffResolver {
        fn resolve(&self, _store: super::super::PayloadStore, _digest: &str) -> super::super::PayloadResolution {
            super::super::PayloadResolution::NotConfigured
        }
    }

    #[test]
    fn s8_row_284_typed_handoff_over_all_fixtures() {
        let resolver = S8TypedHandoffResolver;
        let mut reached = 0;
        for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/receipt_v0")).unwrap() {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            let steps: [fn(&[u8], &dyn PayloadResolver) -> Result<(), RejectionCode>; 6] = [
                step_json_shape, step_canonical_bytes, step_stage_references,
                step_slot_table, step_lifecycle, step_required_fields,
            ];
            if steps.iter().all(|step| step(&bytes, &resolver).is_ok()) {
                reached += 1;
                assert!(serde_json::from_slice::<Row>(&bytes).is_ok(), "{}", path.display());
            }
        }
        assert!(reached > 0, "no fixture reached the typed handoff");
    }
}
