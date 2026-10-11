// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Human projection and instruments (contract e7b70be3, sections 15, 15.1, 16.4).
use super::availability::*;
use super::canonical::*;
use super::records::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn bounded(source: &str, limit: usize) -> (String, bool) {
    let mut end = source.len().min(limit);
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    (source[..end].to_owned(), end < source.len())
}

/// Contract e7b70be3, section 15 field 11: raw UTF-8 byte bound.
pub fn tool_name_display(proposal: &Proposal) -> BoundedToolName {
    let source = match &proposal.tool_name {
        PresentUnavailable::Present(s) => s.as_str(),
        _ => "",
    };
    let (text, truncated) = bounded(source, 64);
    BoundedToolName {
        text,
        len_bytes: source.len() as u64,
        truncated,
    }
}

/// Contract e7b70be3, section 15.1; absent expected effect has no path digest.
pub fn target_display(row: &Row) -> TargetDisplay {
    let expected = match &row.stages.proposal.record.expected_effect {
        PresentAbsentErrored::Present(e) => e,
        _ => {
            return TargetDisplay {
                form: TargetForm::UnresolvedBounded,
                text: String::new(),
                digest: None,
                len_bytes: 0,
                truncated: false,
            }
        }
    };
    let resolved = match &row.stages.attempt {
        Slot::Present { record, .. }
            if matches!(record.target.target_ref, PresentUnavailable::Present(_)) =>
        {
            // An inconsistent target pair is the validator's concern. Keep this total.
            Some(match &record.target.resolved_path {
                PresentUnavailable::Present(s) => s.as_str(),
                _ => "",
            })
        }
        _ => None,
    };
    let (form, source, limit) = match resolved {
        Some(s) => (TargetForm::Resolved, s, 256),
        None => (
            TargetForm::UnresolvedBounded,
            expected.target_path.as_str(),
            64,
        ),
    };
    let (text, truncated) = bounded(source, limit);
    TargetDisplay {
        form,
        text,
        digest: Some(proposed_path_digest(&expected.target_path)),
        len_bytes: expected.target_path.len() as u64,
        truncated,
    }
}

/// Contract e7b70be3, section 15 field 7: renderer text without a projection key.
pub fn detection_display_text(availability: DetectionAvailability) -> Option<&'static str> {
    match availability {
        DetectionAvailability::AbsentByContract => Some("not produced by this deployment"),
        DetectionAvailability::Unavailable => Some("could not be produced"),
        DetectionAvailability::Present => None,
    }
}

/// Contract e7b70be3, section 15 field 6: the closed ten-string map.
pub fn summary_line(summary: Summary) -> &'static str {
    match summary {
        Summary::BlockedBeforeExecution => "blocked before execution",
        Summary::NotAttemptedUnsupported => "not attempted: tool not supported",
        Summary::NotAttemptedNotEnforced => "not attempted: policy enforcement was not active",
        Summary::NotAttemptedNoExecutor => "not attempted: no executor was available",
        Summary::NotAttemptedSink => "not attempted: receipt could not be recorded",
        Summary::NotAttemptedRejected => "not attempted: rejected before execution",
        Summary::Incomplete => "effect verification incomplete",
        Summary::AllowedEffectVerified => "allowed; effect verified",
        Summary::AllowedEffectFailed => "allowed; effect verification failed",
        Summary::AllowedEffectUnknown => "allowed; effect verification inconclusive",
    }
}

// These are concrete closed values with no fallible user-provided serializer.
fn value<T: Serialize>(v: &T) -> Value {
    serde_json::to_value(v).expect("closed receipt value serializes")
}

fn fields(out: &mut Vec<NotPresent>, record: &Value, prefix: &str, names: &[&str]) {
    for name in names {
        let field = &record[name];
        if let Some(availability) = field.get("availability") {
            if availability != "present" {
                out.push(NotPresent {
                    field: format!("{prefix}.{name}"),
                    availability: serde_json::from_value(availability.clone())
                        .expect("typed availability"),
                    reason_code: field["error_code"].as_str().map(str::to_owned),
                });
            }
        } else if *name == "target" {
            fields(
                out,
                field,
                &format!("{prefix}.target"),
                &["target_ref", "resolved_path"],
            );
        }
    }
}

fn slot_fields<T: Serialize, R: Serialize>(
    out: &mut Vec<NotPresent>,
    slot: &Slot<T, R>,
    name: &str,
    names: &[&str],
) {
    match slot {
        Slot::Present { record, .. } => fields(out, &value(record), name, names),
        Slot::AbsentByContract(reason) => out.push(NotPresent {
            field: name.to_owned(),
            availability: Availability::AbsentByContract,
            reason_code: value(reason).as_str().map(str::to_owned),
        }),
    }
}

fn not_present(row: &Row) -> Vec<NotPresent> {
    let mut out = Vec::new();
    fields(
        &mut out,
        &value(&row.stages.proposal.record),
        "proposal",
        &[
            "tool_name",
            "arguments_digest",
            "arguments_len_bytes",
            "expected_effect",
        ],
    );
    fields(
        &mut out,
        &value(&row.stages.decision.record),
        "decision",
        &["code", "detection", "detection_binding_rule"],
    );
    slot_fields(
        &mut out,
        &row.stages.attempt,
        "attempt",
        &["target", "before_state", "content_payload"],
    );
    slot_fields(
        &mut out,
        &row.stages.observation,
        "observation",
        &["target_ref", "observed", "footprint"],
    );
    slot_fields(
        &mut out,
        &row.stages.verification,
        "verification",
        &["comparison"],
    );
    fields(
        &mut out,
        &value(&row.proposal_context.record),
        "proposal_context",
        &[
            "request_ref",
            "tool_call_id",
            "native_arguments",
            "arguments_payload",
            "detection_payload",
        ],
    );
    out
}

/// Contract e7b70be3, section 15: reads carried values, without re-running verification.
pub fn receipt_projection(row: &Row) -> Projection {
    let decision = &row.stages.decision.record;
    let detection = match &decision.detection {
        PresentUnavailableAbsent::Present(d) => ProjectionDetection {
            availability: DetectionAvailability::Present,
            outcome: Some(d.detection_outcome),
            mode: Some(d.mode),
        },
        PresentUnavailableAbsent::AbsentByContract => ProjectionDetection {
            availability: DetectionAvailability::AbsentByContract,
            outcome: None,
            mode: None,
        },
        PresentUnavailableAbsent::Unavailable(_) => ProjectionDetection {
            availability: DetectionAvailability::Unavailable,
            outcome: None,
            mode: None,
        },
    };
    let mut constraint_names: Vec<_> = decision
        .failed_constraints
        .iter()
        .map(|c| c.constraint)
        .collect();
    constraint_names.sort_by_key(|c| value(c).as_str().expect("constraint string").to_owned());
    let attempt = match &row.stages.attempt {
        Slot::Present { record, .. } => Some(record),
        _ => None,
    };
    let stage_reached = if matches!(row.stages.verification, Slot::Present { .. }) {
        StageReached::Verification
    } else if matches!(row.stages.observation, Slot::Present { .. }) {
        StageReached::Observation
    } else if attempt.is_some() {
        StageReached::Attempt
    } else {
        StageReached::Decision
    };
    let verification = match &row.stages.verification {
        Slot::Present { record, .. } => ProjectionVerification {
            availability: Availability::Present,
            outcome: Some(record.outcome),
            reason_code: record.reason_code,
            delta: match &record.comparison {
                PresentAbsent::Present(c) => Some(c.delta),
                _ => None,
            },
        },
        Slot::AbsentByContract(_) => ProjectionVerification {
            availability: Availability::AbsentByContract,
            outcome: None,
            reason_code: None,
            delta: None,
        },
    };
    let bytes = record_bytes(ReceiptRecord::Row(row)).expect("closed row serializes");
    Projection {
        schema_version: ProjectionSchema::V0,
        projection_of: record_digest(&bytes, RecordDomain::Row),
        proposal_id: row.proposal_id.clone(),
        action_id: row.action_id.clone(),
        summary: row.summary,
        line: summary_line(row.summary).to_owned(),
        detection,
        policy: ProjectionPolicy {
            decision: decision.decision,
            reason_code: decision.reason_code,
            constraint_names,
            enforcement: decision.enforcement,
            evaluated_policy: decision.evaluated_policy,
        },
        effect: ProjectionEffect {
            effect_kind: row.stages.proposal.record.effect_kind,
            stage_reached,
            attempt_outcome: attempt.map(|a| a.outcome),
            attempt_error_code: attempt.and_then(|a| a.error_code),
        },
        verification,
        tool_name: tool_name_display(&row.stages.proposal.record),
        target: target_display(row),
        not_present: not_present(row),
    }
}

/// Contract e7b70be3, section 15 and ER1: compact bytes in field-table order.
pub fn projection_bytes(projection: &Projection) -> Vec<u8> {
    serde_json::to_vec(projection).expect("closed projection serializes")
}

fn allowed_keys(path: &str) -> &'static [&'static str] {
    match path {
        "" => &[
            "schema_version",
            "projection_of",
            "proposal_id",
            "action_id",
            "summary",
            "line",
            "detection",
            "policy",
            "effect",
            "verification",
            "tool_name",
            "target",
            "not_present",
        ],
        "detection" => &["availability", "outcome", "mode"],
        "policy" => &[
            "decision",
            "reason_code",
            "constraint_names",
            "enforcement",
            "evaluated_policy",
        ],
        "effect" => &[
            "effect_kind",
            "stage_reached",
            "attempt_outcome",
            "attempt_error_code",
        ],
        "verification" => &["availability", "outcome", "reason_code", "delta"],
        "tool_name" => &["text", "len_bytes", "truncated"],
        "target" => &["form", "text", "digest", "len_bytes", "truncated"],
        "not_present[]" => &["field", "availability", "reason_code"],
        _ => &[],
    }
}

fn keys_allowed(presented: &Value, path: &str) -> bool {
    match presented {
        Value::Object(object) => object.iter().all(|(key, v)| {
            allowed_keys(path).contains(&key.as_str())
                && keys_allowed(
                    v,
                    &if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    },
                )
        }),
        Value::Array(array) => array.iter().all(|v| keys_allowed(v, &format!("{path}[]"))),
        _ => true,
    }
}

/// Contract e7b70be3, sections 15 and 16.4: inspect keys at every depth.
pub fn projection_field_not_allowed(presented: &Value) -> ProjectionInstrument {
    if keys_allowed(presented, "") {
        ProjectionInstrument::Pass
    } else {
        ProjectionInstrument::ProjectionFieldNotAllowed
    }
}

fn leaks(presented: &Value, path: &str, target: &str, tool: &str, forbidden: &[&[u8]]) -> bool {
    match presented {
        Value::String(s) => {
            if (path == "target.text" && s == target) || (path == "tool_name.text" && s == tool) {
                return false;
            }
            forbidden
                .iter()
                .any(|f| !f.is_empty() && s.as_bytes().windows(f.len()).any(|w| w == *f))
        }
        Value::Object(object) => object.iter().any(|(k, v)| {
            leaks(
                v,
                &if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                },
                target,
                tool,
                forbidden,
            )
        }),
        Value::Array(array) => array
            .iter()
            .any(|v| leaks(v, &format!("{path}[]"), target, tool, forbidden)),
        _ => false,
    }
}

/// Contract e7b70be3, sections 15.1 and 16.4: decoded string substring instrument.
/// Empty forbidden sequences are skipped. Exact recomputed display fields are exempt.
pub fn redaction_failure(
    row: &Row,
    presented: &Value,
    forbidden: &[&[u8]],
) -> ProjectionInstrument {
    if leaks(
        presented,
        "",
        &target_display(row).text,
        &tool_name_display(&row.stages.proposal.record).text,
        forbidden,
    ) {
        ProjectionInstrument::RedactionFailure
    } else {
        ProjectionInstrument::Pass
    }
}

/// Contract e7b70be3, sections 15 and 16.4: full value equality, including missing keys.
pub fn projection_mismatch(row: &Row, presented: &Value) -> ProjectionInstrument {
    if value(&receipt_projection(row)) == *presented {
        ProjectionInstrument::Pass
    } else {
        ProjectionInstrument::ProjectionMismatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utf8_prefix_counts_raw_bytes() {
        let source = format!("{}λ", "x".repeat(63));
        assert_eq!(bounded(&source, 64), ("x".repeat(63), true));
    }
    #[test]
    fn nested_foreign_key_is_rejected() {
        assert_eq!(
            projection_field_not_allowed(
                &serde_json::json!({"not_present": [{"field":"attempt", "record":{}}]})
            ),
            ProjectionInstrument::ProjectionFieldNotAllowed
        );
    }
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    /// Contract e7b70be3, section 15: `schema_version`.
    pub schema_version: ProjectionSchema,
    /// Contract e7b70be3, section 15: `projection_of`.
    pub projection_of: String,
    /// Contract e7b70be3, section 15: `proposal_id`.
    pub proposal_id: String,
    /// Contract e7b70be3, section 15: `action_id`.
    pub action_id: Option<String>,
    /// Contract e7b70be3, section 15: `summary`.
    pub summary: Summary,
    /// Contract e7b70be3, section 15: `line`.
    pub line: String,
    /// Contract e7b70be3, section 15: `detection`.
    pub detection: ProjectionDetection,
    /// Contract e7b70be3, section 15: `policy`.
    pub policy: ProjectionPolicy,
    /// Contract e7b70be3, section 15: `effect`.
    pub effect: ProjectionEffect,
    /// Contract e7b70be3, section 15: `verification`.
    pub verification: ProjectionVerification,
    /// Contract e7b70be3, section 15: `tool_name`.
    pub tool_name: BoundedToolName,
    /// Contract e7b70be3, section 15: `target`.
    pub target: TargetDisplay,
    /// Contract e7b70be3, section 15: `not_present`.
    pub not_present: Vec<NotPresent>,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionDetection {
    /// Contract e7b70be3, section 15: `availability`.
    pub availability: DetectionAvailability,
    /// Contract e7b70be3, section 15: `outcome`.
    pub outcome: Option<DetectionOutcome>,
    /// Contract e7b70be3, section 15: `mode`.
    pub mode: Option<DetectionMode>,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionPolicy {
    /// Contract e7b70be3, section 15: `decision`.
    pub decision: DecisionOutcome,
    /// Contract e7b70be3, section 15: `reason_code`.
    pub reason_code: DecisionReason,
    /// Contract e7b70be3, section 15: `constraint_names`.
    pub constraint_names: Vec<ConstraintName>,
    /// Contract e7b70be3, section 15: `enforcement`.
    pub enforcement: Enforcement,
    /// Contract e7b70be3, section 15: `evaluated_policy`.
    pub evaluated_policy: bool,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionEffect {
    /// Contract e7b70be3, section 15: `effect_kind`.
    pub effect_kind: EffectKind,
    /// Contract e7b70be3, section 15: `stage_reached`.
    pub stage_reached: StageReached,
    /// Contract e7b70be3, section 15: `attempt_outcome`.
    pub attempt_outcome: Option<AttemptOutcome>,
    /// Contract e7b70be3, section 15: `attempt_error_code`.
    pub attempt_error_code: Option<AttemptError>,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionVerification {
    /// Contract e7b70be3, section 15: `availability`.
    pub availability: Availability,
    /// Contract e7b70be3, section 15: `outcome`.
    pub outcome: Option<VerificationOutcome>,
    /// Contract e7b70be3, section 15: `reason_code`.
    pub reason_code: Option<VerificationReason>,
    /// Contract e7b70be3, section 15: `delta`.
    pub delta: Option<Delta>,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedToolName {
    /// Contract e7b70be3, section 15: `text`.
    pub text: String,
    /// Contract e7b70be3, section 15: `len_bytes`.
    pub len_bytes: u64,
    /// Contract e7b70be3, section 15: `truncated`.
    pub truncated: bool,
}

/// Contract e7b70be3, section 15 and 15.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetDisplay {
    /// Contract e7b70be3, section 15: `form`.
    pub form: TargetForm,
    /// Contract e7b70be3, section 15: `text`.
    pub text: String,
    /// Contract e7b70be3, section 15: `digest`.
    pub digest: Option<String>,
    /// Contract e7b70be3, section 15: `len_bytes`.
    pub len_bytes: u64,
    /// Contract e7b70be3, section 15: `truncated`.
    pub truncated: bool,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotPresent {
    /// Contract e7b70be3, section 15: `field`.
    pub field: String,
    /// Contract e7b70be3, section 15: `availability`.
    pub availability: Availability,
    /// Contract e7b70be3, section 15: `reason_code`.
    pub reason_code: Option<String>,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionSchema {
    /// Contract e7b70be3, section 15: `parapet-effect-receipt-projection/v0`.
    #[serde(rename = "parapet-effect-receipt-projection/v0")]
    V0,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionAvailability {
    /// Contract e7b70be3, section 15: `present`.
    #[serde(rename = "present")]
    Present,
    /// Contract e7b70be3, section 15: `absent_by_contract`.
    #[serde(rename = "absent_by_contract")]
    AbsentByContract,
    /// Contract e7b70be3, section 15: `unavailable`.
    #[serde(rename = "unavailable")]
    Unavailable,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageReached {
    /// Contract e7b70be3, section 15: `decision`.
    #[serde(rename = "decision")]
    Decision,
    /// Contract e7b70be3, section 15: `attempt`.
    #[serde(rename = "attempt")]
    Attempt,
    /// Contract e7b70be3, section 15: `observation`.
    #[serde(rename = "observation")]
    Observation,
    /// Contract e7b70be3, section 15: `verification`.
    #[serde(rename = "verification")]
    Verification,
}

/// Contract e7b70be3, section 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetForm {
    /// Contract e7b70be3, section 15: `resolved`.
    #[serde(rename = "resolved")]
    Resolved,
    /// Contract e7b70be3, section 15: `unresolved_bounded`.
    #[serde(rename = "unresolved_bounded")]
    UnresolvedBounded,
}

/// Contract e7b70be3, section 16.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionInstrument {
    /// Contract e7b70be3, section 16.4: `pass`.
    #[serde(rename = "pass")]
    Pass,
    /// Contract e7b70be3, section 16.4: `projection_field_not_allowed`.
    #[serde(rename = "projection_field_not_allowed")]
    ProjectionFieldNotAllowed,
    /// Contract e7b70be3, section 16.4: `redaction_failure`.
    #[serde(rename = "redaction_failure")]
    RedactionFailure,
    /// Contract e7b70be3, section 16.4: `projection_mismatch`.
    #[serde(rename = "projection_mismatch")]
    ProjectionMismatch,
}
