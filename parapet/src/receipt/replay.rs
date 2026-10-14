// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Contract e7b70be3, section 17: verified replay inputs and comparison.

use super::*;

/// Contract e7b70be3, sections 7 and 17: first failed payload binding check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCheck {
    /// Contract e7b70be3, section 5.4: lookup.
    Lookup,
    /// Contract e7b70be3, section 5.4: raw_digest.
    RawDigest,
    /// Contract e7b70be3, section 7: length.
    Length,
    /// Contract e7b70be3, sections 8 and 9.2: tagged_digest.
    TaggedDigest,
}

/// Contract e7b70be3, section 17: closed not-comparable diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayReason {
    /// Contract e7b70be3, section 16.2: row_rejected.
    RowRejected(RejectionCode),
    /// Contract e7b70be3, section 9.10: policy_not_evaluated.
    PolicyNotEvaluated,
    /// Contract e7b70be3, section 17: policy_digest_mismatch.
    PolicyDigestMismatch,
    /// Contract e7b70be3, section 9.1: policy_unparsable.
    PolicyUnparsable,
    /// Contract e7b70be3, section 17: arguments_not_present.
    ArgumentsNotPresent,
    /// Contract e7b70be3, section 7.10: arguments_unresolved.
    ArgumentsUnresolved,
    /// Contract e7b70be3, section 7.10: arguments_mismatch.
    ArgumentsMismatch(ReplayCheck),
    /// Contract e7b70be3, section 17: arguments_unparsable.
    ArgumentsUnparsable,
    /// Contract e7b70be3, section 17: detection_not_present.
    DetectionNotPresent,
    /// Contract e7b70be3, section 7.11: detection_unresolved.
    DetectionUnresolved,
    /// Contract e7b70be3, section 7.11: detection_mismatch.
    DetectionMismatch(ReplayCheck),
}

impl ReplayCheck {
    /// Contract e7b70be3, sections 7 and 17: exact check vocabulary.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lookup => "lookup", Self::RawDigest => "raw_digest",
            Self::Length => "length", Self::TaggedDigest => "tagged_digest",
        }
    }
}

impl ReplayReason {
    /// Contract e7b70be3, section 17: exact reason vocabulary.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RowRejected(_) => "row_rejected",
            Self::PolicyNotEvaluated => "policy_not_evaluated",
            Self::PolicyDigestMismatch => "policy_digest_mismatch",
            Self::PolicyUnparsable => "policy_unparsable",
            Self::ArgumentsNotPresent => "arguments_not_present",
            Self::ArgumentsUnresolved => "arguments_unresolved",
            Self::ArgumentsMismatch(_) => "arguments_mismatch",
            Self::ArgumentsUnparsable => "arguments_unparsable",
            Self::DetectionNotPresent => "detection_not_present",
            Self::DetectionUnresolved => "detection_unresolved",
            Self::DetectionMismatch(_) => "detection_mismatch",
        }
    }
}

/// Contract e7b70be3, section 17: three-valued replay result.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayVerdict {
    /// Contract e7b70be3, section 17: match with recomputed fields.
    Match(DecisionOutput),
    /// Contract e7b70be3, section 17: fail with recomputed fields.
    Fail(DecisionOutput),
    /// Contract e7b70be3, section 17: not_comparable without recomputation.
    NotComparable(ReplayReason),
}

impl ReplayVerdict {
    /// Contract e7b70be3, section 17: exact verdict vocabulary.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Match(_) => "match",
            Self::Fail(_) => "fail",
            Self::NotComparable(_) => "not_comparable",
        }
    }
}

/// Contract e7b70be3, section 17: exact field and sequence comparison.
pub fn compare_decision(recorded: &Decision, recomputed: DecisionOutput) -> ReplayVerdict {
    if recorded.decision == recomputed.decision
        && recorded.reason_code == recomputed.reason_code
        && recorded.failed_constraints == recomputed.failed_constraints
    {
        ReplayVerdict::Match(recomputed)
    } else {
        ReplayVerdict::Fail(recomputed)
    }
}

/// Contract e7b70be3, section 17: replayer with injected dependencies.
pub struct DecisionReplayer<'a> {
    function: &'a dyn DecisionFunction,
    resolver: &'a dyn PayloadResolver,
}

impl<'a> DecisionReplayer<'a> {
    /// Contract e7b70be3, sections 9.1 and 17: assign dependencies only.
    pub fn new(function: &'a dyn DecisionFunction, resolver: &'a dyn PayloadResolver) -> Self {
        Self { function, resolver }
    }

    /// Contract e7b70be3, section 17: validate, bind, recompute, then compare.
    pub fn replay(&self, row_bytes: &[u8], policy_bytes: &[u8]) -> ReplayVerdict {
        match self.bind_and_decide(row_bytes, policy_bytes) {
            Ok(verdict) => verdict,
            Err(reason) => ReplayVerdict::NotComparable(reason),
        }
    }

    fn bind_and_decide(&self, bytes: &[u8], policy_bytes: &[u8]) -> Result<ReplayVerdict, ReplayReason> {
        use ReplayReason::*;
        if let ValidationResult::Rejected(code) = validate_receipt(bytes, self.resolver) {
            return Err(RowRejected(code));
        }
        let row: Row = serde_json::from_slice(bytes)
            .map_err(|_| RowRejected(RejectionCode::SchemaError))?;
        let decision = &row.stages.decision.record;
        if !decision.evaluated_policy { return Err(PolicyNotEvaluated); }
        if policy_digest(policy_bytes) != decision.policy.policy_digest {
            return Err(PolicyDigestMismatch);
        }
        let policy: ResolvedActionPolicyV0 = serde_json::from_slice(policy_bytes)
            .map_err(|_| PolicyUnparsable)?;
        let proposal = &row.stages.proposal.record;
        let context = &row.proposal_context.record;
        let PresentUnavailable::Present(reference) = &context.arguments_payload else {
            return Err(ArgumentsNotPresent);
        };
        let argument_bytes = match self.resolver.resolve(reference.store, &reference.digest) {
            PayloadResolution::Found(bytes) => bytes,
            PayloadResolution::NotConfigured => return Err(ArgumentsUnresolved),
            PayloadResolution::NotFound => return Err(ArgumentsMismatch(ReplayCheck::Lookup)),
        };
        if payload_digest(&argument_bytes) != reference.digest {
            return Err(ArgumentsMismatch(ReplayCheck::RawDigest));
        }
        if argument_bytes.len() as u64 != reference.len_bytes
            || proposal.arguments_len_bytes != PresentErrored::Present(argument_bytes.len() as u64)
        {
            return Err(ArgumentsMismatch(ReplayCheck::Length));
        }
        if proposal.arguments_digest != PresentErrored::Present(arguments_digest(&argument_bytes)) {
            return Err(ArgumentsMismatch(ReplayCheck::TaggedDigest));
        }
        let arguments: serde_json::Value = serde_json::from_slice(&argument_bytes)
            .map_err(|_| ArgumentsUnparsable)?;
        if let PresentUnavailableAbsent::Present(detection) = &decision.detection {
            let PresentUnavailableAbsent::Present(reference) = &context.detection_payload else {
                return Err(DetectionNotPresent);
            };
            let detection_bytes = match self.resolver.resolve(reference.store, &reference.digest) {
                PayloadResolution::Found(bytes) => bytes,
                PayloadResolution::NotConfigured => return Err(DetectionUnresolved),
                PayloadResolution::NotFound => return Err(DetectionMismatch(ReplayCheck::Lookup)),
            };
            if payload_digest(&detection_bytes) != reference.digest {
                return Err(DetectionMismatch(ReplayCheck::RawDigest));
            }
            if detection_bytes.len() as u64 != reference.len_bytes
                || detection_bytes.len() as u64 != detection.record_len_bytes
            {
                return Err(DetectionMismatch(ReplayCheck::Length));
            }
            if detection_record_digest(&detection_bytes) != detection.record_digest {
                return Err(DetectionMismatch(ReplayCheck::TaggedDigest));
            }
        }
        // Validation makes this defensive arm unreachable for evaluated rows.
        let PresentUnavailable::Present(tool) = &proposal.tool_name else {
            return Err(PolicyNotEvaluated);
        };
        let recomputed = self.function.decide(&policy, tool, &arguments);
        Ok(compare_decision(decision, recomputed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded() -> Decision {
        serde_json::from_slice(include_bytes!("../../tests/fixtures/receipt_v0/base_decision.json")).unwrap()
    }
    fn output(record: &Decision) -> DecisionOutput {
        DecisionOutput { decision: record.decision, reason_code: record.reason_code,
            failed_constraints: record.failed_constraints.clone() }
    }
    #[test]
    fn comparison_each_field_and_sequence() {
        let mut record = recorded();
        let control = output(&record);
        assert_eq!(compare_decision(&record, control.clone()), ReplayVerdict::Match(control));
        let mut other = output(&record); other.decision = DecisionOutcome::Block;
        assert_eq!(compare_decision(&record, other.clone()), ReplayVerdict::Fail(other));
        let mut other = output(&record); other.reason_code = DecisionReason::Multiple;
        assert_eq!(compare_decision(&record, other.clone()), ReplayVerdict::Fail(other));
        let a = FailedConstraint { argument: Some("a".into()), constraint: ConstraintName::Min };
        let b = FailedConstraint { argument: Some("b".into()), constraint: ConstraintName::Max };
        let mut other = output(&record); other.failed_constraints.push(a.clone());
        assert_eq!(compare_decision(&record, other.clone()), ReplayVerdict::Fail(other));
        record.failed_constraints = vec![a.clone(), b.clone()];
        let mut other = output(&record); other.failed_constraints = vec![b, a.clone()];
        assert_eq!(compare_decision(&record, other.clone()), ReplayVerdict::Fail(other));
        let mut other = output(&record); other.failed_constraints.push(a);
        assert_eq!(compare_decision(&record, other.clone()), ReplayVerdict::Fail(other));
    }
}
