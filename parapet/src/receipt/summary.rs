// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Ordered summary function (contract e7b70be3, section 14.2).
use super::availability::*;
use super::records::*;

/// Contract e7b70be3, section 14.2: first matching line wins.
/// Slot-table inconsistencies are classified deterministically; validation is separate.
pub fn receipt_summary(row: &Row) -> Summary {
    let decision = &row.stages.decision.record;
    if decision.decision == DecisionOutcome::Block {
        return Summary::BlockedBeforeExecution;
    }
    if row.stages.proposal.record.effect_kind == EffectKind::Unsupported {
        return Summary::NotAttemptedUnsupported;
    }
    if decision.enforcement != Enforcement::Enforced {
        return Summary::NotAttemptedNotEnforced;
    }
    if matches!(
        row.stages.attempt,
        Slot::AbsentByContract(AttemptAbsentReason::NoExecutor)
    ) {
        return Summary::NotAttemptedNoExecutor;
    }
    if let Slot::Present { record, .. } = &row.stages.attempt {
        if record.outcome == AttemptOutcome::Rejected {
            if matches!(
                record.error_code,
                Some(
                    AttemptError::SinkRejected
                        | AttemptError::SinkUnavailable
                        | AttemptError::SinkRefMismatch
                )
            ) {
                return Summary::NotAttemptedSink;
            }
            return Summary::NotAttemptedRejected;
        }
    }
    match &row.stages.verification {
        Slot::AbsentByContract(_) => Summary::Incomplete,
        Slot::Present { record, .. } => match record.outcome {
            VerificationOutcome::Verified => Summary::AllowedEffectVerified,
            VerificationOutcome::Failed => Summary::AllowedEffectFailed,
            VerificationOutcome::Unknown => Summary::AllowedEffectUnknown,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_line_precedes_unsupported_and_shadow() {
        let mut row: Row = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/receipt_v0/base_row.json"
        ))
        .unwrap();
        row.stages.decision.record.decision = DecisionOutcome::Block;
        row.stages.decision.record.enforcement = Enforcement::Shadow;
        row.stages.proposal.record.effect_kind = EffectKind::Unsupported;
        assert_eq!(receipt_summary(&row), Summary::BlockedBeforeExecution);
    }
}
