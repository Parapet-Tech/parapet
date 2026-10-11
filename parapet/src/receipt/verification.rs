// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pure comparison and ordered method (contract e7b70be3, sections 12.1, 12.2).
use super::availability::*;
use super::records::*;

/// Contract e7b70be3, section 12.1: change between carried measurements.
/// Missing measurements give unknown; both absent measurements give unchanged.
pub fn receipt_delta(
    before: &PresentUnavailableErroredAbsent<BeforeState, BeforeStateError>,
    observed: &PresentUnavailableErrored<Observed, ObservedError>,
) -> Delta {
    match (before, observed) {
        (PresentUnavailableErroredAbsent::Present(b), PresentUnavailableErrored::Present(o)) => {
            match (b.exists, o.exists) {
                (false, true) => Delta::Created,
                (true, false) => Delta::Removed,
                (false, false) => Delta::Unchanged,
                (true, true) if b.digest == o.digest => Delta::Unchanged,
                (true, true) => Delta::Changed,
            }
        }
        _ => Delta::Unknown,
    }
}

/// Contract e7b70be3, section 12: method output, independent of record identity.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodResult {
    /// Contract e7b70be3, section 12 field 5.
    pub outcome: VerificationOutcome,
    /// Contract e7b70be3, section 12 field 7.
    pub reason_code: Option<VerificationReason>,
    /// Contract e7b70be3, section 12 field 6.
    pub comparison: PresentAbsent<Comparison>,
}

/// Contract e7b70be3, section 12.2 and L4; BQ-9 ruling f8d00d2b.
/// Checks the domain before all ten ordered lines. None denotes only out-of-domain:
/// expected_effect is not present, or the attempt was rejected.
pub fn verify_expected_effect(
    proposal: &Proposal,
    attempt: &Attempt,
    observation: &Observation,
) -> Option<MethodResult> {
    let expected = match &proposal.expected_effect {
        PresentAbsentErrored::Present(e) if attempt.outcome != AttemptOutcome::Rejected => e,
        _ => return None,
    };
    let observed = match &observation.observed {
        PresentUnavailableErrored::Present(o) => Some(o),
        _ => None,
    };
    let before = match &attempt.before_state {
        PresentUnavailableErroredAbsent::Present(b) => Some(b),
        _ => None,
    };
    let delta = receipt_delta(&attempt.before_state, &observation.observed);
    let comparison = match observed {
        Some(o) => PresentAbsent::Present(Comparison {
            expected_digest: expected.content_digest.clone(),
            observed_digest: o.digest.clone(),
            expected_len_bytes: expected.content_len_bytes,
            observed_len_bytes: o.len_bytes,
            within_root: o.resolved_within_root,
            before_digest: before.and_then(|b| b.digest.clone()),
            delta,
        }),
        None => PresentAbsent::AbsentByContract,
    };
    let (outcome, reason_code) = if attempt.outcome == AttemptOutcome::Ambiguous {
        (
            VerificationOutcome::Unknown,
            Some(VerificationReason::AttemptAmbiguous),
        )
    } else if let Some(o) = observed {
        if attempt.outcome == AttemptOutcome::Errored {
            (
                VerificationOutcome::Failed,
                Some(VerificationReason::AttemptErrored),
            )
        } else if !o.resolved_within_root {
            (
                VerificationOutcome::Failed,
                Some(VerificationReason::OutsideRoot),
            )
        } else if !o.exists {
            (
                VerificationOutcome::Failed,
                Some(VerificationReason::TargetMissing),
            )
        } else if o.len_bytes != Some(expected.content_len_bytes) {
            (
                VerificationOutcome::Failed,
                Some(VerificationReason::LengthMismatch),
            )
        } else if o.digest.as_ref() != Some(&expected.content_digest) {
            (
                VerificationOutcome::Failed,
                Some(VerificationReason::DigestMismatch),
            )
        } else if before.is_none() {
            (
                VerificationOutcome::Unknown,
                Some(VerificationReason::BeforeStateMissing),
            )
        } else if delta == Delta::Unchanged {
            (
                VerificationOutcome::Unknown,
                Some(VerificationReason::NoDeltaObserved),
            )
        } else {
            (VerificationOutcome::Verified, None)
        }
    } else {
        (
            VerificationOutcome::Unknown,
            Some(VerificationReason::ObservationMissing),
        )
    };
    Some(MethodResult {
        outcome,
        reason_code,
        comparison,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_before_is_never_clean() {
        let o = PresentUnavailableErrored::Present(Observed {
            exists: true,
            digest: Some("x".into()),
            len_bytes: Some(1),
            resolved_within_root: true,
        });
        assert_eq!(
            receipt_delta(
                &PresentUnavailableErroredAbsent::Unavailable(BeforeStateError::NoReadAccess),
                &o
            ),
            Delta::Unknown
        );
    }
}
