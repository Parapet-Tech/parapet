// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Content-free carrier receipt (carrier Rev1 section 9, Rev2 D3).
//!
//! Records closed outcomes, configured identity, digests, and coordinates
//! only. A rejected receipt carries no envelope claim, since unverified
//! claims are attacker-controlled strings.

use serde::Serialize;

use super::envelope::{Sha256Digest, Source};
use super::projection::{Attribution, ScanReport, ScanVerdict};
use super::treatment::{treatment_of_verified, ExemptionReason, Treatment};
use super::verify::{CarrierError, CarrierIdentity, VerificationOutcome, VerifiedCarrier};

/// Per-segment receipt row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SegmentReceipt {
    pub message_index: usize,
    pub start: usize,
    pub end: usize,
    pub source: Source,
    pub sha256: Option<Sha256Digest>,
    pub scanned: bool,
    pub exemption: Option<ExemptionReason>,
}

/// Closed basis label for projected L1 detection/effect records (Sensor C1/Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedL1Basis {
    Projected,
}

impl ProjectedL1Basis {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Projected => "projected",
        }
    }
}

/// Closed status of projected L1 on a verified carrier request (Sensor C1/Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedL1Status {
    /// L1 ran on the projected basis; per-message L1 signals are not recorded.
    EvaluatedSignalsNotRecorded,
    /// L1 was not enabled in policy for this request.
    NotConfigured,
    /// Verified request exited before projected L1 was evaluated.
    NotEvaluated,
    /// L1 failed closed with a scan error on the projected basis.
    ScanError,
}

impl ProjectedL1Status {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EvaluatedSignalsNotRecorded => "evaluated_signals_not_recorded",
            Self::NotConfigured => "not_configured",
            Self::NotEvaluated => "not_evaluated",
            Self::ScanError => "scan_error",
        }
    }
}

/// Content-free detection/effect record for projected L1 (Sensor C1/Q3).
///
/// Explicitly states the projected-L1 status under the verified
/// profile/build identity and that per-message signals are not recorded.
/// Any positive coordinate is explicitly labeled as a projected-input or
/// projected-unit index and never masquerades as `message_index`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProjectedL1DetectionRecord {
    pub basis: ProjectedL1Basis,
    pub identity: CarrierIdentity,
    pub status: ProjectedL1Status,
    pub signals_recorded: bool,
    pub inputs_scored: usize,
    pub verdict: Option<ScanVerdict>,
    pub positive_projected_input_indices: Vec<usize>,
    pub positive_projected_unit_indices: Vec<usize>,
}

impl ProjectedL1DetectionRecord {
    /// Record for a completed projected L1 [`ScanReport`].
    pub fn from_report(identity: &CarrierIdentity, report: &ScanReport) -> Self {
        Self {
            basis: ProjectedL1Basis::Projected,
            identity: identity.clone(),
            status: ProjectedL1Status::EvaluatedSignalsNotRecorded,
            signals_recorded: false,
            inputs_scored: report.inputs_scored,
            verdict: Some(report.verdict),
            positive_projected_input_indices: report.positive_projected_input_indices.clone(),
            positive_projected_unit_indices: report.positive_projected_unit_indices.clone(),
        }
    }

    /// Record when projected L1 is not enabled in policy.
    pub fn not_configured(identity: &CarrierIdentity) -> Self {
        Self {
            basis: ProjectedL1Basis::Projected,
            identity: identity.clone(),
            status: ProjectedL1Status::NotConfigured,
            signals_recorded: false,
            inputs_scored: 0,
            verdict: None,
            positive_projected_input_indices: Vec::new(),
            positive_projected_unit_indices: Vec::new(),
        }
    }

    /// Record when a verified request exits before projected L1 is evaluated.
    pub fn not_evaluated(identity: &CarrierIdentity) -> Self {
        Self {
            basis: ProjectedL1Basis::Projected,
            identity: identity.clone(),
            status: ProjectedL1Status::NotEvaluated,
            signals_recorded: false,
            inputs_scored: 0,
            verdict: None,
            positive_projected_input_indices: Vec::new(),
            positive_projected_unit_indices: Vec::new(),
        }
    }

    /// Record when projected L1 fails closed with a scan error.
    pub fn scan_error(identity: &CarrierIdentity) -> Self {
        Self {
            basis: ProjectedL1Basis::Projected,
            identity: identity.clone(),
            status: ProjectedL1Status::ScanError,
            signals_recorded: false,
            inputs_scored: 0,
            verdict: None,
            positive_projected_input_indices: Vec::new(),
            positive_projected_unit_indices: Vec::new(),
        }
    }
}

/// Consumer of content-free [`CarrierReceipt`] records emitted by projected scanning.
pub trait CarrierReceiptSink: Send + Sync {
    fn record(&self, receipt: &CarrierReceipt);
}

/// Request-level carrier receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CarrierReceipt {
    pub outcome: VerificationOutcome,
    pub segment_index: Option<usize>,
    pub identity: Option<CarrierIdentity>,
    pub request_body_sha256: Option<Sha256Digest>,
    pub source_map_sha256: Option<Sha256Digest>,
    pub segments: Vec<SegmentReceipt>,
    pub verdict: Option<ScanVerdict>,
    pub inputs_scored: usize,
    pub attributions: Vec<Attribution>,
    pub l1_detection: Option<ProjectedL1DetectionRecord>,
}

impl CarrierReceipt {
    fn segment_receipts(carrier: &VerifiedCarrier) -> Vec<SegmentReceipt> {
        carrier
            .segments
            .iter()
            .map(|s| {
                let treatment = treatment_of_verified(s);
                SegmentReceipt {
                    message_index: s.message_index,
                    start: s.start,
                    end: s.end,
                    source: s.source,
                    sha256: matches!(treatment, Treatment::Exempt(_)).then_some(s.sha256),
                    scanned: treatment.is_untrusted(),
                    exemption: match treatment {
                        Treatment::Exempt(reason) => Some(reason),
                        Treatment::Untrusted => None,
                    },
                }
            })
            .collect()
    }

    /// Receipt for a verified and scanned carrier.
    pub fn verified(carrier: &VerifiedCarrier, report: &ScanReport) -> Self {
        Self::verified_stage(
            carrier,
            Some(report.verdict),
            report.inputs_scored,
            report.positives.clone(),
            ProjectedL1DetectionRecord::from_report(&carrier.identity, report),
        )
    }

    /// Receipt for a verified carrier processed by the projected engine stage.
    pub fn verified_stage(
        carrier: &VerifiedCarrier,
        verdict: Option<ScanVerdict>,
        inputs_scored: usize,
        attributions: Vec<Attribution>,
        l1_detection: ProjectedL1DetectionRecord,
    ) -> Self {
        Self {
            outcome: VerificationOutcome::Verified,
            segment_index: None,
            identity: Some(carrier.identity.clone()),
            request_body_sha256: Some(carrier.request_body_sha256),
            source_map_sha256: Some(carrier.source_map_sha256),
            segments: Self::segment_receipts(carrier),
            verdict,
            inputs_scored,
            attributions,
            l1_detection: Some(l1_detection),
        }
    }

    /// Receipt for a rejected carrier: closed outcome and segment index only.
    pub fn rejected(error: &CarrierError) -> Self {
        Self {
            outcome: error.outcome,
            segment_index: error.segment_index,
            identity: None,
            request_body_sha256: None,
            source_map_sha256: None,
            segments: Vec::new(),
            verdict: None,
            inputs_scored: 0,
            attributions: Vec::new(),
            l1_detection: None,
        }
    }
}
