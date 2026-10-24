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
}

impl CarrierReceipt {
    /// Receipt for a verified and scanned carrier.
    pub fn verified(carrier: &VerifiedCarrier, report: &ScanReport) -> Self {
        let segments = carrier
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
            .collect();
        Self {
            outcome: VerificationOutcome::Verified,
            segment_index: None,
            identity: Some(carrier.identity.clone()),
            request_body_sha256: Some(carrier.request_body_sha256),
            source_map_sha256: Some(carrier.source_map_sha256),
            segments,
            verdict: Some(report.verdict),
            inputs_scored: report.inputs_scored,
            attributions: report.positives.clone(),
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
        }
    }
}
