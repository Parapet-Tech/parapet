// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pre-L0 request capture and the ratified L1 detection binding
//! (Routing contract 50747a52, section 3 step 1; draft 12 sections 5.1 and 9.2).

use crate::config::L1Config;
use crate::layers::l1_detection::{self, DetectionOutcome as L1Outcome, RecordMode};
use crate::message::{Message, Role, TrustLevel};
use crate::receipt::{
    Detection, DetectionBindingRule, DetectionError, DetectionMode, DetectionOutcome,
    DetectionSchema, PresentAbsent, PresentUnavailable, PresentUnavailableAbsent, RequestIdSource,
    RequestRef, RequestRefError,
};

/// Draft 12 section 9.2: the bound detection for every row of one response.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionBindingV0 {
    /// `detection_binding_rule`.
    pub rule: PresentAbsent<DetectionBindingRule>,
    /// `detection`.
    pub detection: PresentUnavailableAbsent<Detection, DetectionError>,
    /// Exact ratified record bytes, retained for replay; never logged.
    pub record_bytes: Option<Vec<u8>>,
}

impl DetectionBindingV0 {
    /// No qualifying message, or L1 not configured.
    pub fn absent() -> Self {
        Self {
            rule: PresentAbsent::AbsentByContract,
            detection: PresentUnavailableAbsent::AbsentByContract,
            record_bytes: None,
        }
    }
}

/// Everything captured from one request before L0 mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestCaptureV0 {
    /// Draft 12 section 5.1.
    pub request_ref: PresentUnavailable<RequestRef, RequestRefError>,
    /// Draft 12 section 7 field 6: model as presented, at most 256 bytes.
    pub model: String,
    /// Draft 12 section 9.2.
    pub detection: DetectionBindingV0,
}

/// Rule `last_untrusted_user_message_before_l0/v0`: in request order, the last
/// message with `Role::User` and `TrustLevel::Untrusted`.
pub fn select_bound_message(messages: &[Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User && m.trust == TrustLevel::Untrusted)
        .map(|m| m.content.as_str())
}

/// Draft 12 section 9.2: call the ratified `scan_text` exactly once when
/// configured and bind the record bytes, never treating them as a decision.
pub fn bind_detection(messages: &[Message], l1: Option<&L1Config>) -> DetectionBindingV0 {
    let (Some(text), Some(cfg)) = (select_bound_message(messages), l1) else {
        return DetectionBindingV0::absent();
    };
    let unavailable = DetectionBindingV0 {
        rule: PresentAbsent::Present(DetectionBindingRule::V0),
        detection: PresentUnavailableAbsent::Unavailable(DetectionError::ScanError),
        record_bytes: None,
    };
    let Ok(record) = l1_detection::scan_text(text, cfg, true) else {
        return unavailable;
    };
    let (Ok(bytes), Ok(digest)) = (
        l1_detection::record_bytes(&record),
        l1_detection::record_digest(&record),
    ) else {
        return unavailable;
    };
    DetectionBindingV0 {
        rule: PresentAbsent::Present(DetectionBindingRule::V0),
        detection: PresentUnavailableAbsent::Present(Detection {
            record_digest: digest,
            record_len_bytes: bytes.len() as u64,
            schema_version: DetectionSchema::V0,
            detection_outcome: match record.detection_outcome {
                L1Outcome::ThresholdBreached => DetectionOutcome::ThresholdBreached,
                L1Outcome::ThresholdNotBreached => DetectionOutcome::ThresholdNotBreached,
            },
            mode: match record.mode {
                RecordMode::Shadow => DetectionMode::Shadow,
                RecordMode::Block => DetectionMode::Block,
            },
            input_digest: record.input_digest.clone(),
        }),
        record_bytes: Some(bytes),
    }
}

/// Draft 12 section 5.1: the client's id when carriable, else `unavailable`;
/// the runtime's generated id is used only when the client sent none.
pub fn request_ref_from_header(
    header: Option<&[u8]>,
    generated: &str,
) -> PresentUnavailable<RequestRef, RequestRefError> {
    let Some(raw) = header else {
        return PresentUnavailable::Present(RequestRef {
            request_id: generated.to_owned(),
            request_id_source: RequestIdSource::Generated,
        });
    };
    let Ok(text) = std::str::from_utf8(raw) else {
        return PresentUnavailable::Unavailable(RequestRefError::RequestIdInvalid);
    };
    let text = text.trim();
    if text.is_empty() {
        return PresentUnavailable::Unavailable(RequestRefError::RequestIdInvalid);
    }
    if text.len() > 256 {
        return PresentUnavailable::Unavailable(RequestRefError::RequestIdOversize);
    }
    PresentUnavailable::Present(RequestRef {
        request_id: text.to_owned(),
        request_id_source: RequestIdSource::Supplied,
    })
}
