// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Effective treatment of verified segments (carrier Rev1 section 6, Rev2 D1).
//!
//! Only verified `vendor_static` is exempt in v1, and only because the
//! verifier already matched its digest against the pinned build manifest.
//! Every other source is untrusted; v1 defines no configured-agent pin.

use serde::Serialize;

use super::envelope::{Segment, Source};

/// Closed v1 exemption reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExemptionReason {
    VendorStaticAttested,
}

/// Effective scan treatment. Coalescing compares treatments, never labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Treatment {
    Untrusted,
    Exempt(ExemptionReason),
}

impl Treatment {
    pub fn is_untrusted(self) -> bool {
        matches!(self, Treatment::Untrusted)
    }
}

/// Treatment of a segment taken from a [`super::VerifiedCarrier`].
pub fn treatment_of_verified(segment: &Segment) -> Treatment {
    match segment.source {
        Source::VendorStatic => Treatment::Exempt(ExemptionReason::VendorStaticAttested),
        Source::ConfiguredAgent
        | Source::Repository
        | Source::User
        | Source::ToolResult
        | Source::Attachment
        | Source::Unknown => Treatment::Untrusted,
    }
}
