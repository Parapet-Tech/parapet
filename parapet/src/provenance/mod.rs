// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! OpenCode provenance carrier (carrier Rev1 + Rev2 delta, Sensor owner
//! input, L1 projection vectors v1).
//!
//! First increment: closed envelope/map verification, effective treatment,
//! projection geometry, block-only scoring through an injected scorer, and
//! content-free receipts. Engine wiring follows the L0 dual-view gate.

pub mod decode;
pub mod envelope;
pub mod projection;
pub mod receipt;
pub mod treatment;
pub mod verify;

pub use decode::*;
pub use envelope::*;
pub use projection::*;
pub use receipt::*;
pub use treatment::*;
pub use verify::*;

/// Whether the required OpenCode profile is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMode {
    /// No required profile: any carrier is ignored.
    Ordinary,
    /// Required OpenCode profile: a verified carrier is mandatory.
    RequiredOpenCode,
}

/// Admission result before L0.
#[derive(Debug)]
pub enum CarrierAdmission {
    /// Current whole-message scan behavior, unchanged.
    WholeMessage,
    /// Verified carrier; scan via [`project`].
    Projected(VerifiedCarrier),
    /// Fail closed: forward zero bytes upstream.
    Reject(CarrierError),
}

/// Admit a request. Outside the required profile the carrier is not parsed.
/// Under it, every failure rejects; there is no whole-message fallback.
pub fn admit(
    mode: ProfileMode,
    verifier: &CarrierVerifier,
    envelope: Option<&[u8]>,
    body: &[u8],
) -> CarrierAdmission {
    match mode {
        ProfileMode::Ordinary => CarrierAdmission::WholeMessage,
        ProfileMode::RequiredOpenCode => match verifier.verify(envelope, body) {
            Ok(verified) => CarrierAdmission::Projected(verified),
            Err(error) => CarrierAdmission::Reject(error),
        },
    }
}
