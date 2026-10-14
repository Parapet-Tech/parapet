// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Types and bytes for effect receipts (contract e7b70be3, sections 3, 5–14).

/// Contract e7b70be3, section 3: availability field and slot shapes.
pub mod availability;
/// Contract e7b70be3, sections 5, 8 and 13: bytes and digest primitives.
pub mod canonical;
/// Contract e7b70be3, sections 5–14: closed stage, row and policy values.
pub mod records;
/// Contract e7b70be3, section 14.2: ordered total summary.
pub mod summary;
/// Contract e7b70be3, section 12: delta and ordered verification method.
pub mod verification;
/// Contract e7b70be3, section 15: human projection and instruments.
pub mod projection;

pub use availability::*;
pub use canonical::*;
pub use records::*;
pub use summary::*;
pub use verification::*;
pub use projection::*;

/// Contract e7b70be3, section 16.2: ordered byte validation.
pub mod validator;
pub use validator::*;

/// Contract e7b70be3, sections 5.4 and 16.2: injected protected-payload lookup.
pub trait PayloadResolver {
    /// Contract e7b70be3, sections 7.10 and 7.11: resolve a store and raw digest.
    fn resolve(&self, store: PayloadStore, digest: &str) -> PayloadResolution;
}

/// Contract e7b70be3, sections 7.10 and 7.11: protected-payload lookup outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadResolution {
    /// Contract e7b70be3, section 5.4: carried payload bytes.
    Found(Vec<u8>),
    /// Contract e7b70be3, sections 7.10 and 7.11: lookup has no matching bytes.
    NotFound,
    /// Contract e7b70be3, sections 7.10 and 7.11: lookup is not configured.
    NotConfigured,
}

/// Contract e7b70be3, section 9.1: injected receipt decision function.
pub trait DecisionFunction {
    /// Contract e7b70be3, sections 9.1 and 17: recompute from verified inputs.
    fn decide(&self, policy: &ResolvedActionPolicyV0, tool_name: &str,
              arguments: &serde_json::Value) -> DecisionOutput;
}

/// Contract e7b70be3, sections 9 and 17: recomputed comparison fields.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutput {
    /// Contract e7b70be3, section 9: decision.
    pub decision: DecisionOutcome,
    /// Contract e7b70be3, section 9: reason.
    pub reason_code: DecisionReason,
    /// Contract e7b70be3, section 9: ordered failures.
    pub failed_constraints: Vec<FailedConstraint>,
}

/// Contract e7b70be3, section 17: input binding and replay comparison.
pub mod replay;
pub use replay::*;
