// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pure byte and digest primitives (contract e7b70be3, sections 5, 8, 13).
use super::records::*;
use sha2::{Digest, Sha256};

/// Contract e7b70be3, section 13. A borrowed, closed canonical record family.
#[derive(Debug, Clone, Copy)]
pub enum ReceiptRecord<'a> {
    /// Contract e7b70be3, section 6.
    Proposal(&'a Proposal),
    /// Contract e7b70be3, section 7.
    ProposalContext(&'a ProposalContext),
    /// Contract e7b70be3, section 9.
    Decision(&'a Decision),
    /// Contract e7b70be3, section 10.
    Attempt(&'a Attempt),
    /// Contract e7b70be3, section 11.
    Observation(&'a Observation),
    /// Contract e7b70be3, section 12.
    Verification(&'a Verification),
    /// Contract e7b70be3, section 14.1.
    Row(&'a Row),
}

/// Contract e7b70be3, ER1–ER3 in section 13. The single table-order serializer.
pub fn record_bytes(record: ReceiptRecord<'_>) -> Result<Vec<u8>, serde_json::Error> {
    match record {
        ReceiptRecord::Proposal(v) => serde_json::to_vec(v),
        ReceiptRecord::ProposalContext(v) => serde_json::to_vec(v),
        ReceiptRecord::Decision(v) => serde_json::to_vec(v),
        ReceiptRecord::Attempt(v) => serde_json::to_vec(v),
        ReceiptRecord::Observation(v) => serde_json::to_vec(v),
        ReceiptRecord::Verification(v) => serde_json::to_vec(v),
        ReceiptRecord::Row(v) => serde_json::to_vec(v),
    }
}

/// Contract e7b70be3, ER2 in section 13. Stage and row digest domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordDomain {
    /// Contract e7b70be3, section 13: stage record.
    Stage,
    /// Contract e7b70be3, section 13: row envelope.
    Row,
}

/// Contract e7b70be3, ER2–ER3 in section 13. Hash carried bytes without serialization.
pub fn record_digest(bytes: &[u8], domain: RecordDomain) -> String {
    tagged_digest(
        match domain {
            RecordDomain::Stage => b"parapet-effect-receipt-stage-digest/v0",
            RecordDomain::Row => b"parapet-effect-receipt-digest/v0",
        },
        bytes,
    )
}

fn tagged_digest(tag: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(tag);
    hash.update([0x0a]);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

/// Contract e7b70be3, C1 in section 8. Compact parsed values with sorted object keys.
/// Wire-form selection and provider normalization belong to the caller.
pub fn canonical_argument_bytes(value: &serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(value)
}

/// Contract e7b70be3, C2 in section 8. Tagged digest of canonical argument bytes.
pub fn arguments_digest(bytes: &[u8]) -> String {
    tagged_digest(b"parapet-effect-proposal-arguments/v0", bytes)
}

/// Contract e7b70be3, section 5.4. Raw, untagged payload digest.
pub fn payload_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Contract e7b70be3, section 5.5. Executable bytes supplied by the caller.
pub fn code_digest(bytes: &[u8]) -> String {
    tagged_digest(b"parapet-engine-code/v0", bytes)
}

/// Contract e7b70be3, section 5.6. Root nonce and canonical absolute path from the caller.
pub fn root_digest(nonce: &[u8; 32], canonical_absolute_path: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"parapet-effect-root/v0\n");
    hash.update(nonce);
    hash.update([0x0a]);
    hash.update(canonical_absolute_path.as_bytes());
    format!("{:x}", hash.finalize())
}

/// Contract e7b70be3, section 5.7. Exact proposed path bytes, without normalization.
pub fn proposed_path_digest(path: &str) -> String {
    tagged_digest(b"parapet-effect-proposed-path/v0", path.as_bytes())
}

/// Contract e7b70be3, section 5.7. Raw root digest and resolved root-relative path bytes.
/// Resolution is the controller's responsibility; this function performs no I/O.
pub fn target_ref(root_digest: &[u8; 32], resolved_path: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"parapet-effect-target/v0\n");
    hash.update(root_digest);
    hash.update([0x00]);
    hash.update(resolved_path.as_bytes());
    format!("{:x}", hash.finalize())
}

/// Contract e7b70be3, section 7.11. Local DR2 recomputation over carried record bytes.
pub fn detection_record_digest(bytes: &[u8]) -> String {
    tagged_digest(b"parapet-l1-detection-record-digest/v0", bytes)
}

/// Contract e7b70be3, section 9.1. Policy serialization failure.
#[derive(Debug, thiserror::Error)]
pub enum PolicyBytesError {
    /// Contract e7b70be3, section 9.1: policy numbers must be finite JSON numbers.
    #[error("non-finite resolved policy bound")]
    NonFinite,
    /// Contract e7b70be3, section 9.1: serialization error.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

/// Contract e7b70be3, section 9.1. C1 policy bytes with every nullable member present.
pub fn canonical_policy_bytes(
    policy: &ResolvedActionPolicyV0,
) -> Result<Vec<u8>, PolicyBytesError> {
    for tool in policy.tools.values() {
        for constraint in tool.constraints.values() {
            if constraint.min.is_some_and(|n| !n.is_finite())
                || constraint.max.is_some_and(|n| !n.is_finite())
            {
                return Err(PolicyBytesError::NonFinite);
            }
        }
    }
    // Unlike stage records, policy objects use C1 sorted key order at every level.
    Ok(canonical_argument_bytes(&serde_json::to_value(policy)?)?)
}

/// Contract e7b70be3, section 9.1. Hash exact canonical policy bytes, without reserialization.
pub fn policy_digest(bytes: &[u8]) -> String {
    tagged_digest(b"parapet-resolved-policy/v0", bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn argument_utf8_and_sorted_keys() {
        let parsed: serde_json::Value =
            serde_json::from_str(r#"{"é":"λ","z":"\n","a":true}"#).unwrap();
        assert_eq!(
            canonical_argument_bytes(&parsed).unwrap(),
            "{\"a\":true,\"z\":\"\\n\",\"é\":\"λ\"}".as_bytes()
        );
    }
}
