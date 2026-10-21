// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Closed record values and table-order fields (contract e7b70be3, sections 5–14).

use super::availability::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptSchema {
    /// Contract e7b70be3, section 6: `parapet-effect-receipt/v0`.
    #[serde(rename = "parapet-effect-receipt/v0")]
    V0,
}

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStage {
    /// Contract e7b70be3, section 6: `proposal`.
    #[serde(rename = "proposal")]
    Proposal,
}

/// Contract e7b70be3, section 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextStage {
    /// Contract e7b70be3, section 7: `proposal_context`.
    #[serde(rename = "proposal_context")]
    ProposalContext,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStage {
    /// Contract e7b70be3, section 9: `decision`.
    #[serde(rename = "decision")]
    Decision,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptStage {
    /// Contract e7b70be3, section 10: `attempt`.
    #[serde(rename = "attempt")]
    Attempt,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationStage {
    /// Contract e7b70be3, section 11: `observation`.
    #[serde(rename = "observation")]
    Observation,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStage {
    /// Contract e7b70be3, section 12: `verification`.
    #[serde(rename = "verification")]
    Verification,
}

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    /// Contract e7b70be3, section 6: `openai`.
    #[serde(rename = "openai")]
    Openai,
    /// Contract e7b70be3, section 6: `anthropic`.
    #[serde(rename = "anthropic")]
    Anthropic,
}

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParseStatus {
    /// Contract e7b70be3, section 6: `parsed`.
    #[serde(rename = "parsed")]
    Parsed,
    /// Contract e7b70be3, section 6: `errored`.
    #[serde(rename = "errored")]
    Errored,
}

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectKind {
    /// Contract e7b70be3, section 6: `write_file`.
    #[serde(rename = "write_file")]
    WriteFile,
    /// Contract e7b70be3, section 6: `unsupported`.
    #[serde(rename = "unsupported")]
    Unsupported,
}

/// Contract e7b70be3, section 5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestIdSource {
    /// Contract e7b70be3, section 5.1: `generated`.
    #[serde(rename = "generated")]
    Generated,
    /// Contract e7b70be3, section 5.1: `supplied`.
    #[serde(rename = "supplied")]
    Supplied,
}

/// Contract e7b70be3, section 5.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadStore {
    /// Contract e7b70be3, section 5.4: `receipt-evidence`.
    #[serde(rename = "receipt-evidence")]
    ReceiptEvidence,
    /// Contract e7b70be3, section 5.4: `root-evidence`.
    #[serde(rename = "root-evidence")]
    RootEvidence,
}

/// Contract e7b70be3, section 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentsWireForm {
    /// Contract e7b70be3, section 7: `string`.
    #[serde(rename = "string")]
    String,
    /// Contract e7b70be3, section 7: `object`.
    #[serde(rename = "object")]
    Object,
    /// Contract e7b70be3, section 7: `null_value`.
    #[serde(rename = "null_value")]
    NullValue,
    /// Contract e7b70be3, section 7: `absent`.
    #[serde(rename = "absent")]
    Absent,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Enforcement {
    /// Contract e7b70be3, section 9: `enforced`.
    #[serde(rename = "enforced")]
    Enforced,
    /// Contract e7b70be3, section 9: `shadow`.
    #[serde(rename = "shadow")]
    Shadow,
    /// Contract e7b70be3, section 9: `not_evaluated`.
    #[serde(rename = "not_evaluated")]
    NotEvaluated,
    /// Contract e7b70be3, section 9: `suppressed_by_response`.
    #[serde(rename = "suppressed_by_response")]
    SuppressedByResponse,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionOutcome {
    /// Contract e7b70be3, section 9: `allow`.
    #[serde(rename = "allow")]
    Allow,
    /// Contract e7b70be3, section 9: `block`.
    #[serde(rename = "block")]
    Block,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionReason {
    /// Contract e7b70be3, section 9: `allowed`.
    #[serde(rename = "allowed")]
    Allowed,
    /// Contract e7b70be3, section 9: `tool_not_allowed`.
    #[serde(rename = "tool_not_allowed")]
    ToolNotAllowed,
    /// Contract e7b70be3, section 9: `argument_missing`.
    #[serde(rename = "argument_missing")]
    ArgumentMissing,
    /// Contract e7b70be3, section 9: `type_check`.
    #[serde(rename = "type_check")]
    TypeCheck,
    /// Contract e7b70be3, section 9: `starts_with`.
    #[serde(rename = "starts_with")]
    StartsWith,
    /// Contract e7b70be3, section 9: `not_contains`.
    #[serde(rename = "not_contains")]
    NotContains,
    /// Contract e7b70be3, section 9: `matches`.
    #[serde(rename = "matches")]
    Matches,
    /// Contract e7b70be3, section 9: `one_of`.
    #[serde(rename = "one_of")]
    OneOf,
    /// Contract e7b70be3, section 9: `max_length`.
    #[serde(rename = "max_length")]
    MaxLength,
    /// Contract e7b70be3, section 9: `min`.
    #[serde(rename = "min")]
    Min,
    /// Contract e7b70be3, section 9: `max`.
    #[serde(rename = "max")]
    Max,
    /// Contract e7b70be3, section 9: `url_host`.
    #[serde(rename = "url_host")]
    UrlHost,
    /// Contract e7b70be3, section 9: `multiple`.
    #[serde(rename = "multiple")]
    Multiple,
    /// Contract e7b70be3, section 9: `proposal_unparsable`.
    #[serde(rename = "proposal_unparsable")]
    ProposalUnparsable,
    /// Contract e7b70be3, section 9: `proposal_shape_invalid`.
    #[serde(rename = "proposal_shape_invalid")]
    ProposalShapeInvalid,
    /// Contract e7b70be3, section 9: `policy_layer_off`.
    #[serde(rename = "policy_layer_off")]
    PolicyLayerOff,
    /// Contract e7b70be3, section 9: `detection_gate`.
    #[serde(rename = "detection_gate")]
    DetectionGate,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintName {
    /// Contract e7b70be3, section 9: `tool_allowed`.
    #[serde(rename = "tool_allowed")]
    ToolAllowed,
    /// Contract e7b70be3, section 9: `argument_present`.
    #[serde(rename = "argument_present")]
    ArgumentPresent,
    /// Contract e7b70be3, section 9: `type_check`.
    #[serde(rename = "type_check")]
    TypeCheck,
    /// Contract e7b70be3, section 9: `starts_with`.
    #[serde(rename = "starts_with")]
    StartsWith,
    /// Contract e7b70be3, section 9: `not_contains`.
    #[serde(rename = "not_contains")]
    NotContains,
    /// Contract e7b70be3, section 9: `matches`.
    #[serde(rename = "matches")]
    Matches,
    /// Contract e7b70be3, section 9: `one_of`.
    #[serde(rename = "one_of")]
    OneOf,
    /// Contract e7b70be3, section 9: `max_length`.
    #[serde(rename = "max_length")]
    MaxLength,
    /// Contract e7b70be3, section 9: `min`.
    #[serde(rename = "min")]
    Min,
    /// Contract e7b70be3, section 9: `max`.
    #[serde(rename = "max")]
    Max,
    /// Contract e7b70be3, section 9: `url_host`.
    #[serde(rename = "url_host")]
    UrlHost,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputConsumed {
    /// Contract e7b70be3, section 9: `policy`.
    #[serde(rename = "policy")]
    Policy,
    /// Contract e7b70be3, section 9: `proposal`.
    #[serde(rename = "proposal")]
    Proposal,
    /// Contract e7b70be3, section 9: `detection`.
    #[serde(rename = "detection")]
    Detection,
}

/// Contract e7b70be3, section 9.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionSchema {
    /// Contract e7b70be3, section 9.2: `l1-detection-record/v0`.
    #[serde(rename = "l1-detection-record/v0")]
    V0,
}

/// Contract e7b70be3, section 9.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionOutcome {
    /// Contract e7b70be3, section 9.2: `threshold_breached`.
    #[serde(rename = "threshold_breached")]
    ThresholdBreached,
    /// Contract e7b70be3, section 9.2: `threshold_not_breached`.
    #[serde(rename = "threshold_not_breached")]
    ThresholdNotBreached,
}

/// Contract e7b70be3, section 9.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMode {
    /// Contract e7b70be3, section 9.2: `shadow`.
    #[serde(rename = "shadow")]
    Shadow,
    /// Contract e7b70be3, section 9.2: `block`.
    #[serde(rename = "block")]
    Block,
}

/// Contract e7b70be3, section 9.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionBindingRule {
    /// Contract e7b70be3, section 9.2: `last_untrusted_user_message_before_l0/v0`.
    #[serde(rename = "last_untrusted_user_message_before_l0/v0")]
    V0,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorName {
    /// Contract e7b70be3, section 10: `parapet-file-controller`.
    #[serde(rename = "parapet-file-controller")]
    FileController,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// Contract e7b70be3, section 10: `atomic_write_no_follow/v0`.
    #[serde(rename = "atomic_write_no_follow/v0")]
    AtomicWriteNoFollowV0,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SinkStatus {
    /// Contract e7b70be3, section 10: `accepted`.
    #[serde(rename = "accepted")]
    Accepted,
    /// Contract e7b70be3, section 10: `rejected`.
    #[serde(rename = "rejected")]
    Rejected,
    /// Contract e7b70be3, section 10: `unavailable`.
    #[serde(rename = "unavailable")]
    Unavailable,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptOutcome {
    /// Contract e7b70be3, section 10: `completed`.
    #[serde(rename = "completed")]
    Completed,
    /// Contract e7b70be3, section 10: `errored`.
    #[serde(rename = "errored")]
    Errored,
    /// Contract e7b70be3, section 10: `ambiguous`.
    #[serde(rename = "ambiguous")]
    Ambiguous,
    /// Contract e7b70be3, section 10: `rejected`.
    #[serde(rename = "rejected")]
    Rejected,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObserverSource {
    /// Contract e7b70be3, section 11: `filesystem_read`.
    #[serde(rename = "filesystem_read")]
    FilesystemRead,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethod {
    /// Contract e7b70be3, section 12: `expected-vs-observed/v0`.
    #[serde(rename = "expected-vs-observed/v0")]
    ExpectedVsObservedV0,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationOutcome {
    /// Contract e7b70be3, section 12: `verified`.
    #[serde(rename = "verified")]
    Verified,
    /// Contract e7b70be3, section 12: `failed`.
    #[serde(rename = "failed")]
    Failed,
    /// Contract e7b70be3, section 12: `unknown`.
    #[serde(rename = "unknown")]
    Unknown,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Delta {
    /// Contract e7b70be3, section 12: `created`.
    #[serde(rename = "created")]
    Created,
    /// Contract e7b70be3, section 12: `changed`.
    #[serde(rename = "changed")]
    Changed,
    /// Contract e7b70be3, section 12: `unchanged`.
    #[serde(rename = "unchanged")]
    Unchanged,
    /// Contract e7b70be3, section 12: `removed`.
    #[serde(rename = "removed")]
    Removed,
    /// Contract e7b70be3, section 12: `unknown`.
    #[serde(rename = "unknown")]
    Unknown,
}

/// Contract e7b70be3, section 14.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Summary {
    /// Contract e7b70be3, section 14.2: `blocked_before_execution`.
    #[serde(rename = "blocked_before_execution")]
    BlockedBeforeExecution,
    /// Contract e7b70be3, section 14.2: `not_attempted_unsupported`.
    #[serde(rename = "not_attempted_unsupported")]
    NotAttemptedUnsupported,
    /// Contract e7b70be3, section 14.2: `not_attempted_not_enforced`.
    #[serde(rename = "not_attempted_not_enforced")]
    NotAttemptedNotEnforced,
    /// Contract e7b70be3, section 14.2: `not_attempted_no_executor`.
    #[serde(rename = "not_attempted_no_executor")]
    NotAttemptedNoExecutor,
    /// Contract e7b70be3, section 14.2: `not_attempted_sink`.
    #[serde(rename = "not_attempted_sink")]
    NotAttemptedSink,
    /// Contract e7b70be3, section 14.2: `not_attempted_rejected`.
    #[serde(rename = "not_attempted_rejected")]
    NotAttemptedRejected,
    /// Contract e7b70be3, section 14.2: `incomplete`.
    #[serde(rename = "incomplete")]
    Incomplete,
    /// Contract e7b70be3, section 14.2: `allowed_effect_verified`.
    #[serde(rename = "allowed_effect_verified")]
    AllowedEffectVerified,
    /// Contract e7b70be3, section 14.2: `allowed_effect_failed`.
    #[serde(rename = "allowed_effect_failed")]
    AllowedEffectFailed,
    /// Contract e7b70be3, section 14.2: `allowed_effect_unknown`.
    #[serde(rename = "allowed_effect_unknown")]
    AllowedEffectUnknown,
}

/// Contract e7b70be3, section 13.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatPolicy {
    /// Contract e7b70be3, section 13: `bit-exact`.
    #[serde(rename = "bit-exact")]
    BitExact,
}

/// Contract e7b70be3, section 18.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Crosswalk {
    /// Contract e7b70be3, section 18: `parapet-effect-receipt-crosswalk/v0`.
    #[serde(rename = "parapet-effect-receipt-crosswalk/v0")]
    V0,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicySchema {
    /// Contract e7b70be3, section 9.1: `parapet-resolved-policy/v0`.
    #[serde(rename = "parapet-resolved-policy/v0")]
    V0,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockAction {
    /// Contract e7b70be3, section 9.1: `error`.
    #[serde(rename = "error")]
    Error,
    /// Contract e7b70be3, section 9.1: `rewrite`.
    #[serde(rename = "rewrite")]
    Rewrite,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeCheck {
    /// Contract e7b70be3, section 9.1: `string`.
    #[serde(rename = "string")]
    String,
    /// Contract e7b70be3, section 9.1: `number`.
    #[serde(rename = "number")]
    Number,
    /// Contract e7b70be3, section 9.1: `boolean`.
    #[serde(rename = "boolean")]
    Boolean,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolNameError {
    /// Contract e7b70be3, section 16.3: `missing_name`.
    #[serde(rename = "missing_name")]
    MissingName,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParseError {
    /// Contract e7b70be3, section 16.3: `parse_error`.
    #[serde(rename = "parse_error")]
    ParseError,
    /// Contract e7b70be3, section 16.3: `missing_name`.
    #[serde(rename = "missing_name")]
    MissingName,
    /// Contract e7b70be3, section 16.3: `missing_id`.
    #[serde(rename = "missing_id")]
    MissingId,
    /// Contract e7b70be3, section 16.3: `not_an_object`.
    #[serde(rename = "not_an_object")]
    NotAnObject,
    /// Contract e7b70be3, section 16.3: `buffer_overflow`.
    #[serde(rename = "buffer_overflow")]
    BufferOverflow,
    /// Contract e7b70be3, section 16.3: `delta_timeout`.
    #[serde(rename = "delta_timeout")]
    DeltaTimeout,
    /// Contract e7b70be3, section 16.3: `stream_aborted`.
    #[serde(rename = "stream_aborted")]
    StreamAborted,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentsError {
    /// Contract e7b70be3, section 16.3: `parse_error`.
    #[serde(rename = "parse_error")]
    ParseError,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedEffectError {
    /// Contract e7b70be3, section 16.3: `not_an_object`.
    #[serde(rename = "not_an_object")]
    NotAnObject,
    /// Contract e7b70be3, section 16.3: `missing_path`.
    #[serde(rename = "missing_path")]
    MissingPath,
    /// Contract e7b70be3, section 16.3: `missing_content`.
    #[serde(rename = "missing_content")]
    MissingContent,
    /// Contract e7b70be3, section 16.3: `wrong_type`.
    #[serde(rename = "wrong_type")]
    WrongType,
    /// Contract e7b70be3, section 16.3: `unsupported_argument`.
    #[serde(rename = "unsupported_argument")]
    UnsupportedArgument,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestRefError {
    /// Contract e7b70be3, section 16.3: `request_id_oversize`.
    #[serde(rename = "request_id_oversize")]
    RequestIdOversize,
    /// Contract e7b70be3, section 16.3: `request_id_invalid`.
    #[serde(rename = "request_id_invalid")]
    RequestIdInvalid,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallIdError {
    /// Contract e7b70be3, section 16.3: `missing_id`.
    #[serde(rename = "missing_id")]
    MissingId,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeArgumentsError {
    /// Contract e7b70be3, section 16.3: `no_string_form`.
    #[serde(rename = "no_string_form")]
    NoStringForm,
    /// Contract e7b70be3, section 16.3: `null_arguments`.
    #[serde(rename = "null_arguments")]
    NullArguments,
    /// Contract e7b70be3, section 16.3: `absent_arguments`.
    #[serde(rename = "absent_arguments")]
    AbsentArguments,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArgumentsPayloadError {
    /// Contract e7b70be3, section 16.3: `store_not_configured`.
    #[serde(rename = "store_not_configured")]
    StoreNotConfigured,
    /// Contract e7b70be3, section 16.3: `parse_error`.
    #[serde(rename = "parse_error")]
    ParseError,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoreError {
    /// Contract e7b70be3, section 16.3: `store_not_configured`.
    #[serde(rename = "store_not_configured")]
    StoreNotConfigured,
    /// Contract 34e79d9e, section 16.3: attempt-scoped failure to establish
    /// and verify a usable persisted content reference with a configured store.
    /// This does not claim that the store holds no copy.
    #[serde(rename = "not_persisted")]
    NotPersisted,
}

/// Contract 34e79d9e, section 16.3: detection payload storage availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionPayloadError {
    /// Contract 34e79d9e, section 16.3: `store_not_configured`.
    #[serde(rename = "store_not_configured")]
    StoreNotConfigured,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeError {
    /// Contract e7b70be3, section 16.3: `no_build_digest`.
    #[serde(rename = "no_build_digest")]
    NoBuildDigest,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionError {
    /// Contract e7b70be3, section 16.3: `scan_error`.
    #[serde(rename = "scan_error")]
    ScanError,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeforeStateError {
    /// Contract e7b70be3, section 16.3: `no_read_access`.
    #[serde(rename = "no_read_access")]
    NoReadAccess,
    /// Contract e7b70be3, section 16.3: `io_error`.
    #[serde(rename = "io_error")]
    IoError,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetError {
    /// Contract e7b70be3, section 16.3: `not_resolved`.
    #[serde(rename = "not_resolved")]
    NotResolved,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptError {
    /// Contract e7b70be3, section 16.3: `action_id_collision`.
    #[serde(rename = "action_id_collision")]
    ActionIdCollision,
    /// Contract e7b70be3, section 16.3: `outside_root`.
    #[serde(rename = "outside_root")]
    OutsideRoot,
    /// Contract e7b70be3, section 16.3: `symlink_component`.
    #[serde(rename = "symlink_component")]
    SymlinkComponent,
    /// Contract e7b70be3, section 16.3: `traversal`.
    #[serde(rename = "traversal")]
    Traversal,
    /// Contract e7b70be3, section 16.3: `policy_drift`.
    #[serde(rename = "policy_drift")]
    PolicyDrift,
    /// Contract e7b70be3, section 16.3: `size_limit`.
    #[serde(rename = "size_limit")]
    SizeLimit,
    /// Contract e7b70be3, section 16.3: `sink_rejected`.
    #[serde(rename = "sink_rejected")]
    SinkRejected,
    /// Contract e7b70be3, section 16.3: `sink_unavailable`.
    #[serde(rename = "sink_unavailable")]
    SinkUnavailable,
    /// Contract e7b70be3, section 16.3: `sink_ref_mismatch`.
    #[serde(rename = "sink_ref_mismatch")]
    SinkRefMismatch,
    /// Contract e7b70be3, section 16.3: `root_unavailable`.
    #[serde(rename = "root_unavailable")]
    RootUnavailable,
    /// Contract e7b70be3, section 16.3: `io_error`.
    #[serde(rename = "io_error")]
    IoError,
    /// Contract e7b70be3, section 16.3: `interrupted`.
    #[serde(rename = "interrupted")]
    Interrupted,
    /// Contract e7b70be3, section 16.3: `rename_unconfirmed`.
    #[serde(rename = "rename_unconfirmed")]
    RenameUnconfirmed,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationTargetError {
    /// Contract e7b70be3, section 16.3: `path_unresolvable`.
    #[serde(rename = "path_unresolvable")]
    PathUnresolvable,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservedError {
    /// Contract e7b70be3, section 16.3: `no_read_access`.
    #[serde(rename = "no_read_access")]
    NoReadAccess,
    /// Contract e7b70be3, section 16.3: `io_error`.
    #[serde(rename = "io_error")]
    IoError,
    /// Contract e7b70be3, section 16.3: `observer_failed`.
    #[serde(rename = "observer_failed")]
    ObserverFailed,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FootprintError {
    /// Contract e7b70be3, section 16.3: `not_configured`.
    #[serde(rename = "not_configured")]
    NotConfigured,
}

/// Contract e7b70be3, section 16.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationReason {
    /// Contract e7b70be3, section 16.3: `attempt_ambiguous`.
    #[serde(rename = "attempt_ambiguous")]
    AttemptAmbiguous,
    /// Contract e7b70be3, section 16.3: `observation_missing`.
    #[serde(rename = "observation_missing")]
    ObservationMissing,
    /// Contract e7b70be3, section 16.3: `attempt_errored`.
    #[serde(rename = "attempt_errored")]
    AttemptErrored,
    /// Contract e7b70be3, section 16.3: `outside_root`.
    #[serde(rename = "outside_root")]
    OutsideRoot,
    /// Contract e7b70be3, section 16.3: `target_missing`.
    #[serde(rename = "target_missing")]
    TargetMissing,
    /// Contract e7b70be3, section 16.3: `length_mismatch`.
    #[serde(rename = "length_mismatch")]
    LengthMismatch,
    /// Contract e7b70be3, section 16.3: `digest_mismatch`.
    #[serde(rename = "digest_mismatch")]
    DigestMismatch,
    /// Contract e7b70be3, section 16.3: `before_state_missing`.
    #[serde(rename = "before_state_missing")]
    BeforeStateMissing,
    /// Contract e7b70be3, section 16.3: `no_delta_observed`.
    #[serde(rename = "no_delta_observed")]
    NoDeltaObserved,
}

/// Contract e7b70be3, section 5.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadRef {
    /// Contract e7b70be3, section 5.4: `digest`.
    pub digest: String,
    /// Contract e7b70be3, section 5.4: `len_bytes`.
    pub len_bytes: u64,
    /// Contract e7b70be3, section 5.4: `media_type`.
    pub media_type: String,
    /// Contract e7b70be3, section 5.4: `store`.
    pub store: PayloadStore,
}

/// Contract e7b70be3, section 5.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestRef {
    /// Contract e7b70be3, section 5.1: `request_id`.
    pub request_id: String,
    /// Contract e7b70be3, section 5.1: `request_id_source`.
    pub request_id_source: RequestIdSource,
}

/// Contract e7b70be3, section 6.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedEffect {
    /// Contract e7b70be3, section 6.1: `target_path`.
    pub target_path: String,
    /// Contract e7b70be3, section 6.1: `content_digest`.
    pub content_digest: String,
    /// Contract e7b70be3, section 6.1: `content_len_bytes`.
    pub content_len_bytes: u64,
    /// Contract e7b70be3, section 6.1: `content_media_type`.
    pub content_media_type: String,
}

/// Contract e7b70be3, section 6.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    /// Contract e7b70be3, section 6: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 6: `stage`.
    pub stage: ProposalStage,
    /// Contract e7b70be3, section 6: `provider`.
    pub provider: Provider,
    /// Contract e7b70be3, section 6: `index`.
    pub index: u32,
    /// Contract e7b70be3, section 6: `tool_name`.
    pub tool_name: PresentUnavailable<String, ToolNameError>,
    /// Contract e7b70be3, section 6: `parse_status`.
    pub parse_status: ParseStatus,
    /// Contract e7b70be3, section 6: `parse_error_code`.
    #[serde(deserialize_with = "required_nullable")]
    pub parse_error_code: Option<ParseError>,
    /// Contract e7b70be3, section 6: `arguments_digest`.
    pub arguments_digest: PresentErrored<String, ArgumentsError>,
    /// Contract e7b70be3, section 6: `arguments_len_bytes`.
    pub arguments_len_bytes: PresentErrored<u64, ArgumentsError>,
    /// Contract e7b70be3, section 6: `effect_kind`.
    pub effect_kind: EffectKind,
    /// Contract e7b70be3, section 6: `expected_effect`.
    pub expected_effect: PresentAbsentErrored<ExpectedEffect, ExpectedEffectError>,
}

/// Contract e7b70be3, section 7.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalContext {
    /// Contract e7b70be3, section 7: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 7: `stage`.
    pub stage: ContextStage,
    /// Contract e7b70be3, section 7: `proposal_ref`.
    pub proposal_ref: String,
    /// Contract e7b70be3, section 7: `request_ref`.
    pub request_ref: PresentUnavailable<RequestRef, RequestRefError>,
    /// Contract e7b70be3, section 7: `choice_index`.
    pub choice_index: u32,
    /// Contract e7b70be3, section 7: `model`.
    pub model: String,
    /// Contract e7b70be3, section 7: `tool_call_id`.
    pub tool_call_id: PresentUnavailable<String, ToolCallIdError>,
    /// Contract e7b70be3, section 7: `arguments_wire_form`.
    pub arguments_wire_form: ArgumentsWireForm,
    /// Contract e7b70be3, section 7: `native_arguments`.
    pub native_arguments: PresentUnavailable<PayloadRef, NativeArgumentsError>,
    /// Contract e7b70be3, section 7: `arguments_payload`.
    pub arguments_payload: PresentUnavailable<PayloadRef, ArgumentsPayloadError>,
    /// Contract e7b70be3, section 7: `detection_payload`.
    pub detection_payload: PresentUnavailableAbsent<PayloadRef, DetectionPayloadError>,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyIdentity {
    /// Contract e7b70be3, section 9.1: `contract_hash`.
    pub contract_hash: String,
    /// Contract e7b70be3, section 9.1: `policy_digest`.
    pub policy_digest: String,
    /// Contract e7b70be3, section 9.1: `policy_version`.
    pub policy_version: String,
}

/// Contract e7b70be3, section 5.5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeIdentity {
    /// Contract e7b70be3, section 5.5: `code_version`.
    pub code_version: String,
    /// Contract e7b70be3, section 5.5: `code_digest`.
    pub code_digest: String,
}

/// Contract e7b70be3, section 9.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detection {
    /// Contract e7b70be3, section 9.2: `record_digest`.
    pub record_digest: String,
    /// Contract e7b70be3, section 9.2: `record_len_bytes`.
    pub record_len_bytes: u64,
    /// Contract e7b70be3, section 9.2: `schema_version`.
    pub schema_version: DetectionSchema,
    /// Contract e7b70be3, section 9.2: `detection_outcome`.
    pub detection_outcome: DetectionOutcome,
    /// Contract e7b70be3, section 9.2: `mode`.
    pub mode: DetectionMode,
    /// Contract e7b70be3, section 9.2: `input_digest`.
    pub input_digest: String,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedConstraint {
    /// Contract e7b70be3, section 9: `argument`.
    #[serde(deserialize_with = "required_nullable")]
    pub argument: Option<String>,
    /// Contract e7b70be3, section 9: `constraint`.
    pub constraint: ConstraintName,
}

/// Contract e7b70be3, section 9.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    /// Contract e7b70be3, section 9: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 9: `stage`.
    pub stage: DecisionStage,
    /// Contract e7b70be3, section 9: `proposal_ref`.
    pub proposal_ref: String,
    /// Contract e7b70be3, section 9: `policy`.
    pub policy: PolicyIdentity,
    /// Contract e7b70be3, section 9: `code`.
    pub code: PresentUnavailable<CodeIdentity, CodeError>,
    /// Contract e7b70be3, section 9: `detection`.
    pub detection: PresentUnavailableAbsent<Detection, DetectionError>,
    /// Contract e7b70be3, section 9: `detection_binding_rule`.
    pub detection_binding_rule: PresentAbsent<DetectionBindingRule>,
    /// Contract e7b70be3, section 9: `enforcement`.
    pub enforcement: Enforcement,
    /// Contract e7b70be3, section 9: `decision`.
    pub decision: DecisionOutcome,
    /// Contract e7b70be3, section 9: `evaluated_policy`.
    pub evaluated_policy: bool,
    /// Contract e7b70be3, section 9: `reason_code`.
    pub reason_code: DecisionReason,
    /// Contract e7b70be3, section 9: `failed_constraints`.
    pub failed_constraints: Vec<FailedConstraint>,
    /// Contract e7b70be3, section 9: `inputs_consumed`.
    pub inputs_consumed: Vec<InputConsumed>,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Executor {
    /// Contract e7b70be3, section 10: `name`.
    pub name: ExecutorName,
    /// Contract e7b70be3, section 10: `version`.
    pub version: String,
}

/// Contract e7b70be3, section 5.6.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Root {
    /// Contract e7b70be3, section 5.6: `root_id`.
    pub root_id: String,
    /// Contract e7b70be3, section 5.6: `root_digest`.
    pub root_digest: String,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Contract e7b70be3, section 10: `proposed_path_digest`.
    pub proposed_path_digest: String,
    /// Contract e7b70be3, section 10: `target_ref`.
    pub target_ref: PresentUnavailable<String, TargetError>,
    /// Contract e7b70be3, section 10: `resolved_path`.
    pub resolved_path: PresentUnavailable<String, TargetError>,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SinkResult {
    /// Contract e7b70be3, section 10: `status`.
    pub status: SinkStatus,
    /// Contract e7b70be3, section 10: `ref`.
    #[serde(deserialize_with = "required_nullable")]
    pub r#ref: Option<String>,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sink {
    /// Contract e7b70be3, section 10: `proposal`.
    pub proposal: SinkResult,
    /// Contract e7b70be3, section 10: `context`.
    pub context: SinkResult,
    /// Contract e7b70be3, section 10: `decision`.
    pub decision: SinkResult,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCheck {
    /// Contract e7b70be3, section 10: `contract_hash`.
    pub contract_hash: String,
    /// Contract e7b70be3, section 10: `policy_digest`.
    pub policy_digest: String,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeforeState {
    /// Contract e7b70be3, section 10: `exists`.
    pub exists: bool,
    /// Contract e7b70be3, section 10: `digest`.
    #[serde(deserialize_with = "required_nullable")]
    pub digest: Option<String>,
    /// Contract e7b70be3, section 10: `len_bytes`.
    #[serde(deserialize_with = "required_nullable")]
    pub len_bytes: Option<u64>,
}

/// Contract e7b70be3, section 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    /// Contract e7b70be3, section 10: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 10: `stage`.
    pub stage: AttemptStage,
    /// Contract e7b70be3, section 10: `decision_ref`.
    pub decision_ref: String,
    /// Contract e7b70be3, section 10: `action_id`.
    pub action_id: String,
    /// Contract e7b70be3, section 10: `executor`.
    pub executor: Executor,
    /// Contract e7b70be3, section 10: `root`.
    pub root: Root,
    /// Contract e7b70be3, section 10: `target`.
    pub target: Target,
    /// Contract e7b70be3, section 10: `operation`.
    pub operation: Operation,
    /// Contract e7b70be3, section 10: `sink`.
    pub sink: Sink,
    /// Contract e7b70be3, section 10: `policy_check`.
    pub policy_check: PolicyCheck,
    /// Contract e7b70be3, section 10: `before_state`.
    pub before_state: PresentUnavailableErroredAbsent<BeforeState, BeforeStateError>,
    /// Contract e7b70be3, section 10: `outcome`.
    pub outcome: AttemptOutcome,
    /// Contract e7b70be3, section 10: `error_code`.
    #[serde(deserialize_with = "required_nullable")]
    pub error_code: Option<AttemptError>,
    /// Contract e7b70be3, section 10: `content_payload`.
    pub content_payload: PresentUnavailableAbsent<PayloadRef, StoreError>,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observer {
    /// Contract e7b70be3, section 11: `name`.
    pub name: String,
    /// Contract e7b70be3, section 11: `version`.
    pub version: String,
    /// Contract e7b70be3, section 11: `source`.
    pub source: ObserverSource,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observed {
    /// Contract e7b70be3, section 11: `exists`.
    pub exists: bool,
    /// Contract e7b70be3, section 11: `digest`.
    #[serde(deserialize_with = "required_nullable")]
    pub digest: Option<String>,
    /// Contract e7b70be3, section 11: `len_bytes`.
    #[serde(deserialize_with = "required_nullable")]
    pub len_bytes: Option<u64>,
    /// Contract e7b70be3, section 11: `resolved_within_root`.
    pub resolved_within_root: bool,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Footprint {
    /// Contract e7b70be3, section 11: `paths_checked`.
    pub paths_checked: Vec<String>,
    /// Contract e7b70be3, section 11: `undeclared_residue`.
    pub undeclared_residue: bool,
}

/// Contract e7b70be3, section 11.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// Contract e7b70be3, section 11: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 11: `stage`.
    pub stage: ObservationStage,
    /// Contract e7b70be3, section 11: `attempt_ref`.
    pub attempt_ref: String,
    /// Contract e7b70be3, section 11: `observer`.
    pub observer: Observer,
    /// Contract e7b70be3, section 11: `target_ref`.
    pub target_ref: PresentUnavailable<String, ObservationTargetError>,
    /// Contract e7b70be3, section 11: `observed`.
    pub observed: PresentUnavailableErrored<Observed, ObservedError>,
    /// Contract e7b70be3, section 11: `footprint`.
    pub footprint: PresentUnavailable<Footprint, FootprintError>,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    /// Contract e7b70be3, section 12: `expected_digest`.
    pub expected_digest: String,
    /// Contract e7b70be3, section 12: `observed_digest`.
    #[serde(deserialize_with = "required_nullable")]
    pub observed_digest: Option<String>,
    /// Contract e7b70be3, section 12: `expected_len_bytes`.
    pub expected_len_bytes: u64,
    /// Contract e7b70be3, section 12: `observed_len_bytes`.
    #[serde(deserialize_with = "required_nullable")]
    pub observed_len_bytes: Option<u64>,
    /// Contract e7b70be3, section 12: `within_root`.
    pub within_root: bool,
    /// Contract e7b70be3, section 12: `before_digest`.
    #[serde(deserialize_with = "required_nullable")]
    pub before_digest: Option<String>,
    /// Contract e7b70be3, section 12: `delta`.
    pub delta: Delta,
}

/// Contract e7b70be3, section 12.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verification {
    /// Contract e7b70be3, section 12: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 12: `stage`.
    pub stage: VerificationStage,
    /// Contract e7b70be3, section 12: `observation_ref`.
    pub observation_ref: String,
    /// Contract e7b70be3, section 12: `method`.
    pub method: VerificationMethod,
    /// Contract e7b70be3, section 12: `outcome`.
    pub outcome: VerificationOutcome,
    /// Contract e7b70be3, section 12: `comparison`.
    pub comparison: PresentAbsent<Comparison>,
    /// Contract e7b70be3, section 12: `reason_code`.
    #[serde(deserialize_with = "required_nullable")]
    pub reason_code: Option<VerificationReason>,
}

/// Contract e7b70be3, section 14.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stages {
    /// Contract e7b70be3, section 14.1: `proposal`.
    pub proposal: PresentSlot<Proposal>,
    /// Contract e7b70be3, section 14.1: `decision`.
    pub decision: PresentSlot<Decision>,
    /// Contract e7b70be3, section 14.1: `attempt`.
    pub attempt: Slot<Attempt, AttemptAbsentReason>,
    /// Contract e7b70be3, section 14.1: `observation`.
    pub observation: Slot<Observation, PostAttemptAbsentReason>,
    /// Contract e7b70be3, section 14.1: `verification`.
    pub verification: Slot<Verification, PostAttemptAbsentReason>,
}

/// Contract e7b70be3, section 14.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Contract e7b70be3, section 14.1: `schema_version`.
    pub schema_version: ReceiptSchema,
    /// Contract e7b70be3, section 14.1: `proposal_id`.
    pub proposal_id: String,
    /// Contract e7b70be3, section 14.1: `action_id`.
    #[serde(deserialize_with = "required_nullable")]
    pub action_id: Option<String>,
    /// Contract e7b70be3, section 14.1: `stages`.
    pub stages: Stages,
    /// Contract e7b70be3, section 14.1: `proposal_context`.
    pub proposal_context: PresentSlot<ProposalContext>,
    /// Contract e7b70be3, section 14.1: `summary`.
    pub summary: Summary,
    /// Contract e7b70be3, section 14.1: `float_policy`.
    pub float_policy: FloatPolicy,
    /// Contract e7b70be3, section 14.1: `crosswalk`.
    pub crosswalk: Crosswalk,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedActionPolicyV0 {
    /// Contract e7b70be3, section 9.1: `schema_version`.
    pub schema_version: PolicySchema,
    /// Contract e7b70be3, section 9.1: `policy_version`.
    pub policy_version: String,
    /// Contract e7b70be3, section 9.1: `l3_outbound`.
    pub l3_outbound: ResolvedL3Outbound,
    /// Contract e7b70be3, section 9.1: `tools`.
    pub tools: BTreeMap<String, ResolvedTool>,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedL3Outbound {
    /// Contract e7b70be3, section 9.1: `present`.
    pub present: bool,
    /// Contract e7b70be3, section 9.1: `mode`.
    #[serde(deserialize_with = "required_nullable")]
    pub mode: Option<String>,
    /// Contract e7b70be3, section 9.1: `block_action_configured`.
    #[serde(deserialize_with = "required_nullable")]
    pub block_action_configured: Option<String>,
    /// Contract e7b70be3, section 9.1: `block_action_effective`.
    pub block_action_effective: BlockAction,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedTool {
    /// Contract e7b70be3, section 9.1: `allowed`.
    pub allowed: bool,
    /// Contract e7b70be3, section 9.1: `constraints`.
    pub constraints: BTreeMap<String, ResolvedConstraints>,
}

/// Contract e7b70be3, section 9.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedConstraints {
    /// Contract e7b70be3, section 9.1: `type_check`.
    #[serde(deserialize_with = "required_nullable")]
    pub type_check: Option<TypeCheck>,
    /// Contract e7b70be3, section 9.1: `starts_with`.
    #[serde(deserialize_with = "required_nullable")]
    pub starts_with: Option<String>,
    /// Contract e7b70be3, section 9.1: `not_contains`.
    #[serde(deserialize_with = "required_nullable")]
    pub not_contains: Option<Vec<String>>,
    /// Contract e7b70be3, section 9.1: `matches`.
    #[serde(deserialize_with = "required_nullable")]
    pub matches: Option<String>,
    /// Contract e7b70be3, section 9.1: `one_of`.
    #[serde(deserialize_with = "required_nullable")]
    pub one_of: Option<Vec<String>>,
    /// Contract e7b70be3, section 9.1: `max_length`.
    #[serde(deserialize_with = "required_nullable")]
    pub max_length: Option<u64>,
    /// Contract e7b70be3, section 9.1: `min`.
    #[serde(deserialize_with = "required_nullable")]
    pub min: Option<f64>,
    /// Contract e7b70be3, section 9.1: `max`.
    #[serde(deserialize_with = "required_nullable")]
    pub max: Option<f64>,
    /// Contract e7b70be3, section 9.1: `url_host`.
    #[serde(deserialize_with = "required_nullable")]
    pub url_host: Option<Vec<String>>,
}

// Nullable fields are required keys, even when their value is null.
fn required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
