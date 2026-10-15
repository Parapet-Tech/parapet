// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Closed 1B interface vocabulary (Routing contract 50747a52, sections 3 and 4.2).
//!
//! No type here carries free-form log text. Every error that crosses an
//! interface is a closed enum mapped to a receipt code, an envelope instrument
//! or an outer runtime/reader status.

use crate::receipt::{
    Attempt, CodeError, CodeIdentity, Decision, ExpectedEffect, Observation, PayloadResolution,
    PayloadStore, PresentUnavailable, Proposal, Provider, RejectionCode, ResolvedActionPolicyV0,
    Row, Sink, SinkResult,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Candidates (section 3 step 2)
// ---------------------------------------------------------------------------

/// Section 3: the exact formed provider coordinate of one raw candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRefV0 {
    /// Draft 12 section 6 field 3.
    pub provider: Provider,
    /// Draft 12 section 5.2: OpenAI `choices[].index`; `0` without choices.
    pub choice_index: u32,
    /// Draft 12 section 5.2: provider position value.
    pub index: u32,
}

impl CandidateRefV0 {
    /// Total order key: provider, then (`choice_index`, `index`).
    pub fn key(&self) -> (u8, u32, u32) {
        let provider = match self.provider {
            Provider::Openai => 0,
            Provider::Anthropic => 1,
        };
        (provider, self.choice_index, self.index)
    }
}

/// Section 4.4: transport a candidate was materialized from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportV0 {
    /// Fully buffered response body.
    Buffered,
    /// Materialized streaming call.
    Streaming,
}

/// Draft 12 section 5.2: why no row coordinate could be formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateErrorV0 {
    /// OpenAI choice without an integer `index`.
    ChoiceIndexUnformable,
    /// Tool call or content block without an integer `index`.
    CallIndexUnformable,
    /// A coordinate outside `[0, 2^31)`.
    IndexOutOfRange,
}

/// Draft 12 section 6 field 7: stream-side failures retained on a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFailureV0 {
    /// `buffer_overflow`.
    BufferOverflow,
    /// `delta_timeout`.
    DeltaTimeout,
    /// `stream_aborted`.
    StreamAborted,
}

/// Draft 12 section 7 field 8: what the provider carried for arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum RawArgumentsV0 {
    /// A JSON string, or the exact concatenation of streaming fragments.
    String(String),
    /// A provider object, or another non-string JSON value.
    Object(serde_json::Value),
    /// An explicit JSON null.
    Null,
    /// No arguments member.
    Absent,
}

/// Section 4.2 `ProposalCandidateEnumeratorV0` output: one raw candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ProposalCandidateV0 {
    /// Stable position in wire order, assigned by the enumerator.
    pub ordinal: u32,
    /// Provider family.
    pub provider: Provider,
    /// Buffered or streaming.
    pub transport: TransportV0,
    /// The real coordinate, or why none could be formed.
    pub coordinate: Result<CandidateRefV0, CoordinateErrorV0>,
    /// Provider tool-call id, when carried as a string.
    pub tool_call_id: Option<String>,
    /// Tool name, when carried as a string.
    pub tool_name: Option<String>,
    /// Provider-native arguments.
    pub arguments: RawArgumentsV0,
    /// Representable stream failure; `None` for a completed materialization.
    pub failure: Option<StreamFailureV0>,
    /// A provider fragment had a shape that cannot be carried; parsing errors.
    pub malformed: bool,
    /// Wire location: (outer array position, inner array position).
    pub location: (u32, u32),
}

/// Section 4.2: raw response or fragments to ordered candidates.
pub trait ProposalCandidateEnumeratorV0 {
    /// Enumerate every candidate of a fully buffered response in wire order.
    fn enumerate_buffered(
        &self,
        provider: Provider,
        body: &serde_json::Value,
    ) -> Vec<ProposalCandidateV0>;
}

// ---------------------------------------------------------------------------
// Policy, identity and decision (section 3 step 4)
// ---------------------------------------------------------------------------

/// Section 4.2 `ResolvedPolicyProviderV0` output.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPolicyV0 {
    /// The landed value.
    pub policy: ResolvedActionPolicyV0,
    /// Landed `canonical_policy_bytes` output.
    pub bytes: Vec<u8>,
    /// Landed `policy_digest` over `bytes`.
    pub digest: String,
    /// `Config.contract_hash` verbatim.
    pub contract_hash: String,
}

/// Closed policy resolution failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyResolveErrorV0 {
    /// A configured `type` outside `string | number | boolean`.
    TypeCheckUnrepresentable,
    /// A non-finite numeric bound.
    NonFiniteBound,
    /// Landed serializer failure.
    Serialization,
}

/// Section 4.2: exact resolved policy from the live configuration.
pub trait ResolvedPolicyProviderV0: Send + Sync {
    /// Resolve the policy value, its canonical bytes and digests.
    fn resolve(&self) -> Result<ResolvedPolicyV0, PolicyResolveErrorV0>;
}

/// Section 4.2: injected build identity with no ambient fallback.
pub trait BuildIdentityProviderV0: Send + Sync {
    /// Present identity, or exactly `unavailable no_build_digest`.
    fn identity(&self) -> PresentUnavailable<CodeIdentity, CodeError>;
}

/// Section 4.3 and 4.4: whether the response is fully buffered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryModeV0 {
    /// Whole-response suppression is possible before serialization.
    Buffered,
    /// Each call is finalized independently and never retroactively suppressed.
    Streaming,
}

// ---------------------------------------------------------------------------
// Protected store and sink (sections 3 steps 8 and 10, 4.3)
// ---------------------------------------------------------------------------

/// Section 4.3: validated run identifier (32 lowercase hex).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunIdV0(String);

impl RunIdV0 {
    /// Parse a run id; only 32 lowercase hex characters are accepted.
    pub fn parse(value: &str) -> Option<Self> {
        (value.len() == 32
            && value
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then(|| Self(value.to_owned()))
    }

    /// A fresh random run id.
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().simple().to_string())
    }

    /// Borrow the id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Section 4.3: closed store failures. No path or OS text is carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreFailureV0 {
    /// Root or directory is a symlink, wrong type, wrong owner or wrong mode.
    UnsafeLayout,
    /// An existing object or file is not a private regular file.
    UnsafeObject,
    /// An existing immutable object has different bytes.
    Collision,
    /// Bytes read back differ in digest.
    DigestMismatch,
    /// Bytes read back differ in length.
    LengthMismatch,
    /// The requested label is not served by this store.
    LabelNotConfigured,
    /// The run manifest is missing.
    ManifestMissing,
    /// The run manifest does not parse or does not validate.
    ManifestCorrupt,
    /// The run is not in a state that admits this transition.
    StateInvalid,
    /// The exclusive store lock could not be taken.
    LockFailed,
    /// A filesystem operation failed.
    Io,
    /// An injected failure point fired.
    Injected,
}

/// Section 4.3: one stored, content-addressed payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredPayloadV0 {
    /// Relative store path, the stable reference.
    pub path: String,
    /// Raw untagged sha256 of the bytes.
    pub digest: String,
    /// Byte length.
    pub len_bytes: u64,
}

/// Section 4.2: content-addressed protected payload store.
pub trait ProtectedPayloadStoreV0: Send + Sync {
    /// Attribute, write and verify payload bytes for one run.
    fn put(
        &self,
        run: &RunIdV0,
        label: PayloadStore,
        bytes: &[u8],
    ) -> Result<StoredPayloadV0, StoreFailureV0>;
    /// Resolve payload bytes by raw digest with digest and length verification.
    fn get(&self, label: PayloadStore, digest: &str) -> PayloadResolution;
}

/// Section 4.2: later records, row and projection persisted at terminal time.
#[derive(Debug, Clone)]
pub struct TerminalBatchV0<'a> {
    /// Canonical attempt, observation and verification bytes that are present.
    pub later_records: Vec<(crate::receipt::RecordDomain, StageKindV0, &'a [u8])>,
    /// Canonical row bytes.
    pub row_bytes: &'a [u8],
    /// Landed row digest of `row_bytes`.
    pub row_digest: &'a str,
    /// Landed projection bytes, unchanged.
    pub projection_bytes: &'a [u8],
}

/// Section 4.3: closed object kinds named by run manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKindV0 {
    /// Protected payload bytes.
    Payload,
    /// Proposal record.
    Proposal,
    /// Proposal context record.
    ProposalContext,
    /// Decision record.
    Decision,
    /// Attempt record.
    Attempt,
    /// Observation record.
    Observation,
    /// Verification record.
    Verification,
    /// Row envelope.
    Row,
    /// Unchanged redacted projection.
    Projection,
}

/// Section 4.2: synchronous receipt sink.
pub trait ReceiptSinkV0: Send + Sync {
    /// Persist exactly proposal, context and decision; always three results.
    fn persist_pre_attempt(&self, run: &RunIdV0, records: [&[u8]; 3]) -> [SinkResult; 3];
    /// Persist later stages, row and projection; any failure is storage-incomplete.
    fn persist_terminal(
        &self,
        run: &RunIdV0,
        batch: &TerminalBatchV0<'_>,
    ) -> Result<Vec<String>, StoreFailureV0>;
}

// ---------------------------------------------------------------------------
// Optional execution and observation (section 3 steps 6-9)
// ---------------------------------------------------------------------------

/// Section 4.5: explicit file-effect profile selection marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileEffectProfileV0 {
    _selected: (),
}

impl FileEffectProfileV0 {
    /// Explicit opt-in; there is no default.
    pub fn select_explicitly() -> Self {
        Self { _selected: () }
    }
}

/// Draft 12 section 5.3: caller-assigned action id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionIdV0(String);

impl ActionIdV0 {
    /// 1 to 128 bytes matching `^[a-z0-9][a-z0-9._:-]*$`.
    pub fn parse(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();
        let valid = (1..=128).contains(&bytes.len())
            && matches!(bytes[0], b'a'..=b'z' | b'0'..=b'9')
            && bytes
                .iter()
                .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b':' | b'-'));
        valid.then(|| Self(value.to_owned()))
    }

    /// Borrow the id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Section 4.5: opaque root capability; 1B never inspects it.
#[derive(Clone)]
pub struct OpaqueRootCapabilityV0(Arc<dyn Any + Send + Sync>);

impl OpaqueRootCapabilityV0 {
    /// Wrap a capability produced outside 1B.
    pub fn new(capability: Arc<dyn Any + Send + Sync>) -> Self {
        Self(capability)
    }

    /// Borrow the wrapped capability for the bound executor or observer.
    pub fn inner(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.0
    }
}

impl std::fmt::Debug for OpaqueRootCapabilityV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpaqueRootCapabilityV0")
    }
}

/// Section 4.2: everything the executor preflight receives.
pub struct ExecutorRequestV0<'a> {
    /// Finalized proposal record.
    pub proposal: &'a Proposal,
    /// Finalized decision record.
    pub decision: &'a Decision,
    /// Landed stage digest of the decision bytes.
    pub decision_ref: &'a str,
    /// Expected digests of proposal, context and decision, in that order.
    pub expected_refs: [&'a str; 3],
    /// Caller action id bound at construction.
    pub action_id: &'a ActionIdV0,
    /// Expected effect of the proposal.
    pub expected_effect: &'a ExpectedEffect,
    /// Content bytes for the effect, never logged.
    pub content: &'a [u8],
    /// Opaque root capability bound at construction.
    pub root: &'a OpaqueRootCapabilityV0,
    /// The complete ordered three-result sink vector.
    pub sink: &'a Sink,
}

/// Section 4.2: optional executor; returns the Sensor attempt shape.
pub trait ActionExecutorV0: Send + Sync {
    /// Run preflight (sink precedence first) and, only if it passes, the operation.
    fn execute(&self, request: &ExecutorRequestV0<'_>) -> Attempt;
}

/// Closed observer failure; the orchestrator writes `observer_failed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverFailureV0 {
    /// The observer could not produce a record.
    ObserverFailed,
}

/// Section 4.2: independent post-attempt observation.
pub trait EffectObserverV0: Send + Sync {
    /// Observer identity written on orchestrator-produced records.
    fn identity(&self) -> crate::receipt::Observer;
    /// Observe the target after a non-rejected attempt.
    fn observe(
        &self,
        proposal: &Proposal,
        attempt: &Attempt,
        attempt_ref: &str,
        root: &OpaqueRootCapabilityV0,
    ) -> Result<Observation, ObserverFailureV0>;
}

/// Section 3 step 6: construction-time executor binding.
#[derive(Clone)]
pub struct ExecutorBindingV0 {
    pub(crate) executor: Arc<dyn ActionExecutorV0>,
    pub(crate) observer: Arc<dyn EffectObserverV0>,
    pub(crate) root: OpaqueRootCapabilityV0,
    pub(crate) action_id: ActionIdV0,
}

impl ExecutorBindingV0 {
    /// Bind exactly an explicit profile, a root capability and a caller action id.
    pub fn new(
        _profile: FileEffectProfileV0,
        root: OpaqueRootCapabilityV0,
        action_id: ActionIdV0,
        executor: Arc<dyn ActionExecutorV0>,
        observer: Arc<dyn EffectObserverV0>,
    ) -> Self {
        Self {
            executor,
            observer,
            root,
            action_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Envelope results (section 3, 4.2 `RuntimeEnvelopeV0`)
// ---------------------------------------------------------------------------

/// Section 3 step 10: outer runtime completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCompletionV0 {
    /// Every terminal record, row, digest and projection is persisted.
    Complete,
    /// Evidence persistence failed; inspection and cleanup are required.
    StorageIncomplete,
}

/// Section 3 step 5: client delivery outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryOutcomeV0 {
    /// Response delivered without tool-call changes.
    Delivered,
    /// Blocked calls removed; refusal text injected.
    Rewritten,
    /// The whole buffered response was replaced before serialization.
    SuppressedWholeResponse,
    /// Each streaming call was finalized and released or withheld independently.
    StreamedPerCall,
}

/// Closed envelope instruments (draft 12 section 16.4 and 1B additions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeInstrumentV0 {
    /// Candidate count differs from row count.
    ProposalCountMismatch,
    /// Two formed candidates share a coordinate.
    ProposalCoordinateDuplicate,
    /// A candidate's coordinate could not be formed; no run.
    RowCoordinateUnformable,
    /// The assembled row did not validate.
    RowValidationRejected(RejectionCode),
    /// Evidence persistence failed for a run.
    StorageIncomplete(StoreFailureV0),
    /// The bound executor returned an attempt inconsistent with its inputs.
    ExecutorContractViolation,
    /// The observer failed; `observer_failed` was written.
    ObserverFailed,
    /// Policy could not be resolved; the response is withheld.
    PolicyUnresolvable(PolicyResolveErrorV0),
    /// The validator could not resolve a referenced payload.
    PayloadUnresolved,
}

impl EnvelopeInstrumentV0 {
    /// Closed telemetry code.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProposalCountMismatch => "proposal_count_mismatch",
            Self::ProposalCoordinateDuplicate => "proposal_coordinate_duplicate",
            Self::RowCoordinateUnformable => "row_coordinate_unformable",
            Self::RowValidationRejected(_) => "row_validation_rejected",
            Self::StorageIncomplete(_) => "storage_incomplete",
            Self::ExecutorContractViolation => "executor_contract_violation",
            Self::ObserverFailed => "observer_failed",
            Self::PolicyUnresolvable(_) => "policy_unresolvable",
            Self::PayloadUnresolved => "payload_unresolved",
        }
    }
}

/// Stable store references of one run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StableRefsV0 {
    /// Payload paths in attribution order.
    pub payloads: Vec<String>,
    /// Stage record paths in lifecycle order (proposal, context, decision, later).
    pub records: Vec<String>,
    /// Row path.
    pub row: Option<String>,
    /// Projection path.
    pub projection: Option<String>,
}

/// One run: one formed candidate, one row.
#[derive(Debug, Clone)]
pub struct RunResultV0 {
    /// Protected run id.
    pub run_id: RunIdV0,
    /// Exact enumerator coordinate.
    pub candidate_ref: CandidateRefV0,
    /// Assembled row (retained in memory even when storage failed).
    pub row: Row,
    /// Canonical row bytes.
    pub row_bytes: Vec<u8>,
    /// Landed row digest.
    pub row_digest: String,
    /// Landed projection bytes, unchanged.
    pub projection_bytes: Vec<u8>,
    /// Landed validator result over `row_bytes`.
    pub validation: crate::receipt::ValidationResult,
    /// Stable references.
    pub refs: StableRefsV0,
    /// Outer status, outside every Sensor value.
    pub completion: RuntimeCompletionV0,
    /// Outer banner when storage is incomplete.
    pub banner: Option<&'static str>,
}

/// Total coordinate-to-run-to-row mapping entry, one per raw candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct MappingEntryV0 {
    /// Enumerator ordinal.
    pub ordinal: u32,
    /// Formed coordinate, if any.
    pub candidate_ref: Option<CandidateRefV0>,
    /// Run id, present exactly when the coordinate formed.
    pub run_id: Option<RunIdV0>,
    /// Row digest, present exactly when the coordinate formed.
    pub row_digest: Option<String>,
    /// `row_coordinate_unformable` when no coordinate formed.
    pub instrument: Option<EnvelopeInstrumentV0>,
}

/// Section 4.2 `RuntimeEnvelopeV0` output for one response.
#[derive(Debug, Clone)]
pub struct EnvelopeResponseResultV0 {
    /// Client delivery outcome.
    pub delivery: DeliveryOutcomeV0,
    /// Total mapping over raw candidates in wire order.
    pub mapping: Vec<MappingEntryV0>,
    /// Runs in wire order.
    pub runs: Vec<RunResultV0>,
    /// Closed instruments.
    pub instruments: Vec<EnvelopeInstrumentV0>,
    /// Outer completion: complete only when every run is complete.
    pub completion: RuntimeCompletionV0,
}

/// Optional consumer of envelope results produced inside the live engine.
pub trait EnvelopeResultSinkV0: Send + Sync {
    /// Receive one response's (or one stream's) result.
    fn accept(&self, result: EnvelopeResponseResultV0);
}

/// Outer banner shown beside, never inside, the Sensor projection.
pub const STORAGE_FAILURE_BANNER: &str =
    "RECEIPT EVIDENCE NOT PERSISTED: storage failed; completion is not established; inspect and clean up";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_reject_free_form_text() {
        assert!(RunIdV0::parse("0123456789abcdef0123456789abcdef").is_some());
        for bad in [
            "",
            "../x",
            "0123456789ABCDEF0123456789ABCDEF",
            "a run id with spaces",
        ] {
            assert!(RunIdV0::parse(bad).is_none(), "{bad}");
        }
        assert!(ActionIdV0::parse("fixture:action-1").is_some());
        for bad in ["", "Upper", "-lead", "has space", "/abs", &"a".repeat(129)] {
            assert!(ActionIdV0::parse(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn instruments_are_closed_codes() {
        let all = [
            EnvelopeInstrumentV0::ProposalCountMismatch,
            EnvelopeInstrumentV0::ProposalCoordinateDuplicate,
            EnvelopeInstrumentV0::RowCoordinateUnformable,
            EnvelopeInstrumentV0::RowValidationRejected(RejectionCode::SchemaError),
            EnvelopeInstrumentV0::StorageIncomplete(StoreFailureV0::Io),
            EnvelopeInstrumentV0::ExecutorContractViolation,
            EnvelopeInstrumentV0::ObserverFailed,
            EnvelopeInstrumentV0::PolicyUnresolvable(PolicyResolveErrorV0::Serialization),
            EnvelopeInstrumentV0::PayloadUnresolved,
        ];
        for i in all {
            let code = i.as_str();
            assert!(
                code.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                "{code}"
            );
        }
        assert!(!STORAGE_FAILURE_BANNER.to_lowercase().contains("incomplete"));
    }
}
