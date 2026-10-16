// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `RuntimeEnvelopeV0`: one run and one row per formed raw candidate
//! (Routing contract 50747a52, sections 3 and 4.2).
//!
//! Row bytes, digests, validation, summary, projection and pure verification
//! are the landed `crate::receipt` functions, called unchanged.

use super::candidate::{RawCandidateEnumeratorV0, StreamCandidateAssemblerV0, StreamTouchV0};
use super::capture::{bind_detection, request_ref_from_header, RequestCaptureV0};
use super::interfaces::*;
use super::policy::{policy_enforced, CompleteConstraintDecisionV0, ConfigPolicyProviderV0};
use super::sink::{sink_precedence, StoreReceiptSinkV0};
use super::store::{ProtectedEvidenceStoreV0, TerminalOutcomeV0};
use crate::config::Config;
use crate::message::Message;
use crate::receipt::*;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::Arc;

const ARGUMENTS_MEDIA_TYPE: &str = "application/json";
const CONTENT_MEDIA_TYPE: &str = "text/plain; charset=utf-8";

/// Construction inputs. The executor binding defaults to `None`.
pub struct EnvelopePartsV0 {
    /// Live resolved configuration (read-only).
    pub config: Arc<Config>,
    /// Private protected evidence store.
    pub store: Arc<ProtectedEvidenceStoreV0>,
    /// Injected build identity.
    pub identity: Arc<dyn BuildIdentityProviderV0>,
    /// Construction-time executor binding; `None` is the production default.
    pub executor: Option<ExecutorBindingV0>,
    /// Sink override (fixtures); defaults to the store sink.
    pub sink: Option<Arc<dyn ReceiptSinkV0>>,
    /// Optional consumer of live results.
    pub results: Option<Arc<dyn EnvelopeResultSinkV0>>,
}

impl EnvelopePartsV0 {
    /// Parts with no executor, the store sink and no result consumer.
    pub fn new(
        config: Arc<Config>,
        store: Arc<ProtectedEvidenceStoreV0>,
        identity: Arc<dyn BuildIdentityProviderV0>,
    ) -> Self {
        Self {
            config,
            store,
            identity,
            executor: None,
            sink: None,
            results: None,
        }
    }
}

/// Section 4.2 `RuntimeEnvelopeV0`.
pub struct RuntimeEnvelopeV0 {
    policy: Box<dyn ResolvedPolicyProviderV0>,
    identity: Arc<dyn BuildIdentityProviderV0>,
    decision: CompleteConstraintDecisionV0,
    config: Arc<Config>,
    store: Arc<ProtectedEvidenceStoreV0>,
    sink: Arc<dyn ReceiptSinkV0>,
    executor: Option<ExecutorBindingV0>,
    results: Option<Arc<dyn EnvelopeResultSinkV0>>,
}

/// Buffered delivery decision applied before serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferedDeliveryV0 {
    /// The whole response must be replaced by an error.
    pub suppressed: bool,
}

/// Streaming per-call release decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallDispositionV0 {
    /// Release the call's buffered lines.
    Release,
    /// Withhold the call; the closed reason code is the only text exposed.
    Withhold(&'static str),
}

#[derive(Debug, Clone)]
struct Parsed {
    tool_name: Option<String>,
    tool_call_id: Option<String>,
    parse_error: Option<ParseError>,
    arguments: Option<Value>,
    canonical: Option<Vec<u8>>,
    wire_form: ArgumentsWireForm,
    native: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct Prepared {
    candidate_ref: CandidateRefV0,
    parsed: Parsed,
    proposal: Proposal,
    proposal_bytes: Vec<u8>,
    proposal_digest: String,
    decision: DecisionOutcome,
    evaluated: bool,
    reason: DecisionReason,
    failed: Vec<FailedConstraint>,
    inputs: Vec<InputConsumed>,
    enforced: bool,
}

fn reason_str(reason: DecisionReason) -> &'static str {
    match reason {
        DecisionReason::Allowed => "allowed",
        DecisionReason::ToolNotAllowed => "tool_not_allowed",
        DecisionReason::ArgumentMissing => "argument_missing",
        DecisionReason::TypeCheck => "type_check",
        DecisionReason::StartsWith => "starts_with",
        DecisionReason::NotContains => "not_contains",
        DecisionReason::Matches => "matches",
        DecisionReason::OneOf => "one_of",
        DecisionReason::MaxLength => "max_length",
        DecisionReason::Min => "min",
        DecisionReason::Max => "max",
        DecisionReason::UrlHost => "url_host",
        DecisionReason::Multiple => "multiple",
        DecisionReason::ProposalUnparsable => "proposal_unparsable",
        DecisionReason::ProposalShapeInvalid => "proposal_shape_invalid",
        DecisionReason::PolicyLayerOff => "policy_layer_off",
        DecisionReason::DetectionGate => "detection_gate",
    }
}

fn parse_candidate(c: &ProposalCandidateV0) -> Parsed {
    let (wire_form, native) = match &c.arguments {
        RawArgumentsV0::String(s) => (ArgumentsWireForm::String, Some(s.as_bytes().to_vec())),
        RawArgumentsV0::Object(_) => (ArgumentsWireForm::Object, None),
        RawArgumentsV0::Null => (ArgumentsWireForm::NullValue, None),
        RawArgumentsV0::Absent => (ArgumentsWireForm::Absent, None),
    };
    let parsed_arguments = match &c.arguments {
        RawArgumentsV0::String(s) if s.is_empty() => Ok(Value::Object(Default::default())),
        RawArgumentsV0::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v) if v.is_object() => Ok(v),
            Ok(_) => Err(ParseError::NotAnObject),
            Err(_) => Err(ParseError::ParseError),
        },
        RawArgumentsV0::Object(v) if v.is_object() => Ok(v.clone()),
        RawArgumentsV0::Object(_) => Err(ParseError::NotAnObject),
        RawArgumentsV0::Null | RawArgumentsV0::Absent => Ok(Value::Object(Default::default())),
    };
    let parse_error = match c.failure {
        Some(StreamFailureV0::BufferOverflow) => Some(ParseError::BufferOverflow),
        Some(StreamFailureV0::DeltaTimeout) => Some(ParseError::DeltaTimeout),
        Some(StreamFailureV0::StreamAborted) => Some(ParseError::StreamAborted),
        None if c.tool_name.is_none() => Some(ParseError::MissingName),
        None if c.tool_call_id.is_none() => Some(ParseError::MissingId),
        None if c.malformed => Some(ParseError::ParseError),
        None => parsed_arguments.as_ref().err().copied(),
    };
    let arguments = parse_error
        .is_none()
        .then(|| parsed_arguments.ok())
        .flatten();
    let canonical = arguments
        .as_ref()
        .map(|v| canonical_argument_bytes(v).expect("parsed value serializes"));
    Parsed {
        tool_name: c.tool_name.clone(),
        tool_call_id: c.tool_call_id.clone(),
        parse_error,
        arguments,
        canonical,
        wire_form,
        native,
    }
}

// Draft 12 section 6.1: exactly `path` and `content`, both strings.
fn expected_effect(arguments: &Value) -> PresentAbsentErrored<ExpectedEffect, ExpectedEffectError> {
    let Some(object) = arguments.as_object() else {
        return PresentAbsentErrored::Errored(ExpectedEffectError::NotAnObject);
    };
    let (Some(path), Some(content)) = (object.get("path"), object.get("content")) else {
        return PresentAbsentErrored::Errored(if object.get("path").is_none() {
            ExpectedEffectError::MissingPath
        } else {
            ExpectedEffectError::MissingContent
        });
    };
    let (Some(path), Some(content)) = (path.as_str(), content.as_str()) else {
        return PresentAbsentErrored::Errored(ExpectedEffectError::WrongType);
    };
    if object.len() != 2 {
        return PresentAbsentErrored::Errored(ExpectedEffectError::UnsupportedArgument);
    }
    PresentAbsentErrored::Present(ExpectedEffect {
        target_path: path.to_owned(),
        content_digest: payload_digest(content.as_bytes()),
        content_len_bytes: content.len() as u64,
        content_media_type: CONTENT_MEDIA_TYPE.to_owned(),
    })
}

fn stage_bytes(record: ReceiptRecord<'_>) -> (Vec<u8>, String) {
    let bytes = record_bytes(record).expect("closed receipt record serializes");
    let digest = record_digest(&bytes, RecordDomain::Stage);
    (bytes, digest)
}

impl RuntimeEnvelopeV0 {
    /// Build an envelope from explicit parts.
    pub fn new(parts: EnvelopePartsV0) -> Self {
        let sink = parts
            .sink
            .unwrap_or_else(|| Arc::new(StoreReceiptSinkV0::new(parts.store.clone())));
        Self {
            policy: Box::new(ConfigPolicyProviderV0::new(parts.config.clone())),
            identity: parts.identity,
            decision: CompleteConstraintDecisionV0::new(),
            config: parts.config,
            store: parts.store,
            sink,
            executor: parts.executor,
            results: parts.results,
        }
    }

    /// The bound protected store.
    pub fn store(&self) -> &Arc<ProtectedEvidenceStoreV0> {
        &self.store
    }

    /// Whether an executor was bound at construction.
    pub fn executor_bound(&self) -> bool {
        self.executor.is_some()
    }

    /// Section 3 step 1: capture before L0 mutation; calls L1 at most once.
    pub fn capture(
        &self,
        messages_before_l0: &[Message],
        request_id_header: Option<&[u8]>,
        generated_request_id: &str,
        model: &str,
    ) -> RequestCaptureV0 {
        RequestCaptureV0 {
            request_ref: request_ref_from_header(request_id_header, generated_request_id),
            model: model.to_owned(),
            detection: bind_detection(messages_before_l0, self.config.policy.layers.l1.as_ref()),
        }
    }

    fn prepare(
        &self,
        policy: &ResolvedPolicyV0,
        candidate: ProposalCandidateV0,
        candidate_ref: CandidateRefV0,
    ) -> Prepared {
        let parsed = parse_candidate(&candidate);
        let parsed_ok = parsed.parse_error.is_none();
        let effect_kind = if parsed_ok && parsed.tool_name.as_deref() == Some("write_file") {
            EffectKind::WriteFile
        } else {
            EffectKind::Unsupported
        };
        let expected = match (effect_kind, &parsed.arguments) {
            (EffectKind::WriteFile, Some(arguments)) => expected_effect(arguments),
            _ => PresentAbsentErrored::AbsentByContract,
        };
        let shape_errored = matches!(expected, PresentAbsentErrored::Errored(_));
        let proposal = Proposal {
            schema_version: ReceiptSchema::V0,
            stage: ProposalStage::Proposal,
            provider: candidate.provider,
            index: candidate_ref.index,
            tool_name: match &parsed.tool_name {
                Some(name) => PresentUnavailable::Present(name.clone()),
                None => PresentUnavailable::Unavailable(ToolNameError::MissingName),
            },
            parse_status: if parsed_ok {
                ParseStatus::Parsed
            } else {
                ParseStatus::Errored
            },
            parse_error_code: parsed.parse_error,
            arguments_digest: match &parsed.canonical {
                Some(bytes) => PresentErrored::Present(arguments_digest(bytes)),
                None => PresentErrored::Errored(ArgumentsError::ParseError),
            },
            arguments_len_bytes: match &parsed.canonical {
                Some(bytes) => PresentErrored::Present(bytes.len() as u64),
                None => PresentErrored::Errored(ArgumentsError::ParseError),
            },
            effect_kind,
            expected_effect: expected,
        };
        let (proposal_bytes, proposal_digest) = stage_bytes(ReceiptRecord::Proposal(&proposal));
        let enforced = policy_enforced(&policy.policy);
        let (decision, evaluated, reason, failed, inputs) = if !parsed_ok {
            (
                DecisionOutcome::Block,
                false,
                DecisionReason::ProposalUnparsable,
                vec![],
                vec![InputConsumed::Proposal],
            )
        } else if shape_errored {
            (
                DecisionOutcome::Block,
                false,
                DecisionReason::ProposalShapeInvalid,
                vec![],
                vec![InputConsumed::Proposal],
            )
        } else if !enforced {
            (
                DecisionOutcome::Allow,
                false,
                DecisionReason::PolicyLayerOff,
                vec![],
                vec![],
            )
        } else {
            let tool = parsed.tool_name.as_deref().expect("parsed name");
            let arguments = parsed.arguments.as_ref().expect("parsed arguments");
            let out = self.decision.decide(&policy.policy, tool, arguments);
            (
                out.decision,
                true,
                out.reason_code,
                out.failed_constraints,
                vec![InputConsumed::Policy, InputConsumed::Proposal],
            )
        };
        Prepared {
            candidate_ref,
            parsed,
            proposal,
            proposal_bytes,
            proposal_digest,
            decision,
            evaluated,
            reason,
            failed,
            inputs,
            enforced,
        }
    }

    fn enforcement(p: &Prepared, suppressed: bool) -> Enforcement {
        if !p.enforced {
            if p.evaluated || p.reason == DecisionReason::PolicyLayerOff {
                Enforcement::NotEvaluated
            } else {
                // A parse- or shape-errored block computed while the layer is
                // off is not applied to the response: computed, not applied.
                Enforcement::Shadow
            }
        } else if suppressed && p.decision == DecisionOutcome::Allow && p.evaluated {
            Enforcement::SuppressedByResponse
        } else {
            Enforcement::Enforced
        }
    }

    fn payload_ref(
        &self,
        run: &RunIdV0,
        bytes: &[u8],
        storage: &mut Option<StoreFailureV0>,
        refs: &mut StableRefsV0,
    ) -> PayloadRef {
        if storage.is_none() {
            match self.store.put(run, PayloadStore::ReceiptEvidence, bytes) {
                Ok(stored) => {
                    if !refs.payloads.contains(&stored.path) {
                        refs.payloads.push(stored.path);
                    }
                }
                Err(e) => *storage = Some(e),
            }
        }
        PayloadRef {
            digest: payload_digest(bytes),
            len_bytes: bytes.len() as u64,
            media_type: ARGUMENTS_MEDIA_TYPE.to_owned(),
            store: PayloadStore::ReceiptEvidence,
        }
    }

    // Section 3 steps 3 and 6-10 for one formed candidate.
    fn finish(
        &self,
        capture: &RequestCaptureV0,
        policy: &ResolvedPolicyV0,
        p: Prepared,
        suppressed: bool,
        instruments: &mut Vec<EnvelopeInstrumentV0>,
    ) -> RunResultV0 {
        let run = RunIdV0::generate();
        let mut storage: Option<StoreFailureV0> = None;
        let mut refs = StableRefsV0::default();
        if let Err(e) = self
            .store
            .create_run(&run, &capture.request_ref, p.candidate_ref)
        {
            storage = Some(e);
        }
        let native_arguments = match (&p.parsed.wire_form, &p.parsed.native) {
            (ArgumentsWireForm::String, Some(bytes)) => {
                PresentUnavailable::Present(self.payload_ref(&run, bytes, &mut storage, &mut refs))
            }
            (ArgumentsWireForm::Object, _) => {
                PresentUnavailable::Unavailable(NativeArgumentsError::NoStringForm)
            }
            (ArgumentsWireForm::NullValue, _) => {
                PresentUnavailable::Unavailable(NativeArgumentsError::NullArguments)
            }
            _ => PresentUnavailable::Unavailable(NativeArgumentsError::AbsentArguments),
        };
        let arguments_payload = match &p.parsed.canonical {
            Some(bytes) => {
                PresentUnavailable::Present(self.payload_ref(&run, bytes, &mut storage, &mut refs))
            }
            None => PresentUnavailable::Unavailable(ArgumentsPayloadError::ParseError),
        };
        let detection_payload = match (
            &capture.detection.detection,
            &capture.detection.record_bytes,
        ) {
            (PresentUnavailableAbsent::Present(_), Some(bytes)) => {
                PresentUnavailableAbsent::Present(self.payload_ref(
                    &run,
                    bytes,
                    &mut storage,
                    &mut refs,
                ))
            }
            _ => PresentUnavailableAbsent::AbsentByContract,
        };
        let context = ProposalContext {
            schema_version: ReceiptSchema::V0,
            stage: ContextStage::ProposalContext,
            proposal_ref: p.proposal_digest.clone(),
            request_ref: capture.request_ref.clone(),
            choice_index: p.candidate_ref.choice_index,
            model: capture.model.clone(),
            tool_call_id: match &p.parsed.tool_call_id {
                Some(id) => PresentUnavailable::Present(id.clone()),
                None => PresentUnavailable::Unavailable(ToolCallIdError::MissingId),
            },
            arguments_wire_form: p.parsed.wire_form,
            native_arguments,
            arguments_payload,
            detection_payload,
        };
        let (context_bytes, context_digest) = stage_bytes(ReceiptRecord::ProposalContext(&context));
        let enforcement = Self::enforcement(&p, suppressed);
        let decision = Decision {
            schema_version: ReceiptSchema::V0,
            stage: DecisionStage::Decision,
            proposal_ref: p.proposal_digest.clone(),
            policy: PolicyIdentity {
                contract_hash: policy.contract_hash.clone(),
                policy_digest: policy.digest.clone(),
                policy_version: policy.policy.policy_version.clone(),
            },
            code: self.identity.identity(),
            detection: capture.detection.detection.clone(),
            detection_binding_rule: capture.detection.rule.clone(),
            enforcement,
            decision: p.decision,
            evaluated_policy: p.evaluated,
            reason_code: p.reason,
            failed_constraints: p.failed.clone(),
            inputs_consumed: p.inputs.clone(),
        };
        let (decision_bytes, decision_digest) = stage_bytes(ReceiptRecord::Decision(&decision));

        // Step 8: synchronous pre-attempt persistence of exactly three records.
        let sink_results = self
            .sink
            .persist_pre_attempt(&run, [&p.proposal_bytes, &context_bytes, &decision_bytes]);
        let [proposal_sink, context_sink, decision_sink] = sink_results;
        let sink = Sink {
            proposal: proposal_sink,
            context: context_sink,
            decision: decision_sink,
        };
        let expected_refs = [
            p.proposal_digest.as_str(),
            context_digest.as_str(),
            decision_digest.as_str(),
        ];
        let sink_code = sink_precedence(&sink, expected_refs);
        if storage.is_none() {
            storage = match sink_code {
                Some(AttemptError::SinkUnavailable) => Some(StoreFailureV0::Io),
                Some(AttemptError::SinkRejected) => Some(StoreFailureV0::Collision),
                Some(_) => Some(StoreFailureV0::DigestMismatch),
                None => None,
            };
        }
        for path in [&p.proposal_digest, &context_digest, &decision_digest] {
            refs.records.push(super::store::layout::record_path(path));
        }

        // Step 7: draft-12 L3 eligibility after the construction-time binding.
        let eligible = p.decision == DecisionOutcome::Allow
            && p.evaluated
            && enforcement == Enforcement::Enforced
            && p.proposal.effect_kind == EffectKind::WriteFile;
        let absent = if p.decision == DecisionOutcome::Block {
            Some(AttemptAbsentReason::Blocked)
        } else if p.proposal.effect_kind == EffectKind::Unsupported {
            Some(AttemptAbsentReason::Unsupported)
        } else if enforcement != Enforcement::Enforced {
            Some(AttemptAbsentReason::NotEnforced)
        } else if !eligible || self.executor.is_none() {
            Some(AttemptAbsentReason::NoExecutor)
        } else {
            None
        };
        let mut later: Vec<(RecordDomain, StageKindV0, Vec<u8>)> = Vec::new();
        let (attempt_slot, observation_slot, verification_slot) = match (absent, &self.executor) {
            (None, Some(binding)) => {
                let PresentAbsentErrored::Present(expected) = &p.proposal.expected_effect else {
                    unreachable!("eligible write_file has a present expected effect");
                };
                let content = p
                    .parsed
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("content"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let attempt = binding.executor.execute(&ExecutorRequestV0 {
                    proposal: &p.proposal,
                    decision: &decision,
                    decision_ref: &decision_digest,
                    expected_refs,
                    action_id: &binding.action_id,
                    expected_effect: expected,
                    content: content.as_bytes(),
                    root: &binding.root,
                    sink: &sink,
                });
                let consistent = attempt.decision_ref == decision_digest
                    && attempt.sink == sink
                    && attempt.action_id == binding.action_id.as_str()
                    && sink_code.is_none_or(|code| {
                        attempt.outcome == AttemptOutcome::Rejected
                            && attempt.error_code == Some(code)
                    });
                if !consistent {
                    instruments.push(EnvelopeInstrumentV0::ExecutorContractViolation);
                }
                let (attempt_bytes, attempt_digest) = stage_bytes(ReceiptRecord::Attempt(&attempt));
                later.push((RecordDomain::Stage, StageKindV0::Attempt, attempt_bytes));
                if attempt.outcome == AttemptOutcome::Rejected {
                    (
                        Slot::Present {
                            digest: attempt_digest,
                            record: attempt,
                        },
                        Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected),
                        Slot::AbsentByContract(PostAttemptAbsentReason::AttemptRejected),
                    )
                } else {
                    // Step 9: independent observer, then the landed pure verifier.
                    let observation = match binding.observer.observe(
                        &p.proposal,
                        &attempt,
                        &attempt_digest,
                        &binding.root,
                    ) {
                        Ok(o) => o,
                        Err(ObserverFailureV0::ObserverFailed) => {
                            instruments.push(EnvelopeInstrumentV0::ObserverFailed);
                            Observation {
                                schema_version: ReceiptSchema::V0,
                                stage: ObservationStage::Observation,
                                attempt_ref: attempt_digest.clone(),
                                observer: binding.observer.identity(),
                                target_ref: PresentUnavailable::Unavailable(
                                    ObservationTargetError::PathUnresolvable,
                                ),
                                observed: PresentUnavailableErrored::Errored(
                                    ObservedError::ObserverFailed,
                                ),
                                footprint: PresentUnavailable::Unavailable(
                                    FootprintError::NotConfigured,
                                ),
                            }
                        }
                    };
                    let (observation_bytes, observation_digest) =
                        stage_bytes(ReceiptRecord::Observation(&observation));
                    later.push((
                        RecordDomain::Stage,
                        StageKindV0::Observation,
                        observation_bytes,
                    ));
                    let method = verify_expected_effect(&p.proposal, &attempt, &observation)
                        .expect("non-rejected attempt with present expected effect");
                    let verification = Verification {
                        schema_version: ReceiptSchema::V0,
                        stage: VerificationStage::Verification,
                        observation_ref: observation_digest.clone(),
                        method: VerificationMethod::ExpectedVsObservedV0,
                        outcome: method.outcome,
                        comparison: method.comparison,
                        reason_code: method.reason_code,
                    };
                    let (verification_bytes, verification_digest) =
                        stage_bytes(ReceiptRecord::Verification(&verification));
                    later.push((
                        RecordDomain::Stage,
                        StageKindV0::Verification,
                        verification_bytes,
                    ));
                    (
                        Slot::Present {
                            digest: attempt_digest,
                            record: attempt,
                        },
                        Slot::Present {
                            digest: observation_digest,
                            record: observation,
                        },
                        Slot::Present {
                            digest: verification_digest,
                            record: verification,
                        },
                    )
                }
            }
            (reason, _) => (
                Slot::AbsentByContract(reason.unwrap_or(AttemptAbsentReason::NoExecutor)),
                Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent),
                Slot::AbsentByContract(PostAttemptAbsentReason::AttemptAbsent),
            ),
        };
        let action_id = match &attempt_slot {
            Slot::Present { record, .. } => Some(record.action_id.clone()),
            Slot::AbsentByContract(_) => None,
        };
        let mut row = Row {
            schema_version: ReceiptSchema::V0,
            proposal_id: p.proposal_digest.clone(),
            action_id,
            stages: Stages {
                proposal: PresentSlot {
                    availability: PresentAvailability::Present,
                    reason: (),
                    digest: p.proposal_digest.clone(),
                    record: p.proposal.clone(),
                },
                decision: PresentSlot {
                    availability: PresentAvailability::Present,
                    reason: (),
                    digest: decision_digest.clone(),
                    record: decision,
                },
                attempt: attempt_slot,
                observation: observation_slot,
                verification: verification_slot,
            },
            proposal_context: PresentSlot {
                availability: PresentAvailability::Present,
                reason: (),
                digest: context_digest,
                record: context,
            },
            summary: Summary::Incomplete,
            float_policy: FloatPolicy::BitExact,
            crosswalk: Crosswalk::V0,
        };
        row.summary = receipt_summary(&row);
        let row_bytes = record_bytes(ReceiptRecord::Row(&row)).expect("closed row serializes");
        let row_digest = record_digest(&row_bytes, RecordDomain::Row);
        let validation = validate_receipt(&row_bytes, self.store.as_ref());
        match validation {
            ValidationResult::Valid => {}
            ValidationResult::Rejected(code) => {
                instruments.push(EnvelopeInstrumentV0::RowValidationRejected(code))
            }
            ValidationResult::ValidUnresolved => {
                instruments.push(EnvelopeInstrumentV0::PayloadUnresolved)
            }
        }
        let projection_bytes = projection_bytes(&receipt_projection(&row));

        // Step 10: terminal records, row, digest and projection, then the terminal.
        let batch = TerminalBatchV0 {
            later_records: later
                .iter()
                .map(|(d, k, b)| (*d, *k, b.as_slice()))
                .collect(),
            row_bytes: &row_bytes,
            row_digest: &row_digest,
            projection_bytes: &projection_bytes,
        };
        match self.sink.persist_terminal(&run, &batch) {
            Ok(paths) => {
                let mut paths = paths.into_iter();
                for _ in 0..later.len() {
                    if let Some(path) = paths.next() {
                        refs.records.push(path);
                    }
                }
                refs.row = paths.next();
                refs.projection = paths.next();
            }
            Err(e) => {
                storage.get_or_insert(e);
            }
        }
        let valid = validation == ValidationResult::Valid;
        let completion = match (storage, valid) {
            (None, true) => {
                match self
                    .store
                    .publish_initial(&run, TerminalOutcomeV0::Complete, &row_digest)
                {
                    Ok(_) => RuntimeCompletionV0::Complete,
                    Err(e) => {
                        instruments.push(EnvelopeInstrumentV0::StorageIncomplete(e));
                        RuntimeCompletionV0::StorageIncomplete
                    }
                }
            }
            (failure, _) => {
                if let Some(e) = failure {
                    instruments.push(EnvelopeInstrumentV0::StorageIncomplete(e));
                }
                // Best effort: record the permanent incomplete outcome.
                let _ = self.store.publish_initial(
                    &run,
                    TerminalOutcomeV0::StorageIncomplete,
                    &row_digest,
                );
                RuntimeCompletionV0::StorageIncomplete
            }
        };
        tracing::debug!(
            schema = "parapet-effect-receipt/v0",
            row_digest = %row_digest,
            row_len_bytes = row_bytes.len(),
            summary = ?row.summary,
            completion = ?completion,
            "effect receipt run settled"
        );
        RunResultV0 {
            run_id: run,
            candidate_ref: p.candidate_ref,
            banner: (completion == RuntimeCompletionV0::StorageIncomplete)
                .then_some(STORAGE_FAILURE_BANNER),
            row,
            row_bytes,
            row_digest,
            projection_bytes,
            validation,
            refs,
            completion,
        }
    }

    fn assemble(
        delivery: DeliveryOutcomeV0,
        mapping: Vec<MappingEntryV0>,
        runs: Vec<RunResultV0>,
        mut instruments: Vec<EnvelopeInstrumentV0>,
    ) -> EnvelopeResponseResultV0 {
        let formed: Vec<_> = mapping.iter().filter_map(|m| m.candidate_ref).collect();
        let unique: BTreeSet<_> = formed.iter().map(CandidateRefV0::key).collect();
        if unique.len() != formed.len() {
            instruments.push(EnvelopeInstrumentV0::ProposalCoordinateDuplicate);
        }
        if formed.len() != runs.len() {
            instruments.push(EnvelopeInstrumentV0::ProposalCountMismatch);
        }
        let complete = runs
            .iter()
            .all(|r| r.completion == RuntimeCompletionV0::Complete)
            && !instruments
                .iter()
                .any(|i| matches!(i, EnvelopeInstrumentV0::ProposalCountMismatch));
        EnvelopeResponseResultV0 {
            delivery,
            mapping,
            runs,
            instruments,
            completion: if complete {
                RuntimeCompletionV0::Complete
            } else {
                RuntimeCompletionV0::StorageIncomplete
            },
        }
    }

    fn unformable_entry(
        c: &ProposalCandidateV0,
        instrument: EnvelopeInstrumentV0,
    ) -> MappingEntryV0 {
        MappingEntryV0 {
            ordinal: c.ordinal,
            candidate_ref: None,
            run_id: None,
            row_digest: None,
            instrument: Some(instrument),
        }
    }

    fn deliver(&self, result: &EnvelopeResponseResultV0) {
        if let Some(results) = &self.results {
            results.accept(result.clone());
        }
    }

    /// Sections 3 steps 2-10 over enumerated candidates of one fully buffered
    /// response; whole-response suppression is decided before any run settles.
    pub fn process_candidates(
        &self,
        capture: &RequestCaptureV0,
        candidates: Vec<ProposalCandidateV0>,
    ) -> (
        EnvelopeResponseResultV0,
        BufferedDeliveryV0,
        Vec<(ProposalCandidateV0, Option<&'static str>)>,
    ) {
        let mut instruments = Vec::new();
        let policy = match self.policy.resolve() {
            Ok(policy) => policy,
            Err(e) => {
                let instrument = EnvelopeInstrumentV0::PolicyUnresolvable(e);
                let mapping = candidates
                    .iter()
                    .map(|c| Self::unformable_entry(c, instrument))
                    .collect();
                let blocked = candidates
                    .into_iter()
                    .map(|c| (c, Some("policy_unresolvable")))
                    .collect();
                let result = Self::assemble(
                    DeliveryOutcomeV0::SuppressedWholeResponse,
                    mapping,
                    vec![],
                    vec![instrument],
                );
                return (result, BufferedDeliveryV0 { suppressed: true }, blocked);
            }
        };
        let enforced = policy_enforced(&policy.policy);
        let mut prepared = Vec::new();
        let mut withheld = Vec::new();
        for c in candidates {
            match c.coordinate {
                Ok(candidate_ref) => {
                    let p = self.prepare(&policy, c.clone(), candidate_ref);
                    let withhold = (enforced && p.decision == DecisionOutcome::Block)
                        .then(|| reason_str(p.reason));
                    withheld.push((c, withhold));
                    prepared.push(Some(p));
                }
                Err(_) => {
                    instruments.push(EnvelopeInstrumentV0::RowCoordinateUnformable);
                    withheld.push((c, enforced.then_some("row_coordinate_unformable")));
                    prepared.push(None);
                }
            }
        }
        let any_withheld = withheld.iter().any(|(_, w)| w.is_some());
        let suppressed =
            any_withheld && policy.policy.l3_outbound.block_action_effective == BlockAction::Error;
        let delivery = if suppressed {
            DeliveryOutcomeV0::SuppressedWholeResponse
        } else if any_withheld {
            DeliveryOutcomeV0::Rewritten
        } else {
            DeliveryOutcomeV0::Delivered
        };
        let mut mapping = Vec::new();
        let mut runs = Vec::new();
        for ((c, _), p) in withheld.iter().zip(prepared) {
            match p {
                Some(p) => {
                    let run = self.finish(capture, &policy, p, suppressed, &mut instruments);
                    mapping.push(MappingEntryV0 {
                        ordinal: c.ordinal,
                        candidate_ref: Some(run.candidate_ref),
                        run_id: Some(run.run_id.clone()),
                        row_digest: Some(run.row_digest.clone()),
                        instrument: None,
                    });
                    runs.push(run);
                }
                None => mapping.push(Self::unformable_entry(
                    c,
                    EnvelopeInstrumentV0::RowCoordinateUnformable,
                )),
            }
        }
        let result = Self::assemble(delivery, mapping, runs, instruments);
        (result, BufferedDeliveryV0 { suppressed }, withheld)
    }

    /// Buffered integration: enumerate raw candidates from the parsed body,
    /// settle every run, and apply delivery to the body before serialization.
    pub fn process_buffered(
        &self,
        capture: &RequestCaptureV0,
        provider: Provider,
        body: &mut Value,
    ) -> (EnvelopeResponseResultV0, BufferedDeliveryV0) {
        use super::interfaces::ProposalCandidateEnumeratorV0;
        let candidates = RawCandidateEnumeratorV0.enumerate_buffered(provider, body);
        let (result, delivery, withheld) = self.process_candidates(capture, candidates);
        if !delivery.suppressed {
            rewrite_buffered(provider, body, &withheld);
        }
        self.deliver(&result);
        (result, delivery)
    }

    /// Streaming integration: a per-response session finalizing calls independently.
    pub fn stream_session(
        self: &Arc<Self>,
        capture: RequestCaptureV0,
        provider: Provider,
    ) -> StreamEnvelopeSessionV0 {
        StreamEnvelopeSessionV0 {
            policy: self.policy.resolve(),
            envelope: self.clone(),
            capture,
            assembler: StreamCandidateAssemblerV0::new(provider),
            mapping: Vec::new(),
            runs: Vec::new(),
            instruments: Vec::new(),
        }
    }
}

// Buffered rewrite: remove withheld calls by wire location and add closed refusal text.
fn rewrite_buffered(
    provider: Provider,
    body: &mut Value,
    withheld: &[(ProposalCandidateV0, Option<&'static str>)],
) {
    let refusal = |codes: &[&str]| {
        codes
            .iter()
            .map(|c| format!("tool call blocked by policy ({c})"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    match provider {
        Provider::Openai => {
            let Some(choices) = body.get_mut("choices").and_then(Value::as_array_mut) else {
                return;
            };
            for (p, choice) in choices.iter_mut().enumerate() {
                let codes: Vec<(u32, &str)> = withheld
                    .iter()
                    .filter(|(c, w)| w.is_some() && c.location.0 == p as u32)
                    .map(|(c, w)| (c.location.1, w.expect("withheld")))
                    .collect();
                if codes.is_empty() {
                    continue;
                }
                let Some(message) = choice.get_mut("message") else {
                    continue;
                };
                if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    let mut i = 0u32;
                    calls.retain(|_| {
                        let keep = !codes.iter().any(|(pos, _)| *pos == i);
                        i += 1;
                        keep
                    });
                }
                let text = refusal(&codes.iter().map(|(_, c)| *c).collect::<Vec<_>>());
                match message.get_mut("content") {
                    Some(Value::String(s)) if s.is_empty() => *s = text,
                    Some(Value::String(s)) => {
                        s.push('\n');
                        s.push_str(&text);
                    }
                    _ => message["content"] = Value::String(text),
                }
            }
        }
        Provider::Anthropic => {
            let Some(blocks) = body.get_mut("content").and_then(Value::as_array_mut) else {
                return;
            };
            let codes: Vec<(u32, &str)> = withheld
                .iter()
                .filter_map(|(c, w)| w.map(|w| (c.location.0, w)))
                .collect();
            if codes.is_empty() {
                return;
            }
            let mut i = 0u32;
            blocks.retain(|_| {
                let keep = !codes.iter().any(|(pos, _)| *pos == i);
                i += 1;
                keep
            });
            let text = refusal(&codes.iter().map(|(_, c)| *c).collect::<Vec<_>>());
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
    }
}

/// One streaming response: each call is finalized and settled independently.
pub struct StreamEnvelopeSessionV0 {
    envelope: Arc<RuntimeEnvelopeV0>,
    policy: Result<ResolvedPolicyV0, PolicyResolveErrorV0>,
    capture: RequestCaptureV0,
    assembler: StreamCandidateAssemblerV0,
    mapping: Vec<MappingEntryV0>,
    runs: Vec<RunResultV0>,
    instruments: Vec<EnvelopeInstrumentV0>,
}

/// Result of one streaming line.
#[derive(Debug, Default)]
pub struct StreamStepV0 {
    /// Calls this line carries bytes for.
    pub touches: Vec<StreamTouchV0>,
    /// Calls finalized by this line with their release decision.
    pub decided: Vec<(StreamTouchV0, CallDispositionV0)>,
    /// A buffer bound was exceeded; the stream aborts after this line.
    pub overflow: bool,
}

impl StreamEnvelopeSessionV0 {
    fn settle(
        &mut self,
        touch: StreamTouchV0,
        c: ProposalCandidateV0,
    ) -> (StreamTouchV0, CallDispositionV0) {
        let policy = match &self.policy {
            Ok(p) => p,
            Err(e) => {
                let instrument = EnvelopeInstrumentV0::PolicyUnresolvable(*e);
                self.instruments.push(instrument);
                self.mapping
                    .push(RuntimeEnvelopeV0::unformable_entry(&c, instrument));
                return (touch, CallDispositionV0::Withhold("policy_unresolvable"));
            }
        };
        let enforced = policy_enforced(&policy.policy);
        let Ok(candidate_ref) = c.coordinate else {
            self.instruments
                .push(EnvelopeInstrumentV0::RowCoordinateUnformable);
            self.mapping.push(RuntimeEnvelopeV0::unformable_entry(
                &c,
                EnvelopeInstrumentV0::RowCoordinateUnformable,
            ));
            let disposition = if enforced {
                CallDispositionV0::Withhold("row_coordinate_unformable")
            } else {
                CallDispositionV0::Release
            };
            return (touch, disposition);
        };
        let ordinal = c.ordinal;
        let prepared = self.envelope.prepare(policy, c, candidate_ref);
        let disposition = if enforced && prepared.decision == DecisionOutcome::Block {
            CallDispositionV0::Withhold(reason_str(prepared.reason))
        } else {
            CallDispositionV0::Release
        };
        // Streaming is never retroactively suppressed.
        let run = self.envelope.finish(
            &self.capture,
            policy,
            prepared,
            false,
            &mut self.instruments,
        );
        self.mapping.push(MappingEntryV0 {
            ordinal,
            candidate_ref: Some(run.candidate_ref),
            run_id: Some(run.run_id.clone()),
            row_digest: Some(run.row_digest.clone()),
            instrument: None,
        });
        self.runs.push(run);
        (touch, disposition)
    }

    /// Ingest one SSE payload; finalized calls are settled before returning.
    pub fn ingest(&mut self, event: Option<&str>, data: &str) -> StreamStepV0 {
        let ingest = self.assembler.ingest(event, data);
        let decided = ingest
            .finalized
            .into_iter()
            .map(|(touch, c)| self.settle(touch, c))
            .collect();
        StreamStepV0 {
            touches: ingest.touches,
            decided,
            overflow: ingest.overflow,
        }
    }

    /// Provider family of this stream.
    pub fn provider(&self) -> Provider {
        self.assembler.provider()
    }

    /// Whether any call is still open.
    pub fn has_open(&self) -> bool {
        self.assembler.has_open()
    }

    /// Finalize every open call with a representable failure.
    pub fn abort(&mut self, failure: StreamFailureV0) -> Vec<(StreamTouchV0, CallDispositionV0)> {
        self.assembler
            .abort(failure)
            .into_iter()
            .map(|(touch, c)| self.settle(touch, c))
            .collect()
    }

    /// Close the stream: open calls become `stream_aborted` rows.
    pub fn finish(mut self) -> EnvelopeResponseResultV0 {
        self.abort(StreamFailureV0::StreamAborted);
        let mut mapping = std::mem::take(&mut self.mapping);
        mapping.sort_by_key(|m| m.ordinal);
        let mut runs = std::mem::take(&mut self.runs);
        let order: Vec<_> = mapping.iter().filter_map(|m| m.run_id.clone()).collect();
        runs.sort_by_key(|r| order.iter().position(|id| *id == r.run_id));
        let result = RuntimeEnvelopeV0::assemble(
            DeliveryOutcomeV0::StreamedPerCall,
            mapping,
            runs,
            std::mem::take(&mut self.instruments),
        );
        self.envelope.deliver(&result);
        result
    }
}
