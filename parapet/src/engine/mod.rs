// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Engine integration -- defined in M1.12
//
// Wires all layers together for non-streaming and streaming requests:
// - Parse request via provider adapter
// - Trust assignment
// - L0 normalization
// - L1 lightweight classifier
// - L2a data payload detection (Prompt Guard 2 + heuristics)
// - L3-inbound scanning
// - Forward to upstream provider
// - L3-outbound tool call validation + rewrite (non-streaming)
// - L5a output redaction
// - Streaming: tool call buffering + validation via StreamProcessor

use base64::Engine as _;
use flate2::read::{DeflateDecoder, GzDecoder};
use std::io::Read as _;

use crate::config::{Config, FailureMode, L2aMode, PatternAction};
use crate::constraint::{ConstraintEvaluator, DslConstraintEvaluator, ToolCallVerdict};
use crate::effect::RequestCaptureV0;
#[cfg(not(windows))]
use crate::effect::RuntimeEnvelopeV0;
use crate::layers::l1::{EnsembleL1Scanner, L1Scanner, L1Verdict, L1Block, SvmModel};
use crate::layers::l1_harness::{L1Harness, L1Model, L1Signal, threshold_outcome, OutcomeView};
use crate::layers::l2a::L2aScanner;
use crate::layers::l3_inbound::{DefaultInboundScanner, InboundScanner, InboundVerdict};
use crate::config::L4Mode;
use crate::layers::l4::{DefaultMultiTurnScanner, L4Verdict, MultiTurnScanner};
use crate::layers::l5a::{L5aScanner, OutputScanner};
use crate::message::ToolCall;
use crate::normalize::{neutralize_role_markers, normalize_messages_with_spans_and_evidence, L0Normalizer, Normalizer};
use crate::provider::{adapter_for, ProviderAdapter, ProviderType};
use crate::proxy::{ProxyError, ProxyRequest, ProxyResponse, Provider, UpstreamClient};
use crate::routing::{L0Evidence, NoopOrthogonalSensorRouter, OrthogonalSensorRouter, RoutingEvidenceContext};
use crate::stream::{
    AnthropicChunkClassifier, OpenAiChunkClassifier, StreamProcessor, ToolCallBlockMode,
    ToolCallValidator, ValidationResult,
};
use crate::signal::combiner::{DefaultVerdictCombiner, VerdictCombiner};
use crate::signal::extractor::{DefaultSignalExtractor, SignalExtractor};
use crate::signal::Signal;
use crate::trust::{RoleTrustAssigner, TrustAssigner};
use async_trait::async_trait;
use axum::body::Body;
use bytes::Bytes;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
#[cfg(test)]
use axum::http::Uri;
use futures_util::stream::{Stream, StreamExt};
use futures_util::TryStreamExt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Request context for structured logging
// ---------------------------------------------------------------------------

struct RequestContext {
    request_id: String,
    contract_hash: String,
    provider_str: &'static str,
    model: String,
}

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

/// Known LLM API hosts — the engine's own allowlist for `X-Parapet-Original-Host`.
///
/// Defense in depth: the SDK only sets the header for hosts in its `LLM_HOSTS`,
/// but the engine must not blindly trust headers from arbitrary clients (curl,
/// other SDKs). An unrecognized host would cause the engine to forward the
/// request — including auth headers — to an attacker-controlled server.
pub const KNOWN_LLM_HOSTS: &[&str] = &[
    "api.openai.com",
    "api.anthropic.com",
    "api.cerebras.ai",
    "api.groq.com",
    "generativelanguage.googleapis.com",
    "api.together.xyz",
];

/// Validated or classified `X-Parapet-Original-Host` header state.
///
/// Manual `Debug` withholds the raw header value to satisfy Z3/Z5.
#[derive(Clone, PartialEq, Eq)]
pub enum OriginalHost<'a> {
    Absent,
    Present(&'a str),
    InvalidHeader {
        rule: &'static str,
        value_count: usize,
        value_lens: Vec<usize>,
    },
}

impl std::fmt::Debug for OriginalHost<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OriginalHost::Absent => write!(f, "OriginalHost::Absent"),
            OriginalHost::Present(s) => {
                write!(f, "OriginalHost::Present(<withheld {} bytes>)", s.len())
            }
            OriginalHost::InvalidHeader {
                rule,
                value_count,
                value_lens,
            } => f
                .debug_struct("OriginalHost::InvalidHeader")
                .field("rule", rule)
                .field("value_count", value_count)
                .field("value_lens", value_lens)
                .finish(),
        }
    }
}

impl<'a> OriginalHost<'a> {
    pub fn from_headers(headers: &'a HeaderMap) -> Self {
        let values: Vec<&HeaderValue> = headers.get_all("x-parapet-original-host").iter().collect();
        match values.as_slice() {
            [] => OriginalHost::Absent,
            [single] => {
                let bytes = single.as_bytes();
                if bytes.is_empty() {
                    OriginalHost::InvalidHeader {
                        rule: "header_empty",
                        value_count: 1,
                        value_lens: vec![0],
                    }
                } else if !bytes.iter().all(|&b| (0x21..=0x7e).contains(&b)) {
                    OriginalHost::InvalidHeader {
                        rule: "header_not_visible_ascii",
                        value_count: 1,
                        value_lens: vec![bytes.len()],
                    }
                } else {
                    match std::str::from_utf8(bytes) {
                        Ok(s) => OriginalHost::Present(s),
                        Err(_) => OriginalHost::InvalidHeader {
                            rule: "header_not_visible_ascii",
                            value_count: 1,
                            value_lens: vec![bytes.len()],
                        },
                    }
                }
            }
            many => OriginalHost::InvalidHeader {
                rule: "header_duplicated",
                value_count: many.len(),
                value_lens: many.iter().map(|v| v.as_bytes().len()).collect(),
            },
        }
    }
}

/// Request-time upstream resolution or destination-binding refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamRefusal {
    HostNotAllowed,
    ConfigInvalid,
    DestinationMismatch,
}

impl std::fmt::Display for UpstreamRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpstreamRefusal::HostNotAllowed => write!(f, "parapet_upstream_host_not_allowed"),
            UpstreamRefusal::ConfigInvalid => write!(f, "parapet_upstream_config_invalid"),
            UpstreamRefusal::DestinationMismatch => {
                write!(f, "parapet_upstream_destination_mismatch")
            }
        }
    }
}

impl std::error::Error for UpstreamRefusal {}

/// Resolves the upstream base URL for a provider.
///
/// Two resolution paths:
/// - `base_url_for_host`: primary path when the SDK sends `X-Parapet-Original-Host`.
///   Supports any OpenAI-compatible provider without hardcoded knowledge.
/// - `base_url`: fallback for zero-SDK mode (no header), maps `Provider` enum to URL.
///
/// Security: `is_allowed_host` gates which hosts are accepted from the header.
pub trait UpstreamResolver: Send + Sync {
    /// Check if a host from `X-Parapet-Original-Host` is allowed.
    ///
    /// Default: matches against `KNOWN_LLM_HOSTS`. Override to extend
    /// (e.g., via env var or config).
    fn is_allowed_host(&self, host: &str) -> bool {
        KNOWN_LLM_HOSTS.contains(&host)
    }

    /// Resolve base URL from the original host header (primary path).
    ///
    /// Default: `https://{host}`. Override to intercept per-host (e.g., internal
    /// routing, staging environments, env var overrides).
    fn base_url_for_host(&self, host: &str) -> String {
        format!("https://{host}")
    }

    /// Fallback: resolve base URL from provider enum (zero-SDK mode, no header).
    fn base_url(&self, provider: Provider) -> String;

    /// Fail-closed resolution used by `build_upstream_url`.
    ///
    /// Disallowed, malformed, or duplicated `X-Parapet-Original-Host` headers
    /// return `Err(UpstreamRefusal::HostNotAllowed)` instead of falling back.
    fn resolve(
        &self,
        provider: Provider,
        original_host: OriginalHost<'_>,
    ) -> Result<String, UpstreamRefusal> {
        match original_host {
            OriginalHost::Absent => Ok(self.base_url(provider)),
            OriginalHost::Present(host) => {
                if self.is_allowed_host(host) {
                    Ok(self.base_url_for_host(host))
                } else {
                    tracing::warn!(
                        rule = "header_not_allowlisted",
                        value_count = 1usize,
                        value_lens = ?[host.len()],
                        "x-parapet-original-host refused"
                    );
                    Err(UpstreamRefusal::HostNotAllowed)
                }
            }
            OriginalHost::InvalidHeader {
                rule,
                value_count,
                value_lens,
            } => {
                tracing::warn!(
                    rule = rule,
                    value_count = value_count,
                    value_lens = ?value_lens,
                    "x-parapet-original-host refused"
                );
                Err(UpstreamRefusal::HostNotAllowed)
            }
        }
    }

    /// Optional startup configuration diagnostic when a resolver was built from an invalid env.
    fn config_error(&self) -> Option<&UpstreamConfigError> {
        None
    }
}

/// Sends HTTP requests to upstream providers.
#[async_trait]
pub trait HttpSender: Send + Sync {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;
}

/// Provides provider adapters for request parsing and error serialization.
pub trait ProviderRegistry: Send + Sync {
    fn adapter_for(&self, provider: Provider) -> Box<dyn ProviderAdapter>;
}

// ---------------------------------------------------------------------------
// Transport types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub timeout_ms: Option<u64>,
    pub stream: bool,
}

pub enum HttpBody {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send>>),
}

pub struct HttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: HttpBody,
}

#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    #[error("upstream request failed: {0}")]
    Transport(String),
    #[error("upstream request timed out: {0}")]
    Timeout(String),
}

// ---------------------------------------------------------------------------
// Startup errors
// ---------------------------------------------------------------------------

/// Error returned when `build_engine_client` cannot construct the engine.
#[derive(Debug)]
pub enum StartupError {
    /// Upstream override environment configuration is invalid or unsupported.
    UpstreamConfig(UpstreamConfigError),
    /// L2a configured in YAML but binary built without `--features l2a`.
    FeatureDisabled(String),
    /// L2a configured but model files missing or invalid.
    ModelMissing(String),
    /// Runtime envelope, store, or file-effect initialization failed.
    #[cfg(not(windows))]
    RuntimeInit(String),
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupError::UpstreamConfig(err) => write!(f, "{err}"),
            StartupError::FeatureDisabled(msg) => write!(f, "{msg}"),
            StartupError::ModelMissing(msg) => write!(f, "{msg}"),
            #[cfg(not(windows))]
            StartupError::RuntimeInit(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for StartupError {}

// ---------------------------------------------------------------------------
// Engine dependencies
// ---------------------------------------------------------------------------

pub struct EngineDeps {
    pub config: Arc<Config>,
    pub http: Arc<dyn HttpSender>,
    pub resolver: Arc<dyn UpstreamResolver>,
    pub registry: Arc<dyn ProviderRegistry>,
    pub normalizer: Arc<dyn Normalizer>,
    pub trust_assigner: Arc<dyn TrustAssigner>,
    pub l1_scanner: Option<Arc<dyn L1Scanner>>,
    pub l1_model: Option<Arc<dyn L1Model>>,
    pub l2a_scanner: Option<Arc<dyn L2aScanner>>,
    pub l2a_semaphore: Option<Arc<tokio::sync::Semaphore>>,
    pub inbound_scanner: Arc<dyn InboundScanner>,
    pub constraint_evaluator: Arc<dyn ConstraintEvaluator>,
    pub output_scanner: Arc<dyn OutputScanner>,
    pub session_store: Option<Arc<dyn crate::session::SessionStore>>,
    pub multi_turn_scanner: Option<Arc<dyn MultiTurnScanner>>,
    pub orthogonal_sensor_router: Arc<dyn OrthogonalSensorRouter>,
    pub signal_extractor: Arc<dyn SignalExtractor>,
    pub verdict_combiner: Arc<dyn VerdictCombiner>,
    /// Emit `x-parapet-l1-signals` JSON header on responses. Eval-only —
    /// production must leave this false to avoid leaking internals and
    /// hitting proxy header-size limits.
    pub emit_l1_signals: bool,
}

// ---------------------------------------------------------------------------
// EngineUpstreamClient
// ---------------------------------------------------------------------------

/// Upstream client that runs the full Parapet pipeline.
pub struct EngineUpstreamClient {
    deps: EngineDeps,
    /// Opt-in effect-receipt runtime envelope; `None` unless explicitly attached.
    #[cfg(not(windows))]
    envelope: Option<Arc<RuntimeEnvelopeV0>>,
}

impl EngineUpstreamClient {
    pub fn new_with(deps: EngineDeps) -> Self {
        Self {
            deps,
            #[cfg(not(windows))]
            envelope: None,
        }
    }

    /// Additive constructor: attach the runtime envelope (Routing contract
    /// 50747a52, section 4). `EngineDeps` and `new_with` are unchanged; the
    /// envelope's executor binding is whatever it was constructed with.
    #[cfg(not(windows))]
    pub fn new_with_envelope(deps: EngineDeps, envelope: Arc<RuntimeEnvelopeV0>) -> Self {
        Self {
            deps,
            envelope: Some(envelope),
        }
    }
}

#[cfg(not(windows))]
fn receipt_provider(provider: Provider) -> crate::receipt::Provider {
    match provider {
        Provider::OpenAi => crate::receipt::Provider::Openai,
        Provider::Anthropic => crate::receipt::Provider::Anthropic,
    }
}

#[async_trait]
impl UpstreamClient for EngineUpstreamClient {
    async fn forward(
        &self,
        provider: Provider,
        request: ProxyRequest,
    ) -> Result<ProxyResponse, ProxyError> {
        let adapter = self.deps.registry.adapter_for(provider);

        // Build request context for structured logging
        let model = extract_model(&request.body);
        let ctx = RequestContext {
            request_id: request_id_from_headers(&request.headers),
            contract_hash: self.deps.config.contract_hash.clone(),
            provider_str: match provider {
                Provider::OpenAi => "openai",
                Provider::Anthropic => "anthropic",
            },
            model,
        };

        tracing::debug!(
            request_id = %ctx.request_id,
            provider = ctx.provider_str,
            model = %ctx.model,
            "processing request"
        );

        // 0.5) C2.2 one-shot effect run: streaming never consumes it, and only
        // the first non-streaming request claims it, before parsing or upstream.
        #[cfg(not(windows))]
        if let Some(envelope) = &self.envelope {
            if envelope.fixed_run_id().is_some() {
                if is_streaming_request(&request.headers, &request.body) {
                    return Ok(effect_runtime_error(
                        adapter.as_ref(),
                        StatusCode::BAD_REQUEST,
                        EFFECT_STREAMING_UNSUPPORTED,
                    ));
                }
                if !envelope.claim_fixed_run() {
                    return Ok(effect_runtime_error(
                        adapter.as_ref(),
                        StatusCode::CONFLICT,
                        EFFECT_RUN_CONSUMED,
                    ));
                }
            }
        }

        // 1) Parse request into messages
        let mut messages = match adapter.parse_request(&request.body) {
            Ok(msgs) => msgs,
            Err(e) => {
                let body = adapter.serialize_error(&format!("invalid request: {e}"), 400);
                return Ok(ProxyResponse::from_bytes(StatusCode::BAD_REQUEST, body));
            }
        };

        // 1.5) Parse trust header spans from SDK (X-Guard-Trust)
        parse_trust_header(&request.headers, &request.body, &mut messages);

        // 2) Trust assignment
        self.deps
            .trust_assigner
            .assign_trust(&mut messages, &self.deps.config);

        // 2.5) Opt-in receipt capture before any L0 mutation (contract 50747a52, step 1).
        #[cfg(not(windows))]
        let capture: Option<RequestCaptureV0> = match &self.envelope {
            Some(envelope) => {
                if ctx.model.len() > 256 {
                    let body =
                        adapter.serialize_error("invalid request: model exceeds 256 bytes", 400);
                    return Ok(ProxyResponse::from_bytes(StatusCode::BAD_REQUEST, body));
                }
                Some(envelope.capture(
                    &messages,
                    request.headers.get("x-request-id").map(|v| v.as_bytes()),
                    &ctx.request_id,
                    &ctx.model,
                ))
            }
            None => None,
        };
        #[cfg(windows)]
        let capture: Option<RequestCaptureV0> = None;

        // 3) L0 normalization (if enabled)
        //
        // When sanitize mode runs, collect content-free L0 evidence (one entry
        // per message) so it can be surfaced to the routing context below. This
        // does not change enforcement: content mutation and span remapping still
        // go through the injected normalizer exactly as before.
        let mut l0_evidence: Option<Vec<L0Evidence>> = None;
        if let Some(layer) = &self.deps.config.policy.layers.l0 {
            if layer.mode == "sanitize" {
                let msg_evidence = normalize_messages_with_spans_and_evidence(
                    self.deps.normalizer.as_ref(),
                    &mut messages,
                );
                // Role marker neutralization: replace chat template tokens in untrusted content.
                // Runs after normalization so span offsets are already remapped.
                let replacements = neutralize_role_markers(&mut messages);
                if !replacements.is_empty() {
                    tracing::info!(
                        request_id = %ctx.request_id,
                        layer = "L0",
                        count = replacements.len(),
                        tokens = ?replacements.iter().map(|r| r.token.as_str()).collect::<Vec<_>>(),
                        "role markers neutralized in untrusted content"
                    );
                }

                // Merge role-marker neutralization counts in by message index.
                let mut role_counts = vec![0usize; messages.len()];
                for r in &replacements {
                    if let Some(slot) = role_counts.get_mut(r.message_index) {
                        *slot += 1;
                    }
                }

                l0_evidence = Some(
                    msg_evidence
                        .into_iter()
                        .map(|m| L0Evidence {
                            message_index: m.message_index,
                            pre_char_len: m.evidence.pre_char_len,
                            post_char_len: m.evidence.post_char_len,
                            pre_byte_len: m.evidence.pre_byte_len,
                            post_byte_len: m.evidence.post_byte_len,
                            removed_invisible_count: m.evidence.removed_invisible_count,
                            confusable_replacement_count: m.evidence.confusable_replacement_count,
                            html_stripped: m.evidence.html_stripped,
                            role_marker_neutralized_count: role_counts
                                .get(m.message_index)
                                .copied()
                                .unwrap_or(0),
                        })
                        .collect(),
                );
            }
        }

        // Collect-then-decide: run all inbound layers before deciding.
        // Store the first block response; later layers still run for signal collection.
        let mut pending_block: Option<ProxyResponse> = None;

        // Hoisted layer results for the verdict processor.
        let mut l1_result_opt: Option<crate::layers::l1::L1Result> = None;
        let mut l1_signals_opt: Option<Vec<L1Signal>> = None;
        let mut inbound_result_opt: Option<crate::layers::l3_inbound::InboundResult> = None;
        let mut l4_result_opt: Option<crate::layers::l4::L4Result> = None;

        let on_failure = &self.deps.config.runtime.engine.on_failure;

        // 3.5) L1 lightweight classifier (if enabled)
        //
        // Two paths: harness (l1_model) or legacy scanner (l1_scanner).
        // Harness path produces Vec<L1Signal> for the signal extractor and
        // derives L1Verdict for backward-compatible blocking/logging.
        // Legacy path produces L1Result directly (ensemble scanners).
        if let (Some(model), Some(l1_config)) = (&self.deps.l1_model, &self.deps.config.policy.layers.l1) {
            let l1_start = Instant::now();
            let l1_scan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                L1Harness::scan(&messages, model.as_ref())
            }));
            let l1_ms = l1_start.elapsed().as_secs_f64() * 1000.0;

            match l1_scan {
                Ok(signals) => {
                    // Derive verdict from harness signals.
                    //
                    // Mention dampening: if attack signal is concentrated in
                    // quoted regions, compare raw_unquoted_score against
                    // threshold instead of raw_score.
                    //
                    // Squash score is emitted to the signal bus for downstream
                    // consumers but does NOT participate in the verdict. At
                    // threshold=0.0 the squash transform generates too many
                    // false positives on benign text (665 extra FPs for 45 TPs
                    // on the holdout set).
                    let verdict = signals.iter()
                        .find_map(|s| {
                            let outcome = threshold_outcome(s, l1_config.threshold);
                            let eff = outcome.effective_raw;
                            if outcome.threshold_breached {
                                let mention = outcome.outcome_view == OutcomeView::Unquoted;
                                let score_label = if mention { "unquoted_score" } else { "score" };
                                Some(L1Verdict::Block(L1Block {
                                    reason: format!(
                                        "L1 {} {:.3} >= threshold {:.3}",
                                        score_label, eff, l1_config.threshold
                                    ),
                                    message_index: s.message_index,
                                    role: s.role.clone(),
                                    score: eff,
                                }))
                            } else {
                                None
                            }
                        })
                        .unwrap_or(L1Verdict::Allow);

                    match &verdict {
                        L1Verdict::Block(block) if l1_config.mode == crate::config::L1Mode::Block => {
                            tracing::info!(
                                request_id = %ctx.request_id,
                                contract_hash = %ctx.contract_hash,
                                provider = ctx.provider_str,
                                model = %ctx.model,
                                layer = "L1",
                                verdict = "block",
                                role = ?block.role,
                                message_index = block.message_index,
                                score = block.score,
                                reason = %block.reason,
                                latency_ms = l1_ms,
                                "L1 classifier blocked"
                            );
                            let body = adapter.serialize_error("request blocked by content policy", 403);
                            pending_block = Some(ProxyResponse::blocked(body, "L1"));
                        }
                        L1Verdict::Block(block) => {
                            tracing::info!(
                                request_id = %ctx.request_id,
                                contract_hash = %ctx.contract_hash,
                                provider = ctx.provider_str,
                                model = %ctx.model,
                                layer = "L1",
                                verdict = "shadow_block",
                                role = ?block.role,
                                message_index = block.message_index,
                                score = block.score,
                                reason = %block.reason,
                                latency_ms = l1_ms,
                                "L1 classifier detected (shadow)"
                            );
                        }
                        L1Verdict::Allow => {
                            tracing::debug!(
                                request_id = %ctx.request_id,
                                contract_hash = %ctx.contract_hash,
                                provider = ctx.provider_str,
                                model = %ctx.model,
                                layer = "L1",
                                verdict = "allow",
                                latency_ms = l1_ms,
                                "L1 classifier passed"
                            );
                        }
                    }

                    l1_signals_opt = Some(signals);
                }
                Err(_) => {
                    if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L1", "scanner_panic", "scanner panicked", &ctx.request_id) {
                        return Ok(resp);
                    }
                }
            }
        } else if let (Some(scanner), Some(l1_config)) = (&self.deps.l1_scanner, &self.deps.config.policy.layers.l1) {
            // Legacy scanner path (ensemble specialists).
            let l1_start = Instant::now();
            let l1_scan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scanner.scan(&messages, l1_config)
            }));
            let l1_ms = l1_start.elapsed().as_secs_f64() * 1000.0;

            match l1_scan {
                Ok(l1_result) => {
                    l1_result_opt = Some(l1_result);
                }
                Err(_) => {
                    if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L1", "scanner_panic", "scanner panicked", &ctx.request_id) {
                        return Ok(resp);
                    }
                }
            }

            if let Some(ref l1_result) = l1_result_opt {
                match &l1_result.verdict {
                    L1Verdict::Block(block) if l1_config.mode == crate::config::L1Mode::Block => {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L1",
                            verdict = "block",
                            role = ?block.role,
                            message_index = block.message_index,
                            score = block.score,
                            reason = %block.reason,
                            latency_ms = l1_ms,
                            "L1 classifier blocked"
                        );
                        let body = adapter.serialize_error("request blocked by content policy", 403);
                        pending_block = Some(ProxyResponse::blocked(body, "L1"));
                    }
                    L1Verdict::Block(block) => {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L1",
                            verdict = "shadow_block",
                            role = ?block.role,
                            message_index = block.message_index,
                            score = block.score,
                            reason = %block.reason,
                            latency_ms = l1_ms,
                            "L1 classifier detected (shadow)"
                        );
                    }
                    L1Verdict::Allow => {
                        tracing::debug!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L1",
                            verdict = "allow",
                            latency_ms = l1_ms,
                            "L1 classifier passed"
                        );
                    }
                }
            }
        }

        // 3.7) L2a data payload detection (if enabled)
        //
        // Runs PG2 + heuristic fusion on untrusted data segments.
        // Timeout and concurrency are engine-level concerns; the scanner is synchronous.
        // Runtime errors respect engine.on_failure (open=skip, closed=503).
        let mut l2a_signals: Vec<Signal> = Vec::new();
        if let (Some(scanner), Some(l2a_config), Some(semaphore)) = (
            &self.deps.l2a_scanner,
            &self.deps.config.policy.layers.l2a,
            &self.deps.l2a_semaphore,
        ) {
            let permit = semaphore.clone().try_acquire_owned();
            if let Ok(permit) = permit {
                let l2a_timeout = Duration::from_millis(l2a_config.timeout_ms);
                let l2a_start = Instant::now();

                let scanner = Arc::clone(scanner);
                let messages_clone = messages.clone();
                let l2a_config_clone = l2a_config.clone();

                // Move owned permit into the blocking task so it is held
                // until the scan actually finishes, not just until the
                // engine stops waiting (on timeout).
                let scan_future = tokio::task::spawn_blocking(move || {
                    let _permit = permit; // held until this closure returns
                    scanner.scan(&messages_clone, &l2a_config_clone)
                });

                match tokio::time::timeout(l2a_timeout, scan_future).await {
                    Ok(Ok(Ok(signals))) => {
                        let l2a_ms = l2a_start.elapsed().as_secs_f64() * 1000.0;
                        l2a_signals = signals;

                        // Per-layer enforcement.
                        let has_block = l2a_signals.iter()
                            .any(|s| s.score >= l2a_config.block_threshold);
                        match (&l2a_config.mode, has_block) {
                            (L2aMode::Block, true) => {
                                tracing::info!(
                                    request_id = %ctx.request_id,
                                    contract_hash = %ctx.contract_hash,
                                    provider = ctx.provider_str,
                                    model = %ctx.model,
                                    layer = "L2a",
                                    verdict = "block",
                                    signal_count = l2a_signals.len(),
                                    latency_ms = l2a_ms,
                                    "L2a blocked"
                                );
                                if pending_block.is_none() {
                                    let body = adapter.serialize_error(
                                        "request blocked by content policy", 403
                                    );
                                    pending_block = Some(ProxyResponse::blocked(body, "L2a"));
                                }
                            }
                            (L2aMode::Shadow, true) => {
                                tracing::info!(
                                    request_id = %ctx.request_id,
                                    contract_hash = %ctx.contract_hash,
                                    provider = ctx.provider_str,
                                    model = %ctx.model,
                                    layer = "L2a",
                                    verdict = "shadow_block",
                                    signal_count = l2a_signals.len(),
                                    latency_ms = l2a_ms,
                                    "L2a detected (shadow)"
                                );
                            }
                            _ => {
                                tracing::debug!(
                                    request_id = %ctx.request_id,
                                    layer = "L2a",
                                    verdict = "allow",
                                    signal_count = l2a_signals.len(),
                                    latency_ms = l2a_ms,
                                    "L2a scan complete"
                                );
                            }
                        }
                    }
                    Ok(Ok(Err(scan_err))) => {
                        // Scanner returned an error (classify failure or cardinality mismatch).
                        if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L2a", "scan_error", &format!("scan error: {scan_err}"), &ctx.request_id) {
                            return Ok(resp);
                        }
                    }
                    Ok(Err(join_err)) => {
                        // Scan task panicked.
                        if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L2a", "scanner_panic", &format!("scan panicked: {join_err}"), &ctx.request_id) {
                            return Ok(resp);
                        }
                    }
                    Err(_) => {
                        // Timeout.
                        if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L2a", "scan_timeout", &format!("scan timed out ({}ms)", l2a_config.timeout_ms), &ctx.request_id) {
                            return Ok(resp);
                        }
                    }
                }
            } else {
                // Concurrency limit reached.
                if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L2a", "concurrency_limit", "concurrency limit reached", &ctx.request_id) {
                    return Ok(resp);
                }
            }
        }

        // 4) L3-inbound scanning (if enabled)
        // Always run the scanner to collect pattern matches (features for signal pipeline).
        // Gate only the block action on mode, not the scan itself.
        //
        // Evidence signals: count evidence-action matches and surface via response
        // headers (x-parapet-evidence-count, x-parapet-evidence-categories) for
        // observability and eval harness consumption.
        let mut evidence_count: usize = 0;
        let mut evidence_categories: Vec<String> = Vec::new();

        if let Some(layer) = &self.deps.config.policy.layers.l3_inbound {
            let l3i_start = Instant::now();
            let scanner_ref = &self.deps.inbound_scanner;
            let config_ref = &self.deps.config;
            let l3i_scan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scanner_ref.scan(&messages, config_ref)
            }));
            let l3i_ms = l3i_start.elapsed().as_secs_f64() * 1000.0;

            match l3i_scan {
                Ok(result) => { inbound_result_opt = Some(result); }
                Err(_) => {
                    if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L3-inbound", "scanner_panic", "scanner panicked", &ctx.request_id) {
                        return Ok(resp);
                    }
                }
            }

            if let Some(ref inbound_result) = inbound_result_opt {
            // Collect evidence signal summary from all matched patterns.
            let mut cats = std::collections::BTreeSet::new();
            for m in &inbound_result.matched_patterns {
                if m.action == PatternAction::Evidence {
                    evidence_count += 1;
                    if let Some(cat) = self.deps.config.policy.block_patterns
                        .get(m.pattern_index)
                        .and_then(|p| p.category.as_ref())
                    {
                        cats.insert(cat.clone());
                    }
                }
            }
            evidence_categories = cats.into_iter().collect();

            match &inbound_result.verdict {
                InboundVerdict::Block(block) if layer.mode == "block" => {
                    tracing::info!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L3-inbound",
                        verdict = "block",
                        role = ?block.role,
                        message_index = block.message_index,
                        reason = %block.reason,
                        matched_patterns = inbound_result.matched_patterns.len(),
                        evidence_count = evidence_count,
                        latency_ms = l3i_ms,
                        "inbound blocked"
                    );
                    if pending_block.is_none() {
                        let body = adapter.serialize_error("request blocked by content policy", 403);
                        pending_block = Some(ProxyResponse::blocked(body, "L3-inbound"));
                    }
                }
                InboundVerdict::Block(block) => {
                    // shadow/signal mode: log but don't block
                    tracing::info!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L3-inbound",
                        verdict = "block_shadow",
                        role = ?block.role,
                        message_index = block.message_index,
                        reason = %block.reason,
                        matched_patterns = inbound_result.matched_patterns.len(),
                        evidence_count = evidence_count,
                        latency_ms = l3i_ms,
                        "inbound would block (shadow mode)"
                    );
                }
                InboundVerdict::Allow => {
                    tracing::info!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L3-inbound",
                        verdict = "allow",
                        evidence_count = evidence_count,
                        latency_ms = l3i_ms,
                        "inbound allowed"
                    );
                }
            }
            } // if let Some(ref inbound_result)
        }

        // 4.5) Target L3 routing scaffold.
        //
        // The default router is a no-op. It runs after normalized messages,
        // lexical evidence, optional payload-scan signals, and PatternGate
        // evidence are available, but before MultiTurnRisk. Future heuristic
        // signals can enter the existing shadow combiner without changing
        // enforcement behavior.
        let routing_context = RoutingEvidenceContext::new(
            &messages,
            l0_evidence.as_deref(),
            l1_signals_opt.as_deref(),
            l1_result_opt.as_ref(),
            &l2a_signals,
            inbound_result_opt.as_ref(),
            &self.deps.config,
        );
        let heuristic_signals = self.deps.orthogonal_sensor_router.route(&routing_context);

        // 5) L4 multi-turn scanning (if enabled)
        if let (Some(scanner), Some(l4_config)) = (&self.deps.multi_turn_scanner, &self.deps.config.policy.layers.l4) {
            let l4_start = Instant::now();
            let l4_scan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scanner.scan(&messages, l4_config)
            }));
            let l4_ms = l4_start.elapsed().as_secs_f64() * 1000.0;

            match l4_scan {
                Ok(result) => { l4_result_opt = Some(result); }
                Err(_) => {
                    if let Some(resp) = handle_layer_error(on_failure, adapter.as_ref(), "L4", "scanner_panic", "scanner panicked", &ctx.request_id) {
                        return Ok(resp);
                    }
                }
            }

            if let Some(ref l4_result) = l4_result_opt {
            let categories: Vec<&str> = l4_result.matched_categories.iter()
                .map(|c| c.category.as_str()).collect();

            match &l4_result.verdict {
                L4Verdict::Block { reason } if l4_config.mode == L4Mode::Block => {
                    tracing::info!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L4",
                        verdict = "block",
                        categories = ?categories,
                        risk_score = l4_result.risk_score,
                        latency_ms = l4_ms,
                        "multi-turn attack blocked"
                    );
                    if pending_block.is_none() {
                        let body = adapter.serialize_error("request blocked by content policy", 403);
                        pending_block = Some(ProxyResponse::blocked(body, "L4"));
                    }
                }
                L4Verdict::Block { .. } => {
                    // Shadow mode: log but don't block
                    tracing::info!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L4",
                        verdict = "shadow_block",
                        categories = ?categories,
                        risk_score = l4_result.risk_score,
                        latency_ms = l4_ms,
                        "multi-turn attack detected (shadow)"
                    );
                }
                L4Verdict::Allow => {
                    tracing::debug!(
                        request_id = %ctx.request_id,
                        contract_hash = %ctx.contract_hash,
                        provider = ctx.provider_str,
                        model = %ctx.model,
                        layer = "L4",
                        verdict = "allow",
                        categories = ?categories,
                        risk_score = l4_result.risk_score,
                        latency_ms = l4_ms,
                        "multi-turn scan complete"
                    );
                }
            }
            } // if let Some(ref l4_result)
        }

        // 5.5) Capture L2a score headers before signals are moved into the combiner.
        let l2a_max_fused = l2a_signals.iter().map(|s| s.score).fold(0.0f32, f32::max);
        let l2a_max_raw = l2a_signals.iter().filter_map(|s| s.raw_model_score).fold(0.0f32, f32::max);
        let l2a_has_scores = !l2a_signals.is_empty();

        // Verdict processor (shadow — log only, no behavioral change).
        {
            let mut signals = Vec::new();
            if let Some(ref l1s) = l1_signals_opt {
                signals.extend(self.deps.signal_extractor.extract_l1_signals(l1s));
            } else if let Some(ref l1r) = l1_result_opt {
                signals.extend(self.deps.signal_extractor.extract_l1(l1r));
            }
            signals.extend(l2a_signals);
            signals.extend(heuristic_signals);
            if let Some(ref l3r) = inbound_result_opt {
                signals.extend(self.deps.signal_extractor.extract_l3(l3r, &*self.deps.config));
            }
            if let Some(ref l4r) = l4_result_opt {
                signals.extend(self.deps.signal_extractor.extract_l4(l4r));
            }
            if !signals.is_empty() {
                let verdict = self.deps.verdict_combiner.combine(&signals);
                tracing::info!(
                    request_id = %ctx.request_id,
                    verdict_action = ?verdict.action,
                    composite_score = verdict.composite_score,
                    signal_count = signals.len(),
                    contributing_count = verdict.contributing.len(),
                    "verdict processor (shadow)"
                );
            }
        }

        // 5.6) Collect-then-decide: if any inbound layer blocked, return now.
        // All layers have run and logged their signals — the first blocker wins.
        if let Some(mut resp) = pending_block {
            attach_evidence_headers(&mut resp.headers, evidence_count, &evidence_categories);
            if self.deps.emit_l1_signals {
                attach_l1_signal_headers(&mut resp.headers, &l1_signals_opt);
            }
            attach_l2a_score_headers(&mut resp.headers, l2a_has_scores, l2a_max_fused, l2a_max_raw);
            return Ok(resp);
        }

        // 6) Forward to upstream
        let stream = is_streaming_request(&request.headers, &request.body);
        let url = match build_upstream_url(self.deps.resolver.as_ref(), provider, &request) {
            Ok(u) => u,
            Err(refusal) => {
                return Ok(upstream_refusal_response(
                    adapter.as_ref(),
                    self.deps.resolver.as_ref(),
                    refusal,
                ));
            }
        };

        // Reverse proxy header hygiene:
        // - x-parapet-original-host: internal routing header, not for upstream.
        // - Host: strip client's Host header (points at 127.0.0.1:9800). reqwest
        //   sets the correct Host from the upstream URL. This is standard reverse
        //   proxy behavior (RFC 7230 §5.4).
        let mut fwd_headers = request.headers.clone();
        fwd_headers.remove("x-parapet-original-host");
        fwd_headers.remove(reqwest::header::HOST);
        if stream {
            force_identity_encoding_for_streaming(&mut fwd_headers);
        }

        let http_req = HttpRequest {
            method: request.method.clone(),
            url,
            headers: fwd_headers,
            body: request.body.clone(),
            timeout_ms: self.deps.config.runtime.engine.timeout_ms,
            stream,
        };

        let upstream = self
            .deps
            .http
            .send(http_req)
            .await
            .map_err(|e| match e {
                HttpError::Timeout(msg) => ProxyError::UpstreamTimeout(msg),
                HttpError::Transport(msg) => ProxyError::UpstreamFailure(msg),
            })?;

        if upstream.status.is_redirection() {
            return Ok(upstream_redirect_refused_response(
                adapter.as_ref(),
                &ctx.request_id,
                upstream.status,
                &upstream.headers,
            ));
        }

        if stream {
            let mut resp = self.handle_streaming_response_with(provider, upstream, &ctx, capture);
            attach_evidence_headers(&mut resp.headers, evidence_count, &evidence_categories);
            if self.deps.emit_l1_signals {
                attach_l1_signal_headers(&mut resp.headers, &l1_signals_opt);
            }
            attach_l2a_score_headers(&mut resp.headers, l2a_has_scores, l2a_max_fused, l2a_max_raw);
            return Ok(resp);
        }

        let mut resp_headers = upstream.headers;
        let body_bytes = match upstream.body {
            HttpBody::Full(b) => b,
            HttpBody::Stream(mut s) => {
                let mut collected = Vec::new();
                while let Some(chunk) = s.next().await {
                    let bytes = chunk.map_err(|e| ProxyError::UpstreamFailure(e.to_string()))?;
                    collected.extend_from_slice(&bytes);
                    if collected.len() as u64 > MAX_RESPONSE_BODY_BYTES {
                        return Err(ProxyError::UpstreamFailure(format!(
                            "response body exceeds {} byte limit",
                            MAX_RESPONSE_BODY_BYTES
                        )));
                    }
                }
                Bytes::from(collected)
            }
        };

        // Decompress if needed so security scanning can parse the JSON.
        // Strips Content-Encoding/Content-Length from forwarded headers.
        let body_bytes = maybe_decompress(&mut resp_headers, body_bytes)?;

        let processed =
            self.handle_non_streaming_response_with(provider, &body_bytes, &ctx, capture.as_ref());

        // The body was re-serialized (JSON parse → process → to_vec), so
        // the upstream framing headers are stale.  Strip them and let
        // hyper set correct values from the actual body.
        // Content-Encoding is NOT stripped here — maybe_decompress already
        // handles it when decompression occurs, and removing it on
        // passthrough (e.g. JSON parse failure) would be incorrect.
        resp_headers.remove(reqwest::header::CONTENT_LENGTH);
        resp_headers.remove(reqwest::header::TRANSFER_ENCODING);

        attach_evidence_headers(&mut resp_headers, evidence_count, &evidence_categories);
        if self.deps.emit_l1_signals {
            attach_l1_signal_headers(&mut resp_headers, &l1_signals_opt);
        }
        attach_l2a_score_headers(&mut resp_headers, l2a_has_scores, l2a_max_fused, l2a_max_raw);
        if let Some(extra) = processed.extra_headers {
            for (key, value) in extra.iter() {
                resp_headers.insert(key, value.clone());
            }
        }
        Ok(ProxyResponse {
            status: processed.status_override.unwrap_or(upstream.status),
            headers: resp_headers,
            body: Body::from(processed.body),
        })
    }
}

impl EngineUpstreamClient {
    #[cfg_attr(windows, allow(unused_variables))]
    fn handle_streaming_response_with(
        &self,
        provider: Provider,
        upstream: HttpResponse,
        ctx: &RequestContext,
        capture: Option<RequestCaptureV0>,
    ) -> ProxyResponse {
        let compressed_stream = is_gzip(&upstream.headers) || is_deflate(&upstream.headers);
        let content_encoding = upstream
            .headers
            .get(reqwest::header::CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        if compressed_stream {
            let compressed_sse_skip_count = increment_streaming_compressed_sse_skip_count();
            tracing::warn!(
                request_id = %ctx.request_id,
                contract_hash = %ctx.contract_hash,
                provider = ctx.provider_str,
                model = %ctx.model,
                content_encoding = content_encoding.as_deref().unwrap_or("unknown"),
                compressed_sse_skip_count,
                "upstream returned compressed SSE despite identity request; streaming redaction will be skipped"
            );
        }

        let (validator, block_mode) = if let Some(layer) = &self.deps.config.policy.layers.l3_outbound {
            if layer.mode == "block" {
                let mode = match layer.block_action.as_deref() {
                    Some("error") => ToolCallBlockMode::Error,
                    _ => ToolCallBlockMode::Rewrite,
                };
                (
                    Arc::new(ConstraintToolCallValidator {
                        evaluator: self.deps.constraint_evaluator.clone(),
                        config: self.deps.config.clone(),
                        request_id: ctx.request_id.clone(),
                        provider_str: ctx.provider_str,
                        model: ctx.model.clone(),
                        contract_hash: ctx.contract_hash.clone(),
                    }) as Arc<dyn ToolCallValidator>,
                    mode,
                )
            } else {
                (Arc::new(AllowAllToolCalls) as Arc<dyn ToolCallValidator>, ToolCallBlockMode::Error)
            }
        } else {
            (Arc::new(AllowAllToolCalls) as Arc<dyn ToolCallValidator>, ToolCallBlockMode::Error)
        };

        let classifier: Arc<dyn crate::stream::ChunkClassifier> = match provider {
            Provider::OpenAi => Arc::new(OpenAiChunkClassifier),
            Provider::Anthropic => Arc::new(AnthropicChunkClassifier),
        };

        let stream: Pin<Box<dyn Stream<Item = Bytes> + Send>> = match upstream.body {
            HttpBody::Full(bytes) => {
                let single = futures_util::stream::once(async move { Ok(bytes) });
                stream_body(single)
            }
            HttpBody::Stream(s) => stream_body(s),
        };

        #[cfg(not(windows))]
        let processor = match (&self.envelope, capture) {
            (Some(envelope), Some(capture)) => StreamProcessor::with_envelope(
                classifier,
                validator,
                block_mode,
                envelope.stream_session(capture, receipt_provider(provider)),
            ),
            _ => StreamProcessor::new(classifier, validator, block_mode),
        };
        #[cfg(windows)]
        let processor = StreamProcessor::new(classifier, validator, block_mode);
        let processed_stream = processor.process(stream);

        let redacted_stream = if let Some(layer) = &self.deps.config.policy.layers.l5a {
            if layer.mode == "redact" && !compressed_stream {
                tracing::info!(
                    request_id = %ctx.request_id,
                    contract_hash = %ctx.contract_hash,
                    provider = ctx.provider_str,
                    model = %ctx.model,
                    layer = "L5a",
                    mode = "streaming",
                    "L5a streaming redaction active"
                );
                apply_streaming_redaction(
                    processed_stream,
                    self.deps.output_scanner.clone(),
                    self.deps.config.clone(),
                    layer.window_chars,
                )
            } else {
                Box::pin(processed_stream) as Pin<Box<dyn Stream<Item = Bytes> + Send>>
            }
        } else {
            Box::pin(processed_stream) as Pin<Box<dyn Stream<Item = Bytes> + Send>>
        };

        let body_stream = redacted_stream.map(|b| Ok::<Bytes, std::io::Error>(b));

        ProxyResponse {
            status: upstream.status,
            headers: upstream.headers,
            body: Body::from_stream(body_stream),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn handle_non_streaming_response(
        &self,
        provider: Provider,
        body: &Bytes,
        ctx: &RequestContext,
    ) -> ProcessedResponse {
        self.handle_non_streaming_response_with(provider, body, ctx, None)
    }

    #[cfg_attr(windows, allow(unused_variables))]
    fn handle_non_streaming_response_with(
        &self,
        provider: Provider,
        body: &Bytes,
        ctx: &RequestContext,
        capture: Option<&RequestCaptureV0>,
    ) -> ProcessedResponse {
        let adapter = self.deps.registry.adapter_for(provider);
        let parsed: Option<serde_json::Value> = serde_json::from_slice(body).ok();

        // C2.2: the one-shot effect run settles exactly one formed candidate;
        // otherwise nothing reaches finish, executor, store or the result sink.
        #[cfg(not(windows))]
        if let (Some(envelope), Some(_)) = (&self.envelope, capture) {
            if envelope.fixed_run_id().is_some()
                && !parsed.as_ref().is_some_and(|json| {
                    envelope.fixed_buffered_cardinality_ok(receipt_provider(provider), json)
                })
            {
                return ProcessedResponse {
                    status_override: Some(StatusCode::BAD_GATEWAY),
                    body: Bytes::from(adapter.serialize_error(EFFECT_CANDIDATE_CARDINALITY, 502)),
                    extra_headers: Some(effect_runtime_headers(EFFECT_CANDIDATE_CARDINALITY)),
                };
            }
        }

        let mut json: serde_json::Value = match parsed {
            Some(v) => v,
            None => {
                return ProcessedResponse {
                    status_override: None,
                    body: body.clone(),
                    extra_headers: None,
                }
            }
        };

        #[cfg(not(windows))]
        let envelope_settled = if let (Some(envelope), Some(capture)) = (&self.envelope, capture) {
            // Whole-response suppression is decided before serialization.
            let (result, delivery) =
                envelope.process_buffered(capture, receipt_provider(provider), &mut json);
            tracing::info!(
                request_id = %ctx.request_id,
                contract_hash = %ctx.contract_hash,
                provider = ctx.provider_str,
                model = %ctx.model,
                layer = "L3-outbound",
                envelope_runs = result.runs.len(),
                envelope_candidates = result.mapping.len(),
                envelope_completion = ?result.completion,
                envelope_delivery = ?result.delivery,
                "effect receipt envelope settled"
            );
            if delivery.suppressed {
                let body = adapter.serialize_error("request blocked by content policy", 403);
                return ProcessedResponse {
                    status_override: Some(StatusCode::FORBIDDEN),
                    body: Bytes::from(body),
                    extra_headers: None,
                };
            }
            true
        } else {
            false
        };
        #[cfg(windows)]
        let envelope_settled = false;

        if envelope_settled {
            // The envelope above settled every candidate; L3-outbound does not rerun.
        } else if let Some(layer) = &self.deps.config.policy.layers.l3_outbound {
            if layer.mode == "block" {
                let l3o_start = Instant::now();
                let block_action = layer.block_action.as_deref().unwrap_or("rewrite");
                let (l3_result, blocked_tools) = match provider {
                    Provider::OpenAi => apply_l3_outbound_openai(
                        &mut json,
                        self.deps.constraint_evaluator.as_ref(),
                        &self.deps.config,
                        block_action,
                    ),
                    Provider::Anthropic => apply_l3_outbound_anthropic(
                        &mut json,
                        self.deps.constraint_evaluator.as_ref(),
                        &self.deps.config,
                        block_action,
                    ),
                };
                let l3o_ms = l3o_start.elapsed().as_secs_f64() * 1000.0;

                match &l3_result {
                    Err(reason) => {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L3-outbound",
                            verdict = "block",
                            block_action = block_action,
                            reason = %reason,
                            latency_ms = l3o_ms,
                            "outbound blocked (error mode)"
                        );
                        // Generic message to client — internal reason already logged above
                        let body = adapter.serialize_error("request blocked by content policy", 403);
                        return ProcessedResponse {
                            status_override: Some(StatusCode::FORBIDDEN),
                            body: Bytes::from(body),
                            extra_headers: None,
                        };
                    }
                    Ok(()) if !blocked_tools.is_empty() => {
                        for tool in &blocked_tools {
                            tracing::info!(
                                request_id = %ctx.request_id,
                                contract_hash = %ctx.contract_hash,
                                provider = ctx.provider_str,
                                model = %ctx.model,
                                layer = "L3-outbound",
                                verdict = "rewrite",
                                tool_name = %tool,
                                latency_ms = l3o_ms,
                                "outbound tool call rewritten"
                            );
                        }
                    }
                    Ok(()) => {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L3-outbound",
                            verdict = "allow",
                            latency_ms = l3o_ms,
                            "outbound allowed"
                        );
                    }
                }
            }
        }

        let serialized = match serde_json::to_vec(&json) {
            Ok(b) => b,
            Err(_) => {
                return ProcessedResponse {
                    status_override: None,
                    body: body.clone(),
                    extra_headers: None,
                }
            }
        };

        // L5a output redaction — wrapped with catch_unwind for on_failure handling.
        let on_failure = &self.deps.config.runtime.engine.on_failure;

        let l5a_layer = match &self.deps.config.policy.layers.l5a {
            Some(layer) if layer.mode == "redact" => Some(layer),
            _ => None,
        };

        if let Some(_layer) = l5a_layer {
            let scanner = &self.deps.output_scanner;
            let config = &self.deps.config;
            let text = String::from_utf8_lossy(&serialized);

            let l5a_start = Instant::now();
            let l5a_scan = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scanner.scan_and_redact(&text, config)
            }));
            let l5a_ms = l5a_start.elapsed().as_secs_f64() * 1000.0;

            match l5a_scan {
                Ok(result) => {
                    if result.redactions.is_empty() {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L5a",
                            verdict = "clean",
                            latency_ms = l5a_ms,
                            "no redactions needed"
                        );
                    } else {
                        for r in &result.redactions {
                            tracing::info!(
                                request_id = %ctx.request_id,
                                contract_hash = %ctx.contract_hash,
                                provider = ctx.provider_str,
                                model = %ctx.model,
                                layer = "L5a",
                                verdict = "redact",
                                pattern = %r.pattern,
                                position = r.position,
                                latency_ms = l5a_ms,
                                "L5a redacted sensitive content"
                            );
                        }
                    }
                    return ProcessedResponse {
                        status_override: None,
                        body: Bytes::from(result.content),
                        extra_headers: None,
                    };
                }
                Err(_) => {
                    // L5a panicked — check on_failure.
                    match on_failure {
                        FailureMode::Closed => {
                            tracing::error!(
                                request_id = %ctx.request_id,
                                layer = "L5a",
                                failure_mode = "closed",
                                error_type = "scanner_panic",
                                "layer runtime error — blocking request (on_failure: closed)"
                            );
                            let err_body = adapter.serialize_error(
                                "service unavailable (guardrail layer error)", 503
                            );
                            let mut hdrs = HeaderMap::new();
                            if let Ok(v) = "error".parse() { hdrs.insert("x-parapet-action", v); }
                            if let Ok(v) = "L5a".parse() { hdrs.insert("x-parapet-layer", v); }
                            if let Ok(v) = "scanner_panic".parse() { hdrs.insert("x-parapet-error", v); }
                            return ProcessedResponse {
                                status_override: Some(StatusCode::SERVICE_UNAVAILABLE),
                                body: Bytes::from(err_body),
                                extra_headers: Some(hdrs),
                            };
                        }
                        FailureMode::Open => {
                            tracing::warn!(
                                request_id = %ctx.request_id,
                                layer = "L5a",
                                failure_mode = "open",
                                error_type = "scanner_panic",
                                "layer runtime error — skipping L5a (on_failure: open)"
                            );
                            // Skip redaction, return original serialized body.
                        }
                    }
                }
            }
        }

        ProcessedResponse {
            status_override: None,
            body: Bytes::from(serialized),
            extra_headers: None,
        }
    }
}

struct ProcessedResponse {
    status_override: Option<StatusCode>,
    body: Bytes,
    /// Extra headers to merge into the final response (e.g. error metadata from on_failure).
    extra_headers: Option<HeaderMap>,
}

// ---------------------------------------------------------------------------
// Tool call validator for streaming (L3-outbound)
// ---------------------------------------------------------------------------

struct ConstraintToolCallValidator {
    evaluator: Arc<dyn ConstraintEvaluator>,
    config: Arc<Config>,
    request_id: String,
    provider_str: &'static str,
    model: String,
    contract_hash: String,
}

impl ToolCallValidator for ConstraintToolCallValidator {
    fn validate(&self, tool_call: &ToolCall) -> ValidationResult {
        let verdicts = self
            .evaluator
            .evaluate_tool_calls(&[tool_call.clone()], &self.config);

        match verdicts.first() {
            Some(ToolCallVerdict::Allow) | None => {
                tracing::info!(
                    request_id = %self.request_id,
                    contract_hash = %self.contract_hash,
                    provider = self.provider_str,
                    model = %self.model,
                    layer = "L3-outbound",
                    mode = "streaming",
                    verdict = "allow",
                    tool_name = %tool_call.name,
                    "streaming tool call allowed"
                );
                ValidationResult::Allow
            }
            Some(ToolCallVerdict::Block { reason, .. }) => {
                tracing::info!(
                    request_id = %self.request_id,
                    contract_hash = %self.contract_hash,
                    provider = self.provider_str,
                    model = %self.model,
                    layer = "L3-outbound",
                    mode = "streaming",
                    verdict = "block",
                    tool_name = %tool_call.name,
                    reason = %reason,
                    "streaming tool call blocked"
                );
                ValidationResult::Block(reason.clone())
            }
        }
    }
}

struct AllowAllToolCalls;

impl ToolCallValidator for AllowAllToolCalls {
    fn validate(&self, _tool_call: &ToolCall) -> ValidationResult {
        ValidationResult::Allow
    }
}

// ---------------------------------------------------------------------------
// Provider registry
// ---------------------------------------------------------------------------

pub struct DefaultProviderRegistry;

impl ProviderRegistry for DefaultProviderRegistry {
    fn adapter_for(&self, provider: Provider) -> Box<dyn ProviderAdapter> {
        let p = match provider {
            Provider::OpenAi => ProviderType::OpenAi,
            Provider::Anthropic => ProviderType::Anthropic,
        };
        adapter_for(p)
    }
}

// ---------------------------------------------------------------------------
// Startup-validated upstream table & env-based upstream resolver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeClass {
    Https,
    Http,
    Other,
}

impl SchemeClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            SchemeClass::Https => "https",
            SchemeClass::Http => "http",
            SchemeClass::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostClass {
    Absent,
    LoopbackLiteral,
    LocalhostName,
    IpLiteral,
    DnsName,
}

impl HostClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            HostClass::Absent => "absent",
            HostClass::LoopbackLiteral => "loopback_literal",
            HostClass::LocalhostName => "localhost_name",
            HostClass::IpLiteral => "ip_literal",
            HostClass::DnsName => "dns_name",
        }
    }
}

/// Closed-vocabulary Z2 tokens for a parseable-but-refused upstream URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedShape {
    pub scheme_class: SchemeClass,
    pub host_class: HostClass,
    pub has_userinfo: bool,
    pub has_query: bool,
    pub has_fragment: bool,
    pub has_port: bool,
}

impl std::fmt::Display for ParsedShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scheme_class={} host_class={}",
            self.scheme_class.as_str(),
            self.host_class.as_str()
        )?;
        if self.has_userinfo {
            write!(f, " has_userinfo=true")?;
        }
        if self.has_query {
            write!(f, " has_query=true")?;
        }
        if self.has_fragment {
            write!(f, " has_fragment=true")?;
        }
        if self.has_port {
            write!(f, " has_port=true")?;
        }
        Ok(())
    }
}

/// Z4 key identifier for `UpstreamConfigError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpstreamKey {
    Static(&'static str),
    ValidatedHostKey(String),
    ExtraHostsEntry(usize),
    HostKeyCollision {
        known_host: Option<&'static str>,
        extra_indices: Vec<usize>,
    },
}

impl UpstreamKey {
    /// Key name emitted in 502 `parapet_upstream_config_invalid` client responses (Z4).
    pub fn client_key_name(&self) -> &str {
        match self {
            UpstreamKey::Static(k) => k,
            UpstreamKey::ValidatedHostKey(k) => k.as_str(),
            UpstreamKey::ExtraHostsEntry(_) | UpstreamKey::HostKeyCollision { .. } => {
                "PARAPET_EXTRA_HOSTS"
            }
        }
    }
}

impl std::fmt::Display for UpstreamKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpstreamKey::Static(k) => write!(f, "{k}"),
            UpstreamKey::ValidatedHostKey(k) => write!(f, "{k}"),
            UpstreamKey::ExtraHostsEntry(idx) => write!(f, "PARAPET_EXTRA_HOSTS entry #{idx}"),
            UpstreamKey::HostKeyCollision {
                known_host: Some(kh),
                extra_indices,
            } => {
                let idx = extra_indices.first().copied().unwrap_or(1);
                write!(
                    f,
                    "PARAPET_EXTRA_HOSTS entry #{idx} (collides with known host {kh})"
                )
            }
            UpstreamKey::HostKeyCollision {
                known_host: None,
                extra_indices,
            } => {
                let first = extra_indices.first().copied().unwrap_or(1);
                let second = extra_indices.get(1).copied().unwrap_or(first);
                write!(f, "PARAPET_EXTRA_HOSTS entries #{first}, #{second}")
            }
        }
    }
}

/// Startup validation error for upstream overrides.
///
/// (Rev3) No field holds or borrows the refused value, or any substring copied
/// or transformed from it. Only `value_len` and `shape` (Z2 classifications)
/// are derived from the value.
#[derive(Clone, PartialEq, Eq)]
pub struct UpstreamConfigError {
    pub key: UpstreamKey,
    pub rule: &'static str,
    pub fix: &'static str,
    pub value_len: usize,
    pub shape: Option<ParsedShape>,
    pub entry_index: Option<usize>,
}

impl std::fmt::Display for UpstreamConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "upstream configuration refused: {}: {} (value withheld, {} bytes",
            self.key, self.rule, self.value_len
        )?;
        if let Some(shape) = &self.shape {
            write!(f, "; {shape}")?;
        }
        write!(f, "). Fix: {}", self.fix)
    }
}

impl std::fmt::Debug for UpstreamConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamConfigError")
            .field("key", &self.key)
            .field("rule", &self.rule)
            .field("fix", &self.fix)
            .field("value_len", &self.value_len)
            .field("shape", &self.shape)
            .field("entry_index", &self.entry_index)
            .finish()
    }
}

impl std::error::Error for UpstreamConfigError {}

const FIX_NON_HTTPS_UPSTREAM: &str =
    "plain-HTTP upstreams are not supported in this release; use an https:// URL (loopback HTTP support is planned behind an explicit flag)";
const FIX_NON_HTTPS_OPENAI_FALLBACK: &str =
    "plain-HTTP upstreams are not supported in this release; use an https:// URL (loopback HTTP support is planned behind an explicit flag); set PARAPET_OPENAI_BASE_URL=https://api.openai.com to keep the hosted default, or remove OPENAI_BASE_URL from the engine's environment";
const FIX_NON_HTTPS_ANTHROPIC_FALLBACK: &str =
    "plain-HTTP upstreams are not supported in this release; use an https:// URL (loopback HTTP support is planned behind an explicit flag); set PARAPET_ANTHROPIC_BASE_URL=https://api.anthropic.com to keep the hosted default, or remove ANTHROPIC_BASE_URL from the engine's environment";
const FIX_EMPTY_URL: &str =
    "unset the environment variable to keep the default https:// upstream, or set it to a valid https:// URL";
const FIX_EMPTY_OPENAI_FALLBACK: &str =
    "unset OPENAI_BASE_URL, or set PARAPET_OPENAI_BASE_URL=https://api.openai.com to keep the hosted default";
const FIX_EMPTY_ANTHROPIC_FALLBACK: &str =
    "unset ANTHROPIC_BASE_URL, or set PARAPET_ANTHROPIC_BASE_URL=https://api.anthropic.com to keep the hosted default";
const FIX_UNPARSEABLE_URL: &str =
    "provide a valid https:// URL with a host and without userinfo, query, or fragment";
const FIX_UNPARSEABLE_OPENAI_FALLBACK: &str =
    "set PARAPET_OPENAI_BASE_URL=https://api.openai.com to keep the hosted default, or remove OPENAI_BASE_URL from the engine's environment";
const FIX_UNPARSEABLE_ANTHROPIC_FALLBACK: &str =
    "set PARAPET_ANTHROPIC_BASE_URL=https://api.anthropic.com to keep the hosted default, or remove ANTHROPIC_BASE_URL from the engine's environment";
const FIX_UNSUPPORTED_SCHEME: &str =
    "only https:// upstream URLs are supported; provide an https:// URL";
const FIX_USERINFO_FORBIDDEN: &str =
    "remove credentials/userinfo from the upstream URL; pass authentication via request headers instead";
const FIX_QUERY_FORBIDDEN: &str = "remove the query string from the upstream base URL";
const FIX_FRAGMENT_FORBIDDEN: &str = "remove the URL fragment (#...) from the upstream base URL";
const FIX_EXTRA_HOST_INVALID: &str =
    "set PARAPET_EXTRA_HOSTS to a comma-separated list of bare lowercase DNS hostnames (no empty entries, scheme, port, path, userinfo, uppercase, or IP literals)";
const FIX_HOST_KEY_COLLISION: &str =
    "remove colliding entries in PARAPET_EXTRA_HOSTS that map to the same PARAPET_<HOST>_BASE_URL environment variable key";
const FIX_UPSTREAM_FLAG_SYNTAX: &str = "pass --upstream openai=<url> or --upstream anthropic=<url>";
const FIX_UPSTREAM_PROVIDER_NAME: &str =
    "supported --upstream providers in Stage 1 are 'openai' and 'anthropic'";
const FIX_DUPLICATE_UPSTREAM_FLAG: &str = "specify --upstream for each provider at most once";
const FIX_FALLBACK_KEY_HTTP: &str =
    "ambient fallback environment variables (OPENAI_BASE_URL, ANTHROPIC_BASE_URL) cannot enable plain http://; pass --upstream <provider>=http://127.0.0.1:<port> or set PARAPET_<PROVIDER>_BASE_URL with --allow-loopback-http-upstream";
const FIX_LOOPBACK_LITERAL_IP: &str =
    "use a literal loopback IP such as http://127.0.0.1:<port> or http://[::1]:<port>; DNS 'localhost' is refused for plain HTTP";
const FIX_NON_LOOPBACK_HTTP: &str =
    "--allow-loopback-http-upstream only permits literal loopback IPs (127.0.0.0/8 or [::1]); use https:// for remote or LAN hosts";
const FIX_SELF_LOOP_UPSTREAM: &str =
    "upstream loopback port matches parapet's inbound listener port; point --upstream at the model server's distinct port, not parapet's own proxy port";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamTransportMode {
    Https,
    LoopbackHttp,
}

impl UpstreamTransportMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Https => "https",
            Self::LoopbackHttp => "loopback-http",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamSource {
    Cli,
    Env,
    Default,
}

impl UpstreamSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Env => "env",
            Self::Default => "default",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamValidationOptions {
    pub cli_upstreams: Vec<String>,
    pub allow_loopback_http_upstream: bool,
    pub listener_port: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamEndpointMetadata {
    /// Sanitized origin ONLY (`scheme://host[:port]`), NEVER including base path, userinfo, query, or fragment.
    pub sanitized_origin: String,
    pub mode: UpstreamTransportMode,
    pub source: UpstreamSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FallbackKeyKind {
    None,
    OpenAi,
    Anthropic,
}

fn known_host_static_env_key(host: &str) -> Option<&'static str> {
    match host {
        "api.openai.com" => Some("PARAPET_API_OPENAI_COM_BASE_URL"),
        "api.anthropic.com" => Some("PARAPET_API_ANTHROPIC_COM_BASE_URL"),
        "api.cerebras.ai" => Some("PARAPET_API_CEREBRAS_AI_BASE_URL"),
        "api.groq.com" => Some("PARAPET_API_GROQ_COM_BASE_URL"),
        "generativelanguage.googleapis.com" => {
            Some("PARAPET_GENERATIVELANGUAGE_GOOGLEAPIS_COM_BASE_URL")
        }
        "api.together.xyz" => Some("PARAPET_API_TOGETHER_XYZ_BASE_URL"),
        _ => None,
    }
}

fn derive_host_env_key(host: &str) -> String {
    format!(
        "PARAPET_{}_BASE_URL",
        host.to_ascii_uppercase().replace(['.', '-'], "_")
    )
}

fn is_valid_extra_host(trimmed: &str) -> bool {
    if trimmed.is_empty()
        || trimmed.starts_with('.')
        || trimmed.ends_with('.')
        || trimmed.starts_with('-')
        || trimmed.ends_with('-')
        || trimmed.contains("..")
    {
        return false;
    }
    if !trimmed
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-')
    {
        return false;
    }
    if !trimmed.contains('.') {
        return false;
    }
    let labels: Vec<&str> = trimmed.split('.').collect();
    if labels
        .iter()
        .any(|l| l.is_empty() || l.starts_with('-') || l.ends_with('-'))
    {
        return false;
    }
    // Reject IP literals: DNS TLDs begin with an ASCII lowercase letter.
    let Some(last_label) = labels.last() else {
        return false;
    };
    if !last_label
        .bytes()
        .next()
        .is_some_and(|b| b.is_ascii_lowercase())
    {
        return false;
    }
    if trimmed.parse::<std::net::IpAddr>().is_ok() {
        return false;
    }
    if matches!(
        url::Host::parse(trimmed),
        Ok(url::Host::Ipv4(_) | url::Host::Ipv6(_))
    ) {
        return false;
    }
    true
}

fn authority_slice(raw: &str) -> Option<&str> {
    let (_, after_scheme) = raw.split_once("://")?;
    let end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    Some(&after_scheme[..end])
}

fn authority_has_at_sign(raw: &str) -> bool {
    authority_slice(raw).is_some_and(|auth| auth.contains('@'))
}

fn authority_has_explicit_port(raw: &str) -> bool {
    let Some(auth) = authority_slice(raw) else {
        return false;
    };
    let host_port = auth.rsplit_once('@').map(|(_, hp)| hp).unwrap_or(auth);
    if let Some((_, after_bracket)) = host_port.rsplit_once(']') {
        after_bracket.starts_with(':')
    } else {
        host_port.contains(':')
    }
}

fn authority_has_zero_port(raw: &str) -> bool {
    let Some(auth) = authority_slice(raw) else {
        return false;
    };
    let host_port = auth.rsplit_once('@').map(|(_, hp)| hp).unwrap_or(auth);
    let port_str = if let Some((_, after_bracket)) = host_port.rsplit_once(']') {
        after_bracket.strip_prefix(':')
    } else {
        host_port.rsplit_once(':').map(|(_, p)| p)
    };
    port_str.is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b == b'0'))
}

/// Extract the raw path slice after the authority (`scheme://authority`) and
/// before any query (`?`) or fragment (`#`), prior to `url::Url::parse` normalization.
/// Never confuses the scheme's `://` with a path `//`.
fn raw_base_path_slice(raw: &str) -> Option<&str> {
    let (_, after_scheme) = raw.split_once("://")?;
    let auth_end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let tail = &after_scheme[auth_end..];
    if !tail.starts_with('/') {
        return Some("");
    }
    let path_end = tail.find(['?', '#']).unwrap_or(tail.len());
    Some(&tail[..path_end])
}

/// Returns `true` if the pre-parse raw base path contains a backslash (`\`),
/// percent sign (`%`, including `%2e`, `%2E`, `%2e%2e`, `%2f`, `%5c`, etc.),
/// double slash (`//`), or an exact `.` or `..` path segment.
fn has_non_canonical_raw_base_path(raw: &str) -> bool {
    let Some(raw_path) = raw_base_path_slice(raw) else {
        return false;
    };
    if raw_path.is_empty() {
        return false;
    }
    if raw_path.contains('\\') || raw_path.contains('%') || raw_path.contains("//") {
        return true;
    }
    raw_path
        .split('/')
        .skip(1)
        .any(|seg| seg == "." || seg == "..")
}

fn is_localhost_domain(host: &str) -> bool {
    let trimmed = host.trim_end_matches('.');
    trimmed.eq_ignore_ascii_case("localhost")
        || trimmed.to_ascii_lowercase().ends_with(".localhost")
}

fn is_strict_loopback_ip(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(v4)) => v4.is_loopback(),
        Some(url::Host::Ipv6(v6)) => v6.is_loopback() && v6.to_ipv4_mapped().is_none(),
        _ => false,
    }
}

fn format_sanitized_origin(url: &url::Url) -> String {
    let scheme = url.scheme();
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{scheme}://{host}:{port}"),
        None => format!("{scheme}://{host}"),
    }
}

fn classify_parsed_url(url: &url::Url, raw: &str) -> ParsedShape {
    let scheme_class = if url.scheme().eq_ignore_ascii_case("https") {
        SchemeClass::Https
    } else if url.scheme().eq_ignore_ascii_case("http") {
        SchemeClass::Http
    } else {
        SchemeClass::Other
    };
    let host_class = match url.host() {
        None | Some(url::Host::Domain("")) => HostClass::Absent,
        Some(url::Host::Ipv4(v4)) => {
            if v4.is_loopback() {
                HostClass::LoopbackLiteral
            } else {
                HostClass::IpLiteral
            }
        }
        Some(url::Host::Ipv6(v6)) => {
            if v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()) {
                HostClass::LoopbackLiteral
            } else {
                HostClass::IpLiteral
            }
        }
        Some(url::Host::Domain(d)) => {
            if d.eq_ignore_ascii_case("localhost") {
                HostClass::LocalhostName
            } else if let Ok(ip) = d.parse::<std::net::IpAddr>() {
                if ip.is_loopback() {
                    HostClass::LoopbackLiteral
                } else {
                    HostClass::IpLiteral
                }
            } else {
                HostClass::DnsName
            }
        }
    };
    let has_userinfo =
        !url.username().is_empty() || url.password().is_some() || authority_has_at_sign(raw);
    let has_query = url.query().is_some() || raw.contains('?');
    let has_fragment = url.fragment().is_some() || raw.contains('#');
    let has_port = url.port().is_some() || authority_has_explicit_port(raw);

    ParsedShape {
        scheme_class,
        host_class,
        has_userinfo,
        has_query,
        has_fragment,
        has_port,
    }
}

fn fix_for_rule(rule: &'static str, fallback: FallbackKeyKind) -> &'static str {
    match (rule, fallback) {
        ("non_https_upstream", FallbackKeyKind::OpenAi) => FIX_NON_HTTPS_OPENAI_FALLBACK,
        ("non_https_upstream", FallbackKeyKind::Anthropic) => FIX_NON_HTTPS_ANTHROPIC_FALLBACK,
        ("non_https_upstream", FallbackKeyKind::None) => FIX_NON_HTTPS_UPSTREAM,
        ("empty", FallbackKeyKind::OpenAi) => FIX_EMPTY_OPENAI_FALLBACK,
        ("empty", FallbackKeyKind::Anthropic) => FIX_EMPTY_ANTHROPIC_FALLBACK,
        ("empty", FallbackKeyKind::None) => FIX_EMPTY_URL,
        ("unparseable", FallbackKeyKind::OpenAi) => FIX_UNPARSEABLE_OPENAI_FALLBACK,
        ("unparseable", FallbackKeyKind::Anthropic) => FIX_UNPARSEABLE_ANTHROPIC_FALLBACK,
        ("unparseable", FallbackKeyKind::None) => FIX_UNPARSEABLE_URL,
        ("unsupported_scheme", _) => FIX_UNSUPPORTED_SCHEME,
        ("userinfo_forbidden", _) => FIX_USERINFO_FORBIDDEN,
        ("query_forbidden", _) => FIX_QUERY_FORBIDDEN,
        ("fragment_forbidden", _) => FIX_FRAGMENT_FORBIDDEN,
        ("invalid_upstream_flag_syntax", _) => FIX_UPSTREAM_FLAG_SYNTAX,
        ("unknown_upstream_provider", _) => FIX_UPSTREAM_PROVIDER_NAME,
        ("duplicate_upstream_flag", _) => FIX_DUPLICATE_UPSTREAM_FLAG,
        ("fallback_key_http_forbidden", _) => FIX_FALLBACK_KEY_HTTP,
        ("loopback_http_requires_literal_ip", _) => FIX_LOOPBACK_LITERAL_IP,
        ("non_loopback_http_upstream", _) => FIX_NON_LOOPBACK_HTTP,
        ("self_loop_upstream", _) => FIX_SELF_LOOP_UPSTREAM,
        _ => FIX_UNPARSEABLE_URL,
    }
}

fn emit_config_refusal_warn(err: &UpstreamConfigError) {
    tracing::warn!(
        rule = err.rule,
        value_len = err.value_len,
        entry_index = ?err.entry_index,
        diagnostic = %err,
        "upstream configuration refused"
    );
}

#[derive(Clone)]
struct ValidatedUpstreamEntry {
    normalized_url: String,
    parsed_url: url::Url,
    key: UpstreamKey,
    value_len: usize,
    shape: ParsedShape,
    mode: UpstreamTransportMode,
    is_strict_loopback: bool,
}

fn validate_upstream_url_with_options(
    raw: &str,
    key: UpstreamKey,
    fallback: FallbackKeyKind,
    options: &UpstreamValidationOptions,
) -> Result<ValidatedUpstreamEntry, UpstreamConfigError> {
    let value_len = raw.len();
    if raw.is_empty() {
        let err = UpstreamConfigError {
            key,
            rule: "empty",
            fix: fix_for_rule("empty", fallback),
            value_len: 0,
            shape: None,
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if raw != raw.trim()
        || raw.bytes().any(|b| b <= 0x20 || b == 0x7f || b == b'\\')
        || has_non_canonical_raw_base_path(raw)
        || authority_has_zero_port(raw)
    {
        let err = UpstreamConfigError {
            key,
            rule: "unparseable",
            fix: fix_for_rule("unparseable", fallback),
            value_len,
            shape: None,
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    let Ok(url) = url::Url::parse(raw) else {
        let err = UpstreamConfigError {
            key,
            rule: "unparseable",
            fix: fix_for_rule("unparseable", fallback),
            value_len,
            shape: None,
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    };
    let shape = classify_parsed_url(&url, raw);
    if shape.scheme_class == SchemeClass::Other {
        let err = UpstreamConfigError {
            key,
            rule: "unsupported_scheme",
            fix: fix_for_rule("unsupported_scheme", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.scheme_class == SchemeClass::Http && !options.allow_loopback_http_upstream {
        let err = UpstreamConfigError {
            key,
            rule: "non_https_upstream",
            fix: fix_for_rule("non_https_upstream", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.scheme_class == SchemeClass::Http
        && options.allow_loopback_http_upstream
        && fallback != FallbackKeyKind::None
    {
        let err = UpstreamConfigError {
            key,
            rule: "fallback_key_http_forbidden",
            fix: fix_for_rule("fallback_key_http_forbidden", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.host_class == HostClass::Absent || url.cannot_be_a_base() {
        let err = UpstreamConfigError {
            key,
            rule: "unparseable",
            fix: fix_for_rule("unparseable", fallback),
            value_len,
            shape: None,
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.has_userinfo {
        let err = UpstreamConfigError {
            key,
            rule: "userinfo_forbidden",
            fix: fix_for_rule("userinfo_forbidden", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.has_query {
        let err = UpstreamConfigError {
            key,
            rule: "query_forbidden",
            fix: fix_for_rule("query_forbidden", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    if shape.has_fragment {
        let err = UpstreamConfigError {
            key,
            rule: "fragment_forbidden",
            fix: fix_for_rule("fragment_forbidden", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }
    let parsed_path = url.path();
    if parsed_path.contains('\\')
        || parsed_path.contains('%')
        || parsed_path.contains("//")
        || parsed_path
            .split('/')
            .skip(1)
            .any(|seg| seg == "." || seg == "..")
    {
        let err = UpstreamConfigError {
            key,
            rule: "unparseable",
            fix: fix_for_rule("unparseable", fallback),
            value_len,
            shape: None,
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }

    let is_strict_loopback = is_strict_loopback_ip(&url);
    let mode = if shape.scheme_class == SchemeClass::Http {
        if !is_strict_loopback {
            let is_localhost = shape.host_class == HostClass::LocalhostName
                || url.host_str().is_some_and(is_localhost_domain);
            let rule = if is_localhost {
                "loopback_http_requires_literal_ip"
            } else {
                "non_loopback_http_upstream"
            };
            let err = UpstreamConfigError {
                key,
                rule,
                fix: fix_for_rule(rule, fallback),
                value_len,
                shape: Some(shape),
                entry_index: None,
            };
            emit_config_refusal_warn(&err);
            return Err(err);
        }
        UpstreamTransportMode::LoopbackHttp
    } else {
        UpstreamTransportMode::Https
    };

    let effective_port = url.port_or_known_default().unwrap_or(80);
    if is_strict_loopback
        && options
            .listener_port
            .is_some_and(|lp| lp != 0 && lp == effective_port)
    {
        let err = UpstreamConfigError {
            key,
            rule: "self_loop_upstream",
            fix: fix_for_rule("self_loop_upstream", fallback),
            value_len,
            shape: Some(shape),
            entry_index: None,
        };
        emit_config_refusal_warn(&err);
        return Err(err);
    }

    let normalized_url = url.as_str().trim_end_matches('/').to_string();
    Ok(ValidatedUpstreamEntry {
        normalized_url,
        parsed_url: url,
        key,
        value_len,
        shape,
        mode,
        is_strict_loopback,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelfLoopCheckItem {
    key: UpstreamKey,
    value_len: usize,
    shape: ParsedShape,
    effective_port: u16,
}

/// Startup-validated, immutable upstream routing table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamTable {
    openai_base_url: String,
    anthropic_base_url: String,
    openai_metadata: UpstreamEndpointMetadata,
    anthropic_metadata: UpstreamEndpointMetadata,
    extra_hosts: Vec<String>,
    host_overrides: std::collections::HashMap<String, String>,
    loopback_entries: Vec<SelfLoopCheckItem>,
}

fn read_process_env_for_upstream_validation(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(value)) => {
            #[cfg(unix)]
            let value_len = {
                use std::os::unix::ffi::OsStrExt as _;
                value.as_os_str().as_bytes().len()
            };
            #[cfg(not(unix))]
            let value_len = value.as_os_str().as_encoded_bytes().len();

            // Preserve only the byte length so the ordinary validation path emits
            // the correct key/rule without retaining or formatting attacker bytes.
            Some(" ".repeat(value_len.max(1)))
        }
    }
}

fn parse_cli_upstreams(
    options: &UpstreamValidationOptions,
) -> Result<
    (
        Option<ValidatedUpstreamEntry>,
        Option<ValidatedUpstreamEntry>,
    ),
    UpstreamConfigError,
> {
    let mut cli_openai: Option<ValidatedUpstreamEntry> = None;
    let mut cli_anthropic: Option<ValidatedUpstreamEntry> = None;

    for raw_arg in &options.cli_upstreams {
        let Some((provider_raw, raw_url)) = raw_arg.split_once('=') else {
            let err = UpstreamConfigError {
                key: UpstreamKey::Static("--upstream"),
                rule: "invalid_upstream_flag_syntax",
                fix: FIX_UPSTREAM_FLAG_SYNTAX,
                value_len: raw_arg.len(),
                shape: None,
                entry_index: None,
            };
            emit_config_refusal_warn(&err);
            return Err(err);
        };
        if provider_raw.is_empty() {
            let err = UpstreamConfigError {
                key: UpstreamKey::Static("--upstream"),
                rule: "invalid_upstream_flag_syntax",
                fix: FIX_UPSTREAM_FLAG_SYNTAX,
                value_len: raw_arg.len(),
                shape: None,
                entry_index: None,
            };
            emit_config_refusal_warn(&err);
            return Err(err);
        }
        match provider_raw {
            "openai" => {
                if cli_openai.is_some() {
                    let err = UpstreamConfigError {
                        key: UpstreamKey::Static("--upstream openai"),
                        rule: "duplicate_upstream_flag",
                        fix: FIX_DUPLICATE_UPSTREAM_FLAG,
                        value_len: raw_url.len(),
                        shape: None,
                        entry_index: None,
                    };
                    emit_config_refusal_warn(&err);
                    return Err(err);
                }
                let entry = validate_upstream_url_with_options(
                    raw_url,
                    UpstreamKey::Static("--upstream openai"),
                    FallbackKeyKind::None,
                    options,
                )?;
                cli_openai = Some(entry);
            }
            "anthropic" => {
                if cli_anthropic.is_some() {
                    let err = UpstreamConfigError {
                        key: UpstreamKey::Static("--upstream anthropic"),
                        rule: "duplicate_upstream_flag",
                        fix: FIX_DUPLICATE_UPSTREAM_FLAG,
                        value_len: raw_url.len(),
                        shape: None,
                        entry_index: None,
                    };
                    emit_config_refusal_warn(&err);
                    return Err(err);
                }
                let entry = validate_upstream_url_with_options(
                    raw_url,
                    UpstreamKey::Static("--upstream anthropic"),
                    FallbackKeyKind::None,
                    options,
                )?;
                cli_anthropic = Some(entry);
            }
            _ => {
                let err = UpstreamConfigError {
                    key: UpstreamKey::Static("--upstream"),
                    rule: "unknown_upstream_provider",
                    fix: FIX_UPSTREAM_PROVIDER_NAME,
                    value_len: raw_arg.len(),
                    shape: None,
                    entry_index: None,
                };
                emit_config_refusal_warn(&err);
                return Err(err);
            }
        }
    }

    Ok((cli_openai, cli_anthropic))
}

impl UpstreamTable {
    pub fn from_env_fn(get: &dyn Fn(&str) -> Option<String>) -> Result<Self, UpstreamConfigError> {
        Self::from_env_fn_with_options(get, &UpstreamValidationOptions::default())
    }

    pub fn from_env_fn_with_options(
        get: &dyn Fn(&str) -> Option<String>,
        options: &UpstreamValidationOptions,
    ) -> Result<Self, UpstreamConfigError> {
        let (cli_openai, cli_anthropic) = parse_cli_upstreams(options)?;
        let mut loopback_entries: Vec<SelfLoopCheckItem> = Vec::new();
        let mut record_loopback = |entry: &ValidatedUpstreamEntry| {
            if entry.is_strict_loopback {
                if let Some(effective_port) = entry.parsed_url.port_or_known_default() {
                    loopback_entries.push(SelfLoopCheckItem {
                        key: entry.key.clone(),
                        value_len: entry.value_len,
                        shape: entry.shape,
                        effective_port,
                    });
                }
            }
        };

        let mut extra_hosts: Vec<String> = Vec::new();
        let mut extra_host_keys: Vec<(usize, String)> = Vec::new();

        if let Some(raw_extra) = get("PARAPET_EXTRA_HOSTS") {
            for (idx0, raw_entry) in raw_extra.split(',').enumerate() {
                let entry_index = idx0 + 1;
                let trimmed = raw_entry.trim();
                if !is_valid_extra_host(trimmed) {
                    let err = UpstreamConfigError {
                        key: UpstreamKey::ExtraHostsEntry(entry_index),
                        rule: "extra_host_invalid",
                        fix: FIX_EXTRA_HOST_INVALID,
                        value_len: raw_entry.len(),
                        shape: None,
                        entry_index: Some(entry_index),
                    };
                    emit_config_refusal_warn(&err);
                    return Err(err);
                }
                let derived_key = derive_host_env_key(trimmed);
                if let Some(&known) = KNOWN_LLM_HOSTS
                    .iter()
                    .find(|&&kh| derive_host_env_key(kh) == derived_key)
                {
                    let err = UpstreamConfigError {
                        key: UpstreamKey::HostKeyCollision {
                            known_host: Some(known),
                            extra_indices: vec![entry_index],
                        },
                        rule: "host_key_collision",
                        fix: FIX_HOST_KEY_COLLISION,
                        value_len: raw_entry.len(),
                        shape: None,
                        entry_index: Some(entry_index),
                    };
                    emit_config_refusal_warn(&err);
                    return Err(err);
                }
                if let Some((prev_idx, _)) =
                    extra_host_keys.iter().find(|(_, pk)| *pk == derived_key)
                {
                    let err = UpstreamConfigError {
                        key: UpstreamKey::HostKeyCollision {
                            known_host: None,
                            extra_indices: vec![*prev_idx, entry_index],
                        },
                        rule: "host_key_collision",
                        fix: FIX_HOST_KEY_COLLISION,
                        value_len: raw_entry.len(),
                        shape: None,
                        entry_index: Some(entry_index),
                    };
                    emit_config_refusal_warn(&err);
                    return Err(err);
                }
                extra_host_keys.push((entry_index, derived_key));
                extra_hosts.push(trimmed.to_string());
            }
        }

        let (mut openai_base_url, mut openai_metadata, openai_explicit) = if let Some(entry) =
            &cli_openai
        {
            record_loopback(entry);
            (
                entry.normalized_url.clone(),
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Cli,
                },
                true,
            )
        } else if let Some(primary) = get("PARAPET_OPENAI_BASE_URL") {
            let entry = validate_upstream_url_with_options(
                &primary,
                UpstreamKey::Static("PARAPET_OPENAI_BASE_URL"),
                FallbackKeyKind::None,
                options,
            )?;
            if get("OPENAI_BASE_URL").is_some() {
                tracing::info!(
                    primary_key = "PARAPET_OPENAI_BASE_URL",
                    shadowed_key = "OPENAI_BASE_URL",
                    "fallback upstream base URL environment variable is shadowed by primary"
                );
            }
            record_loopback(&entry);
            (
                entry.normalized_url,
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Env,
                },
                true,
            )
        } else if let Some(fallback) = get("OPENAI_BASE_URL") {
            let entry = validate_upstream_url_with_options(
                &fallback,
                UpstreamKey::Static("OPENAI_BASE_URL"),
                FallbackKeyKind::OpenAi,
                options,
            )?;
            tracing::warn!(
                key = "OPENAI_BASE_URL",
                "using fallback upstream base URL environment variable; prefer PARAPET_OPENAI_BASE_URL"
            );
            record_loopback(&entry);
            (
                entry.normalized_url,
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Env,
                },
                true,
            )
        } else {
            (
                "https://api.openai.com".to_string(),
                UpstreamEndpointMetadata {
                    sanitized_origin: "https://api.openai.com".to_string(),
                    mode: UpstreamTransportMode::Https,
                    source: UpstreamSource::Default,
                },
                false,
            )
        };

        let (mut anthropic_base_url, mut anthropic_metadata, anthropic_explicit) = if let Some(
            entry,
        ) =
            &cli_anthropic
        {
            record_loopback(entry);
            (
                entry.normalized_url.clone(),
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Cli,
                },
                true,
            )
        } else if let Some(primary) = get("PARAPET_ANTHROPIC_BASE_URL") {
            let entry = validate_upstream_url_with_options(
                &primary,
                UpstreamKey::Static("PARAPET_ANTHROPIC_BASE_URL"),
                FallbackKeyKind::None,
                options,
            )?;
            if get("ANTHROPIC_BASE_URL").is_some() {
                tracing::info!(
                    primary_key = "PARAPET_ANTHROPIC_BASE_URL",
                    shadowed_key = "ANTHROPIC_BASE_URL",
                    "fallback upstream base URL environment variable is shadowed by primary"
                );
            }
            record_loopback(&entry);
            (
                entry.normalized_url,
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Env,
                },
                true,
            )
        } else if let Some(fallback) = get("ANTHROPIC_BASE_URL") {
            let entry = validate_upstream_url_with_options(
                &fallback,
                UpstreamKey::Static("ANTHROPIC_BASE_URL"),
                FallbackKeyKind::Anthropic,
                options,
            )?;
            tracing::warn!(
                key = "ANTHROPIC_BASE_URL",
                "using fallback upstream base URL environment variable; prefer PARAPET_ANTHROPIC_BASE_URL"
            );
            record_loopback(&entry);
            (
                entry.normalized_url,
                UpstreamEndpointMetadata {
                    sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                    mode: entry.mode,
                    source: UpstreamSource::Env,
                },
                true,
            )
        } else {
            (
                "https://api.anthropic.com".to_string(),
                UpstreamEndpointMetadata {
                    sanitized_origin: "https://api.anthropic.com".to_string(),
                    mode: UpstreamTransportMode::Https,
                    source: UpstreamSource::Default,
                },
                false,
            )
        };

        let mut host_overrides = std::collections::HashMap::new();
        if let Some(entry) = &cli_openai {
            host_overrides.insert("api.openai.com".to_string(), entry.normalized_url.clone());
        }
        if let Some(entry) = &cli_anthropic {
            host_overrides.insert(
                "api.anthropic.com".to_string(),
                entry.normalized_url.clone(),
            );
        }

        for &known_host in KNOWN_LLM_HOSTS {
            if (known_host == "api.openai.com" && cli_openai.is_some())
                || (known_host == "api.anthropic.com" && cli_anthropic.is_some())
            {
                continue;
            }
            let env_key = known_host_static_env_key(known_host)
                .expect("every KNOWN_LLM_HOSTS entry has a static env key");
            if let Some(raw) = get(env_key) {
                let entry = validate_upstream_url_with_options(
                    &raw,
                    UpstreamKey::Static(env_key),
                    FallbackKeyKind::None,
                    options,
                )?;
                record_loopback(&entry);
                if known_host == "api.openai.com" && !openai_explicit {
                    openai_base_url = entry.normalized_url.clone();
                    openai_metadata = UpstreamEndpointMetadata {
                        sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                        mode: entry.mode,
                        source: UpstreamSource::Env,
                    };
                }
                if known_host == "api.anthropic.com" && !anthropic_explicit {
                    anthropic_base_url = entry.normalized_url.clone();
                    anthropic_metadata = UpstreamEndpointMetadata {
                        sanitized_origin: format_sanitized_origin(&entry.parsed_url),
                        mode: entry.mode,
                        source: UpstreamSource::Env,
                    };
                }
                host_overrides.insert(known_host.to_string(), entry.normalized_url);
            } else if known_host == "api.openai.com" && openai_explicit {
                host_overrides.insert("api.openai.com".to_string(), openai_base_url.clone());
            } else if known_host == "api.anthropic.com" && anthropic_explicit {
                host_overrides.insert("api.anthropic.com".to_string(), anthropic_base_url.clone());
            }
        }
        for host in &extra_hosts {
            let env_key = derive_host_env_key(host);
            if let Some(raw) = get(&env_key) {
                let entry = validate_upstream_url_with_options(
                    &raw,
                    UpstreamKey::ValidatedHostKey(env_key),
                    FallbackKeyKind::None,
                    options,
                )?;
                record_loopback(&entry);
                host_overrides.insert(host.clone(), entry.normalized_url);
            }
        }

        Ok(Self {
            openai_base_url,
            anthropic_base_url,
            openai_metadata,
            anthropic_metadata,
            extra_hosts,
            host_overrides,
            loopback_entries,
        })
    }

    pub fn from_process_env() -> Result<Self, UpstreamConfigError> {
        Self::from_process_env_with_options(&UpstreamValidationOptions::default())
    }

    pub fn from_process_env_with_options(
        options: &UpstreamValidationOptions,
    ) -> Result<Self, UpstreamConfigError> {
        let table =
            Self::from_env_fn_with_options(&read_process_env_for_upstream_validation, options)?;
        let mut unrelated_count = 0usize;
        for (os_key, _) in std::env::vars_os() {
            let Some(key) = os_key.to_str() else {
                continue;
            };
            if !key.starts_with("PARAPET_") || !key.ends_with("_BASE_URL") {
                continue;
            };
            if key == "PARAPET_OPENAI_BASE_URL" || key == "PARAPET_ANTHROPIC_BASE_URL" {
                continue;
            }
            let is_known = KNOWN_LLM_HOSTS
                .iter()
                .any(|&h| known_host_static_env_key(h) == Some(key));
            let is_extra = table
                .extra_hosts
                .iter()
                .any(|h| derive_host_env_key(h) == key);
            if !is_known && !is_extra {
                unrelated_count += 1;
            }
        }
        if unrelated_count > 0 {
            tracing::warn!(
                ignored_key_count = unrelated_count,
                "ignoring PARAPET_*_BASE_URL environment variable(s) for non-allowlisted host(s)"
            );
        }
        Ok(table)
    }

    pub fn check_no_self_loop(&self, bound_port: u16) -> Result<(), UpstreamConfigError> {
        if bound_port == 0 {
            return Ok(());
        }
        for item in &self.loopback_entries {
            if item.effective_port == bound_port {
                let err = UpstreamConfigError {
                    key: item.key.clone(),
                    rule: "self_loop_upstream",
                    fix: FIX_SELF_LOOP_UPSTREAM,
                    value_len: item.value_len,
                    shape: Some(item.shape),
                    entry_index: None,
                };
                emit_config_refusal_warn(&err);
                return Err(err);
            }
        }
        Ok(())
    }

    pub fn endpoint_metadata(&self, provider: Provider) -> &UpstreamEndpointMetadata {
        match provider {
            Provider::OpenAi => &self.openai_metadata,
            Provider::Anthropic => &self.anthropic_metadata,
        }
    }

    pub fn sanitized_origin(&self, provider: Provider) -> &str {
        &self.endpoint_metadata(provider).sanitized_origin
    }

    pub fn is_allowed_host(&self, host: &str) -> bool {
        KNOWN_LLM_HOSTS.contains(&host) || self.extra_hosts.iter().any(|h| h == host)
    }

    pub fn base_url_for_host(&self, host: &str) -> String {
        self.host_overrides
            .get(host)
            .cloned()
            .unwrap_or_else(|| format!("https://{host}"))
    }

    pub fn base_url(&self, provider: Provider) -> &str {
        match provider {
            Provider::OpenAi => &self.openai_base_url,
            Provider::Anthropic => &self.anthropic_base_url,
        }
    }
}

struct InvalidEnvSnapshot {
    error: UpstreamConfigError,
    openai_legacy: String,
    anthropic_legacy: String,
    extra_hosts: Vec<String>,
    host_overrides: std::collections::HashMap<String, String>,
}

impl std::fmt::Debug for InvalidEnvSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvalidEnvSnapshot")
            .field("error", &self.error)
            .field(
                "openai_legacy",
                &format!("<withheld {} bytes>", self.openai_legacy.len()),
            )
            .field(
                "anthropic_legacy",
                &format!("<withheld {} bytes>", self.anthropic_legacy.len()),
            )
            .finish()
    }
}

pub struct EnvUpstreamResolver {
    state: Result<UpstreamTable, InvalidEnvSnapshot>,
}

impl std::fmt::Debug for EnvUpstreamResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvUpstreamResolver")
            .field("state", &self.state)
            .finish()
    }
}

impl EnvUpstreamResolver {
    pub fn new() -> Self {
        Self::from_env_fn(&read_process_env_for_upstream_validation)
    }

    pub fn from_table(table: UpstreamTable) -> Self {
        Self { state: Ok(table) }
    }

    pub fn from_env_fn(get: &dyn Fn(&str) -> Option<String>) -> Self {
        Self::from_env_fn_with_options(get, &UpstreamValidationOptions::default())
    }

    pub fn from_env_fn_with_options(
        get: &dyn Fn(&str) -> Option<String>,
        options: &UpstreamValidationOptions,
    ) -> Self {
        match UpstreamTable::from_env_fn_with_options(get, options) {
            Ok(table) => Self { state: Ok(table) },
            Err(error) => {
                let openai_legacy = get("PARAPET_OPENAI_BASE_URL")
                    .or_else(|| get("OPENAI_BASE_URL"))
                    .unwrap_or_else(|| "https://api.openai.com".to_string());
                let anthropic_legacy = get("PARAPET_ANTHROPIC_BASE_URL")
                    .or_else(|| get("ANTHROPIC_BASE_URL"))
                    .unwrap_or_else(|| "https://api.anthropic.com".to_string());
                let extra_hosts: Vec<String> = get("PARAPET_EXTRA_HOSTS")
                    .map(|raw| {
                        raw.split(',')
                            .map(str::trim)
                            .filter(|s| is_valid_extra_host(s))
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                let mut host_overrides = std::collections::HashMap::new();
                for &known_host in KNOWN_LLM_HOSTS {
                    if let Some(env_key) = known_host_static_env_key(known_host) {
                        if let Some(val) = get(env_key) {
                            host_overrides.insert(known_host.to_string(), val);
                        }
                    }
                }
                for host in &extra_hosts {
                    let env_key = derive_host_env_key(host);
                    if let Some(val) = get(&env_key) {
                        host_overrides.insert(host.clone(), val);
                    }
                }
                Self {
                    state: Err(InvalidEnvSnapshot {
                        error,
                        openai_legacy,
                        anthropic_legacy,
                        extra_hosts,
                        host_overrides,
                    }),
                }
            }
        }
    }
}

impl Default for EnvUpstreamResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl UpstreamResolver for EnvUpstreamResolver {
    fn is_allowed_host(&self, host: &str) -> bool {
        match &self.state {
            Ok(table) => table.is_allowed_host(host),
            Err(inv) => {
                KNOWN_LLM_HOSTS.contains(&host) || inv.extra_hosts.iter().any(|h| h == host)
            }
        }
    }

    fn base_url_for_host(&self, host: &str) -> String {
        match &self.state {
            Ok(table) => table.base_url_for_host(host),
            Err(inv) => inv
                .host_overrides
                .get(host)
                .cloned()
                .unwrap_or_else(|| format!("https://{host}")),
        }
    }

    fn base_url(&self, provider: Provider) -> String {
        match &self.state {
            Ok(table) => table.base_url(provider).to_owned(),
            Err(inv) => match provider {
                Provider::OpenAi => inv.openai_legacy.clone(),
                Provider::Anthropic => inv.anthropic_legacy.clone(),
            },
        }
    }

    fn resolve(
        &self,
        provider: Provider,
        original_host: OriginalHost<'_>,
    ) -> Result<String, UpstreamRefusal> {
        let table = match &self.state {
            Ok(table) => table,
            Err(_) => return Err(UpstreamRefusal::ConfigInvalid),
        };
        match original_host {
            OriginalHost::Absent => Ok(table.base_url(provider).to_owned()),
            OriginalHost::Present(host) => {
                if table.is_allowed_host(host) {
                    Ok(table.base_url_for_host(host))
                } else {
                    tracing::warn!(
                        rule = "header_not_allowlisted",
                        value_count = 1usize,
                        value_lens = ?[host.len()],
                        "x-parapet-original-host refused"
                    );
                    Err(UpstreamRefusal::HostNotAllowed)
                }
            }
            OriginalHost::InvalidHeader {
                rule,
                value_count,
                value_lens,
            } => {
                tracing::warn!(
                    rule = rule,
                    value_count = value_count,
                    value_lens = ?value_lens,
                    "x-parapet-original-host refused"
                );
                Err(UpstreamRefusal::HostNotAllowed)
            }
        }
    }

    fn config_error(&self) -> Option<&UpstreamConfigError> {
        match &self.state {
            Ok(_) => None,
            Err(inv) => Some(&inv.error),
        }
    }
}

/// Default HTTPS-only upstream resolver bound in `build_engine_client` and
/// `build_engine_client_with_runtime_v0`.
pub type DefaultUpstreamResolver = EnvUpstreamResolver;

// ---------------------------------------------------------------------------
// Reqwest HTTP sender
// ---------------------------------------------------------------------------

/// Build the production `reqwest::Client` with redirect following disabled.
pub fn production_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("build production reqwest client")
}

pub struct ReqwestHttpSender {
    client: reqwest::Client,
}

impl ReqwestHttpSender {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl HttpSender for ReqwestHttpSender {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let mut req = self
            .client
            .request(request.method, &request.url)
            .headers(request.headers)
            .body(request.body);

        if let Some(timeout_ms) = request.timeout_ms {
            req = req.timeout(std::time::Duration::from_millis(timeout_ms));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    HttpError::Timeout(e.to_string())
                } else {
                    HttpError::Transport(e.to_string())
                }
            })?;

        let status = resp.status();
        let headers = resp.headers().clone();

        if request.stream {
            let stream = resp.bytes_stream().map_err(|e| HttpError::Transport(e.to_string()));
            Ok(HttpResponse {
                status,
                headers,
                body: HttpBody::Stream(Box::pin(stream)),
            })
        } else {
            let body = resp
                .bytes()
                .await
                .map_err(|e| HttpError::Transport(e.to_string()))?;
            Ok(HttpResponse {
                status,
                headers,
                body: HttpBody::Full(body),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Public factory for default engine client
// ---------------------------------------------------------------------------

pub fn build_engine_client(config: Arc<Config>) -> Result<EngineUpstreamClient, StartupError> {
    let upstream_table = UpstreamTable::from_process_env().map_err(StartupError::UpstreamConfig)?;
    build_engine_client_with_upstream_table(config, upstream_table)
}

pub fn build_engine_client_with_upstream_table(
    config: Arc<Config>,
    upstream_table: UpstreamTable,
) -> Result<EngineUpstreamClient, StartupError> {
    let session_store: Option<Arc<dyn crate::session::SessionStore>> = None; // Sprint 2

    // L1: harness path for generalist, legacy scanner for ensemble.
    let (l1_model, l1_scanner): (Option<Arc<dyn L1Model>>, Option<Arc<dyn L1Scanner>>) =
        if let Some(l1_config) = &config.policy.layers.l1 {
            if l1_config.specialists.is_empty() {
                (Some(Arc::new(SvmModel)), None)
            } else {
                (None, Some(Arc::new(EnsembleL1Scanner::new(
                    &l1_config.specialists,
                    l1_config.min_agree,
                    l1_config.generalist_solo_threshold,
                ))))
            }
        } else {
            (None, None)
        };

    let multi_turn_scanner: Option<Arc<dyn MultiTurnScanner>> =
        if config.policy.layers.l4.is_some() {
            Some(Arc::new(DefaultMultiTurnScanner::new()))
        } else {
            None
        };

    // L2a scanner construction.
    // Fail-closed at startup: if L2a is configured, the model must be loadable.
    // Feature-gate check: if config enables L2a but feature is disabled, error.
    #[cfg(not(feature = "l2a"))]
    if config.policy.layers.l2a.is_some() {
        return Err(StartupError::FeatureDisabled(
            "policy.layers.l2a is configured but binary was built without \
             --features l2a. Rebuild with: cargo build --features l2a".into()
        ));
    }

    #[cfg(feature = "l2a")]
    let l2a_scanner: Option<Arc<dyn L2aScanner>> =
        if let Some(l2a_config) = &config.policy.layers.l2a {
            use crate::layers::l2a::DefaultL2aScanner;
            use crate::layers::l2a::DefaultHeuristicScanner;
            use crate::layers::l2a_model::OnnxPromptGuard;

            let classifier = OnnxPromptGuard::init(l2a_config)
                .map_err(|e| StartupError::ModelMissing(e.to_string()))?;
            let heuristics = DefaultHeuristicScanner::new();
            Some(Arc::new(DefaultL2aScanner::new(
                Box::new(classifier),
                Box::new(heuristics),
            )))
        } else {
            None
        };

    #[cfg(not(feature = "l2a"))]
    let l2a_scanner: Option<Arc<dyn L2aScanner>> = None;

    let l2a_semaphore = config.policy.layers.l2a.as_ref().map(|l2a_config| {
        Arc::new(tokio::sync::Semaphore::new(l2a_config.max_concurrent_scans))
    });

    let deps = EngineDeps {
        config,
        http: Arc::new(ReqwestHttpSender::new(production_http_client())),
        resolver: Arc::new(EnvUpstreamResolver::from_table(upstream_table)),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        l1_scanner,
        l1_model,
        l2a_scanner,
        l2a_semaphore,
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store,
        multi_turn_scanner,
        orthogonal_sensor_router: Arc::new(NoopOrthogonalSensorRouter::new()),
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
        emit_l1_signals: false,
    };

    Ok(EngineUpstreamClient::new_with(deps))
}

// ---------------------------------------------------------------------------
// POSIX opt-in runtime envelope and deterministic offline upstream (1D)
// ---------------------------------------------------------------------------

/// Deterministic offline demo workflow (1E): proposer, fixtures, and export.
#[cfg(not(windows))]
#[path = "../demo/mod.rs"]
pub mod demo;

/// Schema identifier for the fixed deterministic offline upstream fixture.
#[cfg(not(windows))]
pub const OFFLINE_UPSTREAM_SCHEMA: &str = "parapet-offline-upstream/v0";

/// Fixed deterministic offline upstream fixture selector (`parapet-offline-upstream/v0`).
#[cfg(not(windows))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineUpstreamFixtureV0 {
    /// Generic v0 selector; uses `x-parapet-offline-scenario` header or default.
    V0,
    /// `allow_no_executor` scenario.
    AllowNoExecutor,
    /// `allow_verified_write` scenario.
    AllowVerifiedWrite,
    /// `block_no_effect` scenario.
    BlockNoEffect,
    /// `shadow_parse_error_l3_off` scenario.
    ShadowParseErrorL3Off,
    /// 1E demo journey answered by the document-consuming demo proposer.
    #[cfg(not(windows))]
    Demo(demo::DemoJourneyV0),
}

#[cfg(not(windows))]
impl OfflineUpstreamFixtureV0 {
    /// Parse an offline fixture selector string.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "v0" => Some(Self::V0),
            "allow_no_executor" => Some(Self::AllowNoExecutor),
            "allow_verified_write" => Some(Self::AllowVerifiedWrite),
            "block_no_effect" => Some(Self::BlockNoEffect),
            "shadow_parse_error_l3_off" => Some(Self::ShadowParseErrorL3Off),
            #[cfg(not(windows))]
            "demo_benign_write_v0" => Some(Self::Demo(demo::DemoJourneyV0::BenignWrite)),
            #[cfg(not(windows))]
            "demo_unauthorized_v0" => Some(Self::Demo(demo::DemoJourneyV0::Unauthorized)),
            #[cfg(not(windows))]
            "demo_injection_benign_v0" => Some(Self::Demo(demo::DemoJourneyV0::InjectionBenign)),
            #[cfg(not(windows))]
            "demo_mismatch_v0" => Some(Self::Demo(demo::DemoJourneyV0::Mismatch)),
            _ => None,
        }
    }
}

/// Deterministic offline HTTP sender (`parapet-offline-upstream/v0`).
///
/// Never performs network I/O. Preserves normal `DefaultUpstreamResolver` HTTPS
/// host allowlist checks before `HttpSender::send` is invoked.
#[cfg(not(windows))]
pub struct DeterministicOfflineHttpSenderV0 {
    fixture: OfflineUpstreamFixtureV0,
}

#[cfg(not(windows))]
impl DeterministicOfflineHttpSenderV0 {
    /// Construct a deterministic offline HTTP sender for the selected fixture.
    pub fn new(fixture: OfflineUpstreamFixtureV0) -> Self {
        Self { fixture }
    }

    fn resolve_scenario(&self, request: &HttpRequest) -> OfflineUpstreamFixtureV0 {
        if let Some(hdr) = request
            .headers
            .get("x-parapet-offline-scenario")
            .and_then(|v| v.to_str().ok())
        {
            if let Some(parsed) = OfflineUpstreamFixtureV0::parse(hdr.trim()) {
                // A demo selector in this header is no match: only the demo sender serves it.
                #[cfg(not(windows))]
                let parsed = if matches!(parsed, OfflineUpstreamFixtureV0::Demo(_)) {
                    OfflineUpstreamFixtureV0::V0
                } else {
                    parsed
                };
                if parsed != OfflineUpstreamFixtureV0::V0 {
                    return parsed;
                }
            }
        }
        if self.fixture != OfflineUpstreamFixtureV0::V0 {
            return self.fixture;
        }
        let body_str = std::str::from_utf8(&request.body).unwrap_or("");
        if body_str.contains("shadow_parse_error_l3_off") {
            OfflineUpstreamFixtureV0::ShadowParseErrorL3Off
        } else if body_str.contains("block_no_effect") || body_str.contains("../escape.txt") {
            OfflineUpstreamFixtureV0::BlockNoEffect
        } else if body_str.contains("allow_no_executor") {
            OfflineUpstreamFixtureV0::AllowNoExecutor
        } else {
            OfflineUpstreamFixtureV0::AllowVerifiedWrite
        }
    }
}

#[cfg(not(windows))]
#[async_trait]
impl HttpSender for DeterministicOfflineHttpSenderV0 {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        #[cfg(not(windows))]
        if matches!(self.fixture, OfflineUpstreamFixtureV0::Demo(_)) {
            return Err(HttpError::Transport(
                "offline fixture demo journey requires the demo proposer sender".into(),
            ));
        }
        if !request.url.starts_with("https://") {
            return Err(HttpError::Transport(
                "offline fixture requires HTTPS resolved upstream URL".into(),
            ));
        }
        let scenario = self.resolve_scenario(&request);
        let is_anthropic =
            request.url.contains("anthropic.com") || request.url.ends_with("/v1/messages");
        let payload = if is_anthropic {
            let input_val = match scenario {
                OfflineUpstreamFixtureV0::AllowNoExecutor
                | OfflineUpstreamFixtureV0::AllowVerifiedWrite
                | OfflineUpstreamFixtureV0::V0 => serde_json::json!({
                    "path": "note.txt",
                    "content": "hello parapet\n"
                }),
                OfflineUpstreamFixtureV0::BlockNoEffect => serde_json::json!({
                    "path": "../escape.txt",
                    "content": "blocked"
                }),
                OfflineUpstreamFixtureV0::ShadowParseErrorL3Off => {
                    serde_json::Value::String("{unclosed".to_owned())
                }
                #[cfg(not(windows))]
                OfflineUpstreamFixtureV0::Demo(_) => {
                    return Err(HttpError::Transport(
                        "offline fixture demo journey requires the demo proposer sender".into(),
                    ))
                }
            };
            serde_json::json!({
                "id": "msg_offline_v0",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_offline_v0_1",
                        "name": "write_file",
                        "input": input_val
                    }
                ],
                "stop_reason": "tool_use"
            })
        } else {
            let custom_raw = request
                .headers
                .get("x-parapet-offline-raw-arguments")
                .and_then(|v| v.to_str().ok());
            let arguments_raw = if let Some(raw) = custom_raw {
                raw
            } else {
                match scenario {
                    OfflineUpstreamFixtureV0::AllowNoExecutor
                    | OfflineUpstreamFixtureV0::AllowVerifiedWrite
                    | OfflineUpstreamFixtureV0::V0 => {
                        r#"{"content":"hello parapet\n","path":"note.txt"}"#
                    }
                    OfflineUpstreamFixtureV0::BlockNoEffect => {
                        r#"{"content":"blocked","path":"../escape.txt"}"#
                    }
                    OfflineUpstreamFixtureV0::ShadowParseErrorL3Off => "{unclosed",
                    #[cfg(not(windows))]
                    OfflineUpstreamFixtureV0::Demo(_) => {
                        return Err(HttpError::Transport(
                            "offline fixture demo journey requires the demo proposer sender".into(),
                        ))
                    }
                }
            };
            serde_json::json!({
                "id": "chatcmpl-offline-v0",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": [
                                {
                                    "id": "call_offline_v0_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": arguments_raw
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            })
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|e| HttpError::Transport(format!("offline serialization failed: {e}")))?;
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(HttpResponse {
            status: StatusCode::OK,
            headers,
            body: HttpBody::Full(Bytes::from(bytes)),
        })
    }
}

/// POSIX runtime options for `build_engine_client_with_runtime_v0`.
#[cfg(not(windows))]
pub struct EngineRuntimeOptionsV0 {
    /// Initialized state directory containing `receipt-evidence`.
    pub state_dir: std::path::PathBuf,
    /// Optional new effect root directory for 1C `write_file`.
    pub effect_root: Option<std::path::PathBuf>,
    /// Explicit 1C file-effect profile selection marker.
    pub file_effect_profile: Option<crate::effect::FileEffectProfileV0>,
    /// Maximum content length in bytes (`1..=1_048_576`) when `effect_root` is active.
    pub max_content_len_bytes: Option<u64>,
    /// Optional injected build digest or executable path.
    pub build_digest: Option<String>,
    /// Optional deterministic offline upstream fixture (`parapet-offline-upstream/v0`).
    pub offline_fixture: Option<OfflineUpstreamFixtureV0>,
    /// Optional consumer of live `EnvelopeResponseResultV0` values.
    pub result_sink: Option<Arc<dyn crate::effect::EnvelopeResultSinkV0>>,
}

#[cfg(not(windows))]
struct StaticDigestBuildIdentityV0 {
    identity:
        crate::receipt::PresentUnavailable<crate::receipt::CodeIdentity, crate::receipt::CodeError>,
}

#[cfg(not(windows))]
impl crate::effect::BuildIdentityProviderV0 for StaticDigestBuildIdentityV0 {
    fn identity(
        &self,
    ) -> crate::receipt::PresentUnavailable<crate::receipt::CodeIdentity, crate::receipt::CodeError>
    {
        self.identity.clone()
    }
}

/// Construct an `EngineUpstreamClient` with the POSIX 1B runtime envelope,
/// protected evidence store, optional 1C file-effect binding, and optional
/// deterministic offline upstream fixture.
#[cfg(not(windows))]
#[allow(clippy::type_complexity)]
pub fn build_engine_client_with_runtime_v0(
    config: Arc<Config>,
    options: EngineRuntimeOptionsV0,
) -> Result<
    (
        EngineUpstreamClient,
        Arc<crate::effect::store::ProtectedEvidenceStoreV0>,
        Option<Arc<crate::effect::file::RootCapabilityV0>>,
    ),
    StartupError,
> {
    build_engine_client_with_runtime_and_file_fixture_v0(config, options, None)
}

#[cfg(not(windows))]
#[allow(clippy::type_complexity)]
pub fn build_engine_client_with_runtime_and_upstream_v0(
    config: Arc<Config>,
    options: EngineRuntimeOptionsV0,
    upstream_table: UpstreamTable,
) -> Result<
    (
        EngineUpstreamClient,
        Arc<crate::effect::store::ProtectedEvidenceStoreV0>,
        Option<Arc<crate::effect::file::RootCapabilityV0>>,
    ),
    StartupError,
> {
    build_engine_client_with_runtime_and_file_fixture_and_upstream_v0(
        config,
        options,
        None,
        upstream_table,
    )
}

#[cfg(not(windows))]
#[allow(clippy::type_complexity)]
fn build_engine_client_with_runtime_and_file_fixture_v0(
    config: Arc<Config>,
    options: EngineRuntimeOptionsV0,
    file_fixture_override: Option<crate::effect::file::FileFixtureV0>,
) -> Result<
    (
        EngineUpstreamClient,
        Arc<crate::effect::store::ProtectedEvidenceStoreV0>,
        Option<Arc<crate::effect::file::RootCapabilityV0>>,
    ),
    StartupError,
> {
    let upstream_table = UpstreamTable::from_process_env().map_err(StartupError::UpstreamConfig)?;
    build_engine_client_with_runtime_and_file_fixture_and_upstream_v0(
        config,
        options,
        file_fixture_override,
        upstream_table,
    )
}

#[cfg(not(windows))]
#[allow(clippy::type_complexity)]
fn build_engine_client_with_runtime_and_file_fixture_and_upstream_v0(
    config: Arc<Config>,
    options: EngineRuntimeOptionsV0,
    file_fixture_override: Option<crate::effect::file::FileFixtureV0>,
    upstream_table: UpstreamTable,
) -> Result<
    (
        EngineUpstreamClient,
        Arc<crate::effect::store::ProtectedEvidenceStoreV0>,
        Option<Arc<crate::effect::file::RootCapabilityV0>>,
    ),
    StartupError,
> {
    let session_store: Option<Arc<dyn crate::session::SessionStore>> = None;

    let (l1_model, l1_scanner): (Option<Arc<dyn L1Model>>, Option<Arc<dyn L1Scanner>>) =
        if let Some(l1_config) = &config.policy.layers.l1 {
            if l1_config.specialists.is_empty() {
                (Some(Arc::new(SvmModel)), None)
            } else {
                (
                    None,
                    Some(Arc::new(EnsembleL1Scanner::new(
                        &l1_config.specialists,
                        l1_config.min_agree,
                        l1_config.generalist_solo_threshold,
                    ))),
                )
            }
        } else {
            (None, None)
        };

    let multi_turn_scanner: Option<Arc<dyn MultiTurnScanner>> = if config.policy.layers.l4.is_some()
    {
        Some(Arc::new(DefaultMultiTurnScanner::new()))
    } else {
        None
    };

    #[cfg(not(feature = "l2a"))]
    if config.policy.layers.l2a.is_some() {
        return Err(StartupError::FeatureDisabled(
            "policy.layers.l2a is configured but binary was built without \
             --features l2a. Rebuild with: cargo build --features l2a"
                .into(),
        ));
    }

    #[cfg(feature = "l2a")]
    let l2a_scanner: Option<Arc<dyn L2aScanner>> =
        if let Some(l2a_config) = &config.policy.layers.l2a {
            use crate::layers::l2a::DefaultHeuristicScanner;
            use crate::layers::l2a::DefaultL2aScanner;
            use crate::layers::l2a_model::OnnxPromptGuard;

            let classifier = OnnxPromptGuard::init(l2a_config)
                .map_err(|e| StartupError::ModelMissing(e.to_string()))?;
            let heuristics = DefaultHeuristicScanner::new();
            Some(Arc::new(DefaultL2aScanner::new(
                Box::new(classifier),
                Box::new(heuristics),
            )))
        } else {
            None
        };

    #[cfg(not(feature = "l2a"))]
    let l2a_scanner: Option<Arc<dyn L2aScanner>> = None;

    let l2a_semaphore =
        config.policy.layers.l2a.as_ref().map(|l2a_config| {
            Arc::new(tokio::sync::Semaphore::new(l2a_config.max_concurrent_scans))
        });

    let store_root = options.state_dir.join("receipt-evidence");
    let store = Arc::new(
        crate::effect::store::ProtectedEvidenceStoreV0::open(&store_root)
            .map_err(|e| StartupError::RuntimeInit(format!("store_open_failed:{e:?}")))?,
    );

    let identity: Arc<dyn crate::effect::BuildIdentityProviderV0> = match &options.build_digest {
        None => Arc::new(crate::effect::InjectedBuildIdentityV0::unavailable()),
        Some(val) => {
            let p = std::path::Path::new(val);
            if p.exists() {
                Arc::new(
                    crate::effect::InjectedBuildIdentityV0::from_executable_path(
                        env!("CARGO_PKG_VERSION"),
                        p,
                    ),
                )
            } else {
                let formatted = if val.starts_with("sha256:") {
                    val.clone()
                } else {
                    format!("sha256:{val}")
                };
                Arc::new(StaticDigestBuildIdentityV0 {
                    identity: crate::receipt::PresentUnavailable::Present(
                        crate::receipt::CodeIdentity {
                            code_version: env!("CARGO_PKG_VERSION").to_owned(),
                            code_digest: formatted,
                        },
                    ),
                })
            }
        }
    };

    let (executor_binding, root_cap, run_id) = if let Some(effect_root) = &options.effect_root {
        let profile = options
            .file_effect_profile
            .ok_or_else(|| StartupError::RuntimeInit("missing_file_effect_profile".into()))?;
        let max_bytes = options
            .max_content_len_bytes
            .ok_or_else(|| StartupError::RuntimeInit("missing_max_content_bytes".into()))?;
        use crate::effect::ResolvedPolicyProviderV0;
        let resolved_policy = crate::effect::ConfigPolicyProviderV0::new(config.clone())
            .resolve()
            .map_err(|e| StartupError::RuntimeInit(format!("policy_resolve_failed:{e:?}")))?;
        crate::effect::file::check_mandatory_profile(&resolved_policy.policy, max_bytes)
            .map_err(|e| StartupError::RuntimeInit(format!("profile_error:{e:?}")))?;
        // C2.2: exactly one envelope-allocated id, reserved before the root exists.
        let run_id = RuntimeEnvelopeV0::allocate_run_id();
        let root = match (file_fixture_override, options.offline_fixture) {
            (Some(fixture), _) => crate::effect::file::RootCapabilityV0::create_with(
                effect_root,
                run_id.clone(),
                max_bytes,
                fixture,
            ),
            #[cfg(not(windows))]
            (None, Some(OfflineUpstreamFixtureV0::Demo(demo::DemoJourneyV0::Mismatch))) => {
                use demo::EffectInterferenceV0 as _;
                let interference =
                    demo::OverwriteTargetInterferenceV0::new_default(effect_root.clone())
                        .ok_or_else(|| {
                            StartupError::RuntimeInit("demo_mismatch_interference_invalid".into())
                        })?;
                let hook: crate::effect::file::FileHookV0 = Arc::new(move |phase| {
                    if phase == crate::effect::file::FilePhaseV0::BeforeObservation {
                        interference.before_observation();
                    }
                });
                crate::effect::file::RootCapabilityV0::create_with(
                    effect_root,
                    run_id.clone(),
                    max_bytes,
                    crate::effect::file::FileFixtureV0 {
                        hook: Some(hook),
                        ..Default::default()
                    },
                )
            }
            (None, _) => crate::effect::file::RootCapabilityV0::create(
                effect_root,
                run_id.clone(),
                max_bytes,
            ),
        }
        .map_err(|e| StartupError::RuntimeInit(format!("root_error:{e:?}")))?;
        let exec = crate::effect::file::FileActionExecutorV0::new(root.clone(), &resolved_policy)
            .map_err(|e| StartupError::RuntimeInit(format!("executor_error:{e:?}")))?;
        let obs = crate::effect::file::FilesystemEffectObserverV0::new(root.clone());
        let action_id = crate::effect::ActionIdV0::parse(&format!("act-{}", run_id.as_str()))
            .ok_or_else(|| StartupError::RuntimeInit("invalid_action_id".into()))?;
        let binding = crate::effect::ExecutorBindingV0::new(
            profile,
            root.opaque(),
            action_id,
            Arc::new(exec),
            Arc::new(obs),
        );
        (Some(binding), Some(root), Some(run_id))
    } else {
        (None, None, None)
    };

    let mut parts = crate::effect::EnvelopePartsV0::new(config.clone(), store.clone(), identity);
    parts.executor = executor_binding;
    parts.fixed_run_id = run_id;
    parts.results = options.result_sink;
    let envelope = Arc::new(RuntimeEnvelopeV0::new(parts));

    let http: Arc<dyn HttpSender> = match options.offline_fixture {
        #[cfg(not(windows))]
        Some(OfflineUpstreamFixtureV0::Demo(journey)) => Arc::new(demo::DemoProposerHttpSenderV0::new(
            journey,
            demo::ManifestDemoProposerV0::new_default(),
        )),
        Some(fixture) => Arc::new(DeterministicOfflineHttpSenderV0::new(fixture)),
        None => Arc::new(ReqwestHttpSender::new(production_http_client())),
    };

    let deps = EngineDeps {
        config,
        http,
        resolver: Arc::new(EnvUpstreamResolver::from_table(upstream_table)),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        l1_scanner,
        l1_model,
        l2a_scanner,
        l2a_semaphore,
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store,
        multi_turn_scanner,
        orthogonal_sensor_router: Arc::new(NoopOrthogonalSensorRouter::new()),
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
        emit_l1_signals: false,
    };

    Ok((
        EngineUpstreamClient::new_with_envelope(deps, envelope),
        store,
        root_cap,
    ))
}

/// Alias for `build_engine_client_with_runtime_v0`.
#[cfg(not(windows))]
#[allow(clippy::type_complexity)]
pub fn build_engine_client_with_options(
    config: Arc<Config>,
    options: EngineRuntimeOptionsV0,
) -> Result<
    (
        EngineUpstreamClient,
        Arc<crate::effect::store::ProtectedEvidenceStoreV0>,
        Option<Arc<crate::effect::file::RootCapabilityV0>>,
    ),
    StartupError,
> {
    build_engine_client_with_runtime_v0(config, options)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stable C2.2 one-shot effect runtime codes.
#[cfg(not(windows))]
pub const EFFECT_STREAMING_UNSUPPORTED: &str = "effect_streaming_unsupported";
#[cfg(not(windows))]
pub const EFFECT_RUN_CONSUMED: &str = "effect_run_consumed";
#[cfg(not(windows))]
pub const EFFECT_CANDIDATE_CARDINALITY: &str = "effect_candidate_cardinality";

#[cfg(not(windows))]
fn effect_runtime_headers(code: &'static str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-parapet-action", HeaderValue::from_static("error"));
    headers.insert("x-parapet-error", HeaderValue::from_static(code));
    headers
}

/// Error response carrying a stable effect runtime code in the body message
/// and `X-Parapet-Error`.
#[cfg(not(windows))]
fn effect_runtime_error(
    adapter: &dyn ProviderAdapter,
    status: StatusCode,
    code: &'static str,
) -> ProxyResponse {
    ProxyResponse {
        status,
        headers: effect_runtime_headers(code),
        body: Body::from(adapter.serialize_error(code, status.as_u16())),
    }
}

/// Build a 503 error response for a layer runtime failure (`on_failure: closed`).
///
/// Headers: `X-Parapet-Action: error`, `X-Parapet-Layer: <layer>`, `X-Parapet-Error: <error_type>`.
fn layer_error_503(adapter: &dyn ProviderAdapter, layer: &str, error_type: &str) -> ProxyResponse {
    let body = adapter.serialize_error("service unavailable (guardrail layer error)", 503);
    let mut headers = HeaderMap::new();
    if let Ok(v) = "error".parse() { headers.insert("x-parapet-action", v); }
    if let Ok(v) = layer.parse() { headers.insert("x-parapet-layer", v); }
    if let Ok(v) = error_type.parse() { headers.insert("x-parapet-error", v); }
    ProxyResponse {
        status: StatusCode::SERVICE_UNAVAILABLE,
        headers,
        body: Body::from(body),
    }
}

fn request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

/// Handle a layer runtime error according to `on_failure` config.
///
/// Returns `Some(503 response)` in Closed mode (caller should return early).
/// Returns `None` in Open mode (caller should skip the layer and continue).
fn handle_layer_error(
    on_failure: &FailureMode,
    adapter: &dyn ProviderAdapter,
    layer: &str,
    error_type: &str,
    error: &str,
    request_id: &str,
) -> Option<ProxyResponse> {
    match on_failure {
        FailureMode::Closed => {
            tracing::error!(
                request_id = %request_id,
                layer = layer,
                failure_mode = "closed",
                error_type = error_type,
                error = error,
                "layer runtime error — blocking request (on_failure: closed)"
            );
            Some(layer_error_503(adapter, layer, error_type))
        }
        FailureMode::Open => {
            tracing::warn!(
                request_id = %request_id,
                layer = layer,
                failure_mode = "open",
                error_type = error_type,
                error = error,
                "layer runtime error — skipping layer (on_failure: open)"
            );
            None
        }
    }
}

/// Attach evidence signal headers to a response for observability/eval.
///
/// Only writes headers when evidence_count > 0 to avoid noise on clean requests.
fn attach_evidence_headers(headers: &mut HeaderMap, evidence_count: usize, evidence_categories: &[String]) {
    if evidence_count == 0 {
        return;
    }
    if let Ok(val) = evidence_count.to_string().parse() {
        headers.insert("x-parapet-evidence-count", val);
    }
    if !evidence_categories.is_empty() {
        if let Ok(val) = evidence_categories.join(",").parse() {
            headers.insert("x-parapet-evidence-categories", val);
        }
    }
}

/// Attach L2a score headers for offline threshold sweeps.
///
/// Attach L1 harness signals for eval observability.
/// Serializes the full per-message signal set so offline sweeps can
/// replay both block-threshold and delta-threshold logic exactly.
fn attach_l1_signal_headers(headers: &mut HeaderMap, signals: &Option<Vec<L1Signal>>) {
    let signals = match signals {
        Some(s) if !s.is_empty() => s,
        _ => return,
    };
    if let Ok(json) = serde_json::to_string(signals) {
        if let Ok(v) = json.parse() {
            headers.insert("x-parapet-l1-signals", v);
        }
    }
}

/// Emits both the fused score (what the threshold applies to) and the raw
/// model score (pre-fusion PG2 softmax) so sweeps can target either.
fn attach_l2a_score_headers(headers: &mut HeaderMap, has_scores: bool, max_fused: f32, max_raw: f32) {
    if !has_scores {
        return;
    }
    if let Ok(val) = format!("{max_fused:.6}").parse() {
        headers.insert("x-parapet-l2a-score", val);
    }
    if let Ok(val) = format!("{max_raw:.6}").parse() {
        headers.insert("x-parapet-l2a-pg-score", val);
    }
}

/// Check if the response has gzip Content-Encoding.
fn is_gzip(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("gzip"))
        .unwrap_or(false)
}

/// Check if the response has deflate Content-Encoding.
fn is_deflate(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("deflate"))
        .unwrap_or(false)
}

/// Maximum response body size for non-streaming collection (10 MB).
/// Prevents unbounded memory growth from large upstream responses.
const MAX_RESPONSE_BODY_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum decompressed response body size (50 MB).
/// Prevents decompression bombs from causing OOM.
const MAX_DECOMPRESSED_BYTES: u64 = 50 * 1024 * 1024;

/// Decompress a gzip-encoded body.
fn decompress_gzip(body: &Bytes) -> Result<Bytes, ProxyError> {
    let decoder = GzDecoder::new(&body[..]);
    let mut limited = decoder.take(MAX_DECOMPRESSED_BYTES + 1);
    let mut decompressed = Vec::new();
    limited
        .read_to_end(&mut decompressed)
        .map_err(|e| ProxyError::UpstreamFailure(format!("gzip decompression failed: {e}")))?;
    if decompressed.len() as u64 > MAX_DECOMPRESSED_BYTES {
        return Err(ProxyError::UpstreamFailure(format!(
            "decompressed body exceeds {} byte limit",
            MAX_DECOMPRESSED_BYTES
        )));
    }
    Ok(Bytes::from(decompressed))
}

/// Decompress a deflate-encoded body.
fn decompress_deflate(body: &Bytes) -> Result<Bytes, ProxyError> {
    let decoder = DeflateDecoder::new(&body[..]);
    let mut limited = decoder.take(MAX_DECOMPRESSED_BYTES + 1);
    let mut decompressed = Vec::new();
    limited
        .read_to_end(&mut decompressed)
        .map_err(|e| ProxyError::UpstreamFailure(format!("deflate decompression failed: {e}")))?;
    if decompressed.len() as u64 > MAX_DECOMPRESSED_BYTES {
        return Err(ProxyError::UpstreamFailure(format!(
            "decompressed body exceeds {} byte limit",
            MAX_DECOMPRESSED_BYTES
        )));
    }
    Ok(Bytes::from(decompressed))
}

/// Decompress body if Content-Encoding is set. Strips Content-Encoding and
/// Content-Length from headers (body size changed after decompression).
fn maybe_decompress(headers: &mut HeaderMap, body: Bytes) -> Result<Bytes, ProxyError> {
    let result = if is_gzip(headers) {
        decompress_gzip(&body)?
    } else if is_deflate(headers) {
        decompress_deflate(&body)?
    } else {
        return Ok(body);
    };
    headers.remove(reqwest::header::CONTENT_ENCODING);
    headers.remove(reqwest::header::CONTENT_LENGTH);
    Ok(result)
}

fn emit_destination_mismatch_warn(provider: Provider) {
    tracing::warn!(
        rule = "destination_mismatch",
        provider = ?provider,
        "upstream URL destination mismatch"
    );
}

fn build_upstream_url(
    resolver: &dyn UpstreamResolver,
    provider: Provider,
    req: &ProxyRequest,
) -> Result<String, UpstreamRefusal> {
    let original_host = OriginalHost::from_headers(&req.headers);
    let base = resolver.resolve(provider, original_host.clone())?;
    let expected_raw = match original_host {
        OriginalHost::Present(host) => resolver.base_url_for_host(host),
        OriginalHost::Absent | OriginalHost::InvalidHeader { .. } => resolver.base_url(provider),
    };

    let Ok(expected_url) = url::Url::parse(&expected_raw) else {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    };
    let base_trimmed = base.trim_end_matches('/');
    let Ok(base_url) = url::Url::parse(base_trimmed) else {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    };
    if base_url.host().is_none()
        || !base_url.username().is_empty()
        || base_url.password().is_some()
        || base_url.query().is_some()
        || base_url.fragment().is_some()
        || base_url.scheme() != expected_url.scheme()
        || base_url.host() != expected_url.host()
        || base_url.port_or_known_default() != expected_url.port_or_known_default()
    {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    }

    let raw_path = req.uri.path();
    let raw_query = req.uri.query();
    if !raw_path.starts_with('/') || raw_path.starts_with("//") {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    }

    let canonical_path = if provider == Provider::OpenAi
        && (raw_path == "/openai/v1/chat/completions"
            || raw_path.starts_with("/openai/v1/chat/completions/"))
    {
        raw_path.strip_prefix("/openai").unwrap_or(raw_path)
    } else {
        raw_path
    };

    let canonical_path_and_query = match raw_query {
        Some(q) => format!("{canonical_path}?{q}"),
        None => canonical_path.to_string(),
    };

    let base_path = base_url.path().trim_end_matches('/');
    let effective_base = if (base_path == "/v1" || base_path.ends_with("/v1"))
        && (canonical_path == "/v1" || canonical_path.starts_with("/v1/"))
    {
        base_trimmed.strip_suffix("/v1").unwrap_or(base_trimmed)
    } else {
        base_trimmed
    };

    let joined = format!("{effective_base}{canonical_path_and_query}");
    let Ok(joined_url) = url::Url::parse(&joined) else {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    };
    if joined_url.scheme() != base_url.scheme()
        || joined_url.host() != base_url.host()
        || joined_url.port_or_known_default() != base_url.port_or_known_default()
        || !joined_url.username().is_empty()
        || joined_url.password().is_some()
        || joined_url.fragment().is_some()
    {
        emit_destination_mismatch_warn(provider);
        return Err(UpstreamRefusal::DestinationMismatch);
    }

    Ok(joined)
}

fn upstream_refusal_response(
    adapter: &dyn ProviderAdapter,
    resolver: &dyn UpstreamResolver,
    refusal: UpstreamRefusal,
) -> ProxyResponse {
    let (status, code, message) = match refusal {
        UpstreamRefusal::HostNotAllowed => (
            StatusCode::BAD_REQUEST,
            "parapet_upstream_host_not_allowed",
            "parapet_upstream_host_not_allowed: X-Parapet-Original-Host is not allowlisted or is malformed; configure PARAPET_EXTRA_HOSTS to allow additional upstream hosts".to_string(),
        ),
        UpstreamRefusal::ConfigInvalid => {
            let msg = match resolver.config_error() {
                Some(err) => format!(
                    "parapet_upstream_config_invalid: {}",
                    err.key.client_key_name()
                ),
                None => "parapet_upstream_config_invalid".to_string(),
            };
            (
                StatusCode::BAD_GATEWAY,
                "parapet_upstream_config_invalid",
                msg,
            )
        }
        UpstreamRefusal::DestinationMismatch => (
            StatusCode::BAD_GATEWAY,
            "parapet_upstream_destination_mismatch",
            "parapet_upstream_destination_mismatch: resolved URL origin did not match validated upstream origin".to_string(),
        ),
    };
    let body = adapter.serialize_error(&message, status.as_u16());
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert("x-parapet-error", HeaderValue::from_static(code));
    ProxyResponse {
        status,
        headers,
        body: Body::from(body),
    }
}

fn sanitize_location_host(headers: &HeaderMap) -> String {
    let Some(raw) = headers
        .get(reqwest::header::LOCATION)
        .and_then(|v| v.to_str().ok())
    else {
        return "none".to_string();
    };
    let Some(host) = url::Url::parse(raw)
        .ok()
        .and_then(|u| u.host_str().map(str::to_owned))
    else {
        return "relative_or_unparseable".to_string();
    };
    let mut out = String::new();
    for ch in host.chars().take(253) {
        if ch.is_control() {
            for esc in ch.escape_default() {
                out.push(esc);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn upstream_redirect_refused_response(
    adapter: &dyn ProviderAdapter,
    request_id: &str,
    status: StatusCode,
    upstream_headers: &HeaderMap,
) -> ProxyResponse {
    let location_host = sanitize_location_host(upstream_headers);
    tracing::warn!(
        request_id = %request_id,
        status = status.as_u16(),
        location_host = %location_host,
        rule = "parapet_upstream_redirect_refused",
        "upstream redirect refused"
    );
    let body = adapter.serialize_error(
        "parapet_upstream_redirect_refused: upstream redirects are not followed",
        StatusCode::BAD_GATEWAY.as_u16(),
    );
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert(
        "x-parapet-error",
        HeaderValue::from_static("parapet_upstream_redirect_refused"),
    );
    ProxyResponse {
        status: StatusCode::BAD_GATEWAY,
        headers,
        body: Body::from(body),
    }
}

fn extract_model(body: &Bytes) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(String::from))
        .unwrap_or_default()
}

/// Telemetry counter for streaming responses where upstream still returned
/// compressed SSE and L5a streaming redaction was skipped.
static COMPRESSED_SSE_REDACTION_SKIP_COUNT: AtomicU64 = AtomicU64::new(0);

fn increment_streaming_compressed_sse_skip_count() -> u64 {
    COMPRESSED_SSE_REDACTION_SKIP_COUNT.fetch_add(1, Ordering::Relaxed) + 1
}

#[cfg(test)]
fn streaming_compressed_sse_skip_count() -> u64 {
    COMPRESSED_SSE_REDACTION_SKIP_COUNT.load(Ordering::Relaxed)
}

fn is_streaming_request(headers: &HeaderMap, body: &Bytes) -> bool {
    // Precedence decision: if caller advertises SSE in `Accept`, treat as
    // streaming even when JSON body has `"stream": false`.
    // This keeps behavior compatible with SSE clients that rely on headers.
    if accepts_event_stream(headers) {
        return true;
    }

    let Ok(json) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    json.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn accepts_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(',').any(|part| {
                let token = part.trim();
                token.eq_ignore_ascii_case("text/event-stream")
                    || token
                        .to_ascii_lowercase()
                        .starts_with("text/event-stream;")
            })
        })
        .unwrap_or(false)
}

fn force_identity_encoding_for_streaming(headers: &mut HeaderMap) {
    headers.insert(
        reqwest::header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );
}

/// Extract session ID from W3C Baggage header.
///
/// The SDK sets `baggage: user_id=<value>,role=<value>`. We use `user_id` as
/// the session identifier when present. If no user_id in baggage, generate an
/// ephemeral session ID from a random UUID (best-effort grouping).
///
/// W3C Baggage format: `key1=value1,key2=value2`
/// Values are percent-encoded per RFC 3986.
#[allow(dead_code)] // Sprint 2: session identity
fn extract_session_id(headers: &HeaderMap) -> String {
    if let Some(baggage) = headers.get("baggage").and_then(|v| v.to_str().ok()) {
        for pair in baggage.split(',') {
            let pair = pair.trim();
            if let Some((key, value)) = pair.split_once('=') {
                if key.trim() == "user_id" {
                    let decoded = percent_decode(value.trim());
                    if !decoded.is_empty() {
                        return format!("user:{decoded}");
                    }
                }
            }
        }
    }

    // No user_id in baggage -- generate ephemeral session ID
    format!("ephemeral:{}", Uuid::new_v4())
}

/// Simple percent-decode (handles %XX sequences).
#[allow(dead_code)] // Sprint 2: session identity
fn percent_decode(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

fn stream_body(
    stream: impl Stream<Item = Result<Bytes, HttpError>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Bytes> + Send>> {
    Box::pin(stream.filter_map(|item| async move {
        match item {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                tracing::warn!(error = %e, "dropped error chunk from upstream stream");
                None
            }
        }
    }))
}

/// Compress current request messages into history entries for session state.
///
/// Each message becomes a `HistoryEntry` with a summary of the first 200 chars
/// and tool call names. This keeps session state lightweight while providing
/// enough context for L4 detectors.
#[allow(dead_code)] // Sprint 2: session state
fn compress_to_history(messages: &[crate::message::Message]) -> Vec<crate::session::HistoryEntry> {
    messages
        .iter()
        .map(|msg| {
            let role = match &msg.role {
                crate::message::Role::System => "system",
                crate::message::Role::User => "user",
                crate::message::Role::Assistant => "assistant",
                crate::message::Role::Tool => "tool",
            };
            let summary: String = msg.content.chars().take(200).collect();
            let tool_names: Vec<String> = msg.tool_calls.iter().map(|tc| tc.name.clone()).collect();

            crate::session::HistoryEntry {
                role: role.to_string(),
                content_summary: summary,
                timestamp: chrono::Utc::now(),
                trust_level: msg.trust.clone(),
                tool_calls_summary: tool_names,
            }
        })
        .collect()
}

/// Update session state after a request completes.
///
/// Adds compressed history entries for the current messages,
/// merges any L4 flags, and stores back to the session store.
#[allow(dead_code)] // Sprint 2: session state
fn update_session_after_request(
    store: &dyn crate::session::SessionStore,
    session_id: &str,
    messages: &[crate::message::Message],
    max_history: usize,
) {
    let mut session = store
        .get(session_id)
        .unwrap_or_else(|| crate::session::SessionState::new(session_id));

    for entry in compress_to_history(messages) {
        session.push_history(entry, max_history);
    }

    store.update(session);
}

fn apply_streaming_redaction(
    input: impl Stream<Item = Bytes> + Send + 'static,
    scanner: Arc<dyn OutputScanner>,
    config: Arc<Config>,
    window_override: Option<usize>,
) -> Pin<Box<dyn Stream<Item = Bytes> + Send>> {
    let window = window_override.unwrap_or_else(|| compute_redaction_window(&config));
    let state = (Box::pin(input) as Pin<Box<dyn Stream<Item = Bytes> + Send>>, String::new());

    let stream = futures_util::stream::unfold(state, move |(mut input, mut buffer)| {
        let scanner = scanner.clone();
        let config = config.clone();
        async move {
            loop {
                match input.next().await {
                    Some(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&chunk_str);

                        let redacted = scanner.scan_and_redact(&buffer, &config).content;
                        let redacted_len = redacted.chars().count();
                        if redacted_len > window {
                            let cut = redacted_len - window;
                            let emit = head_chars(&redacted, cut);
                            // Keep the redacted tail so boundary matches stay redacted.
                            buffer = tail_chars(&redacted, window);
                            if !emit.is_empty() {
                                return Some((Bytes::from(emit), (input, buffer)));
                            }
                        }
                    }
                    None => {
                        if buffer.is_empty() {
                            return None;
                        }
                        let redacted = scanner.scan_and_redact(&buffer, &config).content;
                        buffer.clear();
                        if redacted.is_empty() {
                            return None;
                        }
                        return Some((Bytes::from(redacted), (input, buffer)));
                    }
                }
            }
        }
    })
    .filter(|b| futures_util::future::ready(!b.is_empty()));

    Box::pin(stream)
}

fn compute_redaction_window(config: &Config) -> usize {
    let mut max_len = 0usize;
    for token in &config.policy.canary_tokens {
        max_len = max_len.max(token.len());
    }
    for pattern in &config.policy.sensitive_patterns {
        // Regex patterns can match strings much longer than the pattern
        // string itself (e.g. `sk-[a-zA-Z0-9]{20,}` is 22 chars but
        // matches 23+ char strings). Use 4x pattern length as heuristic.
        max_len = max_len.max(pattern.pattern.len().saturating_mul(4));
    }
    max_len = max_len.saturating_add(32);
    max_len.max(1024)
}

fn tail_chars(input: &str, count: usize) -> String {
    if count == 0 {
        return String::new();
    }
    let len = input.chars().count();
    if len <= count {
        return input.to_string();
    }
    input.chars().skip(len - count).collect()
}

fn head_chars(input: &str, count: usize) -> String {
    if count == 0 {
        return String::new();
    }
    let len = input.chars().count();
    if count >= len {
        return input.to_string();
    }
    input.chars().take(count).collect()
}

// ---------------------------------------------------------------------------
// X-Guard-Trust header parsing (SDK -> Engine)
// ---------------------------------------------------------------------------

/// Compact span from the X-Guard-Trust header JSON.
#[derive(serde::Deserialize)]
struct RawTrustHeaderSpan {
    s: usize,
    e: usize,
    src: Option<String>,
}

/// A message's content location within the serialized request body.
struct ContentRegion {
    msg_idx: usize,
    body_start: usize,
    body_end: usize,
    escaped_interior: String,
}

/// Parse X-Guard-Trust header and attach trust spans to messages.
///
/// Format: `inline:<base64-encoded JSON>`
/// JSON: `[{"s": <start>, "e": <end>, "src": "<source>"}, ...]`
///
/// Byte offsets in the header reference positions in the SERIALIZED request body
/// (JSON-escaped). This function:
/// 1. Builds an offset map of where each message's content appears in the body,
///    using sequential search to handle repeated content correctly.
/// 2. Maps body-level offsets to escaped-interior-relative offsets.
/// 3. Converts escaped-interior offsets to raw content offsets via
///    `escaped_byte_to_raw` (handles `\"`, `\\`, `\n`, etc.).
/// 4. Validates final offsets and logs warnings for unmapped/invalid spans.
///
/// If header parsing fails, logs a warning and returns (graceful degradation).
fn parse_trust_header(headers: &HeaderMap, body: &[u8], messages: &mut [crate::message::Message]) {
    let header_value = match headers.get("x-guard-trust").and_then(|v| v.to_str().ok()) {
        Some(v) => v,
        None => return,
    };

    let encoded = match header_value.strip_prefix("inline:") {
        Some(e) => e,
        None => {
            tracing::warn!("X-Guard-Trust header has unknown format (expected 'inline:...')");
            return;
        }
    };

    let decoded = match base64::engine::general_purpose::STANDARD.decode(encoded) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(error = %e, "X-Guard-Trust header base64 decode failed");
            return;
        }
    };

    let spans: Vec<RawTrustHeaderSpan> = match serde_json::from_slice(&decoded) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "X-Guard-Trust header JSON parse failed");
            return;
        }
    };

    if spans.is_empty() {
        return;
    }

    let body_str = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Phase 1: Build offset map for all messages.
    // Search is scoped to the "messages" array to avoid matching "content"
    // keys in unrelated objects (e.g. metadata, tool definitions).
    // Within that array, anchors to "content" JSON keys so non-content
    // fields are never matched. Sequential search_from handles repeats.
    let (msg_array_start, msg_array_end) = match find_messages_array_range(body_str) {
        Some(range) => range,
        None => {
            tracing::debug!("trust header: could not locate messages array in body");
            return;
        }
    };

    let mut regions: Vec<ContentRegion> = Vec::new();
    let mut search_from: usize = msg_array_start;

    for (idx, msg) in messages.iter().enumerate() {
        if msg.content.is_empty() {
            continue;
        }

        let escaped_content = match serde_json::to_string(&msg.content) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let escaped_interior = escaped_content[1..escaped_content.len() - 1].to_string();

        if let Some(body_start) =
            find_content_field_value(body_str, search_from, msg_array_end, &escaped_interior)
        {
            let body_end = body_start + escaped_interior.len();
            regions.push(ContentRegion {
                msg_idx: idx,
                body_start,
                body_end,
                escaped_interior,
            });
            search_from = body_end;
        } else {
            tracing::debug!(
                msg_idx = idx,
                "trust header: could not locate message content in serialized body"
            );
        }
    }

    // Phase 2: Map each raw span to message-relative offsets.
    let mut unmapped_count: usize = 0;

    for span in &spans {
        if span.s >= span.e {
            tracing::warn!(
                start = span.s,
                end = span.e,
                "trust header: span has inverted or empty byte range, skipping"
            );
            unmapped_count += 1;
            continue;
        }

        let mut matched = false;

        for region in &regions {
            if span.s < region.body_end && span.e > region.body_start {
                // Offsets within the escaped interior
                let esc_start = span.s.saturating_sub(region.body_start);
                let esc_end = span.e.min(region.body_end) - region.body_start;

                if esc_start >= esc_end {
                    continue;
                }

                // Convert escaped-interior offsets to raw content offsets
                let raw_start = escaped_byte_to_raw(&region.escaped_interior, esc_start);
                let raw_end = escaped_byte_to_raw(&region.escaped_interior, esc_end);

                let content_len = messages[region.msg_idx].content.len();
                if raw_start >= raw_end || raw_end > content_len {
                    tracing::warn!(
                        msg_idx = region.msg_idx,
                        raw_start,
                        raw_end,
                        content_len,
                        "trust header: span maps to invalid range in message content, skipping"
                    );
                    continue;
                }

                messages[region.msg_idx].trust_spans.push(
                    crate::trust::TrustSpan::untrusted(
                        raw_start,
                        raw_end,
                        span.src.clone().unwrap_or_else(|| "unknown".to_string()),
                    ),
                );
                matched = true;
            }
        }

        if !matched {
            tracing::warn!(
                start = span.s,
                end = span.e,
                "trust header: span does not overlap any message content"
            );
            unmapped_count += 1;
        }
    }

    if unmapped_count > 0 {
        tracing::warn!(
            unmapped = unmapped_count,
            total = spans.len(),
            "trust header: some spans could not be mapped to messages"
        );
    }
}

/// Find the byte range of the top-level `"messages"` JSON array in the body.
///
/// Returns `(start, end)` where `start` is the `[` position and `end` is
/// one past the matching `]`. Only matches `"messages"` at brace depth 1
/// (directly inside the root JSON object). Handles nested arrays/objects
/// and JSON strings correctly via depth tracking with string-escape awareness.
fn find_messages_array_range(body: &str) -> Option<(usize, usize)> {
    let key = b"\"messages\"";

    // Walk the body tracking brace/bracket depth and string state.
    // Only consider a "messages" key match when brace_depth == 1
    // (i.e. we are directly inside the root JSON object).
    let bytes = body.as_bytes();
    let mut brace_depth: usize = 0;
    let mut bracket_depth: usize = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut key_pos: Option<usize> = None;

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        if escape_next {
            escape_next = false;
            i += 1;
            continue;
        }

        if in_string {
            match b {
                b'\\' => escape_next = true,
                b'"' => in_string = false,
                _ => {}
            }
            i += 1;
            continue;
        }

        match b {
            b'"' => {
                // Check if this starts the "messages" key at root level
                if brace_depth == 1 && bracket_depth == 0
                    && i + key.len() <= bytes.len()
                    && &bytes[i..i + key.len()] == key
                {
                    key_pos = Some(i);
                    i += key.len();
                    // in_string is still false because we consumed the
                    // entire quoted key including the closing quote
                    continue;
                }
                in_string = true;
            }
            b'{' => brace_depth += 1,
            b'}' => brace_depth = brace_depth.saturating_sub(1),
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            _ => {}
        }
        i += 1;
    }

    let key_pos = key_pos?;
    let after_key = key_pos + key.len();

    // Skip whitespace, expect colon
    let rest = body[after_key..].trim_start();
    if !rest.starts_with(':') {
        return None;
    }
    let after_colon = after_key + (body[after_key..].len() - rest.len()) + 1;

    // Skip whitespace, expect opening bracket
    let rest2 = body[after_colon..].trim_start();
    if !rest2.starts_with('[') {
        return None;
    }
    let array_start = after_colon + (body[after_colon..].len() - rest2.len());

    // Walk forward to find the matching closing bracket
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, b) in body[array_start..].bytes().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            match b {
                b'\\' => escape_next = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some((array_start, array_start + i + 1));
                }
            }
            _ => {}
        }
    }

    None
}

/// Find the byte offset of a "content" field's string value within `body[from..to]`.
///
/// Searches for `"content"` JSON keys, skips whitespace around the colon,
/// then checks if the string value interior matches `escaped_interior`.
/// Returns the body-level byte offset of the value interior start (after
/// the opening quote).
///
/// The `to` bound scopes the search (e.g. to within the messages array)
/// so that "content" keys in unrelated objects are never matched.
fn find_content_field_value(
    body: &str,
    from: usize,
    to: usize,
    escaped_interior: &str,
) -> Option<usize> {
    let key = "\"content\"";
    let mut pos = from;

    while pos < to {
        let hit = body[pos..to].find(key)?;
        let after_key = pos + hit + key.len();
        if after_key >= to {
            return None;
        }

        // Skip whitespace, expect colon
        let rest = &body[after_key..to];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with(':') {
            pos = after_key;
            continue;
        }
        let after_colon = after_key + (rest.len() - trimmed.len()) + 1;
        if after_colon >= to {
            return None;
        }

        // Skip whitespace after colon, expect opening quote
        let rest2 = &body[after_colon..to];
        let trimmed2 = rest2.trim_start();
        if !trimmed2.starts_with('"') {
            // Value is not a string (null, array, etc.) — skip this key
            pos = after_colon;
            continue;
        }
        let value_interior_start = after_colon + (rest2.len() - trimmed2.len()) + 1;
        if value_interior_start >= to {
            return None;
        }

        if body[value_interior_start..to].starts_with(escaped_interior) {
            return Some(value_interior_start);
        }

        // This "content" key didn't match; keep searching
        pos = value_interior_start;
    }

    None
}

/// Convert a byte position in JSON-escaped string space to the corresponding
/// position in raw (unescaped) string space.
///
/// JSON escape sequences (`\"`, `\\`, `\n`, `\uXXXX`, etc.) occupy more bytes
/// in escaped form than in raw form. Walks the escaped string up to
/// `target_esc_pos` and counts equivalent raw bytes.
///
/// Handles UTF-16 surrogate pairs (`\uD83D\uDE00` → single 4-byte char).
fn escaped_byte_to_raw(escaped: &str, target_esc_pos: usize) -> usize {
    let bytes = escaped.as_bytes();
    let mut esc_pos = 0;
    let mut raw_pos = 0;

    while esc_pos < target_esc_pos && esc_pos < bytes.len() {
        if bytes[esc_pos] == b'\\' && esc_pos + 1 < bytes.len() {
            let next = bytes[esc_pos + 1];
            if matches!(next, b'"' | b'\\' | b'/' | b'n' | b'r' | b't' | b'b' | b'f') {
                // 2-byte escape → 1 raw byte
                if esc_pos + 2 > target_esc_pos {
                    break; // target lands mid-escape; snap to start
                }
                esc_pos += 2;
                raw_pos += 1;
            } else if next == b'u' && esc_pos + 5 < bytes.len() {
                // \uXXXX — check for surrogate pair
                if esc_pos + 6 > target_esc_pos {
                    break;
                }
                let hex_str = std::str::from_utf8(&bytes[esc_pos + 2..esc_pos + 6])
                    .unwrap_or("0000");
                let code_unit = u32::from_str_radix(hex_str, 16).unwrap_or(0);

                // High surrogate (U+D800..U+DBFF) followed by \uXXXX low surrogate?
                if (0xD800..=0xDBFF).contains(&code_unit)
                    && esc_pos + 11 < bytes.len()
                    && bytes[esc_pos + 6] == b'\\'
                    && bytes[esc_pos + 7] == b'u'
                {
                    let low_hex = std::str::from_utf8(&bytes[esc_pos + 8..esc_pos + 12])
                        .unwrap_or("0000");
                    let low_unit = u32::from_str_radix(low_hex, 16).unwrap_or(0);

                    if (0xDC00..=0xDFFF).contains(&low_unit) {
                        // Valid surrogate pair → single code point
                        if esc_pos + 12 > target_esc_pos {
                            break;
                        }
                        let cp = 0x10000
                            + ((code_unit - 0xD800) << 10)
                            + (low_unit - 0xDC00);
                        let char_len = char::from_u32(cp).map_or(4, |c| c.len_utf8());
                        esc_pos += 12;
                        raw_pos += char_len;
                    } else {
                        // High surrogate not followed by valid low — replacement char
                        esc_pos += 6;
                        raw_pos += 3;
                    }
                } else {
                    // Regular BMP code point or lone low surrogate
                    let char_len = char::from_u32(code_unit).map_or(3, |c| c.len_utf8());
                    esc_pos += 6;
                    raw_pos += char_len;
                }
            } else {
                // Unknown escape — treat backslash as literal
                esc_pos += 1;
                raw_pos += 1;
            }
        } else {
            esc_pos += 1;
            raw_pos += 1;
        }
    }

    raw_pos
}

fn apply_l3_outbound_openai(
    response: &mut serde_json::Value,
    evaluator: &dyn ConstraintEvaluator,
    config: &Config,
    block_action: &str,
) -> (Result<(), String>, Vec<String>) {
    let choices = match response.get_mut("choices").and_then(|c| c.as_array_mut()) {
        Some(c) => c,
        None => return (Ok(()), Vec::new()),
    };

    let mut all_blocked_tools = Vec::new();

    for choice in choices {
        let message = match choice.get_mut("message") {
            Some(m) => m,
            None => continue,
        };

        let tool_calls_val = match message.get_mut("tool_calls") {
            Some(tc) => tc,
            None => continue,
        };

        let tool_calls = match tool_calls_val.as_array_mut() {
            Some(arr) => arr,
            None => continue,
        };

        let mut allowed_calls = Vec::new();
        let mut blocked_reasons = Vec::new();

        for tc_val in tool_calls.iter() {
            let tc = match parse_openai_tool_call(tc_val) {
                Ok(t) => t,
                Err(e) => {
                    blocked_reasons.push(format!("invalid tool call: {e}"));
                    continue;
                }
            };
            match evaluator.evaluate_tool_calls(&[tc.clone()], config).first() {
                Some(ToolCallVerdict::Allow) | None => allowed_calls.push(tc_val.clone()),
                Some(ToolCallVerdict::Block { tool_name, reason }) => {
                    all_blocked_tools.push(tool_name.clone());
                    blocked_reasons.push(format!("tool '{tool_name}' blocked: {reason}"));
                }
            }
        }

        if !blocked_reasons.is_empty() {
            if block_action == "error" {
                return (Err(blocked_reasons.join("; ")), all_blocked_tools);
            }

            // Rewrite: remove blocked tool calls and inject refusal text
            *tool_calls = allowed_calls;
            let refusal = blocked_reasons.join(" ");

            match message.get_mut("content") {
                Some(serde_json::Value::String(s)) => {
                    if s.is_empty() {
                        *s = refusal;
                    } else {
                        s.push_str("\n");
                        s.push_str(&refusal);
                    }
                }
                Some(serde_json::Value::Null) | None => {
                    message["content"] = serde_json::Value::String(refusal);
                }
                Some(other) => {
                    // Non-string content: overwrite with refusal to keep format valid
                    *other = serde_json::Value::String(refusal);
                }
            }
        }
    }

    (Ok(()), all_blocked_tools)
}

fn apply_l3_outbound_anthropic(
    response: &mut serde_json::Value,
    evaluator: &dyn ConstraintEvaluator,
    config: &Config,
    block_action: &str,
) -> (Result<(), String>, Vec<String>) {
    let content = match response.get_mut("content").and_then(|c| c.as_array_mut()) {
        Some(c) => c,
        None => return (Ok(()), Vec::new()),
    };

    let mut new_content = Vec::new();
    let mut blocked_reasons = Vec::new();
    let mut blocked_tools = Vec::new();

    for block in content.iter() {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if block_type != "tool_use" {
            new_content.push(block.clone());
            continue;
        }

        let tc = match parse_anthropic_tool_call(block) {
            Ok(t) => t,
            Err(e) => {
                blocked_reasons.push(format!("invalid tool call: {e}"));
                continue;
            }
        };

        match evaluator.evaluate_tool_calls(&[tc.clone()], config).first() {
            Some(ToolCallVerdict::Allow) | None => new_content.push(block.clone()),
            Some(ToolCallVerdict::Block { tool_name, reason }) => {
                blocked_tools.push(tool_name.clone());
                blocked_reasons.push(format!("tool '{tool_name}' blocked: {reason}"));
            }
        }
    }

    if !blocked_reasons.is_empty() {
        if block_action == "error" {
            return (Err(blocked_reasons.join("; ")), blocked_tools);
        }

        let refusal = blocked_reasons.join(" ");
        new_content.push(serde_json::json!({
            "type": "text",
            "text": refusal
        }));
    }

    *content = new_content;
    (Ok(()), blocked_tools)
}

fn parse_openai_tool_call(value: &serde_json::Value) -> Result<ToolCall, String> {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing tool_calls[].id".to_string())?
        .to_string();
    let function = value
        .get("function")
        .ok_or_else(|| "missing tool_calls[].function".to_string())?;
    let name = function
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing tool_calls[].function.name".to_string())?
        .to_string();

    let arguments = parse_openai_arguments(function.get("arguments"))?;

    Ok(ToolCall { id, name, arguments })
}

fn parse_openai_arguments(value: Option<&serde_json::Value>) -> Result<serde_json::Value, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(serde_json::json!({})),
        Some(serde_json::Value::String(s)) => {
            if s.is_empty() {
                return Ok(serde_json::json!({}));
            }
            serde_json::from_str(s).map_err(|e| format!("arguments not valid JSON: {e}"))
        }
        Some(serde_json::Value::Object(_)) => Ok(value.unwrap().clone()),
        Some(other) => Err(format!("arguments has unexpected type: {other}")),
    }
}

fn parse_anthropic_tool_call(value: &serde_json::Value) -> Result<ToolCall, String> {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing content[].id".to_string())?
        .to_string();
    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing content[].name".to_string())?
        .to_string();
    let arguments = match value.get("input") {
        Some(serde_json::Value::Null) | None => serde_json::json!({}),
        Some(v) => v.clone(),
    };

    Ok(ToolCall { id, name, arguments })
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(windows)))]
mod n3_same_builder_tests {
    use super::*;
    use crate::engine::demo::{
        load_verified_manifest, sha256_hex, CompiledDemoFixturesV0, DemoFixtureSourceV0,
        DemoJourneyV0,
    };
    use serde_json::Value;

    #[tokio::test]
    async fn n3_j4_with_interference_hook_disabled_verifies() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        std::fs::create_dir(&state).unwrap();
        std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o700)).unwrap();
        let effect_root = dir.path().join("root");

        let config_bytes = CompiledDemoFixturesV0
            .asset("config_demo_v0.yaml")
            .expect("embedded demo config");
        let config_path = dir.path().join("config_demo_v0.yaml");
        std::fs::write(&config_path, config_bytes).unwrap();
        let config = Arc::new(
            crate::config::load_config(&crate::config::FileSource { path: config_path }).unwrap(),
        );

        let (client, store, root) = build_engine_client_with_runtime_and_file_fixture_v0(
            config,
            EngineRuntimeOptionsV0 {
                state_dir: state,
                effect_root: Some(effect_root.clone()),
                file_effect_profile: Some(crate::effect::FileEffectProfileV0::select_explicitly()),
                max_content_len_bytes: Some(4096),
                build_digest: None,
                offline_fixture: Some(OfflineUpstreamFixtureV0::Demo(DemoJourneyV0::Mismatch)),
                result_sink: None,
            },
            Some(crate::effect::file::FileFixtureV0::default()),
        )
        .expect("builds mismatch engine with default FileFixtureV0");
        let root = root.expect("effect root created");
        let run_id = root.run_id().clone();

        let request_bytes = CompiledDemoFixturesV0
            .asset("request_mismatch.json")
            .expect("embedded mismatch request");
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert(
            "x-parapet-original-host",
            HeaderValue::from_static("api.openai.com"),
        );
        let resp = client
            .forward(
                Provider::OpenAi,
                ProxyRequest {
                    method: Method::POST,
                    uri: "/v1/chat/completions".parse().unwrap(),
                    headers,
                    body: Bytes::from_static(request_bytes),
                },
            )
            .await
            .expect("n3 forward succeeds");
        assert_eq!(resp.status, StatusCode::OK);

        let view = crate::effect::store::RedactedEvidenceReaderV0::new(&store).view(&run_id, None);
        assert!(view.reader_code.is_none());
        assert!(view.projection_code.is_none());
        let proj: Value =
            serde_json::from_slice(&view.projection_bytes.expect("projection bytes")).unwrap();
        assert_eq!(proj["policy"]["decision"], "allow");
        assert_eq!(proj["effect"]["attempt_outcome"], "completed");
        assert_eq!(proj["verification"]["outcome"], "verified");
        assert!(proj["verification"]["reason_code"].is_null());

        let manifest = load_verified_manifest(&CompiledDemoFixturesV0).expect("manifest verifies");
        let entry = manifest
            .journeys
            .iter()
            .find(|j| j.journey == DemoJourneyV0::Mismatch.name())
            .expect("mismatch journey");
        let expected_sha = entry
            .expected_file_sha256
            .as_deref()
            .expect("expected file sha256");
        let target_file = effect_root.join(&entry.expected_target_basename);
        let disk_bytes = std::fs::read(&target_file).expect("target file written");
        assert_eq!(sha256_hex(&disk_bytes), expected_sha);
    }
}
