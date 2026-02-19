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

use crate::config::{Config, L2aMode, PatternAction};
use crate::constraint::{ConstraintEvaluator, DslConstraintEvaluator, ToolCallVerdict};
use crate::layers::l1::{DefaultL1Scanner, L1Scanner, L1Verdict};
use crate::layers::l2a::L2aScanner;
use crate::layers::l3_inbound::{DefaultInboundScanner, InboundScanner, InboundVerdict};
use crate::config::L4Mode;
use crate::layers::l4::{DefaultMultiTurnScanner, L4Verdict, MultiTurnScanner};
use crate::layers::l5a::{L5aScanner, OutputScanner};
use crate::message::ToolCall;
use crate::normalize::{neutralize_role_markers, normalize_messages_with_spans, L0Normalizer, Normalizer};
use crate::provider::{adapter_for, ProviderAdapter, ProviderType};
use crate::proxy::{ProxyError, ProxyRequest, ProxyResponse, Provider, UpstreamClient};
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
use axum::http::{HeaderMap, Method, StatusCode};
#[cfg(test)]
use axum::http::Uri;
use futures_util::stream::{Stream, StreamExt};
use futures_util::TryStreamExt;
use std::pin::Pin;
use std::sync::Arc;
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
const KNOWN_LLM_HOSTS: &[&str] = &[
    "api.openai.com",
    "api.anthropic.com",
    "api.cerebras.ai",
    "api.groq.com",
    "generativelanguage.googleapis.com",
    "api.together.xyz",
];

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
    /// L2a configured in YAML but binary built without `--features l2a`.
    FeatureDisabled(String),
    /// L2a configured but model files missing or invalid.
    ModelMissing(String),
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupError::FeatureDisabled(msg) => write!(f, "{msg}"),
            StartupError::ModelMissing(msg) => write!(f, "{msg}"),
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
    pub l2a_scanner: Option<Arc<dyn L2aScanner>>,
    pub l2a_semaphore: Option<Arc<tokio::sync::Semaphore>>,
    pub inbound_scanner: Arc<dyn InboundScanner>,
    pub constraint_evaluator: Arc<dyn ConstraintEvaluator>,
    pub output_scanner: Arc<dyn OutputScanner>,
    pub session_store: Option<Arc<dyn crate::session::SessionStore>>,
    pub multi_turn_scanner: Option<Arc<dyn MultiTurnScanner>>,
    pub signal_extractor: Arc<dyn SignalExtractor>,
    pub verdict_combiner: Arc<dyn VerdictCombiner>,
}

// ---------------------------------------------------------------------------
// EngineUpstreamClient
// ---------------------------------------------------------------------------

/// Upstream client that runs the full Parapet pipeline.
pub struct EngineUpstreamClient {
    deps: EngineDeps,
}

impl EngineUpstreamClient {
    pub fn new_with(deps: EngineDeps) -> Self {
        Self { deps }
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
            request_id: Uuid::new_v4().to_string(),
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

        // 3) L0 normalization (if enabled)
        if let Some(layer) = &self.deps.config.policy.layers.l0 {
            if layer.mode == "sanitize" {
                normalize_messages_with_spans(self.deps.normalizer.as_ref(), &mut messages);
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
            }
        }

        // Collect-then-decide: run all inbound layers before deciding.
        // Store the first block response; later layers still run for signal collection.
        let mut pending_block: Option<ProxyResponse> = None;

        // Hoisted layer results for the verdict processor.
        let mut l1_result_opt: Option<crate::layers::l1::L1Result> = None;
        let mut inbound_result_opt: Option<crate::layers::l3_inbound::InboundResult> = None;
        let mut l4_result_opt: Option<crate::layers::l4::L4Result> = None;

        // 3.5) L1 lightweight classifier (if enabled)
        if let (Some(scanner), Some(l1_config)) = (&self.deps.l1_scanner, &self.deps.config.policy.layers.l1) {
            let l1_start = Instant::now();
            let l1_result = scanner.scan(&messages, l1_config);
            let l1_ms = l1_start.elapsed().as_secs_f64() * 1000.0;
            l1_result_opt = Some(l1_result);
            let l1_result = l1_result_opt.as_ref().unwrap();

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
                    // Shadow mode: log but don't block
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

        // 3.7) L2a data payload detection (if enabled)
        //
        // Runs PG2 + heuristic fusion on untrusted data segments.
        // Timeout and concurrency are engine-level concerns; the scanner is synchronous.
        // Failure behavior depends on mode: Shadow=fail-open, Block=fail-closed.
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
                        let l2a_ms = l2a_start.elapsed().as_secs_f64() * 1000.0;
                        tracing::error!(
                            request_id = %ctx.request_id,
                            layer = "L2a",
                            error = %scan_err,
                            latency_ms = l2a_ms,
                            "L2a scan error"
                        );
                        if l2a_config.mode == L2aMode::Block {
                            l2a_fail_closed_block(
                                &mut pending_block, adapter.as_ref(), "scan error"
                            );
                        }
                    }
                    Ok(Err(join_err)) => {
                        // Scan task panicked.
                        tracing::error!(
                            request_id = %ctx.request_id,
                            layer = "L2a",
                            error = %join_err,
                            "L2a scan task panicked"
                        );
                        if l2a_config.mode == L2aMode::Block {
                            l2a_fail_closed_block(
                                &mut pending_block, adapter.as_ref(), "scan panicked"
                            );
                        }
                    }
                    Err(_) => {
                        // Timeout.
                        tracing::warn!(
                            request_id = %ctx.request_id,
                            layer = "L2a",
                            timeout_ms = l2a_config.timeout_ms,
                            "L2a scan timed out"
                        );
                        if l2a_config.mode == L2aMode::Block {
                            l2a_fail_closed_block(
                                &mut pending_block, adapter.as_ref(), "scan timed out"
                            );
                        }
                    }
                }
            } else {
                // Concurrency limit reached.
                tracing::warn!(
                    request_id = %ctx.request_id,
                    layer = "L2a",
                    max_concurrent = l2a_config.max_concurrent_scans,
                    "L2a concurrency limit reached"
                );
                if l2a_config.mode == L2aMode::Block {
                    l2a_fail_closed_block(
                        &mut pending_block, adapter.as_ref(), "concurrency limit"
                    );
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
            let inbound_result = self.deps.inbound_scanner.scan(&messages, &self.deps.config);
            let l3i_ms = l3i_start.elapsed().as_secs_f64() * 1000.0;
            inbound_result_opt = Some(inbound_result);
            let inbound_result = inbound_result_opt.as_ref().unwrap();

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
        }

        // 5) L4 multi-turn scanning (if enabled)
        if let (Some(scanner), Some(l4_config)) = (&self.deps.multi_turn_scanner, &self.deps.config.policy.layers.l4) {
            let l4_start = Instant::now();
            let l4_result = scanner.scan(&messages, l4_config);
            let l4_ms = l4_start.elapsed().as_secs_f64() * 1000.0;
            l4_result_opt = Some(l4_result);
            let l4_result = l4_result_opt.as_ref().unwrap();
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
        }

        // 5.5) Verdict processor (shadow — log only, no behavioral change).
        {
            let mut signals = Vec::new();
            if let Some(ref l1r) = l1_result_opt {
                signals.extend(self.deps.signal_extractor.extract_l1(l1r));
            }
            signals.extend(l2a_signals);
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
            return Ok(resp);
        }

        // 6) Forward to upstream
        let stream = is_streaming_request(&request.body);
        let url = build_upstream_url(self.deps.resolver.as_ref(), provider, &request);

        // Reverse proxy header hygiene:
        // - x-parapet-original-host: internal routing header, not for upstream.
        // - Host: strip client's Host header (points at 127.0.0.1:9800). reqwest
        //   sets the correct Host from the upstream URL. This is standard reverse
        //   proxy behavior (RFC 7230 §5.4).
        let mut fwd_headers = request.headers.clone();
        fwd_headers.remove("x-parapet-original-host");
        fwd_headers.remove(reqwest::header::HOST);

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

        if stream {
            // Content-Encoding on SSE is rare (providers send chunked transfer,
            // not gzip'd SSE). Log a warning if we see it, but pass through.
            if is_gzip(&upstream.headers) || is_deflate(&upstream.headers) {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    "streaming response has Content-Encoding; passing through without decompression"
                );
            }
            let mut resp = self.handle_streaming_response(provider, upstream, &ctx);
            attach_evidence_headers(&mut resp.headers, evidence_count, &evidence_categories);
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

        let processed = self.handle_non_streaming_response(provider, &body_bytes, &ctx);

        attach_evidence_headers(&mut resp_headers, evidence_count, &evidence_categories);
        Ok(ProxyResponse {
            status: processed.status_override.unwrap_or(upstream.status),
            headers: resp_headers,
            body: Body::from(processed.body),
        })
    }
}

impl EngineUpstreamClient {
    fn handle_streaming_response(&self, provider: Provider, upstream: HttpResponse, ctx: &RequestContext) -> ProxyResponse {
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

        let processor = StreamProcessor::new(classifier, validator, block_mode);
        let processed_stream = processor.process(stream);

        let redacted_stream = if let Some(layer) = &self.deps.config.policy.layers.l5a {
            if layer.mode == "redact" {
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

    fn handle_non_streaming_response(
        &self,
        provider: Provider,
        body: &Bytes,
        ctx: &RequestContext,
    ) -> ProcessedResponse {
        let adapter = self.deps.registry.adapter_for(provider);
        let mut json: serde_json::Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(_) => {
                return ProcessedResponse {
                    status_override: None,
                    body: body.clone(),
                }
            }
        };

        if let Some(layer) = &self.deps.config.policy.layers.l3_outbound {
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
                }
            }
        };

        let redacted = if let Some(layer) = &self.deps.config.policy.layers.l5a {
            if layer.mode == "redact" {
                let l5a_start = Instant::now();
                let result = self.deps
                    .output_scanner
                    .scan_and_redact(&String::from_utf8_lossy(&serialized), &self.deps.config);
                let l5a_ms = l5a_start.elapsed().as_secs_f64() * 1000.0;

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
                result
            } else {
                return ProcessedResponse {
                    status_override: None,
                    body: Bytes::from(serialized),
                };
            }
        } else {
            return ProcessedResponse {
                status_override: None,
                body: Bytes::from(serialized),
            };
        };

        ProcessedResponse {
            status_override: None,
            body: Bytes::from(redacted.content),
        }
    }
}

struct ProcessedResponse {
    status_override: Option<StatusCode>,
    body: Bytes,
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
// Env-based upstream resolver
// ---------------------------------------------------------------------------

pub struct EnvUpstreamResolver;

impl EnvUpstreamResolver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EnvUpstreamResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl UpstreamResolver for EnvUpstreamResolver {
    fn is_allowed_host(&self, host: &str) -> bool {
        if KNOWN_LLM_HOSTS.contains(&host) {
            return true;
        }
        // PARAPET_EXTRA_HOSTS=api.together.xyz,api.fireworks.ai
        if let Ok(extra) = std::env::var("PARAPET_EXTRA_HOSTS") {
            return extra.split(',').any(|h| h.trim() == host);
        }
        false
    }

    fn base_url_for_host(&self, host: &str) -> String {
        // Check PARAPET_{HOST_UPPER}_BASE_URL, e.g. PARAPET_API_CEREBRAS_AI_BASE_URL
        let env_key = format!(
            "PARAPET_{}_BASE_URL",
            host.to_uppercase().replace(['.', '-'], "_")
        );
        match std::env::var(&env_key) {
            Ok(url) if url.starts_with("https://") => url,
            Ok(url) => {
                tracing::warn!(
                    env_key = %env_key,
                    url = %url,
                    "ignoring non-HTTPS base URL override, falling back to https://{}",
                    host,
                );
                format!("https://{host}")
            }
            Err(_) => format!("https://{host}"),
        }
    }

    fn base_url(&self, provider: Provider) -> String {
        let (primary_key, fallback_key, default_url) = match provider {
            Provider::OpenAi => (
                "PARAPET_OPENAI_BASE_URL",
                Some("OPENAI_BASE_URL"),
                "https://api.openai.com",
            ),
            Provider::Anthropic => (
                "PARAPET_ANTHROPIC_BASE_URL",
                Some("ANTHROPIC_BASE_URL"),
                "https://api.anthropic.com",
            ),
        };

        let url = std::env::var(primary_key)
            .or_else(|_| {
                fallback_key
                    .map(std::env::var)
                    .unwrap_or(Err(std::env::VarError::NotPresent))
            })
            .unwrap_or_else(|_| default_url.to_string());

        if url.starts_with("https://") {
            url
        } else {
            tracing::warn!(
                url = %url,
                "ignoring non-HTTPS base URL override for {:?}, using default",
                provider,
            );
            default_url.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Reqwest HTTP sender
// ---------------------------------------------------------------------------

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
    let session_store: Option<Arc<dyn crate::session::SessionStore>> = None; // Sprint 2

    let l1_scanner: Option<Arc<dyn L1Scanner>> =
        if config.policy.layers.l1.is_some() {
            Some(Arc::new(DefaultL1Scanner::new()))
        } else {
            None
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
        http: Arc::new(ReqwestHttpSender::new(reqwest::Client::new())),
        resolver: Arc::new(EnvUpstreamResolver::new()),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        l1_scanner,
        l2a_scanner,
        l2a_semaphore,
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store,
        multi_turn_scanner,
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
    };

    Ok(EngineUpstreamClient::new_with(deps))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Block the request when L2a enforcement cannot run.
/// Only called in Block mode on scan failure (timeout/concurrency/panic/error).
fn l2a_fail_closed_block(
    pending_block: &mut Option<ProxyResponse>,
    adapter: &dyn ProviderAdapter,
    _reason: &str,
) {
    if pending_block.is_none() {
        let body = adapter.serialize_error(
            "request blocked by content policy (L2a unavailable)", 403
        );
        *pending_block = Some(ProxyResponse::blocked(body, "L2a"));
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

fn build_upstream_url(resolver: &dyn UpstreamResolver, provider: Provider, req: &ProxyRequest) -> String {
    // Primary: use original host from SDK header (any OpenAI-compatible provider).
    // Validate against resolver's allowlist — defense in depth against header
    // injection when the engine is used without the SDK (curl, other clients).
    // Fallback: resolve from Provider enum (zero-SDK mode).
    let base = req
        .headers
        .get("x-parapet-original-host")
        .and_then(|v| v.to_str().ok())
        .and_then(|host| {
            if resolver.is_allowed_host(host) {
                Some(resolver.base_url_for_host(host))
            } else {
                tracing::warn!(
                    host = host,
                    "X-Parapet-Original-Host rejected: not in allowed hosts"
                );
                None
            }
        })
        .unwrap_or_else(|| resolver.base_url(provider));

    let base = base.trim_end_matches('/');
    let path_and_query = req
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(req.uri.path());

    format!("{base}{path_and_query}")
}

fn extract_model(body: &Bytes) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(String::from))
        .unwrap_or_default()
}

fn is_streaming_request(body: &Bytes) -> bool {
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    json.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
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

/// Parse X-Guard-Trust header and attach trust spans to messages.
///
/// Format: `inline:<base64-encoded JSON>`
/// JSON: `[{"s": <start>, "e": <end>, "src": "<source>"}, ...]`
///
/// Byte offsets in the header reference positions in the SERIALIZED request body.
/// Since we've already parsed messages, we need to map body-level offsets to
/// per-message content offsets. We find each message's content string within
/// the serialized body and adjust span offsets relative to the content start.
///
/// If parsing fails, logs a warning and returns (graceful degradation).
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

    // The body is the full JSON request. Trust spans reference byte offsets within
    // this body. For each message, find where its content appears in the body and
    // compute per-message span offsets.
    let body_str = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(_) => return,
    };

    for msg in messages.iter_mut() {
        if msg.content.is_empty() {
            continue;
        }

        // Find the JSON-escaped form of the content in the body.
        // json.dumps adds quotes, so strip them to get the escaped interior.
        let escaped_content = match serde_json::to_string(&msg.content) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let escaped_interior = &escaped_content[1..escaped_content.len() - 1];

        // Find where this content appears in the body (byte offset)
        if let Some(content_byte_offset) = body_str.find(escaped_interior) {
            let content_byte_end = content_byte_offset + escaped_interior.len();

            for span in &spans {
                // Check if this span overlaps with this message's content in the body
                if span.s < content_byte_end && span.e > content_byte_offset {
                    // Compute message-relative offsets
                    let rel_start = span.s.saturating_sub(content_byte_offset);
                    let rel_end = span.e.min(content_byte_end) - content_byte_offset;

                    msg.trust_spans.push(crate::trust::TrustSpan::untrusted(
                        rel_start,
                        rel_end,
                        span.src.clone().unwrap_or_else(|| "unknown".to_string()),
                    ));
                }
            }
        }
    }
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
