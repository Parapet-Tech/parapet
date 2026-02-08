// Engine integration -- defined in M1.12
//
// Wires all layers together for non-streaming and streaming requests:
// - Parse request via provider adapter
// - Trust assignment
// - L0 normalization
// - L3-inbound scanning
// - Forward to upstream provider
// - L3-outbound tool call validation + rewrite (non-streaming)
// - L5a output redaction
// - Streaming: tool call buffering + validation via StreamProcessor

use flate2::read::{DeflateDecoder, GzDecoder};
use std::io::Read as _;

use crate::config::Config;
use crate::constraint::{ConstraintEvaluator, DslConstraintEvaluator, ToolCallVerdict};
use crate::layers::l3_inbound::{DefaultInboundScanner, InboundScanner, InboundVerdict};
use crate::layers::l5a::{L5aScanner, OutputScanner};
use crate::message::ToolCall;
use crate::normalize::{normalize_messages, L0Normalizer, Normalizer};
use crate::provider::{adapter_for, ProviderAdapter, ProviderType};
use crate::proxy::{ProxyError, ProxyRequest, ProxyResponse, Provider, UpstreamClient};
use crate::stream::{
    AnthropicChunkClassifier, OpenAiChunkClassifier, StreamProcessor, ToolCallBlockMode,
    ToolCallValidator, ValidationResult,
};
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
use std::time::Instant;
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
// Engine dependencies
// ---------------------------------------------------------------------------

pub struct EngineDeps {
    pub config: Arc<Config>,
    pub http: Arc<dyn HttpSender>,
    pub resolver: Arc<dyn UpstreamResolver>,
    pub registry: Arc<dyn ProviderRegistry>,
    pub normalizer: Arc<dyn Normalizer>,
    pub trust_assigner: Arc<dyn TrustAssigner>,
    pub inbound_scanner: Arc<dyn InboundScanner>,
    pub constraint_evaluator: Arc<dyn ConstraintEvaluator>,
    pub output_scanner: Arc<dyn OutputScanner>,
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

        // 2) Trust assignment
        self.deps
            .trust_assigner
            .assign_trust(&mut messages, &self.deps.config);

        // 3) L0 normalization (if enabled)
        if let Some(layer) = &self.deps.config.policy.layers.l0 {
            if layer.mode == "sanitize" {
                normalize_messages(self.deps.normalizer.as_ref(), &mut messages);
            }
        }

        // 4) L3-inbound scanning (if enabled)
        if let Some(layer) = &self.deps.config.policy.layers.l3_inbound {
            if layer.mode == "block" {
                let l3i_start = Instant::now();
                let inbound_verdict = self.deps.inbound_scanner.scan(&messages, &self.deps.config);
                let l3i_ms = l3i_start.elapsed().as_secs_f64() * 1000.0;

                match &inbound_verdict {
                    InboundVerdict::Block(block) => {
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
                            latency_ms = l3i_ms,
                            "inbound blocked"
                        );
                        let body = adapter.serialize_error(&block.reason, 403);
                        return Ok(ProxyResponse::from_bytes(StatusCode::FORBIDDEN, body));
                    }
                    InboundVerdict::Allow => {
                        tracing::info!(
                            request_id = %ctx.request_id,
                            contract_hash = %ctx.contract_hash,
                            provider = ctx.provider_str,
                            model = %ctx.model,
                            layer = "L3-inbound",
                            verdict = "allow",
                            latency_ms = l3i_ms,
                            "inbound allowed"
                        );
                    }
                }
            }
        }

        // 5) Forward to upstream
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
            return Ok(self.handle_streaming_response(provider, upstream, &ctx));
        }

        let mut resp_headers = upstream.headers;
        let body_bytes = match upstream.body {
            HttpBody::Full(b) => b,
            HttpBody::Stream(mut s) => {
                let mut collected = Vec::new();
                while let Some(chunk) = s.next().await {
                    let bytes = chunk.map_err(|e| ProxyError::UpstreamFailure(e.to_string()))?;
                    collected.extend_from_slice(&bytes);
                }
                Bytes::from(collected)
            }
        };

        // Decompress if needed so security scanning can parse the JSON.
        // Strips Content-Encoding/Content-Length from forwarded headers.
        let body_bytes = maybe_decompress(&mut resp_headers, body_bytes)?;

        let processed = self.handle_non_streaming_response(provider, &body_bytes, &ctx);
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
                        let body = adapter.serialize_error(reason, 403);
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
        std::env::var(&env_key).unwrap_or_else(|_| format!("https://{host}"))
    }

    fn base_url(&self, provider: Provider) -> String {
        match provider {
            Provider::OpenAi => std::env::var("PARAPET_OPENAI_BASE_URL")
                .or_else(|_| std::env::var("OPENAI_BASE_URL"))
                .unwrap_or_else(|_| "https://api.openai.com".to_string()),
            Provider::Anthropic => std::env::var("PARAPET_ANTHROPIC_BASE_URL")
                .or_else(|_| std::env::var("ANTHROPIC_BASE_URL"))
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
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

pub fn build_engine_client(config: Arc<Config>) -> EngineUpstreamClient {
    let deps = EngineDeps {
        config,
        http: Arc::new(ReqwestHttpSender::new(reqwest::Client::new())),
        resolver: Arc::new(EnvUpstreamResolver::new()),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
    };

    EngineUpstreamClient::new_with(deps)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Decompress a gzip-encoded body.
fn decompress_gzip(body: &Bytes) -> Result<Bytes, ProxyError> {
    let mut decoder = GzDecoder::new(&body[..]);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| ProxyError::UpstreamFailure(format!("gzip decompression failed: {e}")))?;
    Ok(Bytes::from(decompressed))
}

/// Decompress a deflate-encoded body.
fn decompress_deflate(body: &Bytes) -> Result<Bytes, ProxyError> {
    let mut decoder = DeflateDecoder::new(&body[..]);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| ProxyError::UpstreamFailure(format!("deflate decompression failed: {e}")))?;
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

fn stream_body(
    stream: impl Stream<Item = Result<Bytes, HttpError>> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Bytes> + Send>> {
    Box::pin(stream.filter_map(|item| async move { item.ok() }))
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
        max_len = max_len.max(pattern.pattern.len());
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CompiledPattern, ContentPolicy, EngineConfig, LayerConfig, LayerConfigs, PolicyConfig,
        RuntimeConfig, ToolConfig, TrustConfig,
    };
    use crate::message::Message;
    use futures_util::stream;
    use std::collections::HashMap;

    fn base_config_with_canary(canary: &str) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: Vec::new(),
                canary_tokens: vec![canary.to_string()],
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    fn base_config_with_sensitive(pattern: &str) -> Config {
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools: HashMap::new(),
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: vec![CompiledPattern::compile(pattern).unwrap()],
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs::default(),
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    async fn collect_stream(stream: impl Stream<Item = Bytes> + Unpin) -> String {
        let mut output = String::new();
        futures_util::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            output.push_str(&String::from_utf8_lossy(&chunk));
        }
        output
    }

    #[tokio::test]
    async fn streaming_redaction_canary_across_chunk_boundary() {
        let config = Arc::new(base_config_with_canary("SECRET"));
        let scanner: Arc<dyn OutputScanner> = Arc::new(L5aScanner);

        let chunks = vec![
            Bytes::from("start SE"),
            Bytes::from("CRET end"),
        ];
        let input = stream::iter(chunks);

        let redacted = apply_streaming_redaction(input, scanner, config, Some(8));
        let output = collect_stream(redacted).await;

        assert!(!output.contains("SECRET"));
        assert!(output.contains("[REDACTED]"));
        assert!(output.contains("start "));
        assert!(output.contains(" end"));
    }

    #[tokio::test]
    async fn streaming_redaction_regex_across_chunk_boundary() {
        let config = Arc::new(base_config_with_sensitive("API_KEY=[A-Za-z0-9]+"));
        let scanner: Arc<dyn OutputScanner> = Arc::new(L5aScanner);

        let chunks = vec![
            Bytes::from("token API"),
            Bytes::from("_KEY=abc123 here"),
        ];
        let input = stream::iter(chunks);

        let redacted = apply_streaming_redaction(input, scanner, config, Some(16));
        let output = collect_stream(redacted).await;

        assert!(!output.contains("API_KEY=abc123"));
        assert!(output.contains("[REDACTED]"));
        assert!(output.contains("token "));
        assert!(output.contains(" here"));
    }

    // -------------------------------------------------------------------
    // L3-outbound rewrite/error tests (non-streaming)
    // -------------------------------------------------------------------

    struct NoopHttpSender;
    #[async_trait]
    impl HttpSender for NoopHttpSender {
        async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
            Err(HttpError::Transport("not used".to_string()))
        }
    }

    struct NoopResolver;
    impl UpstreamResolver for NoopResolver {
        fn base_url(&self, _provider: Provider) -> String {
            "https://example.com".to_string()
        }
    }

    struct DefaultRegistry;
    impl ProviderRegistry for DefaultRegistry {
        fn adapter_for(&self, provider: Provider) -> Box<dyn ProviderAdapter> {
            let p = match provider {
                Provider::OpenAi => ProviderType::OpenAi,
                Provider::Anthropic => ProviderType::Anthropic,
            };
            adapter_for(p)
        }
    }

    struct NoopNormalizer;
    impl Normalizer for NoopNormalizer {
        fn normalize(&self, input: &str) -> String {
            input.to_string()
        }
    }

    struct NoopTrustAssigner;
    impl TrustAssigner for NoopTrustAssigner {
        fn assign_trust(&self, _messages: &mut [Message], _config: &Config) {}
    }

    struct NoopInboundScanner;
    impl InboundScanner for NoopInboundScanner {
        fn scan(&self, _messages: &[Message], _config: &Config) -> InboundVerdict {
            InboundVerdict::Allow
        }
    }

    fn engine_with_config(config: Config) -> EngineUpstreamClient {
        let deps = EngineDeps {
            config: Arc::new(config),
            http: Arc::new(NoopHttpSender),
            resolver: Arc::new(NoopResolver),
            registry: Arc::new(DefaultRegistry),
            normalizer: Arc::new(NoopNormalizer),
            trust_assigner: Arc::new(NoopTrustAssigner),
            inbound_scanner: Arc::new(NoopInboundScanner),
            constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
            output_scanner: Arc::new(L5aScanner),
        };
        EngineUpstreamClient::new_with(deps)
    }

    fn test_ctx() -> RequestContext {
        RequestContext {
            request_id: "test-request-id".to_string(),
            contract_hash: "sha256:test".to_string(),
            provider_str: "openai",
            model: "test-model".to_string(),
        }
    }

    fn config_with_l3_outbound(block_action: &str) -> Config {
        let mut tools = HashMap::new();
        tools.insert(
            "_default".to_string(),
            ToolConfig {
                allowed: false,
                trust: None,
                constraints: HashMap::new(),
                result_policy: None,
            },
        );
        tools.insert(
            "read_file".to_string(),
            ToolConfig {
                allowed: true,
                trust: None,
                constraints: HashMap::new(),
                result_policy: None,
            },
        );

        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools,
                block_patterns: Vec::new(),
                canary_tokens: Vec::new(),
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs {
                    l0: None,
                    l3_inbound: None,
                    l3_outbound: Some(LayerConfig {
                        mode: "block".to_string(),
                        block_action: Some(block_action.to_string()),
                        window_chars: None,
                    }),
                    l5a: None,
                },
            },
            runtime: RuntimeConfig {
                engine: EngineConfig::default(),
                environment: String::new(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    #[test]
    fn l3_outbound_openai_rewrite_blocks_only_disallowed_tool() {
        let config = config_with_l3_outbound("rewrite");
        let engine = engine_with_config(config);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name":"read_file","arguments":"{}"}
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name":"exec_command","arguments":"{\"cmd\":\"rm -rf /\"}"}
                        }
                    ]
                }
            }]
        });

        let body = Bytes::from(serde_json::to_vec(&response).unwrap());
        let ctx = test_ctx();
        let processed = engine.handle_non_streaming_response(Provider::OpenAi, &body, &ctx);

        assert!(processed.status_override.is_none());
        let output: serde_json::Value = serde_json::from_slice(&processed.body).unwrap();
        let tool_calls = output["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0]["function"]["name"].as_str().unwrap(),
            "read_file"
        );
        let content = output["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert!(content.contains("exec_command"));
        assert!(content.contains("blocked"));
    }

    #[test]
    fn l3_outbound_openai_error_returns_403() {
        let config = config_with_l3_outbound("error");
        let engine = engine_with_config(config);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name":"exec_command","arguments":"{\"cmd\":\"rm -rf /\"}"}
                        }
                    ]
                }
            }]
        });

        let body = Bytes::from(serde_json::to_vec(&response).unwrap());
        let ctx = test_ctx();
        let processed = engine.handle_non_streaming_response(Provider::OpenAi, &body, &ctx);

        assert_eq!(processed.status_override, Some(StatusCode::FORBIDDEN));
        let output: serde_json::Value = serde_json::from_slice(&processed.body).unwrap();
        assert!(output.get("error").is_some());
    }

    #[test]
    fn l3_outbound_anthropic_rewrite_blocks_only_disallowed_tool() {
        let config = config_with_l3_outbound("rewrite");
        let engine = engine_with_config(config);

        let response = serde_json::json!({
            "content": [
                {"type":"tool_use","id":"toolu_1","name":"read_file","input":{}},
                {"type":"tool_use","id":"toolu_2","name":"exec_command","input":{"cmd":"rm -rf /"}}
            ]
        });

        let body = Bytes::from(serde_json::to_vec(&response).unwrap());
        let ctx = test_ctx();
        let processed = engine.handle_non_streaming_response(Provider::Anthropic, &body, &ctx);

        assert!(processed.status_override.is_none());
        let output: serde_json::Value = serde_json::from_slice(&processed.body).unwrap();
        let content = output["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"].as_str().unwrap(), "tool_use");
        assert_eq!(content[0]["name"].as_str().unwrap(), "read_file");
        assert_eq!(content[1]["type"].as_str().unwrap(), "text");
        assert!(content[1]["text"].as_str().unwrap().contains("exec_command"));
    }

    #[test]
    fn l3_outbound_anthropic_error_returns_403() {
        let config = config_with_l3_outbound("error");
        let engine = engine_with_config(config);

        let response = serde_json::json!({
            "content": [
                {"type":"tool_use","id":"toolu_2","name":"exec_command","input":{"cmd":"rm -rf /"}}
            ]
        });

        let body = Bytes::from(serde_json::to_vec(&response).unwrap());
        let ctx = test_ctx();
        let processed = engine.handle_non_streaming_response(Provider::Anthropic, &body, &ctx);

        assert_eq!(processed.status_override, Some(StatusCode::FORBIDDEN));
        let output: serde_json::Value = serde_json::from_slice(&processed.body).unwrap();
        assert_eq!(output["type"].as_str().unwrap(), "error");
    }

    // -------------------------------------------------------------------
    // HttpError variant tests
    // -------------------------------------------------------------------

    #[test]
    fn http_error_timeout_formats_correctly() {
        let err = HttpError::Timeout("request timed out after 5000ms".to_string());
        assert_eq!(
            err.to_string(),
            "upstream request timed out: request timed out after 5000ms"
        );
    }

    #[test]
    fn http_error_transport_formats_correctly() {
        let err = HttpError::Transport("connection refused".to_string());
        assert_eq!(
            err.to_string(),
            "upstream request failed: connection refused"
        );
    }

    // -------------------------------------------------------------------
    // build_upstream_url tests
    // -------------------------------------------------------------------

    fn proxy_request_with_header(path: &str, host: Option<&str>) -> ProxyRequest {
        let uri: Uri = path.parse().unwrap();
        let mut headers = HeaderMap::new();
        if let Some(h) = host {
            headers.insert("x-parapet-original-host", h.parse().unwrap());
        }
        ProxyRequest {
            method: Method::POST,
            uri,
            headers,
            body: Bytes::from_static(b"{}"),
        }
    }

    #[test]
    fn build_upstream_url_with_header_uses_host() {
        let req = proxy_request_with_header(
            "/v1/chat/completions",
            Some("api.cerebras.ai"),
        );
        let url = build_upstream_url(&NoopResolver, Provider::OpenAi, &req);
        assert_eq!(url, "https://api.cerebras.ai/v1/chat/completions");
    }

    #[test]
    fn build_upstream_url_without_header_falls_back_to_resolver() {
        let req = proxy_request_with_header("/v1/chat/completions", None);
        let url = build_upstream_url(&NoopResolver, Provider::OpenAi, &req);
        assert_eq!(url, "https://example.com/v1/chat/completions");
    }

    #[test]
    fn build_upstream_url_with_header_uses_resolver_base_url_for_host() {
        // A resolver that overrides base_url_for_host for staging.
        struct StagingResolver;
        impl UpstreamResolver for StagingResolver {
            fn base_url_for_host(&self, host: &str) -> String {
                if host == "api.cerebras.ai" {
                    "https://staging.cerebras.internal".to_string()
                } else {
                    format!("https://{host}")
                }
            }
            fn base_url(&self, _provider: Provider) -> String {
                "https://example.com".to_string()
            }
        }

        let req = proxy_request_with_header(
            "/v1/chat/completions",
            Some("api.cerebras.ai"),
        );
        let url = build_upstream_url(&StagingResolver, Provider::OpenAi, &req);
        assert_eq!(url, "https://staging.cerebras.internal/v1/chat/completions");
    }

    #[test]
    fn env_upstream_resolver_base_url_for_host_default() {
        let resolver = EnvUpstreamResolver::new();
        // Without env var set, defaults to https://{host}
        let url = resolver.base_url_for_host("api.groq.com");
        assert_eq!(url, "https://api.groq.com");
    }

    #[test]
    fn env_upstream_resolver_base_url_for_host_with_env_override() {
        let resolver = EnvUpstreamResolver::new();
        // Use a distinct host from env_upstream_resolver_base_url_for_host_default
        // to avoid env var race between parallel tests.
        std::env::set_var("PARAPET_API_MISTRAL_AI_BASE_URL", "https://mistral.staging.internal");
        let url = resolver.base_url_for_host("api.mistral.ai");
        std::env::remove_var("PARAPET_API_MISTRAL_AI_BASE_URL");
        assert_eq!(url, "https://mistral.staging.internal");
    }

    // -------------------------------------------------------------------
    // Host allowlist tests (defense in depth)
    // -------------------------------------------------------------------

    #[test]
    fn disallowed_host_header_falls_back_to_provider_resolver() {
        // evil.com is not in KNOWN_LLM_HOSTS — header must be rejected.
        let req = proxy_request_with_header(
            "/v1/chat/completions",
            Some("evil.com"),
        );
        let url = build_upstream_url(&NoopResolver, Provider::OpenAi, &req);
        // Falls back to NoopResolver.base_url() → "https://example.com"
        assert_eq!(url, "https://example.com/v1/chat/completions");
    }

    #[test]
    fn allowed_host_header_is_accepted() {
        let req = proxy_request_with_header(
            "/v1/chat/completions",
            Some("api.groq.com"),
        );
        let url = build_upstream_url(&NoopResolver, Provider::OpenAi, &req);
        assert_eq!(url, "https://api.groq.com/v1/chat/completions");
    }

    #[test]
    fn env_extra_hosts_extends_allowlist() {
        let resolver = EnvUpstreamResolver::new();
        assert!(!resolver.is_allowed_host("api.fireworks.ai"));

        std::env::set_var("PARAPET_EXTRA_HOSTS", "api.fireworks.ai,api.custom.dev");
        assert!(resolver.is_allowed_host("api.fireworks.ai"));
        assert!(resolver.is_allowed_host("api.custom.dev"));
        // Known hosts still allowed
        assert!(resolver.is_allowed_host("api.openai.com"));
        // Random host still blocked
        assert!(!resolver.is_allowed_host("evil.com"));
        std::env::remove_var("PARAPET_EXTRA_HOSTS");
    }

    #[test]
    fn default_is_allowed_host_rejects_unknown() {
        // NoopResolver gets the default is_allowed_host from the trait.
        assert!(NoopResolver.is_allowed_host("api.openai.com"));
        assert!(NoopResolver.is_allowed_host("api.cerebras.ai"));
        assert!(!NoopResolver.is_allowed_host("evil.com"));
        assert!(!NoopResolver.is_allowed_host("api.openai.com.evil.com"));
    }

    // -------------------------------------------------------------------
    // Decompression tests
    // -------------------------------------------------------------------

    fn gzip_compress(data: &[u8]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn deflate_compress(data: &[u8]) -> Vec<u8> {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn is_gzip_detects_gzip_encoding() {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_ENCODING, "gzip".parse().unwrap());
        assert!(is_gzip(&headers));
    }

    #[test]
    fn is_gzip_case_insensitive() {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_ENCODING, "Gzip".parse().unwrap());
        assert!(is_gzip(&headers));
    }

    #[test]
    fn is_gzip_false_for_identity() {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_ENCODING, "identity".parse().unwrap());
        assert!(!is_gzip(&headers));
    }

    #[test]
    fn is_gzip_false_for_no_header() {
        let headers = HeaderMap::new();
        assert!(!is_gzip(&headers));
    }

    #[test]
    fn decompress_gzip_recovers_json() {
        let json = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let compressed = gzip_compress(json.as_bytes());
        let result = decompress_gzip(&Bytes::from(compressed)).unwrap();
        assert_eq!(&result[..], json.as_bytes());
    }

    #[test]
    fn decompress_deflate_recovers_json() {
        let json = r#"{"choices":[{"message":{"content":"hello"}}]}"#;
        let compressed = deflate_compress(json.as_bytes());
        let result = decompress_deflate(&Bytes::from(compressed)).unwrap();
        assert_eq!(&result[..], json.as_bytes());
    }

    #[test]
    fn maybe_decompress_strips_headers_on_gzip() {
        let json = r#"{"ok":true}"#;
        let compressed = gzip_compress(json.as_bytes());
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_ENCODING, "gzip".parse().unwrap());
        headers.insert(reqwest::header::CONTENT_LENGTH, "999".parse().unwrap());

        let result = maybe_decompress(&mut headers, Bytes::from(compressed)).unwrap();
        assert_eq!(&result[..], json.as_bytes());
        assert!(headers.get(reqwest::header::CONTENT_ENCODING).is_none());
        assert!(headers.get(reqwest::header::CONTENT_LENGTH).is_none());
    }

    #[test]
    fn maybe_decompress_passthrough_no_encoding() {
        let body = Bytes::from_static(b"plain text");
        let mut headers = HeaderMap::new();
        let result = maybe_decompress(&mut headers, body.clone()).unwrap();
        assert_eq!(result, body);
    }

    #[test]
    fn non_streaming_gzip_response_still_scannable() {
        // Full integration: gzip'd response body with L5a redaction
        let json = r#"{"choices":[{"message":{"content":"my secret SSN 123-45-6789"}}]}"#;
        let compressed = gzip_compress(json.as_bytes());

        let mut config = base_config_with_sensitive(r"\d{3}-\d{2}-\d{4}");
        config.policy.layers.l5a = Some(LayerConfig {
            mode: "redact".to_string(),
            block_action: None,
            window_chars: None,
        });
        let engine = engine_with_config(config);

        // Simulate: decompress first (as forward() does), then scan
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_ENCODING, "gzip".parse().unwrap());
        let decompressed = maybe_decompress(&mut headers, Bytes::from(compressed)).unwrap();

        let ctx = test_ctx();
        let processed = engine.handle_non_streaming_response(Provider::OpenAi, &decompressed, &ctx);

        let output = String::from_utf8_lossy(&processed.body);
        assert!(!output.contains("123-45-6789"));
        assert!(output.contains("[REDACTED]"));
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
