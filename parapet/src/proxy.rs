// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// HTTP proxy skeleton — M1.3
//
// Responsibilities:
// - Provider detection from request path
// - Request forwarding via injected UpstreamClient trait
// - Response header stripping (X-Guard-Trust*)
// - Heartbeat endpoint
// - 404 for unknown paths

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Wire format detected from the request path.
///
/// This is **not** a vendor enum — it identifies the API wire format.
/// `OpenAi` covers any provider using the OpenAI-compatible chat completions
/// API (Cerebras, Groq, Together, etc.). `Anthropic` covers the Anthropic
/// messages API. Routing to the correct upstream host is handled separately
/// by `UpstreamResolver`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

/// Inbound request data forwarded to the upstream client.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// Response received from the upstream provider.
#[derive(Debug)]
pub struct ProxyResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Body,
}

/// Errors that can occur during upstream forwarding.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("upstream request failed: {0}")]
    UpstreamFailure(String),

    #[error("upstream request timed out: {0}")]
    UpstreamTimeout(String),

    #[error("request body is not valid JSON: {0}")]
    MalformedJson(String),

    #[error("request body is empty")]
    EmptyBody,
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> axum::response::Response {
        let (status, public_message) = match &self {
            ProxyError::UpstreamFailure(_) => (
                StatusCode::BAD_GATEWAY,
                "upstream request failed".to_string(),
            ),
            ProxyError::UpstreamTimeout(_) => (
                StatusCode::GATEWAY_TIMEOUT,
                "upstream request timed out".to_string(),
            ),
            ProxyError::MalformedJson(_) => (
                StatusCode::BAD_REQUEST,
                "request body is not valid JSON".to_string(),
            ),
            ProxyError::EmptyBody => (
                StatusCode::BAD_REQUEST,
                "request body is empty".to_string(),
            ),
        };
        (status, public_message).into_response()
    }
}

// ---------------------------------------------------------------------------
// Trait: UpstreamClient (dependency injection point)
// ---------------------------------------------------------------------------

/// Abstraction over the HTTP client that forwards requests to LLM providers.
///
/// Implementations must be Send + Sync so they can be shared across request
/// handlers via `Arc`.
#[async_trait::async_trait]
pub trait UpstreamClient: Send + Sync {
    async fn forward(
        &self,
        provider: Provider,
        request: ProxyRequest,
    ) -> Result<ProxyResponse, ProxyError>;

    /// Forward a request the proxy received at `received_at`. Clients with a
    /// receipt-bound budget override this; the default ignores the instant.
    async fn forward_received(
        &self,
        provider: Provider,
        request: ProxyRequest,
        received_at: tokio::time::Instant,
    ) -> Result<ProxyResponse, ProxyError> {
        let _ = received_at;
        self.forward(provider, request).await
    }
}

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

/// Classification of an inbound request path before provider detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathClass {
    Supported(Provider),
    /// Exact route of a recognized wire format whose forwarding is disabled.
    Staged(StagedRoute),
    KnownUnsupported,
    NonCanonical,
    Unknown,
}

/// Wire formats with an exact route whose support is narrower than the legacy
/// provider route family. Their handler branch owns the additional admission
/// rules before selecting a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedRoute {
    /// OpenAI Responses API (`crate::provider::responses`).
    OpenAiResponses,
}

impl StagedRoute {
    /// Exact POST paths for this wire format. No aliases, suffixes, or prefixes.
    pub const fn exact_post_paths(self) -> &'static [&'static str] {
        match self {
            Self::OpenAiResponses => &["/v1/responses", "/responses"],
        }
    }

    /// Match `path` exactly against every staged route.
    pub fn for_exact_path(path: &str) -> Option<Self> {
        [Self::OpenAiResponses]
            .into_iter()
            .find(|route| route.exact_post_paths().contains(&path))
    }
}

const KNOWN_UNSUPPORTED_SUBPATHS: &[&str] = &[
    "/v1/responses",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
    "/v1/audio",
    "/v1/images",
    "/v1/files",
    "/v1/batches",
    "/v1/moderations",
    "/v1/assistants",
    "/v1/threads",
    "/v1/realtime",
    "/v1/messages/count_tokens",
    "/v1/messages/batches",
];

fn contains_ascii_case_insensitive(haystack: &str, needle_lower: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle_lower.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return false;
    }
    h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
}

fn decode_dot_segment(seg: &str) -> String {
    let bytes = seg.as_bytes();
    let mut out = String::with_capacity(seg.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 3 <= bytes.len() && bytes[i..i + 3].eq_ignore_ascii_case(b"%2e") {
            out.push('.');
            i += 3;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn is_non_canonical_path(path: &str) -> bool {
    if !path.starts_with('/')
        || path.contains('\\')
        || path.contains("//")
        || path.contains('%')
        || contains_ascii_case_insensitive(path, "%2f")
        || contains_ascii_case_insensitive(path, "%5c")
    {
        return true;
    }
    for seg in path.split('/').skip(1) {
        let decoded = decode_dot_segment(seg);
        if decoded == "." || decoded == ".." {
            return true;
        }
    }
    false
}

fn matches_exact_or_segment_prefix(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

fn is_known_unsupported_path(path: &str) -> bool {
    let normalized = path.strip_prefix("/openai").unwrap_or(path);
    KNOWN_UNSUPPORTED_SUBPATHS
        .iter()
        .any(|prefix| matches_exact_or_segment_prefix(normalized, prefix))
}

/// Classify a request path before `detect_provider`.
pub fn classify_path(path: &str) -> PathClass {
    if is_non_canonical_path(path) {
        return PathClass::NonCanonical;
    }
    if let Some(route) = StagedRoute::for_exact_path(path) {
        return PathClass::Staged(route);
    }
    if is_known_unsupported_path(path) {
        return PathClass::KnownUnsupported;
    }
    if let Some(provider) = detect_provider(path) {
        return PathClass::Supported(provider);
    }
    PathClass::Unknown
}

fn sanitize_path_for_diagnostic(path: &str) -> String {
    let mut byte_end = 0;
    for ch in path.chars() {
        let next = byte_end + ch.len_utf8();
        if next > 128 {
            break;
        }
        byte_end = next;
    }
    let truncated = &path[..byte_end];
    let mut escaped = String::with_capacity(truncated.len());
    for ch in truncated.chars() {
        if ch.is_control() {
            for esc in ch.escape_default() {
                escaped.push(esc);
            }
        } else {
            escaped.push(ch);
        }
    }
    escaped
}

fn path_refusal_response(
    path: &str,
    status: StatusCode,
    err_type: &'static str,
    code: &'static str,
) -> Response<Body> {
    let safe_path = sanitize_path_for_diagnostic(path);
    tracing::warn!(
        path = %safe_path,
        status = status.as_u16(),
        error_type = err_type,
        code = code,
        "request path refused"
    );
    let message = format!(
        "Parapet does not inspect {safe_path}. Supported: POST /v1/chat/completions, POST /openai/v1/chat/completions, POST /v1/messages. See docs/litellm.md and README.md."
    );
    let payload = serde_json::json!({
        "error": {
            "type": err_type,
            "code": code,
            "message": message,
        }
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let mut resp = Response::builder()
        .status(status)
        .body(Body::from(bytes))
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp.headers_mut()
        .insert("x-parapet-error", HeaderValue::from_static(code));
    resp
}

fn method_not_allowed_response(path: &str, method: &Method) -> Response<Body> {
    let safe_path = sanitize_path_for_diagnostic(path);
    tracing::warn!(
        path = %safe_path,
        method = %method,
        status = StatusCode::METHOD_NOT_ALLOWED.as_u16(),
        error_type = "parapet_method_not_allowed",
        code = "method_not_allowed",
        "request method refused"
    );
    let message = format!(
        "Method {method} is not allowed on {safe_path}. Supported: POST /v1/chat/completions, POST /openai/v1/chat/completions, POST /v1/messages."
    );
    let payload = serde_json::json!({
        "error": {
            "type": "parapet_method_not_allowed",
            "code": "method_not_allowed",
            "message": message,
        }
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let mut resp = Response::builder()
        .status(StatusCode::METHOD_NOT_ALLOWED)
        .body(Body::from(bytes))
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp.headers_mut()
        .insert("allow", HeaderValue::from_static("POST"));
    resp.headers_mut().insert(
        "x-parapet-error",
        HeaderValue::from_static("method_not_allowed"),
    );
    resp
}

/// Detect the LLM provider from the request path.
///
/// Returns `None` for paths that do not map to a known provider.
pub fn detect_provider(path: &str) -> Option<Provider> {
    if path.starts_with("/v1/chat/completions")
        || path.starts_with("/openai/v1/chat/completions")
    {
        Some(Provider::OpenAi)
    } else if path.starts_with("/v1/messages") {
        Some(Provider::Anthropic)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Header stripping
// ---------------------------------------------------------------------------

/// Strip response headers whose names start with `X-Guard-Trust` (case-insensitive).
///
/// All other headers are preserved unchanged.
pub fn strip_guard_trust_headers(headers: &mut HeaderMap) {
    let keys_to_remove: Vec<_> = headers
        .keys()
        .filter(|name| {
            name.as_str()
                .to_ascii_lowercase()
                .starts_with("x-guard-trust")
        })
        .cloned()
        .collect();

    for key in keys_to_remove {
        headers.remove(&key);
    }
}

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

use crate::provenance::{
    admit, extract_carrier_header, CarrierAdmission, CarrierError, CarrierVerifier, Clock,
    ProfileMode, ProviderContentDecoder, SessionConfig, SessionManager, SupportedProfile,
    VerificationOutcome, VerifiedCarrier,
};
use crate::provider::ProviderType;

impl Provider {
    /// Exact canonical `POST` path accepted under required-profile carrier admission.
    pub const fn canonical_post_path(self) -> &'static str {
        match self {
            Self::OpenAi => "/v1/chat/completions",
            Self::Anthropic => "/v1/messages",
        }
    }

    /// Corresponding [`ProviderType`] for wire adapter and [`ProviderContentDecoder`] selection.
    pub const fn provider_type(self) -> ProviderType {
        match self {
            Self::OpenAi => ProviderType::OpenAi,
            Self::Anthropic => ProviderType::Anthropic,
        }
    }

    /// Construct the matching [`ProviderContentDecoder`] for this provider.
    pub fn content_decoder(self) -> ProviderContentDecoder {
        ProviderContentDecoder::new(self.provider_type())
    }
}

/// Closed required-profile route and decoder binding for a single [`Provider`].
///
/// Binds one [`Provider`], its exact canonical `POST` path, and its matching
/// [`ProviderContentDecoder`] (`OpenAi` -> `POST /v1/chat/completions`,
/// `Anthropic` -> `POST /v1/messages`) so the session verifier decoder,
/// proxy route gate, and post-verify consistency decoder cannot diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceRouteBinding {
    provider: Provider,
    canonical_post_path: &'static str,
    provider_type: ProviderType,
}

impl ProvenanceRouteBinding {
    /// Closed OpenAI binding: exact `POST /v1/chat/completions` + [`ProviderType::OpenAi`] decoder.
    pub const OPENAI: Self = Self::for_provider(Provider::OpenAi);

    /// Closed Anthropic binding: exact `POST /v1/messages` + [`ProviderType::Anthropic`] decoder.
    pub const ANTHROPIC: Self = Self::for_provider(Provider::Anthropic);

    /// Construct the closed binding for `provider`.
    pub const fn for_provider(provider: Provider) -> Self {
        Self {
            provider,
            canonical_post_path: provider.canonical_post_path(),
            provider_type: provider.provider_type(),
        }
    }

    /// Bound [`Provider`].
    pub const fn provider(self) -> Provider {
        self.provider
    }

    /// Exact canonical `POST` path for the bound [`Provider`].
    pub const fn canonical_post_path(self) -> &'static str {
        self.canonical_post_path
    }

    /// Matching [`ProviderType`] for [`ProviderContentDecoder`].
    pub const fn provider_type(self) -> ProviderType {
        self.provider_type
    }

    /// Construct the matching [`ProviderContentDecoder`].
    pub fn content_decoder(self) -> ProviderContentDecoder {
        ProviderContentDecoder::new(self.provider_type)
    }

    /// Build a [`SessionConfig`] for `profile` and `clock` using this binding's
    /// [`ProviderContentDecoder`].
    pub fn session_config(self, profile: SupportedProfile, clock: Box<dyn Clock>) -> SessionConfig {
        SessionConfig {
            profile,
            clock,
            decoder: Box::new(self.content_decoder()),
        }
    }

    /// Check whether an inbound request matches the exact `POST` method, [`Provider`], and canonical path.
    pub fn matches_post_route(self, method: &Method, provider: Provider, path: &str) -> bool {
        *method == Method::POST && provider == self.provider && path == self.canonical_post_path
    }
}

impl From<Provider> for ProvenanceRouteBinding {
    fn from(provider: Provider) -> Self {
        Self::for_provider(provider)
    }
}

/// Shared state injected into axum handlers.
#[derive(Clone)]
pub struct AppState {
    upstream: Arc<dyn UpstreamClient>,
    admission: Arc<dyn CarrierAdmissionPolicy>,
    verified_next_stage: Option<Arc<dyn VerifiedCarrierNextStage>>,
}

impl AppState {
    /// Construct [`AppState`], panicking if `admission.mode()` is
    /// [`ProfileMode::RequiredOpenCode`] and `verified_next_stage` is `None`
    /// or `admission.route_binding()` is `None`.
    pub fn new(
        upstream: Arc<dyn UpstreamClient>,
        admission: Arc<dyn CarrierAdmissionPolicy>,
        verified_next_stage: Option<Arc<dyn VerifiedCarrierNextStage>>,
    ) -> Self {
        assert!(
            admission.mode() != ProfileMode::RequiredOpenCode || verified_next_stage.is_some(),
            "RequiredOpenCode admission policy requires a VerifiedCarrierNextStage"
        );
        assert!(
            admission.mode() != ProfileMode::RequiredOpenCode
                || admission.route_binding().is_some(),
            "RequiredOpenCode admission policy requires a ProvenanceRouteBinding"
        );
        Self {
            upstream,
            admission,
            verified_next_stage,
        }
    }

    pub fn upstream(&self) -> &Arc<dyn UpstreamClient> {
        &self.upstream
    }

    pub fn admission(&self) -> &Arc<dyn CarrierAdmissionPolicy> {
        &self.admission
    }

    pub fn verified_next_stage(&self) -> Option<&Arc<dyn VerifiedCarrierNextStage>> {
        self.verified_next_stage.as_ref()
    }
}

/// Request-level admission decision before L0/trust/L1 and before any upstream attempt.
#[derive(Debug)]
pub enum RequestAdmissionDecision {
    /// Ordinary mode: proceed through the existing whole-message pipeline.
    WholeMessage,
    /// RequiredOpenCode mode: carrier verified; pass [`VerifiedCarrier`] to the next slice.
    Verified(VerifiedCarrier),
    /// RequiredOpenCode mode: carrier missing or invalid; fail closed with zero upstream calls.
    Reject(CarrierError),
}

/// Injected request-admission interface owned by the proxy/engine boundary.
pub trait CarrierAdmissionPolicy: Send + Sync {
    /// Configured profile mode.
    fn mode(&self) -> ProfileMode;

    /// Closed route/provider binding enforced under [`ProfileMode::RequiredOpenCode`].
    fn route_binding(&self) -> Option<ProvenanceRouteBinding> {
        None
    }

    /// Admit an inbound request after route binding and header extraction.
    fn admit_request(&self, envelope: Option<&[u8]>, body: &[u8]) -> RequestAdmissionDecision;

    /// Build a content-free [`CarrierError`] for transport/route pre-verifier rejections.
    fn fail(&self, outcome: VerificationOutcome) -> CarrierError;
}

/// Default ordinary-mode admission policy: preserves current whole-message behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct OrdinaryAdmissionPolicy;

impl CarrierAdmissionPolicy for OrdinaryAdmissionPolicy {
    fn mode(&self) -> ProfileMode {
        ProfileMode::Ordinary
    }

    fn admit_request(&self, _envelope: Option<&[u8]>, _body: &[u8]) -> RequestAdmissionDecision {
        RequestAdmissionDecision::WholeMessage
    }

    fn fail(&self, outcome: VerificationOutcome) -> CarrierError {
        CarrierError {
            outcome,
            segment_index: None,
            supported_profile: String::new(),
            supported_schema_version: 0,
        }
    }
}

/// Verifier-backed request admission policy for [`ProfileMode::Ordinary`] or [`ProfileMode::RequiredOpenCode`].
///
/// Under [`ProfileMode::RequiredOpenCode`], request bodies are verified against the
/// bound [`ProvenanceRouteBinding`] and its matching [`ProviderContentDecoder`].
pub struct CarrierRequestAdmitter {
    mode: ProfileMode,
    route_binding: ProvenanceRouteBinding,
    verifier: CarrierVerifier,
    supported_profile: String,
    supported_schema_version: u32,
    decoder: ProviderContentDecoder,
}

impl CarrierRequestAdmitter {
    pub fn new(mode: ProfileMode, verifier: CarrierVerifier) -> Self {
        Self::new_for_route(mode, ProvenanceRouteBinding::OPENAI, verifier)
    }

    pub fn new_for_route(
        mode: ProfileMode,
        route_binding: impl Into<ProvenanceRouteBinding>,
        verifier: CarrierVerifier,
    ) -> Self {
        // `CarrierVerifier::verify(None, b"")` short-circuits on its first line
        // (`envelope.ok_or_else(|| self.fail(O::Absent, None))?`), returning only
        // the public `profile_id` and `schema_version` without parsing, reading
        // the clock, invoking the authenticator, or mutating the replay ledger.
        let (supported_profile, supported_schema_version) = match verifier.verify(None, b"") {
            Err(e) => (e.supported_profile, e.supported_schema_version),
            Ok(_) => ("opencode".to_string(), 1),
        };
        Self::with_identity(
            mode,
            route_binding.into(),
            verifier,
            supported_profile,
            supported_schema_version,
        )
    }

    fn with_identity(
        mode: ProfileMode,
        route_binding: ProvenanceRouteBinding,
        verifier: CarrierVerifier,
        supported_profile: impl Into<String>,
        supported_schema_version: u32,
    ) -> Self {
        let decoder = route_binding.content_decoder();
        Self {
            mode,
            route_binding,
            verifier,
            supported_profile: supported_profile.into(),
            supported_schema_version,
            decoder,
        }
    }

    pub fn required_opencode(verifier: CarrierVerifier) -> Self {
        Self::new(ProfileMode::RequiredOpenCode, verifier)
    }

    pub fn required_opencode_for_route(
        route_binding: impl Into<ProvenanceRouteBinding>,
        verifier: CarrierVerifier,
    ) -> Self {
        Self::new_for_route(ProfileMode::RequiredOpenCode, route_binding, verifier)
    }

    pub const fn bound_route(&self) -> ProvenanceRouteBinding {
        self.route_binding
    }
}

impl CarrierAdmissionPolicy for CarrierRequestAdmitter {
    fn mode(&self) -> ProfileMode {
        self.mode
    }

    fn route_binding(&self) -> Option<ProvenanceRouteBinding> {
        Some(self.route_binding)
    }

    fn admit_request(&self, envelope: Option<&[u8]>, body: &[u8]) -> RequestAdmissionDecision {
        match admit(self.mode, &self.verifier, envelope, body) {
            CarrierAdmission::WholeMessage => RequestAdmissionDecision::WholeMessage,
            CarrierAdmission::Projected(verified) => {
                use crate::provenance::MessageContentDecoder as _;
                match self.decoder.decode(body) {
                    Some(decoded) => {
                        let expected: Vec<String> =
                            decoded.into_iter().map(|m| m.content).collect();
                        if verified.messages == expected {
                            RequestAdmissionDecision::Verified(verified)
                        } else {
                            RequestAdmissionDecision::Reject(
                                self.fail(VerificationOutcome::MapOutOfRange),
                            )
                        }
                    }
                    None => RequestAdmissionDecision::Reject(
                        self.fail(VerificationOutcome::MapOutOfRange),
                    ),
                }
            }
            CarrierAdmission::Reject(err) => RequestAdmissionDecision::Reject(err),
        }
    }

    fn fail(&self, outcome: VerificationOutcome) -> CarrierError {
        CarrierError {
            outcome,
            segment_index: None,
            supported_profile: self.supported_profile.clone(),
            supported_schema_version: self.supported_schema_version,
        }
    }
}

/// [`SessionManager`]-backed request admission policy for [`ProfileMode::Ordinary`] or [`ProfileMode::RequiredOpenCode`].
///
/// Binds a closed [`ProvenanceRouteBinding`] (exact [`Provider`], canonical `POST` path, and
/// matching [`ProviderContentDecoder`]). Under [`ProfileMode::RequiredOpenCode`], absence of an
/// active session fails closed with [`VerificationOutcome::SessionMismatch`].
pub struct SessionRequestAdmitter {
    mode: ProfileMode,
    route_binding: ProvenanceRouteBinding,
    sessions: Arc<SessionManager>,
    supported_profile: String,
    supported_schema_version: u32,
    decoder: ProviderContentDecoder,
}

impl SessionRequestAdmitter {
    pub fn new(
        mode: ProfileMode,
        route_binding: impl Into<ProvenanceRouteBinding>,
        sessions: Arc<SessionManager>,
        supported_profile: impl Into<String>,
        supported_schema_version: u32,
    ) -> Self {
        let route_binding = route_binding.into();
        let decoder = route_binding.content_decoder();
        Self {
            mode,
            route_binding,
            sessions,
            supported_profile: supported_profile.into(),
            supported_schema_version,
            decoder,
        }
    }

    pub const fn bound_route(&self) -> ProvenanceRouteBinding {
        self.route_binding
    }

    pub const fn bound_provider(&self) -> Provider {
        self.route_binding.provider()
    }

    pub const fn canonical_post_path(&self) -> &'static str {
        self.route_binding.canonical_post_path()
    }
}

impl CarrierAdmissionPolicy for SessionRequestAdmitter {
    fn mode(&self) -> ProfileMode {
        self.mode
    }

    fn route_binding(&self) -> Option<ProvenanceRouteBinding> {
        Some(self.route_binding)
    }

    fn admit_request(&self, envelope: Option<&[u8]>, body: &[u8]) -> RequestAdmissionDecision {
        match self.mode {
            ProfileMode::Ordinary => RequestAdmissionDecision::WholeMessage,
            ProfileMode::RequiredOpenCode => match self.sessions.verify(envelope, body) {
                None => RequestAdmissionDecision::Reject(
                    self.fail(VerificationOutcome::SessionMismatch),
                ),
                Some(Err(err)) => RequestAdmissionDecision::Reject(err),
                Some(Ok(verified)) => {
                    use crate::provenance::MessageContentDecoder as _;
                    match self.decoder.decode(body) {
                        Some(decoded) => {
                            let expected: Vec<String> =
                                decoded.into_iter().map(|m| m.content).collect();
                            if verified.messages == expected {
                                RequestAdmissionDecision::Verified(verified)
                            } else {
                                RequestAdmissionDecision::Reject(
                                    self.fail(VerificationOutcome::MapOutOfRange),
                                )
                            }
                        }
                        None => RequestAdmissionDecision::Reject(
                            self.fail(VerificationOutcome::MapOutOfRange),
                        ),
                    }
                }
            },
        }
    }

    fn fail(&self, outcome: VerificationOutcome) -> CarrierError {
        CarrierError {
            outcome,
            segment_index: None,
            supported_profile: self.supported_profile.clone(),
            supported_schema_version: self.supported_schema_version,
        }
    }
}

/// Injected next-stage seam receiving a [`VerifiedCarrier`] on successful required-profile admission.
#[async_trait::async_trait]
pub trait VerifiedCarrierNextStage: Send + Sync {
    async fn handle_verified(
        &self,
        provider: Provider,
        request: ProxyRequest,
        verified: VerifiedCarrier,
    ) -> Result<ProxyResponse, ProxyError>;
}

fn carrier_rejection_response(err: &CarrierError) -> Response<Body> {
    let outcome_str = err.outcome.as_str();
    tracing::warn!(
        status = StatusCode::FORBIDDEN.as_u16(),
        error_type = "parapet_carrier_rejected",
        outcome = outcome_str,
        segment_index = ?err.segment_index,
        supported_profile = %err.supported_profile,
        supported_schema_version = err.supported_schema_version,
        "provenance carrier rejected"
    );
    let payload = serde_json::json!({
        "error": {
            "type": "parapet_carrier_rejected",
            "code": outcome_str,
            "outcome": outcome_str,
            "segment_index": err.segment_index,
            "supported_profile": err.supported_profile,
            "supported_schema_version": err.supported_schema_version,
            "message": err.to_string(),
        }
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let mut resp = Response::builder()
        .status(StatusCode::FORBIDDEN)
        .body(Body::from(bytes))
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp.headers_mut()
        .insert("x-parapet-error", HeaderValue::from_static(outcome_str));
    resp.headers_mut().insert(
        "x-parapet-blocked-by",
        HeaderValue::from_static("provenance"),
    );
    resp
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Heartbeat endpoint: GET /v1/heartbeat -> 200 OK
pub async fn heartbeat() -> StatusCode {
    StatusCode::OK
}

/// Proxy handler for provider paths.
///
/// Validates the request, detects the provider, extracts the provenance carrier,
/// enforces request admission, forwards via the injected upstream client (preserving
/// `x-guard-trust*` for whole-message trust parsing while stripping the carrier header)
/// or verified-carrier next-stage seam (with sanitized upstream headers), strips
/// guard-trust headers on responses, and returns the response.
pub async fn proxy_handler(
    State(state): State<AppState>,
    request: Request<Body>,
) -> impl IntoResponse {
    // Request receipt: budgets bound to receipt include classification,
    // body read, admission and validation below.
    let received_at = tokio::time::Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let inbound_headers = request.headers().clone();
    let extraction = extract_carrier_header(&inbound_headers);
    let sanitized_headers = extraction.upstream_headers;
    let carrier = extraction.carrier;
    let path = uri.path();

    // Classify path before reading body or forwarding
    let provider = match classify_path(path) {
        PathClass::Supported(p) => p,
        PathClass::NonCanonical => {
            if state.admission.mode() == ProfileMode::RequiredOpenCode {
                return carrier_rejection_response(
                    &state.admission.fail(VerificationOutcome::SchemaMismatch),
                );
            }
            return path_refusal_response(
                path,
                StatusCode::BAD_REQUEST,
                "parapet_path_not_canonical",
                "path_not_canonical",
            );
        }
        PathClass::Staged(StagedRoute::OpenAiResponses) => {
            if state.admission.mode() == ProfileMode::RequiredOpenCode {
                return carrier_rejection_response(
                    &state.admission.fail(VerificationOutcome::SchemaMismatch),
                );
            }
            if method != Method::POST {
                return method_not_allowed_response(path, &method);
            }
            // Ordinary-mode Responses requests enter the engine's dedicated
            // strict request/output path; streaming uses the activated
            // full-buffer executor. RequiredOpenCode was refused above.
            Provider::OpenAi
        }
        PathClass::KnownUnsupported => {
            if state.admission.mode() == ProfileMode::RequiredOpenCode {
                return carrier_rejection_response(
                    &state.admission.fail(VerificationOutcome::SchemaMismatch),
                );
            }
            return path_refusal_response(
                path,
                StatusCode::NOT_IMPLEMENTED,
                "parapet_unsupported_path",
                "unsupported_wire_format",
            );
        }
        PathClass::Unknown => {
            if state.admission.mode() == ProfileMode::RequiredOpenCode {
                return carrier_rejection_response(
                    &state.admission.fail(VerificationOutcome::SchemaMismatch),
                );
            }
            return path_refusal_response(
                path,
                StatusCode::NOT_FOUND,
                "parapet_unknown_path",
                "unknown_path",
            );
        }
    };

    // RequiredOpenCode binds exclusively to exact POST on the admission policy's bound Provider and canonical path.
    if state.admission.mode() == ProfileMode::RequiredOpenCode {
        let Some(binding) = state.admission.route_binding() else {
            return carrier_rejection_response(
                &state.admission.fail(VerificationOutcome::SchemaMismatch),
            );
        };
        if !binding.matches_post_route(&method, provider, path) {
            return carrier_rejection_response(
                &state.admission.fail(VerificationOutcome::SchemaMismatch),
            );
        }
    }

    if method != Method::POST {
        return method_not_allowed_response(path, &method);
    }

    // Read body
    let body = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("failed to read request body: {e}"),
            )
                .into_response()
        }
    };

    let verified_carrier = match state.admission.mode() {
        ProfileMode::Ordinary => None,
        ProfileMode::RequiredOpenCode => {
            let envelope = match carrier.envelope() {
                Ok(env) => env,
                Err(outcome) => {
                    return carrier_rejection_response(&state.admission.fail(outcome));
                }
            };
            match state.admission.admit_request(envelope, &body) {
                RequestAdmissionDecision::WholeMessage => None,
                RequestAdmissionDecision::Verified(verified) => Some(verified),
                RequestAdmissionDecision::Reject(err) => {
                    return carrier_rejection_response(&err);
                }
            }
        }
    };

    // Validate body
    if body.is_empty() {
        return ProxyError::EmptyBody.into_response();
    }
    if serde_json::from_slice::<serde_json::Value>(&body).is_err() {
        return ProxyError::MalformedJson(
            "request body is not valid JSON".into(),
        )
        .into_response();
    }

    let upstream_result = match verified_carrier {
        Some(verified) => {
            let next_stage = state
                .verified_next_stage
                .as_ref()
                .expect("AppState invariant: RequiredOpenCode requires VerifiedCarrierNextStage");
            let proxy_req = ProxyRequest {
                method,
                uri,
                headers: sanitized_headers,
                body,
            };
            next_stage
                .handle_verified(provider, proxy_req, verified)
                .await
        }
        None => {
            let mut whole_message_headers = inbound_headers;
            whole_message_headers.remove(crate::provenance::PROVENANCE_HEADER);
            let proxy_req = ProxyRequest {
                method,
                uri,
                headers: whole_message_headers,
                body,
            };
            state
                .upstream
                .forward_received(provider, proxy_req, received_at)
                .await
        }
    };

    match upstream_result {
        Ok(mut resp) => {
            strip_guard_trust_headers(&mut resp.headers);

            let mut response = Response::builder().status(resp.status);
            if let Some(h) = response.headers_mut() {
                *h = resp.headers;
            }
            response.body(resp.body).unwrap().into_response()
        }
        Err(e) => e.into_response(),
    }
}

// ---------------------------------------------------------------------------
// Router construction
// ---------------------------------------------------------------------------

/// Build the axum router with all proxy routes and the heartbeat endpoint.
///
/// The upstream client is injected — no side effects, no hard-coded clients.
pub fn build_router(upstream: Arc<dyn UpstreamClient>) -> Router {
    build_router_with_admission(upstream, Arc::new(OrdinaryAdmissionPolicy), None)
}

/// Build the axum router with an injected [`CarrierAdmissionPolicy`] and
/// optional [`VerifiedCarrierNextStage`] seam.
///
/// # Panics
///
/// Panics if `admission.mode()` is [`ProfileMode::RequiredOpenCode`] and
/// `verified_next_stage` is `None`.
pub fn build_router_with_admission(
    upstream: Arc<dyn UpstreamClient>,
    admission: Arc<dyn CarrierAdmissionPolicy>,
    verified_next_stage: Option<Arc<dyn VerifiedCarrierNextStage>>,
) -> Router {
    let state = AppState::new(upstream, admission, verified_next_stage);

    Router::new()
        .route("/v1/heartbeat", get(heartbeat))
        .fallback(proxy_handler)
        .with_state(state)
}

/// The address the proxy binds to. Always localhost, never 0.0.0.0.
pub const BIND_ADDR: ([u8; 4], u16) = ([127, 0, 0, 1], 9800);

impl ProxyResponse {
    pub fn from_bytes(status: StatusCode, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: HeaderMap::new(),
            body: Body::from(body),
        }
    }

    /// Create a 403 response tagged with the layer that blocked it.
    /// Adds `X-Parapet-Blocked-By` header for eval/observability.
    pub fn blocked(body: Vec<u8>, layer: &str) -> Self {
        let mut headers = HeaderMap::new();
        if let Ok(val) = layer.parse() {
            headers.insert("x-parapet-blocked-by", val);
        }
        Self {
            status: StatusCode::FORBIDDEN,
            headers,
            body: Body::from(body),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use bytes::Bytes;
    use std::sync::Arc;
    use tower::ServiceExt; // for oneshot

    // -----------------------------------------------------------------------
    // Mock upstream client
    // -----------------------------------------------------------------------

    /// A mock upstream client for testing. Returns a configurable response.
    /// Proves the DI pattern works: handlers never touch a real HTTP client.
    #[derive(Clone)]
    struct MockUpstreamClient {
        status: StatusCode,
        headers: HeaderMap,
        body: Bytes,
    }

    impl MockUpstreamClient {
        fn new(status: StatusCode, headers: HeaderMap, body: &[u8]) -> Self {
            Self {
                status,
                headers,
                body: Bytes::copy_from_slice(body),
            }
        }

        fn ok_json(body: &str) -> Self {
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            Self::new(StatusCode::OK, headers, body.as_bytes())
        }
    }

    #[async_trait::async_trait]
    impl UpstreamClient for MockUpstreamClient {
        async fn forward(
            &self,
            _provider: Provider,
            _request: ProxyRequest,
        ) -> Result<ProxyResponse, ProxyError> {
            Ok(ProxyResponse {
                status: self.status,
                headers: self.headers.clone(),
                body: Body::from(self.body.clone()),
            })
        }
    }

    /// A mock that captures the provider it was called with.
    struct CapturingClient {
        captured_provider: tokio::sync::Mutex<Option<Provider>>,
    }

    impl CapturingClient {
        fn new() -> Self {
            Self {
                captured_provider: tokio::sync::Mutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl UpstreamClient for CapturingClient {
        async fn forward(
            &self,
            provider: Provider,
            _request: ProxyRequest,
        ) -> Result<ProxyResponse, ProxyError> {
            *self.captured_provider.lock().await = Some(provider);
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            Ok(ProxyResponse {
                status: StatusCode::OK,
                headers,
                body: Body::from(Bytes::from_static(b"{\"ok\":true}")),
            })
        }
    }

    fn json_body(s: &str) -> Body {
        Body::from(s.to_owned())
    }

    fn json_request(method: &str, path: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(json_body(body))
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // Test 1: /v1/chat/completions detected as OpenAI
    // -----------------------------------------------------------------------

    #[test]
    fn detect_openai_provider() {
        assert_eq!(
            detect_provider("/v1/chat/completions"),
            Some(Provider::OpenAi)
        );
    }

    #[test]
    fn detect_openai_provider_with_openai_prefix() {
        assert_eq!(
            detect_provider("/openai/v1/chat/completions"),
            Some(Provider::OpenAi)
        );
    }

    #[tokio::test]
    async fn request_to_chat_completions_routes_as_openai() {
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());

        let req = json_request("POST", "/v1/chat/completions", r#"{"model":"gpt-4o"}"#);
        let _resp = app.oneshot(req).await.unwrap();

        let captured = client.captured_provider.lock().await;
        assert_eq!(*captured, Some(Provider::OpenAi));
    }

    /// A client implementing only `forward` keeps forwarding through the
    /// default `forward_received`, regardless of how long ago receipt was.
    #[tokio::test(start_paused = true)]
    async fn default_forward_received_delegates_to_forward() {
        let client = Arc::new(CapturingClient::new());
        let received_at = tokio::time::Instant::now();
        tokio::time::advance(std::time::Duration::from_secs(3_600)).await;
        let resp = client
            .forward_received(
                Provider::Anthropic,
                ProxyRequest {
                    method: Method::POST,
                    uri: "/v1/messages".parse().unwrap(),
                    headers: HeaderMap::new(),
                    body: Bytes::from_static(b"{}"),
                },
                received_at,
            )
            .await
            .unwrap();
        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(
            *client.captured_provider.lock().await,
            Some(Provider::Anthropic)
        );

        // Through the handler, with the body read stalling on the clock.
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());
        let slow_body = futures_util::stream::once(async {
            tokio::time::sleep(std::time::Duration::from_secs(3_600)).await;
            Ok::<_, std::convert::Infallible>(Bytes::from_static(b"{\"model\":\"gpt-4o\"}"))
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from_stream(slow_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            *client.captured_provider.lock().await,
            Some(Provider::OpenAi)
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: /v1/messages detected as Anthropic
    // -----------------------------------------------------------------------

    #[test]
    fn detect_anthropic_provider() {
        assert_eq!(
            detect_provider("/v1/messages"),
            Some(Provider::Anthropic)
        );
    }

    #[tokio::test]
    async fn request_to_messages_routes_as_anthropic() {
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());

        let req = json_request("POST", "/v1/messages", r#"{"model":"claude-3"}"#);
        let _resp = app.oneshot(req).await.unwrap();

        let captured = client.captured_provider.lock().await;
        assert_eq!(*captured, Some(Provider::Anthropic));
    }

    // -----------------------------------------------------------------------
    // Test 3: Unknown path returns 404
    // -----------------------------------------------------------------------

    #[test]
    fn detect_unknown_path_returns_none() {
        assert_eq!(detect_provider("/v1/unknown"), None);
        assert_eq!(detect_provider("/foo/bar"), None);
        assert_eq!(detect_provider("/"), None);
    }

    #[tokio::test]
    async fn unknown_path_returns_404() {
        let client = Arc::new(MockUpstreamClient::ok_json(r#"{"ok":true}"#));
        let app = build_router(client);

        let req = json_request("POST", "/v1/unknown", r#"{"data":"test"}"#);
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            resp.headers()
                .get("x-parapet-error")
                .and_then(|v| v.to_str().ok()),
            Some("unknown_path")
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "parapet_unknown_path");
        assert_eq!(json["error"]["code"], "unknown_path");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("Parapet does not inspect /v1/unknown."));
        assert!(msg.contains("POST /v1/chat/completions"));
        assert!(msg.contains("POST /openai/v1/chat/completions"));
        assert!(msg.contains("POST /v1/messages"));
    }

    // -----------------------------------------------------------------------
    // Test 4: Response headers X-Guard-Trust* stripped (others preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn strip_guard_trust_headers_removes_matching() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert("x-guard-trust", HeaderValue::from_static("secret"));
        headers.insert(
            "x-guard-trust-level",
            HeaderValue::from_static("high"),
        );
        headers.insert("x-request-id", HeaderValue::from_static("abc123"));

        strip_guard_trust_headers(&mut headers);

        assert!(headers.get("x-guard-trust").is_none());
        assert!(headers.get("x-guard-trust-level").is_none());
        assert_eq!(
            headers.get("content-type").unwrap(),
            "application/json"
        );
        assert_eq!(headers.get("x-request-id").unwrap(), "abc123");
    }

    #[tokio::test]
    async fn proxy_strips_guard_trust_headers_from_response() {
        let mut upstream_headers = HeaderMap::new();
        upstream_headers.insert(
            "content-type",
            HeaderValue::from_static("application/json"),
        );
        upstream_headers.insert(
            "x-guard-trust-level",
            HeaderValue::from_static("high"),
        );
        upstream_headers.insert("x-custom", HeaderValue::from_static("preserved"));

        let client = Arc::new(MockUpstreamClient::new(
            StatusCode::OK,
            upstream_headers,
            b"{\"ok\":true}",
        ));
        let app = build_router(client);

        let req = json_request("POST", "/v1/chat/completions", r#"{"model":"gpt-4o"}"#);
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("x-guard-trust-level").is_none());
        assert_eq!(resp.headers().get("x-custom").unwrap(), "preserved");
    }

    // -----------------------------------------------------------------------
    // Test 5: Binding enforcement — server config specifies 127.0.0.1
    // -----------------------------------------------------------------------

    #[test]
    fn bind_address_is_localhost_only() {
        assert_eq!(BIND_ADDR.0, [127, 0, 0, 1]);
        assert_ne!(BIND_ADDR.0, [0, 0, 0, 0]);
        assert_eq!(BIND_ADDR.1, 9800);
    }

    // -----------------------------------------------------------------------
    // Test 6: Upstream 5xx errors passed through unchanged
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn upstream_5xx_passed_through() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        let error_body = r#"{"error":{"message":"internal server error","type":"server_error"}}"#;
        let client = Arc::new(MockUpstreamClient::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            headers,
            error_body.as_bytes(),
        ));
        let app = build_router(client);

        let req = json_request("POST", "/v1/chat/completions", r#"{"model":"gpt-4o"}"#);
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body, error_body.as_bytes());
    }

    // -----------------------------------------------------------------------
    // Test 7: Malformed JSON request body -> 400 with clear error
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn malformed_json_returns_400() {
        let client = Arc::new(MockUpstreamClient::ok_json(r#"{"ok":true}"#));
        let app = build_router(client);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("this is not json {{{"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(
            body_str.contains("not valid JSON"),
            "error message should mention JSON, got: {body_str}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: Empty request body -> 400
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn empty_body_returns_400() {
        let client = Arc::new(MockUpstreamClient::ok_json(r#"{"ok":true}"#));
        let app = build_router(client);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(
            body_str.contains("empty"),
            "error message should mention empty body, got: {body_str}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Multiple concurrent requests handled correctly
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn concurrent_requests_handled() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Client that counts how many times it was called.
        struct CountingClient {
            count: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl UpstreamClient for CountingClient {
            async fn forward(
                &self,
                _provider: Provider,
                _request: ProxyRequest,
            ) -> Result<ProxyResponse, ProxyError> {
                self.count.fetch_add(1, Ordering::SeqCst);
                // Simulate some upstream latency
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                let mut headers = HeaderMap::new();
                headers.insert(
                    "content-type",
                    HeaderValue::from_static("application/json"),
                );
                Ok(ProxyResponse {
                    status: StatusCode::OK,
                    headers,
                    body: Body::from(Bytes::from_static(b"{\"ok\":true}")),
                })
            }
        }

        let client = Arc::new(CountingClient {
            count: AtomicUsize::new(0),
        });

        let num_requests = 10;
        let mut handles = Vec::new();

        for i in 0..num_requests {
            let app = build_router(client.clone());
            handles.push(tokio::spawn(async move {
                let req = json_request(
                    "POST",
                    "/v1/chat/completions",
                    &format!(r#"{{"request":{i}}}"#),
                );
                let resp = app.oneshot(req).await.unwrap();
                resp.status()
            }));
        }

        for handle in handles {
            let status = handle.await.unwrap();
            assert_eq!(status, StatusCode::OK);
        }

        assert_eq!(client.count.load(Ordering::SeqCst), num_requests);
    }

    // -----------------------------------------------------------------------
    // Additional: heartbeat returns 200
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn heartbeat_returns_200() {
        let client = Arc::new(MockUpstreamClient::ok_json(r#"{"ok":true}"#));
        let app = build_router(client);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/heartbeat")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // -----------------------------------------------------------------------
    // ProxyError::UpstreamTimeout returns 504
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn upstream_timeout_returns_504() {
        struct TimeoutClient;

        #[async_trait::async_trait]
        impl UpstreamClient for TimeoutClient {
            async fn forward(
                &self,
                _provider: Provider,
                _request: ProxyRequest,
            ) -> Result<ProxyResponse, ProxyError> {
                Err(ProxyError::UpstreamTimeout("request timed out after 5000ms".to_string()))
            }
        }

        let client = Arc::new(TimeoutClient);
        let app = build_router(client);

        let req = json_request("POST", "/v1/chat/completions", r#"{"model":"gpt-4o"}"#);
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
    }

    #[test]
    fn proxy_error_upstream_failure_is_502() {
        let err = ProxyError::UpstreamFailure("connection refused".to_string());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn proxy_error_upstream_timeout_is_504() {
        let err = ProxyError::UpstreamTimeout("timed out".to_string());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
    }

    // -----------------------------------------------------------------------
    // Stage 0 path classification tests (T10, T11, T12)
    // -----------------------------------------------------------------------

    #[test]
    fn t10_classify_path_non_canonical() {
        for path in [
            "/v1/chat/completions/../../embeddings",
            "/v1/chat/completions/%2e%2e/%2E%2E/embeddings",
            "/v1//chat/completions",
            "/v1/chat/completions%2f..",
            "/v1\\chat\\completions",
            "/v1/chat/completions/.%2e/embeddings",
            "/v1/chat/completions/%2e./embeddings",
            "/v1/chat/completions/%2e",
            "/v1/chat/completions%5c..",
            "/v1/messages/count%5Ftokens",
            "/v1/messages/%63ount_tokens",
            "/v1/messages/b%61tches",
        ] {
            assert_eq!(
                classify_path(path),
                PathClass::NonCanonical,
                "expected NonCanonical for {path}"
            );
        }
    }

    #[test]
    fn t11_classify_path_known_unsupported() {
        for path in [
            "/v1/embeddings",
            "/v1/completions",
            "/v1/models",
            "/v1/messages/count_tokens",
            "/openai/v1/responses",
            "/v1/audio",
            "/v1/images",
            "/v1/files",
            "/v1/batches",
            "/v1/moderations",
            "/v1/assistants",
            "/v1/threads",
            "/v1/realtime",
            "/v1/messages/batches",
            "/openai/v1/embeddings",
            "/openai/v1/messages/count_tokens",
        ] {
            assert_eq!(
                classify_path(path),
                PathClass::KnownUnsupported,
                "expected KnownUnsupported for {path}"
            );
        }
    }

    #[test]
    fn t12_classify_path_supported_and_pinned_compat() {
        assert_eq!(
            classify_path("/v1/chat/completions"),
            PathClass::Supported(Provider::OpenAi)
        );
        assert_eq!(
            classify_path("/openai/v1/chat/completions"),
            PathClass::Supported(Provider::OpenAi)
        );
        assert_eq!(
            classify_path("/v1/messages"),
            PathClass::Supported(Provider::Anthropic)
        );
        assert_eq!(
            classify_path("/v1/chat/completionsXYZ"),
            PathClass::Supported(Provider::OpenAi)
        );
    }

    #[test]
    fn responses_route_matrix_is_exact() {
        for path in ["/v1/responses", "/responses"] {
            assert_eq!(
                classify_path(path),
                PathClass::Staged(StagedRoute::OpenAiResponses),
                "expected Staged for {path}"
            );
        }

        // Retrieval/sub-resources and the /openai alias stay generic unsupported.
        for path in [
            "/v1/responses/",
            "/v1/responses/resp_1",
            "/v1/responses/resp_1/input_items",
            "/v1/responses/resp_1/cancel",
            "/openai/v1/responses",
            "/openai/v1/responses/resp_1",
        ] {
            assert_eq!(
                classify_path(path),
                PathClass::KnownUnsupported,
                "expected KnownUnsupported for {path}"
            );
        }

        // Suffix, case, and alias confusion never match the staged route.
        for path in [
            "/responses/",
            "/responses/resp_1",
            "/responsesX",
            "/v1/responsesX",
            "/v1/responses.json",
            "/RESPONSES",
            "/v1/Responses",
            "/openai/responses",
            "/v2/responses",
            "/api/v1/responses",
            "/v1/chat/responses",
        ] {
            assert!(
                !matches!(classify_path(path), PathClass::Staged(_)),
                "unexpected Staged for {path}"
            );
            assert!(
                !matches!(classify_path(path), PathClass::Supported(_)),
                "unexpected Supported for {path}"
            );
        }

        // Traversal and encoding tricks are refused as non-canonical first.
        for path in [
            "/v1/../responses",
            "/v1/./responses",
            "/v1//responses",
            "//responses",
            "/v1/%2e%2e/responses",
            "/v1/%72esponses",
            "/%72esponses",
            "/responses%2f",
            "/v1\\responses",
            "/v1/chat/completions/../../responses",
            "/responses/..",
        ] {
            assert_eq!(
                classify_path(path),
                PathClass::NonCanonical,
                "expected NonCanonical for {path}"
            );
        }
    }

    #[tokio::test]
    async fn responses_routes_forward_only_exact_post_in_ordinary_mode() {
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());
        let valid_body = r#"{"model":"gpt-test","instructions":"be brief","input":"hi"}"#;

        for path in ["/v1/responses", "/responses"] {
            let resp = app
                .clone()
                .oneshot(json_request("POST", path, valid_body))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "path={path}");
            assert_eq!(
                *client.captured_provider.lock().await,
                Some(Provider::OpenAi),
                "path={path}"
            );

            for method in ["GET", "PUT", "DELETE", "PATCH"] {
                let resp = app
                    .clone()
                    .oneshot(json_request(method, path, valid_body))
                    .await
                    .unwrap();
                assert_eq!(
                    resp.status(),
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method={method} path={path}"
                );
                assert_eq!(
                    resp.headers().get("allow").and_then(|v| v.to_str().ok()),
                    Some("POST")
                );
            }
        }

        *client.captured_provider.lock().await = None;

        for (path, status) in [
            ("/responses/", StatusCode::NOT_FOUND),
            ("/openai/responses", StatusCode::NOT_FOUND),
            ("/openai/v1/responses", StatusCode::NOT_IMPLEMENTED),
            ("/v1/responses/resp_1", StatusCode::NOT_IMPLEMENTED),
            ("/v1/../responses", StatusCode::BAD_REQUEST),
        ] {
            let resp = app
                .clone()
                .oneshot(json_request("POST", path, valid_body))
                .await
                .unwrap();
            assert_eq!(resp.status(), status, "path={path}");
        }

        assert!(
            client.captured_provider.lock().await.is_none(),
            "Responses aliases, subpaths, and non-canonical paths must not forward"
        );
    }

    #[tokio::test]
    async fn proxy_handler_refuses_unsupported_and_non_canonical_without_forwarding() {
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());

        let resp_501 = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/openai/v1/responses?secret_query=should_not_echo")
                    .body(Body::from("not even json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_501.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            resp_501
                .headers()
                .get("x-parapet-error")
                .and_then(|v| v.to_str().ok()),
            Some("unsupported_wire_format")
        );
        let body_501 = axum::body::to_bytes(resp_501.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json_501: serde_json::Value = serde_json::from_slice(&body_501).unwrap();
        assert_eq!(json_501["error"]["type"], "parapet_unsupported_path");
        assert_eq!(json_501["error"]["code"], "unsupported_wire_format");
        let msg_501 = json_501["error"]["message"].as_str().unwrap();
        assert!(msg_501.contains("Parapet does not inspect /openai/v1/responses."));
        assert!(!msg_501.contains("secret_query"));

        for non_canonical_path in [
            "/v1/chat/completions/../../embeddings",
            "/v1/messages/count%5Ftokens",
            "/v1/messages/%63ount_tokens",
            "/v1/messages/b%61tches",
        ] {
            let resp_400 = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("{non_canonical_path}?secret_query=should_not_echo"))
                        .header("authorization", "Bearer secret-token-should-not-echo")
                        .body(Body::from(r#"{"secret_body":"should_not_echo"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                resp_400.status(),
                StatusCode::BAD_REQUEST,
                "path={non_canonical_path}"
            );
            assert_eq!(
                resp_400
                    .headers()
                    .get("x-parapet-error")
                    .and_then(|v| v.to_str().ok()),
                Some("path_not_canonical")
            );
            let body_400 = axum::body::to_bytes(resp_400.into_body(), 1024 * 1024)
                .await
                .unwrap();
            let json_400: serde_json::Value = serde_json::from_slice(&body_400).unwrap();
            assert_eq!(json_400["error"]["type"], "parapet_path_not_canonical");
            assert_eq!(json_400["error"]["code"], "path_not_canonical");
            let msg_400 = json_400["error"]["message"].as_str().unwrap();
            assert!(!msg_400.contains("secret_query"));
            assert!(!msg_400.contains("secret-token-should-not-echo"));
            assert!(!msg_400.contains("should_not_echo"));
        }

        assert!(client.captured_provider.lock().await.is_none());
    }

    #[tokio::test]
    async fn t_s1_17_proxy_handler_non_post_supported_routes_with_valid_json_return_405_zero_forward(
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingClient {
            count: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl UpstreamClient for CountingClient {
            async fn forward(
                &self,
                _provider: Provider,
                _request: ProxyRequest,
            ) -> Result<ProxyResponse, ProxyError> {
                self.count.fetch_add(1, Ordering::SeqCst);
                Ok(ProxyResponse::from_bytes(
                    StatusCode::OK,
                    b"{\"ok\":true}".to_vec(),
                ))
            }
        }

        let client = Arc::new(CountingClient {
            count: AtomicUsize::new(0),
        });
        let app = build_router(client.clone());
        let valid_json = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

        for path in [
            "/v1/chat/completions",
            "/openai/v1/chat/completions",
            "/v1/messages",
        ] {
            for method in ["GET", "PUT", "DELETE"] {
                let req = Request::builder()
                    .method(method)
                    .uri(format!("{path}?secret_query=should_not_echo"))
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer secret-token-should-not-echo")
                    .body(Body::from(valid_json))
                    .unwrap();
                let resp = app.clone().oneshot(req).await.unwrap();
                assert_eq!(
                    resp.status(),
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method={method} path={path}"
                );
                assert_eq!(
                    resp.headers().get("allow").and_then(|v| v.to_str().ok()),
                    Some("POST"),
                    "method={method} path={path}"
                );
                assert_eq!(
                    resp.headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok()),
                    Some("application/json"),
                    "method={method} path={path}"
                );
                assert_eq!(
                    resp.headers()
                        .get("x-parapet-error")
                        .and_then(|v| v.to_str().ok()),
                    Some("method_not_allowed"),
                    "method={method} path={path}"
                );
                let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(json["error"]["type"], "parapet_method_not_allowed");
                assert_eq!(json["error"]["code"], "method_not_allowed");
                let msg = json["error"]["message"].as_str().unwrap();
                assert!(msg.contains(&format!("Method {method} is not allowed on {path}.")));
                assert!(msg.contains("POST /v1/chat/completions"));
                assert!(msg.contains("POST /openai/v1/chat/completions"));
                assert!(msg.contains("POST /v1/messages"));
                assert!(!msg.contains("secret_query"));
                assert!(!msg.contains("secret-token-should-not-echo"));
            }
        }

        assert_eq!(client.count.load(Ordering::SeqCst), 0);
    }
}
