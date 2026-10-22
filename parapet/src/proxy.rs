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
}

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

/// Classification of an inbound request path before provider detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathClass {
    Supported(Provider),
    KnownUnsupported,
    NonCanonical,
    Unknown,
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

/// Shared state injected into axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub upstream: Arc<dyn UpstreamClient>,
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
/// Validates the request, detects the provider, forwards via the injected
/// upstream client, strips guard-trust headers, and returns the response.
pub async fn proxy_handler(
    State(state): State<AppState>,
    request: Request<Body>,
) -> impl IntoResponse {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let headers = request.headers().clone();
    let path = uri.path();

    // Classify path before reading body or forwarding
    let provider = match classify_path(path) {
        PathClass::Supported(p) => p,
        PathClass::NonCanonical => {
            return path_refusal_response(
                path,
                StatusCode::BAD_REQUEST,
                "parapet_path_not_canonical",
                "path_not_canonical",
            );
        }
        PathClass::KnownUnsupported => {
            return path_refusal_response(
                path,
                StatusCode::NOT_IMPLEMENTED,
                "parapet_unsupported_path",
                "unsupported_wire_format",
            );
        }
        PathClass::Unknown => {
            return path_refusal_response(
                path,
                StatusCode::NOT_FOUND,
                "parapet_unknown_path",
                "unknown_path",
            );
        }
    };

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

    let proxy_req = ProxyRequest {
        method,
        uri,
        headers,
        body,
    };

    // Forward to upstream
    match state.upstream.forward(provider, proxy_req).await {
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
    let state = AppState { upstream };

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
            "/v1/responses",
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

    #[tokio::test]
    async fn proxy_handler_refuses_unsupported_and_non_canonical_without_forwarding() {
        let client = Arc::new(CapturingClient::new());
        let app = build_router(client.clone());

        let resp_501 = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses?secret_query=should_not_echo")
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
        assert!(msg_501.contains("Parapet does not inspect /v1/responses."));
        assert!(!msg_501.contains("secret_query"));

        let resp_400 = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions/../../embeddings")
                    .body(Body::from("not even json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_400.status(), StatusCode::BAD_REQUEST);
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

        assert!(client.captured_provider.lock().await.is_none());
    }
}
