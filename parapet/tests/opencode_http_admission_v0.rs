// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! W3a bounded required-profile HTTP admission and header sanitation seam.
//!
//! Full-handler (`build_router` / `build_router_with_admission`) behavior tests
//! using local doubles only (no network, no model weights).

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use bytes::Bytes;
use parapet::config::{load_config, Config, StringSource};
use parapet::constraint::DslConstraintEvaluator;
use parapet::engine::{
    DefaultProviderRegistry, EngineDeps, EngineUpstreamClient, HttpError, HttpRequest,
    HttpResponse, HttpSender, UpstreamResolver,
};
use parapet::layers::l1_harness::{CalibrationParams, L1Model};
use parapet::layers::l3_inbound::DefaultInboundScanner;
use parapet::layers::l5a::L5aScanner;
use parapet::message::Message;
use parapet::normalize::Normalizer;
use parapet::provenance::*;
use parapet::provider::ProviderType;
use parapet::proxy::{
    build_router, build_router_with_admission, CarrierAdmissionPolicy, CarrierRequestAdmitter,
    ProvenanceRouteBinding, Provider, ProxyError, ProxyRequest, ProxyResponse,
    SessionRequestAdmitter, UpstreamClient, VerifiedCarrierNextStage,
};
use parapet::routing::NoopOrthogonalSensorRouter;
use parapet::signal::combiner::DefaultVerdictCombiner;
use parapet::signal::extractor::DefaultSignalExtractor;
use parapet::trust::TrustAssigner;
use serde_json::{json, Value};
use tower::ServiceExt;

const SESSION_ID: &str = "session-w3a-0001";
const PROFILE_ID: &str = "opencode";
const BUILD_ID: &str = "opencode-2.0.32";
const NOW_MS: u64 = 1_700_000_000_000;
const MAX_VALIDITY_MS: u64 = 60_000;

const CANARY_KEY_BYTES: [u8; SESSION_KEY_LEN] = *b"CANARY-KEY-W3A-SECRET-BYTES-0001";
const CANARY_KEY_STR: &str = "CANARY-KEY-W3A-SECRET-BYTES-0001";
const CANARY_VENDOR: &str = "You are a coding agent. Follow protocol.\n";
const CANARY_REPO: &str = "Repository rule: CANARY-REPO-SEG-8b4e\n";
const CANARY_USER: &str = "Inspect /home/user/secret/project/src/lib.rs CANARY-BODY-TEXT-3c9a";
const CANARY_LOCAL_PATH: &str = "/home/user/secret/project/src/lib.rs";
const CANARY_RAW_ENV_CLAIM: &str = "CANARY-UNVERIFIED-ENVELOPE-CLAIM-9f1d";
const CANARY_TRUST_HEADER: &str = "CANARY-GUARD-TRUST-HEADER-4a2b";
const CANARY_AUTH_HEADER: &str = "Bearer CANARY-UPSTREAM-AUTH-7e5c";

struct FixedClock(u64);

impl Clock for FixedClock {
    fn now_ms(&self) -> u64 {
        self.0
    }
}

#[derive(Default)]
struct CapturingUpstream {
    calls: AtomicUsize,
    last_provider: Mutex<Option<Provider>>,
    last_request: Mutex<Option<ProxyRequest>>,
    response_headers: Mutex<HeaderMap>,
}

impl CapturingUpstream {
    fn new() -> Self {
        Self::default()
    }

    fn with_response_header(key: &'static str, val: &'static str) -> Self {
        let mut h = HeaderMap::new();
        h.insert(key, HeaderValue::from_static(val));
        Self {
            calls: AtomicUsize::new(0),
            last_provider: Mutex::new(None),
            last_request: Mutex::new(None),
            response_headers: Mutex::new(h),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn take_last_request(&self) -> Option<ProxyRequest> {
        self.last_request.lock().unwrap().take()
    }
}

#[async_trait::async_trait]
impl UpstreamClient for CapturingUpstream {
    async fn forward(
        &self,
        provider: Provider,
        request: ProxyRequest,
    ) -> Result<ProxyResponse, ProxyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_provider.lock().unwrap() = Some(provider);
        *self.last_request.lock().unwrap() = Some(request);
        let mut headers = self.response_headers.lock().unwrap().clone();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        Ok(ProxyResponse {
            status: StatusCode::OK,
            headers,
            body: Body::from(Bytes::from_static(b"{\"ok\":true}")),
        })
    }
}

#[derive(Default)]
struct CapturingNextStage {
    calls: AtomicUsize,
    captured: Mutex<Vec<(Provider, ProxyRequest, VerifiedCarrier)>>,
}

impl CapturingNextStage {
    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn take_captured(&self) -> Vec<(Provider, ProxyRequest, VerifiedCarrier)> {
        std::mem::take(&mut *self.captured.lock().unwrap())
    }
}

#[async_trait::async_trait]
impl VerifiedCarrierNextStage for CapturingNextStage {
    async fn handle_verified(
        &self,
        provider: Provider,
        request: ProxyRequest,
        verified: VerifiedCarrier,
    ) -> Result<ProxyResponse, ProxyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.captured
            .lock()
            .unwrap()
            .push((provider, request, verified));
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert(
            "x-guard-trust-response",
            HeaderValue::from_static(CANARY_TRUST_HEADER),
        );
        Ok(ProxyResponse {
            status: StatusCode::OK,
            headers,
            body: Body::from(Bytes::from_static(b"{\"verified_seam\":true}")),
        })
    }
}

fn build_verifier_with_decoder(decoder: Box<dyn MessageContentDecoder>) -> CarrierVerifier {
    let mut builds = BTreeMap::new();
    builds.insert(
        BUILD_ID.to_string(),
        VendorManifest::new([Sha256Digest::of(CANARY_VENDOR.as_bytes())]),
    );
    CarrierVerifier::new(CarrierVerifierDeps {
        profile: SupportedProfile {
            schema_version: SCHEMA_VERSION_V1,
            profile_id: PROFILE_ID.to_string(),
            builds,
            max_validity_ms: MAX_VALIDITY_MS,
        },
        session_id: SESSION_ID.to_string(),
        authenticator: Box::new(HmacSha256Authenticator::new(SessionKey::from_bytes(
            CANARY_KEY_BYTES,
        ))),
        clock: Box::new(FixedClock(NOW_MS)),
        ledger: Box::new(InMemoryReplayLedger::new()),
        decoder,
    })
}

fn build_verifier() -> CarrierVerifier {
    build_verifier_with_decoder(Box::new(ProviderContentDecoder::new(ProviderType::OpenAi)))
}

fn valid_chat_body_and_envelope(sequence: u64) -> (Vec<u8>, Vec<u8>, String) {
    let sys_content = format!("{CANARY_VENDOR}{CANARY_REPO}");
    let body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": CANARY_USER}
        ]
    }))
    .unwrap();

    let v_len = CANARY_VENDOR.len();
    let s_len = sys_content.len();
    let u_len = CANARY_USER.len();
    let source_map = SourceMap {
        segments: vec![
            Segment {
                message_index: 0,
                start: 0,
                end: v_len,
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
            },
            Segment {
                message_index: 0,
                start: v_len,
                end: s_len,
                source: Source::Repository,
                sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: u_len,
                source: Source::User,
                sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
            },
        ],
    };

    let mut env = CarrierEnvelope {
        schema_version: SCHEMA_VERSION_V1,
        profile: PROFILE_ID.to_string(),
        build: BUILD_ID.to_string(),
        session_id: SESSION_ID.to_string(),
        sequence,
        issued_at_ms: NOW_MS - 1_000,
        expires_at_ms: NOW_MS + 10_000,
        request_body_sha256: Sha256Digest::of(&body),
        source_map_sha256: source_map.canonical_digest(),
        source_map,
        authenticator: Authenticator::from_bytes(vec![0u8; 32]),
    };
    let auth = HmacSha256Authenticator::new(SessionKey::from_bytes(CANARY_KEY_BYTES));
    let tag = auth.tag(&env.signed_payload());
    let tag_hex = Sha256Digest::from_bytes(tag).to_hex();
    env.authenticator = Authenticator::from_bytes(tag.to_vec());
    let env_bytes = serde_json::to_vec(&env).unwrap();
    (body, env_bytes, tag_hex)
}

static ACTIVE_TRACE_BUFS: std::sync::LazyLock<Mutex<Vec<Arc<Mutex<Vec<u8>>>>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone, Default)]
struct GlobalTraceWriter;

impl Write for GlobalTraceWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let sinks = ACTIVE_TRACE_BUFS.lock().unwrap().clone();
        for buf in sinks {
            buf.lock().unwrap().extend_from_slice(data);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for GlobalTraceWriter {
    type Writer = GlobalTraceWriter;

    fn make_writer(&'a self) -> Self::Writer {
        GlobalTraceWriter
    }
}

fn ensure_global_trace_subscriber() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_writer(GlobalTraceWriter)
            .without_time()
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

struct ThreadTraceGuard {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl ThreadTraceGuard {
    fn new() -> Self {
        ensure_global_trace_subscriber();
        let buf = Arc::new(Mutex::new(Vec::new()));
        ACTIVE_TRACE_BUFS.lock().unwrap().push(buf.clone());
        Self { buf }
    }
}

impl Drop for ThreadTraceGuard {
    fn drop(&mut self) {
        ACTIVE_TRACE_BUFS
            .lock()
            .unwrap()
            .retain(|b| !Arc::ptr_eq(b, &self.buf));
    }
}

async fn assert_carrier_rejection(resp: axum::response::Response, expected_outcome: &str) -> Value {
    ensure_global_trace_subscriber();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        resp.headers()
            .get("x-parapet-error")
            .and_then(|v| v.to_str().ok()),
        Some(expected_outcome)
    );
    assert_eq!(
        resp.headers()
            .get("x-parapet-blocked-by")
            .and_then(|v| v.to_str().ok()),
        Some("provenance")
    );
    assert!(resp.headers().get(PROVENANCE_HEADER).is_none());
    assert!(!resp
        .headers()
        .keys()
        .any(|k| k.as_str().starts_with("x-guard-trust")));

    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["type"], "parapet_carrier_rejected");
    assert_eq!(json["error"]["code"], expected_outcome);
    assert_eq!(json["error"]["outcome"], expected_outcome);
    assert_eq!(json["error"]["supported_profile"], PROFILE_ID);
    assert_eq!(json["error"]["supported_schema_version"], SCHEMA_VERSION_V1);
    json
}

// ---------------------------------------------------------------------------
// 1. Ordinary mode: absent carrier unchanged; carrier stripped at proxy seam
//    while x-guard-trust* preserved to EngineUpstreamClient and stripped at
//    provider HTTP sender and response boundary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ordinary_mode_absent_and_present_carrier_preserves_behavior_and_strips_headers() {
    let upstream = Arc::new(CapturingUpstream::with_response_header(
        "x-guard-trust-out",
        CANARY_TRUST_HEADER,
    ));
    let app = build_router(upstream.clone());

    let (body, env_bytes, _) = valid_chat_body_and_envelope(1);

    // 1a) Absent carrier with x-guard-trust* headers in request:
    //     proxy_handler preserves x-guard-trust* to the ordinary-mode UpstreamClient
    //     (so EngineUpstreamClient can run parse_trust_header) and strips response
    //     x-guard-trust* headers.
    let req_absent = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .header("X-Guard-Trust-Level", "trusted")
        .body(Body::from(body.clone()))
        .unwrap();

    let resp1 = app.clone().oneshot(req_absent).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);
    assert!(resp1.headers().get("x-guard-trust-out").is_none());
    assert_eq!(upstream.call_count(), 1);
    let fwd1 = upstream.take_last_request().unwrap();
    assert!(fwd1.headers.get(PROVENANCE_HEADER).is_none());
    assert_eq!(
        fwd1.headers.get("x-guard-trust").unwrap(),
        CANARY_TRUST_HEADER
    );
    assert_eq!(fwd1.headers.get("x-guard-trust-level").unwrap(), "trusted");
    assert_eq!(
        fwd1.headers.get("authorization").unwrap(),
        CANARY_AUTH_HEADER
    );
    assert_eq!(fwd1.body.as_ref(), body.as_slice());

    // 1b) Ordinary mode with duplicate/oversize/valid carrier headers still forwards and strips carrier.
    let mut req_with_carrier = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(body.clone()))
        .unwrap();
    req_with_carrier.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_bytes(&env_bytes).unwrap(),
    );
    req_with_carrier.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_static("duplicate-ignored-in-ordinary"),
    );

    let resp2 = app.oneshot(req_with_carrier).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    assert_eq!(upstream.call_count(), 2);
    let fwd2 = upstream.take_last_request().unwrap();
    assert!(fwd2
        .headers
        .get_all(PROVENANCE_HEADER)
        .iter()
        .next()
        .is_none());
    assert_eq!(
        fwd2.headers.get("x-guard-trust").unwrap(),
        CANARY_TRUST_HEADER
    );
}

// ---------------------------------------------------------------------------
// 2. Required mode: runs before L0/trust/L1 and before any upstream attempt
// ---------------------------------------------------------------------------

struct PipelineProbes {
    l0_calls: Arc<AtomicUsize>,
    trust_calls: Arc<AtomicUsize>,
    l1_calls: Arc<AtomicUsize>,
    http_calls: Arc<AtomicUsize>,
}

struct ProbeNormalizer(Arc<AtomicUsize>);
impl Normalizer for ProbeNormalizer {
    fn normalize(&self, input: &str) -> String {
        self.0.fetch_add(1, Ordering::SeqCst);
        input.to_string()
    }
}

struct ProbeTrustAssigner(Arc<AtomicUsize>);
impl TrustAssigner for ProbeTrustAssigner {
    fn assign_trust(&self, _messages: &mut [Message], _config: &Config) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct ProbeL1Model(Arc<AtomicUsize>);
impl L1Model for ProbeL1Model {
    fn score(&self, _text: &str) -> f64 {
        self.0.fetch_add(1, Ordering::SeqCst);
        -5.0
    }
    fn calibration(&self) -> CalibrationParams {
        CalibrationParams { a: 1.0, b: 0.0 }
    }
}

struct ProbeHttpSender(Arc<AtomicUsize>);
#[async_trait::async_trait]
impl HttpSender for ProbeHttpSender {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(HttpError::Transport("must not be called".into()))
    }
}

struct DummyResolver;
impl UpstreamResolver for DummyResolver {
    fn base_url(&self, _provider: Provider) -> String {
        "https://api.openai.com".to_string()
    }
}

fn build_engine_with_probes() -> (Arc<EngineUpstreamClient>, PipelineProbes) {
    ensure_global_trace_subscriber();
    let probes = PipelineProbes {
        l0_calls: Arc::new(AtomicUsize::new(0)),
        trust_calls: Arc::new(AtomicUsize::new(0)),
        l1_calls: Arc::new(AtomicUsize::new(0)),
        http_calls: Arc::new(AtomicUsize::new(0)),
    };
    let yaml = r#"
parapet: v1
use_default_block_patterns: false
use_default_sensitive_patterns: false
layers:
  L0: { mode: sanitize }
  L1: { mode: block, threshold: 0.0 }
  L3_inbound: { mode: block }
"#;
    let config = Arc::new(
        load_config(&StringSource {
            content: yaml.to_string(),
        })
        .unwrap(),
    );
    let deps = EngineDeps {
        config,
        http: Arc::new(ProbeHttpSender(probes.http_calls.clone())),
        resolver: Arc::new(DummyResolver),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(ProbeNormalizer(probes.l0_calls.clone())),
        trust_assigner: Arc::new(ProbeTrustAssigner(probes.trust_calls.clone())),
        l1_scanner: None,
        l1_model: Some(Arc::new(ProbeL1Model(probes.l1_calls.clone()))),
        l2a_scanner: None,
        l2a_semaphore: None,
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store: None,
        multi_turn_scanner: None,
        orthogonal_sensor_router: Arc::new(NoopOrthogonalSensorRouter::new()),
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
        emit_l1_signals: false,
    };
    (Arc::new(EngineUpstreamClient::new_with(deps)), probes)
}

#[tokio::test]
async fn required_missing_carrier_rejects_before_l0_trust_l1_and_zero_forward() {
    let (engine, probes) = build_engine_with_probes();
    let next_stage = Arc::new(CapturingNextStage::default());
    let app = build_router_with_admission(
        engine,
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        Some(next_stage.clone()),
    );

    let (body, _, _) = valid_chat_body_and_envelope(1);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_carrier_rejection(resp, "absent").await;

    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 0);
}

// ---------------------------------------------------------------------------
// 3. Duplicate and oversize carrier headers reject schema_mismatch zero-forward
// ---------------------------------------------------------------------------

#[tokio::test]
async fn required_duplicate_and_oversize_headers_reject_schema_mismatch_zero_forward() {
    let (engine, probes) = build_engine_with_probes();
    let next_stage = Arc::new(CapturingNextStage::default());
    let app = build_router_with_admission(
        engine,
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        Some(next_stage.clone()),
    );

    let (body, env_bytes, _) = valid_chat_body_and_envelope(1);

    // 3a) Duplicate carrier headers (even when both are individually valid).
    let mut dup_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(body.clone()))
        .unwrap();
    dup_req.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_bytes(&env_bytes).unwrap(),
    );
    dup_req.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_bytes(&env_bytes).unwrap(),
    );

    let dup_resp = app.clone().oneshot(dup_req).await.unwrap();
    assert_carrier_rejection(dup_resp, "schema_mismatch").await;

    // 3b) Oversize carrier header (> MAX_ENVELOPE_BYTES).
    let over = "a".repeat(MAX_ENVELOPE_BYTES + 1);
    let over_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(PROVENANCE_HEADER, HeaderValue::from_str(&over).unwrap())
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(body))
        .unwrap();

    let over_resp = app.oneshot(over_req).await.unwrap();
    assert_carrier_rejection(over_resp, "schema_mismatch").await;

    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 0);
}

// ---------------------------------------------------------------------------
// 4. Route/provider binding: only exact POST /v1/chat/completions with OpenAI
//    decoder is admitted; wrong routes/providers reject schema_mismatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn required_wrong_route_or_provider_rejects_schema_mismatch_zero_forward() {
    let (engine, probes) = build_engine_with_probes();
    let next_stage = Arc::new(CapturingNextStage::default());
    let app = build_router_with_admission(
        engine,
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        Some(next_stage.clone()),
    );

    let (body, env_bytes, _) = valid_chat_body_and_envelope(1);

    for wrong_path in [
        "/v1/messages",
        "/openai/v1/chat/completions",
        "/v1/chat/completions/extra",
        "/v1/responses",
        "/v1/unknown",
    ] {
        let req = Request::builder()
            .method(Method::POST)
            .uri(wrong_path)
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_bytes).unwrap(),
            )
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .body(Body::from(body.clone()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_carrier_rejection(resp, "schema_mismatch").await;
    }

    for wrong_method in [Method::GET, Method::PUT, Method::DELETE] {
        for path in ["/v1/chat/completions", "/v1/messages", "/v1/unknown"] {
            let req = Request::builder()
                .method(wrong_method.clone())
                .uri(path)
                .header("content-type", "application/json")
                .header(
                    PROVENANCE_HEADER,
                    HeaderValue::from_bytes(&env_bytes).unwrap(),
                )
                .header("x-guard-trust", CANARY_TRUST_HEADER)
                .body(Body::from(body.clone()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_carrier_rejection(resp, "schema_mismatch").await;
        }
    }

    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 0);

    // Even if a verifier is constructed with the Anthropic decoder and signs an Anthropic body,
    // RequiredOpenCode at /v1/chat/completions enforces the OpenAI decoder and rejects.
    let anthropic_body = serde_json::to_vec(&json!({
        "model": "claude-3-5-sonnet",
        "system": CANARY_VENDOR,
        "messages": [
            {"role": "user", "content": CANARY_USER}
        ]
    }))
    .unwrap();
    let anthropic_map = SourceMap {
        segments: vec![
            Segment {
                message_index: 0,
                start: 0,
                end: CANARY_VENDOR.len(),
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: CANARY_USER.len(),
                source: Source::User,
                sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
            },
        ],
    };
    let mut anthropic_env = CarrierEnvelope {
        schema_version: SCHEMA_VERSION_V1,
        profile: PROFILE_ID.to_string(),
        build: BUILD_ID.to_string(),
        session_id: SESSION_ID.to_string(),
        sequence: 9,
        issued_at_ms: NOW_MS - 1_000,
        expires_at_ms: NOW_MS + 10_000,
        request_body_sha256: Sha256Digest::of(&anthropic_body),
        source_map_sha256: anthropic_map.canonical_digest(),
        source_map: anthropic_map,
        authenticator: Authenticator::from_bytes(vec![0u8; 32]),
    };
    let auth = HmacSha256Authenticator::new(SessionKey::from_bytes(CANARY_KEY_BYTES));
    let tag = auth.tag(&anthropic_env.signed_payload());
    anthropic_env.authenticator = Authenticator::from_bytes(tag.to_vec());
    let anthropic_env_bytes = serde_json::to_vec(&anthropic_env).unwrap();

    let (engine2, probes2) = build_engine_with_probes();
    let next_stage2 = Arc::new(CapturingNextStage::default());
    let app_wrong_decoder = build_router_with_admission(
        engine2,
        Arc::new(CarrierRequestAdmitter::required_opencode(
            build_verifier_with_decoder(Box::new(ProviderContentDecoder::new(
                ProviderType::Anthropic,
            ))),
        )),
        Some(next_stage2.clone()),
    );
    let req_anthropic_on_openai_route = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&anthropic_env_bytes).unwrap(),
        )
        .body(Body::from(anthropic_body))
        .unwrap();
    let resp_wrong_decoder = app_wrong_decoder
        .oneshot(req_anthropic_on_openai_route)
        .await
        .unwrap();
    assert_carrier_rejection(resp_wrong_decoder, "map_out_of_range").await;
    assert_eq!(probes2.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage2.call_count(), 0);
}

// ---------------------------------------------------------------------------
// 5. Valid required carrier reaches only the injected next-stage seam and
//    forwards sanitized headers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn valid_required_carrier_reaches_only_injected_next_stage_with_sanitized_headers() {
    let (engine, probes) = build_engine_with_probes();
    let next_stage = Arc::new(CapturingNextStage::default());
    let app = build_router_with_admission(
        engine,
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        Some(next_stage.clone()),
    );

    let (body, env_bytes, _) = valid_chat_body_and_envelope(1);

    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .header("X-Guard-Trust-Span", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes).unwrap(),
        )
        .body(Body::from(body.clone()))
        .unwrap();
    let _ = &mut req;

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Response x-guard-trust* header from next_stage is also stripped by proxy_handler.
    assert!(resp.headers().get("x-guard-trust-response").is_none());

    // Whole-message engine pipeline and upstream were NOT called; only next_stage was called.
    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 1);

    let captured = next_stage.take_captured();
    let (provider, proxy_req, verified) = &captured[0];
    assert_eq!(*provider, Provider::OpenAi);
    assert!(proxy_req
        .headers
        .get_all(PROVENANCE_HEADER)
        .iter()
        .next()
        .is_none());
    assert!(!proxy_req
        .headers
        .keys()
        .any(|k| k.as_str().starts_with("x-guard-trust")));
    assert_eq!(
        proxy_req.headers.get("authorization").unwrap(),
        CANARY_AUTH_HEADER
    );
    assert_eq!(proxy_req.body.as_ref(), body.as_slice());

    assert_eq!(verified.identity.profile, PROFILE_ID);
    assert_eq!(verified.identity.build, BUILD_ID);
    assert_eq!(verified.identity.schema_version, SCHEMA_VERSION_V1);
    assert_eq!(verified.request_body_sha256, Sha256Digest::of(&body));
    assert_eq!(verified.segments.len(), 3);
    assert_eq!(verified.messages.len(), 2);
}

// ---------------------------------------------------------------------------
// 6. Captured tracing/log assertions (F18.4): no raw envelope, authenticator,
//    key, segment text, body, or local path appears on covered failures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn covered_failures_emit_no_envelope_key_authenticator_segment_body_or_path_in_logs_or_body()
{
    let trace_capture = ThreadTraceGuard::new();

    let upstream = Arc::new(CapturingUpstream::new());
    let next_stage = Arc::new(CapturingNextStage::default());
    let app = build_router_with_admission(
        upstream.clone(),
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        Some(next_stage.clone()),
    );

    let (body, env_bytes, tag_hex) = valid_chat_body_and_envelope(42);
    let raw_env_str = String::from_utf8(env_bytes.clone()).unwrap();

    // Case A: Missing carrier
    let req_missing = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(body.clone()))
        .unwrap();
    let resp_missing = app.clone().oneshot(req_missing).await.unwrap();
    let json_missing = assert_carrier_rejection(resp_missing, "absent").await;

    // Case B: Duplicate carrier headers with canary claim
    let mut req_dup = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.clone()))
        .unwrap();
    req_dup.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_bytes(&env_bytes).unwrap(),
    );
    req_dup.headers_mut().append(
        PROVENANCE_HEADER,
        HeaderValue::from_static(CANARY_RAW_ENV_CLAIM),
    );
    let resp_dup = app.clone().oneshot(req_dup).await.unwrap();
    let json_dup = assert_carrier_rejection(resp_dup, "schema_mismatch").await;

    // Case C: Oversize carrier header containing canary claim
    let oversize_val = format!(
        "{CANARY_RAW_ENV_CLAIM}{}",
        "x".repeat(MAX_ENVELOPE_BYTES + 16)
    );
    let req_over = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_str(&oversize_val).unwrap(),
        )
        .body(Body::from(body.clone()))
        .unwrap();
    let resp_over = app.clone().oneshot(req_over).await.unwrap();
    let json_over = assert_carrier_rejection(resp_over, "schema_mismatch").await;

    // Case D: Wrong route (/v1/messages) with valid carrier
    let req_route = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes).unwrap(),
        )
        .body(Body::from(body.clone()))
        .unwrap();
    let resp_route = app.clone().oneshot(req_route).await.unwrap();
    let json_route = assert_carrier_rejection(resp_route, "schema_mismatch").await;

    // Case E: Invalid signature (tampered authenticator / wrong key)
    let mut bad_sig_val: Value = serde_json::from_slice(&env_bytes).unwrap();
    bad_sig_val["sequence"] = json!(99);
    let bad_sig_bytes = serde_json::to_vec(&bad_sig_val).unwrap();
    let req_sig = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&bad_sig_bytes).unwrap(),
        )
        .body(Body::from(body.clone()))
        .unwrap();
    let resp_sig = app.clone().oneshot(req_sig).await.unwrap();
    let json_sig = assert_carrier_rejection(resp_sig, "signature_invalid").await;

    // Case F: Body digest mismatch (tampered body containing local path & canary)
    // Note: body digest check runs after freshness/replay ledger consumption, so use sequence 42 here
    // and sequence 43 for Case G.
    let mut tampered_body = body.clone();
    tampered_body.push(b' ');
    let req_body_mismatch = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes).unwrap(),
        )
        .body(Body::from(tampered_body))
        .unwrap();
    let resp_body_mismatch = app.clone().oneshot(req_body_mismatch).await.unwrap();
    let json_body_mismatch =
        assert_carrier_rejection(resp_body_mismatch, "body_digest_mismatch").await;

    // Case G: Valid admission then replay rejection (sequence 43)
    let (body_g, env_bytes_g, tag_hex_g) = valid_chat_body_and_envelope(43);
    let req_valid = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes_g).unwrap(),
        )
        .body(Body::from(body_g.clone()))
        .unwrap();
    let resp_valid = app.clone().oneshot(req_valid).await.unwrap();
    assert_eq!(resp_valid.status(), StatusCode::OK);

    let req_replay = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes_g).unwrap(),
        )
        .body(Body::from(body_g))
        .unwrap();
    let resp_replay = app.oneshot(req_replay).await.unwrap();
    let json_replay = assert_carrier_rejection(resp_replay, "stale_or_replayed").await;

    assert_eq!(upstream.call_count(), 0);
    assert_eq!(next_stage.call_count(), 1);

    let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
    for expected_outcome in [
        "absent",
        "schema_mismatch",
        "signature_invalid",
        "body_digest_mismatch",
        "stale_or_replayed",
    ] {
        assert!(
            logs.contains(expected_outcome),
            "logs should record closed outcome {expected_outcome}, got: {logs}"
        );
    }

    let body_str = String::from_utf8(body).unwrap();
    let repo_digest = Sha256Digest::of(CANARY_REPO.as_bytes()).to_hex();
    let user_digest = Sha256Digest::of(CANARY_USER.as_bytes()).to_hex();
    let forbidden = [
        raw_env_str.as_str(),
        &tag_hex,
        &tag_hex_g,
        CANARY_KEY_STR,
        CANARY_VENDOR.trim(),
        CANARY_REPO.trim(),
        "CANARY-REPO-SEG-8b4e",
        CANARY_USER,
        "CANARY-BODY-TEXT-3c9a",
        CANARY_LOCAL_PATH,
        "/home/user",
        CANARY_RAW_ENV_CLAIM,
        CANARY_TRUST_HEADER,
        CANARY_AUTH_HEADER,
        body_str.as_str(),
        &repo_digest,
        &user_digest,
    ];

    let mut surfaces = vec![logs];
    for j in [
        json_missing,
        json_dup,
        json_over,
        json_route,
        json_sig,
        json_body_mismatch,
        json_replay,
    ] {
        surfaces.push(serde_json::to_string(&j).unwrap());
    }

    for surface in &surfaces {
        for needle in forbidden {
            assert!(
                !surface.contains(needle),
                "F18.4 leak of {needle:?} in surface: {surface}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Full-handler SessionRequestAdmitter coverage (no active session, active
//    session + missing carrier, active session + valid freshly signed carrier,
//    and replay rejection) plus side-effect-free fail() metadata verification
// ---------------------------------------------------------------------------

struct DeterministicEntropy {
    calls: AtomicUsize,
}

impl EntropySource for DeterministicEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((call as u8).wrapping_mul(37)).wrapping_add(i as u8 + 1);
        }
        Ok(())
    }
}

fn valid_chat_body_and_envelope_for_session(
    session_id: &str,
    key: &SessionKey,
    sequence: u64,
) -> (Vec<u8>, Vec<u8>) {
    let sys_content = format!("{CANARY_VENDOR}{CANARY_REPO}");
    let body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": CANARY_USER}
        ]
    }))
    .unwrap();

    let v_len = CANARY_VENDOR.len();
    let s_len = sys_content.len();
    let u_len = CANARY_USER.len();
    let source_map = SourceMap {
        segments: vec![
            Segment {
                message_index: 0,
                start: 0,
                end: v_len,
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
            },
            Segment {
                message_index: 0,
                start: v_len,
                end: s_len,
                source: Source::Repository,
                sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: u_len,
                source: Source::User,
                sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
            },
        ],
    };

    let mut env = CarrierEnvelope {
        schema_version: SCHEMA_VERSION_V1,
        profile: PROFILE_ID.to_string(),
        build: BUILD_ID.to_string(),
        session_id: session_id.to_string(),
        sequence,
        issued_at_ms: NOW_MS - 1_000,
        expires_at_ms: NOW_MS + 10_000,
        request_body_sha256: Sha256Digest::of(&body),
        source_map_sha256: source_map.canonical_digest(),
        source_map,
        authenticator: Authenticator::from_bytes(vec![0u8; 32]),
    };
    let auth = HmacSha256Authenticator::new(key.clone());
    let tag = auth.tag(&env.signed_payload());
    env.authenticator = Authenticator::from_bytes(tag.to_vec());
    let env_bytes = serde_json::to_vec(&env).unwrap();
    (body, env_bytes)
}

#[tokio::test]
async fn session_request_admitter_full_handler_lifecycle_and_replay() {
    let sessions = Arc::new(SessionManager::with_entropy(Box::new(
        DeterministicEntropy {
            calls: AtomicUsize::new(0),
        },
    )));
    assert!(!sessions.is_active());

    let (engine, probes) = build_engine_with_probes();
    let next_stage = Arc::new(CapturingNextStage::default());
    let route = ProvenanceRouteBinding::OPENAI;
    assert_eq!(route.provider(), Provider::OpenAi);
    assert_eq!(route.canonical_post_path(), "/v1/chat/completions");
    assert_eq!(route.provider_type(), ProviderType::OpenAi);
    let admitter = Arc::new(SessionRequestAdmitter::new(
        ProfileMode::RequiredOpenCode,
        route,
        sessions.clone(),
        PROFILE_ID,
        SCHEMA_VERSION_V1,
    ));
    assert_eq!(admitter.bound_route(), route);
    assert_eq!(admitter.bound_provider(), Provider::OpenAi);
    assert_eq!(admitter.canonical_post_path(), "/v1/chat/completions");
    let app = build_router_with_admission(engine, admitter, Some(next_stage.clone()));

    let (fallback_body, fallback_env_bytes, _) = valid_chat_body_and_envelope(1);

    // 7a) No active session in RequiredOpenCode rejects session_mismatch with zero
    //     whole-message/L0/trust/L1/upstream/next-stage calls.
    let req_no_session = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&fallback_env_bytes).unwrap(),
        )
        .body(Body::from(fallback_body.clone()))
        .unwrap();
    let resp_no_session = app.clone().oneshot(req_no_session).await.unwrap();
    assert_carrier_rejection(resp_no_session, "session_mismatch").await;
    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 0);

    // Start an active session with deterministic injected entropy and clock.
    let mut builds = BTreeMap::new();
    builds.insert(
        BUILD_ID.to_string(),
        VendorManifest::new([Sha256Digest::of(CANARY_VENDOR.as_bytes())]),
    );
    let grant = sessions
        .start(route.session_config(
            SupportedProfile {
                schema_version: SCHEMA_VERSION_V1,
                profile_id: PROFILE_ID.to_string(),
                builds,
                max_validity_ms: MAX_VALIDITY_MS,
            },
            Box::new(FixedClock(NOW_MS)),
        ))
        .unwrap();
    assert!(sessions.is_active());

    // 7b) Active session + missing carrier rejects absent with zero calls.
    let req_missing = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .body(Body::from(fallback_body))
        .unwrap();
    let resp_missing = app.clone().oneshot(req_missing).await.unwrap();
    assert_carrier_rejection(resp_missing, "absent").await;
    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 0);

    // 7c) Active session + valid freshly signed carrier reaches VerifiedCarrierNextStage
    //     exactly once with sanitized headers.
    let (body, env_bytes) =
        valid_chat_body_and_envelope_for_session(&grant.session_id, &grant.key, 1);
    let req_valid = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .header("X-Guard-Trust-Span", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes).unwrap(),
        )
        .body(Body::from(body.clone()))
        .unwrap();
    let resp_valid = app.clone().oneshot(req_valid).await.unwrap();
    assert_eq!(resp_valid.status(), StatusCode::OK);
    assert!(resp_valid.headers().get("x-guard-trust-response").is_none());
    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 1);

    let captured = next_stage.take_captured();
    assert_eq!(captured.len(), 1);
    let (provider, proxy_req, verified) = &captured[0];
    assert_eq!(*provider, Provider::OpenAi);
    assert!(proxy_req
        .headers
        .get_all(PROVENANCE_HEADER)
        .iter()
        .next()
        .is_none());
    assert!(!proxy_req
        .headers
        .keys()
        .any(|k| k.as_str().starts_with("x-guard-trust")));
    assert_eq!(
        proxy_req.headers.get("authorization").unwrap(),
        CANARY_AUTH_HEADER
    );
    assert_eq!(proxy_req.body.as_ref(), body.as_slice());
    assert_eq!(verified.identity.profile, PROFILE_ID);
    assert_eq!(verified.identity.build, BUILD_ID);
    assert_eq!(verified.identity.schema_version, SCHEMA_VERSION_V1);

    // 7d) Replay of that carrier rejects stale_or_replayed with zero additional
    //     next-stage/upstream calls.
    let req_replay = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&env_bytes).unwrap(),
        )
        .body(Body::from(body))
        .unwrap();
    let resp_replay = app.oneshot(req_replay).await.unwrap();
    assert_carrier_rejection(resp_replay, "stale_or_replayed").await;
    assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
    assert_eq!(next_stage.call_count(), 1);
    assert!(next_stage.take_captured().is_empty());

    // 7e) Verify CarrierRequestAdmitter::new and fail() synthesize public
    //     metadata with zero clock, authenticator, or replay-ledger calls.
    struct CountingClock(Arc<AtomicUsize>);
    impl Clock for CountingClock {
        fn now_ms(&self) -> u64 {
            self.0.fetch_add(1, Ordering::SeqCst);
            NOW_MS
        }
    }
    struct CountingLedger(Arc<AtomicUsize>);
    impl ReplayLedger for CountingLedger {
        fn consume(&self, _: &str, _: u64, _: u64, _: u64) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst);
            true
        }
    }
    let clock_calls = Arc::new(AtomicUsize::new(0));
    let ledger_calls = Arc::new(AtomicUsize::new(0));
    let mut verifier_builds = BTreeMap::new();
    verifier_builds.insert(
        BUILD_ID.to_string(),
        VendorManifest::new([Sha256Digest::of(CANARY_VENDOR.as_bytes())]),
    );
    let probed_verifier = CarrierVerifier::new(CarrierVerifierDeps {
        profile: SupportedProfile {
            schema_version: SCHEMA_VERSION_V1,
            profile_id: PROFILE_ID.to_string(),
            builds: verifier_builds,
            max_validity_ms: MAX_VALIDITY_MS,
        },
        session_id: SESSION_ID.to_string(),
        authenticator: Box::new(HmacSha256Authenticator::new(SessionKey::from_bytes(
            CANARY_KEY_BYTES,
        ))),
        clock: Box::new(CountingClock(clock_calls.clone())),
        ledger: Box::new(CountingLedger(ledger_calls.clone())),
        decoder: Box::new(ProviderContentDecoder::new(ProviderType::OpenAi)),
    });
    let c_admitter = CarrierRequestAdmitter::new(ProfileMode::RequiredOpenCode, probed_verifier);
    let err = c_admitter.fail(VerificationOutcome::SchemaMismatch);
    assert_eq!(err.outcome, VerificationOutcome::SchemaMismatch);
    assert_eq!(err.supported_profile, PROFILE_ID);
    assert_eq!(err.supported_schema_version, SCHEMA_VERSION_V1);
    assert_eq!(clock_calls.load(Ordering::SeqCst), 0);
    assert_eq!(ledger_calls.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------
// 8. R-W3a-1 regression: ordinary mode full-handler preserves x-guard-trust
//    to EngineUpstreamClient so untrusted span > max_length returns 403 with
//    zero provider HTTP sends, while permitted ordinary request strips
//    carrier and x-guard-trust* at the provider HTTP sender
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CapturingHttpSender {
    calls: AtomicUsize,
    requests: Mutex<Vec<HttpRequest>>,
}

#[async_trait::async_trait]
impl HttpSender for CapturingHttpSender {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(request);
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert(
            "x-guard-trust-upstream",
            HeaderValue::from_static(CANARY_TRUST_HEADER),
        );
        Ok(HttpResponse {
            status: StatusCode::OK,
            headers,
            body: parapet::engine::HttpBody::Full(Bytes::from_static(
                br#"{"choices":[{"message":{"content":"ok","tool_calls":[]}}]}"#,
            )),
        })
    }
}

#[tokio::test]
async fn ordinary_mode_engine_trust_parsing_blocks_oversize_untrusted_span_and_sanitizes_provider_wire(
) {
    use base64::Engine as _;
    use parapet::normalize::L0Normalizer;
    use parapet::trust::RoleTrustAssigner;

    ensure_global_trace_subscriber();
    let yaml = r#"
parapet: v1
use_default_block_patterns: false
use_default_sensitive_patterns: false
untrusted_content_policy:
  max_length: 30
trust:
  unknown_trust_policy:
    system: trusted
    user: trusted
    assistant: trusted
    tool: untrusted
layers:
  L0: { mode: sanitize }
  L3_inbound: { mode: block }
"#;
    let config = Arc::new(
        load_config(&StringSource {
            content: yaml.to_string(),
        })
        .unwrap(),
    );
    let sender = Arc::new(CapturingHttpSender::default());
    let deps = EngineDeps {
        config,
        http: sender.clone(),
        resolver: Arc::new(DummyResolver),
        registry: Arc::new(DefaultProviderRegistry),
        normalizer: Arc::new(L0Normalizer::new()),
        trust_assigner: Arc::new(RoleTrustAssigner),
        l1_scanner: None,
        l1_model: None,
        l2a_scanner: None,
        l2a_semaphore: None,
        inbound_scanner: Arc::new(DefaultInboundScanner::new()),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store: None,
        multi_turn_scanner: None,
        orthogonal_sensor_router: Arc::new(NoopOrthogonalSensorRouter::new()),
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
        emit_l1_signals: false,
    };
    let engine = Arc::new(EngineUpstreamClient::new_with(deps));
    let app = build_router(engine);

    // 8a) Ordinary-mode request with an untrusted span > max_length (30) returns 403
    //     and zero provider HTTP sends.
    let long_rag = "This is RAG content from an external database that exceeds thirty characters";
    let blocked_body = format!(
        r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"Safe preamble. {long_rag}"}}]}}"#,
    );
    let offset = blocked_body.find(long_rag).unwrap();
    let end = offset + long_rag.len();
    let spans_json = json!([{"s": offset, "e": end, "src": "rag"}]);
    let encoded =
        base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&spans_json).unwrap());
    let trust_hdr = format!("inline:{encoded}");

    let req_blocked = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", &trust_hdr)
        .header("x-guard-trust-extra", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_static("unverified-ordinary-carrier"),
        )
        .body(Body::from(blocked_body))
        .unwrap();

    let resp_blocked = app.clone().oneshot(req_blocked).await.unwrap();
    assert_eq!(resp_blocked.status(), StatusCode::FORBIDDEN);
    assert_eq!(sender.calls.load(Ordering::SeqCst), 0);

    // 8b) Permitted ordinary request with short untrusted span (<= 30 chars),
    //     x-guard-trust* headers, and a carrier header succeeds and strips all
    //     carrier and x-guard-trust* headers before the provider HTTP sender.
    let short_rag = "short rag span";
    let allowed_body = format!(
        r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"Safe preamble. {short_rag}"}}]}}"#,
    );
    let short_offset = allowed_body.find(short_rag).unwrap();
    let short_end = short_offset + short_rag.len();
    let short_spans = json!([{"s": short_offset, "e": short_end, "src": "rag"}]);
    let short_encoded =
        base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&short_spans).unwrap());
    let short_trust_hdr = format!("inline:{short_encoded}");

    let req_allowed = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", CANARY_AUTH_HEADER)
        .header("x-guard-trust", &short_trust_hdr)
        .header("X-Guard-Trust-Level", CANARY_TRUST_HEADER)
        .header(
            PROVENANCE_HEADER,
            HeaderValue::from_static("unverified-ordinary-carrier"),
        )
        .body(Body::from(allowed_body.clone()))
        .unwrap();

    let resp_allowed = app.oneshot(req_allowed).await.unwrap();
    assert_eq!(resp_allowed.status(), StatusCode::OK);
    assert!(resp_allowed
        .headers()
        .get("x-guard-trust-upstream")
        .is_none());
    assert_eq!(sender.calls.load(Ordering::SeqCst), 1);

    let sent = sender.requests.lock().unwrap();
    assert_eq!(sent.len(), 1);
    let wire_req = &sent[0];
    assert!(wire_req
        .headers
        .get_all(PROVENANCE_HEADER)
        .iter()
        .next()
        .is_none());
    assert!(!wire_req
        .headers
        .keys()
        .any(|k| k.as_str().starts_with("x-guard-trust")));
    assert_eq!(
        wire_req.headers.get("authorization").unwrap(),
        CANARY_AUTH_HEADER
    );
}

// ---------------------------------------------------------------------------
// 9. R-W3a-3 constructor invariant: RequiredOpenCode + None next_stage is
//    impossible at construction, while Ordinary + None remains valid
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "RequiredOpenCode admission policy requires a VerifiedCarrierNextStage")]
fn required_opencode_without_next_stage_panics_at_router_construction() {
    let upstream = Arc::new(CapturingUpstream::new());
    let _ = build_router_with_admission(
        upstream,
        Arc::new(CarrierRequestAdmitter::required_opencode(build_verifier())),
        None,
    );
}

#[test]
fn ordinary_mode_with_none_next_stage_is_valid_at_construction() {
    let upstream = Arc::new(CapturingUpstream::new());
    let _ = build_router_with_admission(
        upstream,
        Arc::new(CarrierRequestAdmitter::new(
            ProfileMode::Ordinary,
            build_verifier(),
        )),
        None,
    );
}

// ---------------------------------------------------------------------------
// 10. W4b-core: owner-only session grant control wire (`parapet-control/v0`)
// ---------------------------------------------------------------------------

#[cfg(not(windows))]
mod w4b_control_wire {
    use super::*;
    use parapet::cli::control::{
        peer_credential_authorized, send_control_request, write_private_file, ControlError,
        ControlRequestV0, ControlStateMetadataV0, DefaultProvenanceSessionClockFactory,
        EngineControlPlane, EngineOwnerRecordV0, ProvenanceSessionClockFactory,
        ProvenanceSessionControl, CONTROL_ACTION_PROVENANCE_SESSION_END,
        CONTROL_ACTION_PROVENANCE_SESSION_START, CONTROL_ERROR_SESSION_UNAVAILABLE, CONTROL_SCHEMA,
        CONTROL_STATE_SCHEMA,
    };
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::path::Path;

    const CANARY_W4B_KEY_1_BYTES: [u8; SESSION_KEY_LEN] = *b"CANARY-W4B-KEY-SESSION-ONE-00001";
    const CANARY_W4B_KEY_1_STR: &str = "CANARY-W4B-KEY-SESSION-ONE-00001";
    const CANARY_W4B_KEY_2_BYTES: [u8; SESSION_KEY_LEN] = *b"CANARY-W4B-KEY-SESSION-TWO-00002";
    const CANARY_W4B_KEY_2_STR: &str = "CANARY-W4B-KEY-SESSION-TWO-00002";
    const CONFIG_HASH: &str =
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct ScriptedSessionEntropy {
        call: AtomicUsize,
    }

    impl ScriptedSessionEntropy {
        fn new() -> Self {
            Self {
                call: AtomicUsize::new(0),
            }
        }
    }

    impl EntropySource for ScriptedSessionEntropy {
        fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
            let idx = self.call.fetch_add(1, Ordering::SeqCst);
            match idx {
                // Session 1: key (32 bytes), id (16 bytes)
                0 => buf.copy_from_slice(&CANARY_W4B_KEY_1_BYTES),
                1 => buf.fill(0xa1),
                // Session 2: key (32 bytes), id (16 bytes)
                2 => buf.copy_from_slice(&CANARY_W4B_KEY_2_BYTES),
                3 => buf.fill(0xb2),
                other => {
                    for (i, b) in buf.iter_mut().enumerate() {
                        *b = (other as u8).wrapping_mul(17).wrapping_add(i as u8 + 1);
                    }
                }
            }
            Ok(())
        }
    }

    struct FailingSessionEntropy;

    impl EntropySource for FailingSessionEntropy {
        fn fill(&self, _buf: &mut [u8]) -> Result<(), EntropyError> {
            Err(EntropyError)
        }
    }

    fn init_private_state_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let meta = ControlStateMetadataV0 {
            schema: CONTROL_STATE_SCHEMA.to_owned(),
            effect_root_must_be_new: true,
        };
        write_private_file(dir, "control.json", &serde_json::to_vec(&meta).unwrap()).unwrap();
    }

    fn supported_profile(max_validity_ms: u64) -> SupportedProfile {
        let mut builds = BTreeMap::new();
        builds.insert(
            BUILD_ID.to_string(),
            VendorManifest::new([Sha256Digest::of(CANARY_VENDOR.as_bytes())]),
        );
        SupportedProfile {
            schema_version: SCHEMA_VERSION_V1,
            profile_id: PROFILE_ID.to_string(),
            builds,
            max_validity_ms,
        }
    }

    fn decode_key_hex(hex: &str) -> SessionKey {
        assert_eq!(hex.len(), SESSION_KEY_LEN * 2);
        assert!(hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        let mut bytes = [0u8; SESSION_KEY_LEN];
        for (i, slot) in bytes.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        SessionKey::from_bytes(bytes)
    }

    fn sign_chat_for_session_at(
        session_id: &str,
        key: &SessionKey,
        sequence: u64,
        issued_at_ms: u64,
        expires_at_ms: u64,
    ) -> (Vec<u8>, Vec<u8>, String) {
        let sys_content = format!("{CANARY_VENDOR}{CANARY_REPO}");
        let body = serde_json::to_vec(&json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": CANARY_USER}
            ]
        }))
        .unwrap();

        let v_len = CANARY_VENDOR.len();
        let s_len = sys_content.len();
        let u_len = CANARY_USER.len();
        let source_map = SourceMap {
            segments: vec![
                Segment {
                    message_index: 0,
                    start: 0,
                    end: v_len,
                    source: Source::VendorStatic,
                    sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
                },
                Segment {
                    message_index: 0,
                    start: v_len,
                    end: s_len,
                    source: Source::Repository,
                    sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
                },
                Segment {
                    message_index: 1,
                    start: 0,
                    end: u_len,
                    source: Source::User,
                    sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
                },
            ],
        };

        let mut env = CarrierEnvelope {
            schema_version: SCHEMA_VERSION_V1,
            profile: PROFILE_ID.to_string(),
            build: BUILD_ID.to_string(),
            session_id: session_id.to_string(),
            sequence,
            issued_at_ms,
            expires_at_ms,
            request_body_sha256: Sha256Digest::of(&body),
            source_map_sha256: source_map.canonical_digest(),
            source_map,
            authenticator: Authenticator::from_bytes(vec![0u8; 32]),
        };
        let auth = HmacSha256Authenticator::new(key.clone());
        let tag = auth.tag(&env.signed_payload());
        let tag_hex = Sha256Digest::from_bytes(tag).to_hex();
        env.authenticator = Authenticator::from_bytes(tag.to_vec());
        let env_bytes = serde_json::to_vec(&env).unwrap();
        (body, env_bytes, tag_hex)
    }

    fn send_control_line_in_process(plane: &EngineControlPlane, line: &str) -> (Value, bool) {
        let (bytes, stop) = plane.handle_request_line(line);
        let text = std::str::from_utf8(&bytes).expect("utf8 control response");
        assert!(text.ends_with('\n'), "control response must end with \\n");
        assert_eq!(text.matches('\n').count(), 1, "single-line JSON response");
        let val: Value = serde_json::from_str(text.trim()).expect("valid JSON control response");
        (val, stop)
    }

    #[tokio::test]
    async fn w4b_no_handle_and_failed_start_return_session_unavailable_and_preserve_ordinary_actions(
    ) {
        let temp = tempfile::tempdir().unwrap();
        let state_no_handle = temp.path().join("state_no_handle");
        init_private_state_dir(&state_no_handle);

        let (plane, _listener, _shutdown_rx) =
            EngineControlPlane::bind(&state_no_handle, 45201, CONFIG_HASH, None).unwrap();
        assert!(plane.provenance_session().is_none());

        let token = std::fs::read_to_string(state_no_handle.join("control.token")).unwrap();
        let pid = std::process::id();

        // Ordinary actions preserve exact wire bytes and behavior.
        let (status_bytes, status_stop) = plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{token}","action":"status"}}"#
        ));
        assert!(!status_stop);
        assert_eq!(
            std::str::from_utf8(&status_bytes).unwrap(),
            format!(
                "{{\"schema\":\"parapet-control/v0\",\"ok\":true,\"pid\":{pid},\"file_capability\":\"not_applicable\",\"file_cleanup\":\"not_applicable\",\"evidence_retained\":true}}\n"
            )
        );

        let (restore_bytes, restore_stop) = plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{token}","action":"restore"}}"#
        ));
        assert!(!restore_stop);
        assert_eq!(
            std::str::from_utf8(&restore_bytes).unwrap(),
            "{\"schema\":\"parapet-control/v0\",\"ok\":true,\"file_cleanup\":\"not_applicable\",\"evidence_retained\":true}\n"
        );

        // No handle configured => both actions return stable content-free session_unavailable.
        let expected_unavailable = format!(
            "{{\"schema\":\"{CONTROL_SCHEMA}\",\"ok\":false,\"error_code\":\"{CONTROL_ERROR_SESSION_UNAVAILABLE}\"}}\n"
        );
        for action in [
            CONTROL_ACTION_PROVENANCE_SESSION_START,
            CONTROL_ACTION_PROVENANCE_SESSION_END,
        ] {
            let (bytes, stop) = plane.handle_request_line(&format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{action}"}}"#
            ));
            assert!(!stop);
            assert_eq!(std::str::from_utf8(&bytes).unwrap(), expected_unavailable);
        }

        let (disable_bytes, disable_stop) = plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{token}","action":"disable"}}"#
        ));
        assert!(disable_stop);
        assert_eq!(
            std::str::from_utf8(&disable_bytes).unwrap(),
            "{\"schema\":\"parapet-control/v0\",\"ok\":true,\"stop_approved\":true,\"file_cleanup\":\"not_applicable\",\"evidence_retained\":true}\n"
        );

        // Configured handle with failing entropy or invalid max_validity_ms also fails closed
        // with session_unavailable and leaves no active session.
        let state_fail = temp.path().join("state_fail");
        init_private_state_dir(&state_fail);
        let failing_sessions = Arc::new(SessionManager::with_entropy(Box::new(
            FailingSessionEntropy,
        )));
        let fail_handle = ProvenanceSessionControl::new(
            ProvenanceRouteBinding::OPENAI,
            failing_sessions.clone(),
            supported_profile(60_000),
        );
        let (fail_plane, _fail_listener, _fail_rx) =
            EngineControlPlane::bind_with_provenance_session(
                &state_fail,
                45202,
                CONFIG_HASH,
                None,
                Some(fail_handle),
            )
            .unwrap();
        let fail_token = std::fs::read_to_string(state_fail.join("control.token")).unwrap();
        let (fail_bytes, fail_stop) = fail_plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{fail_token}","action":"provenance_session_start"}}"#
        ));
        assert!(!fail_stop);
        assert_eq!(
            std::str::from_utf8(&fail_bytes).unwrap(),
            expected_unavailable
        );
        assert!(!failing_sessions.is_active());

        let state_zero = temp.path().join("state_zero_validity");
        init_private_state_dir(&state_zero);
        let zero_sessions = Arc::new(SessionManager::new());
        let zero_handle = ProvenanceSessionControl::new(
            ProvenanceRouteBinding::OPENAI,
            zero_sessions.clone(),
            supported_profile(0),
        );
        let (zero_plane, _zero_listener, _zero_rx) =
            EngineControlPlane::bind_with_provenance_session(
                &state_zero,
                45203,
                CONFIG_HASH,
                None,
                Some(zero_handle),
            )
            .unwrap();
        let zero_token = std::fs::read_to_string(state_zero.join("control.token")).unwrap();
        let (zero_bytes, zero_stop) = zero_plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{zero_token}","action":"provenance_session_start"}}"#
        ));
        assert!(!zero_stop);
        assert_eq!(
            std::str::from_utf8(&zero_bytes).unwrap(),
            expected_unavailable
        );
        assert!(!zero_sessions.is_active());
    }

    #[tokio::test]
    async fn w4b_session_grant_lifecycle_rotation_end_restart_and_no_secret_leak() {
        let trace_capture = ThreadTraceGuard::new();
        let temp = tempfile::tempdir().unwrap();
        let state_dir = temp.path().join("state_w4b");
        init_private_state_dir(&state_dir);

        let sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));
        assert!(!sessions.is_active());

        let profile = supported_profile(MAX_VALIDITY_MS);
        let session_handle = ProvenanceSessionControl::with_clock_factory(
            ProvenanceRouteBinding::OPENAI,
            sessions.clone(),
            profile.clone(),
            || -> Box<dyn Clock> { Box::new(FixedClock(NOW_MS)) },
        );

        let (plane, _listener, _shutdown_rx) = EngineControlPlane::bind_with_provenance_session(
            &state_dir,
            45210,
            CONFIG_HASH,
            None,
            Some(session_handle.clone()),
        )
        .unwrap();

        // Verify control.sock and control.token permissions are 0600 and owner-only.
        let sock_meta = std::fs::symlink_metadata(state_dir.join("control.sock")).unwrap();
        let token_meta = std::fs::symlink_metadata(state_dir.join("control.token")).unwrap();
        assert_eq!(sock_meta.permissions().mode() & 0o7777, 0o600);
        assert_eq!(token_meta.permissions().mode() & 0o7777, 0o600);

        let (engine, probes) = build_engine_with_probes();
        let next_stage = Arc::new(CapturingNextStage::default());
        let admitter = Arc::new(session_handle.request_admitter(ProfileMode::RequiredOpenCode));
        let app = build_router_with_admission(engine, admitter, Some(next_stage.clone()));

        let token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        // 1. Wrong control token cannot start a session.
        let (bad_start_bytes, bad_start_stop) = plane.handle_request_line(
            "{\"schema\":\"parapet-control/v0\",\"token\":\"wrong-token\",\"action\":\"provenance_session_start\"}\n",
        );
        assert!(!bad_start_stop);
        let bad_start_raw = String::from_utf8(bad_start_bytes).unwrap();
        assert_eq!(
            bad_start_raw,
            "{\"schema\":\"parapet-control/v0\",\"ok\":false,\"error_code\":\"unauthorized\"}\n"
        );
        assert!(!sessions.is_active());

        // 2. Valid start yields exact shape/length constraints and activates session.
        let (start1_bytes, start1_stop) = plane.handle_request_line(&format!(
            r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
        ));
        assert!(!start1_stop);
        assert!(sessions.is_active());
        let start1_raw = std::str::from_utf8(&start1_bytes).unwrap();
        let start1: Value = serde_json::from_str(start1_raw.trim()).unwrap();

        let start1_obj = start1.as_object().unwrap();
        let start1_keys: Vec<&str> = start1_obj.keys().map(String::as_str).collect();
        assert_eq!(
            start1_keys,
            [
                "builds",
                "max_validity_ms",
                "ok",
                "profile",
                "schema",
                "schema_version",
                "session_id",
                "session_key_hex",
            ]
        );
        assert_eq!(start1["schema"], CONTROL_SCHEMA);
        assert_eq!(start1["ok"], true);
        assert_eq!(start1["profile"], PROFILE_ID);
        assert_eq!(start1["schema_version"], SCHEMA_VERSION_V1);
        assert_eq!(start1["max_validity_ms"], MAX_VALIDITY_MS);
        assert_eq!(start1["builds"], json!([BUILD_ID]));

        let session_id_1 = start1["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_1 = start1["session_key_hex"].as_str().unwrap().to_owned();
        assert_eq!(
            start1_raw,
            format!(
                "{{\"schema\":\"{CONTROL_SCHEMA}\",\"ok\":true,\"profile\":\"{PROFILE_ID}\",\"schema_version\":{SCHEMA_VERSION_V1},\"session_id\":\"{session_id_1}\",\"session_key_hex\":\"{session_key_hex_1}\",\"max_validity_ms\":{MAX_VALIDITY_MS},\"builds\":[\"{BUILD_ID}\"]}}\n"
            )
        );
        assert_eq!(session_id_1.len(), SESSION_ID_BYTES * 2);
        assert!(is_valid_identifier(&session_id_1));
        assert_eq!(session_key_hex_1.len(), SESSION_KEY_LEN * 2);
        let key1 = decode_key_hex(&session_key_hex_1);
        assert_eq!(key1.expose_bytes(), &CANARY_W4B_KEY_1_BYTES);

        // Sequence 0 is usable in session 1; replay of sequence 0 is rejected.
        let (body1_seq0, env1_seq0, tag1_seq0_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 0, NOW_MS - 1_000, NOW_MS + 10_000);
        let req1_seq0 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env1_seq0).unwrap(),
            )
            .body(Body::from(body1_seq0.clone()))
            .unwrap();
        let resp1_seq0 = app.clone().oneshot(req1_seq0).await.unwrap();
        assert_eq!(resp1_seq0.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 1);

        let req1_seq0_replay = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env1_seq0).unwrap(),
            )
            .body(Body::from(body1_seq0))
            .unwrap();
        let resp1_seq0_replay = app.clone().oneshot(req1_seq0_replay).await.unwrap();
        let json_replay1 = assert_carrier_rejection(resp1_seq0_replay, "stale_or_replayed").await;
        assert_eq!(next_stage.call_count(), 1);

        // 3. Wrong control token cannot rotate or end the active session.
        let (bad_rotate_bytes, bad_rotate_stop) = plane.handle_request_line(
            "{\"schema\":\"parapet-control/v0\",\"token\":\"wrong-token\",\"action\":\"provenance_session_start\"}\n",
        );
        let (bad_end_bytes, bad_end_stop) = plane.handle_request_line(
            "{\"schema\":\"parapet-control/v0\",\"token\":\"wrong-token\",\"action\":\"provenance_session_end\"}\n",
        );
        assert!(!bad_rotate_stop);
        assert!(!bad_end_stop);
        let bad_rotate_raw = String::from_utf8(bad_rotate_bytes).unwrap();
        let bad_end_raw = String::from_utf8(bad_end_bytes).unwrap();
        assert_eq!(
            bad_rotate_raw,
            "{\"schema\":\"parapet-control/v0\",\"ok\":false,\"error_code\":\"unauthorized\"}\n"
        );
        assert_eq!(
            bad_end_raw,
            "{\"schema\":\"parapet-control/v0\",\"ok\":false,\"error_code\":\"unauthorized\"}\n"
        );
        assert!(sessions.is_active());

        // Session 1 remains active and unrotated: sequence 1 under session 1 verifies.
        let (body1_seq1, env1_seq1, tag1_seq1_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 1, NOW_MS - 1_000, NOW_MS + 10_000);
        let req1_seq1 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env1_seq1).unwrap(),
            )
            .body(Body::from(body1_seq1))
            .unwrap();
        let resp1_seq1 = app.clone().oneshot(req1_seq1).await.unwrap();
        assert_eq!(resp1_seq1.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 2);

        // 4. Second valid start rotates atomically: old session rejected, new sequence 0 usable,
        //    replay remains rejected.
        let (start2, start2_stop) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        assert!(!start2_stop);
        assert!(sessions.is_active());

        let session_id_2 = start2["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_2 = start2["session_key_hex"].as_str().unwrap().to_owned();
        assert_eq!(session_id_2.len(), SESSION_ID_BYTES * 2);
        assert!(is_valid_identifier(&session_id_2));
        assert_eq!(session_key_hex_2.len(), SESSION_KEY_LEN * 2);
        assert_ne!(session_id_2, session_id_1);
        assert_ne!(session_key_hex_2, session_key_hex_1);
        let key2 = decode_key_hex(&session_key_hex_2);
        assert_eq!(key2.expose_bytes(), &CANARY_W4B_KEY_2_BYTES);

        // Old session with fresh sequence 2 is rejected as session_mismatch.
        let (body_old, env_old, tag_old_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 2, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_old = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_old).unwrap(),
            )
            .body(Body::from(body_old))
            .unwrap();
        let resp_old = app.clone().oneshot(req_old).await.unwrap();
        let json_old_mismatch = assert_carrier_rejection(resp_old, "session_mismatch").await;
        assert_eq!(next_stage.call_count(), 2);

        // Old key under new session_id is rejected as signature_invalid.
        let (body_wrong_key, env_wrong_key, tag_wrong_key_hex) =
            sign_chat_for_session_at(&session_id_2, &key1, 0, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_wrong_key = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_wrong_key).unwrap(),
            )
            .body(Body::from(body_wrong_key))
            .unwrap();
        let resp_wrong_key = app.clone().oneshot(req_wrong_key).await.unwrap();
        let json_wrong_key = assert_carrier_rejection(resp_wrong_key, "signature_invalid").await;
        assert_eq!(next_stage.call_count(), 2);

        // Sequence 0 (already consumed in session 1) is usable in rotated session 2.
        let (body2_seq0, env2_seq0, tag2_seq0_hex) =
            sign_chat_for_session_at(&session_id_2, &key2, 0, NOW_MS - 1_000, NOW_MS + 10_000);
        let req2_seq0 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env2_seq0).unwrap(),
            )
            .body(Body::from(body2_seq0.clone()))
            .unwrap();
        let resp2_seq0 = app.clone().oneshot(req2_seq0).await.unwrap();
        assert_eq!(resp2_seq0.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 3);

        // Replay of sequence 0 in session 2 remains rejected.
        let req2_seq0_replay = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env2_seq0).unwrap(),
            )
            .body(Body::from(body2_seq0))
            .unwrap();
        let resp2_seq0_replay = app.clone().oneshot(req2_seq0_replay).await.unwrap();
        let json_replay2 = assert_carrier_rejection(resp2_seq0_replay, "stale_or_replayed").await;
        assert_eq!(next_stage.call_count(), 3);

        // 5. provenance_session_end removes the active session.
        let (end_resp, end_stop) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_END}"}}"#
            ),
        );
        assert!(!end_stop);
        assert_eq!(
            end_resp,
            json!({
                "schema": CONTROL_SCHEMA,
                "ok": true,
            })
        );
        assert!(!sessions.is_active());

        let (body_after_end, env_after_end, tag_after_end_hex) =
            sign_chat_for_session_at(&session_id_2, &key2, 1, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_after_end = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_after_end).unwrap(),
            )
            .body(Body::from(body_after_end))
            .unwrap();
        let resp_after_end = app.clone().oneshot(req_after_end).await.unwrap();
        let json_after_end = assert_carrier_rejection(resp_after_end, "session_mismatch").await;
        assert_eq!(next_stage.call_count(), 3);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        // Ordinary control responses (`status`, `restore`, `disable`) on the plane.
        let (status_val, _) = send_control_line_in_process(
            &plane,
            &format!(r#"{{"schema":"parapet-control/v0","token":"{token}","action":"status"}}"#),
        );
        let (restore_val, _) = send_control_line_in_process(
            &plane,
            &format!(r#"{{"schema":"parapet-control/v0","token":"{token}","action":"restore"}}"#),
        );
        let (disable_val, disable_stop) = send_control_line_in_process(
            &plane,
            &format!(r#"{{"schema":"parapet-control/v0","token":"{token}","action":"disable"}}"#),
        );
        assert!(disable_stop);

        // 6. Restart / new SessionManager + new EngineControlPlane starts with no active session,
        //    and DefaultProvenanceSessionClockFactory (SystemClock + OsEntropy) works live.
        drop(plane);
        let restarted_sessions = Arc::new(SessionManager::new());
        assert!(!restarted_sessions.is_active());
        let restarted_handle = ProvenanceSessionControl::new(
            ProvenanceRouteBinding::OPENAI,
            restarted_sessions.clone(),
            profile.clone(),
        );
        let (restarted_plane, _restarted_listener, _restarted_rx) =
            EngineControlPlane::bind_with_provenance_session(
                &state_dir,
                45211,
                CONFIG_HASH,
                None,
                Some(restarted_handle.clone()),
            )
            .unwrap();
        assert!(!restarted_sessions.is_active());
        let restarted_token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        let (engine2, _probes2) = build_engine_with_probes();
        let next_stage2 = Arc::new(CapturingNextStage::default());
        let app2 = build_router_with_admission(
            engine2,
            Arc::new(SessionRequestAdmitter::from_session_control(
                ProfileMode::RequiredOpenCode,
                &restarted_handle,
            )),
            Some(next_stage2.clone()),
        );

        // Before start on restarted manager: old session rejected.
        let (body_restart_pre, env_restart_pre, tag_restart_pre_hex) =
            sign_chat_for_session_at(&session_id_2, &key2, 2, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_restart_pre = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_restart_pre).unwrap(),
            )
            .body(Body::from(body_restart_pre))
            .unwrap();
        let resp_restart_pre = app2.clone().oneshot(req_restart_pre).await.unwrap();
        let json_restart_pre = assert_carrier_rejection(resp_restart_pre, "session_mismatch").await;
        assert_eq!(next_stage2.call_count(), 0);

        // Start session on restarted plane using production OsEntropy + SystemClock.
        let (start3, start3_stop) = send_control_line_in_process(
            &restarted_plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{restarted_token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        assert!(!start3_stop);
        assert!(restarted_sessions.is_active());
        let session_id_3 = start3["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_3 = start3["session_key_hex"].as_str().unwrap().to_owned();
        let key3 = decode_key_hex(&session_key_hex_3);
        let live_now = SystemClock::new().now_ms();
        let (body3, env3, tag3_hex) =
            sign_chat_for_session_at(&session_id_3, &key3, 0, live_now, live_now + 20_000);
        let req3 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env3).unwrap())
            .body(Body::from(body3))
            .unwrap();
        let resp3 = app2.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::OK);
        assert_eq!(next_stage2.call_count(), 1);

        // 7. Verify captured TRACE logs, all state-dir files, owner record, ordinary control
        //    responses, Debug outputs, and forwarded requests do NOT contain session key hex,
        //    raw key bytes, or carrier authenticator tags.
        let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
        let owner_bytes = std::fs::read(state_dir.join("engine.owner.json")).unwrap();
        let owner_record: EngineOwnerRecordV0 = serde_json::from_slice(&owner_bytes).unwrap();
        assert_eq!(owner_record.port, 45211);

        let req_debug = format!(
            "{:?}",
            ControlRequestV0 {
                schema: CONTROL_SCHEMA.to_owned(),
                token: restarted_token.clone(),
                action: CONTROL_ACTION_PROVENANCE_SESSION_START.to_owned(),
            }
        );
        let plane_debug = format!("{restarted_plane:?}");
        assert!(req_debug.contains("<redacted>"));
        assert!(!req_debug.contains(&restarted_token));
        assert!(plane_debug.contains("<redacted>"));
        assert!(!plane_debug.contains(&restarted_token));

        let mut surfaces = vec![
            logs,
            req_debug,
            plane_debug,
            format!("{session_handle:?}"),
            format!("{restarted_handle:?}"),
            format!("{owner_record:?}"),
            serde_json::to_string(&owner_record).unwrap(),
            serde_json::to_string(&status_val).unwrap(),
            serde_json::to_string(&restore_val).unwrap(),
            serde_json::to_string(&disable_val).unwrap(),
            serde_json::to_string(&end_resp).unwrap(),
            bad_start_raw,
            bad_rotate_raw,
            bad_end_raw,
            serde_json::to_string(&json_replay1).unwrap(),
            serde_json::to_string(&json_old_mismatch).unwrap(),
            serde_json::to_string(&json_wrong_key).unwrap(),
            serde_json::to_string(&json_replay2).unwrap(),
            serde_json::to_string(&json_after_end).unwrap(),
            serde_json::to_string(&json_restart_pre).unwrap(),
        ];

        for entry in std::fs::read_dir(&state_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let content = std::fs::read(&path).unwrap();
                surfaces.push(String::from_utf8_lossy(&content).into_owned());
            }
        }

        for (_, fwd_req, _) in next_stage
            .take_captured()
            .into_iter()
            .chain(next_stage2.take_captured())
        {
            assert!(fwd_req.headers.get(PROVENANCE_HEADER).is_none());
            surfaces.push(format!("{:?}", fwd_req.headers));
            surfaces.push(String::from_utf8_lossy(&fwd_req.body).into_owned());
        }

        let forbidden = [
            session_key_hex_1.as_str(),
            session_key_hex_2.as_str(),
            session_key_hex_3.as_str(),
            CANARY_W4B_KEY_1_STR,
            CANARY_W4B_KEY_2_STR,
            tag1_seq0_hex.as_str(),
            tag1_seq1_hex.as_str(),
            tag_old_hex.as_str(),
            tag_wrong_key_hex.as_str(),
            tag2_seq0_hex.as_str(),
            tag_after_end_hex.as_str(),
            tag_restart_pre_hex.as_str(),
            tag3_hex.as_str(),
        ];

        for surface in &surfaces {
            for needle in forbidden {
                assert!(
                    !surface.contains(needle),
                    "W4b secret/authenticator leak of {needle:?} in surface: {surface}"
                );
            }
        }
    }

    #[tokio::test]
    async fn w4b_live_unix_socket_control_request_roundtrip() {
        let trace_capture = ThreadTraceGuard::new();
        let temp = tempfile::tempdir().unwrap();
        let state_dir = temp.path().join("state_live_sock");
        init_private_state_dir(&state_dir);

        let sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));
        assert!(!sessions.is_active());

        let profile = supported_profile(MAX_VALIDITY_MS);
        let session_handle = ProvenanceSessionControl::with_clock_factory(
            ProvenanceRouteBinding::OPENAI,
            sessions,
            profile,
            || -> Box<dyn Clock> { Box::new(FixedClock(NOW_MS)) },
        );
        let sessions = session_handle.sessions().clone();
        let (plane, listener, _shutdown_rx) = EngineControlPlane::bind_with_provenance_session(
            &state_dir,
            45220,
            CONFIG_HASH,
            None,
            Some(session_handle.clone()),
        )
        .unwrap();
        assert!(!sessions.is_active());
        plane.spawn_listener(listener);

        // Wrong-UID accept gate seam (also pinned in cli_v0::control_peer_credential_and_line_bound_rev9).
        let owner_uid = std::fs::symlink_metadata(&state_dir).unwrap().uid();
        assert!(!peer_credential_authorized(
            Err(std::io::Error::other("peer_cred failed")),
            owner_uid,
        ));
        assert!(!peer_credential_authorized(
            Ok(owner_uid.wrapping_add(1)),
            owner_uid,
        ));
        assert!(peer_credential_authorized(Ok(owner_uid), owner_uid));

        // Explicit loosened-mode socket refusal: chmod control.sock away from 0600,
        // assert send_control_request returns UnsafePermissions and no session begins,
        // then restore 0600 before the live lifecycle.
        let sock_path = state_dir.join("control.sock");
        std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o660)).unwrap();
        let state_loose = state_dir.clone();
        let loose_res = tokio::task::spawn_blocking(move || {
            send_control_request(&state_loose, CONTROL_ACTION_PROVENANCE_SESSION_START)
        })
        .await
        .unwrap();
        assert_eq!(loose_res, Err(ControlError::UnsafePermissions));
        assert!(!sessions.is_active());
        std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        // Wrong token over control.sock fails with ControlError::AuthFailed and leaves session inactive.
        write_private_file(&state_dir, "control.token", b"wrong-token").unwrap();
        let state_bad = state_dir.clone();
        let bad_res = tokio::task::spawn_blocking(move || {
            send_control_request(&state_bad, CONTROL_ACTION_PROVENANCE_SESSION_START)
        })
        .await
        .unwrap();
        // In environments that forbid UnixStream::connect with EPERM, send_control_request
        // maps the blocked connect to EngineUnavailable; when connect is permitted, it
        // authenticates over control.sock and returns AuthFailed.
        if bad_res == Err(ControlError::EngineUnavailable) {
            let connect_err = std::os::unix::net::UnixStream::connect(&sock_path).unwrap_err();
            assert_eq!(connect_err.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(!sessions.is_active());
            return;
        }
        assert_eq!(bad_res, Err(ControlError::AuthFailed));
        assert!(!sessions.is_active());

        let (engine, probes) = build_engine_with_probes();
        let next_stage = Arc::new(CapturingNextStage::default());
        let app = build_router_with_admission(
            engine,
            Arc::new(session_handle.request_admitter(ProfileMode::RequiredOpenCode)),
            Some(next_stage.clone()),
        );

        // Restore valid token and perform live start -> rotate -> end over control.sock.
        write_private_file(&state_dir, "control.token", token.as_bytes()).unwrap();
        let state_s1 = state_dir.clone();
        let start1 = tokio::task::spawn_blocking(move || {
            send_control_request(&state_s1, CONTROL_ACTION_PROVENANCE_SESSION_START)
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(start1["ok"], true);
        assert!(sessions.is_active());
        let session_id_1 = start1["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_1 = start1["session_key_hex"].as_str().unwrap().to_owned();
        let key1 = decode_key_hex(&session_key_hex_1);
        assert_eq!(key1.expose_bytes(), &CANARY_W4B_KEY_1_BYTES);

        let (body1, env1, tag1_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 0, NOW_MS - 1_000, NOW_MS + 10_000);
        let req1 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env1).unwrap())
            .body(Body::from(body1.clone()))
            .unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 1);

        let req1_replay = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env1).unwrap())
            .body(Body::from(body1))
            .unwrap();
        let resp1_replay = app.clone().oneshot(req1_replay).await.unwrap();
        let json_replay1 = assert_carrier_rejection(resp1_replay, "stale_or_replayed").await;

        let state_s2 = state_dir.clone();
        let start2 = tokio::task::spawn_blocking(move || {
            send_control_request(&state_s2, CONTROL_ACTION_PROVENANCE_SESSION_START)
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(start2["ok"], true);
        let session_id_2 = start2["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_2 = start2["session_key_hex"].as_str().unwrap().to_owned();
        assert_ne!(session_id_1, session_id_2);
        assert_ne!(session_key_hex_1, session_key_hex_2);
        let key2 = decode_key_hex(&session_key_hex_2);
        assert_eq!(key2.expose_bytes(), &CANARY_W4B_KEY_2_BYTES);

        let (body_old, env_old, tag_old_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 1, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_old = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_old).unwrap(),
            )
            .body(Body::from(body_old))
            .unwrap();
        let resp_old = app.clone().oneshot(req_old).await.unwrap();
        let json_old = assert_carrier_rejection(resp_old, "session_mismatch").await;

        let (body2, env2, tag2_hex) =
            sign_chat_for_session_at(&session_id_2, &key2, 0, NOW_MS - 1_000, NOW_MS + 10_000);
        let req2 = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env2).unwrap())
            .body(Body::from(body2))
            .unwrap();
        let resp2 = app.clone().oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 2);

        let state_end = state_dir.clone();
        let end_resp = tokio::task::spawn_blocking(move || {
            send_control_request(&state_end, CONTROL_ACTION_PROVENANCE_SESSION_END)
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(end_resp, json!({"schema": CONTROL_SCHEMA, "ok": true}));
        assert!(!sessions.is_active());

        let (body_after_end, env_after_end, tag_after_end_hex) =
            sign_chat_for_session_at(&session_id_2, &key2, 1, NOW_MS - 1_000, NOW_MS + 10_000);
        let req_after_end = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_after_end).unwrap(),
            )
            .body(Body::from(body_after_end))
            .unwrap();
        let resp_after_end = app.oneshot(req_after_end).await.unwrap();
        let json_after_end = assert_carrier_rejection(resp_after_end, "session_mismatch").await;
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
        let mut live_surfaces = vec![
            logs,
            format!("{plane:?}"),
            format!("{session_handle:?}"),
            serde_json::to_string(&end_resp).unwrap(),
            serde_json::to_string(&json_replay1).unwrap(),
            serde_json::to_string(&json_old).unwrap(),
            serde_json::to_string(&json_after_end).unwrap(),
        ];
        for entry in std::fs::read_dir(&state_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let content = std::fs::read(&path).unwrap();
                live_surfaces.push(String::from_utf8_lossy(&content).into_owned());
            }
        }
        for (_, fwd_req, _) in next_stage.take_captured() {
            assert!(fwd_req.headers.get(PROVENANCE_HEADER).is_none());
            live_surfaces.push(format!("{:?}", fwd_req.headers));
            live_surfaces.push(String::from_utf8_lossy(&fwd_req.body).into_owned());
        }

        let forbidden = [
            session_key_hex_1.as_str(),
            session_key_hex_2.as_str(),
            CANARY_W4B_KEY_1_STR,
            CANARY_W4B_KEY_2_STR,
            tag1_hex.as_str(),
            tag_old_hex.as_str(),
            tag2_hex.as_str(),
            tag_after_end_hex.as_str(),
        ];
        for surface in &live_surfaces {
            for needle in forbidden {
                assert!(
                    !surface.contains(needle),
                    "live socket TRACE/surface leaked secret or authenticator {needle:?}: {surface}"
                );
            }
        }
    }

    fn sign_anthropic_for_session_at(
        session_id: &str,
        key: &SessionKey,
        sequence: u64,
        issued_at_ms: u64,
        expires_at_ms: u64,
    ) -> (Vec<u8>, Vec<u8>, String) {
        let sys_content = format!("{CANARY_VENDOR}{CANARY_REPO}");
        let body = serde_json::to_vec(&json!({
            "model": "claude-3-5-sonnet",
            "system": sys_content,
            "messages": [
                {"role": "user", "content": CANARY_USER}
            ]
        }))
        .unwrap();

        let v_len = CANARY_VENDOR.len();
        let s_len = sys_content.len();
        let u_len = CANARY_USER.len();
        let source_map = SourceMap {
            segments: vec![
                Segment {
                    message_index: 0,
                    start: 0,
                    end: v_len,
                    source: Source::VendorStatic,
                    sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
                },
                Segment {
                    message_index: 0,
                    start: v_len,
                    end: s_len,
                    source: Source::Repository,
                    sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
                },
                Segment {
                    message_index: 1,
                    start: 0,
                    end: u_len,
                    source: Source::User,
                    sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
                },
            ],
        };

        let mut env = CarrierEnvelope {
            schema_version: SCHEMA_VERSION_V1,
            profile: PROFILE_ID.to_string(),
            build: BUILD_ID.to_string(),
            session_id: session_id.to_string(),
            sequence,
            issued_at_ms,
            expires_at_ms,
            request_body_sha256: Sha256Digest::of(&body),
            source_map_sha256: source_map.canonical_digest(),
            source_map,
            authenticator: Authenticator::from_bytes(vec![0u8; 32]),
        };
        let auth = HmacSha256Authenticator::new(key.clone());
        let tag = auth.tag(&env.signed_payload());
        let tag_hex = Sha256Digest::from_bytes(tag).to_hex();
        env.authenticator = Authenticator::from_bytes(tag.to_vec());
        let env_bytes = serde_json::to_vec(&env).unwrap();
        (body, env_bytes, tag_hex)
    }

    #[tokio::test]
    async fn w4c1_anthropic_bound_session_admission_route_gate_and_failure_closure() {
        let trace_capture = ThreadTraceGuard::new();
        let temp = tempfile::tempdir().unwrap();
        let state_dir = temp.path().join("state_w4c1_anthropic");
        init_private_state_dir(&state_dir);

        let route = ProvenanceRouteBinding::ANTHROPIC;
        assert_eq!(route, ProvenanceRouteBinding::from(Provider::Anthropic));
        assert_eq!(route.provider(), Provider::Anthropic);
        assert_eq!(route.canonical_post_path(), "/v1/messages");
        assert_eq!(route.provider_type(), ProviderType::Anthropic);

        let sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));
        let profile = supported_profile(MAX_VALIDITY_MS);

        // Construct ProvenanceSessionControl and SessionRequestAdmitter from the single
        // closed ProvenanceRouteBinding::ANTHROPIC so verifier and post-verify decoders
        // cannot diverge.
        let session_handle = ProvenanceSessionControl::with_clock_factory(
            route,
            sessions.clone(),
            profile.clone(),
            || -> Box<dyn Clock> { Box::new(FixedClock(NOW_MS)) },
        );
        assert_eq!(session_handle.route_binding(), route);
        let admitter = Arc::new(session_handle.request_admitter(ProfileMode::RequiredOpenCode));
        assert_eq!(admitter.bound_route(), route);
        assert_eq!(admitter.bound_provider(), Provider::Anthropic);
        assert_eq!(admitter.canonical_post_path(), "/v1/messages");

        let (plane, _listener, _shutdown_rx) = EngineControlPlane::bind_with_provenance_session(
            &state_dir,
            45230,
            CONFIG_HASH,
            None,
            Some(session_handle.clone()),
        )
        .unwrap();
        let token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        let (engine, probes) = build_engine_with_probes();
        let next_stage = Arc::new(CapturingNextStage::default());
        let app = build_router_with_admission(engine, admitter, Some(next_stage.clone()));

        let (fallback_body, fallback_env, fallback_tag_hex) = sign_anthropic_for_session_at(
            SESSION_ID,
            &SessionKey::from_bytes(CANARY_KEY_BYTES),
            1,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );

        // 1. No active session on POST /v1/messages rejects session_mismatch with zero sends.
        let req_no_session = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&fallback_env).unwrap(),
            )
            .body(Body::from(fallback_body.clone()))
            .unwrap();
        let resp_no_session = app.clone().oneshot(req_no_session).await.unwrap();
        let json_no_session = assert_carrier_rejection(resp_no_session, "session_mismatch").await;
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
        assert_eq!(next_stage.call_count(), 0);

        // 2. Start Anthropic session via owner-only control wire.
        let (start1, start1_stop) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        assert!(!start1_stop);
        assert_eq!(start1["ok"], true);
        let session_id_1 = start1["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_1 = start1["session_key_hex"].as_str().unwrap().to_owned();
        let key1 = decode_key_hex(&session_key_hex_1);
        assert_eq!(key1.expose_bytes(), &CANARY_W4B_KEY_1_BYTES);

        // 3. Valid Anthropic request + valid carrier on POST /v1/messages reaches
        //    VerifiedCarrierNextStage with Provider::Anthropic and sanitized headers.
        let (anth_body, anth_env, anth_tag_hex) = sign_anthropic_for_session_at(
            &session_id_1,
            &key1,
            0,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let req_valid = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .header("X-Guard-Trust-Span", CANARY_TRUST_HEADER)
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&anth_env).unwrap(),
            )
            .body(Body::from(anth_body.clone()))
            .unwrap();
        let resp_valid = app.clone().oneshot(req_valid).await.unwrap();
        assert_eq!(resp_valid.status(), StatusCode::OK);
        assert!(resp_valid.headers().get("x-guard-trust-response").is_none());
        assert_eq!(probes.l0_calls.load(Ordering::SeqCst), 0);
        assert_eq!(probes.trust_calls.load(Ordering::SeqCst), 0);
        assert_eq!(probes.l1_calls.load(Ordering::SeqCst), 0);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
        assert_eq!(next_stage.call_count(), 1);

        let captured = next_stage.take_captured();
        assert_eq!(captured.len(), 1);
        let (provider, proxy_req, verified) = &captured[0];
        assert_eq!(*provider, Provider::Anthropic);
        assert!(proxy_req
            .headers
            .get_all(PROVENANCE_HEADER)
            .iter()
            .next()
            .is_none());
        assert!(!proxy_req
            .headers
            .keys()
            .any(|k| k.as_str().starts_with("x-guard-trust")));
        assert_eq!(
            proxy_req.headers.get("authorization").unwrap(),
            CANARY_AUTH_HEADER
        );
        assert_eq!(proxy_req.body.as_ref(), anth_body.as_slice());
        assert_eq!(verified.identity.profile, PROFILE_ID);
        assert_eq!(verified.identity.build, BUILD_ID);
        assert_eq!(verified.identity.schema_version, SCHEMA_VERSION_V1);
        assert_eq!(verified.segments.len(), 3);
        assert_eq!(
            verified.messages,
            vec![
                format!("{CANARY_VENDOR}{CANARY_REPO}"),
                CANARY_USER.to_string(),
            ]
        );

        // Replay of sequence 0 on POST /v1/messages rejects stale_or_replayed with zero sends.
        let req_replay = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&anth_env).unwrap(),
            )
            .body(Body::from(anth_body.clone()))
            .unwrap();
        let resp_replay = app.clone().oneshot(req_replay).await.unwrap();
        let json_replay = assert_carrier_rejection(resp_replay, "stale_or_replayed").await;
        assert_eq!(next_stage.call_count(), 1);

        // 4. Wrong routes (including OpenAI /v1/chat/completions) and non-POST methods
        //    on the Anthropic-bound router fail schema_mismatch with zero sends.
        let (anth_body_s1, anth_env_s1, anth_tag_s1_hex) = sign_anthropic_for_session_at(
            &session_id_1,
            &key1,
            1,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let (oai_body_s1, oai_env_s1, oai_tag_s1_hex) =
            sign_chat_for_session_at(&session_id_1, &key1, 1, NOW_MS - 1_000, NOW_MS + 10_000);

        for wrong_path in [
            "/v1/chat/completions",
            "/openai/v1/chat/completions",
            "/v1/messages/extra",
            "/v1/messages/count_tokens",
            "/v1/messages/batches",
            "/v1/responses",
            "/v1/unknown",
            "/v1//messages",
        ] {
            for (b, e) in [(&anth_body_s1, &anth_env_s1), (&oai_body_s1, &oai_env_s1)] {
                let req = Request::builder()
                    .method(Method::POST)
                    .uri(wrong_path)
                    .header("content-type", "application/json")
                    .header(PROVENANCE_HEADER, HeaderValue::from_bytes(e).unwrap())
                    .header("x-guard-trust", CANARY_TRUST_HEADER)
                    .body(Body::from(b.clone()))
                    .unwrap();
                let resp = app.clone().oneshot(req).await.unwrap();
                assert_carrier_rejection(resp, "schema_mismatch").await;
            }
        }

        for wrong_method in [Method::GET, Method::PUT, Method::DELETE] {
            for path in ["/v1/messages", "/v1/chat/completions", "/v1/unknown"] {
                let req = Request::builder()
                    .method(wrong_method.clone())
                    .uri(path)
                    .header("content-type", "application/json")
                    .header(
                        PROVENANCE_HEADER,
                        HeaderValue::from_bytes(&anth_env_s1).unwrap(),
                    )
                    .body(Body::from(anth_body_s1.clone()))
                    .unwrap();
                let resp = app.clone().oneshot(req).await.unwrap();
                assert_carrier_rejection(resp, "schema_mismatch").await;
            }
        }
        assert_eq!(next_stage.call_count(), 1);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        // 5. Anthropic absent, duplicate, oversize, and malformed carrier header rejections
        //    on POST /v1/messages.
        let req_absent = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .body(Body::from(anth_body_s1.clone()))
            .unwrap();
        let resp_absent = app.clone().oneshot(req_absent).await.unwrap();
        let json_absent = assert_carrier_rejection(resp_absent, "absent").await;

        let mut req_dup = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(anth_body_s1.clone()))
            .unwrap();
        req_dup.headers_mut().append(
            PROVENANCE_HEADER,
            HeaderValue::from_bytes(&anth_env_s1).unwrap(),
        );
        req_dup.headers_mut().append(
            PROVENANCE_HEADER,
            HeaderValue::from_static(CANARY_RAW_ENV_CLAIM),
        );
        let resp_dup = app.clone().oneshot(req_dup).await.unwrap();
        let json_dup = assert_carrier_rejection(resp_dup, "schema_mismatch").await;

        let over = format!(
            "{CANARY_RAW_ENV_CLAIM}{}",
            "a".repeat(MAX_ENVELOPE_BYTES + 8)
        );
        let req_over = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(PROVENANCE_HEADER, HeaderValue::from_str(&over).unwrap())
            .body(Body::from(anth_body_s1.clone()))
            .unwrap();
        let resp_over = app.clone().oneshot(req_over).await.unwrap();
        let json_over = assert_carrier_rejection(resp_over, "schema_mismatch").await;

        let req_malformed_env = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_static("{not-valid-carrier-json"),
            )
            .body(Body::from(anth_body_s1.clone()))
            .unwrap();
        let resp_malformed_env = app.clone().oneshot(req_malformed_env).await.unwrap();
        let json_malformed_env = assert_carrier_rejection(resp_malformed_env, "schema_mismatch").await;

        // 6. Anthropic duplicate JSON keys, malformed Anthropic body, and OpenAI-shaped body
        //    on POST /v1/messages reject map_out_of_range with zero sends.
        //    (Note: sequence numbers are consumed before body decode, so use distinct sequences.)
        let req_oai_on_anth = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&oai_env_s1).unwrap(),
            )
            .body(Body::from(oai_body_s1))
            .unwrap();
        let resp_oai_on_anth = app.clone().oneshot(req_oai_on_anth).await.unwrap();
        let json_oai_on_anth =
            assert_carrier_rejection(resp_oai_on_anth, "map_out_of_range").await;

        let dup_key_body = format!(
            r#"{{"model":"claude-3-5-sonnet","system":{sys:?},"messages":[{{"role":"user","content":{u:?},"content":{u:?}}}]}}"#,
            sys = format!("{CANARY_VENDOR}{CANARY_REPO}"),
            u = CANARY_USER,
        )
        .into_bytes();
        let (_, mut dup_key_env_val, _): (Vec<u8>, Vec<u8>, String) =
            sign_anthropic_for_session_at(
                &session_id_1,
                &key1,
                2,
                NOW_MS - 1_000,
                NOW_MS + 10_000,
            );
        let mut dup_env: CarrierEnvelope = serde_json::from_slice(&dup_key_env_val).unwrap();
        dup_env.request_body_sha256 = Sha256Digest::of(&dup_key_body);
        let auth1 = HmacSha256Authenticator::new(key1.clone());
        let dup_tag = auth1.tag(&dup_env.signed_payload());
        let dup_tag_hex = Sha256Digest::from_bytes(dup_tag).to_hex();
        dup_env.authenticator = Authenticator::from_bytes(dup_tag.to_vec());
        dup_key_env_val = serde_json::to_vec(&dup_env).unwrap();

        let req_dup_json_key = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&dup_key_env_val).unwrap(),
            )
            .body(Body::from(dup_key_body))
            .unwrap();
        let resp_dup_json_key = app.clone().oneshot(req_dup_json_key).await.unwrap();
        let json_dup_json_key =
            assert_carrier_rejection(resp_dup_json_key, "map_out_of_range").await;

        // 7. Rotate Anthropic session, verify old session rejects session_mismatch, new session
        //    verifies, end session rejects session_mismatch, injected ProvenanceSessionClockFactory
        //    via ProvenanceSessionControl::new_with enforces Anthropic decoder + deterministic/failing
        //    clocks, and DefaultProvenanceSessionClockFactory works live.
        let (start2, start2_stop) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        assert!(!start2_stop);
        let session_id_2 = start2["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_2 = start2["session_key_hex"].as_str().unwrap().to_owned();
        let key2 = decode_key_hex(&session_key_hex_2);

        let (body_old, env_old, tag_old_hex) = sign_anthropic_for_session_at(
            &session_id_1,
            &key1,
            3,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let req_old = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_old).unwrap(),
            )
            .body(Body::from(body_old))
            .unwrap();
        let resp_old = app.clone().oneshot(req_old).await.unwrap();
        let json_old = assert_carrier_rejection(resp_old, "session_mismatch").await;

        let (body2, env2, tag2_hex) = sign_anthropic_for_session_at(
            &session_id_2,
            &key2,
            0,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let req2 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env2).unwrap())
            .body(Body::from(body2))
            .unwrap();
        let resp2 = app.clone().oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 2);

        let (end_resp, end_stop) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_END}"}}"#
            ),
        );
        assert!(!end_stop);
        assert_eq!(end_resp, json!({"schema": CONTROL_SCHEMA, "ok": true}));
        assert!(!sessions.is_active());

        let (body_after_end, env_after_end, tag_after_end_hex) = sign_anthropic_for_session_at(
            &session_id_2,
            &key2,
            1,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let req_after_end = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_after_end).unwrap(),
            )
            .body(Body::from(body_after_end))
            .unwrap();
        let resp_after_end = app.oneshot(req_after_end).await.unwrap();
        let json_after_end = assert_carrier_rejection(resp_after_end, "session_mismatch").await;
        assert_eq!(next_stage.call_count(), 2);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        // Injected ProvenanceSessionClockFactory seam via ProvenanceSessionControl::new_with:
        // verifies Anthropic content, rejects OpenAI-shaped content with map_out_of_range,
        // and rejects expired carriers with stale_or_replayed when the injected clock advances.
        struct SteppingClockFactory(AtomicUsize);
        impl ProvenanceSessionClockFactory for SteppingClockFactory {
            fn build_clock(&self) -> Box<dyn Clock> {
                match self.0.fetch_add(1, Ordering::SeqCst) {
                    0 => Box::new(FixedClock(NOW_MS)),
                    _ => Box::new(FixedClock(NOW_MS + 120_000)),
                }
            }
        }

        let seam_sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));
        let clock_factory: Arc<dyn ProvenanceSessionClockFactory> =
            Arc::new(SteppingClockFactory(AtomicUsize::new(0)));
        let seam_handle = ProvenanceSessionControl::new_with(
            route,
            seam_sessions.clone(),
            profile.clone(),
            clock_factory,
        );
        let (seam_engine, seam_probes) = build_engine_with_probes();
        let seam_next_stage = Arc::new(CapturingNextStage::default());
        let seam_app = build_router_with_admission(
            seam_engine,
            Arc::new(seam_handle.request_admitter(ProfileMode::RequiredOpenCode)),
            Some(seam_next_stage.clone()),
        );
        let seam_grant_1 = seam_handle.start_session().unwrap();
        let (seam_anth_body, seam_anth_env, seam_anth_tag_hex) = sign_anthropic_for_session_at(
            &seam_grant_1.session_id,
            &seam_grant_1.key,
            0,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let seam_req_anth = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&seam_anth_env).unwrap(),
            )
            .body(Body::from(seam_anth_body))
            .unwrap();
        let seam_resp_anth = seam_app.clone().oneshot(seam_req_anth).await.unwrap();
        assert_eq!(seam_resp_anth.status(), StatusCode::OK);
        assert_eq!(seam_next_stage.call_count(), 1);

        let (seam_oai_body, seam_oai_env, seam_oai_tag_hex) = sign_chat_for_session_at(
            &seam_grant_1.session_id,
            &seam_grant_1.key,
            1,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let seam_req_oai = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&seam_oai_env).unwrap(),
            )
            .body(Body::from(seam_oai_body))
            .unwrap();
        let seam_resp_oai = seam_app.clone().oneshot(seam_req_oai).await.unwrap();
        let json_seam_oai = assert_carrier_rejection(seam_resp_oai, "map_out_of_range").await;
        assert_eq!(seam_next_stage.call_count(), 1);

        // Rotate via start_session() so SteppingClockFactory returns FixedClock(NOW_MS + 120_000),
        // causing an expired Anthropic carrier to fail closed with stale_or_replayed.
        let seam_grant_2 = seam_handle.start_session().unwrap();
        let (seam_exp_body, seam_exp_env, seam_exp_tag_hex) = sign_anthropic_for_session_at(
            &seam_grant_2.session_id,
            &seam_grant_2.key,
            0,
            NOW_MS - 1_000,
            NOW_MS + 10_000,
        );
        let seam_req_exp = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&seam_exp_env).unwrap(),
            )
            .body(Body::from(seam_exp_body))
            .unwrap();
        let seam_resp_exp = seam_app.oneshot(seam_req_exp).await.unwrap();
        let json_seam_exp = assert_carrier_rejection(seam_resp_exp, "stale_or_replayed").await;
        assert_eq!(seam_next_stage.call_count(), 1);
        assert_eq!(seam_probes.http_calls.load(Ordering::SeqCst), 0);

        // Live DefaultProvenanceSessionClockFactory bound to ProvenanceRouteBinding::ANTHROPIC.
        assert_eq!(
            DefaultProvenanceSessionClockFactory,
            DefaultProvenanceSessionClockFactory::default()
        );
        drop(plane);
        let live_sessions = Arc::new(SessionManager::new());
        let live_handle =
            ProvenanceSessionControl::new(route, live_sessions.clone(), profile.clone());
        let (live_plane, _live_listener, _live_rx) =
            EngineControlPlane::bind_with_provenance_session(
                &state_dir,
                45231,
                CONFIG_HASH,
                None,
                Some(live_handle.clone()),
            )
            .unwrap();
        let live_token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();
        let (live_engine, live_probes) = build_engine_with_probes();
        let live_next_stage = Arc::new(CapturingNextStage::default());
        let live_app = build_router_with_admission(
            live_engine,
            Arc::new(live_handle.request_admitter(ProfileMode::RequiredOpenCode)),
            Some(live_next_stage.clone()),
        );
        let (start3, _) = send_control_line_in_process(
            &live_plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{live_token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        let session_id_3 = start3["session_id"].as_str().unwrap().to_owned();
        let session_key_hex_3 = start3["session_key_hex"].as_str().unwrap().to_owned();
        let key3 = decode_key_hex(&session_key_hex_3);
        let live_now = SystemClock::new().now_ms();
        let (body3, env3, tag3_hex) = sign_anthropic_for_session_at(
            &session_id_3,
            &key3,
            0,
            live_now,
            live_now + 20_000,
        );
        let req3 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env3).unwrap())
            .body(Body::from(body3))
            .unwrap();
        let resp3 = live_app.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::OK);
        assert_eq!(live_next_stage.call_count(), 1);
        assert_eq!(live_probes.http_calls.load(Ordering::SeqCst), 0);

        let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
        let surfaces: Vec<String> = [
            logs,
            serde_json::to_string(&json_no_session).unwrap(),
            serde_json::to_string(&json_replay).unwrap(),
            serde_json::to_string(&json_absent).unwrap(),
            serde_json::to_string(&json_dup).unwrap(),
            serde_json::to_string(&json_over).unwrap(),
            serde_json::to_string(&json_malformed_env).unwrap(),
            serde_json::to_string(&json_oai_on_anth).unwrap(),
            serde_json::to_string(&json_dup_json_key).unwrap(),
            serde_json::to_string(&json_old).unwrap(),
            serde_json::to_string(&json_after_end).unwrap(),
            serde_json::to_string(&json_seam_oai).unwrap(),
            serde_json::to_string(&json_seam_exp).unwrap(),
        ]
        .into_iter()
        .collect();

        let forbidden = [
            session_key_hex_1.as_str(),
            session_key_hex_2.as_str(),
            session_key_hex_3.as_str(),
            CANARY_W4B_KEY_1_STR,
            CANARY_W4B_KEY_2_STR,
            CANARY_KEY_STR,
            fallback_tag_hex.as_str(),
            anth_tag_hex.as_str(),
            anth_tag_s1_hex.as_str(),
            oai_tag_s1_hex.as_str(),
            dup_tag_hex.as_str(),
            tag_old_hex.as_str(),
            tag2_hex.as_str(),
            tag_after_end_hex.as_str(),
            seam_anth_tag_hex.as_str(),
            seam_oai_tag_hex.as_str(),
            seam_exp_tag_hex.as_str(),
            tag3_hex.as_str(),
            CANARY_VENDOR.trim(),
            CANARY_REPO.trim(),
            CANARY_USER,
            CANARY_LOCAL_PATH,
            CANARY_RAW_ENV_CLAIM,
            CANARY_TRUST_HEADER,
            CANARY_AUTH_HEADER,
        ];
        for surface in &surfaces {
            for needle in forbidden {
                assert!(
                    !surface.contains(needle),
                    "W4c1 Anthropic surface leaked {needle:?}: {surface}"
                );
            }
        }
    }

    #[tokio::test]
    async fn w4c2b_anthropic_array_system_session_admission_unmodified_opencode_profile_nonvendor_and_unpinned_vendor_reject() {
        let trace_capture = ThreadTraceGuard::new();
        let temp = tempfile::tempdir().unwrap();
        let state_dir = temp.path().join("s_w4c2b_unmod");
        init_private_state_dir(&state_dir);

        let route = ProvenanceRouteBinding::ANTHROPIC;
        let sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));

        // Unmodified production opencode_2_0_32_profile() ("843a6c51", 11 pinned digests).
        // Committed tests do not include OpenCode prompt prose, so against the unmodified
        // pinned manifest:
        // - synthetic CANARY_VENDOR claimed as VendorStatic rejects with vendor_digest_mismatch;
        // - non-vendor array-system blocks (ConfiguredAgent + Repository + User) with
        //   cache_control and Message::content offsets verify and admit.
        let unmodified_profile = opencode_2_0_32_profile();
        let session_handle = ProvenanceSessionControl::with_clock_factory(
            route,
            sessions,
            unmodified_profile,
            || -> Box<dyn Clock> { Box::new(FixedClock(NOW_MS)) },
        );
        let (plane, _listener, _shutdown_rx) = EngineControlPlane::bind_with_provenance_session(
            &state_dir,
            45234,
            CONFIG_HASH,
            None,
            Some(session_handle.clone()),
        )
        .unwrap();
        let token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        let (start1, _) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        let session_id = start1["session_id"].as_str().unwrap().to_owned();
        let session_key_hex = start1["session_key_hex"].as_str().unwrap().to_owned();
        let key = decode_key_hex(&session_key_hex);

        let (engine, probes) = build_engine_with_probes();
        let next_stage = Arc::new(CapturingNextStage::default());
        let app = build_router_with_admission(
            engine,
            Arc::new(session_handle.request_admitter(ProfileMode::RequiredOpenCode)),
            Some(next_stage.clone()),
        );

        let array_body = serde_json::to_vec(&json!({
            "model": "claude-3-5-sonnet",
            "system": [
                {
                    "type": "text",
                    "text": CANARY_VENDOR,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": CANARY_REPO,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CANARY_USER}
                    ]
                }
            ]
        }))
        .unwrap();
        let v_len = CANARY_VENDOR.len();
        let s_len = v_len + CANARY_REPO.len();
        let u_len = CANARY_USER.len();

        let sign_opencode_env =
            |body_bytes: &[u8], segs: Vec<Segment>, seq: u64| -> (Vec<u8>, String) {
                let source_map = SourceMap { segments: segs };
                let mut env = CarrierEnvelope {
                    schema_version: OPENCODE_SCHEMA_VERSION,
                    profile: OPENCODE_PROFILE_ID.to_string(),
                    build: OPENCODE_BUILD.to_string(),
                    session_id: session_id.clone(),
                    sequence: seq,
                    issued_at_ms: NOW_MS - 1_000,
                    expires_at_ms: NOW_MS + 10_000,
                    request_body_sha256: Sha256Digest::of(body_bytes),
                    source_map_sha256: source_map.canonical_digest(),
                    source_map,
                    authenticator: Authenticator::from_bytes(vec![0u8; 32]),
                };
                let auth = HmacSha256Authenticator::new(key.clone());
                let tag = auth.tag(&env.signed_payload());
                let tag_hex = Sha256Digest::from_bytes(tag).to_hex();
                env.authenticator = Authenticator::from_bytes(tag.to_vec());
                (serde_json::to_vec(&env).unwrap(), tag_hex)
            };

        // 1. Unpinned synthetic CANARY_VENDOR claimed as VendorStatic rejects with
        //    vendor_digest_mismatch against the unmodified 843a6c51 manifest (0 sends).
        let unpinned_vendor_segs = vec![
            Segment {
                message_index: 0,
                start: 0,
                end: v_len,
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
            },
            Segment {
                message_index: 0,
                start: v_len,
                end: s_len,
                source: Source::Repository,
                sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: u_len,
                source: Source::User,
                sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
            },
        ];
        let (env_unpinned, tag_unpinned_hex) =
            sign_opencode_env(&array_body, unpinned_vendor_segs.clone(), 0);
        let req_unpinned = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_unpinned).unwrap(),
            )
            .body(Body::from(array_body.clone()))
            .unwrap();
        let resp_unpinned = app.clone().oneshot(req_unpinned).await.unwrap();
        let json_unpinned = assert_carrier_rejection(resp_unpinned, "vendor_digest_mismatch").await;
        assert_eq!(next_stage.call_count(), 0);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        // 2. Non-vendor array-system segments verify and admit against unmodified opencode_2_0_32_profile().
        let mut nonvendor_segs = unpinned_vendor_segs;
        nonvendor_segs[0].source = Source::ConfiguredAgent;
        let (env_nonvendor, tag_nonvendor_hex) =
            sign_opencode_env(&array_body, nonvendor_segs, 1);
        let req_nonvendor = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .header(
                PROVENANCE_HEADER,
                HeaderValue::from_bytes(&env_nonvendor).unwrap(),
            )
            .body(Body::from(array_body))
            .unwrap();
        let resp_nonvendor = app.oneshot(req_nonvendor).await.unwrap();
        assert_eq!(resp_nonvendor.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 1);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);
        let captured = next_stage.take_captured();
        assert_eq!(captured[0].0, Provider::Anthropic);
        assert_eq!(captured[0].2.identity.profile, OPENCODE_PROFILE_ID);
        assert_eq!(captured[0].2.identity.build, OPENCODE_BUILD);
        assert_eq!(
            captured[0].2.messages,
            vec![
                format!("{CANARY_VENDOR}{CANARY_REPO}"),
                CANARY_USER.to_string()
            ]
        );

        let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
        for surface in [logs, serde_json::to_string(&json_unpinned).unwrap()] {
            for needle in [
                session_key_hex.as_str(),
                CANARY_W4B_KEY_1_STR,
                tag_unpinned_hex.as_str(),
                tag_nonvendor_hex.as_str(),
                CANARY_VENDOR.trim(),
                CANARY_REPO.trim(),
                CANARY_USER,
            ] {
                assert!(
                    !surface.contains(needle),
                    "W4c2b unmodified-profile surface leaked {needle:?}: {surface}"
                );
            }
        }
    }

    #[tokio::test]
    async fn w4c2b_anthropic_array_system_session_admission_vendor_static_geometry_with_synthetic_manifest() {
        let trace_capture = ThreadTraceGuard::new();
        let temp = tempfile::tempdir().unwrap();
        let state_dir = temp.path().join("s_w4c2b_synth");
        init_private_state_dir(&state_dir);

        let route = ProvenanceRouteBinding::ANTHROPIC;
        let sessions = Arc::new(SessionManager::with_entropy(Box::new(
            ScriptedSessionEntropy::new(),
        )));

        // Synthetic manifest profile (supported_profile(MAX_VALIDITY_MS) pins CANARY_VENDOR under BUILD_ID):
        // proves vendor_static + Repository + User Message::content geometry on Anthropic array-form
        // system blocks and fail-closed offset/digest rejections without mutating opencode_2_0_32_profile().
        let session_handle = ProvenanceSessionControl::with_clock_factory(
            route,
            sessions.clone(),
            supported_profile(MAX_VALIDITY_MS),
            || -> Box<dyn Clock> { Box::new(FixedClock(NOW_MS)) },
        );
        let (plane, _listener, _shutdown_rx) = EngineControlPlane::bind_with_provenance_session(
            &state_dir,
            45235,
            CONFIG_HASH,
            None,
            Some(session_handle.clone()),
        )
        .unwrap();
        let token = std::fs::read_to_string(state_dir.join("control.token")).unwrap();

        let (start1, _) = send_control_line_in_process(
            &plane,
            &format!(
                r#"{{"schema":"parapet-control/v0","token":"{token}","action":"{CONTROL_ACTION_PROVENANCE_SESSION_START}"}}"#
            ),
        );
        let session_id = start1["session_id"].as_str().unwrap().to_owned();
        let session_key_hex = start1["session_key_hex"].as_str().unwrap().to_owned();
        let key = decode_key_hex(&session_key_hex);

        let (engine, probes) = build_engine_with_probes();
        let next_stage = Arc::new(CapturingNextStage::default());
        let app = build_router_with_admission(
            engine,
            Arc::new(session_handle.request_admitter(ProfileMode::RequiredOpenCode)),
            Some(next_stage.clone()),
        );

        let array_body_json = json!({
            "model": "claude-3-5-sonnet",
            "system": [
                {
                    "type": "text",
                    "text": CANARY_VENDOR,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": CANARY_REPO,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CANARY_USER}
                    ]
                }
            ]
        });
        let array_body = serde_json::to_vec(&array_body_json).unwrap();
        let v_len = CANARY_VENDOR.len();
        let s_len = v_len + CANARY_REPO.len();
        let u_len = CANARY_USER.len();
        let valid_segments = vec![
            Segment {
                message_index: 0,
                start: 0,
                end: v_len,
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(CANARY_VENDOR.as_bytes()),
            },
            Segment {
                message_index: 0,
                start: v_len,
                end: s_len,
                source: Source::Repository,
                sha256: Sha256Digest::of(CANARY_REPO.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: u_len,
                source: Source::User,
                sha256: Sha256Digest::of(CANARY_USER.as_bytes()),
            },
        ];

        let sign_env = |body_bytes: &[u8], segs: Vec<Segment>, seq: u64| -> (Vec<u8>, String) {
            let source_map = SourceMap { segments: segs };
            let mut env = CarrierEnvelope {
                schema_version: SCHEMA_VERSION_V1,
                profile: PROFILE_ID.to_string(),
                build: BUILD_ID.to_string(),
                session_id: session_id.clone(),
                sequence: seq,
                issued_at_ms: NOW_MS - 1_000,
                expires_at_ms: NOW_MS + 10_000,
                request_body_sha256: Sha256Digest::of(body_bytes),
                source_map_sha256: source_map.canonical_digest(),
                source_map,
                authenticator: Authenticator::from_bytes(vec![0u8; 32]),
            };
            let auth = HmacSha256Authenticator::new(key.clone());
            let tag = auth.tag(&env.signed_payload());
            let tag_hex = Sha256Digest::from_bytes(tag).to_hex();
            env.authenticator = Authenticator::from_bytes(tag.to_vec());
            (serde_json::to_vec(&env).unwrap(), tag_hex)
        };

        // 1. Valid Anthropic array-system request with cache_control and Message::content offsets admits
        //    against the synthetic manifest.
        let (env0, tag0_hex) = sign_env(&array_body, valid_segments.clone(), 0);
        let req0 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", CANARY_AUTH_HEADER)
            .header("x-guard-trust", CANARY_TRUST_HEADER)
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env0).unwrap())
            .body(Body::from(array_body.clone()))
            .unwrap();
        let resp0 = app.clone().oneshot(req0).await.unwrap();
        assert_eq!(resp0.status(), StatusCode::OK);
        assert_eq!(next_stage.call_count(), 1);
        let captured = next_stage.take_captured();
        assert_eq!(captured[0].0, Provider::Anthropic);
        assert_eq!(captured[0].2.identity.profile, PROFILE_ID);
        assert_eq!(captured[0].2.identity.build, BUILD_ID);
        assert_eq!(
            captured[0].2.messages,
            vec![
                format!("{CANARY_VENDOR}{CANARY_REPO}"),
                CANARY_USER.to_string()
            ]
        );

        // 2. Raw-wire JSON offset (spanning serialized JSON of system[] instead of Message::content)
        //    fails closed with map_out_of_range and 0 sends.
        let raw_sys = serde_json::to_string(&array_body_json["system"]).unwrap();
        let raw_segs = vec![
            Segment {
                message_index: 0,
                start: 0,
                end: raw_sys.len(),
                source: Source::VendorStatic,
                sha256: Sha256Digest::of(raw_sys.as_bytes()),
            },
            valid_segments[2].clone(),
        ];
        let (env1, tag1_hex) = sign_env(&array_body, raw_segs, 1);
        let req1 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env1).unwrap())
            .body(Body::from(array_body.clone()))
            .unwrap();
        let resp1 = app.clone().oneshot(req1).await.unwrap();
        let json1 = assert_carrier_rejection(resp1, "map_out_of_range").await;

        // 3. Mutated block text fails closed with segment_digest_mismatch and vendor_digest_mismatch.
        let mutated_vendor = "You are a coding agent. Follow protocol!\n";
        let mutated_body = serde_json::to_vec(&json!({
            "model": "claude-3-5-sonnet",
            "system": [
                {"type": "text", "text": mutated_vendor, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": CANARY_REPO, "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ],
            "messages": [{"role": "user", "content": [{"type": "text", "text": CANARY_USER}]}]
        }))
        .unwrap();
        let (env2, tag2_hex) = sign_env(&mutated_body, valid_segments.clone(), 2);
        let req2 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env2).unwrap())
            .body(Body::from(mutated_body.clone()))
            .unwrap();
        let resp2 = app.clone().oneshot(req2).await.unwrap();
        let json2 = assert_carrier_rejection(resp2, "segment_digest_mismatch").await;

        let mut rehashed_segs = valid_segments.clone();
        rehashed_segs[0].sha256 = Sha256Digest::of(mutated_vendor.as_bytes());
        let (env3, tag3_hex) = sign_env(&mutated_body, rehashed_segs, 3);
        let req3 = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header(PROVENANCE_HEADER, HeaderValue::from_bytes(&env3).unwrap())
            .body(Body::from(mutated_body))
            .unwrap();
        let resp3 = app.clone().oneshot(req3).await.unwrap();
        let json3 = assert_carrier_rejection(resp3, "vendor_digest_mismatch").await;
        assert_eq!(next_stage.call_count(), 1);
        assert_eq!(probes.http_calls.load(Ordering::SeqCst), 0);

        let logs = String::from_utf8(trace_capture.buf.lock().unwrap().clone()).unwrap();
        for surface in [
            logs,
            serde_json::to_string(&json1).unwrap(),
            serde_json::to_string(&json2).unwrap(),
            serde_json::to_string(&json3).unwrap(),
        ] {
            for needle in [
                session_key_hex.as_str(),
                CANARY_W4B_KEY_1_STR,
                tag0_hex.as_str(),
                tag1_hex.as_str(),
                tag2_hex.as_str(),
                tag3_hex.as_str(),
                CANARY_VENDOR.trim(),
                CANARY_REPO.trim(),
                CANARY_USER,
            ] {
                assert!(
                    !surface.contains(needle),
                    "W4c2b synthetic-manifest surface leaked {needle:?}: {surface}"
                );
            }
        }
    }
}
