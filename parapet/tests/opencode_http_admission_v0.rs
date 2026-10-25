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
    Provider, ProxyError, ProxyRequest, ProxyResponse, SessionRequestAdmitter, UpstreamClient,
    VerifiedCarrierNextStage,
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

thread_local! {
    static THREAD_TRACE_BUF: std::cell::RefCell<Option<Arc<Mutex<Vec<u8>>>>> =
        const { std::cell::RefCell::new(None) };
}

#[derive(Clone, Default)]
struct GlobalTraceWriter;

impl Write for GlobalTraceWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        THREAD_TRACE_BUF.with(|slot| {
            if let Some(buf) = slot.borrow().as_ref() {
                buf.lock().unwrap().extend_from_slice(data);
            }
        });
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
        THREAD_TRACE_BUF.with(|slot| {
            *slot.borrow_mut() = Some(buf.clone());
        });
        Self { buf }
    }
}

impl Drop for ThreadTraceGuard {
    fn drop(&mut self) {
        THREAD_TRACE_BUF.with(|slot| {
            *slot.borrow_mut() = None;
        });
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
    let admitter = Arc::new(SessionRequestAdmitter::new(
        ProfileMode::RequiredOpenCode,
        sessions.clone(),
        PROFILE_ID,
        SCHEMA_VERSION_V1,
    ));
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
        .start(SessionConfig {
            profile: SupportedProfile {
                schema_version: SCHEMA_VERSION_V1,
                profile_id: PROFILE_ID.to_string(),
                builds,
                max_validity_ms: MAX_VALIDITY_MS,
            },
            clock: Box::new(FixedClock(NOW_MS)),
            decoder: Box::new(ProviderContentDecoder::new(ProviderType::OpenAi)),
        })
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
