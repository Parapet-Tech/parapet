// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Engine tests

use super::*;
use crate::config::{
    CompiledPattern, ContentPolicy, EngineConfig, LayerConfig, LayerConfigs, PolicyConfig,
    RuntimeConfig, ToolConfig, TrustConfig,
};
use crate::message::{Message, Role, TrustLevel};
use crate::session::SessionStore;
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
    fn scan(&self, _messages: &[Message], _config: &Config) -> crate::layers::l3_inbound::InboundResult {
        crate::layers::l3_inbound::InboundResult {
            verdict: InboundVerdict::Allow,
            matched_patterns: vec![],
        }
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
        l1_scanner: None,
        l2a_scanner: None,
        l2a_semaphore: None,
        inbound_scanner: Arc::new(NoopInboundScanner),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store: None,
        multi_turn_scanner: None,
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
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
                l1: None,
                l2a: None,
                l3_inbound: None,
                l3_outbound: Some(LayerConfig {
                    mode: "block".to_string(),
                    block_action: Some(block_action.to_string()),
                    window_chars: None,
                }),
                l5a: None,
                l4: None,
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

// -------------------------------------------------------------------
// X-Guard-Trust header parsing tests
// -------------------------------------------------------------------

#[test]
fn parse_trust_header_attaches_spans_to_messages() {
    // Build a mock JSON body with a user message
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello untrusted data here"}
        ]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();

    // Find where "untrusted data" appears in the body bytes
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let content_in_body = body_str.find("untrusted data").unwrap();

    // Create the trust header with a span covering "untrusted data"
    let spans = serde_json::json!([
        {"s": content_in_body, "e": content_in_body + "untrusted data".len(), "src": "rag"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());
    let header_value = format!("inline:{encoded}");

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", header_value.parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "Hello untrusted data here")];

    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert!(!messages[0].trust_spans.is_empty());
    assert_eq!(messages[0].trust_spans[0].level, TrustLevel::Untrusted);
    assert_eq!(messages[0].trust_spans[0].source.as_deref(), Some("rag"));
}

#[test]
fn parse_trust_header_missing_header_is_noop() {
    let headers = HeaderMap::new();
    let body = b"{}";
    let mut messages = vec![Message::new(Role::User, "hello")];

    parse_trust_header(&headers, body, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_malformed_base64_warns_gracefully() {
    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", "inline:not-valid-base64!!!".parse().unwrap());
    let body = b"{}";
    let mut messages = vec![Message::new(Role::User, "hello")];

    parse_trust_header(&headers, body, &mut messages);

    // Should not panic, messages unchanged
    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_unknown_format_is_noop() {
    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", "external:http://example.com".parse().unwrap());
    let body = b"{}";
    let mut messages = vec![Message::new(Role::User, "hello")];

    parse_trust_header(&headers, body, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_empty_spans_array_is_noop() {
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(b"[]");
    let header_value = format!("inline:{encoded}");

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", header_value.parse().unwrap());

    let body = b"{}";
    let mut messages = vec![Message::new(Role::User, "hello")];

    parse_trust_header(&headers, body, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_span_without_src_defaults_to_unknown() {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "test data"}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let data_offset = body_str.find("test data").unwrap();

    let spans = serde_json::json!([
        {"s": data_offset, "e": data_offset + "test data".len()}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "test data")];

    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert_eq!(messages[0].trust_spans.len(), 1);
    assert_eq!(messages[0].trust_spans[0].level, TrustLevel::Untrusted);
    assert_eq!(messages[0].trust_spans[0].source.as_deref(), Some("unknown"));
}

// -------------------------------------------------------------------
// V4-01: Trust-span correctness regression tests
// -------------------------------------------------------------------

#[test]
fn parse_trust_header_repeated_content_maps_to_correct_message() {
    // Two messages with identical content. Span targets the second one.
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "same text"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "same text"}
        ]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // Find the SECOND occurrence of "same text" in the body
    let first_pos = body_str.find("same text").unwrap();
    let second_pos = body_str[first_pos + 1..].find("same text").unwrap() + first_pos + 1;

    let spans = serde_json::json!([
        {"s": second_pos, "e": second_pos + "same text".len(), "src": "rag"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![
        Message::new(Role::User, "same text"),
        Message::new(Role::Assistant, "reply"),
        Message::new(Role::User, "same text"),
    ];

    parse_trust_header(&headers, &body_bytes, &mut messages);

    // First "same text" message should have NO spans
    assert!(
        messages[0].trust_spans.is_empty(),
        "first message should not have spans: got {:?}",
        messages[0].trust_spans
    );
    // Second "same text" message (index 2) should have the span
    assert_eq!(messages[2].trust_spans.len(), 1);
    assert_eq!(messages[2].trust_spans[0].start, 0);
    assert_eq!(messages[2].trust_spans[0].end, "same text".len());
    assert_eq!(messages[2].trust_spans[0].source.as_deref(), Some("rag"));
}

#[test]
fn parse_trust_header_out_of_bounds_span_is_not_attached() {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();

    // Span at offset 9999, well beyond any content
    let spans = serde_json::json!([
        {"s": 9999, "e": 10010, "src": "bogus"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "hello")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_inverted_range_is_rejected() {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();

    // s > e → inverted
    let spans = serde_json::json!([
        {"s": 50, "e": 10, "src": "bad"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "hello")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_escaped_quotes_map_to_raw_offsets() {
    // Content with double-quotes: raw is 13 bytes, escaped interior is 15 bytes
    let raw_content = r#"say "hello" ok"#; // 14 bytes: s a y   " h e l l o "   o k
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": raw_content}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // In the serialized body, the content is: say \"hello\" ok (16 escaped bytes)
    // We want to mark "hello" as untrusted.
    // In escaped body: \"hello\" starts at offset 4 from content start
    // In escaped form: \  "  h  e  l  l  o  \  "
    //                  4  5  6  7  8  9  10 11 12
    // So "hello" in escaped space is at escaped offset 6..11 from content start
    let escaped_content = serde_json::to_string(raw_content).unwrap();
    let escaped_interior = &escaped_content[1..escaped_content.len() - 1];
    let content_start = body_str.find(escaped_interior).unwrap();

    // Find "hello" in the escaped interior
    let hello_esc_start = escaped_interior.find("hello").unwrap();
    let hello_esc_end = hello_esc_start + "hello".len();

    let spans = serde_json::json!([
        {"s": content_start + hello_esc_start, "e": content_start + hello_esc_end, "src": "test"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, raw_content)];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert_eq!(messages[0].trust_spans.len(), 1);
    let span = &messages[0].trust_spans[0];
    // "hello" in raw content is at byte offset 5..10
    let slice = &raw_content[span.start..span.end];
    assert_eq!(slice, "hello", "span should cover 'hello' in raw content, got '{slice}'");
}

#[test]
fn parse_trust_header_escaped_newline_maps_to_raw_offset() {
    // Content with a newline: raw "line1\nline2" (11 bytes), escaped "line1\\nline2" (12 bytes)
    let raw_content = "line1\nline2";
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": raw_content}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    let escaped_content = serde_json::to_string(raw_content).unwrap();
    let escaped_interior = &escaped_content[1..escaped_content.len() - 1];
    let content_start = body_str.find(escaped_interior).unwrap();

    // Mark "line2" as untrusted. In escaped interior: "line1\nline2"
    // "line2" starts at escaped offset 7 (l i n e 1 \ n l i n e 2)
    let line2_esc_start = escaped_interior.find("line2").unwrap();
    let line2_esc_end = line2_esc_start + "line2".len();

    let spans = serde_json::json!([
        {"s": content_start + line2_esc_start, "e": content_start + line2_esc_end, "src": "test"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, raw_content)];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert_eq!(messages[0].trust_spans.len(), 1);
    let span = &messages[0].trust_spans[0];
    let slice = &raw_content[span.start..span.end];
    assert_eq!(slice, "line2", "span should cover 'line2' in raw content, got '{slice}'");
}

#[test]
fn parse_trust_header_content_in_non_message_field_not_matched() {
    // "gpt-4" appears in "model" field before "content" field.
    // Span targets the "content" field occurrence — should not bind to "model".
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "gpt-4"}
        ]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // The SDK would find "gpt-4" in the content field. Find it anchored to "content":
    // We need the byte offset of "gpt-4" inside the "content" field, not the "model" field.
    let content_key_pos = body_str.find(r#""content":"gpt-4""#)
        .or_else(|| body_str.find(r#""content": "gpt-4""#))
        .expect("content field should exist");
    let content_value_pos = body_str[content_key_pos..].find("gpt-4").unwrap() + content_key_pos;

    let spans = serde_json::json!([
        {"s": content_value_pos, "e": content_value_pos + "gpt-4".len(), "src": "rag"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "gpt-4")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    assert_eq!(messages[0].trust_spans.len(), 1);
    let span = &messages[0].trust_spans[0];
    assert_eq!(span.start, 0);
    assert_eq!(span.end, "gpt-4".len());
}

#[test]
fn parse_trust_header_non_message_content_key_not_matched() {
    // Exact reproduction from review: "content" key exists outside messages array.
    // {"meta":{"content":"abcd"},"messages":[{"role":"user","content":"abc"}]}
    // find_content_field_value must not match meta.content.
    let body_str = r#"{"meta":{"content":"abcd"},"messages":[{"role":"user","content":"abc"}]}"#;
    let body_bytes = body_str.as_bytes();

    // Span targets "abc" inside the messages content field
    let msg_content_pos = body_str.rfind("\"abc\"").unwrap() + 1; // skip opening quote
    let spans = serde_json::json!([
        {"s": msg_content_pos, "e": msg_content_pos + 3, "src": "test"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "abc")];
    parse_trust_header(&headers, body_bytes, &mut messages);

    // Should map to the message, not the meta field
    assert_eq!(messages[0].trust_spans.len(), 1);
    let span = &messages[0].trust_spans[0];
    assert_eq!(span.start, 0);
    assert_eq!(span.end, 3);
    assert_eq!(&"abc"[span.start..span.end], "abc");
}

#[test]
fn parse_trust_header_meta_content_span_not_attached() {
    // Span points at meta.content value (outside messages array).
    // Should NOT be attached to any message.
    let body_str = r#"{"meta":{"content":"abcd"},"messages":[{"role":"user","content":"hello"}]}"#;
    let body_bytes = body_str.as_bytes();

    let meta_pos = body_str.find("abcd").unwrap();
    let spans = serde_json::json!([
        {"s": meta_pos, "e": meta_pos + 4, "src": "bad"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "hello")];
    parse_trust_header(&headers, body_bytes, &mut messages);

    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_model_field_span_not_attached_to_message() {
    // Span points at "model" field value, NOT the "content" field.
    // Should not be attached to any message.
    let body = serde_json::json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "hello world"}
        ]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // Span targets "gpt-4" in the model field
    let model_pos = body_str.find("gpt-4").unwrap();

    let spans = serde_json::json!([
        {"s": model_pos, "e": model_pos + "gpt-4".len(), "src": "bad"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "hello world")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    // "gpt-4" span is in the model field, not overlapping the content region
    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn parse_trust_header_nested_messages_key_ignored() {
    // A nested "messages" key inside another object must NOT be used
    // as the messages array boundary. Only the top-level "messages" counts.
    let body_str = r#"{"meta":{"messages":[{"content":"trap"}]},"messages":[{"role":"user","content":"real"}]}"#;
    let body_bytes = body_str.as_bytes();

    // Span points at "real" inside the top-level messages array
    let real_pos = body_str.rfind("real").unwrap();

    let spans = serde_json::json!([
        {"s": real_pos, "e": real_pos + "real".len(), "src": "rag"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "real")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    // The span should attach to the message, mapping to content offsets 0..4
    assert_eq!(messages[0].trust_spans.len(), 1);
    assert_eq!(messages[0].trust_spans[0].start, 0);
    assert_eq!(messages[0].trust_spans[0].end, 4);
    assert_eq!(messages[0].trust_spans[0].source.as_deref(), Some("rag"));
}

#[test]
fn parse_trust_header_nested_messages_trap_not_matched() {
    // Span points at "trap" inside a nested (non-top-level) "messages" key.
    // This should NOT be attached to any message.
    let body_str = r#"{"meta":{"messages":[{"content":"trap"}]},"messages":[{"role":"user","content":"real"}]}"#;
    let body_bytes = body_str.as_bytes();

    // Span targets "trap" inside meta.messages
    let trap_pos = body_str.find("trap").unwrap();

    let spans = serde_json::json!([
        {"s": trap_pos, "e": trap_pos + "trap".len(), "src": "bad"}
    ]);
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(serde_json::to_vec(&spans).unwrap());

    let mut headers = HeaderMap::new();
    headers.insert("x-guard-trust", format!("inline:{encoded}").parse().unwrap());

    let mut messages = vec![Message::new(Role::User, "real")];
    parse_trust_header(&headers, &body_bytes, &mut messages);

    // "trap" is in meta.messages, outside the top-level messages array range,
    // so no span should be attached
    assert!(messages[0].trust_spans.is_empty());
}

#[test]
fn escaped_byte_to_raw_surrogate_pair() {
    // \uD83D\uDE00 = 😀 (U+1F600), which is 4 bytes in UTF-8
    // Escaped: 12 bytes (\uD83D\uDE00), raw: 4 bytes
    let escaped = r#"\uD83D\uDE00"#;
    assert_eq!(escaped_byte_to_raw(escaped, 0), 0);
    assert_eq!(escaped_byte_to_raw(escaped, 12), 4); // full pair → 4-byte char
}

#[test]
fn escaped_byte_to_raw_surrogate_pair_with_trailing_text() {
    // \uD83D\uDE00x → 😀x → 5 raw bytes (4 for emoji + 1 for 'x')
    let escaped = r#"\uD83D\uDE00x"#;
    assert_eq!(escaped_byte_to_raw(escaped, 12), 4); // after pair
    assert_eq!(escaped_byte_to_raw(escaped, 13), 5); // after 'x'
}

#[test]
fn escaped_byte_to_raw_bmp_unicode_escape() {
    // \u00E9 = é (U+00E9), which is 2 bytes in UTF-8
    let escaped = r#"caf\u00E9"#;
    assert_eq!(escaped_byte_to_raw(escaped, 3), 3);  // 'caf'
    assert_eq!(escaped_byte_to_raw(escaped, 9), 5);  // 'caf' + é (2 UTF-8 bytes)
}

#[test]
fn escaped_byte_to_raw_no_escapes() {
    assert_eq!(escaped_byte_to_raw("hello world", 0), 0);
    assert_eq!(escaped_byte_to_raw("hello world", 5), 5);
    assert_eq!(escaped_byte_to_raw("hello world", 11), 11);
}

#[test]
fn escaped_byte_to_raw_quote_escapes() {
    // escaped: say \"hi\"   (positions: s=0 a=1 y=2 ' '=3 \=4 "=5 h=6 i=7 \=8 "=9)
    // raw:     say "hi"     (positions: s=0 a=1 y=2 ' '=3 "=4 h=5 i=6 "=7)
    let escaped = r#"say \"hi\""#;
    assert_eq!(escaped_byte_to_raw(escaped, 0), 0); // 's'
    assert_eq!(escaped_byte_to_raw(escaped, 4), 4); // start of \"
    assert_eq!(escaped_byte_to_raw(escaped, 6), 5); // 'h' — after first escape
    assert_eq!(escaped_byte_to_raw(escaped, 8), 7); // start of second \"
    assert_eq!(escaped_byte_to_raw(escaped, 10), 8); // end
}

#[test]
fn escaped_byte_to_raw_newline_escape() {
    // escaped: "a\nb" (positions: a=0 \=1 n=2 b=3)
    // raw:     "a\nb" (positions: a=0 \n=1 b=2)
    let escaped = r#"a\nb"#;
    assert_eq!(escaped_byte_to_raw(escaped, 0), 0);
    assert_eq!(escaped_byte_to_raw(escaped, 1), 1); // start of \n
    assert_eq!(escaped_byte_to_raw(escaped, 3), 2); // 'b'
    assert_eq!(escaped_byte_to_raw(escaped, 4), 3); // end
}

#[test]
fn escaped_byte_to_raw_multiple_escapes() {
    // escaped: \\\"  (4 bytes: backslash-backslash-backslash-quote)
    // raw:     \"    (2 bytes: literal backslash + literal quote)
    let escaped = r#"\\\""#;
    assert_eq!(escaped_byte_to_raw(escaped, 0), 0);
    assert_eq!(escaped_byte_to_raw(escaped, 2), 1); // after first escape
    assert_eq!(escaped_byte_to_raw(escaped, 4), 2); // after second escape
}

// -------------------------------------------------------------------
// Session ID extraction from W3C Baggage header (M8)
// -------------------------------------------------------------------

#[test]
fn extract_session_id_from_baggage_user_id() {
    let mut headers = HeaderMap::new();
    headers.insert("baggage", "user_id=user_123,role=admin".parse().unwrap());
    let id = extract_session_id(&headers);
    assert_eq!(id, "user:user_123");
}

#[test]
fn extract_session_id_percent_encoded() {
    let mut headers = HeaderMap::new();
    headers.insert("baggage", "user_id=user%20with%20spaces".parse().unwrap());
    let id = extract_session_id(&headers);
    assert_eq!(id, "user:user with spaces");
}

#[test]
fn extract_session_id_no_baggage_generates_ephemeral() {
    let headers = HeaderMap::new();
    let id = extract_session_id(&headers);
    assert!(id.starts_with("ephemeral:"));
}

#[test]
fn extract_session_id_baggage_without_user_id_generates_ephemeral() {
    let mut headers = HeaderMap::new();
    headers.insert("baggage", "role=admin".parse().unwrap());
    let id = extract_session_id(&headers);
    assert!(id.starts_with("ephemeral:"));
}

#[test]
fn extract_session_id_empty_user_id_generates_ephemeral() {
    let mut headers = HeaderMap::new();
    headers.insert("baggage", "user_id=".parse().unwrap());
    let id = extract_session_id(&headers);
    assert!(id.starts_with("ephemeral:"));
}

#[test]
fn percent_decode_basic() {
    assert_eq!(percent_decode("hello%20world"), "hello world");
    assert_eq!(percent_decode("no%20encoding%21"), "no encoding!");
    assert_eq!(percent_decode("plain"), "plain");
    assert_eq!(percent_decode(""), "");
}

// -----------------------------------------------------------------------
// M11: Session state update + history compression
// -----------------------------------------------------------------------

#[test]
fn compress_to_history_captures_role_and_summary() {
    let messages = vec![
        Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
            trust: TrustLevel::Trusted,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust_spans: Vec::new(),
        },
        Message {
            role: Role::Assistant,
            content: "Hi there! How can I help?".to_string(),
            trust: TrustLevel::Trusted,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust_spans: Vec::new(),
        },
    ];

    let entries = compress_to_history(&messages);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].role, "user");
    assert_eq!(entries[0].content_summary, "Hello, world!");
    assert_eq!(entries[1].role, "assistant");
    assert_eq!(entries[1].content_summary, "Hi there! How can I help?");
}

#[test]
fn compress_to_history_truncates_at_200_chars() {
    let long_content = "a".repeat(500);
    let messages = vec![Message {
        role: Role::User,
        content: long_content,
        trust: TrustLevel::Trusted,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_name: None,
        trust_spans: Vec::new(),
    }];

    let entries = compress_to_history(&messages);
    assert_eq!(entries[0].content_summary.len(), 200);
}

#[test]
fn compress_to_history_captures_tool_call_names() {
    let messages = vec![Message {
        role: Role::Assistant,
        content: String::new(),
        trust: TrustLevel::Trusted,
        tool_calls: vec![
            crate::message::ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({}),
            },
            crate::message::ToolCall {
                id: "call_2".to_string(),
                name: "exec_command".to_string(),
                arguments: serde_json::json!({}),
            },
        ],
        tool_call_id: None,
        tool_name: None,
        trust_spans: Vec::new(),
    }];

    let entries = compress_to_history(&messages);
    assert_eq!(entries[0].tool_calls_summary, vec!["read_file", "exec_command"]);
}

#[test]
fn update_session_after_request_creates_session_if_missing() {
    let store = crate::session::InMemorySessionStore::new(std::time::Duration::from_secs(3600));
    let messages = vec![Message {
        role: Role::User,
        content: "hello".to_string(),
        trust: TrustLevel::Trusted,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_name: None,
        trust_spans: Vec::new(),
    }];

    assert!(store.get("new_session").is_none());
    update_session_after_request(&store, "new_session", &messages, 50);

    let session = store.get("new_session").unwrap();
    assert_eq!(session.turn_count, 1);
    assert_eq!(session.history.len(), 1);
    assert_eq!(session.history[0].role, "user");
}

#[test]
fn update_session_after_request_appends_to_existing() {
    let store = crate::session::InMemorySessionStore::new(std::time::Duration::from_secs(3600));
    let msg1 = vec![Message {
        role: Role::User,
        content: "turn 1".to_string(),
        trust: TrustLevel::Trusted,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_name: None,
        trust_spans: Vec::new(),
    }];
    update_session_after_request(&store, "sess", &msg1, 50);

    let msg2 = vec![Message {
        role: Role::Assistant,
        content: "turn 2".to_string(),
        trust: TrustLevel::Trusted,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_name: None,
        trust_spans: Vec::new(),
    }];
    update_session_after_request(&store, "sess", &msg2, 50);

    let session = store.get("sess").unwrap();
    assert_eq!(session.turn_count, 2);
    assert_eq!(session.history.len(), 2);
}

#[test]
fn update_session_respects_max_history() {
    let store = crate::session::InMemorySessionStore::new(std::time::Duration::from_secs(3600));
    for i in 0..10 {
        let msg = vec![Message {
            role: Role::User,
            content: format!("turn {i}"),
            trust: TrustLevel::Trusted,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust_spans: Vec::new(),
        }];
        update_session_after_request(&store, "sess", &msg, 5);
    }

    let session = store.get("sess").unwrap();
    assert_eq!(session.turn_count, 10);
    assert_eq!(session.history.len(), 5);
    // Oldest remaining should be turn 5
    assert_eq!(session.history[0].content_summary, "turn 5");
}

#[test]
fn compress_to_history_preserves_trust_level() {
    let messages = vec![
        Message {
            role: Role::User,
            content: "trusted".to_string(),
            trust: TrustLevel::Trusted,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust_spans: Vec::new(),
        },
        Message {
            role: Role::User,
            content: "untrusted".to_string(),
            trust: TrustLevel::Untrusted,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            trust_spans: Vec::new(),
        },
    ];

    let entries = compress_to_history(&messages);
    assert_eq!(entries[0].trust_level, TrustLevel::Trusted);
    assert_eq!(entries[1].trust_level, TrustLevel::Untrusted);
}

// -------------------------------------------------------------------
// StartupError tests
// -------------------------------------------------------------------

#[test]
fn startup_error_display_feature_disabled() {
    let e = StartupError::FeatureDisabled("missing l2a feature".into());
    assert_eq!(e.to_string(), "missing l2a feature");
}

#[test]
fn startup_error_display_model_missing() {
    let e = StartupError::ModelMissing("model not found".into());
    assert_eq!(e.to_string(), "model not found");
}

#[test]
fn startup_error_is_error_trait() {
    let e: Box<dyn std::error::Error> =
        Box::new(StartupError::ModelMissing("test".into()));
    assert!(e.to_string().contains("test"));
}

// -------------------------------------------------------------------
// Feature-gate test (we build without `l2a` by default)
// -------------------------------------------------------------------

#[cfg(not(feature = "l2a"))]
#[test]
fn build_engine_client_rejects_l2a_config_without_feature() {
    use crate::config::{L2aConfig, L2aMode};

    let mut config = base_config_with_canary("dummy");
    config.policy.layers.l2a = Some(L2aConfig {
        mode: L2aMode::Shadow,
        model: "pg2-86m".to_string(),
        model_dir: None,
        pg_threshold: 0.5,
        block_threshold: 0.7,
        heuristic_weight: 0.3,
        fusion_confidence_agreement: 0.9,
        fusion_confidence_pg_only: 0.7,
        fusion_confidence_heuristic_only: 0.4,
        max_segments: 16,
        timeout_ms: 200,
        max_concurrent_scans: 4,
    });
    let result = build_engine_client(Arc::new(config));
    match result {
        Err(StartupError::FeatureDisabled(msg)) => {
            assert!(msg.contains("--features l2a"));
        }
        Err(_) => panic!("expected FeatureDisabled, got different error"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

// -------------------------------------------------------------------
// L2a pipeline integration (mock scanner)
// -------------------------------------------------------------------

use crate::layers::l2a::{L2aScanError, L2aScanner};
use crate::config::{L2aConfig, L2aMode};
use crate::signal::{LayerId, Signal, SignalKind};

struct MockL2aScanner {
    signals: Vec<Signal>,
}

impl L2aScanner for MockL2aScanner {
    fn scan(
        &self,
        _messages: &[crate::message::Message],
        _config: &L2aConfig,
    ) -> Result<Vec<Signal>, L2aScanError> {
        Ok(self.signals.clone())
    }
}

struct FailingL2aScanner;

impl L2aScanner for FailingL2aScanner {
    fn scan(
        &self,
        _messages: &[crate::message::Message],
        _config: &L2aConfig,
    ) -> Result<Vec<Signal>, L2aScanError> {
        Err(L2aScanError::ClassifyFailed("mock classify failure".into()))
    }
}

/// Scanner that sleeps longer than the configured timeout.
struct SlowL2aScanner {
    delay: std::time::Duration,
}

impl L2aScanner for SlowL2aScanner {
    fn scan(
        &self,
        _messages: &[crate::message::Message],
        _config: &L2aConfig,
    ) -> Result<Vec<Signal>, L2aScanError> {
        std::thread::sleep(self.delay);
        Ok(vec![])
    }
}

/// Scanner that panics (simulates ONNX runtime failure).
struct PanickingL2aScanner;

impl L2aScanner for PanickingL2aScanner {
    fn scan(
        &self,
        _messages: &[crate::message::Message],
        _config: &L2aConfig,
    ) -> Result<Vec<Signal>, L2aScanError> {
        panic!("onnx session exploded");
    }
}

/// HTTP sender that returns a valid OpenAI-style 200 response.
struct SuccessHttpSender;

#[async_trait]
impl HttpSender for SuccessHttpSender {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }]
        });
        Ok(HttpResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            body: HttpBody::Full(Bytes::from(serde_json::to_vec(&body).unwrap())),
        })
    }
}

fn l2a_test_config() -> L2aConfig {
    L2aConfig {
        mode: L2aMode::Shadow,
        model: "pg2-86m".to_string(),
        model_dir: None,
        pg_threshold: 0.5,
        block_threshold: 0.7,
        heuristic_weight: 0.3,
        fusion_confidence_agreement: 0.9,
        fusion_confidence_pg_only: 0.7,
        fusion_confidence_heuristic_only: 0.4,
        max_segments: 16,
        timeout_ms: 200,
        max_concurrent_scans: 4,
    }
}

fn l2a_signal(score: f32, category: &str) -> Signal {
    Signal {
        layer: LayerId::L2a,
        kind: SignalKind::Evidence,
        score,
        confidence: 0.9,
        category: Some(category.to_string()),
        message_index: Some(0),
        segment_id: None,
    }
}

fn engine_with_l2a_scanner(
    config: Config,
    scanner: Arc<dyn L2aScanner>,
    semaphore_permits: usize,
) -> EngineUpstreamClient {
    let deps = EngineDeps {
        config: Arc::new(config),
        http: Arc::new(SuccessHttpSender),
        resolver: Arc::new(NoopResolver),
        registry: Arc::new(DefaultRegistry),
        normalizer: Arc::new(NoopNormalizer),
        trust_assigner: Arc::new(NoopTrustAssigner),
        l1_scanner: None,
        l2a_scanner: Some(scanner),
        l2a_semaphore: Some(Arc::new(tokio::sync::Semaphore::new(semaphore_permits))),
        inbound_scanner: Arc::new(NoopInboundScanner),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store: None,
        multi_turn_scanner: None,
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
    };
    EngineUpstreamClient::new_with(deps)
}

fn config_with_l2a(mode: L2aMode) -> Config {
    let mut config = base_config_with_canary("dummy");
    config.policy.layers.l2a = Some(L2aConfig {
        mode,
        ..l2a_test_config()
    });
    config
}

fn openai_request() -> ProxyRequest {
    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });
    ProxyRequest {
        method: Method::POST,
        uri: Uri::from_static("/v1/chat/completions"),
        headers: HeaderMap::new(),
        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
    }
}

fn blocked_by(resp: &ProxyResponse) -> Option<String> {
    resp.headers
        .get("x-parapet-blocked-by")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

#[test]
fn l2a_fail_closed_block_sets_pending() {
    use crate::provider::adapter_for;

    let adapter = adapter_for(ProviderType::OpenAi);
    let mut pending: Option<ProxyResponse> = None;
    l2a_fail_closed_block(&mut pending, adapter.as_ref(), "test");
    assert!(pending.is_some());
    let resp = pending.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
}

#[test]
fn l2a_fail_closed_block_first_block_wins() {
    use crate::provider::adapter_for;

    let adapter = adapter_for(ProviderType::OpenAi);
    // Pre-set a block from another layer.
    let existing = ProxyResponse::blocked(
        adapter.serialize_error("blocked by L1", 403),
        "L1",
    );
    let mut pending = Some(existing);
    l2a_fail_closed_block(&mut pending, adapter.as_ref(), "test");
    // Should not overwrite.
    let resp = pending.unwrap();
    assert_eq!(blocked_by(&resp).as_deref(), Some("L1"));
}

// -------------------------------------------------------------------
// L2a runtime-path tests via forward()
// -------------------------------------------------------------------

#[tokio::test]
async fn l2a_shadow_scan_success_does_not_block() {
    let scanner = Arc::new(MockL2aScanner {
        signals: vec![l2a_signal(0.9, "semantic_injection")],
    });
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Shadow), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Shadow mode: signals above block_threshold logged but not blocked.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_block_scan_above_threshold_blocks() {
    let scanner = Arc::new(MockL2aScanner {
        signals: vec![l2a_signal(0.9, "semantic_injection")],
    });
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Block), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_by(&resp).as_deref(), Some("L2a"));
}

#[tokio::test]
async fn l2a_block_scan_below_threshold_allows() {
    let scanner = Arc::new(MockL2aScanner {
        signals: vec![l2a_signal(0.3, "semantic_injection")],
    });
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Block), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Score 0.3 < block_threshold 0.7 → allow.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_shadow_scanner_error_fails_open() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(FailingL2aScanner);
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Shadow), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Shadow: fail-open — request proceeds.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_block_scanner_error_fails_closed() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(FailingL2aScanner);
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Block), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_by(&resp).as_deref(), Some("L2a"));
}

#[tokio::test]
async fn l2a_shadow_timeout_fails_open() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(SlowL2aScanner {
        delay: std::time::Duration::from_secs(5),
    });
    let mut cfg = config_with_l2a(L2aMode::Shadow);
    cfg.policy.layers.l2a.as_mut().unwrap().timeout_ms = 10; // 10ms timeout
    let engine = engine_with_l2a_scanner(cfg, scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Shadow: fail-open on timeout.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_block_timeout_fails_closed() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(SlowL2aScanner {
        delay: std::time::Duration::from_secs(5),
    });
    let mut cfg = config_with_l2a(L2aMode::Block);
    cfg.policy.layers.l2a.as_mut().unwrap().timeout_ms = 10; // 10ms timeout
    let engine = engine_with_l2a_scanner(cfg, scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_by(&resp).as_deref(), Some("L2a"));
}

#[tokio::test]
async fn l2a_shadow_panic_fails_open() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(PanickingL2aScanner);
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Shadow), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Shadow: fail-open on panic.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_block_panic_fails_closed() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(PanickingL2aScanner);
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Block), scanner, 4);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_by(&resp).as_deref(), Some("L2a"));
}

#[tokio::test]
async fn l2a_shadow_concurrency_limit_fails_open() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(MockL2aScanner { signals: vec![] });
    // 0 permits = immediately exhausted.
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Shadow), scanner, 0);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    // Shadow: fail-open when semaphore exhausted.
    assert_eq!(resp.status, StatusCode::OK);
}

#[tokio::test]
async fn l2a_block_concurrency_limit_fails_closed() {
    let scanner: Arc<dyn L2aScanner> = Arc::new(MockL2aScanner { signals: vec![] });
    // 0 permits = immediately exhausted.
    let engine = engine_with_l2a_scanner(config_with_l2a(L2aMode::Block), scanner, 0);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_by(&resp).as_deref(), Some("L2a"));
}

#[tokio::test]
async fn l2a_no_scanner_configured_passes_through() {
    // L2a config is None, scanner is None → zero overhead path.
    let config = base_config_with_canary("dummy");
    let deps = EngineDeps {
        config: Arc::new(config),
        http: Arc::new(SuccessHttpSender),
        resolver: Arc::new(NoopResolver),
        registry: Arc::new(DefaultRegistry),
        normalizer: Arc::new(NoopNormalizer),
        trust_assigner: Arc::new(NoopTrustAssigner),
        l1_scanner: None,
        l2a_scanner: None,
        l2a_semaphore: None,
        inbound_scanner: Arc::new(NoopInboundScanner),
        constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
        output_scanner: Arc::new(L5aScanner),
        session_store: None,
        multi_turn_scanner: None,
        signal_extractor: Arc::new(DefaultSignalExtractor::new()),
        verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
    };
    let engine = EngineUpstreamClient::new_with(deps);
    let resp = engine.forward(Provider::OpenAi, openai_request()).await.unwrap();
    assert_eq!(resp.status, StatusCode::OK);
}
