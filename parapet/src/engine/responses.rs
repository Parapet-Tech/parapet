// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Output-policy seams for OpenAI Responses bodies and collected streams.
//!
//! Exact non-streaming Responses routes use this closed, structured path
//! instead of reusing the Chat Completions handler, whose legacy parse failures
//! pass the upstream body through unchanged.
//!
//! The streaming bridge (D-0027 OA-R3) is called only by the R4b1 stream
//! executor, which the engine's ordinary Responses stream route drives. It
//! judges one completed [`CollectedStream`] as a whole and either
//! releases its exact canonical SSE bytes or refuses with a fixed decision.
//! It never rewrites, re-serializes, or partially releases a stream.

use crate::config::{Config, FailureMode};
use crate::constraint::{ConstraintEvaluator, ToolCallVerdict};
use crate::layers::l5a::{OutputScanner, ScanResult};
use crate::provider::responses_output::{
    DecodedResponse, OpenAiResponsesOutputDecoder, ResponseDecoder,
};
use crate::provider::responses_sse::CollectedStream;
use crate::provider::responses_stream_state::Terminal;
use axum::http::StatusCode;
use bytes::Bytes;
use std::fmt;

/// Content-free result of applying output policy to a Responses body.
///
/// Refusal variants intentionally carry no upstream bytes, decoder messages,
/// tool names, arguments, or scanner findings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum ResponsesOutputDecision {
    Allowed(Vec<u8>),
    InvalidUpstream,
    BlockedByL3Outbound,
    BlockedByL5aToolArguments,
    LayerFailure,
}

/// Content-free result of applying output policy to a collected stream.
///
/// `Release` holds the collector's canonical SSE bytes, unchanged; every
/// other variant suppresses the whole stream. `Debug` shows only the byte
/// count, so neither text nor opaque reasoning ciphertext can leak through it.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum ResponsesStreamDecision {
    Release(Bytes),
    BlockedByL3Outbound,
    BlockedByL5a,
    LayerFailure,
}

impl fmt::Debug for ResponsesStreamDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Release(bytes) => f
                .debug_struct("Release")
                .field("bytes", &bytes.len())
                .finish(),
            Self::BlockedByL3Outbound => f.write_str("BlockedByL3Outbound"),
            Self::BlockedByL5a => f.write_str("BlockedByL5a"),
            Self::LayerFailure => f.write_str("LayerFailure"),
        }
    }
}

/// Why a shared policy check refused. Each path maps it to its own decision.
enum PolicyRefusal {
    L3Outbound,
    L5a,
    LayerFailure,
}

/// Validate and enforce a complete non-streaming Responses body.
///
/// Only `200 OK` with a body accepted by the strict Responses decoder can be
/// returned. L3 evaluates every complete function call. L5a rewrites each
/// structured text segment independently; tool arguments are scanned as a
/// canonical JSON value and cause whole-response suppression on a finding,
/// because mutating a serialized tool call would change its semantics.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn enforce_non_streaming_responses(
    upstream_status: StatusCode,
    body: &[u8],
    config: &Config,
    evaluator: &dyn ConstraintEvaluator,
    scanner: &dyn OutputScanner,
) -> ResponsesOutputDecision {
    if upstream_status != StatusCode::OK {
        return ResponsesOutputDecision::InvalidUpstream;
    }

    let decoded = match OpenAiResponsesOutputDecoder::new().decode_response(body) {
        Ok(decoded) => decoded,
        Err(_) => return ResponsesOutputDecision::InvalidUpstream,
    };

    match check_l3(&decoded, config, evaluator) {
        Ok(()) => {}
        Err(PolicyRefusal::L3Outbound) => return ResponsesOutputDecision::BlockedByL3Outbound,
        Err(_) => return ResponsesOutputDecision::LayerFailure,
    }

    if !l5a_enabled(config) {
        return ResponsesOutputDecision::Allowed(decoded.to_json_bytes());
    }

    let on_panic = &config.runtime.engine.on_failure;
    match check_tool_calls_l5a(&decoded, config, scanner, on_panic) {
        Ok(()) => {}
        Err(PolicyRefusal::L5a) => return ResponsesOutputDecision::BlockedByL5aToolArguments,
        Err(_) => return ResponsesOutputDecision::LayerFailure,
    }

    let mut replacements = Vec::with_capacity(decoded.texts().len());
    for segment in decoded.texts() {
        match scan(scanner, segment.text(), config, on_panic) {
            Ok(Some(result)) => {
                if result.redactions.is_empty() && result.content != segment.text() {
                    return ResponsesOutputDecision::LayerFailure;
                }
                replacements.push(result.content);
            }
            Ok(None) => replacements.push(segment.text().to_string()),
            Err(()) => return ResponsesOutputDecision::LayerFailure,
        }
    }

    match decoded.with_replaced_texts(&replacements) {
        Ok(rewritten) => ResponsesOutputDecision::Allowed(rewritten.to_json_bytes()),
        Err(_) => ResponsesOutputDecision::LayerFailure,
    }
}

/// Enforce output policy on one collected `completed` or `incomplete` stream.
///
/// Policy runs on the assembled terminal response, which the assembler has
/// already bound to every streamed delta and done event. Only a wholly clean
/// verdict releases [`CollectedStream::canonical`], byte for byte. Any L3
/// block, L5a finding (redaction included), scanner contract drift,
/// evaluator cardinality mismatch, or scanner panic suppresses the entire
/// stream: a stream cannot be redacted in place. Scanner panics fail closed
/// regardless of [`FailureMode`], because nothing safe can be released.
///
/// L5a scans, in order: every text segment in wire order, one exact
/// concatenation of those segments (so a finding split across adjacent
/// segments is caught), then each tool name, canonical arguments, and every
/// argument key and string leaf. Opaque reasoning `encrypted_content` is
/// never a scanner input: the decoded response withholds it. Request-echo
/// fields in lifecycle snapshots are client-origin bytes, not model output;
/// they leave only inside an unchanged canonical release.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn enforce_streaming_responses(
    collected: CollectedStream,
    config: &Config,
    evaluator: &dyn ConstraintEvaluator,
    scanner: &dyn OutputScanner,
) -> ResponsesStreamDecision {
    let (terminal, canonical) = collected.into_parts();
    let (Terminal::Completed(stream) | Terminal::Incomplete(stream)) = &terminal else {
        // The collector never finishes with these; refuse rather than trust it.
        return ResponsesStreamDecision::LayerFailure;
    };
    match check_stream(stream.response(), config, evaluator, scanner) {
        Ok(()) => ResponsesStreamDecision::Release(canonical),
        Err(PolicyRefusal::L3Outbound) => ResponsesStreamDecision::BlockedByL3Outbound,
        Err(PolicyRefusal::L5a) => ResponsesStreamDecision::BlockedByL5a,
        Err(PolicyRefusal::LayerFailure) => ResponsesStreamDecision::LayerFailure,
    }
}

fn check_stream(
    response: &DecodedResponse,
    config: &Config,
    evaluator: &dyn ConstraintEvaluator,
    scanner: &dyn OutputScanner,
) -> Result<(), PolicyRefusal> {
    check_l3(response, config, evaluator)?;
    if !l5a_enabled(config) {
        return Ok(());
    }
    let mut aggregate = String::new();
    for segment in response.texts() {
        detect(scanner, segment.text(), config, &FailureMode::Closed)?;
        aggregate.push_str(segment.text());
    }
    detect(scanner, &aggregate, config, &FailureMode::Closed)?;
    check_tool_calls_l5a(response, config, scanner, &FailureMode::Closed)
}

/// L3 over every complete function call. A verdict count that differs from
/// the call count is a layer failure.
fn check_l3(
    response: &DecodedResponse,
    config: &Config,
    evaluator: &dyn ConstraintEvaluator,
) -> Result<(), PolicyRefusal> {
    let l3_enabled = config
        .policy
        .layers
        .l3_outbound
        .as_ref()
        .is_some_and(|layer| layer.mode == "block");
    if !l3_enabled {
        return Ok(());
    }
    let verdicts = evaluator.evaluate_tool_calls(response.tool_calls(), config);
    if verdicts.len() != response.tool_calls().len() {
        return Err(PolicyRefusal::LayerFailure);
    }
    if verdicts
        .iter()
        .any(|verdict| matches!(verdict, ToolCallVerdict::Block { .. }))
    {
        return Err(PolicyRefusal::L3Outbound);
    }
    Ok(())
}

fn l5a_enabled(config: &Config) -> bool {
    config
        .policy
        .layers
        .l5a
        .as_ref()
        .is_some_and(|layer| layer.mode == "redact")
}

/// Arguments are complete, strictly parsed objects here. Keep the canonical
/// JSON scan for patterns that span structure, and also scan the tool name
/// plus every object key and string leaf. Canonical JSON escaping must not
/// hide quote- or whitespace-separated secrets from L5a. Never splice scanner
/// output back into an executable tool call: any finding refuses.
fn check_tool_calls_l5a(
    response: &DecodedResponse,
    config: &Config,
    scanner: &dyn OutputScanner,
    on_panic: &FailureMode,
) -> Result<(), PolicyRefusal> {
    for tool_call in response.tool_calls() {
        let arguments = serde_json::to_string(&tool_call.arguments)
            .expect("serializing a serde_json::Value cannot fail");
        for content in [&*tool_call.name, &*arguments] {
            detect(scanner, content, config, on_panic)?;
        }
        match json_strings_detected(&tool_call.arguments, scanner, config, on_panic) {
            Ok(true) => return Err(PolicyRefusal::L5a),
            Ok(false) => {}
            Err(()) => return Err(PolicyRefusal::LayerFailure),
        }
    }
    Ok(())
}

/// `Ok(None)` means an open-mode scanner panic and therefore preserves the
/// validated original field. Closed mode maps the same panic to a fixed
/// content-free layer failure.
#[cfg_attr(not(test), allow(dead_code))]
fn scan(
    scanner: &dyn OutputScanner,
    content: &str,
    config: &Config,
    on_panic: &FailureMode,
) -> Result<Option<ScanResult>, ()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scanner.scan_and_redact(content, config)
    })) {
        Ok(result) => Ok(Some(result)),
        Err(_) if *on_panic == FailureMode::Open => Ok(None),
        Err(_) => Err(()),
    }
}

fn scan_detects(
    scanner: &dyn OutputScanner,
    content: &str,
    config: &Config,
    on_panic: &FailureMode,
) -> Result<bool, ()> {
    match scan(scanner, content, config, on_panic)? {
        Some(result) if result.redactions.is_empty() && result.content != content => Err(()),
        Some(result) => Ok(!result.redactions.is_empty()),
        None => Ok(false),
    }
}

fn detect(
    scanner: &dyn OutputScanner,
    content: &str,
    config: &Config,
    on_panic: &FailureMode,
) -> Result<(), PolicyRefusal> {
    match scan_detects(scanner, content, config, on_panic) {
        Ok(false) => Ok(()),
        Ok(true) => Err(PolicyRefusal::L5a),
        Err(()) => Err(PolicyRefusal::LayerFailure),
    }
}

fn json_strings_detected(
    value: &serde_json::Value,
    scanner: &dyn OutputScanner,
    config: &Config,
    on_panic: &FailureMode,
) -> Result<bool, ()> {
    match value {
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                if scan_detects(scanner, key, config, on_panic)?
                    || json_strings_detected(value, scanner, config, on_panic)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        serde_json::Value::Array(values) => {
            for value in values {
                if json_strings_detected(value, scanner, config, on_panic)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        serde_json::Value::String(text) => scan_detects(scanner, text, config, on_panic),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ContentPolicy, EngineConfig, LayerConfig, LayerConfigs, PolicyConfig, RuntimeConfig,
        ToolConfig, TrustConfig,
    };
    use crate::constraint::DslConstraintEvaluator;
    use crate::layers::l5a::L5aScanner;
    use crate::message::TrustLevel;
    use serde_json::Value;
    use std::collections::HashMap;

    const MARKER: &str = "RESPONSES_SECRET_7f3a";

    fn config(l3: bool, l5a: bool, failure: FailureMode) -> Config {
        let mut tools = HashMap::new();
        tools.insert(
            "save_note".to_string(),
            ToolConfig {
                allowed: true,
                trust: Some(TrustLevel::Untrusted),
                constraints: HashMap::new(),
                result_policy: None,
            },
        );
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools,
                block_patterns: Vec::new(),
                canary_tokens: vec![MARKER.to_string()],
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs {
                    l3_outbound: l3.then(|| LayerConfig {
                        mode: "block".to_string(),
                        block_action: Some("rewrite".to_string()),
                        window_chars: None,
                    }),
                    l5a: l5a.then(|| LayerConfig {
                        mode: "redact".to_string(),
                        block_action: None,
                        window_chars: None,
                    }),
                    ..LayerConfigs::default()
                },
            },
            runtime: RuntimeConfig {
                engine: EngineConfig {
                    on_failure: failure,
                    timeout_ms: None,
                },
                environment: "test".to_string(),
                responses_stream: crate::config::ResponsesStreamConfig::default(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    fn config_with_default_sensitive_patterns() -> Config {
        let mut config = config(false, true, FailureMode::Open);
        config.policy.canary_tokens.clear();
        config.policy.sensitive_patterns = crate::config::default_sensitive_patterns();
        config
    }

    fn response(text_marker: bool, argument_marker: bool) -> Vec<u8> {
        let text = if text_marker { MARKER } else { "safe" };
        let argument = if argument_marker { MARKER } else { "note.txt" };
        serde_json::to_vec(&serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "error": null,
            "output": [
                {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                 "content": [
                    {"type": "output_text", "text": text, "annotations": [], "logprobs": []},
                    {"type": "refusal", "refusal": text}
                 ]},
                {"type": "reasoning", "id": "rs_1", "status": "completed",
                 "summary": [{"type": "summary_text", "text": text}],
                 "content": [{"type": "reasoning_text", "text": text}],
                 "encrypted_content": null},
                {"type": "function_call", "id": "fc_1", "status": "completed",
                 "call_id": "call_1", "name": "save_note",
                 "arguments": serde_json::to_string(&serde_json::json!({"path": argument})).unwrap()}
            ]
        }))
        .unwrap()
    }

    fn response_with_arguments(arguments: Value) -> Vec<u8> {
        let mut response: Value = serde_json::from_slice(&response(false, false)).unwrap();
        response["output"][2]["arguments"] =
            Value::String(serde_json::to_string(&arguments).unwrap());
        serde_json::to_vec(&response).unwrap()
    }

    struct PanicScanner;
    impl OutputScanner for PanicScanner {
        fn scan_and_redact(&self, _: &str, _: &Config) -> ScanResult {
            panic!("{MARKER}")
        }
    }

    struct MutatingScanner;
    impl OutputScanner for MutatingScanner {
        fn scan_and_redact(&self, _: &str, _: &Config) -> ScanResult {
            ScanResult {
                content: "changed without evidence".to_string(),
                redactions: Vec::new(),
            }
        }
    }

    struct BlockAll;
    impl ConstraintEvaluator for BlockAll {
        fn evaluate_tool_calls(
            &self,
            calls: &[crate::message::ToolCall],
            _: &Config,
        ) -> Vec<ToolCallVerdict> {
            calls
                .iter()
                .map(|call| ToolCallVerdict::Block {
                    tool_name: call.name.clone(),
                    reason: MARKER.to_string(),
                })
                .collect()
        }
    }

    #[test]
    fn non_success_and_undecodable_bodies_fail_closed_without_echo() {
        let config = config(false, false, FailureMode::Open);
        let evaluator = DslConstraintEvaluator::new();
        let scanner = L5aScanner;
        for (status, body) in [
            (StatusCode::UNAUTHORIZED, MARKER.as_bytes()),
            (StatusCode::OK, MARKER.as_bytes()),
            (
                StatusCode::OK,
                br#"{"object":"response","output":[],"output":[]}"#,
            ),
            (
                StatusCode::OK,
                br#"{"object":"response","output":[{"type":"web_search_call"}]}"#,
            ),
        ] {
            let result =
                enforce_non_streaming_responses(status, body, &config, &evaluator, &scanner);
            assert_eq!(result, ResponsesOutputDecision::InvalidUpstream);
            assert!(!format!("{result:?}").contains(MARKER));
        }
    }

    #[test]
    fn l3_blocks_complete_tool_calls_without_exposing_evaluator_reason() {
        let result = enforce_non_streaming_responses(
            StatusCode::OK,
            &response(false, false),
            &config(true, false, FailureMode::Open),
            &BlockAll,
            &L5aScanner,
        );
        assert_eq!(result, ResponsesOutputDecision::BlockedByL3Outbound);
        assert!(!format!("{result:?}").contains(MARKER));
    }

    #[test]
    fn l5a_redacts_every_structured_text_kind_and_keeps_valid_json() {
        let result = enforce_non_streaming_responses(
            StatusCode::OK,
            &response(true, false),
            &config(false, true, FailureMode::Open),
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        );
        let ResponsesOutputDecision::Allowed(body) = result else {
            panic!("expected allowed response")
        };
        assert!(!String::from_utf8_lossy(&body).contains(MARKER));
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["output"][0]["content"][0]["text"], "[REDACTED]");
        assert_eq!(value["output"][0]["content"][1]["refusal"], "[REDACTED]");
        assert_eq!(value["output"][1]["summary"][0]["text"], "[REDACTED]");
        assert_eq!(value["output"][1]["content"][0]["text"], "[REDACTED]");
    }

    #[test]
    fn l5a_suppresses_tool_arguments_instead_of_mutating_executable_json() {
        let result = enforce_non_streaming_responses(
            StatusCode::OK,
            &response(false, true),
            &config(false, true, FailureMode::Open),
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        );
        assert_eq!(result, ResponsesOutputDecision::BlockedByL5aToolArguments);
        assert!(!format!("{result:?}").contains(MARKER));
    }

    #[test]
    fn l5a_scans_tool_argument_keys_and_nested_string_leaves_before_json_escaping() {
        let hex64 = "a".repeat(64);
        let aws40 = "A".repeat(40);
        let bearer = "B".repeat(24);
        let cases = [
            serde_json::json!({"value": format!("password=\"{hex64}")}),
            serde_json::json!({"value": format!("password:\n{hex64}")}),
            serde_json::json!({"value": format!("Bearer\t{bearer}")}),
            serde_json::json!({"value": format!("aws_secret_access_key=\"{aws40}")}),
            serde_json::json!({format!("password=\"{hex64}"): "value"}),
            serde_json::json!({"nested": ["safe", {"value": format!("token:\n{hex64}")}]}),
        ];

        for arguments in cases {
            let body = response_with_arguments(arguments);
            let result = enforce_non_streaming_responses(
                StatusCode::OK,
                &body,
                &config_with_default_sensitive_patterns(),
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            );
            assert_eq!(result, ResponsesOutputDecision::BlockedByL5aToolArguments);
            assert!(!format!("{result:?}").contains(&hex64));
        }
    }

    #[test]
    fn scanner_panics_follow_failure_mode_and_contract_drift_fails_closed() {
        let body = response(false, false);
        let evaluator = DslConstraintEvaluator::new();
        assert!(matches!(
            enforce_non_streaming_responses(
                StatusCode::OK,
                &body,
                &config(false, true, FailureMode::Open),
                &evaluator,
                &PanicScanner,
            ),
            ResponsesOutputDecision::Allowed(_)
        ));
        assert_eq!(
            enforce_non_streaming_responses(
                StatusCode::OK,
                &body,
                &config(false, true, FailureMode::Closed),
                &evaluator,
                &PanicScanner,
            ),
            ResponsesOutputDecision::LayerFailure
        );
        assert_eq!(
            enforce_non_streaming_responses(
                StatusCode::OK,
                &body,
                &config(false, true, FailureMode::Open),
                &evaluator,
                &MutatingScanner,
            ),
            ResponsesOutputDecision::LayerFailure
        );
    }

    #[test]
    fn l3_evaluator_cardinality_drift_fails_closed() {
        struct EmptyEvaluator;
        impl ConstraintEvaluator for EmptyEvaluator {
            fn evaluate_tool_calls(
                &self,
                _: &[crate::message::ToolCall],
                _: &Config,
            ) -> Vec<ToolCallVerdict> {
                Vec::new()
            }
        }
        assert_eq!(
            enforce_non_streaming_responses(
                StatusCode::OK,
                &response(false, false),
                &config(true, false, FailureMode::Open),
                &EmptyEvaluator,
                &L5aScanner,
            ),
            ResponsesOutputDecision::LayerFailure
        );
    }

    #[test]
    fn clean_valid_response_is_deterministic_and_preserves_tool_arguments() {
        let input = response(false, false);
        let result = enforce_non_streaming_responses(
            StatusCode::OK,
            &input,
            &config(true, true, FailureMode::Open),
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        );
        let ResponsesOutputDecision::Allowed(first) = result else {
            panic!("expected allowed response")
        };
        let second = enforce_non_streaming_responses(
            StatusCode::OK,
            &input,
            &config(true, true, FailureMode::Open),
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        );
        assert_eq!(second, ResponsesOutputDecision::Allowed(first.clone()));
        let json: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(json["output"][2]["arguments"], "{\"path\":\"note.txt\"}");
    }

    // -- streaming policy bridge (D-0027 OA-R3) -----------------------------

    use crate::provider::responses_sse::{ResponsesSseCollector, SseCollector};
    use serde_json::json;
    use std::sync::Mutex;

    const CIPHERTEXT: &str = "EncOpaqueCiphertextZZZZ+/==";

    /// Model output carried by one test stream. Every text is non-empty so
    /// each part streams exactly one delta.
    #[derive(Clone)]
    struct StreamSpec {
        output_text: String,
        refusal: String,
        summary: String,
        reasoning: String,
        tool_name: String,
        arguments: String,
        encrypted: Option<&'static str>,
        incomplete: bool,
    }

    impl StreamSpec {
        fn safe() -> Self {
            Self {
                output_text: "safe text".to_string(),
                refusal: "safe refusal".to_string(),
                summary: "safe summary".to_string(),
                reasoning: "safe reasoning".to_string(),
                tool_name: "save_note".to_string(),
                arguments: r#"{"path":"note.txt","meta":{"tags":["a"]}}"#.to_string(),
                encrypted: None,
                incomplete: false,
            }
        }

        fn with_arguments(arguments: Value) -> Self {
            Self {
                arguments: serde_json::to_string(&arguments).unwrap(),
                ..Self::safe()
            }
        }
    }

    fn reasoning_item(spec: &StreamSpec, filled: bool) -> Value {
        let (summary, content) = if filled {
            (
                json!([{"type": "summary_text", "text": spec.summary}]),
                json!([{"type": "reasoning_text", "text": spec.reasoning}]),
            )
        } else {
            (json!([]), json!([]))
        };
        let mut item =
            json!({"type": "reasoning", "id": "rs_1", "summary": summary, "content": content});
        if let Some(enc) = spec.encrypted {
            item["encrypted_content"] = json!(enc);
        }
        item
    }

    fn message_item(content: Value) -> Value {
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": content})
    }

    fn function_item(spec: &StreamSpec, arguments: &str) -> Value {
        json!({"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "call_1",
            "name": spec.tool_name, "arguments": arguments})
    }

    /// Canonical SSE for one message (output_text + refusal), one reasoning
    /// item (summary + plaintext content), and one function call.
    fn stream_sse(spec: &StreamSpec) -> String {
        let text_part = |t: &str| json!({"type": "output_text", "text": t, "annotations": []});
        let refusal_part = |t: &str| json!({"type": "refusal", "refusal": t});
        let content_part = |phase: &str, index: usize, part: Value| {
            json!({"type": format!("response.content_part.{phase}"), "item_id": "msg_1",
                "output_index": 0, "content_index": index, "part": part})
        };
        let summary_part = |phase: &str, t: &str| {
            json!({"type": format!("response.reasoning_summary_part.{phase}"), "item_id": "rs_1",
                "output_index": 1, "summary_index": 0, "part": {"type": "summary_text", "text": t}})
        };
        // Delta then done for one text part; `family` picks the index and
        // done-value keys.
        let text = |family: &str, item: &str, output: usize, index: usize, t: &str| {
            let index_key = if family == "reasoning_summary_text" {
                "summary_index"
            } else {
                "content_index"
            };
            let done_key = if family == "refusal" {
                "refusal"
            } else {
                "text"
            };
            [
                json!({"type": format!("response.{family}.delta"), "item_id": item,
                    "output_index": output, index_key: index, "delta": t}),
                json!({"type": format!("response.{family}.done"), "item_id": item,
                    "output_index": output, index_key: index, done_key: t}),
            ]
        };
        let item = |phase: &str, index: usize, item: Value| json!({"type": format!("response.output_item.{phase}"), "output_index": index, "item": item});
        let message = message_item(json!([
            text_part(&spec.output_text),
            refusal_part(&spec.refusal)
        ]));
        let output = vec![
            message.clone(),
            reasoning_item(spec, true),
            function_item(spec, &spec.arguments),
        ];
        let status = if spec.incomplete {
            "incomplete"
        } else {
            "completed"
        };
        let response = |status: &str, output: Vec<Value>| json!({"id": "resp_1", "object": "response", "status": status, "error": null, "output": output});
        let mut events = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            item("added", 0, message_item(json!([]))),
            content_part("added", 0, text_part("")),
        ];
        events.extend(text("output_text", "msg_1", 0, 0, &spec.output_text));
        events.extend([
            content_part("done", 0, text_part(&spec.output_text)),
            content_part("added", 1, refusal_part("")),
        ]);
        events.extend(text("refusal", "msg_1", 0, 1, &spec.refusal));
        events.extend([
            content_part("done", 1, refusal_part(&spec.refusal)),
            item("done", 0, message),
            item("added", 1, reasoning_item(spec, false)),
            summary_part("added", ""),
        ]);
        events.extend(text("reasoning_summary_text", "rs_1", 1, 0, &spec.summary));
        events.push(summary_part("done", &spec.summary));
        events.extend(text("reasoning_text", "rs_1", 1, 0, &spec.reasoning));
        events.extend([
            item("done", 1, reasoning_item(spec, true)),
            item("added", 2, function_item(spec, "")),
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1",
                "output_index": 2, "delta": spec.arguments}),
            json!({"type": "response.function_call_arguments.done", "item_id": "fc_1",
                "output_index": 2, "name": spec.tool_name, "arguments": spec.arguments}),
            item("done", 2, function_item(spec, &spec.arguments)),
            json!({"type": format!("response.{status}"), "response": response(status, output)}),
        ]);
        events
            .into_iter()
            .enumerate()
            .map(|(seq, mut event)| {
                event["sequence_number"] = json!(seq);
                let kind = event["type"].as_str().unwrap().to_string();
                format!(
                    "event: {kind}\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap()
                )
            })
            .collect()
    }

    fn collect_stream(spec: &StreamSpec) -> (String, CollectedStream) {
        let sse = stream_sse(spec);
        let mut collector = ResponsesSseCollector::new();
        collector
            .push(sse.as_bytes())
            .expect("fixture stream is valid");
        let collected = collector.finish().expect("fixture stream finishes");
        assert_eq!(
            collected.canonical().as_ref(),
            sse.as_bytes(),
            "fixture is already canonical"
        );
        (sse, collected)
    }

    fn enforce_stream(
        spec: &StreamSpec,
        config: &Config,
        evaluator: &dyn ConstraintEvaluator,
        scanner: &dyn OutputScanner,
    ) -> ResponsesStreamDecision {
        let (_, collected) = collect_stream(spec);
        enforce_streaming_responses(collected, config, evaluator, scanner)
    }

    /// Records every scanner input and reports no findings.
    #[derive(Default)]
    struct RecordingScanner(Mutex<Vec<String>>);
    impl OutputScanner for RecordingScanner {
        fn scan_and_redact(&self, content: &str, _: &Config) -> ScanResult {
            self.0.lock().unwrap().push(content.to_string());
            ScanResult {
                content: content.to_string(),
                redactions: Vec::new(),
            }
        }
    }

    /// Reports a redaction but returns the input unchanged.
    struct RedactionOnlyScanner;
    impl OutputScanner for RedactionOnlyScanner {
        fn scan_and_redact(&self, content: &str, _: &Config) -> ScanResult {
            ScanResult {
                content: content.to_string(),
                redactions: vec![crate::layers::l5a::Redaction {
                    pattern: MARKER.to_string(),
                    position: 0,
                }],
            }
        }
    }

    fn assert_suppressed(result: &ResponsesStreamDecision, expected: ResponsesStreamDecision) {
        assert_eq!(*result, expected);
        assert!(!matches!(result, ResponsesStreamDecision::Release(_)));
        let debug = format!("{result:?}");
        for leak in [MARKER, "RESPONSES_SEC", "event:", "resp_1", CIPHERTEXT] {
            assert!(!debug.contains(leak), "decision leaks {leak}: {debug}");
        }
    }

    #[test]
    fn stream_safe_completed_and_incomplete_release_exact_canonical_bytes() {
        for incomplete in [false, true] {
            let spec = StreamSpec {
                incomplete,
                ..StreamSpec::safe()
            };
            let (sse, collected) = collect_stream(&spec);
            let canonical = collected.canonical().clone();
            let result = enforce_streaming_responses(
                collected,
                &config(true, true, FailureMode::Open),
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            );
            assert_eq!(result, ResponsesStreamDecision::Release(canonical));
            let ResponsesStreamDecision::Release(bytes) = result else {
                unreachable!()
            };
            assert_eq!(bytes.as_ref(), sse.as_bytes());
        }
    }

    #[test]
    fn stream_scans_every_model_output_field_in_order_and_never_ciphertext() {
        let spec = StreamSpec {
            encrypted: Some(CIPHERTEXT),
            ..StreamSpec::safe()
        };
        let (sse, collected) = collect_stream(&spec);
        assert!(sse.contains(CIPHERTEXT), "fixture carries ciphertext");
        let scanner = RecordingScanner::default();
        let result = enforce_streaming_responses(
            collected,
            &config(true, true, FailureMode::Closed),
            &DslConstraintEvaluator::new(),
            &scanner,
        );
        let ResponsesStreamDecision::Release(bytes) = &result else {
            panic!("expected release, got {result:?}")
        };
        assert_eq!(bytes.as_ref(), sse.as_bytes());
        assert!(!format!("{result:?}").contains(CIPHERTEXT));

        let inputs = scanner.0.into_inner().unwrap();
        let expected: Vec<String> = [
            "safe text",
            "safe refusal",
            "safe summary",
            "safe reasoning",
            "safe textsafe refusalsafe summarysafe reasoning",
            "save_note",
            r#"{"meta":{"tags":["a"]},"path":"note.txt"}"#,
            "meta",
            "tags",
            "a",
            "path",
            "note.txt",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(inputs, expected);
        assert!(inputs.iter().all(|input| !input.contains(CIPHERTEXT)));
    }

    #[test]
    fn stream_l3_block_and_cardinality_drift_suppress_all_bytes() {
        struct EmptyEvaluator;
        impl ConstraintEvaluator for EmptyEvaluator {
            fn evaluate_tool_calls(
                &self,
                _: &[crate::message::ToolCall],
                _: &Config,
            ) -> Vec<ToolCallVerdict> {
                Vec::new()
            }
        }
        let config = config(true, false, FailureMode::Open);
        for incomplete in [false, true] {
            let spec = StreamSpec {
                incomplete,
                ..StreamSpec::safe()
            };
            assert_suppressed(
                &enforce_stream(&spec, &config, &BlockAll, &L5aScanner),
                ResponsesStreamDecision::BlockedByL3Outbound,
            );
            assert_suppressed(
                &enforce_stream(&spec, &config, &EmptyEvaluator, &L5aScanner),
                ResponsesStreamDecision::LayerFailure,
            );
        }
        // A tool the policy disallows is denied by the real evaluator.
        let mut denying = config;
        denying
            .policy
            .tools
            .get_mut("save_note")
            .expect("fixture tool is configured")
            .allowed = false;
        assert_suppressed(
            &enforce_stream(
                &StreamSpec::safe(),
                &denying,
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            ),
            ResponsesStreamDecision::BlockedByL3Outbound,
        );
    }

    #[test]
    fn stream_l5a_finding_in_any_model_output_field_suppresses_all_bytes() {
        let marked = |field: &str| {
            let mut spec = StreamSpec::safe();
            let target = match field {
                "output_text" => &mut spec.output_text,
                "refusal" => &mut spec.refusal,
                "summary" => &mut spec.summary,
                "reasoning" => &mut spec.reasoning,
                "tool_name" => &mut spec.tool_name,
                _ => unreachable!(),
            };
            *target = format!("x{MARKER}");
            spec
        };
        let mut specs: Vec<StreamSpec> = [
            "output_text",
            "refusal",
            "summary",
            "reasoning",
            "tool_name",
        ]
        .into_iter()
        .map(marked)
        .collect();
        specs.push(StreamSpec::with_arguments(json!({"path": MARKER})));
        specs.push(StreamSpec::with_arguments(json!({MARKER: 1})));
        specs.push(StreamSpec::with_arguments(json!({"n": [{"deep": MARKER}]})));
        for incomplete in [false, true] {
            for spec in &specs {
                let spec = StreamSpec {
                    incomplete,
                    ..spec.clone()
                };
                assert_suppressed(
                    &enforce_stream(
                        &spec,
                        &config(false, true, FailureMode::Open),
                        &DslConstraintEvaluator::new(),
                        &L5aScanner,
                    ),
                    ResponsesStreamDecision::BlockedByL5a,
                );
            }
        }
    }

    #[test]
    fn stream_scans_argument_keys_and_leaves_before_json_escaping() {
        let hex64 = "a".repeat(64);
        let cases = [
            json!({"value": format!("password=\"{hex64}")}),
            json!({format!("password=\"{hex64}"): "value"}),
            json!({"nested": ["safe", {"value": format!("token:\n{hex64}")}]}),
        ];
        for arguments in cases {
            let spec = StreamSpec::with_arguments(arguments);
            let result = enforce_stream(
                &spec,
                &config_with_default_sensitive_patterns(),
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            );
            assert_suppressed(&result, ResponsesStreamDecision::BlockedByL5a);
            assert!(!format!("{result:?}").contains(&hex64));
        }
    }

    #[test]
    fn stream_signal_split_across_adjacent_segments_is_caught_by_aggregate() {
        let (head, tail) = MARKER.split_at(8);
        let spec = StreamSpec {
            output_text: format!("safe {head}"),
            refusal: format!("{tail} safe"),
            ..StreamSpec::safe()
        };
        // No single field carries the marker.
        let sse = stream_sse(&spec);
        assert!(!sse.contains(MARKER));
        assert_suppressed(
            &enforce_stream(
                &spec,
                &config(false, true, FailureMode::Open),
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            ),
            ResponsesStreamDecision::BlockedByL5a,
        );
    }

    #[test]
    fn stream_redaction_or_mutation_never_yields_rewritten_output() {
        let config = config(false, true, FailureMode::Open);
        let evaluator = DslConstraintEvaluator::new();
        let safe = StreamSpec::safe();
        assert_suppressed(
            &enforce_stream(&safe, &config, &evaluator, &RedactionOnlyScanner),
            ResponsesStreamDecision::BlockedByL5a,
        );
        assert_suppressed(
            &enforce_stream(&safe, &config, &evaluator, &MutatingScanner),
            ResponsesStreamDecision::LayerFailure,
        );
        let marked = StreamSpec {
            output_text: MARKER.to_string(),
            ..StreamSpec::safe()
        };
        assert_suppressed(
            &enforce_stream(&marked, &config, &evaluator, &L5aScanner),
            ResponsesStreamDecision::BlockedByL5a,
        );
    }

    #[test]
    fn stream_scanner_panic_fails_closed_in_open_and_closed_modes() {
        for mode in [FailureMode::Open, FailureMode::Closed] {
            for incomplete in [false, true] {
                let spec = StreamSpec {
                    incomplete,
                    encrypted: Some(CIPHERTEXT),
                    ..StreamSpec::safe()
                };
                assert_suppressed(
                    &enforce_stream(
                        &spec,
                        &config(true, true, mode.clone()),
                        &DslConstraintEvaluator::new(),
                        &PanicScanner,
                    ),
                    ResponsesStreamDecision::LayerFailure,
                );
            }
        }
    }

    #[test]
    fn stream_with_l5a_disabled_never_invokes_scanner() {
        let spec = StreamSpec {
            output_text: MARKER.to_string(),
            ..StreamSpec::safe()
        };
        let (sse, collected) = collect_stream(&spec);
        let scanner = RecordingScanner::default();
        let result = enforce_streaming_responses(
            collected,
            &config(true, false, FailureMode::Closed),
            &DslConstraintEvaluator::new(),
            &scanner,
        );
        assert_eq!(
            result,
            ResponsesStreamDecision::Release(Bytes::from(sse.into_bytes()))
        );
        assert!(scanner.0.into_inner().unwrap().is_empty());
        // PanicScanner would fail the stream if it were ever called.
        let result = enforce_stream(
            &spec,
            &config(false, false, FailureMode::Closed),
            &DslConstraintEvaluator::new(),
            &PanicScanner,
        );
        assert!(matches!(result, ResponsesStreamDecision::Release(_)));
    }

    #[test]
    fn stream_bridge_has_no_production_caller() {
        fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    rust_files(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let this = src.join("engine").join("responses.rs");
        // The R4b1 executor is the bridge's only caller.
        let executor = src.join("engine").join("responses_stream_exec.rs");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        assert!(files.contains(&this) && files.contains(&executor));
        for file in files
            .iter()
            .filter(|file| **file != this && **file != executor)
        {
            let text = std::fs::read_to_string(file).unwrap();
            for symbol in ["enforce_streaming_responses", "ResponsesStreamDecision"] {
                assert!(
                    !text.contains(symbol),
                    "{} references {symbol}",
                    file.display()
                );
            }
        }
        let text = std::fs::read_to_string(&this).unwrap();
        let production = text.split("#[cfg(test)]\nmod tests").next().unwrap();
        assert_eq!(
            production.matches("enforce_streaming_responses(").count(),
            1
        );
        let text = std::fs::read_to_string(&executor).unwrap();
        let production = text.split("#[cfg(test)]\nmod tests").next().unwrap();
        assert_eq!(
            production.matches("enforce_streaming_responses(").count(),
            1
        );
    }
}
