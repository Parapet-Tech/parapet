// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Output-policy seam for non-streaming OpenAI Responses bodies.
//!
//! This module is deliberately not wired to the proxy route yet. It gives the
//! activation slice one closed, structured path instead of reusing the Chat
//! Completions handler, whose legacy parse failures pass the upstream body
//! through unchanged.

use crate::config::{Config, FailureMode};
use crate::constraint::{ConstraintEvaluator, ToolCallVerdict};
use crate::layers::l5a::{OutputScanner, ScanResult};
use crate::provider::responses_output::{OpenAiResponsesOutputDecoder, ResponseDecoder};
use axum::http::StatusCode;

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

    if config
        .policy
        .layers
        .l3_outbound
        .as_ref()
        .is_some_and(|layer| layer.mode == "block")
    {
        let verdicts = evaluator.evaluate_tool_calls(decoded.tool_calls(), config);
        if verdicts.len() != decoded.tool_calls().len() {
            return ResponsesOutputDecision::LayerFailure;
        }
        if verdicts
            .iter()
            .any(|verdict| matches!(verdict, ToolCallVerdict::Block { .. }))
        {
            return ResponsesOutputDecision::BlockedByL3Outbound;
        }
    }

    let l5a_enabled = config
        .policy
        .layers
        .l5a
        .as_ref()
        .is_some_and(|layer| layer.mode == "redact");
    if !l5a_enabled {
        return ResponsesOutputDecision::Allowed(decoded.to_json_bytes());
    }

    // Arguments are complete, strictly parsed objects here. Scan their
    // canonical JSON for detection, but never splice scanner output back into
    // an executable tool call. Any finding suppresses the entire response.
    for tool_call in decoded.tool_calls() {
        let arguments = serde_json::to_string(&tool_call.arguments)
            .expect("serializing a serde_json::Value cannot fail");
        match scan(scanner, &arguments, config) {
            Ok(Some(result)) if !result.redactions.is_empty() => {
                return ResponsesOutputDecision::BlockedByL5aToolArguments;
            }
            Ok(Some(result)) if result.content != arguments => {
                return ResponsesOutputDecision::LayerFailure;
            }
            Ok(_) => {}
            Err(()) => return ResponsesOutputDecision::LayerFailure,
        }
    }

    let mut replacements = Vec::with_capacity(decoded.texts().len());
    for segment in decoded.texts() {
        match scan(scanner, segment.text(), config) {
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

/// `Ok(None)` means an open-mode scanner panic and therefore preserves the
/// validated original field. Closed mode maps the same panic to a fixed
/// content-free layer failure.
#[cfg_attr(not(test), allow(dead_code))]
fn scan(
    scanner: &dyn OutputScanner,
    content: &str,
    config: &Config,
) -> Result<Option<ScanResult>, ()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scanner.scan_and_redact(content, config)
    })) {
        Ok(result) => Ok(Some(result)),
        Err(_) if config.runtime.engine.on_failure == FailureMode::Open => Ok(None),
        Err(_) => Err(()),
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
            },
            contract_hash: "sha256:test".to_string(),
        }
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
}
