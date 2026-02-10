// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Tests for M1.5 — SSE streaming passthrough with tool call buffering
//
// Written BEFORE implementation per project protocol.
// Tests cover:
//  1. OpenAI text content chunks forwarded in order
//  2. Anthropic text content chunks forwarded in order
//  3. Text content flows through immediately (not buffered)
//  4. Tool call delta chunks buffered until complete (OpenAI)
//  5. Tool call delta chunks buffered until complete (Anthropic)
//  6. Complete tool call passed to validator before release
//  7. Tool call buffer exceeding 1MB -> error response
//  8. Interleaved text and tool_use blocks -> text flows, tool_use buffered
//  9. Non-SSE response body passed through as-is
// 10. data: [DONE] event forwarded as-is

use super::*;
use crate::message::ToolCall;
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tokio::time::{self, Duration};

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

/// A validator that always allows tool calls.
struct AllowAllValidator;

impl ToolCallValidator for AllowAllValidator {
    fn validate(&self, _tool_call: &ToolCall) -> ValidationResult {
        ValidationResult::Allow
    }
}

/// A validator that always blocks tool calls with a reason.
struct BlockAllValidator {
    reason: String,
}

impl BlockAllValidator {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl ToolCallValidator for BlockAllValidator {
    fn validate(&self, _tool_call: &ToolCall) -> ValidationResult {
        ValidationResult::Block(self.reason.clone())
    }
}

/// A validator that records which tool calls it was asked to validate.
struct RecordingValidator {
    calls: Mutex<Vec<ToolCall>>,
    result: ValidationResult,
}

impl RecordingValidator {
    fn allowing() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            result: ValidationResult::Allow,
        }
    }

    fn validated_calls(&self) -> Vec<ToolCall> {
        self.calls.lock().unwrap().clone()
    }
}

impl ToolCallValidator for RecordingValidator {
    fn validate(&self, tool_call: &ToolCall) -> ValidationResult {
        self.calls.lock().unwrap().push(tool_call.clone());
        self.result.clone()
    }
}

// ---------------------------------------------------------------------------
// Helper: create a stream from SSE lines
// ---------------------------------------------------------------------------

/// Build an in-memory byte stream from a list of SSE line strings.
/// Each string becomes a separate chunk (simulating streaming).
fn sse_stream(lines: Vec<&str>) -> impl tokio_stream::Stream<Item = Bytes> + Unpin + Send {
    let chunks: Vec<Bytes> = lines
        .into_iter()
        .map(|l| Bytes::from(format!("{l}\n")))
        .collect();
    tokio_stream::iter(chunks)
}

/// Build an in-memory byte stream from raw bytes (for non-SSE testing).
fn raw_stream(data: Vec<&[u8]>) -> impl tokio_stream::Stream<Item = Bytes> + Unpin + Send {
    let chunks: Vec<Bytes> = data.into_iter().map(|d| Bytes::copy_from_slice(d)).collect();
    tokio_stream::iter(chunks)
}

/// Build a stream backed by a channel for time-controlled tests.
fn channel_stream() -> (tokio::sync::mpsc::Sender<Bytes>, ReceiverStream<Bytes>) {
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    (tx, ReceiverStream::new(rx))
}

/// Collect all output from a stream processor into a single string.
async fn collect_output(
    stream: impl tokio_stream::Stream<Item = Bytes> + Unpin,
) -> String {
    let mut output = String::new();
    tokio::pin!(stream);
    while let Some(chunk) = stream.next().await {
        output.push_str(&String::from_utf8_lossy(&chunk));
    }
    output
}

// ---------------------------------------------------------------------------
// Test 1: OpenAI text content chunks forwarded in order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_text_content_chunks_forwarded_in_order() {
    let input = sse_stream(vec![
        r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    // Both text chunks should appear in order
    let hello_pos = output.find("\"content\":\"Hello\"").expect("Should contain Hello");
    let world_pos = output
        .find("\"content\":\" world\"")
        .expect("Should contain world");
    assert!(
        hello_pos < world_pos,
        "Hello should appear before world in output"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Anthropic text content chunks forwarded in order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_text_content_chunks_forwarded_in_order() {
    let input = sse_stream(vec![
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
    ]);

    let processor = StreamProcessor::new(
        Arc::new(AnthropicChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    let hello_pos = output.find("\"text\":\"Hello\"").expect("Should contain Hello");
    let world_pos = output.find("\"text\":\" world\"").expect("Should contain world");
    assert!(
        hello_pos < world_pos,
        "Hello should appear before world in output"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Text content flows through immediately (ordering proof)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_content_not_buffered_flows_immediately() {
    // Send text content chunks. If they were buffered, order might change
    // or they would be delayed. We verify they appear in output in the same
    // order as input.
    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"C"},"finish_reason":null}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    let a_pos = output.find("\"content\":\"A\"").expect("Should contain A");
    let b_pos = output.find("\"content\":\"B\"").expect("Should contain B");
    let c_pos = output.find("\"content\":\"C\"").expect("Should contain C");

    assert!(a_pos < b_pos, "A before B");
    assert!(b_pos < c_pos, "B before C");
}

// ---------------------------------------------------------------------------
// Test 4: OpenAI tool call delta chunks buffered until complete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_tool_call_deltas_buffered_until_complete() {
    let validator = Arc::new(RecordingValidator::allowing());

    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"/tmp\"}"}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        validator.clone(),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    // The validator should have been called with the complete tool call
    let calls = validator.validated_calls();
    assert_eq!(calls.len(), 1, "Validator should be called exactly once");
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments, serde_json::json!({"path": "/tmp"}));

    // The tool call data should appear in output (since validator allows)
    assert!(
        output.contains("read_file"),
        "Allowed tool call should appear in output"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Anthropic tool call delta chunks buffered until complete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_tool_call_deltas_buffered_until_complete() {
    let validator = Arc::new(RecordingValidator::allowing());

    let input = sse_stream(vec![
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"th\":\"/tmp\"}"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
    ]);

    let processor = StreamProcessor::new(
        Arc::new(AnthropicChunkClassifier),
        validator.clone(),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    // The validator should have been called with the complete tool call
    let calls = validator.validated_calls();
    assert_eq!(calls.len(), 1, "Validator should be called exactly once");
    assert_eq!(calls[0].id, "toolu_1");
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments, serde_json::json!({"path": "/tmp"}));

    // Tool call chunks should appear in output (allowed)
    assert!(
        output.contains("read_file"),
        "Allowed tool call should appear in output"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Complete tool call passed to validator before release
// ---------------------------------------------------------------------------

#[tokio::test]
async fn complete_tool_call_validated_before_release_blocked() {
    let validator = Arc::new(BlockAllValidator::new("not in allowlist"));

    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec_command","arguments":""}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"rm -rf /\"}"}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        validator,
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    // The blocked tool call should NOT have its deltas in the output
    // but should have an error message
    assert!(
        output.contains("error"),
        "Blocked tool call should produce error in output"
    );
    assert!(
        output.contains("not in allowlist"),
        "Error should contain the block reason"
    );

    // The original tool call arguments should NOT be forwarded
    // (the delta lines with exec_command args should be suppressed)
    assert!(
        !output.contains("rm -rf"),
        "Blocked tool call arguments should not appear in output"
    );
}

#[tokio::test]
async fn validator_receives_tool_call_with_correct_fields() {
    let validator = Arc::new(RecordingValidator::allowing());

    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"write_file","arguments":""}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"/tmp/out.txt\",\"content\":\"hello\"}"}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        validator.clone(),
        ToolCallBlockMode::Error,
    );

    let _ = collect_output(processor.process(input)).await;

    let calls = validator.validated_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_abc");
    assert_eq!(calls[0].name, "write_file");
    assert_eq!(
        calls[0].arguments,
        serde_json::json!({"path": "/tmp/out.txt", "content": "hello"})
    );
}

// ---------------------------------------------------------------------------
// Test 7: Tool call buffer exceeding 1MB -> error response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_call_buffer_overflow_produces_error() {
    // Create a tool call with arguments that exceed 1MB
    let large_chunk = "a".repeat(600 * 1024); // 600KB per chunk -> overflow after two

    let line1 = format!(
        r#"data: {{"id":"c1","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"id":"call_big","type":"function","function":{{"name":"big_tool","arguments":""}}}}]}},"finish_reason":null}}]}}"#
    );
    let line2 = format!(
        r#"data: {{"id":"c1","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{large_chunk}"}}}}]}},"finish_reason":null}}]}}"#
    );
    let line3 = format!(
        r#"data: {{"id":"c1","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{large_chunk}"}}}}]}},"finish_reason":null}}]}}"#
    );

    let input = sse_stream(vec![&line1, "", &line2, "", &line3, ""]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    assert!(
        output.contains("error"),
        "Buffer overflow should produce error in output"
    );
    assert!(
        output.contains("overflow") || output.contains("exceeded") || output.contains("limit"),
        "Error should mention overflow/exceeded/limit, got: {output}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Interleaved text and tool_use blocks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interleaved_text_and_tool_use_text_flows_tool_buffered() {
    let validator = Arc::new(RecordingValidator::allowing());

    // Anthropic format with text block (index 0) then tool_use block (index 1)
    let input = sse_stream(vec![
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}"#,
        "",
        // Text block
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me read that file."}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        // Tool use block
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp/test.txt\"}"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
    ]);

    let processor = StreamProcessor::new(
        Arc::new(AnthropicChunkClassifier),
        validator.clone(),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    // Text content should be in the output
    assert!(
        output.contains("Let me read that file."),
        "Text content should flow through: {output}"
    );

    // Tool call should have been validated
    let calls = validator.validated_calls();
    assert_eq!(calls.len(), 1, "One tool call should be validated");
    assert_eq!(calls[0].name, "read_file");

    // Tool call data should also be in output (since allowed)
    assert!(
        output.contains("read_file"),
        "Allowed tool call should be in output"
    );

    // Verify text appears before the released tool call data in the output.
    // Text is passed through immediately, tool call is buffered then released.
    let text_pos = output.find("Let me read that file.").unwrap();
    // The released buffered lines containing read_file come after the text
    // Find the position of the buffered tool call start (which contains read_file in the data)
    // We look for the content_block_start with tool_use, which is part of the buffered lines
    let tool_start_line = output.find("\"type\":\"tool_use\"");
    if let Some(tool_pos) = tool_start_line {
        assert!(
            text_pos < tool_pos,
            "Text should appear before released tool call data"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: Non-SSE response body passed through as-is
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_sse_response_passed_through_as_is() {
    // Simulate a 429 rate limit JSON error response
    let json_error = r#"{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;

    let input = raw_stream(vec![json_error.as_bytes()]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    assert_eq!(
        output.trim(),
        json_error,
        "Non-SSE response should pass through unchanged"
    );
}

#[tokio::test]
async fn non_sse_html_error_passed_through() {
    let html = b"<html><body><h1>502 Bad Gateway</h1></body></html>";

    let input = raw_stream(vec![html]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    assert!(
        output.contains("502 Bad Gateway"),
        "HTML error should pass through"
    );
}

// ---------------------------------------------------------------------------
// Test 10: data: [DONE] event forwarded as-is
// ---------------------------------------------------------------------------

#[tokio::test]
async fn done_event_forwarded_as_is() {
    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let output = collect_output(processor.process(input)).await;

    assert!(
        output.contains("data: [DONE]"),
        "DONE event should be in output: {output}"
    );
}

// ---------------------------------------------------------------------------
// Additional streaming behavior tests
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn tool_call_delta_timeout_emits_error() {
    let (tx, input) = channel_stream();

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let mut output_stream = processor.process(input);

    // Send a tool call delta to start buffering.
    let _ = tx
        .send(Bytes::from(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"slow_tool\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n",
        ))
        .await;

    tokio::task::yield_now().await;

    // Advance time past the timeout.
    time::advance(Duration::from_secs(31)).await;

    // Send another chunk to trigger the timeout check.
    let _ = tx
        .send(Bytes::from(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n",
        ))
        .await;

    drop(tx);

    let mut output = String::new();
    while let Some(chunk) = output_stream.next().await {
        output.push_str(&String::from_utf8_lossy(&chunk));
    }

    assert!(
        output.contains("delta timeout"),
        "Expected timeout error in stream output"
    );
}

#[tokio::test]
async fn slow_consumer_does_not_break_stream() {
    let mut chunks: Vec<Bytes> = Vec::new();
    for i in 0..100 {
        chunks.push(Bytes::from(format!(
            r#"data: {{"id":"c1","choices":[{{"index":0,"delta":{{"content":"chunk{i}"}},"finish_reason":null}}]}}"#,
        )));
        chunks.push(Bytes::from("\n"));
    }
    let input = tokio_stream::iter(chunks);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let mut output = String::new();
    let mut stream = processor.process(input);

    while let Some(chunk) = stream.next().await {
        output.push_str(&String::from_utf8_lossy(&chunk));
        time::sleep(Duration::from_millis(5)).await;
    }

    assert!(output.contains("chunk0"));
    assert!(output.contains("chunk99"));
}

#[tokio::test]
async fn client_disconnect_stops_processing() {
    let (tx, input) = channel_stream();

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        Arc::new(AllowAllValidator),
        ToolCallBlockMode::Error,
    );

    let mut output_stream = processor.process(input);

    let _ = tx
        .send(Bytes::from(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n",
        ))
        .await;

    let _ = output_stream.next().await;

    drop(output_stream);

    let _ = tx
        .send(Bytes::from(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n",
        ))
        .await;

    tokio::task::yield_now().await;

    let closed = time::timeout(Duration::from_secs(1), tx.closed()).await;
    assert!(
        closed.is_ok(),
        "processor should stop when client disconnects"
    );
}

// ---------------------------------------------------------------------------
// Rewrite mode tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_blocked_tool_call_rewrite_injects_text() {
    let validator = Arc::new(BlockAllValidator::new("not allowed"));

    let input = sse_stream(vec![
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"exec_command","arguments":""}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\"rm -rf /\"}"}}]},"finish_reason":null}]}"#,
        "",
        r#"data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "",
        "data: [DONE]",
    ]);

    let processor = StreamProcessor::new(
        Arc::new(OpenAiChunkClassifier),
        validator,
        ToolCallBlockMode::Rewrite,
    );

    let output = collect_output(processor.process(input)).await;

    assert!(
        output.contains("Tool 'exec_command' was blocked"),
        "Rewrite should inject refusal text"
    );
    assert!(
        !output.contains("rm -rf"),
        "Blocked tool arguments should not appear"
    );
}

#[tokio::test]
async fn anthropic_blocked_tool_call_rewrite_injects_text_block() {
    let validator = Arc::new(BlockAllValidator::new("not allowed"));

    let input = sse_stream(vec![
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022"}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"exec_command","input":{}}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"rm -rf /\"}"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
    ]);

    let processor = StreamProcessor::new(
        Arc::new(AnthropicChunkClassifier),
        validator,
        ToolCallBlockMode::Rewrite,
    );

    let output = collect_output(processor.process(input)).await;

    assert!(
        output.contains("content_block_start"),
        "Rewrite should emit a synthetic text block start"
    );
    assert!(
        output.contains("Tool 'exec_command' was blocked"),
        "Rewrite should inject refusal text"
    );
    assert!(
        !output.contains("rm -rf"),
        "Blocked tool arguments should not appear"
    );
}

// ---------------------------------------------------------------------------
// Classifier unit tests
// ---------------------------------------------------------------------------

mod classifier_tests {
    use super::super::classifier::*;
    use super::super::types::*;

    #[test]
    fn openai_classify_text_content() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: r#"{"id":"c1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::TextContent);
    }

    #[test]
    fn openai_classify_tool_call_delta() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: r#"{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read"}}]},"finish_reason":null}]}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::ToolCallDelta(0));
    }

    #[test]
    fn openai_classify_tool_call_complete() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: r#"{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#
                .to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::ToolCallComplete(0));
    }

    #[test]
    fn openai_classify_done() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: "[DONE]".to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Done);
    }

    #[test]
    fn openai_classify_stop() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: r#"{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#
                .to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Control);
    }

    #[test]
    fn openai_classify_role_delta() {
        let classifier = OpenAiChunkClassifier;
        let chunk = SseChunk {
            event: None,
            data: r#"{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Control);
    }

    #[test]
    fn anthropic_classify_text_delta() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("content_block_delta".to_string()),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::TextContent);
    }

    #[test]
    fn anthropic_classify_tool_use_start() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("content_block_start".to_string()),
            data: r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::ToolCallDelta(1));
    }

    #[test]
    fn anthropic_classify_input_json_delta() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("content_block_delta".to_string()),
            data: r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::ToolCallDelta(1));
    }

    #[test]
    fn anthropic_classify_content_block_stop() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("content_block_stop".to_string()),
            data: r#"{"type":"content_block_stop","index":1}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::ToolCallComplete(1));
    }

    #[test]
    fn anthropic_classify_message_start_is_control() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("message_start".to_string()),
            data: r#"{"type":"message_start","message":{}}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Control);
    }

    #[test]
    fn anthropic_classify_message_stop_is_control() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("message_stop".to_string()),
            data: r#"{"type":"message_stop"}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Control);
    }

    #[test]
    fn anthropic_classify_text_block_start_is_control() {
        let classifier = AnthropicChunkClassifier;
        let chunk = SseChunk {
            event: Some("content_block_start".to_string()),
            data: r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.to_string(),
        };
        assert_eq!(classifier.classify(&chunk), ChunkType::Control);
    }
}

// ---------------------------------------------------------------------------
// ToolCallBuffer unit tests
// ---------------------------------------------------------------------------

mod buffer_tests {
    use super::super::types::*;

    #[test]
    fn buffer_accumulates_arguments() {
        let mut buf = ToolCallBuffer::new("call_1", "read_file");
        buf.append_arguments("{\"pa").unwrap();
        buf.append_arguments("th\":\"/tmp\"}").unwrap();
        assert_eq!(buf.arguments_json, "{\"path\":\"/tmp\"}");
    }

    #[test]
    fn buffer_parse_arguments_valid() {
        let mut buf = ToolCallBuffer::new("call_1", "read_file");
        buf.append_arguments("{\"path\":\"/tmp\"}").unwrap();
        let args = buf.parse_arguments().unwrap();
        assert_eq!(args, serde_json::json!({"path": "/tmp"}));
    }

    #[test]
    fn buffer_parse_empty_arguments() {
        let buf = ToolCallBuffer::new("call_1", "no_args");
        let args = buf.parse_arguments().unwrap();
        assert_eq!(args, serde_json::json!({}));
    }

    #[test]
    fn buffer_overflow_detected() {
        let mut buf = ToolCallBuffer::new("call_1", "big_tool");
        let big = "x".repeat(MAX_TOOL_CALL_BUFFER_BYTES + 1);
        let result = buf.append_arguments(&big);
        assert!(result.is_err());
        match result.unwrap_err() {
            StreamError::BufferOverflow { tool_name, .. } => {
                assert_eq!(tool_name, "big_tool");
            }
            other => panic!("Expected BufferOverflow, got: {:?}", other),
        }
    }

    #[test]
    fn buffer_tracks_total_bytes() {
        let mut buf = ToolCallBuffer::new("call_1", "tool");
        buf.append_arguments("hello").unwrap(); // 5 bytes
        assert_eq!(buf.total_bytes, 5);
        buf.append_arguments(" world").unwrap(); // 6 bytes
        assert_eq!(buf.total_bytes, 11);
    }

    #[test]
    fn buffer_raw_line_storage() {
        let mut buf = ToolCallBuffer::new("call_1", "tool");
        buf.buffer_raw_line("data: {\"test\": 1}");
        buf.buffer_raw_line("data: {\"test\": 2}");
        assert_eq!(buf.buffered_chunks.len(), 2);
        assert_eq!(buf.buffered_chunks[0], "data: {\"test\": 1}");
    }
}

// ---------------------------------------------------------------------------
// SSE parsing unit tests
// ---------------------------------------------------------------------------

mod parse_tests {
    use super::super::classifier::*;

    #[test]
    fn parse_sse_data_line() {
        let chunk = parse_sse_line("data: {\"content\":\"hello\"}").unwrap();
        assert_eq!(chunk.event, None);
        assert_eq!(chunk.data, "{\"content\":\"hello\"}");
    }

    #[test]
    fn parse_sse_data_line_no_space() {
        let chunk = parse_sse_line("data:{\"content\":\"hello\"}").unwrap();
        assert_eq!(chunk.data, "{\"content\":\"hello\"}");
    }

    #[test]
    fn parse_sse_done() {
        let chunk = parse_sse_line("data: [DONE]").unwrap();
        assert_eq!(chunk.data, "[DONE]");
    }

    #[test]
    fn parse_sse_empty_line_returns_none() {
        assert!(parse_sse_line("").is_none());
    }

    #[test]
    fn parse_sse_comment_returns_none() {
        assert!(parse_sse_line(": this is a comment").is_none());
    }

    #[test]
    fn is_sse_detects_data_prefix() {
        assert!(is_sse_content(b"data: {\"test\": true}"));
    }

    #[test]
    fn is_sse_detects_event_prefix() {
        assert!(is_sse_content(b"event: message_start\n"));
    }

    #[test]
    fn is_sse_rejects_json() {
        assert!(!is_sse_content(b"{\"error\": \"rate limit\"}"));
    }

    #[test]
    fn is_sse_rejects_html() {
        assert!(!is_sse_content(b"<html><body>Error</body></html>"));
    }

    #[test]
    fn is_sse_handles_leading_whitespace() {
        assert!(is_sse_content(b"  data: test"));
    }

    #[test]
    fn is_sse_handles_invalid_utf8() {
        assert!(!is_sse_content(&[0xFF, 0xFE, 0x00]));
    }
}

// ---------------------------------------------------------------------------
// StreamError display tests
// ---------------------------------------------------------------------------

mod error_tests {
    use super::super::types::*;

    #[test]
    fn buffer_overflow_display() {
        let err = StreamError::BufferOverflow {
            tool_name: "big_tool".to_string(),
            tool_id: "call_1".to_string(),
            limit: 1_048_576,
        };
        let msg = err.to_string();
        assert!(msg.contains("big_tool"));
        assert!(msg.contains("1048576"));
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn delta_timeout_display() {
        let err = StreamError::DeltaTimeout {
            tool_name: "slow_tool".to_string(),
            tool_id: "call_2".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("slow_tool"));
        assert!(msg.contains("30"));
    }

    #[test]
    fn malformed_tool_call_display() {
        let err = StreamError::MalformedToolCall {
            tool_name: "bad_tool".to_string(),
            tool_id: "call_3".to_string(),
            reason: "invalid JSON".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("bad_tool"));
        assert!(msg.contains("invalid JSON"));
    }

    #[test]
    fn tool_call_blocked_display() {
        let err = StreamError::ToolCallBlocked {
            tool_name: "exec_command".to_string(),
            tool_id: "call_4".to_string(),
            reason: "not in allowlist".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("exec_command"));
        assert!(msg.contains("not in allowlist"));
    }
}

// ---------------------------------------------------------------------------
// ValidationResult tests
// ---------------------------------------------------------------------------

mod validation_tests {
    use super::super::types::*;

    #[test]
    fn allow_variant() {
        let result = ValidationResult::Allow;
        assert_eq!(result, ValidationResult::Allow);
    }

    #[test]
    fn block_variant_with_reason() {
        let result = ValidationResult::Block("forbidden".to_string());
        match result {
            ValidationResult::Block(reason) => assert_eq!(reason, "forbidden"),
            _ => panic!("Expected Block"),
        }
    }
}
