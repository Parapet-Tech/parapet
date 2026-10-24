// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Chunk classifiers — M1.5
//
// Determine how each SSE chunk should be handled: pass through,
// buffer as tool call delta, or release as complete tool call.
//
// Two implementations: OpenAI and Anthropic, each understanding
// their respective SSE formats.

use super::types::{ChunkType, SseChunk};

// ---------------------------------------------------------------------------
// Trait: ChunkClassifier
// ---------------------------------------------------------------------------

/// Determines what to do with each SSE chunk.
///
/// Implementations understand provider-specific SSE formats and classify
/// each chunk into a `ChunkType` that the stream processor acts on.
pub trait ChunkClassifier: Send + Sync {
    fn classify(&self, chunk: &SseChunk) -> ChunkType;
}

// ---------------------------------------------------------------------------
// OpenAI classifier
// ---------------------------------------------------------------------------

/// Classifies OpenAI SSE chunks.
///
/// OpenAI format:
/// - `data: {"choices":[{"delta":{"content":"Hello"}}]}` -> TextContent
/// - `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read"}}]}}]}` -> ToolCallDelta(0)
/// - `data: {"choices":[{"finish_reason":"tool_calls",...}]}` -> ToolCallComplete for all in-progress
/// - `data: {"choices":[{"finish_reason":"stop",...}]}` -> Control
/// - `data: [DONE]` -> Done
pub struct OpenAiChunkClassifier;

impl ChunkClassifier for OpenAiChunkClassifier {
    fn classify(&self, chunk: &SseChunk) -> ChunkType {
        let data = chunk.data.trim();

        // Stream terminator
        if data == "[DONE]" {
            return ChunkType::Done;
        }

        // Try to parse as JSON
        let json: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return ChunkType::NonSse,
        };

        // Navigate to choices[0]
        let choice = match json.get("choices").and_then(|c| c.get(0)) {
            Some(c) => c,
            None => return ChunkType::Control,
        };

        // Check finish_reason first — if "tool_calls", the tool call is complete
        if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            if finish == "tool_calls" {
                // The finish_reason signals all tool calls are complete.
                // We return ToolCallComplete(0) as a sentinel — the processor
                // should complete all in-progress tool call buffers.
                return ChunkType::ToolCallComplete(0);
            }
            if finish == "stop" {
                return ChunkType::Control;
            }
        }

        // Check delta for tool_calls or content
        if let Some(delta) = choice.get("delta") {
            // Tool call deltas
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                if let Some(first) = tool_calls.first() {
                    let index = first
                        .get("index")
                        .and_then(|i| i.as_u64())
                        .unwrap_or(0) as usize;
                    return ChunkType::ToolCallDelta(index);
                }
            }

            // Text content
            if delta.get("content").is_some() {
                return ChunkType::TextContent;
            }

            // Role-only delta (first chunk often just has {"role":"assistant"})
            if delta.get("role").is_some() && delta.get("content").is_none() {
                return ChunkType::Control;
            }
        }

        ChunkType::Control
    }
}

// ---------------------------------------------------------------------------
// Anthropic classifier
// ---------------------------------------------------------------------------

/// Classifies Anthropic SSE chunks.
///
/// Anthropic format uses `event:` + `data:` pairs:
/// - `event: content_block_delta` + `delta.type: "text_delta"` -> TextContent
/// - `event: content_block_start` + `content_block.type: "tool_use"` -> ToolCallDelta (start)
/// - `event: content_block_delta` + `delta.type: "input_json_delta"` -> ToolCallDelta
/// - `event: content_block_stop` -> ToolCallComplete for that index
/// - `event: message_start`, `message_stop`, `message_delta`, `ping` -> Control
/// - `event: content_block_start` + `content_block.type: "text"` -> Control (text block start)
pub struct AnthropicChunkClassifier;

impl ChunkClassifier for AnthropicChunkClassifier {
    fn classify(&self, chunk: &SseChunk) -> ChunkType {
        let event = chunk.event.as_deref().unwrap_or("");
        let data = chunk.data.trim();

        match event {
            "content_block_start" => {
                // Parse data to determine if this is a text or tool_use block
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let block_type = json
                        .get("content_block")
                        .and_then(|cb| cb.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");

                    if block_type == "tool_use" {
                        let index = json
                            .get("index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        return ChunkType::ToolCallDelta(index);
                    }
                }
                // Text block start or unknown — pass through
                ChunkType::Control
            }
            "content_block_delta" => {
                // Parse delta type
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let delta_type = json
                        .get("delta")
                        .and_then(|d| d.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");

                    match delta_type {
                        "text_delta" => return ChunkType::TextContent,
                        "input_json_delta" => {
                            let index = json
                                .get("index")
                                .and_then(|i| i.as_u64())
                                .unwrap_or(0) as usize;
                            return ChunkType::ToolCallDelta(index);
                        }
                        _ => {}
                    }
                }
                ChunkType::Control
            }
            "content_block_stop" => {
                // Parse to get the index
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let index = json
                        .get("index")
                        .and_then(|i| i.as_u64())
                        .unwrap_or(0) as usize;
                    return ChunkType::ToolCallComplete(index);
                }
                ChunkType::Control
            }
            "message_start" | "message_stop" | "message_delta" | "ping" => ChunkType::Control,
            "" => {
                // No event type — could be non-SSE content
                if data.is_empty() {
                    ChunkType::Control
                } else {
                    ChunkType::NonSse
                }
            }
            _ => ChunkType::Control,
        }
    }
}

// ---------------------------------------------------------------------------
// SSE line parsing
// ---------------------------------------------------------------------------

/// Parse raw SSE lines into an SseChunk.
///
/// SSE format:
/// ```text
/// event: <event_type>\n
/// data: <payload>\n
/// \n
/// ```
///
/// OpenAI omits the event line. Anthropic includes it.
/// Multiple `data:` lines for the same event are concatenated with newlines.
pub fn parse_sse_line(line: &str) -> Option<SseChunk> {
    let trimmed = line.trim();

    // Empty line is an event separator in SSE, not a chunk
    if trimmed.is_empty() {
        return None;
    }

    // Comment lines start with ':'
    if trimmed.starts_with(':') {
        return None;
    }

    // data: line (OpenAI format, no event prefix)
    if let Some(data) = trimmed.strip_prefix("data: ").or_else(|| trimmed.strip_prefix("data:")) {
        return Some(SseChunk {
            event: None,
            data: data.to_string(),
        });
    }

    None
}

/// Maximum characters buffered while determining initial SSE vs non-SSE mode.
pub(crate) const MAX_INITIAL_SSE_PROBE_CHARS: usize = 64 * 1024;

/// Tri-state result when probing initial stream bytes for SSE framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InitialSseMode {
    Sse,
    NonSse,
    NeedMore,
}

/// Classify accumulated UTF-8 text from the start of a stream as SSE, NonSse,
/// or NeedMore (when a leading BOM, whitespace/blank lines, or split SSE field
/// prefix has not yet completed).
pub(crate) fn classify_initial_sse_str(buffer: &str, eof: bool) -> InitialSseMode {
    let without_bom = buffer.strip_prefix('\u{FEFF}').unwrap_or(buffer);
    let trimmed = without_bom.trim_start_matches([' ', '\t', '\n', '\r']);
    if trimmed.is_empty() {
        if !eof && buffer.chars().count() <= MAX_INITIAL_SSE_PROBE_CHARS {
            return InitialSseMode::NeedMore;
        }
        return InitialSseMode::NonSse;
    }

    if trimmed.starts_with("data:")
        || trimmed.starts_with("event:")
        || trimmed.starts_with("id:")
        || trimmed.starts_with("retry:")
        || trimmed.starts_with(':')
    {
        return InitialSseMode::Sse;
    }

    if !eof
        && !trimmed.contains('\n')
        && !trimmed.contains('\r')
        && ["data:", "event:", "id:", "retry:"]
            .iter()
            .any(|prefix| prefix.starts_with(trimmed))
        && buffer.chars().count() <= MAX_INITIAL_SSE_PROBE_CHARS
    {
        return InitialSseMode::NeedMore;
    }

    InitialSseMode::NonSse
}

/// Classify raw initial bytes from a stream as SSE, NonSse, or NeedMore.
/// Handles multi-byte UTF-8 sequences (such as a 3-byte UTF-8 BOM) split across chunks.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn classify_initial_sse_bytes(bytes: &[u8], eof: bool) -> InitialSseMode {
    match std::str::from_utf8(bytes) {
        Ok(text) => classify_initial_sse_str(text, eof),
        Err(err) => {
            if !eof && err.error_len().is_none() {
                let valid_prefix = std::str::from_utf8(&bytes[..err.valid_up_to()]).unwrap_or("");
                match classify_initial_sse_str(valid_prefix, false) {
                    InitialSseMode::NeedMore => InitialSseMode::NeedMore,
                    decided => decided,
                }
            } else {
                InitialSseMode::NonSse
            }
        }
    }
}

/// Decode a transport chunk into UTF-8, holding back up to 3 trailing bytes
/// when a multi-byte UTF-8 codepoint (such as a UTF-8 BOM) is split across chunks.
pub(crate) fn decode_utf8_chunk(utf8_tail: &mut Vec<u8>, chunk: &[u8]) -> String {
    let combined: std::borrow::Cow<[u8]> = if utf8_tail.is_empty() {
        std::borrow::Cow::Borrowed(chunk)
    } else {
        let mut v = std::mem::take(utf8_tail);
        v.extend_from_slice(chunk);
        std::borrow::Cow::Owned(v)
    };
    let bytes = combined.as_ref();
    let check_start = bytes.len().saturating_sub(3);
    let mut incomplete_len = 0;
    for idx in check_start..bytes.len() {
        if let Err(err) = std::str::from_utf8(&bytes[idx..]) {
            if err.valid_up_to() == 0 && err.error_len().is_none() {
                incomplete_len = bytes.len() - idx;
                break;
            }
        }
    }
    let valid_end = bytes.len() - incomplete_len;
    if incomplete_len > 0 {
        utf8_tail.extend_from_slice(&bytes[valid_end..]);
    }
    String::from_utf8_lossy(&bytes[..valid_end]).into_owned()
}

/// Detect whether a response body looks like an SSE stream.
///
/// SSE streams start with lines like `data:`, `event:`, `id:`, `retry:`, or `:`.
/// Non-SSE responses (JSON error bodies, HTML) do not.
#[cfg_attr(not(test), allow(dead_code))]
pub fn is_sse_content(first_bytes: &[u8]) -> bool {
    classify_initial_sse_bytes(first_bytes, true) == InitialSseMode::Sse
}
