// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Stream types — M1.5
//
// Core types for SSE streaming: chunks, classification, validation,
// tool call buffering, and errors.

use crate::message::ToolCall;
use std::fmt;

// ---------------------------------------------------------------------------
// SSE chunk representation
// ---------------------------------------------------------------------------

/// A parsed SSE chunk from the upstream response stream.
///
/// Each chunk represents one `data:` line from the SSE stream, optionally
/// preceded by an `event:` line (Anthropic format).
#[derive(Debug, Clone, PartialEq)]
pub struct SseChunk {
    /// The SSE event type, if present (e.g., "content_block_start").
    /// OpenAI streams do not use event types; Anthropic does.
    pub event: Option<String>,
    /// The raw data payload (everything after `data: `).
    pub data: String,
}

// ---------------------------------------------------------------------------
// Chunk classification
// ---------------------------------------------------------------------------

/// Classification of an SSE chunk, determining how the stream processor
/// should handle it.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    /// Text content chunk — pass through immediately, no buffering.
    TextContent,
    /// Tool call delta — buffer by tool call index until complete.
    ToolCallDelta(usize),
    /// Tool call is complete — release for validation before forwarding.
    ToolCallComplete(usize),
    /// SSE control event (e.g., message_start, message_stop) — pass through.
    Control,
    /// Not SSE at all (e.g., JSON error body) — pass through as-is.
    NonSse,
    /// Stream terminator (e.g., `data: [DONE]`) — pass through.
    Done,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of validating a complete tool call before release.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Tool call is allowed — forward to client.
    Allow,
    /// Tool call is blocked — include reason.
    Block(String),
}

/// How to handle blocked tool calls in streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallBlockMode {
    /// Emit an error event and suppress tool call chunks.
    Error,
    /// Suppress tool call chunks and inject a refusal text chunk.
    Rewrite,
}

/// Validates complete tool calls before they are released to the client.
///
/// Implementations are injected into the stream processor, enabling
/// L3-outbound validation without coupling the streaming module to
/// the constraint evaluator.
pub trait ToolCallValidator: Send + Sync {
    fn validate(&self, tool_call: &ToolCall) -> ValidationResult;
}

// ---------------------------------------------------------------------------
// Tool call buffer
// ---------------------------------------------------------------------------

/// Maximum size of a single tool call buffer in bytes.
pub const MAX_TOOL_CALL_BUFFER_BYTES: usize = 1_048_576; // 1 MB

/// Timeout for tool call delta arrival in seconds.
pub const TOOL_CALL_DELTA_TIMEOUT_SECS: u64 = 30;

/// Accumulates tool call deltas until the tool call is complete.
///
/// Each in-progress tool call gets its own buffer. The buffer tracks:
/// - The tool call ID and name (from the first delta)
/// - Accumulated arguments string (concatenated deltas)
/// - Total byte size for memory bounding
#[derive(Debug, Clone)]
pub struct ToolCallBuffer {
    /// Tool call ID (e.g., "call_abc123" for OpenAI, "toolu_abc" for Anthropic).
    pub id: String,
    /// Tool name (e.g., "read_file").
    pub name: String,
    /// Accumulated arguments JSON string, built from deltas.
    pub arguments_json: String,
    /// Total bytes accumulated (for memory bounding).
    pub total_bytes: usize,
    /// The original SSE chunks that make up this tool call, for replay on release.
    pub buffered_chunks: Vec<String>,
}

impl ToolCallBuffer {
    /// Create a new tool call buffer with the given ID and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments_json: String::new(),
            total_bytes: 0,
            buffered_chunks: Vec::new(),
        }
    }

    /// Append an arguments delta to the buffer.
    ///
    /// Returns an error if the buffer would exceed the maximum size.
    pub fn append_arguments(&mut self, delta: &str) -> Result<(), StreamError> {
        let new_size = self.total_bytes + delta.len();
        if new_size > MAX_TOOL_CALL_BUFFER_BYTES {
            return Err(StreamError::BufferOverflow {
                tool_name: self.name.clone(),
                tool_id: self.id.clone(),
                limit: MAX_TOOL_CALL_BUFFER_BYTES,
            });
        }
        self.arguments_json.push_str(delta);
        self.total_bytes = new_size;
        Ok(())
    }

    /// Buffer a raw SSE line for later replay.
    pub fn buffer_raw_line(&mut self, line: &str) {
        self.buffered_chunks.push(line.to_string());
    }

    /// Try to parse the accumulated arguments JSON into a serde_json::Value.
    ///
    /// This is called when the tool call is complete. If parsing fails,
    /// the tool call is released as an error.
    pub fn parse_arguments(&self) -> Result<serde_json::Value, StreamError> {
        if self.arguments_json.is_empty() {
            return Ok(serde_json::Value::Object(serde_json::Map::new()));
        }
        serde_json::from_str(&self.arguments_json).map_err(|e| StreamError::MalformedToolCall {
            tool_name: self.name.clone(),
            tool_id: self.id.clone(),
            reason: format!("invalid JSON arguments: {e}"),
        })
    }

}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during SSE stream processing.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamError {
    /// Tool call buffer exceeded the maximum size.
    BufferOverflow {
        tool_name: String,
        tool_id: String,
        limit: usize,
    },
    /// Tool call deltas stopped arriving (timeout).
    DeltaTimeout {
        tool_name: String,
        tool_id: String,
    },
    /// Tool call arguments could not be parsed as valid JSON.
    MalformedToolCall {
        tool_name: String,
        tool_id: String,
        reason: String,
    },
    /// Tool call was blocked by validation.
    ToolCallBlocked {
        tool_name: String,
        tool_id: String,
        reason: String,
    },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::BufferOverflow {
                tool_name,
                tool_id,
                limit,
            } => {
                write!(
                    f,
                    "tool call buffer overflow: tool '{tool_name}' (id: {tool_id}) \
                     exceeded {limit} byte limit"
                )
            }
            StreamError::DeltaTimeout {
                tool_name,
                tool_id,
            } => {
                write!(
                    f,
                    "tool call delta timeout: tool '{tool_name}' (id: {tool_id}) \
                     deltas stopped arriving for >{} seconds",
                    TOOL_CALL_DELTA_TIMEOUT_SECS
                )
            }
            StreamError::MalformedToolCall {
                tool_name,
                tool_id,
                reason,
            } => {
                write!(
                    f,
                    "malformed tool call: tool '{tool_name}' (id: {tool_id}): {reason}"
                )
            }
            StreamError::ToolCallBlocked {
                tool_name,
                tool_id,
                reason,
            } => {
                write!(
                    f,
                    "tool call blocked: tool '{tool_name}' (id: {tool_id}): {reason}"
                )
            }
        }
    }
}

impl std::error::Error for StreamError {}
