// Stream processor — M1.5
//
// Takes an input byte stream, classifies SSE chunks, buffers tool
// call deltas, validates complete tool calls, and produces an output
// byte stream with text passed through immediately and tool calls
// released after validation.

use super::classifier::{is_sse_content, parse_sse_line, ChunkClassifier};
use super::types::{
    ChunkType, StreamError, ToolCallBlockMode, ToolCallBuffer, ToolCallValidator, ValidationResult,
    TOOL_CALL_DELTA_TIMEOUT_SECS,
};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

/// Processes an SSE stream, buffering tool calls for validation.
///
/// The processor is constructed with a classifier (provider-specific)
/// and a validator (L3-outbound). It consumes an input stream of bytes
/// and produces an output stream of bytes.
pub struct StreamProcessor {
    classifier: Arc<dyn ChunkClassifier>,
    validator: Arc<dyn ToolCallValidator>,
    block_mode: ToolCallBlockMode,
}

impl StreamProcessor {
    /// Create a new stream processor with injected dependencies.
    pub fn new(
        classifier: Arc<dyn ChunkClassifier>,
        validator: Arc<dyn ToolCallValidator>,
        block_mode: ToolCallBlockMode,
    ) -> Self {
        Self {
            classifier,
            validator,
            block_mode,
        }
    }

    /// Process an input byte stream, producing an output byte stream.
    ///
    /// Text content is passed through immediately. Tool call deltas are
    /// buffered until complete, validated, then released or blocked.
    ///
    /// Non-SSE responses are detected and passed through without parsing.
    pub fn process(
        &self,
        mut input: impl Stream<Item = Bytes> + Unpin + Send + 'static,
    ) -> impl Stream<Item = Bytes> {
        let classifier = self.classifier.clone();
        let validator = self.validator.clone();

        let (tx, rx) = mpsc::channel::<Bytes>(64);

        let block_mode = self.block_mode;

        tokio::spawn(async move {
            // Accumulate the first chunk to detect SSE vs non-SSE
            let first_chunk = match input.next().await {
                Some(chunk) => chunk,
                None => return, // Empty stream
            };

            // Check if this is an SSE stream
            if !is_sse_content(&first_chunk) {
                // Non-SSE: pass through everything
                if tx.send(first_chunk).await.is_err() {
                    return;
                }
                while let Some(chunk) = input.next().await {
                    if tx.send(chunk).await.is_err() {
                        break; // Client disconnected
                    }
                }
                return;
            }

            // SSE stream processing state
            let mut state = ProcessingState::new();

            // Process first chunk
            state.line_buffer.push_str(&String::from_utf8_lossy(&first_chunk));

            // Drain and process complete lines
            if drain_lines(
                &mut state,
                classifier.as_ref(),
                validator.as_ref(),
                block_mode,
                &tx,
            )
                .await
                .is_err()
            {
                return;
            }

            // Process remaining chunks
            let timeout = Duration::from_secs(TOOL_CALL_DELTA_TIMEOUT_SECS);
            while let Some(chunk) = input.next().await {
                // Check timeout for in-progress tool calls
                if !state.tool_buffers.is_empty() && state.last_delta_time.elapsed() > timeout {
                    if let Some((_idx, buffer)) = state.tool_buffers.iter().next() {
                        let err = StreamError::DeltaTimeout {
                            tool_name: buffer.name.clone(),
                            tool_id: buffer.id.clone(),
                        };
                        if send_error(&tx, &err).await.is_err() {
                            return;
                        }
                        return;
                    }
                }

                state
                    .line_buffer
                    .push_str(&String::from_utf8_lossy(&chunk));

                if drain_lines(
                    &mut state,
                    classifier.as_ref(),
                    validator.as_ref(),
                    block_mode,
                    &tx,
                )
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // Process any remaining partial line
            if !state.line_buffer.is_empty() {
                let line = std::mem::take(&mut state.line_buffer);
                let _ = process_single_line(
                    &line,
                    &mut state,
                    classifier.as_ref(),
                    validator.as_ref(),
                    block_mode,
                    &tx,
                )
                .await;
            }
        });

        ReceiverStream::new(rx)
    }
}

// ---------------------------------------------------------------------------
// Processing state
// ---------------------------------------------------------------------------

/// Mutable state carried through the SSE processing loop.
struct ProcessingState {
    /// Partial line buffer (data between newlines).
    line_buffer: String,
    /// Pending event line for Anthropic-style event+data pairs.
    /// Stored so we can buffer it with tool call data lines.
    pending_event_line: Option<String>,
    /// In-progress tool call buffers, keyed by content block index.
    tool_buffers: HashMap<usize, ToolCallBuffer>,
    /// Track which indices are tool_use blocks (vs text).
    tool_use_indices: std::collections::HashSet<usize>,
    /// Time of last tool call delta (for timeout detection).
    last_delta_time: Instant,
}

impl ProcessingState {
    fn new() -> Self {
        Self {
            line_buffer: String::new(),
            pending_event_line: None,
            tool_buffers: HashMap::new(),
            tool_use_indices: Default::default(),
            last_delta_time: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Line processing
// ---------------------------------------------------------------------------

/// Drain all complete lines from the buffer and process each one.
async fn drain_lines(
    state: &mut ProcessingState,
    classifier: &dyn ChunkClassifier,
    validator: &dyn ToolCallValidator,
    block_mode: ToolCallBlockMode,
    tx: &mpsc::Sender<Bytes>,
) -> Result<(), ()> {
    loop {
        if let Some(newline_pos) = state.line_buffer.find('\n') {
            let line = state.line_buffer[..newline_pos].to_string();
            state.line_buffer = state.line_buffer[newline_pos + 1..].to_string();

            process_single_line(line.as_str(), state, classifier, validator, block_mode, tx)
                .await?;
        } else {
            break;
        }
    }
    Ok(())
}

/// Process a single SSE line.
///
/// Handles Anthropic event+data pairing: event: lines are stored, data: lines
/// are classified together with the pending event. For tool call chunks, both
/// event and data lines are buffered together.
async fn process_single_line(
    line: &str,
    state: &mut ProcessingState,
    classifier: &dyn ChunkClassifier,
    validator: &dyn ToolCallValidator,
    block_mode: ToolCallBlockMode,
    tx: &mpsc::Sender<Bytes>,
) -> Result<(), ()> {
    let trimmed = line.trim();

    // Empty line: SSE event separator
    if trimmed.is_empty() {
        if tx.send(Bytes::from("\n")).await.is_err() {
            return Err(());
        }
        return Ok(());
    }

    // Anthropic event: line — store it, do NOT pass through yet.
    // We will emit it when we see the matching data: line and know
    // whether to pass through or buffer.
    if trimmed.starts_with("event:") {
        state.pending_event_line = Some(line.to_string());
        return Ok(());
    }

    // data: line — might be standalone (OpenAI) or paired with event (Anthropic)
    if let Some(sse_chunk) = parse_sse_line(trimmed) {
        // Take the pending event line, if any
        let event_line = state.pending_event_line.take();

        // Build the classified chunk
        let chunk = if let Some(ref ev_line) = event_line {
            let event_str = ev_line
                .trim()
                .strip_prefix("event: ")
                .or_else(|| ev_line.trim().strip_prefix("event:"))
                .unwrap_or("")
                .to_string();
            super::types::SseChunk {
                event: Some(event_str),
                data: sse_chunk.data,
            }
        } else {
            sse_chunk
        };

        let chunk_type = classifier.classify(&chunk);

        match chunk_type {
            ChunkType::TextContent | ChunkType::Control | ChunkType::Done | ChunkType::NonSse => {
                // Pass through: emit event line (if any) then data line
                if let Some(ev_line) = &event_line {
                    if tx.send(Bytes::from(format!("{ev_line}\n"))).await.is_err() {
                        return Err(());
                    }
                }
                if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
                    return Err(());
                }
            }
            ChunkType::ToolCallDelta(index) => {
                state.last_delta_time = Instant::now();
                state.tool_use_indices.insert(index);

                // Parse the data payload
                let data: serde_json::Value = match serde_json::from_str(chunk.data.trim()) {
                    Ok(v) => v,
                    Err(_) => {
                        // Can't parse — pass through both lines
                        if let Some(ev_line) = &event_line {
                            if tx.send(Bytes::from(format!("{ev_line}\n"))).await.is_err() {
                                return Err(());
                            }
                        }
                        if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
                            return Err(());
                        }
                        return Ok(());
                    }
                };

                // Get or create tool call buffer
                if !state.tool_buffers.contains_key(&index) {
                    let (id, name) = extract_tool_call_start(&data, chunk.event.as_deref());
                    state
                        .tool_buffers
                        .insert(index, ToolCallBuffer::new(id, name));
                }

                let buffer = state.tool_buffers.get_mut(&index).unwrap();

                // Extract and append arguments delta
                let args_delta = extract_arguments_delta(&data, chunk.event.as_deref());
                if !args_delta.is_empty() {
                    if let Err(e) = buffer.append_arguments(&args_delta) {
                        // Buffer overflow — send error, abort stream
                        send_error(tx, &e).await?;
                        return Err(());
                    }
                }

                // Buffer both event and data lines for replay on release
                if let Some(ev_line) = &event_line {
                    buffer.buffer_raw_line(ev_line);
                }
                buffer.buffer_raw_line(line);
            }
            ChunkType::ToolCallComplete(index) => {
                // Determine which indices to complete
                let indices_to_complete: Vec<usize> = if chunk.event.is_some() {
                    // Anthropic: complete the specific index, but only if it is
                    // a tool_use block (text blocks also get content_block_stop)
                    if state.tool_use_indices.contains(&index) {
                        vec![index]
                    } else {
                        // Not a tool_use block — pass through
                        if let Some(ev_line) = &event_line {
                            if tx.send(Bytes::from(format!("{ev_line}\n"))).await.is_err() {
                                return Err(());
                            }
                        }
                        if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
                            return Err(());
                        }
                        return Ok(());
                    }
                } else {
                    // OpenAI: complete all in-progress buffers
                    state.tool_buffers.keys().cloned().collect()
                };

                let mut any_allowed = false;
                for idx in indices_to_complete {
                    if let Some(buffer) = state.tool_buffers.remove(&idx) {
                        state.tool_use_indices.remove(&idx);
                        let buffered_lines = buffer.buffered_chunks.clone();

                        match buffer.parse_arguments() {
                            Ok(arguments) => {
                                let tool_call = crate::message::ToolCall {
                                    id: buffer.id.clone(),
                                    name: buffer.name.clone(),
                                    arguments,
                                };

                                match validator.validate(&tool_call) {
                                    ValidationResult::Allow => {
                                        any_allowed = true;
                                        // Release all buffered lines (event+data pairs)
                                        for buffered_line in &buffered_lines {
                                            if tx
                                                .send(Bytes::from(format!("{buffered_line}\n")))
                                                .await
                                                .is_err()
                                            {
                                                return Err(());
                                            }
                                        }
                                    }
                                    ValidationResult::Block(reason) => {
                                        match block_mode {
                                            ToolCallBlockMode::Error => {
                                                let err = StreamError::ToolCallBlocked {
                                                    tool_name: tool_call.name,
                                                    tool_id: tool_call.id,
                                                    reason,
                                                };
                                                send_error(tx, &err).await?;
                                                // Do NOT emit the buffered lines --
                                                // tool call is suppressed
                                            }
                                            ToolCallBlockMode::Rewrite => {
                                                let refusal = format!(
                                                    "Tool '{}' was blocked by policy: {}",
                                                    tool_call.name, reason
                                                );
                                                send_rewrite(
                                                    tx,
                                                    &refusal,
                                                    &event_line,
                                                    idx,
                                                )
                                                .await?;
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                send_error(tx, &e).await?;
                            }
                        }
                    }
                }

                // Emit the completion event itself.
                //
                // For Anthropic: content_block_stop is per-block, so only
                // emit it if the tool call was allowed.
                //
                // For OpenAI: finish_reason is a structural event that the
                // client needs regardless -- always emit it.
                if chunk.event.is_some() {
                    // Anthropic: only emit if the tool call was allowed
                    if any_allowed {
                        if let Some(ev_line) = &event_line {
                            if tx.send(Bytes::from(format!("{ev_line}\n"))).await.is_err() {
                                return Err(());
                            }
                        }
                        if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
                            return Err(());
                        }
                    }
                } else {
                    // OpenAI: always emit the finish_reason line
                    if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
                        return Err(());
                    }
                }
            }
        }
    } else {
        // Not a recognized SSE line — pass through, plus any pending event
        if let Some(ev_line) = state.pending_event_line.take() {
            if tx.send(Bytes::from(format!("{ev_line}\n"))).await.is_err() {
                return Err(());
            }
        }
        if tx.send(Bytes::from(format!("{line}\n"))).await.is_err() {
            return Err(());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Send an error as an SSE data line.
async fn send_error(tx: &mpsc::Sender<Bytes>, err: &StreamError) -> Result<(), ()> {
    let error_line = format!(
        "data: {{\"error\": \"{}\"}}\n",
        err.to_string().replace('"', "\\\"")
    );
    tx.send(Bytes::from(error_line)).await.map_err(|_| ())
}

async fn send_rewrite(
    tx: &mpsc::Sender<Bytes>,
    refusal: &str,
    event_line: &Option<String>,
    index: usize,
) -> Result<(), ()> {
    if event_line.is_some() {
        // Anthropic SSE: emit a synthetic text block (start + delta + stop).
        let start_ev = "event: content_block_start\n";
        let start_data = format!(
            "data: {{\"type\":\"content_block_start\",\"index\":{index},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n"
        );
        let delta_ev = "event: content_block_delta\n";
        let delta_data = format!(
            "data: {{\"type\":\"content_block_delta\",\"index\":{index},\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n",
            refusal.replace('"', "\\\"")
        );
        let stop_ev = "event: content_block_stop\n";
        let stop_data = format!(
            "data: {{\"type\":\"content_block_stop\",\"index\":{index}}}\n"
        );
        tx.send(Bytes::from(start_ev)).await.map_err(|_| ())?;
        tx.send(Bytes::from(start_data)).await.map_err(|_| ())?;
        tx.send(Bytes::from(delta_ev)).await.map_err(|_| ())?;
        tx.send(Bytes::from(delta_data)).await.map_err(|_| ())?;
        tx.send(Bytes::from(stop_ev)).await.map_err(|_| ())?;
        tx.send(Bytes::from(stop_data)).await.map_err(|_| ())?;
    } else {
        // OpenAI SSE: emit a text delta chunk.
        let data = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n",
            refusal.replace('"', "\\\"")
        );
        tx.send(Bytes::from(data)).await.map_err(|_| ())?;
    }
    Ok(())
}

/// Extract tool call ID and name from the first delta chunk.
fn extract_tool_call_start(data: &serde_json::Value, event: Option<&str>) -> (String, String) {
    match event {
        Some("content_block_start") => {
            // Anthropic: content_block.{id, name}
            let id = data
                .get("content_block")
                .and_then(|cb| cb.get("id"))
                .and_then(|id| id.as_str())
                .unwrap_or("")
                .to_string();
            let name = data
                .get("content_block")
                .and_then(|cb| cb.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            (id, name)
        }
        _ => {
            // OpenAI: choices[0].delta.tool_calls[0].{id, function.name}
            let tool_call = data
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("tool_calls"))
                .and_then(|tc| tc.get(0));

            let id = tool_call
                .and_then(|tc| tc.get("id"))
                .and_then(|id| id.as_str())
                .unwrap_or("")
                .to_string();
            let name = tool_call
                .and_then(|tc| tc.get("function"))
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            (id, name)
        }
    }
}

/// Extract arguments delta string from an SSE data payload.
fn extract_arguments_delta(data: &serde_json::Value, event: Option<&str>) -> String {
    match event {
        Some("content_block_delta") => {
            // Anthropic: delta.partial_json
            data.get("delta")
                .and_then(|d| d.get("partial_json"))
                .and_then(|pj| pj.as_str())
                .unwrap_or("")
                .to_string()
        }
        _ => {
            // OpenAI: choices[0].delta.tool_calls[0].function.arguments
            data.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("tool_calls"))
                .and_then(|tc| tc.get(0))
                .and_then(|tc| tc.get("function"))
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("")
                .to_string()
        }
    }
}
