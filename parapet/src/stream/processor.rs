// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

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
#[cfg(not(windows))]
use crate::effect::{CallDispositionV0, StreamEnvelopeSessionV0, StreamFailureV0, StreamTouchV0};
use bytes::Bytes;
use std::collections::HashMap;
#[cfg(not(windows))]
use std::collections::VecDeque;
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
    /// Opt-in runtime envelope session; `None` for every existing constructor.
    #[cfg(not(windows))]
    envelope: std::sync::Mutex<Option<StreamEnvelopeSessionV0>>,
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
            #[cfg(not(windows))]
            envelope: std::sync::Mutex::new(None),
        }
    }

    /// Envelope-aware constructor (Routing contract 50747a52, section 4.4):
    /// every materialized call coordinate is finalized and settled by the
    /// session independently; an earlier released allow is never retracted.
    #[cfg(not(windows))]
    pub fn with_envelope(
        classifier: Arc<dyn ChunkClassifier>,
        validator: Arc<dyn ToolCallValidator>,
        block_mode: ToolCallBlockMode,
        session: StreamEnvelopeSessionV0,
    ) -> Self {
        let processor = Self::new(classifier, validator, block_mode);
        *processor.envelope.lock().expect("envelope slot") = Some(session);
        processor
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

        #[cfg(not(windows))]
        if let Some(session) = self.envelope.lock().expect("envelope slot").take() {
            tokio::spawn(envelope_task(input, session, block_mode, tx));
            return ReceiverStream::new(rx);
        }

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

// ---------------------------------------------------------------------------
// Envelope-aware processing (opt-in)
// ---------------------------------------------------------------------------

/// One complete SSE event, field lines plus its blank terminator, awaiting the
/// decisions of every call it touches. An event is released or dropped whole.
#[cfg(not(windows))]
struct PendingEvent {
    lines: Vec<String>,
    data: String,
    terminated: bool,
    touches: Vec<StreamTouchV0>,
}

#[cfg(not(windows))]
struct EnvelopeState {
    openai: bool,
    line_buffer: String,
    event_lines: Vec<String>,
    pending: VecDeque<PendingEvent>,
    decided: HashMap<StreamTouchV0, CallDispositionV0>,
    last_delta_time: Instant,
}

#[cfg(not(windows))]
async fn emit(tx: &mpsc::Sender<Bytes>, text: String) -> Result<(), ()> {
    tx.send(Bytes::from(text)).await.map_err(|_| ())
}

// Emit one event as a unit; the terminator is restored only if the source sent it.
#[cfg(not(windows))]
async fn emit_event(
    tx: &mpsc::Sender<Bytes>,
    lines: &[String],
    terminated: bool,
) -> Result<(), ()> {
    let mut text = String::new();
    for line in lines {
        text.push_str(line);
        text.push('\n');
    }
    if terminated {
        text.push('\n');
    }
    emit(tx, text).await
}

#[cfg(not(windows))]
async fn emit_closed_error(tx: &mpsc::Sender<Bytes>, text: &str) -> Result<(), ()> {
    emit(tx, format!("data: {{\"error\": \"{text}\"}}\n\n")).await
}

// Closed refusal for a withheld call, as complete SSE events.
#[cfg(not(windows))]
fn rewrite_events(refusal: &str, anthropic: bool, index: u32) -> Vec<Vec<String>> {
    let data = |value: serde_json::Value| format!("data: {value}");
    if anthropic {
        vec![
            vec![
                "event: content_block_start".to_string(),
                data(
                    serde_json::json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}}),
                ),
            ],
            vec![
                "event: content_block_delta".to_string(),
                data(
                    serde_json::json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": refusal}}),
                ),
            ],
            vec![
                "event: content_block_stop".to_string(),
                data(serde_json::json!({"type": "content_block_stop", "index": index})),
            ],
        ]
    } else {
        vec![vec![data(
            serde_json::json!({"choices": [{"delta": {"content": refusal},
            "finish_reason": null}]}),
        )]]
    }
}

// Queue an event that touches no call; it is released in arrival order.
#[cfg(not(windows))]
fn queue_event(state: &mut EnvelopeState, lines: Vec<String>) {
    state.pending.push_back(PendingEvent {
        lines,
        data: String::new(),
        terminated: true,
        touches: Vec::new(),
    });
}

// Value of an SSE field line (`name:` with one optional leading space).
#[cfg(not(windows))]
fn sse_field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let rest = line.trim().strip_prefix(name)?.strip_prefix(':')?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

// Remove withheld calls from a multi-call OpenAI payload; `None` drops the event.
#[cfg(not(windows))]
fn filter_openai_data(
    data: &str,
    decided: &HashMap<StreamTouchV0, CallDispositionV0>,
) -> Option<String> {
    let mut json: serde_json::Value = serde_json::from_str(data.trim()).ok()?;
    let mut kept = 0usize;
    for choice in json.get_mut("choices")?.as_array_mut()?.iter_mut() {
        let choice_index = choice.get("index").and_then(|i| i.as_u64());
        if let Some(calls) = choice
            .get_mut("delta")
            .and_then(|d| d.get_mut("tool_calls"))
            .and_then(|c| c.as_array_mut())
        {
            calls.retain(|call| {
                let index = call.get("index").and_then(|i| i.as_u64());
                let released = match (choice_index, index) {
                    (Some(c), Some(i)) => matches!(
                        decided.get(&StreamTouchV0::Formed(c as u32, i as u32)),
                        Some(CallDispositionV0::Release)
                    ),
                    _ => false,
                };
                kept += usize::from(released);
                released
            });
        }
    }
    (kept > 0).then(|| serde_json::to_string(&json).ok().unwrap_or_default())
}

// Release or drop queued events in arrival order; stops at the first event
// whose calls are not all decided, so released events keep their order.
#[cfg(not(windows))]
async fn flush_pending(state: &mut EnvelopeState, tx: &mpsc::Sender<Bytes>) -> Result<(), ()> {
    while let Some(pending) = state.pending.front() {
        let dispositions: Option<Vec<&CallDispositionV0>> = pending
            .touches
            .iter()
            .map(|t| state.decided.get(t))
            .collect();
        let Some(dispositions) = dispositions else {
            break;
        };
        let all_released = dispositions
            .iter()
            .all(|d| **d == CallDispositionV0::Release);
        let any_released = dispositions
            .iter()
            .any(|d| **d == CallDispositionV0::Release);
        let pending = state.pending.pop_front().expect("front event");
        if all_released {
            emit_event(tx, &pending.lines, pending.terminated).await?;
        } else if state.openai && any_released {
            if let Some(filtered) = filter_openai_data(&pending.data, &state.decided) {
                let mut lines = Vec::new();
                let mut placed = false;
                for line in &pending.lines {
                    if sse_field(line, "data").is_none() {
                        lines.push(line.clone());
                    } else if !placed {
                        lines.push(format!("data: {filtered}"));
                        placed = true;
                    }
                }
                emit_event(tx, &lines, pending.terminated).await?;
            }
        }
    }
    Ok(())
}

// Record finalized calls; each withheld call queues its closed notice.
#[cfg(not(windows))]
fn record_decisions(
    state: &mut EnvelopeState,
    decided: Vec<(StreamTouchV0, CallDispositionV0)>,
    block_mode: ToolCallBlockMode,
) {
    for (touch, disposition) in decided {
        if let CallDispositionV0::Withhold(code) = &disposition {
            match block_mode {
                ToolCallBlockMode::Error => {
                    let text = format!("tool call blocked by policy ({code})");
                    queue_event(state, vec![format!("data: {{\"error\": \"{text}\"}}")]);
                }
                ToolCallBlockMode::Rewrite => {
                    let index = match touch {
                        StreamTouchV0::Formed(_, i) => i,
                        StreamTouchV0::Unformable(_) => 0,
                    };
                    let refusal = format!("Tool call was blocked by policy ({code})");
                    for lines in rewrite_events(&refusal, !state.openai, index) {
                        queue_event(state, lines);
                    }
                }
            }
        }
        state.decided.insert(touch, disposition);
    }
}

// Settle one complete event; returns whether a call overflowed its buffer.
#[cfg(not(windows))]
async fn envelope_event(
    lines: Vec<String>,
    terminated: bool,
    state: &mut EnvelopeState,
    session: &mut StreamEnvelopeSessionV0,
    block_mode: ToolCallBlockMode,
    tx: &mpsc::Sender<Bytes>,
) -> Result<bool, ()> {
    let mut event: Option<String> = None;
    let mut data: Option<String> = None;
    for line in &lines {
        if let Some(name) = sse_field(line, "event") {
            event = Some(name.trim().to_string());
        } else if let Some(value) = sse_field(line, "data") {
            match &mut data {
                Some(joined) => {
                    joined.push('\n');
                    joined.push_str(value);
                }
                None => data = Some(value.to_string()),
            }
        }
    }
    let data = data.filter(|d| d.trim() != "[DONE]");
    let step = match &data {
        Some(data) => session.ingest(event.as_deref(), data),
        None => Default::default(),
    };
    record_decisions(state, step.decided, block_mode);
    if !step.touches.is_empty() {
        state.last_delta_time = Instant::now();
    }
    // Pass-through events (no touches) queue too, so arrival order is kept.
    state.pending.push_back(PendingEvent {
        lines,
        data: data.unwrap_or_default(),
        terminated,
        touches: step.touches,
    });
    flush_pending(state, tx).await?;
    Ok(step.overflow)
}

// Collect field lines; a blank line completes the event under construction.
#[cfg(not(windows))]
async fn envelope_line(
    line: String,
    state: &mut EnvelopeState,
    session: &mut StreamEnvelopeSessionV0,
    block_mode: ToolCallBlockMode,
    tx: &mpsc::Sender<Bytes>,
) -> Result<bool, ()> {
    if !line.trim().is_empty() {
        state.event_lines.push(line);
        return Ok(false);
    }
    let lines = std::mem::take(&mut state.event_lines);
    if lines.is_empty() {
        // A stray blank line dispatches nothing; it keeps its place in order.
        queue_event(state, lines);
        flush_pending(state, tx).await?;
        return Ok(false);
    }
    envelope_event(lines, true, state, session, block_mode, tx).await
}

#[cfg(not(windows))]
async fn envelope_abort(
    state: &mut EnvelopeState,
    session: &mut StreamEnvelopeSessionV0,
    failure: StreamFailureV0,
    block_mode: ToolCallBlockMode,
    tx: &mpsc::Sender<Bytes>,
) {
    let decided = session.abort(failure);
    record_decisions(state, decided, block_mode);
    let _ = flush_pending(state, tx).await;
}

#[cfg(not(windows))]
async fn envelope_task(
    mut input: impl Stream<Item = Bytes> + Unpin + Send + 'static,
    mut session: StreamEnvelopeSessionV0,
    block_mode: ToolCallBlockMode,
    tx: mpsc::Sender<Bytes>,
) {
    let Some(first_chunk) = input.next().await else {
        session.finish();
        return;
    };
    if !is_sse_content(&first_chunk) {
        if tx.send(first_chunk).await.is_ok() {
            while let Some(chunk) = input.next().await {
                if tx.send(chunk).await.is_err() {
                    break;
                }
            }
        }
        session.finish();
        return;
    }
    let mut state = EnvelopeState {
        openai: session.provider() == crate::receipt::Provider::Openai,
        line_buffer: String::new(),
        event_lines: Vec::new(),
        pending: VecDeque::new(),
        decided: HashMap::new(),
        last_delta_time: Instant::now(),
    };
    let timeout = Duration::from_secs(TOOL_CALL_DELTA_TIMEOUT_SECS);
    let mut chunk = Some(first_chunk);
    while let Some(bytes) = chunk.take() {
        state.line_buffer.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(pos) = state.line_buffer.find('\n') {
            let line = state.line_buffer[..pos].to_string();
            state.line_buffer.drain(..=pos);
            match envelope_line(line, &mut state, &mut session, block_mode, &tx).await {
                Ok(false) => {}
                Ok(true) => {
                    envelope_abort(
                        &mut state,
                        &mut session,
                        StreamFailureV0::StreamAborted,
                        block_mode,
                        &tx,
                    )
                    .await;
                    let _ = emit_closed_error(&tx, "tool call buffer overflow").await;
                    session.finish();
                    return;
                }
                Err(()) => {
                    envelope_abort(
                        &mut state,
                        &mut session,
                        StreamFailureV0::StreamAborted,
                        block_mode,
                        &tx,
                    )
                    .await;
                    session.finish();
                    return;
                }
            }
        }
        // While a call is open, the next chunk races the delta deadline so a
        // stalled provider cannot hold the candidate open indefinitely.
        chunk = if session.has_open() {
            let deadline = state.last_delta_time + timeout;
            match tokio::time::timeout_at(deadline, input.next()).await {
                Ok(next) => next,
                Err(_) => {
                    envelope_abort(
                        &mut state,
                        &mut session,
                        StreamFailureV0::DeltaTimeout,
                        block_mode,
                        &tx,
                    )
                    .await;
                    let _ = emit_closed_error(&tx, "tool call delta timeout").await;
                    session.finish();
                    return;
                }
            }
        } else {
            input.next().await
        };
    }
    if !state.line_buffer.trim().is_empty() {
        let line = std::mem::take(&mut state.line_buffer);
        state.event_lines.push(line);
    }
    let lines = std::mem::take(&mut state.event_lines);
    if !lines.is_empty() {
        let _ = envelope_event(lines, false, &mut state, &mut session, block_mode, &tx).await;
    }
    envelope_abort(
        &mut state,
        &mut session,
        StreamFailureV0::StreamAborted,
        block_mode,
        &tx,
    )
    .await;
    session.finish();
}
