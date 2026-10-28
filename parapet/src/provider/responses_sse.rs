// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API SSE framer/collector -- D-0027 OA-R1
//
// Responsibilities:
// - Incrementally consume arbitrary upstream byte chunks and frame them as
//   Server-Sent Events: LF and CRLF line endings across any chunk boundary,
//   `:` comment lines ignored (comment-only keepalive frames never reach the
//   decoder), multiline `data:` joined by one LF, and `event_name = None`
//   when a frame has no `event:` field
// - Enforce injected inclusive bounds on total stream bytes and on bytes per
//   frame BEFORE any buffer grows past them (sensor C-7, D-1)
// - Accept and drop `id:` and `retry:` fields; fail closed on a bare CR,
//   malformed UTF-8 field names or event values, duplicate `event:` fields,
//   missing or empty data on a data frame, any other field, truncated EOF, a
//   stream with no terminal event, and any byte after the terminal frame
// - Pass every data frame through an injected [`StreamEventDecoder`] and
//   [`StreamAssembler`]; any error poisons the collector
// - Retain only canonical validated frames, never upstream bytes: each
//   accepted event becomes `event: <payload type>` plus its joined data
//   payload, in event order; comments, `id:`, `retry:`, and line-ending
//   choices disappear. Opaque reasoning `encrypted_content` accepted by the
//   default decoder stays only inside these original payloads (and the
//   assembler's withheld terminal continuation); it is never rewritten
// - Release canonical frames from [`SseCollector::finish`] only for a
//   `response.completed` or `response.incomplete` terminal; `response.failed`
//   and top-level `error` finish with a fixed refusal and release nothing
//
// Not in scope: route/proxy activation, request rewriting, output policy,
// concurrent-stream caps (D-2), and incremental release (D-5). This module is
// an inert seam; nothing calls it on live traffic yet. A clean finish means
// the stream is well framed and self-consistent, NOT that it is safe.

use super::responses_stream::{OpenAiResponsesStreamDecoder, StreamEventDecoder, StreamEventKind};
use super::responses_stream_state::{
    Progress, ResponsesStreamAssembler, StreamAssembler, StreamStateError, Terminal,
};
use bytes::{Bytes, BytesMut};
use std::fmt;

/// Frames, decodes, and assembles one upstream Responses SSE stream.
pub trait SseCollector {
    /// Absorb one upstream chunk of any size, including empty. The result is
    /// structural only. Any error poisons the collector; every later call
    /// fails.
    fn push(&mut self, chunk: &[u8]) -> Result<Progress, SseCollectError>;

    /// Signal clean upstream EOF. Fails unless the stream ended exactly at the
    /// end of a validated `completed` or `incomplete` terminal frame and no
    /// error occurred.
    fn finish(self) -> Result<CollectedStream, SseCollectError>
    where
        Self: Sized;
}

/// Hard byte bounds. Every bound is inclusive: exactly `max` is accepted,
/// `max + 1` fails closed.
///
/// [`Default`] values are conservative local safety limits for a single
/// proxied stream, not a claim about any provider's maximums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SseLimits {
    /// Total upstream bytes, including comments and line endings. Also bounds
    /// the retained canonical frames.
    pub max_stream_bytes: usize,
    /// Bytes of one frame: every non-blank line of the frame, comments
    /// included, with its line ending. The terminating blank line is not
    /// counted. A frame's joined data payload is always within this bound.
    pub max_frame_bytes: usize,
}

impl SseLimits {
    /// Reviewed defaults; also the ceiling the runtime config may tighten.
    pub const DEFAULT: Self = Self {
        max_stream_bytes: 40 * 1024 * 1024,
        max_frame_bytes: 16 * 1024 * 1024,
    };
}

impl Default for SseLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why the collector refused. Fixed messages only; never carries stream
/// bytes, ids, text, or digests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SseCollectError {
    #[error("sse: collector already failed")]
    Poisoned,
    #[error("sse: stream byte limit exceeded")]
    StreamTooLarge,
    #[error("sse: frame byte limit exceeded")]
    FrameTooLarge,
    #[error("sse: carriage return not followed by line feed")]
    BareCarriageReturn,
    #[error("sse: field name is not valid UTF-8")]
    MalformedFieldName,
    #[error("sse: event field is empty or not valid UTF-8")]
    InvalidEventName,
    #[error("sse: frame has more than one event field")]
    DuplicateEvent,
    #[error("sse: frame has an unsupported field")]
    UnsupportedField,
    #[error("sse: event frame has no data field")]
    MissingData,
    #[error("sse: frame data is empty")]
    EmptyData,
    #[error("sse: frame data is not an accepted Responses event")]
    Decode,
    #[error("sse: {0}")]
    Assembly(StreamStateError),
    #[error("sse: content after the terminal event")]
    AfterTerminal,
    #[error("sse: stream ended inside a frame")]
    Truncated,
    #[error("sse: stream ended without a terminal event")]
    NoTerminal,
    #[error("sse: upstream response failed")]
    UpstreamFailed,
    #[error("sse: upstream stream reported an error")]
    UpstreamError,
}

/// A clean stream that ended in `response.completed` or
/// `response.incomplete`: the assembler's [`Terminal`] plus the canonical
/// validated frames it was assembled from.
///
/// `Debug` is redacted to the terminal kind and byte count.
pub struct CollectedStream {
    terminal: Terminal,
    canonical: Bytes,
}

impl CollectedStream {
    /// Always [`Terminal::Completed`] or [`Terminal::Incomplete`].
    pub fn terminal(&self) -> &Terminal {
        &self.terminal
    }

    /// Every accepted event, in order, as `event: <type>\ndata: <payload>\n\n`.
    /// A payload joined from several `data:` lines is re-split at each LF so
    /// it re-frames to the same payload.
    pub fn canonical(&self) -> &Bytes {
        &self.canonical
    }

    pub fn into_parts(self) -> (Terminal, Bytes) {
        (self.terminal, self.canonical)
    }
}

impl fmt::Debug for CollectedStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terminal = match self.terminal {
            Terminal::Completed(_) => "completed",
            Terminal::Incomplete(_) => "incomplete",
            Terminal::Failed | Terminal::Errored => "refused",
        };
        f.debug_struct("CollectedStream")
            .field("terminal", &terminal)
            .field("canonical_bytes", &self.canonical.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Frame {
    /// Counted bytes so far (see [`SseLimits::max_frame_bytes`]).
    bytes: usize,
    /// Whether any field or comment line was seen.
    any_line: bool,
    event: Option<String>,
    data: Option<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Open,
    Terminal,
    Poisoned,
}

/// Reference [`SseCollector`] for OpenAI Responses streams.
///
/// `Debug` is redacted: it shows stage, byte counters, and limits only.
pub struct ResponsesSseCollector<D = OpenAiResponsesStreamDecoder, A = ResponsesStreamAssembler>
where
    D: StreamEventDecoder,
    A: StreamAssembler,
{
    limits: SseLimits,
    decoder: D,
    assembler: A,
    stage: Stage,
    /// Upstream bytes consumed so far.
    stream_bytes: usize,
    /// Canonical frames of accepted events; emptied on a failed terminal.
    canonical: BytesMut,
    /// Current line content, line ending excluded.
    line: Vec<u8>,
    /// A CR was read and its LF is still expected.
    pending_cr: bool,
    frame: Frame,
}

impl ResponsesSseCollector {
    /// Collector with [`SseLimits::default`], the strict decoder, and an
    /// assembler with default limits.
    pub fn new() -> Self {
        Self::with_limits(SseLimits::default())
    }

    pub fn with_limits(limits: SseLimits) -> Self {
        Self::with_parts(
            limits,
            OpenAiResponsesStreamDecoder::new(),
            ResponsesStreamAssembler::new(),
        )
    }
}

impl Default for ResponsesSseCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: StreamEventDecoder, A: StreamAssembler> ResponsesSseCollector<D, A> {
    pub fn with_parts(limits: SseLimits, decoder: D, assembler: A) -> Self {
        Self {
            limits,
            decoder,
            assembler,
            stage: Stage::Open,
            stream_bytes: 0,
            canonical: BytesMut::new(),
            line: Vec::new(),
            pending_cr: false,
            frame: Frame::default(),
        }
    }
}

impl<D: StreamEventDecoder, A: StreamAssembler> fmt::Debug for ResponsesSseCollector<D, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage = match self.stage {
            Stage::Open => "open",
            Stage::Terminal => "terminal",
            Stage::Poisoned => "poisoned",
        };
        f.debug_struct("ResponsesSseCollector")
            .field("stage", &stage)
            .field("stream_bytes", &self.stream_bytes)
            .field("canonical_bytes", &self.canonical.len())
            .field("frame_bytes", &(self.frame.bytes + self.line.len()))
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl<D: StreamEventDecoder, A: StreamAssembler> SseCollector for ResponsesSseCollector<D, A> {
    fn push(&mut self, chunk: &[u8]) -> Result<Progress, SseCollectError> {
        let result = self.accept(chunk);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn finish(self) -> Result<CollectedStream, SseCollectError> {
        match self.stage {
            Stage::Poisoned => Err(SseCollectError::Poisoned),
            Stage::Open if self.pending_cr || !self.line.is_empty() || self.frame.any_line => {
                Err(SseCollectError::Truncated)
            }
            Stage::Open => Err(SseCollectError::NoTerminal),
            Stage::Terminal => match self.assembler.finish().map_err(SseCollectError::Assembly)? {
                Terminal::Failed => Err(SseCollectError::UpstreamFailed),
                Terminal::Errored => Err(SseCollectError::UpstreamError),
                terminal => Ok(CollectedStream {
                    terminal,
                    canonical: self.canonical.freeze(),
                }),
            },
        }
    }
}

impl<D: StreamEventDecoder, A: StreamAssembler> ResponsesSseCollector<D, A> {
    fn poison(&mut self) {
        self.stage = Stage::Poisoned;
        self.canonical = BytesMut::new();
        self.line = Vec::new();
        self.frame = Frame::default();
    }

    fn accept(&mut self, chunk: &[u8]) -> Result<Progress, SseCollectError> {
        match self.stage {
            Stage::Poisoned => return Err(SseCollectError::Poisoned),
            Stage::Terminal if !chunk.is_empty() => return Err(SseCollectError::AfterTerminal),
            Stage::Terminal => return Ok(Progress::Terminal),
            Stage::Open => {}
        }
        if chunk.len() > self.limits.max_stream_bytes - self.stream_bytes {
            return Err(SseCollectError::StreamTooLarge);
        }
        self.stream_bytes += chunk.len();
        for &byte in chunk {
            // The terminal frame's blank line was earlier in this chunk.
            if self.stage == Stage::Terminal {
                return Err(SseCollectError::AfterTerminal);
            }
            if self.pending_cr {
                if byte != b'\n' {
                    return Err(SseCollectError::BareCarriageReturn);
                }
                self.pending_cr = false;
                self.end_line(2)?;
                continue;
            }
            match byte {
                b'\r' => self.pending_cr = true,
                b'\n' => self.end_line(1)?,
                _ => {
                    if self.frame.bytes + self.line.len() >= self.limits.max_frame_bytes {
                        return Err(SseCollectError::FrameTooLarge);
                    }
                    self.line.push(byte);
                }
            }
        }
        Ok(if self.stage == Stage::Terminal {
            Progress::Terminal
        } else {
            Progress::Open
        })
    }

    /// Complete the current line, whose ending was `ending` bytes long.
    fn end_line(&mut self, ending: usize) -> Result<(), SseCollectError> {
        if self.line.is_empty() {
            return self.dispatch();
        }
        let bytes = self.frame.bytes + self.line.len() + ending;
        if bytes > self.limits.max_frame_bytes {
            return Err(SseCollectError::FrameTooLarge);
        }
        self.frame.bytes = bytes;
        self.frame.any_line = true;
        let line = std::mem::take(&mut self.line);
        let result = self.field(&line);
        // Reuse the line allocation; it never exceeds the frame bound.
        self.line = line;
        self.line.clear();
        result
    }

    /// Apply one non-blank line to the current frame.
    fn field(&mut self, line: &[u8]) -> Result<(), SseCollectError> {
        if line[0] == b':' {
            return Ok(());
        }
        let (name, value) = match line.iter().position(|&b| b == b':') {
            Some(colon) => {
                let value = &line[colon + 1..];
                (&line[..colon], value.strip_prefix(b" ").unwrap_or(value))
            }
            None => (line, &[][..]),
        };
        let name = std::str::from_utf8(name).map_err(|_| SseCollectError::MalformedFieldName)?;
        match name {
            "event" => {
                if self.frame.event.is_some() {
                    return Err(SseCollectError::DuplicateEvent);
                }
                let event = std::str::from_utf8(value)
                    .ok()
                    .filter(|e| !e.is_empty())
                    .ok_or(SseCollectError::InvalidEventName)?;
                self.frame.event = Some(event.to_string());
            }
            "data" => match &mut self.frame.data {
                Some(data) => {
                    data.push(b'\n');
                    data.extend_from_slice(value);
                }
                None => self.frame.data = Some(value.to_vec()),
            },
            // Reconnection hints; this collector never reconnects.
            "id" | "retry" => {}
            _ => return Err(SseCollectError::UnsupportedField),
        }
        Ok(())
    }

    /// A blank line ends the frame. Comment-, `id:`-, and `retry:`-only frames
    /// are keepalives and never reach the decoder.
    fn dispatch(&mut self) -> Result<(), SseCollectError> {
        let frame = std::mem::take(&mut self.frame);
        let (event, data) = match (frame.event, frame.data) {
            (None, None) => return Ok(()),
            (Some(_), None) => return Err(SseCollectError::MissingData),
            (_, Some(data)) if data.is_empty() => return Err(SseCollectError::EmptyData),
            (event, Some(data)) => (event, data),
        };
        let decoded = self
            .decoder
            .decode_event(event.as_deref(), &data)
            .map_err(|_| SseCollectError::Decode)?;
        let wire_type = decoded.kind.wire_type();
        let refused = matches!(
            decoded.kind,
            StreamEventKind::ResponseFailed | StreamEventKind::Error
        );
        let progress = self
            .assembler
            .push(decoded)
            .map_err(SseCollectError::Assembly)?;
        if refused {
            self.canonical = BytesMut::new();
        } else {
            self.append_canonical(wire_type, &data)?;
        }
        if progress == Progress::Terminal {
            self.stage = Stage::Terminal;
        }
        Ok(())
    }

    /// Append `event: <wire_type>` and one `data: ` line per LF-separated
    /// segment of `data`, then the blank line.
    fn append_canonical(&mut self, wire_type: &str, data: &[u8]) -> Result<(), SseCollectError> {
        const EVENT: &[u8] = b"event: ";
        const DATA: &[u8] = b"data: ";
        let segments = data.split(|&b| b == b'\n');
        let lines = data.iter().filter(|&&b| b == b'\n').count() + 1;
        // Every LF of `data` stays a line ending, the last line gains one, and
        // each line gains a prefix; then the blank line.
        let size = EVENT.len() + wire_type.len() + 1 + data.len() + 1 + lines * DATA.len() + 1;
        if size > self.limits.max_stream_bytes - self.canonical.len() {
            return Err(SseCollectError::StreamTooLarge);
        }
        let start = self.canonical.len();
        self.canonical.reserve(size);
        self.canonical.extend_from_slice(EVENT);
        self.canonical.extend_from_slice(wire_type.as_bytes());
        self.canonical.extend_from_slice(b"\n");
        for segment in segments {
            self.canonical.extend_from_slice(DATA);
            self.canonical.extend_from_slice(segment);
            self.canonical.extend_from_slice(b"\n");
        }
        self.canonical.extend_from_slice(b"\n");
        debug_assert_eq!(self.canonical.len() - start, size);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::responses_output::TextSegmentKind;
    use serde_json::{json, Value};
    use SseCollectError as E;

    const MARKER: &str = "ATTACKER_BYTES_7f3a";

    // -- event builders (sequence numbers are assigned by `events`) ---------

    fn response(status: &str, output: Vec<Value>) -> Value {
        json!({"id": "resp_1", "object": "response", "status": status, "error": null, "output": output})
    }
    fn msg(text: &str) -> Value {
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": if text.is_empty() { json!([]) } else { json!([{"type": "output_text", "text": text, "annotations": []}]) }})
    }
    fn part(phase: &str, text: &str) -> Value {
        json!({"type": format!("response.content_part.{phase}"), "item_id": "msg_1",
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []}})
    }
    fn delta(d: &str) -> Value {
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
            "content_index": 0, "delta": d})
    }
    fn text_done(text: &str) -> Value {
        json!({"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0,
            "content_index": 0, "text": text})
    }

    /// Consistent single-part message stream with `deltas`, terminated by
    /// `response.completed`, numbered from 0.
    fn events(deltas: &[&str]) -> Vec<Value> {
        let full = deltas.concat();
        let mut events = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": msg("")}),
            part("added", ""),
        ];
        events.extend(deltas.iter().map(|d| delta(d)));
        events.extend([
            text_done(&full),
            part("done", &full),
            json!({"type": "response.output_item.done", "output_index": 0, "item": msg(&full)}),
            json!({"type": "response.completed", "response": response("completed", vec![msg(&full)])}),
        ]);
        number(events)
    }

    fn number(mut events: Vec<Value>) -> Vec<Value> {
        for (seq, v) in events.iter_mut().enumerate() {
            v["sequence_number"] = json!(seq);
        }
        events
    }

    /// One SSE frame. `named` adds an `event:` line; `eol` is the line ending.
    fn frame(v: &Value, named: bool, eol: &str) -> String {
        let data = serde_json::to_string(v).unwrap();
        let event = if named {
            format!("event: {}{eol}", v["type"].as_str().unwrap())
        } else {
            String::new()
        };
        format!("{event}data: {data}{eol}{eol}")
    }

    fn sse(events: &[Value], named: bool, eol: &str) -> String {
        events.iter().map(|v| frame(v, named, eol)).collect()
    }

    fn stream() -> String {
        sse(&events(&["He", "llo"]), true, "\n")
    }

    fn collect_with(limits: SseLimits, chunks: &[&[u8]]) -> Result<CollectedStream, E> {
        let mut c = ResponsesSseCollector::with_limits(limits);
        for chunk in chunks {
            c.push(chunk)?;
        }
        c.finish()
    }

    fn collect(bytes: &[u8]) -> Result<CollectedStream, E> {
        collect_with(SseLimits::default(), &[bytes])
    }

    fn assert_no_echo(label: &str, text: &str) {
        assert!(
            !text.contains(MARKER),
            "{label} echoes stream bytes: {text}"
        );
        assert!(!text.contains("msg_1"), "{label} echoes an id: {text}");
        assert!(!text.contains("resp_1"), "{label} echoes an id: {text}");
    }

    /// Push `bytes` in one chunk; assert the first error, poisoning, and no
    /// echo anywhere.
    fn expect_err(bytes: &[u8], expected: E) {
        let mut c = ResponsesSseCollector::new();
        let err = match c.push(bytes) {
            Ok(_) => c.finish().expect_err("stream must fail closed"),
            Err(err) => {
                assert_no_echo("collector", &format!("{c:?}"));
                assert_eq!(c.push(b""), Err(E::Poisoned), "error poisons");
                assert_eq!(c.finish().unwrap_err(), E::Poisoned);
                err
            }
        };
        assert_eq!(err, expected);
        assert_no_echo("error", &format!("{err} {err:?}"));
    }

    fn completed_text(collected: &CollectedStream) -> String {
        let Terminal::Completed(stream) = collected.terminal() else {
            panic!("completed")
        };
        let [text] = stream.response().texts() else {
            panic!("one text")
        };
        assert_eq!(text.kind(), TextSegmentKind::OutputText);
        text.text().to_string()
    }

    // -- clean streams ------------------------------------------------------

    #[test]
    fn clean_finish_returns_canonical_frames_and_assembled_terminal() {
        let bytes = format!(": keepalive\n\nid: 7\nretry: 10\n\n{}", stream());
        let collected = collect(bytes.as_bytes()).expect("clean stream");
        // Comments, id, and retry are gone; frames are otherwise unchanged.
        assert_eq!(collected.canonical().as_ref(), stream().as_bytes());
        assert_eq!(completed_text(&collected), "Hello");
        let Terminal::Completed(s) = collected.terminal() else {
            panic!("completed")
        };
        assert_eq!(s.items()[0].content_parts()[0].deltas(), "Hello");
        assert_no_echo("collected", &format!("{collected:?}"));
        let (terminal, canonical) = collected.into_parts();
        assert!(matches!(terminal, Terminal::Completed(_)));
        assert_eq!(canonical.as_ref(), stream().as_bytes());
    }

    #[test]
    fn canonical_frames_name_every_event_and_drop_framing_noise() {
        let evs = events(&["x"]);
        let expected = sse(&evs, true, "\n");
        // Unnamed CRLF frames with id/retry/comments inside each frame.
        let noisy: String = evs
            .iter()
            .map(|v| {
                format!(
                    ": c\r\nid: {MARKER}\r\nretry: x\r\n{}",
                    frame(v, false, "\r\n")
                )
            })
            .collect();
        let c = collect(noisy.as_bytes()).expect("noisy stream");
        assert_eq!(c.canonical().as_ref(), expected.as_bytes());
        // Canonical output re-collects to itself.
        let again = collect(c.canonical()).expect("canonical re-collects");
        assert_eq!(again.canonical(), c.canonical());
    }

    #[test]
    fn incomplete_terminal_returns_canonical_frames() {
        let mut evs = events(&["x"]);
        let last = evs.len() - 1;
        evs[last]["type"] = json!("response.incomplete");
        evs[last]["response"]["status"] = json!("incomplete");
        evs[last]["response"]["incomplete_details"] = json!({"reason": "max_output_tokens"});
        let bytes = sse(&evs, false, "\n");
        let c = collect(bytes.as_bytes()).expect("incomplete stream");
        assert!(matches!(c.terminal(), Terminal::Incomplete(_)));
        assert_eq!(c.canonical().as_ref(), sse(&evs, true, "\n").as_bytes());
    }

    #[test]
    fn every_split_and_one_byte_fragmentation_is_identical() {
        for eol in ["\n", "\r\n"] {
            let bytes = format!(":ka\r\n\n{}", sse(&events(&["a", "b\nc"]), true, eol));
            let bytes = bytes.as_bytes();
            let ones: Vec<&[u8]> = bytes.chunks(1).collect();
            let expected = sse(&events(&["a", "b\nc"]), true, "\n");
            let c = collect_with(SseLimits::default(), &ones).expect("1-byte chunks");
            assert_eq!(c.canonical().as_ref(), expected.as_bytes());
            assert_eq!(completed_text(&c), "ab\nc");
            for split in 0..=bytes.len() {
                let (a, b) = bytes.split_at(split);
                let c = collect_with(SseLimits::default(), &[a, b, b""])
                    .unwrap_or_else(|e| panic!("split {split} {eol:?}: {e}"));
                assert_eq!(c.canonical().as_ref(), expected.as_bytes());
            }
        }
    }

    #[test]
    fn lf_crlf_and_mixed_line_endings_frame_identically() {
        let evs = events(&["x"]);
        for eol in ["\n", "\r\n"] {
            for named in [true, false] {
                let c = collect(sse(&evs, named, eol).as_bytes()).expect("frames");
                assert_eq!(completed_text(&c), "x");
            }
        }
        let mixed: String = evs
            .iter()
            .enumerate()
            .map(|(i, v)| frame(v, i % 2 == 0, if i % 3 == 0 { "\r\n" } else { "\n" }))
            .collect();
        assert_eq!(completed_text(&collect(mixed.as_bytes()).unwrap()), "x");
    }

    #[test]
    fn bare_carriage_return_fails_closed() {
        let s = stream();
        expect_err(s.replacen('\n', "\r", 1).as_bytes(), E::BareCarriageReturn);
        expect_err(b"\r:x\n\n", E::BareCarriageReturn);
        // A trailing CR at a chunk boundary waits for its LF; EOF there is
        // truncation.
        let mut c = ResponsesSseCollector::new();
        assert_eq!(c.push(b": a\r"), Ok(Progress::Open));
        assert_eq!(c.finish().unwrap_err(), E::Truncated);
    }

    #[test]
    fn multiline_data_joins_with_one_lf_and_missing_event_is_none() {
        let evs = events(&["x"]);
        // Split the first event's JSON across three data lines at JSON
        // whitespace positions, with one empty data line in between.
        let json = serde_json::to_string(&evs[0]).unwrap();
        let (a, rest) = json.split_at(json.find(',').unwrap() + 1);
        let multi = format!("data: {a}\ndata:\ndata:{rest}\n\n");
        let tail = sse(&evs[1..], false, "\n");
        let c = collect(format!("{multi}{tail}").as_bytes()).expect("multiline data");
        assert_eq!(completed_text(&c), "x");
        // Canonical output keeps one `data:` line per payload line.
        let head = format!("event: response.created\ndata: {a}\ndata: \ndata: {rest}\n\n");
        assert!(c.canonical().starts_with(head.as_bytes()));
        assert_eq!(
            collect(c.canonical()).unwrap().canonical(),
            c.canonical(),
            "multiline canonical re-collects to itself"
        );

        // Recording decoder proves the exact joined payload and `None`.
        #[derive(Default)]
        struct Recording(std::sync::Mutex<Vec<(Option<String>, Vec<u8>)>>);
        impl StreamEventDecoder for Recording {
            fn decode_event(
                &self,
                name: Option<&str>,
                data: &[u8],
            ) -> Result<
                crate::provider::responses_stream::StreamEvent,
                crate::provider::ProviderError,
            > {
                self.0
                    .lock()
                    .unwrap()
                    .push((name.map(str::to_string), data.to_vec()));
                OpenAiResponsesStreamDecoder::new().decode_event(name, data)
            }
        }
        let rec = Recording::default();
        let mut c = ResponsesSseCollector::with_parts(
            SseLimits::default(),
            &rec,
            ResponsesStreamAssembler::new(),
        );
        c.push(format!("{multi}event: response.output_item.added\ndata: ").as_bytes())
            .unwrap();
        let seen = rec.0.lock().unwrap().clone();
        assert_eq!(seen, vec![(None, format!("{a}\n\n{rest}").into_bytes())]);
    }

    impl<T: StreamEventDecoder> StreamEventDecoder for &T {
        fn decode_event(
            &self,
            name: Option<&str>,
            data: &[u8],
        ) -> Result<crate::provider::responses_stream::StreamEvent, crate::provider::ProviderError>
        {
            (**self).decode_event(name, data)
        }
    }

    #[test]
    fn id_and_retry_are_accepted_and_dropped() {
        let mut c = ResponsesSseCollector::new();
        assert_eq!(
            c.push(b"id: 1\n\nretry: 10\nid\nretry\n\n"),
            Ok(Progress::Open)
        );
        assert_eq!(c.finish().unwrap_err(), E::NoTerminal);
        let s: String = events(&["x"])
            .iter()
            .map(|v| format!("retry: 5\n\nid: {MARKER}\n{}", frame(v, true, "\n")))
            .collect();
        let c = collect(s.as_bytes()).expect("id/retry dropped");
        assert_eq!(
            c.canonical().as_ref(),
            sse(&events(&["x"]), true, "\n").as_bytes()
        );
    }

    #[test]
    fn comments_are_ignored_inside_and_between_frames() {
        let evs = events(&["x"]);
        let mut s = String::from(":\n: ping\n\n\n");
        for v in &evs {
            s.push_str(&format!(": before\n{}", frame(v, true, "\n")));
        }
        let collected = collect(s.as_bytes()).expect("comments ignored");
        assert_eq!(
            collected.canonical().as_ref(),
            sse(&evs, true, "\n").as_bytes()
        );
        // A comment-only frame alone is a keepalive, not an event.
        let mut c = ResponsesSseCollector::new();
        assert_eq!(c.push(b": ping\n\n"), Ok(Progress::Open));
        assert_eq!(c.finish().unwrap_err(), E::NoTerminal);
    }

    #[test]
    fn malformed_fields_fail_closed() {
        let data = serde_json::to_string(&events(&["x"])[0]).unwrap();
        let named = format!("event: response.created\ndata: {data}\n\n");
        expect_err(
            format!("event: response.created\n{named}").as_bytes(),
            E::DuplicateEvent,
        );
        expect_err(b"Id: 1\n\n", E::UnsupportedField);
        expect_err(b"retries: 1\n\n", E::UnsupportedField);
        // id/retry are dropped, so they cannot stand in for data.
        expect_err(b"event: response.created\nid: 1\n\n", E::MissingData);
        expect_err(format!("{MARKER}: x\n\n").as_bytes(), E::UnsupportedField);
        expect_err(b"Data: {}\n\n", E::UnsupportedField);
        expect_err(b" data: {}\n\n", E::UnsupportedField);
        expect_err(b"ev\xffent: x\n\n", E::MalformedFieldName);
        expect_err(b"event: resp\xffonse\n\n", E::InvalidEventName);
        expect_err(b"event:\ndata: {}\n\n", E::InvalidEventName);
        expect_err(b"event: response.created\n\n", E::MissingData);
        expect_err(b"event: response.created\n: c\n\n", E::MissingData);
        expect_err(b"data:\n\n", E::EmptyData);
        expect_err(b"data\n\n", E::EmptyData);
        expect_err(b"event: response.created\ndata: \n\n", E::EmptyData);
    }

    #[test]
    fn decoder_rejections_fail_closed_without_echo() {
        let mut created = events(&["x"])[0].clone();
        created["response"]["metadata"] = json!(MARKER);
        expect_err(frame(&created, true, "\n").as_bytes(), E::Decode);
        let dup = format!(
            "data: {{\"type\":\"error\",\"type\":\"error\",\"sequence_number\":0,\"message\":\"{MARKER}\"}}\n\n"
        );
        expect_err(dup.as_bytes(), E::Decode);
        let unknown =
            json!({"type": "response.audio.delta", "sequence_number": 0, "delta": MARKER});
        expect_err(frame(&unknown, false, "\n").as_bytes(), E::Decode);
        expect_err(b"data: [DONE]\n\n", E::Decode);
        // Event name disagreeing with the payload type.
        let s = frame(&events(&["x"])[0], false, "\n");
        expect_err(format!("event: error\n{s}").as_bytes(), E::Decode);
        // Malformed UTF-8 inside data is the decoder's to refuse.
        expect_err(b"data: {\"type\":\"\xff\"}\n\n", E::Decode);
    }

    #[test]
    fn assembler_rejections_poison_the_collector() {
        let mut evs = events(&["x"]);
        evs[1]["sequence_number"] = json!(5);
        expect_err(
            sse(&evs, true, "\n").as_bytes(),
            E::Assembly(StreamStateError::SequenceOutOfOrder),
        );
        let mut evs = events(&["a", "b"]);
        let done = evs
            .iter()
            .position(|v| v["type"] == "response.output_text.done");
        evs[done.unwrap()]["text"] = json!(MARKER);
        expect_err(
            sse(&evs, false, "\n").as_bytes(),
            E::Assembly(StreamStateError::TextDoneMismatch),
        );
    }

    #[test]
    fn nothing_is_accepted_after_terminal() {
        let s = stream();
        for trailing in [
            &b"\n"[..],
            b"\r\n",
            b": keepalive\n\n",
            b"x",
            format!("data: {MARKER}\n\n").as_bytes(),
        ] {
            // Same chunk as the terminal frame.
            let mut both = s.clone().into_bytes();
            both.extend_from_slice(trailing);
            expect_err(&both, E::AfterTerminal);
            // A later chunk.
            let mut c = ResponsesSseCollector::new();
            assert_eq!(c.push(s.as_bytes()), Ok(Progress::Terminal));
            assert_eq!(c.push(b""), Ok(Progress::Terminal));
            assert_eq!(c.push(trailing), Err(E::AfterTerminal));
            assert_eq!(c.finish().unwrap_err(), E::Poisoned);
        }
    }

    #[test]
    fn eof_without_terminal_or_inside_a_frame_fails() {
        let s = stream();
        let last = s.trim_end_matches('\n').rfind("\n\n").unwrap() + 2;
        expect_err(&s.as_bytes()[..last], E::NoTerminal);
        expect_err(b"", E::NoTerminal);
        // Every proper prefix past the last complete pre-terminal frame is
        // truncated or missing its terminal.
        for cut in last + 1..s.len() {
            expect_err(&s.as_bytes()[..cut], E::Truncated);
        }
        expect_err(b": partial comment", E::Truncated);
    }

    #[test]
    fn frame_bound_accepts_exact_and_rejects_one_more() {
        let s = stream();
        // The largest frame bounds the limit; count lines + endings, not the
        // blank line.
        let max = s
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .map(|f| f.len() + 1)
            .max()
            .unwrap();
        let limits = |max_frame_bytes| SseLimits {
            max_frame_bytes,
            ..SseLimits::default()
        };
        let ones: Vec<&[u8]> = s.as_bytes().chunks(1).collect();
        collect_with(limits(max), &[s.as_bytes()]).expect("exact frame bound");
        collect_with(limits(max), &ones).expect("exact frame bound, 1-byte");
        assert_eq!(
            collect_with(limits(max - 1), &[s.as_bytes()]).unwrap_err(),
            E::FrameTooLarge
        );
        assert_eq!(
            collect_with(limits(max - 1), &ones).unwrap_err(),
            E::FrameTooLarge
        );
        // CRLF endings count two bytes each; comments count too.
        let comment = b": 123456\r\n\r\n";
        let mut c = ResponsesSseCollector::with_limits(limits(10));
        assert_eq!(c.push(comment), Ok(Progress::Open));
        let mut c = ResponsesSseCollector::with_limits(limits(9));
        assert_eq!(c.push(comment), Err(E::FrameTooLarge));
        // The line buffer never grows past the bound, even without a line
        // ending.
        let mut c = ResponsesSseCollector::with_limits(limits(4));
        assert_eq!(c.push(b"data"), Ok(Progress::Open));
        assert_eq!(c.push(b":"), Err(E::FrameTooLarge));
    }

    #[test]
    fn stream_bound_accepts_exact_and_rejects_one_more() {
        let s = stream();
        let limits = |max_stream_bytes| SseLimits {
            max_stream_bytes,
            ..SseLimits::default()
        };
        let c = collect_with(limits(s.len()), &[s.as_bytes()]).expect("exact stream bound");
        assert_eq!(c.canonical().len(), s.len());
        let (a, b) = s.as_bytes().split_at(s.len() / 2);
        assert_eq!(
            collect_with(limits(s.len() - 1), &[a, b]).unwrap_err(),
            E::StreamTooLarge
        );
        let mut c = ResponsesSseCollector::with_limits(limits(s.len() - 1));
        assert_eq!(c.push(a), Ok(Progress::Open));
        assert_eq!(c.push(b), Err(E::StreamTooLarge));
        assert_no_echo("collector", &format!("{c:?}"));
    }

    #[test]
    fn canonical_frames_share_the_stream_bound() {
        // Unnamed frames grow by an `event:` line when canonicalized.
        let s = sse(&events(&["x"]), false, "\n");
        let canonical = sse(&events(&["x"]), true, "\n");
        let limits = |max_stream_bytes| SseLimits {
            max_stream_bytes,
            ..SseLimits::default()
        };
        let c = collect_with(limits(canonical.len()), &[s.as_bytes()]).expect("exact");
        assert_eq!(c.canonical().len(), canonical.len());
        assert_eq!(
            collect_with(limits(canonical.len() - 1), &[s.as_bytes()]).unwrap_err(),
            E::StreamTooLarge
        );
    }

    #[test]
    fn default_limits_are_the_carried_bounds() {
        assert_eq!(
            SseLimits::default(),
            SseLimits {
                max_stream_bytes: 40 * 1024 * 1024,
                max_frame_bytes: 16 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn failed_and_error_terminals_refuse_without_content() {
        let failed = number(vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.failed", "response": {"id": "resp_1", "object": "response",
                "status": "failed", "error": {"code": "server_error", "message": MARKER}, "output": []}}),
        ]);
        let errored = number(vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "error", "code": "server_error", "message": MARKER, "param": null}),
        ]);
        for (evs, expected) in [(failed, E::UpstreamFailed), (errored, E::UpstreamError)] {
            for (named, eol) in [(true, "\n"), (false, "\r\n")] {
                let mut c = ResponsesSseCollector::new();
                assert_eq!(
                    c.push(sse(&evs, named, eol).as_bytes()),
                    Ok(Progress::Terminal)
                );
                let debug = format!("{c:?}");
                assert!(debug.contains("canonical_bytes: 0"), "{debug}");
                assert_no_echo("collector", &debug);
                let err = c.finish().unwrap_err();
                assert_eq!(err, expected);
                assert_no_echo("error", &format!("{err} {err:?}"));
            }
        }
    }

    #[test]
    fn failed_and_error_after_content_release_nothing() {
        // Accepted canonical frames carrying the marker are discarded when a
        // failure terminal arrives mid-stream.
        let head = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": msg("")}),
            part("added", ""),
            delta(MARKER),
        ];
        let failed = json!({"type": "response.failed", "response": {"id": "resp_1",
            "object": "response", "status": "failed",
            "error": {"code": "server_error", "message": MARKER}, "output": []}});
        let errored = json!({"type": "error", "code": "server_error", "message": MARKER,
            "param": null});
        for (last, expected) in [(failed, E::UpstreamFailed), (errored, E::UpstreamError)] {
            let mut evs = head.clone();
            evs.push(last);
            let bytes = sse(&number(evs), true, "\n");
            let mut c = ResponsesSseCollector::new();
            let (a, b) = bytes
                .as_bytes()
                .split_at(bytes.find("response.output_text.delta").unwrap());
            assert_eq!(c.push(a), Ok(Progress::Open));
            assert_eq!(c.push(b), Ok(Progress::Terminal));
            let debug = format!("{c:?}");
            assert!(debug.contains("canonical_bytes: 0"), "{debug}");
            assert_no_echo("collector", &debug);
            let err = c.finish().unwrap_err();
            assert_eq!(err, expected);
            assert_no_echo("error", &format!("{err} {err:?}"));
        }
    }

    #[test]
    fn lifecycle_response_ids_must_be_non_empty_and_exact() {
        let base = events(&["x"]);
        let last = base.len() - 1;
        let with_id = |index: usize, id: Value| {
            let mut evs = base.clone();
            evs[index]["response"]["id"] = id;
            sse(&evs, true, "\n")
        };
        for id in [json!(""), Value::Null] {
            expect_err(
                with_id(0, id.clone()).as_bytes(),
                E::Assembly(StreamStateError::MissingResponseId),
            );
            expect_err(
                with_id(last, id).as_bytes(),
                E::Assembly(StreamStateError::MissingResponseId),
            );
        }
        for id in ["resp_", "resp_10", "RESP_1"] {
            expect_err(
                with_id(last, json!(id)).as_bytes(),
                E::Assembly(StreamStateError::ResponseIdMismatch),
            );
        }
        // in_progress binds to the same id as created and the terminal.
        let mut evs = base.clone();
        evs.insert(
            1,
            json!({"type": "response.in_progress", "response": response("in_progress", vec![])}),
        );
        let good = number(evs.clone());
        let c = collect(sse(&good, false, "\n").as_bytes()).expect("same id throughout");
        assert_eq!(c.canonical().as_ref(), sse(&good, true, "\n").as_bytes());
        evs[1]["response"]["id"] = json!("resp_2");
        expect_err(
            sse(&number(evs), true, "\n").as_bytes(),
            E::Assembly(StreamStateError::ResponseIdMismatch),
        );
    }

    // -- opaque encrypted_content (OA-R2b1) ---------------------------------

    const ADDED_ENC: &str = "EncAddedAAAA";
    const DONE_ENC: &str = "EncDoneBBBBBBBB+/==";
    const TERM_ENC: &str = "EncTermCCCCCCCC-_==";

    fn reasoning(enc: &str, summary: &[&str]) -> Value {
        let summary: Vec<Value> = summary
            .iter()
            .map(|t| json!({"type": "summary_text", "text": t}))
            .collect();
        json!({"type": "reasoning", "id": "rs_1", "summary": summary, "encrypted_content": enc})
    }

    /// `reasoning_summary_part.{phase}` on `rs_1`; `status` is added as given.
    fn summary_part(phase: &str, index: usize, text: &str, status: Option<&str>) -> Value {
        let mut v = json!({"type": format!("response.reasoning_summary_part.{phase}"),
            "item_id": "rs_1", "output_index": 0, "summary_index": index,
            "part": {"type": "summary_text", "text": text}});
        if let Some(status) = status {
            v["status"] = json!(status);
        }
        v
    }

    /// Reasoning item whose opaque value rotates across added, done, and the
    /// terminal snapshot; `last` replaces the terminal event when given.
    fn encrypted_events(last: Option<Value>) -> Vec<Value> {
        let mut evs = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0,
                "item": reasoning(ADDED_ENC, &[])}),
            summary_part("added", 0, "", None),
            json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1",
                "output_index": 0, "summary_index": 0, "delta": "S"}),
            json!({"type": "response.reasoning_summary_text.done", "item_id": "rs_1",
                "output_index": 0, "summary_index": 0, "text": "S"}),
            summary_part("done", 0, "S", None),
            json!({"type": "response.output_item.done", "output_index": 0,
                "item": reasoning(DONE_ENC, &["S"])}),
        ];
        evs.push(last.unwrap_or_else(|| {
            json!({"type": "response.completed",
                "response": response("completed", vec![reasoning(TERM_ENC, &["S"])])})
        }));
        number(evs)
    }

    #[test]
    fn canonical_frames_keep_every_original_opaque_value_unchanged() {
        let evs = encrypted_events(None);
        let bytes = sse(&evs, false, "\r\n");
        let collected = collect(bytes.as_bytes()).expect("rotation is accepted");
        // Each released frame is the original validated payload, so every
        // snapshot keeps its own value; nothing is merged or rewritten.
        assert_eq!(
            collected.canonical().as_ref(),
            sse(&evs, true, "\n").as_bytes()
        );
        let canonical = std::str::from_utf8(collected.canonical()).unwrap();
        for enc in [ADDED_ENC, DONE_ENC, TERM_ENC] {
            assert_eq!(canonical.matches(enc).count(), 1, "{enc}");
        }
        let debug = format!("{collected:?}");
        let Terminal::Completed(stream) = collected.terminal() else {
            panic!("completed")
        };
        assert_eq!(
            stream
                .response()
                .encrypted_continuation(0)
                .map(|c| c.expose_ciphertext()),
            Some(TERM_ENC)
        );
        assert_eq!(stream.response().texts()[0].text(), "S");
        let terminal_debug = format!("{:?}", collected.terminal());
        for enc in [ADDED_ENC, DONE_ENC, TERM_ENC] {
            assert!(!debug.contains(enc) && !terminal_debug.contains(enc));
        }
    }

    #[test]
    fn failed_and_error_after_opaque_release_nothing() {
        let failed = json!({"type": "response.failed", "response": {"id": "resp_1",
            "object": "response", "status": "failed",
            "error": {"code": "server_error", "message": MARKER}, "output": []}});
        let errored = json!({"type": "error", "code": "server_error", "message": MARKER,
            "param": null});
        for (last, expected) in [(failed, E::UpstreamFailed), (errored, E::UpstreamError)] {
            let bytes = sse(&encrypted_events(Some(last)), true, "\n");
            let mut c = ResponsesSseCollector::new();
            assert_eq!(c.push(bytes.as_bytes()), Ok(Progress::Terminal));
            let debug = format!("{c:?}");
            assert!(debug.contains("canonical_bytes: 0"), "{debug}");
            let err = c.finish().unwrap_err();
            assert_eq!(err, expected);
            for enc in [ADDED_ENC, DONE_ENC, TERM_ENC] {
                assert!(!format!("{err} {err:?} {debug}").contains(enc));
            }
        }
    }

    #[test]
    fn opaque_presence_mismatch_poisons_and_releases_nothing() {
        let last = json!({"type": "response.completed",
            "response": response("completed", vec![json!({"type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "S"}]})])});
        let bytes = sse(&encrypted_events(Some(last)), true, "\n");
        let mut c = ResponsesSseCollector::new();
        let err = c.push(bytes.as_bytes()).unwrap_err();
        assert_eq!(
            err,
            E::Assembly(StreamStateError::EncryptedPresenceMismatch)
        );
        assert!(format!("{c:?}").contains("canonical_bytes: 0"));
        assert_eq!(c.finish().unwrap_err(), E::Poisoned);
    }

    // -- reasoning summary wrapper (OA-R2b2a) -------------------------------

    /// Two wrapped summary parts on an opaque reasoning item; the second
    /// wrapper done carries `status` as given. `last` replaces the terminal.
    fn summary_events(status: Option<&str>, last: Option<Value>) -> Vec<Value> {
        let text = |phase: &str, index: usize, key: &str, s: &str| {
            json!({"type": format!("response.reasoning_summary_text.{phase}"),
                "item_id": "rs_1", "output_index": 0, "summary_index": index, key: s})
        };
        let mut evs = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0,
                "item": reasoning(ADDED_ENC, &[])}),
            summary_part("added", 0, "", None),
            text("delta", 0, "delta", "Su"),
            text("delta", 0, "delta", "m"),
            text("done", 0, "text", "Sum"),
            summary_part("done", 0, "Sum", None),
            summary_part("added", 1, "", None),
            text("delta", 1, "delta", "Two"),
            text("done", 1, "text", "Two"),
            summary_part("done", 1, "Two", status),
            json!({"type": "response.output_item.done", "output_index": 0,
                "item": reasoning(DONE_ENC, &["Sum", "Two"])}),
        ];
        evs.push(last.unwrap_or_else(|| {
            json!({"type": "response.completed", "response":
                response("completed", vec![reasoning(TERM_ENC, &["Sum", "Two"])])})
        }));
        number(evs)
    }

    #[test]
    fn summary_wrappers_release_original_frames_and_project_once() {
        for status in [None, Some("incomplete")] {
            let evs = summary_events(status, None);
            let collected = collect(sse(&evs, false, "\r\n").as_bytes())
                .unwrap_or_else(|e| panic!("{status:?}: {e}"));
            // Canonical frames are the validated originals, status included.
            assert_eq!(
                collected.canonical().as_ref(),
                sse(&evs, true, "\n").as_bytes()
            );
            let canonical = std::str::from_utf8(collected.canonical()).unwrap();
            assert_eq!(
                canonical.matches("\"status\":\"incomplete\"").count(),
                usize::from(status.is_some())
            );
            let Terminal::Completed(stream) = collected.terminal() else {
                panic!("completed")
            };
            // Exactly one projection per summary part, from the terminal.
            let texts: Vec<_> = stream
                .response()
                .texts()
                .iter()
                .map(|s| (s.kind(), s.part_index(), s.text()))
                .collect();
            assert_eq!(
                texts,
                [
                    (TextSegmentKind::ReasoningSummary, 0, "Sum"),
                    (TextSegmentKind::ReasoningSummary, 1, "Two")
                ]
            );
            let parts: Vec<_> = stream.items()[0]
                .summary_parts()
                .iter()
                .map(|p| (p.added_text(), p.text_done(), p.part_done_text()))
                .collect();
            assert_eq!(
                parts,
                [
                    (Some(""), "Sum", Some("Sum")),
                    (Some(""), "Two", Some("Two"))
                ]
            );
            assert_eq!(
                stream
                    .response()
                    .encrypted_continuation(0)
                    .map(|c| c.expose_ciphertext()),
                Some(TERM_ENC)
            );
        }
    }

    #[test]
    fn summary_wrapper_violations_poison_and_release_nothing() {
        type Mutation = fn(&mut Vec<Value>);
        let cases: Vec<(&str, Mutation, E)> = vec![
            (
                "added text not empty",
                |e| e[2]["part"]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::PartAddedTextNotEmpty),
            ),
            (
                "wrapper done mismatch",
                |e| e[6]["part"]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::PartDoneMismatch),
            ),
            (
                "wrapper done before text done",
                |e| e.swap(5, 6),
                E::Assembly(StreamStateError::PartTextOpen),
            ),
            (
                "wrapper done dropped",
                |e| {
                    e.remove(10);
                },
                E::Assembly(StreamStateError::ChildrenOpen),
            ),
            (
                "wrapper added dropped",
                |e| {
                    e.remove(2);
                },
                E::Assembly(StreamStateError::UnknownPart),
            ),
            (
                "summary index gap",
                |e| e[7]["summary_index"] = json!(2),
                E::Assembly(StreamStateError::PartIndexGap),
            ),
            (
                "wrong item id",
                |e| e[2]["item_id"] = json!("rs_2"),
                E::Assembly(StreamStateError::UnknownItem),
            ),
            (
                "item done text mismatch",
                |e| e[11]["item"]["summary"][1]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::ItemDoneTextMismatch),
            ),
            (
                "terminal text mismatch",
                |e| e[12]["response"]["output"][0]["summary"][0]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::SnapshotTextMismatch),
            ),
            ("status null", |e| e[10]["status"] = json!(null), E::Decode),
            (
                "status other",
                |e| e[10]["status"] = json!("completed"),
                E::Decode,
            ),
            (
                "status on added",
                |e| e[2]["status"] = json!("incomplete"),
                E::Decode,
            ),
            (
                "extra part field",
                |e| e[6]["part"]["annotations"] = json!([]),
                E::Decode,
            ),
            (
                "wrong part type",
                |e| e[6]["part"]["type"] = json!("reasoning_text"),
                E::Decode,
            ),
        ];
        for (label, mutate, expected) in cases {
            let mut evs = summary_events(Some("incomplete"), None);
            mutate(&mut evs);
            let evs = number(evs);
            let mut c = ResponsesSseCollector::new();
            let err = c.push(sse(&evs, true, "\n").as_bytes()).unwrap_err();
            assert_eq!(err, expected, "{label}");
            let debug = format!("{c:?}");
            assert!(debug.contains("canonical_bytes: 0"), "{label}: {debug}");
            assert_no_echo(label, &format!("{err} {err:?} {debug}"));
            assert_eq!(c.finish().unwrap_err(), E::Poisoned, "{label}");
        }
    }

    // -- reasoning content wrapper (OA-3) ------------------------------------

    fn reasoning_content(content: &[&str]) -> Value {
        let content: Vec<Value> = content
            .iter()
            .map(|t| json!({"type": "reasoning_text", "text": t}))
            .collect();
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "content": content})
    }

    /// Reasoning item: content index 0 wrapped ("Re" + "a"), index 1 bare.
    fn reasoning_content_events() -> Vec<Value> {
        let text = |phase: &str, index: usize, key: &str, s: &str| {
            json!({"type": format!("response.reasoning_text.{phase}"), "item_id": "rs_1",
                "output_index": 0, "content_index": index, key: s})
        };
        let wrapper = |phase: &str, s: &str| {
            json!({"type": format!("response.content_part.{phase}"), "item_id": "rs_1",
                "output_index": 0, "content_index": 0,
                "part": {"type": "reasoning_text", "text": s}})
        };
        number(vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0,
                "item": reasoning_content(&[])}),
            wrapper("added", ""),
            text("delta", 0, "delta", "Re"),
            text("delta", 0, "delta", "a"),
            text("done", 0, "text", "Rea"),
            wrapper("done", "Rea"),
            text("delta", 1, "delta", "B"),
            text("done", 1, "text", "B"),
            json!({"type": "response.output_item.done", "output_index": 0,
                "item": reasoning_content(&["Rea", "B"])}),
            json!({"type": "response.completed", "response":
                response("completed", vec![reasoning_content(&["Rea", "B"])])}),
        ])
    }

    #[test]
    fn reasoning_content_wrappers_release_original_frames_and_project_once() {
        let evs = reasoning_content_events();
        let collected = collect(sse(&evs, false, "\r\n").as_bytes()).expect("wrapped + bare");
        assert_eq!(
            collected.canonical().as_ref(),
            sse(&evs, true, "\n").as_bytes()
        );
        let canonical = std::str::from_utf8(collected.canonical()).unwrap();
        // Both wrapper frames are retained as their original payloads.
        assert_eq!(
            canonical
                .matches(r#""type":"response.content_part."#)
                .count(),
            2
        );
        let Terminal::Completed(stream) = collected.terminal() else {
            panic!("completed")
        };
        let texts: Vec<_> = stream
            .response()
            .texts()
            .iter()
            .map(|s| (s.kind(), s.part_index(), s.text()))
            .collect();
        assert_eq!(
            texts,
            [
                (TextSegmentKind::ReasoningContent, 0, "Rea"),
                (TextSegmentKind::ReasoningContent, 1, "B")
            ]
        );
    }

    #[test]
    fn reasoning_content_wrapper_violations_poison_and_release_nothing() {
        type Mutation = fn(&mut Vec<Value>);
        let cases: Vec<(&str, Mutation, E)> = vec![
            (
                "added text not empty",
                |e| e[2]["part"]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::PartAddedTextNotEmpty),
            ),
            (
                "wrapper done mismatch",
                |e| e[6]["part"]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::PartDoneMismatch),
            ),
            (
                "wrapper done dropped",
                |e| {
                    e.remove(6);
                },
                E::Assembly(StreamStateError::ChildrenOpen),
            ),
            (
                "delta before wrapper added",
                |e| e.swap(2, 3),
                E::Assembly(StreamStateError::PartIndexGap),
            ),
            (
                "terminal text mismatch",
                |e| e[10]["response"]["output"][0]["content"][0]["text"] = json!(MARKER),
                E::Assembly(StreamStateError::SnapshotTextMismatch),
            ),
            (
                "wrong part type",
                |e| e[6]["part"]["type"] = json!("summary_text"),
                E::Decode,
            ),
            (
                "extra part field",
                |e| e[2]["part"]["annotations"] = json!([]),
                E::Decode,
            ),
        ];
        for (label, mutate, expected) in cases {
            let mut evs = reasoning_content_events();
            mutate(&mut evs);
            let evs = number(evs);
            let mut c = ResponsesSseCollector::new();
            let err = c.push(sse(&evs, true, "\n").as_bytes()).unwrap_err();
            assert_eq!(err, expected, "{label}");
            let debug = format!("{c:?}");
            assert!(debug.contains("canonical_bytes: 0"), "{label}: {debug}");
            assert_no_echo(label, &format!("{err} {err:?} {debug}"));
            assert_eq!(c.finish().unwrap_err(), E::Poisoned, "{label}");
        }
    }
}
