// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Raw candidate enumeration before `ToolCall` parsing
//! (Routing contract 50747a52, sections 3 step 2 and 4.4; draft 12 sections 5.2 and 7).
//!
//! Every choice and every call is enumerated in stable wire order with its real
//! coordinate. A coordinate that cannot be formed is never defaulted to 0.

use super::interfaces::{
    CandidateRefV0, CoordinateErrorV0, ProposalCandidateEnumeratorV0, ProposalCandidateV0,
    RawArgumentsV0, StreamFailureV0, TransportV0,
};
use crate::receipt::Provider;
use crate::stream::MAX_TOOL_CALL_BUFFER_BYTES;
use serde_json::Value;
use std::collections::BTreeMap;

const COORDINATE_LIMIT: u64 = 1 << 31;

fn coordinate(value: Option<&Value>, missing: CoordinateErrorV0) -> Result<u32, CoordinateErrorV0> {
    match value.and_then(Value::as_u64) {
        Some(n) if n < COORDINATE_LIMIT => Ok(n as u32),
        Some(_) => Err(CoordinateErrorV0::IndexOutOfRange),
        None => Err(missing),
    }
}

fn position(n: usize) -> Result<u32, CoordinateErrorV0> {
    if (n as u64) < COORDINATE_LIMIT {
        Ok(n as u32)
    } else {
        Err(CoordinateErrorV0::IndexOutOfRange)
    }
}

fn raw_arguments(value: Option<&Value>) -> RawArgumentsV0 {
    match value {
        None => RawArgumentsV0::Absent,
        Some(Value::Null) => RawArgumentsV0::Null,
        Some(Value::String(s)) => RawArgumentsV0::String(s.clone()),
        Some(other) => RawArgumentsV0::Object(other.clone()),
    }
}

fn string_field(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}

/// Section 4.2 `ProposalCandidateEnumeratorV0` for buffered responses.
#[derive(Debug, Default, Clone, Copy)]
pub struct RawCandidateEnumeratorV0;

impl ProposalCandidateEnumeratorV0 for RawCandidateEnumeratorV0 {
    fn enumerate_buffered(&self, provider: Provider, body: &Value) -> Vec<ProposalCandidateV0> {
        let mut out = Vec::new();
        match provider {
            Provider::Openai => {
                let Some(choices) = body.get("choices").and_then(Value::as_array) else {
                    return out;
                };
                for (p, choice) in choices.iter().enumerate() {
                    let choice_index = coordinate(
                        choice.get("index"),
                        CoordinateErrorV0::ChoiceIndexUnformable,
                    );
                    let Some(calls) = choice
                        .get("message")
                        .and_then(|m| m.get("tool_calls"))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };
                    for (i, call) in calls.iter().enumerate() {
                        let function = call.get("function");
                        let coordinate = choice_index.and_then(|c| {
                            position(i).map(|index| CandidateRefV0 {
                                provider,
                                choice_index: c,
                                index,
                            })
                        });
                        out.push(ProposalCandidateV0 {
                            ordinal: out.len() as u32,
                            provider,
                            transport: TransportV0::Buffered,
                            coordinate,
                            tool_call_id: string_field(call.get("id")),
                            tool_name: string_field(function.and_then(|f| f.get("name"))),
                            arguments: raw_arguments(function.and_then(|f| f.get("arguments"))),
                            failure: None,
                            malformed: function.is_some_and(|f| !f.is_object()),
                            location: (p as u32, i as u32),
                        });
                    }
                }
            }
            Provider::Anthropic => {
                let Some(blocks) = body.get("content").and_then(Value::as_array) else {
                    return out;
                };
                for (p, block) in blocks.iter().enumerate() {
                    if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                        continue;
                    }
                    out.push(ProposalCandidateV0 {
                        ordinal: out.len() as u32,
                        provider,
                        transport: TransportV0::Buffered,
                        coordinate: position(p).map(|index| CandidateRefV0 {
                            provider,
                            choice_index: 0,
                            index,
                        }),
                        tool_call_id: string_field(block.get("id")),
                        tool_name: string_field(block.get("name")),
                        arguments: raw_arguments(block.get("input")),
                        failure: None,
                        malformed: false,
                        location: (p as u32, 0),
                    });
                }
            }
        }
        out
    }
}

/// Which call a streaming line touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamTouchV0 {
    /// A formed (`choice_index`, `index`) coordinate.
    Formed(u32, u32),
    /// A fragment whose coordinate could not be formed, by ordinal.
    Unformable(u32),
}

/// Result of ingesting one SSE data payload.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct StreamIngestV0 {
    /// Calls this line carries bytes for; empty means pass-through.
    pub touches: Vec<StreamTouchV0>,
    /// Candidates finalized by this line, in ordinal order.
    pub finalized: Vec<(StreamTouchV0, ProposalCandidateV0)>,
    /// A call exceeded the buffer bound; the stream must abort.
    pub overflow: bool,
}

#[derive(Debug, Clone)]
struct CallBuffer {
    ordinal: u32,
    id: Option<String>,
    name: Option<String>,
    fragments: Option<String>,
    start_input: Option<RawArgumentsV0>,
    malformed: bool,
}

/// Streaming half of the enumerator: materializes every call coordinate from
/// raw SSE payloads, independently of the delivery classifier.
#[derive(Debug)]
pub struct StreamCandidateAssemblerV0 {
    provider: Provider,
    next_ordinal: u32,
    open: BTreeMap<(u32, u32), CallBuffer>,
    seen: BTreeMap<(u32, u32), u32>,
}

impl StreamCandidateAssemblerV0 {
    /// A new assembler for one provider stream.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            next_ordinal: 0,
            open: BTreeMap::new(),
            seen: BTreeMap::new(),
        }
    }

    /// Provider family of this stream.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Whether any call is still being materialized.
    pub fn has_open(&self) -> bool {
        !self.open.is_empty()
    }

    fn ordinal(&mut self) -> u32 {
        let o = self.next_ordinal;
        self.next_ordinal += 1;
        o
    }

    fn buffer(&mut self, key: (u32, u32)) -> &mut CallBuffer {
        if !self.open.contains_key(&key) {
            let ordinal = self.ordinal();
            self.seen.insert(key, ordinal);
            self.open.insert(
                key,
                CallBuffer {
                    ordinal,
                    id: None,
                    name: None,
                    fragments: None,
                    start_input: None,
                    malformed: false,
                },
            );
        }
        self.open.get_mut(&key).expect("buffer")
    }

    fn unformable(&mut self, error: CoordinateErrorV0, fragment: &Value, out: &mut StreamIngestV0) {
        let ordinal = self.ordinal();
        let touch = StreamTouchV0::Unformable(ordinal);
        out.touches.push(touch);
        out.finalized.push((
            touch,
            ProposalCandidateV0 {
                ordinal,
                provider: self.provider,
                transport: TransportV0::Streaming,
                coordinate: Err(error),
                tool_call_id: string_field(fragment.get("id")),
                tool_name: string_field(fragment.get("function").and_then(|f| f.get("name"))),
                arguments: RawArgumentsV0::Absent,
                failure: None,
                malformed: true,
                location: (0, 0),
            },
        ));
    }

    fn append(&mut self, key: (u32, u32), fragment: Option<&Value>, out: &mut StreamIngestV0) {
        let buffer = self.buffer(key);
        match fragment {
            None => {}
            Some(Value::String(s)) => buffer.fragments.get_or_insert_with(String::new).push_str(s),
            Some(_) => buffer.malformed = true,
        }
        if buffer
            .fragments
            .as_ref()
            .is_some_and(|f| f.len() > MAX_TOOL_CALL_BUFFER_BYTES)
        {
            out.overflow = true;
            if let Some(c) = self.finalize(key, Some(StreamFailureV0::BufferOverflow)) {
                out.finalized.push(c);
            }
        }
    }

    fn finalize(
        &mut self,
        key: (u32, u32),
        failure: Option<StreamFailureV0>,
    ) -> Option<(StreamTouchV0, ProposalCandidateV0)> {
        let buffer = self.open.remove(&key)?;
        let arguments = match (buffer.fragments, buffer.start_input) {
            (Some(s), _) => RawArgumentsV0::String(s),
            (None, Some(start)) => start,
            (None, None) => match self.provider {
                // OpenAI streams carry arguments only as fragments.
                Provider::Openai => RawArgumentsV0::String(String::new()),
                Provider::Anthropic => RawArgumentsV0::Absent,
            },
        };
        Some((
            StreamTouchV0::Formed(key.0, key.1),
            ProposalCandidateV0 {
                ordinal: buffer.ordinal,
                provider: self.provider,
                transport: TransportV0::Streaming,
                coordinate: Ok(CandidateRefV0 {
                    provider: self.provider,
                    choice_index: key.0,
                    index: key.1,
                }),
                tool_call_id: buffer.id,
                tool_name: buffer.name,
                arguments,
                failure,
                malformed: buffer.malformed,
                location: key,
            },
        ))
    }

    /// Ingest one SSE payload (`event` is Anthropic's event name).
    pub fn ingest(&mut self, event: Option<&str>, data: &str) -> StreamIngestV0 {
        let mut out = StreamIngestV0::default();
        if data.trim() == "[DONE]" {
            return out;
        }
        let Ok(json) = serde_json::from_str::<Value>(data.trim()) else {
            self.malformed_payload(&mut out);
            return out;
        };
        match self.provider {
            Provider::Openai => self.ingest_openai(&json, &mut out),
            Provider::Anthropic => self.ingest_anthropic(event, &json, &mut out),
        }
        out.finalized.sort_by_key(|(_, c)| c.ordinal);
        out
    }

    // An outer payload that is not JSON cannot be attributed: it fails closed.
    // Any open call may have lost bytes to it and becomes a parse failure; the
    // payload itself is an unformable candidate, so it never passes through.
    fn malformed_payload(&mut self, out: &mut StreamIngestV0) {
        for (key, buffer) in self.open.iter_mut() {
            buffer.malformed = true;
            out.touches.push(StreamTouchV0::Formed(key.0, key.1));
        }
        let error = match self.provider {
            Provider::Openai => CoordinateErrorV0::ChoiceIndexUnformable,
            Provider::Anthropic => CoordinateErrorV0::CallIndexUnformable,
        };
        self.unformable(error, &Value::Null, out);
    }

    fn ingest_openai(&mut self, json: &Value, out: &mut StreamIngestV0) {
        let Some(choices) = json.get("choices").and_then(Value::as_array) else {
            return;
        };
        for choice in choices {
            let calls = choice
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(Value::as_array);
            let choice_index = coordinate(
                choice.get("index"),
                CoordinateErrorV0::ChoiceIndexUnformable,
            );
            for call in calls.into_iter().flatten() {
                let key = choice_index.and_then(|c| {
                    coordinate(call.get("index"), CoordinateErrorV0::CallIndexUnformable)
                        .map(|i| (c, i))
                });
                let key = match key {
                    Ok(key) => key,
                    Err(e) => {
                        self.unformable(e, call, out);
                        continue;
                    }
                };
                out.touches.push(StreamTouchV0::Formed(key.0, key.1));
                let buffer = self.buffer(key);
                if buffer.id.is_none() {
                    buffer.id = string_field(call.get("id"));
                }
                let function = call.get("function");
                if buffer.name.is_none() {
                    buffer.name = string_field(function.and_then(|f| f.get("name")));
                }
                self.append(key, function.and_then(|f| f.get("arguments")), out);
            }
            let finished = choice.get("finish_reason").is_some_and(|f| !f.is_null());
            if let (true, Ok(c)) = (finished, choice_index) {
                let keys: Vec<_> = self.open.keys().filter(|k| k.0 == c).copied().collect();
                for key in keys {
                    if let Some(done) = self.finalize(key, None) {
                        out.finalized.push(done);
                    }
                }
            }
        }
    }

    fn ingest_anthropic(&mut self, event: Option<&str>, json: &Value, out: &mut StreamIngestV0) {
        let index = || coordinate(json.get("index"), CoordinateErrorV0::CallIndexUnformable);
        match event.unwrap_or("") {
            "content_block_start" => {
                let block = json.get("content_block");
                if block.and_then(|b| b.get("type")).and_then(Value::as_str) != Some("tool_use") {
                    return;
                }
                let block = block.expect("tool_use block");
                match index() {
                    Ok(i) => {
                        out.touches.push(StreamTouchV0::Formed(0, i));
                        let buffer = self.buffer((0, i));
                        buffer.id = string_field(block.get("id"));
                        buffer.name = string_field(block.get("name"));
                        buffer.start_input = Some(raw_arguments(block.get("input")));
                    }
                    Err(e) => self.unformable(e, block, out),
                }
            }
            "content_block_delta" => {
                let delta = json.get("delta");
                if delta.and_then(|d| d.get("type")).and_then(Value::as_str)
                    != Some("input_json_delta")
                {
                    return;
                }
                match index() {
                    Ok(i) => {
                        out.touches.push(StreamTouchV0::Formed(0, i));
                        self.append((0, i), delta.and_then(|d| d.get("partial_json")), out);
                    }
                    Err(e) => self.unformable(e, json, out),
                }
            }
            "content_block_stop" => match index() {
                Ok(i) if self.open.contains_key(&(0, i)) => {
                    out.touches.push(StreamTouchV0::Formed(0, i));
                    if let Some(done) = self.finalize((0, i), None) {
                        out.finalized.push(done);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Finalize every open call with a representable failure.
    pub fn abort(&mut self, failure: StreamFailureV0) -> Vec<(StreamTouchV0, ProposalCandidateV0)> {
        let keys: Vec<_> = self.open.keys().copied().collect();
        let mut out: Vec<_> = keys
            .into_iter()
            .filter_map(|k| self.finalize(k, Some(failure)))
            .collect();
        out.sort_by_key(|(_, c)| c.ordinal);
        out
    }

    /// The ordinal assigned to a formed coordinate, if seen.
    pub fn ordinal_of(&self, choice_index: u32, index: u32) -> Option<u32> {
        self.seen.get(&(choice_index, index)).copied()
    }
}

/// Materialize every candidate of a complete SSE body (fixtures and replay).
pub fn enumerate_stream(provider: Provider, sse: &str) -> Vec<ProposalCandidateV0> {
    let mut assembler = StreamCandidateAssemblerV0::new(provider);
    let mut event: Option<String> = None;
    let mut out = Vec::new();
    for line in sse.lines() {
        let line = line.trim();
        if let Some(e) = line.strip_prefix("event:") {
            event = Some(e.trim().to_owned());
        } else if let Some(data) = line.strip_prefix("data:") {
            let ingest = assembler.ingest(event.as_deref(), data.trim());
            out.extend(ingest.finalized.into_iter().map(|(_, c)| c));
            event = None;
        }
    }
    out.extend(
        assembler
            .abort(StreamFailureV0::StreamAborted)
            .into_iter()
            .map(|(_, c)| c),
    );
    out.sort_by_key(|c| c.ordinal);
    out
}
