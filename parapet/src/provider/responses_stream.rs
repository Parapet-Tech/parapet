// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API SSE event decoder -- D-0027 Stage 2C1
//
// Responsibilities:
// - Decode ONE already-framed Responses SSE event (optional `event:` name plus
//   one `data:` JSON payload) through the strict duplicate-rejecting
//   [`JsonParser`] into a closed [`StreamEventKind`]
// - Require the payload `type` to equal the SSE event name when one is given
// - Validate every event against a closed per-event field allowlist; nested
//   response/item/part objects reuse the non-stream closed validators
// - Fail closed without echoing payload bytes on `[DONE]`, unknown event
//   types, hosted-tool/audio/image/file events, `obfuscation` padding, and
//   any unexpected model-visible field
//
// Not in scope: SSE framing, cross-event state (delta assembly, index/id
// continuity, sequence ordering), L5a/L3-outbound policy, rewriting,
// passthrough, and route activation. Nothing here makes a stream safe on its
// own; a later state machine must consume these events.

use super::responses::{
    allow_keys, empty_array_or_absent, invalid, optional_string, required_string, JsonParseError,
    JsonParser, StrictJsonParser,
};
use super::responses_output::{
    decode_message_part, decode_response_value, non_empty_string, DecodedResponse, OutputItemKind,
    OutputView, ResponseStatus, TextSegment, TextSegmentKind,
};
use super::ProviderError;
use crate::message::ToolCall;
use serde_json::{Map, Value};

/// Decodes one framed Responses SSE event.
pub trait StreamEventDecoder: Send + Sync {
    /// `event_name` is the SSE `event:` field, if the frame had one; `data`
    /// is the single `data:` payload.
    fn decode_event(
        &self,
        event_name: Option<&str>,
        data: &[u8],
    ) -> Result<StreamEvent, ProviderError>;
}

/// One decoded event: its provider sequence number plus the closed payload.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    pub sequence_number: u64,
    pub kind: StreamEventKind,
}

/// Whether an `*.added` or `*.done` event opened or closed its object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Added,
    Done,
}

/// Closed set of accepted Responses stream events. Every field is decoder
/// validated; provider error text never appears here.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEventKind {
    /// `response.created` / `response.in_progress`: status `queued` or
    /// `in_progress`, empty `output`.
    ResponsePending { response: DecodedResponse },
    /// `response.completed`: full closed response object.
    ResponseCompleted { response: DecodedResponse },
    /// `response.incomplete`: full closed response object.
    ResponseIncomplete { response: DecodedResponse },
    /// `response.failed`: shape validated, contents withheld.
    ResponseFailed,
    /// `response.output_item.added` / `.done`.
    OutputItem {
        phase: Phase,
        output_index: usize,
        item: StreamItem,
    },
    /// `response.content_part.added` / `.done` on a message item.
    ContentPart {
        phase: Phase,
        item_id: String,
        output_index: usize,
        content_index: usize,
        kind: TextSegmentKind,
        text: String,
    },
    /// `response.{output_text,refusal,reasoning_summary_text,reasoning_text}.delta`.
    /// `part_index` is `summary_index` for [`TextSegmentKind::ReasoningSummary`]
    /// and `content_index` otherwise.
    TextDelta {
        kind: TextSegmentKind,
        item_id: String,
        output_index: usize,
        part_index: usize,
        delta: String,
    },
    /// `response.{output_text,refusal,reasoning_summary_text,reasoning_text}.done`.
    TextDone {
        kind: TextSegmentKind,
        item_id: String,
        output_index: usize,
        part_index: usize,
        text: String,
    },
    /// `response.function_call_arguments.delta`.
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: usize,
        delta: String,
    },
    /// `response.function_call_arguments.done`. `arguments` is the exact wire
    /// string, unparsed, for later strict assembly checks.
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: usize,
        name: Option<String>,
        arguments: String,
    },
    /// Top-level `error` event: shape validated, contents withheld.
    Error,
}

/// A validated `output_item.added` / `.done` item.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamItem {
    pub kind: OutputItemKind,
    pub id: String,
    /// Text segments carried by the item, addressed at the event's
    /// `output_index`.
    pub texts: Vec<TextSegment>,
    /// Present for `function_call` items. Arguments are strictly parsed; an
    /// empty wire string (typical on `added`) parses as `{}`.
    pub tool_call: Option<ToolCall>,
}

/// Decoder for OpenAI Responses SSE event payloads.
pub struct OpenAiResponsesStreamDecoder<P: JsonParser = StrictJsonParser> {
    parser: P,
}

impl OpenAiResponsesStreamDecoder {
    pub fn new() -> Self {
        Self::with_parser(StrictJsonParser)
    }
}

impl<P: JsonParser> OpenAiResponsesStreamDecoder<P> {
    pub fn with_parser(parser: P) -> Self {
        Self { parser }
    }
}

impl Default for OpenAiResponsesStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Text event families: (type prefix, segment kind, index field, done field).
const TEXT_EVENTS: &[(&str, TextSegmentKind, &str, &str)] = &[
    (
        "response.output_text",
        TextSegmentKind::OutputText,
        "content_index",
        "text",
    ),
    (
        "response.refusal",
        TextSegmentKind::Refusal,
        "content_index",
        "refusal",
    ),
    (
        "response.reasoning_summary_text",
        TextSegmentKind::ReasoningSummary,
        "summary_index",
        "text",
    ),
    (
        "response.reasoning_text",
        TextSegmentKind::ReasoningContent,
        "content_index",
        "text",
    ),
];

impl<P: JsonParser> StreamEventDecoder for OpenAiResponsesStreamDecoder<P> {
    fn decode_event(
        &self,
        event_name: Option<&str>,
        data: &[u8],
    ) -> Result<StreamEvent, ProviderError> {
        if data.trim_ascii() == b"[DONE]" {
            return Err(invalid("[DONE] is not a Responses stream event"));
        }
        let root = self.parser.parse(data).map_err(|e| match e {
            JsonParseError::Syntax(msg) => ProviderError::InvalidJson(msg),
            JsonParseError::DuplicateKey => invalid("event contains a duplicate object key"),
        })?;
        let Value::Object(mut obj) = root else {
            return Err(invalid("event must be a JSON object"));
        };
        let event_type = required_string(&obj, "type", "event")?.to_string();
        if event_name.is_some_and(|name| name != event_type) {
            return Err(invalid("event name does not match event type"));
        }
        let sequence_number = index_u64(&obj, "sequence_number")?;
        let kind = self.decode_kind(&event_type, &mut obj)?;
        Ok(StreamEvent {
            sequence_number,
            kind,
        })
    }
}

impl<P: JsonParser> OpenAiResponsesStreamDecoder<P> {
    fn decode_kind(
        &self,
        event_type: &str,
        obj: &mut Map<String, Value>,
    ) -> Result<StreamEventKind, ProviderError> {
        let lifecycle = match event_type {
            "response.created" | "response.in_progress" => Some(ResponseStatus::Pending),
            "response.completed" => Some(ResponseStatus::Completed),
            "response.incomplete" => Some(ResponseStatus::Incomplete),
            "response.failed" => Some(ResponseStatus::Failed),
            _ => None,
        };
        if let Some(status) = lifecycle {
            return self.decode_lifecycle(obj, status);
        }
        let added_done = |prefix: &str| match event_type.strip_prefix(prefix) {
            Some(".added") => Some(Phase::Added),
            Some(".done") => Some(Phase::Done),
            _ => None,
        };
        if let Some(phase) = added_done("response.output_item") {
            return self.decode_output_item(obj, phase);
        }
        if let Some(phase) = added_done("response.content_part") {
            return decode_content_part(obj, phase);
        }
        for &(prefix, kind, index_key, done_key) in TEXT_EVENTS {
            match event_type.strip_prefix(prefix) {
                Some(".delta") => return decode_text_delta(obj, kind, index_key),
                Some(".done") => return decode_text_done(obj, kind, index_key, done_key),
                _ => {}
            }
        }
        match event_type {
            "response.function_call_arguments.delta" => decode_arguments_delta(obj),
            "response.function_call_arguments.done" => decode_arguments_done(obj),
            "error" => decode_error(obj),
            _ => Err(invalid("event has an unsupported type")),
        }
    }

    fn decode_lifecycle(
        &self,
        obj: &mut Map<String, Value>,
        status: ResponseStatus,
    ) -> Result<StreamEventKind, ProviderError> {
        allow_keys(obj, &["type", "sequence_number", "response"], "event")?;
        let root = obj
            .remove("response")
            .ok_or_else(|| ProviderError::MissingField("event.response".to_string()))?;
        let response = decode_response_value(&self.parser, root, status)?;
        Ok(match status {
            ResponseStatus::Pending => StreamEventKind::ResponsePending { response },
            ResponseStatus::Completed => StreamEventKind::ResponseCompleted { response },
            ResponseStatus::Incomplete => StreamEventKind::ResponseIncomplete { response },
            ResponseStatus::Failed => StreamEventKind::ResponseFailed,
            ResponseStatus::Final => unreachable!("Final is not a stream lifecycle status"),
        })
    }

    fn decode_output_item(
        &self,
        obj: &Map<String, Value>,
        phase: Phase,
    ) -> Result<StreamEventKind, ProviderError> {
        allow_keys(
            obj,
            &["type", "sequence_number", "output_index", "item"],
            "event",
        )?;
        let output_index = index(obj, "output_index")?;
        let item = obj
            .get("item")
            .ok_or_else(|| ProviderError::MissingField("event.item".to_string()))?;
        let mut view = OutputView::default();
        let kind = view.decode_item(&self.parser, item, output_index)?;
        // Stream items are addressed by `item_id` in later events.
        let id = item
            .as_object()
            .map(|o| non_empty_string(o, "id", "event.item"))
            .expect("decode_item accepted an object")?
            .to_string();
        Ok(StreamEventKind::OutputItem {
            phase,
            output_index,
            item: StreamItem {
                kind,
                id,
                texts: view.texts,
                tool_call: view.tool_calls.pop(),
            },
        })
    }
}

fn decode_content_part(
    obj: &Map<String, Value>,
    phase: Phase,
) -> Result<StreamEventKind, ProviderError> {
    allow_keys(
        obj,
        &[
            "type",
            "sequence_number",
            "item_id",
            "output_index",
            "content_index",
            "part",
        ],
        "event",
    )?;
    let part = obj
        .get("part")
        .ok_or_else(|| ProviderError::MissingField("event.part".to_string()))?;
    let (kind, _, text) = decode_message_part(part, "event.part")?;
    Ok(StreamEventKind::ContentPart {
        phase,
        item_id: non_empty_string(obj, "item_id", "event")?.to_string(),
        output_index: index(obj, "output_index")?,
        content_index: index(obj, "content_index")?,
        kind,
        text: text.to_string(),
    })
}

fn decode_text_delta(
    obj: &Map<String, Value>,
    kind: TextSegmentKind,
    index_key: &str,
) -> Result<StreamEventKind, ProviderError> {
    let mut allowed = vec![
        "type",
        "sequence_number",
        "item_id",
        "output_index",
        index_key,
        "delta",
    ];
    if kind == TextSegmentKind::OutputText {
        allowed.push("logprobs");
    }
    allow_keys(obj, &allowed, "event")?;
    empty_array_or_absent(obj, "logprobs", "event")?;
    Ok(StreamEventKind::TextDelta {
        kind,
        item_id: non_empty_string(obj, "item_id", "event")?.to_string(),
        output_index: index(obj, "output_index")?,
        part_index: index(obj, index_key)?,
        delta: required_string(obj, "delta", "event")?.to_string(),
    })
}

fn decode_text_done(
    obj: &Map<String, Value>,
    kind: TextSegmentKind,
    index_key: &str,
    done_key: &str,
) -> Result<StreamEventKind, ProviderError> {
    let mut allowed = vec![
        "type",
        "sequence_number",
        "item_id",
        "output_index",
        index_key,
        done_key,
    ];
    if kind == TextSegmentKind::OutputText {
        allowed.push("logprobs");
    }
    allow_keys(obj, &allowed, "event")?;
    empty_array_or_absent(obj, "logprobs", "event")?;
    Ok(StreamEventKind::TextDone {
        kind,
        item_id: non_empty_string(obj, "item_id", "event")?.to_string(),
        output_index: index(obj, "output_index")?,
        part_index: index(obj, index_key)?,
        text: required_string(obj, done_key, "event")?.to_string(),
    })
}

fn decode_arguments_delta(obj: &Map<String, Value>) -> Result<StreamEventKind, ProviderError> {
    allow_keys(
        obj,
        &[
            "type",
            "sequence_number",
            "item_id",
            "output_index",
            "delta",
        ],
        "event",
    )?;
    Ok(StreamEventKind::FunctionCallArgumentsDelta {
        item_id: non_empty_string(obj, "item_id", "event")?.to_string(),
        output_index: index(obj, "output_index")?,
        delta: required_string(obj, "delta", "event")?.to_string(),
    })
}

fn decode_arguments_done(obj: &Map<String, Value>) -> Result<StreamEventKind, ProviderError> {
    allow_keys(
        obj,
        &[
            "type",
            "sequence_number",
            "item_id",
            "output_index",
            "name",
            "arguments",
        ],
        "event",
    )?;
    let name = match obj.get("name") {
        None => None,
        Some(_) => Some(non_empty_string(obj, "name", "event")?.to_string()),
    };
    Ok(StreamEventKind::FunctionCallArgumentsDone {
        item_id: non_empty_string(obj, "item_id", "event")?.to_string(),
        output_index: index(obj, "output_index")?,
        name,
        arguments: required_string(obj, "arguments", "event")?.to_string(),
    })
}

/// Top-level `error`: validate shape, expose nothing.
fn decode_error(obj: &Map<String, Value>) -> Result<StreamEventKind, ProviderError> {
    allow_keys(
        obj,
        &["type", "sequence_number", "code", "message", "param"],
        "event",
    )?;
    optional_string(obj, "code", "event")?;
    optional_string(obj, "param", "event")?;
    required_string(obj, "message", "event")?;
    Ok(StreamEventKind::Error)
}

/// Required non-negative integer. Negative, fractional, exponent, and
/// beyond-u64 forms all fail.
fn index_u64(obj: &Map<String, Value>, key: &str) -> Result<u64, ProviderError> {
    match obj.get(key) {
        Some(v) => v
            .as_u64()
            .ok_or_else(|| invalid(format!("event.{key} must be a non-negative integer"))),
        None => Err(ProviderError::MissingField(format!("event.{key}"))),
    }
}

fn index(obj: &Map<String, Value>, key: &str) -> Result<usize, ProviderError> {
    usize::try_from(index_u64(obj, key)?)
        .map_err(|_| invalid(format!("event.{key} is out of range")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MARKER: &str = "ATTACKER_BYTES_7f3a";

    const RESPONSE_PENDING: &str = r#"{"id":"resp_1","object":"response","created_at":1761650000,"status":"in_progress","error":null,"incomplete_details":null,"model":"gpt-test-2026","output":[],"tools":[],"tool_choice":"auto","usage":null,"metadata":{}}"#;

    /// Exact raw payload for every accepted event type.
    fn accepted_fixtures() -> Vec<(&'static str, String)> {
        let completed = r#"{"id":"resp_1","object":"response","status":"completed","error":null,"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi","annotations":[],"logprobs":[]}]},{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"get","arguments":"{\"a\":1}"}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}"#;
        let incomplete = r#"{"id":"resp_1","object":"response","status":"incomplete","error":null,"incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","id":"msg_1","status":"incomplete","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}]}"#;
        let failed = r#"{"id":"resp_1","object":"response","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}"#;
        vec![
            ("response.created", format!(r#"{{"type":"response.created","sequence_number":0,"response":{RESPONSE_PENDING}}}"#)),
            ("response.in_progress", format!(r#"{{"type":"response.in_progress","sequence_number":1,"response":{}}}"#, RESPONSE_PENDING.replace("in_progress", "queued"))),
            ("response.completed", format!(r#"{{"type":"response.completed","sequence_number":20,"response":{completed}}}"#)),
            ("response.incomplete", format!(r#"{{"type":"response.incomplete","sequence_number":20,"response":{incomplete}}}"#)),
            ("response.failed", format!(r#"{{"type":"response.failed","sequence_number":20,"response":{failed}}}"#)),
            ("response.output_item.added", r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}"#.to_string()),
            ("response.output_item.done", r#"{"type":"response.output_item.done","sequence_number":9,"output_index":1,"item":{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"get","arguments":"{\"a\":1}"}}"#.to_string()),
            ("response.content_part.added", r#"{"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"logprobs":[]}}"#.to_string()),
            ("response.content_part.done", r#"{"type":"response.content_part.done","sequence_number":6,"item_id":"msg_1","output_index":0,"content_index":1,"part":{"type":"refusal","refusal":"No."}}"#.to_string()),
            ("response.output_text.delta", r#"{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi","logprobs":[]}"#.to_string()),
            ("response.output_text.done", r#"{"type":"response.output_text.done","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":0,"text":"Hi","logprobs":[]}"#.to_string()),
            ("response.refusal.delta", r#"{"type":"response.refusal.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":1,"delta":"No"}"#.to_string()),
            ("response.refusal.done", r#"{"type":"response.refusal.done","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":1,"refusal":"No."}"#.to_string()),
            ("response.reasoning_summary_text.delta", r#"{"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_1","output_index":0,"summary_index":2,"delta":"S"}"#.to_string()),
            ("response.reasoning_summary_text.done", r#"{"type":"response.reasoning_summary_text.done","sequence_number":5,"item_id":"rs_1","output_index":0,"summary_index":2,"text":"S1"}"#.to_string()),
            ("response.reasoning_text.delta", r#"{"type":"response.reasoning_text.delta","sequence_number":4,"item_id":"rs_1","output_index":0,"content_index":0,"delta":"R"}"#.to_string()),
            ("response.reasoning_text.done", r#"{"type":"response.reasoning_text.done","sequence_number":5,"item_id":"rs_1","output_index":0,"content_index":0,"text":"R1"}"#.to_string()),
            ("response.function_call_arguments.delta", r#"{"type":"response.function_call_arguments.delta","sequence_number":7,"item_id":"fc_1","output_index":1,"delta":"{\"a\""}"#.to_string()),
            ("response.function_call_arguments.done", r#"{"type":"response.function_call_arguments.done","sequence_number":8,"item_id":"fc_1","output_index":1,"name":"get","arguments":"{\"a\":1}"}"#.to_string()),
            ("error", r#"{"type":"error","sequence_number":3,"code":"rate_limit_exceeded","message":"slow down","param":null}"#.to_string()),
        ]
    }

    fn decode(name: Option<&str>, data: &str) -> Result<StreamEvent, ProviderError> {
        OpenAiResponsesStreamDecoder::new().decode_event(name, data.as_bytes())
    }

    fn decode_value(name: &str, data: &Value) -> Result<StreamEvent, ProviderError> {
        decode(Some(name), &serde_json::to_string(data).unwrap())
    }

    fn fixture(name: &str) -> String {
        accepted_fixtures()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no fixture {name}"))
            .1
    }

    fn fixture_value(name: &str) -> Value {
        serde_json::from_str(&fixture(name)).unwrap()
    }

    fn assert_fails_without_echo(result: Result<StreamEvent, ProviderError>, label: &str) {
        let err = match result {
            Ok(ev) => panic!("{label}: expected fail-closed decode, got {ev:?}"),
            Err(e) => e,
        };
        assert!(
            !err.to_string().contains(MARKER) && !format!("{err:?}").contains(MARKER),
            "{label}: error echoes payload bytes: {err}"
        );
    }

    #[test]
    fn every_accepted_event_decodes_with_and_without_event_name() {
        for (name, data) in accepted_fixtures() {
            let named = decode(Some(name), &data).unwrap_or_else(|e| panic!("{name}: {e}"));
            let unnamed = decode(None, &data).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(named, unnamed, "{name}");
        }
    }

    #[test]
    fn accepted_events_expose_decoder_owned_fields() {
        use StreamEventKind::*;
        use TextSegmentKind::*;
        let ev = |name: &str| decode(Some(name), &fixture(name)).unwrap();

        let created = ev("response.created");
        assert_eq!(created.sequence_number, 0);
        let ResponsePending { response } = created.kind else {
            panic!("created")
        };
        assert!(response.texts().is_empty() && response.tool_calls().is_empty());
        assert!(matches!(
            ev("response.in_progress").kind,
            ResponsePending { .. }
        ));

        let ResponseCompleted { response } = ev("response.completed").kind else {
            panic!("completed")
        };
        assert_eq!(response.texts()[0].text(), "Hi");
        assert_eq!(response.tool_calls()[0].arguments, json!({"a": 1}));
        assert!(matches!(
            ev("response.incomplete").kind,
            ResponseIncomplete { .. }
        ));
        assert_eq!(ev("response.failed").kind, ResponseFailed);
        assert_eq!(ev("error").kind, Error);

        let added = ev("response.output_item.added");
        assert_eq!(
            added.kind,
            OutputItem {
                phase: Phase::Added,
                output_index: 0,
                item: StreamItem {
                    kind: OutputItemKind::Message,
                    id: "msg_1".into(),
                    texts: vec![],
                    tool_call: None,
                },
            }
        );
        let OutputItem {
            phase,
            output_index,
            item,
        } = ev("response.output_item.done").kind
        else {
            panic!("item done")
        };
        assert_eq!(
            (phase, output_index, item.kind),
            (Phase::Done, 1, OutputItemKind::FunctionCall)
        );
        assert_eq!(
            item.tool_call,
            Some(ToolCall {
                id: "call_1".into(),
                name: "get".into(),
                arguments: json!({"a": 1}),
            })
        );

        assert_eq!(
            ev("response.content_part.done").kind,
            ContentPart {
                phase: Phase::Done,
                item_id: "msg_1".into(),
                output_index: 0,
                content_index: 1,
                kind: Refusal,
                text: "No.".into(),
            }
        );
        assert!(matches!(
            ev("response.content_part.added").kind,
            ContentPart {
                phase: Phase::Added,
                kind: OutputText,
                ..
            }
        ));

        let delta = |kind, item: &str, part_index, s: &str| TextDelta {
            kind,
            item_id: item.into(),
            output_index: 0,
            part_index,
            delta: s.into(),
        };
        let done = |kind, item: &str, part_index, s: &str| TextDone {
            kind,
            item_id: item.into(),
            output_index: 0,
            part_index,
            text: s.into(),
        };
        assert_eq!(
            ev("response.output_text.delta").kind,
            delta(OutputText, "msg_1", 0, "Hi")
        );
        assert_eq!(
            ev("response.output_text.done").kind,
            done(OutputText, "msg_1", 0, "Hi")
        );
        assert_eq!(
            ev("response.refusal.delta").kind,
            delta(Refusal, "msg_1", 1, "No")
        );
        assert_eq!(
            ev("response.refusal.done").kind,
            done(Refusal, "msg_1", 1, "No.")
        );
        assert_eq!(
            ev("response.reasoning_summary_text.delta").kind,
            delta(ReasoningSummary, "rs_1", 2, "S")
        );
        assert_eq!(
            ev("response.reasoning_summary_text.done").kind,
            done(ReasoningSummary, "rs_1", 2, "S1")
        );
        assert_eq!(
            ev("response.reasoning_text.delta").kind,
            delta(ReasoningContent, "rs_1", 0, "R")
        );
        assert_eq!(
            ev("response.reasoning_text.done").kind,
            done(ReasoningContent, "rs_1", 0, "R1")
        );
        assert_eq!(
            ev("response.function_call_arguments.delta").kind,
            FunctionCallArgumentsDelta {
                item_id: "fc_1".into(),
                output_index: 1,
                delta: "{\"a\"".into(),
            }
        );
    }

    #[test]
    fn function_arguments_done_preserves_exact_wire_string() {
        // Whitespace, key order, escapes, and even non-JSON survive verbatim;
        // strict parsing belongs to the later assembly stage.
        for args in [
            "{\"a\":1}",
            " { \"b\" : [1, 2] , \"a\" : \"\\u0041\\n\" } ",
            "{\"a\":1,\"a\":2}",
            "",
            "{\"unterminated\"",
        ] {
            let mut v = fixture_value("response.function_call_arguments.done");
            v["arguments"] = json!(args);
            v.as_object_mut().unwrap().remove("name");
            let ev = decode_value("response.function_call_arguments.done", &v).unwrap();
            assert_eq!(
                ev.kind,
                StreamEventKind::FunctionCallArgumentsDone {
                    item_id: "fc_1".into(),
                    output_index: 1,
                    name: None,
                    arguments: args.to_string(),
                }
            );
        }
    }

    #[test]
    fn deltas_preserve_newlines_quotes_and_escapes() {
        let raw = r#"{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"line1\nline2 \"q\" \\ é \r\n\n"}"#;
        let ev = decode(Some("response.output_text.delta"), raw).unwrap();
        let StreamEventKind::TextDelta { delta, .. } = ev.kind else {
            panic!()
        };
        assert_eq!(delta, "line1\nline2 \"q\" \\ \u{e9} \r\n\n");
    }

    #[test]
    fn event_name_must_equal_type() {
        let data = fixture("response.output_text.delta");
        for name in [
            "response.output_text.done",
            "",
            "response.output_text.delta ",
            "RESPONSE.OUTPUT_TEXT.DELTA",
            MARKER,
        ] {
            let result = decode(Some(name), &data);
            let msg = result
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_default();
            assert!(msg.contains("does not match"), "{name:?}: {msg}");
            assert_fails_without_echo(result, name);
        }
    }

    #[test]
    fn done_sentinel_and_non_events_fail_closed() {
        let truncated = format!("{{\"type\":\"{MARKER}\"");
        let trailing = format!("{} {MARKER}", fixture("response.refusal.delta"));
        let two_payloads = format!(
            "{}\n{}",
            fixture("response.refusal.delta"),
            fixture("response.refusal.done")
        );
        let array = format!("[{}]", fixture("response.refusal.delta"));
        for data in [
            "[DONE]",
            " [DONE]\n",
            "",
            "null",
            "\"response.created\"",
            truncated.as_str(),
            trailing.as_str(),
            two_payloads.as_str(),
            array.as_str(),
        ] {
            assert_fails_without_echo(decode(None, data), data);
        }
        assert!(decode(None, "[DONE]")
            .unwrap_err()
            .to_string()
            .contains("[DONE]"));
    }

    #[test]
    fn unknown_and_unsupported_event_types_fail_closed() {
        let types = [
            "response.queued",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.delta",
            "response.audio.delta",
            "response.audio.transcript.delta",
            "response.image_generation_call.partial_image",
            "response.image_generation_call.completed",
            "response.web_search_call.in_progress",
            "response.file_search_call.searching",
            "response.code_interpreter_call_code.delta",
            "response.mcp_call_arguments.delta",
            "response.mcp_list_tools.completed",
            "response.custom_tool_call_input.delta",
            "response.output_text.annotation.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.output_text",
            "response.output_text.added",
            "response.output_item.delta",
            "response.function_call_arguments.added",
            "response.output_text.delta.extra",
            "Response.Completed",
            MARKER,
        ];
        for t in types {
            let data = json!({"type": t, "sequence_number": 1, "item_id": "x",
                "output_index": 0, "content_index": 0, "delta": MARKER});
            assert_fails_without_echo(decode_value(t, &data), t);
        }
        for bad in [
            json!({}),
            json!({"type": 1, "sequence_number": 0}),
            json!({"type": null}),
        ] {
            assert_fails_without_echo(decode(None, &bad.to_string()), "bad type");
        }
    }

    #[test]
    fn duplicate_keys_including_escaped_aliases_fail_closed() {
        let cases = [
            format!(
                r#"{{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"ok","delta":"{MARKER}"}}"#
            ),
            format!(
                r#"{{"type":"response.output_text.delta","type":"error","sequence_number":4,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"{MARKER}"}}"#
            ),
            format!(
                r#"{{"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{{"type":"output_text","text":"","text":"{MARKER}"}}}}"#
            ),
            format!(
                r#"{{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{{"type":"message","id":"msg_1","role":"assistant","content":[],"role":"{MARKER}"}}}}"#
            ),
            format!(
                r#"{{"type":"response.created","sequence_number":0,"response":{{"object":"response","status":"queued","output":[],"output":["{MARKER}"]}}}}"#
            ),
        ];
        for data in cases {
            serde_json::from_str::<Value>(&data).unwrap();
            let result = decode(None, &data);
            let msg = result
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_default();
            assert!(msg.contains("duplicate"), "{msg}");
            assert_fails_without_echo(result, "duplicate");
        }
    }

    #[test]
    fn indexes_must_be_non_negative_integers() {
        let base = fixture("response.output_text.delta");
        let bad = [
            "-1",
            "1.5",
            "1.0",
            "1e2",
            "18446744073709551616",
            "\"0\"",
            "null",
            "true",
            "[0]",
        ];
        for key in ["sequence_number", "output_index", "content_index"] {
            for value in bad {
                let mut v: Value = serde_json::from_str(&base).unwrap();
                v[key] = json!(0);
                let raw = serde_json::to_string(&v)
                    .unwrap()
                    .replace(&format!("\"{key}\":0"), &format!("\"{key}\":{value}"));
                assert_fails_without_echo(decode(None, &raw), &format!("{key}={value}"));
            }
            let mut v: Value = serde_json::from_str(&base).unwrap();
            v.as_object_mut().unwrap().remove(key);
            assert_fails_without_echo(decode(None, &v.to_string()), &format!("missing {key}"));
        }
        let mut v = fixture_value("response.output_text.delta");
        v["sequence_number"] = json!(u64::MAX);
        v["output_index"] = json!(0);
        assert_eq!(
            decode(None, &v.to_string()).unwrap().sequence_number,
            u64::MAX
        );
    }

    #[test]
    fn ids_and_names_must_be_non_empty_strings() {
        let cases: Vec<(&str, &str, Value)> = vec![
            ("response.output_text.delta", "/item_id", json!("")),
            ("response.output_text.delta", "/item_id", json!(1)),
            ("response.refusal.done", "/item_id", json!(null)),
            ("response.content_part.added", "/item_id", json!("")),
            (
                "response.function_call_arguments.delta",
                "/item_id",
                json!(""),
            ),
            (
                "response.function_call_arguments.done",
                "/item_id",
                json!(""),
            ),
            ("response.function_call_arguments.done", "/name", json!("")),
            (
                "response.function_call_arguments.done",
                "/name",
                json!(null),
            ),
            ("response.output_item.added", "/item/id", json!("")),
            ("response.output_item.added", "/item/id", json!(null)),
            ("response.output_item.done", "/item/name", json!("")),
            ("response.output_item.done", "/item/call_id", json!("")),
        ];
        for (name, path, value) in cases {
            let mut v = fixture_value(name);
            *v.pointer_mut(path).unwrap() = value;
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} {path}"));
        }
        let mut v = fixture_value("response.output_item.added");
        v["item"].as_object_mut().unwrap().remove("id");
        assert_fails_without_echo(decode_value("response.output_item.added", &v), "item id");
    }

    #[test]
    fn non_string_text_fields_fail_closed() {
        let cases = [
            ("response.output_text.delta", "/delta"),
            ("response.output_text.done", "/text"),
            ("response.refusal.delta", "/delta"),
            ("response.refusal.done", "/refusal"),
            ("response.reasoning_summary_text.done", "/text"),
            ("response.reasoning_text.delta", "/delta"),
            ("response.function_call_arguments.delta", "/delta"),
            ("response.function_call_arguments.done", "/arguments"),
            ("response.content_part.added", "/part/text"),
            ("error", "/message"),
        ];
        for (name, path) in cases {
            for value in [json!({"v": MARKER}), json!([MARKER]), json!(null), json!(1)] {
                let mut v = fixture_value(name);
                *v.pointer_mut(path).unwrap() = value;
                assert_fails_without_echo(decode_value(name, &v), &format!("{name} {path}"));
            }
            let mut v = fixture_value(name);
            let (parent, key) = path.rsplit_once('/').unwrap();
            v.pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} missing {path}"));
        }
    }

    #[test]
    fn obfuscation_and_logprob_content_fail_closed() {
        for name in [
            "response.output_text.delta",
            "response.refusal.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        ] {
            let mut v = fixture_value(name);
            v["obfuscation"] = json!(MARKER);
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} obfuscation"));
        }
        for name in ["response.output_text.delta", "response.output_text.done"] {
            let mut v = fixture_value(name);
            v["logprobs"] = json!([{"token": MARKER, "logprob": -0.1}]);
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} logprobs"));
        }
        for name in ["response.refusal.delta", "response.reasoning_text.done"] {
            let mut v = fixture_value(name);
            v["logprobs"] = json!([]);
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} logprobs"));
        }
        let mut v = fixture_value("response.content_part.added");
        v["part"]["annotations"] = json!([{"type": "url_citation", "url": MARKER}]);
        assert_fails_without_echo(
            decode_value("response.content_part.added", &v),
            "annotations",
        );
    }

    #[test]
    fn unsupported_items_and_parts_fail_closed() {
        for t in [
            "web_search_call",
            "file_search_call",
            "image_generation_call",
            "code_interpreter_call",
            "computer_call",
            "mcp_call",
            "custom_tool_call",
            MARKER,
        ] {
            let mut v = fixture_value("response.output_item.added");
            v["item"] = json!({"type": t, "id": "x", "status": "in_progress", "result": MARKER});
            assert_fails_without_echo(decode_value("response.output_item.added", &v), t);
        }
        let mut v = fixture_value("response.output_item.done");
        v["item"] = json!({"type": "reasoning", "id": "rs_1", "summary": [],
            "encrypted_content": MARKER});
        assert_fails_without_echo(decode_value("response.output_item.done", &v), "encrypted");

        for part in [
            json!({"type": "output_audio", "data": MARKER, "transcript": MARKER}),
            json!({"type": "output_image", "image_url": MARKER}),
            json!({"type": "output_file", "file_id": MARKER}),
            json!({"type": "reasoning_text", "text": MARKER}),
            json!({"type": MARKER, "text": MARKER}),
            json!(MARKER),
        ] {
            let mut v = fixture_value("response.content_part.added");
            v["part"] = part;
            assert_fails_without_echo(decode_value("response.content_part.added", &v), "part");
        }
    }

    #[test]
    fn lifecycle_response_shapes_fail_closed() {
        let item = json!({"type": "message", "id": "m", "role": "assistant",
            "content": [{"type": "output_text", "text": MARKER}]});
        let cases: Vec<(&str, &str, Value)> = vec![
            ("response.created", "/response/output", json!([item])),
            ("response.in_progress", "/response/output", json!([item])),
            ("response.created", "/response/status", json!("completed")),
            (
                "response.created",
                "/response/error",
                json!({"code": "x", "message": MARKER}),
            ),
            (
                "response.completed",
                "/response/status",
                json!("incomplete"),
            ),
            ("response.completed", "/response/status", json!("failed")),
            (
                "response.completed",
                "/response/error",
                json!({"code": "x", "message": MARKER}),
            ),
            (
                "response.completed",
                "/response/output/1",
                json!({"type": "web_search_call", "id": "w", "action": {"query": MARKER}}),
            ),
            (
                "response.completed",
                "/response/output/0/content/0/annotations",
                json!([{"type": "url_citation", "title": MARKER}]),
            ),
            (
                "response.completed",
                "/response/output/1/arguments",
                json!(format!("{{\"a\":1,\"a\":\"{MARKER}\"}}")),
            ),
            (
                "response.incomplete",
                "/response/status",
                json!("completed"),
            ),
            ("response.failed", "/response/status", json!("completed")),
            ("response.failed", "/response/error", json!(null)),
            ("response.failed", "/response/error", json!(MARKER)),
            (
                "response.failed",
                "/response/error/message",
                json!({"v": MARKER}),
            ),
            ("response.failed", "/response/error/code", json!([MARKER])),
            ("response.completed", "/response", json!(MARKER)),
            ("response.completed", "/response", json!(null)),
        ];
        for (name, path, value) in cases {
            let mut v = fixture_value(name);
            *v.pointer_mut(path)
                .unwrap_or_else(|| panic!("{name} {path}")) = value;
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} {path}"));
        }
        for (name, key) in [
            ("response.created", "status"),
            ("response.completed", "status"),
            ("response.failed", "error"),
            ("response.completed", "output"),
        ] {
            let mut v = fixture_value(name);
            v["response"].as_object_mut().unwrap().remove(key);
            assert_fails_without_echo(decode_value(name, &v), &format!("{name} missing {key}"));
        }
        let mut v = fixture_value("response.completed");
        v.as_object_mut().unwrap().remove("response");
        assert_fails_without_echo(decode_value("response.completed", &v), "missing response");
    }

    #[test]
    fn failed_and_error_events_never_expose_provider_text() {
        let mut failed = fixture_value("response.failed");
        failed["response"]["error"] = json!({"code": MARKER, "message": MARKER});
        let mut error = fixture_value("error");
        error["code"] = json!(MARKER);
        error["message"] = json!(MARKER);
        error["param"] = json!(MARKER);
        for (name, v) in [("response.failed", failed), ("error", error)] {
            let ev = decode_value(name, &v).unwrap();
            assert!(!format!("{ev:?}").contains(MARKER), "{name}");
        }
    }

    /// Mutation sweep: inject a model-visible-looking field at every object
    /// boundary of every accepted fixture; each must fail closed without echo.
    #[test]
    fn unexpected_fields_at_every_object_boundary_fail_closed() {
        fn object_pointers(v: &Value, at: String, out: &mut Vec<String>) {
            match v {
                Value::Object(obj) => {
                    for (k, child) in obj {
                        object_pointers(child, format!("{at}/{k}"), out);
                    }
                    out.push(at);
                }
                Value::Array(items) => {
                    for (i, child) in items.iter().enumerate() {
                        object_pointers(child, format!("{at}/{i}"), out);
                    }
                }
                _ => {}
            }
        }
        // Free-form maps whose keys are caller data, not schema.
        let open_maps = ["/response/metadata"];
        let mut swept = 0;
        for (name, data) in accepted_fixtures() {
            let base: Value = serde_json::from_str(&data).unwrap();
            let mut boundaries = Vec::new();
            object_pointers(&base, String::new(), &mut boundaries);
            for at in boundaries
                .iter()
                .filter(|p| !open_maps.contains(&p.as_str()))
            {
                for field in ["smuggled", "obfuscation", "transcript", "text_extra"] {
                    let mut v = base.clone();
                    v.pointer_mut(at)
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .insert(field.to_string(), json!(MARKER));
                    assert_fails_without_echo(
                        decode_value(name, &v),
                        &format!("{name}: {field} at {at:?}"),
                    );
                    swept += 1;
                }
            }
        }
        assert!(swept > 100, "sweep covered only {swept} boundaries");
    }

    struct RecordingParser(std::sync::Mutex<Vec<Vec<u8>>>);

    impl JsonParser for RecordingParser {
        fn parse(&self, bytes: &[u8]) -> Result<Value, JsonParseError> {
            self.0.lock().unwrap().push(bytes.to_vec());
            StrictJsonParser.parse(bytes)
        }
    }

    #[test]
    fn decoder_reads_payload_and_nested_arguments_through_injected_parser() {
        let decoder =
            OpenAiResponsesStreamDecoder::with_parser(RecordingParser(Default::default()));
        let data = fixture("response.output_item.done");
        decoder
            .decode_event(Some("response.output_item.done"), data.as_bytes())
            .unwrap();
        // Arguments-done stays unparsed: only the payload goes through.
        let args_done = fixture("response.function_call_arguments.done");
        decoder.decode_event(None, args_done.as_bytes()).unwrap();
        let seen = decoder.parser.0.lock().unwrap();
        assert_eq!(
            *seen,
            vec![
                data.into_bytes(),
                b"{\"a\":1}".to_vec(),
                args_done.into_bytes()
            ]
        );
    }

    struct RefusingParser;

    impl JsonParser for RefusingParser {
        fn parse(&self, _: &[u8]) -> Result<Value, JsonParseError> {
            Err(JsonParseError::DuplicateKey)
        }
    }

    #[test]
    fn injected_parser_verdict_is_authoritative() {
        let decoder = OpenAiResponsesStreamDecoder::with_parser(RefusingParser);
        let err = decoder
            .decode_event(None, fixture("response.refusal.delta").as_bytes())
            .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }
}
