// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API non-stream output decoder -- D-0027 Stage 2B2
//
// Responsibilities:
// - Decode a non-streaming `response` object through the strict
//   duplicate-rejecting [`JsonParser`] and keep the validated object intact
// - Expose every model-produced text segment in wire order (`output[]` items
//   in array order, parts in array order): message `output_text.text` and
//   `refusal.refusal`, reasoning `summary_text.text` then `reasoning_text.text`
// - Expose every `function_call` item as an internal [`ToolCall`] with
//   strictly parsed object arguments
// - Rewrite text segments one-for-one and reserialize compact JSON in which
//   only the addressed text fields changed
// - Fail closed without echoing response bytes on any unknown or malformed
//   item, part, or top-level field, on non-empty annotations/logprobs, on
//   reasoning `encrypted_content`, and on non-success status/error shapes
//
// Not in scope: SSE, L5a/L3-outbound policy, and route activation. This is a
// pure seam; the `/v1/responses` route still refuses to forward.

use super::responses::{
    allow_keys, decode_function_tool, empty_array_or_absent, enum_value, invalid, nullable_bool,
    nullable_enum, nullable_object, nullable_string, optional_string, parse_arguments,
    required_string, validate_metadata, validate_reasoning_config, validate_text_config,
    validate_tool_choice, JsonParseError, JsonParser, StrictJsonParser,
};
use super::ProviderError;
use crate::message::ToolCall;
use serde_json::{Map, Value};

/// Decodes a provider response body into an inspectable, rewritable value.
pub trait ResponseDecoder: Send + Sync {
    fn decode_response(&self, body: &[u8]) -> Result<DecodedResponse, ProviderError>;
}

/// Which wire field a [`TextSegment`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSegmentKind {
    /// `message` item, `output_text.text`
    OutputText,
    /// `message` item, `refusal.refusal`
    Refusal,
    /// `reasoning` item, `summary[].text`
    ReasoningSummary,
    /// `reasoning` item, `content[].text`
    ReasoningContent,
}

/// Where a segment lives inside `output[]`. Every component is either an
/// index or a decoder-owned constant, so the JSON pointer needs no escaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegmentLocation {
    item: usize,
    container: &'static str,
    part: usize,
    field: &'static str,
}

impl SegmentLocation {
    fn pointer(&self) -> String {
        format!(
            "/output/{}/{}/{}/{}",
            self.item, self.container, self.part, self.field
        )
    }
}

/// One model-produced text field, in wire order.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSegment {
    kind: TextSegmentKind,
    text: String,
    location: SegmentLocation,
}

impl TextSegment {
    pub fn kind(&self) -> TextSegmentKind {
        self.kind
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Index of the owning item in `output[]`.
    pub fn item_index(&self) -> usize {
        self.location.item
    }
}

/// Why [`DecodedResponse::with_replaced_texts`] refused. Never carries text.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RewriteError {
    #[error("expected {expected} replacement texts, got {actual}")]
    CardinalityMismatch { expected: usize, actual: usize },
    #[error("text segment {0} does not address a text field")]
    UnaddressableSegment(usize),
}

/// A validated non-stream Responses object plus its inspection view.
///
/// The full wire object is kept as validated; texts and tool calls are
/// read-only projections of it. Rewriting yields a new value and leaves this
/// one untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedResponse {
    root: Value,
    texts: Vec<TextSegment>,
    tool_calls: Vec<ToolCall>,
}

impl DecodedResponse {
    /// Model-produced text segments in wire order.
    pub fn texts(&self) -> &[TextSegment] {
        &self.texts
    }

    /// Function calls in `output[]` order.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Provider-assigned response `id`, if present and non-null.
    pub(super) fn response_id(&self) -> Option<&str> {
        self.root.get("id").and_then(Value::as_str)
    }

    /// `(kind, id)` of every `output[]` item in array order. Decoding already
    /// restricted `type` to the accepted kinds.
    pub(super) fn output_identities(&self) -> Vec<(OutputItemKind, Option<&str>)> {
        let items = self.root.get("output").and_then(Value::as_array);
        items
            .into_iter()
            .flatten()
            .map(|item| {
                let kind = match item.get("type").and_then(Value::as_str) {
                    Some("message") => OutputItemKind::Message,
                    Some("function_call") => OutputItemKind::FunctionCall,
                    Some("reasoning") => OutputItemKind::Reasoning,
                    _ => unreachable!("decode_response_value accepted only known item types"),
                };
                (kind, item.get("id").and_then(Value::as_str))
            })
            .collect()
    }

    /// Replace every text segment, in [`Self::texts`] order, with the
    /// matching entry of `replacements`. Exactly one replacement per segment
    /// is required; nothing outside the addressed text fields changes.
    pub fn with_replaced_texts(&self, replacements: &[String]) -> Result<Self, RewriteError> {
        if replacements.len() != self.texts.len() {
            return Err(RewriteError::CardinalityMismatch {
                expected: self.texts.len(),
                actual: replacements.len(),
            });
        }
        let mut root = self.root.clone();
        let mut texts = self.texts.clone();
        for (i, (segment, replacement)) in texts.iter_mut().zip(replacements).enumerate() {
            match root.pointer_mut(&segment.location.pointer()) {
                Some(Value::String(field)) => field.clone_from(replacement),
                _ => return Err(RewriteError::UnaddressableSegment(i)),
            }
            segment.text.clone_from(replacement);
        }
        Ok(Self {
            root,
            texts,
            tool_calls: self.tool_calls.clone(),
        })
    }

    /// Compact JSON of the (possibly rewritten) response object. Object keys
    /// serialize in sorted order, so output is deterministic.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.root).expect("serializing a serde_json::Value cannot fail")
    }
}

/// Decoder for the OpenAI Responses non-stream `response` object.
///
/// Accepted output items: `message` (role `assistant`; parts `output_text`
/// with absent/null/empty `annotations` and `logprobs`, and `refusal`),
/// `function_call`, and `reasoning` (`summary_text` / `reasoning_text` parts,
/// `encrypted_content` absent or null). Every other item type -- hosted tool
/// calls, computer/shell calls, image generation, MCP, custom tools -- and
/// every other part type fails closed, since its model-visible payload is not
/// inspected here.
///
/// Top-level fields are a closed allowlist of response metadata and request
/// echoes (`instructions`, `tools`, `tool_choice`, `reasoning`, `text`, ...),
/// each restricted to the same shapes the request decoder accepts. `status`
/// must be `completed` or `incomplete` and `error` must be null.
pub struct OpenAiResponsesOutputDecoder<P: JsonParser = StrictJsonParser> {
    parser: P,
}

impl OpenAiResponsesOutputDecoder {
    pub fn new() -> Self {
        Self::with_parser(StrictJsonParser)
    }
}

impl<P: JsonParser> OpenAiResponsesOutputDecoder<P> {
    pub fn with_parser(parser: P) -> Self {
        Self { parser }
    }
}

impl Default for OpenAiResponsesOutputDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: JsonParser> ResponseDecoder for OpenAiResponsesOutputDecoder<P> {
    fn decode_response(&self, body: &[u8]) -> Result<DecodedResponse, ProviderError> {
        let root = self.parser.parse(body).map_err(|e| match e {
            JsonParseError::Syntax(msg) => ProviderError::InvalidJson(msg),
            JsonParseError::DuplicateKey => {
                invalid("response body contains a duplicate object key")
            }
        })?;
        decode_response_value(&self.parser, root, ResponseStatus::Final)
    }
}

/// Which `status` / `error` combination a response object may carry. The
/// non-stream body is always [`ResponseStatus::Final`]; SSE lifecycle events
/// carry the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseStatus {
    /// `queued` or `in_progress`, `error` null, `output` empty.
    Pending,
    /// `completed` or `incomplete`, `error` null.
    Final,
    /// Exactly `completed`, `error` null.
    Completed,
    /// Exactly `incomplete`, `error` null.
    Incomplete,
    /// `failed` with a non-null `{code, message}` error object.
    Failed,
}

impl ResponseStatus {
    fn statuses(self) -> &'static [&'static str] {
        match self {
            Self::Pending => &["queued", "in_progress"],
            Self::Final => &["completed", "incomplete"],
            Self::Completed => &["completed"],
            Self::Incomplete => &["incomplete"],
            Self::Failed => &["failed"],
        }
    }
}

/// Validate an already-parsed response object and build its inspection view.
pub(super) fn decode_response_value(
    parser: &dyn JsonParser,
    root: Value,
    expected: ResponseStatus,
) -> Result<DecodedResponse, ProviderError> {
    let obj = root
        .as_object()
        .ok_or_else(|| invalid("response body must be a JSON object"))?;

    validate_top_level(obj, expected)?;

    let items = match obj.get("output") {
        Some(Value::Array(items)) => items,
        Some(_) => return Err(invalid("output must be an array")),
        None => return Err(ProviderError::MissingField("output".to_string())),
    };
    if expected == ResponseStatus::Pending && !items.is_empty() {
        return Err(invalid(
            "output must be empty while the response is pending",
        ));
    }
    let mut view = OutputView::default();
    for (i, item) in items.iter().enumerate() {
        view.decode_item(parser, item, i)?;
    }

    Ok(DecodedResponse {
        root,
        texts: view.texts,
        tool_calls: view.tool_calls,
    })
}

// ---------------------------------------------------------------------------
// Top-level fields
// ---------------------------------------------------------------------------

fn validate_top_level(
    root: &Map<String, Value>,
    expected: ResponseStatus,
) -> Result<(), ProviderError> {
    enum_value(
        root.get("object").unwrap_or(&Value::Null),
        &["response"],
        "object",
    )?;
    for (key, value) in root {
        let key = key.as_str();
        match key {
            "object" | "output" | "tools" | "tool_choice" => {}
            "id" | "model" | "instructions" | "user" | "safety_identifier" | "prompt_cache_key" => {
                nullable_string(value, key)?
            }
            "created_at" | "completed_at" | "max_output_tokens" | "max_tool_calls"
            | "top_logprobs" => nullable_count(value, key)?,
            "temperature" | "top_p" => {
                if !(value.is_null() || value.is_number()) {
                    return Err(invalid(format!("{key} must be a number or null")));
                }
            }
            "status" => enum_value(value, expected.statuses(), key)?,
            "error" => validate_error(value, expected)?,
            "previous_response_id" | "conversation" | "prompt" => {
                if !value.is_null() {
                    return Err(invalid(format!("{key} must be null")));
                }
            }
            "incomplete_details" => {
                if let Some(obj) = nullable_object(value, key)? {
                    allow_keys(obj, &["reason"], key)?;
                    nullable_enum(
                        obj.get("reason").unwrap_or(&Value::Null),
                        &["max_output_tokens", "content_filter"],
                        "incomplete_details.reason",
                    )?;
                }
            }
            "parallel_tool_calls" | "store" => nullable_bool(value, key)?,
            "background" => {
                if !matches!(value, Value::Null | Value::Bool(false)) {
                    return Err(invalid("background must be false or null"));
                }
            }
            "truncation" => nullable_enum(value, &["auto", "disabled"], key)?,
            "service_tier" => nullable_enum(
                value,
                &["auto", "default", "flex", "scale", "priority"],
                key,
            )?,
            "prompt_cache_retention" => nullable_enum(value, &["in-memory", "24h"], key)?,
            "metadata" => validate_metadata(value)?,
            "reasoning" => validate_reasoning_config(value)?,
            "text" => validate_text_config(value)?,
            "usage" => validate_usage(value)?,
            _ => return Err(invalid("response has an unsupported top-level field")),
        }
    }

    // Stream lifecycle events bind their phase to an explicit status; a
    // failed response must also say why.
    if expected != ResponseStatus::Final && !root.contains_key("status") {
        return Err(ProviderError::MissingField("status".to_string()));
    }
    if expected == ResponseStatus::Failed && !root.contains_key("error") {
        return Err(ProviderError::MissingField("error".to_string()));
    }

    // Request echo: same shapes the request decoder accepts.
    let mut tool_names = Vec::new();
    match root.get("tools") {
        None | Some(Value::Null) => {}
        Some(Value::Array(tools)) => {
            for (i, tool) in tools.iter().enumerate() {
                let (name, _) = decode_function_tool(tool, &format!("tools[{i}]"))?;
                if tool_names.contains(&name) {
                    return Err(invalid(format!("tools[{i}].name is a duplicate")));
                }
                tool_names.push(name);
            }
        }
        Some(_) => return Err(invalid("tools must be an array or null")),
    }
    validate_tool_choice(root.get("tool_choice"), &tool_names)
}

/// `error` is null except on a failed response, where it must be a
/// `{code, message}` object. Provider error text is validated, never exposed.
fn validate_error(value: &Value, expected: ResponseStatus) -> Result<(), ProviderError> {
    if expected != ResponseStatus::Failed {
        return if value.is_null() {
            Ok(())
        } else {
            Err(invalid("error must be null"))
        };
    }
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("error must be an object on a failed response"))?;
    allow_keys(obj, &["code", "message"], "error")?;
    optional_string(obj, "code", "error")?;
    required_string(obj, "message", "error")?;
    Ok(())
}

fn validate_usage(value: &Value) -> Result<(), ProviderError> {
    let Some(obj) = nullable_object(value, "usage")? else {
        return Ok(());
    };
    allow_keys(
        obj,
        &[
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
        ],
        "usage",
    )?;
    for key in ["input_tokens", "output_tokens", "total_tokens"] {
        nullable_count(
            obj.get(key).unwrap_or(&Value::Null),
            &format!("usage.{key}"),
        )?;
    }
    for (key, count) in [
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ] {
        let at = format!("usage.{key}");
        if let Some(details) = nullable_object(obj.get(key).unwrap_or(&Value::Null), &at)? {
            allow_keys(details, &[count], &at)?;
            nullable_count(
                details.get(count).unwrap_or(&Value::Null),
                &format!("{at}.{count}"),
            )?;
        }
    }
    Ok(())
}

fn nullable_count(value: &Value, at: &str) -> Result<(), ProviderError> {
    if value.is_null() || value.is_u64() {
        Ok(())
    } else {
        Err(invalid(format!(
            "{at} must be a non-negative integer or null"
        )))
    }
}

// ---------------------------------------------------------------------------
// Output items
// ---------------------------------------------------------------------------

const ITEM_STATUSES: &[&str] = &["in_progress", "completed", "incomplete"];

/// Inspection view accumulated over one or more output items.
#[derive(Default)]
pub(super) struct OutputView {
    pub(super) texts: Vec<TextSegment>,
    pub(super) tool_calls: Vec<ToolCall>,
}

impl OutputView {
    /// Validate one `output[]` item at `index`, appending its text segments
    /// and tool call. Returns the item kind.
    pub(super) fn decode_item(
        &mut self,
        parser: &dyn JsonParser,
        item: &Value,
        index: usize,
    ) -> Result<OutputItemKind, ProviderError> {
        let at = format!("output[{index}]");
        let obj = item
            .as_object()
            .ok_or_else(|| invalid(format!("{at} must be an object")))?;
        match required_string(obj, "type", &at)? {
            "message" => self
                .decode_message(obj, index, &at)
                .map(|()| OutputItemKind::Message),
            "function_call" => self
                .decode_function_call(parser, obj, &at)
                .map(|()| OutputItemKind::FunctionCall),
            "reasoning" => self
                .decode_reasoning(obj, index, &at)
                .map(|()| OutputItemKind::Reasoning),
            _ => Err(invalid(format!("{at} has an unsupported output item type"))),
        }
    }

    fn decode_message(
        &mut self,
        obj: &Map<String, Value>,
        index: usize,
        at: &str,
    ) -> Result<(), ProviderError> {
        allow_keys(obj, &["type", "id", "status", "role", "content"], at)?;
        validate_item_meta(obj, at)?;
        if required_string(obj, "role", at)? != "assistant" {
            return Err(ProviderError::InvalidRole(format!("{at}.role")));
        }
        let parts = required_array(obj, "content", at)?;
        for (p, part) in parts.iter().enumerate() {
            let (kind, field, text) = decode_message_part(part, &format!("{at}.content[{p}]"))?;
            self.push_text(kind, text, index, "content", p, field);
        }
        Ok(())
    }

    fn decode_function_call(
        &mut self,
        parser: &dyn JsonParser,
        obj: &Map<String, Value>,
        at: &str,
    ) -> Result<(), ProviderError> {
        allow_keys(
            obj,
            &["type", "id", "status", "call_id", "name", "arguments"],
            at,
        )?;
        validate_item_meta(obj, at)?;
        let call_id = non_empty_string(obj, "call_id", at)?;
        let name = non_empty_string(obj, "name", at)?;
        let arguments = parse_arguments(parser, required_string(obj, "arguments", at)?, at)?;
        if !arguments.is_object() {
            return Err(invalid(format!("{at}.arguments must be a JSON object")));
        }
        if self.tool_calls.iter().any(|c| c.id == call_id) {
            return Err(invalid(format!("{at}.call_id is a duplicate")));
        }
        self.tool_calls.push(ToolCall {
            id: call_id.to_string(),
            name: name.to_string(),
            arguments,
        });
        Ok(())
    }

    fn decode_reasoning(
        &mut self,
        obj: &Map<String, Value>,
        index: usize,
        at: &str,
    ) -> Result<(), ProviderError> {
        allow_keys(
            obj,
            &[
                "type",
                "id",
                "status",
                "summary",
                "content",
                "encrypted_content",
            ],
            at,
        )?;
        validate_item_meta(obj, at)?;
        // Opaque ciphertext; refuse any non-null value (including "").
        if obj.get("encrypted_content").is_some_and(|v| !v.is_null()) {
            return Err(invalid(format!(
                "{at}.encrypted_content cannot be inspected"
            )));
        }
        let summary = required_array(obj, "summary", at)?;
        self.decode_reasoning_parts(
            summary,
            index,
            "summary",
            "summary_text",
            TextSegmentKind::ReasoningSummary,
            at,
        )?;
        match obj.get("content") {
            None | Some(Value::Null) => {}
            Some(Value::Array(parts)) => self.decode_reasoning_parts(
                parts,
                index,
                "content",
                "reasoning_text",
                TextSegmentKind::ReasoningContent,
                at,
            )?,
            Some(_) => return Err(invalid(format!("{at}.content must be an array or null"))),
        }
        Ok(())
    }

    fn decode_reasoning_parts(
        &mut self,
        parts: &[Value],
        index: usize,
        container: &'static str,
        expected_type: &str,
        kind: TextSegmentKind,
        at: &str,
    ) -> Result<(), ProviderError> {
        for (p, part) in parts.iter().enumerate() {
            let at = format!("{at}.{container}[{p}]");
            let part = part
                .as_object()
                .ok_or_else(|| invalid(format!("{at} must be an object")))?;
            allow_keys(part, &["type", "text"], &at)?;
            if required_string(part, "type", &at)? != expected_type {
                return Err(invalid(format!("{at}.type must be \"{expected_type}\"")));
            }
            let text = required_string(part, "text", &at)?;
            self.push_text(kind, text, index, container, p, "text");
        }
        Ok(())
    }

    fn push_text(
        &mut self,
        kind: TextSegmentKind,
        text: &str,
        item: usize,
        container: &'static str,
        part: usize,
        field: &'static str,
    ) {
        self.texts.push(TextSegment {
            kind,
            text: text.to_string(),
            location: SegmentLocation {
                item,
                container,
                part,
                field,
            },
        });
    }
}

/// Which accepted `output[]` item type was decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputItemKind {
    Message,
    FunctionCall,
    Reasoning,
}

/// Validate one assistant `message` content part. Returns its segment kind,
/// text field name, and text.
pub(super) fn decode_message_part<'a>(
    part: &'a Value,
    at: &str,
) -> Result<(TextSegmentKind, &'static str, &'a str), ProviderError> {
    let part = part
        .as_object()
        .ok_or_else(|| invalid(format!("{at} must be an object")))?;
    let (kind, field) = match required_string(part, "type", at)? {
        "output_text" => {
            allow_keys(part, &["type", "text", "annotations", "logprobs"], at)?;
            // Annotation titles/URLs and logprob tokens are text-bearing and
            // not segmented; only empty forms pass.
            empty_array_or_absent(part, "annotations", at)?;
            empty_array_or_absent(part, "logprobs", at)?;
            (TextSegmentKind::OutputText, "text")
        }
        "refusal" => {
            allow_keys(part, &["type", "refusal"], at)?;
            (TextSegmentKind::Refusal, "refusal")
        }
        _ => return Err(invalid(format!("{at} has an unsupported content type"))),
    };
    Ok((kind, field, required_string(part, field, at)?))
}

/// `id` and `status` on every output item: provider-assigned, not model text.
fn validate_item_meta(obj: &Map<String, Value>, at: &str) -> Result<(), ProviderError> {
    optional_string(obj, "id", at)?;
    nullable_enum(
        obj.get("status").unwrap_or(&Value::Null),
        ITEM_STATUSES,
        &format!("{at}.status"),
    )
}

fn required_array<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<&'a [Value], ProviderError> {
    match obj.get(key) {
        Some(Value::Array(a)) => Ok(a),
        Some(_) => Err(invalid(format!("{at}.{key} must be an array"))),
        None => Err(ProviderError::MissingField(format!("{at}.{key}"))),
    }
}

pub(super) fn non_empty_string<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<&'a str, ProviderError> {
    let s = required_string(obj, key, at)?;
    if s.is_empty() {
        return Err(invalid(format!("{at}.{key} must be non-empty")));
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MARKER: &str = "ATTACKER_BYTES_7f3a";

    /// Benign non-stream response: reasoning, mixed message parts, two
    /// function calls, a trailing message, plus every accepted metadata field.
    const FIXTURE: &str = r#"{
  "id": "resp_123",
  "object": "response",
  "created_at": 1761650000,
  "completed_at": 1761650003,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": "You are terse.",
  "model": "gpt-test-2026",
  "output": [
    {"type": "reasoning", "id": "rs_1", "status": "completed",
     "summary": [{"type": "summary_text", "text": "S1"}, {"type": "summary_text", "text": "S2"}],
     "content": [{"type": "reasoning_text", "text": "R1"}],
     "encrypted_content": null},
    {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
     "content": [
       {"type": "output_text", "text": "T1", "annotations": [], "logprobs": []},
       {"type": "refusal", "refusal": "F1"},
       {"type": "output_text", "text": "T2", "annotations": []}
     ]},
    {"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "call_a",
     "name": "read_file",
     "arguments": "{\"path\":\"/tmp/a\",\"opts\":{\"depth\":2,\"tags\":[\"x\",{\"k\":null}]}}"},
    {"type": "function_call", "id": "fc_2", "status": "completed", "call_id": "call_b",
     "name": "list_dir", "arguments": ""},
    {"type": "message", "id": "msg_2", "status": "completed", "role": "assistant",
     "content": [{"type": "output_text", "text": "T3"}]}
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {"effort": "low", "summary": "auto"},
  "store": false,
  "background": false,
  "service_tier": "default",
  "temperature": 1.0,
  "text": {"format": {"type": "text"}, "verbosity": "medium"},
  "tool_choice": "auto",
  "tools": [{"type": "function", "name": "read_file", "description": "Read a file",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
             "strict": true},
            {"type": "function", "name": "list_dir", "parameters": null, "strict": null}],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 30},
  "user": null,
  "metadata": {"trace": "abc"}
}"#;

    fn decode_raw(body: &str) -> Result<DecodedResponse, ProviderError> {
        OpenAiResponsesOutputDecoder::new().decode_response(body.as_bytes())
    }

    fn decode(body: &Value) -> Result<DecodedResponse, ProviderError> {
        decode_raw(&serde_json::to_string(body).unwrap())
    }

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn minimal(output: Value) -> Value {
        json!({"object": "response", "status": "completed", "output": output})
    }

    fn assert_fails_without_echo(result: Result<DecodedResponse, ProviderError>, label: &str) {
        let err = match result {
            Ok(_) => panic!("{label}: expected fail-closed decode"),
            Err(e) => e,
        };
        assert!(
            !err.to_string().contains(MARKER) && !format!("{err:?}").contains(MARKER),
            "{label}: error echoes response bytes: {err}"
        );
    }

    fn texts(decoded: &DecodedResponse) -> Vec<(TextSegmentKind, &str, usize)> {
        decoded
            .texts()
            .iter()
            .map(|s| (s.kind(), s.text(), s.item_index()))
            .collect()
    }

    #[test]
    fn mixed_text_refusal_and_reasoning_segments_in_wire_order() {
        use TextSegmentKind::*;
        let decoded = decode_raw(FIXTURE).unwrap();
        assert_eq!(
            texts(&decoded),
            vec![
                (ReasoningSummary, "S1", 0),
                (ReasoningSummary, "S2", 0),
                (ReasoningContent, "R1", 0),
                (OutputText, "T1", 1),
                (Refusal, "F1", 1),
                (OutputText, "T2", 1),
                (OutputText, "T3", 4),
            ]
        );
    }

    #[test]
    fn multiple_function_calls_with_nested_arguments() {
        let decoded = decode_raw(FIXTURE).unwrap();
        assert_eq!(
            decoded.tool_calls(),
            &[
                ToolCall {
                    id: "call_a".to_string(),
                    name: "read_file".to_string(),
                    arguments: json!({"path": "/tmp/a",
                        "opts": {"depth": 2, "tags": ["x", {"k": null}]}}),
                },
                ToolCall {
                    id: "call_b".to_string(),
                    name: "list_dir".to_string(),
                    arguments: json!({}),
                },
            ]
        );
    }

    #[test]
    fn round_trip_preserves_every_field_and_benign_metadata() {
        let decoded = decode_raw(FIXTURE).unwrap();
        let reserialized: Value = serde_json::from_slice(&decoded.to_json_bytes()).unwrap();
        assert_eq!(reserialized, fixture());
        // Deterministic and stable under re-decode.
        let again = decode(&reserialized).unwrap();
        assert_eq!(again, decoded);
        assert_eq!(again.to_json_bytes(), decoded.to_json_bytes());
        // Identity rewrite is byte-identical.
        let same: Vec<String> = decoded
            .texts()
            .iter()
            .map(|s| s.text().to_string())
            .collect();
        assert_eq!(
            decoded.with_replaced_texts(&same).unwrap().to_json_bytes(),
            decoded.to_json_bytes()
        );
    }

    #[test]
    fn rewrite_replaces_exactly_the_text_fields() {
        let protected = "PROTECTED_SECRET_sk-live-0000";
        let mut body = fixture();
        body["output"][1]["content"][0]["text"] = json!(format!("leak {protected}"));
        body["output"][0]["summary"][1]["text"] = json!(protected);
        body["output"][1]["content"][1]["refusal"] = json!(protected);
        let decoded = decode(&body).unwrap();

        let replacements: Vec<String> = (0..decoded.texts().len())
            .map(|i| format!("[r{i}]"))
            .collect();
        let rewritten = decoded.with_replaced_texts(&replacements).unwrap();
        let bytes = rewritten.to_json_bytes();
        let out = String::from_utf8(bytes.clone()).unwrap();
        assert!(!out.contains(protected));
        assert!(!out.contains('\n'), "serialization must be compact");

        let mut expected = body.clone();
        let paths = [
            "/output/0/summary/0/text",
            "/output/0/summary/1/text",
            "/output/0/content/0/text",
            "/output/1/content/0/text",
            "/output/1/content/1/refusal",
            "/output/1/content/2/text",
            "/output/4/content/0/text",
        ];
        for (i, path) in paths.iter().enumerate() {
            *expected.pointer_mut(path).unwrap() = json!(format!("[r{i}]"));
        }
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);

        let new_texts: Vec<&str> = rewritten.texts().iter().map(|s| s.text()).collect();
        assert_eq!(
            new_texts,
            replacements.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(rewritten.tool_calls(), decoded.tool_calls());
        // The source value is immutable.
        assert_eq!(decoded, decode(&body).unwrap());
    }

    #[test]
    fn rewrite_refuses_cardinality_mismatch() {
        let decoded = decode_raw(FIXTURE).unwrap();
        let n = decoded.texts().len();
        for actual in [0, n - 1, n + 1] {
            let replacements = vec![String::new(); actual];
            assert_eq!(
                decoded.with_replaced_texts(&replacements),
                Err(RewriteError::CardinalityMismatch {
                    expected: n,
                    actual
                })
            );
        }
        let empty = decode(&minimal(json!([]))).unwrap();
        assert!(empty.texts().is_empty() && empty.tool_calls().is_empty());
        assert_eq!(
            empty.with_replaced_texts(&[]).unwrap().to_json_bytes(),
            empty.to_json_bytes()
        );
        assert!(empty.with_replaced_texts(&["x".to_string()]).is_err());
    }

    #[test]
    fn empty_text_parts_are_still_segments() {
        let decoded = decode(&minimal(json!([
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": ""}]}
        ])))
        .unwrap();
        assert_eq!(decoded.texts().len(), 1);
        assert_eq!(decoded.texts()[0].text(), "");
    }

    #[test]
    fn duplicate_keys_fail_closed_at_every_depth_without_echo() {
        let marker_key = format!("{MARKER}_k");
        let cases: Vec<(&str, String)> = vec![
            (
                "top-level escaped alias",
                format!(r#"{{"object":"response","output":[],"\u006fbject":"response","{marker_key}":1}}"#),
            ),
            (
                "output item",
                r#"{"object":"response","output":[{"type":"message","role":"assistant","content":[],"t\u0079pe":"message"}]}"#.to_string(),
            ),
            (
                "content part text",
                format!(r#"{{"object":"response","output":[{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"benign","te\u0078t":"{MARKER}"}}]}}]}}"#),
            ),
            (
                "function_call arguments",
                format!(r#"{{"object":"response","output":[{{"type":"function_call","call_id":"c","name":"n","arguments":"{{\"cmd\":\"ls\",\"\\u0063md\":\"{MARKER}\"}}"}}]}}"#),
            ),
            (
                "nested arguments object",
                format!(r#"{{"object":"response","output":[{{"type":"function_call","call_id":"c","name":"n","arguments":"{{\"a\":{{\"b\":1,\"b\":\"{MARKER}\"}}}}"}}]}}"#),
            ),
        ];
        for (label, body) in cases {
            // Each case must be well-formed apart from the duplicate.
            serde_json::from_str::<Value>(&body).unwrap_or_else(|e| panic!("{label}: {e}"));
            let result = decode_raw(&body);
            let msg = result
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_default();
            assert!(msg.contains("duplicate"), "{label}: {msg}");
            assert_fails_without_echo(result, label);
        }
    }

    #[test]
    fn non_inspectable_output_items_fail_closed_without_echo() {
        let items = [
            "web_search_call",
            "file_search_call",
            "computer_call",
            "image_generation_call",
            "code_interpreter_call",
            "local_shell_call",
            "shell_call",
            "apply_patch_call",
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "custom_tool_call",
            "function_call_output",
            "compaction",
            MARKER,
        ];
        for t in items {
            let body = minimal(json!([{"type": t, "id": "x", "result": MARKER,
                "output": MARKER, "action": {"query": MARKER}}]));
            assert_fails_without_echo(decode(&body), t);
        }
        for bad in [json!({}), json!("message"), json!({"type": 1}), json!(null)] {
            assert_fails_without_echo(decode(&minimal(json!([bad]))), "malformed item");
        }
    }

    #[test]
    fn non_inspectable_content_parts_fail_closed_without_echo() {
        let parts = vec![
            json!({"type": "output_image", "image_url": MARKER}),
            json!({"type": "output_audio", "data": MARKER, "transcript": MARKER}),
            json!({"type": "output_file", "file_id": MARKER}),
            json!({"type": "input_image", "image_url": MARKER}),
            json!({"type": "input_file", "file_data": MARKER}),
            json!({"type": "input_text", "text": MARKER}),
            json!({"type": "reasoning_text", "text": MARKER}),
            json!({"type": MARKER, "text": MARKER}),
            json!({"text": MARKER}),
            json!(MARKER),
        ];
        for part in parts {
            let body = minimal(json!([{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}, part.clone()]}]));
            assert_fails_without_echo(decode(&body), &part.to_string().replace(MARKER, "*"));
        }
    }

    #[test]
    fn encrypted_reasoning_fails_closed_unless_null() {
        for enc in [json!(MARKER), json!(""), json!({"data": MARKER}), json!([])] {
            let body = minimal(json!([{"type": "reasoning", "summary": [],
                "encrypted_content": enc}]));
            assert_fails_without_echo(decode(&body), "encrypted_content");
        }
        let ok = minimal(json!([{"type": "reasoning", "summary": [], "encrypted_content": null}]));
        assert!(decode(&ok).unwrap().texts().is_empty());
    }

    /// Mutation sweep: inject a model-visible-looking field at every object
    /// boundary of the benign fixture; each must fail closed without echo.
    #[test]
    fn unexpected_fields_at_every_boundary_fail_closed() {
        let boundaries = [
            "",
            "/output/0",
            "/output/0/summary/0",
            "/output/0/content/0",
            "/output/1",
            "/output/1/content/0",
            "/output/1/content/1",
            "/output/2",
            "/usage",
            "/usage/input_tokens_details",
            "/usage/output_tokens_details",
            "/reasoning",
            "/text",
            "/text/format",
            "/tools/0",
        ];
        for field in ["smuggled", "output_text", "annotations_text", "transcript"] {
            for at in boundaries {
                let mut body = fixture();
                body.pointer_mut(at)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert(field.to_string(), json!(MARKER));
                assert_fails_without_echo(decode(&body), &format!("{field} at {at:?}"));
            }
        }
    }

    #[test]
    fn model_visible_shape_mutations_fail_closed() {
        let mutations: Vec<(&str, &str, Value)> = vec![
            (
                "annotations",
                "/output/1/content/0/annotations",
                json!([{"type": "url_citation", "title": MARKER, "url": MARKER}]),
            ),
            (
                "logprobs",
                "/output/1/content/0/logprobs",
                json!([{"token": MARKER, "logprob": -0.1}]),
            ),
            (
                "annotations type",
                "/output/1/content/0/annotations",
                json!(MARKER),
            ),
            (
                "text non-string",
                "/output/1/content/0/text",
                json!({"v": MARKER}),
            ),
            (
                "refusal non-string",
                "/output/1/content/1/refusal",
                json!([MARKER]),
            ),
            ("summary non-string", "/output/0/summary/0/text", json!(1)),
            (
                "summary part type",
                "/output/0/summary/0/type",
                json!("reasoning_text"),
            ),
            (
                "content part type",
                "/output/0/content/0/type",
                json!("summary_text"),
            ),
            (
                "reasoning content shape",
                "/output/0/content",
                json!(MARKER),
            ),
            ("message role", "/output/1/role", json!("user")),
            ("message content string", "/output/1/content", json!(MARKER)),
            ("item status", "/output/1/status", json!(MARKER)),
            ("item id", "/output/1/id", json!({"v": MARKER})),
            ("output shape", "/output", json!(MARKER)),
            ("status failed", "/status", json!("failed")),
            ("status in_progress", "/status", json!("in_progress")),
            (
                "error",
                "/error",
                json!({"code": "server_error", "message": MARKER}),
            ),
            (
                "incomplete reason",
                "/incomplete_details",
                json!({"reason": MARKER}),
            ),
            (
                "instructions array",
                "/instructions",
                json!([{"role": "developer", "content": MARKER}]),
            ),
            ("object", "/object", json!(MARKER)),
            ("metadata nested", "/metadata", json!({"k": {"v": MARKER}})),
            (
                "previous_response_id",
                "/previous_response_id",
                json!(MARKER),
            ),
            ("background", "/background", json!(true)),
            (
                "tools hosted",
                "/tools/0",
                json!({"type": "web_search", "name": MARKER}),
            ),
            (
                "tool_choice undeclared",
                "/tool_choice",
                json!({"type": "function", "name": MARKER}),
            ),
            (
                "text json_schema",
                "/text/format",
                json!({"type": "json_schema", "schema": {"description": MARKER}}),
            ),
            ("usage non-integer", "/usage/total_tokens", json!(MARKER)),
        ];
        for (label, path, value) in mutations {
            let mut body = fixture();
            *body.pointer_mut(path).unwrap_or_else(|| panic!("{label}")) = value;
            assert_fails_without_echo(decode(&body), label);
        }

        let mut no_output = fixture();
        no_output.as_object_mut().unwrap().remove("output");
        assert_fails_without_echo(decode(&no_output), "missing output");
        for top in ["output_text", "conversation_text", MARKER] {
            let mut body = fixture();
            body.as_object_mut()
                .unwrap()
                .insert(top.to_string(), json!(MARKER));
            assert_fails_without_echo(decode(&body), top);
        }
    }

    #[test]
    fn malformed_function_calls_fail_closed_without_echo() {
        let call = |patch: Value| {
            let mut c = json!({"type": "function_call", "call_id": "c1", "name": "n",
                "arguments": "{}"});
            for (k, v) in patch.as_object().unwrap() {
                c[k] = v.clone();
            }
            minimal(json!([c]))
        };
        let cases = vec![
            (
                "arguments invalid json",
                call(json!({"arguments": format!("{{\"a\":\"{MARKER}")})),
            ),
            (
                "arguments non-object",
                call(json!({"arguments": format!("\"{MARKER}\"")})),
            ),
            (
                "arguments array",
                call(json!({"arguments": format!("[\"{MARKER}\"]")})),
            ),
            (
                "arguments not string",
                call(json!({"arguments": {"a": MARKER}})),
            ),
            (
                "arguments trailing",
                call(json!({"arguments": format!("{{}} {MARKER}")})),
            ),
            ("empty name", call(json!({"name": ""}))),
            ("name non-string", call(json!({"name": [MARKER]}))),
            ("empty call_id", call(json!({"call_id": ""}))),
            ("missing arguments", {
                let mut b = call(json!({}));
                b["output"][0].as_object_mut().unwrap().remove("arguments");
                b
            }),
        ];
        for (label, body) in cases {
            assert_fails_without_echo(decode(&body), label);
        }

        let dup = minimal(json!([
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            {"type": "function_call", "call_id": "c1", "name": "b", "arguments": "{}"}
        ]));
        assert_fails_without_echo(decode(&dup), "duplicate call_id");
    }

    #[test]
    fn invalid_json_and_non_object_bodies_fail_closed() {
        let truncated = format!("{{\"{MARKER}\"");
        for body in ["", "{", "[]", "\"response\"", "null", truncated.as_str()] {
            assert_fails_without_echo(decode_raw(body), body);
        }
    }

    struct RecordingParser(std::sync::Mutex<Vec<Vec<u8>>>);

    impl JsonParser for RecordingParser {
        fn parse(&self, bytes: &[u8]) -> Result<Value, JsonParseError> {
            self.0.lock().unwrap().push(bytes.to_vec());
            StrictJsonParser.parse(bytes)
        }
    }

    #[test]
    fn decoder_reads_body_and_arguments_through_the_injected_parser() {
        let decoder =
            OpenAiResponsesOutputDecoder::with_parser(RecordingParser(Default::default()));
        let body = minimal(json!([
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{\"x\":1}"}
        ]));
        let raw = serde_json::to_vec(&body).unwrap();
        decoder.decode_response(&raw).unwrap();
        let seen = decoder.parser.0.lock().unwrap();
        assert_eq!(*seen, vec![raw, b"{\"x\":1}".to_vec()]);
    }
}
