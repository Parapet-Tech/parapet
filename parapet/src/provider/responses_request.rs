// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses API streaming request preparer -- D-0027 OA-R2a
//
// Responsibilities:
// - Enforce an injected inclusive request-body byte bound BEFORE parsing,
//   and the same bound again on the canonical upstream bytes AFTER rewrite
//   (canonicalization can grow the body)
// - Parse the body exactly once through the injected [`JsonParser`]
//   (duplicate-safe by default) before any value is inspected or mutated
// - Require `stream` to be exactly `true`; a non-stream request is refused,
//   never converted
// - Accept non-null `encrypted_content` only on `input[]` items whose `type`
//   is `reasoning`, and only as a JSON string of ASCII `[A-Za-z0-9+/=_-]`
//   within injected per-field and aggregate bounds. An empty string is
//   present (withheld and restored like any other); null or missing is
//   absent. The
//   value is opaque: it is withheld from the scan view, so decoded messages
//   carry only `summary` / `reasoning_text` exactly as the live decoder
//   projects them, and it reappears verbatim only in the canonical upstream
//   body. `encrypted_content` anywhere else is refused by the decoder's
//   closed schemas.
// - Decode the scan view with the existing [`OpenAiResponsesDecoder`], so
//   every other field keeps its current closed validation
// - Force `stream_options.include_obfuscation = false` in the canonical
//   upstream body and serialize it deterministically (sorted keys, compact)
// - Return an explicit typed upstream `Accept-Encoding: identity` contract
//
// Fixed no-echo errors only; `Debug` of every result/config/state type is
// redacted to counts and limits.
//
// Not in scope: response-side encrypted handling, route/proxy activation,
// trust, credentials, and provider-host inference. This module is an inert
// seam; nothing calls it on live traffic, and the live decoder still refuses
// every non-null `encrypted_content`.

use super::responses::{
    DecodedResponsesRequest, JsonParseError, JsonParser, OpenAiResponsesDecoder, ScanContentClass,
    StrictJsonParser,
};
use super::responses_output::is_opaque_byte;
use crate::message::Message;
use serde_json::{Map, Value};
use std::fmt;

/// Prepares a client streaming request for future upstream forwarding.
pub trait StreamRequestPreparer: Send + Sync {
    fn prepare(&self, body: &[u8]) -> Result<PreparedStreamRequest, StreamRequestError>;
}

/// Hard byte bounds. Every bound is inclusive: exactly `max` is accepted,
/// `max + 1` fails closed. Opaque bytes are counted inside the body bound,
/// not on top of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRequestLimits {
    /// Whole request body, checked before parsing and again on the
    /// canonical upstream bytes after rewrite. The default matches the
    /// proxy's request `to_bytes` cap.
    pub max_body_bytes: usize,
    /// One reasoning `encrypted_content` string, in bytes.
    pub max_opaque_field_bytes: usize,
    /// Sum of every reasoning `encrypted_content` string, in bytes.
    pub max_opaque_total_bytes: usize,
}

impl StreamRequestLimits {
    /// Reviewed defaults; also the ceiling the runtime config may tighten.
    pub const DEFAULT: Self = Self {
        max_body_bytes: 10 * 1024 * 1024,
        max_opaque_field_bytes: 4 * 1024 * 1024,
        max_opaque_total_bytes: 32 * 1024 * 1024,
    };
}

impl Default for StreamRequestLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Upstream `Accept-Encoding` the caller must send with the prepared body.
///
/// Only `identity` exists: the streaming response must arrive uncompressed so
/// the SSE collector bounds and inspects the bytes the upstream actually sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamAcceptEncoding {
    Identity,
}

impl UpstreamAcceptEncoding {
    pub fn header_value(self) -> &'static str {
        match self {
            Self::Identity => "identity",
        }
    }
}

/// Why the preparer refused. Fixed messages only; never carries request
/// bytes, keys, values, or positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamRequestError {
    #[error("responses request: body byte limit exceeded")]
    BodyTooLarge,
    #[error("responses request: body is not valid JSON")]
    InvalidJson,
    #[error("responses request: body contains a duplicate object key")]
    DuplicateKey,
    #[error("responses request: body must be a JSON object")]
    NotObject,
    #[error("responses request: stream must be true")]
    StreamNotTrue,
    #[error("responses request: reasoning encrypted_content must be a string")]
    OpaqueNotString,
    #[error("responses request: reasoning encrypted_content has an unsupported character")]
    OpaqueCharset,
    #[error("responses request: reasoning encrypted_content field byte limit exceeded")]
    OpaqueFieldTooLarge,
    #[error("responses request: reasoning encrypted_content aggregate byte limit exceeded")]
    OpaqueTotalTooLarge,
    #[error("responses request: request failed Responses validation")]
    Invalid,
    #[error("responses request: upstream body could not be serialized")]
    Serialize,
}

/// A validated streaming request: the scan view plus the canonical upstream
/// body and its required upstream encoding.
///
/// `Debug` is redacted to counts.
pub struct PreparedStreamRequest {
    decoded: DecodedResponsesRequest,
    upstream_body: Vec<u8>,
    accept_encoding: UpstreamAcceptEncoding,
}

impl PreparedStreamRequest {
    /// Scan-internal messages; never contain opaque `encrypted_content`.
    pub fn messages(&self) -> &[Message] {
        &self.decoded.messages
    }

    /// Index-aligned with [`Self::messages`].
    pub fn scan_classes(&self) -> &[ScanContentClass] {
        &self.decoded.scan_classes
    }

    /// Canonical compact JSON with sorted keys and
    /// `stream_options.include_obfuscation = false`.
    pub fn upstream_body(&self) -> &[u8] {
        &self.upstream_body
    }

    pub fn accept_encoding(&self) -> UpstreamAcceptEncoding {
        self.accept_encoding
    }

    pub fn into_parts(self) -> (DecodedResponsesRequest, Vec<u8>, UpstreamAcceptEncoding) {
        (self.decoded, self.upstream_body, self.accept_encoding)
    }
}

impl fmt::Debug for PreparedStreamRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedStreamRequest")
            .field("messages", &self.decoded.messages.len())
            .field("upstream_body_bytes", &self.upstream_body.len())
            .field("accept_encoding", &self.accept_encoding)
            .finish()
    }
}

/// Reference [`StreamRequestPreparer`] for OpenAI Responses streaming
/// requests.
///
/// `Debug` shows limits only.
pub struct OpenAiResponsesStreamRequestPreparer<P: JsonParser = StrictJsonParser> {
    decoder: OpenAiResponsesDecoder<P>,
    limits: StreamRequestLimits,
}

impl OpenAiResponsesStreamRequestPreparer {
    pub fn new() -> Self {
        Self::with_parts(StrictJsonParser, StreamRequestLimits::default())
    }
}

impl Default for OpenAiResponsesStreamRequestPreparer {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: JsonParser> OpenAiResponsesStreamRequestPreparer<P> {
    /// `parser` reads the body and every function_call `arguments` string.
    pub fn with_parts(parser: P, limits: StreamRequestLimits) -> Self {
        Self {
            decoder: OpenAiResponsesDecoder::with_parser(parser),
            limits,
        }
    }
}

impl<P: JsonParser> fmt::Debug for OpenAiResponsesStreamRequestPreparer<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiResponsesStreamRequestPreparer")
            .field("limits", &self.limits)
            .finish()
    }
}

impl<P: JsonParser> StreamRequestPreparer for OpenAiResponsesStreamRequestPreparer<P> {
    fn prepare(&self, body: &[u8]) -> Result<PreparedStreamRequest, StreamRequestError> {
        if body.len() > self.limits.max_body_bytes {
            return Err(StreamRequestError::BodyTooLarge);
        }
        let mut root = self.decoder.parser().parse(body).map_err(|e| match e {
            JsonParseError::Syntax(_) => StreamRequestError::InvalidJson,
            JsonParseError::DuplicateKey => StreamRequestError::DuplicateKey,
        })?;
        let obj = root.as_object_mut().ok_or(StreamRequestError::NotObject)?;
        if obj.get("stream") != Some(&Value::Bool(true)) {
            return Err(StreamRequestError::StreamNotTrue);
        }

        let withheld = withhold_opaque(obj, &self.limits)?;
        let decoded = self
            .decoder
            .decode_value(&root)
            .map_err(|_| StreamRequestError::Invalid)?;

        let obj = root.as_object_mut().ok_or(StreamRequestError::NotObject)?;
        restore_opaque(obj, withheld);
        force_no_obfuscation(obj);
        let upstream_body = serde_json::to_vec(&root).map_err(|_| StreamRequestError::Serialize)?;
        if upstream_body.len() > self.limits.max_body_bytes {
            return Err(StreamRequestError::BodyTooLarge);
        }

        Ok(PreparedStreamRequest {
            decoded,
            upstream_body,
            accept_encoding: UpstreamAcceptEncoding::Identity,
        })
    }
}

// ---------------------------------------------------------------------------
// Opaque reasoning content
// ---------------------------------------------------------------------------

/// Opaque strings taken out of the scan view, keyed by `input` index.
/// Not `Debug`: it holds the values themselves.
struct WithheldOpaque(Vec<(usize, String)>);

/// Validate and remove (replace with `null`) every non-null
/// `encrypted_content` on a `type: "reasoning"` input item, including the
/// empty string, which counts zero aggregate bytes. Other locations
/// are left untouched for the decoder's closed schemas to refuse.
fn withhold_opaque(
    root: &mut Map<String, Value>,
    limits: &StreamRequestLimits,
) -> Result<WithheldOpaque, StreamRequestError> {
    let mut withheld = Vec::new();
    let Some(Value::Array(items)) = root.get_mut("input") else {
        return Ok(WithheldOpaque(withheld));
    };
    let mut total = 0usize;
    for (index, item) in items.iter_mut().enumerate() {
        let Some(item) = item.as_object_mut() else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("reasoning") {
            continue;
        }
        let Some(slot) = item.get_mut("encrypted_content") else {
            continue;
        };
        let opaque = match slot {
            Value::Null => continue,
            Value::String(s) => s,
            _ => return Err(StreamRequestError::OpaqueNotString),
        };
        if opaque.len() > limits.max_opaque_field_bytes {
            return Err(StreamRequestError::OpaqueFieldTooLarge);
        }
        if !opaque.bytes().all(is_opaque_byte) {
            return Err(StreamRequestError::OpaqueCharset);
        }
        total = total
            .checked_add(opaque.len())
            .filter(|t| *t <= limits.max_opaque_total_bytes)
            .ok_or(StreamRequestError::OpaqueTotalTooLarge)?;
        let Value::String(opaque) = slot.take() else {
            unreachable!("slot was matched as a string above");
        };
        withheld.push((index, opaque));
    }
    Ok(WithheldOpaque(withheld))
}

/// Put each withheld string back at the `input` index it was taken from.
fn restore_opaque(root: &mut Map<String, Value>, withheld: WithheldOpaque) {
    let Some(Value::Array(items)) = root.get_mut("input") else {
        return;
    };
    for (index, opaque) in withheld.0 {
        if let Some(Value::Object(item)) = items.get_mut(index) {
            item.insert("encrypted_content".to_string(), Value::String(opaque));
        }
    }
}

/// `stream_options` was already validated as `null` or a closed object
/// whose only key is a boolean-or-null `include_obfuscation`.
fn force_no_obfuscation(root: &mut Map<String, Value>) {
    let options = root.entry("stream_options").or_insert(Value::Null);
    if !options.is_object() {
        *options = Value::Object(Map::new());
    }
    if let Value::Object(options) = options {
        options.insert("include_obfuscation".to_string(), Value::Bool(false));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;
    use crate::provider::responses::RequestDecoder;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use StreamRequestError as E;

    /// In the opaque alphabet so only no-echo handling keeps it out of output.
    const MARKER: &str = "OPAQUExMARKERx7f3a";

    fn preparer() -> OpenAiResponsesStreamRequestPreparer {
        OpenAiResponsesStreamRequestPreparer::new()
    }

    fn small(field: usize, total: usize) -> OpenAiResponsesStreamRequestPreparer {
        OpenAiResponsesStreamRequestPreparer::with_parts(
            StrictJsonParser,
            StreamRequestLimits {
                max_body_bytes: 1024 * 1024,
                max_opaque_field_bytes: field,
                max_opaque_total_bytes: total,
            },
        )
    }

    fn reasoning(opaque: Value) -> Value {
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "sum"}],
            "content": [{"type": "reasoning_text", "text": "think"}],
            "encrypted_content": opaque
        })
    }

    fn request(input: Vec<Value>) -> Value {
        json!({
            "model": "gpt-5",
            "stream": true,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "high", "summary": "auto"},
            "instructions": "be brief",
            "stream_options": {"include_obfuscation": false},
            "input": input
        })
    }

    fn bytes(v: &Value) -> Vec<u8> {
        serde_json::to_vec(v).unwrap()
    }

    fn upstream(prepared: &PreparedStreamRequest) -> Value {
        serde_json::from_slice(prepared.upstream_body()).unwrap()
    }

    fn assert_no_marker(text: &str) {
        assert!(!text.contains(MARKER), "marker leaked: {text}");
    }

    fn assert_error_no_marker(err: E) {
        assert_no_marker(&err.to_string());
        assert_no_marker(&format!("{err:?}"));
    }

    // -- parse boundary --------------------------------------------------

    #[test]
    fn duplicate_top_level_key_refuses_before_rewrite() {
        let body = br#"{"model":"gpt-5","stream":true,"stream":false,"input":"hi"}"#;
        assert_eq!(preparer().prepare(body).unwrap_err(), E::DuplicateKey);
    }

    #[test]
    fn duplicate_nested_reasoning_key_refuses_before_rewrite() {
        let body = format!(
            r#"{{"stream":true,"input":[{{"type":"reasoning","summary":[],"encrypted_content":"AAAA","encrypted_content":"{MARKER}"}}]}}"#
        );
        let err = preparer().prepare(body.as_bytes()).unwrap_err();
        assert_eq!(err, E::DuplicateKey);
        assert_error_no_marker(err);
    }

    #[test]
    fn invalid_json_and_non_object_refuse_without_position() {
        assert_eq!(
            preparer().prepare(b"{\"stream\":").unwrap_err(),
            E::InvalidJson
        );
        assert_eq!(preparer().prepare(b"[true]").unwrap_err(), E::NotObject);
    }

    /// Records whether it was called, then delegates to the strict parser.
    struct CountingParser(Arc<AtomicUsize>);

    impl JsonParser for CountingParser {
        fn parse(&self, bytes: &[u8]) -> Result<Value, JsonParseError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            StrictJsonParser.parse(bytes)
        }
    }

    #[test]
    fn body_bound_is_inclusive_and_checked_before_parsing() {
        let body = bytes(&request(vec![reasoning(json!("AAAA"))]));
        let calls = Arc::new(AtomicUsize::new(0));
        let limits = |max_body_bytes| StreamRequestLimits {
            max_body_bytes,
            ..StreamRequestLimits::default()
        };

        let at = OpenAiResponsesStreamRequestPreparer::with_parts(
            CountingParser(calls.clone()),
            limits(body.len()),
        );
        assert!(at.prepare(&body).is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let under = OpenAiResponsesStreamRequestPreparer::with_parts(
            CountingParser(calls.clone()),
            limits(body.len() - 1),
        );
        assert_eq!(under.prepare(&body).unwrap_err(), E::BodyTooLarge);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "parser ran on oversized body"
        );
    }

    #[test]
    fn canonical_upstream_bytes_are_bounded_after_rewrite() {
        // No stream_options in the input, so canonicalization grows the body.
        let mut req = request(vec![reasoning(json!(MARKER))]);
        req.as_object_mut().unwrap().remove("stream_options");
        let body = bytes(&req);
        let canonical = preparer().prepare(&body).unwrap().upstream_body().len();
        assert!(canonical > body.len() + 1, "rewrite did not grow the body");

        let calls = Arc::new(AtomicUsize::new(0));
        let with_cap = |max_body_bytes| {
            OpenAiResponsesStreamRequestPreparer::with_parts(
                CountingParser(calls.clone()),
                StreamRequestLimits {
                    max_body_bytes,
                    ..StreamRequestLimits::default()
                },
            )
        };

        let at = with_cap(canonical).prepare(&body).unwrap();
        assert_eq!(at.upstream_body().len(), canonical);

        let err = with_cap(canonical - 1).prepare(&body).unwrap_err();
        assert_eq!(err, E::BodyTooLarge);
        assert_error_no_marker(err);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "input passed the pre-parse bound; the post-rewrite bound refused"
        );
    }

    #[test]
    fn opaque_bytes_count_inside_body_bound() {
        let opaque = "A".repeat(64);
        let body = bytes(&request(vec![reasoning(json!(opaque))]));
        let without = bytes(&request(vec![reasoning(Value::Null)]));
        let p = OpenAiResponsesStreamRequestPreparer::with_parts(
            StrictJsonParser,
            StreamRequestLimits {
                max_body_bytes: without.len() + 32,
                ..StreamRequestLimits::default()
            },
        );
        assert!(p.prepare(&without).is_ok());
        assert_eq!(p.prepare(&body).unwrap_err(), E::BodyTooLarge);
    }

    #[test]
    fn default_limits_match_contract() {
        let limits = StreamRequestLimits::default();
        assert_eq!(limits.max_body_bytes, 10 * 1024 * 1024);
        assert_eq!(limits.max_opaque_field_bytes, 4 * 1024 * 1024);
        assert_eq!(limits.max_opaque_total_bytes, 32 * 1024 * 1024);
    }

    // -- stream ------------------------------------------------------------

    #[test]
    fn stream_must_be_exactly_true() {
        let mut req = json!({"model": "gpt-5", "input": "hi"});
        assert_eq!(
            preparer().prepare(&bytes(&req)).unwrap_err(),
            E::StreamNotTrue
        );
        for bad in [
            json!(null),
            json!(false),
            json!("true"),
            json!(1),
            json!({}),
        ] {
            req["stream"] = bad;
            assert_eq!(
                preparer().prepare(&bytes(&req)).unwrap_err(),
                E::StreamNotTrue
            );
        }
        req["stream"] = json!(true);
        let prepared = preparer().prepare(&bytes(&req)).unwrap();
        assert_eq!(upstream(&prepared)["stream"], json!(true));
    }

    // -- stream_options ------------------------------------------------------

    #[test]
    fn stream_options_canonicalize_to_one_false_obfuscation_flag() {
        let base = json!({"model": "gpt-5", "stream": true, "input": "hi"});
        let mut cases = vec![base.clone()];
        for options in [
            json!(null),
            json!({}),
            json!({"include_obfuscation": null}),
            json!({"include_obfuscation": true}),
            json!({"include_obfuscation": false}),
        ] {
            let mut req = base.clone();
            req["stream_options"] = options;
            cases.push(req);
        }
        for req in cases {
            let prepared = preparer().prepare(&bytes(&req)).unwrap();
            let text = std::str::from_utf8(prepared.upstream_body()).unwrap();
            assert_eq!(text.matches("include_obfuscation").count(), 1, "{text}");
            assert_eq!(
                upstream(&prepared)["stream_options"],
                json!({"include_obfuscation": false})
            );
        }
    }

    #[test]
    fn unknown_or_malformed_stream_options_refuse() {
        for options in [
            json!({"include_obfuscation": false, "include_usage": true}),
            json!({"include_obfuscation": "false"}),
            json!(true),
        ] {
            let req = json!({"stream": true, "input": "hi", "stream_options": options});
            assert_eq!(preparer().prepare(&bytes(&req)).unwrap_err(), E::Invalid);
        }
    }

    // -- opaque encrypted_content -------------------------------------------

    #[test]
    fn base64_and_base64url_opaque_pass_at_inclusive_bounds() {
        // 8 bytes each; field bound 8, aggregate bound 16.
        let std_b64 = "ab+/AZ==";
        let url_b64 = "ab-_09xy";
        let req = request(vec![reasoning(json!(std_b64)), reasoning(json!(url_b64))]);
        let prepared = small(8, 16).prepare(&bytes(&req)).unwrap();
        let up = upstream(&prepared);
        assert_eq!(up["input"][0]["encrypted_content"], json!(std_b64));
        assert_eq!(up["input"][1]["encrypted_content"], json!(url_b64));
    }

    #[test]
    fn opaque_field_bound_plus_one_refuses() {
        let req = request(vec![reasoning(json!(format!("{MARKER}A")))]);
        let field = MARKER.len();
        let err = small(field, field * 4).prepare(&bytes(&req)).unwrap_err();
        assert_eq!(err, E::OpaqueFieldTooLarge);
        assert_error_no_marker(err);
    }

    #[test]
    fn opaque_aggregate_bound_plus_one_refuses() {
        let req = request(vec![
            reasoning(json!(MARKER)),
            reasoning(json!(format!("{MARKER}A"))),
        ]);
        let field = MARKER.len() + 1;
        let err = small(field, MARKER.len() * 2)
            .prepare(&bytes(&req))
            .unwrap_err();
        assert_eq!(err, E::OpaqueTotalTooLarge);
        assert_error_no_marker(err);
    }

    #[test]
    fn opaque_bad_charset_refuses_without_echo() {
        for bad in [
            format!("{MARKER}!"),
            format!("{MARKER} "),
            format!("{MARKER}\n"),
            format!("{MARKER}é"),
            format!("{MARKER}."),
        ] {
            let req = request(vec![reasoning(json!(bad))]);
            let err = preparer().prepare(&bytes(&req)).unwrap_err();
            assert_eq!(err, E::OpaqueCharset);
            assert_error_no_marker(err);
        }
    }

    #[test]
    fn opaque_non_string_refuses() {
        for bad in [json!(1), json!(true), json!([MARKER]), json!({"v": MARKER})] {
            let req = request(vec![reasoning(bad)]);
            let err = preparer().prepare(&bytes(&req)).unwrap_err();
            assert_eq!(err, E::OpaqueNotString);
            assert_error_no_marker(err);
        }
    }

    #[test]
    fn empty_opaque_passes_is_withheld_and_restored_at_zero_bytes() {
        // Aggregate bound 8 is filled exactly by the second item, so the
        // empty string must count zero bytes.
        let req = request(vec![reasoning(json!("")), reasoning(json!("AAAAAAAA"))]);
        let prepared = small(8, 8).prepare(&bytes(&req)).unwrap();

        assert_eq!(upstream(&prepared), req);
        assert_eq!(
            upstream(&prepared)["input"][0]["encrypted_content"],
            json!("")
        );

        let without = request(vec![reasoning(Value::Null), reasoning(Value::Null)]);
        let live = OpenAiResponsesDecoder::new()
            .decode_request(&bytes(&without))
            .unwrap();
        assert_eq!(prepared.messages(), live.as_slice());
    }

    #[test]
    fn null_or_missing_opaque_stays_absent() {
        let mut missing = reasoning(Value::Null);
        missing.as_object_mut().unwrap().remove("encrypted_content");
        for (item, expected) in [(reasoning(Value::Null), Some(Value::Null)), (missing, None)] {
            let req = request(vec![item]);
            let prepared = small(0, 0).prepare(&bytes(&req)).unwrap();
            let up = upstream(&prepared);
            assert_eq!(up["input"][0].get("encrypted_content"), expected.as_ref());
            assert_eq!(up["input"], req["input"]);
        }
    }

    #[test]
    fn opaque_outside_reasoning_items_refuses_without_echo() {
        let wrong = [
            json!({"type": "message", "role": "user", "content": "hi", "encrypted_content": MARKER}),
            json!({"role": "user", "content": "hi", "encrypted_content": MARKER}),
            json!({"type": "function_call_output", "call_id": "c", "output": "o", "encrypted_content": MARKER}),
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi", "encrypted_content": MARKER}
            ]}),
        ];
        for item in wrong {
            let req = request(vec![item]);
            let err = preparer().prepare(&bytes(&req)).unwrap_err();
            assert_eq!(err, E::Invalid);
            assert_error_no_marker(err);
        }
        let mut top = request(vec![]);
        top["encrypted_content"] = json!(MARKER);
        assert_eq!(preparer().prepare(&bytes(&top)).unwrap_err(), E::Invalid);
        let mut nested = request(vec![]);
        nested["reasoning"]["encrypted_content"] = json!(MARKER);
        assert_eq!(preparer().prepare(&bytes(&nested)).unwrap_err(), E::Invalid);
    }

    #[test]
    fn canonical_body_preserves_opaque_and_semantics_while_scan_view_excludes_it() {
        let req = json!({
            "model": "gpt-5",
            "stream": true,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "high", "summary": "auto"},
            "metadata": {"k": "v"},
            "temperature": 0.25,
            "max_output_tokens": 512,
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            "tool_choice": "auto",
            "instructions": "be brief",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                reasoning(json!(MARKER)),
                {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"q\":1}"},
                {"type": "function_call_output", "call_id": "c1", "output": "done"}
            ]
        });
        let prepared = preparer().prepare(&bytes(&req)).unwrap();

        let mut expected = req.clone();
        expected["stream_options"] = json!({"include_obfuscation": false});
        assert_eq!(upstream(&prepared), expected);
        assert_eq!(prepared.upstream_body(), bytes(&expected).as_slice());

        let reasoning_msg = &prepared.messages()[3];
        assert_eq!(reasoning_msg.role, Role::Assistant);
        assert_eq!(reasoning_msg.content, "sumthink");
        assert_no_marker(&format!("{:?}", prepared.messages()));
        assert_eq!(
            prepared.scan_classes(),
            &[
                ScanContentClass::Conversation,
                ScanContentClass::ToolDefinition,
                ScanContentClass::Conversation,
                ScanContentClass::Conversation,
                ScanContentClass::Conversation,
                ScanContentClass::Conversation,
            ]
        );
    }

    #[test]
    fn scan_view_matches_live_decoder_without_opaque() {
        let with = request(vec![reasoning(json!(MARKER))]);
        let without = request(vec![reasoning(Value::Null)]);
        let prepared = preparer().prepare(&bytes(&with)).unwrap();
        let live = OpenAiResponsesDecoder::new()
            .decode_request(&bytes(&without))
            .unwrap();
        assert_eq!(prepared.messages(), live.as_slice());
    }

    #[test]
    fn live_decoder_still_refuses_non_null_opaque() {
        let body = bytes(&request(vec![reasoning(json!(MARKER))]));
        assert!(preparer().prepare(&body).is_ok());
        let err = OpenAiResponsesDecoder::new()
            .decode_request(&body)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid message format: input[0].encrypted_content cannot be inspected"
        );
        assert_no_marker(&err.to_string());
    }

    // -- contract and determinism ------------------------------------------

    #[test]
    fn result_requires_identity_upstream_encoding() {
        let req = json!({"stream": true, "input": "hi"});
        let prepared = preparer().prepare(&bytes(&req)).unwrap();
        assert_eq!(prepared.accept_encoding(), UpstreamAcceptEncoding::Identity);
        assert_eq!(prepared.accept_encoding().header_value(), "identity");
    }

    #[test]
    fn canonical_bytes_are_deterministic_for_equivalent_input() {
        let a = format!(
            r#"{{"stream":true,"model":"gpt-5","input":[{{"type":"reasoning","encrypted_content":"{MARKER}","summary":[]}}],"stream_options":{{"include_obfuscation":true}}}}"#
        );
        let b = format!(
            "{{ \"input\" : [ {{ \"summary\" : [ ], \"type\" : \"reasoning\", \"encrypted_content\" : \"{MARKER}\" }} ],\n \"model\":\"gpt-\\u0035\", \"stream\" : true }}"
        );
        let pa = preparer().prepare(a.as_bytes()).unwrap();
        let pb = preparer().prepare(b.as_bytes()).unwrap();
        assert_eq!(pa.upstream_body(), pb.upstream_body());
        assert_eq!(
            pa.upstream_body(),
            preparer().prepare(a.as_bytes()).unwrap().upstream_body()
        );
    }

    #[test]
    fn debug_and_display_never_expose_opaque() {
        let p = preparer();
        assert_no_marker(&format!("{p:?}"));
        let prepared = p
            .prepare(&bytes(&request(vec![reasoning(json!(MARKER))])))
            .unwrap();
        let debug = format!("{prepared:?}");
        assert_no_marker(&debug);
        assert!(debug.contains("upstream_body_bytes"));
        assert!(std::str::from_utf8(prepared.upstream_body())
            .unwrap()
            .contains(MARKER));
    }
}
