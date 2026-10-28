// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// OpenAI Responses streaming runtime settings.
//
// One validated source for every bound the future streaming route needs:
// request preparation, opaque reasoning, SSE framing, event assembly,
// concurrency, and wall/idle timeouts. Each component is built from the
// same resolved values, so request and response bounds cannot drift.
//
// Inert: resolved during config loading, but nothing outside this module
// and its tests builds components from it yet.

use std::time::Duration;

use crate::provider::responses::StrictJsonParser;
use crate::provider::responses_output::{EncryptedReasoningPolicy, OpaqueReasoningLimits};
use crate::provider::responses_request::{
    OpenAiResponsesStreamRequestPreparer, StreamRequestLimits,
};
use crate::provider::responses_sse::{ResponsesSseCollector, SseLimits};
use crate::provider::responses_stream::OpenAiResponsesStreamDecoder;
use crate::provider::responses_stream_state::{ResponsesStreamAssembler, StreamLimits};

use super::error::ConfigError;
use super::raw::RawResponsesStreamConfig;

/// Validated OpenAI Responses streaming settings and the factory for every
/// streaming component.
///
/// Every bound is inclusive and may only be tightened: a configured value
/// must be in `1..=max`, where `max` is the reviewed default. `Debug` shows
/// numbers only; no headers, hosts, credentials, or stream content are held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponsesStreamConfig {
    concurrent_streams: usize,
    wall_clock_ms: u64,
    upstream_idle_ms: u64,
    request: StreamRequestLimits,
    sse: SseLimits,
    assembler: StreamLimits,
    resident_ceiling_bytes: u64,
}

impl ResponsesStreamConfig {
    pub const DEFAULT_CONCURRENT_STREAMS: usize = 4;
    pub const MAX_CONCURRENT_STREAMS: usize = 8;
    /// Strictly below the 300 s client header timeout.
    pub const DEFAULT_WALL_CLOCK_MS: u64 = 295_000;
    pub const MAX_WALL_CLOCK_MS: u64 = 299_000;
    pub const DEFAULT_UPSTREAM_IDLE_MS: u64 = 60_000;
    pub const MAX_UPSTREAM_IDLE_MS: u64 = 120_000;
    /// Resident bytes per stream byte (raw frames, canonical copy, assembled
    /// state).
    pub const RESIDENT_FACTOR: u64 = 3;

    /// Reviewed defaults. Resident ceiling: 4 streams * 40 MiB * 3 =
    /// 480 MiB. The largest accepted config (8 streams) is 960 MiB.
    pub const DEFAULT: Self = Self {
        concurrent_streams: Self::DEFAULT_CONCURRENT_STREAMS,
        wall_clock_ms: Self::DEFAULT_WALL_CLOCK_MS,
        upstream_idle_ms: Self::DEFAULT_UPSTREAM_IDLE_MS,
        request: StreamRequestLimits::DEFAULT,
        sse: SseLimits::DEFAULT,
        assembler: StreamLimits::DEFAULT,
        resident_ceiling_bytes: match resident_ceiling(
            Self::DEFAULT_CONCURRENT_STREAMS,
            SseLimits::DEFAULT.max_stream_bytes,
        ) {
            Some(bytes) => bytes,
            None => panic!("default resident ceiling overflows"),
        },
    };

    /// Resolve the optional raw block. Omitted fields take their defaults;
    /// zero or anything above the reviewed maximum is refused with an error
    /// that names the field only.
    pub(crate) fn resolve(raw: Option<RawResponsesStreamConfig>) -> Result<Self, ConfigError> {
        let raw = raw.unwrap_or_default();
        let req = StreamRequestLimits::DEFAULT;
        let sse = SseLimits::DEFAULT;
        let asm = StreamLimits::DEFAULT;

        let concurrent_streams = bounded_usize(
            "concurrent_streams",
            raw.concurrent_streams,
            Self::DEFAULT_CONCURRENT_STREAMS,
            Self::MAX_CONCURRENT_STREAMS,
        )?;
        let sse = SseLimits {
            max_stream_bytes: ceiling(
                "max_sse_stream_bytes",
                raw.max_sse_stream_bytes,
                sse.max_stream_bytes,
            )?,
            max_frame_bytes: ceiling(
                "max_sse_frame_bytes",
                raw.max_sse_frame_bytes,
                sse.max_frame_bytes,
            )?,
        };
        let resident_ceiling_bytes = resident_ceiling(concurrent_streams, sse.max_stream_bytes)
            .ok_or_else(|| out_of_range("concurrent_streams"))?;

        Ok(Self {
            concurrent_streams,
            wall_clock_ms: bounded(
                "wall_clock_ms",
                raw.wall_clock_ms,
                Self::DEFAULT_WALL_CLOCK_MS,
                Self::MAX_WALL_CLOCK_MS,
            )?,
            upstream_idle_ms: bounded(
                "upstream_idle_ms",
                raw.upstream_idle_ms,
                Self::DEFAULT_UPSTREAM_IDLE_MS,
                Self::MAX_UPSTREAM_IDLE_MS,
            )?,
            request: StreamRequestLimits {
                max_body_bytes: ceiling("max_body_bytes", raw.max_body_bytes, req.max_body_bytes)?,
                max_opaque_field_bytes: ceiling(
                    "max_opaque_field_bytes",
                    raw.max_opaque_field_bytes,
                    req.max_opaque_field_bytes,
                )?,
                max_opaque_total_bytes: ceiling(
                    "max_opaque_total_bytes",
                    raw.max_opaque_total_bytes,
                    req.max_opaque_total_bytes,
                )?,
            },
            sse,
            assembler: StreamLimits {
                max_events: ceiling("max_events", raw.max_events, asm.max_events)?,
                max_items: ceiling("max_items", raw.max_items, asm.max_items)?,
                max_parts: ceiling("max_parts", raw.max_parts, asm.max_parts)?,
                max_text_bytes: ceiling("max_text_bytes", raw.max_text_bytes, asm.max_text_bytes)?,
                max_argument_bytes: ceiling(
                    "max_argument_bytes",
                    raw.max_argument_bytes,
                    asm.max_argument_bytes,
                )?,
                max_snapshot_bytes: ceiling(
                    "max_snapshot_bytes",
                    raw.max_snapshot_bytes,
                    asm.max_snapshot_bytes,
                )?,
            },
            resident_ceiling_bytes,
        })
    }

    // -- resolved values ----------------------------------------------------

    /// Maximum simultaneous streams for the future route.
    pub fn concurrent_streams(&self) -> usize {
        self.concurrent_streams
    }

    /// Whole-stream wall-clock budget.
    pub fn wall_clock(&self) -> Duration {
        Duration::from_millis(self.wall_clock_ms)
    }

    /// Longest gap between upstream chunks.
    pub fn upstream_idle(&self) -> Duration {
        Duration::from_millis(self.upstream_idle_ms)
    }

    /// `concurrent_streams * max_sse_stream_bytes * RESIDENT_FACTOR`.
    pub fn resident_ceiling_bytes(&self) -> u64 {
        self.resident_ceiling_bytes
    }

    pub fn request_limits(&self) -> StreamRequestLimits {
        self.request
    }

    /// The same opaque bounds the request preparer enforces.
    pub fn opaque_limits(&self) -> OpaqueReasoningLimits {
        OpaqueReasoningLimits {
            max_field_bytes: self.request.max_opaque_field_bytes,
            max_total_bytes: self.request.max_opaque_total_bytes,
        }
    }

    pub fn encrypted_policy(&self) -> EncryptedReasoningPolicy {
        EncryptedReasoningPolicy::AcceptOpaque(self.opaque_limits())
    }

    pub fn sse_limits(&self) -> SseLimits {
        self.sse
    }

    pub fn stream_limits(&self) -> StreamLimits {
        self.assembler
    }

    // -- component factory --------------------------------------------------

    pub fn request_preparer(&self) -> OpenAiResponsesStreamRequestPreparer {
        OpenAiResponsesStreamRequestPreparer::with_parts(StrictJsonParser, self.request)
    }

    pub fn stream_decoder(&self) -> OpenAiResponsesStreamDecoder {
        OpenAiResponsesStreamDecoder::with_policy(StrictJsonParser, self.encrypted_policy())
    }

    pub fn stream_assembler(&self) -> ResponsesStreamAssembler {
        ResponsesStreamAssembler::with_parser(self.assembler, StrictJsonParser)
    }

    /// A collector whose decoder and assembler come from this same config.
    pub fn sse_collector(&self) -> ResponsesSseCollector {
        ResponsesSseCollector::with_parts(self.sse, self.stream_decoder(), self.stream_assembler())
    }
}

impl Default for ResponsesStreamConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// Request and response opaque bounds share one config field each; their
// type defaults must agree so the shared ceiling is the reviewed one.
const _: () = {
    assert!(
        StreamRequestLimits::DEFAULT.max_opaque_field_bytes
            == OpaqueReasoningLimits::DEFAULT.max_field_bytes
    );
    assert!(
        StreamRequestLimits::DEFAULT.max_opaque_total_bytes
            == OpaqueReasoningLimits::DEFAULT.max_total_bytes
    );
    assert!(ResponsesStreamConfig::MAX_WALL_CLOCK_MS < 300_000);
    assert!(
        ResponsesStreamConfig::DEFAULT_CONCURRENT_STREAMS
            <= ResponsesStreamConfig::MAX_CONCURRENT_STREAMS
    );
    assert!(
        ResponsesStreamConfig::DEFAULT_WALL_CLOCK_MS <= ResponsesStreamConfig::MAX_WALL_CLOCK_MS
    );
    assert!(
        ResponsesStreamConfig::DEFAULT_UPSTREAM_IDLE_MS
            <= ResponsesStreamConfig::MAX_UPSTREAM_IDLE_MS
    );
};

const fn resident_ceiling(concurrent_streams: usize, max_stream_bytes: usize) -> Option<u64> {
    match (concurrent_streams as u64).checked_mul(max_stream_bytes as u64) {
        Some(bytes) => bytes.checked_mul(ResponsesStreamConfig::RESIDENT_FACTOR),
        None => None,
    }
}

fn out_of_range(field: &'static str) -> ConfigError {
    ConfigError::Validation(format!("responses_stream.{field} is out of range"))
}

fn bounded(
    field: &'static str,
    value: Option<u64>,
    default: u64,
    max: u64,
) -> Result<u64, ConfigError> {
    match value {
        None => Ok(default),
        Some(v) if (1..=max).contains(&v) => Ok(v),
        Some(_) => Err(out_of_range(field)),
    }
}

fn bounded_usize(
    field: &'static str,
    value: Option<u64>,
    default: usize,
    max: usize,
) -> Result<usize, ConfigError> {
    let max64 = u64::try_from(max).map_err(|_| out_of_range(field))?;
    let default64 = u64::try_from(default).map_err(|_| out_of_range(field))?;
    let v = bounded(field, value, default64, max64)?;
    usize::try_from(v).map_err(|_| out_of_range(field))
}

/// A byte/count bound whose default is also its maximum.
fn ceiling(field: &'static str, value: Option<u64>, max: usize) -> Result<usize, ConfigError> {
    bounded_usize(field, value, max, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{load_config, StringSource};
    use crate::provider::responses_request::{StreamRequestError, StreamRequestPreparer};
    use crate::provider::responses_sse::{SseCollectError, SseCollector};
    use crate::provider::responses_stream::StreamEventDecoder;
    use crate::provider::responses_stream_state::{StreamLimit, StreamStateError};
    use serde_json::{json, Value};

    const MIB: u64 = 1024 * 1024;

    /// Every field with its reviewed maximum.
    const FIELDS: &[(&str, u64)] = &[
        ("concurrent_streams", 8),
        ("wall_clock_ms", 299_000),
        ("upstream_idle_ms", 120_000),
        ("max_body_bytes", 10 * MIB),
        ("max_opaque_field_bytes", 4 * MIB),
        ("max_opaque_total_bytes", 32 * MIB),
        ("max_sse_stream_bytes", 40 * MIB),
        ("max_sse_frame_bytes", 16 * MIB),
        ("max_events", 262_144),
        ("max_items", 256),
        ("max_parts", 1_024),
        ("max_text_bytes", 4 * MIB),
        ("max_argument_bytes", 4 * MIB),
        ("max_snapshot_bytes", 32 * MIB),
    ];

    fn load(yaml: &str) -> Result<ResponsesStreamConfig, ConfigError> {
        load_config(&StringSource {
            content: format!("parapet: v1\n{yaml}"),
        })
        .map(|c| c.runtime.responses_stream)
    }

    fn with(fields: &[(&str, u64)]) -> Result<ResponsesStreamConfig, ConfigError> {
        let body: String = fields
            .iter()
            .map(|(k, v)| format!("  {k}: {v}\n"))
            .collect();
        load(&format!("responses_stream:\n{body}"))
    }

    fn refusal(fields: &[(&str, u64)]) -> String {
        match with(fields) {
            Err(ConfigError::Validation(msg)) => msg,
            other => panic!("{fields:?}: expected validation error, got {other:?}"),
        }
    }

    // -- resolution -----------------------------------------------------------

    #[test]
    fn omitted_block_resolves_to_every_safe_default() {
        for cfg in [load("").unwrap(), load("responses_stream:\n").unwrap()] {
            assert_eq!(cfg, ResponsesStreamConfig::default());
            assert_eq!(cfg.concurrent_streams(), 4);
            assert_eq!(cfg.wall_clock(), Duration::from_millis(295_000));
            assert_eq!(cfg.upstream_idle(), Duration::from_millis(60_000));
            assert_eq!(
                cfg.request_limits(),
                StreamRequestLimits {
                    max_body_bytes: 10 << 20,
                    max_opaque_field_bytes: 4 << 20,
                    max_opaque_total_bytes: 32 << 20,
                }
            );
            assert_eq!(
                cfg.opaque_limits(),
                OpaqueReasoningLimits {
                    max_field_bytes: 4 << 20,
                    max_total_bytes: 32 << 20,
                }
            );
            assert_eq!(
                cfg.encrypted_policy(),
                EncryptedReasoningPolicy::AcceptOpaque(cfg.opaque_limits())
            );
            assert_eq!(
                cfg.sse_limits(),
                SseLimits {
                    max_stream_bytes: 40 << 20,
                    max_frame_bytes: 16 << 20,
                }
            );
            assert_eq!(
                cfg.stream_limits(),
                StreamLimits {
                    max_events: 262_144,
                    max_items: 256,
                    max_parts: 1_024,
                    max_text_bytes: 4 << 20,
                    max_argument_bytes: 4 << 20,
                    max_snapshot_bytes: 32 << 20,
                }
            );
            assert_eq!(cfg.resident_ceiling_bytes(), 480 * MIB);
        }
    }

    #[test]
    fn defaults_and_maxima_are_accepted() {
        let defaults: Vec<(&str, u64)> = FIELDS
            .iter()
            .map(|&(k, max)| match k {
                "concurrent_streams" => (k, 4),
                "wall_clock_ms" => (k, 295_000),
                "upstream_idle_ms" => (k, 60_000),
                _ => (k, max),
            })
            .collect();
        assert_eq!(with(&defaults).unwrap(), ResponsesStreamConfig::default());

        let maxed = with(FIELDS).unwrap();
        assert_eq!(maxed.concurrent_streams(), 8);
        assert_eq!(maxed.wall_clock(), Duration::from_millis(299_000));
        assert_eq!(maxed.upstream_idle(), Duration::from_millis(120_000));
        assert_eq!(maxed.request_limits(), StreamRequestLimits::default());
        assert_eq!(maxed.sse_limits(), SseLimits::default());
        assert_eq!(maxed.stream_limits(), StreamLimits::default());
        for &(k, max) in FIELDS {
            assert!(with(&[(k, max)]).is_ok(), "{k} at max");
            assert!(with(&[(k, 1)]).is_ok(), "{k} at 1");
        }
    }

    #[test]
    fn zero_and_max_plus_one_are_refused_naming_only_the_field() {
        for &(k, max) in FIELDS {
            let expected = format!("responses_stream.{k} is out of range");
            for bad in [0, max + 1, u64::MAX] {
                let msg = refusal(&[(k, bad)]);
                assert_eq!(msg, expected, "{k}={bad}");
                assert!(!msg.contains(&bad.to_string()), "{k} echoes its value");
            }
        }
    }

    #[test]
    fn wall_clock_at_client_timeout_and_nine_streams_are_refused() {
        assert_eq!(
            refusal(&[("wall_clock_ms", 300_000)]),
            "responses_stream.wall_clock_ms is out of range"
        );
        assert_eq!(
            refusal(&[("concurrent_streams", 9)]),
            "responses_stream.concurrent_streams is out of range"
        );
    }

    #[test]
    fn unknown_and_ill_typed_fields_are_refused() {
        assert!(matches!(
            load("responses_stream:\n  max_bytes: 1\n"),
            Err(ConfigError::YamlError(_))
        ));
        assert!(matches!(
            load("responses_stream:\n  max_events: -1\n"),
            Err(ConfigError::YamlError(_))
        ));
        assert!(matches!(
            load("responses_stream:\n  max_events: many\n"),
            Err(ConfigError::YamlError(_))
        ));
    }

    #[test]
    fn resident_ceiling_is_checked_and_documented() {
        assert_eq!(
            ResponsesStreamConfig::DEFAULT.resident_ceiling_bytes(),
            480 * MIB
        );
        assert_eq!(
            with(&[("concurrent_streams", 8)])
                .unwrap()
                .resident_ceiling_bytes(),
            960 * MIB
        );
        let tight = with(&[("concurrent_streams", 2), ("max_sse_stream_bytes", 1_000)]).unwrap();
        assert_eq!(tight.resident_ceiling_bytes(), 6_000);
        assert_eq!(resident_ceiling(usize::MAX, 2), None);
        assert_eq!(resident_ceiling(1, usize::MAX), None);
    }

    #[test]
    fn debug_is_numbers_only() {
        let text = format!("{:?}", with(FIELDS).unwrap());
        for needle in ["http", "Bearer", "authorization", "host", "\""] {
            assert!(
                !text
                    .to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase()),
                "{needle}"
            );
        }
    }

    // -- factory boundary probes ---------------------------------------------

    fn reasoning(enc: &str) -> Value {
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": enc})
    }

    fn request(input: Vec<Value>) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "model": "gpt-5",
            "stream": true,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "stream_options": {"include_obfuscation": false},
            "input": input
        }))
        .unwrap()
    }

    #[test]
    fn request_preparer_enforces_tightened_limits() {
        let body = request(vec![reasoning("AAAA"), reasoning("AAAA")]);
        let n = body.len() as u64;
        let exact = [
            ("max_body_bytes", n),
            ("max_opaque_field_bytes", 4),
            ("max_opaque_total_bytes", 8),
        ];
        let at = |over: Option<&str>| {
            let fields: Vec<_> = exact
                .iter()
                .map(|&(k, v)| (k, if Some(k) == over { v - 1 } else { v }))
                .collect();
            with(&fields).unwrap().request_preparer().prepare(&body)
        };
        assert!(at(None).is_ok(), "exact bounds");
        assert_eq!(
            at(Some("max_body_bytes")).unwrap_err(),
            StreamRequestError::BodyTooLarge
        );
        assert_eq!(
            at(Some("max_opaque_field_bytes")).unwrap_err(),
            StreamRequestError::OpaqueFieldTooLarge
        );
        assert_eq!(
            at(Some("max_opaque_total_bytes")).unwrap_err(),
            StreamRequestError::OpaqueTotalTooLarge
        );
    }

    #[test]
    fn decoder_shares_the_request_opaque_field_limit() {
        let event = |enc: &str| {
            json!({"type": "response.output_item.added", "sequence_number": 2,
                "output_index": 0, "item": reasoning(enc)})
            .to_string()
        };
        let cfg = with(&[("max_opaque_field_bytes", 4)]).unwrap();
        let decoder = cfg.stream_decoder();
        assert!(decoder.decode_event(None, event("AAAA").as_bytes()).is_ok());
        assert!(decoder
            .decode_event(None, event("AAAAA").as_bytes())
            .is_err());
        assert_eq!(
            cfg.request_preparer()
                .prepare(&request(vec![reasoning("AAAAA")]))
                .unwrap_err(),
            StreamRequestError::OpaqueFieldTooLarge
        );
    }

    fn msg(text: &str) -> Value {
        let content = if text.is_empty() {
            json!([])
        } else {
            json!([{"type": "output_text", "text": text, "annotations": []}])
        };
        json!({"type": "message", "id": "msg_1", "status": "completed",
            "role": "assistant", "content": content})
    }

    fn response(status: &str, output: Vec<Value>) -> Value {
        json!({"id": "resp_1", "object": "response", "status": status,
            "error": null, "output": output})
    }

    /// One-message, two-delta stream ("He" + "llo") as canonical named SSE
    /// frames: 9 events, 5 text bytes, plus the largest frame's byte count.
    fn stream() -> (String, u64) {
        let part = |phase: &str, text: &str| {
            json!({"type": format!("response.content_part.{phase}"), "item_id": "msg_1",
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}})
        };
        let delta = |d: &str| {
            json!({"type": "response.output_text.delta", "item_id": "msg_1",
                "output_index": 0, "content_index": 0, "delta": d})
        };
        let mut events = vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": msg("")}),
            part("added", ""),
            delta("He"),
            delta("llo"),
            json!({"type": "response.output_text.done", "item_id": "msg_1",
                "output_index": 0, "content_index": 0, "text": "Hello"}),
            part("done", "Hello"),
            json!({"type": "response.output_item.done", "output_index": 0, "item": msg("Hello")}),
            json!({"type": "response.completed", "response": response("completed", vec![msg("Hello")])}),
        ];
        for (seq, v) in events.iter_mut().enumerate() {
            v["sequence_number"] = json!(seq);
        }
        let frames: Vec<String> = events
            .iter()
            .map(|v| format!("event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap()))
            .collect();
        // Frame bytes exclude the terminating blank line.
        let max_frame = frames.iter().map(|f| f.len() - 1).max().unwrap() as u64;
        (frames.concat(), max_frame)
    }

    #[test]
    fn collector_decoder_and_assembler_enforce_tightened_limits() {
        let (s, max_frame) = stream();
        let exact = [
            ("max_sse_stream_bytes", s.len() as u64),
            ("max_sse_frame_bytes", max_frame),
            ("max_events", 9),
            ("max_items", 1),
            ("max_parts", 1),
            ("max_text_bytes", 5),
        ];
        let run = |over: Option<&str>| {
            let fields: Vec<_> = exact
                .iter()
                .map(|&(k, v)| (k, if Some(k) == over { v - 1 } else { v }))
                .collect();
            let mut c = with(&fields).unwrap().sse_collector();
            c.push(s.as_bytes())?;
            c.finish()
        };
        run(None).expect("exact bounds");
        assert_eq!(
            run(Some("max_sse_stream_bytes")).unwrap_err(),
            SseCollectError::StreamTooLarge
        );
        assert_eq!(
            run(Some("max_sse_frame_bytes")).unwrap_err(),
            SseCollectError::FrameTooLarge
        );
        assert_eq!(
            run(Some("max_events")).unwrap_err(),
            SseCollectError::Assembly(StreamStateError::LimitExceeded(StreamLimit::Events))
        );
        assert_eq!(
            run(Some("max_text_bytes")).unwrap_err(),
            SseCollectError::Assembly(StreamStateError::LimitExceeded(StreamLimit::TextBytes))
        );

        // The collector's decoder is the config's decoder: a reasoning item
        // with a too-long opaque value is refused mid-stream.
        let opaque = json!({"type": "response.output_item.added", "sequence_number": 1,
            "output_index": 0, "item": reasoning("AAAAA")});
        let created = json!({"type": "response.created", "sequence_number": 0,
            "response": response("queued", vec![])});
        let mut c = with(&[("max_opaque_field_bytes", 4)])
            .unwrap()
            .sse_collector();
        c.push(format!("data: {created}\n\n").as_bytes()).unwrap();
        assert_eq!(
            c.push(format!("data: {opaque}\n\n").as_bytes()),
            Err(SseCollectError::Decode)
        );
    }

    #[test]
    fn standalone_assembler_uses_configured_limits() {
        let cfg = with(&[("max_items", 1)]).unwrap();
        let decoder = cfg.stream_decoder();
        let mut asm = cfg.stream_assembler();
        let events = [
            json!({"type": "response.created", "sequence_number": 0, "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": msg("")}),
            json!({"type": "response.output_item.added", "sequence_number": 2, "output_index": 1,
                "item": {"type": "message", "id": "msg_2", "status": "in_progress", "role": "assistant", "content": []}}),
        ];
        use crate::provider::responses_stream_state::StreamAssembler;
        let mut results = events.iter().map(|v| {
            asm.push(
                decoder
                    .decode_event(None, v.to_string().as_bytes())
                    .unwrap(),
            )
        });
        assert!(results.next().unwrap().is_ok());
        assert!(results.next().unwrap().is_ok());
        assert_eq!(
            results.next().unwrap().unwrap_err(),
            StreamStateError::LimitExceeded(StreamLimit::Items)
        );
    }

    // -- reachability ---------------------------------------------------------

    /// R4a is inert: outside the config module, only the inert R4b1 stream
    /// executor reads the resolved streaming settings or calls the factory
    /// in production code.
    #[test]
    fn factory_has_no_production_caller() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let executor = src.join("engine").join("responses_stream_exec.rs");
        assert!(executor.is_file());
        let mut stack = vec![src.clone()];
        let mut offenders = Vec::new();
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    if path != src.join("config") {
                        stack.push(path);
                    }
                    continue;
                }
                // `tests.rs` files are `#[cfg(test)]` modules in their entirety.
                if path.extension().and_then(|e| e.to_str()) != Some("rs")
                    || path.file_name().and_then(|n| n.to_str()) == Some("tests.rs")
                    || path == executor
                {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let production = text.split("#[cfg(test)]").next().unwrap();
                if production.contains("ResponsesStreamConfig")
                    || production.contains(".responses_stream")
                {
                    offenders.push(path.display().to_string());
                }
            }
        }
        assert!(offenders.is_empty(), "production callers: {offenders:?}");
    }
}
