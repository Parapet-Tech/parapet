// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Full-buffer executor for OpenAI Responses streams (D-0027 OA-R4b1).
//!
//! Admits one stream under the configured concurrency cap, opens an injected
//! upstream, collects every chunk under the wall-clock and idle bounds,
//! validates the terminal, and applies the streaming policy bridge. It then
//! yields either the collector's exact canonical bytes or a fixed
//! content-free refusal. Nothing is released before the whole stream has
//! passed; there is no callback or partial output.
//!
//! Cancellation is by drop: the executor starts no task, so dropping its
//! future drops the upstream open future or chunk stream and the admission
//! permit.
//!
//! Inert: nothing outside this module and its tests calls it yet.

use crate::config::{Config, ResponsesStreamConfig};
use crate::constraint::ConstraintEvaluator;
use crate::layers::l5a::OutputScanner;
use crate::provider::responses_sse::{SseCollectError, SseCollector};
use axum::http::StatusCode;
use bytes::Bytes;
use futures_util::future::BoxFuture;
use futures_util::stream::{BoxStream, StreamExt};
use std::fmt;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{timeout, timeout_at, Instant};

use super::responses::{enforce_streaming_responses, ResponsesStreamDecision};

/// A content-free upstream transport failure. The transport maps its own
/// error to this unit so no upstream message can reach an outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct UpstreamFault;

/// An established upstream response: status plus body chunks.
pub(super) struct UpstreamResponse {
    pub(super) status: StatusCode,
    pub(super) body: BoxStream<'static, Result<Bytes, UpstreamFault>>,
}

impl fmt::Debug for UpstreamResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpstreamResponse")
            .field("status", &self.status.as_u16())
            .finish_non_exhaustive()
    }
}

/// One-shot upstream factory. `open` is called at most once, and only after
/// admission; dropping the returned future or the response body cancels it.
pub(super) trait ResponsesStreamUpstream: Send {
    fn open(self) -> BoxFuture<'static, Result<UpstreamResponse, UpstreamFault>>;
}

/// Output policy inputs for the streaming bridge.
#[derive(Clone, Copy)]
pub(super) struct StreamPolicy<'a> {
    pub(super) config: &'a Config,
    pub(super) evaluator: &'a dyn ConstraintEvaluator,
    pub(super) scanner: &'a dyn OutputScanner,
}

/// Content-free result of one stream execution.
///
/// Only `Release` carries bytes: the canonical SSE stream, unchanged. `Debug`
/// shows its byte count only.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum ResponsesStreamOutcome {
    Release(Bytes),
    /// The executor's stream bounds differ from the policy config's
    /// `runtime.responses_stream`; nothing was admitted or opened.
    ConfigMismatch,
    /// Every stream slot was taken; upstream was not called.
    ConcurrencyLimit,
    /// The wall-clock budget, measured from request receipt, ran out.
    WallClockTimeout,
    /// No upstream chunk arrived within the idle bound.
    IdleTimeout,
    /// Upstream answered with a status other than `200 OK`.
    UpstreamStatus,
    /// Upstream could not be reached or its body failed mid-stream.
    UpstreamUnavailable,
    /// The stream ended before a terminal event or inside a frame.
    Truncated,
    /// The stream ended in `response.failed` or a top-level `error` event.
    UpstreamFailed,
    /// Framing, decoding, assembly, or a byte/count limit refused the stream.
    InvalidStream,
    BlockedByL3Outbound,
    BlockedByL5a,
    LayerFailure,
}

impl fmt::Debug for ResponsesStreamOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Release(bytes) => {
                return f
                    .debug_struct("Release")
                    .field("bytes", &bytes.len())
                    .finish()
            }
            Self::ConfigMismatch => "ConfigMismatch",
            Self::ConcurrencyLimit => "ConcurrencyLimit",
            Self::WallClockTimeout => "WallClockTimeout",
            Self::IdleTimeout => "IdleTimeout",
            Self::UpstreamStatus => "UpstreamStatus",
            Self::UpstreamUnavailable => "UpstreamUnavailable",
            Self::Truncated => "Truncated",
            Self::UpstreamFailed => "UpstreamFailed",
            Self::InvalidStream => "InvalidStream",
            Self::BlockedByL3Outbound => "BlockedByL3Outbound",
            Self::BlockedByL5a => "BlockedByL5a",
            Self::LayerFailure => "LayerFailure",
        };
        f.write_str(name)
    }
}

impl From<SseCollectError> for ResponsesStreamOutcome {
    fn from(err: SseCollectError) -> Self {
        match err {
            SseCollectError::Truncated | SseCollectError::NoTerminal => Self::Truncated,
            SseCollectError::UpstreamFailed | SseCollectError::UpstreamError => {
                Self::UpstreamFailed
            }
            _ => Self::InvalidStream,
        }
    }
}

impl From<ResponsesStreamDecision> for ResponsesStreamOutcome {
    fn from(decision: ResponsesStreamDecision) -> Self {
        match decision {
            ResponsesStreamDecision::Release(bytes) => Self::Release(bytes),
            ResponsesStreamDecision::BlockedByL3Outbound => Self::BlockedByL3Outbound,
            ResponsesStreamDecision::BlockedByL5a => Self::BlockedByL5a,
            ResponsesStreamDecision::LayerFailure => Self::LayerFailure,
        }
    }
}

/// Concurrency gate plus the full-buffer stream pipeline. One instance is
/// shared by every stream on a route; its semaphore is the stream cap.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct ResponsesStreamExecutor {
    config: ResponsesStreamConfig,
    slots: Arc<Semaphore>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl ResponsesStreamExecutor {
    pub(super) fn new(config: ResponsesStreamConfig) -> Self {
        Self {
            config,
            slots: Arc::new(Semaphore::new(config.concurrent_streams())),
        }
    }

    /// Run one stream to a final outcome.
    ///
    /// `received_at` is when the client request arrived; the wall-clock
    /// budget runs from it, not from this call. A policy config whose
    /// `runtime.responses_stream` differs from the executor's bounds (for
    /// example a later snapshot than the one that built the executor), an
    /// expired budget, or a full gate returns before `upstream` is opened.
    /// The permit is held until this future completes or is dropped.
    ///
    /// Collector finish and policy run synchronously on the polling runtime
    /// worker and cannot be preempted; a result that completes past the
    /// deadline is refused, never released. Before a route calls this, R4b2
    /// must release-benchmark worst-case finish plus policy on a max-size
    /// stream against the `300 s - wall_clock` margin, and record that up
    /// to `concurrent_streams` tails can occupy runtime workers at once
    /// (N-R4b1-2).
    pub(super) async fn execute<U: ResponsesStreamUpstream>(
        &self,
        received_at: Instant,
        upstream: U,
        policy: StreamPolicy<'_>,
    ) -> ResponsesStreamOutcome {
        if self.config != policy.config.runtime.responses_stream {
            return ResponsesStreamOutcome::ConfigMismatch;
        }
        let deadline = received_at + self.config.wall_clock();
        if Instant::now() >= deadline {
            return ResponsesStreamOutcome::WallClockTimeout;
        }
        let Ok(_permit) = Arc::clone(&self.slots).try_acquire_owned() else {
            return ResponsesStreamOutcome::ConcurrencyLimit;
        };
        let outcome = match timeout_at(deadline, self.run(upstream, policy)).await {
            Ok(outcome) => outcome,
            Err(_) => return ResponsesStreamOutcome::WallClockTimeout,
        };
        // Collection and policy are synchronous and cannot be preempted;
        // anything that finishes past the deadline is still refused.
        if Instant::now() >= deadline {
            return ResponsesStreamOutcome::WallClockTimeout;
        }
        outcome
    }

    async fn run<U: ResponsesStreamUpstream>(
        &self,
        upstream: U,
        policy: StreamPolicy<'_>,
    ) -> ResponsesStreamOutcome {
        let Ok(response) = upstream.open().await else {
            return ResponsesStreamOutcome::UpstreamUnavailable;
        };
        if response.status != StatusCode::OK {
            return ResponsesStreamOutcome::UpstreamStatus;
        }
        let mut body = response.body;
        let mut collector = self.config.sse_collector();
        loop {
            match timeout(self.config.upstream_idle(), body.next()).await {
                Err(_) => return ResponsesStreamOutcome::IdleTimeout,
                Ok(None) => break,
                Ok(Some(Err(UpstreamFault))) => return ResponsesStreamOutcome::UpstreamUnavailable,
                Ok(Some(Ok(chunk))) => {
                    if let Err(err) = collector.push(&chunk) {
                        return err.into();
                    }
                }
            }
        }
        drop(body);
        let collected = match collector.finish() {
            Ok(collected) => collected,
            Err(err) => return err.into(),
        };
        enforce_streaming_responses(collected, policy.config, policy.evaluator, policy.scanner)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        load_config, ContentPolicy, EngineConfig, FailureMode, LayerConfig, LayerConfigs,
        PolicyConfig, RuntimeConfig, StringSource, ToolConfig, TrustConfig,
    };
    use crate::constraint::{DslConstraintEvaluator, ToolCallVerdict};
    use crate::layers::l5a::{L5aScanner, ScanResult};
    use crate::message::TrustLevel;
    use futures_util::stream;
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    const MARKER: &str = "EXEC_SECRET_9c1d";
    const CIPHERTEXT: &str = "EncExecCiphertextQQQQ+/==";

    // -- config ---------------------------------------------------------------

    fn stream_config(fields: &[(&str, u64)]) -> ResponsesStreamConfig {
        let body: String = fields
            .iter()
            .map(|(k, v)| format!("  {k}: {v}\n"))
            .collect();
        load_config(&StringSource {
            content: format!("parapet: v1\nresponses_stream:\n{body}"),
        })
        .expect("valid stream config")
        .runtime
        .responses_stream
    }

    fn policy_config(l3: bool, l5a: bool) -> Config {
        let mut tools = HashMap::new();
        tools.insert(
            "save_note".to_string(),
            ToolConfig {
                allowed: true,
                trust: Some(TrustLevel::Untrusted),
                constraints: HashMap::new(),
                result_policy: None,
            },
        );
        Config {
            policy: PolicyConfig {
                version: "v1".to_string(),
                tools,
                block_patterns: Vec::new(),
                canary_tokens: vec![MARKER.to_string()],
                sensitive_patterns: Vec::new(),
                untrusted_content_policy: ContentPolicy::default(),
                trust: TrustConfig::default(),
                layers: LayerConfigs {
                    l3_outbound: l3.then(|| LayerConfig {
                        mode: "block".to_string(),
                        block_action: Some("rewrite".to_string()),
                        window_chars: None,
                    }),
                    l5a: l5a.then(|| LayerConfig {
                        mode: "redact".to_string(),
                        block_action: None,
                        window_chars: None,
                    }),
                    ..LayerConfigs::default()
                },
            },
            runtime: RuntimeConfig {
                engine: EngineConfig {
                    on_failure: FailureMode::Open,
                    timeout_ms: None,
                },
                environment: "test".to_string(),
                responses_stream: ResponsesStreamConfig::default(),
            },
            contract_hash: "sha256:test".to_string(),
        }
    }

    /// A policy config whose stream bounds are `fields`, as one loaded
    /// `Config` would carry them.
    fn bound_config(l3: bool, l5a: bool, fields: &[(&str, u64)]) -> Config {
        let mut config = policy_config(l3, l5a);
        config.runtime.responses_stream = stream_config(fields);
        config
    }

    /// The executor a route builds from the same `Config` it enforces.
    fn executor_for(config: &Config) -> ResponsesStreamExecutor {
        ResponsesStreamExecutor::new(config.runtime.responses_stream)
    }

    // -- policy doubles -------------------------------------------------------

    struct BlockAll;
    impl ConstraintEvaluator for BlockAll {
        fn evaluate_tool_calls(
            &self,
            calls: &[crate::message::ToolCall],
            _: &Config,
        ) -> Vec<ToolCallVerdict> {
            calls
                .iter()
                .map(|call| ToolCallVerdict::Block {
                    tool_name: call.name.clone(),
                    reason: MARKER.to_string(),
                })
                .collect()
        }
    }

    struct EmptyEvaluator;
    impl ConstraintEvaluator for EmptyEvaluator {
        fn evaluate_tool_calls(
            &self,
            _: &[crate::message::ToolCall],
            _: &Config,
        ) -> Vec<ToolCallVerdict> {
            Vec::new()
        }
    }

    struct PanicScanner;
    impl OutputScanner for PanicScanner {
        fn scan_and_redact(&self, _: &str, _: &Config) -> ScanResult {
            panic!("{MARKER}")
        }
    }

    /// Blocks the thread until a real-time instant, then reports no finding.
    struct SlowScanner {
        until: std::time::Instant,
        calls: AtomicUsize,
    }
    impl OutputScanner for SlowScanner {
        fn scan_and_redact(&self, content: &str, _: &Config) -> ScanResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let now = std::time::Instant::now();
            if now < self.until {
                std::thread::sleep(self.until - now);
            }
            ScanResult {
                content: content.to_string(),
                redactions: Vec::new(),
            }
        }
    }

    /// Asserts the gate is still held while policy runs.
    struct SlotProbeScanner {
        slots: Arc<Semaphore>,
        held: AtomicBool,
    }
    impl OutputScanner for SlotProbeScanner {
        fn scan_and_redact(&self, content: &str, _: &Config) -> ScanResult {
            if self.slots.available_permits() == 0 {
                self.held.store(true, Ordering::SeqCst);
            }
            ScanResult {
                content: content.to_string(),
                redactions: Vec::new(),
            }
        }
    }

    // -- stream fixtures ------------------------------------------------------

    #[derive(Clone)]
    struct Spec {
        text: String,
        encrypted: Option<&'static str>,
        incomplete: bool,
    }

    impl Spec {
        fn safe() -> Self {
            Self {
                text: "safe text".to_string(),
                encrypted: Some(CIPHERTEXT),
                incomplete: false,
            }
        }
    }

    fn response(status: &str, output: Vec<Value>) -> Value {
        json!({"id": "resp_1", "object": "response", "status": status, "error": null,
            "output": output})
    }

    fn reasoning(spec: &Spec) -> Value {
        let mut item = json!({"type": "reasoning", "id": "rs_1", "summary": []});
        if let Some(enc) = spec.encrypted {
            item["encrypted_content"] = json!(enc);
        }
        item
    }

    fn message(text: Option<&str>) -> Value {
        let content = match text {
            Some(t) => json!([{"type": "output_text", "text": t, "annotations": []}]),
            None => json!([]),
        };
        json!({"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": content})
    }

    fn function(arguments: &str) -> Value {
        json!({"type": "function_call", "id": "fc_1", "status": "completed",
            "call_id": "call_1", "name": "save_note", "arguments": arguments})
    }

    fn number(events: Vec<Value>) -> String {
        events
            .into_iter()
            .enumerate()
            .map(|(seq, mut event)| {
                event["sequence_number"] = json!(seq);
                let kind = event["type"].as_str().unwrap().to_string();
                format!("event: {kind}\ndata: {event}\n\n")
            })
            .collect()
    }

    /// Events up to (not including) the terminal: reasoning with opaque
    /// ciphertext, one text message, and one function call.
    fn body_events(spec: &Spec) -> Vec<Value> {
        let part = |phase: &str, t: &str| {
            json!({"type": format!("response.content_part.{phase}"), "item_id": "msg_1",
                "output_index": 1, "content_index": 0,
                "part": {"type": "output_text", "text": t, "annotations": []}})
        };
        let args = r#"{"path":"note.txt"}"#;
        vec![
            json!({"type": "response.created", "response": response("queued", vec![])}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": reasoning(spec)}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning(spec)}),
            json!({"type": "response.output_item.added", "output_index": 1, "item": message(None)}),
            part("added", ""),
            json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 1,
                "content_index": 0, "delta": spec.text}),
            json!({"type": "response.output_text.done", "item_id": "msg_1", "output_index": 1,
                "content_index": 0, "text": spec.text}),
            part("done", &spec.text),
            json!({"type": "response.output_item.done", "output_index": 1,
                "item": message(Some(&spec.text))}),
            json!({"type": "response.output_item.added", "output_index": 2, "item": function("")}),
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1",
                "output_index": 2, "delta": args}),
            json!({"type": "response.function_call_arguments.done", "item_id": "fc_1",
                "output_index": 2, "name": "save_note", "arguments": args}),
            json!({"type": "response.output_item.done", "output_index": 2, "item": function(args)}),
        ]
    }

    fn sse(spec: &Spec) -> String {
        let mut events = body_events(spec);
        let status = if spec.incomplete {
            "incomplete"
        } else {
            "completed"
        };
        let output = vec![
            reasoning(spec),
            message(Some(&spec.text)),
            function(r#"{"path":"note.txt"}"#),
        ];
        events.push(
            json!({"type": format!("response.{status}"), "response": response(status, output)}),
        );
        number(events)
    }

    fn with_terminal(spec: &Spec, last: Value) -> String {
        let mut events = body_events(spec);
        events.push(last);
        number(events)
    }

    // -- upstream doubles -----------------------------------------------------

    /// Sets its flag when dropped.
    struct DropProbe(Arc<AtomicBool>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    enum Step {
        Chunk(Bytes),
        Fault,
        Delay(Duration),
        Pending,
    }

    /// Scripted upstream. Counts `open` calls and drops `probe` with the
    /// open future (if still pending) or with the body.
    struct Scripted {
        opens: Arc<AtomicUsize>,
        status: StatusCode,
        open_delay: Option<Duration>,
        open_fault: bool,
        steps: Vec<Step>,
        probe: Option<Arc<AtomicBool>>,
    }

    impl Scripted {
        fn new(steps: Vec<Step>) -> Self {
            Self {
                opens: Arc::new(AtomicUsize::new(0)),
                status: StatusCode::OK,
                open_delay: None,
                open_fault: false,
                steps,
                probe: None,
            }
        }

        fn chunks(body: &str, size: usize) -> Self {
            Self::new(
                body.as_bytes()
                    .chunks(size)
                    .map(|c| Step::Chunk(Bytes::copy_from_slice(c)))
                    .collect(),
            )
        }
    }

    impl ResponsesStreamUpstream for Scripted {
        fn open(self) -> BoxFuture<'static, Result<UpstreamResponse, UpstreamFault>> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            let probe = self.probe.map(DropProbe);
            Box::pin(async move {
                if let Some(delay) = self.open_delay {
                    tokio::time::sleep(delay).await;
                }
                if self.open_fault {
                    return Err(UpstreamFault);
                }
                let body = stream::unfold(
                    (self.steps.into_iter(), probe),
                    |(mut steps, probe)| async move {
                        loop {
                            match steps.next()? {
                                Step::Chunk(bytes) => return Some((Ok(bytes), (steps, probe))),
                                Step::Fault => return Some((Err(UpstreamFault), (steps, probe))),
                                Step::Delay(d) => tokio::time::sleep(d).await,
                                Step::Pending => std::future::pending::<()>().await,
                            }
                        }
                    },
                );
                Ok(UpstreamResponse {
                    status: self.status,
                    body: body.boxed(),
                })
            })
        }
    }

    // -- helpers --------------------------------------------------------------

    fn policy<'a>(
        config: &'a Config,
        evaluator: &'a dyn ConstraintEvaluator,
        scanner: &'a dyn OutputScanner,
    ) -> StreamPolicy<'a> {
        StreamPolicy {
            config,
            evaluator,
            scanner,
        }
    }

    fn assert_refused(outcome: &ResponsesStreamOutcome, expected: ResponsesStreamOutcome) {
        assert_eq!(*outcome, expected);
        let debug = format!("{outcome:?}");
        for leak in [
            MARKER,
            CIPHERTEXT,
            "event",
            "resp_1",
            "save_note",
            "note.txt",
            "{",
        ] {
            assert!(!debug.contains(leak), "outcome leaks {leak}: {debug}");
        }
    }

    async fn run_with(
        executor: &ResponsesStreamExecutor,
        upstream: Scripted,
        config: &Config,
        evaluator: &dyn ConstraintEvaluator,
        scanner: &dyn OutputScanner,
    ) -> (ResponsesStreamOutcome, usize) {
        let opens = Arc::clone(&upstream.opens);
        let outcome = executor
            .execute(Instant::now(), upstream, policy(config, evaluator, scanner))
            .await;
        (outcome, opens.load(Ordering::SeqCst))
    }

    async fn run_default(upstream: Scripted) -> (ResponsesStreamOutcome, usize) {
        let executor = ResponsesStreamExecutor::new(ResponsesStreamConfig::default());
        let config = policy_config(true, true);
        run_with(
            &executor,
            upstream,
            &config,
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        )
        .await
    }

    // -- release --------------------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn safe_completed_and_incomplete_release_exact_canonical_bytes() {
        for incomplete in [false, true] {
            let spec = Spec {
                incomplete,
                ..Spec::safe()
            };
            let body = sse(&spec);
            assert!(body.contains(CIPHERTEXT), "fixture carries ciphertext");
            for size in [1, 7, body.len()] {
                let (outcome, opens) = run_default(Scripted::chunks(&body, size)).await;
                assert_eq!(opens, 1);
                let ResponsesStreamOutcome::Release(bytes) = &outcome else {
                    panic!("expected release, got {outcome:?}")
                };
                assert_eq!(bytes.as_ref(), body.as_bytes());
                let debug = format!("{outcome:?}");
                assert_eq!(debug, format!("Release {{ bytes: {} }}", body.len()));
                assert!(!debug.contains(CIPHERTEXT));
            }
        }
    }

    // -- policy refusals ------------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn every_policy_refusal_suppresses_all_bytes() {
        let marked = Spec {
            text: format!("x{MARKER}"),
            ..Spec::safe()
        };
        let mut denying = policy_config(true, false);
        denying.policy.tools.get_mut("save_note").unwrap().allowed = false;
        let l3 = policy_config(true, false);
        let l5a = policy_config(false, true);
        let dsl = DslConstraintEvaluator::new();
        let safe = Spec::safe();
        let cases: Vec<(
            &Spec,
            &Config,
            &dyn ConstraintEvaluator,
            &dyn OutputScanner,
            _,
        )> = vec![
            (
                &safe,
                &l3,
                &BlockAll,
                &L5aScanner,
                ResponsesStreamOutcome::BlockedByL3Outbound,
            ),
            (
                &safe,
                &denying,
                &dsl,
                &L5aScanner,
                ResponsesStreamOutcome::BlockedByL3Outbound,
            ),
            (
                &safe,
                &l3,
                &EmptyEvaluator,
                &L5aScanner,
                ResponsesStreamOutcome::LayerFailure,
            ),
            (
                &marked,
                &l5a,
                &dsl,
                &L5aScanner,
                ResponsesStreamOutcome::BlockedByL5a,
            ),
            (
                &safe,
                &l5a,
                &dsl,
                &PanicScanner,
                ResponsesStreamOutcome::LayerFailure,
            ),
        ];
        let executor = ResponsesStreamExecutor::new(ResponsesStreamConfig::default());
        for (spec, config, evaluator, scanner, expected) in cases {
            for incomplete in [false, true] {
                let spec = Spec {
                    incomplete,
                    ..spec.clone()
                };
                let upstream = Scripted::chunks(&sse(&spec), 64);
                let (outcome, opens) =
                    run_with(&executor, upstream, config, evaluator, scanner).await;
                assert_eq!(opens, 1);
                assert_refused(&outcome, expected.clone());
            }
        }
        assert_eq!(executor.slots.available_permits(), 4);
    }

    // -- upstream and collector refusals --------------------------------------

    #[tokio::test(start_paused = true)]
    async fn upstream_failures_and_every_collector_error_suppress_all_bytes() {
        let spec = Spec::safe();
        let body = sse(&spec);
        let failed = json!({"type": "response.failed", "response": {"id": "resp_1",
            "object": "response", "status": "failed",
            "error": {"code": "server_error", "message": MARKER}, "output": []}});
        let errored =
            json!({"type": "error", "code": "server_error", "message": MARKER, "param": null});

        let mut rejected = Scripted::chunks(&body, 64);
        rejected.status = StatusCode::UNAUTHORIZED;
        let mut unreachable = Scripted::chunks(&body, 64);
        unreachable.open_fault = true;
        let mut mid_fault = Scripted::chunks(&body[..body.len() / 2], 64);
        mid_fault.steps.push(Step::Fault);

        let cases = [
            (rejected, ResponsesStreamOutcome::UpstreamStatus),
            (unreachable, ResponsesStreamOutcome::UpstreamUnavailable),
            (mid_fault, ResponsesStreamOutcome::UpstreamUnavailable),
            // EOF inside a frame and EOF after whole frames without a terminal.
            (
                Scripted::chunks(&body[..body.len() - 1], 64),
                ResponsesStreamOutcome::Truncated,
            ),
            (
                Scripted::chunks(&number(body_events(&spec)), 64),
                ResponsesStreamOutcome::Truncated,
            ),
            (Scripted::new(Vec::new()), ResponsesStreamOutcome::Truncated),
            (
                Scripted::chunks(&with_terminal(&spec, failed), 64),
                ResponsesStreamOutcome::UpstreamFailed,
            ),
            (
                Scripted::chunks(&with_terminal(&spec, errored), 64),
                ResponsesStreamOutcome::UpstreamFailed,
            ),
            // Framing, decoding, and bytes after the terminal.
            (
                Scripted::chunks("event: x\rdata: {}\n\n", 64),
                ResponsesStreamOutcome::InvalidStream,
            ),
            (
                Scripted::chunks("data: not json\n\n", 64),
                ResponsesStreamOutcome::InvalidStream,
            ),
            (
                Scripted::chunks(&format!("{body}data: {{}}\n\n"), 64),
                ResponsesStreamOutcome::InvalidStream,
            ),
        ];
        for (upstream, expected) in cases {
            let (outcome, opens) = run_default(upstream).await;
            assert_eq!(opens, 1);
            assert_refused(&outcome, expected);
        }
        let collector_errors = [
            SseCollectError::Poisoned,
            SseCollectError::StreamTooLarge,
            SseCollectError::FrameTooLarge,
            SseCollectError::BareCarriageReturn,
            SseCollectError::MalformedFieldName,
            SseCollectError::InvalidEventName,
            SseCollectError::DuplicateEvent,
            SseCollectError::UnsupportedField,
            SseCollectError::MissingData,
            SseCollectError::EmptyData,
            SseCollectError::Decode,
            SseCollectError::Assembly(
                crate::provider::responses_stream_state::StreamStateError::LimitExceeded(
                    crate::provider::responses_stream_state::StreamLimit::Events,
                ),
            ),
            SseCollectError::AfterTerminal,
            SseCollectError::Truncated,
            SseCollectError::NoTerminal,
            SseCollectError::UpstreamFailed,
            SseCollectError::UpstreamError,
        ];
        for err in collector_errors {
            let outcome = ResponsesStreamOutcome::from(err);
            assert!(!matches!(outcome, ResponsesStreamOutcome::Release(_)));
            assert_refused(&outcome, outcome.clone());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn configured_collector_limits_apply() {
        let body = sse(&Spec::safe());
        for (fields, expected) in [
            (
                vec![("max_sse_stream_bytes", body.len() as u64 - 1)],
                ResponsesStreamOutcome::InvalidStream,
            ),
            (
                vec![("max_events", 13)],
                ResponsesStreamOutcome::InvalidStream,
            ),
            (
                vec![("max_opaque_field_bytes", 4)],
                ResponsesStreamOutcome::InvalidStream,
            ),
        ] {
            let config = bound_config(false, false, &fields);
            let executor = executor_for(&config);
            let (outcome, _) = run_with(
                &executor,
                Scripted::chunks(&body, 64),
                &config,
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            )
            .await;
            assert_refused(&outcome, expected);
        }
        let config = bound_config(
            false,
            false,
            &[
                ("max_sse_stream_bytes", body.len() as u64),
                ("max_events", 14),
            ],
        );
        let exact = executor_for(&config);
        let (outcome, _) = run_with(
            &exact,
            Scripted::chunks(&body, 64),
            &config,
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        )
        .await;
        assert!(matches!(outcome, ResponsesStreamOutcome::Release(_)));
    }

    // -- admission ------------------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn full_gate_refuses_before_upstream_and_releases_on_completion() {
        let config = bound_config(true, true, &[("concurrent_streams", 1)]);
        let executor = executor_for(&config);
        let evaluator = DslConstraintEvaluator::new();
        let body = sse(&Spec::safe());

        // Hold the only slot with a stream parked mid-body.
        let mut parked = Scripted::chunks(&body[..10], 64);
        parked.steps.push(Step::Pending);
        let mut first = Box::pin(executor.execute(
            Instant::now(),
            parked,
            policy(&config, &evaluator, &L5aScanner),
        ));
        assert!(futures_util::poll!(first.as_mut()).is_pending());
        assert_eq!(executor.slots.available_permits(), 0);

        let (outcome, opens) = run_with(
            &executor,
            Scripted::chunks(&body, 64),
            &config,
            &evaluator,
            &L5aScanner,
        )
        .await;
        assert_eq!(opens, 0, "overflow must not call upstream");
        assert_refused(&outcome, ResponsesStreamOutcome::ConcurrencyLimit);
        assert_eq!(executor.slots.available_permits(), 0);

        drop(first);
        assert_eq!(executor.slots.available_permits(), 1);

        // The permit is held through collector finish and policy.
        let probe = SlotProbeScanner {
            slots: Arc::clone(&executor.slots),
            held: AtomicBool::new(false),
        };
        let (outcome, _) = run_with(
            &executor,
            Scripted::chunks(&body, 64),
            &config,
            &evaluator,
            &probe,
        )
        .await;
        assert!(matches!(outcome, ResponsesStreamOutcome::Release(_)));
        assert!(
            probe.held.load(Ordering::SeqCst),
            "permit held during policy"
        );
        assert_eq!(executor.slots.available_permits(), 1, "released on return");

        // A completed stream frees its slot for the next one.
        let (outcome, opens) = run_with(
            &executor,
            Scripted::chunks(&body, 64),
            &config,
            &evaluator,
            &L5aScanner,
        )
        .await;
        assert_eq!(opens, 1);
        assert!(matches!(outcome, ResponsesStreamOutcome::Release(_)));
        assert_eq!(executor.slots.available_permits(), 1);
    }

    // -- config coherence -----------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn config_mismatch_refuses_before_admission_and_upstream() {
        let config = policy_config(true, true);
        let body = sse(&Spec::safe());
        for fields in [
            vec![("concurrent_streams", 1)],
            vec![("wall_clock_ms", 1_000)],
            vec![("max_events", 13)],
        ] {
            let executor = ResponsesStreamExecutor::new(stream_config(&fields));
            let (outcome, opens) = run_with(
                &executor,
                Scripted::chunks(&body, 64),
                &config,
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            )
            .await;
            assert_eq!(opens, 0, "mismatch must not call upstream");
            assert_refused(&outcome, ResponsesStreamOutcome::ConfigMismatch);
            assert_eq!(format!("{outcome:?}"), "ConfigMismatch");
            assert_eq!(
                executor.slots.available_permits(),
                executor.config.concurrent_streams(),
                "mismatch must not take a permit"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn config_mismatch_precedes_expired_budget_and_full_gate() {
        let bound = bound_config(false, false, &[("concurrent_streams", 1)]);
        let executor = executor_for(&bound);
        let other = policy_config(false, false);
        let evaluator = DslConstraintEvaluator::new();
        let body = sse(&Spec::safe());

        // Fill the only slot with a matching stream parked mid-body.
        let mut parked = Scripted::chunks(&body[..10], 64);
        parked.steps.push(Step::Pending);
        let mut first = Box::pin(executor.execute(
            Instant::now(),
            parked,
            policy(&bound, &evaluator, &L5aScanner),
        ));
        assert!(futures_util::poll!(first.as_mut()).is_pending());
        assert_eq!(executor.slots.available_permits(), 0);

        // A full gate and an expired budget would each refuse; the
        // mismatch is reported first, without touching upstream.
        let received_at = Instant::now();
        tokio::time::advance(executor.config.wall_clock()).await;
        let upstream = Scripted::chunks(&body, 64);
        let opens = Arc::clone(&upstream.opens);
        let outcome = executor
            .execute(
                received_at,
                upstream,
                policy(&other, &evaluator, &L5aScanner),
            )
            .await;
        assert_eq!(opens.load(Ordering::SeqCst), 0);
        assert_refused(&outcome, ResponsesStreamOutcome::ConfigMismatch);
        assert_eq!(executor.slots.available_permits(), 0);

        // The same full gate refuses a matching config as ConcurrencyLimit.
        let (outcome, opens) = run_with(
            &executor,
            Scripted::chunks(&body, 64),
            &bound,
            &evaluator,
            &L5aScanner,
        )
        .await;
        assert_eq!(opens, 0);
        assert_refused(&outcome, ResponsesStreamOutcome::ConcurrencyLimit);

        drop(first);
        assert_eq!(executor.slots.available_permits(), 1);
    }

    /// Every refusal prints a fixed name only. The match is exhaustive, so a
    /// new outcome cannot be added without listing it here (and in R4b2's
    /// status mapping).
    #[test]
    fn every_refusal_debug_is_a_fixed_name() {
        use ResponsesStreamOutcome as O;
        let refusals = [
            (O::ConfigMismatch, "ConfigMismatch"),
            (O::ConcurrencyLimit, "ConcurrencyLimit"),
            (O::WallClockTimeout, "WallClockTimeout"),
            (O::IdleTimeout, "IdleTimeout"),
            (O::UpstreamStatus, "UpstreamStatus"),
            (O::UpstreamUnavailable, "UpstreamUnavailable"),
            (O::Truncated, "Truncated"),
            (O::UpstreamFailed, "UpstreamFailed"),
            (O::InvalidStream, "InvalidStream"),
            (O::BlockedByL3Outbound, "BlockedByL3Outbound"),
            (O::BlockedByL5a, "BlockedByL5a"),
            (O::LayerFailure, "LayerFailure"),
        ];
        let mut seen = std::collections::HashSet::new();
        for (outcome, name) in refusals {
            match outcome {
                O::Release(_) => unreachable!("release is not a refusal"),
                O::ConfigMismatch
                | O::ConcurrencyLimit
                | O::WallClockTimeout
                | O::IdleTimeout
                | O::UpstreamStatus
                | O::UpstreamUnavailable
                | O::Truncated
                | O::UpstreamFailed
                | O::InvalidStream
                | O::BlockedByL3Outbound
                | O::BlockedByL5a
                | O::LayerFailure => {}
            }
            assert_eq!(format!("{outcome:?}"), name);
            assert_refused(&outcome, outcome.clone());
            assert!(seen.insert(name), "duplicate {name}");
        }
        assert_eq!(seen.len(), 12);
    }

    // -- wall clock -----------------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn expired_budget_never_calls_upstream() {
        let config = bound_config(false, false, &[("wall_clock_ms", 1_000)]);
        let executor = executor_for(&config);
        let evaluator = DslConstraintEvaluator::new();
        let received_at = Instant::now();
        tokio::time::advance(Duration::from_millis(1_000)).await;
        let upstream = Scripted::chunks(&sse(&Spec::safe()), 64);
        let opens = Arc::clone(&upstream.opens);
        let outcome = executor
            .execute(
                received_at,
                upstream,
                policy(&config, &evaluator, &L5aScanner),
            )
            .await;
        assert_refused(&outcome, ResponsesStreamOutcome::WallClockTimeout);
        assert_eq!(opens.load(Ordering::SeqCst), 0);
        assert_eq!(executor.slots.available_permits(), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn wall_clock_runs_from_request_receipt_not_executor_entry() {
        let config = bound_config(
            false,
            false,
            &[("wall_clock_ms", 1_000), ("upstream_idle_ms", 120_000)],
        );
        let executor = executor_for(&config);
        let evaluator = DslConstraintEvaluator::new();
        let body = sse(&Spec::safe());
        let slow = || {
            let mut upstream = Scripted::chunks(&body, 64);
            upstream
                .steps
                .insert(0, Step::Delay(Duration::from_millis(300)));
            upstream
        };

        let received_at = Instant::now();
        tokio::time::advance(Duration::from_millis(800)).await;
        // 300 ms of work fits a fresh budget but not the remaining 200 ms.
        let late = executor
            .execute(
                received_at,
                slow(),
                policy(&config, &evaluator, &L5aScanner),
            )
            .await;
        assert_refused(&late, ResponsesStreamOutcome::WallClockTimeout);
        assert_eq!(Instant::now() - received_at, Duration::from_millis(1_000));

        let fresh = executor
            .execute(
                Instant::now(),
                slow(),
                policy(&config, &evaluator, &L5aScanner),
            )
            .await;
        assert!(matches!(fresh, ResponsesStreamOutcome::Release(_)));
        assert_eq!(executor.slots.available_permits(), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn wall_clock_expiry_during_establishment_or_chunks_fails_closed() {
        let config = bound_config(
            false,
            false,
            &[("wall_clock_ms", 1_000), ("upstream_idle_ms", 120_000)],
        );
        let executor = executor_for(&config);
        let body = sse(&Spec::safe());

        let probe = Arc::new(AtomicBool::new(false));
        let mut establishing = Scripted::chunks(&body, 64);
        establishing.open_delay = Some(Duration::from_secs(5));
        establishing.probe = Some(Arc::clone(&probe));

        // Each chunk is well within idle, but together they exceed the budget.
        let steps = body
            .as_bytes()
            .chunks(body.len() / 4 + 1)
            .flat_map(|c| {
                [
                    Step::Delay(Duration::from_millis(400)),
                    Step::Chunk(Bytes::copy_from_slice(c)),
                ]
            })
            .collect();
        let dribbling = Scripted::new(steps);

        for upstream in [establishing, dribbling] {
            let start = Instant::now();
            let (outcome, opens) = run_with(
                &executor,
                upstream,
                &config,
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            )
            .await;
            assert_eq!(opens, 1);
            assert_refused(&outcome, ResponsesStreamOutcome::WallClockTimeout);
            assert_eq!(Instant::now() - start, Duration::from_millis(1_000));
            assert_eq!(executor.slots.available_permits(), 4);
        }
        assert!(
            probe.load(Ordering::SeqCst),
            "pending open dropped on timeout"
        );
    }

    /// Real time, because a synchronous scanner cannot advance a paused
    /// clock. Sleeps a few hundred milliseconds at most.
    #[tokio::test]
    async fn wall_clock_expiry_during_policy_fails_closed() {
        let config = bound_config(false, true, &[("wall_clock_ms", 300)]);
        let executor = executor_for(&config);
        let evaluator = DslConstraintEvaluator::new();
        let received_at = Instant::now();
        let scanner = SlowScanner {
            until: (received_at + Duration::from_millis(320)).into_std(),
            calls: AtomicUsize::new(0),
        };
        let outcome = executor
            .execute(
                received_at,
                Scripted::chunks(&sse(&Spec::safe()), 64),
                policy(&config, &evaluator, &scanner),
            )
            .await;
        assert!(scanner.calls.load(Ordering::SeqCst) > 0, "policy ran");
        assert_refused(&outcome, ResponsesStreamOutcome::WallClockTimeout);
        assert_eq!(executor.slots.available_permits(), 4);
    }

    // -- idle -----------------------------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn idle_bound_applies_before_first_byte_and_between_chunks() {
        let config = bound_config(
            false,
            false,
            &[("wall_clock_ms", 299_000), ("upstream_idle_ms", 1_000)],
        );
        let executor = executor_for(&config);
        let body = sse(&Spec::safe());

        let silent = Scripted::new(vec![Step::Pending]);
        let mut stalled = Scripted::chunks(&body[..body.len() / 2], 64);
        stalled.steps.push(Step::Pending);
        let mut first_late = Scripted::chunks(&body, 64);
        first_late
            .steps
            .insert(0, Step::Delay(Duration::from_millis(1_001)));

        for upstream in [silent, stalled, first_late] {
            let start = Instant::now();
            let (outcome, _) = run_with(
                &executor,
                upstream,
                &config,
                &DslConstraintEvaluator::new(),
                &L5aScanner,
            )
            .await;
            assert_refused(&outcome, ResponsesStreamOutcome::IdleTimeout);
            assert!(Instant::now() - start <= Duration::from_millis(1_001));
        }

        // Gaps just inside the bound keep the stream alive.
        let steps = body
            .as_bytes()
            .chunks(64)
            .flat_map(|c| {
                [
                    Step::Delay(Duration::from_millis(999)),
                    Step::Chunk(Bytes::copy_from_slice(c)),
                ]
            })
            .collect();
        let (outcome, _) = run_with(
            &executor,
            Scripted::new(steps),
            &config,
            &DslConstraintEvaluator::new(),
            &L5aScanner,
        )
        .await;
        assert!(matches!(outcome, ResponsesStreamOutcome::Release(_)));
    }

    // -- cancellation ---------------------------------------------------------

    async fn assert_drop_cancels<F: Future<Output = ResponsesStreamOutcome>>(
        fut: F,
        slots: &Semaphore,
        probe: &AtomicBool,
    ) {
        let mut fut = Box::pin(fut);
        assert!(futures_util::poll!(fut.as_mut()).is_pending());
        assert_eq!(slots.available_permits(), 0);
        assert!(!probe.load(Ordering::SeqCst));
        drop(fut);
        assert!(probe.load(Ordering::SeqCst), "upstream dropped");
        assert_eq!(slots.available_permits(), 1, "permit released");
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_executor_future_drops_upstream_and_releases_permit() {
        let config = bound_config(false, false, &[("concurrent_streams", 1)]);
        let executor = executor_for(&config);
        let evaluator = DslConstraintEvaluator::new();
        let body = sse(&Spec::safe());

        // Pending establishment.
        let opening_probe = Arc::new(AtomicBool::new(false));
        let mut opening = Scripted::chunks(&body, 64);
        opening.open_delay = Some(Duration::from_secs(10));
        opening.probe = Some(Arc::clone(&opening_probe));

        // Body parked mid-stream.
        let body_probe = Arc::new(AtomicBool::new(false));
        let mut parked = Scripted::chunks(&body[..body.len() / 2], 64);
        parked.steps.push(Step::Pending);
        parked.probe = Some(Arc::clone(&body_probe));

        for (upstream, probe) in [(opening, opening_probe), (parked, body_probe)] {
            let fut = executor.execute(
                Instant::now(),
                upstream,
                policy(&config, &evaluator, &L5aScanner),
            );
            assert_drop_cancels(fut, &executor.slots, &probe).await;

            // The released slot admits the next execution.
            let (outcome, opens) = run_with(
                &executor,
                Scripted::chunks(&body, 64),
                &config,
                &evaluator,
                &L5aScanner,
            )
            .await;
            assert_eq!(opens, 1);
            assert!(matches!(outcome, ResponsesStreamOutcome::Release(_)));
        }
    }

    #[test]
    fn execute_future_is_send() {
        fn assert_send<T: Send>(_: &T) {}
        let executor = ResponsesStreamExecutor::new(ResponsesStreamConfig::default());
        let config = policy_config(false, false);
        let evaluator = DslConstraintEvaluator::new();
        let fut = executor.execute(
            Instant::now(),
            Scripted::new(Vec::new()),
            policy(&config, &evaluator, &L5aScanner),
        );
        assert_send(&fut);
    }

    // -- reachability ---------------------------------------------------------

    /// R4b1 is inert: no production code outside this module names the
    /// executor, and this module spawns nothing.
    #[test]
    fn executor_has_no_production_caller_and_spawns_nothing() {
        fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    rust_files(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let this = src.join("engine").join("responses_stream_exec.rs");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        assert!(files.contains(&this));
        for file in files.iter().filter(|file| **file != this) {
            let text = std::fs::read_to_string(file).unwrap();
            for symbol in [
                "responses_stream_exec::",
                "ResponsesStreamExecutor",
                "ResponsesStreamOutcome",
                "ResponsesStreamUpstream",
            ] {
                assert!(
                    !text.contains(symbol),
                    "{} references {symbol}",
                    file.display()
                );
            }
        }
        let text = std::fs::read_to_string(&this).unwrap();
        let production = text.split("#[cfg(test)]\nmod tests").next().unwrap();
        for forbidden in ["spawn", "std::thread", "std::process", "reqwest", "hyper"] {
            assert!(!production.contains(forbidden), "executor uses {forbidden}");
        }
    }
}
