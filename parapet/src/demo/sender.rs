// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `HttpSender` that answers with the demo proposer's call
//! (`parapet-offline-upstream/v0`, demo selectors).

use super::proposer::{DemoProposerV0, ProposerErrorV0, ProposerInputV0};
use super::DemoJourneyV0;
use crate::engine::{HttpBody, HttpError, HttpRequest, HttpResponse, HttpSender};
use crate::message::Role;
use crate::provider::{adapter_for, ProviderType};
use async_trait::async_trait;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::Bytes;

/// Deterministic offline sender driven by an untrusted document.
///
/// Never performs network I/O and ignores every request header, so a request
/// cannot steer the proposal. Like the 1D sender, it rejects a non-HTTPS
/// resolved upstream URL.
pub struct DemoProposerHttpSenderV0<P: DemoProposerV0> {
    journey: DemoJourneyV0,
    proposer: P,
}

impl<P: DemoProposerV0> DemoProposerHttpSenderV0<P> {
    /// Bind one journey to a proposer.
    pub fn new(journey: DemoJourneyV0, proposer: P) -> Self {
        Self { journey, proposer }
    }
}

/// The document: content of the last `user` message in request order. The
/// sender sees the forwarded body without the engine's trust assignment, and
/// demo requests carry exactly one user message.
fn bound_document(provider: ProviderType, body: &[u8]) -> Option<String> {
    adapter_for(provider)
        .parse_request(body)
        .ok()?
        .into_iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content)
}

fn proposer_error(e: ProposerErrorV0) -> HttpError {
    HttpError::Transport(format!("demo_proposer:{}", e.code()))
}

#[async_trait]
impl<P: DemoProposerV0> HttpSender for DemoProposerHttpSenderV0<P> {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        if !request.url.starts_with("https://") {
            return Err(HttpError::Transport(
                "offline fixture requires HTTPS resolved upstream URL".into(),
            ));
        }
        let is_anthropic =
            request.url.contains("anthropic.com") || request.url.ends_with("/v1/messages");
        let provider = if is_anthropic {
            ProviderType::Anthropic
        } else {
            ProviderType::OpenAi
        };
        let document = bound_document(provider, &request.body)
            .ok_or_else(|| proposer_error(ProposerErrorV0::DocumentMissing))?;
        let call = self
            .proposer
            .propose(&ProposerInputV0 {
                journey: self.journey,
                document: &document,
            })
            .map_err(proposer_error)?;
        let payload = if is_anthropic {
            let input: serde_json::Value = serde_json::from_str(&call.arguments_json)
                .map_err(|e| HttpError::Transport(format!("offline serialization failed: {e}")))?;
            serde_json::json!({
                "id": "msg_offline_demo_v0",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_offline_demo_v0_1",
                        "name": call.tool_name,
                        "input": input
                    }
                ],
                "stop_reason": "tool_use"
            })
        } else {
            serde_json::json!({
                "id": "chatcmpl-offline-demo-v0",
                "object": "chat.completion",
                "created": 1_700_000_000u64,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": serde_json::Value::Null,
                            "tool_calls": [
                                {
                                    "id": "call_offline_demo_v0_1",
                                    "type": "function",
                                    "function": {
                                        "name": call.tool_name,
                                        "arguments": call.arguments_json
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            })
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|e| HttpError::Transport(format!("offline serialization failed: {e}")))?;
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(HttpResponse {
            status: StatusCode::OK,
            headers,
            body: HttpBody::Full(Bytes::from(bytes)),
        })
    }
}
