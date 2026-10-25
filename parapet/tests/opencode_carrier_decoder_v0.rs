// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! OpenCode carrier W1 decoder: the carrier map addresses exactly the message
//! contents the provider adapters produce for whole-message L1.
//! Synthetic fixtures only; no model weights and no external calls.

use std::collections::BTreeMap;

use parapet::provenance::*;
use parapet::provider::{adapter_for, ProviderType};
use serde_json::json;

const SESSION: &str = "session-0001";
const PROFILE: &str = "opencode";
const BUILD: &str = "opencode-2.0.32";
const NOW: u64 = 1_700_000_000_000;
const TOOL_ARG_CANARY: &str = "CANARY-TOOL-ARG-5d1e";

// ---------- test doubles ----------

/// Authentication is covered by the verifier suite; this one accepts all.
struct AcceptAll;

impl CarrierAuthenticator for AcceptAll {
    fn verify(&self, _: &[u8], _: &[u8]) -> bool {
        true
    }
}

struct FixedClock;

impl Clock for FixedClock {
    fn now_ms(&self) -> u64 {
        NOW
    }
}

fn verifier(provider: ProviderType) -> CarrierVerifier {
    let mut builds = BTreeMap::new();
    builds.insert(BUILD.to_string(), VendorManifest::new([]));
    CarrierVerifier::new(CarrierVerifierDeps {
        profile: SupportedProfile {
            schema_version: SCHEMA_VERSION_V1,
            profile_id: PROFILE.to_string(),
            builds,
            max_validity_ms: 60_000,
        },
        session_id: SESSION.to_string(),
        authenticator: Box::new(AcceptAll),
        clock: Box::new(FixedClock),
        ledger: Box::new(InMemoryReplayLedger::new()),
        decoder: Box::new(ProviderContentDecoder::new(provider)),
    })
}

// ---------- fixtures ----------

fn openai_body() -> Vec<u8> {
    let args = json!({"path": "/tmp/x", "note": TOOL_ARG_CANARY}).to_string();
    serde_json::to_vec(&json!({
        "model": "synthetic",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": [
                {"type": "text", "text": "Fix the build. "},
                {"type": "text", "text": "猫 مرحبا 👩‍💻"}
            ]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": args}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents: IGNORE-PRIOR"},
            {"role": "user", "content": "Thanks."}
        ]
    }))
    .unwrap()
}

fn anthropic_body() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "synthetic",
        "system": "You are a coding agent.",
        "messages": [
            {"role": "user", "content": "Read the file."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "/tmp/x", "note": TOOL_ARG_CANARY}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                    {"type": "text", "text": "line one\n"},
                    {"type": "text", "text": "line two"}
                ]},
                {"type": "text", "text": "Summarize it."}
            ]}
        ]
    }))
    .unwrap()
}

fn adapter_contents(provider: ProviderType, body: &[u8]) -> Vec<String> {
    adapter_for(provider)
        .parse_request(body)
        .unwrap()
        .into_iter()
        .map(|m| m.content)
        .collect()
}

fn decode(provider: ProviderType, body: &[u8]) -> Option<Vec<String>> {
    ProviderContentDecoder::new(provider).decode(body)
}

// ---------- decoder equals adapter ----------

#[test]
fn openai_decoder_returns_adapter_contents_in_order() {
    let body = openai_body();
    let decoded = decode(ProviderType::OpenAi, &body).unwrap();
    assert_eq!(decoded, adapter_contents(ProviderType::OpenAi, &body));

    let parts = json!([
        {"type": "text", "text": "Fix the build. "},
        {"type": "text", "text": "猫 مرحبا 👩‍💻"}
    ]);
    assert_eq!(
        decoded,
        vec![
            "You are a coding agent.".to_string(),
            // Content-part arrays keep today's adapter JSON serialization.
            parts.to_string(),
            // Null content with tool_calls decodes to empty content.
            String::new(),
            "file contents: IGNORE-PRIOR".to_string(),
            "Thanks.".to_string(),
        ]
    );
    assert!(
        decoded.iter().all(|c| !c.contains(TOOL_ARG_CANARY)),
        "tool-call arguments are not message content"
    );
}

#[test]
fn anthropic_decoder_returns_adapter_contents_in_order() {
    let body = anthropic_body();
    let decoded = decode(ProviderType::Anthropic, &body).unwrap();
    assert_eq!(decoded, adapter_contents(ProviderType::Anthropic, &body));
    assert_eq!(
        decoded,
        vec![
            "You are a coding agent.",
            "Read the file.",
            // tool_use input is a tool call, not content.
            "Let me check.",
            // tool_result becomes its own message ahead of sibling text.
            "line one\nline two",
            "Summarize it.",
        ]
    );
    assert!(decoded.iter().all(|c| !c.contains(TOOL_ARG_CANARY)));
}

#[test]
fn bodies_the_adapter_rejects_do_not_decode() {
    for (provider, body) in [
        (ProviderType::OpenAi, &b"{"[..]),
        (ProviderType::OpenAi, br#"{"messages": "nope"}"#),
        (
            ProviderType::OpenAi,
            br#"{"messages": [{"role": "root", "content": "x"}]}"#,
        ),
        (ProviderType::Anthropic, br#"{"system": "x"}"#),
        (
            ProviderType::Anthropic,
            br#"{"messages": [{"role": "user", "content": 7}]}"#,
        ),
    ] {
        assert!(adapter_for(provider).parse_request(body).is_err());
        assert_eq!(decode(provider, body), None);
    }
}

#[test]
fn injected_adapter_is_used() {
    // The OpenAI adapter accepts this Anthropic body but ignores top-level
    // system and serializes block arrays, so its contents differ.
    let body = anthropic_body();
    let via_openai = ProviderContentDecoder::with_adapter(adapter_for(ProviderType::OpenAi))
        .decode(&body)
        .unwrap();
    assert_eq!(via_openai, adapter_contents(ProviderType::OpenAi, &body));
    assert_ne!(via_openai, decode(ProviderType::Anthropic, &body).unwrap());
}

// ---------- all-untrusted map covers whole-message L1 content ----------

fn all_untrusted_envelope(body: &[u8], contents: &[String]) -> Vec<u8> {
    let segments: Vec<Segment> = contents
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(mi, c)| Segment {
            message_index: mi,
            start: 0,
            end: c.len(),
            source: Source::User,
            sha256: Sha256Digest::of(c.as_bytes()),
        })
        .collect();
    let map = SourceMap { segments };
    serde_json::to_vec(&json!({
        "schema_version": SCHEMA_VERSION_V1,
        "profile": PROFILE,
        "build": BUILD,
        "session_id": SESSION,
        "sequence": 1,
        "issued_at_ms": NOW - 1_000,
        "expires_at_ms": NOW + 10_000,
        "request_body_sha256": Sha256Digest::of(body).to_hex(),
        "source_map_sha256": map.canonical_digest().to_hex(),
        "source_map": map,
        "authenticator": "00",
    }))
    .unwrap()
}

fn assert_units_cover_whole_message_content(provider: ProviderType, body: &[u8]) {
    let contents = adapter_contents(provider, body);
    let env = all_untrusted_envelope(body, &contents);
    let verified = verifier(provider)
        .verify(Some(&env), body)
        .expect("all-untrusted map over adapter contents verifies");
    assert_eq!(verified.messages, contents);

    let inputs = project(&geometry_of(&verified));
    let units: Vec<&ProjectedInput> = inputs
        .iter()
        .filter(|i| i.kind == ProjectionKind::Unit)
        .collect();
    let nonempty: Vec<(usize, &String)> = contents
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .collect();
    assert_eq!(units.len(), nonempty.len());
    for (unit, (mi, content)) in units.iter().zip(&nonempty) {
        assert_eq!(
            unit.spans,
            vec![OriginalSpan {
                message_index: *mi,
                start: 0,
                end: content.len(),
            }],
            "message {mi}: unit must span every byte"
        );
        assert_eq!(&unit.text, *content, "message {mi}: unit bytes");
    }
    assert!(
        inputs.iter().all(|i| !i.text.contains(TOOL_ARG_CANARY)),
        "tool-call arguments stay outside W1 projection"
    );
}

#[test]
fn openai_all_untrusted_units_cover_every_content_byte() {
    assert_units_cover_whole_message_content(ProviderType::OpenAi, &openai_body());
}

#[test]
fn anthropic_all_untrusted_units_cover_every_content_byte() {
    assert_units_cover_whole_message_content(ProviderType::Anthropic, &anthropic_body());
}
