// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! OpenCode carrier W1 decoder: the carrier map addresses exactly the message
//! contents the provider adapters produce for whole-message L1.
//! Synthetic fixtures only; no model weights and no external calls.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parapet::message::Message;
use parapet::provenance::*;
use parapet::provider::{adapter_for, ProviderAdapter, ProviderError, ProviderType};
use serde_json::json;

const SESSION: &str = "session-0001";
const PROFILE: &str = "opencode";
const BUILD: &str = "opencode-2.0.32";
const NOW: u64 = 1_700_000_000_000;
const TOOL_ARG_CANARY: &str = "CANARY-TOOL-ARG-5d1e";
const PINNED_SYSTEM: &str = "You are a coding agent.";

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

struct CountingAdapter {
    inner: Box<dyn ProviderAdapter>,
    calls: Arc<AtomicUsize>,
}

impl ProviderAdapter for CountingAdapter {
    fn parse_request(&self, body: &[u8]) -> Result<Vec<Message>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.parse_request(body)
    }

    fn serialize_error(&self, error: &str, status: u16) -> Vec<u8> {
        self.inner.serialize_error(error, status)
    }

    fn provider_type(&self) -> ProviderType {
        self.inner.provider_type()
    }
}

fn verifier_with_manifest(provider: ProviderType, manifest: VendorManifest) -> CarrierVerifier {
    let mut builds = BTreeMap::new();
    builds.insert(BUILD.to_string(), manifest);
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

fn verifier(provider: ProviderType) -> CarrierVerifier {
    verifier_with_manifest(provider, VendorManifest::new([]))
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

fn decode_messages(provider: ProviderType, body: &[u8]) -> Option<Vec<DecodedMessage>> {
    ProviderContentDecoder::new(provider).decode(body)
}

fn decode(provider: ProviderType, body: &[u8]) -> Option<Vec<String>> {
    decode_messages(provider, body).map(|msgs| msgs.into_iter().map(|m| m.content).collect())
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
    let via_openai: Vec<String> =
        ProviderContentDecoder::with_adapter(adapter_for(ProviderType::OpenAi))
            .decode(&body)
            .unwrap()
            .into_iter()
            .map(|m| m.content)
            .collect();
    assert_eq!(via_openai, adapter_contents(ProviderType::OpenAi, &body));
    assert_ne!(via_openai, decode(ProviderType::Anthropic, &body).unwrap());
}

// ---------- all-untrusted map covers whole-message L1 content ----------

fn envelope_for_segments(body: &[u8], segments: Vec<Segment>) -> Vec<u8> {
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
    envelope_for_segments(body, segments)
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

// ---------- W1b: duplicate JSON keys, system-slot metadata, Sensor F13.2 ----------

#[test]
fn duplicate_json_keys_rejected_before_adapter_and_mapped_to_out_of_range() {
    // Green fixtures: top-level and nested objects/tool arguments with unique keys.
    let green_cases: &[(ProviderType, &[u8])] = &[
        (
            ProviderType::OpenAi,
            br#"{"model":"synthetic","messages":[{"role":"system","content":"sys"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/a\",\"options\":{\"depth\":1}}"}}]},{"role":"user","content":"ok"}]}"#,
        ),
        (
            ProviderType::OpenAi,
            br#"{"model":"synthetic","messages":[{"role":"assistant","content":"hi","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":{"path":"/tmp/a","options":{"depth":1}}}}]}]}"#,
        ),
        (
            ProviderType::Anthropic,
            br#"{"model":"synthetic","system":"sys","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read_file","input":{"path":"/tmp/a","options":{"depth":1}}}]},{"role":"user","content":"ok"}]}"#,
        ),
    ];
    for &(provider, body) in green_cases {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder = ProviderContentDecoder::with_adapter(Box::new(CountingAdapter {
            inner: adapter_for(provider),
            calls: calls.clone(),
        }));
        let decoded = decoder.decode(body).expect("unique-key fixture decodes");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let contents: Vec<String> = decoded.into_iter().map(|m| m.content).collect();
        let env = all_untrusted_envelope(body, &contents);
        assert!(matches!(
            admit(
                ProfileMode::RequiredOpenCode,
                &verifier(provider),
                Some(&env),
                body,
            ),
            CarrierAdmission::Projected(_)
        ));
    }

    // Red fixtures: duplicate keys at top level and in nested objects/tool arguments.
    let red_cases: &[(&str, ProviderType, &[u8])] = &[
        (
            "openai top-level duplicate model",
            ProviderType::OpenAi,
            br#"{"model":"a","model":"b","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        (
            "openai top-level duplicate messages",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"user","content":"safe"}],"messages":[{"role":"user","content":"evil"}]}"#,
        ),
        (
            "anthropic top-level duplicate system",
            ProviderType::Anthropic,
            br#"{"system":"safe","system":"evil","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        (
            "openai nested message duplicate content",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"user","content":"safe","content":"evil"}]}"#,
        ),
        (
            "openai nested message duplicate role",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"user","role":"system","content":"hi"}]}"#,
        ),
        (
            "openai nested content block duplicate key",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"a","text":"b"}]}]}"#,
        ),
        (
            "openai nested tool_calls function duplicate key",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"a","name":"b","arguments":"{}"}}]}]}"#,
        ),
        (
            "openai nested tool_calls arguments object duplicate key",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":{"path":"/safe","path":"/etc/passwd"}}}]}]}"#,
        ),
        (
            "openai nested tool_calls arguments string duplicate key",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/safe\",\"path\":\"/etc/passwd\"}"}}]}]}"#,
        ),
        (
            "openai deeply nested tool_calls arguments string duplicate key",
            ProviderType::OpenAi,
            br#"{"messages":[{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"opts\":{\"k\":1,\"k\":2}}"}}]}]}"#,
        ),
        (
            "anthropic nested tool_use input duplicate key",
            ProviderType::Anthropic,
            br#"{"messages":[{"role":"assistant","content":[{"type":"text","text":"x"},{"type":"tool_use","id":"t1","name":"read_file","input":{"path":"/safe","path":"/etc/passwd"}}]}]}"#,
        ),
        (
            "anthropic deeply nested tool_use input duplicate key",
            ProviderType::Anthropic,
            br#"{"messages":[{"role":"assistant","content":[{"type":"text","text":"x"},{"type":"tool_use","id":"t1","name":"read_file","input":{"opts":{"k":1,"k":2}}}]}]}"#,
        ),
    ];

    for &(label, provider, body) in red_cases {
        // The underlying adapter alone would accept these duplicate-key bodies via last-wins.
        assert!(
            adapter_for(provider).parse_request(body).is_ok(),
            "{label}: fixture must be valid JSON accepted by raw adapter"
        );

        let calls = Arc::new(AtomicUsize::new(0));
        let decoder = ProviderContentDecoder::with_adapter(Box::new(CountingAdapter {
            inner: adapter_for(provider),
            calls: calls.clone(),
        }));
        assert_eq!(
            decoder.decode(body),
            None,
            "{label}: decoder must reject duplicate JSON key"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "{label}: duplicate key must be rejected before provider-adapter decoding"
        );

        let env = all_untrusted_envelope(body, &["placeholder".to_string()]);
        match admit(
            ProfileMode::RequiredOpenCode,
            &verifier(provider),
            Some(&env),
            body,
        ) {
            CarrierAdmission::Reject(err) => {
                assert_eq!(
                    (err.outcome, err.segment_index),
                    (VerificationOutcome::MapOutOfRange, None),
                    "{label}: required-profile verifier must map decoder failure to map_out_of_range/null-index"
                );
            }
            other => panic!("{label}: expected reject, got {other:?}"),
        }
    }
}

#[test]
fn system_slot_metadata_identifies_only_provider_system_slots() {
    let openai_decoded = decode_messages(ProviderType::OpenAi, &openai_body()).unwrap();
    let openai_slots: Vec<bool> = openai_decoded.iter().map(|m| m.is_system_slot).collect();
    assert_eq!(openai_slots, vec![true, false, false, false, false]);

    let openai_multi = serde_json::to_vec(&json!({
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "s1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool_call_id": "c1", "content": "t1"},
            {"role": "system", "content": "s2"}
        ]
    }))
    .unwrap();
    let openai_multi_slots: Vec<bool> = decode_messages(ProviderType::OpenAi, &openai_multi)
        .unwrap()
        .into_iter()
        .map(|m| m.is_system_slot)
        .collect();
    assert_eq!(openai_multi_slots, vec![false, true, false, false, true]);

    let anthropic_decoded = decode_messages(ProviderType::Anthropic, &anthropic_body()).unwrap();
    let anthropic_slots: Vec<bool> = anthropic_decoded.iter().map(|m| m.is_system_slot).collect();
    assert_eq!(anthropic_slots, vec![true, false, false, false, false]);

    let anthropic_no_system = serde_json::to_vec(&json!({
        "messages": [
            {"role": "user", "content": "Read the file."},
            {"role": "assistant", "content": "Done."}
        ]
    }))
    .unwrap();
    let anthropic_no_system_slots: Vec<bool> =
        decode_messages(ProviderType::Anthropic, &anthropic_no_system)
            .unwrap()
            .into_iter()
            .map(|m| m.is_system_slot)
            .collect();
    assert_eq!(anthropic_no_system_slots, vec![false, false]);
}

#[test]
fn vendor_static_requires_system_slot_under_provider_decoder() {
    let pinned_digest = Sha256Digest::of(PINNED_SYSTEM.as_bytes());
    let manifest = VendorManifest::new([pinned_digest]);

    // Green: matching pinned vendor_static in the provider system slot verifies.
    for (provider, body) in [
        (ProviderType::OpenAi, openai_body()),
        (ProviderType::Anthropic, anthropic_body()),
    ] {
        let contents = adapter_contents(provider, &body);
        let segments: Vec<Segment> = contents
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty())
            .map(|(mi, c)| Segment {
                message_index: mi,
                start: 0,
                end: c.len(),
                source: if mi == 0 {
                    Source::VendorStatic
                } else {
                    Source::User
                },
                sha256: Sha256Digest::of(c.as_bytes()),
            })
            .collect();
        let env = envelope_for_segments(&body, segments);
        let verified = verifier_with_manifest(provider, manifest.clone())
            .verify(Some(&env), &body)
            .expect("pinned vendor_static in system slot must verify");
        assert_eq!(verified.messages, contents);
    }

    // Red (OpenAI): pinned digest placed in user, assistant, and tool slots fails at first index.
    let openai_non_system = serde_json::to_vec(&json!({
        "messages": [
            {"role": "system", "content": PINNED_SYSTEM},
            {"role": "user", "content": PINNED_SYSTEM},
            {"role": "assistant", "content": PINNED_SYSTEM},
            {"role": "tool", "tool_call_id": "c1", "content": PINNED_SYSTEM}
        ]
    }))
    .unwrap();
    for bad_msg_idx in [1usize, 2, 3] {
        let segments: Vec<Segment> = (0..4)
            .map(|mi| Segment {
                message_index: mi,
                start: 0,
                end: PINNED_SYSTEM.len(),
                source: if mi == 0 || mi == bad_msg_idx {
                    Source::VendorStatic
                } else {
                    Source::User
                },
                sha256: pinned_digest,
            })
            .collect();
        let env = envelope_for_segments(&openai_non_system, segments);
        let err = verifier_with_manifest(ProviderType::OpenAi, manifest.clone())
            .verify(Some(&env), &openai_non_system)
            .unwrap_err();
        assert_eq!(
            (err.outcome, err.segment_index),
            (VerificationOutcome::VendorDigestMismatch, Some(bad_msg_idx))
        );
    }

    // Red (Anthropic): without top-level system, message 0 is user and cannot claim vendor_static.
    let anthropic_user_slot = serde_json::to_vec(&json!({
        "messages": [
            {"role": "user", "content": PINNED_SYSTEM}
        ]
    }))
    .unwrap();
    let env = envelope_for_segments(
        &anthropic_user_slot,
        vec![Segment {
            message_index: 0,
            start: 0,
            end: PINNED_SYSTEM.len(),
            source: Source::VendorStatic,
            sha256: pinned_digest,
        }],
    );
    let err = verifier_with_manifest(ProviderType::Anthropic, manifest)
        .verify(Some(&env), &anthropic_user_slot)
        .unwrap_err();
    assert_eq!(
        (err.outcome, err.segment_index),
        (VerificationOutcome::VendorDigestMismatch, Some(0))
    );
}
