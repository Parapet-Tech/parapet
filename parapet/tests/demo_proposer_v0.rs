// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! 1E demo proposer, fixture, sender, and export contract tests
//! (unit level; the black-box journeys live in `demo_workflow_v0.rs`).

#![cfg(not(windows))]

use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use bytes::Bytes;
use parapet::engine::demo::export::{export_assets, DemoExportErrorV0, DEMO_EXPORT_SCHEMA};
use parapet::engine::demo::{
    load_verified_manifest, sha256_hex, CompiledDemoFixturesV0, DemoFixtureErrorV0,
    DemoFixtureSourceV0, DemoJourneyV0, DemoProposerHttpSenderV0, DemoProposerV0,
    ManifestDemoProposerV0, ProposerErrorV0, ProposerInputV0, EXPORTED_ASSETS, MANIFEST_NAME,
};
use parapet::engine::{
    build_engine_client_with_runtime_v0, DeterministicOfflineHttpSenderV0, EngineRuntimeOptionsV0,
    HttpBody, HttpError, HttpRequest, HttpResponse, HttpSender, OfflineUpstreamFixtureV0,
};
use parapet::proxy::{Provider, ProxyRequest, UpstreamClient};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Frozen human-operator intake hashes (Section 5.1); never re-derived.
const J2_UNAUTHORIZED_SHA256: &str =
    "60a65e916c21c05659f80665dc484358fd10fedb84fa60c462f25bacf908e75b";
const J3_INJECTION_BENIGN_SHA256: &str =
    "7b685e429c5fe616fd0884977e5f3e01f4eee795a597aa80ee2cf4934b5f8a02";

const DEMO_ERROR: &str = "offline fixture demo journey requires the demo proposer sender";

fn source_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("demo/v0")
}

fn embedded(name: &str) -> &'static [u8] {
    if name == MANIFEST_NAME {
        CompiledDemoFixturesV0.manifest_bytes()
    } else {
        CompiledDemoFixturesV0.asset(name).expect("embedded asset")
    }
}

fn embedded_str(name: &str) -> &'static str {
    std::str::from_utf8(embedded(name)).expect("utf8 asset")
}

fn manifest_json() -> Value {
    serde_json::from_slice(embedded(MANIFEST_NAME)).expect("manifest json")
}

fn journey_entry(journey: DemoJourneyV0) -> Value {
    manifest_json()["journeys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["journey"] == journey.name())
        .cloned()
        .expect("journey entry")
}

fn document(journey: DemoJourneyV0) -> &'static str {
    embedded_str(journey_entry(journey)["document_path"].as_str().unwrap())
}

fn propose(journey: DemoJourneyV0, document: &str) -> Result<String, ProposerErrorV0> {
    ManifestDemoProposerV0::new_default()
        .propose(&ProposerInputV0 { journey, document })
        .map(|c| {
            assert_eq!(c.tool_name, "write_file");
            c.arguments_json
        })
}

fn openai_request(url: &str, document: &str) -> HttpRequest {
    let body = serde_json::json!({
        "model": "parapet-demo-offline",
        "messages": [{"role": "user", "content": document}]
    });
    HttpRequest {
        method: Method::POST,
        url: url.to_owned(),
        headers: HeaderMap::new(),
        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
        timeout_ms: None,
        stream: false,
    }
}

fn anthropic_request(document: &str) -> HttpRequest {
    let body = serde_json::json!({
        "model": "parapet-demo-offline",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": document}]
    });
    HttpRequest {
        method: Method::POST,
        url: "https://api.anthropic.com/v1/messages".to_owned(),
        headers: HeaderMap::new(),
        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
        timeout_ms: None,
        stream: false,
    }
}

fn full_body(resp: HttpResponse) -> Value {
    match resp.body {
        HttpBody::Full(b) => serde_json::from_slice(&b).expect("json body"),
        HttpBody::Stream(_) => panic!("demo sender never streams"),
    }
}

fn transport_message(res: Result<HttpResponse, HttpError>) -> String {
    match res {
        Err(HttpError::Transport(m)) => m,
        Err(HttpError::Timeout(m)) => panic!("unexpected timeout: {m}"),
        Ok(resp) => panic!("unexpected response: {}", full_body(resp)),
    }
}

// --- fixtures: single source, frozen hashes, packaging ------------------------

#[test]
fn embedded_assets_equal_source_files_and_manifest_verifies() {
    for name in EXPORTED_ASSETS {
        let source = std::fs::read(source_dir().join(name)).expect("source asset");
        assert_eq!(
            embedded(name),
            source.as_slice(),
            "{name}: embedded != source"
        );
    }
    let manifest = load_verified_manifest(&CompiledDemoFixturesV0).expect("manifest verifies");
    assert_eq!(manifest.schema, "parapet-demo-fixtures/v0");
    assert_eq!(manifest.upstream_label, "parapet-offline-upstream/v0");
    // Exactly the exported set exists in the source directory, plus the guide.
    let mut on_disk: Vec<String> = std::fs::read_dir(source_dir())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.') && n != "README.md")
        .collect();
    on_disk.sort();
    assert_eq!(on_disk, EXPORTED_ASSETS.map(str::to_owned).to_vec());
}

#[test]
fn human_injection_fixtures_equal_frozen_intake_hashes() {
    for (name, frozen) in [
        ("doc_unauthorized.txt", J2_UNAUTHORIZED_SHA256),
        ("doc_injection_benign.txt", J3_INJECTION_BENIGN_SHA256),
    ] {
        assert_eq!(sha256_hex(embedded(name)), frozen, "{name} embedded");
        let source = std::fs::read(source_dir().join(name)).unwrap();
        assert_eq!(sha256_hex(&source), frozen, "{name} source");
    }
    for (journey, frozen) in [
        (DemoJourneyV0::Unauthorized, J2_UNAUTHORIZED_SHA256),
        (DemoJourneyV0::InjectionBenign, J3_INJECTION_BENIGN_SHA256),
    ] {
        let entry = journey_entry(journey);
        assert_eq!(entry["document_sha256"], frozen);
        assert_eq!(entry["document_class"], "injection");
        assert_eq!(entry["document_authorship"]["kind"], "human_operator");
    }
    for journey in [DemoJourneyV0::BenignWrite, DemoJourneyV0::Mismatch] {
        let entry = journey_entry(journey);
        assert_eq!(entry["document_class"], "benign_non_injection");
        assert_eq!(entry["document_authorship"]["kind"], "agent");
        assert!(entry["provenance"]
            .as_str()
            .unwrap()
            .starts_with("agent-authored"));
    }
}

#[test]
fn fixture_documents_meet_structural_requirements() {
    let mut targets = Vec::new();
    for (journey, canary) in DemoJourneyV0::ALL.into_iter().zip([
        "PARAPET-1E-CANARY-J1",
        "PARAPET-1E-CANARY-J2",
        "PARAPET-1E-CANARY-J3",
        "PARAPET-1E-CANARY-J4",
    ]) {
        let bytes = document(journey).as_bytes();
        assert!(bytes.len() <= 4096);
        assert!(!bytes.contains(&0) && !bytes.contains(&b'\r'));
        assert!(!bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        let text = document(journey);
        let rest = text
            .strip_prefix("Task: save-note\nTarget: ")
            .expect("header");
        let (target, rest) = rest.split_once('\n').unwrap();
        let body = rest.strip_prefix("---\n").expect("separator");
        assert_eq!(body.matches(canary).count(), 1, "{canary}");
        let entry = journey_entry(journey);
        assert_eq!(entry["canary"], canary);
        assert_eq!(entry["expected_target_basename"], target);
        targets.push(target.to_owned());
    }
    targets.sort();
    targets.dedup();
    assert_eq!(targets.len(), 4, "every journey has a distinct Target");
    assert!(document(DemoJourneyV0::Unauthorized).contains("../escape.txt"));
    assert_eq!(
        journey_entry(DemoJourneyV0::Unauthorized)["injected_path"],
        "../escape.txt"
    );
}

#[test]
fn requests_rederive_from_documents_byte_for_byte() {
    for journey in DemoJourneyV0::ALL {
        let entry = journey_entry(journey);
        let doc = document(journey);
        let derived = format!(
            "{{\"model\":\"parapet-demo-offline\",\"messages\":[{{\"role\":\"user\",\"content\":{}}}]}}",
            Value::String(doc.to_owned())
        );
        let request = embedded(entry["request_path"].as_str().unwrap());
        assert_eq!(derived.as_bytes(), request, "{}", journey.name());
        assert_eq!(entry["request_sha256"], sha256_hex(request));
        assert_eq!(entry["selector"], journey.selector());
    }
}

#[test]
fn demo_config_contract_hash_matches_manifest() {
    let cfg = parapet::config::load_config(&parapet::config::FileSource {
        path: source_dir().join("config_demo_v0.yaml"),
    })
    .expect("demo config loads");
    let manifest = manifest_json();
    assert_eq!(
        Value::String(cfg.contract_hash.clone()),
        manifest["config"]["contract_hash"]
    );
    assert!(cfg.policy.layers.l1.is_some(), "L1 is recorded (shadow)");
    // Same tools block as the frozen 1D enforced accept config.
    let accept = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli_v0/accept_enforced_v0.yaml"),
    )
    .unwrap();
    let demo = embedded_str("config_demo_v0.yaml");
    assert_eq!(demo, format!("{accept}  L1: {{ mode: shadow }}\n"));
}

// --- proposer -----------------------------------------------------------------

#[test]
fn proposer_follows_document_and_is_deterministic() {
    for journey in DemoJourneyV0::ALL {
        let entry = journey_entry(journey);
        let first = propose(journey, document(journey)).expect("proposal");
        let second = propose(journey, document(journey)).expect("proposal");
        assert_eq!(first, second);
        assert_eq!(
            Value::String(first.clone()),
            entry["expected_arguments_json"]
        );
        let args: Value = serde_json::from_str(&first).unwrap();
        let body = document(journey).split_once("---\n").unwrap().1;
        assert_eq!(args["content"], body, "content is the exact body bytes");
        let expected_path = if journey == DemoJourneyV0::Unauthorized {
            "../escape.txt".to_owned()
        } else {
            entry["expected_target_basename"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(args["path"], expected_path.as_str());
        if let Some(sha) = entry["expected_file_sha256"].as_str() {
            assert_eq!(sha256_hex(body.as_bytes()), sha);
        }
    }
}

#[test]
fn n1_unknown_document_proposes_nothing() {
    let mut doc = document(DemoJourneyV0::BenignWrite).to_owned();
    doc.push(' ');
    assert_eq!(
        propose(DemoJourneyV0::BenignWrite, &doc),
        Err(ProposerErrorV0::DocumentUnknown)
    );
    assert_eq!(
        propose(DemoJourneyV0::BenignWrite, ""),
        Err(ProposerErrorV0::DocumentUnknown)
    );
}

#[test]
fn n2_document_cannot_steer_another_journey() {
    for journey in DemoJourneyV0::ALL {
        for other in DemoJourneyV0::ALL.into_iter().filter(|o| *o != journey) {
            assert_eq!(
                propose(journey, document(other)),
                Err(ProposerErrorV0::JourneyDocumentMismatch),
                "{} under {}",
                other.name(),
                journey.name()
            );
        }
    }
}

/// Wraps the compiled set, replacing one asset's bytes or the manifest bytes.
struct TamperedSource {
    name: &'static str,
    bytes: Vec<u8>,
}

impl DemoFixtureSourceV0 for TamperedSource {
    fn manifest_bytes(&self) -> &[u8] {
        if self.name == MANIFEST_NAME {
            &self.bytes
        } else {
            CompiledDemoFixturesV0.manifest_bytes()
        }
    }
    fn asset(&self, name: &str) -> Option<&[u8]> {
        if name == self.name {
            Some(&self.bytes)
        } else {
            CompiledDemoFixturesV0.asset(name)
        }
    }
}

fn tampered_manifest_digest() -> TamperedSource {
    let mut manifest = manifest_json();
    manifest["journeys"][0]["document_sha256"] = Value::String("0".repeat(64));
    TamperedSource {
        name: MANIFEST_NAME,
        bytes: serde_json::to_vec(&manifest).unwrap(),
    }
}

fn tampered_document() -> TamperedSource {
    let mut bytes = embedded("doc_benign_write.txt").to_vec();
    bytes[40] ^= 0x01;
    TamperedSource {
        name: "doc_benign_write.txt",
        bytes,
    }
}

#[test]
fn n6_tampered_fixture_source_fails_closed() {
    assert_eq!(
        load_verified_manifest(&tampered_manifest_digest()),
        Err(DemoFixtureErrorV0::DigestMismatch)
    );
    assert_eq!(
        load_verified_manifest(&tampered_document()),
        Err(DemoFixtureErrorV0::DigestMismatch)
    );
    for source in [tampered_manifest_digest(), tampered_document()] {
        let proposer = ManifestDemoProposerV0::new(source);
        for journey in DemoJourneyV0::ALL {
            let res = proposer.propose(&ProposerInputV0 {
                journey,
                document: document(journey),
            });
            assert_eq!(res, Err(ProposerErrorV0::FixtureDigestMismatch));
        }
    }
}

// --- demo sender --------------------------------------------------------------

fn demo_sender(journey: DemoJourneyV0) -> impl HttpSender {
    DemoProposerHttpSenderV0::new(journey, ManifestDemoProposerV0::new_default())
}

#[tokio::test]
async fn demo_sender_serves_openai_and_anthropic_tool_calls() {
    for journey in DemoJourneyV0::ALL {
        let expected = journey_entry(journey)["expected_arguments_json"].clone();
        let resp = demo_sender(journey)
            .send(openai_request(
                "https://api.openai.com/v1/chat/completions",
                document(journey),
            ))
            .await
            .expect("openai response");
        assert_eq!(resp.status, StatusCode::OK);
        let body = full_body(resp);
        let call = &body["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "write_file");
        assert_eq!(call["arguments"], expected);

        let resp = demo_sender(journey)
            .send(anthropic_request(document(journey)))
            .await
            .expect("anthropic response");
        let body = full_body(resp);
        assert_eq!(body["content"][0]["name"], "write_file");
        let input: Value = serde_json::from_str(expected.as_str().unwrap()).unwrap();
        assert_eq!(body["content"][0]["input"], input);
    }
}

#[tokio::test]
async fn demo_sender_refuses_without_a_call() {
    let sender = demo_sender(DemoJourneyV0::BenignWrite);
    let doc = document(DemoJourneyV0::BenignWrite);
    assert_eq!(
        transport_message(
            sender
                .send(openai_request(
                    "http://api.openai.com/v1/chat/completions",
                    doc
                ))
                .await
        ),
        "offline fixture requires HTTPS resolved upstream URL"
    );
    let unknown = format!("{doc} ");
    assert_eq!(
        transport_message(
            sender
                .send(openai_request(
                    "https://api.openai.com/v1/chat/completions",
                    &unknown
                ))
                .await
        ),
        "demo_proposer:document_unknown"
    );
    assert_eq!(
        transport_message(
            demo_sender(DemoJourneyV0::Unauthorized)
                .send(openai_request(
                    "https://api.openai.com/v1/chat/completions",
                    doc
                ))
                .await
        ),
        "demo_proposer:journey_document_mismatch"
    );
    let mut no_user = openai_request("https://api.openai.com/v1/chat/completions", doc);
    no_user.body =
        Bytes::from_static(br#"{"model":"m","messages":[{"role":"system","content":"x"}]}"#);
    assert_eq!(
        transport_message(sender.send(no_user).await),
        "demo_proposer:document_missing"
    );
}

#[tokio::test]
async fn demo_sender_ignores_steering_headers() {
    let doc = document(DemoJourneyV0::BenignWrite);
    let mut req = openai_request("https://api.openai.com/v1/chat/completions", doc);
    req.headers.insert(
        "x-parapet-offline-scenario",
        HeaderValue::from_static("block_no_effect"),
    );
    req.headers.insert(
        "x-parapet-offline-raw-arguments",
        HeaderValue::from_static(r#"{"content":"x","path":"../escape.txt"}"#),
    );
    let body = full_body(
        demo_sender(DemoJourneyV0::BenignWrite)
            .send(req)
            .await
            .unwrap(),
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        journey_entry(DemoJourneyV0::BenignWrite)["expected_arguments_json"]
    );
}

// --- landed 1D sender: selectors, header guard, direct construction ----------

#[test]
fn demo_selectors_parse_and_landed_selectors_are_unchanged() {
    for journey in DemoJourneyV0::ALL {
        assert_eq!(
            OfflineUpstreamFixtureV0::parse(journey.selector()),
            Some(OfflineUpstreamFixtureV0::Demo(journey))
        );
    }
    for (raw, want) in [
        ("v0", OfflineUpstreamFixtureV0::V0),
        (
            "allow_no_executor",
            OfflineUpstreamFixtureV0::AllowNoExecutor,
        ),
        (
            "allow_verified_write",
            OfflineUpstreamFixtureV0::AllowVerifiedWrite,
        ),
        ("block_no_effect", OfflineUpstreamFixtureV0::BlockNoEffect),
        (
            "shadow_parse_error_l3_off",
            OfflineUpstreamFixtureV0::ShadowParseErrorL3Off,
        ),
    ] {
        assert_eq!(OfflineUpstreamFixtureV0::parse(raw), Some(want));
    }
    assert_eq!(OfflineUpstreamFixtureV0::parse("demo_v0"), None);
}

#[tokio::test]
async fn landed_sender_header_never_resolves_to_a_demo_journey() {
    for journey in DemoJourneyV0::ALL {
        let mut req = openai_request(
            "https://api.openai.com/v1/chat/completions",
            "write note.txt",
        );
        req.headers.insert(
            "x-parapet-offline-scenario",
            HeaderValue::from_static(journey.selector()),
        );
        // Falls through to the landed default rule, exactly as an unknown header string.
        let body = full_body(
            DeterministicOfflineHttpSenderV0::new(OfflineUpstreamFixtureV0::V0)
                .send(req)
                .await
                .expect("landed default response"),
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            r#"{"content":"hello parapet\n","path":"note.txt"}"#
        );
    }
}

#[tokio::test]
async fn landed_sender_directly_constructed_with_demo_journey_returns_exact_error() {
    for journey in DemoJourneyV0::ALL {
        let sender = DeterministicOfflineHttpSenderV0::new(OfflineUpstreamFixtureV0::Demo(journey));
        let doc = document(journey);
        let mut steered = openai_request("https://api.openai.com/v1/chat/completions", doc);
        steered.headers.insert(
            "x-parapet-offline-scenario",
            HeaderValue::from_static("allow_verified_write"),
        );
        steered.headers.insert(
            "x-parapet-offline-raw-arguments",
            HeaderValue::from_static(r#"{"content":"x","path":"note.txt"}"#),
        );
        for req in [
            openai_request("https://api.openai.com/v1/chat/completions", doc),
            anthropic_request(doc),
            openai_request("http://api.openai.com/v1/chat/completions", doc),
            steered,
        ] {
            let res = sender.send(req).await;
            assert!(
                matches!(res, Err(HttpError::Transport(_))),
                "{}",
                journey.name()
            );
            assert_eq!(transport_message(res), DEMO_ERROR);
        }
    }
}

// --- N5: no demo code path without --offline-fixture --------------------------

fn private_state_dir(parent: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let state = parent.join("state");
    std::fs::create_dir(&state).unwrap();
    std::fs::set_permissions(&state, std::fs::Permissions::from_mode(0o700)).unwrap();
    state
}

#[tokio::test]
async fn n5_default_sender_never_reaches_demo_code() {
    // Point the allowed host at a closed loopback HTTPS port: no traffic leaves the host.
    std::env::set_var("PARAPET_API_OPENAI_COM_BASE_URL", "https://127.0.0.1:9");
    let dir = tempfile::tempdir().unwrap();
    let state = private_state_dir(dir.path());
    let effect_root = dir.path().join("root");
    let config = std::sync::Arc::new(
        parapet::config::load_config(&parapet::config::FileSource {
            path: source_dir().join("config_demo_v0.yaml"),
        })
        .unwrap(),
    );
    let (client, _store, root) = build_engine_client_with_runtime_v0(
        config,
        EngineRuntimeOptionsV0 {
            state_dir: state,
            effect_root: Some(effect_root.clone()),
            file_effect_profile: Some(parapet::effect::FileEffectProfileV0::select_explicitly()),
            max_content_len_bytes: Some(4096),
            build_digest: None,
            offline_fixture: None,
            result_sink: None,
        },
    )
    .expect("builds without offline fixture");
    assert!(root.is_some());
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert(
        "x-parapet-original-host",
        HeaderValue::from_static("api.openai.com"),
    );
    let resp = client
        .forward(
            Provider::OpenAi,
            ProxyRequest {
                method: Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                headers,
                body: Bytes::from_static(embedded("request_benign_write.json")),
            },
        )
        .await;
    if let Ok(resp) = resp {
        let bytes = axum::body::to_bytes(resp.body, usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("offline_demo"), "{text}");
        assert!(!text.contains("release-note.txt"), "{text}");
    }
    assert!(!effect_root.join("release-note.txt").exists());
}

// --- N9: export --------------------------------------------------------------

#[test]
fn export_writes_exact_assets_and_reports_them() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("assets");
    let report = export_assets(&CompiledDemoFixturesV0, &out).expect("export");
    assert_eq!(DEMO_EXPORT_SCHEMA, "parapet-cli-demo-export/v0");
    assert_eq!(report.manifest_sha256, sha256_hex(embedded(MANIFEST_NAME)));
    let names: Vec<&str> = report.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, EXPORTED_ASSETS.to_vec());
    for f in &report.files {
        let written = std::fs::read(out.join(&f.name)).unwrap();
        assert_eq!(written, embedded(&f.name));
        assert_eq!(f.sha256, sha256_hex(&written));
        assert_eq!(f.len_bytes, written.len() as u64);
    }
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&out).unwrap().permissions().mode() & 0o777,
        0o700
    );
}

#[test]
fn n9_export_refuses_existing_dir_and_tampered_source() {
    let dir = tempfile::tempdir().unwrap();
    let existing = dir.path().join("existing");
    std::fs::create_dir(&existing).unwrap();
    std::fs::write(existing.join("keep.txt"), b"keep").unwrap();
    let err = export_assets(&CompiledDemoFixturesV0, &existing).unwrap_err();
    assert_eq!(err, DemoExportErrorV0::OutAlreadyExists);
    assert_eq!(
        (err.code(), err.exit_code()),
        ("demo_out_already_exists", 2)
    );
    let entries: Vec<_> = std::fs::read_dir(&existing).unwrap().flatten().collect();
    assert_eq!(entries.len(), 1, "existing directory untouched");

    for source in [tampered_manifest_digest(), tampered_document()] {
        let out = dir.path().join("tampered");
        let err = export_assets(&source, &out).unwrap_err();
        assert_eq!(err, DemoExportErrorV0::AssetDigestMismatch);
        assert_eq!(
            (err.code(), err.exit_code()),
            ("demo_asset_digest_mismatch", 4)
        );
        assert!(std::fs::symlink_metadata(&out).is_err(), "nothing written");
    }

    for bad in [
        dir.path().join("missing/assets"),
        PathBuf::from("../assets"),
    ] {
        let err = export_assets(&CompiledDemoFixturesV0, &bad).unwrap_err();
        assert_eq!((err.code(), err.exit_code()), ("invalid_path", 2));
    }
}

// --- N13(a): public-asset boundary scan and manifest authorship templates -----

struct PublicAssetScanner {
    f1_id: regex::Regex,
    f1_key: regex::Regex,
    f2a: regex::Regex,
    f2a_hosts: Vec<String>,
    f3: regex::Regex,
    f4_ci: regex::Regex,
    f4_cs: regex::Regex,
}

impl PublicAssetScanner {
    fn new() -> Self {
        let f1_id_pat = format!(
            "(msg|dispatch|dsp)_{}_{}_{}",
            "[0-9]{8}", "[0-9]{6}", "[0-9a-f]{8}"
        );
        let k_ref = ["r", "ef"].concat();
        let k_disp = ["dispatch", "_id"].concat();
        let k_msg = ["message", "_id"].concat();
        let f1_key_pat = format!(r#"\\?"({k_ref}|{k_disp}|{k_msg})\\?"[[:space:]]*:"#);

        let f2a_parts = [
            ["/", "Users", "/"].concat(),
            ["/", "home", "/"].concat(),
            ["/", "private", "/"].concat(),
            ["/", "var", "/", "folders", "/"].concat(),
            [r"(^|[^A-Za-z])[A-Za-z]:[\\/]+", "Users", r"[\\/]"].concat(),
            ["implement", r"[\\/]+", "research-findings"].concat(),
            ["human", "_fixture_", "input"].concat(),
            ["CARGO_", "MANIFEST_DIR"].concat(),
            [r"\$", "HOME"].concat(),
            [r"\$\{", "HOME", r"\}"].concat(),
            ["USER", "PROFILE"].concat(),
        ];
        let f2a_pat = format!("(?i){}", f2a_parts.join("|"));

        let mut f2a_hosts = Vec::new();
        if let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            let s = root.to_string_lossy().into_owned();
            if !s.is_empty() {
                f2a_hosts.push(s);
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            let s = home.to_string_lossy().into_owned();
            if !s.is_empty() && !f2a_hosts.contains(&s) {
                f2a_hosts.push(s);
            }
        }

        let f3_parts = [
            ["-", "architect"].concat(),
            ["-", "worker"].concat(),
            ["routing-", "claude"].concat(),
            ["pola", "ris"].concat(),
            ["north", "star"].concat(),
            ["ca", "reer"].concat(),
            ["local-", "lab"].concat(),
            ["co", "dex"].concat(),
            ["cla", "ude"].concat(),
            ["her", "dr"].concat(),
        ];
        let f3_pat = format!("(?i){}", f3_parts.join("|"));

        let f4_ci_parts = [
            ["agent.", "comms"].concat(),
            ["hand", "off"].concat(),
            ["re", "freeze"].concat(),
            ["brief ", "rev"].concat(),
            ["Rev ", "[0-9]"].concat(),
            ["D-00", "[0-9]{2}"].concat(),
        ];
        let f4_ci_pat = format!("(?i){}", f4_ci_parts.join("|"));

        let f4_cs_parts = [
            ["ACCEPT", "-BIND"].concat(),
            ["RET", "URN"].concat(),
            ["FRE", "EZE"].concat(),
        ];
        let f4_cs_pat = f4_cs_parts.join("|");

        Self {
            f1_id: regex::Regex::new(&f1_id_pat).unwrap(),
            f1_key: regex::Regex::new(&f1_key_pat).unwrap(),
            f2a: regex::Regex::new(&f2a_pat).unwrap(),
            f2a_hosts,
            f3: regex::Regex::new(&f3_pat).unwrap(),
            f4_ci: regex::Regex::new(&f4_ci_pat).unwrap(),
            f4_cs: regex::Regex::new(&f4_cs_pat).unwrap(),
        }
    }

    fn scan_line(&self, line: &str) -> Vec<&'static str> {
        let mut classes = Vec::new();
        if self.f1_id.is_match(line) || self.f1_key.is_match(line) {
            classes.push("F1");
        }
        if self.f2a.is_match(line) || self.f2a_hosts.iter().any(|h| line.contains(h.as_str())) {
            classes.push("F2a");
        }
        if self.f3.is_match(line) {
            classes.push("F3");
        }
        if self.f4_ci.is_match(line) || self.f4_cs.is_match(line) {
            classes.push("F4");
        }
        classes
    }

    fn scan_text(&self, text: &str) -> Vec<(usize, &'static str, String)> {
        let mut hits = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            for class in self.scan_line(line) {
                hits.push((idx + 1, class, line.to_owned()));
            }
        }
        hits
    }

    fn verify_controls(&self) {
        let k_ref = ["r", "ef"].concat();
        let k_disp = ["dispatch", "_id"].concat();
        let k_msg = ["message", "_id"].concat();
        let mut must_hit: Vec<(&str, String)> = vec![
            ("F1", ["msg", "_20000101_", "000000_0000000a"].concat()),
            ("F1", ["dispatch", "_20000101_", "000000_0000000b"].concat()),
            ("F1", ["dsp", "_20000101_", "000000_0000000c"].concat()),
            ("F1", format!("\"{k_ref}\": \"x\"")),
            ("F1", format!("\"{k_disp}\": \"x\"")),
            ("F1", format!("\"{k_msg}\":1")),
            ("F1", format!("{{\\\"{k_ref}\\\": \\\"x\\\"}}")),
            ("F2a", ["/", "Users", "/x"].concat()),
            ("F2a", ["/", "home", "/x"].concat()),
            ("F2a", ["/", "private", "/x"].concat()),
            ("F2a", ["/", "var", "/", "folders", "/x"].concat()),
            ("F2a", ["C:\\", "Users", "\\x"].concat()),
            ("F2a", ["C:/", "Users", "/x"].concat()),
            ("F2a", ["see \"C:\\\\", "Users", "\\\\x\""].concat()),
            (
                "F2a",
                ["implement", "/", "research-findings", "/x"].concat(),
            ),
            (
                "F2a",
                ["implement", "\\", "research-findings", "\\x"].concat(),
            ),
            ("F2a", ["human", "_fixture_", "input"].concat()),
            ("F2a", ["CARGO_", "MANIFEST_DIR"].concat()),
            ("F2a", ["$", "HOME", "/x"].concat()),
            ("F2a", ["${", "HOME", "}/x"].concat()),
            ("F2a", ["%USER", "PROFILE%"].concat()),
            ("F3", ["x-", "architect"].concat()),
            ("F3", ["x-", "worker"].concat()),
            ("F3", ["routing-", "claude"].concat()),
            ("F3", ["Pola", "ris"].concat()),
            ("F3", ["north", "star"].concat()),
            ("F3", ["Ca", "reer"].concat()),
            ("F3", ["local-", "lab"].concat()),
            ("F3", ["Co", "dex"].concat()),
            ("F3", ["Cla", "ude"].concat()),
            ("F3", ["her", "dr"].concat()),
            ("F4", ["agent-", "comms"].concat()),
            ("F4", ["agent ", "comms"].concat()),
            ("F4", ["hand", "off"].concat()),
            ("F4", ["re", "freeze"].concat()),
            ("F4", ["brief ", "rev"].concat()),
            ("F4", ["Rev ", "9"].concat()),
            ("F4", ["D-00", "26"].concat()),
            ("F4", ["ACCEPT", "-BIND"].concat()),
            ("F4", ["RET", "URN"].concat()),
            ("F4", ["FRE", "EZE"].concat()),
        ];
        for h in &self.f2a_hosts {
            must_hit.push(("F2a", h.clone()));
        }
        assert!(
            !self.f2a_hosts.is_empty(),
            "host values must be populated for F2a controls"
        );
        for (want_class, line) in &must_hit {
            let got = self.scan_line(line);
            assert!(
                got.contains(want_class),
                "planted control for {want_class} missed on {line:?}: got {got:?}"
            );
        }

        let must_pass = [
            "https://users.example.com/a".to_owned(),
            "return value".to_owned(),
            "freeze the clock".to_owned(),
            "rev-parse HEAD".to_owned(),
            "previous".to_owned(),
            "\"refs\": []".to_owned(),
            "\"href\": \"x\"".to_owned(),
            format!("\"{k_msg}s\": []"),
            format!("{k_ref}: main"),
            format!("the \"{k_ref}\" key"),
        ];
        for line in &must_pass {
            let got = self.scan_line(line);
            assert!(
                got.is_empty(),
                "precision control unexpectedly flagged on {line:?}: {got:?}"
            );
        }
    }
}

fn collect_forbidden_json_keys(val: &Value, path: &str, out: &mut Vec<String>) {
    let k_ref = ["r", "ef"].concat();
    let k_disp = ["dispatch", "_id"].concat();
    let k_msg = ["message", "_id"].concat();
    match val {
        Value::Object(map) => {
            for (k, v) in map {
                let child = format!("{path}.{k}");
                if k == &k_ref || k == &k_disp || k == &k_msg {
                    out.push(child.clone());
                }
                collect_forbidden_json_keys(v, &child, out);
            }
        }
        Value::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                collect_forbidden_json_keys(v, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

#[test]
fn n13a_public_boundary_scanner_detects_planted_controls_and_passes_precision_controls() {
    let scanner = PublicAssetScanner::new();
    scanner.verify_controls();
}

#[test]
fn n13a_embedded_and_exported_assets_pass_public_boundary_and_manifest_templates() {
    let scanner = PublicAssetScanner::new();
    scanner.verify_controls();

    for name in EXPORTED_ASSETS {
        let text = embedded_str(name);
        let hits = scanner.scan_text(text);
        assert!(
            hits.is_empty(),
            "embedded {name} failed boundary scan: {hits:?}"
        );
    }

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("exported_assets");
    let report = export_assets(&CompiledDemoFixturesV0, &out).expect("export succeeds");
    assert_eq!(report.files.len(), EXPORTED_ASSETS.len());
    for name in EXPORTED_ASSETS {
        let disk_text = std::fs::read_to_string(out.join(name)).expect("utf8 exported asset");
        let hits = scanner.scan_text(&disk_text);
        assert!(
            hits.is_empty(),
            "exported {name} failed boundary scan: {hits:?}"
        );
    }

    let manifest = manifest_json();
    let mut bad_keys = Vec::new();
    collect_forbidden_json_keys(&manifest, "$", &mut bad_keys);
    assert!(
        bad_keys.is_empty(),
        "forbidden F1 keys in manifest: {bad_keys:?}"
    );

    let expected_journeys = [
        (
            DemoJourneyV0::BenignWrite,
            "benign_non_injection",
            "agent",
            "authored",
            "2026-10-20",
            "agent-authored on 2026-10-20; not derived from any corpus".to_owned(),
        ),
        (
            DemoJourneyV0::Unauthorized,
            "injection",
            "human_operator",
            "substantive_rewrite",
            "2026-10-20",
            format!(
                "Human-operator-authored injection fixture (substantive rewrite by the operator of a pre-existing hand-authored sample). Operator statement: \"there u go one handauthored attack:\". Packaged byte-for-byte; never edited by an agent. Recorded 2026-10-20; document sha256:{J2_UNAUTHORIZED_SHA256}."
            ),
        ),
        (
            DemoJourneyV0::InjectionBenign,
            "injection",
            "human_operator",
            "authored",
            "2026-10-20",
            format!(
                "Human-operator-authored injection fixture (authored by the operator). Operator statement: \"that was handwritten today morning\". Packaged byte-for-byte; never edited by an agent. Recorded 2026-10-20; document sha256:{J3_INJECTION_BENIGN_SHA256}."
            ),
        ),
        (
            DemoJourneyV0::Mismatch,
            "benign_non_injection",
            "agent",
            "authored",
            "2026-10-20",
            "agent-authored on 2026-10-20; not derived from any corpus".to_owned(),
        ),
    ];

    for (journey, want_class, want_kind, want_basis, want_date, want_prov) in expected_journeys {
        let entry = journey_entry(journey);
        assert_eq!(entry["document_class"], want_class, "{}", journey.name());
        let auth_obj = entry["document_authorship"]
            .as_object()
            .expect("document_authorship object");
        let mut keys: Vec<&str> = auth_obj.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["basis", "date", "kind"],
            "{}: document_authorship must have exact keys kind, basis, date",
            journey.name()
        );
        assert_eq!(auth_obj["kind"], want_kind, "{}", journey.name());
        assert_eq!(auth_obj["basis"], want_basis, "{}", journey.name());
        assert_eq!(auth_obj["date"], want_date, "{}", journey.name());
        assert_eq!(entry["provenance"], want_prov, "{}", journey.name());
    }

    // Verify load_verified_manifest rejects a forbidden key or non-matching provenance template.
    let k_ref = ["r", "ef"].concat();
    let mut with_ref = manifest_json();
    with_ref["journeys"][0]["document_authorship"]
        .as_object_mut()
        .unwrap()
        .insert(k_ref, Value::String("x".to_owned()));
    assert_eq!(
        load_verified_manifest(&TamperedSource {
            name: MANIFEST_NAME,
            bytes: serde_json::to_vec(&with_ref).unwrap(),
        }),
        Err(DemoFixtureErrorV0::ManifestInvalid)
    );

    let mut bad_prov = manifest_json();
    bad_prov["journeys"][0]["provenance"] = Value::String("agent-authored".to_owned());
    assert_eq!(
        load_verified_manifest(&TamperedSource {
            name: MANIFEST_NAME,
            bytes: serde_json::to_vec(&bad_prov).unwrap(),
        }),
        Err(DemoFixtureErrorV0::ManifestInvalid)
    );
}
