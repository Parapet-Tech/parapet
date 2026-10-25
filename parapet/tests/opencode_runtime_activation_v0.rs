// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! A1 explicit `--provenance-profile opencode` runtime activation on the
//! `parapet run` and `parapet-engine` entry surfaces.
//!
//! Subprocess tests against a local loopback fake upstream (no external
//! network). Carriers are signed with the session key returned once by the
//! live `provenance_session_start` control action.

#![cfg(not(windows))]

use parapet::cli::control::{
    send_control_request, CONTROL_ACTION_PROVENANCE_SESSION_END,
    CONTROL_ACTION_PROVENANCE_SESSION_START, CONTROL_ERROR_SESSION_UNAVAILABLE,
};
use parapet::engine::KNOWN_LLM_HOSTS;
use parapet::provenance::{
    Authenticator, CarrierEnvelope, HmacSha256Authenticator, Segment, SessionKey, Sha256Digest,
    Source, SourceMap, OPENCODE_BUILD, OPENCODE_PROFILE_ID, OPENCODE_SCHEMA_VERSION,
    PROVENANCE_HEADER, SESSION_KEY_LEN,
};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const CONFIG_YAML: &str = include_str!("fixtures/cli_v0/accept_enforced_v0.yaml");
const AGENT_TEXT: &str = "You are a helpful coding assistant.";
const USER_TEXT: &str = "Please summarize the README file.";

// ---------------------------------------------------------------------------
// Fake Anthropic upstream
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CapturedRequest {
    target: String,
    headers: Vec<(String, String)>,
}

struct FakeUpstream {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl FakeUpstream {
    async fn spawn() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let Some(req) = read_request(&mut stream).await else {
                    continue;
                };
                let body = if req.target.ends_with("/chat/completions") {
                    json!({
                        "id": "chatcmpl-a1",
                        "object": "chat.completion",
                        "created": 1_700_000_000u64,
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "a1 ok"},
                            "finish_reason": "stop"
                        }]
                    })
                } else {
                    json!({
                        "id": "msg_a1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-sonnet",
                        "content": [{"type": "text", "text": "a1 ok"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": null,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    })
                }
                .to_string();
                captured.lock().unwrap().push(req);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        Self {
            addr,
            requests,
            handle,
        }
    }

    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    fn last(&self) -> CapturedRequest {
        self.requests.lock().unwrap().last().unwrap().clone()
    }
}

impl Drop for FakeUpstream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn read_request(stream: &mut TcpStream) -> Option<CapturedRequest> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let target = lines.next()?.split_whitespace().nth(1)?.to_string();
    let mut headers = Vec::new();
    let mut content_len = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_string(), v.trim().to_string());
            if k.eq_ignore_ascii_case("content-length") {
                content_len = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    while buf.len() < header_end + content_len {
        match stream.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    Some(CapturedRequest { target, headers })
}

// ---------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------

struct ChildGuard(Option<Child>);

impl ChildGuard {
    /// Wait for exit and return `(stdout, stderr)`.
    fn finish(mut self) -> (String, String) {
        let child = self.0.take().unwrap();
        let out = child.wait_with_output().unwrap();
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn clear_upstream_env(cmd: &mut Command) {
    for key in [
        "PARAPET_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "PARAPET_ANTHROPIC_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "PARAPET_EXTRA_HOSTS",
        "PARAPET_UPSTREAM_URL",
        "PARAPET_PROVENANCE_PROFILE",
    ] {
        cmd.env_remove(key);
    }
    for &host in KNOWN_LLM_HOSTS {
        cmd.env_remove(format!(
            "PARAPET_{}_BASE_URL",
            host.to_ascii_uppercase().replace(['.', '-'], "_")
        ));
    }
}

fn private_state_dir(base: &Path, name: &str) -> PathBuf {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let dir = base.join(name);
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
        .unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let control_json = dir.join("control.json");
    std::fs::write(
        &control_json,
        br#"{"schema":"parapet-control-state/v0","effect_root_must_be_new":true}"#,
    )
    .unwrap();
    std::fs::set_permissions(&control_json, std::fs::Permissions::from_mode(0o600)).unwrap();
    dir
}

fn write_config(dir: &Path) -> PathBuf {
    let cfg = dir.join("parapet.yaml");
    std::fs::write(&cfg, CONFIG_YAML).unwrap();
    cfg
}

#[derive(Clone, Copy)]
enum Surface {
    ParapetRun,
    ParapetEngine,
}

fn spawn_surface(
    surface: Surface,
    state_dir: &Path,
    cfg: &Path,
    upstream: &FakeUpstream,
    extra: &[&str],
    env: &[(&str, &str)],
) -> ChildGuard {
    let _ = std::fs::remove_file(state_dir.join("ready.port"));
    let port = upstream.addr.port();
    let anthropic = format!("anthropic=http://127.0.0.1:{port}");
    let openai = format!("openai=http://127.0.0.1:{port}");
    let mut cmd = match surface {
        Surface::ParapetRun => {
            let mut c = Command::new(env!("CARGO_BIN_EXE_parapet"));
            c.args(["run", "--json"]);
            c
        }
        Surface::ParapetEngine => Command::new(env!("CARGO_BIN_EXE_parapet-engine")),
    };
    clear_upstream_env(&mut cmd);
    cmd.envs(env.iter().copied())
        .args([
            "--port",
            "0",
            "--config",
            cfg.to_str().unwrap(),
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--ready-file",
            "ready.port",
            "--upstream",
            &anthropic,
            "--upstream",
            &openai,
            "--allow-loopback-http-upstream",
        ])
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    ChildGuard(Some(cmd.spawn().expect("spawn runtime surface")))
}

async fn wait_ready(state_dir: &Path) -> u16 {
    for _ in 0..400 {
        if let Ok(text) = std::fs::read_to_string(state_dir.join("ready.port")) {
            if let Ok(p) = text.trim().parse::<u16>() {
                if p != 0 {
                    return p;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for ready file");
}

fn control(state_dir: &Path, action: &str) -> Value {
    send_control_request(state_dir, action).expect("control request")
}

fn disable(state_dir: &Path) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
    clear_upstream_env(&mut cmd);
    let out = cmd
        .args(["disable", "--json", "--state-dir"])
        .arg(state_dir)
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "disable failed: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

fn read_all_files(root: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for entry in rd.flatten() {
            let path = entry.path();
            let meta = std::fs::symlink_metadata(&path).unwrap();
            if meta.is_dir() {
                out.extend(read_all_files(&path));
            } else if meta.is_file() {
                out.extend(std::fs::read(&path).unwrap_or_default());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Carrier helpers
// ---------------------------------------------------------------------------

struct Grant {
    session_id: String,
    key_hex: String,
}

impl Grant {
    fn from_start(resp: &Value) -> Self {
        assert_eq!(resp["ok"], true, "start failed: {resp}");
        assert_eq!(resp["profile"], OPENCODE_PROFILE_ID);
        assert_eq!(resp["schema_version"], OPENCODE_SCHEMA_VERSION);
        assert_eq!(resp["builds"], json!([OPENCODE_BUILD]));
        let key_hex = resp["session_key_hex"].as_str().unwrap().to_owned();
        assert_eq!(key_hex.len(), SESSION_KEY_LEN * 2);
        Self {
            session_id: resp["session_id"].as_str().unwrap().to_owned(),
            key_hex,
        }
    }

    fn key(&self) -> SessionKey {
        let mut bytes = [0u8; SESSION_KEY_LEN];
        for (i, slot) in bytes.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&self.key_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        SessionKey::from_bytes(bytes)
    }
}

fn anthropic_body() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "claude-3-5-sonnet",
        "max_tokens": 16,
        "system": AGENT_TEXT,
        "messages": [{"role": "user", "content": USER_TEXT}]
    }))
    .unwrap()
}

fn openai_body() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": AGENT_TEXT},
            {"role": "user", "content": USER_TEXT}
        ]
    }))
    .unwrap()
}

/// Sign `body` (system segment as configured agent, user segment as user)
/// with the live grant; returns `(envelope_bytes, tag_hex)`.
fn sign(grant: &Grant, body: &[u8], sequence: u64) -> (Vec<u8>, String) {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    sign_at(grant, body, sequence, now_ms)
}

fn sign_at(grant: &Grant, body: &[u8], sequence: u64, now_ms: u64) -> (Vec<u8>, String) {
    let source_map = SourceMap {
        segments: vec![
            Segment {
                message_index: 0,
                start: 0,
                end: AGENT_TEXT.len(),
                source: Source::ConfiguredAgent,
                sha256: Sha256Digest::of(AGENT_TEXT.as_bytes()),
            },
            Segment {
                message_index: 1,
                start: 0,
                end: USER_TEXT.len(),
                source: Source::User,
                sha256: Sha256Digest::of(USER_TEXT.as_bytes()),
            },
        ],
    };
    let mut env = CarrierEnvelope {
        schema_version: OPENCODE_SCHEMA_VERSION,
        profile: OPENCODE_PROFILE_ID.to_string(),
        build: OPENCODE_BUILD.to_string(),
        session_id: grant.session_id.clone(),
        sequence,
        issued_at_ms: now_ms - 1_000,
        expires_at_ms: now_ms + 20_000,
        request_body_sha256: Sha256Digest::of(body),
        source_map_sha256: source_map.canonical_digest(),
        source_map,
        authenticator: Authenticator::from_bytes(vec![0u8; 32]),
    };
    let tag = HmacSha256Authenticator::new(grant.key()).tag(&env.signed_payload());
    env.authenticator = Authenticator::from_bytes(tag.to_vec());
    (
        serde_json::to_vec(&env).unwrap(),
        Sha256Digest::from_bytes(tag).to_hex(),
    )
}

async fn post(
    port: u16,
    path: &str,
    body: &[u8],
    carrier: Option<&[u8]>,
) -> (reqwest::StatusCode, String) {
    let mut req = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}{path}"))
        .header("content-type", "application/json")
        .body(body.to_vec());
    if let Some(env) = carrier {
        req = req.header(PROVENANCE_HEADER, env.to_vec());
    }
    let resp = req.send().await.expect("send request");
    let status = resp.status();
    (status, resp.text().await.unwrap())
}

fn assert_carrier_rejected(status: reqwest::StatusCode, body: &str, outcome: Option<&str>) {
    assert_eq!(status, reqwest::StatusCode::FORBIDDEN, "body={body}");
    let v: Value = serde_json::from_str(body).unwrap();
    assert_eq!(
        v["error"]["type"], "parapet_carrier_rejected",
        "body={body}"
    );
    assert_eq!(v["error"]["supported_profile"], OPENCODE_PROFILE_ID);
    if let Some(o) = outcome {
        assert_eq!(v["error"]["outcome"], o, "body={body}");
    }
}

fn assert_no_secret(surfaces: &[(&str, &str)], needles: &[&str]) {
    for (label, surface) in surfaces {
        for needle in needles {
            assert!(
                !surface.contains(needle),
                "{label} leaked secret material {needle:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_run_without_profile_keeps_ordinary_routing_and_ignores_env() {
    let upstream = FakeUpstream::spawn().await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = write_config(tmp.path());
    let state_dir = private_state_dir(tmp.path(), "state-default");
    // An environment variable must never implicitly activate the profile.
    let child = spawn_surface(
        Surface::ParapetRun,
        &state_dir,
        &cfg,
        &upstream,
        &[],
        &[("PARAPET_PROVENANCE_PROFILE", "opencode")],
    );
    let port = wait_ready(&state_dir).await;

    let (status, body) = post(port, "/v1/messages", &anthropic_body(), None).await;
    assert_eq!(status, reqwest::StatusCode::OK, "body={body}");
    assert_eq!(upstream.count(), 1);
    let (status, body) = post(port, "/v1/chat/completions", &openai_body(), None).await;
    assert_eq!(status, reqwest::StatusCode::OK, "body={body}");
    assert_eq!(upstream.count(), 2);

    let start = control(&state_dir, CONTROL_ACTION_PROVENANCE_SESSION_START);
    assert_eq!(start["ok"], false);
    assert_eq!(start["error_code"], CONTROL_ERROR_SESSION_UNAVAILABLE);
    let end = control(&state_dir, CONTROL_ACTION_PROVENANCE_SESSION_END);
    assert_eq!(end["error_code"], CONTROL_ERROR_SESSION_UNAVAILABLE);
    assert_eq!(control(&state_dir, "status")["ok"], true);

    disable(&state_dir);
    child.finish();
}

#[test]
fn unknown_profile_and_missing_state_dir_are_rejected_before_listening() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = write_config(tmp.path());
    let state_dir = private_state_dir(tmp.path(), "state-reject");

    for value in ["openai", "OpenCode", ""] {
        let mut run = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_upstream_env(&mut run);
        let out = run
            .args(["run", "--json", "--port", "0", "--config"])
            .arg(&cfg)
            .arg("--state-dir")
            .arg(&state_dir)
            .args(["--ready-file", "ready.port", "--provenance-profile", value])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "run value={value:?}");
        let v: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(v["error_code"], "invalid_arguments");

        let mut eng = Command::new(env!("CARGO_BIN_EXE_parapet-engine"));
        clear_upstream_env(&mut eng);
        let out = eng
            .args(["--port", "0", "--config"])
            .arg(&cfg)
            .arg("--state-dir")
            .arg(&state_dir)
            .args(["--ready-file", "ready.port", "--provenance-profile", value])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "engine value={value:?}");
    }

    // parapet-engine requires --state-dir for the control-plane session grant.
    let mut eng = Command::new(env!("CARGO_BIN_EXE_parapet-engine"));
    clear_upstream_env(&mut eng);
    let out = eng
        .args(["--port", "0", "--config"])
        .arg(&cfg)
        .args(["--provenance-profile", "opencode"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("runtime options require --state-dir"));

    assert!(!state_dir.join("ready.port").exists());
    assert!(!state_dir.join("control.sock").exists());
    assert!(!state_dir.join("engine.owner.json").exists());
}

async fn exercise_opencode_activation(surface: Surface, tag: &str) {
    let upstream = FakeUpstream::spawn().await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = write_config(tmp.path());
    let state_dir = private_state_dir(tmp.path(), &format!("state-{tag}"));
    let child = spawn_surface(
        surface,
        &state_dir,
        &cfg,
        &upstream,
        &["--provenance-profile", "opencode"],
        &[],
    );
    let port = wait_ready(&state_dir).await;
    let anthropic = anthropic_body();
    let openai = openai_body();
    let mut responses = Vec::new();

    // No active session: missing carrier fails closed before upstream.
    let (s, b) = post(port, "/v1/messages", &anthropic, None).await;
    assert_carrier_rejected(s, &b, Some("session_mismatch"));
    responses.push(b);
    // OpenAI is never implicitly valid.
    let (s, b) = post(port, "/v1/chat/completions", &openai, None).await;
    assert_carrier_rejected(s, &b, Some("schema_mismatch"));
    responses.push(b);
    assert_eq!(upstream.count(), 0);

    // Start returns the key exactly once, in the control response only.
    let start1 = control(&state_dir, CONTROL_ACTION_PROVENANCE_SESSION_START);
    let grant1 = Grant::from_start(&start1);

    let (s, b) = post(port, "/v1/messages", &anthropic, None).await;
    assert_carrier_rejected(s, &b, None);
    responses.push(b);
    assert_eq!(upstream.count(), 0);

    // A verified carrier reaches the projected engine path and upstream once,
    // without forwarding the carrier header.
    let (env, tag1) = sign(&grant1, &anthropic, 0);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env)).await;
    assert_eq!(s, reqwest::StatusCode::OK, "body={b}");
    assert_eq!(upstream.count(), 1);
    let forwarded = upstream.last();
    assert_eq!(forwarded.target, "/v1/messages");
    assert!(forwarded
        .headers
        .iter()
        .all(|(k, _)| !k.eq_ignore_ascii_case(PROVENANCE_HEADER)));

    // Replay of the same carrier is rejected.
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env)).await;
    assert_carrier_rejected(s, &b, None);
    responses.push(b);
    // Wrong route/provider with a validly keyed carrier is rejected.
    let (env_oa, tag_oa) = sign(&grant1, &openai, 1);
    let (s, b) = post(port, "/v1/chat/completions", &openai, Some(&env_oa)).await;
    assert_carrier_rejected(s, &b, Some("schema_mismatch"));
    responses.push(b);
    let (env_alias, tag_alias) = sign(&grant1, &anthropic, 2);
    let (s, b) = post(port, "/anthropic/v1/messages", &anthropic, Some(&env_alias)).await;
    assert_carrier_rejected(s, &b, Some("schema_mismatch"));
    responses.push(b);
    assert_eq!(upstream.count(), 1);

    // Rotation through control invalidates the old key at request admission.
    let start2 = control(&state_dir, CONTROL_ACTION_PROVENANCE_SESSION_START);
    let grant2 = Grant::from_start(&start2);
    assert_ne!(grant1.key_hex, grant2.key_hex);
    assert_ne!(grant1.session_id, grant2.session_id);
    let (env_old, tag_old) = sign(&grant1, &anthropic, 3);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env_old)).await;
    assert_carrier_rejected(s, &b, None);
    responses.push(b);
    let (env_new, tag2) = sign(&grant2, &anthropic, 0);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env_new)).await;
    assert_eq!(s, reqwest::StatusCode::OK, "body={b}");
    assert_eq!(upstream.count(), 2);

    // End invalidates the shared session.
    let end = control(&state_dir, CONTROL_ACTION_PROVENANCE_SESSION_END);
    assert_eq!(end, json!({"schema": "parapet-control/v0", "ok": true}));
    let (env_after_end, tag_end) = sign(&grant2, &anthropic, 1);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env_after_end)).await;
    assert_carrier_rejected(s, &b, Some("session_mismatch"));
    responses.push(b);
    assert_eq!(upstream.count(), 2);

    let status = control(&state_dir, "status").to_string();
    let end_text = end.to_string();
    disable(&state_dir);
    let (stdout1, stderr1) = child.finish();

    // Restart on the same state-dir: a fresh manager rejects the old grant.
    let child = spawn_surface(
        surface,
        &state_dir,
        &cfg,
        &upstream,
        &["--provenance-profile", "opencode"],
        &[],
    );
    let port = wait_ready(&state_dir).await;
    let (env_restart, tag_restart) = sign(&grant2, &anthropic, 2);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env_restart)).await;
    assert_carrier_rejected(s, &b, Some("session_mismatch"));
    responses.push(b);
    let grant3 = Grant::from_start(&control(
        &state_dir,
        CONTROL_ACTION_PROVENANCE_SESSION_START,
    ));
    assert_ne!(grant3.key_hex, grant2.key_hex);
    let (s, b) = post(port, "/v1/messages", &anthropic, Some(&env_restart)).await;
    assert_carrier_rejected(s, &b, None);
    responses.push(b);
    assert_eq!(upstream.count(), 2);
    disable(&state_dir);
    let (stdout2, stderr2) = child.finish();

    // The key appears only in its one-time start response, never in logs,
    // errors, status, receipts, or state-dir files.
    let files = String::from_utf8_lossy(&read_all_files(&state_dir)).into_owned();
    let responses = responses.join("\n");
    let surfaces = [
        ("stdout", stdout1.as_str()),
        ("stderr", stderr1.as_str()),
        ("restart stdout", stdout2.as_str()),
        ("restart stderr", stderr2.as_str()),
        ("rejection bodies", responses.as_str()),
        ("status", status.as_str()),
        ("end response", end_text.as_str()),
        ("state-dir files", files.as_str()),
    ];
    assert_no_secret(
        &surfaces,
        &[
            &grant1.key_hex,
            &grant2.key_hex,
            &grant3.key_hex,
            &tag1,
            &tag_oa,
            &tag_alias,
            &tag_old,
            &tag2,
            &tag_end,
            &tag_restart,
        ],
    );
    assert!(!start1.to_string().contains(&grant2.key_hex));
}

#[tokio::test]
async fn parapet_run_opencode_profile_wires_required_admission_and_session_control() {
    exercise_opencode_activation(Surface::ParapetRun, "run").await;
}

#[tokio::test]
async fn parapet_engine_opencode_profile_wires_required_admission_and_session_control() {
    exercise_opencode_activation(Surface::ParapetEngine, "engine").await;
}

// ---------------------------------------------------------------------------
// In-process wiring (no sockets): shared manager, route binding, projected stage
// ---------------------------------------------------------------------------

mod in_process {
    use super::*;
    use axum::body::Body;
    use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
    use bytes::Bytes;
    use parapet::cli::activation::{build_runtime_routing_with, RuntimeRouting};
    use parapet::cli::control::{ProvenanceRouteBinding, ProvenanceSessionClockFactory};
    use parapet::cli::ProvenanceProfileArg;
    use parapet::config::{load_config, StringSource};
    use parapet::constraint::DslConstraintEvaluator;
    use parapet::engine::{
        DefaultProviderRegistry, EngineDeps, EngineUpstreamClient, HttpBody, HttpError,
        HttpRequest, HttpResponse, HttpSender, UpstreamResolver,
    };
    use parapet::layers::l3_inbound::DefaultInboundScanner;
    use parapet::layers::l5a::L5aScanner;
    use parapet::normalize::L0Normalizer;
    use parapet::provenance::{opencode_2_0_32_profile, Clock, SessionManager};
    use parapet::proxy::Provider;
    use parapet::routing::NoopOrthogonalSensorRouter;
    use parapet::signal::combiner::DefaultVerdictCombiner;
    use parapet::signal::extractor::DefaultSignalExtractor;
    use parapet::trust::RoleTrustAssigner;
    use tower::ServiceExt;

    const NOW_MS: u64 = 1_700_000_000_000;

    struct FixedClock;
    impl Clock for FixedClock {
        fn now_ms(&self) -> u64 {
            NOW_MS
        }
    }

    #[derive(Default)]
    struct CountingSender {
        urls: Mutex<Vec<String>>,
    }

    impl CountingSender {
        fn count(&self) -> usize {
            self.urls.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl HttpSender for CountingSender {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            assert!(!request.headers.contains_key(PROVENANCE_HEADER));
            self.urls.lock().unwrap().push(request.url.clone());
            let body = if request.url.ends_with("/chat/completions") {
                br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#.to_vec()
            } else {
                br#"{"id":"msg_a1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#.to_vec()
            };
            let mut headers = HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            Ok(HttpResponse {
                status: StatusCode::OK,
                headers,
                body: HttpBody::Full(Bytes::from(body)),
            })
        }
    }

    struct LoopbackResolver;
    impl UpstreamResolver for LoopbackResolver {
        fn base_url(&self, provider: Provider) -> String {
            match provider {
                Provider::OpenAi => "http://openai.invalid".to_string(),
                Provider::Anthropic => "http://anthropic.invalid".to_string(),
            }
        }
    }

    fn engine(sender: Arc<CountingSender>) -> Arc<EngineUpstreamClient> {
        let config = Arc::new(
            load_config(&StringSource {
                content: CONFIG_YAML.to_string(),
            })
            .unwrap(),
        );
        Arc::new(EngineUpstreamClient::new_with(EngineDeps {
            config,
            http: sender,
            resolver: Arc::new(LoopbackResolver),
            registry: Arc::new(DefaultProviderRegistry),
            normalizer: Arc::new(L0Normalizer::new()),
            trust_assigner: Arc::new(RoleTrustAssigner),
            l1_scanner: None,
            l1_model: None,
            l2a_scanner: None,
            l2a_semaphore: None,
            inbound_scanner: Arc::new(DefaultInboundScanner::new()),
            constraint_evaluator: Arc::new(DslConstraintEvaluator::new()),
            output_scanner: Arc::new(L5aScanner),
            session_store: None,
            multi_turn_scanner: None,
            orthogonal_sensor_router: Arc::new(NoopOrthogonalSensorRouter::new()),
            signal_extractor: Arc::new(DefaultSignalExtractor::new()),
            verdict_combiner: Arc::new(DefaultVerdictCombiner::new()),
            emit_l1_signals: false,
        }))
    }

    fn routing(
        profile: Option<ProvenanceProfileArg>,
    ) -> (
        RuntimeRouting,
        Arc<SessionManager>,
        Arc<CountingSender>,
        Arc<EngineUpstreamClient>,
    ) {
        let sender = Arc::new(CountingSender::default());
        let sessions = Arc::new(SessionManager::new());
        let clock: Arc<dyn ProvenanceSessionClockFactory> =
            Arc::new(|| -> Box<dyn Clock> { Box::new(FixedClock) });
        let engine = engine(sender.clone());
        let routing = build_runtime_routing_with(profile, engine.clone(), sessions.clone(), clock);
        (routing, sessions, sender, engine)
    }

    async fn send(
        app: &axum::Router,
        path: &str,
        body: &[u8],
        carrier: Option<&[u8]>,
    ) -> (StatusCode, String) {
        let mut req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json");
        if let Some(env) = carrier {
            req = req.header(PROVENANCE_HEADER, HeaderValue::from_bytes(env).unwrap());
        }
        let resp = app
            .clone()
            .oneshot(req.body(Body::from(body.to_vec())).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    fn reject(status: StatusCode, body: &str, outcome: Option<&str>) {
        let status = reqwest::StatusCode::from_u16(status.as_u16()).unwrap();
        assert_carrier_rejected(status, body, outcome);
    }

    fn grant_of(control: &parapet::cli::control::ProvenanceSessionControl) -> Grant {
        let grant = control.start_session().expect("start session");
        let key_hex: String = grant
            .key
            .expose_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Grant {
            session_id: grant.session_id.clone(),
            key_hex,
        }
    }

    #[tokio::test]
    async fn without_profile_routing_is_ordinary_and_exposes_no_session_control() {
        let (routing, sessions, sender, engine) = routing(None);
        assert!(routing.provenance_session.is_none());
        let (s, b) = send(&routing.router, "/v1/messages", &anthropic_body(), None).await;
        assert_eq!(s, StatusCode::OK, "body={b}");
        let (s, b) = send(
            &routing.router,
            "/v1/chat/completions",
            &openai_body(),
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "body={b}");
        assert_eq!(sender.count(), 2);
        assert!(!sessions.is_active());
        assert!(engine.carrier_receipts().is_empty());
    }

    #[tokio::test]
    async fn opencode_profile_binds_exact_anthropic_route_shared_manager_and_projected_stage() {
        let (routing, sessions, sender, engine) = routing(Some(ProvenanceProfileArg::OpenCode));
        let control = routing.provenance_session.expect("session control");
        let app = routing.router;
        let anthropic = anthropic_body();
        let openai = openai_body();

        // Exact identity: Anthropic binding, unmodified opencode profile, shared manager.
        assert_eq!(control.route_binding(), ProvenanceRouteBinding::ANTHROPIC);
        let expected = opencode_2_0_32_profile();
        assert_eq!(control.profile().profile_id, expected.profile_id);
        assert_eq!(control.profile().schema_version, expected.schema_version);
        assert_eq!(control.profile().max_validity_ms, expected.max_validity_ms);
        assert_eq!(control.profile().builds, expected.builds);
        assert!(Arc::ptr_eq(control.sessions(), &sessions));

        let (s, b) = send(&app, "/v1/messages", &anthropic, None).await;
        reject(s, &b, Some("session_mismatch"));
        let (s, b) = send(&app, "/v1/chat/completions", &openai, None).await;
        reject(s, &b, Some("schema_mismatch"));
        assert_eq!(sender.count(), 0);

        let g1 = grant_of(&control);
        assert!(sessions.is_active());
        let (s, b) = send(&app, "/v1/messages", &anthropic, None).await;
        reject(s, &b, None);
        let (env, _) = sign_at(&g1, &openai, 0, NOW_MS);
        let (s, b) = send(&app, "/v1/chat/completions", &openai, Some(&env)).await;
        reject(s, &b, Some("schema_mismatch"));
        assert_eq!(sender.count(), 0);

        let (env, _) = sign_at(&g1, &anthropic, 1, NOW_MS);
        let (s, b) = send(&app, "/v1/messages", &anthropic, Some(&env)).await;
        assert_eq!(s, StatusCode::OK, "body={b}");
        assert_eq!(sender.count(), 1);
        assert!(sender.urls.lock().unwrap()[0].starts_with("http://anthropic.invalid"));
        // The verified carrier took the projected engine path, which records a carrier receipt.
        assert_eq!(engine.carrier_receipts().len(), 1);

        // Rotation and end through the control handle act on request admission.
        let g2 = grant_of(&control);
        let (env, _) = sign_at(&g1, &anthropic, 2, NOW_MS);
        let (s, b) = send(&app, "/v1/messages", &anthropic, Some(&env)).await;
        reject(s, &b, None);
        let (env, _) = sign_at(&g2, &anthropic, 0, NOW_MS);
        let (s, b) = send(&app, "/v1/messages", &anthropic, Some(&env)).await;
        assert_eq!(s, StatusCode::OK, "body={b}");
        assert_eq!(sender.count(), 2);

        control.end_session();
        assert!(!sessions.is_active());
        let (env, _) = sign_at(&g2, &anthropic, 1, NOW_MS);
        let (s, b) = send(&app, "/v1/messages", &anthropic, Some(&env)).await;
        reject(s, &b, Some("session_mismatch"));
        assert_eq!(sender.count(), 2);

        let debug = format!("{control:?}");
        for key in [&g1.key_hex, &g2.key_hex] {
            assert!(!debug.contains(key.as_str()));
        }
    }
}
