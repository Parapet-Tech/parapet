// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! D-0026 Integration Stage 0 — Fail-Closed Upstream + Zero-Leak Diagnostics (R1–R11).

#![cfg(not(windows))]

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use bytes::Bytes;
use parapet::config::{load_config, StringSource};
use parapet::constraint::DslConstraintEvaluator;
use parapet::effect::{EnvelopePartsV0, InjectedBuildIdentityV0, RuntimeEnvelopeV0};
use parapet::engine::{
    production_http_client, DefaultProviderRegistry, EngineDeps, EngineUpstreamClient,
    EnvUpstreamResolver, HttpRequest, HttpSender, ReqwestHttpSender, UpstreamResolver,
    UpstreamTable, KNOWN_LLM_HOSTS,
};
use parapet::layers::l3_inbound::DefaultInboundScanner;
use parapet::layers::l5a::L5aScanner;
use parapet::normalize::L0Normalizer;
use parapet::proxy::{build_router, Provider, ProxyRequest, UpstreamClient};
use parapet::routing::NoopOrthogonalSensorRouter;
use parapet::signal::combiner::DefaultVerdictCombiner;
use parapet::signal::extractor::DefaultSignalExtractor;
use parapet::trust::RoleTrustAssigner;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tower::ServiceExt;

const MANDATORY_PROFILE_YAML: &str = include_str!("fixtures/cli_v0/accept_enforced_v0.yaml");

fn clear_all_upstream_env(cmd: &mut Command) {
    for key in [
        "PARAPET_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
        "PARAPET_ANTHROPIC_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "PARAPET_EXTRA_HOSTS",
        "PARAPET_UPSTREAM_URL",
    ] {
        cmd.env_remove(key);
    }
    for &host in KNOWN_LLM_HOSTS {
        let key = format!(
            "PARAPET_{}_BASE_URL",
            host.to_ascii_uppercase().replace(['.', '-'], "_")
        );
        cmd.env_remove(key);
    }
    for (os_k, _) in std::env::vars_os() {
        if let Some(k) = os_k.to_str() {
            if k.starts_with("PARAPET_") && k.ends_with("_BASE_URL") {
                cmd.env_remove(k);
            }
        }
    }
}

fn write_private_state_dir(base: &Path, name: &str) -> PathBuf {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    let dir = base.join(name);
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)
        .expect("create private state dir");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .expect("chmod 0700 state dir");
    let control_json = dir.join("control.json");
    std::fs::write(
        &control_json,
        br#"{"schema":"parapet-control-state/v0","effect_root_must_be_new":true}"#,
    )
    .expect("write control.json");
    std::fs::set_permissions(&control_json, std::fs::Permissions::from_mode(0o600))
        .expect("chmod 0600 control.json");
    dir
}

fn write_test_config(dir: &Path) -> PathBuf {
    let cfg = dir.join("parapet.yaml");
    std::fs::write(&cfg, MANDATORY_PROFILE_YAML).expect("write parapet.yaml");
    cfg
}

fn list_dir_entries_recursive(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !root.exists() {
        return out;
    }
    fn walk(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if let Ok(rel) = p.strip_prefix(root) {
                    out.push(rel.to_path_buf());
                }
                if p.is_dir() {
                    walk(&p, root, out);
                }
            }
        }
    }
    walk(root, root, &mut out);
    out.sort();
    out
}

fn store_receipt_entry_count(store_root: &Path) -> usize {
    list_dir_entries_recursive(store_root)
        .into_iter()
        .filter(|rel| rel != Path::new(parapet::effect::store::layout::LOCK_NAME))
        .count()
}

fn collect_all_files_bytes(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    if !root.exists() {
        return out;
    }
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.is_file() {
                    if let Ok(bytes) = std::fs::read(&p) {
                        out.push((p, bytes));
                    }
                }
            }
        }
    }
    walk(root, &mut out);
    out
}

fn walk_json_strings(val: &serde_json::Value, out: &mut Vec<String>) {
    match val {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                walk_json_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                out.push(k.clone());
                walk_json_strings(v, out);
            }
        }
        _ => {}
    }
}

#[derive(Clone)]
struct SentinelToken {
    full: String,
    suffix: String,
}

struct SentinelSet {
    tokens: Vec<SentinelToken>,
    c1: String,
    c2: String,
    c3: String,
    c4: String,
    c5: String,
    c6: String,
    c7: String,
    c8: String,
    c9: String,
    c10: String,
    h1: Vec<u8>,
    h2: Vec<u8>,
    h3: Vec<u8>,
    h4_second: Vec<u8>,
    h5: Vec<u8>,
}

impl SentinelSet {
    fn generate() -> Self {
        let seed_base = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5a17_c0de)
            ^ (std::process::id() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut tokens = Vec::new();
        let mut next_token = |prefix: &str, tag: &str, idx: u64| -> String {
            let mixed = seed_base
                .wrapping_add(idx.wrapping_mul(0xbf58_476d_1ce4_e5b9))
                .rotate_left(17)
                ^ 0x94d0_49bb_1331_11eb;
            let digits = [b'1', b'2', b'3', b'4', b'6', b'7', b'8', b'9'];
            let letters = [b'a', b'b', b'c', b'd', b'e', b'f'];
            let mut suffix = String::with_capacity(6);
            for pos in 0..3 {
                let d = digits[((mixed >> (pos * 8)) as usize) % digits.len()] as char;
                let l = letters[((mixed >> (pos * 8 + 4)) as usize) % letters.len()] as char;
                suffix.push(d);
                suffix.push(l);
            }
            let full = format!("{prefix}-{tag}-{suffix}");
            tokens.push(SentinelToken {
                full: full.clone(),
                suffix,
            });
            full
        };

        let c1_u = next_token("SNTLu", "c1u", 1);
        let c1_p = next_token("SNTLp", "c1p", 2);
        let c1_path = next_token("SNTLpath", "c1path", 3);
        let c1 = format!("https://{c1_u}:{c1_p}@gw.example/{c1_path}");

        let c2_raw = next_token("SNTLraw", "c2", 4);
        let c2 = format!("{c2_raw} ::/ not a url");

        let c3_host = next_token("SNTLhost", "c3h", 5);
        let c3_path = next_token("SNTLpath", "c3p", 6);
        let c3 = format!("http://{c3_host}.example:4000/{c3_path}");

        let c4_q = next_token("SNTLq", "c4q", 7);
        let c4_qv = next_token("SNTLqv", "c4v", 8);
        let c4 = format!("https://gw.example/?{c4_q}={c4_qv}");

        let c5_frag = next_token("SNTLfrag", "c5f", 9);
        let c5 = format!("https://gw.example/#{c5_frag}");

        let c6_scheme = next_token("SNTLscheme", "c6s", 10);
        let c6 = format!("{c6_scheme}://gw.example");

        let c7_long = next_token("SNTLlong", "c7l", 11);
        let repeat_count = (4096 / c7_long.len()) + 4;
        let c7 = format!("https://[{}", c7_long.repeat(repeat_count));

        let c8_self = next_token("SNTLself", "c8s", 12);
        let c8 = format!("http://{c8_self}.local:9800");

        let c9_xh = next_token("SNTLxh", "c9x", 13);
        let c9 = format!("api.ok.dev,{c9_xh}@evil.dev");

        let c10_xs = next_token("SNTLxs", "c10x", 14);
        let c10 = format!("https://{c10_xs}.dev");

        let h1_tok = next_token("SNTLh", "h1", 15);
        let h1 = format!("{h1_tok}.evil.com").into_bytes();

        let h2_tok = next_token("SNTLh", "h2", 16);
        let h2 = format!("api.openai.com.{h2_tok}").into_bytes();

        let h3_tok = next_token("SNTLh", "h3", 17);
        let mut h3 = h3_tok.into_bytes();
        h3.extend_from_slice(&[0xFF, 0xFE]);

        let h4_tok = next_token("SNTLh", "h4", 18);
        let h4_second = h4_tok.into_bytes();

        let h5_tok = next_token("SNTLh", "h5", 19);
        let h5 = format!("{h5_tok}\tevil.com").into_bytes();

        Self {
            tokens,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            h1,
            h2,
            h3,
            h4_second,
            h5,
        }
    }

    fn percent_encode_all(s: &str) -> String {
        let mut out = String::with_capacity(s.len() * 3);
        for b in s.as_bytes() {
            out.push_str(&format!("%{b:02x}"));
        }
        out
    }

    fn scan_bytes(&self, surface: &[u8]) -> Option<String> {
        let lower: Vec<u8> = surface.iter().map(|b| b.to_ascii_lowercase()).collect();
        if lower.windows(4).any(|w| w == b"sntl") {
            return Some("matched literal prefix SNTL".to_string());
        }
        for tok in &self.tokens {
            let full_lower = tok.full.to_ascii_lowercase();
            let pct_lower = Self::percent_encode_all(&tok.full).to_ascii_lowercase();
            let idna_lower = url::Host::parse(&tok.full)
                .ok()
                .map(|h| h.to_string().to_ascii_lowercase())
                .unwrap_or_else(|| full_lower.clone());
            let suffix_lower = tok.suffix.to_ascii_lowercase();
            for (kind, needle) in [
                ("raw", full_lower.as_bytes()),
                ("percent_encoded", pct_lower.as_bytes()),
                ("idna_lowered", idna_lower.as_bytes()),
                ("hex_suffix", suffix_lower.as_bytes()),
            ] {
                if !needle.is_empty()
                    && lower.len() >= needle.len()
                    && lower.windows(needle.len()).any(|w| w == needle)
                {
                    return Some(format!("matched {kind} for {}", tok.full));
                }
            }
        }
        None
    }

    fn assert_clean(&self, label: &str, surface: &[u8]) {
        if let Some(reason) = self.scan_bytes(surface) {
            panic!("sentinel leak on {label}: {reason}");
        }
    }

    fn assert_state_dir_clean(&self, label: &str, state_dir: &Path) {
        for (path, bytes) in collect_all_files_bytes(state_dir) {
            self.assert_clean(&format!("{label}:{}:content", path.display()), &bytes);
            self.assert_clean(
                &format!("{label}:{}:path", path.display()),
                path.to_string_lossy().as_bytes(),
            );
        }
    }
}

#[test]
fn sentinel_set_matcher_positive_control() {
    let s = SentinelSet::generate();
    assert!(s
        .scan_bytes(b"completely benign diagnostic output")
        .is_none());
    let raw_leak = format!("diagnostic leaked {}", s.tokens[0].full);
    assert!(s.scan_bytes(raw_leak.as_bytes()).is_some());
    let pct_leak = format!(
        "diagnostic leaked {}",
        SentinelSet::percent_encode_all(&s.tokens[0].full)
    );
    assert!(s.scan_bytes(pct_leak.as_bytes()).is_some());
    let idna_leak = format!(
        "diagnostic leaked {}",
        url::Host::parse(&s.tokens[0].full)
            .unwrap()
            .to_string()
            .to_ascii_lowercase()
    );
    assert!(s.scan_bytes(idna_leak.as_bytes()).is_some());
    let suffix_leak = format!("diagnostic leaked {}", s.tokens[0].suffix);
    assert!(s.scan_bytes(suffix_leak.as_bytes()).is_some());
    let prefix_leak = b"truncated sNtL-leak";
    assert!(s.scan_bytes(prefix_leak).is_some());
}

struct LoopbackRecorder {
    addr: SocketAddr,
    connections: Arc<AtomicUsize>,
    handle: tokio::task::JoinHandle<()>,
}

impl LoopbackRecorder {
    async fn spawn_silent() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind loopback recorder");
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let conn_clone = connections.clone();
        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                conn_clone.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
            }
        });
        Self {
            addr,
            connections,
            handle,
        }
    }

    async fn spawn_redirect_to(target_url: String) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind redirect recorder");
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let conn_clone = connections.clone();
        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                conn_clone.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 2048];
                let _ = stream.read(&mut buf).await;
                let resp = format!(
                    "HTTP/1.1 307 Temporary Redirect\r\nLocation: {target_url}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        });
        Self {
            addr,
            connections,
            handle,
        }
    }

    fn count(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
}

impl Drop for LoopbackRecorder {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[derive(Clone, Default)]
struct ScopedTraceBuffer {
    buf: Arc<Mutex<Vec<u8>>>,
}

struct ScopedTraceWriter {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl Write for ScopedTraceWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for ScopedTraceBuffer {
    type Writer = ScopedTraceWriter;
    fn make_writer(&'a self) -> Self::Writer {
        ScopedTraceWriter {
            buf: self.buf.clone(),
        }
    }
}

#[tokio::test]
async fn r1_parapet_run_json_refuses_http_upstream_without_connection_or_leak() {
    let recorder = LoopbackRecorder::spawn_silent().await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let state_dir = write_private_state_dir(tmp.path(), "state-r1");
    let before_entries = list_dir_entries_recursive(&state_dir);

    let bad_url = format!("http://127.0.0.1:{}", recorder.addr.port());
    let port_str = recorder.addr.port().to_string();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
    clear_all_upstream_env(&mut cmd);
    let out = cmd
        .env("PARAPET_OPENAI_BASE_URL", &bad_url)
        .args([
            "run",
            "--json",
            "--config",
            cfg_path.to_str().unwrap(),
            "--state-dir",
            state_dir.to_str().unwrap(),
        ])
        .output()
        .expect("spawn parapet run --json");

    assert_eq!(out.status.code(), Some(4));
    assert_eq!(recorder.count(), 0);
    assert_eq!(
        list_dir_entries_recursive(&state_dir),
        before_entries,
        "refused startup must leave no new state-dir artifacts"
    );

    let stdout_str = String::from_utf8(out.stdout).unwrap();
    let stderr_str = String::from_utf8(out.stderr).unwrap();
    let json: serde_json::Value = serde_json::from_str(stdout_str.trim()).expect("valid json");
    assert_eq!(json["error_code"], "upstream_config_invalid");
    assert!(stderr_str.contains("PARAPET_OPENAI_BASE_URL"));
    assert!(stderr_str.contains("non_https_upstream"));
    assert!(stderr_str.contains("Fix:"));

    for surface in [&stdout_str, &stderr_str] {
        assert!(
            !surface.contains("127.0.0.1"),
            "leaked 127.0.0.1: {surface}"
        );
        assert!(
            !surface.contains(&port_str),
            "leaked recorder port {port_str}: {surface}"
        );
    }
}

#[tokio::test]
async fn r2_parapet_run_human_mode_and_openai_base_url_self_loop_refused() {
    let recorder = LoopbackRecorder::spawn_silent().await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let state_dir = write_private_state_dir(tmp.path(), "state-r2");

    // Human mode with PARAPET_OPENAI_BASE_URL
    let bad_url = format!("http://127.0.0.1:{}", recorder.addr.port());
    let port_str = recorder.addr.port().to_string();
    let mut cmd1 = Command::new(env!("CARGO_BIN_EXE_parapet"));
    clear_all_upstream_env(&mut cmd1);
    let out1 = cmd1
        .env("PARAPET_OPENAI_BASE_URL", &bad_url)
        .args([
            "run",
            "--config",
            cfg_path.to_str().unwrap(),
            "--state-dir",
            state_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out1.status.code(), Some(4));
    assert_eq!(recorder.count(), 0);
    let stdout1 = String::from_utf8_lossy(&out1.stdout);
    let stderr1 = String::from_utf8_lossy(&out1.stderr);
    assert!(stderr1.contains("upstream configuration refused"));
    assert!(stderr1.contains("PARAPET_OPENAI_BASE_URL"));
    assert!(stderr1.contains("non_https_upstream"));
    assert!(!stdout1.contains("127.0.0.1") && !stderr1.contains("127.0.0.1"));
    assert!(!stdout1.contains(&port_str) && !stderr1.contains(&port_str));

    // Self-loop control with OPENAI_BASE_URL=http://127.0.0.1:9800
    let mut cmd2 = Command::new(env!("CARGO_BIN_EXE_parapet"));
    clear_all_upstream_env(&mut cmd2);
    let out2 = cmd2
        .env("OPENAI_BASE_URL", "http://127.0.0.1:9800")
        .args([
            "run",
            "--config",
            cfg_path.to_str().unwrap(),
            "--state-dir",
            state_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out2.status.code(), Some(4));
    let stdout2 = String::from_utf8_lossy(&out2.stdout);
    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(stderr2.contains("OPENAI_BASE_URL"));
    assert!(stderr2.contains("PARAPET_OPENAI_BASE_URL"));
    assert!(!stdout2.contains("9800") && !stderr2.contains("9800"));
    assert!(!stdout2.contains("127.0.0.1") && !stderr2.contains("127.0.0.1"));
}

#[tokio::test]
async fn r3_parapet_engine_refuses_old_litellm_doc_config_and_doc_contract_holds() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let state_dir = write_private_state_dir(tmp.path(), "state-r3");

    // Pick an unused local port for parapet-engine --port
    let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let engine_port = probe.local_addr().unwrap().port();
    drop(probe);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet-engine"));
    clear_all_upstream_env(&mut cmd);
    let out = cmd
        .env("PARAPET_API_OPENAI_COM_BASE_URL", "http://127.0.0.1:4000")
        .args([
            "--config",
            cfg_path.to_str().unwrap(),
            "--port",
            &engine_port.to_string(),
            "--state-dir",
            state_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run parapet-engine");

    assert_eq!(out.status.code(), Some(1));
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", engine_port)).is_err(),
        "listener must not be bound when startup fails"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stderr.contains(
        "upstream configuration refused: PARAPET_API_OPENAI_COM_BASE_URL: non_https_upstream"
    ));
    assert!(!stderr.contains("127.0.0.1:4000"));
    assert!(!stdout.contains("127.0.0.1:4000"));

    // Verify docs/litellm.md content contract (§5)
    let doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/litellm.md"));
    assert!(!doc.contains("PARAPET_API_OPENAI_COM_BASE_URL = \"http://127.0.0.1:4000\""));
    assert!(doc.contains("upstream_config_invalid"));
    assert!(doc.contains("https://litellm.internal.example"));
    assert!(doc.contains("PARAPET_OPENAI_BASE_URL"));
    assert!(doc.contains("OPENAI_BASE_URL"));
    assert!(doc.contains("not yet supported"));
}

#[tokio::test]
async fn r4_credential_prompt_and_header_misrouting_refused_with_zero_leak() {
    let recorder_a = LoopbackRecorder::spawn_silent().await;
    let s = SentinelSet::generate();
    let tmp = tempfile::tempdir().unwrap();
    let state_dir = write_private_state_dir(tmp.path(), "state-r4");
    let store_root = state_dir.join("receipt-evidence");
    let store = Arc::new(
        parapet::effect::store::ProtectedEvidenceStoreV0::open(&store_root).expect("open store"),
    );

    struct TestHttpsResolver {
        target: String,
    }
    impl UpstreamResolver for TestHttpsResolver {
        fn base_url(&self, _provider: Provider) -> String {
            self.target.clone()
        }
    }

    let cfg = Arc::new(
        load_config(&StringSource {
            content: MANDATORY_PROFILE_YAML.to_string(),
        })
        .unwrap(),
    );
    let parts = EnvelopePartsV0::new(
        cfg.clone(),
        store.clone(),
        Arc::new(InjectedBuildIdentityV0::unavailable()),
    );
    let envelope = Arc::new(RuntimeEnvelopeV0::new(parts));

    let deps = EngineDeps {
        config: cfg,
        http: Arc::new(ReqwestHttpSender::new(production_http_client())),
        resolver: Arc::new(TestHttpsResolver {
            target: format!("https://127.0.0.1:{}", recorder_a.addr.port()),
        }),
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
    };
    let app = build_router(Arc::new(EngineUpstreamClient::new_with_envelope(
        deps, envelope,
    )));

    let trace_buf = ScopedTraceBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_writer(trace_buf.clone())
        .without_time()
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let single_headers: Vec<Vec<u8>> = vec![
        b"evil.com".to_vec(),
        s.h1.clone(),
        s.h2.clone(),
        s.h3.clone(),
        s.h5.clone(),
    ];

    for hdr in single_headers {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer sentinel-AUTH-7f3")
            .header(
                "x-parapet-original-host",
                HeaderValue::from_bytes(&hdr).unwrap(),
            )
            .body(Body::from(
                r#"{"model":"gpt-4o","messages":[{"role":"user","content":"sentinel-PROMPT-91c"}]}"#,
            ))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.headers()
                .get("x-parapet-error")
                .and_then(|v| v.to_str().ok()),
            Some("parapet_upstream_host_not_allowed")
        );
        for (k, v) in resp.headers() {
            s.assert_clean(k.as_str(), v.as_bytes());
            let vs = String::from_utf8_lossy(v.as_bytes());
            assert!(!vs.contains("sentinel-AUTH-7f3"));
            assert!(!vs.contains("sentinel-PROMPT-91c"));
        }
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        s.assert_clean("R4 response body", &body);
        let bs = String::from_utf8_lossy(&body);
        assert!(!bs.contains("sentinel-AUTH-7f3"));
        assert!(!bs.contains("sentinel-PROMPT-91c"));
    }

    // H4 duplicate headers
    let req_h4 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sentinel-AUTH-7f3")
        .header("x-parapet-original-host", "api.openai.com")
        .header(
            "x-parapet-original-host",
            HeaderValue::from_bytes(&s.h4_second).unwrap(),
        )
        .body(Body::from(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"sentinel-PROMPT-91c"}]}"#,
        ))
        .unwrap();
    let resp_h4 = app.oneshot(req_h4).await.unwrap();
    assert_eq!(resp_h4.status(), StatusCode::BAD_REQUEST);
    let body_h4 = axum::body::to_bytes(resp_h4.into_body(), 1024 * 1024)
        .await
        .unwrap();
    s.assert_clean("R4 H4 body", &body_h4);

    assert_eq!(recorder_a.count(), 0);
    assert_eq!(store_receipt_entry_count(&store_root), 0);

    let captured_logs = trace_buf.buf.lock().unwrap().clone();
    s.assert_clean("R4 tracing capture", &captured_logs);
    let log_str = String::from_utf8_lossy(&captured_logs);
    assert!(!log_str.contains("sentinel-AUTH-7f3"));
    assert!(!log_str.contains("sentinel-PROMPT-91c"));

    s.assert_state_dir_clean("R4 state dir", &state_dir);
    for (_, file_bytes) in collect_all_files_bytes(&state_dir) {
        let text = String::from_utf8_lossy(&file_bytes);
        assert!(!text.contains("sentinel-AUTH-7f3"));
        assert!(!text.contains("sentinel-PROMPT-91c"));
    }
}

#[tokio::test]
async fn r5_redirect_not_followed_by_sender_and_refused_with_502_by_engine() {
    let recorder_b = LoopbackRecorder::spawn_silent().await;
    let steal_url = format!("http://127.0.0.1:{}/steal", recorder_b.addr.port());
    let recorder_a = LoopbackRecorder::spawn_redirect_to(steal_url.clone()).await;

    // 1. Drive ReqwestHttpSender::send directly
    let sender = ReqwestHttpSender::new(production_http_client());
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sentinel-AUTH"),
    );
    let direct_resp = sender
        .send(HttpRequest {
            method: Method::POST,
            url: format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                recorder_a.addr.port()
            ),
            headers,
            body: Bytes::from_static(b"{}"),
            timeout_ms: Some(2000),
            stream: false,
        })
        .await
        .expect("direct sender call");
    assert_eq!(direct_resp.status, StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(recorder_a.count(), 1);
    assert_eq!(recorder_b.count(), 0);

    // 2. Drive the engine forward path through a test resolver
    struct LoopbackTestResolver {
        base: String,
    }
    impl UpstreamResolver for LoopbackTestResolver {
        fn base_url(&self, _provider: Provider) -> String {
            self.base.clone()
        }
    }

    let cfg = Arc::new(
        load_config(&StringSource {
            content: MANDATORY_PROFILE_YAML.to_string(),
        })
        .unwrap(),
    );
    let deps = EngineDeps {
        config: cfg,
        http: Arc::new(ReqwestHttpSender::new(production_http_client())),
        resolver: Arc::new(LoopbackTestResolver {
            base: format!("http://127.0.0.1:{}", recorder_a.addr.port()),
        }),
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
    };
    let engine = EngineUpstreamClient::new_with(deps);
    let mut req_headers = HeaderMap::new();
    req_headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sentinel-AUTH"),
    );
    let resp = engine
        .forward(
            Provider::OpenAi,
            ProxyRequest {
                method: Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                headers: req_headers,
                body: Bytes::from_static(
                    br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                ),
            },
        )
        .await
        .expect("engine forward");

    assert_eq!(resp.status, StatusCode::BAD_GATEWAY);
    assert_eq!(
        resp.headers
            .get("x-parapet-error")
            .and_then(|v| v.to_str().ok()),
        Some("parapet_upstream_redirect_refused")
    );
    assert!(
        resp.headers.get(reqwest::header::LOCATION).is_none(),
        "Location header must never be relayed"
    );
    assert_eq!(recorder_b.count(), 0);
}

#[tokio::test]
async fn r6_unsupported_unknown_and_non_canonical_paths_return_json_without_forwarding() {
    let recorder = LoopbackRecorder::spawn_silent().await;
    struct LoopbackTestResolver {
        base: String,
    }
    impl UpstreamResolver for LoopbackTestResolver {
        fn base_url(&self, _provider: Provider) -> String {
            self.base.clone()
        }
    }

    let cfg = Arc::new(
        load_config(&StringSource {
            content: MANDATORY_PROFILE_YAML.to_string(),
        })
        .unwrap(),
    );
    let deps = EngineDeps {
        config: cfg,
        http: Arc::new(ReqwestHttpSender::new(production_http_client())),
        resolver: Arc::new(LoopbackTestResolver {
            base: format!("http://127.0.0.1:{}", recorder.addr.port()),
        }),
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
    };
    let app = build_router(Arc::new(EngineUpstreamClient::new_with(deps)));

    let probes = [
        (
            "POST",
            "/v1/responses",
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_wire_format",
        ),
        (
            "POST",
            "/v1/embeddings",
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_wire_format",
        ),
        (
            "GET",
            "/v1/models",
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_wire_format",
        ),
        ("POST", "/foo", StatusCode::NOT_FOUND, "unknown_path"),
        (
            "POST",
            "/v1/chat/completions/../../embeddings",
            StatusCode::BAD_REQUEST,
            "path_not_canonical",
        ),
    ];

    for (method, path, expected_status, expected_code) in probes {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .body(Body::from("{}"))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), expected_status, "path={path}");
        assert_eq!(
            resp.headers()
                .get("x-parapet-error")
                .and_then(|v| v.to_str().ok()),
            Some(expected_code)
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], expected_code);
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("POST /v1/chat/completions"));
        assert!(msg.contains("POST /openai/v1/chat/completions"));
        assert!(msg.contains("POST /v1/messages"));
    }

    assert_eq!(recorder.count(), 0);
}

#[test]
fn r7_1d_1e_accept_and_demo_byte_identity_with_upstream_env_cleared() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let shadow_cfg_path = tmp.path().join("parapet_shadow.yaml");
    std::fs::write(
        &shadow_cfg_path,
        include_str!("fixtures/cli_v0/accept_shadow_l3_off_v0.yaml"),
    )
    .expect("write shadow yaml");

    // 1. parapet accept --json across all 4 valid scenarios
    for (scenario, needs_er, cfg) in [
        ("allow_no_executor", false, &cfg_path),
        ("allow_verified_write", true, &cfg_path),
        ("block_no_effect", true, &cfg_path),
        ("shadow_parse_error_l3_off", false, &shadow_cfg_path),
    ] {
        let state_dir = tmp.path().join(format!("state-r7-{scenario}"));
        let er_dir = tmp.path().join(format!("er-r7-{scenario}"));
        let mut accept_cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut accept_cmd);
        accept_cmd.args([
            "accept",
            "--json",
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--config",
            cfg.to_str().unwrap(),
            "--scenario",
            scenario,
        ]);
        if needs_er {
            accept_cmd.args(["--effect-root", er_dir.to_str().unwrap()]);
        }
        let accept_out = accept_cmd.output().expect("run parapet accept");
        assert_eq!(
            accept_out.status.code(),
            Some(0),
            "scenario={scenario} stderr={}",
            String::from_utf8_lossy(&accept_out.stderr)
        );
        let accept_json: serde_json::Value =
            serde_json::from_slice(&accept_out.stdout).expect("parse accept json");
        assert_eq!(accept_json["schema"], "parapet-cli-accept/v0");
        assert_eq!(accept_json["scenario"], scenario);
        assert_eq!(accept_json["real_model_contacted"], false);
    }

    // 2. parapet demo export deterministic byte-identity across two runs
    let export_a = tmp.path().join("export-a");
    let export_b = tmp.path().join("export-b");
    for dest in [&export_a, &export_b] {
        let mut demo_cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut demo_cmd);
        let demo_out = demo_cmd
            .args(["demo", "--json", "export", "--out", dest.to_str().unwrap()])
            .output()
            .expect("run parapet demo export");
        assert_eq!(
            demo_out.status.code(),
            Some(0),
            "stderr={}",
            String::from_utf8_lossy(&demo_out.stderr)
        );
    }
    let files_a = collect_all_files_bytes(&export_a);
    let files_b = collect_all_files_bytes(&export_b);
    assert!(!files_a.is_empty());
    assert_eq!(files_a.len(), files_b.len());
    for ((pa, ba), (pb, bb)) in files_a.iter().zip(files_b.iter()) {
        assert_eq!(
            pa.strip_prefix(&export_a).unwrap(),
            pb.strip_prefix(&export_b).unwrap()
        );
        assert_eq!(ba, bb, "demo export bytes must be identical");
    }
}

#[test]
fn r8_offline_fixture_with_invalid_env_refuses_startup_with_zero_leak() {
    let s = SentinelSet::generate();
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());

    for (idx, val, expected_rule) in [
        (1, "http://x", "non_https_upstream"),
        (2, s.c2.as_str(), "unparseable"),
    ] {
        let state_dir = write_private_state_dir(tmp.path(), &format!("state-r8-{idx}"));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut cmd);
        let out = cmd
            .env("PARAPET_OPENAI_BASE_URL", val)
            .args([
                "run",
                "--json",
                "--config",
                cfg_path.to_str().unwrap(),
                "--state-dir",
                state_dir.to_str().unwrap(),
                "--offline-fixture",
                "v0",
            ])
            .output()
            .expect("run parapet run --offline-fixture v0");
        assert_eq!(out.status.code(), Some(4));
        s.assert_clean("R8 stdout", &out.stdout);
        s.assert_clean("R8 stderr", &out.stderr);
        s.assert_state_dir_clean("R8 state_dir", &state_dir);

        let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(json["error_code"], "upstream_config_invalid");
        let stderr_str = String::from_utf8_lossy(&out.stderr);
        assert!(stderr_str.contains(expected_rule), "stderr={stderr_str}");
        assert!(!stderr_str.contains("http://x"));
    }
}

#[test]
fn r8_non_utf8_upstream_env_refuses_startup_with_zero_leak() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let s = SentinelSet::generate();
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let mut non_utf8_value = s.c2.as_bytes().to_vec();
    non_utf8_value.extend_from_slice(b"http://evil.example/\xff\xfe");
    let expected_len = non_utf8_value.len();
    let os_value = OsStr::from_bytes(&non_utf8_value);

    for (tag, env_key, expected_prefix) in [
        (
            "provider_url",
            "PARAPET_OPENAI_BASE_URL",
            "PARAPET_OPENAI_BASE_URL: unparseable",
        ),
        (
            "fallback_url",
            "OPENAI_BASE_URL",
            "OPENAI_BASE_URL: unparseable",
        ),
        (
            "per_host_url",
            "PARAPET_API_OPENAI_COM_BASE_URL",
            "PARAPET_API_OPENAI_COM_BASE_URL: unparseable",
        ),
        (
            "extra_hosts",
            "PARAPET_EXTRA_HOSTS",
            "PARAPET_EXTRA_HOSTS entry #1: extra_host_invalid",
        ),
    ] {
        let state_dir = write_private_state_dir(tmp.path(), &format!("state-r8-nonutf8-{tag}"));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut cmd);
        let out = cmd
            .env(env_key, os_value)
            .args([
                "run",
                "--json",
                "--config",
                cfg_path.to_str().unwrap(),
                "--state-dir",
                state_dir.to_str().unwrap(),
                "--offline-fixture",
                "v0",
            ])
            .output()
            .expect("run parapet run --offline-fixture v0 with non-UTF-8 env");

        assert_eq!(out.status.code(), Some(4), "case={tag}");
        s.assert_clean(&format!("R8:nonutf8:{tag}:stdout"), &out.stdout);
        s.assert_clean(&format!("R8:nonutf8:{tag}:stderr"), &out.stderr);
        s.assert_state_dir_clean(&format!("R8:nonutf8:{tag}:state_dir"), &state_dir);

        for surface in [&out.stdout, &out.stderr] {
            assert!(
                !surface.contains(&0xff) && !surface.contains(&0xfe),
                "raw non-UTF-8 bytes leaked on {tag}"
            );
            let lossy = String::from_utf8_lossy(surface);
            assert!(
                !lossy.contains("evil.example") && !lossy.contains("http://evil"),
                "leaked URL substring on {tag}: {lossy}"
            );
        }

        let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(json["error_code"], "upstream_config_invalid");
        let mut json_strings = Vec::new();
        walk_json_strings(&json, &mut json_strings);
        for field in &json_strings {
            s.assert_clean(&format!("R8:nonutf8:{tag}:json_field"), field.as_bytes());
            assert!(!field.contains("evil.example"));
        }

        let expected_diag = format!("{expected_prefix} (value withheld, {expected_len} bytes)");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&expected_diag),
            "case={tag} expected '{expected_diag}' in stderr={stderr}"
        );
    }
}

#[tokio::test]
async fn r9_config_sentinel_sweep_across_all_real_binaries_and_surfaces() {
    let recorder = LoopbackRecorder::spawn_silent().await;
    let s = SentinelSet::generate();
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());

    let cases: Vec<(&str, &str, &str, &str)> = vec![
        ("c1", "PARAPET_OPENAI_BASE_URL", &s.c1, "userinfo_forbidden"),
        ("c2", "PARAPET_OPENAI_BASE_URL", &s.c2, "unparseable"),
        ("c3", "PARAPET_OPENAI_BASE_URL", &s.c3, "non_https_upstream"),
        ("c4", "PARAPET_OPENAI_BASE_URL", &s.c4, "query_forbidden"),
        ("c5", "PARAPET_OPENAI_BASE_URL", &s.c5, "fragment_forbidden"),
        ("c6", "PARAPET_OPENAI_BASE_URL", &s.c6, "unsupported_scheme"),
        ("c7", "PARAPET_OPENAI_BASE_URL", &s.c7, "unparseable"),
        ("c8", "OPENAI_BASE_URL", &s.c8, "non_https_upstream"),
        (
            "c9",
            "PARAPET_EXTRA_HOSTS",
            &s.c9,
            "PARAPET_EXTRA_HOSTS entry #2: extra_host_invalid",
        ),
        (
            "c10",
            "PARAPET_EXTRA_HOSTS",
            &s.c10,
            "PARAPET_EXTRA_HOSTS entry #1: extra_host_invalid",
        ),
    ];

    for (tag, env_key, env_val, expected_substr) in cases {
        // 1. parapet run --json
        let sd_run_json = write_private_state_dir(tmp.path(), &format!("sd-{tag}-run-json"));
        let mut cmd_run_json = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut cmd_run_json);
        let out_run_json = cmd_run_json
            .env(env_key, env_val)
            .args([
                "run",
                "--json",
                "--config",
                cfg_path.to_str().unwrap(),
                "--state-dir",
                sd_run_json.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(out_run_json.status.code(), Some(4), "case={tag} run --json");
        s.assert_clean(&format!("{tag}:run_json:stdout"), &out_run_json.stdout);
        s.assert_clean(&format!("{tag}:run_json:stderr"), &out_run_json.stderr);
        s.assert_state_dir_clean(&format!("{tag}:run_json:state_dir"), &sd_run_json);
        let parsed_run_json: serde_json::Value =
            serde_json::from_slice(&out_run_json.stdout).unwrap();
        let mut json_strings = Vec::new();
        walk_json_strings(&parsed_run_json, &mut json_strings);
        for js in &json_strings {
            s.assert_clean(&format!("{tag}:run_json:field"), js.as_bytes());
        }
        assert_eq!(parsed_run_json["error_code"], "upstream_config_invalid");
        assert!(
            String::from_utf8_lossy(&out_run_json.stderr).contains(expected_substr),
            "case={tag} stderr={}",
            String::from_utf8_lossy(&out_run_json.stderr)
        );

        // 2. parapet run (human)
        let sd_run_human = write_private_state_dir(tmp.path(), &format!("sd-{tag}-run-human"));
        let mut cmd_run_human = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut cmd_run_human);
        let out_run_human = cmd_run_human
            .env(env_key, env_val)
            .args([
                "run",
                "--config",
                cfg_path.to_str().unwrap(),
                "--state-dir",
                sd_run_human.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(out_run_human.status.code(), Some(4), "case={tag} run human");
        s.assert_clean(&format!("{tag}:run_human:stdout"), &out_run_human.stdout);
        s.assert_clean(&format!("{tag}:run_human:stderr"), &out_run_human.stderr);
        s.assert_state_dir_clean(&format!("{tag}:run_human:state_dir"), &sd_run_human);
        assert!(
            String::from_utf8_lossy(&out_run_human.stderr).contains(expected_substr),
            "case={tag} stderr={}",
            String::from_utf8_lossy(&out_run_human.stderr)
        );

        // 3. parapet accept --json
        let sd_accept = write_private_state_dir(tmp.path(), &format!("sd-{tag}-accept"));
        let mut cmd_accept = Command::new(env!("CARGO_BIN_EXE_parapet"));
        clear_all_upstream_env(&mut cmd_accept);
        let out_accept = cmd_accept
            .env(env_key, env_val)
            .args([
                "accept",
                "--json",
                "--state-dir",
                sd_accept.to_str().unwrap(),
                "--config",
                cfg_path.to_str().unwrap(),
                "--scenario",
                "allow_no_executor",
            ])
            .output()
            .unwrap();
        assert_eq!(
            out_accept.status.code(),
            Some(4),
            "case={tag} accept --json"
        );
        s.assert_clean(&format!("{tag}:accept:stdout"), &out_accept.stdout);
        s.assert_clean(&format!("{tag}:accept:stderr"), &out_accept.stderr);
        s.assert_state_dir_clean(&format!("{tag}:accept:state_dir"), &sd_accept);
        let parsed_accept: serde_json::Value = serde_json::from_slice(&out_accept.stdout).unwrap();
        let mut accept_strings = Vec::new();
        walk_json_strings(&parsed_accept, &mut accept_strings);
        for js in &accept_strings {
            s.assert_clean(&format!("{tag}:accept:field"), js.as_bytes());
        }
        assert_eq!(parsed_accept["error_code"], "upstream_config_invalid");
        assert!(
            String::from_utf8_lossy(&out_accept.stderr).contains(expected_substr),
            "case={tag} accept stderr={}",
            String::from_utf8_lossy(&out_accept.stderr)
        );

        // 4. parapet-engine with RUST_LOG=trace
        let sd_engine = write_private_state_dir(tmp.path(), &format!("sd-{tag}-engine"));
        let mut cmd_engine = Command::new(env!("CARGO_BIN_EXE_parapet-engine"));
        clear_all_upstream_env(&mut cmd_engine);
        let out_engine = cmd_engine
            .env("RUST_LOG", "trace")
            .env(env_key, env_val)
            .args([
                "--config",
                cfg_path.to_str().unwrap(),
                "--port",
                "0",
                "--state-dir",
                sd_engine.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(
            out_engine.status.code(),
            Some(1),
            "case={tag} parapet-engine"
        );
        s.assert_clean(&format!("{tag}:engine:stdout"), &out_engine.stdout);
        s.assert_clean(&format!("{tag}:engine:stderr"), &out_engine.stderr);
        s.assert_state_dir_clean(&format!("{tag}:engine:state_dir"), &sd_engine);
        assert!(
            String::from_utf8_lossy(&out_engine.stderr).contains(expected_substr),
            "case={tag} engine stderr={}",
            String::from_utf8_lossy(&out_engine.stderr)
        );
    }

    assert_eq!(recorder.count(), 0);
}

async fn send_raw_http_request(addr: SocketAddr, raw_req: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect(addr).await.expect("connect to engine");
    stream.write_all(raw_req).await.expect("write raw request");
    let mut resp = Vec::new();
    stream
        .read_to_end(&mut resp)
        .await
        .expect("read raw response");
    resp
}

#[tokio::test]
async fn r10_header_sentinel_sweep_on_running_engine_via_raw_tcp() {
    let recorder = LoopbackRecorder::spawn_silent().await;
    let s = SentinelSet::generate();
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = write_test_config(tmp.path());
    let state_dir = write_private_state_dir(tmp.path(), "state-r10");

    // Spawn real parapet-engine with RUST_LOG=trace, valid table (upstream env cleared),
    // and offline-fixture so startup requires no external network.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_parapet-engine"));
    clear_all_upstream_env(&mut cmd);
    let mut child = cmd
        .env("RUST_LOG", "trace")
        .args([
            "--config",
            cfg_path.to_str().unwrap(),
            "--port",
            "0",
            "--state-dir",
            state_dir.to_str().unwrap(),
            "--offline-fixture",
            "v0",
            "--ready-file",
            "ready.port",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn parapet-engine");

    let ready_path = state_dir.join("ready.port");
    let mut bound_port: Option<u16> = None;
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(&ready_path) {
            if let Ok(p) = text.trim().parse::<u16>() {
                bound_port = Some(p);
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let port = bound_port.expect("engine wrote ready.port");
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    let endpoints = [
        (
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}"#,
        ),
        (
            "/v1/messages",
            r#"{"model":"claude-3-5-sonnet-20241022","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}"#,
        ),
    ];

    for (path, body) in endpoints {
        // H1, H2, H3, H5 single header lines
        for (h_name, raw_val) in [("H1", &s.h1), ("H2", &s.h2), ("H3", &s.h3), ("H5", &s.h5)] {
            let mut raw = Vec::new();
            raw.extend_from_slice(
                format!(
                    "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nX-Parapet-Original-Host: "
                )
                .as_bytes(),
            );
            raw.extend_from_slice(raw_val);
            raw.extend_from_slice(
                format!(
                    "\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );

            let resp_bytes = send_raw_http_request(addr, &raw).await;
            let resp_str = String::from_utf8_lossy(&resp_bytes);
            assert!(
                resp_str.starts_with("HTTP/1.1 400 "),
                "{h_name} on {path} expected 400, got: {resp_str}"
            );
            assert!(resp_str.contains("parapet_upstream_host_not_allowed"));
            s.assert_clean(&format!("R10:{path}:{h_name}:response"), &resp_bytes);
        }

        // H4 duplicate header lines
        let mut raw_h4 = Vec::new();
        raw_h4.extend_from_slice(
            format!(
                "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nX-Parapet-Original-Host: api.openai.com\r\nX-Parapet-Original-Host: "
            )
            .as_bytes(),
        );
        raw_h4.extend_from_slice(&s.h4_second);
        raw_h4.extend_from_slice(
            format!(
                "\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        );
        let resp_h4 = send_raw_http_request(addr, &raw_h4).await;
        let resp_h4_str = String::from_utf8_lossy(&resp_h4);
        assert!(
            resp_h4_str.starts_with("HTTP/1.1 400 "),
            "H4 on {path} expected 400, got: {resp_h4_str}"
        );
        s.assert_clean(&format!("R10:{path}:H4:response"), &resp_h4);
    }

    let _ = child.kill();
    let out = child.wait_with_output().expect("wait engine");
    s.assert_clean("R10:engine:stdout", &out.stdout);
    s.assert_clean("R10:engine:stderr", &out.stderr);
    s.assert_state_dir_clean("R10:state_dir", &state_dir);
    assert_eq!(recorder.count(), 0);
}

#[tokio::test]
async fn r11_request_time_upstream_refusals_write_zero_receipt_rows() {
    let recorder_b = LoopbackRecorder::spawn_silent().await;
    let steal_url = format!("http://127.0.0.1:{}/steal", recorder_b.addr.port());
    let recorder_redirect = LoopbackRecorder::spawn_redirect_to(steal_url).await;

    let tmp = tempfile::tempdir().unwrap();
    let state_dir = write_private_state_dir(tmp.path(), "state-r11");
    let store_root = state_dir.join("receipt-evidence");
    let store = Arc::new(
        parapet::effect::store::ProtectedEvidenceStoreV0::open(&store_root).expect("open store"),
    );

    let cfg = Arc::new(
        load_config(&StringSource {
            content: MANDATORY_PROFILE_YAML.to_string(),
        })
        .unwrap(),
    );

    // Helper to build an envelope-backed EngineUpstreamClient sharing `store`
    let build_client = |resolver: Arc<dyn UpstreamResolver>, sender: Arc<dyn HttpSender>| {
        let parts = EnvelopePartsV0::new(
            cfg.clone(),
            store.clone(),
            Arc::new(InjectedBuildIdentityV0::unavailable()),
        );
        let envelope = Arc::new(RuntimeEnvelopeV0::new(parts));
        let deps = EngineDeps {
            config: cfg.clone(),
            http: sender,
            resolver,
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
        };
        EngineUpstreamClient::new_with_envelope(deps, envelope)
    };

    let valid_body =
        Bytes::from_static(br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#);

    // 1. Disallowed original-host (400)
    let valid_table = UpstreamTable::from_env_fn(&|_| None).unwrap();
    let client_host_refusal = build_client(
        Arc::new(EnvUpstreamResolver::from_table(valid_table)),
        Arc::new(ReqwestHttpSender::new(production_http_client())),
    );
    let mut bad_host_headers = HeaderMap::new();
    bad_host_headers.insert(
        "x-parapet-original-host",
        HeaderValue::from_static("evil.example.com"),
    );
    let resp_400 = client_host_refusal
        .forward(
            Provider::OpenAi,
            ProxyRequest {
                method: Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                headers: bad_host_headers,
                body: valid_body.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(resp_400.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        resp_400
            .headers
            .get("x-parapet-error")
            .and_then(|v| v.to_str().ok()),
        Some("parapet_upstream_host_not_allowed")
    );
    assert_eq!(store_receipt_entry_count(&store_root), 0);

    // 2. Config-invalid refusal through an injected resolver (502)
    let invalid_resolver = EnvUpstreamResolver::from_env_fn(&|k| match k {
        "PARAPET_OPENAI_BASE_URL" => Some("http://127.0.0.1:4000".to_string()),
        _ => None,
    });
    let client_cfg_refusal = build_client(
        Arc::new(invalid_resolver),
        Arc::new(ReqwestHttpSender::new(production_http_client())),
    );
    let resp_502_cfg = client_cfg_refusal
        .forward(
            Provider::OpenAi,
            ProxyRequest {
                method: Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                headers: HeaderMap::new(),
                body: valid_body.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(resp_502_cfg.status, StatusCode::BAD_GATEWAY);
    assert_eq!(
        resp_502_cfg
            .headers
            .get("x-parapet-error")
            .and_then(|v| v.to_str().ok()),
        Some("parapet_upstream_config_invalid")
    );
    let body_502_cfg = axum::body::to_bytes(resp_502_cfg.body, 1024 * 1024)
        .await
        .unwrap();
    let body_502_str = String::from_utf8_lossy(&body_502_cfg);
    assert!(body_502_str.contains("parapet_upstream_config_invalid"));
    assert!(body_502_str.contains("PARAPET_OPENAI_BASE_URL"));
    assert!(!body_502_str.contains("non_https_upstream"));
    assert!(!body_502_str.contains("4000"));
    assert_eq!(store_receipt_entry_count(&store_root), 0);

    // 3. Upstream redirect refusal (502)
    struct RedirectTestResolver {
        base: String,
    }
    impl UpstreamResolver for RedirectTestResolver {
        fn base_url(&self, _provider: Provider) -> String {
            self.base.clone()
        }
    }
    let client_redirect = build_client(
        Arc::new(RedirectTestResolver {
            base: format!("http://127.0.0.1:{}", recorder_redirect.addr.port()),
        }),
        Arc::new(ReqwestHttpSender::new(production_http_client())),
    );
    let resp_502_redir = client_redirect
        .forward(
            Provider::OpenAi,
            ProxyRequest {
                method: Method::POST,
                uri: "/v1/chat/completions".parse().unwrap(),
                headers: HeaderMap::new(),
                body: valid_body,
            },
        )
        .await
        .unwrap();
    assert_eq!(resp_502_redir.status, StatusCode::BAD_GATEWAY);
    assert_eq!(
        resp_502_redir
            .headers
            .get("x-parapet-error")
            .and_then(|v| v.to_str().ok()),
        Some("parapet_upstream_redirect_refused")
    );
    assert_eq!(recorder_b.count(), 0);
    assert_eq!(
        store_receipt_entry_count(&store_root),
        0,
        "upstream policy refusals must write zero receipt rows"
    );
}
