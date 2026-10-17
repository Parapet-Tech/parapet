// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Owner-only POSIX control channel (`parapet-control/v0`) and file-cleanup
//! shutdown gate (Brief Sections 6.1–6.3).

#![cfg(not(windows))]

use crate::effect::{FileCleanupResultV0, FileEffectCleanupV0, RootCapabilityV0};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Control wire protocol schema constant.
pub const CONTROL_SCHEMA: &str = "parapet-control/v0";
/// State directory control metadata schema constant.
pub const CONTROL_STATE_SCHEMA: &str = "parapet-control-state/v0";
/// Engine ownership metadata schema constant.
pub const ENGINE_OWNER_SCHEMA: &str = "parapet-engine-owner/v0";
/// Maximum line length accepted on `control.sock`.
pub const MAX_CONTROL_LINE_BYTES: usize = 4096;
/// Control request/response timeout (`5,000 ms`).
pub const CONTROL_TIMEOUT: Duration = Duration::from_millis(5000);

/// Persistent state metadata written by `parapet init` to `<state-dir>/control.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlStateMetadataV0 {
    pub schema: String,
    pub effect_root_must_be_new: bool,
}

/// Ownership record written by an owned engine to `<state-dir>/engine.owner.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineOwnerRecordV0 {
    pub schema: String,
    pub pid: u32,
    pub port: u16,
    pub config_hash: String,
    pub file_capability_active: bool,
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned: Option<bool>,
}

/// Control request message over `control.sock`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlRequestV0 {
    pub schema: String,
    pub token: String,
    pub action: String,
}

#[derive(Serialize)]
struct ControlErrorResponseV0<'a> {
    schema: &'static str,
    ok: bool,
    error_code: &'a str,
}

#[derive(Serialize)]
struct ControlStatusResponseV0<'a> {
    schema: &'static str,
    ok: bool,
    pid: u32,
    file_capability: &'a str,
    file_cleanup: &'a str,
    evidence_retained: bool,
}

#[derive(Serialize)]
struct ControlRestoreResponseV0<'a> {
    schema: &'static str,
    ok: bool,
    file_cleanup: &'a str,
    evidence_retained: bool,
}

#[derive(Serialize)]
struct ControlDisableResponseV0<'a> {
    schema: &'static str,
    ok: bool,
    stop_approved: bool,
    file_cleanup: &'a str,
    evidence_retained: bool,
}

/// Closed control errors (never carrying OS error text or filesystem paths).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlError {
    InvalidStateDir,
    UnsafePermissions,
    IoFailed,
    EngineUnavailable,
    AuthFailed,
    ProtocolError,
}

impl ControlError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidStateDir => "invalid_state_dir",
            Self::UnsafePermissions => "unsafe_state_dir_permissions",
            Self::IoFailed => "control_io_failed",
            Self::EngineUnavailable => "engine_unavailable",
            Self::AuthFailed => "control_auth_failed",
            Self::ProtocolError => "control_protocol_error",
        }
    }
}

/// Tracked in-memory file-cleanup state inside the live owning engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackedFileCleanupState {
    NotApplicable,
    NotRun,
    Settled(FileCleanupResultV0),
}

impl TrackedFileCleanupState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::NotRun => "not_run",
            Self::Settled(r) => r.as_str(),
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn generate_token_hex() -> String {
    format!(
        "{:032x}{:032x}",
        uuid::Uuid::new_v4().as_u128(),
        uuid::Uuid::new_v4().as_u128()
    )
}

/// Verify `<state_dir>` exists, is a non-symlink directory, has mode `0700`,
/// and is owned by the current process user.
pub fn verify_private_state_dir(state_dir: &Path) -> Result<u32, ControlError> {
    let meta = std::fs::symlink_metadata(state_dir).map_err(|_| ControlError::InvalidStateDir)?;
    if !meta.file_type().is_dir() || meta.file_type().is_symlink() {
        return Err(ControlError::UnsafePermissions);
    }
    if (meta.mode() & 0o7777) != 0o700 {
        return Err(ControlError::UnsafePermissions);
    }
    Ok(meta.uid())
}

/// Write a private `0600` file inside `state_dir` atomically via a temporary file.
pub fn write_private_file(state_dir: &Path, name: &str, bytes: &[u8]) -> Result<(), ControlError> {
    let expected_uid = verify_private_state_dir(state_dir)?;
    let tmp_name = format!(".tmp-{name}-{}", uuid::Uuid::new_v4().simple());
    let tmp_path = state_dir.join(&tmp_name);
    let final_path = state_dir.join(name);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp_path)
        .map_err(|_| ControlError::IoFailed)?;
    let meta = file.metadata().map_err(|_| ControlError::IoFailed)?;
    if meta.uid() != expected_uid || (meta.mode() & 0o7777) != 0o600 {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(ControlError::UnsafePermissions);
    }
    if file.write_all(bytes).is_err() || file.sync_all().is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(ControlError::IoFailed);
    }
    drop(file);
    if std::fs::rename(&tmp_path, &final_path).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(ControlError::IoFailed);
    }
    Ok(())
}

/// Read a private `0600` file from `state_dir`.
pub fn read_private_file(state_dir: &Path, name: &str) -> Result<Vec<u8>, ControlError> {
    let expected_uid = verify_private_state_dir(state_dir)?;
    let path = state_dir.join(name);
    let meta = std::fs::symlink_metadata(&path).map_err(|_| ControlError::EngineUnavailable)?;
    if !meta.file_type().is_file()
        || meta.file_type().is_symlink()
        || meta.uid() != expected_uid
        || (meta.mode() & 0o7777) != 0o600
    {
        return Err(ControlError::UnsafePermissions);
    }
    std::fs::read(&path).map_err(|_| ControlError::IoFailed)
}

/// Remove transient owned engine control files while preserving `control.json`
/// and `receipt-evidence`.
pub fn remove_owned_transient_files(state_dir: &Path) {
    for name in [
        "control.sock",
        "control.token",
        "engine.owner.json",
        "engine.pid",
        "engine.meta",
    ] {
        let _ = std::fs::remove_file(state_dir.join(name));
    }
}

/// Live control plane held inside the owning engine process.
pub struct EngineControlPlane {
    state_dir: PathBuf,
    owner_uid: u32,
    token: String,
    capability: Option<Arc<RootCapabilityV0>>,
    cleanup_state: Mutex<TrackedFileCleanupState>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

impl EngineControlPlane {
    /// Initialize `control.token`, `control.sock`, and `engine.owner.json` inside `state_dir`.
    pub fn bind(
        state_dir: &Path,
        port: u16,
        config_hash: &str,
        capability: Option<Arc<RootCapabilityV0>>,
    ) -> Result<
        (
            Arc<Self>,
            tokio::net::UnixListener,
            tokio::sync::watch::Receiver<bool>,
        ),
        ControlError,
    > {
        let expected_uid = verify_private_state_dir(state_dir)?;
        let token = generate_token_hex();
        write_private_file(state_dir, "control.token", token.as_bytes())?;

        let sock_path = state_dir.join("control.sock");
        if std::fs::symlink_metadata(&sock_path).is_ok() {
            let _ = std::fs::remove_file(&sock_path);
        }
        let std_listener = std::os::unix::net::UnixListener::bind(&sock_path)
            .map_err(|_| ControlError::IoFailed)?;
        std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o600))
            .map_err(|_| ControlError::IoFailed)?;
        let sock_meta =
            std::fs::symlink_metadata(&sock_path).map_err(|_| ControlError::IoFailed)?;
        if !sock_meta.file_type().is_socket()
            || sock_meta.uid() != expected_uid
            || (sock_meta.mode() & 0o7777) != 0o600
        {
            let _ = std::fs::remove_file(&sock_path);
            return Err(ControlError::UnsafePermissions);
        }
        std_listener
            .set_nonblocking(true)
            .map_err(|_| ControlError::IoFailed)?;
        let listener =
            tokio::net::UnixListener::from_std(std_listener).map_err(|_| ControlError::IoFailed)?;

        let normalized_hash = config_hash
            .strip_prefix("sha256:")
            .unwrap_or(config_hash)
            .to_owned();
        let owner_record = EngineOwnerRecordV0 {
            schema: ENGINE_OWNER_SCHEMA.to_owned(),
            pid: std::process::id(),
            port,
            config_hash: normalized_hash,
            file_capability_active: capability.is_some(),
            run_id: capability.as_ref().map(|c| c.run_id().as_str().to_owned()),
            owned: None,
        };
        let owner_bytes = serde_json::to_vec(&owner_record).map_err(|_| ControlError::IoFailed)?;
        write_private_file(state_dir, "engine.owner.json", &owner_bytes)?;

        let initial_state = if capability.is_some() {
            TrackedFileCleanupState::NotRun
        } else {
            TrackedFileCleanupState::NotApplicable
        };
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let plane = Arc::new(Self {
            state_dir: state_dir.to_path_buf(),
            owner_uid: expected_uid,
            token,
            capability,
            cleanup_state: Mutex::new(initial_state),
            shutdown_tx,
        });
        Ok((plane, listener, shutdown_rx))
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    pub fn tracked_cleanup_state(&self) -> TrackedFileCleanupState {
        *self.cleanup_state.lock().expect("cleanup state lock")
    }

    /// Observational stop gate: never invokes `FileEffectCleanupV0::cleanup()`.
    /// Returns `(stop_approved, file_cleanup_code)`.
    pub fn evaluate_stop_gate(&self) -> (bool, &'static str) {
        let state = self.tracked_cleanup_state();
        match (&self.capability, state) {
            (None, _) => (true, "not_applicable"),
            (Some(cap), TrackedFileCleanupState::Settled(FileCleanupResultV0::Clear))
                if cap.is_cleared() =>
            {
                (true, "clear")
            }
            (Some(_), TrackedFileCleanupState::Settled(FileCleanupResultV0::Clear)) => {
                (false, "not_run")
            }
            (Some(_), TrackedFileCleanupState::NotApplicable) => (false, "not_run"),
            (Some(_), other) => (false, other.as_str()),
        }
    }

    /// Execute authenticated `restore` on the retained capability.
    pub fn run_restore(&self) -> &'static str {
        let Some(ref cap) = self.capability else {
            return "not_applicable";
        };
        let cleanup = FileEffectCleanupV0::new(cap.clone());
        let result = cleanup.cleanup();
        let mut guard = self.cleanup_state.lock().expect("cleanup state lock");
        *guard = TrackedFileCleanupState::Settled(result);
        result.as_str()
    }

    /// Process a single JSON request line and return `(response_json_bytes, approve_shutdown)`.
    pub fn handle_request_line(&self, line: &str) -> (Vec<u8>, bool) {
        let parsed: Result<ControlRequestV0, _> = serde_json::from_str(line.trim());
        let Ok(req) = parsed else {
            let body = ControlErrorResponseV0 {
                schema: CONTROL_SCHEMA,
                ok: false,
                error_code: "invalid_request",
            };
            let mut out = serde_json::to_vec(&body).unwrap_or_default();
            out.push(b'\n');
            return (out, false);
        };
        if req.schema != CONTROL_SCHEMA
            || !constant_time_eq(req.token.as_bytes(), self.token.as_bytes())
        {
            let body = ControlErrorResponseV0 {
                schema: CONTROL_SCHEMA,
                ok: false,
                error_code: "unauthorized",
            };
            let mut out = serde_json::to_vec(&body).unwrap_or_default();
            out.push(b'\n');
            return (out, false);
        }

        let cap_status = if self.capability.is_some() {
            "created"
        } else {
            "not_applicable"
        };

        match req.action.as_str() {
            "status" => {
                let (_, cleanup_code) = self.evaluate_stop_gate();
                let body = ControlStatusResponseV0 {
                    schema: CONTROL_SCHEMA,
                    ok: true,
                    pid: std::process::id(),
                    file_capability: cap_status,
                    file_cleanup: cleanup_code,
                    evidence_retained: true,
                };
                let mut out = serde_json::to_vec(&body).unwrap_or_default();
                out.push(b'\n');
                (out, false)
            }
            "restore" => {
                let result_code = self.run_restore();
                let ok = matches!(result_code, "clear" | "not_applicable");
                let body = ControlRestoreResponseV0 {
                    schema: CONTROL_SCHEMA,
                    ok,
                    file_cleanup: result_code,
                    evidence_retained: true,
                };
                let mut out = serde_json::to_vec(&body).unwrap_or_default();
                out.push(b'\n');
                (out, false)
            }
            "disable" => {
                let (approved, cleanup_code) = self.evaluate_stop_gate();
                let body = ControlDisableResponseV0 {
                    schema: CONTROL_SCHEMA,
                    ok: approved,
                    stop_approved: approved,
                    file_cleanup: cleanup_code,
                    evidence_retained: true,
                };
                let mut out = serde_json::to_vec(&body).unwrap_or_default();
                out.push(b'\n');
                (out, approved)
            }
            _ => {
                let body = ControlErrorResponseV0 {
                    schema: CONTROL_SCHEMA,
                    ok: false,
                    error_code: "unknown_action",
                };
                let mut out = serde_json::to_vec(&body).unwrap_or_default();
                out.push(b'\n');
                (out, false)
            }
        }
    }

    /// Spawn the background Unix socket accept loop.
    pub fn spawn_listener(self: &Arc<Self>, listener: tokio::net::UnixListener) {
        let plane = self.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let plane_conn = plane.clone();
                tokio::spawn(async move {
                    if let Ok(cred) = stream.peer_cred() {
                        if cred.uid() != plane_conn.owner_uid {
                            return;
                        }
                    }
                    let (reader, mut writer) = stream.split();
                    let mut buf_reader = tokio::io::BufReader::new(reader);
                    let mut line = String::new();
                    let read_res =
                        tokio::time::timeout(CONTROL_TIMEOUT, buf_reader.read_line(&mut line))
                            .await;
                    let Ok(Ok(n)) = read_res else {
                        return;
                    };
                    if n == 0 || line.len() > MAX_CONTROL_LINE_BYTES {
                        return;
                    }
                    let (resp, approve_shutdown) = plane_conn.handle_request_line(&line);
                    let _ = writer.write_all(&resp).await;
                    let _ = writer.flush().await;
                    let _ = writer.shutdown().await;
                    if approve_shutdown {
                        let _ = plane_conn.shutdown_tx.send(true);
                    }
                });
            }
        });
    }

    /// Spawn `SIGINT` and `SIGTERM` handlers bound to the same observational
    /// file-cleanup gate as authenticated `disable` (Section 6.3.1).
    pub fn spawn_signal_gate(self: &Arc<Self>) {
        use tokio::signal::unix::{signal, SignalKind};
        let Ok(mut sigint) = signal(SignalKind::interrupt()) else {
            return;
        };
        let Ok(mut sigterm) = signal(SignalKind::terminate()) else {
            return;
        };
        let plane = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = sigint.recv() => {}
                    _ = sigterm.recv() => {}
                }
                let (approved, cleanup_code) = plane.evaluate_stop_gate();
                if approved {
                    let _ = plane.shutdown_tx.send(true);
                    break;
                }
                tracing::warn!(
                    schema = "parapet-file-cleanup/v0",
                    result = cleanup_code,
                    "signal shutdown refused until file cleanup is clear"
                );
            }
        });
    }
}

/// Ownership classification of `<state-dir>` for `parapet disable`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipCheck {
    Owned(EngineOwnerRecordV0),
    NotOwned,
    Unavailable,
}

fn is_pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if pid == std::process::id() {
        return true;
    }
    std::process::Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn is_port_listening(port: u16) -> bool {
    if port == 0 {
        return false;
    }
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok()
}

/// Validate `engine.owner.json` against `<state_dir>` metadata and live process state
/// before contacting `control.sock` (Section 5.2.9 and `AC-1D-21`).
pub fn check_state_dir_ownership(state_dir: &Path) -> OwnershipCheck {
    if verify_private_state_dir(state_dir).is_err() {
        return OwnershipCheck::Unavailable;
    }
    let owner_path = state_dir.join("engine.owner.json");
    let pid_file_val = std::fs::read_to_string(state_dir.join("engine.pid"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let meta_val = std::fs::read_to_string(state_dir.join("engine.meta"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let meta_port_val = meta_val
        .as_ref()
        .and_then(|v| v.get("port").and_then(Value::as_u64))
        .and_then(|p| u16::try_from(p).ok());
    let meta_pid_val = meta_val
        .as_ref()
        .and_then(|v| v.get("pid").and_then(Value::as_u64))
        .and_then(|p| u32::try_from(p).ok());
    let meta_owned_val = meta_val
        .as_ref()
        .and_then(|v| v.get("owned").and_then(Value::as_bool));
    let meta_hash_val = meta_val
        .as_ref()
        .and_then(|v| {
            v.get("configHash")
                .or_else(|| v.get("config_hash"))
                .and_then(Value::as_str)
        })
        .map(|s| s.strip_prefix("sha256:").unwrap_or(s).to_owned());

    if std::fs::symlink_metadata(&owner_path).is_err() {
        if meta_owned_val == Some(false)
            || pid_file_val.is_some_and(is_pid_alive)
            || meta_pid_val.is_some_and(is_pid_alive)
            || meta_port_val.is_some_and(is_port_listening)
        {
            return OwnershipCheck::NotOwned;
        }
        return OwnershipCheck::Unavailable;
    }

    let Ok(owner_bytes) = read_private_file(state_dir, "engine.owner.json") else {
        return OwnershipCheck::Unavailable;
    };
    let Ok(owner) = serde_json::from_slice::<EngineOwnerRecordV0>(&owner_bytes) else {
        return OwnershipCheck::Unavailable;
    };
    if owner.schema != ENGINE_OWNER_SCHEMA {
        return OwnershipCheck::Unavailable;
    }
    if owner.owned == Some(false) || meta_owned_val == Some(false) {
        return OwnershipCheck::NotOwned;
    }
    if let Some(pid_from_file) = pid_file_val {
        if pid_from_file != owner.pid && (is_pid_alive(pid_from_file) || is_pid_alive(owner.pid)) {
            return OwnershipCheck::NotOwned;
        }
    }
    if let Some(pid_from_meta) = meta_pid_val {
        if pid_from_meta != owner.pid && (is_pid_alive(pid_from_meta) || is_pid_alive(owner.pid)) {
            return OwnershipCheck::NotOwned;
        }
    }
    if let Some(port_from_meta) = meta_port_val {
        if port_from_meta != owner.port
            && (is_port_listening(port_from_meta) || is_pid_alive(owner.pid))
        {
            return OwnershipCheck::NotOwned;
        }
    }
    if let Some(ref hash_from_meta) = meta_hash_val {
        let normalized_owner_hash = owner
            .config_hash
            .strip_prefix("sha256:")
            .unwrap_or(&owner.config_hash);
        if hash_from_meta != normalized_owner_hash && is_pid_alive(owner.pid) {
            return OwnershipCheck::NotOwned;
        }
    }
    let has_control = std::fs::symlink_metadata(state_dir.join("control.sock")).is_ok()
        && std::fs::symlink_metadata(state_dir.join("control.token")).is_ok();
    if !has_control {
        if is_pid_alive(owner.pid) || is_port_listening(owner.port) {
            return OwnershipCheck::NotOwned;
        }
        return OwnershipCheck::Unavailable;
    }
    if !is_pid_alive(owner.pid) {
        return OwnershipCheck::Unavailable;
    }
    OwnershipCheck::Owned(owner)
}

/// Send an authenticated control request over `<state_dir>/control.sock` with a 5,000 ms timeout.
pub fn send_control_request(state_dir: &Path, action: &str) -> Result<Value, ControlError> {
    let expected_uid = verify_private_state_dir(state_dir)?;
    let token_bytes = read_private_file(state_dir, "control.token")?;
    let token = String::from_utf8(token_bytes)
        .map_err(|_| ControlError::ProtocolError)?
        .trim()
        .to_owned();
    let sock_path = state_dir.join("control.sock");
    let sock_meta =
        std::fs::symlink_metadata(&sock_path).map_err(|_| ControlError::EngineUnavailable)?;
    if !sock_meta.file_type().is_socket()
        || sock_meta.uid() != expected_uid
        || (sock_meta.mode() & 0o7777) != 0o600
    {
        return Err(ControlError::UnsafePermissions);
    }

    let mut stream =
        UnixStream::connect(&sock_path).map_err(|_| ControlError::EngineUnavailable)?;
    let _ = stream.set_read_timeout(Some(CONTROL_TIMEOUT));
    let _ = stream.set_write_timeout(Some(CONTROL_TIMEOUT));

    let req = ControlRequestV0 {
        schema: CONTROL_SCHEMA.to_owned(),
        token,
        action: action.to_owned(),
    };
    let mut payload = serde_json::to_vec(&req).map_err(|_| ControlError::ProtocolError)?;
    payload.push(b'\n');
    stream
        .write_all(&payload)
        .map_err(|_| ControlError::EngineUnavailable)?;
    stream
        .flush()
        .map_err(|_| ControlError::EngineUnavailable)?;

    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    let read_bytes = reader
        .read_line(&mut resp_line)
        .map_err(|_| ControlError::EngineUnavailable)?;
    if read_bytes == 0 || resp_line.len() > MAX_CONTROL_LINE_BYTES {
        return Err(ControlError::EngineUnavailable);
    }
    let val: Value =
        serde_json::from_str(resp_line.trim()).map_err(|_| ControlError::ProtocolError)?;
    if val.get("schema").and_then(Value::as_str) != Some(CONTROL_SCHEMA) {
        return Err(ControlError::ProtocolError);
    }
    if val.get("error_code").and_then(Value::as_str) == Some("unauthorized") {
        return Err(ControlError::AuthFailed);
    }
    Ok(val)
}

/// Wait up to `timeout` for the owned engine process `pid` to exit after an approved `disable`.
pub fn wait_for_pid_exit(pid: u32, timeout: Duration) -> bool {
    if pid == std::process::id() {
        return true;
    }
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !is_pid_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    !is_pid_alive(pid)
}

/// Wait up to `timeout` for the owned engine in `state_dir` (`pid`) to shut down
/// and remove its transient control files after an approved `disable`.
pub fn wait_for_owned_shutdown(state_dir: &Path, pid: u32, timeout: Duration) -> bool {
    if pid == std::process::id() {
        return true;
    }
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !is_pid_alive(pid)
            || (std::fs::symlink_metadata(state_dir.join("control.sock")).is_err()
                && std::fs::symlink_metadata(state_dir.join("engine.owner.json")).is_err())
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    !is_pid_alive(pid)
        || (std::fs::symlink_metadata(state_dir.join("control.sock")).is_err()
            && std::fs::symlink_metadata(state_dir.join("engine.owner.json")).is_err())
}
