// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Parapet 1D CLI command surface (`parapet`), schemas, and lifecycle handlers
//! (Frozen 1D Brief `81621f3c`, Sections 5–7).

#[cfg(not(windows))]
pub mod control;
pub mod render;

use crate::config;
use crate::layers::l1_detection::{
    self, AttributionStatus, DetectionOutcome, RecordMode, ScanError,
};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use std::io::{Read, Write};
use std::path::PathBuf;

#[cfg(not(windows))]
use crate::cli::control::{
    check_state_dir_ownership, remove_owned_transient_files, send_control_request,
    verify_private_state_dir, wait_for_owned_shutdown, write_private_file, ControlStateMetadataV0,
    EngineControlPlane, OwnershipCheck, CONTROL_STATE_SCHEMA, CONTROL_TIMEOUT,
};
#[cfg(not(windows))]
use crate::cli::render::{
    availability_code, enforcement_code, enforcement_label, evidence_cleanup_result_label,
    file_cleanup_lifecycle_label, projection_verification_label, render_inspect_human,
    splice_inspect_json, store_failure_code, store_failure_label, summary_code, summary_label,
    verification_error_code, verification_outcome_code,
};
#[cfg(not(windows))]
use crate::effect::store::{
    cleanup::CleanupRefusalV0, CleanupResultV0, ProtectedEvidenceStoreV0, RedactedEvidenceReaderV0,
    TerminalOutcomeV0,
};
#[cfg(not(windows))]
use crate::effect::{EnvelopeResponseResultV0, EnvelopeResultSinkV0, FileEffectProfileV0, RunIdV0};
#[cfg(not(windows))]
use crate::engine::{
    build_engine_client_with_runtime_v0, EngineRuntimeOptionsV0, OfflineUpstreamFixtureV0,
};
#[cfg(not(windows))]
use crate::receipt::receipt_projection;

const SENSITIVE_RECORD_JSON_HELP: &str = "SENSITIVE EVIDENCE EXPORT: emits exact canonical l1-detection-record/v0 bytes for caller-supplied input, which may contain matched feature substrings and character spans. Never use on secret or regulated input without appropriate custody.";

#[derive(Parser)]
#[command(
    name = "parapet",
    version = env!("CARGO_PKG_VERSION"),
    about = "Parapet local CLI and protected evidence lifecycle adapter"
)]
struct Cli {
    /// Emit machine-readable JSON on stdout
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print CLI/engine version and L1 model identity
    Info,
    /// Scan caller-supplied text with L1 detection
    Scan(ScanArgs),
    /// Initialize a private state directory for receipt evidence and engine control
    Init(InitArgs),
    /// Run the local Parapet proxy engine inside an initialized state directory
    Run(RunArgs),
    /// Run a deterministic offline Rust CLI acceptance scenario across the full pipeline
    Accept(AcceptArgs),
    /// Inspect a run's redacted receipt projection and authority status
    Inspect(InspectArgs),
    /// Execute idempotent file-effect cleanup through the live owning engine
    Restore(RestoreArgs),
    /// Protected evidence store operations
    Evidence(EvidenceArgs),
    /// Stop the engine owned by a state directory after file-cleanup verification
    Disable(DisableArgs),
}

#[derive(Args)]
struct ScanArgs {
    /// Read UTF-8 input from file path (mutually exclusive with --stdin)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Read UTF-8 input from standard input (mutually exclusive with --input)
    #[arg(long)]
    stdin: bool,

    /// Optional parapet.yaml configuration path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Maximum input length in bytes (default: 1048576)
    #[arg(long, default_value_t = 1_048_576)]
    max_bytes: usize,

    /// Disable L0 normalization prior to L1 scan
    #[arg(long)]
    no_normalize: bool,

    #[arg(long, help = SENSITIVE_RECORD_JSON_HELP)]
    record_json: bool,
}

#[derive(Args)]
struct InitArgs {
    /// Disposable private state directory path
    #[arg(long)]
    state_dir: PathBuf,

    /// Verify that a future effect root path does not yet exist
    #[arg(long)]
    effect_root: Option<PathBuf>,
}

#[derive(Args)]
struct RunArgs {
    /// Initialized state directory path
    #[arg(long)]
    state_dir: PathBuf,

    /// Path to parapet.yaml configuration file
    #[arg(long)]
    config: PathBuf,

    /// Port to listen on (0 for ephemeral port)
    #[arg(long, default_value_t = 9800)]
    port: u16,

    /// Optional new effect root directory for 1C write_file
    #[arg(long)]
    effect_root: Option<PathBuf>,

    /// Explicit 1C file-effect profile selector (v0)
    #[arg(long)]
    file_effect_profile: Option<String>,

    /// Maximum content length in bytes (1..=1048576) for 1C write_file
    #[arg(long)]
    max_content_bytes: Option<u64>,

    /// Optional deterministic offline upstream fixture selector (v0)
    #[arg(long)]
    offline_fixture: Option<String>,

    /// Optional relative filename inside state-dir written when ready
    #[arg(long)]
    ready_file: Option<String>,
}

#[derive(Args)]
struct AcceptArgs {
    /// Disposable state directory path
    #[arg(long)]
    state_dir: Option<PathBuf>,

    /// Exact scenario config (parapet/tests/fixtures/cli_v0/accept_*_v0.yaml bytes)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Acceptance scenario (allow_no_executor | allow_verified_write | block_no_effect | shadow_parse_error_l3_off)
    #[arg(long)]
    scenario: Option<String>,

    /// New effect root path (required for allow_verified_write and block_no_effect; forbidden otherwise)
    #[arg(long)]
    effect_root: Option<PathBuf>,
}

#[derive(Args)]
struct InspectArgs {
    /// Initialized state directory path
    #[arg(long)]
    state_dir: PathBuf,

    /// 32-character lowercase hex run identifier
    #[arg(long)]
    run_id: String,

    /// Optional claimed terminal outcome to corroborate (complete | storage_incomplete)
    #[arg(long)]
    claimed_outcome: Option<String>,

    /// Omit target.text and display only digest, len_bytes, and truncated
    #[arg(long)]
    inert_fallback: bool,
}

#[derive(Args)]
struct RestoreArgs {
    /// Initialized state directory path
    #[arg(long)]
    state_dir: PathBuf,
}

#[derive(Args)]
struct EvidenceArgs {
    #[command(subcommand)]
    command: EvidenceCommands,
}

#[derive(Subcommand)]
enum EvidenceCommands {
    /// Release protected store objects for a run while retaining control evidence
    Release(EvidenceReleaseArgs),
}

#[derive(Args)]
struct EvidenceReleaseArgs {
    /// Initialized state directory path
    #[arg(long)]
    state_dir: PathBuf,

    /// 32-character lowercase hex run identifier
    #[arg(long)]
    run_id: String,
}

#[derive(Args)]
struct DisableArgs {
    /// Initialized state directory path
    #[arg(long)]
    state_dir: PathBuf,
}

#[derive(serde::Serialize)]
struct CliErrorJsonV0<'a> {
    schema: &'static str,
    command: &'a str,
    error_code: &'a str,
    exit_code: i32,
}

#[derive(serde::Serialize)]
struct CliInfoJsonV0<'a> {
    schema: &'static str,
    cli_version: &'a str,
    engine_version: &'a str,
    l1_model: l1_detection::ModelIdentity,
}

#[derive(serde::Serialize)]
struct CliScanJsonV0<'a> {
    schema: &'static str,
    record_schema_version: &'a str,
    record_digest: &'a str,
    model_id: &'a str,
    model_digest: &'a str,
    mode: &'a str,
    detection_outcome: &'a str,
    effective_raw: f64,
    threshold: f64,
    input_digest: &'a str,
    input_len_bytes: usize,
    attribution_status: &'a str,
    feature_count_matched: Option<usize>,
}

#[derive(serde::Serialize)]
struct CliInitJsonV0 {
    schema: &'static str,
    state_status: &'static str,
    owned_control_entries: [&'static str; 2],
    effect_root_policy: &'static str,
    file_capability: &'static str,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliRunJsonV0<'a> {
    schema: &'static str,
    port: u16,
    run_id: Option<&'a str>,
    file_capability: &'static str,
    file_cleanup: &'static str,
    file_cleanup_label: &'static str,
    evidence_retained: bool,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliAcceptJsonV0<'a> {
    schema: &'static str,
    scenario: &'a str,
    requester: &'static str,
    requester_version: &'static str,
    run_id: &'a str,
    config_contract_hash: &'a str,
    real_model_contacted: bool,
    upstream_label: &'static str,
    http_status: u16,
    summary: &'static str,
    summary_label: &'static str,
    enforcement: &'static str,
    enforcement_label: &'static str,
    verification_availability: &'static str,
    verification_outcome: Option<&'static str>,
    verification_error_code: Option<&'static str>,
    verification_label: &'a str,
    replay_verdict: Option<&'static str>,
    replay_reason: Option<&'static str>,
    replay_label: Option<&'static str>,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliAcceptConfigMismatchJsonV0<'a> {
    schema: &'static str,
    command: &'static str,
    error_code: &'static str,
    exit_code: i32,
    required_config: &'static str,
    required_config_contract_hash: &'static str,
    observed_config_contract_hash: &'a str,
}

/// Machine requester identity of `parapet accept`: the installed Rust CLI.
#[cfg(not(windows))]
const ACCEPT_REQUESTER: &str = "rust_cli";

/// Immutable acceptance scenario configs (Rev 9 D5): repository-relative public
/// fixture path and exact `Config.contract_hash`.
#[cfg(not(windows))]
const ACCEPT_ENFORCED_CONFIG: (&str, &str) = (
    "parapet/tests/fixtures/cli_v0/accept_enforced_v0.yaml",
    "sha256:44628d6064ca4ebc86b79883b7796df84fdcaf11a68c459145d6d1b43bc39d2d",
);
#[cfg(not(windows))]
const ACCEPT_SHADOW_L3_OFF_CONFIG: (&str, &str) = (
    "parapet/tests/fixtures/cli_v0/accept_shadow_l3_off_v0.yaml",
    "sha256:211c6e9f0501273e39220a478e64154dea277367f1c7c66268b5ce67e7a7feed",
);

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliRestoreJsonV0<'a> {
    schema: &'static str,
    result: &'a str,
    label: &'static str,
    a6_booking_required: bool,
    evidence_retained: bool,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliEvidenceReleaseJsonV0<'a> {
    schema: &'static str,
    run_id: &'a str,
    store_open: &'static str,
    store_failure_code: Option<&'static str>,
    result: Option<&'static str>,
    label: &'static str,
    control_evidence_retained: bool,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliDisableJsonV0 {
    schema: &'static str,
    status: &'static str,
    refusal_code: Option<&'static str>,
    file_cleanup: Option<&'static str>,
    file_cleanup_label: Option<&'static str>,
    ownership_label: &'static str,
    owned_control_removed: bool,
    evidence_retained: bool,
}

fn has_parent_dir_component(path: &std::path::Path) -> bool {
    path.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
}

fn emit_json_line<T: serde::Serialize + ?Sized>(val: &T) {
    if let Ok(mut bytes) = serde_json::to_vec(val) {
        bytes.push(b'\n');
        let _ = std::io::stdout().write_all(&bytes);
        let _ = std::io::stdout().flush();
    }
}

fn emit_cli_error(
    command: &str,
    error_code: &str,
    exit_code: i32,
    human_msg: &str,
    json: bool,
) -> i32 {
    eprintln!("parapet {command}: error: {human_msg}");
    if json {
        let body = CliErrorJsonV0 {
            schema: "parapet-cli-error/v0",
            command,
            error_code,
            exit_code,
        };
        emit_json_line(&body);
    }
    exit_code
}

fn detect_subcommand(raw_args: &[String]) -> &'static str {
    for arg in raw_args.iter().skip(1) {
        match arg.as_str() {
            "info" => return "info",
            "scan" => return "scan",
            "init" => return "init",
            "run" => return "run",
            "accept" => return "accept",
            "inspect" => return "inspect",
            "restore" => return "restore",
            "evidence" => {
                if raw_args.iter().skip(1).any(|a| a == "release") {
                    return "evidence release";
                }
                return "evidence";
            }
            "disable" => return "disable",
            _ => {}
        }
    }
    "cli"
}

/// Entrypoint called by `src/bin/parapet.rs`.
pub fn run_from_env() -> i32 {
    let raw_args: Vec<String> = std::env::args().collect();
    let wants_json = raw_args.iter().any(|a| a == "--json");
    let wants_record_json = raw_args.iter().any(|a| a == "--record-json");

    let cli = match Cli::try_parse_from(&raw_args) {
        Ok(c) => c,
        Err(err) => {
            use clap::error::ErrorKind;
            if matches!(
                err.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) {
                let _ = err.print();
                return 0;
            }
            let subcmd = detect_subcommand(&raw_args);
            if subcmd == "accept" {
                return emit_cli_error(
                    "accept",
                    "invalid_accept_combination",
                    2,
                    "invalid or missing accept arguments",
                    wants_json,
                );
            }
            return emit_cli_error(
                subcmd,
                "invalid_arguments",
                2,
                "invalid command-line arguments",
                wants_json,
            );
        }
    };

    if wants_json && wants_record_json {
        return emit_cli_error(
            "scan",
            "conflicting_output_flags",
            2,
            "--json and --record-json are mutually exclusive",
            true,
        );
    }

    let json_mode = cli.json;
    match cli.command {
        Commands::Info => run_info(json_mode),
        Commands::Scan(args) => run_scan(args, json_mode),
        Commands::Init(args) => run_init(args, json_mode),
        Commands::Run(args) => run_engine_cmd(args, json_mode),
        Commands::Accept(args) => run_accept(args, json_mode),
        Commands::Inspect(args) => run_inspect(args, json_mode),
        Commands::Restore(args) => run_restore(args, json_mode),
        Commands::Evidence(ev) => match ev.command {
            EvidenceCommands::Release(args) => run_evidence_release(args, json_mode),
        },
        Commands::Disable(args) => run_disable(args, json_mode),
    }
}

fn run_info(json_mode: bool) -> i32 {
    let identity = l1_detection::model_identity();
    let pkg_version = env!("CARGO_PKG_VERSION");
    if json_mode {
        let out = CliInfoJsonV0 {
            schema: "parapet-cli-info/v0",
            cli_version: pkg_version,
            engine_version: pkg_version,
            l1_model: identity,
        };
        emit_json_line(&out);
    } else {
        println!("parapet {pkg_version} (engine {pkg_version})");
        println!("l1_model.schema_version: {}", identity.schema_version);
        println!("l1_model.model_id: {}", identity.model_id);
        println!("l1_model.model_digest: {}", identity.model_digest);
        println!("l1_model.code_version: {}", identity.code_version);
    }
    0
}

fn run_scan(args: ScanArgs, json_mode: bool) -> i32 {
    if json_mode && args.record_json {
        return emit_cli_error(
            "scan",
            "conflicting_output_flags",
            2,
            "--json and --record-json are mutually exclusive",
            true,
        );
    }
    match (&args.input, args.stdin) {
        (Some(_), true) | (None, false) => {
            return emit_cli_error(
                "scan",
                "invalid_scan_input_source",
                2,
                "exactly one of --input <path> or --stdin must be specified",
                json_mode,
            );
        }
        _ => {}
    }
    if args.max_bytes == 0 {
        return emit_cli_error(
            "scan",
            "invalid_max_bytes",
            2,
            "--max-bytes must be at least 1",
            json_mode,
        );
    }

    let limit = (args.max_bytes as u64).saturating_add(1);
    let mut raw_bytes = Vec::new();
    let read_res = if let Some(ref path) = args.input {
        match std::fs::File::open(path) {
            Ok(file) => file.take(limit).read_to_end(&mut raw_bytes),
            Err(_) => {
                return emit_cli_error(
                    "scan",
                    "input_read_failed",
                    3,
                    "unable to read input file",
                    json_mode,
                );
            }
        }
    } else {
        std::io::stdin()
            .lock()
            .take(limit)
            .read_to_end(&mut raw_bytes)
    };
    if read_res.is_err() {
        return emit_cli_error(
            "scan",
            "input_read_failed",
            3,
            "unable to read input stream",
            json_mode,
        );
    }
    if raw_bytes.len() > args.max_bytes {
        return emit_cli_error(
            "scan",
            "input_too_large",
            3,
            "input exceeds --max-bytes limit",
            json_mode,
        );
    }
    let Ok(text) = String::from_utf8(raw_bytes) else {
        return emit_cli_error(
            "scan",
            "invalid_utf8_input",
            3,
            "input is not valid UTF-8",
            json_mode,
        );
    };

    let loaded_config = if let Some(path) = args.config {
        config::load_config(&config::FileSource { path })
    } else {
        config::load_config(&config::StringSource {
            content: "parapet: v1\n".to_string(),
        })
    };
    let Ok(cfg) = loaded_config else {
        return emit_cli_error(
            "scan",
            "invalid_config",
            3,
            "failed to load configuration",
            json_mode,
        );
    };
    let Some(l1_cfg) = cfg.policy.layers.l1.as_ref() else {
        return emit_cli_error(
            "scan",
            "invalid_config",
            3,
            "L1 configuration is not enabled",
            json_mode,
        );
    };

    let record = match l1_detection::scan_text(&text, l1_cfg, !args.no_normalize) {
        Ok(r) => r,
        Err(ScanError::EmptyInput) => {
            return emit_cli_error(
                "scan",
                "empty_input_after_normalization",
                3,
                "input is empty after normalization",
                json_mode,
            );
        }
        Err(ScanError::NonFinite) => {
            return emit_cli_error(
                "scan",
                "non_finite_score",
                3,
                "scanner produced a non-finite score",
                json_mode,
            );
        }
    };

    if args.record_json {
        let Ok(bytes) = l1_detection::record_bytes(&record) else {
            return emit_cli_error(
                "scan",
                "record_serialization_failed",
                3,
                "failed to serialize detection record",
                false,
            );
        };
        let _ = std::io::stdout().write_all(&bytes);
        let _ = std::io::stdout().flush();
        return 0;
    }

    let Ok(digest) = l1_detection::record_digest(&record) else {
        return emit_cli_error(
            "scan",
            "record_serialization_failed",
            3,
            "failed to compute detection record digest",
            json_mode,
        );
    };

    let mode_str = match record.mode {
        RecordMode::Shadow => "shadow",
        RecordMode::Block => "block",
    };
    let outcome_str = match record.detection_outcome {
        DetectionOutcome::ThresholdBreached => "threshold_breached",
        DetectionOutcome::ThresholdNotBreached => "threshold_not_breached",
    };
    let attr_str = match record.attribution_status {
        AttributionStatus::Present => "present",
        AttributionStatus::AbsentByContract => "absent_by_contract",
        AttributionStatus::Errored => "errored",
    };

    if json_mode {
        let out = CliScanJsonV0 {
            schema: "parapet-cli-scan/v0",
            record_schema_version: &record.schema_version,
            record_digest: &digest,
            model_id: &record.model_id,
            model_digest: &record.model_digest,
            mode: mode_str,
            detection_outcome: outcome_str,
            effective_raw: record.effective_raw,
            threshold: record.threshold,
            input_digest: &record.input_digest,
            input_len_bytes: record.input_len_bytes,
            attribution_status: attr_str,
            feature_count_matched: record.feature_count_matched,
        };
        emit_json_line(&out);
    } else {
        println!("record_schema_version: {}", record.schema_version);
        println!("record_digest: {digest}");
        println!("model_id: {}", record.model_id);
        println!("model_digest: {}", record.model_digest);
        println!("mode: {mode_str}");
        println!("detection_outcome: {outcome_str}");
        println!("effective_raw: {}", record.effective_raw);
        println!("threshold: {}", record.threshold);
        println!("input_digest: {}", record.input_digest);
        println!("input_len_bytes: {}", record.input_len_bytes);
        println!("attribution_status: {attr_str}");
        match record.feature_count_matched {
            Some(n) => println!("feature_count_matched: {n}"),
            None => println!("feature_count_matched: none"),
        }
    }
    0
}

#[cfg(windows)]
fn run_init(_args: InitArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "init",
        "unsupported_platform",
        6,
        "init is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_init(args: InitArgs, json_mode: bool) -> i32 {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    if has_parent_dir_component(&args.state_dir)
        || args
            .effect_root
            .as_deref()
            .is_some_and(has_parent_dir_component)
    {
        return emit_cli_error(
            "init",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    if let Some(ref er) = args.effect_root {
        if std::fs::symlink_metadata(er).is_ok() {
            return emit_cli_error(
                "init",
                "effect_root_already_exists",
                2,
                "effect-root must not exist before engine startup",
                json_mode,
            );
        }
    }

    if std::fs::symlink_metadata(&args.state_dir).is_err() {
        if std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&args.state_dir)
            .is_err()
        {
            return emit_cli_error(
                "init",
                "state_dir_create_failed",
                4,
                "unable to create private state directory",
                json_mode,
            );
        }
        let _ = std::fs::set_permissions(&args.state_dir, std::fs::Permissions::from_mode(0o700));
    }

    if verify_private_state_dir(&args.state_dir).is_err() {
        return emit_cli_error(
            "init",
            "unsafe_state_dir",
            4,
            "state-dir must be a private 0700 directory owned by current user",
            json_mode,
        );
    }

    let store_path = args.state_dir.join("receipt-evidence");
    if let Err(sf) = ProtectedEvidenceStoreV0::open(&store_path) {
        return emit_cli_error(
            "init",
            store_failure_code(sf),
            4,
            store_failure_label(sf),
            json_mode,
        );
    }

    let meta = ControlStateMetadataV0 {
        schema: CONTROL_STATE_SCHEMA.to_owned(),
        effect_root_must_be_new: true,
    };
    let Ok(meta_bytes) = serde_json::to_vec(&meta) else {
        return emit_cli_error(
            "init",
            "control_write_failed",
            4,
            "failed to serialize control metadata",
            json_mode,
        );
    };
    if write_private_file(&args.state_dir, "control.json", &meta_bytes).is_err() {
        return emit_cli_error(
            "init",
            "control_write_failed",
            4,
            "failed to write control.json",
            json_mode,
        );
    }

    if json_mode {
        let out = CliInitJsonV0 {
            schema: "parapet-cli-init/v0",
            state_status: "initialized",
            owned_control_entries: ["control.json", "receipt-evidence"],
            effect_root_policy: "must_be_new_on_run",
            file_capability: "not_applicable",
        };
        emit_json_line(&out);
    } else {
        println!("state_status: initialized");
        println!("owned_control_entries: control.json, receipt-evidence");
        println!("effect_root_policy: must_be_new_on_run");
        println!("file_capability: not_applicable");
    }
    0
}

#[cfg(windows)]
fn run_engine_cmd(_args: RunArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "run",
        "unsupported_platform",
        6,
        "run with state-dir is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_engine_cmd(args: RunArgs, json_mode: bool) -> i32 {
    if has_parent_dir_component(&args.state_dir)
        || has_parent_dir_component(&args.config)
        || args
            .effect_root
            .as_deref()
            .is_some_and(has_parent_dir_component)
    {
        return emit_cli_error(
            "run",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    if verify_private_state_dir(&args.state_dir).is_err()
        || !args.state_dir.join("control.json").exists()
    {
        return emit_cli_error(
            "run",
            "state_dir_not_initialized",
            4,
            "state-dir is not initialized (run 'parapet init' first)",
            json_mode,
        );
    }

    let (profile, max_bytes) = if let Some(ref er) = args.effect_root {
        if std::fs::symlink_metadata(er).is_ok() {
            return emit_cli_error(
                "run",
                "effect_root_already_exists",
                2,
                "effect-root must be a new non-existent path",
                json_mode,
            );
        }
        if args.file_effect_profile.as_deref() != Some("v0") {
            return emit_cli_error(
                "run",
                "invalid_effect_arguments",
                2,
                "--file-effect-profile v0 is required when --effect-root is set",
                json_mode,
            );
        }
        let Some(mb) = args.max_content_bytes else {
            return emit_cli_error(
                "run",
                "invalid_effect_arguments",
                2,
                "--max-content-bytes is required when --effect-root is set",
                json_mode,
            );
        };
        if !(1..=1_048_576).contains(&mb) {
            return emit_cli_error(
                "run",
                "invalid_effect_arguments",
                2,
                "--max-content-bytes must be in 1..=1048576",
                json_mode,
            );
        }
        (Some(FileEffectProfileV0::select_explicitly()), Some(mb))
    } else {
        if args.file_effect_profile.is_some() || args.max_content_bytes.is_some() {
            return emit_cli_error(
                "run",
                "invalid_effect_arguments",
                2,
                "--file-effect-profile and --max-content-bytes require --effect-root",
                json_mode,
            );
        }
        (None, None)
    };

    let offline_fixture = match args.offline_fixture.as_deref() {
        None => None,
        Some(raw) => match OfflineUpstreamFixtureV0::parse(raw) {
            Some(f) => Some(f),
            None => {
                return emit_cli_error(
                    "run",
                    "invalid_offline_fixture",
                    2,
                    "invalid --offline-fixture value",
                    json_mode,
                );
            }
        },
    };

    if let Some(ref ready) = args.ready_file {
        if ready.is_empty()
            || ready.contains('/')
            || ready.contains('\\')
            || ready == "."
            || ready == ".."
        {
            return emit_cli_error(
                "run",
                "invalid_ready_file",
                2,
                "--ready-file must be a single relative filename",
                json_mode,
            );
        }
    }

    let Ok(cfg) = config::load_config(&config::FileSource {
        path: args.config.clone(),
    }) else {
        return emit_cli_error(
            "run",
            "invalid_config",
            3,
            "failed to load configuration file",
            json_mode,
        );
    };
    let config = std::sync::Arc::new(cfg);

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => {
            return emit_cli_error(
                "run",
                "runtime_init_failed",
                4,
                "failed to initialize async runtime",
                json_mode,
            );
        }
    };

    rt.block_on(async move {
        let options = EngineRuntimeOptionsV0 {
            state_dir: args.state_dir.clone(),
            effect_root: args.effect_root.clone(),
            file_effect_profile: profile,
            max_content_len_bytes: max_bytes,
            build_digest: None,
            offline_fixture,
            result_sink: None,
        };
        let contract_hash = config.contract_hash.clone();
        let (client, _store, root_cap) =
            match build_engine_client_with_runtime_v0(config, options) {
                Ok(parts) => parts,
                Err(_) => {
                    return emit_cli_error(
                        "run",
                        "engine_startup_failed",
                        4,
                        "engine runtime initialization failed",
                        json_mode,
                    );
                }
            };
        let has_cap = root_cap.is_some();
        // Capture the engine-generated identity before the capability moves into the control plane.
        let run_id = root_cap.as_ref().map(|cap| cap.run_id().clone());
        let upstream: std::sync::Arc<dyn crate::proxy::UpstreamClient> =
            std::sync::Arc::new(client);
        let app = crate::proxy::build_router(upstream);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], args.port));
        let Ok(listener) = tokio::net::TcpListener::bind(addr).await else {
            return emit_cli_error(
                "run",
                "bind_failed",
                4,
                "failed to bind local TCP listener",
                json_mode,
            );
        };
        let bound_port = listener.local_addr().map(|a| a.port()).unwrap_or(args.port);

        let Ok((plane, unix_listener, mut shutdown_rx)) = EngineControlPlane::bind(
            &args.state_dir,
            bound_port,
            &contract_hash,
            root_cap,
        ) else {
            return emit_cli_error(
                "run",
                "control_bind_failed",
                4,
                "failed to bind control channel",
                json_mode,
            );
        };
        plane.spawn_listener(unix_listener);
        plane.spawn_signal_gate();

        if let Some(ref ready_name) = args.ready_file {
            let _ = write_private_file(
                &args.state_dir,
                ready_name,
                format!("{bound_port}\n").as_bytes(),
            );
        }

        let serve_res = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.changed().await;
            })
            .await;

        remove_owned_transient_files(&args.state_dir);

        if serve_res.is_err() {
            return emit_cli_error(
                "run",
                "server_error",
                4,
                "proxy server terminated with an error",
                json_mode,
            );
        }

        let (cap_str, cleanup_str) = if has_cap {
            ("created", "clear")
        } else {
            ("not_applicable", "not_applicable")
        };
        let cleanup_lbl =
            file_cleanup_lifecycle_label(cleanup_str).unwrap_or("file cleanup: not applicable");

        if json_mode {
            let out = CliRunJsonV0 {
                schema: "parapet-cli-run/v0",
                port: bound_port,
                run_id: run_id.as_ref().map(RunIdV0::as_str),
                file_capability: cap_str,
                file_cleanup: cleanup_str,
                file_cleanup_label: cleanup_lbl,
                evidence_retained: true,
            };
            emit_json_line(&out);
        } else {
            println!("port: {bound_port}");
            println!(
                "run_id: {}",
                run_id.as_ref().map_or("not_applicable", RunIdV0::as_str)
            );
            println!("file_capability: {cap_str}");
            println!("{cleanup_lbl}");
            println!(
                "protected evidence: retained (use 'parapet evidence release' to release store objects)"
            );
        }
        0
    })
}

#[cfg(not(windows))]
struct AcceptSink {
    captured: std::sync::Mutex<Option<EnvelopeResponseResultV0>>,
}

#[cfg(not(windows))]
impl EnvelopeResultSinkV0 for AcceptSink {
    fn accept(&self, result: EnvelopeResponseResultV0) {
        *self.captured.lock().expect("accept sink lock") = Some(result);
    }
}

#[cfg(windows)]
fn run_accept(_args: AcceptArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "accept",
        "unsupported_platform",
        6,
        "accept is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_accept(args: AcceptArgs, json_mode: bool) -> i32 {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let (Some(state_dir), Some(config_path), Some(scenario)) =
        (args.state_dir, args.config, args.scenario)
    else {
        return emit_cli_error(
            "accept",
            "invalid_accept_combination",
            2,
            "missing required --state-dir, --config, or --scenario",
            json_mode,
        );
    };

    if has_parent_dir_component(&state_dir)
        || has_parent_dir_component(&config_path)
        || args
            .effect_root
            .as_deref()
            .is_some_and(has_parent_dir_component)
    {
        return emit_cli_error(
            "accept",
            "invalid_accept_combination",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    let (fixture, (required_config, required_hash)) = match scenario.as_str() {
        "allow_no_executor" => (
            OfflineUpstreamFixtureV0::AllowNoExecutor,
            ACCEPT_ENFORCED_CONFIG,
        ),
        "allow_verified_write" => (
            OfflineUpstreamFixtureV0::AllowVerifiedWrite,
            ACCEPT_ENFORCED_CONFIG,
        ),
        "block_no_effect" => (
            OfflineUpstreamFixtureV0::BlockNoEffect,
            ACCEPT_ENFORCED_CONFIG,
        ),
        "shadow_parse_error_l3_off" => (
            OfflineUpstreamFixtureV0::ShadowParseErrorL3Off,
            ACCEPT_SHADOW_L3_OFF_CONFIG,
        ),
        _ => {
            return emit_cli_error(
                "accept",
                "invalid_accept_combination",
                2,
                "unsupported acceptance scenario",
                json_mode,
            );
        }
    };

    let requires_effect_root = matches!(
        scenario.as_str(),
        "allow_verified_write" | "block_no_effect"
    );
    match (requires_effect_root, &args.effect_root) {
        (true, None) => {
            return emit_cli_error(
                "accept",
                "invalid_accept_combination",
                2,
                "scenario requires a new --effect-root",
                json_mode,
            );
        }
        (true, Some(er)) if std::fs::symlink_metadata(er).is_ok() => {
            return emit_cli_error(
                "accept",
                "invalid_accept_combination",
                2,
                "--effect-root must not already exist",
                json_mode,
            );
        }
        (false, Some(_)) => {
            return emit_cli_error(
                "accept",
                "invalid_accept_combination",
                2,
                "scenario forbids --effect-root",
                json_mode,
            );
        }
        _ => {}
    }

    // The caller bytes are loaded once and the loaded `Config` is never mutated.
    let Ok(cfg) = config::load_config(&config::FileSource { path: config_path }) else {
        return emit_cli_error(
            "accept",
            "invalid_config",
            3,
            "failed to load configuration file",
            json_mode,
        );
    };

    // Rev 9 D5: the exact immutable scenario config is required before any
    // state, effect, store, socket, or listener side effect.
    if cfg.contract_hash != required_hash {
        eprintln!(
            "parapet accept: error: config contract hash {} does not match required scenario config {required_config} ({required_hash})",
            cfg.contract_hash
        );
        if json_mode {
            emit_json_line(&CliAcceptConfigMismatchJsonV0 {
                schema: "parapet-cli-error/v0",
                command: "accept",
                error_code: "accept_config_mismatch",
                exit_code: 3,
                required_config,
                required_config_contract_hash: required_hash,
                observed_config_contract_hash: &cfg.contract_hash,
            });
        }
        return 3;
    }

    let max_content_len_bytes = cfg
        .policy
        .tools
        .get("write_file")
        .and_then(|t| t.constraints.get("content"))
        .and_then(|c| c.max_length)
        .map_or(4096u64, |len| len as u64);
    let config_contract_hash = cfg.contract_hash.clone();
    let config = std::sync::Arc::new(cfg);

    if !state_dir.exists() {
        if std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&state_dir)
            .is_err()
        {
            return emit_cli_error(
                "accept",
                "state_dir_create_failed",
                4,
                "unable to create state-dir",
                json_mode,
            );
        }
        let _ = std::fs::set_permissions(&state_dir, std::fs::Permissions::from_mode(0o700));
    }
    if verify_private_state_dir(&state_dir).is_err() {
        return emit_cli_error(
            "accept",
            "unsafe_state_dir",
            4,
            "state-dir must be a private 0700 directory",
            json_mode,
        );
    }
    if !state_dir.join("control.json").exists() {
        let meta = ControlStateMetadataV0 {
            schema: CONTROL_STATE_SCHEMA.to_owned(),
            effect_root_must_be_new: true,
        };
        if let Ok(bytes) = serde_json::to_vec(&meta) {
            let _ = write_private_file(&state_dir, "control.json", &bytes);
        }
    }

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => {
            return emit_cli_error(
                "accept",
                "runtime_init_failed",
                4,
                "failed to initialize async runtime",
                json_mode,
            );
        }
    };

    rt.block_on(async move {
        let sink = std::sync::Arc::new(AcceptSink {
            captured: std::sync::Mutex::new(None),
        });
        let options = EngineRuntimeOptionsV0 {
            state_dir: state_dir.clone(),
            effect_root: args.effect_root.clone(),
            file_effect_profile: args
                .effect_root
                .as_ref()
                .map(|_| FileEffectProfileV0::select_explicitly()),
            max_content_len_bytes: args.effect_root.as_ref().map(|_| max_content_len_bytes),
            build_digest: None,
            offline_fixture: Some(fixture),
            result_sink: Some(sink.clone()),
        };
        let (client, _store, _root_cap) = match build_engine_client_with_runtime_v0(config, options)
        {
            Ok(parts) => parts,
            Err(_) => {
                return emit_cli_error(
                    "accept",
                    "engine_startup_failed",
                    4,
                    "failed to build engine client for acceptance scenario",
                    json_mode,
                );
            }
        };
        let upstream: std::sync::Arc<dyn crate::proxy::UpstreamClient> =
            std::sync::Arc::new(client);
        let app = crate::proxy::build_router(upstream);
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
        let Ok(listener) = tokio::net::TcpListener::bind(addr).await else {
            return emit_cli_error(
                "accept",
                "bind_failed",
                4,
                "failed to bind ephemeral acceptance listener",
                json_mode,
            );
        };
        let bound_port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server_task = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = stop_rx.await;
                })
                .await;
        });

        let url = format!("http://127.0.0.1:{bound_port}/v1/chat/completions");
        let req_body = json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": format!("parapet accept scenario {scenario}")
                }
            ]
        });
        let http_client = reqwest::Client::new();
        let resp = http_client
            .post(&url)
            .header("content-type", "application/json")
            .header("x-parapet-original-host", "api.openai.com")
            .header("x-parapet-offline-scenario", scenario.as_str())
            .json(&req_body)
            .send()
            .await;

        let _ = stop_tx.send(());
        let _ = server_task.await;

        let Ok(resp) = resp else {
            return emit_cli_error(
                "accept",
                "request_failed",
                4,
                "offline acceptance request failed",
                json_mode,
            );
        };
        let http_status = resp.status().as_u16();
        let _ = resp.bytes().await;

        let captured = sink.captured.lock().expect("accept sink lock").clone();
        let Some(env_res) = captured else {
            return emit_cli_error(
                "accept",
                "no_envelope_result",
                4,
                "no receipt envelope result captured",
                json_mode,
            );
        };
        let Some(first_run) = env_res.runs.first() else {
            return emit_cli_error(
                "accept",
                "no_run_captured",
                4,
                "no candidate run captured",
                json_mode,
            );
        };

        let proj = receipt_projection(&first_run.row);
        let sum_str = summary_code(proj.summary);
        let sum_lbl = summary_label(proj.summary);
        let enf_str = enforcement_code(proj.policy.enforcement);
        let enf_lbl = enforcement_label(proj.policy.enforcement);

        let ver_avail = availability_code(proj.verification.availability);
        let ver_outcome = proj.verification.outcome.map(verification_outcome_code);
        let ver_err = verification_error_code(&proj.verification);
        let ver_lbl = projection_verification_label(&proj.verification);
        let upstream_lbl = "upstream: deterministic offline fixture (no real model contacted)";
        let requester_version = env!("CARGO_PKG_VERSION");

        if json_mode {
            let out = CliAcceptJsonV0 {
                schema: "parapet-cli-accept/v0",
                scenario: &scenario,
                requester: ACCEPT_REQUESTER,
                requester_version,
                run_id: first_run.run_id.as_str(),
                config_contract_hash: &config_contract_hash,
                real_model_contacted: false,
                upstream_label: upstream_lbl,
                http_status,
                summary: sum_str,
                summary_label: sum_lbl,
                enforcement: enf_str,
                enforcement_label: enf_lbl,
                verification_availability: ver_avail,
                verification_outcome: ver_outcome,
                verification_error_code: ver_err,
                verification_label: &ver_lbl,
                replay_verdict: None,
                replay_reason: None,
                replay_label: None,
            };
            emit_json_line(&out);
        } else {
            println!("scenario: {scenario}");
            println!("requester: Rust CLI {requester_version}");
            println!("run_id: {}", first_run.run_id.as_str());
            println!("config_contract_hash: {config_contract_hash}");
            println!("real_model_contacted: false");
            println!("{upstream_lbl}");
            println!("http_status: {http_status}");
            println!("summary: {sum_lbl}");
            println!("{enf_lbl}");
            println!("{ver_lbl}");
        }
        0
    })
}

#[cfg(windows)]
fn run_inspect(_args: InspectArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "inspect",
        "unsupported_platform",
        6,
        "inspect is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_inspect(args: InspectArgs, json_mode: bool) -> i32 {
    if has_parent_dir_component(&args.state_dir) {
        return emit_cli_error(
            "inspect",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    let Some(run_id) = RunIdV0::parse(&args.run_id) else {
        return emit_cli_error(
            "inspect",
            "invalid_run_id",
            2,
            "--run-id must be 32 lowercase hex characters",
            json_mode,
        );
    };

    let claimed = match args.claimed_outcome.as_deref() {
        None => None,
        Some("complete") => Some(TerminalOutcomeV0::Complete),
        Some("storage_incomplete") => Some(TerminalOutcomeV0::StorageIncomplete),
        Some(_) => {
            return emit_cli_error(
                "inspect",
                "invalid_claimed_outcome",
                2,
                "--claimed-outcome must be complete or storage_incomplete",
                json_mode,
            );
        }
    };

    let store_path = args.state_dir.join("receipt-evidence");
    let store = match ProtectedEvidenceStoreV0::open(&store_path) {
        Ok(s) => s,
        Err(sf) => {
            return emit_cli_error(
                "inspect",
                store_failure_code(sf),
                4,
                store_failure_label(sf),
                json_mode,
            );
        }
    };

    // The redacted view is the sole source of every rendered field and code.
    let view = RedactedEvidenceReaderV0::new(&store).view(&run_id, claimed);

    let inert_fallback = args.inert_fallback;

    if json_mode {
        let bytes = splice_inspect_json(
            run_id.as_str(),
            view.outcome,
            view.cleanup,
            view.banner,
            view.reader_code,
            view.projection_code,
            view.projection_bytes.as_deref(),
        );
        let _ = std::io::stdout().write_all(&bytes);
        let _ = std::io::stdout().flush();
    } else {
        let text = render_inspect_human(
            run_id.as_str(),
            view.outcome,
            view.cleanup,
            view.banner,
            view.reader_code,
            view.projection_code,
            view.projection_bytes.as_deref(),
            inert_fallback,
        );
        let _ = std::io::stdout().write_all(text.as_bytes());
        let _ = std::io::stdout().flush();
    }

    if view.reader_code.is_some()
        || view.projection_code.is_some()
        || view.outcome != Some(TerminalOutcomeV0::Complete)
    {
        4
    } else {
        0
    }
}

#[cfg(windows)]
fn run_restore(_args: RestoreArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "restore",
        "unsupported_platform",
        6,
        "restore is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_restore(args: RestoreArgs, json_mode: bool) -> i32 {
    if has_parent_dir_component(&args.state_dir) {
        return emit_cli_error(
            "restore",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    let result_code = if matches!(
        check_state_dir_ownership(&args.state_dir),
        OwnershipCheck::Owned(_)
    ) {
        match send_control_request(&args.state_dir, "restore") {
            Ok(val) => val
                .get("file_cleanup")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("engine_unavailable")
                .to_owned(),
            Err(_) => "engine_unavailable".to_owned(),
        }
    } else {
        "engine_unavailable".to_owned()
    };

    let label = file_cleanup_lifecycle_label(&result_code)
        .unwrap_or("cleanup: not run (owning engine unavailable)");
    let a6_booking_required = result_code == "indeterminate";
    let exit_code = match result_code.as_str() {
        "clear" | "not_applicable" => 0,
        "indeterminate" => 5,
        _ => 4,
    };

    if json_mode {
        let out = CliRestoreJsonV0 {
            schema: "parapet-cli-restore/v0",
            result: &result_code,
            label,
            a6_booking_required,
            evidence_retained: true,
        };
        emit_json_line(&out);
    } else {
        println!("{label}");
        println!("a6_booking_required: {a6_booking_required}");
        println!(
            "protected evidence: retained (use 'parapet evidence release' to release store objects)"
        );
    }
    exit_code
}

#[cfg(windows)]
fn run_evidence_release(_args: EvidenceReleaseArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "evidence release",
        "unsupported_platform",
        6,
        "evidence release is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_evidence_release(args: EvidenceReleaseArgs, json_mode: bool) -> i32 {
    if has_parent_dir_component(&args.state_dir) {
        return emit_cli_error(
            "evidence release",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    let Some(run_id) = RunIdV0::parse(&args.run_id) else {
        return emit_cli_error(
            "evidence release",
            "invalid_run_id",
            2,
            "--run-id must be 32 lowercase hex characters",
            json_mode,
        );
    };

    let store_path = args.state_dir.join("receipt-evidence");
    let store = match ProtectedEvidenceStoreV0::open(&store_path) {
        Ok(s) => s,
        Err(sf) => {
            let sf_code = store_failure_code(sf);
            let sf_lbl = store_failure_label(sf);
            if json_mode {
                let out = CliEvidenceReleaseJsonV0 {
                    schema: "parapet-cli-evidence-release/v0",
                    run_id: run_id.as_str(),
                    store_open: "failed",
                    store_failure_code: Some(sf_code),
                    result: None,
                    label: sf_lbl,
                    control_evidence_retained: true,
                };
                emit_json_line(&out);
            } else {
                println!("run_id: {}", run_id.as_str());
                println!("{sf_lbl}");
                println!("control_evidence_retained: true");
            }
            return 4;
        }
    };

    let cleanup_res = store.cleanup_run(&run_id);
    let (res_str, exit_code) = match cleanup_res {
        CleanupResultV0::Clear => ("clear", 0),
        CleanupResultV0::Fail => ("fail", 4),
        CleanupResultV0::Indeterminate => ("indeterminate", 5),
        CleanupResultV0::Refused(CleanupRefusalV0::LockFailed) => ("refused_lock_failed", 4),
        CleanupResultV0::Refused(CleanupRefusalV0::ManifestUnusable) => {
            ("refused_manifest_unusable", 4)
        }
        CleanupResultV0::Refused(CleanupRefusalV0::TerminalUnsettled) => {
            ("refused_terminal_unsettled", 4)
        }
    };
    let lbl = evidence_cleanup_result_label(cleanup_res);

    if json_mode {
        let out = CliEvidenceReleaseJsonV0 {
            schema: "parapet-cli-evidence-release/v0",
            run_id: run_id.as_str(),
            store_open: "ok",
            store_failure_code: None,
            result: Some(res_str),
            label: lbl,
            control_evidence_retained: true,
        };
        emit_json_line(&out);
    } else {
        println!("run_id: {}", run_id.as_str());
        println!("{lbl}");
        println!(
            "control evidence retained: lock, run/{}/manifest.json, run/{}/terminal/*.json",
            run_id.as_str(),
            run_id.as_str()
        );
    }
    exit_code
}

#[cfg(windows)]
fn run_disable(_args: DisableArgs, json_mode: bool) -> i32 {
    emit_cli_error(
        "disable",
        "unsupported_platform",
        6,
        "disable is only supported on POSIX platforms",
        json_mode,
    )
}

#[cfg(not(windows))]
fn run_disable(args: DisableArgs, json_mode: bool) -> i32 {
    if has_parent_dir_component(&args.state_dir) {
        return emit_cli_error(
            "disable",
            "invalid_path",
            2,
            "path arguments must not contain parent directory traversal",
            json_mode,
        );
    }

    match check_state_dir_ownership(&args.state_dir) {
        OwnershipCheck::NotOwned => {
            let ownership_lbl = "engine ownership: not owned; no stop attempted";
            if json_mode {
                let out = CliDisableJsonV0 {
                    schema: "parapet-cli-disable/v0",
                    status: "refused",
                    refusal_code: Some("not_owned"),
                    file_cleanup: None,
                    file_cleanup_label: None,
                    ownership_label: ownership_lbl,
                    owned_control_removed: false,
                    evidence_retained: true,
                };
                emit_json_line(&out);
            } else {
                println!("status: refused");
                println!("{ownership_lbl}");
                println!(
                    "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                );
            }
            4
        }
        OwnershipCheck::Unavailable => {
            let ownership_lbl = "engine ownership: unavailable";
            let cleanup_lbl = "cleanup: not run (owning engine unavailable)";
            if json_mode {
                let out = CliDisableJsonV0 {
                    schema: "parapet-cli-disable/v0",
                    status: "refused",
                    refusal_code: Some("engine_unavailable"),
                    file_cleanup: Some("engine_unavailable"),
                    file_cleanup_label: Some(cleanup_lbl),
                    ownership_label: ownership_lbl,
                    owned_control_removed: false,
                    evidence_retained: true,
                };
                emit_json_line(&out);
            } else {
                println!("status: refused");
                println!("{ownership_lbl}");
                println!("{cleanup_lbl}");
                println!(
                    "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                );
            }
            4
        }
        OwnershipCheck::Owned(owner) => {
            let ownership_lbl = "engine ownership: owned";
            let Ok(resp) = send_control_request(&args.state_dir, "disable") else {
                let cleanup_lbl = "cleanup: not run (owning engine unavailable)";
                if json_mode {
                    let out = CliDisableJsonV0 {
                        schema: "parapet-cli-disable/v0",
                        status: "refused",
                        refusal_code: Some("engine_unavailable"),
                        file_cleanup: Some("engine_unavailable"),
                        file_cleanup_label: Some(cleanup_lbl),
                        ownership_label: "engine ownership: unavailable",
                        owned_control_removed: false,
                        evidence_retained: true,
                    };
                    emit_json_line(&out);
                } else {
                    println!("status: refused");
                    println!("engine ownership: unavailable");
                    println!("{cleanup_lbl}");
                    println!(
                        "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                    );
                }
                return 4;
            };

            let stop_approved = resp
                .get("stop_approved")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let file_cleanup_raw = resp
                .get("file_cleanup")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("not_run");
            let file_cleanup = match file_cleanup_raw {
                "clear" => "clear",
                "not_applicable" => "not_applicable",
                "fail" => "fail",
                "indeterminate" => "indeterminate",
                _ => "not_run",
            };
            let cleanup_lbl = file_cleanup_lifecycle_label(file_cleanup)
                .unwrap_or("file cleanup: not run (restore required)");

            if stop_approved {
                // Rev 9 L4: `stopped` and control removal only after confirmed exit;
                // a bounded wait that times out preserves every owned control.
                if !wait_for_owned_shutdown(&args.state_dir, owner.pid, CONTROL_TIMEOUT) {
                    let stop_lbl =
                        "engine stop: exit not confirmed within 5000 ms; owned controls preserved";
                    if json_mode {
                        let out = CliDisableJsonV0 {
                            schema: "parapet-cli-disable/v0",
                            status: "refused",
                            refusal_code: Some("stop_unconfirmed"),
                            file_cleanup: Some(file_cleanup),
                            file_cleanup_label: Some(cleanup_lbl),
                            ownership_label: ownership_lbl,
                            owned_control_removed: false,
                            evidence_retained: true,
                        };
                        emit_json_line(&out);
                    } else {
                        println!("status: refused");
                        println!("{ownership_lbl}");
                        println!("{cleanup_lbl}");
                        println!("{stop_lbl}");
                        println!(
                            "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                        );
                    }
                    return 4;
                }
                remove_owned_transient_files(&args.state_dir);
                if json_mode {
                    let out = CliDisableJsonV0 {
                        schema: "parapet-cli-disable/v0",
                        status: "stopped",
                        refusal_code: None,
                        file_cleanup: Some(file_cleanup),
                        file_cleanup_label: Some(cleanup_lbl),
                        ownership_label: ownership_lbl,
                        owned_control_removed: true,
                        evidence_retained: true,
                    };
                    emit_json_line(&out);
                } else {
                    println!("status: stopped");
                    println!("{ownership_lbl}");
                    println!("{cleanup_lbl}");
                    println!(
                        "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                    );
                }
                0
            } else {
                let (refusal_code, exit_code) = match file_cleanup {
                    "fail" => ("file_cleanup_fail", 4),
                    "indeterminate" => ("file_cleanup_indeterminate", 5),
                    _ => ("file_cleanup_not_run", 4),
                };
                if json_mode {
                    let out = CliDisableJsonV0 {
                        schema: "parapet-cli-disable/v0",
                        status: "refused",
                        refusal_code: Some(refusal_code),
                        file_cleanup: Some(file_cleanup),
                        file_cleanup_label: Some(cleanup_lbl),
                        ownership_label: ownership_lbl,
                        owned_control_removed: false,
                        evidence_retained: true,
                    };
                    emit_json_line(&out);
                } else {
                    println!("status: refused");
                    println!("{ownership_lbl}");
                    println!("{cleanup_lbl}");
                    println!(
                        "protected evidence: retained (use 'parapet evidence release' to release store objects)"
                    );
                }
                exit_code
            }
        }
    }
}
