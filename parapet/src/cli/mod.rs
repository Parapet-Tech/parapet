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
    availability_code, enforcement_code, enforcement_label, evidence_cleanup_lifecycle_label,
    evidence_cleanup_result_label, evidence_cleanup_view_label, file_cleanup_lifecycle_label,
    join_lines, next_step_line, original_decision_line, projection_verification_label,
    render_inspect_human, render_run_startup_human, splice_inspect_json, store_failure_code,
    store_failure_label, summary_code, summary_label, verification_error_code,
    verification_outcome_code, LifecycleHumanV0, NextStepV0, ORIGINAL_DECISION_NO_RUN,
    ORIGINAL_DECISION_UNKNOWN,
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
    build_engine_client_with_runtime_and_upstream_v0, build_engine_client_with_runtime_v0,
    EngineRuntimeOptionsV0, OfflineUpstreamFixtureV0, UpstreamTable, UpstreamValidationOptions,
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
    /// Deterministic offline demo assets
    Demo(DemoArgs),
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

    /// Override upstream base URL for a provider (`openai=<url>` or `anthropic=<url>`).
    /// May be passed at most once per provider.
    #[arg(long = "upstream", value_name = "PROVIDER=URL")]
    upstream: Vec<String>,

    /// Explicitly allow plain HTTP (`http://`) upstreams to literal loopback IPs (`127.0.0.0/8` or `[::1]`).
    #[arg(long = "allow-loopback-http-upstream", default_value_t = false)]
    allow_loopback_http_upstream: bool,
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

#[derive(Args)]
struct DemoArgs {
    #[command(subcommand)]
    command: DemoCommands,
}

#[derive(Subcommand)]
enum DemoCommands {
    /// Write the embedded, hash-verified demo assets into a new directory
    Export(DemoExportArgs),
}

#[derive(Args)]
struct DemoExportArgs {
    /// New directory to create (must not exist; its parent must exist)
    #[arg(long)]
    out: PathBuf,
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
struct CliUpstreamEndpointReadyJsonV0<'a> {
    origin: &'a str,
    mode: &'static str,
    source: &'static str,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliUpstreamsReadyMapJsonV0<'a> {
    openai: CliUpstreamEndpointReadyJsonV0<'a>,
    anthropic: CliUpstreamEndpointReadyJsonV0<'a>,
}

#[cfg(not(windows))]
#[derive(serde::Serialize)]
struct CliUpstreamReadyJsonV0<'a> {
    schema: &'static str,
    listen_addr: &'a str,
    allow_loopback_http_upstream: bool,
    upstreams: CliUpstreamsReadyMapJsonV0<'a>,
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
        Commands::Demo(demo) => {
            let DemoCommands::Export(_args) = demo.command;
            #[cfg(not(windows))]
            return crate::engine::demo::export::export(&_args.out, json_mode);
            #[cfg(windows)]
            return emit_cli_error(
                "demo export",
                "unsupported_platform",
                6,
                "demo export is only supported on POSIX platforms",
                json_mode,
            );
        }
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
        // Verdict first; clean and flagged results print the same 13 fields in the
        // same order. The verdict comes only from the record's raw-f64 outcome.
        let verdict = match record.detection_outcome {
            DetectionOutcome::ThresholdBreached => "flagged",
            DetectionOutcome::ThresholdNotBreached => "clean",
        };
        let (effective_raw, threshold) =
            format_scan_score_threshold_human(record.effective_raw, record.threshold);
        let features = record
            .feature_count_matched
            .map_or_else(|| "none".to_owned(), |n| n.to_string());
        println!("verdict: {verdict}");
        println!("detection_outcome: {outcome_str}");
        println!("effective_raw: {effective_raw}");
        println!("threshold: {threshold}");
        println!("mode: {mode_str}");
        println!("record_schema_version: {}", record.schema_version);
        println!("record_digest: {digest}");
        println!("model_id: {}", record.model_id);
        println!("model_digest: {}", record.model_digest);
        println!("input_digest: {}", record.input_digest);
        println!("input_len_bytes: {}", record.input_len_bytes);
        println!("attribution_status: {attr_str}");
        println!("feature_count_matched: {features}");
    }
    0
}

/// Display-only human rendering of a finite scan `f64`: the exact binary64
/// value rounded to 8 decimal places with ties to even, trailing zeros and any
/// terminal decimal point trimmed, and zero always shown as `0`. Total on
/// non-finite inputs, which the scan path rejects before rendering.
fn format_scan_f64_human(v: f64) -> String {
    const SCALE: u128 = 100_000_000;
    if v.is_nan() {
        return "non_finite_nan".to_owned();
    }
    if v == f64::INFINITY {
        return "non_finite_pos_inf".to_owned();
    }
    if v == f64::NEG_INFINITY {
        return "non_finite_neg_inf".to_owned();
    }

    // |v| = m * 2^e exactly, so |v| * 10^8 = (m * 5^8) * 2^(e + 8).
    let bits = v.abs().to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac_bits = bits & 0x000f_ffff_ffff_ffff;
    let (m, e) = if exp_bits == 0 {
        (frac_bits, -1074)
    } else {
        ((1_u64 << 52) | frac_bits, exp_bits - 1075)
    };
    let n = u128::from(m) * 390_625;
    let shift = e + 8;
    let q = if shift >= 0 {
        if shift > 55 {
            // |v| >= 2^47 is already an integer; std formatting is exact here.
            let s = format!("{v:.8}");
            return s.trim_end_matches(".00000000").to_owned();
        }
        n << shift
    } else {
        let k = (-shift) as u32;
        if k >= 128 {
            0
        } else {
            let q = n >> k;
            let half = (n >> (k - 1)) & 1;
            let sticky = n & ((1_u128 << (k - 1)) - 1);
            if half == 1 && (sticky != 0 || q & 1 == 1) {
                q + 1
            } else {
                q
            }
        }
    };

    if q == 0 {
        return "0".to_owned();
    }
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let int_part = q / SCALE;
    let frac_part = q % SCALE;
    if frac_part == 0 {
        return format!("{sign}{int_part}");
    }
    let frac = format!("{frac_part:08}");
    format!("{sign}{int_part}.{}", frac.trim_end_matches('0'))
}

/// Human `effective_raw` and `threshold` values. When both round to the same
/// digits, the score carries the raw relation that decided the verdict.
fn format_scan_score_threshold_human(effective_raw: f64, threshold: f64) -> (String, String) {
    use std::cmp::Ordering;

    let score = format_scan_f64_human(effective_raw);
    let thresh = format_scan_f64_human(threshold);
    if score != thresh {
        return (score, thresh);
    }
    // IEEE comparison, so -0.0 == 0.0 matches `effective_raw >= threshold`.
    let relation = match effective_raw.partial_cmp(&threshold) {
        Some(Ordering::Less) => "<",
        Some(Ordering::Equal) => "==",
        Some(Ordering::Greater) => ">",
        None => return (score, thresh),
    };
    (
        format!("{score} (raw_relation: effective_raw {relation} threshold)"),
        thresh,
    )
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

    if offline_fixture.is_some() && (!args.upstream.is_empty() || args.allow_loopback_http_upstream)
    {
        return emit_cli_error(
            "run",
            "invalid_offline_fixture_combination",
            2,
            "--offline-fixture cannot be combined with --upstream or --allow-loopback-http-upstream",
            json_mode,
        );
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

    let upstream_opts = UpstreamValidationOptions {
        cli_upstreams: args.upstream.clone(),
        allow_loopback_http_upstream: args.allow_loopback_http_upstream,
        listener_port: if args.port != 0 {
            Some(args.port)
        } else {
            None
        },
    };
    let upstream_table = match UpstreamTable::from_process_env_with_options(&upstream_opts) {
        Ok(table) => table,
        Err(diag) => {
            return emit_cli_error(
                "run",
                "upstream_config_invalid",
                4,
                &diag.to_string(),
                json_mode,
            );
        }
    };

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
        if let Err(diag) = upstream_table.check_no_self_loop(bound_port) {
            drop(listener);
            return emit_cli_error(
                "run",
                "upstream_config_invalid",
                4,
                &diag.to_string(),
                json_mode,
            );
        }

        let openai_meta = upstream_table
            .endpoint_metadata(crate::proxy::Provider::OpenAi)
            .clone();
        let anthropic_meta = upstream_table
            .endpoint_metadata(crate::proxy::Provider::Anthropic)
            .clone();

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
            match build_engine_client_with_runtime_and_upstream_v0(config, options, upstream_table)
            {
                Ok(parts) => parts,
                Err(crate::engine::StartupError::UpstreamConfig(diag)) => {
                    return emit_cli_error(
                        "run",
                        "upstream_config_invalid",
                        4,
                        &diag.to_string(),
                        json_mode,
                    );
                }
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

        if offline_fixture.is_none() {
            if json_mode {
                let listen_addr = format!("127.0.0.1:{bound_port}");
                let ready_json = CliUpstreamReadyJsonV0 {
                    schema: "parapet-upstream-ready/v0",
                    listen_addr: &listen_addr,
                    allow_loopback_http_upstream: args.allow_loopback_http_upstream,
                    upstreams: CliUpstreamsReadyMapJsonV0 {
                        openai: CliUpstreamEndpointReadyJsonV0 {
                            origin: &openai_meta.sanitized_origin,
                            mode: openai_meta.mode.as_str(),
                            source: openai_meta.source.as_str(),
                        },
                        anthropic: CliUpstreamEndpointReadyJsonV0 {
                            origin: &anthropic_meta.sanitized_origin,
                            mode: anthropic_meta.mode.as_str(),
                            source: anthropic_meta.source.as_str(),
                        },
                    },
                };
                if let Ok(mut bytes) = serde_json::to_vec(&ready_json) {
                    bytes.push(b'\n');
                    let _ = std::io::stderr().write_all(&bytes);
                    let _ = std::io::stderr().flush();
                }
            } else {
                eprintln!(
                    "parapet listening on 127.0.0.1:{bound_port} (openai={} [mode={} source={}], anthropic={} [mode={} source={}])",
                    openai_meta.sanitized_origin,
                    openai_meta.mode.as_str(),
                    openai_meta.source.as_str(),
                    anthropic_meta.sanitized_origin,
                    anthropic_meta.mode.as_str(),
                    anthropic_meta.source.as_str()
                );
                let _ = std::io::stderr().flush();
            }
        }

        if let Some(ref ready_name) = args.ready_file {
            let _ = write_private_file(
                &args.state_dir,
                ready_name,
                format!("{bound_port}\n").as_bytes(),
            );
        }

        // Readiness is durable (control plane bound, ready file written): tell
        // the human operator the port, run ID, and mode before serve blocks.
        // JSON stdout stays the single `parapet-cli-run/v0` line at exit.
        if !json_mode {
            let startup =
                render_run_startup_human(bound_port, run_id.as_ref().map(RunIdV0::as_str));
            let mut stdout = std::io::stdout().lock();
            let _ = stdout.write_all(startup.as_bytes());
            let _ = stdout.flush();
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
            println!("status: stopped");
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
            Err(crate::engine::StartupError::UpstreamConfig(diag)) => {
                return emit_cli_error(
                    "accept",
                    "upstream_config_invalid",
                    4,
                    &diag.to_string(),
                    json_mode,
                );
            }
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

    let (result_code, run) = match check_state_dir_ownership(&args.state_dir) {
        OwnershipCheck::Owned(owner) => match send_control_request(&args.state_dir, "restore") {
            Ok(val) => (
                val.get("file_cleanup")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("engine_unavailable")
                    .to_owned(),
                KnownRunV0::from_owner(owner.run_id),
            ),
            Err(_) => ("engine_unavailable".to_owned(), KnownRunV0::Unknown),
        },
        _ => ("engine_unavailable".to_owned(), KnownRunV0::Unknown),
    };

    let label = file_cleanup_lifecycle_label(&result_code)
        .unwrap_or("cleanup: not run (owning engine unavailable)");
    // JSON v0 field retained for compatibility; human output states the next step instead.
    let a6_booking_required = result_code == "indeterminate";
    let (exit_code, next) = match result_code.as_str() {
        "clear" | "not_applicable" => (0, NextStepV0::Disable),
        "indeterminate" => (5, NextStepV0::ReviewEffectRoot),
        "fail" => (4, NextStepV0::Restore),
        _ => (4, NextStepV0::Inspect),
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
        let observed = observe_run(&args.state_dir, &run);
        let mut lines = observed
            .lifecycle(file_cleanup_human_line(&result_code))
            .lines();
        lines.push(next_step_line(next, run.run_id()));
        let _ = std::io::stdout().write_all(join_lines(&lines).as_bytes());
        let _ = std::io::stdout().flush();
    }
    exit_code
}

/// What a lifecycle command knows about the engine's file-effect run.
#[cfg(not(windows))]
enum KnownRunV0 {
    /// The owning engine reported this engine-generated run ID.
    Run(String),
    /// The owning engine has no file-effect capability, hence no run.
    NoRun,
    /// No owning engine answered, so the run is not identified.
    Unknown,
}

#[cfg(not(windows))]
impl KnownRunV0 {
    fn from_owner(run_id: Option<String>) -> Self {
        run_id.map_or(Self::NoRun, Self::Run)
    }

    fn run_id(&self) -> Option<&str> {
        match self {
            Self::Run(run) => Some(run),
            Self::NoRun | Self::Unknown => None,
        }
    }
}

/// Read-only observation of a run's original decision and evidence cleanup.
#[cfg(not(windows))]
struct ObservedRunV0<'a> {
    run_id: Option<&'a str>,
    decision: String,
    evidence_cleanup: &'static str,
}

#[cfg(not(windows))]
impl<'a> ObservedRunV0<'a> {
    fn lifecycle<'b>(&'b self, file_cleanup: &'b str) -> LifecycleHumanV0<'b> {
        LifecycleHumanV0 {
            run_id: self.run_id,
            decision: &self.decision,
            file_cleanup,
            evidence_cleanup: self.evidence_cleanup,
        }
    }
}

/// Observe a run through the redacted reader (the same authority as `inspect`).
#[cfg(not(windows))]
fn observe_run<'a>(state_dir: &std::path::Path, run: &'a KnownRunV0) -> ObservedRunV0<'a> {
    let unknown_evidence =
        evidence_cleanup_lifecycle_label("unknown").unwrap_or("evidence cleanup: unknown");
    let parsed = match run {
        KnownRunV0::Run(raw) => RunIdV0::parse(raw),
        KnownRunV0::NoRun => {
            return ObservedRunV0 {
                run_id: None,
                decision: ORIGINAL_DECISION_NO_RUN.to_owned(),
                evidence_cleanup: evidence_cleanup_lifecycle_label("not_applicable")
                    .unwrap_or("evidence cleanup: not applicable"),
            };
        }
        KnownRunV0::Unknown => None,
    };
    let Some(run_id) = parsed else {
        return ObservedRunV0 {
            run_id: None,
            decision: ORIGINAL_DECISION_UNKNOWN.to_owned(),
            evidence_cleanup: unknown_evidence,
        };
    };
    let observed_run_id = run.run_id();
    let Ok(store) = ProtectedEvidenceStoreV0::open(&state_dir.join("receipt-evidence")) else {
        return ObservedRunV0 {
            run_id: observed_run_id,
            decision: "original decision: not available (evidence store unavailable)".to_owned(),
            evidence_cleanup: unknown_evidence,
        };
    };
    let view = RedactedEvidenceReaderV0::new(&store).view(&run_id, None);
    ObservedRunV0 {
        run_id: observed_run_id,
        decision: original_decision_line(
            view.projection_bytes.as_deref(),
            view.reader_code,
            view.projection_code,
        ),
        evidence_cleanup: evidence_cleanup_view_label(view.cleanup, view.reader_code),
    }
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
                println!("control evidence retained: true");
            }
            return 4;
        }
    };

    // The original decision is read before release removes the projection object.
    let decision = (!json_mode).then(|| {
        let view = RedactedEvidenceReaderV0::new(&store).view(&run_id, None);
        original_decision_line(
            view.projection_bytes.as_deref(),
            view.reader_code,
            view.projection_code,
        )
    });
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
        let next = match cleanup_res {
            CleanupResultV0::Clear => Some(NextStepV0::Inspect),
            CleanupResultV0::Fail
            | CleanupResultV0::Indeterminate
            | CleanupResultV0::Refused(CleanupRefusalV0::LockFailed) => {
                Some(NextStepV0::EvidenceRelease)
            }
            CleanupResultV0::Refused(CleanupRefusalV0::TerminalUnsettled) => {
                Some(NextStepV0::Inspect)
            }
            CleanupResultV0::Refused(CleanupRefusalV0::ManifestUnusable) => None,
        };
        let mut lines = LifecycleHumanV0 {
            run_id: Some(run_id.as_str()),
            decision: decision.as_deref().unwrap_or(ORIGINAL_DECISION_UNKNOWN),
            file_cleanup: file_cleanup_lifecycle_label("not_recorded")
                .unwrap_or("file cleanup: not recorded in protected evidence"),
            evidence_cleanup: lbl,
        }
        .lines();
        lines.push(format!(
            "control evidence retained: lock, run/{}/manifest.json, run/{}/terminal/*.json",
            run_id.as_str(),
            run_id.as_str()
        ));
        if let Some(step) = next {
            lines.push(next_step_line(step, Some(run_id.as_str())));
        }
        let _ = std::io::stdout().write_all(join_lines(&lines).as_bytes());
        let _ = std::io::stdout().flush();
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

    let outcome = evaluate_disable(&args.state_dir);
    if json_mode {
        emit_json_line(&CliDisableJsonV0 {
            schema: "parapet-cli-disable/v0",
            status: outcome.status,
            refusal_code: outcome.refusal_code,
            file_cleanup: outcome.file_cleanup,
            file_cleanup_label: outcome.file_cleanup_label,
            ownership_label: outcome.ownership_label,
            owned_control_removed: outcome.owned_control_removed,
            evidence_retained: true,
        });
    } else {
        let mut lines = vec![
            format!("status: {}", outcome.status),
            outcome.ownership_label.to_owned(),
        ];
        if let Some(stop_lbl) = outcome.stop_label {
            lines.push(stop_lbl.to_owned());
        }
        let observed = observe_run(&args.state_dir, &outcome.run);
        let file_cleanup = outcome.file_cleanup.map_or(
            "file cleanup: not observed (engine not owned)",
            file_cleanup_human_line,
        );
        lines.extend(observed.lifecycle(file_cleanup).lines());
        if let Some(step) = outcome.next {
            lines.push(next_step_line(step, outcome.run.run_id()));
        }
        let _ = std::io::stdout().write_all(join_lines(&lines).as_bytes());
        let _ = std::io::stdout().flush();
    }
    outcome.exit_code
}

/// Settled result of `parapet disable`, rendered as JSON v0 or human lines.
#[cfg(not(windows))]
struct DisableOutcomeV0 {
    status: &'static str,
    refusal_code: Option<&'static str>,
    file_cleanup: Option<&'static str>,
    file_cleanup_label: Option<&'static str>,
    ownership_label: &'static str,
    owned_control_removed: bool,
    stop_label: Option<&'static str>,
    run: KnownRunV0,
    next: Option<NextStepV0>,
    exit_code: i32,
}

#[cfg(not(windows))]
fn evaluate_disable(state_dir: &std::path::Path) -> DisableOutcomeV0 {
    let unavailable = || DisableOutcomeV0 {
        status: "refused",
        refusal_code: Some("engine_unavailable"),
        file_cleanup: Some("engine_unavailable"),
        file_cleanup_label: Some("cleanup: not run (owning engine unavailable)"),
        ownership_label: "engine ownership: unavailable",
        owned_control_removed: false,
        stop_label: None,
        run: KnownRunV0::Unknown,
        next: Some(NextStepV0::Inspect),
        exit_code: 4,
    };
    let owner = match check_state_dir_ownership(state_dir) {
        OwnershipCheck::NotOwned => {
            return DisableOutcomeV0 {
                status: "refused",
                refusal_code: Some("not_owned"),
                file_cleanup: None,
                file_cleanup_label: None,
                ownership_label: "engine ownership: not owned; no stop attempted",
                owned_control_removed: false,
                stop_label: None,
                run: KnownRunV0::Unknown,
                next: None,
                exit_code: 4,
            };
        }
        OwnershipCheck::Unavailable => return unavailable(),
        OwnershipCheck::Owned(owner) => owner,
    };
    let Ok(resp) = send_control_request(state_dir, "disable") else {
        return unavailable();
    };

    let stop_approved = resp
        .get("stop_approved")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let file_cleanup = match resp
        .get("file_cleanup")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("not_run")
    {
        "clear" => "clear",
        "not_applicable" => "not_applicable",
        "fail" => "fail",
        "indeterminate" => "indeterminate",
        _ => "not_run",
    };
    let cleanup_lbl = file_cleanup_lifecycle_label(file_cleanup)
        .unwrap_or("file cleanup: not run (restore required)");
    let owned =
        |status, refusal_code, removed, stop_label, run, next, exit_code| DisableOutcomeV0 {
            status,
            refusal_code,
            file_cleanup: Some(file_cleanup),
            file_cleanup_label: Some(cleanup_lbl),
            ownership_label: "engine ownership: owned",
            owned_control_removed: removed,
            stop_label,
            run,
            next,
            exit_code,
        };
    let run = KnownRunV0::from_owner(owner.run_id.clone());

    if !stop_approved {
        let (refusal_code, next, exit_code) = match file_cleanup {
            "fail" => ("file_cleanup_fail", NextStepV0::Restore, 4),
            "indeterminate" => (
                "file_cleanup_indeterminate",
                NextStepV0::ReviewEffectRoot,
                5,
            ),
            _ => ("file_cleanup_not_run", NextStepV0::Restore, 4),
        };
        return owned(
            "refused",
            Some(refusal_code),
            false,
            None,
            run,
            Some(next),
            exit_code,
        );
    }
    // Rev 9 L4: `stopped` and control removal only after confirmed exit;
    // a bounded wait that times out preserves every owned control.
    if !wait_for_owned_shutdown(state_dir, owner.pid, CONTROL_TIMEOUT) {
        return owned(
            "refused",
            Some("stop_unconfirmed"),
            false,
            Some("engine stop: exit not confirmed within 5000 ms; owned controls preserved"),
            run,
            Some(NextStepV0::Disable),
            4,
        );
    }
    remove_owned_transient_files(state_dir);
    let next = match run {
        KnownRunV0::Run(_) => NextStepV0::Inspect,
        KnownRunV0::NoRun | KnownRunV0::Unknown => NextStepV0::Done,
    };
    owned("stopped", None, true, None, run, Some(next), 0)
}

/// Human file-cleanup line: the lifecycle label, always prefixed `file cleanup:`
/// so it never reads as protected-evidence cleanup.
#[cfg(not(windows))]
fn file_cleanup_human_line(code: &str) -> &'static str {
    match code {
        "engine_unavailable" => "file cleanup: not run (owning engine unavailable)",
        other => file_cleanup_lifecycle_label(other).unwrap_or("file cleanup: unknown"),
    }
}

#[cfg(test)]
mod tests {
    use super::{format_scan_f64_human, format_scan_score_threshold_human};

    fn assert_formats(cases: &[(&str, u64, &str)]) {
        for &(id, bits, expected) in cases {
            assert_eq!(
                format_scan_f64_human(f64::from_bits(bits)),
                expected,
                "{id} ({bits:#018x})"
            );
        }
    }

    #[test]
    fn scan_f64_human_long_tail_values() {
        assert_formats(&[
            ("LT-ALPHA-CLEAN", 0xbfe4_ab58_e41c_ca90, "-0.64591641"),
            ("LT-ALPHA-FLAGGED", 0x3ff9_7277_e1e3_e03a, "1.59044636"),
            ("LT-CLI-FLAGGED-PROMPT", 0x4004_7520_3178_c8e8, "2.55719031"),
            ("LT-BINARY-01-02", 0x3fd3_3333_3333_3334, "0.3"),
            ("LT-TRAILING-ZERO", 0x3ff3_c083_12b3_4fbd, "1.2345"),
            ("LT-TERM-DOT-ZERO", 0x0000_0000_0000_0000, "0"),
            ("LT-TERM-DOT-ONE", 0x3ff0_0000_0000_0000, "1"),
            ("LT-TERM-DOT-NEG-TWO", 0xc000_0000_0022_5c18, "-2"),
        ]);
        assert_eq!((0.1_f64 + 0.2_f64).to_bits(), 0x3fd3_3333_3333_3334);
        assert_eq!(
            (-0.645_916_409_999_999_9_f64).to_bits(),
            0xbfe4_ab58_e41c_ca90
        );
        assert_eq!(2.557_190_309_999_999_3_f64.to_bits(), 0x4004_7520_3178_c8e8);
    }

    #[test]
    fn scan_f64_human_rounds_half_ties_to_even() {
        assert_formats(&[
            (
                "HT-EVEN-DOWN",
                0x42a0_0000_0000_0001,
                "8796093022208.00195312",
            ),
            ("HT-ODD-UP", 0x42a0_0000_0000_0003, "8796093022208.00585938"),
            ("HT-SMALL-EVEN-DOWN", 0x3f60_0000_0000_0000, "0.00195312"),
            ("HT-SMALL-ODD-UP", 0x3f78_0000_0000_0000, "0.00585938"),
            ("HT-DEC-BELOW", 0x3e5a_d7f2_9abc_af48, "0.00000002"),
            ("HT-DEC-ABOVE", 0x3e62_ca5d_05ea_7ab3, "0.00000004"),
            (
                "HT-NEG-ODD-UP",
                0xc2a0_0000_0000_0003,
                "-8796093022208.00585938",
            ),
        ]);
        assert_eq!(0.000_000_025_f64.to_bits(), 0x3e5a_d7f2_9abc_af48);
        assert_eq!(0.000_000_035_f64.to_bits(), 0x3e62_ca5d_05ea_7ab3);
    }

    #[test]
    fn scan_f64_human_never_emits_negative_zero() {
        assert_formats(&[
            ("NZ-IEEE-NEG-ZERO", 0x8000_0000_0000_0000, "0"),
            ("NZ-IEEE-POS-ZERO", 0x0000_0000_0000_0000, "0"),
            ("NZ-TINY-NEG", 0xbe11_2e0b_e826_d695, "0"),
            ("NZ-TINY-POS", 0x3e11_2e0b_e826_d695, "0"),
        ]);
        assert_eq!((-1e-9_f64).to_bits(), 0xbe11_2e0b_e826_d695);
        // Smallest subnormal and an integer beyond the exact-shift range.
        assert_eq!(format_scan_f64_human(f64::from_bits(1)), "0");
        assert_eq!(format_scan_f64_human(-1e20), "-100000000000000000000");
    }

    #[test]
    fn scan_f64_human_is_total_on_non_finite_values() {
        assert_formats(&[
            ("NF-UNIT-NAN", 0x7ff8_0000_0000_0000, "non_finite_nan"),
            (
                "NF-UNIT-POS-INF",
                0x7ff0_0000_0000_0000,
                "non_finite_pos_inf",
            ),
            (
                "NF-UNIT-NEG-INF",
                0xfff0_0000_0000_0000,
                "non_finite_neg_inf",
            ),
        ]);
    }

    #[test]
    fn scan_score_threshold_human_states_raw_relation_on_collision() {
        let s = f64::from_bits(0xbfe5_34bf_44a3_f3e6);
        let cases: &[(&str, f64, f64, &str, &str)] = &[
            (
                "COL-CLEAN-LT",
                s,
                f64::from_bits(0xbfe5_34bf_441a_8386),
                "-0.66268886 (raw_relation: effective_raw < threshold)",
                "-0.66268886",
            ),
            (
                "COL-FLAGGED-GT",
                s,
                f64::from_bits(0xbfe5_34bf_452d_6445),
                "-0.66268886 (raw_relation: effective_raw > threshold)",
                "-0.66268886",
            ),
            (
                "COL-FLAGGED-EQ",
                s,
                s,
                "-0.66268886 (raw_relation: effective_raw == threshold)",
                "-0.66268886",
            ),
            (
                "COL-ZERO-SIGN-EQ",
                -0.0,
                0.0,
                "0 (raw_relation: effective_raw == threshold)",
                "0",
            ),
            (
                "COL-ZERO-LT",
                -1e-9,
                0.0,
                "0 (raw_relation: effective_raw < threshold)",
                "0",
            ),
            (
                "NO-COLLISION-CLEAN",
                -0.645_916_409_999_999_9,
                0.0,
                "-0.64591641",
                "0",
            ),
            ("NO-COLLISION-FLAGGED", 1.590_446_36, 0.0, "1.59044636", "0"),
        ];
        for &(id, score, threshold, want_score, want_threshold) in cases {
            assert_eq!(
                format_scan_score_threshold_human(score, threshold),
                (want_score.to_owned(), want_threshold.to_owned()),
                "{id}"
            );
        }
        assert_eq!((-0.662_688_859_f64).to_bits(), 0xbfe5_34bf_441a_8386);
        assert_eq!((-0.662_688_861_f64).to_bits(), 0xbfe5_34bf_452d_6445);
    }
}
