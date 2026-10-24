// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Human receipt renderer (Draft 12 section 15.1 / F71), one-to-one state-to-label
//! map (Section 5.4), and byte-exact JSON projection splicer.

#[cfg(not(windows))]
use crate::effect::store::{
    cleanup::CleanupRefusalV0, CleanupResultV0, CleanupStateV0, ReaderErrorV0, TerminalOutcomeV0,
};
#[cfg(not(windows))]
use crate::effect::{FileCleanupResultV0, StoreFailureV0};
use crate::receipt::{
    Availability, Enforcement, Projection, ProjectionVerification, ReplayReason, ReplayVerdict,
    Summary, TargetForm, VerificationOutcome, VerificationReason,
};
use serde::Serialize;

/// Exact machine code for `Summary`.
pub fn summary_code(summary: Summary) -> &'static str {
    match summary {
        Summary::BlockedBeforeExecution => "blocked_before_execution",
        Summary::NotAttemptedUnsupported => "not_attempted_unsupported",
        Summary::NotAttemptedNotEnforced => "not_attempted_not_enforced",
        Summary::NotAttemptedNoExecutor => "not_attempted_no_executor",
        Summary::NotAttemptedSink => "not_attempted_sink",
        Summary::NotAttemptedRejected => "not_attempted_rejected",
        Summary::Incomplete => "incomplete",
        Summary::AllowedEffectVerified => "allowed_effect_verified",
        Summary::AllowedEffectFailed => "allowed_effect_failed",
        Summary::AllowedEffectUnknown => "allowed_effect_unknown",
    }
}

/// Frozen one-to-one human label for `Summary` (Section 5.4).
pub fn summary_label(summary: Summary) -> &'static str {
    match summary {
        Summary::BlockedBeforeExecution => "blocked before execution",
        Summary::NotAttemptedUnsupported => "not attempted: unsupported",
        Summary::NotAttemptedNotEnforced => "not attempted: not enforced",
        Summary::NotAttemptedNoExecutor => "not attempted: no executor",
        Summary::NotAttemptedSink => "not attempted: evidence sink",
        Summary::NotAttemptedRejected => "not attempted: rejected",
        Summary::Incomplete => "incomplete",
        Summary::AllowedEffectVerified => "allowed; effect verified",
        Summary::AllowedEffectFailed => "allowed; effect verification failed",
        Summary::AllowedEffectUnknown => "allowed; effect verification unknown",
    }
}

/// Exact machine code for `Enforcement`.
pub fn enforcement_code(enforcement: Enforcement) -> &'static str {
    match enforcement {
        Enforcement::Enforced => "enforced",
        Enforcement::Shadow => "shadow",
        Enforcement::NotEvaluated => "not_evaluated",
        Enforcement::SuppressedByResponse => "suppressed_by_response",
    }
}

/// Frozen one-to-one human label for `Enforcement` (Section 5.4).
pub fn enforcement_label(enforcement: Enforcement) -> &'static str {
    match enforcement {
        Enforcement::Enforced => "enforcement: enforced",
        Enforcement::Shadow => "enforcement: shadow (block computed, not applied)",
        Enforcement::NotEvaluated => "enforcement: not evaluated",
        Enforcement::SuppressedByResponse => "enforcement: suppressed by response",
    }
}

/// Exact machine code for `Availability`.
pub fn availability_code(availability: Availability) -> &'static str {
    match availability {
        Availability::Present => "present",
        Availability::AbsentByContract => "absent_by_contract",
        Availability::Unavailable => "unavailable",
        Availability::Errored => "errored",
    }
}

/// Exact machine code for `VerificationOutcome`.
pub fn verification_outcome_code(outcome: VerificationOutcome) -> &'static str {
    match outcome {
        VerificationOutcome::Verified => "verified",
        VerificationOutcome::Failed => "failed",
        VerificationOutcome::Unknown => "unknown",
    }
}

/// Frozen one-to-one human label for `VerificationOutcome` (Section 5.4).
pub fn verification_outcome_label(outcome: VerificationOutcome) -> &'static str {
    match outcome {
        VerificationOutcome::Verified => "verification: verified",
        VerificationOutcome::Failed => "verification: failed",
        VerificationOutcome::Unknown => "verification: unknown",
    }
}

/// Frozen one-to-one human label for verification availability and outcome (Section 5.4).
pub fn verification_availability_label(
    availability: Availability,
    outcome: Option<VerificationOutcome>,
    error_code: Option<&str>,
) -> String {
    match availability {
        Availability::Present => match outcome {
            Some(o) => verification_outcome_label(o).to_owned(),
            None => "verification: unknown".to_owned(),
        },
        Availability::AbsentByContract => "verification: absent by contract".to_owned(),
        Availability::Unavailable => {
            let code = error_code.unwrap_or("unavailable");
            format!("verification: unavailable ({code})")
        }
        Availability::Errored => {
            let code = error_code.unwrap_or("errored");
            format!("verification: errored ({code})")
        }
    }
}

/// Exact machine code for `VerificationReason` (its serialized form).
pub fn verification_reason_code(reason: VerificationReason) -> &'static str {
    match reason {
        VerificationReason::AttemptAmbiguous => "attempt_ambiguous",
        VerificationReason::ObservationMissing => "observation_missing",
        VerificationReason::AttemptErrored => "attempt_errored",
        VerificationReason::OutsideRoot => "outside_root",
        VerificationReason::TargetMissing => "target_missing",
        VerificationReason::LengthMismatch => "length_mismatch",
        VerificationReason::DigestMismatch => "digest_mismatch",
        VerificationReason::BeforeStateMissing => "before_state_missing",
        VerificationReason::NoDeltaObserved => "no_delta_observed",
    }
}

/// `verification_error_code` (Rev 9 L5): exactly the projection verification
/// `reason_code`, null only when that field is `None`.
pub fn verification_error_code(v: &ProjectionVerification) -> Option<&'static str> {
    v.reason_code.map(verification_reason_code)
}

/// Frozen one-to-one human label for `ProjectionVerification`.
pub fn projection_verification_label(v: &ProjectionVerification) -> String {
    verification_availability_label(v.availability, v.outcome, None)
}

/// Frozen one-to-one human label for `ReplayVerdict` (Section 5.4).
pub fn replay_verdict_label(verdict: &ReplayVerdict) -> String {
    match verdict {
        ReplayVerdict::Match(_) => "replay: match".to_owned(),
        ReplayVerdict::Fail(_) => "replay: mismatch".to_owned(),
        ReplayVerdict::NotComparable(reason) => match reason {
            ReplayReason::ArgumentsMismatch(check) | ReplayReason::DetectionMismatch(check) => {
                format!(
                    "replay: not comparable ({}:{})",
                    reason.as_str(),
                    check.as_str()
                )
            }
            other => format!("replay: not comparable ({})", other.as_str()),
        },
    }
}

/// Exact machine code for `FileCleanupResultV0`.
#[cfg(not(windows))]
pub fn file_cleanup_code(result: FileCleanupResultV0) -> &'static str {
    match result {
        FileCleanupResultV0::Clear => "clear",
        FileCleanupResultV0::Fail => "fail",
        FileCleanupResultV0::Indeterminate => "indeterminate",
    }
}

/// Frozen one-to-one human label for `FileCleanupResultV0` (Section 5.4).
#[cfg(not(windows))]
pub fn file_cleanup_label(result: FileCleanupResultV0) -> &'static str {
    match result {
        FileCleanupResultV0::Clear => "file cleanup: clear",
        FileCleanupResultV0::Fail => "file cleanup: failed",
        FileCleanupResultV0::Indeterminate => "file cleanup: indeterminate",
    }
}

/// Frozen one-to-one human label for CLI file-cleanup lifecycle states (Section 5.4).
pub fn file_cleanup_lifecycle_label(state: &str) -> Option<&'static str> {
    match state {
        "clear" => Some("file cleanup: clear"),
        "fail" => Some("file cleanup: failed"),
        "indeterminate" => Some("file cleanup: indeterminate"),
        "not_applicable" => Some("file cleanup: not applicable"),
        "not_run" => Some("file cleanup: not run (restore required)"),
        "engine_unavailable" => Some("cleanup: not run (owning engine unavailable)"),
        "not_recorded" => Some("file cleanup: not recorded in protected evidence"),
        _ => None,
    }
}

/// Frozen one-to-one human label for CLI process ownership states (Section 5.4).
pub fn ownership_label(state: &str) -> Option<&'static str> {
    match state {
        "owned" => Some("engine ownership: owned"),
        "not_owned" => Some("engine ownership: not owned; no stop attempted"),
        "unavailable" => Some("engine ownership: unavailable"),
        _ => None,
    }
}

/// Exact machine code for `CleanupResultV0`.
#[cfg(not(windows))]
pub fn evidence_cleanup_result_code(result: CleanupResultV0) -> &'static str {
    match result {
        CleanupResultV0::Clear => "clear",
        CleanupResultV0::Fail => "fail",
        CleanupResultV0::Indeterminate => "indeterminate",
        CleanupResultV0::Refused(CleanupRefusalV0::LockFailed) => "refused_lock_failed",
        CleanupResultV0::Refused(CleanupRefusalV0::ManifestUnusable) => "refused_manifest_unusable",
        CleanupResultV0::Refused(CleanupRefusalV0::TerminalUnsettled) => {
            "refused_terminal_unsettled"
        }
    }
}

/// Frozen one-to-one human label for `CleanupResultV0` (Section 5.4).
#[cfg(not(windows))]
pub fn evidence_cleanup_result_label(result: CleanupResultV0) -> &'static str {
    match result {
        CleanupResultV0::Clear => "evidence cleanup: clear",
        CleanupResultV0::Fail => "evidence cleanup: failed",
        CleanupResultV0::Indeterminate => "evidence cleanup: indeterminate",
        CleanupResultV0::Refused(CleanupRefusalV0::LockFailed) => {
            "evidence cleanup: not started (lock failed)"
        }
        CleanupResultV0::Refused(CleanupRefusalV0::ManifestUnusable) => {
            "evidence cleanup: not started (manifest unusable)"
        }
        CleanupResultV0::Refused(CleanupRefusalV0::TerminalUnsettled) => {
            "evidence cleanup: not started (terminal unsettled)"
        }
    }
}

/// Exact machine code for `CleanupStateV0`.
#[cfg(not(windows))]
pub fn evidence_cleanup_state_code(state: CleanupStateV0) -> &'static str {
    match state {
        CleanupStateV0::NotStarted => "not_started",
        CleanupStateV0::InProgress => "in_progress",
        CleanupStateV0::Clear => "clear",
        CleanupStateV0::Fail => "fail",
        CleanupStateV0::Indeterminate => "indeterminate",
    }
}

/// Frozen one-to-one human label for `CleanupStateV0` (Section 5.4).
#[cfg(not(windows))]
pub fn evidence_cleanup_state_label(state: CleanupStateV0) -> &'static str {
    match state {
        CleanupStateV0::NotStarted => "evidence cleanup: not started",
        CleanupStateV0::InProgress => "evidence cleanup: in progress",
        CleanupStateV0::Clear => "evidence cleanup: clear",
        CleanupStateV0::Fail => "evidence cleanup: failed",
        CleanupStateV0::Indeterminate => "evidence cleanup: indeterminate",
    }
}

/// Frozen one-to-one human label for CLI evidence-cleanup lifecycle states (Section 5.4).
pub fn evidence_cleanup_lifecycle_label(state: &str) -> Option<&'static str> {
    match state {
        "clear" => Some("evidence cleanup: clear"),
        "fail" => Some("evidence cleanup: failed"),
        "indeterminate" => Some("evidence cleanup: indeterminate"),
        "refused_lock_failed" => Some("evidence cleanup: not started (lock failed)"),
        "refused_manifest_unusable" => Some("evidence cleanup: not started (manifest unusable)"),
        "refused_terminal_unsettled" => Some("evidence cleanup: not started (terminal unsettled)"),
        "not_applicable" => Some("evidence cleanup: not applicable"),
        "not_started" => Some("evidence cleanup: not started"),
        "in_progress" => Some("evidence cleanup: in progress"),
        "unknown" => Some("evidence cleanup: unknown (not changed by this command)"),
        _ => None,
    }
}

/// The next valid operator command after a lifecycle step. Commands name the
/// state directory by placeholder only, so human output never echoes a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextStepV0 {
    /// Send the one file-effect request, then restore.
    RequestThenRestore,
    /// Stop a proxy-only engine.
    StopProxy,
    Restore,
    Disable,
    Inspect,
    EvidenceRelease,
    /// Indeterminate file cleanup: undeclared residue needs operator review.
    ReviewEffectRoot,
    Done,
}

/// Frozen human `next:` line for a `NextStepV0`; `<run-id>` when the run is unknown.
pub fn next_step_line(step: NextStepV0, run_id: Option<&str>) -> String {
    let run = run_id.unwrap_or("<run-id>");
    match step {
        NextStepV0::RequestThenRestore => {
            "next: send one request, then parapet restore --state-dir <state-dir>".to_owned()
        }
        NextStepV0::StopProxy => "next: parapet disable --state-dir <state-dir>".to_owned(),
        NextStepV0::Restore => "next: parapet restore --state-dir <state-dir>".to_owned(),
        NextStepV0::Disable => "next: parapet disable --state-dir <state-dir>".to_owned(),
        NextStepV0::Inspect => {
            format!("next: parapet inspect --state-dir <state-dir> --run-id {run}")
        }
        NextStepV0::EvidenceRelease => format!(
            "next: parapet evidence release --state-dir <state-dir> --run-id {run} (optional)"
        ),
        NextStepV0::ReviewEffectRoot => "next: remove undeclared files from the effect root, then parapet restore --state-dir <state-dir>".to_owned(),
        NextStepV0::Done => "next: none".to_owned(),
    }
}

/// Human startup status for `parapet run`, flushed once the engine is ready.
/// Carries only the loopback port, the engine-generated run ID, and the mode.
pub fn render_run_startup_human(port: u16, run_id: Option<&str>) -> String {
    let (mode, next) = match run_id {
        Some(_) => ("protected", NextStepV0::RequestThenRestore),
        None => ("proxy", NextStepV0::StopProxy),
    };
    format!(
        "status: ready\nport: {port}\nrun_id: {}\nmode: {mode}\n{}\n",
        run_id.unwrap_or("not_applicable"),
        next_step_line(next, run_id)
    )
}

/// Human `original decision:` line: the receipt summary a cleanup never changes.
#[cfg(not(windows))]
pub fn original_decision_line(
    projection_bytes: Option<&[u8]>,
    reader_code: Option<ReaderErrorV0>,
    projection_code: Option<ReaderErrorV0>,
) -> String {
    // A reader error withholds authority, so no projection field is shown.
    if reader_code.is_none() {
        if let Some(projection) =
            projection_bytes.and_then(|b| serde_json::from_slice::<Projection>(b).ok())
        {
            return format!("original decision: {}", summary_label(projection.summary));
        }
    }
    match reader_code.or(projection_code) {
        // No manifest: the engine has not recorded a request for this run.
        Some(ReaderErrorV0::RunManifestMissing) => "original decision: not recorded".to_owned(),
        // The run is open: its terminal decision has not been published.
        Some(code @ (ReaderErrorV0::TerminalMissing | ReaderErrorV0::TerminalPending)) => {
            format!("original decision: not settled ({})", code.as_str())
        }
        Some(code) => format!("original decision: not available ({})", code.as_str()),
        None => "original decision: not available".to_owned(),
    }
}

/// Human `evidence cleanup:` line for a redacted run view.
#[cfg(not(windows))]
pub fn evidence_cleanup_view_label(
    cleanup: Option<CleanupStateV0>,
    reader_code: Option<ReaderErrorV0>,
) -> &'static str {
    match (cleanup, reader_code) {
        (Some(state), _) => evidence_cleanup_state_label(state),
        (None, Some(ReaderErrorV0::RunManifestMissing)) => "evidence cleanup: not applicable",
        (None, _) => "evidence cleanup: unknown (not changed by this command)",
    }
}

/// `original decision:` line when the engine has no file-effect run.
pub const ORIGINAL_DECISION_NO_RUN: &str = "original decision: not applicable (no file-effect run)";

/// `original decision:` line when no run could be identified.
pub const ORIGINAL_DECISION_UNKNOWN: &str = "original decision: unknown (run not identified)";

/// Lifecycle block shared by human `restore`, `disable`, `inspect`, and
/// `evidence release`: the original decision, the current file-effect cleanup,
/// and the protected-evidence cleanup are always three separate lines.
pub struct LifecycleHumanV0<'a> {
    pub run_id: Option<&'a str>,
    pub decision: &'a str,
    pub file_cleanup: &'a str,
    pub evidence_cleanup: &'a str,
}

impl LifecycleHumanV0<'_> {
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(4);
        if let Some(run) = self.run_id {
            lines.push(format!("run_id: {run}"));
        }
        lines.push(self.decision.to_owned());
        lines.push(self.file_cleanup.to_owned());
        lines.push(self.evidence_cleanup.to_owned());
        lines
    }
}

/// Join human lines with a trailing newline.
pub fn join_lines(lines: &[String]) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// Next step after `parapet inspect`, when one is known.
#[cfg(not(windows))]
pub fn inspect_next_step(
    outcome: Option<TerminalOutcomeV0>,
    cleanup: Option<CleanupStateV0>,
    reader_code: Option<ReaderErrorV0>,
) -> Option<NextStepV0> {
    match (cleanup, reader_code) {
        (Some(CleanupStateV0::Clear), _) => Some(NextStepV0::Done),
        (Some(CleanupStateV0::NotStarted), None)
            if outcome == Some(TerminalOutcomeV0::Complete) =>
        {
            Some(NextStepV0::EvidenceRelease)
        }
        (Some(CleanupStateV0::Fail | CleanupStateV0::Indeterminate), _) => {
            Some(NextStepV0::EvidenceRelease)
        }
        _ => None,
    }
}

/// Exact snake_case code for `StoreFailureV0` (Sensor N7).
#[cfg(not(windows))]
pub fn store_failure_code(failure: StoreFailureV0) -> &'static str {
    match failure {
        StoreFailureV0::UnsafeLayout => "unsafe_layout",
        StoreFailureV0::UnsafeObject => "unsafe_object",
        StoreFailureV0::Collision => "collision",
        StoreFailureV0::DigestMismatch => "digest_mismatch",
        StoreFailureV0::LengthMismatch => "length_mismatch",
        StoreFailureV0::LabelNotConfigured => "label_not_configured",
        StoreFailureV0::ManifestMissing => "manifest_missing",
        StoreFailureV0::ManifestCorrupt => "manifest_corrupt",
        StoreFailureV0::StateInvalid => "state_invalid",
        StoreFailureV0::LockFailed => "lock_failed",
        StoreFailureV0::Io => "io",
        StoreFailureV0::Injected => "injected",
    }
}

/// Frozen one-to-one human label for `StoreFailureV0` (Sensor N7, Section 5.4).
#[cfg(not(windows))]
pub fn store_failure_label(failure: StoreFailureV0) -> &'static str {
    match failure {
        StoreFailureV0::UnsafeLayout => "evidence store open failed: unsafe_layout",
        StoreFailureV0::UnsafeObject => "evidence store open failed: unsafe_object",
        StoreFailureV0::Collision => "evidence store open failed: collision",
        StoreFailureV0::DigestMismatch => "evidence store open failed: digest_mismatch",
        StoreFailureV0::LengthMismatch => "evidence store open failed: length_mismatch",
        StoreFailureV0::LabelNotConfigured => "evidence store open failed: label_not_configured",
        StoreFailureV0::ManifestMissing => "evidence store open failed: manifest_missing",
        StoreFailureV0::ManifestCorrupt => "evidence store open failed: manifest_corrupt",
        StoreFailureV0::StateInvalid => "evidence store open failed: state_invalid",
        StoreFailureV0::LockFailed => "evidence store open failed: lock_failed",
        StoreFailureV0::Io => "evidence store open failed: io",
        StoreFailureV0::Injected => "evidence store open failed: injected",
    }
}

/// Exact code for `ReaderErrorV0`.
#[cfg(not(windows))]
pub fn reader_error_code(error: ReaderErrorV0) -> &'static str {
    error.as_str()
}

/// Frozen one-to-one human label for `ReaderErrorV0` (Section 5.4).
#[cfg(not(windows))]
pub fn reader_error_label(error: ReaderErrorV0) -> &'static str {
    match error {
        ReaderErrorV0::RunManifestMissing => "reader: run_manifest_missing",
        ReaderErrorV0::RunManifestCorrupt => "reader: run_manifest_corrupt",
        ReaderErrorV0::RefNotListed => "reader: ref_not_listed",
        ReaderErrorV0::ObjectPending => "reader: object_pending",
        ReaderErrorV0::ObjectReleased => "reader: object_released",
        ReaderErrorV0::ObjectMissing => "reader: object_missing",
        ReaderErrorV0::ObjectDigestMismatch => "reader: object_digest_mismatch",
        ReaderErrorV0::ObjectLengthMismatch => "reader: object_length_mismatch",
        ReaderErrorV0::TerminalPending => "reader: terminal_pending",
        ReaderErrorV0::TerminalMissing => "reader: terminal_missing",
        ReaderErrorV0::TerminalCorrupt => "reader: terminal_corrupt",
        ReaderErrorV0::TerminalBindingMismatch => "reader: terminal_binding_mismatch",
        ReaderErrorV0::TerminalStatusMismatch => "reader: terminal_status_mismatch",
        ReaderErrorV0::CleanupInProgress => "reader: cleanup_in_progress",
        ReaderErrorV0::StorageIncomplete => "reader: storage_incomplete",
    }
}

/// Frozen one-to-one human label for a redacted `projection_code` (Rev 9 D2-C4).
#[cfg(not(windows))]
pub fn projection_error_label(error: ReaderErrorV0) -> &'static str {
    match error {
        ReaderErrorV0::RunManifestMissing => "projection: run_manifest_missing",
        ReaderErrorV0::RunManifestCorrupt => "projection: run_manifest_corrupt",
        ReaderErrorV0::RefNotListed => "projection: ref_not_listed",
        ReaderErrorV0::ObjectPending => "projection: object_pending",
        ReaderErrorV0::ObjectReleased => "projection: object_released",
        ReaderErrorV0::ObjectMissing => "projection: object_missing",
        ReaderErrorV0::ObjectDigestMismatch => "projection: object_digest_mismatch",
        ReaderErrorV0::ObjectLengthMismatch => "projection: object_length_mismatch",
        ReaderErrorV0::TerminalPending => "projection: terminal_pending",
        ReaderErrorV0::TerminalMissing => "projection: terminal_missing",
        ReaderErrorV0::TerminalCorrupt => "projection: terminal_corrupt",
        ReaderErrorV0::TerminalBindingMismatch => "projection: terminal_binding_mismatch",
        ReaderErrorV0::TerminalStatusMismatch => "projection: terminal_status_mismatch",
        ReaderErrorV0::CleanupInProgress => "projection: cleanup_in_progress",
        ReaderErrorV0::StorageIncomplete => "projection: storage_incomplete",
    }
}

fn is_bidi_or_format_control(c: char) -> bool {
    matches!(
        c as u32,
        0x0080..=0x009F
            | 0x061C
            | 0x200B..=0x200F
            | 0x2028..=0x202E
            | 0x2060..=0x206F
            | 0xFEFF
            | 0xFFF9..=0xFFFB
    ) || c.is_control()
}

/// Draft 12 section 15.1 / F71 deterministic inert plaintext sanitizer.
///
/// Replaces any ASCII control byte (`< 0x20` or `0x7f`, including `0x1b` ESC,
/// `\n`, `\r`, `\t`, NUL) with `\xHH` and any non-ASCII Unicode control /
/// bidi / format character with `\u{HHHH}`. Printable ASCII characters
/// (including Markdown/HTML punctuation `*`, `_`, `<`, `>`, `/`, `[`, `]`)
/// are emitted verbatim as literal plaintext.
pub fn render_inert_plaintext(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let code = c as u32;
        if code < 0x20 || code == 0x7f {
            out.push_str(&format!("\\x{:02x}", code));
        } else if is_bidi_or_format_control(c) {
            out.push_str(&format!("\\u{{{:04x}}}", code));
        } else {
            out.push(c);
        }
    }
    out
}

/// Render a parsed `Projection` into human-readable lines per Section 5.3 / F71.
pub fn render_projection_human_lines(projection: &Projection, inert_fallback: bool) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("receipt_row_digest: {}", projection.projection_of));
    lines.push(format!("receipt_line: {}", projection.line));
    lines.push(format!(
        "summary_label: {}",
        summary_label(projection.summary)
    ));
    lines.push(format!(
        "enforcement_label: {}",
        enforcement_label(projection.policy.enforcement)
    ));
    lines.push(format!(
        "verification_label: {}",
        projection_verification_label(&projection.verification)
    ));
    if !inert_fallback {
        lines.push(format!(
            "tool_name.text: {}",
            render_inert_plaintext(&projection.tool_name.text)
        ));
    }
    lines.push(format!(
        "tool_name.len_bytes: {}",
        projection.tool_name.len_bytes
    ));
    lines.push(format!(
        "tool_name.truncated: {}",
        projection.tool_name.truncated
    ));
    match projection.target.form {
        TargetForm::UnresolvedBounded => {
            lines.push("target.form: unresolved_bounded [unverified]".to_owned());
            if !inert_fallback {
                lines.push(format!(
                    "target.text [unverified]: {}",
                    render_inert_plaintext(&projection.target.text)
                ));
            }
        }
        TargetForm::Resolved => {
            lines.push("target.form: resolved".to_owned());
            if !inert_fallback {
                lines.push(format!(
                    "target.text: {}",
                    render_inert_plaintext(&projection.target.text)
                ));
            }
        }
    }
    lines.push(format!(
        "target.digest: {}",
        projection.target.digest.as_deref().unwrap_or("none")
    ));
    lines.push(format!("target.len_bytes: {}", projection.target.len_bytes));
    lines.push(format!("target.truncated: {}", projection.target.truncated));
    lines
}

/// Render full `parapet inspect` human output (Section 5.3).
#[cfg(not(windows))]
pub fn render_inspect_human(
    run_id: &str,
    outcome: Option<TerminalOutcomeV0>,
    cleanup: Option<CleanupStateV0>,
    banner: Option<&str>,
    reader_code: Option<ReaderErrorV0>,
    projection_code: Option<ReaderErrorV0>,
    projection_bytes: Option<&[u8]>,
    inert_fallback: bool,
) -> String {
    let outcome_str = match outcome {
        Some(TerminalOutcomeV0::Complete) => "complete",
        Some(TerminalOutcomeV0::StorageIncomplete) => "storage_incomplete",
        None => "none",
    };
    let decision = original_decision_line(projection_bytes, reader_code, projection_code);
    let evidence_cleanup = evidence_cleanup_view_label(cleanup, reader_code);
    let mut lines = LifecycleHumanV0 {
        run_id: Some(run_id),
        decision: &decision,
        file_cleanup: "file cleanup: not recorded in protected evidence",
        evidence_cleanup,
    }
    .lines();
    lines.insert(1, format!("terminal_outcome: {outcome_str}"));
    if let Some(b) = banner {
        lines.push(format!("banner: {b}"));
    }
    if let Some(code) = reader_code {
        lines.push(reader_error_label(code).to_owned());
    }
    if let Some(code) = projection_code {
        lines.push(projection_error_label(code).to_owned());
    }
    if reader_code.is_none() {
        if let Some(bytes) = projection_bytes {
            if let Ok(projection) = serde_json::from_slice::<Projection>(bytes) {
                lines.extend(render_projection_human_lines(&projection, inert_fallback));
            }
        }
    }
    if let Some(step) = inspect_next_step(outcome, cleanup, reader_code) {
        lines.push(next_step_line(step, Some(run_id)));
    }
    join_lines(&lines)
}

#[derive(Serialize)]
struct InspectLabels<'a> {
    summary: Option<&'a str>,
    enforcement: Option<&'a str>,
    verification: Option<String>,
    evidence_cleanup: Option<&'a str>,
    reader: Option<&'a str>,
    // Not `projection`: that key is reserved for the spliced projection bytes.
    projection_code: Option<&'a str>,
}

#[derive(Serialize)]
struct InspectPrefix<'a> {
    schema: &'static str,
    run_id: &'a str,
    outcome: Option<&'static str>,
    cleanup: Option<&'static str>,
    banner: Option<&'a str>,
    reader_code: Option<&'static str>,
    projection_code: Option<&'static str>,
    labels: InspectLabels<'a>,
}

/// Construct `parapet-cli-inspect/v0` JSON bytes by splicing `projection_bytes`
/// verbatim as raw bytes after `"projection":` without reserializing (Section 5.2.6).
#[cfg(not(windows))]
pub fn splice_inspect_json(
    run_id: &str,
    outcome: Option<TerminalOutcomeV0>,
    cleanup: Option<CleanupStateV0>,
    banner: Option<&str>,
    reader_code: Option<ReaderErrorV0>,
    projection_code: Option<ReaderErrorV0>,
    projection_bytes: Option<&[u8]>,
) -> Vec<u8> {
    let outcome_str = match outcome {
        Some(TerminalOutcomeV0::Complete) => Some("complete"),
        Some(TerminalOutcomeV0::StorageIncomplete) => Some("storage_incomplete"),
        None => None,
    };
    let cleanup_str = cleanup.map(evidence_cleanup_state_code);
    let parsed_projection =
        projection_bytes.and_then(|b| serde_json::from_slice::<Projection>(b).ok());
    let summary_lbl = parsed_projection.as_ref().map(|p| summary_label(p.summary));
    let enforcement_lbl = parsed_projection
        .as_ref()
        .map(|p| enforcement_label(p.policy.enforcement));
    let verification_lbl = parsed_projection
        .as_ref()
        .map(|p| projection_verification_label(&p.verification));
    let evidence_cleanup_lbl = cleanup.map(evidence_cleanup_state_label);
    let reader_lbl = reader_code.map(reader_error_label);
    let projection_lbl = projection_code.map(projection_error_label);

    let prefix = InspectPrefix {
        schema: "parapet-cli-inspect/v0",
        run_id,
        outcome: outcome_str,
        cleanup: cleanup_str,
        banner,
        reader_code: reader_code.map(reader_error_code),
        projection_code: projection_code.map(reader_error_code),
        labels: InspectLabels {
            summary: summary_lbl,
            enforcement: enforcement_lbl,
            verification: verification_lbl,
            evidence_cleanup: evidence_cleanup_lbl,
            reader: reader_lbl,
            projection_code: projection_lbl,
        },
    };
    let mut out = serde_json::to_vec(&prefix).expect("inspect prefix serializes");
    // Replace trailing '}' with ',"projection":' and splice raw projection bytes verbatim.
    let popped = out.pop();
    debug_assert_eq!(popped, Some(b'}'));
    out.extend_from_slice(b",\"projection\":");
    match projection_bytes {
        Some(raw) => out.extend_from_slice(raw),
        None => out.extend_from_slice(b"null"),
    }
    out.extend_from_slice(b"}\n");
    out
}
