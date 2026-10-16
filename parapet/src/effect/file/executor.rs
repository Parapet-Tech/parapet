// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `FileActionExecutorV0`: total first-match preflight, then
//! `atomic_write_no_follow/v0` (1C brief b76e6b1b sections 6.1, 6.2 and 7, as
//! corrected by brief f17b49a6 sections 3 and 4; draft 12 sections 10 and 10.1).

use super::evidence::persist_payload;
use super::path::{classify_raw_path, BASENAME_MAX_LEN, BASENAME_PATTERN};
use super::root::{read_target, FileFaultV0, FilePhaseV0, RootCapabilityV0, TargetReadV0};
use crate::effect::interfaces::{
    ActionExecutorV0, ActionIdV0, ExecutorRequestV0, ResolvedPolicyV0,
};
use crate::effect::policy::policy_enforced;
use crate::effect::sink::sink_precedence;
use crate::receipt::*;
use rustix::fs::{fstat, openat, renameat, statat, unlinkat, AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

/// Draft 12 section 10 field 5 version owned by 1C.
pub const EXECUTOR_VERSION: &str = "v0";

const TEMP_PREFIX: &str = ".parapet-tmp-v0-";
const TEMP_TAG: &[u8] = b"parapet-file-temp/v0\n";

/// Closed mandatory-profile construction failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileErrorV0 {
    /// `policy.layers.l3_outbound` is absent or its mode is not `block`.
    NotEnforced,
    /// `write_file` is not exactly the mandatory first-use profile.
    ProfileMismatch,
}

fn string_constraint(matches: Option<&str>, max_length: u64) -> ResolvedConstraints {
    ResolvedConstraints {
        type_check: Some(TypeCheck::String),
        starts_with: None,
        not_contains: None,
        matches: matches.map(str::to_owned),
        one_of: None,
        max_length: Some(max_length),
        min: None,
        max: None,
        url_host: None,
    }
}

/// The exact resolved `write_file` profile for a content limit.
pub fn mandatory_write_file_tool(max_content_len_bytes: u64) -> ResolvedTool {
    ResolvedTool {
        allowed: true,
        constraints: BTreeMap::from([
            (
                "content".to_owned(),
                string_constraint(None, max_content_len_bytes),
            ),
            (
                "path".to_owned(),
                string_constraint(Some(BASENAME_PATTERN), BASENAME_MAX_LEN),
            ),
        ]),
    }
}

/// The mandatory profile cannot be relaxed: L3 enforces and `write_file` is exact.
pub fn check_mandatory_profile(
    policy: &ResolvedActionPolicyV0,
    max_content_len_bytes: u64,
) -> Result<(), ProfileErrorV0> {
    if !policy_enforced(policy) {
        return Err(ProfileErrorV0::NotEnforced);
    }
    if policy.tools.get("write_file") != Some(&mandatory_write_file_tool(max_content_len_bytes)) {
        return Err(ProfileErrorV0::ProfileMismatch);
    }
    Ok(())
}

/// The concrete file controller bound to one root and one resolved policy.
pub struct FileActionExecutorV0 {
    root: Arc<RootCapabilityV0>,
    contract_hash: String,
    policy_digest: String,
}

impl std::fmt::Debug for FileActionExecutorV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileActionExecutorV0")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl FileActionExecutorV0 {
    /// Bind the controller to a live root and the resolved mandatory profile.
    pub fn new(
        root: Arc<RootCapabilityV0>,
        policy: &ResolvedPolicyV0,
    ) -> Result<Self, ProfileErrorV0> {
        check_mandatory_profile(&policy.policy, root.max_content_len_bytes())?;
        Ok(Self {
            root,
            contract_hash: policy.contract_hash.clone(),
            policy_digest: policy.digest.clone(),
        })
    }

    fn rejected(&self, req: &ExecutorRequestV0<'_>, code: AttemptError) -> Attempt {
        Attempt {
            schema_version: ReceiptSchema::V0,
            stage: AttemptStage::Attempt,
            decision_ref: req.decision_ref.to_owned(),
            action_id: req.action_id.as_str().to_owned(),
            executor: Executor {
                name: ExecutorName::FileController,
                version: EXECUTOR_VERSION.to_owned(),
            },
            root: self.root.root(),
            target: Target {
                proposed_path_digest: proposed_path_digest(&req.expected_effect.target_path),
                target_ref: PresentUnavailable::Unavailable(TargetError::NotResolved),
                resolved_path: PresentUnavailable::Unavailable(TargetError::NotResolved),
            },
            operation: Operation::AtomicWriteNoFollowV0,
            sink: req.sink.clone(),
            policy_check: PolicyCheck {
                contract_hash: self.contract_hash.clone(),
                policy_digest: self.policy_digest.clone(),
            },
            before_state: PresentUnavailableErroredAbsent::AbsentByContract,
            outcome: AttemptOutcome::Rejected,
            error_code: Some(code),
            content_payload: PresentUnavailableAbsent::AbsentByContract,
        }
    }

    // Section 6.1 rows 1-11 in first-match order; `Ok` carries the resolved basename.
    fn preflight<'r>(&self, req: &'r ExecutorRequestV0<'_>) -> Result<&'r str, Attempt> {
        if let Some(code) = sink_precedence(req.sink, req.expected_refs) {
            return Err(self.rejected(req, code));
        }
        if req.decision.policy.contract_hash != self.contract_hash
            || req.decision.policy.policy_digest != self.policy_digest
        {
            return Err(self.rejected(req, AttemptError::PolicyDrift));
        }
        // The first use reserves the id; no later outcome releases it.
        if !self.root.reserve(req.action_id) {
            return Err(self.rejected(req, AttemptError::ActionIdCollision));
        }
        let supplied = RootCapabilityV0::from_opaque(req.root);
        if !supplied.is_some_and(|r| r.same(&self.root)) || !self.root.is_live() {
            return Err(self.rejected(req, AttemptError::RootUnavailable));
        }
        let basename = classify_raw_path(&req.expected_effect.target_path)
            .map_err(|row| self.rejected(req, row.code()))?;
        let resolution = self
            .root
            .with_fd(|fd| statat(fd, basename, AtFlags::SYMLINK_NOFOLLOW).map(|s| s.st_mode));
        match resolution {
            None => return Err(self.rejected(req, AttemptError::RootUnavailable)),
            Some(Ok(mode)) if FileType::from_raw_mode(mode) == FileType::Symlink => {
                return Err(self.rejected(req, AttemptError::SymlinkComponent))
            }
            Some(_) => {}
        }
        if req.content.len() as u64 > self.root.max_content_len_bytes() {
            let mut attempt = self.rejected(req, AttemptError::SizeLimit);
            attempt.target = self.target(req, basename);
            return Err(attempt);
        }
        Ok(basename)
    }

    fn target(&self, req: &ExecutorRequestV0<'_>, basename: &str) -> Target {
        Target {
            proposed_path_digest: proposed_path_digest(&req.expected_effect.target_path),
            target_ref: PresentUnavailable::Present(self.root.target_ref_for(basename)),
            resolved_path: PresentUnavailable::Present(basename.to_owned()),
        }
    }

    fn pre_read(
        &self,
        basename: &str,
    ) -> PresentUnavailableErroredAbsent<BeforeState, BeforeStateError> {
        let read = self.root.with_fd(|fd| {
            self.root.count_target_call();
            if self.root.fault(FileFaultV0::PreReadNoAccess) {
                return TargetReadV0::NoAccess;
            }
            if self.root.fault(FileFaultV0::PreReadIo) {
                return TargetReadV0::Error;
            }
            read_target(fd, basename)
        });
        match read {
            Some(TargetReadV0::Missing) => PresentUnavailableErroredAbsent::Present(BeforeState {
                exists: false,
                digest: None,
                len_bytes: None,
            }),
            Some(TargetReadV0::Bytes(bytes)) => {
                PresentUnavailableErroredAbsent::Present(BeforeState {
                    exists: true,
                    digest: Some(payload_digest(&bytes)),
                    len_bytes: Some(bytes.len() as u64),
                })
            }
            Some(TargetReadV0::NoAccess) => {
                PresentUnavailableErroredAbsent::Unavailable(BeforeStateError::NoReadAccess)
            }
            _ => PresentUnavailableErroredAbsent::Errored(BeforeStateError::IoError),
        }
    }

    // Root nonce, action id and fresh randomness, hashed: the name discloses none.
    fn temp_name(&self, action_id: &ActionIdV0) -> String {
        let mut hash = Sha256::new();
        hash.update(TEMP_TAG);
        hash.update(self.root.nonce());
        hash.update([0x00]);
        hash.update(action_id.as_str().as_bytes());
        hash.update([0x00]);
        hash.update(uuid::Uuid::new_v4().as_bytes());
        let digest = format!("{:x}", hash.finalize());
        format!("{TEMP_PREFIX}{}", &digest[..32])
    }

    // Section 6.2 steps 3-6; the outcome follows the observed phase (Sensor C1).
    fn write_and_rename(
        &self,
        req: &ExecutorRequestV0<'_>,
        basename: &str,
    ) -> (AttemptOutcome, Option<AttemptError>) {
        let errored = |code| (AttemptOutcome::Errored, Some(code));
        let temp = self.temp_name(req.action_id);
        self.root.declare(&temp);
        let fault = |f| self.root.fault(f);
        let done = self.root.with_fd(|fd| {
            let discard = || {
                let _ = unlinkat(fd, temp.as_str(), AtFlags::empty());
            };
            self.root.count_target_call();
            if fault(FileFaultV0::TempCreate) {
                return errored(AttemptError::IoError);
            }
            let flags =
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            let Ok(created) = openat(fd, temp.as_str(), flags, Mode::from_raw_mode(0o600)) else {
                return errored(AttemptError::IoError);
            };
            let mut file = File::from(created);
            self.root.count_target_call();
            let bytes = if fault(FileFaultV0::PartialWrite) {
                &req.content[..req.content.len() / 2]
            } else {
                req.content
            };
            let written = if fault(FileFaultV0::WriteError) {
                Err(std::io::Error::from(std::io::ErrorKind::Other))
            } else {
                file.write_all(bytes)
            };
            let length_ok =
                fstat(&file).is_ok_and(|s| s.st_size as u64 == req.content.len() as u64);
            if written.is_err() || !length_ok {
                discard();
                return errored(AttemptError::IoError);
            }
            if fault(FileFaultV0::InterruptBeforeRename) {
                discard();
                return errored(AttemptError::Interrupted);
            }
            self.root.count_target_call();
            if fault(FileFaultV0::Fsync) || file.sync_all().is_err() {
                discard();
                return errored(AttemptError::IoError);
            }
            drop(file);
            self.root.phase(FilePhaseV0::BeforeRename);
            self.root.count_target_call();
            if fault(FileFaultV0::RenameError) {
                discard();
                return errored(AttemptError::IoError);
            }
            if fault(FileFaultV0::RenameInterrupted) {
                return (AttemptOutcome::Ambiguous, Some(AttemptError::Interrupted));
            }
            // Source and destination are both relative to the retained fd.
            match renameat(fd, temp.as_str(), fd, basename) {
                Ok(()) if fault(FileFaultV0::RenameUnconfirmed) => (
                    AttemptOutcome::Ambiguous,
                    Some(AttemptError::RenameUnconfirmed),
                ),
                Ok(()) => (AttemptOutcome::Completed, None),
                Err(Errno::INTR) => (AttemptOutcome::Ambiguous, Some(AttemptError::Interrupted)),
                Err(_) => {
                    discard();
                    errored(AttemptError::IoError)
                }
            }
        });
        done.unwrap_or((AttemptOutcome::Errored, Some(AttemptError::IoError)))
    }

    fn operate(&self, req: &ExecutorRequestV0<'_>, basename: &str) -> Attempt {
        let mut attempt = self.rejected(req, AttemptError::IoError);
        attempt.target = self.target(req, basename);
        attempt.outcome = AttemptOutcome::Errored;
        attempt.content_payload =
            PresentUnavailableAbsent::Unavailable(StoreError::StoreNotConfigured);
        attempt.before_state = self.pre_read(basename);
        if !matches!(
            attempt.before_state,
            PresentUnavailableErroredAbsent::Present(_)
        ) {
            return attempt;
        }
        let payload = if self.root.fault(FileFaultV0::EvidencePersist) {
            None
        } else {
            persist_payload(
                &self.root,
                req.content,
                &req.expected_effect.content_media_type,
            )
        };
        let Some(payload) = payload else {
            return attempt;
        };
        attempt.content_payload = PresentUnavailableAbsent::Present(payload);
        let (outcome, code) = self.write_and_rename(req, basename);
        attempt.outcome = outcome;
        attempt.error_code = code;
        attempt
    }
}

impl ActionExecutorV0 for FileActionExecutorV0 {
    fn execute(&self, req: &ExecutorRequestV0<'_>) -> Attempt {
        let attempt = match self.preflight(req) {
            Err(rejected) => rejected,
            Ok(basename) => {
                self.root.declare(basename);
                self.root.phase(FilePhaseV0::AfterResolution);
                self.operate(req, basename)
            }
        };
        tracing::debug!(
            schema = "parapet-file-controller/v0",
            outcome = ?attempt.outcome,
            error_code = ?attempt.error_code,
            "file effect attempt settled"
        );
        attempt
    }
}
