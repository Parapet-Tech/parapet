// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Immutable terminal snapshots `parapet-runtime-terminal/v0`
//! (Routing contract 50747a52, section 4.3 steps 3, 4 and 7).

use super::layout::{split, terminal_path};
use super::manifest::{PendingTerminalV0, RunManifestV0, RunStateV0, TerminalPointerV0};
use super::object::{raw_digest, write_immutable};
use super::{FaultPointV0, ManifestRead, ProtectedEvidenceStoreV0};
use crate::effect::interfaces::{CandidateRefV0, RunIdV0, StoreFailureV0};
use crate::receipt::{PresentUnavailable, RequestRef, RequestRefError};
use serde::{Deserialize, Deserializer, Serialize};

/// Terminal schema constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalSchemaV0 {
    /// `parapet-runtime-terminal/v0`.
    #[serde(rename = "parapet-runtime-terminal/v0")]
    V0,
}

/// Section 4.3: workflow outcome, monotone across generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcomeV0 {
    /// Every terminal record, row, digest and projection persisted.
    Complete,
    /// Permanent storage failure.
    StorageIncomplete,
}

/// Section 4.3: cleanup state carried by a terminal snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupStateV0 {
    /// No cleanup attempted.
    NotStarted,
    /// Cleanup preflight passed; releases in progress.
    InProgress,
    /// Every target entry released and every unshared object absent.
    Clear,
    /// Cleanup failed.
    Fail,
    /// Cleanup could not establish a result.
    Indeterminate,
}

fn required<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}

/// Section 4.3: exact ordered terminal fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTerminalV0 {
    /// Schema constant.
    pub schema: TerminalSchemaV0,
    /// Run id.
    pub run_id: RunIdV0,
    /// Draft-12 request reference, verbatim.
    pub request_ref: PresentUnavailable<RequestRef, RequestRefError>,
    /// Exact formed coordinate.
    pub candidate_ref: CandidateRefV0,
    /// Raw sha256 of the prior authoritative snapshot; null only initially.
    #[serde(deserialize_with = "required")]
    pub previous_terminal_digest: Option<String>,
    /// Workflow outcome.
    pub outcome: TerminalOutcomeV0,
    /// Landed digest of the run's single row.
    pub row_digest: String,
    /// Cleanup state.
    pub cleanup: CleanupStateV0,
}

impl RuntimeTerminalV0 {
    /// Compact bytes in field order.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("closed terminal serializes")
    }

    /// Parse canonical terminal bytes.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let terminal: Self = serde_json::from_slice(bytes).ok()?;
        (terminal.to_bytes() == bytes).then_some(terminal)
    }

    fn initial(manifest: &RunManifestV0, outcome: TerminalOutcomeV0, row_digest: &str) -> Self {
        Self {
            schema: TerminalSchemaV0::V0,
            run_id: manifest.run_id.clone(),
            request_ref: manifest.request_ref.clone(),
            candidate_ref: manifest.candidate_ref,
            previous_terminal_digest: None,
            outcome,
            row_digest: row_digest.to_owned(),
            cleanup: CleanupStateV0::NotStarted,
        }
    }
}

/// Recovery result for a run found without a settled terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryOutcomeV0 {
    /// An authoritative terminal already exists; nothing changed.
    AlreadySettled(TerminalOutcomeV0),
    /// The exact pending initial `complete` bytes were re-published.
    ResumedComplete,
    /// A `storage_incomplete` terminal was published.
    PublishedStorageIncomplete,
    /// No row digest is attributable; no terminal can be bound.
    Unbindable,
    /// The manifest is missing or corrupt.
    ManifestUnusable,
}

impl ProtectedEvidenceStoreV0 {
    /// Read and parse the authoritative terminal named by `pointer`.
    pub(crate) fn read_terminal_locked(
        &self,
        pointer_path: &str,
        digest: &str,
        len: u64,
    ) -> Option<RuntimeTerminalV0> {
        let bytes = self.read_at(pointer_path).ok()??;
        if raw_digest(&bytes) != digest || bytes.len() as u64 != len {
            return None;
        }
        RuntimeTerminalV0::parse(&bytes)
    }

    /// Section 4.3 step 3 under the lock: pending pointer, immutable write,
    /// re-read, then pointer move into `next_state`.
    pub(crate) fn publish_locked(
        &self,
        manifest: &mut RunManifestV0,
        terminal: &RuntimeTerminalV0,
        next_state: RunStateV0,
    ) -> Result<TerminalPointerV0, StoreFailureV0> {
        let generation = manifest.terminal.as_ref().map_or(1, |t| t.generation + 1);
        let path = terminal_path(&manifest.run_id, generation);
        let bytes = terminal.to_bytes();
        let digest = raw_digest(&bytes);
        manifest.pending_terminal = Some(PendingTerminalV0 {
            path: path.clone(),
            digest: digest.clone(),
            len_bytes: bytes.len() as u64,
            generation,
            next_state,
        });
        let prior_state = manifest.state;
        manifest.state = RunStateV0::TerminalPending;
        let written = self
            .faults()
            .check(FaultPointV0::TerminalPendingWrite)
            .and_then(|()| self.write_manifest_locked(manifest, false));
        if let Err(e) = written {
            manifest.pending_terminal = None;
            manifest.state = prior_state;
            return Err(e);
        }
        self.faults().check(FaultPointV0::TerminalPendingSet)?;
        self.write_terminal_bytes_locked(&path, &bytes)?;
        self.faults().check(FaultPointV0::TerminalWritten)?;
        self.move_pointer_locked(manifest)
    }

    fn write_terminal_bytes_locked(&self, path: &str, bytes: &[u8]) -> Result<(), StoreFailureV0> {
        let (dir, name) = split(path).ok_or(StoreFailureV0::UnsafeLayout)?;
        let fd = self
            .dir(dir, false)?
            .ok_or(StoreFailureV0::ManifestMissing)?;
        write_immutable(&fd, name, bytes, self.owner())
    }

    fn move_pointer_locked(
        &self,
        manifest: &mut RunManifestV0,
    ) -> Result<TerminalPointerV0, StoreFailureV0> {
        let pending = manifest
            .pending_terminal
            .clone()
            .ok_or(StoreFailureV0::StateInvalid)?;
        self.faults().check(FaultPointV0::TerminalPointerMove)?;
        let pointer = TerminalPointerV0 {
            path: pending.path,
            digest: pending.digest,
            len_bytes: pending.len_bytes,
            generation: pending.generation,
        };
        manifest.terminal = Some(pointer.clone());
        manifest.pending_terminal = None;
        manifest.state = pending.next_state;
        self.write_manifest_locked(manifest, false)?;
        Ok(pointer)
    }

    /// Section 3 step 10: the originating live envelope publishes the initial
    /// terminal for its run before returning.
    pub fn publish_initial(
        &self,
        run: &RunIdV0,
        outcome: TerminalOutcomeV0,
        row_digest: &str,
    ) -> Result<TerminalPointerV0, StoreFailureV0> {
        let _lock = self.lock()?;
        let mut manifest = self.valid_manifest_locked(run)?;
        if manifest.state != RunStateV0::Open {
            return Err(StoreFailureV0::StateInvalid);
        }
        let terminal = RuntimeTerminalV0::initial(&manifest, outcome, row_digest);
        let next = match outcome {
            TerminalOutcomeV0::Complete => RunStateV0::Complete,
            TerminalOutcomeV0::StorageIncomplete => RunStateV0::StorageIncomplete,
        };
        self.publish_locked(&mut manifest, &terminal, next)
    }

    /// Publish a later generation copying outcome, row digest and refs.
    pub(crate) fn publish_successor_locked(
        &self,
        manifest: &mut RunManifestV0,
        prior: &RuntimeTerminalV0,
        cleanup: super::CleanupStateV0,
        next_state: RunStateV0,
    ) -> Result<TerminalPointerV0, StoreFailureV0> {
        let prior_digest = manifest.terminal.as_ref().map(|t| t.digest.clone());
        let terminal = RuntimeTerminalV0 {
            previous_terminal_digest: prior_digest,
            cleanup,
            ..prior.clone()
        };
        self.publish_locked(manifest, &terminal, next_state)
    }

    /// Section 4.3 step 4: recovery never upgrades unproven completion.
    pub fn recover_run(&self, run: &RunIdV0) -> RecoveryOutcomeV0 {
        let Ok(_lock) = self.lock() else {
            return RecoveryOutcomeV0::ManifestUnusable;
        };
        let mut manifest = match self.read_manifest_locked(run) {
            ManifestRead::Valid(m) => m,
            _ => return RecoveryOutcomeV0::ManifestUnusable,
        };
        if let (Some(t), None) = (&manifest.terminal, &manifest.pending_terminal) {
            return match self.read_terminal_locked(&t.path, &t.digest, t.len_bytes) {
                Some(terminal) => RecoveryOutcomeV0::AlreadySettled(terminal.outcome),
                None => RecoveryOutcomeV0::ManifestUnusable,
            };
        }
        let Some(row_digest) = manifest.row_digest().map(str::to_owned) else {
            return RecoveryOutcomeV0::Unbindable;
        };
        if let Some(pending) = manifest.pending_terminal.clone() {
            // An authoritative prior terminal: finish only an exact-byte successor.
            if let Some(prior) = &manifest.terminal {
                let exact = self
                    .read_terminal_locked(&pending.path, &pending.digest, pending.len_bytes)
                    .is_some();
                let prior_ok =
                    self.read_terminal_locked(&prior.path, &prior.digest, prior.len_bytes);
                return match (exact, prior_ok) {
                    (true, Some(p)) if self.move_pointer_locked(&mut manifest).is_ok() => {
                        RecoveryOutcomeV0::AlreadySettled(p.outcome)
                    }
                    (_, Some(p)) => {
                        manifest.pending_terminal = None;
                        manifest.state = state_after(&p);
                        match self.write_manifest_locked(&mut manifest, false) {
                            Ok(()) => RecoveryOutcomeV0::AlreadySettled(p.outcome),
                            Err(_) => RecoveryOutcomeV0::ManifestUnusable,
                        }
                    }
                    _ => RecoveryOutcomeV0::ManifestUnusable,
                };
            }
            // Initial pending publication: only the exact expected bytes may finish.
            let outcome = match pending.next_state {
                RunStateV0::Complete => Some(TerminalOutcomeV0::Complete),
                RunStateV0::StorageIncomplete => Some(TerminalOutcomeV0::StorageIncomplete),
                _ => None,
            };
            let all_retained = manifest
                .objects
                .iter()
                .all(|e| e.state == super::manifest::ObjectStateV0::Retained);
            let outcome =
                outcome.filter(|o| *o == TerminalOutcomeV0::StorageIncomplete || all_retained);
            if let (1, Some(outcome)) = (pending.generation, outcome) {
                let expected = RuntimeTerminalV0::initial(&manifest, outcome, &row_digest);
                let bytes = expected.to_bytes();
                if raw_digest(&bytes) == pending.digest
                    && bytes.len() as u64 == pending.len_bytes
                    && self
                        .write_terminal_bytes_locked(&pending.path, &bytes)
                        .is_ok()
                    && self
                        .read_terminal_locked(&pending.path, &pending.digest, pending.len_bytes)
                        .is_some()
                    && self.move_pointer_locked(&mut manifest).is_ok()
                {
                    return match outcome {
                        TerminalOutcomeV0::Complete => RecoveryOutcomeV0::ResumedComplete,
                        TerminalOutcomeV0::StorageIncomplete => {
                            RecoveryOutcomeV0::PublishedStorageIncomplete
                        }
                    };
                }
            }
            // Abandon the unproven publication; its generation file, if any, is
            // left unreferenced and the incomplete outcome takes a new generation.
            manifest.pending_terminal = None;
            manifest.state = RunStateV0::Open;
            if self.write_manifest_locked(&mut manifest, false).is_err() {
                return RecoveryOutcomeV0::ManifestUnusable;
            }
        }
        let terminal = RuntimeTerminalV0::initial(
            &manifest,
            TerminalOutcomeV0::StorageIncomplete,
            &row_digest,
        );
        let generation_taken = self
            .read_at(&terminal_path(run, 1))
            .ok()
            .flatten()
            .is_some_and(|existing| existing != terminal.to_bytes());
        if generation_taken {
            // Generation 1 holds unproven bytes; never overwrite an immutable name.
            return RecoveryOutcomeV0::Unbindable;
        }
        match self.publish_locked(&mut manifest, &terminal, RunStateV0::StorageIncomplete) {
            Ok(_) => RecoveryOutcomeV0::PublishedStorageIncomplete,
            Err(_) => RecoveryOutcomeV0::ManifestUnusable,
        }
    }
}

fn state_after(terminal: &RuntimeTerminalV0) -> RunStateV0 {
    match (terminal.cleanup, terminal.outcome) {
        (CleanupStateV0::NotStarted, TerminalOutcomeV0::Complete) => RunStateV0::Complete,
        (CleanupStateV0::NotStarted, TerminalOutcomeV0::StorageIncomplete) => {
            RunStateV0::StorageIncomplete
        }
        (CleanupStateV0::InProgress, _) => RunStateV0::Cleaning,
        (CleanupStateV0::Clear, _) => RunStateV0::Cleaned,
        (CleanupStateV0::Fail, _) => RunStateV0::CleanupFailed,
        (CleanupStateV0::Indeterminate, _) => RunStateV0::CleanupIndeterminate,
    }
}

pub(crate) fn expected_state(terminal: &RuntimeTerminalV0) -> RunStateV0 {
    state_after(terminal)
}
