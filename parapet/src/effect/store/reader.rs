// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Authority reader and redacted wrapper (Routing contract 50747a52, sections 4.2 and 4.3).

use super::layout::terminal_path;
use super::manifest::{ObjectStateV0, RunManifestV0, RunStateV0};
use super::object::raw_digest;
use super::terminal::{expected_state, CleanupStateV0, RuntimeTerminalV0, TerminalOutcomeV0};
use super::{ManifestRead, ProtectedEvidenceStoreV0};
use crate::effect::interfaces::{CandidateRefV0, RunIdV0, StageKindV0, STORAGE_FAILURE_BANNER};
use crate::receipt::{
    record_digest, PresentUnavailable, RecordDomain, RequestRef, RequestRefError,
};

/// Section 4.2: closed outer reader errors. None carries raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderErrorV0 {
    /// `run_manifest_missing`.
    RunManifestMissing,
    /// `run_manifest_corrupt`.
    RunManifestCorrupt,
    /// `ref_not_listed`.
    RefNotListed,
    /// `object_pending`.
    ObjectPending,
    /// `object_released`.
    ObjectReleased,
    /// `object_missing`.
    ObjectMissing,
    /// `object_digest_mismatch`.
    ObjectDigestMismatch,
    /// `object_length_mismatch`.
    ObjectLengthMismatch,
    /// `terminal_pending`.
    TerminalPending,
    /// `terminal_missing`.
    TerminalMissing,
    /// `terminal_corrupt`.
    TerminalCorrupt,
    /// `terminal_binding_mismatch`.
    TerminalBindingMismatch,
    /// `terminal_status_mismatch`.
    TerminalStatusMismatch,
    /// `cleanup_in_progress`.
    CleanupInProgress,
    /// `storage_incomplete`.
    StorageIncomplete,
}

impl ReaderErrorV0 {
    /// Closed outer code.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RunManifestMissing => "run_manifest_missing",
            Self::RunManifestCorrupt => "run_manifest_corrupt",
            Self::RefNotListed => "ref_not_listed",
            Self::ObjectPending => "object_pending",
            Self::ObjectReleased => "object_released",
            Self::ObjectMissing => "object_missing",
            Self::ObjectDigestMismatch => "object_digest_mismatch",
            Self::ObjectLengthMismatch => "object_length_mismatch",
            Self::TerminalPending => "terminal_pending",
            Self::TerminalMissing => "terminal_missing",
            Self::TerminalCorrupt => "terminal_corrupt",
            Self::TerminalBindingMismatch => "terminal_binding_mismatch",
            Self::TerminalStatusMismatch => "terminal_status_mismatch",
            Self::CleanupInProgress => "cleanup_in_progress",
            Self::StorageIncomplete => "storage_incomplete",
        }
    }
}

/// Bindings a caller expects the authoritative history to carry.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedBindingV0 {
    /// Verbatim request reference.
    pub request_ref: PresentUnavailable<RequestRef, RequestRefError>,
    /// Formed coordinate.
    pub candidate_ref: CandidateRefV0,
    /// Selected row digest.
    pub row_digest: String,
}

/// Authoritative terminal evidence for one run.
#[derive(Debug, Clone)]
pub(crate) struct Authority {
    pub manifest: RunManifestV0,
    pub terminal: RuntimeTerminalV0,
}

impl ProtectedEvidenceStoreV0 {
    // Precedence: manifest; pending/missing/corrupt terminal; history binding.
    pub(crate) fn authority_locked(
        &self,
        run: &RunIdV0,
        expected: Option<&ExpectedBindingV0>,
    ) -> Result<Authority, ReaderErrorV0> {
        let manifest = match self.read_manifest_locked(run) {
            ManifestRead::Valid(m) => m,
            ManifestRead::Missing => return Err(ReaderErrorV0::RunManifestMissing),
            ManifestRead::Corrupt => return Err(ReaderErrorV0::RunManifestCorrupt),
        };
        if manifest.pending_terminal.is_some() {
            return Err(ReaderErrorV0::TerminalPending);
        }
        let pointer = manifest
            .terminal
            .clone()
            .ok_or(ReaderErrorV0::TerminalMissing)?;
        let terminal = self
            .read_terminal_locked(&pointer.path, &pointer.digest, pointer.len_bytes)
            .ok_or(ReaderErrorV0::TerminalCorrupt)?;
        let bound = |t: &RuntimeTerminalV0| {
            &t.run_id == run
                && t.request_ref == manifest.request_ref
                && t.candidate_ref == manifest.candidate_ref
                && t.row_digest == terminal.row_digest
                && t.outcome == terminal.outcome
                && manifest.row_digest().is_none_or(|d| d == t.row_digest)
        };
        if !bound(&terminal) || expected_state(&terminal) != manifest.state {
            return Err(ReaderErrorV0::TerminalBindingMismatch);
        }
        if let Some(expected) = expected {
            if expected.request_ref != terminal.request_ref
                || expected.candidate_ref != terminal.candidate_ref
                || expected.row_digest != terminal.row_digest
            {
                return Err(ReaderErrorV0::TerminalBindingMismatch);
            }
        }
        // Walk the immutable chain to the initial snapshot.
        let mut current = terminal.clone();
        let mut generation = pointer.generation;
        while let Some(previous) = current.previous_terminal_digest.clone() {
            if generation <= 1 {
                return Err(ReaderErrorV0::TerminalBindingMismatch);
            }
            generation -= 1;
            let bytes = self
                .read_at(&terminal_path(run, generation))
                .ok()
                .flatten()
                .ok_or(ReaderErrorV0::TerminalBindingMismatch)?;
            if raw_digest(&bytes) != previous {
                return Err(ReaderErrorV0::TerminalBindingMismatch);
            }
            let prior =
                RuntimeTerminalV0::parse(&bytes).ok_or(ReaderErrorV0::TerminalBindingMismatch)?;
            if !bound(&prior) {
                return Err(ReaderErrorV0::TerminalBindingMismatch);
            }
            current = prior;
        }
        if generation != 1 {
            return Err(ReaderErrorV0::TerminalBindingMismatch);
        }
        Ok(Authority { manifest, terminal })
    }

    // Then: authoritative storage_incomplete; supplied status; cleanup state.
    fn status_locked(
        &self,
        run: &RunIdV0,
        expected: Option<&ExpectedBindingV0>,
        claimed: Option<TerminalOutcomeV0>,
    ) -> Result<Authority, ReaderErrorV0> {
        let authority = self.authority_locked(run, expected)?;
        if authority.terminal.outcome == TerminalOutcomeV0::StorageIncomplete {
            return Err(ReaderErrorV0::StorageIncomplete);
        }
        if claimed.is_some_and(|c| c != authority.terminal.outcome) {
            return Err(ReaderErrorV0::TerminalStatusMismatch);
        }
        if matches!(
            authority.manifest.state,
            RunStateV0::Cleaning | RunStateV0::CleanupFailed | RunStateV0::CleanupIndeterminate
        ) {
            return Err(ReaderErrorV0::CleanupInProgress);
        }
        Ok(authority)
    }

    // Finally: object authorization, hash, then length.
    pub(crate) fn object_locked(
        &self,
        manifest: &RunManifestV0,
        object_ref: &str,
    ) -> Result<Vec<u8>, ReaderErrorV0> {
        let entry = manifest
            .objects
            .iter()
            .find(|e| e.path == object_ref)
            .ok_or(ReaderErrorV0::RefNotListed)?;
        match entry.state {
            ObjectStateV0::Pending => return Err(ReaderErrorV0::ObjectPending),
            ObjectStateV0::Released => return Err(ReaderErrorV0::ObjectReleased),
            ObjectStateV0::Retained => {}
        }
        let bytes = match self.read_at(&entry.path) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Err(ReaderErrorV0::ObjectMissing),
            Err(_) => return Err(ReaderErrorV0::ObjectDigestMismatch),
        };
        let digest = match entry.kind {
            StageKindV0::Payload | StageKindV0::Projection => raw_digest(&bytes),
            StageKindV0::Row => record_digest(&bytes, RecordDomain::Row),
            _ => record_digest(&bytes, RecordDomain::Stage),
        };
        if digest != entry.digest {
            return Err(ReaderErrorV0::ObjectDigestMismatch);
        }
        if bytes.len() as u64 != entry.len_bytes {
            return Err(ReaderErrorV0::ObjectLengthMismatch);
        }
        Ok(bytes)
    }
}

/// Section 4.2 `ProtectedEvidenceReaderV0`: raw bytes only under authority.
/// Not for public adapters; they use `RedactedEvidenceReaderV0`.
pub struct ProtectedEvidenceReaderV0<'a> {
    store: &'a ProtectedEvidenceStoreV0,
}

impl<'a> ProtectedEvidenceReaderV0<'a> {
    /// Bind the reader to a store.
    pub fn new(store: &'a ProtectedEvidenceStoreV0) -> Self {
        Self { store }
    }

    /// Return protected bytes only for an authoritative complete retained object.
    pub fn read(
        &self,
        run: &RunIdV0,
        object_ref: &str,
        expected: Option<&ExpectedBindingV0>,
        claimed: Option<TerminalOutcomeV0>,
    ) -> Result<Vec<u8>, ReaderErrorV0> {
        let _lock = self
            .store
            .lock()
            .map_err(|_| ReaderErrorV0::RunManifestCorrupt)?;
        let authority = self.store.status_locked(run, expected, claimed)?;
        self.store.object_locked(&authority.manifest, object_ref)
    }
}

/// Section 4.2: outer display model; the projection bytes are carried unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactedRunViewV0 {
    /// Authoritative workflow outcome, when one exists.
    pub outcome: Option<TerminalOutcomeV0>,
    /// Cleanup state; `clear` only for a validated `cleaned` manifest.
    pub cleanup: Option<CleanupStateV0>,
    /// Storage-failure banner, outside the projection.
    pub banner: Option<&'static str>,
    /// Closed reader code, when authority was not established.
    pub reader_code: Option<ReaderErrorV0>,
    /// Exact landed projection bytes, when authorized.
    pub projection_bytes: Option<Vec<u8>>,
}

/// Section 4.2: the only reader a public adapter receives.
pub struct RedactedEvidenceReaderV0<'a> {
    store: &'a ProtectedEvidenceStoreV0,
}

impl<'a> RedactedEvidenceReaderV0<'a> {
    /// Bind the redacted reader to a store.
    pub fn new(store: &'a ProtectedEvidenceStoreV0) -> Self {
        Self { store }
    }

    /// Outer outcome, banner, reader code and unchanged projection bytes.
    pub fn view(&self, run: &RunIdV0, claimed: Option<TerminalOutcomeV0>) -> RedactedRunViewV0 {
        let failed = |code| RedactedRunViewV0 {
            outcome: None,
            cleanup: None,
            banner: None,
            reader_code: Some(code),
            projection_bytes: None,
        };
        let Ok(_lock) = self.store.lock() else {
            return failed(ReaderErrorV0::RunManifestCorrupt);
        };
        let authority = match self.store.authority_locked(run, None) {
            Ok(a) => a,
            Err(code) => return failed(code),
        };
        let outcome = authority.terminal.outcome;
        let cleanup = match authority.terminal.cleanup {
            CleanupStateV0::Clear if authority.manifest.state != RunStateV0::Cleaned => None,
            other => Some(other),
        };
        let banner =
            (outcome == TerminalOutcomeV0::StorageIncomplete).then_some(STORAGE_FAILURE_BANNER);
        let reader_code = match self.store.status_locked(run, None, claimed) {
            Ok(_) => None,
            Err(code) => Some(code),
        };
        let mismatch = reader_code == Some(ReaderErrorV0::TerminalStatusMismatch);
        let projection_bytes = (!mismatch)
            .then(|| {
                let path = super::layout::projection_path(&authority.terminal.row_digest);
                self.store.object_locked(&authority.manifest, &path).ok()
            })
            .flatten();
        RedactedRunViewV0 {
            outcome: Some(outcome),
            cleanup: if mismatch { None } else { cleanup },
            banner,
            reader_code,
            projection_bytes,
        }
    }
}
