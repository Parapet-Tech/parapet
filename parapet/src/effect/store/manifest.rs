// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Run manifest schema `parapet-protected-run-manifest/v0` (Routing contract 50747a52, section 4.3).

use super::layout::{is_digest, object_path, terminal_path};
use crate::effect::interfaces::{CandidateRefV0, RunIdV0, StageKindV0};
use crate::receipt::{PayloadStore, PresentUnavailable, RequestRef, RequestRefError};
use serde::{Deserialize, Deserializer, Serialize};

/// Manifest schema constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestSchemaV0 {
    /// `parapet-protected-run-manifest/v0`.
    #[serde(rename = "parapet-protected-run-manifest/v0")]
    V0,
}

/// Section 4.3: run state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStateV0 {
    /// Empty manifest created before any run object.
    Open,
    /// A terminal publication is pending.
    TerminalPending,
    /// Initial terminal `complete` published.
    Complete,
    /// Authoritative terminal outcome `storage_incomplete`.
    StorageIncomplete,
    /// Cleanup in progress.
    Cleaning,
    /// Cleanup `clear` published.
    Cleaned,
    /// Cleanup `fail` published.
    CleanupFailed,
    /// Cleanup `indeterminate` published.
    CleanupIndeterminate,
}

/// Section 4.3: object entry state, monotone `pending -> retained -> released`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStateV0 {
    /// Attributed before the object write.
    Pending,
    /// Written and verified.
    Retained,
    /// Released by cleanup; never live.
    Released,
}

impl ObjectStateV0 {
    /// A live reference is exactly a pending or retained entry.
    pub fn is_live(self) -> bool {
        matches!(self, Self::Pending | Self::Retained)
    }
}

/// Section 4.3: `{path, store, kind, digest, len_bytes, state}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectEntryV0 {
    /// Relative object path.
    pub path: String,
    /// Closed store label.
    pub store: PayloadStore,
    /// Closed object kind.
    pub kind: StageKindV0,
    /// Integrity digest: raw sha256 for payloads and projections, the landed
    /// tagged digest for stage records and rows.
    pub digest: String,
    /// Byte length.
    pub len_bytes: u64,
    /// Entry state.
    pub state: ObjectStateV0,
}

/// Section 4.3: `{path, digest, len_bytes, generation}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalPointerV0 {
    /// Relative terminal path.
    pub path: String,
    /// Raw sha256 of the terminal bytes.
    pub digest: String,
    /// Byte length.
    pub len_bytes: u64,
    /// Terminal generation, starting at 1.
    pub generation: u64,
}

/// Section 4.3: the terminal pointer fields plus `next_state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingTerminalV0 {
    /// Relative terminal path.
    pub path: String,
    /// Raw sha256 of the expected terminal bytes.
    pub digest: String,
    /// Expected byte length.
    pub len_bytes: u64,
    /// Terminal generation being published.
    pub generation: u64,
    /// State entered when the pointer moves.
    pub next_state: RunStateV0,
}

fn required<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}

/// Section 4.3: exact ordered manifest fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV0 {
    /// Schema constant.
    pub schema: ManifestSchemaV0,
    /// Run id; equals the directory name.
    pub run_id: RunIdV0,
    /// Draft-12 availability-bearing request reference, verbatim.
    pub request_ref: PresentUnavailable<RequestRef, RequestRefError>,
    /// Exact formed coordinate.
    pub candidate_ref: CandidateRefV0,
    /// Manifest revision, incremented on every write.
    pub generation: u64,
    /// Run state.
    pub state: RunStateV0,
    /// Path-sorted object entries.
    pub objects: Vec<ObjectEntryV0>,
    /// Authoritative terminal pointer.
    #[serde(deserialize_with = "required")]
    pub terminal: Option<TerminalPointerV0>,
    /// Pending terminal publication.
    #[serde(deserialize_with = "required")]
    pub pending_terminal: Option<PendingTerminalV0>,
}

impl RunManifestV0 {
    /// The empty `open` manifest written before any run object.
    pub fn open(
        run_id: RunIdV0,
        request_ref: PresentUnavailable<RequestRef, RequestRefError>,
        candidate_ref: CandidateRefV0,
    ) -> Self {
        Self {
            schema: ManifestSchemaV0::V0,
            run_id,
            request_ref,
            candidate_ref,
            generation: 1,
            state: RunStateV0::Open,
            objects: Vec::new(),
            terminal: None,
            pending_terminal: None,
        }
    }

    /// Compact manifest bytes in field order.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("closed manifest serializes")
    }

    /// Parse and validate manifest bytes for the run named by its directory.
    pub fn parse(bytes: &[u8], run: &RunIdV0) -> Option<Self> {
        let manifest: Self = serde_json::from_slice(bytes).ok()?;
        (manifest.to_bytes() == bytes && manifest.valid_for(run)).then_some(manifest)
    }

    fn valid_for(&self, run: &RunIdV0) -> bool {
        if &self.run_id != run || self.generation == 0 {
            return false;
        }
        let sorted = self.objects.windows(2).all(|w| w[0].path < w[1].path);
        let entries = self.objects.iter().all(|e| {
            let key = if e.kind == StageKindV0::Projection {
                e.path
                    .strip_prefix("projection/sha256/")
                    .and_then(|p| p.strip_suffix(".json"))
                    .unwrap_or("")
            } else {
                e.digest.as_str()
            };
            e.store == PayloadStore::ReceiptEvidence
                && is_digest(&e.digest)
                && is_digest(key)
                && e.path == object_path(e.kind, key)
        });
        let pointer = self.terminal.as_ref().is_none_or(|t| {
            t.generation > 0 && is_digest(&t.digest) && t.path == terminal_path(run, t.generation)
        });
        let pending = self.pending_terminal.as_ref().is_none_or(|p| {
            p.generation == self.terminal.as_ref().map_or(1, |t| t.generation + 1)
                && is_digest(&p.digest)
                && p.path == terminal_path(run, p.generation)
                && !matches!(p.next_state, RunStateV0::Open | RunStateV0::TerminalPending)
        });
        let state = match self.state {
            RunStateV0::Open => self.terminal.is_none() && self.pending_terminal.is_none(),
            RunStateV0::TerminalPending => self.pending_terminal.is_some(),
            _ => self.terminal.is_some() && self.pending_terminal.is_none(),
        };
        let cleaned = self.state != RunStateV0::Cleaned
            || self
                .objects
                .iter()
                .all(|e| e.state == ObjectStateV0::Released);
        sorted && entries && pointer && pending && state && cleaned
    }

    /// Insert or find an entry, keeping path order. Returns its index.
    pub fn attribute(&mut self, entry: ObjectEntryV0) -> Result<usize, ()> {
        match self
            .objects
            .binary_search_by(|e| e.path.as_str().cmp(&entry.path))
        {
            Ok(i) => {
                let existing = &self.objects[i];
                if existing.digest != entry.digest
                    || existing.len_bytes != entry.len_bytes
                    || existing.kind != entry.kind
                    || existing.state == ObjectStateV0::Released
                {
                    return Err(());
                }
                Ok(i)
            }
            Err(i) => {
                self.objects.insert(i, entry);
                Ok(i)
            }
        }
    }

    /// The row digest named by this run, if a row entry exists.
    pub fn row_digest(&self) -> Option<&str> {
        self.objects
            .iter()
            .find(|e| e.kind == StageKindV0::Row)
            .map(|e| e.digest.as_str())
    }
}
