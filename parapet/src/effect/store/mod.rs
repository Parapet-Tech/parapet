// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Protected first-use evidence store (Routing contract 50747a52, section 4.3).
//!
//! The caller supplies a private `receipt-evidence` root outside any effect
//! root. Absolute paths never leave this module: every operation after opening
//! the root is relative to its retained directory descriptor.

/// Locked cleanup with full-manifest preflight.
pub mod cleanup;
/// Exact relative layout and private-directory checks.
pub mod layout;
/// Run manifest schema and validation.
pub mod manifest;
/// No-follow private file primitives.
pub mod object;
/// Authority reader and redacted wrapper.
pub mod reader;
/// Immutable terminal snapshots, publication and recovery.
pub mod terminal;

use crate::effect::interfaces::{
    CandidateRefV0, ProtectedPayloadStoreV0, RunIdV0, StageKindV0, StoreFailureV0, StoredPayloadV0,
};
use crate::receipt::{
    PayloadResolution, PayloadStore, PresentUnavailable, RequestRef, RequestRefError,
};
use layout::{split, Owner, LOCK_NAME};
use manifest::{ObjectEntryV0, ObjectStateV0, RunManifestV0};
use rustix::fs::{fstat, openat, FileType, Mode, OFlags};
use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Mutex;

pub use cleanup::CleanupResultV0;
pub use manifest::RunStateV0;
pub use reader::{
    ExpectedBindingV0, ProtectedEvidenceReaderV0, ReaderErrorV0, RedactedEvidenceReaderV0,
    RedactedRunViewV0,
};
pub use terminal::{CleanupStateV0, RecoveryOutcomeV0, RuntimeTerminalV0, TerminalOutcomeV0};

/// Named crash points for fault-injection fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultPointV0 {
    /// Any manifest write.
    ManifestWrite,
    /// After a pending entry is attributed, before the object write.
    ObjectPendingAdded,
    /// After the object is written, before the entry becomes retained.
    ObjectWritten,
    /// The manifest write that would persist `pending_terminal`.
    TerminalPendingWrite,
    /// After `pending_terminal` is persisted, before the terminal write.
    TerminalPendingSet,
    /// After the terminal snapshot is written, before the pointer moves.
    TerminalWritten,
    /// The pointer-moving manifest write.
    TerminalPointerMove,
    /// After an entry is released, before its unlink.
    CleanupAfterRelease,
    /// After unlinks, before the final cleanup terminal.
    CleanupAfterUnlink,
}

/// Fixture-only owner substitution, to exercise the owner checks without privileges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignOwnerV0 {
    /// The root appears owned by another user.
    Root,
    /// Directories and objects below a valid root appear owned by another user.
    Objects,
}

/// Injected failures: each `(point, skip)` fires once after `skip` passes.
#[derive(Debug, Default)]
pub struct FaultPlanV0 {
    points: Mutex<Vec<(FaultPointV0, u32)>>,
    foreign_owner: Option<ForeignOwnerV0>,
}

impl FaultPlanV0 {
    /// No injected failures.
    pub fn none() -> Self {
        Self::default()
    }

    /// Fail at the `skip`-th later occurrence (0 = the next) of `point`.
    pub fn fail_at(point: FaultPointV0, skip: u32) -> Self {
        Self {
            points: Mutex::new(vec![(point, skip)]),
            foreign_owner: None,
        }
    }

    /// Treat the root or its contents as owned by another user.
    pub fn foreign_owner(which: ForeignOwnerV0) -> Self {
        Self {
            points: Mutex::new(Vec::new()),
            foreign_owner: Some(which),
        }
    }

    pub(crate) fn check(&self, point: FaultPointV0) -> Result<(), StoreFailureV0> {
        let mut points = self.points.lock().expect("fault plan lock");
        if let Some(i) = points.iter().position(|(p, _)| *p == point) {
            if points[i].1 == 0 {
                points.remove(i);
                return Err(StoreFailureV0::Injected);
            }
            points[i].1 -= 1;
        }
        Ok(())
    }
}

/// Held exclusive store lock; released on drop.
pub(crate) struct StoreLock(File);

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

/// Manifest read outcome under the lock.
#[derive(Debug, Clone)]
pub(crate) enum ManifestRead {
    Missing,
    Corrupt,
    Valid(RunManifestV0),
}

/// Section 4.2 `ProtectedPayloadStoreV0` and the 4.3 run lifecycle.
pub struct ProtectedEvidenceStoreV0 {
    root: OwnedFd,
    owner: Owner,
    faults: FaultPlanV0,
}

impl std::fmt::Debug for ProtectedEvidenceStoreV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProtectedEvidenceStoreV0")
    }
}

impl ProtectedEvidenceStoreV0 {
    /// Create or verify a private `0700` root owned by this process's user.
    pub fn open(root: &Path) -> Result<Self, StoreFailureV0> {
        Self::open_with_faults(root, FaultPlanV0::none())
    }

    /// As `open`, with injected failure points for fixtures.
    pub fn open_with_faults(root: &Path, faults: FaultPlanV0) -> Result<Self, StoreFailureV0> {
        let foreign = faults.foreign_owner;
        let (root, mut owner) =
            layout::open_root(root, true, foreign == Some(ForeignOwnerV0::Root))?;
        if foreign == Some(ForeignOwnerV0::Objects) {
            owner = Owner(owner.0.wrapping_add(1));
        }
        let store = Self {
            root,
            owner,
            faults,
        };
        drop(store.lock()?);
        Ok(store)
    }

    pub(crate) fn faults(&self) -> &FaultPlanV0 {
        &self.faults
    }

    pub(crate) fn lock(&self) -> Result<StoreLock, StoreFailureV0> {
        let flags = OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let fd = openat(
            &self.root,
            LOCK_NAME,
            flags,
            Mode::from_raw_mode(layout::FILE_MODE as _),
        )
        .map_err(|_| StoreFailureV0::LockFailed)?;
        let stat = fstat(&fd).map_err(|_| StoreFailureV0::LockFailed)?;
        let private = FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
            && (stat.st_mode as u32) & 0o7777 == layout::FILE_MODE
            && stat.st_uid == self.owner.0;
        if !private {
            return Err(StoreFailureV0::UnsafeLayout);
        }
        let file = File::from(fd);
        file.lock().map_err(|_| StoreFailureV0::LockFailed)?;
        Ok(StoreLock(file))
    }

    pub(crate) fn dir(
        &self,
        relative: &str,
        create: bool,
    ) -> Result<Option<OwnedFd>, StoreFailureV0> {
        layout::open_dir(&self.root, relative, self.owner, create)
    }

    pub(crate) fn owner(&self) -> Owner {
        self.owner
    }

    pub(crate) fn read_at(&self, path: &str) -> Result<Option<Vec<u8>>, StoreFailureV0> {
        let (dir, name) = split(path).ok_or(StoreFailureV0::UnsafeLayout)?;
        match self.dir(dir, false)? {
            Some(fd) => object::read_private(&fd, name, self.owner),
            None => Ok(None),
        }
    }

    pub(crate) fn read_manifest_locked(&self, run: &RunIdV0) -> ManifestRead {
        match self.read_at(&layout::manifest_path(run)) {
            Ok(Some(bytes)) => match RunManifestV0::parse(&bytes, run) {
                Some(m) => ManifestRead::Valid(m),
                None => ManifestRead::Corrupt,
            },
            Ok(None) => ManifestRead::Missing,
            Err(_) => ManifestRead::Corrupt,
        }
    }

    pub(crate) fn valid_manifest_locked(
        &self,
        run: &RunIdV0,
    ) -> Result<RunManifestV0, StoreFailureV0> {
        match self.read_manifest_locked(run) {
            ManifestRead::Valid(m) => Ok(m),
            ManifestRead::Missing => Err(StoreFailureV0::ManifestMissing),
            ManifestRead::Corrupt => Err(StoreFailureV0::ManifestCorrupt),
        }
    }

    pub(crate) fn write_manifest_locked(
        &self,
        manifest: &mut RunManifestV0,
        first: bool,
    ) -> Result<(), StoreFailureV0> {
        self.faults.check(FaultPointV0::ManifestWrite)?;
        if !first {
            manifest.generation += 1;
        }
        let dir = self
            .dir(&format!("run/{}", manifest.run_id.as_str()), false)?
            .ok_or(StoreFailureV0::ManifestMissing)?;
        object::write_replace(&dir, "manifest.json", &manifest.to_bytes(), self.owner)
    }

    /// Section 4.3 step 1: atomically create the empty `open` manifest.
    pub fn create_run(
        &self,
        run: &RunIdV0,
        request_ref: &PresentUnavailable<RequestRef, RequestRefError>,
        candidate_ref: CandidateRefV0,
    ) -> Result<(), StoreFailureV0> {
        let _lock = self.lock()?;
        let base = format!("run/{}", run.as_str());
        let dir = self.dir(&base, true)?.ok_or(StoreFailureV0::Io)?;
        if object::exists(&dir, "manifest.json")? {
            return Err(StoreFailureV0::StateInvalid);
        }
        self.dir(&format!("{base}/terminal"), true)?;
        let mut manifest = RunManifestV0::open(run.clone(), request_ref.clone(), candidate_ref);
        self.write_manifest_locked(&mut manifest, true)
    }

    /// Section 4.3 step 2: attribute `pending`, write and verify, then `retained`.
    /// `key_digest` names the path; `digest` is the integrity digest of `bytes`.
    pub fn put_object(
        &self,
        run: &RunIdV0,
        kind: StageKindV0,
        key_digest: &str,
        digest: &str,
        bytes: &[u8],
    ) -> Result<String, StoreFailureV0> {
        let _lock = self.lock()?;
        let mut manifest = self.valid_manifest_locked(run)?;
        if !matches!(manifest.state, RunStateV0::Open) {
            return Err(StoreFailureV0::StateInvalid);
        }
        let path = layout::object_path(kind, key_digest);
        let entry = ObjectEntryV0 {
            path: path.clone(),
            store: PayloadStore::ReceiptEvidence,
            kind,
            digest: digest.to_owned(),
            len_bytes: bytes.len() as u64,
            state: ObjectStateV0::Pending,
        };
        let index = manifest
            .attribute(entry)
            .map_err(|()| StoreFailureV0::Collision)?;
        if manifest.objects[index].state == ObjectStateV0::Pending {
            self.write_manifest_locked(&mut manifest, false)?;
            self.faults.check(FaultPointV0::ObjectPendingAdded)?;
        }
        let (dir, name) = split(&path).ok_or(StoreFailureV0::UnsafeLayout)?;
        let fd = self.dir(dir, true)?.ok_or(StoreFailureV0::Io)?;
        object::write_immutable(&fd, name, bytes, self.owner)?;
        self.faults.check(FaultPointV0::ObjectWritten)?;
        if manifest.objects[index].state == ObjectStateV0::Pending {
            manifest.objects[index].state = ObjectStateV0::Retained;
            self.write_manifest_locked(&mut manifest, false)?;
        }
        Ok(path)
    }

    /// Locked manifest snapshot for inspection.
    pub fn manifest(&self, run: &RunIdV0) -> Result<RunManifestV0, StoreFailureV0> {
        let _lock = self.lock()?;
        self.valid_manifest_locked(run)
    }

    /// Every run directory name, validated as a run id where possible.
    pub(crate) fn run_names_locked(&self) -> Result<Vec<Result<RunIdV0, ()>>, StoreFailureV0> {
        let Some(dir) = self.dir("run", false)? else {
            return Ok(Vec::new());
        };
        let mut names = Vec::new();
        let mut entries = rustix::fs::Dir::read_from(&dir).map_err(|_| StoreFailureV0::Io)?;
        while let Some(entry) = entries.next() {
            let entry = entry.map_err(|_| StoreFailureV0::Io)?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "." || name == ".." {
                continue;
            }
            names.push(RunIdV0::parse(&name).ok_or(()));
        }
        names.sort();
        Ok(names)
    }
}

impl ProtectedPayloadStoreV0 for ProtectedEvidenceStoreV0 {
    fn put(
        &self,
        run: &RunIdV0,
        label: PayloadStore,
        bytes: &[u8],
    ) -> Result<StoredPayloadV0, StoreFailureV0> {
        if label != PayloadStore::ReceiptEvidence {
            return Err(StoreFailureV0::LabelNotConfigured);
        }
        let digest = object::raw_digest(bytes);
        let path = self.put_object(run, StageKindV0::Payload, &digest, &digest, bytes)?;
        Ok(StoredPayloadV0 {
            path,
            digest,
            len_bytes: bytes.len() as u64,
        })
    }

    fn get(&self, label: PayloadStore, digest: &str) -> PayloadResolution {
        if label != PayloadStore::ReceiptEvidence {
            return PayloadResolution::NotConfigured;
        }
        if !layout::is_digest(digest) {
            return PayloadResolution::NotFound;
        }
        let Ok(_lock) = self.lock() else {
            return PayloadResolution::NotFound;
        };
        match self.read_at(&layout::payload_path(digest)) {
            Ok(Some(bytes)) if object::raw_digest(&bytes) == digest => {
                PayloadResolution::Found(bytes)
            }
            _ => PayloadResolution::NotFound,
        }
    }
}

impl crate::receipt::PayloadResolver for ProtectedEvidenceStoreV0 {
    fn resolve(&self, store: PayloadStore, digest: &str) -> PayloadResolution {
        self.get(store, digest)
    }
}
