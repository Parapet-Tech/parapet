// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `RootCapabilityV0`: one new private root, one retained directory fd
//! (1C brief b76e6b1b sections 5 and 6.2; draft 12 sections 5.3 and 5.6).
//!
//! The absolute path is read once, at construction, to compute the draft-12
//! root digest. Every later operation is relative to the retained fd; the path
//! is never stored, logged or reopened.

use crate::effect::interfaces::{ActionIdV0, OpaqueRootCapabilityV0, RunIdV0};
use crate::effect::store::layout::{self, Owner};
use crate::effect::store::object;
use crate::receipt::{root_digest, target_ref, Root};
use rustix::fs::{fstat, mkdirat, openat, statat, AtFlags, FileType, Mode, OFlags, CWD};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Controller-only root-evidence namespace inside the root, mode `0700`.
pub const ROOT_EVIDENCE_DIR: &str = ".parapet-root-evidence-v0";
/// The root-evidence namespace as a declared footprint entry.
pub const ROOT_EVIDENCE_ENTRY: &str = ".parapet-root-evidence-v0/";
/// Private root metadata inside the root-evidence namespace.
pub const ROOT_METADATA_NAME: &str = "root.json";
/// Content-addressed protected copies inside the root-evidence namespace.
pub const ROOT_PAYLOAD_DIR: &str = "payload/sha256";
/// Largest admissible `max_content_len_bytes`.
pub const MAX_CONTENT_LIMIT: u64 = 1_048_576;

const DIR_MODE: u32 = 0o700;
const METADATA_SCHEMA: &str = "parapet-file-root-metadata/v0";
const SENTINEL_SCHEMA: &str = "parapet-file-run-sentinel/v0";

/// `.parapet-run-<run-id>.sentinel`.
pub fn sentinel_name(run: &RunIdV0) -> String {
    format!(".parapet-run-{}.sentinel", run.as_str())
}

/// Closed root construction failures. No path or OS text is carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootErrorV0 {
    /// `max_content_len_bytes` is outside `1..=1_048_576`.
    ContentLimitOutOfRange,
    /// The root path already exists; a root is always new.
    AlreadyExists,
    /// The created root is not a private directory owned by this process.
    UnsafeLayout,
    /// A filesystem operation failed.
    Io,
}

/// Deterministic fixture fault points (tests only; none by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileFaultV0 {
    /// Pre-read reports `no_read_access`.
    PreReadNoAccess,
    /// Pre-read reports `io_error`.
    PreReadIo,
    /// Root-evidence payload persistence fails.
    EvidencePersist,
    /// Exclusive temp creation fails.
    TempCreate,
    /// The temp write fails.
    WriteError,
    /// The temp write stops after half of the bytes.
    PartialWrite,
    /// The operation is interrupted before the rename is issued.
    InterruptBeforeRename,
    /// The temp fsync fails.
    Fsync,
    /// The rename returns a definite error.
    RenameError,
    /// The rename call is interrupted after issuance; its result is unknown.
    RenameInterrupted,
    /// The rename is issued and its confirmation is lost.
    RenameUnconfirmed,
    /// The observer's read reports `no_read_access`.
    ObserveNoAccess,
    /// The observer's read reports `io_error`.
    ObserveIo,
    /// The observer cannot produce a record.
    ObserverFails,
    /// Footprint enumeration fails during observation.
    FootprintUnavailable,
    /// Cleanup enumeration fails.
    CleanupEnumerate,
    /// A declared cleanup removal fails.
    CleanupUnlink,
}

/// Fixture race points at which a test hook runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilePhaseV0 {
    /// After successful no-follow resolution, before the pre-read.
    AfterResolution,
    /// After the temp is written and fsynced, before the rename.
    BeforeRename,
    /// At the start of observation, before independent resolution.
    BeforeObservation,
}

/// Fixture race hook.
pub type FileHookV0 = Arc<dyn Fn(FilePhaseV0) + Send + Sync>;

/// Deterministic fixture inputs; the default is a fresh nonce and no faults.
#[derive(Clone, Default)]
pub struct FileFixtureV0 {
    /// Fixed nonce, for digest fixtures.
    pub nonce: Option<[u8; 32]>,
    /// Injected fault points.
    pub faults: BTreeSet<FileFaultV0>,
    /// Race hook.
    pub hook: Option<FileHookV0>,
}

impl std::fmt::Debug for FileFixtureV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileFixtureV0")
            .field("faults", &self.faults)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RootMetadataV0 {
    pub schema: String,
    pub root_id: String,
    pub root_digest: String,
    pub nonce: String,
    pub run_id: String,
    pub max_content_len_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunSentinelV0 {
    pub schema: String,
    pub run_id: String,
    pub root_id: String,
    pub root_digest: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest_bytes(digest: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&digest[2 * i..2 * i + 2], 16).unwrap_or_default();
    }
    out
}

fn fresh_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    nonce[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    nonce[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    nonce
}

fn dir_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}

fn private_dir(stat: &rustix::fs::Stat, owner: Option<Owner>) -> bool {
    FileType::from_raw_mode(stat.st_mode) == FileType::Directory
        && (stat.st_mode as u32) & 0o7777 == DIR_MODE
        && owner.is_none_or(|o| stat.st_uid == o.0)
}

/// Result of one no-follow read of a basename relative to the retained fd.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TargetReadV0 {
    /// The name does not exist.
    Missing,
    /// A regular file with these bytes.
    Bytes(Vec<u8>),
    /// The final component is a symlink and was not followed.
    Symlink,
    /// Permission denied.
    NoAccess,
    /// Any other failure, including a non-regular object.
    Error,
}

pub(crate) fn read_target(dir: &OwnedFd, basename: &str) -> TargetReadV0 {
    use std::io::Read;
    // Non-blocking, so a FIFO planted at the name cannot stall the read.
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let fd = match openat(dir, basename, flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(Errno::NOENT) => return TargetReadV0::Missing,
        Err(Errno::LOOP) => return TargetReadV0::Symlink,
        Err(Errno::ACCESS | Errno::PERM) => return TargetReadV0::NoAccess,
        Err(_) => return TargetReadV0::Error,
    };
    match fstat(&fd) {
        Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile => {}
        _ => return TargetReadV0::Error,
    }
    let mut bytes = Vec::new();
    match std::fs::File::from(fd).read_to_end(&mut bytes) {
        Ok(_) => TargetReadV0::Bytes(bytes),
        Err(_) => TargetReadV0::Error,
    }
}

// Sorted entry names of a directory, excluding `.` and `..`; `None` when
// enumeration is uncertain.
pub(crate) fn list_entries(dir: &OwnedFd) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for entry in rustix::fs::Dir::read_from(dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        out.push(String::from_utf8_lossy(name).into_owned());
    }
    out.sort();
    Some(out)
}

// Whether a root entry is in the declared footprint.
pub(crate) fn is_declared(name: &str, declared: &[String]) -> bool {
    declared.iter().any(|d| d == name)
        || (name == ROOT_EVIDENCE_DIR && declared.iter().any(|d| d == ROOT_EVIDENCE_ENTRY))
}

/// One new private root for one acceptance run.
pub struct RootCapabilityV0 {
    fd: Mutex<Option<OwnedFd>>,
    identity: (u64, u64),
    owner: Owner,
    root_id: String,
    nonce: [u8; 32],
    root_digest: String,
    max_content_len_bytes: u64,
    run_id: RunIdV0,
    registry: Mutex<BTreeSet<String>>,
    footprint: Mutex<BTreeSet<String>>,
    cleared: AtomicBool,
    target_calls: AtomicU64,
    fixture: FileFixtureV0,
}

impl std::fmt::Debug for RootCapabilityV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootCapabilityV0")
            .field("root_id", &self.root_id)
            .field("max_content_len_bytes", &self.max_content_len_bytes)
            .field("live", &self.is_live())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for RootCapabilityV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "file-effect root {}", self.root_id)
    }
}

impl RootCapabilityV0 {
    /// Create a new private root at `path` with a fresh nonce.
    pub fn create(
        path: &Path,
        run_id: RunIdV0,
        max_content_len_bytes: u64,
    ) -> Result<Arc<Self>, RootErrorV0> {
        Self::create_with(
            path,
            run_id,
            max_content_len_bytes,
            FileFixtureV0::default(),
        )
    }

    /// Create a new private root with deterministic fixture inputs.
    pub fn create_with(
        path: &Path,
        run_id: RunIdV0,
        max_content_len_bytes: u64,
        fixture: FileFixtureV0,
    ) -> Result<Arc<Self>, RootErrorV0> {
        if !(1..=MAX_CONTENT_LIMIT).contains(&max_content_len_bytes) {
            return Err(RootErrorV0::ContentLimitOutOfRange);
        }
        match mkdirat(CWD, path, Mode::from_raw_mode(DIR_MODE as _)) {
            Ok(()) => {}
            Err(Errno::EXIST) => return Err(RootErrorV0::AlreadyExists),
            Err(_) => return Err(RootErrorV0::Io),
        }
        let fd =
            openat(CWD, path, dir_flags(), Mode::empty()).map_err(|_| RootErrorV0::UnsafeLayout)?;
        let stat = fstat(&fd).map_err(|_| RootErrorV0::Io)?;
        if !private_dir(&stat, None) {
            return Err(RootErrorV0::UnsafeLayout);
        }
        let identity = (stat.st_dev as u64, stat.st_ino as u64);
        // The canonical absolute path must name the inode just opened.
        let canonical = std::fs::canonicalize(path).map_err(|_| RootErrorV0::Io)?;
        let named =
            statat(CWD, &canonical, AtFlags::SYMLINK_NOFOLLOW).map_err(|_| RootErrorV0::Io)?;
        if (named.st_dev as u64, named.st_ino as u64) != identity {
            return Err(RootErrorV0::UnsafeLayout);
        }
        let canonical = canonical.to_str().ok_or(RootErrorV0::UnsafeLayout)?;
        let nonce = fixture.nonce.unwrap_or_else(fresh_nonce);
        let root_id = format!("root-{}", uuid::Uuid::new_v4().simple());
        let digest = root_digest(&nonce, canonical);

        // Protected namespace; the process owner is that of the directory it creates.
        mkdirat(&fd, ROOT_EVIDENCE_DIR, Mode::from_raw_mode(DIR_MODE as _))
            .map_err(|_| RootErrorV0::Io)?;
        let evidence = openat(&fd, ROOT_EVIDENCE_DIR, dir_flags(), Mode::empty())
            .map_err(|_| RootErrorV0::UnsafeLayout)?;
        let evidence_stat = fstat(&evidence).map_err(|_| RootErrorV0::Io)?;
        let owner = Owner(evidence_stat.st_uid);
        if !private_dir(&evidence_stat, Some(owner)) || stat.st_uid != owner.0 {
            return Err(RootErrorV0::UnsafeLayout);
        }
        let payload_path = format!("{ROOT_EVIDENCE_DIR}/{ROOT_PAYLOAD_DIR}");
        layout::open_dir(&fd, &payload_path, owner, true)
            .map_err(|_| RootErrorV0::UnsafeLayout)?
            .ok_or(RootErrorV0::Io)?;
        let metadata = RootMetadataV0 {
            schema: METADATA_SCHEMA.to_owned(),
            root_id: root_id.clone(),
            root_digest: digest.clone(),
            nonce: hex(&nonce),
            run_id: run_id.as_str().to_owned(),
            max_content_len_bytes,
        };
        let bytes = serde_json::to_vec(&metadata).map_err(|_| RootErrorV0::Io)?;
        object::write_immutable(&evidence, ROOT_METADATA_NAME, &bytes, owner)
            .map_err(|_| RootErrorV0::Io)?;
        let sentinel = RunSentinelV0 {
            schema: SENTINEL_SCHEMA.to_owned(),
            run_id: run_id.as_str().to_owned(),
            root_id: root_id.clone(),
            root_digest: digest.clone(),
        };
        let bytes = serde_json::to_vec(&sentinel).map_err(|_| RootErrorV0::Io)?;
        object::write_immutable(&fd, sentinel_name(&run_id).as_str(), &bytes, owner)
            .map_err(|_| RootErrorV0::Io)?;
        Ok(Arc::new(Self {
            fd: Mutex::new(Some(fd)),
            identity,
            owner,
            root_id,
            nonce,
            root_digest: digest,
            max_content_len_bytes,
            run_id,
            registry: Mutex::new(BTreeSet::new()),
            footprint: Mutex::new(BTreeSet::new()),
            cleared: AtomicBool::new(false),
            target_calls: AtomicU64::new(0),
            fixture,
        }))
    }

    /// Wrap this capability for the landed `ExecutorBindingV0`.
    pub fn opaque(self: &Arc<Self>) -> OpaqueRootCapabilityV0 {
        OpaqueRootCapabilityV0::new(self.clone())
    }

    /// Draft 12 section 10 field 6.
    pub fn root(&self) -> Root {
        Root {
            root_id: self.root_id.clone(),
            root_digest: self.root_digest.clone(),
        }
    }

    /// Generated root id.
    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    /// Draft 12 section 5.6 root digest.
    pub fn root_digest(&self) -> &str {
        &self.root_digest
    }

    /// Construction-bound UTF-8 byte limit for content.
    pub fn max_content_len_bytes(&self) -> u64 {
        self.max_content_len_bytes
    }

    /// Caller-supplied acceptance-run id of the sentinel.
    pub fn run_id(&self) -> &RunIdV0 {
        &self.run_id
    }

    /// Whether the retained fd is open, valid and not yet cleaned.
    pub fn is_live(&self) -> bool {
        self.with_fd(|_| ()).is_some()
    }

    /// Close the retained fd; the capability is never reopened.
    pub fn close(&self) {
        self.fd.lock().expect("root fd").take();
    }

    /// Count of target operations: pre-read, temp create/write/fsync and rename.
    pub fn target_calls(&self) -> u64 {
        self.target_calls.load(Ordering::SeqCst)
    }

    /// Whether `action_id` has been reserved in this live capability.
    pub fn is_reserved(&self, action_id: &ActionIdV0) -> bool {
        self.registry
            .lock()
            .expect("registry")
            .contains(action_id.as_str())
    }

    /// The exact declared cleanup footprint, sorted: final targets, generated
    /// temps, the root-evidence namespace and the run sentinel.
    pub fn declared_footprint(&self) -> Vec<String> {
        let mut out: BTreeSet<String> = self.footprint.lock().expect("footprint").clone();
        out.insert(ROOT_EVIDENCE_ENTRY.to_owned());
        out.insert(sentinel_name(&self.run_id));
        out.into_iter().collect()
    }

    /// Draft 12 section 5.7 over this root and a canonical basename.
    pub fn target_ref_for(&self, basename: &str) -> String {
        target_ref(&digest_bytes(&self.root_digest), basename)
    }

    pub(crate) fn from_opaque(opaque: &OpaqueRootCapabilityV0) -> Option<&Self> {
        (**opaque.inner()).downcast_ref::<Self>()
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    // The retained fd, only while it still names the private root it opened.
    pub(crate) fn with_fd<R>(&self, f: impl FnOnce(&OwnedFd) -> R) -> Option<R> {
        if self.cleared.load(Ordering::SeqCst) {
            return None;
        }
        self.with_fd_for_cleanup(f)
    }

    pub(crate) fn with_fd_for_cleanup<R>(&self, f: impl FnOnce(&OwnedFd) -> R) -> Option<R> {
        let guard = self.fd.lock().expect("root fd");
        let fd = guard.as_ref()?;
        let stat = fstat(fd).ok()?;
        let valid = private_dir(&stat, Some(self.owner))
            && (stat.st_dev as u64, stat.st_ino as u64) == self.identity;
        valid.then(|| f(fd))
    }

    pub(crate) fn owner(&self) -> Owner {
        self.owner
    }

    pub(crate) fn nonce(&self) -> &[u8; 32] {
        &self.nonce
    }

    pub(crate) fn metadata(&self) -> RootMetadataV0 {
        RootMetadataV0 {
            schema: METADATA_SCHEMA.to_owned(),
            root_id: self.root_id.clone(),
            root_digest: self.root_digest.clone(),
            nonce: hex(&self.nonce),
            run_id: self.run_id.as_str().to_owned(),
            max_content_len_bytes: self.max_content_len_bytes,
        }
    }

    pub(crate) fn sentinel(&self) -> RunSentinelV0 {
        RunSentinelV0 {
            schema: SENTINEL_SCHEMA.to_owned(),
            run_id: self.run_id.as_str().to_owned(),
            root_id: self.root_id.clone(),
            root_digest: self.root_digest.clone(),
        }
    }

    // Atomic reservation; false means the id was already used in this root.
    pub(crate) fn reserve(&self, action_id: &ActionIdV0) -> bool {
        self.registry
            .lock()
            .expect("registry")
            .insert(action_id.as_str().to_owned())
    }

    pub(crate) fn declare(&self, basename: &str) {
        self.footprint
            .lock()
            .expect("footprint")
            .insert(basename.to_owned());
    }

    pub(crate) fn count_target_call(&self) {
        self.target_calls.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn fault(&self, fault: FileFaultV0) -> bool {
        self.fixture.faults.contains(&fault)
    }

    pub(crate) fn phase(&self, phase: FilePhaseV0) {
        if let Some(hook) = &self.fixture.hook {
            hook(phase);
        }
    }

    pub(crate) fn mark_cleared(&self) {
        self.cleared.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_cleared(&self) -> bool {
        self.cleared.load(Ordering::SeqCst)
    }
}
