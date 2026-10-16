// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Relative layout and private-directory checks (Routing contract 50747a52, section 4.3).

use crate::effect::interfaces::{RunIdV0, StageKindV0, StoreFailureV0};
use rustix::fs::{fstat, mkdirat, openat, FileType, Mode, OFlags, CWD};
use rustix::io::Errno;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;

/// Exclusive transition lock, mode `0600`.
pub const LOCK_NAME: &str = ".parapet-store.lock";
/// Private directory mode.
pub const DIR_MODE: u32 = 0o700;
/// Private file mode.
pub const FILE_MODE: u32 = 0o600;

/// `payload/sha256/<64-lowercase-hex>`.
pub fn payload_path(digest: &str) -> String {
    format!("payload/sha256/{digest}")
}

/// `record/sha256/<64-lowercase-hex>.json`.
pub fn record_path(digest: &str) -> String {
    format!("record/sha256/{digest}.json")
}

/// `projection/sha256/<row-digest>.json`.
pub fn projection_path(row_digest: &str) -> String {
    format!("projection/sha256/{row_digest}.json")
}

/// `run/<run-id>/manifest.json`.
pub fn manifest_path(run: &RunIdV0) -> String {
    format!("run/{}/manifest.json", run.as_str())
}

/// `run/<run-id>/terminal/<generation>.json`.
pub fn terminal_path(run: &RunIdV0, generation: u64) -> String {
    format!("run/{}/terminal/{generation}.json", run.as_str())
}

/// Bare lowercase 64-hex check.
pub fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The exact path an object of `kind` and `digest` must occupy.
/// Projection objects are keyed by their row digest, which the caller supplies.
pub fn object_path(kind: StageKindV0, key_digest: &str) -> String {
    match kind {
        StageKindV0::Payload => payload_path(key_digest),
        StageKindV0::Projection => projection_path(key_digest),
        _ => record_path(key_digest),
    }
}

/// Split a validated relative object path into its directory and file name.
pub fn split(path: &str) -> Option<(&str, &str)> {
    let (dir, name) = path.rsplit_once('/')?;
    let valid = !name.is_empty()
        && path
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != ".." && !c.starts_with(".tmp-"));
    valid.then_some((dir, name))
}

/// Owner identity of files this process creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Owner(pub u32);

pub(crate) fn errno_failure(errno: Errno) -> StoreFailureV0 {
    if errno == Errno::LOOP || errno == Errno::NOTDIR || errno == Errno::MLINK {
        StoreFailureV0::UnsafeLayout
    } else {
        StoreFailureV0::Io
    }
}

fn check_dir(fd: &OwnedFd, owner: Option<Owner>) -> Result<(), StoreFailureV0> {
    let stat = fstat(fd).map_err(|_| StoreFailureV0::Io)?;
    let private = FileType::from_raw_mode(stat.st_mode) == FileType::Directory
        && (stat.st_mode as u32) & 0o7777 == DIR_MODE
        && owner.is_none_or(|o| stat.st_uid == o.0);
    if private {
        Ok(())
    } else {
        Err(StoreFailureV0::UnsafeLayout)
    }
}

/// Open or create the caller-supplied root; it must be a private directory.
pub fn open_root(
    root: &Path,
    create: bool,
    foreign_owner: bool,
) -> Result<(OwnedFd, Owner), StoreFailureV0> {
    if create {
        match mkdirat(CWD, root, Mode::from_raw_mode(DIR_MODE as _)) {
            Ok(()) | Err(Errno::EXIST) => {}
            Err(e) => return Err(errno_failure(e)),
        }
    }
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = openat(CWD, root, flags, Mode::empty()).map_err(errno_failure)?;
    check_dir(&fd, None)?;
    let mut owner = probe_owner(&fd)?;
    if foreign_owner {
        owner = Owner(owner.0.wrapping_add(1));
    }
    let stat = fstat(&fd).map_err(|_| StoreFailureV0::Io)?;
    if stat.st_uid != owner.0 {
        return Err(StoreFailureV0::UnsafeLayout);
    }
    Ok((fd, owner))
}

// The process owner is learned from a file it creates, using only `fs` APIs.
fn probe_owner(root: &OwnedFd) -> Result<Owner, StoreFailureV0> {
    let name = format!(".tmp-probe-{}", uuid::Uuid::new_v4().simple());
    let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = openat(
        root,
        name.as_str(),
        flags,
        Mode::from_raw_mode(FILE_MODE as _),
    )
    .map_err(errno_failure)?;
    let uid = fstat(&fd).map_err(|_| StoreFailureV0::Io)?.st_uid;
    drop(fd);
    rustix::fs::unlinkat(root, name.as_str(), rustix::fs::AtFlags::empty())
        .map_err(|_| StoreFailureV0::Io)?;
    Ok(Owner(uid))
}

/// Open (and optionally create) a relative directory, component by component,
/// refusing symlinks and any directory that is not private to `owner`.
/// `Ok(None)` means a component is missing and `create` was false.
pub fn open_dir<Fd: AsFd>(
    root: Fd,
    relative: &str,
    owner: Owner,
    create: bool,
) -> Result<Option<OwnedFd>, StoreFailureV0> {
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut current = openat(root, ".", flags, Mode::empty()).map_err(errno_failure)?;
    for component in relative.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            return Err(StoreFailureV0::UnsafeLayout);
        }
        if create {
            match mkdirat(&current, component, Mode::from_raw_mode(DIR_MODE as _)) {
                Ok(()) | Err(Errno::EXIST) => {}
                Err(e) => return Err(errno_failure(e)),
            }
        }
        let next = match openat(&current, component, flags, Mode::empty()) {
            Ok(fd) => fd,
            Err(Errno::NOENT) => return Ok(None),
            Err(e) => return Err(errno_failure(e)),
        };
        check_dir(&next, Some(owner))?;
        current = next;
    }
    Ok(Some(current))
}
