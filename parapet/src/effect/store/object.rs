// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! No-follow private file primitives (Routing contract 50747a52, section 4.3).
//!
//! Every write is a random same-directory exclusive no-follow temporary, a full
//! write, a file fsync, a relative atomic rename, then a re-read with digest and
//! length checks. No directory fsync is performed or claimed.

use super::layout::{errno_failure, Owner, FILE_MODE};
use crate::effect::interfaces::StoreFailureV0;
use rustix::fs::{
    fstat, openat, renameat, renameat_with, unlinkat, AtFlags, FileType, Mode, OFlags, RenameFlags,
};
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::OwnedFd;

/// Raw untagged sha256 of bytes.
pub fn raw_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn open_private(dir: &OwnedFd, name: &str, owner: Owner) -> Result<Option<File>, StoreFailureV0> {
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = match openat(dir, name, flags, Mode::empty()) {
        Ok(fd) => fd,
        Err(Errno::NOENT) => return Ok(None),
        Err(Errno::LOOP) => return Err(StoreFailureV0::UnsafeObject),
        Err(e) => return Err(errno_failure(e)),
    };
    let stat = fstat(&fd).map_err(|_| StoreFailureV0::Io)?;
    let private = FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
        && (stat.st_mode as u32) & 0o7777 == FILE_MODE
        && stat.st_uid == owner.0;
    if !private {
        return Err(StoreFailureV0::UnsafeObject);
    }
    Ok(Some(File::from(fd)))
}

/// Read a private regular file without following a final symlink.
/// `Ok(None)` means the name does not exist.
pub fn read_private(
    dir: &OwnedFd,
    name: &str,
    owner: Owner,
) -> Result<Option<Vec<u8>>, StoreFailureV0> {
    let Some(mut file) = open_private(dir, name, owner)? else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|_| StoreFailureV0::Io)?;
    Ok(Some(bytes))
}

fn write_temp(dir: &OwnedFd, bytes: &[u8]) -> Result<String, StoreFailureV0> {
    let name = format!(".tmp-{}", uuid::Uuid::new_v4().simple());
    let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = openat(
        dir,
        name.as_str(),
        flags,
        Mode::from_raw_mode(FILE_MODE as _),
    )
    .map_err(errno_failure)?;
    let mut file = File::from(fd);
    let written = file.write_all(bytes).and_then(|()| file.sync_all());
    if written.is_err() {
        let _ = unlinkat(dir, name.as_str(), AtFlags::empty());
        return Err(StoreFailureV0::Io);
    }
    Ok(name)
}

fn verify_reread(
    dir: &OwnedFd,
    name: &str,
    bytes: &[u8],
    owner: Owner,
) -> Result<(), StoreFailureV0> {
    let reread = read_private(dir, name, owner)?.ok_or(StoreFailureV0::Io)?;
    if raw_digest(&reread) != raw_digest(bytes) {
        return Err(StoreFailureV0::DigestMismatch);
    }
    if reread.len() != bytes.len() {
        return Err(StoreFailureV0::LengthMismatch);
    }
    Ok(())
}

/// Atomically replace a mutable control file (manifests).
pub fn write_replace(
    dir: &OwnedFd,
    name: &str,
    bytes: &[u8],
    owner: Owner,
) -> Result<(), StoreFailureV0> {
    // An existing name must itself be a private regular file.
    open_private(dir, name, owner)?;
    let temp = write_temp(dir, bytes)?;
    if renameat(dir, temp.as_str(), dir, name).is_err() {
        let _ = unlinkat(dir, temp.as_str(), AtFlags::empty());
        return Err(StoreFailureV0::Io);
    }
    verify_reread(dir, name, bytes, owner)
}

/// Create an immutable object; an existing name is verified, never overwritten.
pub fn write_immutable(
    dir: &OwnedFd,
    name: &str,
    bytes: &[u8],
    owner: Owner,
) -> Result<(), StoreFailureV0> {
    if let Some(existing) = read_private(dir, name, owner)? {
        return if existing == bytes {
            Ok(())
        } else {
            Err(StoreFailureV0::Collision)
        };
    }
    let temp = write_temp(dir, bytes)?;
    match renameat_with(dir, temp.as_str(), dir, name, RenameFlags::NOREPLACE) {
        Ok(()) => {}
        Err(Errno::EXIST) => {
            let _ = unlinkat(dir, temp.as_str(), AtFlags::empty());
            let existing = read_private(dir, name, owner)?.ok_or(StoreFailureV0::Io)?;
            return if existing == bytes {
                Ok(())
            } else {
                Err(StoreFailureV0::Collision)
            };
        }
        Err(_) => {
            let _ = unlinkat(dir, temp.as_str(), AtFlags::empty());
            return Err(StoreFailureV0::Io);
        }
    }
    verify_reread(dir, name, bytes, owner)
}

/// Remove an object name; a missing name is already removed.
pub fn unlink(dir: &OwnedFd, name: &str) -> Result<(), StoreFailureV0> {
    match unlinkat(dir, name, AtFlags::empty()) {
        Ok(()) | Err(Errno::NOENT) => Ok(()),
        Err(_) => Err(StoreFailureV0::Io),
    }
}

/// Whether a name exists without following it.
pub fn exists(dir: &OwnedFd, name: &str) -> Result<bool, StoreFailureV0> {
    match rustix::fs::statat(dir, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(true),
        Err(Errno::NOENT) => Ok(false),
        Err(_) => Err(StoreFailureV0::Io),
    }
}
