// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `FileEffectCleanupV0`: idempotent cleanup through the retained capability
//! (1C brief b76e6b1b section 6.4).
//!
//! Cleanup owns no ambient path authority. It verifies the run sentinel, root
//! id, root digest and nonce-bound metadata, then enumerates the root before
//! any removal. Directly observed undeclared residue is retained and forces
//! `indeterminate`; uncertain identity, enumeration or removal never yields
//! `clear`. Cleanup state is outside the Sensor row and changes no receipt.

use super::root::{
    is_declared, list_entries, sentinel_name, FileFaultV0, RootCapabilityV0, RootMetadataV0,
    RunSentinelV0, ROOT_EVIDENCE_DIR, ROOT_EVIDENCE_ENTRY, ROOT_METADATA_NAME,
};
use crate::effect::store::{layout, object};
use rustix::fs::{openat, statat, unlinkat, AtFlags, FileType, Mode, OFlags};
use rustix::io::Errno;
use std::os::fd::OwnedFd;
use std::sync::Arc;

const MAX_EVIDENCE_DEPTH: usize = 4;

/// Closed cleanup results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileCleanupResultV0 {
    /// Identity verified, every declared entry removed, nothing else observed.
    Clear,
    /// A declared entry could not be removed.
    Fail,
    /// Identity, enumeration or residue makes the root state uncertain.
    Indeterminate,
}

impl FileCleanupResultV0 {
    /// Closed telemetry code.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Fail => "fail",
            Self::Indeterminate => "indeterminate",
        }
    }
}

/// Cleanup bound to one root capability.
pub struct FileEffectCleanupV0 {
    root: Arc<RootCapabilityV0>,
}

impl std::fmt::Debug for FileEffectCleanupV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileEffectCleanupV0")
            .field("root", &self.root)
            .finish()
    }
}

fn remove_entry(dir: &OwnedFd, name: &str) -> bool {
    match unlinkat(dir, name, AtFlags::empty()) {
        Ok(()) | Err(Errno::NOENT) => true,
        Err(_) => false,
    }
}

// Remove the contents of a controller-owned directory without following links.
fn remove_tree(dir: &OwnedFd, depth: usize, keep: Option<&str>, owner: layout::Owner) -> bool {
    if depth > MAX_EVIDENCE_DEPTH {
        return false;
    }
    let Some(entries) = list_entries(dir) else {
        return false;
    };
    for name in entries.iter().filter(|n| Some(n.as_str()) != keep) {
        let Ok(stat) = statat(dir, name.as_str(), AtFlags::SYMLINK_NOFOLLOW) else {
            return false;
        };
        if FileType::from_raw_mode(stat.st_mode) == FileType::Directory {
            let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            let Ok(child) = openat(dir, name.as_str(), flags, Mode::empty()) else {
                return false;
            };
            if stat.st_uid != owner.0
                || !remove_tree(&child, depth + 1, None, owner)
                || unlinkat(dir, name.as_str(), AtFlags::REMOVEDIR).is_err()
            {
                return false;
            }
        } else if !remove_entry(dir, name) {
            return false;
        }
    }
    true
}

impl FileEffectCleanupV0 {
    /// Bind cleanup to a capability; no path is accepted.
    pub fn new(root: Arc<RootCapabilityV0>) -> Self {
        Self { root }
    }

    /// Run cleanup. A previous in-process `clear` stays `clear` and touches nothing.
    pub fn cleanup(&self) -> FileCleanupResultV0 {
        let result = if self.root.is_cleared() {
            FileCleanupResultV0::Clear
        } else {
            self.root
                .with_fd_for_cleanup(|fd| self.run(fd))
                .unwrap_or(FileCleanupResultV0::Indeterminate)
        };
        if result == FileCleanupResultV0::Clear {
            self.root.mark_cleared();
        }
        tracing::debug!(
            schema = "parapet-file-cleanup/v0",
            result = result.as_str(),
            "file effect cleanup settled"
        );
        result
    }

    fn identity_matches(&self, fd: &OwnedFd) -> bool {
        let owner = self.root.owner();
        let Ok(Some(evidence)) = layout::open_dir(fd, ROOT_EVIDENCE_DIR, owner, false) else {
            return false;
        };
        let metadata = object::read_private(&evidence, ROOT_METADATA_NAME, owner)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice::<RootMetadataV0>(&b).ok());
        let sentinel = object::read_private(fd, &sentinel_name(self.root.run_id()), owner)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice::<RunSentinelV0>(&b).ok());
        metadata.as_ref() == Some(&self.root.metadata())
            && sentinel.as_ref() == Some(&self.root.sentinel())
    }

    fn remove_declared(&self, fd: &OwnedFd, name: &str) -> bool {
        !self.root.fault(FileFaultV0::CleanupUnlink) && remove_entry(fd, name)
    }

    fn run(&self, fd: &OwnedFd) -> FileCleanupResultV0 {
        use FileCleanupResultV0::*;
        // Identity and complete footprint preflight before any deletion.
        if !self.identity_matches(fd) || self.root.fault(FileFaultV0::CleanupEnumerate) {
            return Indeterminate;
        }
        let Some(entries) = list_entries(fd) else {
            return Indeterminate;
        };
        let declared = self.root.declared_footprint();
        let sentinel = sentinel_name(self.root.run_id());
        let residue = entries.iter().any(|name| !is_declared(name, &declared));
        for name in declared
            .iter()
            .filter(|n| n.as_str() != ROOT_EVIDENCE_ENTRY && **n != sentinel)
        {
            if !self.remove_declared(fd, name) {
                return Fail;
            }
        }
        // Residue keeps the identity anchor and evidence for inspection.
        if residue {
            return Indeterminate;
        }
        let owner = self.root.owner();
        match layout::open_dir(fd, ROOT_EVIDENCE_DIR, owner, false) {
            Ok(Some(evidence)) => {
                if !remove_tree(&evidence, 0, Some(ROOT_METADATA_NAME), owner)
                    || !remove_entry(&evidence, ROOT_METADATA_NAME)
                    || unlinkat(fd, ROOT_EVIDENCE_DIR, AtFlags::REMOVEDIR).is_err()
                {
                    return Fail;
                }
            }
            _ => return Indeterminate,
        }
        if !self.remove_declared(fd, &sentinel) {
            return Fail;
        }
        match list_entries(fd) {
            Some(rest) if rest.is_empty() => Clear,
            _ => Indeterminate,
        }
    }
}
