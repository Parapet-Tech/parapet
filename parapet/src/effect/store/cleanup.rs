// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Locked evidence-store cleanup (Routing contract 50747a52, section 4.3 steps 5-7).
//!
//! The store lock, run manifests and terminal history are retained control
//! evidence and are never cleanup targets.

use super::layout::split;
use super::manifest::{ObjectStateV0, RunManifestV0, RunStateV0};
use super::object::{exists, unlink};
use super::terminal::CleanupStateV0;
use super::{FaultPointV0, ManifestRead, ProtectedEvidenceStoreV0};
use crate::effect::interfaces::{RunIdV0, StoreFailureV0};
use std::collections::BTreeSet;

/// Why cleanup did not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupRefusalV0 {
    /// The store lock could not be taken.
    LockFailed,
    /// The target manifest is missing, corrupt or not authoritative.
    ManifestUnusable,
    /// The target has no settled terminal; recover it first.
    TerminalUnsettled,
}

/// Section 4.3 step 7: cleanup result bound by a new terminal generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupResultV0 {
    /// Every target entry released; every unshared named object absent.
    Clear,
    /// A failed unlink, mismatch or control-evidence failure.
    Fail,
    /// An unreadable peer, observed residue, or an unestablished result.
    Indeterminate,
    /// Cleanup did not start; nothing changed.
    Refused(CleanupRefusalV0),
}

enum Preflight {
    Readable(Vec<RunManifestV0>),
    Unreadable,
}

impl ProtectedEvidenceStoreV0 {
    fn preflight_locked(&self) -> Result<Preflight, StoreFailureV0> {
        let mut manifests = Vec::new();
        for name in self.run_names_locked()? {
            let Ok(run) = name else {
                return Ok(Preflight::Unreadable);
            };
            match self.read_manifest_locked(&run) {
                ManifestRead::Valid(m) => manifests.push(m),
                ManifestRead::Missing | ManifestRead::Corrupt => return Ok(Preflight::Unreadable),
            }
        }
        Ok(Preflight::Readable(manifests))
    }

    fn live_paths(manifests: &[RunManifestV0]) -> BTreeSet<String> {
        manifests
            .iter()
            .flat_map(|m| m.objects.iter())
            .filter(|e| e.state.is_live())
            .map(|e| e.path.clone())
            .collect()
    }

    fn object_exists_locked(&self, path: &str) -> Result<bool, StoreFailureV0> {
        let (dir, name) = split(path).ok_or(StoreFailureV0::UnsafeLayout)?;
        match self.dir(dir, false)? {
            Some(fd) => exists(&fd, name),
            None => Ok(false),
        }
    }

    fn unlink_locked(&self, path: &str) -> Result<(), StoreFailureV0> {
        let (dir, name) = split(path).ok_or(StoreFailureV0::UnsafeLayout)?;
        match self.dir(dir, false)? {
            Some(fd) => unlink(&fd, name),
            None => Ok(()),
        }
    }

    // Undeclared residue inside the run's control directories.
    fn residue_locked(&self, manifest: &RunManifestV0) -> Result<bool, StoreFailureV0> {
        let base = format!("run/{}", manifest.run_id.as_str());
        let generation = manifest.terminal.as_ref().map_or(0, |t| t.generation);
        let allowed_terminal: BTreeSet<String> =
            (1..=generation).map(|g| format!("{g}.json")).collect();
        for (relative, allowed) in [
            (
                base.clone(),
                ["manifest.json", "terminal"]
                    .map(String::from)
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
            ),
            (format!("{base}/terminal"), allowed_terminal),
        ] {
            let Some(dir) = self.dir(&relative, false)? else {
                return Ok(true);
            };
            let mut entries = rustix::fs::Dir::read_from(&dir).map_err(|_| StoreFailureV0::Io)?;
            while let Some(entry) = entries.next() {
                let entry = entry.map_err(|_| StoreFailureV0::Io)?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name != "." && name != ".." && !allowed.contains(&name) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Section 4.3 steps 5-7 under one held lock.
    pub fn cleanup_run(&self, run: &RunIdV0) -> CleanupResultV0 {
        let Ok(_lock) = self.lock() else {
            return CleanupResultV0::Refused(CleanupRefusalV0::LockFailed);
        };
        let Ok(authority) = self.authority_locked(run, None) else {
            return match self.read_manifest_locked(run) {
                ManifestRead::Valid(m) if m.terminal.is_none() || m.pending_terminal.is_some() => {
                    CleanupResultV0::Refused(CleanupRefusalV0::TerminalUnsettled)
                }
                _ => CleanupResultV0::Refused(CleanupRefusalV0::ManifestUnusable),
            };
        };
        let mut manifest = authority.manifest;
        let mut terminal = authority.terminal;
        if manifest.state == RunStateV0::Cleaned {
            return CleanupResultV0::Clear;
        }
        let finish = |store: &Self,
                      manifest: &mut RunManifestV0,
                      terminal: &super::RuntimeTerminalV0,
                      result| {
            let (cleanup, state) = match result {
                CleanupResultV0::Clear => (CleanupStateV0::Clear, RunStateV0::Cleaned),
                CleanupResultV0::Fail => (CleanupStateV0::Fail, RunStateV0::CleanupFailed),
                _ => (
                    CleanupStateV0::Indeterminate,
                    RunStateV0::CleanupIndeterminate,
                ),
            };
            match store.publish_successor_locked(manifest, terminal, cleanup, state) {
                Ok(_) => result,
                Err(_) => CleanupResultV0::Fail,
            }
        };
        // Step 5: full readable-manifest preflight before any release or unlink.
        match self.preflight_locked() {
            Ok(Preflight::Readable(_)) => {}
            Ok(Preflight::Unreadable) | Err(_) => {
                return finish(
                    self,
                    &mut manifest,
                    &terminal,
                    CleanupResultV0::Indeterminate,
                );
            }
        }
        // Step 6: publish `in_progress` and enter `cleaning`.
        if terminal.cleanup != CleanupStateV0::InProgress || manifest.state != RunStateV0::Cleaning
        {
            if self
                .publish_successor_locked(
                    &mut manifest,
                    &terminal,
                    CleanupStateV0::InProgress,
                    RunStateV0::Cleaning,
                )
                .is_err()
            {
                return CleanupResultV0::Fail;
            }
            let pointer = manifest.terminal.clone().expect("published pointer");
            match self.read_terminal_locked(&pointer.path, &pointer.digest, pointer.len_bytes) {
                Some(t) => terminal = t,
                None => return CleanupResultV0::Fail,
            }
        }
        let paths: Vec<String> = manifest.objects.iter().map(|e| e.path.clone()).collect();
        for path in paths {
            let index = manifest
                .objects
                .iter()
                .position(|e| e.path == path)
                .expect("entry");
            if manifest.objects[index].state != ObjectStateV0::Released {
                manifest.objects[index].state = ObjectStateV0::Released;
                if self.write_manifest_locked(&mut manifest, false).is_err() {
                    return CleanupResultV0::Fail;
                }
                if self
                    .faults()
                    .check(FaultPointV0::CleanupAfterRelease)
                    .is_err()
                {
                    return CleanupResultV0::Fail;
                }
            }
            let peers = match self.preflight_locked() {
                Ok(Preflight::Readable(m)) => m,
                _ => {
                    return finish(
                        self,
                        &mut manifest,
                        &terminal,
                        CleanupResultV0::Indeterminate,
                    )
                }
            };
            if !Self::live_paths(&peers).contains(&path) && self.unlink_locked(&path).is_err() {
                return finish(self, &mut manifest, &terminal, CleanupResultV0::Fail);
            }
        }
        if self
            .faults()
            .check(FaultPointV0::CleanupAfterUnlink)
            .is_err()
        {
            return CleanupResultV0::Fail;
        }
        // Step 7: declared-path, residue and control-evidence checks.
        let peers = match self.preflight_locked() {
            Ok(Preflight::Readable(m)) => m,
            _ => {
                return finish(
                    self,
                    &mut manifest,
                    &terminal,
                    CleanupResultV0::Indeterminate,
                )
            }
        };
        let live = Self::live_paths(&peers);
        for entry in &manifest.objects {
            match self.object_exists_locked(&entry.path) {
                Ok(present) if present && !live.contains(&entry.path) => {
                    return finish(self, &mut manifest, &terminal, CleanupResultV0::Fail);
                }
                Ok(_) => {}
                Err(_) => {
                    return finish(
                        self,
                        &mut manifest,
                        &terminal,
                        CleanupResultV0::Indeterminate,
                    )
                }
            }
        }
        match self.residue_locked(&manifest) {
            Ok(false) => {}
            _ => {
                return finish(
                    self,
                    &mut manifest,
                    &terminal,
                    CleanupResultV0::Indeterminate,
                )
            }
        }
        if self.authority_locked(run, None).is_err() {
            return finish(self, &mut manifest, &terminal, CleanupResultV0::Fail);
        }
        finish(self, &mut manifest, &terminal, CleanupResultV0::Clear)
    }
}
