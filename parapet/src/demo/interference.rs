// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pre-observation fault-injection interference for the 1E Mismatch journey
//! (`1E-ROUTING-DETERMINISTIC-DEMO`, Section 4.2 and Section 4.4).
//!
//! Simulates a concurrent external writer at `FilePhaseV0::BeforeObservation`.
//! It never reads policy, detection, or the receipt and makes no verification
//! decision: it overwrites the target basename in the effect root with fixed
//! replacement bytes of identical length and swallows any I/O failure.

use super::fixtures::{
    load_verified_manifest, parse_document, CompiledDemoFixturesV0, DemoFixtureSourceV0,
};
use super::DemoJourneyV0;
use crate::effect::file::is_user_basename;
use rustix::fs::{openat, Mode, OFlags, CWD};
use std::io::Write;
use std::path::PathBuf;

/// Fixed fill byte for replacement content (`#`, `0x23`), matching the
/// manifest's `same_length_0x23_fill` specification.
pub const INTERFERENCE_FILL_BYTE: u8 = 0x23;

/// Pre-observation interference used only by the Mismatch journey.
pub trait EffectInterferenceV0: Send + Sync {
    /// Run at `FilePhaseV0::BeforeObservation`, before the observer reads the target.
    fn before_observation(&self);
}

/// Default mismatch interference: overwrites `<effect_root>/<basename>` with
/// fixed replacement bytes of the same length as the expected content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverwriteTargetInterferenceV0 {
    pub effect_root: PathBuf,
    pub basename: String,
    pub bytes: Vec<u8>,
}

impl OverwriteTargetInterferenceV0 {
    /// Construct interference with explicit fields (constructor assigns fields only).
    pub fn new(effect_root: PathBuf, basename: String, bytes: Vec<u8>) -> Self {
        Self {
            effect_root,
            basename,
            bytes,
        }
    }

    /// Construct the default J4 mismatch interference from the verified
    /// compiled-in manifest and `doc_mismatch.txt` body length.
    pub fn new_default(effect_root: PathBuf) -> Option<Self> {
        let manifest = load_verified_manifest(&CompiledDemoFixturesV0).ok()?;
        let entry = manifest.entry(DemoJourneyV0::Mismatch)?;
        let basename = entry.expected_target_basename.clone();
        if !is_user_basename(&basename) {
            return None;
        }
        let doc_bytes = CompiledDemoFixturesV0.asset(&entry.document_path)?;
        let doc = std::str::from_utf8(doc_bytes).ok()?;
        let (_target, body) = parse_document(doc)?;
        let replacement = vec![INTERFERENCE_FILL_BYTE; body.len()];
        Some(Self::new(effect_root, basename, replacement))
    }
}

impl EffectInterferenceV0 for OverwriteTargetInterferenceV0 {
    fn before_observation(&self) {
        if !is_user_basename(&self.basename) {
            return;
        }
        let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let Ok(dir_fd) = openat(CWD, &self.effect_root, dir_flags, Mode::empty()) else {
            return;
        };
        let file_flags = OFlags::WRONLY | OFlags::NOFOLLOW | OFlags::TRUNC | OFlags::CLOEXEC;
        let Ok(file_fd) = openat(&dir_fd, &self.basename, file_flags, Mode::empty()) else {
            return;
        };
        let mut file = std::fs::File::from(file_fd);
        let _ = file.write_all(&self.bytes);
    }
}
