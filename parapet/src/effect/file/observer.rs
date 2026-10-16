// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `FilesystemEffectObserverV0`: independent post-attempt measurement
//! (1C brief b76e6b1b section 6.3; draft 12 section 11).
//!
//! The observer is constructed separately from the executor, resolves the
//! proposal's own path through the retained fd, and never reads the attempt's
//! claimed target. It asserts a measurement, not effect truth: only the landed
//! `verify_expected_effect` produces a verification.

use super::path::classify_raw_path;
use super::root::{
    is_declared, list_entries, read_target, FileFaultV0, FilePhaseV0, RootCapabilityV0,
    TargetReadV0,
};
use crate::effect::interfaces::{EffectObserverV0, ObserverFailureV0, OpaqueRootCapabilityV0};
use crate::receipt::*;
use std::sync::Arc;

/// Draft 12 section 11 field 4 observer name.
pub const OBSERVER_NAME: &str = "parapet-file-observer";
/// Draft 12 section 11 field 4 observer version.
pub const OBSERVER_VERSION: &str = "v0";

/// The filesystem observer bound to one root.
pub struct FilesystemEffectObserverV0 {
    root: Arc<RootCapabilityV0>,
}

impl std::fmt::Debug for FilesystemEffectObserverV0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilesystemEffectObserverV0")
            .field("root", &self.root)
            .finish()
    }
}

impl FilesystemEffectObserverV0 {
    /// Construct an observer over a live root; it shares no state with the executor.
    pub fn new(root: Arc<RootCapabilityV0>) -> Self {
        Self { root }
    }

    fn footprint(&self) -> PresentUnavailable<Footprint, FootprintError> {
        let unavailable = PresentUnavailable::Unavailable(FootprintError::NotConfigured);
        if self.root.fault(FileFaultV0::FootprintUnavailable) {
            return unavailable;
        }
        let declared = self.root.declared_footprint();
        match self.root.with_fd(list_entries) {
            Some(Some(entries)) => PresentUnavailable::Present(Footprint {
                undeclared_residue: entries.iter().any(|name| !is_declared(name, &declared)),
                paths_checked: declared,
            }),
            _ => unavailable,
        }
    }
}

fn outside_root() -> PresentUnavailableErrored<Observed, ObservedError> {
    PresentUnavailableErrored::Present(Observed {
        exists: false,
        digest: None,
        len_bytes: None,
        resolved_within_root: false,
    })
}

impl EffectObserverV0 for FilesystemEffectObserverV0 {
    fn identity(&self) -> Observer {
        Observer {
            name: OBSERVER_NAME.to_owned(),
            version: OBSERVER_VERSION.to_owned(),
            source: ObserverSource::FilesystemRead,
        }
    }

    fn observe(
        &self,
        proposal: &Proposal,
        _attempt: &Attempt,
        attempt_ref: &str,
        root: &OpaqueRootCapabilityV0,
    ) -> Result<Observation, ObserverFailureV0> {
        self.root.phase(FilePhaseV0::BeforeObservation);
        let supplied = RootCapabilityV0::from_opaque(root);
        if self.root.fault(FileFaultV0::ObserverFails)
            || !supplied.is_some_and(|r| r.same(&self.root))
        {
            return Err(ObserverFailureV0::ObserverFailed);
        }
        let PresentAbsentErrored::Present(expected) = &proposal.expected_effect else {
            return Err(ObserverFailureV0::ObserverFailed);
        };
        let unresolvable =
            PresentUnavailable::Unavailable(ObservationTargetError::PathUnresolvable);
        let (target_ref, observed) = match classify_raw_path(&expected.target_path) {
            Err(_) => (unresolvable, outside_root()),
            Ok(basename) => {
                let read = self.root.with_fd(|fd| {
                    if self.root.fault(FileFaultV0::ObserveNoAccess) {
                        TargetReadV0::NoAccess
                    } else if self.root.fault(FileFaultV0::ObserveIo) {
                        TargetReadV0::Error
                    } else {
                        read_target(fd, basename)
                    }
                });
                let resolved = PresentUnavailable::Present(self.root.target_ref_for(basename));
                match read.ok_or(ObserverFailureV0::ObserverFailed)? {
                    TargetReadV0::Missing => (
                        resolved,
                        PresentUnavailableErrored::Present(Observed {
                            exists: false,
                            digest: None,
                            len_bytes: None,
                            resolved_within_root: true,
                        }),
                    ),
                    TargetReadV0::Bytes(bytes) => (
                        resolved,
                        PresentUnavailableErrored::Present(Observed {
                            exists: true,
                            digest: Some(payload_digest(&bytes)),
                            len_bytes: Some(bytes.len() as u64),
                            resolved_within_root: true,
                        }),
                    ),
                    // A final symlink is not followed: the name does not resolve within the root.
                    TargetReadV0::Symlink => (unresolvable, outside_root()),
                    TargetReadV0::NoAccess => (
                        resolved,
                        PresentUnavailableErrored::Unavailable(ObservedError::NoReadAccess),
                    ),
                    TargetReadV0::Error => (
                        resolved,
                        PresentUnavailableErrored::Errored(ObservedError::IoError),
                    ),
                }
            }
        };
        Ok(Observation {
            schema_version: ReceiptSchema::V0,
            stage: ObservationStage::Observation,
            attempt_ref: attempt_ref.to_owned(),
            observer: self.identity(),
            target_ref,
            observed,
            footprint: self.footprint(),
        })
    }
}
