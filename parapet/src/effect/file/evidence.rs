// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Root-evidence payload copies (1C brief b76e6b1b section 6.2; draft 12
//! sections 5.4 and 10 field 14).
//!
//! Content-addressed private copies live at
//! `.parapet-root-evidence-v0/payload/sha256/<digest>` and are written with the
//! landed protected-store primitives: relative no-follow directories, an
//! exclusive private temp, full write, file fsync, atomic rename, re-read and
//! digest/length check. A correctly named existing object is verified, never
//! overwritten.

use super::root::{RootCapabilityV0, ROOT_EVIDENCE_DIR, ROOT_PAYLOAD_DIR};
use crate::effect::store::{layout, object};
use crate::receipt::{payload_digest, PayloadRef, PayloadStore};

/// Persist `bytes` as a root-evidence payload; any failure is `None`.
pub(crate) fn persist_payload(
    root: &RootCapabilityV0,
    bytes: &[u8],
    media_type: &str,
) -> Option<PayloadRef> {
    let digest = payload_digest(bytes);
    let relative = format!("{ROOT_EVIDENCE_DIR}/{ROOT_PAYLOAD_DIR}");
    root.with_fd(|fd| {
        let dir = layout::open_dir(fd, &relative, root.owner(), false).ok()??;
        object::write_immutable(&dir, &digest, bytes, root.owner()).ok()?;
        Some(())
    })??;
    Some(PayloadRef {
        digest,
        len_bytes: bytes.len() as u64,
        media_type: media_type.to_owned(),
        store: PayloadStore::RootEvidence,
    })
}

/// Read a protected copy back by digest, re-checking digest and length.
pub fn read_root_payload(root: &RootCapabilityV0, digest: &str) -> Option<Vec<u8>> {
    if !layout::is_digest(digest) {
        return None;
    }
    let relative = format!("{ROOT_EVIDENCE_DIR}/{ROOT_PAYLOAD_DIR}");
    let bytes = root.with_fd(|fd| {
        let dir = layout::open_dir(fd, &relative, root.owner(), false).ok()??;
        object::read_private(&dir, digest, root.owner()).ok()?
    })??;
    (payload_digest(&bytes) == digest).then_some(bytes)
}
