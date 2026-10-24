// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, document-driven proposer that stands in for the model.
//!
//! It never reads policy, detection, or the receipt and decides nothing: it
//! maps one hash-pinned document to one `write_file` call.

use super::fixtures::{
    load_verified_manifest, parse_document, sha256_hex, write_file_arguments_json,
    CompiledDemoFixturesV0, DemoFixtureErrorV0, DemoFixtureSourceV0, DemoManifestV0,
};
use super::DemoJourneyV0;
use std::sync::OnceLock;

/// Pure input: the journey and the untrusted document as the engine forwarded it.
pub struct ProposerInputV0<'a> {
    pub journey: DemoJourneyV0,
    pub document: &'a str,
}

/// One proposed tool call, arguments as exact JSON bytes (keys in fixed order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedCallV0 {
    pub tool_name: String,
    pub arguments_json: String,
}

/// Why no call was proposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposerErrorV0 {
    /// The request carried no user message.
    DocumentMissing,
    /// The document's SHA-256 is not pinned by the manifest.
    DocumentUnknown,
    /// The document is pinned, but for a different journey.
    JourneyDocumentMismatch,
    /// The fixture source failed integrity verification.
    FixtureDigestMismatch,
}

impl ProposerErrorV0 {
    /// Stable snake_case code.
    pub fn code(self) -> &'static str {
        match self {
            ProposerErrorV0::DocumentMissing => "document_missing",
            ProposerErrorV0::DocumentUnknown => "document_unknown",
            ProposerErrorV0::JourneyDocumentMismatch => "journey_document_mismatch",
            ProposerErrorV0::FixtureDigestMismatch => "fixture_digest_mismatch",
        }
    }
}

/// Deterministic, side-effect-free proposer. Same input → same bytes, always.
pub trait DemoProposerV0: Send + Sync {
    fn propose(&self, input: &ProposerInputV0<'_>) -> Result<ProposedCallV0, ProposerErrorV0>;
}

/// Proposer bound to a fixture source; verifies the source at first use.
pub struct ManifestDemoProposerV0<S: DemoFixtureSourceV0> {
    source: S,
    verified: OnceLock<Result<DemoManifestV0, DemoFixtureErrorV0>>,
}

impl<S: DemoFixtureSourceV0> ManifestDemoProposerV0<S> {
    /// Bind `source`; verification is deferred to the first `propose`.
    pub fn new(source: S) -> Self {
        Self {
            source,
            verified: OnceLock::new(),
        }
    }
}

impl ManifestDemoProposerV0<CompiledDemoFixturesV0> {
    /// Proposer over the compiled-in fixture set.
    pub fn new_default() -> Self {
        Self::new(CompiledDemoFixturesV0)
    }
}

impl<S: DemoFixtureSourceV0> DemoProposerV0 for ManifestDemoProposerV0<S> {
    fn propose(&self, input: &ProposerInputV0<'_>) -> Result<ProposedCallV0, ProposerErrorV0> {
        let manifest = self
            .verified
            .get_or_init(|| load_verified_manifest(&self.source))
            .as_ref()
            .map_err(|_| ProposerErrorV0::FixtureDigestMismatch)?;
        let digest = sha256_hex(input.document.as_bytes());
        let entry = manifest
            .entry(input.journey)
            .ok_or(ProposerErrorV0::FixtureDigestMismatch)?;
        if entry.document_sha256 != digest {
            return Err(
                if manifest
                    .journeys
                    .iter()
                    .any(|j| j.document_sha256 == digest)
                {
                    ProposerErrorV0::JourneyDocumentMismatch
                } else {
                    ProposerErrorV0::DocumentUnknown
                },
            );
        }
        // The digest matched a verified entry, so the header parses.
        let (target, body) =
            parse_document(input.document).ok_or(ProposerErrorV0::DocumentUnknown)?;
        let path = match input.journey {
            // A compromised agent obeys the injected instruction.
            DemoJourneyV0::Unauthorized => entry
                .injected_path
                .as_deref()
                .ok_or(ProposerErrorV0::FixtureDigestMismatch)?,
            DemoJourneyV0::BenignWrite
            | DemoJourneyV0::InjectionBenign
            | DemoJourneyV0::Mismatch
            | DemoJourneyV0::OrdinaryOverreach => target,
        };
        Ok(ProposedCallV0 {
            tool_name: entry.expected_tool_name.clone(),
            arguments_json: write_file_arguments_json(path, body),
        })
    }
}
