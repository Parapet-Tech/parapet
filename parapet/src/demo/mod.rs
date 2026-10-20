// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Deterministic offline demo workflow (`1E-ROUTING-DETERMINISTIC-DEMO`).
//!
//! A document-consuming proposer that stands in for the model behind the
//! landed `HttpSender` seam, the hash-pinned public fixture set it reads, and
//! `parapet demo export`. POSIX-only, like the 1D runtime. Declared from
//! `engine/mod.rs`, so the public path is `parapet::engine::demo`.

pub mod export;
pub mod fixtures;
pub mod interference;
pub mod proposer;
pub mod sender;

pub use fixtures::{
    load_verified_manifest, sha256_hex, CompiledDemoFixturesV0, DemoAuthorshipV0,
    DemoFixtureErrorV0, DemoFixtureSourceV0, DemoJourneyEntryV0, DemoManifestV0,
    DEMO_FIXTURES_SCHEMA, EXPORTED_ASSETS, MANIFEST_NAME,
};
pub use interference::{
    EffectInterferenceV0, OverwriteTargetInterferenceV0, INTERFERENCE_FILL_BYTE,
};
pub use proposer::{
    DemoProposerV0, ManifestDemoProposerV0, ProposedCallV0, ProposerErrorV0, ProposerInputV0,
};
pub use sender::DemoProposerHttpSenderV0;

/// Closed set of 1E workflow journeys that carry a proposer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoJourneyV0 {
    /// J1: benign allowed write with fixed bytes.
    BenignWrite,
    /// J2: injected instruction to write outside the root; policy must block it.
    Unauthorized,
    /// J3: injection-looking benign document; allowed.
    InjectionBenign,
    /// J4: allowed write whose observed bytes are overwritten before observation.
    Mismatch,
}

impl DemoJourneyV0 {
    /// Every journey, in manifest order.
    pub const ALL: [DemoJourneyV0; 4] = [
        DemoJourneyV0::BenignWrite,
        DemoJourneyV0::Unauthorized,
        DemoJourneyV0::InjectionBenign,
        DemoJourneyV0::Mismatch,
    ];

    /// Manifest `journey` name.
    pub fn name(self) -> &'static str {
        match self {
            DemoJourneyV0::BenignWrite => "benign_write",
            DemoJourneyV0::Unauthorized => "unauthorized",
            DemoJourneyV0::InjectionBenign => "injection_benign",
            DemoJourneyV0::Mismatch => "mismatch",
        }
    }

    /// Public `--offline-fixture` selector.
    pub fn selector(self) -> &'static str {
        match self {
            DemoJourneyV0::BenignWrite => "demo_benign_write_v0",
            DemoJourneyV0::Unauthorized => "demo_unauthorized_v0",
            DemoJourneyV0::InjectionBenign => "demo_injection_benign_v0",
            DemoJourneyV0::Mismatch => "demo_mismatch_v0",
        }
    }
}
