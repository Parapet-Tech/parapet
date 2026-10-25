// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Carrier verification before L0 (carrier Rev1 sections 4, 5 and 8).
//!
//! Check order is fixed: presence, closed schema/profile/build, session,
//! map digest and authenticator, freshness, single-use sequence, exact body
//! digest, map geometry, segment digests, pinned vendor digests. Envelope
//! claims are never echoed into errors.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Mutex;

use super::envelope::{CarrierEnvelope, Segment, Sha256Digest, Source};

/// Injected wall clock, milliseconds since the Unix epoch.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// Injected session authenticator. Key custody is a separate Sensor freeze
/// item; implementations must compare tags in constant time.
pub trait CarrierAuthenticator: Send + Sync {
    fn verify(&self, signed_payload: &[u8], tag: &[u8]) -> bool;
}

/// Injected single-use sequence ledger.
pub trait ReplayLedger: Send + Sync {
    /// Record `sequence` for `session_id`. Returns `false` when it was
    /// already consumed within its validity interval.
    fn consume(&self, session_id: &str, sequence: u64, expires_at_ms: u64, now_ms: u64) -> bool;
}

/// One decoded request message content item in adapter order, with the
/// minimal metadata needed to verify system-slot provenance rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedMessage {
    pub content: String,
    pub is_system_slot: bool,
}

impl DecodedMessage {
    pub fn new(content: impl Into<String>, is_system_slot: bool) -> Self {
        Self {
            content: content.into(),
            is_system_slot,
        }
    }
}

/// Injected request decoder: exact body bytes to decoded message contents
/// and system-slot metadata, in request message order.
pub trait MessageContentDecoder: Send + Sync {
    fn decode(&self, body: &[u8]) -> Option<Vec<DecodedMessage>>;
}

/// In-memory [`ReplayLedger`]. Entries expire with their envelope, so the
/// ledger stays bounded by the validity interval.
#[derive(Default)]
pub struct InMemoryReplayLedger {
    state: Mutex<ReplayState>,
}

#[derive(Default)]
struct ReplayState {
    consumed: HashMap<(String, u64), u64>,
    high_water_ms: u64,
}

impl InMemoryReplayLedger {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ReplayLedger for InMemoryReplayLedger {
    fn consume(&self, session_id: &str, sequence: u64, expires_at_ms: u64, now_ms: u64) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        // Never reopen a replay window after an entry was pruned at a later
        // wall-clock reading. A clock rollback fails closed until the session
        // is restarted or time catches back up.
        if now_ms < state.high_water_ms {
            return false;
        }
        state.high_water_ms = now_ms;
        state.consumed.retain(|_, expiry| *expiry > now_ms);
        let key = (session_id.to_string(), sequence);
        if state.consumed.contains_key(&key) {
            return false;
        }
        state.consumed.insert(key, expires_at_ms);
        true
    }
}

/// Exact pinned vendor-template digests for one supported build.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VendorManifest {
    pinned: BTreeSet<Sha256Digest>,
}

impl VendorManifest {
    pub fn new(pinned: impl IntoIterator<Item = Sha256Digest>) -> Self {
        Self {
            pinned: pinned.into_iter().collect(),
        }
    }

    pub fn contains(&self, digest: &Sha256Digest) -> bool {
        self.pinned.contains(digest)
    }
}

/// Closed supported integration identity. A build without a manifest entry
/// is unsupported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportedProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub builds: BTreeMap<String, VendorManifest>,
    pub max_validity_ms: u64,
}

/// Closed v1 verification outcome (carrier Rev1 section 8, Sensor section 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    Verified,
    Absent,
    SchemaMismatch,
    SessionMismatch,
    StaleOrReplayed,
    BodyDigestMismatch,
    MapIncomplete,
    MapOutOfRange,
    MapOverlap,
    SegmentDigestMismatch,
    SignatureInvalid,
    VendorDigestMismatch,
}

impl VerificationOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Absent => "absent",
            Self::SchemaMismatch => "schema_mismatch",
            Self::SessionMismatch => "session_mismatch",
            Self::StaleOrReplayed => "stale_or_replayed",
            Self::BodyDigestMismatch => "body_digest_mismatch",
            Self::MapIncomplete => "map_incomplete",
            Self::MapOutOfRange => "map_out_of_range",
            Self::MapOverlap => "map_overlap",
            Self::SegmentDigestMismatch => "segment_digest_mismatch",
            Self::SignatureInvalid => "signature_invalid",
            Self::VendorDigestMismatch => "vendor_digest_mismatch",
        }
    }
}

/// Typed, content-free verification failure. Carries only the closed
/// outcome, an optional segment index, and the configured (not claimed)
/// supported identity for operator recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarrierError {
    pub outcome: VerificationOutcome,
    pub segment_index: Option<usize>,
    pub supported_profile: String,
    pub supported_schema_version: u32,
}

impl fmt::Display for CarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provenance carrier rejected: {}", self.outcome.as_str())?;
        if let Some(i) = self.segment_index {
            write!(f, " (segment {i})")?;
        }
        write!(
            f,
            "; supported integration {} schema v{}",
            self.supported_profile, self.supported_schema_version
        )
    }
}

impl std::error::Error for CarrierError {}

/// Verified schema/profile/build identity.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CarrierIdentity {
    pub schema_version: u32,
    pub profile: String,
    pub build: String,
}

/// A carrier that passed every check, with the decoded message contents it
/// was verified against.
#[derive(Clone, PartialEq, Eq)]
pub struct VerifiedCarrier {
    pub identity: CarrierIdentity,
    pub request_body_sha256: Sha256Digest,
    pub source_map_sha256: Sha256Digest,
    pub segments: Vec<Segment>,
    pub messages: Vec<String>,
}

impl fmt::Debug for VerifiedCarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedCarrier")
            .field("identity", &self.identity)
            .field("request_body_sha256", &self.request_body_sha256)
            .field("source_map_sha256", &self.source_map_sha256)
            .field("segments", &self.segments)
            .field(
                "messages",
                &format_args!("<{} redacted>", self.messages.len()),
            )
            .finish()
    }
}

/// Injected verifier dependencies.
pub struct CarrierVerifierDeps {
    pub profile: SupportedProfile,
    pub session_id: String,
    pub authenticator: Box<dyn CarrierAuthenticator>,
    pub clock: Box<dyn Clock>,
    pub ledger: Box<dyn ReplayLedger>,
    pub decoder: Box<dyn MessageContentDecoder>,
}

/// Envelope/map verifier bound to one Parapet session.
pub struct CarrierVerifier {
    deps: CarrierVerifierDeps,
}

impl CarrierVerifier {
    pub fn new(deps: CarrierVerifierDeps) -> Self {
        Self { deps }
    }

    /// Verify `envelope` against the exact raw request `body`.
    pub fn verify(
        &self,
        envelope: Option<&[u8]>,
        body: &[u8],
    ) -> Result<VerifiedCarrier, CarrierError> {
        use VerificationOutcome as O;
        let d = &self.deps;

        let raw = envelope.ok_or_else(|| self.fail(O::Absent, None))?;
        let env = CarrierEnvelope::parse(raw).ok_or_else(|| self.fail(O::SchemaMismatch, None))?;

        let manifest = (env.schema_version == d.profile.schema_version
            && env.profile == d.profile.profile_id)
            .then(|| d.profile.builds.get(&env.build))
            .flatten()
            .ok_or_else(|| self.fail(O::SchemaMismatch, None))?;

        if env.session_id != d.session_id {
            return Err(self.fail(O::SessionMismatch, None));
        }

        if env.source_map.canonical_digest() != env.source_map_sha256
            || !d
                .authenticator
                .verify(&env.signed_payload(), env.authenticator.as_bytes())
        {
            return Err(self.fail(O::SignatureInvalid, None));
        }

        let now = d.clock.now_ms();
        let fresh = env.issued_at_ms < env.expires_at_ms
            && env.expires_at_ms - env.issued_at_ms <= d.profile.max_validity_ms
            && env.issued_at_ms <= now
            && now < env.expires_at_ms;
        if !fresh
            || !d
                .ledger
                .consume(&env.session_id, env.sequence, env.expires_at_ms, now)
        {
            return Err(self.fail(O::StaleOrReplayed, None));
        }

        if Sha256Digest::of(body) != env.request_body_sha256 {
            return Err(self.fail(O::BodyDigestMismatch, None));
        }

        // A body that matches its digest but does not decode has no content
        // for the map to address.
        let decoded = d
            .decoder
            .decode(body)
            .ok_or_else(|| self.fail(O::MapOutOfRange, None))?;

        let segments = env.source_map.segments;
        check_geometry(&segments, &decoded).map_err(|(o, i)| self.fail(o, i))?;

        for (i, s) in segments.iter().enumerate() {
            let bytes = &decoded[s.message_index].content.as_bytes()[s.start..s.end];
            if Sha256Digest::of(bytes) != s.sha256 {
                return Err(self.fail(O::SegmentDigestMismatch, Some(i)));
            }
        }

        for (i, s) in segments.iter().enumerate() {
            if s.source == Source::VendorStatic
                && (!decoded[s.message_index].is_system_slot || !manifest.contains(&s.sha256))
            {
                return Err(self.fail(O::VendorDigestMismatch, Some(i)));
            }
        }

        let messages = decoded.into_iter().map(|m| m.content).collect();

        Ok(VerifiedCarrier {
            identity: CarrierIdentity {
                schema_version: env.schema_version,
                profile: env.profile,
                build: env.build,
            },
            request_body_sha256: env.request_body_sha256,
            source_map_sha256: env.source_map_sha256,
            segments,
            messages,
        })
    }

    fn fail(&self, outcome: VerificationOutcome, segment_index: Option<usize>) -> CarrierError {
        CarrierError {
            outcome,
            segment_index,
            supported_profile: self.deps.profile.profile_id.clone(),
            supported_schema_version: self.deps.profile.schema_version,
        }
    }
}

/// Segments must form a complete, ordered, non-overlapping partition of every
/// decoded message. Revisiting an already covered position (including a
/// reordered segment or earlier message) is an overlap; skipping bytes or
/// leaving a nonempty message uncovered is incomplete. Empty segments,
/// out-of-bounds offsets, and offsets inside a UTF-8 code point are out of
/// range.
fn check_geometry(
    segments: &[Segment],
    messages: &[DecodedMessage],
) -> Result<(), (VerificationOutcome, Option<usize>)> {
    use VerificationOutcome as O;

    let mut msg = 0usize;
    let mut cursor = 0usize;
    for (i, s) in segments.iter().enumerate() {
        let content = messages
            .get(s.message_index)
            .map(|m| m.content.as_str())
            .ok_or((O::MapOutOfRange, Some(i)))?;
        if s.start >= s.end
            || s.end > content.len()
            || !content.is_char_boundary(s.start)
            || !content.is_char_boundary(s.end)
        {
            return Err((O::MapOutOfRange, Some(i)));
        }
        if s.message_index < msg {
            return Err((O::MapOverlap, Some(i)));
        }
        if s.message_index > msg {
            let skipped_nonempty = cursor != messages[msg].content.len()
                || messages[msg + 1..s.message_index]
                    .iter()
                    .any(|m| !m.content.is_empty());
            if skipped_nonempty {
                return Err((O::MapIncomplete, Some(i)));
            }
            msg = s.message_index;
            cursor = 0;
        }
        if s.start < cursor {
            return Err((O::MapOverlap, Some(i)));
        }
        if s.start > cursor {
            return Err((O::MapIncomplete, Some(i)));
        }
        cursor = s.end;
    }

    let tail_uncovered = match messages.get(msg) {
        Some(current) => {
            cursor != current.content.len()
                || messages[msg + 1..].iter().any(|m| !m.content.is_empty())
        }
        None => false,
    };
    if tail_uncovered {
        return Err((O::MapIncomplete, None));
    }
    Ok(())
}
