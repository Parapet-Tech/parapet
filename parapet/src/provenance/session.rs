// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! In-memory carrier session lifecycle.
//!
//! At most one session is active. Starting a session draws a fresh 32-byte
//! HMAC key and a fresh 128-bit session id from injected entropy and builds a
//! new verifier with an empty replay ledger, replacing any previous session
//! wholesale. Nothing is persisted: a new manager (process restart) begins
//! with no session. Key material is returned once in the grant and is never
//! rendered by `Debug`.

use std::fmt;
use std::sync::Mutex;

use super::auth::{HmacSha256Authenticator, SessionKey, SESSION_KEY_LEN};
use super::verify::{
    CarrierError, CarrierVerifier, CarrierVerifierDeps, Clock, InMemoryReplayLedger,
    MessageContentDecoder, SupportedProfile, VerifiedCarrier,
};

/// Session id entropy in bytes (128 bits), hex-encoded on the wire.
pub const SESSION_ID_BYTES: usize = 16;

/// Upper bound on `max_validity_ms` accepted for a session.
pub const MAX_SESSION_VALIDITY_MS: u64 = 60_000;

/// Injected entropy source.
pub trait EntropySource: Send + Sync {
    /// Fill `buf` entirely with fresh random bytes.
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError>;
}

/// The entropy source could not supply bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropyError;

/// Operating-system CSPRNG.
#[derive(Debug, Default, Clone, Copy)]
pub struct OsEntropy;

impl EntropySource for OsEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
        getrandom::fill(buf).map_err(|_| EntropyError)
    }
}

/// Per-session verifier inputs other than key, id and ledger.
pub struct SessionConfig {
    pub profile: SupportedProfile,
    pub clock: Box<dyn Clock>,
    pub decoder: Box<dyn MessageContentDecoder>,
}

/// Material handed to the session's carrier issuer.
pub struct SessionGrant {
    pub session_id: String,
    pub key: SessionKey,
}

impl fmt::Debug for SessionGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionGrant")
            .field("session_id", &self.session_id)
            .field("key", &self.key)
            .finish()
    }
}

/// Why a session could not be started. Any previous session is ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    /// `max_validity_ms` outside `1..=60000`.
    InvalidMaxValidity,
    /// Entropy source failure.
    Entropy,
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaxValidity => write!(
                f,
                "provenance session max_validity_ms must be in 1..={MAX_SESSION_VALIDITY_MS}"
            ),
            Self::Entropy => f.write_str("provenance session entropy unavailable"),
        }
    }
}

impl std::error::Error for SessionError {}

/// Holder of the single active carrier session.
pub struct SessionManager {
    entropy: Box<dyn EntropySource>,
    active: Mutex<Option<CarrierVerifier>>,
}

impl SessionManager {
    /// Manager backed by the OS CSPRNG, with no active session.
    pub fn new() -> Self {
        Self::with_entropy(Box::new(OsEntropy))
    }

    /// Manager with injected entropy, with no active session.
    pub fn with_entropy(entropy: Box<dyn EntropySource>) -> Self {
        Self {
            entropy,
            active: Mutex::new(None),
        }
    }

    /// Start a session, replacing any active one. On error no session is
    /// active (fail closed).
    pub fn start(&self, config: SessionConfig) -> Result<SessionGrant, SessionError> {
        let mut active = self.lock();
        *active = None;

        if !(1..=MAX_SESSION_VALIDITY_MS).contains(&config.profile.max_validity_ms) {
            return Err(SessionError::InvalidMaxValidity);
        }

        let mut key = SessionKey::from_bytes([0u8; SESSION_KEY_LEN]);
        let mut id = [0u8; SESSION_ID_BYTES];
        self.entropy
            .fill(key.expose_bytes_mut())
            .and_then(|()| self.entropy.fill(&mut id))
            .map_err(|EntropyError| SessionError::Entropy)?;
        let session_id = lower_hex(&id);

        *active = Some(CarrierVerifier::new(CarrierVerifierDeps {
            profile: config.profile,
            session_id: session_id.clone(),
            authenticator: Box::new(HmacSha256Authenticator::new(key.clone())),
            clock: config.clock,
            ledger: Box::new(InMemoryReplayLedger::new()),
            decoder: config.decoder,
        }));
        Ok(SessionGrant { session_id, key })
    }

    /// End the active session, if any.
    pub fn end(&self) {
        *self.lock() = None;
    }

    /// Whether a session is active.
    pub fn is_active(&self) -> bool {
        self.lock().is_some()
    }

    /// Verify against the active session. `None` when no session is active;
    /// callers in the required profile must reject.
    pub fn verify(
        &self,
        envelope: Option<&[u8]>,
        body: &[u8],
    ) -> Option<Result<VerifiedCarrier, CarrierError>> {
        // Verification holds the lock so a rotation cannot race a request
        // onto the replaced session's key or ledger.
        self.lock().as_ref().map(|v| v.verify(envelope, body))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<CarrierVerifier>> {
        self.active.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU8, Ordering};

    use super::*;
    use crate::provenance::envelope::{
        Authenticator, CarrierEnvelope, Segment, Sha256Digest, Source, SourceMap, SCHEMA_VERSION_V1,
    };
    use crate::provenance::verify::{DecodedMessage, VendorManifest, VerificationOutcome};

    const NOW: u64 = 1_700_000_000_000;
    const BODY: &str = "hello carrier";

    /// Each fill writes one distinct byte value, starting at 1.
    struct CountingEntropy(AtomicU8);

    impl EntropySource for CountingEntropy {
        fn fill(&self, buf: &mut [u8]) -> Result<(), EntropyError> {
            let v = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            buf.fill(v);
            Ok(())
        }
    }

    struct FailingEntropy;

    impl EntropySource for FailingEntropy {
        fn fill(&self, _: &mut [u8]) -> Result<(), EntropyError> {
            Err(EntropyError)
        }
    }

    struct FixedClock;

    impl Clock for FixedClock {
        fn now_ms(&self) -> u64 {
            NOW
        }
    }

    /// The whole body is one message.
    struct WholeBody;

    impl MessageContentDecoder for WholeBody {
        fn decode(&self, body: &[u8]) -> Option<Vec<DecodedMessage>> {
            Some(vec![DecodedMessage::new(
                String::from_utf8(body.to_vec()).ok()?,
                false,
            )])
        }
    }

    fn counting() -> SessionManager {
        SessionManager::with_entropy(Box::new(CountingEntropy(AtomicU8::new(0))))
    }

    fn config(max_validity_ms: u64) -> SessionConfig {
        let mut builds = BTreeMap::new();
        builds.insert("b1".to_string(), VendorManifest::default());
        SessionConfig {
            profile: SupportedProfile {
                schema_version: SCHEMA_VERSION_V1,
                profile_id: "opencode".to_string(),
                builds,
                max_validity_ms,
            },
            clock: Box::new(FixedClock),
            decoder: Box::new(WholeBody),
        }
    }

    fn envelope(session_id: &str, key: &SessionKey, sequence: u64) -> Vec<u8> {
        let source_map = SourceMap {
            segments: vec![Segment {
                message_index: 0,
                start: 0,
                end: BODY.len(),
                source: Source::User,
                sha256: Sha256Digest::of(BODY.as_bytes()),
            }],
        };
        let mut env = CarrierEnvelope {
            schema_version: SCHEMA_VERSION_V1,
            profile: "opencode".to_string(),
            build: "b1".to_string(),
            session_id: session_id.to_string(),
            sequence,
            issued_at_ms: NOW - 1,
            expires_at_ms: NOW + 1_000,
            request_body_sha256: Sha256Digest::of(BODY.as_bytes()),
            source_map_sha256: source_map.canonical_digest(),
            source_map,
            authenticator: Authenticator::from_bytes(Vec::new()),
        };
        let tag = HmacSha256Authenticator::new(key.clone()).tag(&env.signed_payload());
        env.authenticator = Authenticator::from_bytes(tag.to_vec());
        serde_json::to_vec(&env).unwrap()
    }

    fn outcome(m: &SessionManager, env: &[u8]) -> Option<VerificationOutcome> {
        m.verify(Some(env), BODY.as_bytes())
            .map(|r| r.map_or_else(|e| e.outcome, |_| VerificationOutcome::Verified))
    }

    #[test]
    fn new_manager_has_no_session() {
        let m = counting();
        assert!(!m.is_active());
        assert!(m.verify(Some(b"{}"), BODY.as_bytes()).is_none());
    }

    #[test]
    fn start_draws_key_and_128_bit_id_from_injected_entropy() {
        let m = counting();
        let grant = m.start(config(60_000)).unwrap();
        assert_eq!(grant.key.expose_bytes(), &[1u8; SESSION_KEY_LEN]);
        assert_eq!(grant.session_id, "02".repeat(SESSION_ID_BYTES));
        assert!(m.is_active());
        let env = envelope(&grant.session_id, &grant.key, 1);
        assert_eq!(outcome(&m, &env), Some(VerificationOutcome::Verified));
        assert_eq!(
            outcome(&m, &env),
            Some(VerificationOutcome::StaleOrReplayed)
        );
    }

    #[test]
    fn max_validity_bounds_are_1_to_60000_inclusive() {
        let m = counting();
        for ok in [1, 60_000] {
            assert!(m.start(config(ok)).is_ok(), "{ok}");
        }
        for bad in [0, 60_001, u64::MAX] {
            m.start(config(60_000)).unwrap();
            assert_eq!(
                m.start(config(bad)).unwrap_err(),
                SessionError::InvalidMaxValidity
            );
            assert!(!m.is_active(), "rejected start must end prior session");
        }
    }

    #[test]
    fn rotation_replaces_key_id_and_ledger() {
        let m = counting();
        let old = m.start(config(60_000)).unwrap();
        let old_env = envelope(&old.session_id, &old.key, 7);
        assert_eq!(outcome(&m, &old_env), Some(VerificationOutcome::Verified));

        let new = m.start(config(60_000)).unwrap();
        assert_ne!(new.session_id, old.session_id);
        assert_ne!(new.key.expose_bytes(), old.key.expose_bytes());

        // Old session's envelope no longer verifies.
        assert_eq!(
            outcome(&m, &envelope(&old.session_id, &old.key, 8)),
            Some(VerificationOutcome::SessionMismatch)
        );
        // Old key under the new id fails authentication.
        assert_eq!(
            outcome(&m, &envelope(&new.session_id, &old.key, 8)),
            Some(VerificationOutcome::SignatureInvalid)
        );
        // Fresh ledger: the sequence consumed in the old session is usable.
        assert_eq!(
            outcome(&m, &envelope(&new.session_id, &new.key, 7)),
            Some(VerificationOutcome::Verified)
        );
    }

    #[test]
    fn end_clears_session() {
        let m = counting();
        let grant = m.start(config(60_000)).unwrap();
        m.end();
        assert!(!m.is_active());
        assert!(outcome(&m, &envelope(&grant.session_id, &grant.key, 1)).is_none());
    }

    #[test]
    fn entropy_failure_fails_closed() {
        let m = SessionManager::with_entropy(Box::new(FailingEntropy));
        assert_eq!(m.start(config(60_000)).unwrap_err(), SessionError::Entropy);
        assert!(!m.is_active());
    }

    #[test]
    fn restart_begins_with_no_session() {
        let first = counting();
        let grant = first.start(config(60_000)).unwrap();
        drop(first);
        let restarted = counting();
        assert!(!restarted.is_active());
        assert!(outcome(&restarted, &envelope(&grant.session_id, &grant.key, 1)).is_none());
    }

    #[test]
    fn os_entropy_yields_distinct_sessions() {
        let m = SessionManager::new();
        let a = m.start(config(60_000)).unwrap();
        let b = m.start(config(60_000)).unwrap();
        assert_eq!(a.session_id.len(), SESSION_ID_BYTES * 2);
        assert!(crate::provenance::envelope::is_valid_identifier(
            &a.session_id
        ));
        assert_ne!(a.session_id, b.session_id);
        assert_ne!(a.key.expose_bytes(), b.key.expose_bytes());
    }

    #[test]
    fn grant_debug_never_renders_key() {
        let m = SessionManager::with_entropy(Box::new(CountingEntropy(AtomicU8::new(0xaa))));
        let grant = m.start(config(60_000)).unwrap();
        let rendered = format!("{grant:?}");
        assert!(rendered.contains("redacted"));
        assert!(!rendered.contains("abab"));
        assert!(!rendered.contains("171"));
    }
}
