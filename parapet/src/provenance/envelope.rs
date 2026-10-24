// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Closed carrier envelope and canonical source map (carrier Rev1 sections 4-5).
//!
//! Parsing is closed: unknown fields, unknown source values, malformed digests
//! and out-of-charset identifiers are all rejected. Canonical byte encodings
//! are line-oriented and contain only closed enum names, integers, lowercase
//! hex digests and charset-restricted identifiers.

use std::fmt;

use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Closed v1 carrier schema version.
pub const SCHEMA_VERSION_V1: u32 = 1;

/// Upper bound on envelope bytes accepted for parsing.
pub const MAX_ENVELOPE_BYTES: usize = 1 << 20;

/// Upper bound on identifier length (profile, build, session).
pub const MAX_IDENTIFIER_LEN: usize = 128;

const SIGNED_PAYLOAD_DOMAIN: &str = "parapet-opencode-carrier-v1";
const SOURCE_MAP_DOMAIN: &str = "parapet-opencode-source-map-v1";

/// SHA-256 digest serialized as 64 lowercase hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Digest of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Wrap raw digest bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Parse exactly 64 lowercase hex characters.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let raw = decode_lower_hex(hex)?;
        let bytes: [u8; 32] = raw.try_into().ok()?;
        Some(Self(bytes))
    }

    /// Lowercase hex encoding.
    pub fn to_hex(&self) -> String {
        encode_lower_hex(&self.0)
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.to_hex())
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).ok_or_else(|| de::Error::custom("invalid sha256 digest"))
    }
}

/// Closed v1 source enum (carrier Rev1 section 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    VendorStatic,
    ConfiguredAgent,
    Repository,
    User,
    ToolResult,
    Attachment,
    Unknown,
}

impl Source {
    /// Closed wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Source::VendorStatic => "vendor_static",
            Source::ConfiguredAgent => "configured_agent",
            Source::Repository => "repository",
            Source::User => "user",
            Source::ToolResult => "tool_result",
            Source::Attachment => "attachment",
            Source::Unknown => "unknown",
        }
    }
}

/// One `[start,end)` byte range of decoded message content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub message_index: usize,
    pub start: usize,
    pub end: usize,
    pub source: Source,
    pub sha256: Sha256Digest,
}

/// Canonical source map: ordered segments only, no descriptive strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMap {
    pub segments: Vec<Segment>,
}

impl SourceMap {
    /// Canonical bytes covered by `source_map_sha256`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::with_capacity(32 + self.segments.len() * 96);
        out.push_str(SOURCE_MAP_DOMAIN);
        out.push('\n');
        for s in &self.segments {
            out.push_str(&format!(
                "{} {} {} {} {}\n",
                s.message_index,
                s.start,
                s.end,
                s.source.as_str(),
                s.sha256.to_hex()
            ));
        }
        out.into_bytes()
    }

    /// SHA-256 of [`SourceMap::canonical_bytes`].
    pub fn canonical_digest(&self) -> Sha256Digest {
        Sha256Digest::of(&self.canonical_bytes())
    }
}

/// Authenticator tag bytes. Never rendered by `Debug`.
#[derive(Clone, PartialEq, Eq)]
pub struct Authenticator(Vec<u8>);

impl Authenticator {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Authenticator(<redacted>)")
    }
}

impl Serialize for Authenticator {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode_lower_hex(&self.0))
    }
}

impl<'de> Deserialize<'de> for Authenticator {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match decode_lower_hex(&s) {
            Some(raw) if !raw.is_empty() => Ok(Self(raw)),
            _ => Err(de::Error::custom("invalid authenticator encoding")),
        }
    }
}

/// Closed carrier envelope (carrier Rev1 section 4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarrierEnvelope {
    pub schema_version: u32,
    pub profile: String,
    pub build: String,
    pub session_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub request_body_sha256: Sha256Digest,
    pub source_map_sha256: Sha256Digest,
    pub source_map: SourceMap,
    pub authenticator: Authenticator,
}

impl CarrierEnvelope {
    /// Closed parse. `None` for oversize input, malformed JSON, unknown
    /// fields or enum values, malformed digests, or out-of-charset identifiers.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_ENVELOPE_BYTES {
            return None;
        }
        let envelope: Self = serde_json::from_slice(bytes).ok()?;
        let identifiers_ok = [&envelope.profile, &envelope.build, &envelope.session_id]
            .iter()
            .all(|id| is_valid_identifier(id));
        identifiers_ok.then_some(envelope)
    }

    /// Canonical bytes the authenticator covers: every field except the
    /// authenticator itself. The source map is covered via its digest.
    pub fn signed_payload(&self) -> Vec<u8> {
        format!(
            "{SIGNED_PAYLOAD_DOMAIN}\n\
             schema_version={}\n\
             profile={}\n\
             build={}\n\
             session_id={}\n\
             sequence={}\n\
             issued_at_ms={}\n\
             expires_at_ms={}\n\
             request_body_sha256={}\n\
             source_map_sha256={}\n",
            self.schema_version,
            self.profile,
            self.build,
            self.session_id,
            self.sequence,
            self.issued_at_ms,
            self.expires_at_ms,
            self.request_body_sha256.to_hex(),
            self.source_map_sha256.to_hex(),
        )
        .into_bytes()
    }
}

/// Identifiers are 1..=128 chars of `[A-Za-z0-9._:+-]`, so canonical
/// payload lines cannot be forged by embedded separators.
pub fn is_valid_identifier(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_IDENTIFIER_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'+' | b'-'))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn decode_lower_hex(hex: &str) -> Option<Vec<u8>> {
    fn nibble(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        }
    }
    let bytes = hex.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    bytes
        .chunks_exact(2)
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}
