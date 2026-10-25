// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Production carrier authenticator: HMAC-SHA256 (RFC 2104) under a 32-byte
//! per-session key, with exactly 32-byte tags compared in constant time.
//!
//! Key bytes are never rendered by `Debug` and are overwritten on drop.

use std::fmt;

use sha2::{Digest as _, Sha256};
use subtle::ConstantTimeEq;

use super::verify::CarrierAuthenticator;

/// Session key length in bytes.
pub const SESSION_KEY_LEN: usize = 32;

/// HMAC-SHA256 tag length in bytes. Any other length is rejected.
pub const CARRIER_TAG_LEN: usize = 32;

const SHA256_BLOCK_LEN: usize = 64;

/// 32-byte per-session HMAC key.
pub struct SessionKey([u8; SESSION_KEY_LEN]);

impl SessionKey {
    pub fn from_bytes(bytes: [u8; SESSION_KEY_LEN]) -> Self {
        Self(bytes)
    }

    /// Raw key bytes, for handing to the session's carrier issuer only.
    pub fn expose_bytes(&self) -> &[u8; SESSION_KEY_LEN] {
        &self.0
    }

    pub(super) fn expose_bytes_mut(&mut self) -> &mut [u8; SESSION_KEY_LEN] {
        &mut self.0
    }
}

impl Clone for SessionKey {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SessionKey(<redacted>)")
    }
}

impl Drop for SessionKey {
    fn drop(&mut self) {
        wipe(&mut self.0);
    }
}

/// HMAC-SHA256 [`CarrierAuthenticator`].
#[derive(Clone)]
pub struct HmacSha256Authenticator {
    key: SessionKey,
}

impl HmacSha256Authenticator {
    pub fn new(key: SessionKey) -> Self {
        Self { key }
    }

    /// HMAC-SHA256 tag over `payload`.
    pub fn tag(&self, payload: &[u8]) -> [u8; CARRIER_TAG_LEN] {
        hmac_sha256(self.key.expose_bytes(), payload)
    }
}

impl fmt::Debug for HmacSha256Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HmacSha256Authenticator(<redacted>)")
    }
}

impl CarrierAuthenticator for HmacSha256Authenticator {
    fn verify(&self, signed_payload: &[u8], tag: &[u8]) -> bool {
        if tag.len() != CARRIER_TAG_LEN {
            return false;
        }
        let mut expected = self.tag(signed_payload);
        let ok: bool = expected.ct_eq(tag).into();
        wipe(&mut expected);
        ok
    }
}

/// RFC 2104 HMAC over SHA-256 for a key no longer than one block.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; CARRIER_TAG_LEN] {
    assert!(key.len() <= SHA256_BLOCK_LEN);
    let mut ipad = [0x36u8; SHA256_BLOCK_LEN];
    let mut opad = [0x5cu8; SHA256_BLOCK_LEN];
    for (i, k) in key.iter().enumerate() {
        ipad[i] ^= k;
        opad[i] ^= k;
    }
    let inner = Sha256::new()
        .chain_update(ipad)
        .chain_update(message)
        .finalize();
    let tag = Sha256::new()
        .chain_update(opad)
        .chain_update(inner)
        .finalize()
        .into();
    wipe(&mut ipad);
    wipe(&mut opad);
    tag
}

/// Overwrite secret bytes so the store is not elided as dead.
fn wipe(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, aligned, exclusive reference.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn key(byte: u8) -> SessionKey {
        SessionKey::from_bytes([byte; SESSION_KEY_LEN])
    }

    #[test]
    fn hmac_matches_rfc4231_vectors() {
        let cases: [(Vec<u8>, Vec<u8>, &str); 4] = [
            (
                vec![0x0b; 20],
                b"Hi There".to_vec(),
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe".to_vec(),
                b"what do ya want for nothing?".to_vec(),
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            (
                vec![0xaa; 20],
                vec![0xdd; 50],
                "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
            ),
            (
                (1..=25).collect(),
                vec![0xcd; 50],
                "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
            ),
        ];
        for (k, msg, expected) in cases {
            assert_eq!(hmac_sha256(&k, &msg).to_vec(), hex(expected));
        }
    }

    #[test]
    fn session_key_tag_matches_reference() {
        let mut bytes = [0u8; SESSION_KEY_LEN];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let auth = HmacSha256Authenticator::new(SessionKey::from_bytes(bytes));
        assert_eq!(
            auth.tag(b"parapet").to_vec(),
            hex("a670cceb9e83682b429514a0dc23f5c2d5a4689d6c9f96fb3a6a32775c2898fd")
        );
    }

    #[test]
    fn verify_accepts_exact_tag_under_same_key() {
        let auth = HmacSha256Authenticator::new(key(7));
        let tag = auth.tag(b"payload");
        assert!(auth.verify(b"payload", &tag));
    }

    #[test]
    fn verify_rejects_31_and_33_byte_tags() {
        let auth = HmacSha256Authenticator::new(key(7));
        let tag = auth.tag(b"payload");
        assert!(!auth.verify(b"payload", &tag[..31]));
        let mut long = tag.to_vec();
        long.push(0);
        assert!(!auth.verify(b"payload", &long));
        assert!(!auth.verify(b"payload", &[]));
    }

    #[test]
    fn verify_rejects_correct_tag_under_wrong_key() {
        let signer = HmacSha256Authenticator::new(key(7));
        let other = HmacSha256Authenticator::new(key(8));
        let tag = signer.tag(b"payload");
        assert!(!other.verify(b"payload", &tag));
    }

    #[test]
    fn verify_rejects_tampered_payload_or_tag() {
        let auth = HmacSha256Authenticator::new(key(7));
        let mut tag = auth.tag(b"payload");
        assert!(!auth.verify(b"payloae", &tag));
        tag[31] ^= 1;
        assert!(!auth.verify(b"payload", &tag));
    }

    #[test]
    fn debug_never_renders_key_bytes() {
        let k = SessionKey::from_bytes([0xab; SESSION_KEY_LEN]);
        let auth = HmacSha256Authenticator::new(k.clone());
        for rendered in [format!("{k:?}"), format!("{auth:?}")] {
            assert!(rendered.contains("redacted"));
            assert!(!rendered.contains("ab"));
            assert!(!rendered.contains("171"));
        }
    }
}
