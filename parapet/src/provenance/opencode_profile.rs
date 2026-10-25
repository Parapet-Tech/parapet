// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Pinned OpenCode 2.0.32 supported profile (Sensor W4 item 1).
//!
//! Digest-only: the manifest holds SHA-256 digests of vendor templates and
//! never the template prose. Constants are decoded in const context, so a
//! malformed or duplicated digest fails the build instead of being dropped.
//! Not reachable from any startup path yet (C3).

use std::collections::BTreeMap;

use super::envelope::Sha256Digest;
use super::verify::{SupportedProfile, VendorManifest};

pub const OPENCODE_SCHEMA_VERSION: u32 = 1;
pub const OPENCODE_PROFILE_ID: &str = "opencode";
pub const OPENCODE_BUILD: &str = "843a6c51";
pub const OPENCODE_MAX_VALIDITY_MS: u64 = 60_000;

const OPENCODE_VENDOR_DIGESTS_HEX: [&str; 11] = [
    "3e02ded4dc469e98492398560506ff36c5a9c8867ac834db5705d1449956de19",
    "9876f53ad077e81d56ceb42ebd0313da8577d2947af44271faf973c454ad66fd",
    "9615947d74b9721b32526cc9098e2d5e24a8452775ec1e506f3cdfac4e827f7f",
    "d110fe79c05880b0a738ef3942fb549b5776e4012112b30a65cd4f8d457e11cf",
    "6086cb52d6d95c7c607afefd77be849807369e1f973039449713148c75d3361a",
    "996648310cf03de88c250168959cecdc49bd6c11d5df14f4f95b372318c66908",
    "f6b8a3d366d47d8a54f3e9ab6bbc0e1406a5241e487b02038a0b902e2ffe1570",
    "1ffd1d46e14c2699759ad6671aa13aae9d41d31391cdc24b32b97d2007a01cdf",
    "166c8993ad8a9d0ece132d6f411b99ddbaaef164a0b02bf3678c44be3ca3e595",
    "d2f45520a3d99f523c139cde9409420ed6662514b506ca4aba76df659bfe5997",
    "d7d778c42104edf39b30284adf458e38a0ae1d97fd02d3d51721057562651627",
];

/// Decoded pinned digests; evaluated at compile time.
const OPENCODE_VENDOR_DIGESTS: [[u8; 32]; 11] = decode_all(&OPENCODE_VENDOR_DIGESTS_HEX);

const fn hex_nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("pinned digest must be lowercase hex"),
    }
}

const fn decode_digest(hex: &str) -> [u8; 32] {
    let raw = hex.as_bytes();
    assert!(raw.len() == 64, "pinned digest must be 64 hex characters");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (hex_nibble(raw[2 * i]) << 4) | hex_nibble(raw[2 * i + 1]);
        i += 1;
    }
    out
}

const fn digests_equal(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn decode_all<const N: usize>(hex: &[&str; N]) -> [[u8; 32]; N] {
    let mut out = [[0u8; 32]; N];
    let mut i = 0;
    while i < N {
        out[i] = decode_digest(hex[i]);
        let mut j = 0;
        while j < i {
            assert!(
                !digests_equal(&out[i], &out[j]),
                "pinned digests must be distinct"
            );
            j += 1;
        }
        i += 1;
    }
    out
}

/// The exact supported OpenCode 2.0.32 profile: one build, eleven pinned
/// vendor-template digests.
pub fn opencode_2_0_32_profile() -> SupportedProfile {
    let manifest = VendorManifest::new(
        OPENCODE_VENDOR_DIGESTS
            .iter()
            .map(|bytes| Sha256Digest::from_bytes(*bytes)),
    );
    let mut builds = BTreeMap::new();
    builds.insert(OPENCODE_BUILD.to_string(), manifest);
    SupportedProfile {
        schema_version: OPENCODE_SCHEMA_VERSION,
        profile_id: OPENCODE_PROFILE_ID.to_string(),
        builds,
        max_validity_ms: OPENCODE_MAX_VALIDITY_MS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_has_exact_identity_and_validity_bound() {
        let profile = opencode_2_0_32_profile();
        assert_eq!(profile.schema_version, 1);
        assert_eq!(profile.profile_id, "opencode");
        assert_eq!(profile.max_validity_ms, 60_000);
    }

    #[test]
    fn profile_supports_exactly_one_build() {
        let profile = opencode_2_0_32_profile();
        let builds: Vec<&str> = profile.builds.keys().map(String::as_str).collect();
        assert_eq!(builds, vec!["843a6c51"]);
    }

    #[test]
    fn manifest_accepts_all_eleven_pinned_digests() {
        let profile = opencode_2_0_32_profile();
        let manifest = &profile.builds["843a6c51"];
        for hex in OPENCODE_VENDOR_DIGESTS_HEX {
            let digest = Sha256Digest::from_hex(hex).expect("runtime parser agrees");
            assert!(manifest.contains(&digest), "missing pinned digest {hex}");
        }
        assert_eq!(
            *manifest,
            VendorManifest::new(
                OPENCODE_VENDOR_DIGESTS_HEX
                    .iter()
                    .map(|h| Sha256Digest::from_hex(h).unwrap()),
            )
        );
    }

    #[test]
    fn manifest_has_exactly_eleven_members() {
        let profile = opencode_2_0_32_profile();
        let manifest = &profile.builds["843a6c51"];
        // Dropping any one pinned digest must change the manifest.
        for skip in 0..OPENCODE_VENDOR_DIGESTS_HEX.len() {
            let fewer = VendorManifest::new(
                OPENCODE_VENDOR_DIGESTS_HEX
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip)
                    .map(|(_, h)| Sha256Digest::from_hex(h).unwrap()),
            );
            assert_ne!(*manifest, fewer);
        }
    }

    #[test]
    fn manifest_rejects_unknown_digest() {
        let profile = opencode_2_0_32_profile();
        let manifest = &profile.builds["843a6c51"];
        assert!(!manifest.contains(&Sha256Digest::of(b"not a pinned template")));
        assert!(!manifest.contains(&Sha256Digest::from_bytes([0u8; 32])));
    }

    #[test]
    fn const_decoder_matches_runtime_parser() {
        for (hex, bytes) in OPENCODE_VENDOR_DIGESTS_HEX
            .iter()
            .zip(OPENCODE_VENDOR_DIGESTS.iter())
        {
            assert_eq!(
                Sha256Digest::from_hex(hex),
                Some(Sha256Digest::from_bytes(*bytes))
            );
        }
    }
}
