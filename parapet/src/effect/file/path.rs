// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Raw single-basename namespace (1C brief b76e6b1b section 6.1 rows 7-9, as
//! corrected by brief f17b49a6 section 3).
//!
//! The decision layer compares normalized text, so an allowed decision does not
//! prove that the raw path is ASCII. These checks read the exact raw bytes and
//! never normalize: no folded or stripped value becomes an accepted basename.

use crate::receipt::AttemptError;

/// Mandatory profile `matches` source pattern for `write_file.path`.
pub const BASENAME_PATTERN: &str = "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$";
/// Mandatory profile `max_length` for `write_file.path`.
pub const BASENAME_MAX_LEN: u64 = 128;

/// First-match controller row for a raw path outside the user namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathRejectionV0 {
    /// Row 7: the path is absolute.
    Absolute,
    /// Row 8: a parsed component is `.` or `..`.
    Traversal,
    /// Row 9: any other value outside the single-basename namespace.
    OutsideNamespace,
}

impl PathRejectionV0 {
    /// The row of the section 6.1 table.
    pub fn row(self) -> u8 {
        match self {
            Self::Absolute => 7,
            Self::Traversal => 8,
            Self::OutsideNamespace => 9,
        }
    }

    /// The closed draft-12 attempt code of the row.
    pub fn code(self) -> AttemptError {
        match self {
            Self::Absolute | Self::OutsideNamespace => AttemptError::OutsideRoot,
            Self::Traversal => AttemptError::Traversal,
        }
    }
}

/// Whether raw bytes are exactly one user basename:
/// `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` over ASCII bytes.
pub fn is_user_basename(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    (1..=BASENAME_MAX_LEN as usize).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Rows 7-9 in first-match order over the raw proposed path.
pub fn classify_raw_path(raw: &str) -> Result<&str, PathRejectionV0> {
    if raw.starts_with('/') {
        return Err(PathRejectionV0::Absolute);
    }
    if raw.split('/').any(|c| c == "." || c == "..") {
        return Err(PathRejectionV0::Traversal);
    }
    if !is_user_basename(raw) {
        return Err(PathRejectionV0::OutsideNamespace);
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_follow_first_match_order() {
        assert_eq!(classify_raw_path("a.txt"), Ok("a.txt"));
        assert_eq!(classify_raw_path("/../a"), Err(PathRejectionV0::Absolute));
        assert_eq!(classify_raw_path("a/../b"), Err(PathRejectionV0::Traversal));
        assert_eq!(classify_raw_path("."), Err(PathRejectionV0::Traversal));
        for raw in [
            "a/b",
            "a\\b",
            "a\0b",
            ".hidden",
            "\u{FF41}.txt",
            "a\u{200B}.txt",
            "",
        ] {
            assert_eq!(
                classify_raw_path(raw),
                Err(PathRejectionV0::OutsideNamespace),
                "{raw:?}"
            );
        }
        assert!(is_user_basename(&"a".repeat(128)));
        assert!(!is_user_basename(&"a".repeat(129)));
    }
}
