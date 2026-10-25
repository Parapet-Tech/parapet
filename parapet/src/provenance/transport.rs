// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Carrier header transport (routing v1): one `x-parapet-provenance` header
//! carrying the serialized envelope.
//!
//! Extraction is pure. The upstream header copy never contains the carrier
//! header or any `x-guard-trust*` header, whatever the outcome. Header values
//! are never rendered by `Debug`.

use std::fmt;

use axum::http::HeaderMap;

use super::envelope::MAX_ENVELOPE_BYTES;
use super::verify::VerificationOutcome;
use crate::proxy::strip_guard_trust_headers;

/// Carrier header name (lowercase, as stored by `HeaderMap`).
pub const PROVENANCE_HEADER: &str = "x-parapet-provenance";

/// Carrier header as found on the inbound request.
#[derive(Clone, PartialEq, Eq)]
pub enum CarrierHeader {
    /// No carrier header.
    Absent,
    /// Exactly one carrier header within the size bound.
    Present(Vec<u8>),
    /// Duplicate or oversize carrier header: `schema_mismatch`.
    Rejected(VerificationOutcome),
}

impl CarrierHeader {
    /// Envelope bytes for [`super::CarrierVerifier::verify`], or the outcome
    /// to reject with.
    pub fn envelope(&self) -> Result<Option<&[u8]>, VerificationOutcome> {
        match self {
            Self::Absent => Ok(None),
            Self::Present(bytes) => Ok(Some(bytes)),
            Self::Rejected(outcome) => Err(*outcome),
        }
    }
}

impl fmt::Debug for CarrierHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => f.write_str("Absent"),
            Self::Present(bytes) => write!(f, "Present(<{} bytes redacted>)", bytes.len()),
            Self::Rejected(outcome) => write!(f, "Rejected({})", outcome.as_str()),
        }
    }
}

/// Result of [`extract_carrier_header`].
pub struct CarrierExtraction {
    pub carrier: CarrierHeader,
    /// Inbound headers minus every carrier and `x-guard-trust*` header.
    pub upstream_headers: HeaderMap,
}

impl fmt::Debug for CarrierExtraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.upstream_headers.keys().map(|k| k.as_str()).collect();
        f.debug_struct("CarrierExtraction")
            .field("carrier", &self.carrier)
            .field("upstream_header_names", &names)
            .finish()
    }
}

/// Read the carrier header from `inbound` and build the sanitized upstream
/// header copy.
pub fn extract_carrier_header(inbound: &HeaderMap) -> CarrierExtraction {
    let mut values = inbound.get_all(PROVENANCE_HEADER).iter();
    let carrier = match (values.next(), values.next()) {
        (None, _) => CarrierHeader::Absent,
        (Some(_), Some(_)) => CarrierHeader::Rejected(VerificationOutcome::SchemaMismatch),
        (Some(value), None) if value.len() > MAX_ENVELOPE_BYTES => {
            CarrierHeader::Rejected(VerificationOutcome::SchemaMismatch)
        }
        (Some(value), None) => CarrierHeader::Present(value.as_bytes().to_vec()),
    };

    let mut upstream_headers = inbound.clone();
    upstream_headers.remove(PROVENANCE_HEADER);
    strip_guard_trust_headers(&mut upstream_headers);

    CarrierExtraction {
        carrier,
        upstream_headers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const SECRET: &str = "CANARY-HEADER-VALUE-5e1d";

    fn base() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("content-type", HeaderValue::from_static("application/json"));
        h.insert("authorization", HeaderValue::from_static("Bearer upstream"));
        h.insert("x-guard-trust", HeaderValue::from_static("trusted"));
        h.insert("X-Guard-Trust-Level", HeaderValue::from_static("high"));
        h
    }

    fn assert_sanitized(x: &CarrierExtraction) {
        let up = &x.upstream_headers;
        assert!(up.get_all(PROVENANCE_HEADER).iter().next().is_none());
        assert!(!up.keys().any(|k| k.as_str().starts_with("x-guard-trust")));
        assert_eq!(up.get("content-type").unwrap(), "application/json");
        assert_eq!(up.get("authorization").unwrap(), "Bearer upstream");
        assert_eq!(up.len(), 2);
    }

    #[test]
    fn absent_header_is_absent_and_sanitized() {
        let x = extract_carrier_header(&base());
        assert_eq!(x.carrier, CarrierHeader::Absent);
        assert_eq!(x.carrier.envelope(), Ok(None));
        assert_sanitized(&x);
    }

    #[test]
    fn single_header_is_extracted_verbatim_and_removed() {
        let mut h = base();
        h.insert(
            "X-Parapet-Provenance",
            HeaderValue::from_static(r#"{"a":1}"#),
        );
        let x = extract_carrier_header(&h);
        assert_eq!(x.carrier.envelope(), Ok(Some(&br#"{"a":1}"#[..])));
        assert_sanitized(&x);
    }

    #[test]
    fn duplicate_headers_reject_as_schema_mismatch_and_are_all_removed() {
        let mut h = base();
        h.append(PROVENANCE_HEADER, HeaderValue::from_static("{}"));
        h.append(PROVENANCE_HEADER, HeaderValue::from_static("{}"));
        h.append(PROVENANCE_HEADER, HeaderValue::from_static("{}"));
        let x = extract_carrier_header(&h);
        assert_eq!(
            x.carrier,
            CarrierHeader::Rejected(VerificationOutcome::SchemaMismatch)
        );
        assert_eq!(
            x.carrier.envelope(),
            Err(VerificationOutcome::SchemaMismatch)
        );
        assert_sanitized(&x);
    }

    #[test]
    fn size_bound_is_inclusive_and_oversize_rejects() {
        let at_limit = "a".repeat(MAX_ENVELOPE_BYTES);
        let mut h = base();
        h.insert(PROVENANCE_HEADER, HeaderValue::from_str(&at_limit).unwrap());
        let x = extract_carrier_header(&h);
        assert!(matches!(&x.carrier, CarrierHeader::Present(b) if b.len() == MAX_ENVELOPE_BYTES));
        assert_sanitized(&x);

        let over = "a".repeat(MAX_ENVELOPE_BYTES + 1);
        h.insert(PROVENANCE_HEADER, HeaderValue::from_str(&over).unwrap());
        let x = extract_carrier_header(&h);
        assert_eq!(
            x.carrier,
            CarrierHeader::Rejected(VerificationOutcome::SchemaMismatch)
        );
        assert_sanitized(&x);
    }

    #[test]
    fn empty_value_is_present_and_left_to_closed_parse() {
        let mut h = base();
        h.insert(PROVENANCE_HEADER, HeaderValue::from_static(""));
        let x = extract_carrier_header(&h);
        assert_eq!(x.carrier.envelope(), Ok(Some(&b""[..])));
        assert_sanitized(&x);
    }

    #[test]
    fn inbound_map_is_not_mutated() {
        let mut h = base();
        h.insert(PROVENANCE_HEADER, HeaderValue::from_static("{}"));
        let before = h.clone();
        let _ = extract_carrier_header(&h);
        assert_eq!(h, before);
    }

    #[test]
    fn debug_never_renders_header_values() {
        let mut h = base();
        h.insert(PROVENANCE_HEADER, HeaderValue::from_static(SECRET));
        let rendered = format!("{:?}", extract_carrier_header(&h));
        assert!(!rendered.contains(SECRET));
        assert!(!rendered.contains("Bearer upstream"));
        assert!(rendered.contains("redacted"));
        assert!(rendered.contains("authorization"));
    }
}
