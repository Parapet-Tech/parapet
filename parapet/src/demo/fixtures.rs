// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! Hash-pinned demo fixture set (`parapet-demo-fixtures/v0`).
//!
//! `parapet/demo/v0/` is the single source of every demo asset byte. The files
//! are embedded once here; the proposer and `parapet demo export` both read
//! them through [`DemoFixtureSourceV0`] and only after
//! [`load_verified_manifest`] has checked every digest and length.

use super::DemoJourneyV0;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Manifest schema identifier.
pub const DEMO_FIXTURES_SCHEMA: &str = "parapet-demo-fixtures/v0";

/// Manifest file name.
pub const MANIFEST_NAME: &str = "manifest.json";

/// Every exported asset, sorted by name.
pub const EXPORTED_ASSETS: [&str; 12] = [
    "config_demo_v0.yaml",
    "doc_benign_write.txt",
    "doc_injection_benign.txt",
    "doc_mismatch.txt",
    "doc_ordinary_overreach.txt",
    "doc_unauthorized.txt",
    "manifest.json",
    "request_benign_write.json",
    "request_injection_benign.json",
    "request_mismatch.json",
    "request_ordinary_overreach.json",
    "request_unauthorized.json",
];

/// Fixed document header (`Task` line); the `Target` line and `---` follow.
const DOCUMENT_TASK_LINE: &str = "Task: save-note\n";
const DOCUMENT_TARGET_PREFIX: &str = "Target: ";
const DOCUMENT_SEPARATOR: &str = "---\n";

/// Source of hash-pinned fixture bytes (compiled-in by default; injectable in tests).
pub trait DemoFixtureSourceV0: Send + Sync {
    /// Exact `manifest.json` bytes.
    fn manifest_bytes(&self) -> &[u8];
    /// Exact bytes of a non-manifest asset, by file name.
    fn asset(&self, name: &str) -> Option<&[u8]>;
}

/// Fixture integrity failure. Always fails closed; never a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoFixtureErrorV0 {
    /// `manifest.json` does not parse or violates the v0 shape.
    ManifestInvalid,
    /// A manifest-named or exported asset is absent from the source.
    AssetMissing,
    /// An asset's SHA-256 or length differs from the manifest.
    DigestMismatch,
}

impl DemoFixtureErrorV0 {
    /// Stable snake_case code.
    pub fn code(self) -> &'static str {
        match self {
            DemoFixtureErrorV0::ManifestInvalid => "manifest_invalid",
            DemoFixtureErrorV0::AssetMissing => "asset_missing",
            DemoFixtureErrorV0::DigestMismatch => "digest_mismatch",
        }
    }
}

/// Parsed `manifest.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoManifestV0 {
    pub schema: String,
    pub packet: String,
    pub upstream_label: String,
    pub measurement_status: String,
    pub config: DemoConfigEntryV0,
    pub journeys: Vec<DemoJourneyEntryV0>,
    pub interference: serde_json::Value,
    pub negative_controls: serde_json::Value,
}

/// Manifest entry for `config_demo_v0.yaml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoConfigEntryV0 {
    pub path: String,
    pub sha256: String,
    pub len_bytes: u64,
    pub contract_hash: String,
}

/// Fixture authorship disclosure.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoAuthorshipV0 {
    /// `human_operator` or `agent`.
    pub kind: String,
    /// `authored` or `substantive_rewrite`.
    pub basis: String,
    /// `YYYY-MM-DD` intake or authoring date.
    pub date: String,
}

/// One journey's pinned document, request, and expectations.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoJourneyEntryV0 {
    pub journey: String,
    pub selector: String,
    pub document_path: String,
    pub document_sha256: String,
    pub document_len_bytes: u64,
    /// `injection` or `benign_non_injection`.
    pub document_class: String,
    pub document_authorship: DemoAuthorshipV0,
    pub request_path: String,
    pub request_sha256: String,
    pub request_len_bytes: u64,
    /// `measured` once a live run froze `expected`, else `predeclared_unmeasured`.
    pub measurement_status: String,
    pub expected_http_status: u16,
    pub expected_tool_name: String,
    pub expected_arguments_json: String,
    pub expected_target_basename: String,
    /// Unauthorized journey only: the out-of-root path the document injects.
    pub injected_path: Option<String>,
    pub canary: String,
    pub expected_file_sha256: Option<String>,
    pub expected: serde_json::Value,
    pub provenance: String,
}

impl DemoManifestV0 {
    /// The entry pinned for `journey`.
    pub fn entry(&self, journey: DemoJourneyV0) -> Option<&DemoJourneyEntryV0> {
        self.journeys.iter().find(|j| j.journey == journey.name())
    }
}

/// Lowercase hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Split a fixture document into `(target, body)`; `body` is every byte after
/// the first `---\n`. `None` if the fixed header block is absent.
pub fn parse_document(document: &str) -> Option<(&str, &str)> {
    let rest = document.strip_prefix(DOCUMENT_TASK_LINE)?;
    let rest = rest.strip_prefix(DOCUMENT_TARGET_PREFIX)?;
    let (target, rest) = rest.split_once('\n')?;
    let body = rest.strip_prefix(DOCUMENT_SEPARATOR)?;
    Some((target, body))
}

/// `write_file` arguments as exact JSON bytes, keys in fixed order.
pub fn write_file_arguments_json(path: &str, content: &str) -> String {
    format!(
        "{{\"content\":{},\"path\":{}}}",
        serde_json::Value::String(content.to_owned()),
        serde_json::Value::String(path.to_owned())
    )
}

fn check_pinned(
    source: &dyn DemoFixtureSourceV0,
    name: &str,
    sha256: &str,
    len_bytes: u64,
) -> Result<(), DemoFixtureErrorV0> {
    let bytes = source.asset(name).ok_or(DemoFixtureErrorV0::AssetMissing)?;
    if bytes.len() as u64 != len_bytes || sha256_hex(bytes) != sha256 {
        return Err(DemoFixtureErrorV0::DigestMismatch);
    }
    Ok(())
}

fn is_iso_date(date: &str) -> bool {
    let b = date.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[8..10].iter().all(u8::is_ascii_digit)
}

fn check_authorship_and_provenance(
    journey: DemoJourneyV0,
    entry: &DemoJourneyEntryV0,
) -> Result<(), DemoFixtureErrorV0> {
    let auth = &entry.document_authorship;
    if !is_iso_date(&auth.date) {
        return Err(DemoFixtureErrorV0::ManifestInvalid);
    }
    match journey {
        DemoJourneyV0::BenignWrite | DemoJourneyV0::Mismatch | DemoJourneyV0::OrdinaryOverreach => {
            let expected_prov = format!(
                "agent-authored on {}; not derived from any corpus",
                auth.date
            );
            if entry.document_class != "benign_non_injection"
                || auth.kind != "agent"
                || auth.basis != "authored"
                || entry.provenance != expected_prov
            {
                return Err(DemoFixtureErrorV0::ManifestInvalid);
            }
        }
        DemoJourneyV0::Unauthorized | DemoJourneyV0::InjectionBenign => {
            if entry.document_class != "injection" || auth.kind != "human_operator" {
                return Err(DemoFixtureErrorV0::ManifestInvalid);
            }
            let basis_text = match auth.basis.as_str() {
                "authored" => "authored by the operator",
                "substantive_rewrite" => {
                    "substantive rewrite by the operator of a pre-existing hand-authored sample"
                }
                _ => return Err(DemoFixtureErrorV0::ManifestInvalid),
            };
            let prefix = format!(
                "Human-operator-authored injection fixture ({basis_text}). Operator statement: \""
            );
            let suffix = format!(
                "\". Packaged byte-for-byte; never edited by an agent. Recorded {}; document sha256:{}.",
                auth.date, entry.document_sha256
            );
            if !entry
                .provenance
                .strip_prefix(&prefix)
                .and_then(|s| s.strip_suffix(&suffix))
                .is_some_and(|stmt| !stmt.is_empty())
            {
                return Err(DemoFixtureErrorV0::ManifestInvalid);
            }
        }
    }
    Ok(())
}

fn check_journey(
    source: &dyn DemoFixtureSourceV0,
    journey: DemoJourneyV0,
    entry: &DemoJourneyEntryV0,
) -> Result<(), DemoFixtureErrorV0> {
    if entry.selector != journey.selector() || entry.expected_tool_name != "write_file" {
        return Err(DemoFixtureErrorV0::ManifestInvalid);
    }
    check_pinned(
        source,
        &entry.document_path,
        &entry.document_sha256,
        entry.document_len_bytes,
    )?;
    check_pinned(
        source,
        &entry.request_path,
        &entry.request_sha256,
        entry.request_len_bytes,
    )?;
    check_authorship_and_provenance(journey, entry)?;
    let document = source
        .asset(&entry.document_path)
        .and_then(|b| std::str::from_utf8(b).ok())
        .ok_or(DemoFixtureErrorV0::ManifestInvalid)?;
    let (target, body) = parse_document(document).ok_or(DemoFixtureErrorV0::ManifestInvalid)?;
    if target != entry.expected_target_basename || body.matches(&entry.canary).count() != 1 {
        return Err(DemoFixtureErrorV0::ManifestInvalid);
    }
    let path = match (journey, entry.injected_path.as_deref()) {
        (DemoJourneyV0::Unauthorized, Some(injected)) if body.contains(injected) => injected,
        (DemoJourneyV0::Unauthorized, _) | (_, Some(_)) => {
            return Err(DemoFixtureErrorV0::ManifestInvalid)
        }
        (_, None) => target,
    };
    if write_file_arguments_json(path, body) != entry.expected_arguments_json {
        return Err(DemoFixtureErrorV0::ManifestInvalid);
    }
    Ok(())
}

/// Parse the source's manifest and verify every pinned asset against it.
pub fn load_verified_manifest(
    source: &dyn DemoFixtureSourceV0,
) -> Result<DemoManifestV0, DemoFixtureErrorV0> {
    let manifest: DemoManifestV0 = serde_json::from_slice(source.manifest_bytes())
        .map_err(|_| DemoFixtureErrorV0::ManifestInvalid)?;
    if manifest.schema != DEMO_FIXTURES_SCHEMA
        || manifest.journeys.len() != DemoJourneyV0::ALL.len()
    {
        return Err(DemoFixtureErrorV0::ManifestInvalid);
    }
    check_pinned(
        source,
        &manifest.config.path,
        &manifest.config.sha256,
        manifest.config.len_bytes,
    )?;
    for journey in DemoJourneyV0::ALL {
        let entry = manifest
            .entry(journey)
            .ok_or(DemoFixtureErrorV0::ManifestInvalid)?;
        check_journey(source, journey, entry)?;
    }
    for name in EXPORTED_ASSETS {
        if name != MANIFEST_NAME && source.asset(name).is_none() {
            return Err(DemoFixtureErrorV0::AssetMissing);
        }
    }
    Ok(manifest)
}

/// The compiled-in fixture set: `parapet/demo/v0/` except `README.md`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompiledDemoFixturesV0;

impl DemoFixtureSourceV0 for CompiledDemoFixturesV0 {
    fn manifest_bytes(&self) -> &[u8] {
        include_bytes!("../../demo/v0/manifest.json")
    }

    fn asset(&self, name: &str) -> Option<&[u8]> {
        let bytes: &'static [u8] = match name {
            "config_demo_v0.yaml" => include_bytes!("../../demo/v0/config_demo_v0.yaml"),
            "doc_benign_write.txt" => include_bytes!("../../demo/v0/doc_benign_write.txt"),
            "doc_injection_benign.txt" => include_bytes!("../../demo/v0/doc_injection_benign.txt"),
            "doc_mismatch.txt" => include_bytes!("../../demo/v0/doc_mismatch.txt"),
            "doc_ordinary_overreach.txt" => {
                include_bytes!("../../demo/v0/doc_ordinary_overreach.txt")
            }
            "doc_unauthorized.txt" => include_bytes!("../../demo/v0/doc_unauthorized.txt"),
            "request_benign_write.json" => {
                include_bytes!("../../demo/v0/request_benign_write.json")
            }
            "request_injection_benign.json" => {
                include_bytes!("../../demo/v0/request_injection_benign.json")
            }
            "request_mismatch.json" => include_bytes!("../../demo/v0/request_mismatch.json"),
            "request_ordinary_overreach.json" => {
                include_bytes!("../../demo/v0/request_ordinary_overreach.json")
            }
            "request_unauthorized.json" => {
                include_bytes!("../../demo/v0/request_unauthorized.json")
            }
            _ => return None,
        };
        Some(bytes)
    }
}
