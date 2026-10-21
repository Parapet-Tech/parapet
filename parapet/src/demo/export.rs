// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

//! `parapet demo export --out <new-dir> [--json]`
//! (schema `parapet-cli-demo-export/v0`).
//!
//! Writes the embedded, hash-verified demo assets into a new directory so the
//! demo runs from an installed binary in any working directory. No network
//! I/O; it never reads the working directory, the checkout, or a state dir.

use super::fixtures::{
    load_verified_manifest, sha256_hex, CompiledDemoFixturesV0, DemoFixtureSourceV0,
    EXPORTED_ASSETS, MANIFEST_NAME,
};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

/// Output schema identifier.
pub const DEMO_EXPORT_SCHEMA: &str = "parapet-cli-demo-export/v0";

const COMMAND: &str = "demo export";

/// One exported file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DemoExportFileV0 {
    pub name: String,
    pub sha256: String,
    pub len_bytes: u64,
}

/// Successful export: files sorted by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemoExportReportV0 {
    pub out_dir: PathBuf,
    pub manifest_sha256: String,
    pub files: Vec<DemoExportFileV0>,
}

/// Export failure; nothing outside this invocation's own files is touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoExportErrorV0 {
    /// `--out` already exists.
    OutAlreadyExists,
    /// `--out` is empty, traverses a parent, or its parent is not a directory.
    InvalidPath,
    /// An embedded asset failed verification; nothing was written.
    AssetDigestMismatch,
    /// Writing failed; files this invocation created were removed.
    WriteFailed,
}

impl DemoExportErrorV0 {
    /// Stable `parapet-cli-error/v0` `error_code`.
    pub fn code(self) -> &'static str {
        match self {
            DemoExportErrorV0::OutAlreadyExists => "demo_out_already_exists",
            DemoExportErrorV0::InvalidPath => "invalid_path",
            DemoExportErrorV0::AssetDigestMismatch => "demo_asset_digest_mismatch",
            DemoExportErrorV0::WriteFailed => "demo_export_write_failed",
        }
    }

    /// Exit code (2 usage/path, 4 runtime).
    pub fn exit_code(self) -> i32 {
        match self {
            DemoExportErrorV0::OutAlreadyExists | DemoExportErrorV0::InvalidPath => 2,
            DemoExportErrorV0::AssetDigestMismatch | DemoExportErrorV0::WriteFailed => 4,
        }
    }

    fn message(self) -> &'static str {
        match self {
            DemoExportErrorV0::OutAlreadyExists => "--out must be a new non-existent path",
            DemoExportErrorV0::InvalidPath => {
                "--out must name a new directory inside an existing directory, without parent traversal"
            }
            DemoExportErrorV0::AssetDigestMismatch => {
                "embedded demo assets failed digest verification; nothing was written"
            }
            DemoExportErrorV0::WriteFailed => {
                "failed to write demo assets; files created by this invocation were removed"
            }
        }
    }
}

fn check_out_path(out: &Path) -> Result<(), DemoExportErrorV0> {
    if out.as_os_str().is_empty() || out.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(DemoExportErrorV0::InvalidPath);
    }
    if std::fs::symlink_metadata(out).is_ok() {
        return Err(DemoExportErrorV0::OutAlreadyExists);
    }
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    match std::fs::metadata(parent) {
        Ok(meta) if meta.is_dir() => Ok(()),
        _ => Err(DemoExportErrorV0::InvalidPath),
    }
}

/// Outcome of one exclusive write.
enum WriteOutcomeV0 {
    Written,
    /// The open failed, so nothing was created.
    NotCreated,
    /// The file was created, then writing it failed.
    CreatedThenFailed,
}

fn write_new_file(path: &Path, bytes: &[u8]) -> WriteOutcomeV0 {
    let Ok(mut file) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32)
        .open(path)
    else {
        return WriteOutcomeV0::NotCreated;
    };
    match file.write_all(bytes).and_then(|()| file.sync_all()) {
        Ok(()) => WriteOutcomeV0::Written,
        Err(_) => WriteOutcomeV0::CreatedThenFailed,
    }
}

/// Verify `source`, then write every exported asset into the new directory `out`.
pub fn export_assets(
    source: &dyn DemoFixtureSourceV0,
    out: &Path,
) -> Result<DemoExportReportV0, DemoExportErrorV0> {
    check_out_path(out)?;
    load_verified_manifest(source).map_err(|_| DemoExportErrorV0::AssetDigestMismatch)?;
    let mut assets: Vec<(&str, &[u8])> = Vec::with_capacity(EXPORTED_ASSETS.len());
    for name in EXPORTED_ASSETS {
        let bytes = if name == MANIFEST_NAME {
            source.manifest_bytes()
        } else {
            source
                .asset(name)
                .ok_or(DemoExportErrorV0::AssetDigestMismatch)?
        };
        assets.push((name, bytes));
    }

    match std::fs::DirBuilder::new().mode(0o700).create(out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(DemoExportErrorV0::OutAlreadyExists)
        }
        Err(_) => return Err(DemoExportErrorV0::WriteFailed),
    }
    let mut created: Vec<PathBuf> = Vec::with_capacity(assets.len());
    for (name, bytes) in &assets {
        let path = out.join(name);
        let outcome = write_new_file(&path, bytes);
        if !matches!(outcome, WriteOutcomeV0::NotCreated) {
            created.push(path);
        }
        if !matches!(outcome, WriteOutcomeV0::Written) {
            for p in &created {
                let _ = std::fs::remove_file(p);
            }
            // Succeeds only if the directory this invocation created is now empty.
            let _ = std::fs::remove_dir(out);
            return Err(DemoExportErrorV0::WriteFailed);
        }
    }

    Ok(DemoExportReportV0 {
        out_dir: out.to_path_buf(),
        manifest_sha256: sha256_hex(source.manifest_bytes()),
        files: assets
            .iter()
            .map(|(name, bytes)| DemoExportFileV0 {
                name: (*name).to_owned(),
                sha256: sha256_hex(bytes),
                len_bytes: bytes.len() as u64,
            })
            .collect(),
    })
}

#[derive(serde::Serialize)]
struct DemoExportJsonV0<'a> {
    schema: &'static str,
    out_dir: &'a str,
    manifest_sha256: &'a str,
    files: &'a [DemoExportFileV0],
}

#[derive(serde::Serialize)]
struct DemoExportErrorJsonV0 {
    schema: &'static str,
    command: &'static str,
    error_code: &'static str,
    exit_code: i32,
}

fn emit_json_line<T: serde::Serialize>(val: &T) {
    if let Ok(mut bytes) = serde_json::to_vec(val) {
        bytes.push(b'\n');
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(&bytes);
        let _ = stdout.flush();
    }
}

/// CLI entry for `parapet demo export` over the compiled-in fixture set.
pub fn export(out: &Path, json: bool) -> i32 {
    match export_assets(&CompiledDemoFixturesV0, out) {
        Ok(report) => {
            let out_dir = report.out_dir.to_string_lossy();
            if json {
                emit_json_line(&DemoExportJsonV0 {
                    schema: DEMO_EXPORT_SCHEMA,
                    out_dir: &out_dir,
                    manifest_sha256: &report.manifest_sha256,
                    files: &report.files,
                });
            } else {
                println!("out_dir: {out_dir}");
                println!("manifest_sha256: {}", report.manifest_sha256);
                for f in &report.files {
                    println!(
                        "file: {} sha256:{} len_bytes:{}",
                        f.name, f.sha256, f.len_bytes
                    );
                }
            }
            0
        }
        Err(e) => {
            eprintln!("parapet {COMMAND}: error: {}", e.message());
            if json {
                emit_json_line(&DemoExportErrorJsonV0 {
                    schema: "parapet-cli-error/v0",
                    command: COMMAND,
                    error_code: e.code(),
                    exit_code: e.exit_code(),
                });
            }
            e.exit_code()
        }
    }
}
