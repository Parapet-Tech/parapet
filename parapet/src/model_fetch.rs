// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// Model fetching: download, verify, and install PG2 model files.
//
// Used by the `parapet-fetch` CLI binary.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Known model definitions with download URLs and expected checksums.
#[derive(Debug)]
pub struct ModelManifest {
    pub name: &'static str,
    pub files: &'static [ModelFile],
}

#[derive(Debug)]
pub struct ModelFile {
    pub filename: &'static str,
    pub url: &'static str,
    /// Expected SHA256 hex digest. Empty string means checksum not yet known.
    pub sha256: &'static str,
}

impl ModelFile {
    /// Whether a real checksum is available for verification.
    pub fn has_checksum(&self) -> bool {
        !self.sha256.is_empty()
    }
}

/// Hardcoded model manifests.
///
/// Checksums are empty until PG2 model hosting is finalized.
/// `parapet-fetch` will refuse to download without `--skip-checksum`
/// until real checksums are populated.
pub static MODELS: &[ModelManifest] = &[
    ModelManifest {
        name: "pg2-86m",
        files: &[
            ModelFile {
                filename: "model.onnx",
                url: "https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-86M-onnx/resolve/main/model.onnx",
                sha256: "",
            },
            ModelFile {
                filename: "tokenizer.json",
                url: "https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-86M-onnx/resolve/main/tokenizer.json",
                sha256: "",
            },
        ],
    },
    ModelManifest {
        name: "pg2-22m",
        files: &[
            ModelFile {
                filename: "model.onnx",
                url: "https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/resolve/main/model.quant.onnx",
                sha256: "",
            },
            ModelFile {
                filename: "tokenizer.json",
                url: "https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/resolve/main/tokenizer.json",
                sha256: "",
            },
        ],
    },
];

/// Errors from model fetching.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("unknown model \"{0}\". known models: pg2-86m, pg2-22m")]
    UnknownModel(String),

    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("download failed for {url}: {source}")]
    Download {
        url: String,
        source: reqwest::Error,
    },

    #[error("HTTP {status} for {url}")]
    HttpStatus { url: String, status: u16 },

    #[error("checksum mismatch for {filename}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        filename: String,
        expected: String,
        actual: String,
    },

    #[error(
        "no checksum available for {filename} — \
         pass --skip-checksum to download without verification (not recommended)"
    )]
    NoChecksum { filename: String },

    #[error("failed to write {path}: {source}")]
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("could not determine model directory — set $PARAPET_MODEL_DIR or --model-dir")]
    NoModelDir,
}

/// Look up a model manifest by name.
pub fn find_model(name: &str) -> Result<&'static ModelManifest, FetchError> {
    MODELS
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| FetchError::UnknownModel(name.to_string()))
}

/// Download and verify a single model file.
///
/// When `skip_checksum` is false:
/// - If a checksum is available, verify it. Mismatch = error.
/// - If no checksum is available (empty string), return `NoChecksum` error.
///
/// When `skip_checksum` is true, no verification is performed.
pub async fn fetch_file(
    file: &ModelFile,
    dest_dir: &Path,
    skip_checksum: bool,
) -> Result<PathBuf, FetchError> {
    let dest_path = dest_dir.join(file.filename);

    // Fail early if we can't verify and the user didn't opt out.
    if !skip_checksum && !file.has_checksum() {
        return Err(FetchError::NoChecksum {
            filename: file.filename.to_string(),
        });
    }

    let client = reqwest::Client::new();
    let mut request = client.get(file.url);
    if let Some(token) = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGINGFACE_API_KEY"))
        .ok()
    {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let response = request
        .send()
        .await
        .map_err(|e| FetchError::Download {
            url: file.url.to_string(),
            source: e,
        })?;

    // Validate HTTP status before reading body.
    let status = response.status();
    if !status.is_success() {
        return Err(FetchError::HttpStatus {
            url: file.url.to_string(),
            status: status.as_u16(),
        });
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| FetchError::Download {
            url: file.url.to_string(),
            source: e,
        })?;

    // Verify checksum before writing.
    if !skip_checksum {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = format!("{:x}", hasher.finalize());
        if actual != file.sha256 {
            return Err(FetchError::ChecksumMismatch {
                filename: file.filename.to_string(),
                expected: file.sha256.to_string(),
                actual,
            });
        }
    }

    // Write file.
    let mut f = fs::File::create(&dest_path).map_err(|e| FetchError::WriteFile {
        path: dest_path.clone(),
        source: e,
    })?;
    f.write_all(&bytes).map_err(|e| FetchError::WriteFile {
        path: dest_path.clone(),
        source: e,
    })?;

    Ok(dest_path)
}

/// Download all files for a model and write a manifest.json alongside them.
pub async fn fetch_model(
    manifest: &ModelManifest,
    dest_dir: &Path,
    skip_checksum: bool,
) -> Result<(), FetchError> {
    let model_dir = dest_dir.join(manifest.name);
    fs::create_dir_all(&model_dir).map_err(|e| FetchError::CreateDir {
        path: model_dir.clone(),
        source: e,
    })?;

    for file in manifest.files {
        fetch_file(file, &model_dir, skip_checksum).await?;
    }

    // Write manifest.json
    let manifest_json = serde_json::json!({
        "model": manifest.name,
        "files": manifest.files.iter().map(|f| serde_json::json!({
            "filename": f.filename,
            "sha256": f.sha256,
        })).collect::<Vec<_>>(),
        "fetched_at": chrono::Utc::now().to_rfc3339(),
    });
    let manifest_path = model_dir.join("manifest.json");
    let mut f = fs::File::create(&manifest_path).map_err(|e| FetchError::WriteFile {
        path: manifest_path.clone(),
        source: e,
    })?;
    f.write_all(
        serde_json::to_string_pretty(&manifest_json)
            .unwrap()
            .as_bytes(),
    )
    .map_err(|e| FetchError::WriteFile {
        path: manifest_path,
        source: e,
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn find_model_known() {
        let m = find_model("pg2-86m").unwrap();
        assert_eq!(m.name, "pg2-86m");
        assert_eq!(m.files.len(), 2);
    }

    #[test]
    fn find_model_unknown() {
        let err = find_model("pg2-999b").unwrap_err();
        assert!(err.to_string().contains("pg2-999b"));
        assert!(err.to_string().contains("pg2-86m"));
    }

    #[test]
    fn find_model_unknown_lists_known() {
        let err = find_model("nonexistent").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pg2-86m"));
        assert!(msg.contains("pg2-22m"));
    }

    #[test]
    fn pg2_22m_urls_point_to_22m_model() {
        let m = find_model("pg2-22m").unwrap();
        for file in m.files {
            assert!(
                file.url.contains("22M"),
                "pg2-22m file {} URL should reference 22M, got: {}",
                file.filename,
                file.url
            );
        }
    }

    #[test]
    fn pg2_86m_urls_point_to_86m_model() {
        let m = find_model("pg2-86m").unwrap();
        for file in m.files {
            assert!(
                file.url.contains("86M"),
                "pg2-86m file {} URL should reference 86M, got: {}",
                file.filename,
                file.url
            );
        }
    }

    #[test]
    fn has_checksum_false_for_empty() {
        let f = ModelFile {
            filename: "test",
            url: "http://x",
            sha256: "",
        };
        assert!(!f.has_checksum());
    }

    #[test]
    fn has_checksum_true_for_real_hash() {
        let f = ModelFile {
            filename: "test",
            url: "http://x",
            sha256: "abc123",
        };
        assert!(f.has_checksum());
    }

    // -----------------------------------------------------------------
    // fetch_file tests (wiremock)
    // -----------------------------------------------------------------

    fn test_file_with_checksum(url: &str, sha256: &str) -> ModelFile {
        // Leak the string to get 'static lifetime for test purposes.
        let url: &'static str = Box::leak(url.to_string().into_boxed_str());
        let sha256: &'static str = Box::leak(sha256.to_string().into_boxed_str());
        ModelFile {
            filename: "test.bin",
            url,
            sha256,
        }
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    #[tokio::test]
    async fn fetch_file_success_with_valid_checksum() {
        let server = MockServer::start().await;
        let body = b"model data here";
        let checksum = sha256_hex(body);

        Mock::given(method("GET"))
            .and(path("/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_slice()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let file = test_file_with_checksum(
            &format!("{}/model.bin", server.uri()),
            &checksum,
        );

        let result = fetch_file(&file, dir.path(), false).await;
        assert!(result.is_ok());
        let written = std::fs::read(dir.path().join("test.bin")).unwrap();
        assert_eq!(written, body);
    }

    #[tokio::test]
    async fn fetch_file_checksum_mismatch() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"actual data"))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let file = test_file_with_checksum(
            &format!("{}/model.bin", server.uri()),
            "0000000000000000000000000000000000000000000000000000000000000000",
        );

        let err = fetch_file(&file, dir.path(), false).await.unwrap_err();
        assert!(matches!(err, FetchError::ChecksumMismatch { .. }));
    }

    #[tokio::test]
    async fn fetch_file_skip_checksum_bypasses_verification() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"data"))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let file = test_file_with_checksum(
            &format!("{}/model.bin", server.uri()),
            "wrong_checksum",
        );

        // With skip_checksum=true, wrong checksum should not cause error.
        let result = fetch_file(&file, dir.path(), true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn fetch_file_no_checksum_without_skip_returns_error() {
        let file = ModelFile {
            filename: "test.bin",
            url: "http://localhost:1/unused",
            sha256: "",
        };
        let dir = tempfile::tempdir().unwrap();

        // Should fail before even making an HTTP request.
        let err = fetch_file(&file, dir.path(), false).await.unwrap_err();
        assert!(matches!(err, FetchError::NoChecksum { .. }));
        assert!(err.to_string().contains("--skip-checksum"));
    }

    #[tokio::test]
    async fn fetch_file_no_checksum_with_skip_proceeds() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"data"))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        // Leak for 'static
        let url: &'static str =
            Box::leak(format!("{}/model.bin", server.uri()).into_boxed_str());
        let file = ModelFile {
            filename: "test.bin",
            url,
            sha256: "",
        };

        let result = fetch_file(&file, dir.path(), true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn fetch_file_http_404_returns_error() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/missing.bin"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let file = test_file_with_checksum(
            &format!("{}/missing.bin", server.uri()),
            "abc",
        );

        let err = fetch_file(&file, dir.path(), true).await.unwrap_err();
        assert!(matches!(err, FetchError::HttpStatus { status: 404, .. }));
    }

    #[tokio::test]
    async fn fetch_file_http_500_returns_error() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/error.bin"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let file = test_file_with_checksum(
            &format!("{}/error.bin", server.uri()),
            "abc",
        );

        let err = fetch_file(&file, dir.path(), true).await.unwrap_err();
        assert!(matches!(err, FetchError::HttpStatus { status: 500, .. }));
    }

    // -----------------------------------------------------------------
    // fetch_model tests
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn fetch_model_writes_all_files_and_manifest() {
        let server = MockServer::start().await;
        let body_a = b"file_a_content";
        let body_b = b"file_b_content";
        let hash_a = sha256_hex(body_a);
        let hash_b = sha256_hex(body_b);

        Mock::given(method("GET"))
            .and(path("/a.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body_a.as_slice()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/b.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body_b.as_slice()))
            .mount(&server)
            .await;

        let url_a: &'static str =
            Box::leak(format!("{}/a.bin", server.uri()).into_boxed_str());
        let url_b: &'static str =
            Box::leak(format!("{}/b.bin", server.uri()).into_boxed_str());
        let hash_a: &'static str = Box::leak(hash_a.into_boxed_str());
        let hash_b: &'static str = Box::leak(hash_b.into_boxed_str());

        let files: &'static [ModelFile] = Box::leak(Box::new([
            ModelFile {
                filename: "a.bin",
                url: url_a,
                sha256: hash_a,
            },
            ModelFile {
                filename: "b.bin",
                url: url_b,
                sha256: hash_b,
            },
        ]));

        let manifest = ModelManifest {
            name: "test-model",
            files,
        };

        let dir = tempfile::tempdir().unwrap();
        fetch_model(&manifest, dir.path(), false).await.unwrap();

        // Check files written.
        let model_dir = dir.path().join("test-model");
        assert_eq!(std::fs::read(model_dir.join("a.bin")).unwrap(), body_a);
        assert_eq!(std::fs::read(model_dir.join("b.bin")).unwrap(), body_b);

        // Check manifest.json written.
        let manifest_content = std::fs::read_to_string(model_dir.join("manifest.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
        assert_eq!(parsed["model"], "test-model");
        assert!(parsed["fetched_at"].is_string());
        assert_eq!(parsed["files"].as_array().unwrap().len(), 2);
    }
}
