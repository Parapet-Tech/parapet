// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// L2a model infrastructure: trait for PG2 classification and feature-gated
// ONNX implementation.
//
// The `PromptGuardClassifier` trait, `ClassifyError`, and `ModelInitError`
// are always compiled so that tests and other code can reference them
// without the `l2a` feature.
// Only `OnnxPromptGuard` is behind `#[cfg(feature = "l2a")]`.

// ---------------------------------------------------------------------------
// Always-compiled: trait + error types
// ---------------------------------------------------------------------------

/// Error during classification (tokenization or inference).
#[derive(Debug)]
pub enum ClassifyError {
    /// Tokenizer failed to encode input text.
    Tokenize(String),
    /// ONNX inference or output extraction failed.
    Inference(String),
}

impl std::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassifyError::Tokenize(msg) => write!(f, "tokenize error: {msg}"),
            ClassifyError::Inference(msg) => write!(f, "inference error: {msg}"),
        }
    }
}

impl std::error::Error for ClassifyError {}

/// Classifies text segments for prompt injection probability.
///
/// Returns scores in [0.0, 1.0] where higher = more likely injection.
/// All methods are fallible — callers decide policy (fail-open / fail-closed)
/// based on L2a mode.
pub trait PromptGuardClassifier: Send + Sync {
    /// Classify a single text segment.
    fn classify(&self, text: &str) -> Result<f32, ClassifyError>;

    /// Classify a batch of text segments in one call.
    fn classify_batch(&self, texts: &[&str]) -> Result<Vec<f32>, ClassifyError>;
}

/// Error returned by model initialization.
///
/// Defined in `layers/`, not `engine/` — layers don't know about engine types.
/// The engine maps this to `StartupError` at the boundary (Chunk E).
#[derive(Debug)]
pub enum ModelInitError {
    /// Model files not found at the expected path.
    ModelNotFound(String),
    /// SHA256 checksum of a model file does not match the expected value.
    ChecksumMismatch { expected: String, actual: String },
    /// ONNX runtime error during session creation or warmup.
    OnnxError(String),
}

impl std::fmt::Display for ModelInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelInitError::ModelNotFound(msg) => write!(f, "{msg}"),
            ModelInitError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "model checksum mismatch: expected {expected}, got {actual}"
                )
            }
            ModelInitError::OnnxError(msg) => write!(f, "ONNX error: {msg}"),
        }
    }
}

impl std::error::Error for ModelInitError {}

/// Known model names accepted by the L2a layer.
/// Used by config validation and ONNX initialization.
pub const KNOWN_MODELS: &[&str] = &["pg2-86m", "pg2-22m"];

// ---------------------------------------------------------------------------
// Feature-gated: OnnxPromptGuard
// ---------------------------------------------------------------------------

#[cfg(feature = "l2a")]
mod onnx_impl {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use ndarray::Array2;
    use ort::session::Session;
    use sha2::{Digest, Sha256};
    use tokenizers::Tokenizer;

    use crate::config::L2aConfig;
    use crate::model_fetch;
    use crate::model_path;

    use super::{ClassifyError, ModelInitError, PromptGuardClassifier};

    /// ONNX Runtime-backed Prompt Guard 2 classifier.
    ///
    /// Initialized once at startup via `init()`. Holds the ONNX session
    /// and tokenizer for the lifetime of the process. The session is
    /// wrapped in a Mutex because ort v2 `session.run()` requires `&mut self`.
    pub struct OnnxPromptGuard {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
    }

    impl OnnxPromptGuard {
        /// Borrow the tokenizer (diagnostic use).
        pub fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        /// Lock the session (diagnostic use).
        pub fn session_lock(&self) -> std::sync::MutexGuard<'_, Session> {
            self.session.lock().expect("session lock")
        }

        /// Initialize the ONNX model and tokenizer.
        ///
        /// Performs:
        /// 1. Model path resolution (config > env > default)
        /// 2. File existence check with actionable error
        /// 3. SHA256 verification against hardcoded manifest
        /// 4. ONNX session creation
        /// 5. Tokenizer loading
        /// 6. Warmup inference to pre-allocate buffers
        pub fn init(config: &L2aConfig) -> Result<Self, ModelInitError> {
            // Resolve model directory.
            let model_dir = model_path::model_dir_for(
                config.model_dir.as_deref(),
                &config.model,
            )
            .ok_or_else(|| {
                ModelInitError::ModelNotFound(
                    "could not determine model directory — \
                     set model_dir in config, $PARAPET_MODEL_DIR, or \
                     run `parapet-fetch`"
                        .to_string(),
                )
            })?;

            let onnx_path = model_dir.join("model.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");

            // Check file existence with actionable message.
            if !onnx_path.exists() {
                return Err(ModelInitError::ModelNotFound(format!(
                    "L2a configured but model not found at {} — \
                     run `parapet-fetch --model {}`",
                    onnx_path.display(),
                    config.model,
                )));
            }
            if !tokenizer_path.exists() {
                return Err(ModelInitError::ModelNotFound(format!(
                    "L2a configured but tokenizer not found at {} — \
                     run `parapet-fetch --model {}`",
                    tokenizer_path.display(),
                    config.model,
                )));
            }

            // SHA256 verification against hardcoded manifest.
            verify_checksum(&onnx_path, &config.model, "model.onnx")?;
            verify_checksum(&tokenizer_path, &config.model, "tokenizer.json")?;

            // Create ONNX session.
            let session = Session::builder()
                .and_then(|b| b.commit_from_file(&onnx_path))
                .map_err(|e| ModelInitError::OnnxError(e.to_string()))?;

            // Load tokenizer.
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| ModelInitError::OnnxError(format!("tokenizer: {e}")))?;

            let pg = Self { session: Mutex::new(session), tokenizer };

            // Warmup inference to pre-allocate buffers.
            // Errors here surface as ModelInitError, not a panic.
            pg.classify("warmup").map_err(|e| {
                ModelInitError::OnnxError(format!("warmup inference failed: {e}"))
            })?;

            Ok(pg)
        }
    }

    /// Verify SHA256 of a file against the hardcoded manifest.
    ///
    /// Fail-closed: unknown model name → error (defense in depth;
    /// config validation in loader.rs should have caught this earlier).
    ///
    /// Empty checksum → warn and skip. All checksums are currently empty
    /// until PG2 hosting is finalized — same known gap as `parapet-fetch`.
    fn verify_checksum(
        path: &PathBuf,
        model_name: &str,
        filename: &str,
    ) -> Result<(), ModelInitError> {
        // Look up expected checksum from manifest.
        let manifest = model_fetch::find_model(model_name).map_err(|_| {
            ModelInitError::ModelNotFound(format!(
                "model \"{model_name}\" not in hardcoded manifest — \
                 cannot verify integrity"
            ))
        })?;

        let expected = manifest
            .files
            .iter()
            .find(|f| f.filename == filename)
            .map(|f| f.sha256);

        match expected {
            Some(hash) if !hash.is_empty() => {
                let data = std::fs::read(path).map_err(|e| {
                    ModelInitError::OnnxError(format!(
                        "failed to read {} for checksum: {e}",
                        path.display()
                    ))
                })?;
                let mut hasher = Sha256::new();
                hasher.update(&data);
                let actual = format!("{:x}", hasher.finalize());
                if actual != hash {
                    return Err(ModelInitError::ChecksumMismatch {
                        expected: hash.to_string(),
                        actual,
                    });
                }
                Ok(())
            }
            _ => {
                // No checksum populated yet. Warn so operators know
                // integrity is not enforced. Will become an error once
                // checksums are finalized in model_fetch::MODELS.
                tracing::warn!(
                    model = model_name,
                    file = filename,
                    "no checksum available — integrity verification skipped"
                );
                Ok(())
            }
        }
    }

    impl PromptGuardClassifier for OnnxPromptGuard {
        fn classify(&self, text: &str) -> Result<f32, ClassifyError> {
            self.classify_batch(&[text])
                .map(|v| v.into_iter().next().unwrap_or(0.0))
        }

        fn classify_batch(&self, texts: &[&str]) -> Result<Vec<f32>, ClassifyError> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            // Tokenize all texts.
            let encodings: Result<Vec<_>, _> = texts
                .iter()
                .map(|t| {
                    self.tokenizer
                        .encode(*t, true)
                        .map_err(|e| ClassifyError::Tokenize(e.to_string()))
                })
                .collect();
            let encodings = encodings?;

            // Find max length for padding.
            // Safe: encodings is non-empty (texts is non-empty).
            let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();
            let batch_size = texts.len();

            // Build padded input_ids and attention_mask arrays.
            let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
            let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));

            for (i, encoding) in encodings.iter().enumerate() {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                for (j, &id) in ids.iter().enumerate() {
                    input_ids[[i, j]] = id as i64;
                }
                for (j, &m) in mask.iter().enumerate() {
                    attention_mask[[i, j]] = m as i64;
                }
            }

            // Run ONNX inference.
            let input_ids_tensor = ort::value::Tensor::from_array(input_ids)
                .map_err(|e| ClassifyError::Inference(format!("input_ids tensor: {e}")))?;
            let attention_mask_tensor = ort::value::Tensor::from_array(attention_mask)
                .map_err(|e| ClassifyError::Inference(format!("attention_mask tensor: {e}")))?;
            let mut session = self.session.lock()
                .map_err(|e| ClassifyError::Inference(format!("session lock poisoned: {e}")))?;
            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
                .map_err(|e| ClassifyError::Inference(format!("session.run: {e}")))?;

            // Extract logits and apply softmax.
            // PG2 output shape: [batch_size, 3] — classes: benign, injection, jailbreak.
            // We return P(injection) + P(jailbreak) as the combined score.
            let (shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| ClassifyError::Inference(format!("extract logits: {e}")))?;

            let num_classes = if shape.len() >= 2 { shape[1] as usize } else { 0 };
            Ok(logits_data
                .chunks(num_classes)
                .map(|row| {
                    let probs = softmax(row);
                    // PG2 full model: 3 classes — benign, injection, jailbreak
                    // PG2 quantized:  2 classes — benign, malicious
                    match probs.len() {
                        n if n >= 3 => probs[1] + probs[2],
                        2 => probs[1],
                        _ => 0.0,
                    }
                })
                .collect())
        }
    }

    /// Softmax over a slice of logits.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }
}

#[cfg(feature = "l2a")]
pub use onnx_impl::OnnxPromptGuard;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock classifier for testing pipeline logic without ONNX.
    struct MockClassifier {
        score: f32,
    }

    impl MockClassifier {
        fn new(score: f32) -> Self {
            Self { score }
        }
    }

    impl PromptGuardClassifier for MockClassifier {
        fn classify(&self, _text: &str) -> Result<f32, ClassifyError> {
            Ok(self.score)
        }

        fn classify_batch(&self, texts: &[&str]) -> Result<Vec<f32>, ClassifyError> {
            Ok(vec![self.score; texts.len()])
        }
    }

    /// Mock classifier that always fails.
    struct FailingClassifier;

    impl PromptGuardClassifier for FailingClassifier {
        fn classify(&self, _text: &str) -> Result<f32, ClassifyError> {
            Err(ClassifyError::Inference("mock failure".to_string()))
        }

        fn classify_batch(&self, _texts: &[&str]) -> Result<Vec<f32>, ClassifyError> {
            Err(ClassifyError::Inference("mock batch failure".to_string()))
        }
    }

    #[test]
    fn mock_classify_returns_configured_score() {
        let c = MockClassifier::new(0.85);
        assert!((c.classify("test").unwrap() - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn mock_classify_batch_returns_correct_count() {
        let c = MockClassifier::new(0.5);
        let results = c.classify_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!((*r - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn mock_classify_batch_empty_input() {
        let c = MockClassifier::new(0.5);
        let results = c.classify_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn trait_object_works() {
        let c: Box<dyn PromptGuardClassifier> = Box::new(MockClassifier::new(0.9));
        assert!((c.classify("test").unwrap() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn failing_classifier_returns_error() {
        let c = FailingClassifier;
        let err = c.classify("test").unwrap_err();
        assert!(err.to_string().contains("mock failure"));
    }

    #[test]
    fn failing_classifier_batch_returns_error() {
        let c = FailingClassifier;
        let err = c.classify_batch(&["a", "b"]).unwrap_err();
        assert!(err.to_string().contains("mock batch failure"));
    }

    #[test]
    fn classify_error_display_tokenize() {
        let e = ClassifyError::Tokenize("bad input".to_string());
        assert!(e.to_string().contains("tokenize"));
        assert!(e.to_string().contains("bad input"));
    }

    #[test]
    fn classify_error_display_inference() {
        let e = ClassifyError::Inference("ort crashed".to_string());
        assert!(e.to_string().contains("inference"));
        assert!(e.to_string().contains("ort crashed"));
    }

    #[test]
    fn classify_error_is_error_trait() {
        let e: Box<dyn std::error::Error> =
            Box::new(ClassifyError::Inference("test".to_string()));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn model_init_error_display_not_found() {
        let e = ModelInitError::ModelNotFound("model missing".to_string());
        assert_eq!(e.to_string(), "model missing");
    }

    #[test]
    fn model_init_error_display_checksum() {
        let e = ModelInitError::ChecksumMismatch {
            expected: "aaa".to_string(),
            actual: "bbb".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("aaa"));
        assert!(msg.contains("bbb"));
    }

    #[test]
    fn model_init_error_display_onnx() {
        let e = ModelInitError::OnnxError("session failed".to_string());
        assert!(e.to_string().contains("session failed"));
    }

    #[test]
    fn model_init_error_is_error_trait() {
        let e: Box<dyn std::error::Error> =
            Box::new(ModelInitError::OnnxError("test".to_string()));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn known_models_contains_expected_values() {
        assert!(KNOWN_MODELS.contains(&"pg2-86m"));
        assert!(KNOWN_MODELS.contains(&"pg2-22m"));
        assert!(!KNOWN_MODELS.contains(&"pg2-999b"));
    }
}
