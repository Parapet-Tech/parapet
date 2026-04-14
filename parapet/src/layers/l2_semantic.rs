// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

// L2 semantic classifier: MiniLM embedding + logistic regression head.
//
// Replaces PromptGuard (PG2) in the L2a slot. Uses the same ONNX
// infrastructure (ort, tokenizers, ndarray) behind the `l2a` feature flag.
//
// Inference pipeline:
//   1. Head+tail tokenization (128 + SEP + 127 = 256 tokens max)
//   2. ONNX forward pass → token embeddings [seq_len, 384]
//   3. Attention-mask-weighted mean pooling → [384]
//   4. L2 normalization → [384]
//   5. Standardize + clamp 5 harness signals → [5]
//   6. Concatenate → [389]
//   7. Dot product with LR weights + bias
//   8. Sigmoid → score in [0.0, 1.0]

use crate::layers::l2a_model::{ClassifyError, L2Context, SemanticClassifier};
use crate::layers::l2_weights::{
    EMBEDDING_DIM, HARNESS_DIM, L2_BIAS, L2_HARNESS_MEAN, L2_HARNESS_SCALE, L2_WEIGHTS,
    TOTAL_DIM,
};

/// Maximum token count for MiniLM input.
const MAX_TOKENS: usize = 256;
/// Head portion of head+tail truncation.
const HEAD_TOKENS: usize = 128;
/// Tail portion (MAX_TOKENS - HEAD_TOKENS - 1 for separator).
const TAIL_TOKENS: usize = MAX_TOKENS - HEAD_TOKENS - 1; // 127

/// Clamp bound for standardized harness signals.
const HARNESS_CLAMP: f32 = 10.0;

// ---------------------------------------------------------------------------
// Feature-gated implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "l2a")]
mod onnx_impl {
    use std::sync::Mutex;

    use ndarray::{Array1, Array2, Axis};
    use ort::session::Session;
    use tokenizers::Tokenizer;

    use crate::config::L2aConfig;
    use crate::layers::l2a_model::{ClassifyError, L2Context, ModelInitError, SemanticClassifier};
    use crate::model_path;

    use super::*;

    /// ONNX-backed MiniLM semantic classifier.
    pub struct OnnxMiniLmClassifier {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
    }

    impl OnnxMiniLmClassifier {
        /// Initialize the ONNX model and tokenizer.
        pub fn init(config: &L2aConfig) -> Result<Self, ModelInitError> {
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

            // Try int8 first, fall back to fp32.
            let onnx_path = {
                let int8 = model_dir.join("model_int8.onnx");
                let fp32 = model_dir.join("model_fp32.onnx");
                let legacy = model_dir.join("model.onnx");
                if int8.exists() {
                    int8
                } else if fp32.exists() {
                    fp32
                } else if legacy.exists() {
                    legacy
                } else {
                    return Err(ModelInitError::ModelNotFound(format!(
                        "L2 configured for {} but no ONNX model found in {} — \
                         expected model_int8.onnx, model_fp32.onnx, or model.onnx",
                        config.model,
                        model_dir.display(),
                    )));
                }
            };

            let tokenizer_path = model_dir.join("tokenizer.json");
            if !tokenizer_path.exists() {
                return Err(ModelInitError::ModelNotFound(format!(
                    "L2 configured for {} but tokenizer not found at {}",
                    config.model,
                    tokenizer_path.display(),
                )));
            }

            let session = Session::builder()
                .and_then(|b| b.commit_from_file(&onnx_path))
                .map_err(|e| ModelInitError::OnnxError(e.to_string()))?;

            // Load tokenizer. Disable padding (we handle it ourselves).
            // Do NOT set truncation here — we implement head+tail manually.
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| ModelInitError::OnnxError(format!("tokenizer: {e}")))?;
            tokenizer.with_padding(None);

            let cls = Self {
                session: Mutex::new(session),
                tokenizer,
            };

            // Warmup inference.
            let warmup_ctx = L2Context {
                raw_score: 0.0,
                raw_unquoted_score: 0.0,
                raw_squash_score: 0.0,
                raw_score_delta: 0.0,
                quote_detected: false,
            };
            cls.classify("warmup", &warmup_ctx).map_err(|e| {
                ModelInitError::OnnxError(format!("warmup inference failed: {e}"))
            })?;

            Ok(cls)
        }

        /// Tokenize with head+tail truncation.
        ///
        /// If the input exceeds MAX_TOKENS, take the first HEAD_TOKENS and
        /// last TAIL_TOKENS with a separator in between. This preserves
        /// framing context at the start and payloads at the end.
        fn tokenize_head_tail(
            &self,
            text: &str,
        ) -> Result<(Vec<i64>, Vec<i64>), ClassifyError> {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| ClassifyError::Tokenize(e.to_string()))?;

            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            if ids.len() <= MAX_TOKENS {
                let input_ids: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
                let attention_mask: Vec<i64> = mask.iter().map(|&m| m as i64).collect();
                return Ok((input_ids, attention_mask));
            }

            // Head + SEP + tail truncation.
            // The tokenizer adds [CLS] at start and [SEP] at end by default.
            // We take first HEAD_TOKENS tokens (including [CLS]), add a [SEP],
            // then take the last TAIL_TOKENS tokens.
            let sep_id = self
                .tokenizer
                .token_to_id("[SEP]")
                .unwrap_or(102) as i64; // 102 is standard BERT SEP

            let mut input_ids = Vec::with_capacity(MAX_TOKENS);
            let mut attention_mask = Vec::with_capacity(MAX_TOKENS);

            // Head
            for &id in &ids[..HEAD_TOKENS] {
                input_ids.push(id as i64);
                attention_mask.push(1);
            }

            // Separator
            input_ids.push(sep_id);
            attention_mask.push(1);

            // Tail
            let tail_start = ids.len().saturating_sub(TAIL_TOKENS);
            for &id in &ids[tail_start..] {
                input_ids.push(id as i64);
                attention_mask.push(1);
            }

            Ok((input_ids, attention_mask))
        }

        /// Run ONNX inference and return the pooled, L2-normalized embedding.
        fn embed(&self, text: &str) -> Result<Array1<f32>, ClassifyError> {
            let (input_ids_vec, attention_mask_vec) = self.tokenize_head_tail(text)?;
            let seq_len = input_ids_vec.len();

            // Build [1, seq_len] tensors.
            let input_ids = Array2::from_shape_vec((1, seq_len), input_ids_vec)
                .map_err(|e| ClassifyError::Inference(format!("input_ids shape: {e}")))?;
            let attention_mask =
                Array2::from_shape_vec((1, seq_len), attention_mask_vec.clone())
                    .map_err(|e| {
                        ClassifyError::Inference(format!("attention_mask shape: {e}"))
                    })?;

            let input_ids_tensor = ort::value::Tensor::from_array(input_ids)
                .map_err(|e| ClassifyError::Inference(format!("input_ids tensor: {e}")))?;
            let attention_mask_tensor =
                ort::value::Tensor::from_array(attention_mask)
                    .map_err(|e| {
                        ClassifyError::Inference(format!("attention_mask tensor: {e}"))
                    })?;

            let mut session = self
                .session
                .lock()
                .map_err(|e| ClassifyError::Inference(format!("session lock: {e}")))?;

            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
                .map_err(|e| ClassifyError::Inference(format!("session.run: {e}")))?;

            // Output shape: [1, seq_len, 384] (token embeddings).
            let (shape, embeddings_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    ClassifyError::Inference(format!("extract embeddings: {e}"))
                })?;

            if shape.len() < 3 || (shape[2] as usize) != EMBEDDING_DIM {
                return Err(ClassifyError::Inference(format!(
                    "unexpected embedding shape: {:?}, expected [1, {}, {}]",
                    shape, seq_len, EMBEDDING_DIM,
                )));
            }

            // Mean pooling with attention mask.
            let token_embs = Array2::from_shape_vec(
                (seq_len, EMBEDDING_DIM),
                embeddings_data.to_vec(),
            )
            .map_err(|e| {
                ClassifyError::Inference(format!("reshape embeddings: {e}"))
            })?;

            let mask_f32: Vec<f32> =
                attention_mask_vec.iter().map(|&m| m as f32).collect();
            let mask_arr = Array1::from_vec(mask_f32);

            // Weighted sum: sum(token_embs * mask, axis=0) / sum(mask)
            let mut pooled = Array1::<f32>::zeros(EMBEDDING_DIM);
            for (i, row) in token_embs.axis_iter(Axis(0)).enumerate() {
                let w = mask_arr[i];
                pooled = pooled + &row.mapv(|v| v * w);
            }
            let count = mask_arr.sum().max(1.0);
            pooled.mapv_inplace(|v| v / count);

            // L2 normalization.
            let norm = pooled.dot(&pooled).sqrt().max(1e-12);
            pooled.mapv_inplace(|v| v / norm);

            Ok(pooled)
        }
    }

    impl SemanticClassifier for OnnxMiniLmClassifier {
        fn classify(
            &self,
            text: &str,
            ctx: &L2Context,
        ) -> Result<f32, ClassifyError> {
            // Step 1-4: embed.
            let embedding = self.embed(text)?;

            // Step 5: standardize + clamp harness signals.
            let harness_raw = [
                ctx.raw_score as f32,
                ctx.raw_unquoted_score as f32,
                ctx.raw_squash_score as f32,
                ctx.raw_score_delta as f32,
                if ctx.quote_detected { 1.0 } else { 0.0 },
            ];

            let mut harness_scaled = [0.0f32; HARNESS_DIM];
            for i in 0..HARNESS_DIM {
                let scaled =
                    (harness_raw[i] - L2_HARNESS_MEAN[i]) / L2_HARNESS_SCALE[i];
                harness_scaled[i] = scaled.clamp(-HARNESS_CLAMP, HARNESS_CLAMP);
            }

            // Step 6: concatenate embedding + scaled harness → [389].
            // Step 7: dot product with LR weights + bias.
            let mut logit = L2_BIAS;
            for i in 0..EMBEDDING_DIM {
                logit += embedding[i] * L2_WEIGHTS[i];
            }
            for i in 0..HARNESS_DIM {
                logit += harness_scaled[i] * L2_WEIGHTS[EMBEDDING_DIM + i];
            }

            // Step 8: sigmoid.
            let score = 1.0 / (1.0 + (-logit).exp());

            Ok(score)
        }
    }
}

#[cfg(feature = "l2a")]
pub use onnx_impl::OnnxMiniLmClassifier;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::l2a_model::L2Context;

    fn test_ctx() -> L2Context {
        L2Context {
            raw_score: 1.5,
            raw_unquoted_score: 0.3,
            raw_squash_score: 2.0,
            raw_score_delta: 1.2,
            quote_detected: true,
        }
    }

    #[test]
    fn harness_scaling_and_clamping() {
        // With placeholder weights (all zeros), score should be sigmoid(0) = 0.5.
        // This tests the math path without ONNX.
        let harness_raw = [1.5f32, 0.3, 2.0, 1.2, 1.0];
        let mut scaled = [0.0f32; HARNESS_DIM];
        for i in 0..HARNESS_DIM {
            let s = (harness_raw[i] - L2_HARNESS_MEAN[i]) / L2_HARNESS_SCALE[i];
            scaled[i] = s.clamp(-HARNESS_CLAMP, HARNESS_CLAMP);
        }
        // With default mean=0, scale=1: scaled == raw (all within clamp range).
        assert!((scaled[0] - 1.5).abs() < 1e-6);
        assert!((scaled[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn harness_clamping_extreme_value() {
        let extreme = 50.0f32;
        let scaled = (extreme - L2_HARNESS_MEAN[0]) / L2_HARNESS_SCALE[0];
        let clamped = scaled.clamp(-HARNESS_CLAMP, HARNESS_CLAMP);
        assert!((clamped - HARNESS_CLAMP).abs() < 1e-6);
    }

    #[test]
    fn lr_scoring_with_zero_weights() {
        // All weights are zero → logit = bias = 0 → sigmoid = 0.5.
        let logit = L2_BIAS;
        let score = 1.0 / (1.0 + (-logit).exp());
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn constants_have_correct_dimensions() {
        assert_eq!(L2_WEIGHTS.len(), TOTAL_DIM);
        assert_eq!(L2_HARNESS_MEAN.len(), HARNESS_DIM);
        assert_eq!(L2_HARNESS_SCALE.len(), HARNESS_DIM);
        assert_eq!(TOTAL_DIM, EMBEDDING_DIM + HARNESS_DIM);
        assert_eq!(TOTAL_DIM, 389);
    }

    #[test]
    fn token_window_constants() {
        assert_eq!(MAX_TOKENS, 256);
        assert_eq!(HEAD_TOKENS, 128);
        assert_eq!(TAIL_TOKENS, 127);
        assert_eq!(HEAD_TOKENS + 1 + TAIL_TOKENS, MAX_TOKENS);
    }
}
