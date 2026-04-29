# %% [markdown]
# # L2 latency bench — Kaggle / Colab entry-point
#
# Runs `parapet_runner.latency_bench` against a candidate ONNX classifier
# under single-thread, no-batching conditions and writes a manifest-bearing
# JSON result.
#
# **Convert this file to a notebook** with `jupytext --to ipynb` or paste
# each `# %%` cell into a Kaggle/Colab notebook. The cells are intentionally
# small and reference real CLI tools — bench logic lives in repo code, not
# in this notebook.
#
# **Inputs** (mounted by Kaggle as `/kaggle/input/<dataset>/...`):
#   * Curated train/val text corpus (NOT eval/holdout — see direction.md
#     Phase 3 split discipline).
# **Outputs** land under `/kaggle/working/...`.
#
# This notebook does the ONNX export and quantization; the bench itself only
# loads, measures, and writes results. If you re-run the bench with the same
# exported artifact, skip the export cells.

# %% [markdown]
# ## 1. Install dependencies
#
# `parapet-runner[bench]` pulls onnxruntime, transformers, sentencepiece.
# `optimum[exporters]` is for the one-shot HF→ONNX export below; if you re-run
# the bench against an existing artifact, you can skip it.

# %%
# !pip install -q "git+https://github.com/<org>/parapet.git#subdirectory=parapet-runner&egg=parapet-runner[bench]"
# !pip install -q "optimum[exporters]>=1.20"
# # On a GPU runtime, prefer onnxruntime-gpu and uninstall the CPU build first:
# # !pip uninstall -y onnxruntime
# # !pip install -q onnxruntime-gpu

# %% [markdown]
# ## 2. Export mDeBERTa to ONNX
#
# One-shot. Skip on subsequent bench reruns.

# %%
# !optimum-cli export onnx \
#     --model microsoft/mdeberta-v3-base \
#     --task feature-extraction \
#     --opset 17 \
#     /kaggle/working/mdeberta-onnx

# %% [markdown]
# ## 3. (Optional) Quantize to int8
#
# Direction.md Phase 0.6 measures int8 specifically. Use ORT's dynamic
# quantizer; it does not require a calibration dataset.

# %%
# !python -c "
# from onnxruntime.quantization import quantize_dynamic, QuantType
# quantize_dynamic(
#     '/kaggle/working/mdeberta-onnx/model.onnx',
#     '/kaggle/working/mdeberta-onnx/model.int8.onnx',
#     weight_type=QuantType.QInt8,
# )
# "

# %% [markdown]
# ## 4. Hash the ONNX model
#
# `parapet_runner.latency_bench --onnx-sha256` requires the digest so the
# adapter refuses to load a swapped artifact.

# %%
# !python -c "
# import hashlib, sys
# path = '/kaggle/working/mdeberta-onnx/model.int8.onnx'
# h = hashlib.sha256()
# with open(path, 'rb') as f:
#     while chunk := f.read(65536):
#         h.update(chunk)
# print(h.hexdigest())
# " | tee /kaggle/working/mdeberta-onnx/model.int8.sha256

# %% [markdown]
# ## 5. Pin the corpus
#
# Real curated train/val text, not holdout. Replace the dataset/file names
# with whatever you mounted as a Kaggle input.

# %%
# CORPUS_PATH = "/kaggle/input/<your-curated-train-dataset>/train.jsonl"

# %% [markdown]
# ## 6. Run the bench
#
# All bench logic is in `parapet_runner.latency_bench`; this cell just
# invokes it with the artifacts pinned above.

# %%
# import subprocess
# result = subprocess.run([
#     "python", "-m", "parapet_runner.latency_bench",
#     "--model-path", "/kaggle/working/mdeberta-onnx/model.int8.onnx",
#     "--tokenizer-path", "/kaggle/working/mdeberta-onnx",
#     "--model-revision", "microsoft/mdeberta-v3-base@<git_sha_from_HF>",
#     "--onnx-sha256", open("/kaggle/working/mdeberta-onnx/model.int8.sha256").read().strip(),
#     "--quant", "int8",
#     "--provider", "CPUExecutionProvider",   # or CUDAExecutionProvider on GPU runtimes
#     "--corpus", CORPUS_PATH,
#     "--output", "/kaggle/working/latency_result.json",
#     "--environment", "kaggle",
# ], check=True)

# %% [markdown]
# ## 7. Inspect the result
#
# Look for `end_to_end.p50_ms` and `end_to_end.p99_ms`. Direction.md Phase 0.6
# gates: p50 ≤ 25ms, p99 ≤ 100ms. If the candidate misses both by a wide
# margin, the model is not shippable as the default L2a path — fall back to
# MiniLM and update direction.md accordingly.

# %%
# import json
# result = json.loads(open("/kaggle/working/latency_result.json").read())
# print("end-to-end percentiles (ms):", result["end_to_end"])
# print("token length histogram:", result["token_length_histogram"])
# print("manifest:", json.dumps(result["manifest"], indent=2))
