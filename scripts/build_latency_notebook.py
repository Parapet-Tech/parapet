"""Builds notebooks/latency_bench_kaggle.ipynb deterministically.

Run from the parapet/ project root:

    python scripts/build_latency_notebook.py

The notebook is regenerated from the cell definitions below. Edit this script,
not the .ipynb, when changing notebook content. Treat the .ipynb as a generated
artifact: round-trip diffs against this script should be empty.
"""

from __future__ import annotations

import json
from pathlib import Path


def md_cell(cell_id: str, lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _with_newlines(lines),
    }


def code_cell(cell_id: str, lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _with_newlines(lines),
    }


def _with_newlines(lines: list[str]) -> list[str]:
    """Re-attach trailing newlines to all but the last line.

    nbformat stores cell.source as a list of strings, where every line except
    the last ends in '\\n'. Round-tripping through Jupyter expects this shape.
    """
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


CELLS = [
    md_cell("00-title", [
        "# L2 latency feasibility — single Kaggle notebook",
        "",
        "Runs end-to-end in one notebook with internet on:",
        "",
        "1. set `MODEL_ID` and bench knobs (`MAX_SEQ_LEN`, `INTRA_THREAD_SWEEP`) in the config cell",
        "2. install build deps (`optimum[onnx]`, `huggingface_hub`)",
        "3. export `MODEL_ID` to ONNX",
        "4. dynamic int8 quantize",
        "5. hash the int8 model",
        "6. resolve immutable HF commit SHA",
        "7. fail-fast validation",
        "8. install `parapet-runner[bench]` from the public repo (pinned for apples-to-apples bench code)",
        "9. run the bench once per `intra_op_threads` value in `INTRA_THREAD_SWEEP` (truncating inputs at `MAX_SEQ_LEN`)",
        "10. inspect each result, with PASS/FAIL against the `direction.md` Phase 0.6 gates",
        "",
        "**Inputs needed:** one Kaggle dataset containing `l2_latency_v8_train_val_stratified.jsonl` (slug `parapet-l2-latency`). The notebook glob-discovers the corpus under `/kaggle/input/`, so the actual mount path doesn't matter.",
        "",
        "**Note on truncation vs filtering:** `MAX_SEQ_LEN` truncates over-length inputs at the tokenizer (deployment-realistic) — it does NOT drop them from the corpus. The token-length histogram is post-truncation.",
        "",
        "**Runtime:** CPU. **Internet:** on.",
    ]),
    code_cell("05-config", [
        "# Single source of truth for what we're benching.",
        "# Swap MODEL_ID to bench a different candidate; the rest of the notebook follows.",
        "import glob",
        "",
        'MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"',
        'EXPORT_DIR = "onnx-export"',
        "",
        "# Bench knobs.",
        "# MAX_SEQ_LEN truncates the tokenizer (deployment-realistic), it does not filter the corpus.",
        "# INTRA_THREAD_SWEEP runs one bench per value, all against the same exported model.",
        "MAX_SEQ_LEN = 128",
        "INTRA_THREAD_SWEEP = [1, 4]",
        "",
        "# Glob-discover the corpus. Kaggle's actual mount path varies",
        "# (e.g. /kaggle/input/datasets/<username>/<slug>/...), so we search.",
        "_matches = glob.glob(",
        '    "/kaggle/input/**/l2_latency_v8_train_val_stratified.jsonl",',
        "    recursive=True,",
        ")",
        'assert len(_matches) == 1, f"expected exactly one corpus match, got: {_matches}"',
        "CORPUS_PATH = _matches[0]",
        "",
        'print(f"MODEL_ID            = {MODEL_ID}")',
        'print(f"EXPORT_DIR          = {EXPORT_DIR}")',
        'print(f"CORPUS_PATH         = {CORPUS_PATH}")',
        'print(f"MAX_SEQ_LEN         = {MAX_SEQ_LEN}")',
        'print(f"INTRA_THREAD_SWEEP  = {INTRA_THREAD_SWEEP}")',
    ]),
    code_cell("10-install-build-deps", [
        '!pip install -q "optimum[onnx]>=1.20" "huggingface_hub>=0.25" "onnxruntime>=1.17"',
    ]),
    code_cell("20-export", [
        "!optimum-cli export onnx \\",
        "    --model {MODEL_ID} \\",
        "    --task feature-extraction \\",
        "    --opset 17 \\",
        "    {EXPORT_DIR}",
    ]),
    code_cell("30-quantize", [
        "from onnxruntime.quantization import quantize_dynamic, QuantType",
        "quantize_dynamic(",
        '    f"{EXPORT_DIR}/model.onnx",',
        '    f"{EXPORT_DIR}/model.int8.onnx",',
        "    weight_type=QuantType.QInt8,",
        ")",
        'print("quantized OK")',
    ]),
    code_cell("40-hash", [
        "import hashlib",
        "h = hashlib.sha256()",
        'with open(f"{EXPORT_DIR}/model.int8.onnx", "rb") as f:',
        "    while chunk := f.read(65536):",
        "        h.update(chunk)",
        "ONNX_SHA = h.hexdigest()",
        'open(f"{EXPORT_DIR}/model.int8.sha256", "w").write(ONNX_SHA)',
        'print(f"int8 SHA-256: {ONNX_SHA}")',
    ]),
    code_cell("50-revision", [
        "from huggingface_hub import HfApi",
        "HF_REV = HfApi().model_info(MODEL_ID).sha",
        'open(f"{EXPORT_DIR}/revision.txt", "w").write(HF_REV)',
        'print(f"{MODEL_ID} revision: {HF_REV}")',
    ]),
    code_cell("60-validate", [
        "import os, re",
        "import numpy as np",
        "import onnxruntime as ort",
        "from transformers import AutoTokenizer",
        "",
        "for required in [",
        '    f"{EXPORT_DIR}/model.int8.onnx",',
        '    f"{EXPORT_DIR}/model.int8.sha256",',
        '    f"{EXPORT_DIR}/revision.txt",',
        "]:",
        '    assert os.path.isfile(required), f"missing artifact: {required}"',
        "",
        "tok = AutoTokenizer.from_pretrained(EXPORT_DIR, local_files_only=True, use_fast=True)",
        'encoded = tok("hello world", return_tensors="np", padding=False, truncation=True, max_length=128)',
        'assert encoded["input_ids"].shape[1] > 0',
        "",
        'sess = ort.InferenceSession(f"{EXPORT_DIR}/model.int8.onnx", providers=["CPUExecutionProvider"])',
        "input_names = {i.name for i in sess.get_inputs()}",
        "feed = {n: np.asarray(encoded[n]) for n in input_names if n in encoded}",
        "missing = input_names - feed.keys()",
        'assert not missing, f"tokenizer missing required ONNX inputs: {missing}"',
        "out = sess.run(None, feed)",
        "assert out and out[0].size > 0",
        "",
        'rev = open(f"{EXPORT_DIR}/revision.txt").read().strip()',
        'assert re.fullmatch(r"[0-9a-fA-F]{40}", rev), f"revision.txt not a 40-char hex SHA: {rev!r}"',
        'print(f"validation OK  (inputs={sorted(input_names)}, out={out[0].shape}, rev={rev})")',
    ]),
    code_cell("70-install-bench", [
        "# Pin the bench commit so MiniLM and mDeBERTa runs share identical bench code.",
        "# Do not bump this casually; bumping changes what we're measuring.",
        '!pip install -q "parapet-runner[bench] @ git+https://github.com/Parapet-Tech/parapet.git@9ac341e#subdirectory=parapet-runner"',
    ]),
    code_cell("80-run", [
        "# One bench run per intra_op_threads value, all against the same exported model.",
        "# Outputs land in /kaggle/working/ with config-derived filenames so multiple runs",
        "# in the same session don't clobber each other.",
        "import subprocess",
        "",
        'with open(f"{EXPORT_DIR}/revision.txt") as _f:',
        "    _hf_rev = _f.read().strip()",
        'with open(f"{EXPORT_DIR}/model.int8.sha256") as _f:',
        "    _onnx_sha = _f.read().strip()",
        "",
        "RESULT_FILES = []",
        "for _intra in INTRA_THREAD_SWEEP:",
        '    out_path = f"/kaggle/working/latency_result_seq{MAX_SEQ_LEN}_intra{_intra}.json"',
        '    print(f"\\n=== running max_seq_len={MAX_SEQ_LEN}, intra_op_threads={_intra} -> {out_path} ===")',
        "    subprocess.run(",
        "        [",
        '            "python", "-m", "parapet_runner.latency_bench",',
        '            "--model-path", f"{EXPORT_DIR}/model.int8.onnx",',
        '            "--tokenizer-path", EXPORT_DIR,',
        '            "--model-revision", f"{MODEL_ID}@{_hf_rev}",',
        '            "--onnx-sha256", _onnx_sha,',
        '            "--quant", "int8",',
        '            "--provider", "CPUExecutionProvider",',
        '            "--corpus", CORPUS_PATH,',
        '            "--output", out_path,',
        '            "--environment", "kaggle",',
        '            "--max-seq-len", str(MAX_SEQ_LEN),',
        '            "--intra-op-threads", str(_intra),',
        "        ],",
        "        check=True,",
        "    )",
        "    RESULT_FILES.append(out_path)",
        "",
        'print(f"\\nresult files: {RESULT_FILES}")',
    ]),
    code_cell("90-inspect", [
        "import json",
        "",
        "P50_GATE_MS = 25",
        "P99_GATE_MS = 100",
        "",
        "for _path in RESULT_FILES:",
        "    r = json.loads(open(_path).read())",
        "    cfg = r['config']",
        "    print(f\"\\n=== {_path} ===\")",
        "    print(f\"max_seq_len      = {cfg['max_seq_len']}\")",
        "    print(f\"intra_op_threads = {cfg['intra_op_threads']}\")",
        "    print(f\"end_to_end (ms)  : {r['end_to_end']}\")",
        "    print(f\"infer      (ms)  : {r['infer']}\")",
        "    print(f\"tokenize   (ms)  : {r['tokenize']}\")",
        "    print(f\"token length histogram (post-truncation): {r['token_length_histogram']}\")",
        "    p50 = r['end_to_end']['p50_ms']",
        "    p99 = r['end_to_end']['p99_ms']",
        "    p50_verdict = 'PASS' if p50 <= P50_GATE_MS else 'FAIL'",
        "    p99_verdict = 'PASS' if p99 <= P99_GATE_MS else 'FAIL'",
        "    print(f\"p50 = {p50:.2f} ms (\u2264 {P50_GATE_MS})   {p50_verdict}\")",
        "    print(f\"p99 = {p99:.2f} ms (\u2264 {P99_GATE_MS})  {p99_verdict}\")",
        "",
        "print('\\n--- summary ---')",
        "print(f\"{'intra':>6}  {'p50_ms':>10}  {'p99_ms':>10}  verdict\")",
        "for _path in RESULT_FILES:",
        "    r = json.loads(open(_path).read())",
        "    intra = r['config']['intra_op_threads']",
        "    p50 = r['end_to_end']['p50_ms']",
        "    p99 = r['end_to_end']['p99_ms']",
        "    verdict = 'PASS' if (p50 <= P50_GATE_MS and p99 <= P99_GATE_MS) else 'FAIL'",
        "    print(f\"{intra:>6}  {p50:>10.2f}  {p99:>10.2f}  {verdict}\")",
    ]),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "notebooks" / "latency_bench_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
