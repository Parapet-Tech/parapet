"""Tests for parapet-runner/scripts/phase1_truncation_policy_spike.py.

Avoids loading the real MiniLM-L12 model — uses synthetic encoder + tokenizer
fakes so the test suite stays under a second.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "phase1_truncation_policy_spike.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase1_truncation_policy_spike", script_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / "trunc_spike" / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------


def test_stratified_subsample_preserves_label_language_ratios():
    m = _load_module()
    rows = (
        [{"label": "benign", "language": "EN"} for _ in range(60)]
        + [{"label": "benign", "language": "RU"} for _ in range(20)]
        + [{"label": "malicious", "language": "EN"} for _ in range(15)]
        + [{"label": "malicious", "language": "ZH"} for _ in range(5)]
    )
    out = m.stratified_subsample(rows, n=20, seed=42)
    assert len(out) == 20
    # Approximately 60% benign-EN, 20% benign-RU, 15% mal-EN, 5% mal-ZH
    counts = {}
    for r in out:
        counts[(r["label"], r["language"])] = counts.get((r["label"], r["language"]), 0) + 1
    # Largest cell preserved
    assert counts.get(("benign", "EN"), 0) >= 10  # ~60% of 20


def test_stratified_subsample_returns_full_when_n_geq_total():
    m = _load_module()
    rows = [{"label": "benign", "language": "EN"} for _ in range(5)]
    out = m.stratified_subsample(rows, n=100, seed=42)
    assert len(out) == 5


def test_stratified_subsample_deterministic_for_seed():
    m = _load_module()
    rows = [{"label": "benign", "language": "EN"} for _ in range(50)]
    out_a = m.stratified_subsample(rows, n=10, seed=42)
    out_b = m.stratified_subsample(rows, n=10, seed=42)
    out_c = m.stratified_subsample(rows, n=10, seed=99)
    assert out_a == out_b
    # Different seed -> different selection (not guaranteed but very likely with 50 rows)
    # so we just assert sizes match
    assert len(out_c) == 10


# ---------------------------------------------------------------------------
# Fold split
# ---------------------------------------------------------------------------


def test_split_by_fold_partitions_correctly():
    m = _load_module()
    rows = [
        {"fold_id": 0, "id": "a"}, {"fold_id": 1, "id": "b"},
        {"fold_id": 2, "id": "c"}, {"fold_id": 3, "id": "d"},
        {"fold_id": 4, "id": "e"}, {"fold_id": 4, "id": "f"},
    ]
    train, val = m.split_by_fold(rows, val_fold=4)
    assert [r["id"] for r in train] == ["a", "b", "c", "d"]
    assert [r["id"] for r in val] == ["e", "f"]


def test_split_by_fold_handles_empty_val():
    m = _load_module()
    rows = [{"fold_id": 0}, {"fold_id": 1}]
    train, val = m.split_by_fold(rows, val_fold=4)
    assert len(train) == 2
    assert val == []


# ---------------------------------------------------------------------------
# build_truncated_strings (uses real tokenization shape but synthetic ids)
# ---------------------------------------------------------------------------


def test_build_truncated_strings_head_128():
    m = _load_module()
    text = "x " * 200  # 200 char-positioned tokens
    token_ids = list(range(200))
    offsets = [(2 * i, 2 * i + 1) for i in range(200)]  # each char "x"
    out = m.build_truncated_strings(text, token_ids, offsets, policy="head_128")
    assert len(out) == 1
    # Should cover the first 128 tokens' span: chars 0..255 (last char of token 127 = end 255)
    assert len(out[0]) == 255  # 0 -> 255 exclusive of end? str slice len = 255
    # Must be a prefix of original text
    assert text.startswith(out[0])


def test_build_truncated_strings_tail_128():
    m = _load_module()
    text = "x " * 200
    token_ids = list(range(200))
    offsets = [(2 * i, 2 * i + 1) for i in range(200)]
    out = m.build_truncated_strings(text, token_ids, offsets, policy="tail_128")
    assert len(out) == 1
    # Should cover the last 128 tokens (positions 72..199)
    assert text.endswith(out[0])


def test_build_truncated_strings_head_tail_64_64():
    m = _load_module()
    text = "x " * 200
    token_ids = list(range(200))
    offsets = [(2 * i, 2 * i + 1) for i in range(200)]
    out = m.build_truncated_strings(text, token_ids, offsets, policy="head_tail_64_64")
    assert len(out) == 1
    # The format is "<head_text> ... <tail_text>"
    assert " ... " in out[0]


def test_build_truncated_strings_full_512_chunks_for_long_input():
    m = _load_module()
    text = "x" * 1024
    token_ids = list(range(512))
    offsets = [(i * 2, i * 2 + 1) for i in range(512)]
    out = m.build_truncated_strings(text, token_ids, offsets, policy="full_512")
    # 512 tokens chunked at 128 with stride 64 = ~7 chunks
    assert len(out) > 1
    assert all(s.strip() for s in out)


def test_build_truncated_strings_full_512_single_chunk_when_short():
    m = _load_module()
    text = "x " * 50
    token_ids = list(range(50))
    offsets = [(2 * i, 2 * i + 1) for i in range(50)]
    out = m.build_truncated_strings(text, token_ids, offsets, policy="full_512")
    assert len(out) == 1


# ---------------------------------------------------------------------------
# recall_at_fpr
# ---------------------------------------------------------------------------


def test_recall_at_fpr_perfect_separator():
    m = _load_module()
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    recall, fpr, _ = m.recall_at_fpr(y, s, target_fpr=0.10)
    assert recall == 1.0
    assert fpr <= 0.10


def test_recall_at_fpr_random_scores_under_constraint():
    m = _load_module()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.6, 0.5, 0.4, 0.3, 0.7, 0.6, 0.5, 0.4])
    recall, fpr, _ = m.recall_at_fpr(y, s, target_fpr=0.25)
    assert 0 <= recall <= 1
    assert fpr <= 0.25


# ---------------------------------------------------------------------------
# evaluate_policy with synthetic embeddings
# ---------------------------------------------------------------------------


def test_evaluate_policy_separable_embeddings():
    m = _load_module()
    rng = np.random.default_rng(0)
    train_emb = np.vstack([
        rng.normal(loc=-1.0, size=(40, 8)).astype(np.float32),
        rng.normal(loc=1.0, size=(40, 8)).astype(np.float32),
    ])
    train_labels = np.array([0] * 40 + [1] * 40)
    val_emb = np.vstack([
        rng.normal(loc=-1.0, size=(20, 8)).astype(np.float32),
        rng.normal(loc=1.0, size=(20, 8)).astype(np.float32),
    ])
    val_rows = (
        [{"label": "benign", "language": "EN"} for _ in range(20)]
        + [{"label": "malicious", "language": "EN"} for _ in range(20)]
    )
    out = m.evaluate_policy(
        val_rows, val_emb, train_emb, train_labels,
        target_fpr=0.10, seed=0,
    )
    assert out["recall_at_fpr"] > 0.5  # separable -> easy
    assert out["roc_auc"] > 0.8
    assert "EN" in out["per_language"]


# ---------------------------------------------------------------------------
# load_or_encode cache
# ---------------------------------------------------------------------------


def _fake_tokenize(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    # Char-based tokenization: one "token" per non-space char.
    tokens = []
    offsets = []
    for i, ch in enumerate(text):
        if ch != " ":
            tokens.append(ord(ch) % 1000)
            offsets.append((i, i + 1))
    return tokens, offsets


def _fake_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic 8-dim embedding from text length + first-byte hash."""
    out = np.zeros((len(texts), 8), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, 0] = float(len(t)) / 1000.0
        out[i, 1] = float(ord(t[0]) if t else 0) / 256.0
    return out


def test_load_or_encode_caches_and_reuses():
    m = _load_module()
    output_dir = _new_output_dir("cache_reuse")
    rows = [{"content": f"row {i} text"} for i in range(5)]

    call_count = {"n": 0}

    def counting_encoder(texts):
        call_count["n"] += 1
        return _fake_encoder(texts)

    emb1 = m.load_or_encode(
        rows, policy="head_128", split="train",
        output_dir=output_dir, tokenize_fn=_fake_tokenize,
        encoder=counting_encoder,
    )
    assert emb1.shape == (5, 8)
    n_after_first = call_count["n"]
    assert n_after_first > 0

    emb2 = m.load_or_encode(
        rows, policy="head_128", split="train",
        output_dir=output_dir, tokenize_fn=_fake_tokenize,
        encoder=counting_encoder,
    )
    np.testing.assert_array_equal(emb1, emb2)
    # Cache should have prevented further encoder calls.
    assert call_count["n"] == n_after_first


def test_load_or_encode_force_re_encodes():
    m = _load_module()
    output_dir = _new_output_dir("cache_force")
    rows = [{"content": "row text"} for _ in range(3)]
    call_count = {"n": 0}

    def counting_encoder(texts):
        call_count["n"] += 1
        return _fake_encoder(texts)

    m.load_or_encode(
        rows, policy="tail_128", split="train",
        output_dir=output_dir, tokenize_fn=_fake_tokenize,
        encoder=counting_encoder,
    )
    n1 = call_count["n"]
    m.load_or_encode(
        rows, policy="tail_128", split="train",
        output_dir=output_dir, tokenize_fn=_fake_tokenize,
        encoder=counting_encoder, force=True,
    )
    assert call_count["n"] > n1


# ---------------------------------------------------------------------------
# Legacy path guard in main()
# ---------------------------------------------------------------------------


def test_main_refuses_legacy_holdout_path():
    """Per the user's instruction: no legacy schema/eval YAML may enter the
    decision path of the truncation spike."""
    m = _load_module()
    output_dir = _new_output_dir("guard_holdout")
    fake_legacy = output_dir / "schema" / "eval" / "l1_holdout.yaml"
    fake_legacy.parent.mkdir(parents=True, exist_ok=True)
    fake_legacy.write_text("[]", encoding="utf-8")
    rc = m.main([
        "--residuals", str(fake_legacy),
        "--output-dir", str(output_dir / "out"),
    ])
    assert rc == 1


def test_main_refuses_legacy_challenge_path():
    m = _load_module()
    output_dir = _new_output_dir("guard_challenge")
    fake_legacy = output_dir / "schema" / "eval" / "challenges" / "x.yaml"
    fake_legacy.parent.mkdir(parents=True, exist_ok=True)
    fake_legacy.write_text("[]", encoding="utf-8")
    rc = m.main([
        "--residuals", str(fake_legacy),
        "--output-dir", str(output_dir / "out"),
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# End-to-end: synthetic encoder, no model load
# ---------------------------------------------------------------------------


def test_main_e2e_with_fake_encoder(monkeypatch):
    """Smoke test of main() with monkey-patched encoder + tokenizer.

    Verifies the wiring from residuals.jsonl through encode -> linear head ->
    summary/manifest works end-to-end, without loading MiniLM.
    """
    m = _load_module()
    output_dir = _new_output_dir("e2e_fake")
    residuals = output_dir / "residuals.jsonl"
    rows = []
    rng = np.random.default_rng(0)
    # Build a residual file with both classes, EN+RU, folds 0..4.
    for i in range(80):
        label = "malicious" if i % 4 == 0 else "benign"
        lang = "EN" if i % 3 else "RU"
        fold = i % 5
        rows.append({
            "content": f"row {i} {'attack ignore previous' if label == 'malicious' else 'benign text'}",
            "label": label, "language": lang,
            "reason": "instruction_override" if label == "malicious" else "discussion_benign",
            "source": "src", "format_bin": "prose", "length_bin": "short",
            "content_hash": str(i), "fold_id": fold,
            "l1_score": 0.5, "l1_decision": "allow",
            "raw_score": 0.0,
        })
    with residuals.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Patch the encoder + tokenizer so no model is loaded.
    monkeypatch.setattr(m, "make_default_encoder",
                        lambda *a, **k: _fake_encoder)

    class FakeTokenizer:
        def __call__(self, text, return_offsets_mapping=False,
                     add_special_tokens=False, truncation=False):
            ids, offsets = _fake_tokenize(text or " ")
            return {"input_ids": ids, "offset_mapping": offsets}

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(model_id: str):
            return FakeTokenizer()

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTok)

    rc = m.main([
        "--residuals", str(residuals),
        "--output-dir", str(output_dir / "out"),
        "--policy", "head_128",
        "--policy", "tail_128",
        "--sample-size", "0", "--full",
        "--target-fpr", "0.20",
    ])
    assert rc == 0
    out = output_dir / "out"
    assert (out / "summary.json").exists()
    assert (out / "summary.md").exists()
    assert (out / "manifest.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["ranking"]) == 2
    policies_in_ranking = {e["policy"] for e in summary["ranking"]}
    assert policies_in_ranking == {"head_128", "tail_128"}
    # Manifest carries the ranking-only caveat
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "ranking only" in manifest["spike_scope"].lower()
