"""Evaluate the L1 -> L2 cascade end-to-end on the Phase 4 challenge set.

Simulates the cascade routing:
  1. L1 scores every sample (from Phase 4 eval.json)
  2. Routing: confident allow / confident block / route to L2
  3. L2 decides on routed traffic (backend selectable)
  4. Reports end-to-end metrics by source family

Backends:
  --l2-backend l2b        XGBoost/RF/LR over L1 harness + structural + TF-IDF
                          (requires --l2b-model-dir)
  --l2-backend tinybert   TinyBERT ONNX sequence classifier
                          (requires --tinybert-dir)

Usage:
    cd parapet
    python scripts/eval_cascade.py \
      --phase4-eval parapet-runner/runs/phase4_final/run/_eval_holdout_0p0/eval.json \
      --challenge-attack schema/eval/challenges/tough_attack_v2/tough_attack_v6_novel.yaml \
      --challenge-neutral schema/eval/challenges/tough_neutral_v2/tough_neutral_v6_novel.yaml \
      --l2-backend tinybert --tinybert-dir models/tinybert_l2 \
      --output-dir parapet-runner/runs/cascade_eval_tinybert
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import hstack, csr_matrix


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Structural features (must match train_l2b.py exactly)
# ---------------------------------------------------------------------------

QUOTE_PATTERNS = [
    re.compile(r'"[^"]{10,}"'),
    re.compile(r"'[^']{10,}'"),
    re.compile(r'`[^`]{10,}`'),
    re.compile(r'```[\s\S]*?```'),
    re.compile(r'^>\s', re.MULTILINE),
]

CODE_MARKERS = re.compile(
    r'(?:def |class |import |function |var |let |const |if\s*\(|for\s*\(|while\s*\(|return |print\()',
    re.IGNORECASE,
)

FRAMING_MARKERS = re.compile(
    r'\b(?:example|for instance|such as|e\.g\.|here is|the following|'
    r'writeup|write-up|analysis|CTF|challenge|security|discuss|'
    r'prompt injection|jailbreak|attack technique|red team)\b',
    re.IGNORECASE,
)


def compute_structural_features(text: str) -> dict[str, float]:
    lines = text.split("\n")
    n_lines = len(lines)
    n_chars = len(text)
    n_words = len(text.split())
    n_punct = sum(1 for c in text if c in string.punctuation)
    n_digits = sum(1 for c in text if c.isdigit())
    n_upper = sum(1 for c in text if c.isupper())
    n_spaces = sum(1 for c in text if c.isspace())
    n_quotes = sum(len(pat.findall(text)) for pat in QUOTE_PATTERNS)
    n_code_markers = len(CODE_MARKERS.findall(text))
    n_framing_markers = len(FRAMING_MARKERS.findall(text))

    if n_chars > 0:
        char_counts = Counter(text)
        entropy = -sum(
            (count / n_chars) * math.log2(count / n_chars)
            for count in char_counts.values()
        )
    else:
        entropy = 0.0

    words = text.lower().split()
    if words:
        word_counts = Counter(words)
        repeated = sum(1 for w, c in word_counts.items() if c > 1)
        repeated_ratio = repeated / len(word_counts)
    else:
        repeated_ratio = 0.0

    n_delimiters = sum(1 for c in text if c in "{}[]()<>|/\\;:=+*&^%$#@!")
    safe_chars = max(n_chars, 1)
    safe_words = max(n_words, 1)

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": n_lines,
        "avg_line_length": n_chars / max(n_lines, 1),
        "punct_ratio": n_punct / safe_chars,
        "digit_ratio": n_digits / safe_chars,
        "upper_ratio": n_upper / safe_chars,
        "space_ratio": n_spaces / safe_chars,
        "delimiter_density": n_delimiters / safe_chars,
        "quote_count": n_quotes,
        "code_marker_count": n_code_markers,
        "framing_marker_count": n_framing_markers,
        "entropy": entropy,
        "repeated_token_ratio": repeated_ratio,
        "avg_word_length": n_chars / safe_words,
    }


# ---------------------------------------------------------------------------
# Load challenge data with L1 harness signals from Phase 4 eval
# ---------------------------------------------------------------------------

def load_challenge_with_signals(
    eval_json_path: Path,
    attack_yaml_path: Path,
    neutral_yaml_path: Path,
) -> list[dict[str, Any]]:
    """Join Phase 4 eval results with challenge set metadata."""
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    results = eval_data["results"]

    attack = yaml.safe_load(attack_yaml_path.read_text(encoding="utf-8"))
    neutral = yaml.safe_load(neutral_yaml_path.read_text(encoding="utf-8"))

    # Build id -> source mapping
    id_to_meta: dict[str, dict] = {}
    for row in attack:
        desc = row.get("description", "")
        source = desc.split("source=")[1].split(" ")[0] if "source=" in desc else "unknown"
        id_to_meta[row["id"]] = {"source": source, "true_label": "malicious", "content": row["content"]}
    for row in neutral:
        desc = row.get("description", "")
        source = desc.split("source=")[1].split(" ")[0] if "source=" in desc else "unknown"
        id_to_meta[row["id"]] = {"source": source, "true_label": "benign", "content": row["content"]}

    samples = []
    for r in results:
        case_id = r["case_id"]
        meta = id_to_meta.get(case_id, {})

        l1_raw = r.get("l1_signals") or []
        l1 = l1_raw[0] if isinstance(l1_raw, list) and l1_raw else {}

        samples.append({
            "case_id": case_id,
            "content": meta.get("content", ""),
            "true_label": meta.get("true_label", "unknown"),
            "source": meta.get("source", "unknown"),
            "l1_expected": r["expected"],
            "l1_actual": r["actual"],
            "l1_correct": r["correct"],
            "raw_score": l1.get("raw_score", 0.0),
            "raw_unquoted_score": l1.get("raw_unquoted_score", 0.0),
            "raw_squash_score": l1.get("raw_squash_score", 0.0),
            "raw_score_delta": l1.get("raw_score_delta", 0.0),
            "quote_detected": l1.get("quote_detected", False),
            "score": l1.get("score", 0.0),
            "unquoted_score": l1.get("unquoted_score", 0.0),
            "squash_score": l1.get("squash_score", 0.0),
        })

    return samples


# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------

def simulate_cascade(
    samples: list[dict[str, Any]],
    rf_model: Any,
    tfidf: Any,
    t_allow: float,
    t_block: float,
    families: set[str] | None = None,
    extract_fn: Any = None,
) -> list[dict[str, Any]]:
    """Simulate L1 -> routing -> L2b cascade.

    Routing based on L1 raw_score:
      raw_score < t_allow  -> allow (L1 confident benign)
      raw_score >= t_block -> block (L1 confident attack)
      otherwise            -> route to L2b
    """
    if families is None:
        families = {"harness", "structural", "tfidf"}

    routed_indices = []
    decisions = [""] * len(samples)

    for i, s in enumerate(samples):
        raw = s["raw_score"]
        if raw < t_allow:
            decisions[i] = "l1_allow"
        elif raw >= t_block:
            decisions[i] = "l1_block"
        else:
            routed_indices.append(i)
            decisions[i] = "routed"

    if routed_indices and extract_fn is not None:
        routed_samples = [samples[i] for i in routed_indices]
        X, _, _ = extract_fn(routed_samples, families, tfidf, fit=False)

        l2b_preds = rf_model.predict(X)
        l2b_probs = rf_model.predict_proba(X)[:, 1]

        for j, i in enumerate(routed_indices):
            if l2b_preds[j] == 1:
                decisions[i] = "l2b_block"
            else:
                decisions[i] = "l2b_allow"
            samples[i]["l2b_prob"] = float(l2b_probs[j])

    # Compute final predictions
    results = []
    for i, s in enumerate(samples):
        decision = decisions[i]
        final_pred = "malicious" if decision in ("l1_block", "l2b_block") else "benign"
        results.append({
            **s,
            "cascade_decision": decision,
            "cascade_pred": final_pred,
        })

    return results


def compute_metrics(
    results: list[dict[str, Any]],
    label: str = "",
) -> dict[str, Any]:
    """Compute confusion matrix and metrics."""
    tp = fp = fn = tn = 0
    for r in results:
        true = r["true_label"]
        pred = r["cascade_pred"]
        if true == "malicious" and pred == "malicious":
            tp += 1
        elif true == "benign" and pred == "malicious":
            fp += 1
        elif true == "malicious" and pred == "benign":
            fn += 1
        else:
            tn += 1

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    n_pos = tp + fn
    n_neg = fp + tn
    fpr = fp / n_neg if n_neg > 0 else 0.0

    route_count = sum(1 for r in results if r["cascade_decision"].startswith("l2b_"))
    route_rate = route_count / n if n > 0 else 0.0

    return {
        "label": label,
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "route_count": route_count,
        "route_rate": route_rate,
    }


# ---------------------------------------------------------------------------
# L2 backends
# ---------------------------------------------------------------------------


class TinyBertONNXModel:
    """Sklearn-shaped wrapper around a TinyBERT ONNX model directory.

    Expects `<model_dir>/model.onnx` and a HuggingFace tokenizer dump
    (tokenizer.json + tokenizer_config.json). Pair with
    `tinybert_extract_fn` so `simulate_cascade` can reuse the same code
    path as the L2b backend.
    """

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"))
        self.max_length = max_length
        self.batch_size = batch_size
        self.threshold = threshold

    def _probs(self, samples: list[dict]) -> np.ndarray:
        texts = [s["content"] for s in samples]
        out = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            logits = self.session.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            out[i : i + self.batch_size] = ex[:, 1] / ex.sum(axis=1)
        return out

    def predict_proba(self, samples: list[dict]) -> np.ndarray:
        p = self._probs(samples)
        return np.column_stack([1.0 - p, p])

    def predict(self, samples: list[dict]) -> np.ndarray:
        return (self._probs(samples) >= self.threshold).astype(int)


def tinybert_extract_fn(samples, families, tfidf, fit=False):
    """Pass-through extractor: TinyBertONNXModel consumes sample dicts directly."""
    return samples, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate L1+L2 cascade end-to-end")
    parser.add_argument("--phase4-eval", type=Path, required=True)
    parser.add_argument("--challenge-attack", type=Path, required=True)
    parser.add_argument("--challenge-neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--l2-backend",
        choices=["l2b", "tinybert"],
        default="l2b",
        help="L2 scorer: 'l2b' = XGBoost/RF/LR, 'tinybert' = ONNX transformer",
    )
    parser.add_argument("--l2b-model-dir", type=Path, default=None,
                        help="Required when --l2-backend=l2b")
    parser.add_argument("--tinybert-dir", type=Path, default=None,
                        help="Required when --l2-backend=tinybert")
    parser.add_argument("--tinybert-threshold", type=float, default=0.5)
    parser.add_argument("--tinybert-batch-size", type=int, default=32)
    parser.add_argument("--tinybert-max-length", type=int, default=512)
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Evaluate the L2 backend alone, bypassing L1 routing. "
             "Scores every challenge sample and reports per-source metrics. "
             "This is the right mode for answering 'is this L2 model any good?'",
    )
    args = parser.parse_args()

    if args.l2_backend == "l2b" and args.l2b_model_dir is None:
        parser.error("--l2b-model-dir required when --l2-backend=l2b")
    if args.l2_backend == "tinybert" and args.tinybert_dir is None:
        parser.error("--tinybert-dir required when --l2-backend=tinybert")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 4 eval with harness signals
    print("Loading Phase 4 challenge eval...")
    samples = load_challenge_with_signals(
        Path(args.phase4_eval).resolve(),
        Path(args.challenge_attack).resolve(),
        Path(args.challenge_neutral).resolve(),
    )
    print(f"  {len(samples)} samples")
    n_attack = sum(1 for s in samples if s["true_label"] == "malicious")
    n_benign = sum(1 for s in samples if s["true_label"] == "benign")
    print(f"  {n_attack} attack, {n_benign} benign")

    # Load L2 backend
    tfidf = None
    families: set[str] = set()
    extract_fn: Any = None

    if args.l2_backend == "l2b":
        print("\nLoading L2b model...")
        model_dir = Path(args.l2b_model_dir).resolve()

        # Auto-detect model type
        model = None
        model_name = "unknown"
        for name, path in [("xgb", "xgb_model.pkl"), ("rf", "rf_model.pkl"), ("lr", "lr_model.pkl")]:
            model_path = model_dir / path
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                model_name = name
                break
        if model is None:
            print("ERROR: no model found in", model_dir, file=sys.stderr)
            return 1
        print(f"  Loaded {model_name} model")

        # Load feature config
        feature_config_path = model_dir / "feature_config.json"
        if feature_config_path.exists():
            feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
            families = set(feature_config["families"])
        else:
            families = {"harness", "structural", "tfidf"}
            print("  WARNING: no feature_config.json, assuming all families")

        # Load TF-IDF if needed
        if "tfidf" in families:
            tfidf_path = model_dir / "tfidf.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, "rb") as f:
                    tfidf = pickle.load(f)
            else:
                print("  WARNING: tfidf family requested but tfidf.pkl missing")
                families.discard("tfidf")

        # Import extract_features from train_l2b
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from train_l2b import extract_features
        extract_fn = extract_features
        rf_model = model
    else:
        print(f"\nLoading TinyBERT ONNX from {args.tinybert_dir}...")
        onnx_model = TinyBertONNXModel(
            Path(args.tinybert_dir).resolve(),
            max_length=args.tinybert_max_length,
            batch_size=args.tinybert_batch_size,
            threshold=args.tinybert_threshold,
        )
        extract_fn = tinybert_extract_fn
        print(f"  threshold={args.tinybert_threshold} "
              f"max_length={args.tinybert_max_length} "
              f"batch_size={args.tinybert_batch_size}")

        # Pre-score every sample that could possibly be routed, ONCE.
        # In cascade mode: only samples inside the widest routing window matter.
        # In standalone mode: L2 decides every sample, so score them all.
        import time
        sweep_t_allow_min = -1.0
        sweep_t_block_max = 3.0
        if args.standalone:
            routable = samples
        else:
            routable = [
                s for s in samples
                if sweep_t_allow_min <= s["raw_score"] < sweep_t_block_max
            ]
        print(f"  Pre-scoring {len(routable)}/{len(samples)} routable samples "
              f"(raw_score in [{sweep_t_allow_min}, {sweep_t_block_max}))...")
        t0 = time.time()
        precomputed = onnx_model._probs(routable) if routable else np.zeros(0, dtype=np.float32)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s "
              f"({dt/max(len(routable),1)*1000:.1f}ms/sample)")

        prob_by_id = {s["case_id"]: float(precomputed[i]) for i, s in enumerate(routable)}

        class _CachedTinyBert:
            """Lookup-only L2 scorer backed by precomputed TinyBERT probabilities."""

            def __init__(self, probs: dict[str, float], threshold: float) -> None:
                self._probs = probs
                self._threshold = threshold

            def _lookup(self, batch: list[dict]) -> np.ndarray:
                return np.array(
                    [self._probs.get(s["case_id"], 0.0) for s in batch],
                    dtype=np.float32,
                )

            def predict_proba(self, batch: list[dict]) -> np.ndarray:
                p = self._lookup(batch)
                return np.column_stack([1.0 - p, p])

            def predict(self, batch: list[dict]) -> np.ndarray:
                return (self._lookup(batch) >= self._threshold).astype(int)

        rf_model = _CachedTinyBert(prob_by_id, args.tinybert_threshold)

    # L1 standalone baseline
    print("\n=== L1 Standalone (Phase 4 result) ===")
    l1_baseline = compute_metrics([
        {**s, "cascade_decision": "l1_only",
         "cascade_pred": "malicious" if s["l1_actual"] == "blocked" else "benign"}
        for s in samples
    ], label="L1 standalone")
    print(f"  F1={l1_baseline['f1']:.4f}  P={l1_baseline['precision']:.4f}  "
          f"R={l1_baseline['recall']:.4f}  FPR={l1_baseline['fpr']:.4f}")

    # --- Standalone L2 mode: skip the sweep, evaluate L2 on every sample. ---
    if args.standalone:
        print("\n=== L2 Standalone (L1 bypassed; every sample scored by L2) ===")
        standalone = simulate_cascade(
            samples, rf_model, tfidf,
            t_allow=float("-inf"), t_block=float("inf"),
            families=families, extract_fn=extract_fn,
        )
        m = compute_metrics(standalone, label="L2 standalone")
        print(f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  FPR={m['fpr']:.4f}")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
        print(f"  vs L1: d_F1={m['f1']-l1_baseline['f1']:+.4f}  "
              f"d_FPR={m['fpr']-l1_baseline['fpr']:+.4f}")

        print("\n=== Per-Source Breakdown (L2 standalone) ===")
        by_source: dict[str, list[dict]] = defaultdict(list)
        for r in standalone:
            by_source[r["source"]].append(r)

        print(f"{'Source':<40} {'N':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'FPR':>6} {'Recall':>6}")
        print("-" * 85)
        for source in sorted(by_source.keys()):
            rows = by_source[source]
            sm = compute_metrics(rows, label=source)
            fpr_str = f"{sm['fpr']:.2f}" if (sm['tn'] + sm['fp']) > 0 else "n/a"
            rec_str = f"{sm['recall']:.2f}" if (sm['tp'] + sm['fn']) > 0 else "n/a"
            print(f"{source:<40} {sm['n']:>5} {sm['tp']:>4} {sm['fp']:>4} "
                  f"{sm['fn']:>4} {sm['tn']:>4} {fpr_str:>6} {rec_str:>6}")

        (output_dir / "l2_standalone.json").write_text(
            json.dumps(m, indent=2), encoding="utf-8"
        )
        (output_dir / "l1_baseline.json").write_text(
            json.dumps(l1_baseline, indent=2), encoding="utf-8"
        )
        print(f"\nResults saved to {output_dir}")
        return 0

    # Sweep routing thresholds
    print("\n=== Cascade Threshold Sweep ===")
    print(f"{'t_allow':>8} {'t_block':>8} | {'F1':>6} {'P':>6} {'R':>6} "
          f"{'FPR':>6} {'Route%':>7} | {'d_F1':>6} {'d_FPR':>7}")
    print("-" * 80)

    sweep_results = []
    for t_allow in [-1.0, -0.5, -0.3, 0.0]:
        for t_block in [0.5, 1.0, 1.5, 2.0, 3.0]:
            if t_allow >= t_block:
                continue

            cascade = simulate_cascade(samples, rf_model, tfidf, t_allow, t_block, families=families, extract_fn=extract_fn)
            m = compute_metrics(cascade, label=f"t_allow={t_allow},t_block={t_block}")

            delta_f1 = m["f1"] - l1_baseline["f1"]
            delta_fpr = m["fpr"] - l1_baseline["fpr"]

            print(f"{t_allow:>8.1f} {t_block:>8.1f} | {m['f1']:>6.4f} {m['precision']:>6.4f} "
                  f"{m['recall']:>6.4f} {m['fpr']:>6.4f} {m['route_rate']*100:>6.1f}% | "
                  f"{delta_f1:>+6.4f} {delta_fpr:>+7.4f}")

            sweep_results.append({
                "t_allow": t_allow,
                "t_block": t_block,
                **m,
                "delta_f1": delta_f1,
                "delta_fpr": delta_fpr,
            })

    # Find best operating point (maximize F1 with FPR <= L1 baseline)
    viable = [r for r in sweep_results if r["recall"] >= 0.75]
    if viable:
        best = max(viable, key=lambda r: r["f1"])
    else:
        best = max(sweep_results, key=lambda r: r["f1"])

    print(f"\n=== Best Operating Point ===")
    print(f"  t_allow={best['t_allow']}, t_block={best['t_block']}")
    print(f"  F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    print(f"  FPR={best['fpr']:.4f}  Route rate={best['route_rate']*100:.1f}%")
    print(f"  vs L1: d_F1={best['delta_f1']:+.4f}  d_FPR={best['delta_fpr']:+.4f}")

    # Per-source breakdown at best operating point
    print(f"\n=== Per-Source Breakdown (best point) ===")
    cascade_best = simulate_cascade(
        samples, rf_model, tfidf, best["t_allow"], best["t_block"],
        families=families, extract_fn=extract_fn,
    )

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in cascade_best:
        by_source[r["source"]].append(r)

    print(f"{'Source':<40} {'N':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'FPR':>6} {'Recall':>6} {'Routed':>6}")
    print("-" * 95)

    for source in sorted(by_source.keys()):
        rows = by_source[source]
        sm = compute_metrics(rows, label=source)
        n_routed = sum(1 for r in rows if r["cascade_decision"].startswith("l2b_"))
        fpr_str = f"{sm['fpr']:.2f}" if (sm['tn'] + sm['fp']) > 0 else "n/a"
        rec_str = f"{sm['recall']:.2f}" if (sm['tp'] + sm['fn']) > 0 else "n/a"
        print(f"{source:<40} {sm['n']:>5} {sm['tp']:>4} {sm['fp']:>4} "
              f"{sm['fn']:>4} {sm['tn']:>4} {fpr_str:>6} {rec_str:>6} {n_routed:>6}")

    # Save results
    (output_dir / "sweep_results.json").write_text(
        json.dumps(sweep_results, indent=2), encoding="utf-8"
    )
    (output_dir / "best_operating_point.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    (output_dir / "l1_baseline.json").write_text(
        json.dumps(l1_baseline, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
