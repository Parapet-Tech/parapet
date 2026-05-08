"""Phase 1 truncation-policy spike — frozen MiniLM-L12 + linear head, ranking only.

Compares 5 truncation policies on the Phase 1 v8 residual artifact:

    head_128 / tail_128 / head_tail_64_64 / suspicious_span_128 / full_512

The encoder is frozen sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(matches §0.6.1 production candidate). Per-policy linear head is trained on
fold-id ∈ {0,1,2,3} residuals and scored on fold-id == 4 residuals.
Reports recall at fixed FPR and ROC-AUC for ranking.

Inputs are restricted to runs/phase1_v8_residuals/residuals.jsonl. The
script does NOT load any legacy schema/eval YAML. The spike is
ranking-only and produces no acceptance numbers.

For full_512 the encoder's native max length is 128, so the policy is
implemented as overlapping 128-token chunk encoding (stride 64) with
mean-pooled chunk embeddings — an honest "what if we saw all 512 tokens"
proxy that respects the model's training distribution.

Output (under --output-dir, default runs/phase1_truncation_spike/):
    embeddings/<policy>_train.npy        cached embeddings
    embeddings/<policy>_val.npy
    predictions/<policy>_val.jsonl       per-row scores + meta
    summary.json                         policy ranking
    summary.md                           human-readable
    manifest.json                        provenance + script args + caveats

Usage:
    cd parapet/parapet-runner
    python scripts/phase1_truncation_policy_spike.py            # default 2000-row sample
    python scripts/phase1_truncation_policy_spike.py --full     # all 11,192 rows (slow on CPU)
    python scripts/phase1_truncation_policy_spike.py --policy head_128 --policy tail_128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from parapet_runner import truncation as trunc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TARGET_FPR = 0.10
DEFAULT_VAL_FOLD = 4
DEFAULT_SAMPLE_SIZE = 2000
RANDOM_SEED = 42

CHUNK_SIZE = 128
CHUNK_STRIDE = 64
FULL_512_TOKEN_BUDGET = 512


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_residuals(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stratified_subsample(
    rows: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    keys: tuple[str, ...] = ("label", "language"),
) -> list[dict[str, Any]]:
    """Largest-remainder per-cell sample preserving (label, language) ratios."""
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    cells: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = tuple(str(r.get(k) or "unknown") for k in keys)
        cells[key].append(i)
    total = len(rows)
    raw = {k: len(v) * n / total for k, v in cells.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    deficit = n - sum(alloc.values())
    if deficit > 0:
        ranked = sorted(cells.keys(), key=lambda k: (-(raw[k] - alloc[k]), str(k)))
        for k in ranked[:deficit]:
            alloc[k] += 1
    for k in alloc:
        alloc[k] = min(alloc[k], len(cells[k]))
    sampled: list[int] = []
    for k in sorted(cells.keys(), key=str):
        n_k = alloc.get(k, 0)
        if n_k > 0:
            sampled.extend(rng.sample(cells[k], n_k))
    sampled.sort()
    return [rows[i] for i in sampled]


def split_by_fold(
    rows: list[dict[str, Any]], val_fold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for r in rows:
        if r.get("fold_id") == val_fold:
            val.append(r)
        else:
            train.append(r)
    return train, val


def label_to_int(label: str) -> int:
    return 1 if label == "malicious" else 0


# ---------------------------------------------------------------------------
# Truncated-text construction (offsets-aware)
# ---------------------------------------------------------------------------


def build_truncated_strings(
    text: str,
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    *,
    policy: str,
) -> list[str]:
    """Return the list of strings to encode for this policy.

    For all policies except full_512 we return a single string. For full_512
    we return a list of overlapping 128-token chunks (stride 64) covering
    up to the first 512 tokens, to be encoded separately and mean-pooled.
    """
    if policy == "full_512":
        budget = min(len(token_ids), FULL_512_TOKEN_BUDGET)
        if budget <= CHUNK_SIZE:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < budget:
            end = min(start + CHUNK_SIZE, budget)
            chunk_text = text[offsets[start][0]:offsets[end - 1][1]]
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end >= budget:
                break
            start += CHUNK_STRIDE
        return chunks or [text]

    truncated_ids = trunc.apply_policy(
        policy, token_ids=token_ids, char_offsets=offsets, text=text,
    )
    if not truncated_ids:
        return [text[:1] or " "]  # encoder needs non-empty input
    if policy == "head_128":
        end_char = offsets[len(truncated_ids) - 1][1]
        return [text[:end_char]]
    if policy == "tail_128":
        start_char = offsets[len(token_ids) - len(truncated_ids)][0]
        return [text[start_char:]]
    if policy == "head_tail_64_64":
        head_n = min(64, len(token_ids))
        tail_n = min(64, len(token_ids) - head_n)
        if tail_n <= 0:
            return [text[:offsets[head_n - 1][1]]]
        head_text = text[:offsets[head_n - 1][1]]
        tail_text = text[offsets[len(token_ids) - tail_n][0]:]
        # Single string with separator so the encoder sees a coherent input.
        return [head_text + " ... " + tail_text]
    if policy == "suspicious_span_128":
        # Locate the chosen window's start in the full token list by
        # matching the returned subsequence (token_ids are unique enough
        # in practice; if they're not, this still gets the leftmost match
        # which is fine for ranking).
        n_trunc = len(truncated_ids)
        for s in range(0, len(token_ids) - n_trunc + 1):
            if token_ids[s:s + n_trunc] == truncated_ids:
                return [text[offsets[s][0]:offsets[s + n_trunc - 1][1]]]
        # Defensive fallback if the linear search didn't find a match:
        return [text[:offsets[min(n_trunc, len(offsets)) - 1][1]]]
    raise ValueError(f"unknown policy {policy!r}")


# ---------------------------------------------------------------------------
# Encoder protocol + default sentence-transformers backend
# ---------------------------------------------------------------------------


EncoderFn = Callable[[list[str]], np.ndarray]
"""A callable that encodes texts -> (n, dim) float32 array."""


def make_default_encoder(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    batch_size: int = 32,
) -> EncoderFn:
    """Lazily build a sentence-transformers-backed encoder."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_id)
    model.eval()

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return emb.astype(np.float32)

    return encode


# ---------------------------------------------------------------------------
# Per-row encoding pipeline
# ---------------------------------------------------------------------------


def encode_rows(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    tokenize_fn: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
    encoder: EncoderFn,
    progress_every: int = 500,
) -> np.ndarray:
    """Apply policy + encode -> (n_rows, dim) embedding array.

    For full_512 we encode multiple chunks per row and mean-pool them
    (numpy mean over axis 0). Embedding dim is inferred from the first row.
    """
    embeddings: list[np.ndarray] = []
    for i, r in enumerate(rows):
        text = r.get("content") or ""
        token_ids, offsets = tokenize_fn(text)
        if not token_ids:
            # Empty content: use a placeholder so dimension stays consistent.
            embeddings.append(encoder([" "])[0])
            continue
        chunks = build_truncated_strings(text, token_ids, offsets, policy=policy)
        chunk_emb = encoder(chunks)
        if chunk_emb.shape[0] > 1:
            row_emb = chunk_emb.mean(axis=0)
        else:
            row_emb = chunk_emb[0]
        embeddings.append(row_emb)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{policy}] encoded {i + 1}/{len(rows)} rows", flush=True)
    return np.vstack(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Linear head + metrics
# ---------------------------------------------------------------------------


def fit_linear_head(
    train_emb: np.ndarray, train_labels: np.ndarray, *, seed: int,
) -> Any:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    clf.fit(train_emb, train_labels)
    return clf


def recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float,
) -> tuple[float, float, float]:
    """Pick the threshold giving the largest recall at FPR <= target.

    Returns (recall, achieved_fpr, threshold). Falls back to (0,0,inf)
    when no threshold satisfies the constraint.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, scores)
    eligible = fpr <= target_fpr
    if not eligible.any():
        return 0.0, 0.0, float("inf")
    # Largest TPR among eligible thresholds.
    idx = int(np.argmax(np.where(eligible, tpr, -1)))
    return float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def evaluate_policy(
    val_rows: list[dict[str, Any]],
    val_emb: np.ndarray,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    *,
    target_fpr: float,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    clf = fit_linear_head(train_emb, train_labels, seed=seed)
    scores = clf.predict_proba(val_emb)[:, 1]
    y_true = np.array([label_to_int(r.get("label") or "benign") for r in val_rows])

    if len(np.unique(y_true)) < 2:
        # Degenerate val set (one class only). Return placeholders.
        return {
            "recall_at_fpr": 0.0, "achieved_fpr": 0.0, "threshold": float("inf"),
            "roc_auc": float("nan"), "n_val": int(len(y_true)),
            "n_val_malicious": int(y_true.sum()),
            "per_language": {},
            "scores": scores.tolist(),
        }

    recall, fpr, thr = recall_at_fpr(y_true, scores, target_fpr)
    auc = float(roc_auc_score(y_true, scores))

    # Per-language slice (recall@FPR computed per-lang on the val fold).
    per_lang: dict[str, dict[str, float]] = {}
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(val_rows):
        by_lang[str(r.get("language") or "unknown")].append(i)
    for lang, idxs in by_lang.items():
        idxs_arr = np.array(idxs)
        y_l = y_true[idxs_arr]
        s_l = scores[idxs_arr]
        if len(np.unique(y_l)) < 2:
            per_lang[lang] = {
                "n": int(len(y_l)), "n_malicious": int(y_l.sum()),
                "recall_at_fpr": float("nan"), "roc_auc": float("nan"),
            }
            continue
        r_l, f_l, _ = recall_at_fpr(y_l, s_l, target_fpr)
        per_lang[lang] = {
            "n": int(len(y_l)),
            "n_malicious": int(y_l.sum()),
            "recall_at_fpr": r_l,
            "achieved_fpr": f_l,
            "roc_auc": float(roc_auc_score(y_l, s_l)),
        }

    return {
        "recall_at_fpr": recall,
        "achieved_fpr": fpr,
        "threshold": thr,
        "roc_auc": auc,
        "n_val": int(len(y_true)),
        "n_val_malicious": int(y_true.sum()),
        "per_language": per_lang,
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(output_dir: Path, policy: str, split: str) -> Path:
    return output_dir / "embeddings" / f"{policy}_{split}.npy"


def load_or_encode(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    split: str,
    output_dir: Path,
    tokenize_fn: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
    encoder: EncoderFn,
    force: bool = False,
) -> np.ndarray:
    path = cache_path(output_dir, policy, split)
    if path.exists() and not force:
        emb = np.load(path)
        if emb.shape[0] == len(rows):
            return emb
        print(f"  [{policy}/{split}] cache size mismatch — re-encoding", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{policy}/{split}] encoding {len(rows)} rows ...", flush=True)
    emb = encode_rows(rows, policy=policy, tokenize_fn=tokenize_fn, encoder=encoder)
    np.save(path, emb)
    return emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--residuals", type=Path,
        default=Path("runs/phase1_v8_residuals/residuals.jsonl"),
        help="Phase 1 v8 residuals (sole input). Legacy eval YAMLs are not loadable.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase1_truncation_spike"),
    )
    parser.add_argument(
        "--policy", action="append", default=None,
        choices=trunc.POLICIES,
        help="Policy to evaluate (repeatable). Default: all 5.",
    )
    parser.add_argument(
        "--val-fold", type=int, default=DEFAULT_VAL_FOLD,
    )
    parser.add_argument(
        "--target-fpr", type=float, default=DEFAULT_TARGET_FPR,
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Stratified subsample size. Set to 0 with --full for the entire residual.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on the full residual (slow on CPU).",
    )
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
    )
    parser.add_argument(
        "--force-encode", action="store_true",
        help="Ignore cached embeddings and re-encode.",
    )
    args = parser.parse_args(argv)

    # ---- Hard guard: refuse legacy eval paths ---------------------------
    residuals_path = args.residuals.resolve()
    forbidden = ("schema/eval/l1_holdout", "schema/eval/challenges")
    if any(s in str(residuals_path).replace("\\", "/") for s in forbidden):
        print(
            f"ERROR: --residuals points at a legacy eval path ({residuals_path}). "
            "Refusing to run; truncation spike must consume only "
            "phase1_v8_residuals/residuals.jsonl.",
            file=sys.stderr,
        )
        return 1

    if not residuals_path.exists():
        print(f"ERROR: {residuals_path} not found.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    policies: list[str] = list(args.policy or trunc.POLICIES)

    print(f"Loading residuals from {residuals_path} ...")
    all_rows = load_residuals(residuals_path)
    print(f"  {len(all_rows):,} rows")

    if not args.full and args.sample_size > 0 and args.sample_size < len(all_rows):
        rows = stratified_subsample(all_rows, n=args.sample_size, seed=args.seed)
        print(f"  Subsampled to {len(rows):,} rows (stratified by label x language)")
    else:
        rows = all_rows
        if args.full:
            print("  --full: using complete residual")

    train_rows, val_rows = split_by_fold(rows, args.val_fold)
    print(f"  Train (fold != {args.val_fold}): {len(train_rows):,} rows")
    print(f"  Val   (fold == {args.val_fold}): {len(val_rows):,} rows")

    train_label_counts = Counter(r.get("label") for r in train_rows)
    val_label_counts = Counter(r.get("label") for r in val_rows)
    print(f"  Train labels: {dict(train_label_counts)}")
    print(f"  Val labels:   {dict(val_label_counts)}")

    if not val_rows or len(val_label_counts) < 2:
        print(
            "ERROR: val fold has insufficient rows / single-class label distribution.",
            file=sys.stderr,
        )
        return 1

    # ---- Tokenizer + encoder --------------------------------------------
    print(f"Loading model {args.model_id} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def tokenize_fn(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        out = tokenizer(
            text or " ",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = [(int(s), int(e)) for s, e in out["offset_mapping"]]
        return list(map(int, out["input_ids"])), offsets

    encoder = make_default_encoder(args.model_id)

    # ---- Per-policy encode + evaluate -----------------------------------
    train_labels = np.array([label_to_int(r.get("label") or "benign") for r in train_rows])
    results: dict[str, Any] = {}

    for policy in policies:
        print(f"\n=== Policy: {policy} ===")
        train_emb = load_or_encode(
            train_rows, policy=policy, split="train",
            output_dir=output_dir, tokenize_fn=tokenize_fn, encoder=encoder,
            force=args.force_encode,
        )
        val_emb = load_or_encode(
            val_rows, policy=policy, split="val",
            output_dir=output_dir, tokenize_fn=tokenize_fn, encoder=encoder,
            force=args.force_encode,
        )
        print(f"  Fitting linear head ...")
        result = evaluate_policy(
            val_rows, val_emb, train_emb, train_labels,
            target_fpr=args.target_fpr, seed=args.seed,
        )
        results[policy] = result
        print(
            f"  recall@FPR={args.target_fpr:.2f}: {result['recall_at_fpr']:.3f} "
            f"(achieved FPR={result['achieved_fpr']:.3f})  "
            f"AUC={result['roc_auc']:.3f}"
        )

        # Persist per-row predictions for reproducibility.
        pred_path = output_dir / "predictions" / f"{policy}_val.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for r, s in zip(val_rows, result["scores"]):
                f.write(json.dumps({
                    "content_hash": r.get("content_hash"),
                    "label": r.get("label"),
                    "language": r.get("language"),
                    "reason": r.get("reason"),
                    "fold_id": r.get("fold_id"),
                    "score": float(s),
                }, ensure_ascii=False) + "\n")

    # ---- Ranking + outputs -----------------------------------------------
    ranked = sorted(
        results.items(),
        key=lambda kv: (-kv[1]["recall_at_fpr"], -kv[1]["roc_auc"]),
    )
    summary = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "model_id": args.model_id,
        "target_fpr": args.target_fpr,
        "val_fold": args.val_fold,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "ranking": [
            {
                "rank": i + 1,
                "policy": policy,
                "recall_at_fpr": r["recall_at_fpr"],
                "achieved_fpr": r["achieved_fpr"],
                "roc_auc": r["roc_auc"],
                "n_val_malicious": r["n_val_malicious"],
            }
            for i, (policy, r) in enumerate(ranked)
        ],
        "per_policy": {
            policy: {k: v for k, v in r.items() if k != "scores"}
            for policy, r in results.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_render_md(summary), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "spike_scope": "ranking only — NOT acceptance evidence",
        "script_args": {
            "residuals": str(args.residuals),
            "output_dir": str(args.output_dir),
            "policies": policies,
            "val_fold": args.val_fold,
            "target_fpr": args.target_fpr,
            "sample_size": args.sample_size,
            "full": args.full,
            "model_id": args.model_id,
            "seed": args.seed,
        },
        "inputs": {
            "residuals": {
                "path": str(residuals_path),
                "sha256": sha256_file(residuals_path),
                "n_rows_total": len(all_rows),
                "n_rows_used": len(rows),
            },
        },
        "caveats": [
            "Encoder is frozen sentence-transformers MiniLM-L12-v2 with native "
            "max_seq_len=128. The full_512 policy is implemented as overlapping "
            "128-token chunk encoding (stride 64) with mean-pooled embeddings — "
            "an honest 'what if we saw all 512 tokens' proxy, not literal 512-token "
            "attention.",
            "Linear head is sklearn LogisticRegression class_weight='balanced'. Per "
            "direction.md §0.2, this is the cheapest faithful proxy for class "
            "imbalance; not the recipe a final L2 will use.",
            "suspicious_span_128 uses a small EN-heavy attack-trigger lexicon. If it "
            "underperforms on non-EN, that itself is a useful signal — naive English "
            "keyword spotting cannot drive multilingual routing.",
            "This artifact is for policy ranking only. Per direction.md §3 / §6, "
            "no acceptance claims may be made against this run.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Ranking ===")
    for entry in summary["ranking"]:
        print(
            f"  {entry['rank']}. {entry['policy']:<22} "
            f"recall@FPR={entry['recall_at_fpr']:.3f}  AUC={entry['roc_auc']:.3f}"
        )
    print(f"\nManifest: {output_dir / 'manifest.json'}")
    print(f"Summary:  {output_dir / 'summary.md'}")
    return 0


def _render_md(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# Phase 1 truncation-policy spike (ranking only)", ""]
    lines.append(f"Created: {summary['created_utc']}")
    lines.append(f"Model: {summary['model_id']}")
    lines.append(
        f"Target FPR: {summary['target_fpr']} | val fold: {summary['val_fold']} | "
        f"n_train={summary['n_train']:,}, n_val={summary['n_val']:,}"
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("| rank | policy | recall@FPR | achieved FPR | ROC-AUC |")
    lines.append("|------|--------|------------|--------------|---------|")
    for entry in summary["ranking"]:
        lines.append(
            f"| {entry['rank']} | {entry['policy']} | "
            f"{entry['recall_at_fpr']:.3f} | {entry['achieved_fpr']:.3f} | "
            f"{entry['roc_auc']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-language detail")
    for policy, r in summary["per_policy"].items():
        lines.append(f"### {policy}")
        lines.append(
            f"recall@FPR={r['recall_at_fpr']:.3f}, AUC={r['roc_auc']:.3f}"
        )
        lines.append("")
        lines.append("| language | n | n_mal | recall@FPR | AUC |")
        lines.append("|----------|---|-------|------------|-----|")
        for lang, m in sorted(r["per_language"].items()):
            lines.append(
                f"| {lang} | {m['n']} | {m['n_malicious']} | "
                f"{m.get('recall_at_fpr', float('nan')):.3f} | "
                f"{m.get('roc_auc', float('nan')):.3f} |"
            )
        lines.append("")
    lines.append("## Decision rules (from direction.md)")
    lines.append("")
    lines.append("- head_128 within ~1-2 recall points of full_512 → use head_128.")
    lines.append("- head_tail_64_64 recovers suffix attacks at similar latency → "
                 "prefer over head-only.")
    lines.append("- full_512 materially beats all 128-token policies → MiniLM is "
                 "slow-lane only, not default hot-path L2.")
    lines.append("- suspicious_span_128 beats both → cheap mechanical scan selects "
                 "what semantic L2 sees; this is the Parapet-specific shape.")
    lines.append("")
    lines.append("This is ranking only. No acceptance numbers may be claimed from "
                 "this artifact.")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
