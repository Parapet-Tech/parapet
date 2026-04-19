"""Pass 1: benchmark off-the-shelf PI classifiers on tough_{attack,neutral}_v2.

Loads both challenge yamls, runs a roster of Hugging Face classifiers, writes
per-sample scores to `parapet-runner/runs/l2_pass1_ots/results/<slug>.jsonl`
for downstream aggregation.

Design:
- Inference only, no training.
- Label conventions differ across models. We read `model.config.id2label` at
  load time and resolve which class index corresponds to the positive
  (injection / jailbreak / malicious) signal.
- Llama-Prompt-Guard-2 emits 3 classes (BENIGN / INJECTION / JAILBREAK). We
  combine INJECTION + JAILBREAK probabilities as the positive score.
- Source parsed from the `description` field ("source=XXX ..."), matching the
  convention in `scripts/eval_l2_challenge.py`.

Run:
    cd parapet
    python scripts/eval_pass1_ots_classifiers.py \
        --out parapet-runner/runs/l2_pass1_ots \
        [--models <slug> ...] [--limit N] [--batch-size 32] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


# Roster is intentionally small and focused. Each entry declares the HF slug
# and a hint about how to interpret outputs. Everything else is resolved at
# load time from model.config.
@dataclass(frozen=True)
class ModelSpec:
    slug: str
    # "binary" => single positive class; "multiclass_pg2" => combine classes
    # whose label name contains "inject" or "jailbreak"; "binary_jailbreak"
    # => positive class label contains "jailbreak".
    output_kind: str
    gated: bool = False
    notes: str = ""
    # "hf" => AutoModelForSequenceClassification; "onnx" => local ONNX dir.
    backend: str = "hf"
    local_path: str | None = None  # absolute path for ONNX backend
    # For ONNX models that lack id2label; defaults to conventional index 1.
    positive_index: int = 1


ROSTER: list[ModelSpec] = [
    ModelSpec("meta-llama/Llama-Prompt-Guard-2-22M", "multiclass_pg2", gated=True,
              notes="Meta PG2, current L2a baseline."),
    ModelSpec("meta-llama/Llama-Prompt-Guard-2-86M", "multiclass_pg2", gated=True,
              notes="Meta PG2 large. Confirmed only marginal over 22M previously."),
    ModelSpec("protectai/deberta-v3-base-prompt-injection-v2", "binary",
              notes="Popular DeBERTa-v3-base PI classifier."),
    ModelSpec("deepset/deberta-v3-base-injection", "binary",
              notes="Older deepset DeBERTa-v3 injection classifier."),
    ModelSpec("fmops/distilbert-prompt-injection", "binary",
              notes="Smallest/fastest PI classifier; latency anchor."),
    ModelSpec("jackhhao/jailbreak-classifier", "binary_jailbreak",
              notes="DistilBERT trained on jailbreak_classification corpus."),
    ModelSpec("vibhorag101/jailbreak-classifier", "binary_jailbreak",
              notes="RoBERTa-base trained on jailbreaks."),
]


# ---------- Data ----------

SOURCE_RE = re.compile(r"source=(\S+)")


def parse_source(description: str) -> str:
    m = SOURCE_RE.search(description or "")
    return m.group(1) if m else "unknown"


def load_challenges(root: Path) -> list[dict]:
    """Load tough_attack_v2 + tough_neutral_v2 into a unified, INTERLEAVED list.

    Interleaving matters because callers frequently use `--limit N` for smoke
    tests; sequential concatenation would return N attacks and zero benigns
    (or vice versa) for small N. Interleaving makes small samples stratified.

    Each record is {id, source, true_label ('malicious'|'benign'), content}.
    """
    def _load(path: Path, assumed_label: str) -> list[dict]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        out = []
        for r in data:
            out.append({
                "id": r["id"],
                "source": parse_source(r.get("description", "")),
                "true_label": r.get("label", assumed_label),
                "content": r.get("content", ""),
            })
        return out

    attacks = _load(
        root / "schema/eval/challenges/tough_attack_v2/tough_attack_v6_novel.yaml",
        "malicious",
    )
    benigns = _load(
        root / "schema/eval/challenges/tough_neutral_v2/tough_neutral_v6_novel.yaml",
        "benign",
    )

    interleaved: list[dict] = []
    n = max(len(attacks), len(benigns))
    for i in range(n):
        if i < len(attacks):
            interleaved.append(attacks[i])
        if i < len(benigns):
            interleaved.append(benigns[i])
    return interleaved


# ---------- Inference ----------

def resolve_positive_indices(id2label: dict[int, str], kind: str) -> list[int]:
    """Return indices whose label names indicate a positive (malicious) class.

    id2label arrives from transformers as str keys; normalize to int.
    """
    normalized = {int(k): str(v).lower() for k, v in id2label.items()}

    if kind == "multiclass_pg2":
        positive = [i for i, name in normalized.items()
                    if "inject" in name or "jailbreak" in name]
    elif kind == "binary_jailbreak":
        positive = [i for i, name in normalized.items() if "jailbreak" in name]
    else:  # binary
        # Accept "injection", "unsafe", "malicious", "label_1", etc.
        positive = [i for i, name in normalized.items()
                    if any(tok in name for tok in ("inject", "unsafe", "malicious", "attack"))]
        if not positive:
            # Fallback: assume index 1 is positive (common binary convention).
            positive = [i for i in normalized if i == 1]

    if not positive:
        raise RuntimeError(f"Could not resolve positive class in id2label={id2label}")
    return positive


def run_model(spec: ModelSpec, rows: Sequence[dict], *,
              batch_size: int, device: str,
              hf_token: str | None) -> tuple[list[dict], dict]:
    """Run a single model over all rows. Returns (records, meta)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[{spec.slug}] loading...", flush=True)
    auth = {"token": hf_token} if spec.gated and hf_token else {}
    try:
        tok = AutoTokenizer.from_pretrained(spec.slug, **auth)
        model = AutoModelForSequenceClassification.from_pretrained(spec.slug, **auth)
    except Exception as e:
        return [], {"error": f"load_failed: {type(e).__name__}: {e}"}

    model = model.to(device).eval()
    positive_idx = resolve_positive_indices(model.config.id2label, spec.output_kind)

    records: list[dict] = []
    latencies_ms: list[float] = []

    texts = [r["content"] for r in rows]
    # Warm-up to stabilize latency numbers.
    with torch.inference_mode():
        _ = model(**tok(["warmup"], return_tensors="pt", truncation=True, max_length=512).to(device))

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]

        t0 = time.perf_counter()
        enc = tok(batch_texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pos = probs[:, positive_idx].sum(dim=-1)
        scores = pos.detach().cpu().tolist()
        elapsed = (time.perf_counter() - t0) * 1000.0

        per_sample_ms = elapsed / max(1, len(batch_rows))
        latencies_ms.extend([per_sample_ms] * len(batch_rows))

        for r, s in zip(batch_rows, scores):
            records.append({
                "id": r["id"],
                "source": r["source"],
                "true_label": r["true_label"],
                "score": float(s),
                "latency_ms": per_sample_ms,
            })

        if (start // batch_size) % 10 == 0:
            print(f"[{spec.slug}] {start + len(batch_rows)}/{len(rows)} "
                  f"(batch={len(batch_rows)}, {per_sample_ms:.1f} ms/sample)",
                  flush=True)

    meta = {
        "slug": spec.slug,
        "output_kind": spec.output_kind,
        "id2label": dict(model.config.id2label),
        "positive_indices": positive_idx,
        "device": device,
        "batch_size": batch_size,
        "n_rows": len(records),
        "latency_ms": {
            "mean": _mean(latencies_ms),
            "p50": _pct(latencies_ms, 0.5),
            "p95": _pct(latencies_ms, 0.95),
            "p99": _pct(latencies_ms, 0.99),
        },
    }

    # Free GPU memory before next model.
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return records, meta


def _pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = int(p * (len(xs_sorted) - 1))
    return float(xs_sorted[k])


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# ---------- ONNX support ----------

def _ensure_onnx_dir(path: Path) -> Path:
    """Accept an ONNX model directory OR a .zip; return the directory.

    Zip files are extracted once to a sibling `_extracted_<stem>/` directory
    and cached on subsequent runs.
    """
    if path.is_dir():
        return path
    if path.suffix.lower() == ".zip":
        import zipfile
        extract_root = path.parent / f"_extracted_{path.stem}"
        if not extract_root.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(extract_root)
        # Walk for the actual dir containing model.onnx.
        for cand in extract_root.rglob("model.onnx"):
            return cand.parent
        raise FileNotFoundError(f"No model.onnx inside extracted {path}")
    raise ValueError(f"Expected directory or .zip, got: {path}")


def run_onnx_model(spec: ModelSpec, rows: Sequence[dict], *,
                   batch_size: int, device: str) -> tuple[list[dict], dict]:
    """Run a locally-stored ONNX classifier. Mirrors run_model's contract."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = Path(spec.local_path).resolve()
    print(f"[{spec.slug}] loading ONNX from {model_dir}...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as e:
        return [], {"error": f"tokenizer_load_failed: {type(e).__name__}: {e}"}

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.startswith("cuda")
                 else ["CPUExecutionProvider"])
    try:
        sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=providers)
    except Exception as e:
        return [], {"error": f"onnx_load_failed: {type(e).__name__}: {e}"}

    expected_inputs = {i.name for i in sess.get_inputs()}

    def tok_to_feed(texts: list[str]) -> dict:
        enc = tok(texts, return_tensors="np", padding=True,
                  truncation=True, max_length=512)
        # Pass only what the ONNX graph actually wants; some exports omit
        # token_type_ids, some require it.
        feed = {k: v for k, v in enc.items() if k in expected_inputs}
        # If the graph expects token_type_ids but tokenizer didn't emit them
        # (e.g., RoBERTa-family), fill with zeros.
        if "token_type_ids" in expected_inputs and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"])
        return feed

    # Warm-up.
    sess.run(None, tok_to_feed(["warmup"]))

    records: list[dict] = []
    latencies_ms: list[float] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        texts = [r["content"] for r in batch_rows]

        t0 = time.perf_counter()
        logits = sess.run(None, tok_to_feed(texts))[0]
        # Numerically stable softmax.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        scores = probs[:, spec.positive_index].tolist()
        elapsed = (time.perf_counter() - t0) * 1000.0

        per_sample_ms = elapsed / max(1, len(batch_rows))
        latencies_ms.extend([per_sample_ms] * len(batch_rows))

        for r, s in zip(batch_rows, scores):
            records.append({
                "id": r["id"],
                "source": r["source"],
                "true_label": r["true_label"],
                "score": float(s),
                "latency_ms": per_sample_ms,
            })

        if (start // batch_size) % 10 == 0:
            print(f"[{spec.slug}] {start + len(batch_rows)}/{len(rows)} "
                  f"(batch={len(batch_rows)}, {per_sample_ms:.1f} ms/sample)",
                  flush=True)

    meta = {
        "slug": spec.slug,
        "backend": "onnx",
        "output_kind": spec.output_kind,
        "positive_index": spec.positive_index,
        "local_path": str(model_dir),
        "device": device,
        "batch_size": batch_size,
        "n_rows": len(records),
        "latency_ms": {
            "mean": _mean(latencies_ms),
            "p50": _pct(latencies_ms, 0.5),
            "p95": _pct(latencies_ms, 0.95),
            "p99": _pct(latencies_ms, 0.99),
        },
    }
    return records, meta


# ---------- Entry ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="parapet-runner/runs/l2_pass1_ots",
                    help="Output directory (relative to parapet/).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Filter to these HF slugs (subset of ROSTER).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap challenge rows for quick smoke tests.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip models whose results file already exists.")
    ap.add_argument("--local-onnx", nargs="*", default=[],
                    help="Paths to local ONNX model dirs or .zip files to include.")
    ap.add_argument("--local-positive-index", type=int, default=1,
                    help="Positive class index for local ONNX models (default 1).")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent  # parapet/
    rows = load_challenges(base)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} challenge rows "
          f"({sum(1 for r in rows if r['true_label']=='malicious')} mal, "
          f"{sum(1 for r in rows if r['true_label']=='benign')} ben)")

    out_dir = base / args.out
    results_dir = out_dir / "results"
    meta_dir = out_dir / "meta"
    results_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Device selection.
    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    hf_token = os.environ.get("HF_TOKEN")

    # Build local ONNX specs from --local-onnx paths.
    extras: list[ModelSpec] = []
    for p in args.local_onnx:
        src_path = Path(p).resolve()
        try:
            model_dir = _ensure_onnx_dir(src_path)
        except Exception as e:
            print(f"[local-onnx] SKIP {src_path}: {e}")
            continue
        extras.append(ModelSpec(
            slug=f"local/{src_path.stem}",
            output_kind="binary",
            backend="onnx",
            local_path=str(model_dir),
            positive_index=args.local_positive_index,
            notes=f"Local ONNX from {src_path}",
        ))

    selected = ROSTER + extras
    if args.models:
        wanted = set(args.models)
        selected = [m for m in selected if m.slug in wanted]
        missing = wanted - {m.slug for m in selected}
        if missing:
            print(f"WARN: not in roster/extras: {sorted(missing)}")

    for spec in selected:
        slug_safe = spec.slug.replace("/", "__")
        result_path = results_dir / f"{slug_safe}.jsonl"
        meta_path = meta_dir / f"{slug_safe}.json"

        if args.skip_existing and result_path.exists() and meta_path.exists():
            print(f"[{spec.slug}] skip (exists)")
            continue

        if spec.gated and not hf_token:
            print(f"[{spec.slug}] SKIP: gated model, HF_TOKEN not set")
            meta_path.write_text(json.dumps(
                {"slug": spec.slug, "error": "gated_no_token"}, indent=2
            ), encoding="utf-8")
            continue

        if spec.backend == "onnx":
            records, meta = run_onnx_model(spec, rows,
                                           batch_size=args.batch_size,
                                           device=device)
        else:
            records, meta = run_model(spec, rows,
                                      batch_size=args.batch_size,
                                      device=device,
                                      hf_token=hf_token)

        if meta.get("error"):
            print(f"[{spec.slug}] ERROR: {meta['error']}")
            meta_path.write_text(json.dumps({"slug": spec.slug, **meta}, indent=2),
                                 encoding="utf-8")
            continue

        with result_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{spec.slug}] wrote {len(records)} rows -> {result_path.name} "
              f"(p50={meta['latency_ms']['p50']:.1f}ms, p95={meta['latency_ms']['p95']:.1f}ms)")

    print(f"\nDone. Results in {out_dir}")
    print(f"Next: python scripts/pass1_aggregate.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
