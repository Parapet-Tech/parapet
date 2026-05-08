"""Phase 1 v8 residual export — direction.md §0.1 schema-conformant L2 training input.

Re-exports the existing 5-fold OOF run at runs/l1b_residuals_v8/ into the
canonical §0.1 residual schema, plus a per-axis distribution report (Phase
1 §2). Optional canonical-eval gate (Phase 1 §3) only fires when the
caller passes one or more --canonical-eval paths.

No re-training. Reads only on-disk fold artifacts.

Per-fold sourcing:
    - run/_eval_holdout_*/eval.json              verdicts, scores, case_ids
    - run/_eval_holdout_*/dataset/holdout.yaml   case_id -> row index pairing
    - run/_eval_holdout_*/eval_config_l1_threshold.yaml  applied L1 threshold
    - curated/holdout.yaml                       row metadata (lang, reason, source, ...)

The script ignores the on-disk fold_assignments.json file. The existing
runner emits it as a 10-row preview ("... (N total)" suffix), not a
machine-readable artifact. fold_id is derived from the enclosing fold_*
dir name instead.

Canonical-eval gate (--canonical-eval):
    Pass one or more paths to the active L2 acceptance eval set. Overlap
    of any v8 train content_hash with these targets is a hard fail (rc=2
    unless --allow-leakage). With no --canonical-eval provided, no gate
    runs and the manifest records gate_status = "no_canonical_eval_defined".
    The legacy schema/eval/{l1_holdout,challenges/...} YAMLs predate the
    JSONL curation pipeline and SHOULD NOT be passed here without
    regeneration; for diagnosing those, use audit_legacy_eval_leakage.py.

Output (under --output-dir):
    residuals.jsonl              FN U FP U near-boundary  (L2 training input)
    baseline_correct.jsonl       sidecar, NOT training input
    distribution_report.json     Phase 1 §2
    distribution_report.md
    leakage_check.json           Phase 1 §3 — only when --canonical-eval given
    manifest.json                provenance + fold thresholds + counts + usability

Usage:
    cd parapet/parapet-runner
    python scripts/phase1_v8_residual_export.py \
        --source-run-dir runs/l1b_residuals_v8 \
        --curation-dir ../parapet-data/curated/v8_experiment \
        --output-dir runs/phase1_v8_residuals
        # add --canonical-eval <path> when an active eval set exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_REASONS: frozenset[str] = frozenset({
    "instruction_override",
    "roleplay_jailbreak",
    "meta_probe",
    "exfiltration",
    "adversarial_suffix",
    "indirect_injection",
    "obfuscation",
    "constraint_bypass",
    "uncategorized",
})

# Near-boundary uses raw_score (the SVM decision_function value), NOT the
# post-pipeline `score` we expose as l1_score. This matches the existing
# scripts/export_l1_residuals.py:391 selection so the residual cut is
# comparable across runs. Recorded in manifest.json for transparency.
DEFAULT_MARGIN_BAND_RAW_SCORE = 1.0
DEFAULT_BASELINE_SAMPLE_RATE = 0.10

FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")
EVAL_HOLDOUT_DIR_RE = re.compile(r"^_eval_holdout_")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml_list(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected YAML list, got {type(raw).__name__}")
    return raw


# ---------------------------------------------------------------------------
# Fold discovery
# ---------------------------------------------------------------------------


def discover_fold_dirs(source_run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(fold_id, fold_dir), ...] sorted by fold_id ascending."""
    found: list[tuple[int, Path]] = []
    for child in source_run_dir.iterdir():
        if not child.is_dir():
            continue
        m = FOLD_DIR_RE.match(child.name)
        if not m:
            continue
        found.append((int(m.group(1)), child))
    found.sort(key=lambda pair: pair[0])
    if not found:
        raise FileNotFoundError(f"{source_run_dir}: no fold_N/ subdirectories")
    return found


def locate_eval_holdout_dir(fold_dir: Path) -> Path:
    """Find the single _eval_holdout_*/ inside fold_N/run/. Asserts exactly one."""
    run_dir = fold_dir / "run"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{fold_dir}: missing run/ subdir")
    candidates = [
        d for d in run_dir.iterdir()
        if d.is_dir() and EVAL_HOLDOUT_DIR_RE.match(d.name)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{run_dir}: expected exactly one _eval_holdout_*/ dir, "
            f"found {len(candidates)}: {[d.name for d in candidates]}"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Per-fold load + join
# ---------------------------------------------------------------------------


def load_fold_threshold(eval_holdout_dir: Path) -> tuple[float, Path]:
    """Read layers.L1.threshold from eval_config_l1_threshold.yaml."""
    cfg_path = eval_holdout_dir / "eval_config_l1_threshold.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        threshold = cfg["layers"]["L1"]["threshold"]
    except (TypeError, KeyError) as e:
        raise ValueError(f"{cfg_path}: missing layers.L1.threshold ({e})")
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"{cfg_path}: threshold must be numeric, got {type(threshold).__name__}")
    return float(threshold), cfg_path


def load_eval_results_by_index(eval_json_path: Path, expected_n: int) -> dict[int, dict[str, Any]]:
    """Index eval.json results by 0-based row index (case_id holdout_N -> N-1).

    Mirrors scripts/export_l1_residuals.py:301 case_id parsing rules.
    """
    data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{eval_json_path}: results is not a list")
    if len(results) != expected_n:
        raise ValueError(
            f"{eval_json_path}: results count {len(results)} != expected_n {expected_n}"
        )
    indexed: dict[int, dict[str, Any]] = {}
    for r in results:
        case_id = str(r.get("case_id") or "")
        prefix, _, suffix = case_id.rpartition("_")
        if prefix != "holdout" or not suffix.isdigit():
            raise ValueError(f"{eval_json_path}: unexpected case_id {case_id!r}")
        idx = int(suffix) - 1
        if idx < 0 or idx >= expected_n:
            raise ValueError(
                f"{eval_json_path}: case_id {case_id!r} out of bounds for {expected_n} rows"
            )
        if idx in indexed:
            raise ValueError(f"{eval_json_path}: duplicate case_id {case_id!r}")
        indexed[idx] = r
    if len(indexed) != expected_n:
        raise ValueError(
            f"{eval_json_path}: indexed {len(indexed)} of {expected_n} rows"
        )
    return indexed


def join_fold(fold_id: int, fold_dir: Path) -> tuple[list[dict[str, Any]], float, dict[str, Path]]:
    """Build the joined per-row view for one fold.

    Returns:
        rows: list of dicts with both row metadata and L1 verdict/scores
        threshold: applied L1 threshold for this fold's holdout eval
        sources: dict naming the artifact paths consumed (for manifest)
    """
    eval_holdout_dir = locate_eval_holdout_dir(fold_dir)
    dataset_holdout = load_yaml_list(eval_holdout_dir / "dataset" / "holdout.yaml")
    curated_holdout = load_yaml_list(fold_dir / "curated" / "holdout.yaml")

    if len(dataset_holdout) != len(curated_holdout):
        raise ValueError(
            f"fold_{fold_id}: dataset/holdout.yaml has {len(dataset_holdout)} rows, "
            f"curated/holdout.yaml has {len(curated_holdout)} rows"
        )
    # Full content equality check kills the entire "wrong row metadata" bug
    # class. Cheap (~5k str compares per fold).
    for i, (d, c) in enumerate(zip(dataset_holdout, curated_holdout)):
        if d.get("content") != c.get("content"):
            raise ValueError(
                f"fold_{fold_id}: row {i}: dataset/holdout.yaml content != "
                f"curated/holdout.yaml content"
            )

    eval_json_path = eval_holdout_dir / "eval.json"
    indexed_results = load_eval_results_by_index(eval_json_path, len(dataset_holdout))
    threshold, threshold_path = load_fold_threshold(eval_holdout_dir)

    rows: list[dict[str, Any]] = []
    for idx, curated in enumerate(curated_holdout):
        result = indexed_results[idx]
        l1_signals = result.get("l1_signals") or []
        l1 = l1_signals[0] if isinstance(l1_signals, list) and l1_signals else {}

        expected = result.get("expected")
        actual = result.get("actual")
        true_label = "malicious" if expected == "blocked" else "benign"
        pred_label = "malicious" if actual == "blocked" else "benign"
        if true_label == "malicious" and pred_label == "benign":
            error_type = "false_negative"
        elif true_label == "benign" and pred_label == "malicious":
            error_type = "false_positive"
        else:
            error_type = "correct"

        rows.append({
            # row metadata (curated/holdout.yaml)
            "content": curated.get("content"),
            "label": true_label,
            "language": curated.get("language"),
            "reason_raw": curated.get("reason"),
            "source": curated.get("source"),
            "format_bin": curated.get("format_bin"),
            "length_bin": curated.get("length_bin"),
            # L1 verdict (eval.json)
            "pred_label": pred_label,
            "correct": bool(result.get("correct")),
            "error_type": error_type,
            # L1 signals (eval.json -> l1_signals[0])
            "score": l1.get("score"),
            "raw_score": l1.get("raw_score"),
            "raw_unquoted_score": l1.get("raw_unquoted_score"),
            "raw_squash_score": l1.get("raw_squash_score"),
            "raw_score_delta": l1.get("raw_score_delta"),
            "unquoted_score": l1.get("unquoted_score"),
            "squash_score": l1.get("squash_score"),
            "quote_detected": l1.get("quote_detected"),
            "fold_id": fold_id,
            "applied_threshold": threshold,
        })

    sources = {
        "eval_json": eval_json_path,
        "dataset_holdout": eval_holdout_dir / "dataset" / "holdout.yaml",
        "curated_holdout": fold_dir / "curated" / "holdout.yaml",
        "eval_config_l1_threshold": threshold_path,
    }
    return rows, threshold, sources


# ---------------------------------------------------------------------------
# §0.1 schema mapping + reason normalization
# ---------------------------------------------------------------------------


def normalize_reason(value: Any) -> tuple[str, bool]:
    """Return (canonical_reason, was_normalized).

    Pass-through if value is in CANONICAL_REASONS. Anything else (None, "",
    non-canonical strings) becomes "uncategorized" and was_normalized=True.
    """
    if isinstance(value, str) and value in CANONICAL_REASONS:
        return value, False
    return "uncategorized", True


def to_phase1_row(joined: dict[str, Any]) -> dict[str, Any]:
    """Map a joined per-fold row to the §0.1 canonical residual schema.

    Training input fields:  content, label, language, reason, source, content_hash
    Metadata fields:        l1_score, l1_decision, l1_thresholds, routed_reason,
                            fold_id, format_bin, length_bin, error_type
    Diagnostic fields:      raw_score and friends (kept for residual cut + analysis)
    """
    reason, _was_norm = normalize_reason(joined.get("reason_raw"))
    content = joined.get("content") or ""
    pred_label = joined.get("pred_label")
    return {
        # training input
        "content": content,
        "label": joined.get("label"),
        "language": joined.get("language"),
        "reason": reason,
        "source": joined.get("source"),
        "content_hash": content_hash(content),
        # metadata
        "l1_score": joined.get("score"),
        "l1_decision": "block" if pred_label == "malicious" else "allow",
        "l1_thresholds": {"l1": joined.get("applied_threshold")},
        "routed_reason": None,
        "fold_id": joined.get("fold_id"),
        "format_bin": joined.get("format_bin"),
        "length_bin": joined.get("length_bin"),
        "error_type": joined.get("error_type"),
        # diagnostic (carried through from existing residual export)
        "raw_score": joined.get("raw_score"),
        "raw_unquoted_score": joined.get("raw_unquoted_score"),
        "raw_squash_score": joined.get("raw_squash_score"),
        "raw_score_delta": joined.get("raw_score_delta"),
        "unquoted_score": joined.get("unquoted_score"),
        "squash_score": joined.get("squash_score"),
        "quote_detected": joined.get("quote_detected"),
    }


# ---------------------------------------------------------------------------
# Partition (FN, FP, near-boundary, baseline-correct)
# ---------------------------------------------------------------------------


def partition_rows(
    joined_rows: list[dict[str, Any]],
    *,
    margin_band_raw_score: float,
    baseline_sample_rate: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split joined per-fold rows into the 4-bucket residual partition.

    Near-boundary uses raw_score (matches existing script:391), not l1_score.
    Baseline-correct is sampled deterministically (rng).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {
        "false_negative": [],
        "false_positive": [],
        "near_boundary_benign": [],
        "baseline_correct": [],
    }
    for row in joined_rows:
        et = row["error_type"]
        if et == "false_negative":
            buckets["false_negative"].append(row)
        elif et == "false_positive":
            buckets["false_positive"].append(row)
        elif et == "correct":
            raw = row.get("raw_score")
            if (
                row["label"] == "benign"
                and isinstance(raw, (int, float))
                and abs(raw) <= margin_band_raw_score
            ):
                buckets["near_boundary_benign"].append(row)
            elif rng.random() < baseline_sample_rate:
                buckets["baseline_correct"].append(row)
    return buckets


# ---------------------------------------------------------------------------
# Distribution report (Phase 1 §2)
# ---------------------------------------------------------------------------


_AXES = ("language", "reason", "source", "length_bin", "format_bin")


def _axis_share(rows: Iterable[dict[str, Any]], axis: str) -> dict[str, dict[str, Any]]:
    counter: Counter[str] = Counter()
    for r in rows:
        counter[str(r.get(axis) or "unknown")] += 1
    total = sum(counter.values())
    out: dict[str, dict[str, Any]] = {}
    for k, v in counter.most_common():
        out[k] = {"count": v, "share": (v / total) if total else 0.0}
    return out


def _joint_top(
    rows: Iterable[dict[str, Any]], axes: tuple[str, ...], top_n: int
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for r in rows:
        key = tuple(str(r.get(a) or "unknown") for a in axes)
        counter[key] += 1
    total = sum(counter.values())
    return [
        {**dict(zip(axes, key)), "count": count, "share": (count / total) if total else 0.0}
        for key, count in counter.most_common(top_n)
    ]


def build_distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fn = [r for r in rows if r["error_type"] == "false_negative"]
    fp = [r for r in rows if r["error_type"] == "false_positive"]
    return {
        "totals": {
            "all_predictions": len(rows),
            "false_negative": len(fn),
            "false_positive": len(fp),
        },
        "false_negative": {
            **{axis: _axis_share(fn, axis) for axis in _AXES},
            "top_language_x_reason": _joint_top(fn, ("language", "reason"), 20),
            "top_language_x_source": _joint_top(fn, ("language", "source"), 20),
        },
        "false_positive": {
            **{axis: _axis_share(fp, axis) for axis in _AXES},
            "top_language_x_reason": _joint_top(fp, ("language", "reason"), 20),
            "top_language_x_source": _joint_top(fp, ("language", "source"), 20),
        },
    }


def render_distribution_md(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Phase 1 v8 residual distribution report", ""]
    t = report["totals"]
    lines.append(
        f"Total predictions: {t['all_predictions']:,} | "
        f"FN: {t['false_negative']:,} | FP: {t['false_positive']:,}"
    )
    lines.append("")
    for kind in ("false_negative", "false_positive"):
        section = report[kind]
        lines.append(f"## {kind}")
        for axis in _AXES:
            lines.append(f"### by {axis}")
            for k, v in section[axis].items():
                lines.append(f"- {k}: {v['count']:,} ({v['share']:.1%})")
            lines.append("")
        lines.append("### top language × reason")
        for entry in section["top_language_x_reason"]:
            lines.append(
                f"- {entry['language']} / {entry['reason']}: "
                f"{entry['count']:,} ({entry['share']:.1%})"
            )
        lines.append("")
        lines.append("### top language × source")
        for entry in section["top_language_x_source"]:
            lines.append(
                f"- {entry['language']} / {entry['source']}: "
                f"{entry['count']:,} ({entry['share']:.1%})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Leakage check (Phase 1 §3)
# ---------------------------------------------------------------------------


def hashes_for_yaml(path: Path) -> set[str]:
    """Return content_hashes of rows in a YAML list. Strict: raises if not a list."""
    rows = load_yaml_list(path)
    out: set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            c = r.get("content")
            if isinstance(c, str):
                out.add(content_hash(c))
    return out


def discover_challenge_yamls(schema_eval_dir: Path) -> list[Path]:
    """Recurse under schema/eval/challenges/**/*.yaml, skipping any dir whose
    name starts with `_` (e.g. `_excluded/` tombstones)."""
    challenges_dir = schema_eval_dir / "challenges"
    if not challenges_dir.is_dir():
        return []
    found: list[Path] = []
    for p in challenges_dir.rglob("*.yaml"):
        if any(part.startswith("_") for part in p.relative_to(challenges_dir).parts[:-1]):
            continue
        found.append(p)
    return sorted(found)


def compute_leakage(
    train_hashes: set[str],
    target_hashes: set[str],
) -> list[str]:
    return sorted(train_hashes & target_hashes)


# ---------------------------------------------------------------------------
# fold_assignments.json detection (informational only)
# ---------------------------------------------------------------------------


def detect_fold_assignments_status(source_run_dir: Path) -> dict[str, Any]:
    path = source_run_dir / "fold_assignments.json"
    status: dict[str, Any] = {"path": str(path), "present": path.exists()}
    if not path.exists():
        return status
    try:
        json.loads(path.read_text(encoding="utf-8"))
        status["machine_readable"] = True
    except json.JSONDecodeError:
        status["machine_readable"] = False
        status["note"] = (
            "Detected as preview-style (truncated, not valid JSON). "
            "fold_id derived from fold_*/run/_eval_holdout_*/ artifacts instead."
        )
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export §0.1-canonical Phase 1 v8 residuals from existing 5-fold OOF run."
    )
    parser.add_argument(
        "--source-run-dir", type=Path, default=Path("runs/l1b_residuals_v8"),
        help="Existing 5-fold residual run dir.",
    )
    parser.add_argument(
        "--curation-dir", type=Path, default=Path("../parapet-data/curated/v8_experiment"),
        help="v8 curation dir (used as the content_hash leakage source set).",
    )
    parser.add_argument(
        "--canonical-eval", type=Path, action="append", default=None,
        help="Path to current canonical L2 eval YAML/JSONL (repeatable). "
             "Leakage between v8 train and these is a hard fail (rc=2 unless "
             "--allow-leakage). If omitted, no gate runs and manifest records "
             "gate_status = 'no_canonical_eval_defined'. Do NOT pass legacy "
             "schema/eval/{l1_holdout,challenges/...} YAMLs here without "
             "regeneration — use audit_legacy_eval_leakage.py for those.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase1_v8_residuals"),
    )
    parser.add_argument(
        "--margin-band-raw-score", type=float, default=DEFAULT_MARGIN_BAND_RAW_SCORE,
    )
    parser.add_argument(
        "--baseline-sample-rate", type=float, default=DEFAULT_BASELINE_SAMPLE_RATE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-leakage", action="store_true",
        help="Do not exit non-zero if leakage is detected. Off by default.",
    )
    args = parser.parse_args(argv)

    source_run_dir = args.source_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source run: {source_run_dir}")
    print(f"Output:     {output_dir}")

    fold_assignments_status = detect_fold_assignments_status(source_run_dir)
    if fold_assignments_status.get("present") and not fold_assignments_status.get("machine_readable", True):
        print(f"  [info] fold_assignments.json: {fold_assignments_status['note']}")

    # ---- per-fold join ---------------------------------------------------
    fold_dirs = discover_fold_dirs(source_run_dir)
    print(f"Folds discovered: {[fid for fid, _ in fold_dirs]}")

    all_joined: list[dict[str, Any]] = []
    fold_thresholds: dict[str, float] = {}
    fold_threshold_sources: dict[str, str] = {}
    fold_input_paths: dict[str, dict[str, str]] = {}

    for fold_id, fold_dir in fold_dirs:
        rows, threshold, sources = join_fold(fold_id, fold_dir)
        all_joined.extend(rows)
        fold_thresholds[str(fold_id)] = threshold
        fold_threshold_sources[str(fold_id)] = str(sources["eval_config_l1_threshold"])
        fold_input_paths[str(fold_id)] = {k: str(v) for k, v in sources.items()}
        print(f"  fold_{fold_id}: {len(rows):,} rows  threshold={threshold}")

    # ---- cross-checks ----------------------------------------------------
    summary_path = source_run_dir / "summary.json"
    expected_total = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_total = summary.get("total_predictions")
    if expected_total is not None and expected_total != len(all_joined):
        raise ValueError(
            f"row count mismatch vs summary.json: joined={len(all_joined)} "
            f"summary.total_predictions={expected_total}"
        )

    all_predictions_path = source_run_dir / "all_predictions.jsonl"
    if all_predictions_path.exists():
        with all_predictions_path.open("r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines != len(all_joined):
            raise ValueError(
                f"row count mismatch vs all_predictions.jsonl: "
                f"joined={len(all_joined)} jsonl_lines={n_lines}"
            )
    print(f"Total joined rows: {len(all_joined):,}")

    # ---- partition + emit residuals --------------------------------------
    buckets = partition_rows(
        all_joined,
        margin_band_raw_score=args.margin_band_raw_score,
        baseline_sample_rate=args.baseline_sample_rate,
        seed=args.seed,
    )

    noncanonical_counter: Counter[str] = Counter()
    for row in all_joined:
        rr = row.get("reason_raw")
        canonical, was_norm = normalize_reason(rr)
        if was_norm and rr is not None and rr != "":
            noncanonical_counter[str(rr)] += 1

    residuals_path = output_dir / "residuals.jsonl"
    with residuals_path.open("w", encoding="utf-8") as f:
        residual_count = 0
        for category in ("false_negative", "false_positive", "near_boundary_benign"):
            for row in buckets[category]:
                phase1 = to_phase1_row(row)
                phase1["residual_category"] = category
                f.write(json.dumps(phase1, ensure_ascii=False) + "\n")
                residual_count += 1
    baseline_path = output_dir / "baseline_correct.jsonl"
    with baseline_path.open("w", encoding="utf-8") as f:
        for row in buckets["baseline_correct"]:
            phase1 = to_phase1_row(row)
            phase1["residual_category"] = "baseline_correct"
            f.write(json.dumps(phase1, ensure_ascii=False) + "\n")
    print(f"Residuals: {residuals_path} ({residual_count} rows)")
    print(f"Baseline:  {baseline_path} ({len(buckets['baseline_correct'])} rows)")

    # ---- distribution report --------------------------------------------
    distribution = build_distribution_report(all_joined)
    (output_dir / "distribution_report.json").write_text(
        json.dumps(distribution, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "distribution_report.md").write_text(
        render_distribution_md(distribution), encoding="utf-8"
    )

    # ---- canonical-eval leakage gate (only when caller provides targets) ---
    train_yaml = args.curation_dir / "train.yaml"
    canonical_evals: list[Path] = list(args.canonical_eval or [])
    leakage_path: Path | None = None
    overall_overlap = 0

    if canonical_evals:
        train_hashes = hashes_for_yaml(train_yaml)
        leakage: dict[str, Any] = {
            "train_yaml": str(train_yaml),
            "train_yaml_sha256": sha256_file(train_yaml) if train_yaml.exists() else None,
            "n_train_hashes": len(train_hashes),
            "targets": [],
            "skipped_targets": [],
        }
        for target in canonical_evals:
            if not target.exists():
                leakage["skipped_targets"].append(
                    {"path": str(target), "reason": "file not found"}
                )
                print(f"  [skip] {target}: file not found", file=sys.stderr)
                continue
            try:
                target_hashes = hashes_for_yaml(target)
            except ValueError as exc:
                leakage["skipped_targets"].append({"path": str(target), "reason": str(exc)})
                print(f"  [skip] {target}: {exc}", file=sys.stderr)
                continue
            overlap = compute_leakage(train_hashes, target_hashes)
            leakage["targets"].append({
                "path": str(target),
                "sha256": sha256_file(target),
                "n_target_hashes": len(target_hashes),
                "overlap_count": len(overlap),
                "overlap_sample": overlap[:50],
            })
            overall_overlap += len(overlap)
        leakage["overall_overlap_count"] = overall_overlap

        leakage_path = output_dir / "leakage_check.json"
        leakage_path.write_text(json.dumps(leakage, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Leakage check: {leakage_path} (overlap={overall_overlap})")
    else:
        print(
            "No --canonical-eval provided. Skipping leakage gate. "
            "Manifest will record gate_status='no_canonical_eval_defined'."
        )

    # gate_status drives both manifest reporting and exit code
    if not canonical_evals:
        gate_status = "no_canonical_eval_defined"
    elif overall_overlap == 0:
        gate_status = "passed"
    elif args.allow_leakage:
        gate_status = "failed_allowed"
    else:
        gate_status = "failed_blocked"

    # ---- manifest --------------------------------------------------------
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now_iso(),
        "script_args": {
            "source_run_dir": str(args.source_run_dir),
            "curation_dir": str(args.curation_dir),
            "canonical_eval": [str(p) for p in canonical_evals],
            "output_dir": str(args.output_dir),
            "margin_band_raw_score": args.margin_band_raw_score,
            "baseline_sample_rate": args.baseline_sample_rate,
            "seed": args.seed,
            "allow_leakage": args.allow_leakage,
        },
        "fold_assignments_json_status": fold_assignments_status,
        "near_boundary_definition": {
            "selector_field": "raw_score",
            "margin_band": args.margin_band_raw_score,
            "note": (
                "Near-boundary selection uses raw_score (SVM decision_function) "
                "with abs(raw_score) <= margin_band, matching the existing "
                "scripts/export_l1_residuals.py:391 selection. l1_score is "
                "score (post-pipeline), used for the verdict + threshold pairing."
            ),
        },
        "schema_directionmd_0_1": {
            "training_input_fields": [
                "content", "label", "language", "reason", "source", "content_hash",
            ],
            "metadata_fields": [
                "l1_score", "l1_decision", "l1_thresholds", "routed_reason",
                "fold_id", "format_bin", "length_bin", "error_type",
                "residual_category",
            ],
            "diagnostic_fields": [
                "raw_score", "raw_unquoted_score", "raw_squash_score",
                "raw_score_delta", "unquoted_score", "squash_score", "quote_detected",
            ],
            "routed_reason_note": (
                "Reserved for future cascade/router integration. Always null in "
                "Phase 1 export. Do not duplicate `reason` here — that blurs "
                "source taxonomy with routing behavior."
            ),
        },
        "inputs": {
            "summary_json": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path) if summary_path.exists() else None,
            },
            "all_predictions_jsonl": {
                "path": str(all_predictions_path),
                "sha256": sha256_file(all_predictions_path) if all_predictions_path.exists() else None,
            },
            "curation_manifest": {
                "path": str(args.curation_dir / "manifest.json"),
                "sha256": sha256_file(args.curation_dir / "manifest.json")
                          if (args.curation_dir / "manifest.json").exists() else None,
            },
            "per_fold": fold_input_paths,
        },
        "fold_thresholds": fold_thresholds,
        "fold_threshold_sources": fold_threshold_sources,
        "row_counts": {
            "total_predictions": len(all_joined),
            "false_negative": len(buckets["false_negative"]),
            "false_positive": len(buckets["false_positive"]),
            "near_boundary_benign": len(buckets["near_boundary_benign"]),
            "baseline_correct": len(buckets["baseline_correct"]),
            "residuals_emitted": residual_count,
        },
        "outputs": {
            "residuals": str(residuals_path),
            "baseline_correct": str(baseline_path),
            "distribution_report_json": str(output_dir / "distribution_report.json"),
            "distribution_report_md": str(output_dir / "distribution_report.md"),
            "leakage_check": str(leakage_path) if leakage_path else None,
        },
        "warnings": {
            "noncanonical_reasons": dict(noncanonical_counter.most_common()),
        },
        "gate": {
            "status": gate_status,
            "canonical_eval": [str(p) for p in canonical_evals],
            "overall_overlap_count": overall_overlap,
            "leakage_check_path": str(leakage_path) if leakage_path else None,
        },
        "usability": {
            "for_l2_training": True,
            "for_truncation_policy_spike": True,
            "for_mechanical_detector_simulation": True,
            "for_phase4_acceptance": gate_status == "passed",
            "caveats": [
                (
                    "No canonical L2 acceptance eval was passed at export time "
                    "(--canonical-eval omitted). This artifact is usable as "
                    "L2 training input, for the truncation-policy spike, and "
                    "for mechanical-detector simulation. It is NOT validated "
                    "against any acceptance gate. Phase 4 acceptance must be "
                    "measured against a clean, content-hash-disjoint eval "
                    "produced by the JSONL curation pipeline."
                ) if gate_status == "no_canonical_eval_defined" else (
                    f"Canonical-eval gate status: {gate_status}. "
                    f"Overall overlap = {overall_overlap}. "
                    "Acceptance use is conditional on this status being 'passed'."
                ),
                (
                    "Legacy schema/eval YAMLs (l1_holdout, "
                    "tough_*_mirror_v2_novel) are NOT acceptance gates for L2. "
                    "They predate the JSONL curation pipeline. A diagnostic "
                    "audit against them lives under legacy_audit/ if present "
                    "and is informational only."
                ),
            ],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest:  {output_dir / 'manifest.json'}")

    if gate_status == "failed_blocked":
        print(
            f"FAIL: canonical-eval leakage (overlap={overall_overlap}). "
            f"See {leakage_path}. Re-run with --allow-leakage only after triage.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
