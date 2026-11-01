"""Calibration contract for a specialist paired with frozen generalist scores."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _retention(kept: int, total: int) -> dict[str, Any]:
    if not total:
        return {"kept": kept, "total": total, "retention": None, "wilson95": None}
    z = 1.959963984540054
    p = kept / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"kept": kept, "total": total, "retention": p, "wilson95": [max(0., center - half), min(1., center + half)]}


def constrained_retention(
    rows: Sequence[Mapping[str, Any]], *, retention_floor: float = .99,
    generalist_threshold: float = 0., generalist_block_alone: float = 3.,
) -> dict[str, Any]:
    """Maximize benign relief over distinct specialist scores; ties use lowest t.

    Each row supplies label (benign/malicious), specialist_score, generalist_score.
    Empty calibration or no generalist-blocked attacks cannot define retention.
    With finite scores, the lowest candidate retains every blocked attack, so a
    feasible threshold always exists for a floor in [0, 1].
    """
    if not math.isfinite(retention_floor) or not 0 <= retention_floor <= 1:
        raise ValueError("retention floor must be in [0, 1]")
    if not all(map(math.isfinite, [generalist_threshold, generalist_block_alone])) or generalist_block_alone < generalist_threshold:
        raise ValueError("invalid generalist thresholds")
    for row in rows:
        if row["label"] not in ("malicious", "benign"):
            raise ValueError("invalid calibration label")
        if not all(math.isfinite(row[k]) for k in ("generalist_score", "specialist_score")):
            raise ValueError("calibration scores must be finite")
    attacks = [r for r in rows if r["label"] == "malicious" and r["generalist_score"] >= generalist_threshold]
    band = [r for r in attacks if r["generalist_score"] < generalist_block_alone]
    best = None
    for threshold in sorted({r["specialist_score"] for r in rows}):
        kept = sum(r["generalist_score"] >= generalist_block_alone or r["specialist_score"] >= threshold for r in attacks)
        if not attacks or kept / len(attacks) < retention_floor:
            continue
        relief = sum(r["label"] == "benign" and generalist_threshold <= r["generalist_score"] < generalist_block_alone and r["specialist_score"] < threshold for r in rows)
        if best is None or relief > best["relief_count"]:
            best = {"threshold": threshold, "relief_count": relief,
                    "total": _retention(kept, len(attacks)),
                    "deciding_band": _retention(sum(r["specialist_score"] >= threshold for r in band), len(band))}
    if best is None:
        raise ValueError("no feasible threshold: calibration requires generalist-blocked attacks")
    return best


def weights_identity(path: Path) -> str:
    """Stable generated-table identity, omitting only the timestamp line."""
    return hashlib.sha256(b"".join(line for line in path.read_bytes().splitlines(keepends=True)
                                  if not line.startswith(b"// Trained:"))).hexdigest()


def score_candidate(path: Path, rows: Sequence[Mapping[str, Any]], config: Any) -> list[float]:
    """Score the actual pruned Rust artifact, with runtime binary raw/squash features.

    This calibration path supports char_wb, the runtime specialist contract.
    L0 uses the same transform as the trainer, before both scoring passes.
    """
    import importlib.util
    import re
    if config.analyzer != "char_wb":
        raise ValueError("retention calibration requires char_wb")
    module_spec = importlib.util.spec_from_file_location("l1_calibration_transform",
        Path(__file__).resolve().parents[2] / "scripts" / "train_l1_specialist.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    text = path.read_text(encoding="utf-8")
    bias_match = re.search(r"pub const BIAS: f64 = ([-+0-9.eE]+)_f64;", text)
    if bias_match is None:
        raise ValueError("candidate artifact has no bias")
    weights = {}
    for match in re.finditer(r'    "((?:[^"\\]|\\.)*)" => ([-+0-9.eE]+)_f64,', text):
        key = re.sub(r'\\([\\"])', lambda m: m[1], match[1])
        weights[key] = float(match[2])
    if not weights:
        raise ValueError("candidate artifact has no weights")
    bias = float(bias_match[1])
    def score(value):
        features = set()
        for word in value.lower().split():
            padded = " " + word + " "
            for n in range(config.ngram_min, config.ngram_max + 1):
                if len(padded) < n:
                    features.add(padded)
                else:
                    features.update(padded[i:i+n] for i in range(len(padded)-n+1))
        return bias + math.fsum(weights.get(f, 0.) for f in sorted(features))
    results = []
    for row in rows:
        value = row["content"]
        if config.apply_l0_transform:
            value = module.apply_l0_transform(value)
        results.append(max(score(value), score("".join(c for c in value.lower() if c.isalnum()))))
    return results


def calibrate_pinned_scores(config: Any, validation_rows: Sequence[Mapping[str, Any]], *,
                            model_artifact: Path) -> dict[str, Any]:
    """Bind frozen generalist scores to content, then score the trained candidate."""
    if weights_identity(Path(config.generalist_weights)) != config.generalist_weights_sha256:
        raise ValueError("generalist weights identity mismatch")
    path = Path(config.calibration_scores)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config.calibration_scores_sha256:
        raise ValueError("calibration score sha256 mismatch")
    scores = [json.loads(line) for line in data.splitlines() if line.strip()]
    expected = {r["id"]: (r["label"], hashlib.sha256(r["content"].encode()).hexdigest()) for r in validation_rows}
    actual = {r["id"]: (r["label"], r.get("content_sha256")) for r in scores}
    if len(expected) != len(validation_rows) or len(actual) != len(scores) or actual != expected:
        raise ValueError("calibration scores must match validation ids, labels and content hashes exactly")
    if any(r.get("generalist_weights_sha256") != config.generalist_weights_sha256 for r in scores):
        raise ValueError("generalist score identity mismatch")
    if any("specialist_score" in r for r in scores):
        raise ValueError("externally supplied specialist scores are forbidden")
    by_id = {r["id"]: r for r in scores}
    candidate_scores = score_candidate(model_artifact, validation_rows, config)
    joined = [dict(by_id[r["id"]], specialist_score=s) for r, s in zip(validation_rows, candidate_scores)]
    result = constrained_retention(joined, retention_floor=config.retention_floor,
        generalist_threshold=config.generalist_threshold, generalist_block_alone=config.generalist_block_alone)
    return dict(result, scores_sha256=config.calibration_scores_sha256,
                candidate_weights_sha256=weights_identity(model_artifact),
                generalist_weights_sha256=config.generalist_weights_sha256)
