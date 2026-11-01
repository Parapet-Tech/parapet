"""Runtime-arbitrated calibration for a specialist paired with the frozen generalist."""
from __future__ import annotations

import hashlib
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
    """Propose thresholds from the pruned artifact using an approximate surrogate.

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
    if not weights and not re.search(r"pub static WEIGHTS:.*?phf_map!\s*\{\s*\};", text, re.S):
        raise ValueError("candidate artifact has no valid weight table")
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


def calibrate_runtime(config: Any, validation_rows: Sequence[Mapping[str, Any]], *,
                      model_artifact: Path, runtime: Any) -> dict[str, Any]:
    """The injected runtime supplies ID -> blocked verdicts; surrogate proposes only."""
    if weights_identity(Path(config.generalist_weights)) != config.generalist_weights_sha256:
        raise ValueError("generalist weights identity mismatch")
    scores = score_candidate(model_artifact, validation_rows, config)
    candidates = sorted(set(scores))
    if not candidates or not all(math.isfinite(s) for s in candidates):
        raise ValueError("no finite calibration candidates")
    ids = {r["id"] for r in validation_rows}
    if len(ids) != len(validation_rows):
        raise ValueError("duplicate validation ids")

    def verdicts(t, generalist_only=False):
        result = runtime(t, generalist_only=generalist_only)
        if set(result) != ids or any(type(v) is not bool for v in result.values()):
            raise ValueError("runtime must return exactly one boolean verdict per validation id")
        return result

    baseline = verdicts(config.generalist_threshold, True)
    # A second generalist-only run at the solo cutoff measures score >= cutoff
    # through the same dual views. Its complement among baseline blocks is [0,3).
    solo = verdicts(config.generalist_block_alone, True)
    attacks = [r["id"] for r in validation_rows if r["label"] == "malicious" and baseline[r["id"]]]
    band = [i for i in attacks if not solo[i]]
    benign = [r["id"] for r in validation_rows if r["label"] == "benign" and baseline[r["id"]]]
    if not attacks:
        raise ValueError("no feasible threshold: no generalist-blocked attacks")
    evaluations = {}

    def evaluate(index):
        t = candidates[index]
        if index not in evaluations:
            blocked = verdicts(t)
            evaluations[index] = dict(threshold=t,
                total=_retention(sum(blocked[i] for i in attacks), len(attacks)),
                deciding_band=_retention(sum(blocked[i] for i in band), len(band)),
                relief_count=sum(not blocked[i] for i in benign))
        return evaluations[index]

    low, high = 0, len(candidates) - 1
    feasible = None
    while low <= high:
        mid = (low + high) // 2
        if evaluate(mid)["total"]["retention"] >= config.retention_floor:
            feasible = mid
            low = mid + 1
        else:
            high = mid - 1
    if feasible is None:
        raise ValueError("no feasible runtime threshold")
    relief = evaluate(feasible)["relief_count"]
    low, high = 0, feasible
    while low < high:
        mid = (low + high) // 2
        if evaluate(mid)["relief_count"] >= relief:
            high = mid
        else:
            low = mid + 1
    chosen = evaluate(low)
    # This prediction is diagnostic, never an input to the runtime search.
    t = chosen["threshold"]
    prediction = {
        "retained": sum(s >= t or solo[r["id"]] for r, s in zip(validation_rows, scores)
                        if r["id"] in attacks),
        "relief_count": sum(s < t and not solo[r["id"]] for r, s in zip(validation_rows, scores)
                            if r["id"] in benign),
    }
    return dict(chosen, runtime_evaluations=list(evaluations.values()),
        surrogate_prediction=prediction,
        candidate_weights_sha256=weights_identity(model_artifact),
        generalist_weights_sha256=config.generalist_weights_sha256)
