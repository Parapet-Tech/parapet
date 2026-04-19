"""Golden contract extraction and drift checking for curated corpora."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .curated_artifact import load_curated_entries


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_curated_entries(curated_dir: Path) -> list[dict[str, Any]]:
    return load_curated_entries(curated_dir)


def _normalize_rel(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _stats(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": count,
            "share": round((count / total) if total else 0.0, 6),
        }
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def _count_by(entries: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    counter: Counter[str] = Counter()
    for entry in entries:
        value = entry.get(field)
        if value is not None:
            counter[str(value)] += 1
    return _stats(counter, len(entries))


def _count_by_label(entries: list[dict[str, Any]], field: str) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for entry in entries:
        label = str(entry.get("label", ""))
        value = entry.get(field)
        if not label or value is None:
            continue
        grouped[label][str(value)] += 1
        totals[label] += 1
    return {
        label: _stats(counter, totals[label])
        for label, counter in sorted(grouped.items())
    }


def find_matching_run_manifests(
    semantic_hash: str,
    runner_root: Path,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(runner_root.rglob("run_manifest.json")):
        try:
            manifest = _load_json(path)
        except Exception:
            continue
        curation = manifest.get("curation", {})
        runtime = manifest.get("runtime", {})
        if curation.get("semantic_hash") != semantic_hash:
            continue
        matches.append(
            {
                "path": _normalize_rel(path, project_root),
                "run_id": manifest.get("run_id"),
                "git_sha": runtime.get("git_sha"),
                "trainer_script_hash": runtime.get("trainer_script_hash"),
                "parapet_eval_hash": runtime.get("parapet_eval_hash"),
                "spec_hash": curation.get("spec_hash"),
            }
        )
    return matches


def build_golden_contract(
    curated_dir: Path,
    *,
    run_manifest_path: Path | None = None,
    runner_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    curated_dir = curated_dir.resolve()
    manifest_path = curated_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    entries = _load_curated_entries(curated_dir)

    matching_runs: list[dict[str, Any]] = []
    if run_manifest_path is not None:
        run_manifest = _load_json(run_manifest_path)
        matching_runs = [
            {
                "path": _normalize_rel(run_manifest_path.resolve(), project_root),
                "run_id": run_manifest.get("run_id"),
                "git_sha": run_manifest.get("runtime", {}).get("git_sha"),
                "trainer_script_hash": run_manifest.get("runtime", {}).get("trainer_script_hash"),
                "parapet_eval_hash": run_manifest.get("runtime", {}).get("parapet_eval_hash"),
                "spec_hash": run_manifest.get("curation", {}).get("spec_hash"),
            }
        ]
    elif runner_root is not None:
        matching_runs = find_matching_run_manifests(
            manifest["semantic_hash"],
            runner_root.resolve(),
            project_root=project_root,
        )

    contract = {
        "contract_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "curated_dir": _normalize_rel(curated_dir, project_root),
        "manifest_path": _normalize_rel(manifest_path, project_root),
        "spec_name": manifest.get("spec_name"),
        "spec_version": manifest.get("spec_version"),
        "spec_hash": manifest.get("spec_hash"),
        "seed": manifest.get("seed"),
        "semantic_hash": manifest.get("semantic_hash"),
        "output_hash": manifest.get("output_hash"),
        "totals": {
            "samples": len(entries),
            "attack_samples": sum(1 for e in entries if e.get("label") == "malicious"),
            "benign_samples": sum(1 for e in entries if e.get("label") == "benign"),
        },
        "distributions": {
            "label": _count_by(entries, "label"),
            "reason": _count_by(entries, "reason"),
            "language": _count_by(entries, "language"),
            "source": _count_by(entries, "source"),
            "reason_by_label": _count_by_label(entries, "reason"),
            "language_by_label": _count_by_label(entries, "language"),
            "source_by_label": _count_by_label(entries, "source"),
        },
        "source_hashes": manifest.get("source_hashes", {}),
        "source_metadata": manifest.get("source_metadata", {}),
        "matching_runs": matching_runs,
        "unique_trainer_script_hashes": sorted(
            {
                run.get("trainer_script_hash")
                for run in matching_runs
                if run.get("trainer_script_hash")
            }
        ),
    }
    return contract


def write_contract(contract: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def compare_contract(
    baseline: dict[str, Any],
    observed: dict[str, Any],
    *,
    max_source_share_delta: float = 0.05,
    max_language_share_delta: float = 0.03,
    min_monitored_source_share: float = 0.005,
    check_source_hashes: bool = False,
) -> list[str]:
    violations: list[str] = []

    baseline_lang = baseline["distributions"]["language"]
    observed_lang = observed["distributions"]["language"]
    for language, expected in baseline_lang.items():
        baseline_share = float(expected["share"])
        observed_share = float(observed_lang.get(language, {}).get("share", 0.0))
        delta = abs(observed_share - baseline_share)
        if delta > max_language_share_delta:
            violations.append(
                f"language share drift: {language} baseline={baseline_share:.3f} "
                f"observed={observed_share:.3f} delta={delta:.3f}"
            )

    baseline_sources = baseline["distributions"]["source_by_label"]
    observed_sources = observed["distributions"]["source_by_label"]
    for label, expected_sources in baseline_sources.items():
        actual_sources = observed_sources.get(label, {})
        for source, expected in expected_sources.items():
            baseline_share = float(expected["share"])
            if baseline_share < min_monitored_source_share:
                continue
            observed_share = float(actual_sources.get(source, {}).get("share", 0.0))
            delta = abs(observed_share - baseline_share)
            if delta > max_source_share_delta:
                violations.append(
                    f"source share drift [{label}]: {source} baseline={baseline_share:.3f} "
                    f"observed={observed_share:.3f} delta={delta:.3f}"
                )

    if check_source_hashes:
        baseline_hashes = baseline.get("source_hashes", {})
        observed_hashes = observed.get("source_hashes", {})
        for source, expected_hash in baseline_hashes.items():
            actual_hash = observed_hashes.get(source)
            if actual_hash != expected_hash:
                violations.append(
                    f"source hash mismatch: {source} baseline={expected_hash} observed={actual_hash}"
                )

    return violations
