"""ProtectAI-style baseline recipe materialization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


Label = Literal["malicious", "benign"]
BaselineFamily = Literal["protectai_repro", "protectai_size_matched"]


@dataclass(frozen=True)
class ProtectAIRecipeSource:
    """One source file in the ProtectAI recipe."""

    path: Path
    label: Label
    cap: int | None = None


@dataclass(frozen=True)
class ProtectAIRecipeMaterialization:
    """Materialized baseline dataset and identity hashes."""

    train_path: Path
    val_path: Path
    holdout_path: Path
    recipe_manifest_path: Path
    baseline_family: BaselineFamily
    baseline_recipe_hash: str
    baseline_data_hash: str
    baseline_data_size: int


DEFAULT_PROTECTAI_RECIPE_SOURCES: tuple[ProtectAIRecipeSource, ...] = (
    # Attack sources
    ProtectAIRecipeSource(Path("malicious/opensource_gandalf_attacks.yaml"), "malicious"),
    ProtectAIRecipeSource(Path("malicious/opensource_chatgpt_jailbreak_attacks.yaml"), "malicious"),
    ProtectAIRecipeSource(Path("malicious/opensource_imoxto_attacks.yaml"), "malicious"),
    ProtectAIRecipeSource(Path("malicious/opensource_hackaprompt_attacks.yaml"), "malicious"),
    ProtectAIRecipeSource(Path("malicious/opensource_jailbreak_cls_attacks.yaml"), "malicious"),
    # Benign sources
    ProtectAIRecipeSource(Path("benign/opensource_notinject_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_wildguardmix_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_awesome_chatgpt_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_teven_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_dahoas_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_chatgpt_prompts_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_hf_instruction_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_no_robots_benign.yaml"), "benign"),
    ProtectAIRecipeSource(Path("benign/opensource_ultrachat_benign.yaml"), "benign"),
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _normalize_source_label(raw: Any, *, fallback: Label) -> Label:
    label = str(raw).strip().lower()
    if label in {"malicious", "attack", "positive"}:
        return "malicious"
    if label in {"benign", "negative"}:
        return "benign"
    return fallback


def _load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"Invalid recipe source format: expected list at {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row at {path}#{index}: expected mapping")
        rows.append(dict(row))
    return rows


def _resolve_source_path(data_root: Path, source_path: Path) -> Path:
    if source_path.is_absolute():
        return source_path

    candidates: list[Path] = [data_root / source_path]
    if source_path.parent in {Path("."), Path("")}:
        candidates.extend(
            [
                data_root / "malicious" / source_path.name,
                data_root / "benign" / source_path.name,
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (data_root / source_path).resolve()


def _deterministic_sample(
    entries: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError(f"Sample count must be >= 0, got {count}")
    if count > len(entries):
        raise ValueError(f"Cannot sample {count} rows from only {len(entries)} entries")
    if count == len(entries):
        return list(entries)
    indices = sorted(rng.sample(range(len(entries)), count))
    return [entries[i] for i in indices]


def _split_train_val(
    entries: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not entries:
        return [], []
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = random.Random(seed)
    by_label: dict[Label, list[dict[str, Any]]] = {"malicious": [], "benign": []}
    for entry in entries:
        label = _normalize_source_label(entry.get("label"), fallback="benign")
        by_label[label].append(entry)

    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for label in ("malicious", "benign"):
        group = list(by_label[label])  # copy
        rng.shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        proposed = int(round(len(group) * val_ratio))
        val_count = max(1, proposed)
        if val_count >= len(group):
            val_count = len(group) - 1
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _to_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep JSONL strictly parseable even when content includes U+2028/U+2029.
    lines = [json.dumps(entry, ensure_ascii=True) for entry in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_holdout_entries(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        entries: list[dict[str, Any]] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid holdout JSONL record at {path}:{line_no}")
            entries.append(dict(payload))
        return entries

    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise ValueError(f"Invalid holdout YAML payload at {path}: expected list")
        entries = []
        for index, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Invalid holdout row at {path}#{index}: expected mapping")
            entries.append(dict(row))
        return entries

    raise ValueError(f"Unsupported holdout format: {path}")


def materialize_protectai_recipe(
    *,
    data_root: Path,
    output_dir: Path,
    holdout_path: Path,
    baseline_family: BaselineFamily,
    seed: int,
    target_size: int | None,
    recipe_sources: tuple[ProtectAIRecipeSource, ...] = DEFAULT_PROTECTAI_RECIPE_SOURCES,
    contamination_denylist: tuple[str, ...] = (),
) -> ProtectAIRecipeMaterialization:
    """
    Materialize deterministic ProtectAI baseline splits and recipe manifest.

    `target_size` is required for `protectai_size_matched` and ignored for `protectai_repro`.
    """

    if baseline_family == "protectai_size_matched":
        if target_size is None:
            raise ValueError("target_size is required for protectai_size_matched")
        if target_size <= 0:
            raise ValueError(f"target_size must be > 0, got {target_size}")
    else:
        target_size = None

    denylisted = {item.strip() for item in contamination_denylist if item.strip()}
    rng = random.Random(seed)

    source_counts: dict[str, dict[str, int]] = {}
    missing: list[Path] = []
    all_entries: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for source in recipe_sources:
        source_rel = source.path.as_posix()
        if source_rel in denylisted or source.path.name in denylisted:
            continue

        source_path = _resolve_source_path(data_root, source.path)
        if not source_path.exists():
            missing.append(source_path)
            continue

        raw_rows = _load_yaml_rows(source_path)
        cleaned: list[dict[str, Any]] = []
        for index, row in enumerate(raw_rows, start=1):
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            row_label = _normalize_source_label(row.get("label"), fallback=source.label)
            mapped_label = source.label if row_label != source.label else row_label
            entry = {
                "id": str(row.get("id") or f"{source_path.stem}-{index}"),
                "label": mapped_label,
                "description": str(row.get("description") or f"{source_path.name}:{index}"),
                "content": content,
                "source": source_rel,
            }
            cleaned.append(entry)

        if source.cap is not None and len(cleaned) > source.cap:
            cleaned = _deterministic_sample(cleaned, source.cap, rng=rng)

        source_counts[source_rel] = {
            "loaded": len(raw_rows),
            "usable": len(cleaned),
            "cap": source.cap or 0,
        }

        for entry in cleaned:
            content_hash = _content_hash(entry["content"])
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            all_entries.append(entry)

    if missing:
        missing_paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "ProtectAI recipe source files are missing:\n"
            f"{missing_paths}\n"
            "Expected under schema/eval/benign or schema/eval/malicious; "
            "use fetch scripts or move files before running this baseline."
        )

    if not all_entries:
        raise ValueError("ProtectAI recipe produced zero entries after filtering/dedup")

    selected = list(all_entries)
    if baseline_family == "protectai_size_matched":
        assert target_size is not None
        if target_size > len(all_entries):
            raise ValueError(
                f"Cannot size-match ProtectAI baseline to {target_size} rows with only "
                f"{len(all_entries)} available recipe rows"
            )

        attacks = [entry for entry in all_entries if entry["label"] == "malicious"]
        benign = [entry for entry in all_entries if entry["label"] == "benign"]
        if not attacks or not benign:
            raise ValueError("ProtectAI recipe must contain both malicious and benign entries")

        attack_target = int(round(target_size * (len(attacks) / len(all_entries))))
        attack_target = max(1, min(attack_target, len(attacks)))
        benign_target = target_size - attack_target
        if benign_target <= 0:
            benign_target = 1
            attack_target = target_size - 1
        if benign_target > len(benign):
            deficit = benign_target - len(benign)
            benign_target = len(benign)
            attack_target += deficit
        if attack_target > len(attacks):
            deficit = attack_target - len(attacks)
            attack_target = len(attacks)
            benign_target += deficit
        if attack_target + benign_target != target_size:
            raise ValueError(
                "Internal allocation error while size-matching ProtectAI baseline "
                f"(attack_target={attack_target}, benign_target={benign_target}, target={target_size})"
            )

        selected_attacks = _deterministic_sample(attacks, attack_target, rng=rng)
        selected_benign = _deterministic_sample(benign, benign_target, rng=rng)
        selected = selected_attacks + selected_benign
        rng.shuffle(selected)

    train_entries, val_entries = _split_train_val(selected, seed=seed, val_ratio=0.1)
    holdout_entries = _load_holdout_entries(holdout_path)
    if not holdout_entries:
        raise ValueError(f"Canonical holdout is empty: {holdout_path}")

    baseline_dir = output_dir
    train_path = baseline_dir / "train.jsonl"
    val_path = baseline_dir / "val.jsonl"
    holdout_out = baseline_dir / "holdout.jsonl"
    recipe_manifest_path = baseline_dir / "recipe_manifest.json"

    _to_jsonl(train_path, train_entries)
    _to_jsonl(val_path, val_entries)
    _to_jsonl(holdout_out, holdout_entries)

    recipe_payload: dict[str, Any] = {
        "baseline_family": baseline_family,
        "seed": seed,
        "data_root": data_root.as_posix(),
        "target_size": target_size,
        "dataset_allowlist": [source.path.as_posix() for source in recipe_sources],
        "per_source_caps": {
            source.path.as_posix(): source.cap
            for source in recipe_sources
            if source.cap is not None
        },
        "label_mapping": {
            "attack": "malicious",
            "malicious": "malicious",
            "positive": "malicious",
            "benign": "benign",
            "negative": "benign",
        },
        "contamination_denylist": sorted(denylisted),
        "source_counts": source_counts,
        "counts": {
            "selected_total": len(selected),
            "selected_malicious": sum(1 for entry in selected if entry["label"] == "malicious"),
            "selected_benign": sum(1 for entry in selected if entry["label"] == "benign"),
            "train": len(train_entries),
            "val": len(val_entries),
            "holdout": len(holdout_entries),
        },
    }

    baseline_recipe_hash = _canonical_hash(
        {
            "baseline_family": recipe_payload["baseline_family"],
            "seed": recipe_payload["seed"],
            "target_size": recipe_payload["target_size"],
            "dataset_allowlist": recipe_payload["dataset_allowlist"],
            "per_source_caps": recipe_payload["per_source_caps"],
            "label_mapping": recipe_payload["label_mapping"],
            "contamination_denylist": recipe_payload["contamination_denylist"],
        }
    )
    baseline_data_hash = _canonical_hash(
        {
            "train_hashes": sorted(_content_hash(entry["content"]) for entry in train_entries),
            "val_hashes": sorted(_content_hash(entry["content"]) for entry in val_entries),
            "holdout_hashes": sorted(_content_hash(str(entry.get("content", ""))) for entry in holdout_entries),
        }
    )
    recipe_payload["baseline_recipe_hash"] = baseline_recipe_hash
    recipe_payload["baseline_data_hash"] = baseline_data_hash

    recipe_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_manifest_path.write_text(
        json.dumps(recipe_payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return ProtectAIRecipeMaterialization(
        train_path=train_path,
        val_path=val_path,
        holdout_path=holdout_out,
        recipe_manifest_path=recipe_manifest_path,
        baseline_family=baseline_family,
        baseline_recipe_hash=baseline_recipe_hash,
        baseline_data_hash=baseline_data_hash,
        baseline_data_size=len(train_entries) + len(val_entries),
    )
