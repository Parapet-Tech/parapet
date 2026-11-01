"""Pinned, lossless JSONL import, independent of MirrorSpec sampling."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import CurationManifest, SplitManifest, compute_semantic_hash


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_column: str
    id_column: str
    label_column: str | None = None
    fixed_label: Literal["benign", "malicious"] | None = None
    group_column: str | None = None
    stratify_column: str | None = None
    weight_column: str | None = None

    @model_validator(mode="after")
    def require_label(self):
        if self.label_column is None and self.fixed_label is None:
            raise ValueError("label_column or fixed_label required")
        return self


class AssignmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SelectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm: str


class FixedImportSpec(BaseModel):
    """Contract: exact selected rows, canonical labels, group-hashed splits."""
    model_config = ConfigDict(extra="forbid")
    mode: Literal["fixed_import"]
    name: str
    inputs: list[InputSpec] = Field(min_length=1)
    assignment: AssignmentSpec | None = None
    selection: SelectionSpec | None = None
    expected_counts: dict[Literal["benign", "malicious"], int] | None = None
    seed: int = 42
    splits: dict[str, float] = Field(default_factory=lambda: {"train": 1., "val": 0., "holdout": 0.})

    @model_validator(mode="after")
    def validate_contract(self):
        if len({s.name for s in self.inputs}) != len(self.inputs):
            raise ValueError("duplicate input name")
        if set(self.splits) != {"train", "val", "holdout"} or any(
            not 0 <= v <= 1 for v in self.splits.values()
        ) or abs(sum(self.splits.values()) - 1) > 1e-9:
            raise ValueError("splits must be train/val/holdout fractions summing to one")
        return self


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _pinned(path: Path, pin: str, base: Path) -> bytes:
    data = (base / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError(f"sha256 mismatch: {path}")
    return data


def compose_fixed(spec: FixedImportSpec, output: Path, base: Path) -> CurationManifest:
    """Validate all inputs before writing deterministic runner-compatible artifacts."""
    selection = None
    if spec.selection:
        arms = json.loads(_pinned(spec.selection.path, spec.selection.sha256, base))
        selection = arms[spec.selection.arm]
        if set(selection) != {s.name for s in spec.inputs}:
            raise ValueError("selection parts must match input names")
        for ids in selection.values():
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError("selection ids must be lists of strings")
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate selected id")
    assignment = None
    if spec.assignment:
        assignment = json.loads(_pinned(spec.assignment.path, spec.assignment.sha256, base))
        if not isinstance(assignment, dict) or any(v not in ("train", "calibration") for v in assignment.values()):
            raise ValueError("assignment must map ids to train or calibration")
    if spec.selection and spec.splits["train"] < 1 and assignment is None:
        raise ValueError("arm selection with calibration/holdout requires pinned full-population assignment")
    rows = []
    seen = set()
    split_rows = {k: [] for k in ("train", "val", "holdout")}
    grouped = {}
    full_group_assignments = {}
    row_ids = {}
    for source in spec.inputs:
        wanted = set(selection[source.name]) if selection is not None else None
        found = set()
        for line in _pinned(source.path, source.sha256, base).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row[source.id_column]
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("id must be a nonempty string")
            if identifier in seen:
                raise ValueError(f"duplicate id: {identifier}")
            seen.add(identifier)
            label = row[source.label_column] if source.label_column else source.fixed_label
            if label not in ("benign", "malicious") or (
                source.fixed_label is not None and label != source.fixed_label
            ) or ("label" in row and row["label"] != label):
                raise ValueError(f"label drift: {identifier}")
            text = row[source.text_column]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a nonempty string")
            if assignment is not None and identifier not in assignment:
                raise ValueError("assignment missing population id")
            group = row.get(source.group_column) if source.group_column else identifier
            if group is None:
                group = ["row", identifier]
            elif not isinstance(group, str) or not group:
                raise ValueError("group must be a nonempty string")
            group_key = json.dumps(group, sort_keys=True)
            if assignment is not None:
                value = assignment[identifier]
                if group_key in full_group_assignments and full_group_assignments[group_key] != value:
                    raise ValueError("assignment splits a group")
                full_group_assignments[group_key] = value
            if wanted is not None and identifier in wanted:
                found.add(identifier)
            if wanted is not None and identifier not in wanted and (assignment is None or assignment[identifier] != "calibration"):
                continue
            # Preserve all original fields, including any noncanonical content field.
            if "content" in row and row["content"] != text:
                raise ValueError("content column conflicts with text column")
            if "id" in row and row["id"] != identifier:
                raise ValueError("id column conflicts with canonical id")
            item = dict(row, id=identifier, content=text, label=label)
            if source.weight_column:
                weight = row[source.weight_column]
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("sample weight must be finite and positive")
                if "sample_weight" in row and row["sample_weight"] != weight:
                    raise ValueError("sample_weight conflicts with weight column")
                item["sample_weight"] = weight
            row_ids[id(item)] = identifier
            group_key = json.dumps(group, sort_keys=True)
            stratum = row[source.stratify_column] if source.stratify_column else None
            if source.stratify_column and (not isinstance(stratum, str) or not stratum):
                raise ValueError("stratum must be a nonempty string")
            if group_key in grouped and grouped[group_key][0] != stratum:
                raise ValueError("group spans inconsistent strata")
            grouped.setdefault(group_key, (stratum, []))[1].append(item)
            score = Fraction(int(hashlib.sha256(_json_bytes([spec.seed, group])).hexdigest(), 16), 2**256)
            cumulative = Fraction(0)
            for split in ("train", "val", "holdout"):
                cumulative += Fraction(str(spec.splits[split]))
                if score < cumulative or split == "holdout":
                    split_rows[split].append(item)
                    break
            rows.append(item)
        if wanted is not None and found != wanted:
            raise ValueError(f"selected id missing: {sorted(wanted - found)}")
    if not rows:
        raise ValueError("selected population must be nonempty")
    counts = Counter(r["label"] for r in rows)
    if spec.expected_counts is not None and any(counts[k] != v for k, v in spec.expected_counts.items()):
        raise ValueError(f"count mismatch: {dict(counts)}")
    if assignment is not None:
        if set(assignment) != seen:
            raise ValueError("assignment must match full population ids")
        split_rows = {"train": [], "val": [], "holdout": []}
        for row in rows:
            split_rows["val" if assignment[row["id"]] == "calibration" else "train"].append(row)
    elif any(source.stratify_column for source in spec.inputs):
        # Allocate whole groups within each stratum, ordered by seeded hashes.
        split_rows = {k: [] for k in split_rows}
        strata = {}
        for key, (stratum, entries) in grouped.items():
            strata.setdefault(stratum, []).append((key, entries))
        for groups in strata.values():
            groups.sort(key=lambda pair: hashlib.sha256(_json_bytes([spec.seed, pair[0]])).digest())
            train_end = round(len(groups) * spec.splits["train"])
            val_end = round(len(groups) * (spec.splits["train"] + spec.splits["val"]))
            for index, (_, entries) in enumerate(groups):
                name = "train" if index < train_end else "val" if index < val_end else "holdout"
                split_rows[name].extend(entries)
    output.mkdir(parents=True, exist_ok=True)
    def write(name, entries):
        data = b"".join((json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n").encode() for r in entries)
        (output / f"{name}.jsonl").write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    hashes = lambda entries: sorted(hashlib.sha256(r["content"].strip().encode()).hexdigest() for r in entries)
    splits = {}
    for name, entries in split_rows.items():
        digest = write(name, entries)
        splits[name] = SplitManifest(sha256=digest, name=name, sample_count=len(entries), content_hashes=hashes(entries), artifact_path=Path(f"{name}.jsonl"))
    manifest = CurationManifest(
        spec_name=spec.name, spec_version="fixed-import-v1", spec_hash=hashlib.sha256(_json_bytes(spec.model_dump(mode="json"))).hexdigest(),
        seed=spec.seed, timestamp="1970-01-01T00:00:00+00:00", source_hashes={s.name: s.sha256 for s in spec.inputs},
        output_hash=write("curated", rows), semantic_hash=compute_semantic_hash(hashes(rows), {}),
        total_samples=len(rows), attack_samples=counts["malicious"], benign_samples=counts["benign"],
        splits=splits, cell_fills={}, gaps=[], cross_contamination_dropped=0, output_path=Path("curated.jsonl"),
    )
    split_ids = _json_bytes({name: sorted(row_ids[id(r)] for r in entries) for name, entries in split_rows.items()})
    (output / "split_ids.json").write_bytes(split_ids)
    from .models import ArtifactReceipt
    manifest.split_ids = ArtifactReceipt(path=Path("split_ids.json"), sha256=hashlib.sha256(split_ids).hexdigest())
    manifest.fixed_import = spec.model_dump(mode="json")
    receipt = manifest.model_dump(mode="json")
    (output / "manifest.json").write_bytes(_json_bytes(receipt))
    (output / "composition.json").write_bytes(_json_bytes({"counts": dict(counts)}))
    return manifest
