"""Experiment orchestration with strict dependency injection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from .baseline import CommandExecutor, SubprocessCommandExecutor, parse_eval_result_json
from .config import ThresholdPolicy, TrainConfig
from .manifest import (
    CurationManifest,
    EvalResult,
    RunManifest,
    RuntimeIdentity,
    compute_metric_delta,
    compute_semantic_parity_hash,
)


class ResolvedSplits(BaseModel):
    """Runner-resolved split artifacts for one curated dataset."""

    train_path: Path
    val_path: Path
    holdout_path: Path
    holdout_source: str
    dataset_dir: Path
    content_hashes: list[str] = Field(default_factory=list)
    per_cell_counts: dict[str, Any] = Field(default_factory=dict)


class SplitResolver(Protocol):
    def resolve(self, curation: CurationManifest) -> ResolvedSplits:
        ...


class Trainer(Protocol):
    def train(self, *, train_split: Path, config: TrainConfig, output_dir: Path) -> Path:
        ...


class Evaluator(Protocol):
    def evaluate(
        self,
        *,
        model_artifact: Path,
        split_path: Path,
        threshold: float,
        split_name: str,
        output_dir: Path,
    ) -> EvalResult:
        ...


class BaselineProvider(Protocol):
    def run(self, *, holdout: ResolvedSplits, output_dir: Path) -> EvalResult:
        ...


class ErrorAnalyzer(Protocol):
    def write(
        self,
        *,
        eval_result: EvalResult,
        baseline_result: EvalResult | None,
        output_dir: Path,
    ) -> Path:
        ...


class ThresholdCalibrator(Protocol):
    def calibrate(
        self,
        *,
        evaluator: Evaluator,
        model_artifact: Path,
        val_split: Path,
        output_dir: Path,
    ) -> float:
        ...


class RuntimeIdentityProvider(Protocol):
    def collect(self) -> RuntimeIdentity:
        ...


class ArtifactVerifier(Protocol):
    def verify(self, curation: CurationManifest) -> None:
        ...


@dataclass(frozen=True)
class ExperimentDependencies:
    split_resolver: SplitResolver
    trainer: Trainer
    evaluator: Evaluator
    baseline_provider: BaselineProvider
    error_analyzer: ErrorAnalyzer
    runtime_identity_provider: RuntimeIdentityProvider
    threshold_calibrator: ThresholdCalibrator | None = None
    artifact_verifier: ArtifactVerifier | None = None


class StaticSplitResolver:
    """Simple resolver for early integration while parapet-data evolves."""

    def __init__(self, resolved: ResolvedSplits) -> None:
        self._resolved = resolved

    def resolve(self, curation: CurationManifest) -> ResolvedSplits:  # noqa: ARG002
        return self._resolved


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _normalize_label(raw_label: Any, *, context: str) -> str:
    label = str(raw_label).strip().lower()
    if label in {"malicious", "attack", "positive"}:
        return "malicious"
    if label in {"benign", "negative"}:
        return "benign"
    raise ValueError(f"Unsupported label '{raw_label}' in {context}")


def _load_labeled_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Split artifact not found: {path}")

    if path.suffix.lower() == ".jsonl":
        entries: list[dict[str, Any]] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Invalid JSONL record at {path}:{line_no}: expected object")
            entries.append(dict(payload))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            entries = []
        elif isinstance(payload, list):
            if not all(isinstance(item, Mapping) for item in payload):
                raise ValueError(f"Invalid YAML list in {path}: all entries must be objects")
            entries = [dict(item) for item in payload]
        else:
            raise ValueError(f"Invalid YAML payload in {path}: expected list of objects")
    else:
        raise ValueError(f"Unsupported split format for {path} (expected .jsonl/.yaml/.yml)")

    validated: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        context = f"{path} entry #{idx}"
        if "content" not in entry:
            raise ValueError(f"Missing 'content' in {context}")
        content = entry["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Invalid empty content in {context}")
        label = _normalize_label(entry.get("label"), context=context)
        validated.append(
            {
                **entry,
                "content": content,
                "label": label,
            }
        )
    return validated


def _write_yaml_entries(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(list(entries), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _resolve_relative_path(path: Path, *, base_dir: Path | None) -> Path:
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def _split_content_hashes(path: Path) -> list[str]:
    return sorted(_content_hash(entry["content"]) for entry in _load_labeled_entries(path))


def _to_cell_fill_dict(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
        raise TypeError(f"Unsupported cell fill model dump type: {type(dumped)}")
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(f"Unsupported cell fill value type: {type(raw)}")


class ManifestSplitResolver:
    """Resolve train/val/holdout artifacts from CurationManifest with hash checks."""

    def __init__(
        self,
        *,
        root_dir: Path | None = None,
        verify_content_hashes: bool = True,
    ) -> None:
        self._root_dir = root_dir
        self._verify_content_hashes = verify_content_hashes

    def resolve(self, curation: CurationManifest) -> ResolvedSplits:
        required = ("train", "val", "holdout")
        missing = [name for name in required if name not in curation.splits]
        if missing:
            raise ValueError(f"Curation manifest missing required splits: {missing}")

        output_path = Path(curation.output_path)
        output_parent: Path | None = None
        if output_path.is_absolute():
            output_parent = output_path.parent
        elif self._root_dir is not None:
            output_parent = (self._root_dir / output_path).resolve().parent

        resolved_paths: dict[str, Path] = {}
        for split_name in required:
            split_manifest = curation.splits[split_name]
            raw_path = Path(split_manifest.artifact_path)

            path = raw_path
            if not path.is_absolute():
                candidates: list[Path] = []
                if output_parent is not None:
                    candidates.append((output_parent / raw_path).resolve())
                if self._root_dir is not None:
                    candidates.append((self._root_dir / raw_path).resolve())
                if candidates:
                    existing = next((candidate for candidate in candidates if candidate.exists()), None)
                    path = existing or candidates[0]
                else:
                    path = raw_path.resolve()

            if not path.exists():
                raise FileNotFoundError(
                    f"Split artifact for '{split_name}' not found: {path} "
                    f"(manifest path={raw_path})"
                )

            if self._verify_content_hashes:
                expected = sorted(str(h) for h in split_manifest.content_hashes)
                actual = _split_content_hashes(path)
                if actual != expected:
                    raise ValueError(
                        f"Content hash mismatch in split '{split_name}': "
                        f"expected {len(expected)} hashes, got {len(actual)}"
                    )

            resolved_paths[split_name] = path

        all_hashes: list[str] = []
        for split_name in required:
            all_hashes.extend(str(h) for h in curation.splits[split_name].content_hashes)

        per_cell_counts = {str(k): _to_cell_fill_dict(v) for k, v in curation.cell_fills.items()}

        holdout_path = resolved_paths["holdout"]
        return ResolvedSplits(
            train_path=resolved_paths["train"],
            val_path=resolved_paths["val"],
            holdout_path=holdout_path,
            holdout_source=holdout_path.stem,
            dataset_dir=holdout_path.parent,
            content_hashes=all_hashes,
            per_cell_counts=per_cell_counts,
        )


class TrainScriptTrainer:
    """Concrete trainer adapter around scripts/train_l1_specialist_variant.py."""

    def __init__(
        self,
        *,
        trainer_script: Path,
        workspace_root: Path,
        command_executor: CommandExecutor | None = None,
        python_bin: str = "python",
    ) -> None:
        self._trainer_script = trainer_script
        self._workspace_root = workspace_root
        self._executor = command_executor or SubprocessCommandExecutor()
        self._python_bin = python_bin

    def train(self, *, train_split: Path, config: TrainConfig, output_dir: Path) -> Path:
        entries = _load_labeled_entries(train_split)
        attacks = [entry for entry in entries if entry["label"] == "malicious"]
        benign = [entry for entry in entries if entry["label"] == "benign"]
        if not attacks:
            raise ValueError(f"Train split has no malicious samples: {train_split}")
        if not benign:
            raise ValueError(f"Train split has no benign samples: {train_split}")

        prepared_dir = output_dir / "_prepared" / "train"
        attack_file = prepared_dir / "attacks.yaml"
        benign_file = prepared_dir / "benign.yaml"
        _write_yaml_entries(attack_file, attacks)
        _write_yaml_entries(benign_file, benign)

        model_artifact = output_dir / "model" / f"l1_weights_{config.specialist}.rs"
        holdout_artifact = output_dir / "model" / f"{config.specialist}_train_holdout.yaml"

        args = [
            self._python_bin,
            "-u",
            str(self._trainer_script),
            *config.to_train_script_args(
                attack_files=[attack_file],
                benign_files=[benign_file],
                out_path=model_artifact,
                holdout_path=holdout_artifact,
            ),
        ]
        result = self._executor.run(args, cwd=self._workspace_root)

        # Persist training output for reproducibility
        log_dir = output_dir / "model"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{config.specialist}_train.log"
        log_path.write_text(
            result.stderr + "\n" + result.stdout,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Training command failed (log: {log_path}):\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        if not model_artifact.exists():
            raise FileNotFoundError(
                f"Training succeeded but model artifact missing: {model_artifact}"
            )
        return model_artifact


def _build_eval_cases(entries: Sequence[Mapping[str, Any]], *, split_name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        case_id = str(entry.get("id") or f"{split_name}_{idx}")
        description = str(entry.get("description") or f"{split_name} sample {idx}")
        cases.append(
            {
                "id": case_id,
                "layer": "l1",
                "label": _normalize_label(entry.get("label"), context=f"{split_name}:{case_id}"),
                "description": description,
                "content": str(entry["content"]),
            }
        )
    return cases


def _set_l1_threshold(config: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    payload = dict(config)
    layers_raw = payload.get("layers")
    if not isinstance(layers_raw, Mapping):
        raise ValueError("Eval config missing 'layers' mapping")
    layers = dict(layers_raw)

    l1_key = next((key for key in layers.keys() if str(key).lower() == "l1"), None)
    if l1_key is None:
        l1_key = "L1"
        layers[l1_key] = {"mode": "block"}
    l1_raw = layers.get(l1_key)
    if not isinstance(l1_raw, Mapping):
        raise ValueError("Eval config L1 entry must be a mapping")
    l1 = dict(l1_raw)
    l1["threshold"] = float(threshold)
    layers[l1_key] = l1
    payload["layers"] = layers
    return payload


class ParapetEvalEvaluator:
    """Concrete evaluator adapter around parapet-eval for L1 model metrics."""

    def __init__(
        self,
        *,
        parapet_eval_bin: Path,
        eval_config: Path,
        workspace_root: Path,
        command_executor: CommandExecutor | None = None,
        accepted_returncodes: Sequence[int] = (0, 1),
    ) -> None:
        self._parapet_eval_bin = parapet_eval_bin
        self._eval_config = eval_config
        self._workspace_root = workspace_root
        self._executor = command_executor or SubprocessCommandExecutor()
        self._accepted_returncodes = set(accepted_returncodes)

    def evaluate(
        self,
        *,
        model_artifact: Path,
        split_path: Path,
        threshold: float,
        split_name: str,
        output_dir: Path,
    ) -> EvalResult:
        if not model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_artifact}")

        run_dir = output_dir / f"_eval_{split_name}_{str(threshold).replace('.', 'p').replace('-', 'm')}"
        dataset_dir = run_dir / "dataset"
        eval_cases_path = dataset_dir / f"{split_name}.yaml"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        split_entries = _load_labeled_entries(split_path)
        eval_cases = _build_eval_cases(split_entries, split_name=split_name)
        _write_yaml_entries(eval_cases_path, eval_cases)

        base_config = yaml.safe_load(self._eval_config.read_text(encoding="utf-8"))
        if not isinstance(base_config, Mapping):
            raise ValueError(f"Invalid eval config format: {self._eval_config}")
        threshold_config = _set_l1_threshold(base_config, threshold)
        config_path = run_dir / "eval_config_l1_threshold.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(threshold_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        output_json = run_dir / "eval.json"
        command = [
            str(self._parapet_eval_bin),
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_dir),
            "--source",
            split_name,
            "--layer",
            "l1",
            "--json",
            "--output",
            str(output_json),
        ]
        result = self._executor.run(command, cwd=self._workspace_root)
        if result.returncode not in self._accepted_returncodes:
            raise RuntimeError(
                "Evaluator command failed:\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        if not output_json.exists():
            raise FileNotFoundError(
                "Evaluator did not produce output JSON:\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return parse_eval_result_json(payload, threshold_fallback=threshold)


class ParapetEvalPG2BaselineProvider:
    """Run PG2 baseline via parapet-eval on the exact same holdout split."""

    def __init__(
        self,
        *,
        parapet_eval_bin: Path,
        eval_config: Path,
        workspace_root: Path,
        command_executor: CommandExecutor | None = None,
        accepted_returncodes: Sequence[int] = (0, 1),
    ) -> None:
        self._parapet_eval_bin = parapet_eval_bin
        self._eval_config = eval_config
        self._workspace_root = workspace_root
        self._executor = command_executor or SubprocessCommandExecutor()
        self._accepted_returncodes = set(accepted_returncodes)

    def run(self, *, holdout: ResolvedSplits, output_dir: Path) -> EvalResult:
        baseline_dir = output_dir / "_baseline_pg2"
        dataset_dir = baseline_dir / "dataset"
        source_name = holdout.holdout_source
        eval_cases_path = dataset_dir / f"{source_name}.yaml"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        split_entries = _load_labeled_entries(holdout.holdout_path)
        eval_cases = _build_eval_cases(split_entries, split_name=source_name)
        _write_yaml_entries(eval_cases_path, eval_cases)

        output_json = baseline_dir / "pg2_eval.json"
        command = [
            str(self._parapet_eval_bin),
            "--config",
            str(self._eval_config),
            "--dataset",
            str(dataset_dir),
            "--source",
            source_name,
            "--layer",
            "l1",
            "--remap-layer",
            "l2a",
            "--json",
            "--output",
            str(output_json),
        ]
        result = self._executor.run(command, cwd=self._workspace_root)
        if result.returncode not in self._accepted_returncodes:
            raise RuntimeError(
                "PG2 baseline command failed:\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )
        if not output_json.exists():
            raise FileNotFoundError(
                "PG2 baseline did not produce output JSON:\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return parse_eval_result_json(payload, threshold_fallback=0.0)


class F1GridSearchCalibrator:
    """Calibrate threshold by maximizing F1 on validation split only."""

    def __init__(self, thresholds: Sequence[float] | None = None) -> None:
        self._thresholds = list(thresholds) if thresholds else [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

    def calibrate(
        self,
        *,
        evaluator: Evaluator,
        model_artifact: Path,
        val_split: Path,
        output_dir: Path,
    ) -> float:
        best_threshold = self._thresholds[0]
        best_f1 = -1.0
        for threshold in self._thresholds:
            result = evaluator.evaluate(
                model_artifact=model_artifact,
                split_path=val_split,
                threshold=threshold,
                split_name="val",
                output_dir=output_dir,
            )
            if result.f1 > best_f1:
                best_f1 = result.f1
                best_threshold = threshold
        return best_threshold


class YamlErrorAnalyzer:
    """Emit a compact, human-readable post-eval error summary YAML."""

    def write(
        self,
        *,
        eval_result: EvalResult,
        baseline_result: EvalResult | None,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "errors.yaml"
        payload: dict[str, Any] = {
            "eval": {
                "threshold": eval_result.threshold,
                "f1": eval_result.f1,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "false_positives": eval_result.false_positives,
                "false_negatives": eval_result.false_negatives,
                "holdout_size": eval_result.holdout_size,
            }
        }
        if baseline_result is not None:
            payload["baseline"] = {
                "threshold": baseline_result.threshold,
                "f1": baseline_result.f1,
                "precision": baseline_result.precision,
                "recall": baseline_result.recall,
                "false_positives": baseline_result.false_positives,
                "false_negatives": baseline_result.false_negatives,
                "holdout_size": baseline_result.holdout_size,
            }
            payload["delta"] = compute_metric_delta(eval_result, baseline_result)
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return out_path


class RuntimeIdentityCollector:
    """Collect immutable runtime identity with lockfile-first environment hashing."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        trainer_script: Path,
        parapet_eval_binary: Path,
        pg2_model_id: str,
        eval_config: Path,
        command_executor: CommandExecutor | None = None,
        lockfile_names: Sequence[str] = ("uv.lock", "poetry.lock", "requirements.txt"),
        allow_pip_freeze_fallback: bool = True,
    ) -> None:
        self._workspace_root = workspace_root
        self._trainer_script = trainer_script
        self._parapet_eval_binary = parapet_eval_binary
        self._pg2_model_id = pg2_model_id
        self._eval_config = eval_config
        self._executor = command_executor or SubprocessCommandExecutor()
        self._lockfile_names = tuple(lockfile_names)
        self._allow_pip_freeze_fallback = allow_pip_freeze_fallback

    def collect(self) -> RuntimeIdentity:
        return RuntimeIdentity(
            git_sha=self._git_sha(),
            trainer_script_hash=_sha256_file(self._trainer_script),
            parapet_eval_hash=_sha256_file(self._parapet_eval_binary),
            pg2_model_id=self._pg2_model_id,
            eval_config_hash=_sha256_file(self._eval_config),
            env_hash=self._environment_hash(),
        )

    def _git_sha(self) -> str:
        result = self._executor.run(["git", "rev-parse", "HEAD"], cwd=self._workspace_root)
        if result.returncode != 0:
            raise RuntimeError(f"Unable to resolve git SHA: {result.stderr.strip()}")
        return result.stdout.strip()

    def _environment_hash(self) -> str:
        for name in self._lockfile_names:
            lockfile = self._workspace_root / name
            if lockfile.exists():
                return _sha256_file(lockfile)

        if not self._allow_pip_freeze_fallback:
            raise FileNotFoundError(
                "No lockfile found for environment hash. Looked for: "
                + ", ".join(self._lockfile_names)
            )

        freeze = self._executor.run(["python", "-m", "pip", "freeze", "--all"], cwd=self._workspace_root)
        if freeze.returncode != 0:
            raise RuntimeError(f"pip freeze failed: {freeze.stderr.strip()}")
        normalized = "\n".join(sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip()))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class OutputHashVerifier:
    """Verify curation output path resolves to the expected output hash."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root_dir = root_dir

    def verify(self, curation: CurationManifest) -> None:
        output_path = Path(curation.output_path)
        if not output_path.is_absolute() and self._root_dir is not None:
            output_path = (self._root_dir / output_path).resolve()
        if not output_path.exists():
            raise FileNotFoundError(f"Curation artifact not found: {output_path}")
        actual = _sha256_file(output_path)
        if actual != curation.output_hash:
            raise ValueError(
                f"Curation artifact hash mismatch for {output_path}: "
                f"expected {curation.output_hash}, got {actual}"
            )


def assert_no_leakage(
    train_hashes: set[str],
    eval_hashes: set[str],
    *,
    context: str = "",
) -> None:
    """Assert that no training content appears in the eval set.

    Raises ValueError with overlap details if leakage is detected.
    """
    overlap = train_hashes & eval_hashes
    if overlap:
        pct = 100 * len(overlap) / max(len(train_hashes), 1)
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"Data leakage detected{' (' + context + ')' if context else ''}: "
            f"{len(overlap):,} of {len(train_hashes):,} train samples ({pct:.1f}%) "
            f"appear in the eval set ({len(eval_hashes):,} samples). "
            f"First hashes: {sample}"
        )


class ExperimentRunner:
    """Coordinates train -> val calibration -> holdout eval -> baseline -> manifests."""

    def __init__(self, deps: ExperimentDependencies) -> None:
        self._deps = deps

    def run_experiment(
        self,
        *,
        curation_manifest: CurationManifest,
        train_config: TrainConfig,
        output_dir: Path,
        run_id: str | None = None,
    ) -> RunManifest:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._deps.artifact_verifier is not None:
            self._deps.artifact_verifier.verify(curation_manifest)

        splits = self._deps.split_resolver.resolve(curation_manifest)
        model_artifact = self._deps.trainer.train(
            train_split=splits.train_path,
            config=train_config,
            output_dir=output_dir,
        )

        threshold = train_config.threshold_value
        if train_config.threshold_policy == ThresholdPolicy.CALIBRATE_F1:
            if self._deps.threshold_calibrator is None:
                raise ValueError("threshold_calibrator is required for CALIBRATE_F1 policy")
            threshold = self._deps.threshold_calibrator.calibrate(
                evaluator=self._deps.evaluator,
                model_artifact=model_artifact,
                val_split=splits.val_path,
                output_dir=output_dir,
            )

        eval_result = self._deps.evaluator.evaluate(
            model_artifact=model_artifact,
            split_path=splits.holdout_path,
            threshold=threshold,
            split_name="holdout",
            output_dir=output_dir,
        )
        baseline_result = self._deps.baseline_provider.run(holdout=splits, output_dir=output_dir)
        error_file = self._deps.error_analyzer.write(
            eval_result=eval_result,
            baseline_result=baseline_result,
            output_dir=output_dir,
        )

        semantic_hash: str | None = None
        if splits.content_hashes:
            semantic_hash = compute_semantic_parity_hash(
                splits.content_hashes,
                splits.per_cell_counts,
            )

        run_manifest = RunManifest(
            run_id=run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"),
            runtime=self._deps.runtime_identity_provider.collect(),
            curation=curation_manifest,
            train_config=train_config,
            eval_result=eval_result,
            pg2_baseline=baseline_result,
            delta=compute_metric_delta(eval_result, baseline_result),
            error_file=error_file,
            semantic_parity_hash=semantic_hash,
        )
        return run_manifest


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cli_semantic_hash(content_hashes_file: Path, counts_json_file: Path) -> int:
    hashes = [
        line.strip()
        for line in content_hashes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per_cell_counts = json.loads(counts_json_file.read_text(encoding="utf-8"))
    print(compute_semantic_parity_hash(hashes, per_cell_counts))
    return 0


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _default_workspace_root() -> Path:
    # .../parapet/parapet-runner/parapet_runner/runner.py -> .../parapet
    return Path(__file__).resolve().parents[2]


def _parse_thresholds(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("calibration thresholds cannot be empty")
    return [float(value) for value in values]


def _cli_run(args: argparse.Namespace) -> int:
    curation_manifest_path = Path(args.curation_manifest).resolve()
    train_config_path = Path(args.train_config).resolve()
    output_dir = Path(args.output_dir).resolve()

    workspace_root = (
        Path(args.workspace_root).resolve()
        if args.workspace_root is not None
        else _default_workspace_root()
    )
    trainer_script = _resolve_relative_path(Path(args.trainer_script), base_dir=workspace_root)
    parapet_eval_bin = _resolve_relative_path(Path(args.parapet_eval_bin), base_dir=workspace_root)
    l1_eval_config = _resolve_relative_path(Path(args.l1_eval_config), base_dir=workspace_root)
    pg2_eval_config = _resolve_relative_path(Path(args.pg2_eval_config), base_dir=workspace_root)

    curation_raw = _load_structured_file(curation_manifest_path)
    train_config_raw = _load_structured_file(train_config_path)
    curation_manifest = CurationManifest.model_validate(curation_raw)
    train_config = TrainConfig.model_validate(train_config_raw)

    thresholds = _parse_thresholds(args.calibration_thresholds)
    executor = SubprocessCommandExecutor()
    deps = ExperimentDependencies(
        split_resolver=ManifestSplitResolver(
            root_dir=curation_manifest_path.parent,
            verify_content_hashes=not args.skip_split_hash_verify,
        ),
        trainer=TrainScriptTrainer(
            trainer_script=trainer_script,
            workspace_root=workspace_root,
            command_executor=executor,
            python_bin=args.python_bin,
        ),
        evaluator=ParapetEvalEvaluator(
            parapet_eval_bin=parapet_eval_bin,
            eval_config=l1_eval_config,
            workspace_root=workspace_root,
            command_executor=executor,
        ),
        baseline_provider=ParapetEvalPG2BaselineProvider(
            parapet_eval_bin=parapet_eval_bin,
            eval_config=pg2_eval_config,
            workspace_root=workspace_root,
            command_executor=executor,
        ),
        error_analyzer=YamlErrorAnalyzer(),
        runtime_identity_provider=RuntimeIdentityCollector(
            workspace_root=workspace_root,
            trainer_script=trainer_script,
            parapet_eval_binary=parapet_eval_bin,
            pg2_model_id=args.pg2_model_id,
            eval_config=l1_eval_config,
            command_executor=executor,
        ),
        threshold_calibrator=F1GridSearchCalibrator(thresholds=thresholds),
        artifact_verifier=(
            None
            if args.skip_output_hash_verify
            else OutputHashVerifier(root_dir=curation_manifest_path.parent)
        ),
    )

    run_manifest = ExperimentRunner(deps).run_experiment(
        curation_manifest=curation_manifest,
        train_config=train_config,
        output_dir=output_dir,
        run_id=args.run_id,
    )

    run_manifest_path = (
        Path(args.run_manifest_path).resolve()
        if args.run_manifest_path
        else (output_dir / "run_manifest.json")
    )
    write_run_manifest(run_manifest_path, run_manifest)
    print(run_manifest_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parapet runner utilities")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run train -> val calibration -> holdout eval pipeline")
    run.add_argument("--curation-manifest", type=Path, required=True)
    run.add_argument("--train-config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--run-id", type=str)
    run.add_argument("--run-manifest-path", type=Path)
    run.add_argument("--workspace-root", type=Path)
    run.add_argument("--trainer-script", type=Path, default=Path("scripts/train_l1_specialist_variant.py"))
    run.add_argument("--parapet-eval-bin", type=Path, default=Path("target/release/parapet-eval.exe"))
    run.add_argument("--l1-eval-config", type=Path, default=Path("schema/eval/eval_config_l1_only.yaml"))
    run.add_argument("--pg2-eval-config", type=Path, default=Path("schema/eval/eval_config_l2a_only.yaml"))
    run.add_argument("--pg2-model-id", type=str, default="pg2-22m")
    run.add_argument("--python-bin", type=str, default="python")
    run.add_argument(
        "--calibration-thresholds",
        type=str,
        default="-1.5,-1.0,-0.5,0.0,0.5,1.0",
        help="Comma-separated threshold grid for F1 calibration on validation split",
    )
    run.add_argument(
        "--skip-output-hash-verify",
        action="store_true",
        help="Skip curation output artifact hash verification",
    )
    run.add_argument(
        "--skip-split-hash-verify",
        action="store_true",
        help="Skip split content-hash verification against CurationManifest",
    )

    semantic_hash = subparsers.add_parser(
        "semantic-hash",
        help="Compute SHA256(sorted content hashes + per-cell counts)",
    )
    semantic_hash.add_argument("--content-hashes-file", type=Path, required=True)
    semantic_hash.add_argument("--cell-counts-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _cli_run(args)
    if args.command == "semantic-hash":
        return _cli_semantic_hash(args.content_hashes_file, args.cell_counts_json)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
