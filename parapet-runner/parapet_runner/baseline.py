"""PG2 baseline execution and eval result parsing."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .manifest import EvalResult


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandExecutor(Protocol):
    """Injectable command execution boundary."""

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
        ...


class SubprocessCommandExecutor:
    """Default executor used in production wiring."""

    def run(self, args: Sequence[str], *, cwd: Path | None = None) -> CommandResult:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        return CommandResult(
            args=tuple(str(a) for a in args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _find_first(payload: Any, candidates: set[str]) -> Any | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in candidates:
                return value
        for value in payload.values():
            found = _find_first(value, candidates)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first(item, candidates)
            if found is not None:
                return found
    return None


def parse_eval_result_json(payload: Mapping[str, Any], *, threshold_fallback: float = -0.5) -> EvalResult:
    """Parse parapet-eval JSON payload into EvalResult."""

    f1 = _find_first(payload, {"f1", "f1_score"})
    precision = _find_first(payload, {"precision"})
    recall = _find_first(payload, {"recall"})
    false_positives = _find_first(payload, {"false_positives", "fp"})
    false_negatives = _find_first(payload, {"false_negatives", "fn", "fn_count"})
    threshold = _find_first(payload, {"threshold"})
    holdout_size = _find_first(payload, {"holdout_size", "total", "count", "n", "cases"})

    if f1 is None or precision is None or recall is None:
        raise ValueError("Could not parse required metrics (f1/precision/recall) from eval payload")

    return EvalResult(
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        false_positives=int(false_positives or 0),
        false_negatives=int(false_negatives or 0),
        threshold=float(threshold if threshold is not None else threshold_fallback),
        holdout_size=int(holdout_size or 0),
    )


class PG2BaselineRunner:
    """Runs PG2 on a holdout split and returns EvalResult."""

    def __init__(self, executor: CommandExecutor) -> None:
        self._executor = executor

    def run(
        self,
        *,
        parapet_eval_bin: Path,
        eval_config: Path,
        dataset_dir: Path,
        source: str,
        output_json: Path,
        cwd: Path,
    ) -> EvalResult:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(parapet_eval_bin),
            "--config",
            str(eval_config),
            "--dataset",
            str(dataset_dir),
            "--source",
            source,
            "--layer",
            "l3_inbound",
            "--remap-layer",
            "l2a",
            "--json",
            "--output",
            str(output_json),
        ]

        result = self._executor.run(command, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(
                "PG2 baseline command failed:\n"
                f"args={result.args}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return parse_eval_result_json(payload)
