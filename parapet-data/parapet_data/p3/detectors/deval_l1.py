"""D_eval member: frozen parapet L1 (generalist), via the parapet-eval Rust binary.

Wraps the already-wired `parapet-eval --l1-scores` scoring path (option (i): no Rust
engine changes, no heavy Python deps; stdlib subprocess/csv/json/math only). The
deployed L1 is currently the generalist specialist (ZH specialist WIP), so the
per-event surface_signal is the `generalist` column's raw SVM margin, calibrated to
[0,1] by L1's in-tree sigmoid.

BATCH-ONLY: one binary invocation scores many events. Never one process per event.

FAILS CLOSED: a subprocess failure, a missing CSV, a missing `generalist` column
(schema drift), or a missing row yields error DetectorResults (score=None), never a
silent 0.0. Every result records engine_sha + binary_path (which build produced it).

Contract reference (locked 2026-06-06 against the real binary):
  parapet-eval --config schema/eval/eval_config_l1_ensemble.yaml --dataset <dir>
               --source <name> --l1-scores --output <csv>
  CSV header: case_id,label,source,generalist,roleplay_jailbreak,instruction_override,
              meta_probe,exfiltration,adversarial_suffix,indirect_injection  (raw margins)
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
from typing import Optional

from parapet_data.p3.detectors.interface import FAMILY_LINEAR, DetectorResult, EventContext, clamp_unit

# Mirrors in-tree L1 calibration (parapet/src/layers/l1.rs): P = 1 / (1 + exp(-A * (raw + B))).
# Update if the engine calibration changes.
L1_SIGMOID_A = 0.6
L1_SIGMOID_B = 0.0

DETECTOR_ID = "deval_l1_generalist"
DEFAULT_SPECIALIST = "generalist"
DEFAULT_BINARY_REL = os.path.join("parapet", "target", "release", "parapet-eval")
DEFAULT_CONFIG_REL = os.path.join("schema", "eval", "eval_config_l1_ensemble.yaml")
_META_COLS = {"case_id", "label", "source"}  # CSV columns that are not specialists


def calibrate(raw: float) -> float:
    """L1 raw SVM margin -> [0,1] via the in-tree sigmoid (constants above)."""
    return clamp_unit(1.0 / (1.0 + math.exp(-L1_SIGMOID_A * (raw + L1_SIGMOID_B))))


class DEvalL1:
    family = FAMILY_LINEAR
    detector_id = DETECTOR_ID

    def __init__(
        self,
        *,
        engine_root: str,
        binary_path: Optional[str] = None,
        config_path: Optional[str] = None,
        specialist: str = DEFAULT_SPECIALIST,
        engine_sha: Optional[str] = None,
        run_fn=None,
        timeout: float = 300.0,
    ):
        self.engine_root = engine_root  # cwd for the binary; config/weight paths resolve here
        self.binary_path = binary_path or os.path.join(engine_root, DEFAULT_BINARY_REL)
        self.config_path = config_path or os.path.join(engine_root, DEFAULT_CONFIG_REL)
        self.specialist = specialist
        self.model_id = f"parapet-l1-{specialist}"
        self.engine_sha = engine_sha
        self._run = run_fn or self._run_binary
        self.timeout = timeout

    def _result(self, score, error, rationale=None) -> DetectorResult:
        return DetectorResult(
            score=score, family=self.family, model_id=self.model_id,
            detector_id=self.detector_id, rationale=rationale, error=error,
            engine_sha=self.engine_sha, binary_path=self.binary_path,
        )

    def _run_binary(self, dataset_dir: str, source: str, out_csv: str) -> None:
        cmd = [
            self.binary_path,
            "--config", self.config_path,
            "--dataset", dataset_dir,
            "--source", source,
            "--l1-scores",
            "--output", out_csv,
        ]
        subprocess.run(
            cmd, cwd=self.engine_root, capture_output=True, text=True,
            timeout=self.timeout, check=True,
        )

    def score(self, event_text: str, context: Optional[EventContext] = None) -> DetectorResult:
        return self.score_batch([event_text], [context])[0]

    def score_batch(self, texts: list, contexts=None) -> list:
        n = len(texts)
        results: list = [None] * n
        # Empty texts never reach the binary; mirror the judge's empty handling.
        send_idx = []
        for i, t in enumerate(texts):
            if not t or not str(t).strip():
                results[i] = self._result(None, "empty_event_text")
            else:
                send_idx.append(i)
        if not send_idx:
            return results

        source = "p3_events"
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = os.path.join(tmp, "ds")
            os.makedirs(ds_dir, exist_ok=True)
            cases = [{
                "id": f"ev{i}", "layer": "l1", "label": "unknown",
                "description": "p3 event", "content": texts[i],
            } for i in send_idx]
            # JSON is valid YAML; keeps this stdlib-only (no pyyaml). Verified against the binary.
            with open(os.path.join(ds_dir, f"{source}.yaml"), "w") as fh:
                json.dump(cases, fh)
            out_csv = os.path.join(tmp, "scores.csv")

            try:
                self._run(ds_dir, source, out_csv)
            except subprocess.CalledProcessError as exc:
                tail = ((exc.stderr or "").strip().splitlines() or [""])[-1]
                return self._fill(results, send_idx, f"l1_subprocess_failed: {tail[:160]}")
            except (subprocess.TimeoutExpired, OSError) as exc:
                return self._fill(results, send_idx, f"l1_run_error: {type(exc).__name__}")

            if not os.path.isfile(out_csv):
                return self._fill(results, send_idx, "l1_no_output")

            by_id, header_error = self._parse_csv(out_csv)

        if header_error:
            return self._fill(results, send_idx, header_error)

        for i in send_idx:
            raw = by_id.get(f"ev{i}", "missing")
            if raw == "missing":
                results[i] = self._result(None, "l1_missing_row")
            elif raw is None:
                results[i] = self._result(None, "l1_bad_value")
            else:
                results[i] = self._result(
                    calibrate(raw), None, rationale=f"{self.specialist} raw margin {raw:.4f}",
                )
        return results

    def _fill(self, results: list, idxs: list, error: str) -> list:
        for i in idxs:
            results[i] = self._result(None, error)
        return results

    def _parse_csv(self, out_csv: str):
        """Return (case_id -> raw float or None, header_error or None). Fails closed."""
        try:
            with open(out_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                cols = reader.fieldnames or []
                if self.specialist not in cols:
                    specialists = [c for c in cols if c not in _META_COLS]
                    return {}, f"l1_column_drift: '{self.specialist}' not in {specialists}"
                by_id = {}
                for row in reader:
                    try:
                        by_id[row.get("case_id")] = float(row.get(self.specialist))
                    except (TypeError, ValueError):
                        by_id[row.get("case_id")] = None
                return by_id, None
        except (OSError, csv.Error) as exc:
            return {}, f"l1_csv_unreadable: {type(exc).__name__}"
