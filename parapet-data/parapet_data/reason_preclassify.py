"""Preclassify heuristic attack sources into explicit-reason staged JSONL."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from .classifiers import classify_reason
from .extractors import get_extractor
from .filters import content_hash, passes_label_filter
from .models import AttackReason, MirrorSpec, ReasonProvenance, SourceRef
from .sampler import load_source
from .staged_artifact import write_staged_rows

# Routing bucket for rows the heuristic classifier could not score above
# the confidence floor. Kept as a real reason value so the staged contract
# stays honest (every row has a reason); not a mirror taxonomy peer.
_UNCATEGORIZED_REASON = AttackReason.UNCATEGORIZED.value


@dataclass
class ReasonPreclassifyReport:
    """Summary for one source preclassification pass.

    Counters split classifier-attributable rows from the routing bucket so
    audits can see, per source, how much of the corpus the calibrated mirror
    taxonomy actually covers.

    - ``classified``: rows that scored above the classifier floor and
      received a calibrated mirror reason.
    - ``uncategorized``: rows that did not score above the floor and were
      written with ``reason: uncategorized`` so the data is preserved.
    - ``empty_text``: rows dropped because the extractor returned no text.
    - ``rows_written``: total rows in the JSONL output (``classified`` +
      ``uncategorized``).
    """

    source_name: str
    input_path: str
    output_path: Path
    rows_read: int = 0
    rows_after_label_filter: int = 0
    rows_with_text: int = 0
    rows_written: int = 0
    empty_text: int = 0
    classified: int = 0
    uncategorized: int = 0
    output_sha256: str = ""
    by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def keep_rate(self) -> float:
        """Fraction of text-bearing rows written to disk (always 1.0 today)."""
        if self.rows_with_text == 0:
            return 0.0
        return self.rows_written / self.rows_with_text

    @property
    def classified_rate(self) -> float:
        """Fraction of text-bearing rows that received a calibrated reason."""
        if self.rows_with_text == 0:
            return 0.0
        return self.classified / self.rows_with_text


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def preclassified_source_filename(source: SourceRef) -> str:
    """Return a stable filename for a preclassified source artifact."""
    safe_name = _SAFE_FILENAME_RE.sub("_", source.name).strip("._")
    if not safe_name:
        safe_name = "source"
    return f"{source.language.value.lower()}_{safe_name}_heuristic_staged.jsonl"


def _reason_value(classification: object) -> str:
    reason = getattr(classification, "reason", None)
    if hasattr(reason, "value"):
        reason = getattr(reason, "value")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("classify_reason returned a classification without a reason")
    return reason.strip()


def _preclassified_rows(
    source: SourceRef,
    report: ReasonPreclassifyReport,
    *,
    base_dir: Path | None,
    label: str,
) -> Iterator[dict]:
    extractor = get_extractor(source.extractor)
    counts: Counter[str] = Counter()

    for row in load_source(source, base_dir=base_dir):
        report.rows_read += 1
        if not passes_label_filter(row, source.label_filter):
            continue
        report.rows_after_label_filter += 1

        text = extractor(row)
        if not text:
            report.empty_text += 1
            continue
        report.rows_with_text += 1

        classification = classify_reason(text)
        if classification is None:
            # Preserve the row with the uncategorized routing bucket so
            # downstream consumers (e.g. supplement lanes) can still use
            # this known-malicious sample. Audit fields stay null so it's
            # obvious the classifier did not score this row.
            report.uncategorized += 1
            counts[_UNCATEGORIZED_REASON] += 1
            report.rows_written += 1
            yield {
                "content": text,
                "label": label,
                "language": source.language.value,
                "source": source.name,
                "reason": _UNCATEGORIZED_REASON,
                "reason_confidence": None,
                "reason_signals": [],
                "content_hash": content_hash(text),
            }
            continue

        reason = _reason_value(classification)
        counts[reason] += 1
        report.classified += 1
        report.rows_written += 1
        yield {
            "content": text,
            "label": label,
            "language": source.language.value,
            "source": source.name,
            "reason": reason,
            "reason_confidence": getattr(classification, "confidence", None),
            "reason_signals": list(getattr(classification, "signals", ())),
            "content_hash": content_hash(text),
        }

    report.by_reason = dict(sorted(counts.items()))


def preclassify_reason_source(
    source: SourceRef,
    output_path: Path,
    *,
    base_dir: Path | None = None,
    label: str = "malicious",
) -> ReasonPreclassifyReport:
    """Classify a heuristic attack source once and write explicit-reason JSONL.

    Every label-passing row with text is written to the output. Rows that
    score above the classifier floor get their calibrated mirror reason;
    rows that do not get ``reason: uncategorized`` so the known-malicious
    signal is preserved for downstream supplement/pool consumption rather
    than silently discarded. Audit fields (``reason_confidence``,
    ``reason_signals``) are populated only for classified rows.
    """
    if label != "malicious":
        raise ValueError("reason preclassification only supports malicious sources")
    if source.reason_provenance not in {
        ReasonProvenance.HEURISTIC,
        ReasonProvenance.HEURISTIC_STAGED,
    }:
        raise ValueError(
            f"{source.name}: expected heuristic reason_provenance, got "
            f"{source.reason_provenance}"
        )

    report = ReasonPreclassifyReport(
        source_name=source.name,
        input_path=str(source.path),
        output_path=output_path,
    )
    report.output_sha256 = write_staged_rows(
        output_path,
        _preclassified_rows(source, report, base_dir=base_dir, label=label),
    )
    return report


def heuristic_attack_sources(
    spec: MirrorSpec,
    *,
    source_names: Sequence[str] | None = None,
) -> list[SourceRef]:
    """Return unique heuristic attack sources from a spec in stable order."""
    requested = set(source_names or [])
    sources_by_name: dict[str, SourceRef] = {}

    for cell in spec.cells:
        for source in cell.attack_sources:
            if source.reason_provenance != ReasonProvenance.HEURISTIC:
                continue
            if requested and source.name not in requested:
                continue
            existing = sources_by_name.get(source.name)
            if existing is None:
                sources_by_name[source.name] = source
                continue
            if existing.model_dump(mode="json") != source.model_dump(mode="json"):
                raise ValueError(
                    f"source name {source.name!r} appears with conflicting definitions"
                )

    missing = requested.difference(sources_by_name)
    if missing:
        raise ValueError(
            "requested source(s) not found as heuristic attack sources: "
            + ", ".join(sorted(missing))
        )

    return list(sources_by_name.values())


def preclassify_reason_sources_from_spec(
    spec: MirrorSpec,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
    source_names: Sequence[str] | None = None,
) -> list[ReasonPreclassifyReport]:
    """Preclassify every selected heuristic attack source in a spec."""
    reports: list[ReasonPreclassifyReport] = []
    for source in heuristic_attack_sources(spec, source_names=source_names):
        output_path = output_dir / preclassified_source_filename(source)
        reports.append(
            preclassify_reason_source(
                source,
                output_path,
                base_dir=base_dir,
                label="malicious",
            )
        )
    return reports


def reports_to_json(reports: Sequence[ReasonPreclassifyReport]) -> str:
    """Serialize reports for CLI sidecars/tests."""
    return json.dumps(
        [
            {
                "source_name": report.source_name,
                "input_path": report.input_path,
                "output_path": str(report.output_path),
                "rows_read": report.rows_read,
                "rows_after_label_filter": report.rows_after_label_filter,
                "rows_with_text": report.rows_with_text,
                "rows_written": report.rows_written,
                "empty_text": report.empty_text,
                "classified": report.classified,
                "uncategorized": report.uncategorized,
                "keep_rate": report.keep_rate,
                "classified_rate": report.classified_rate,
                "output_sha256": report.output_sha256,
                "by_reason": report.by_reason,
            }
            for report in reports
        ],
        indent=2,
        sort_keys=True,
    )
