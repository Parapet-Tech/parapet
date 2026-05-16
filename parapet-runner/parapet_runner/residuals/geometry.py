"""L2 score-geometry analysis for residual rows."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable


MENTION_RAW_DELTA_THRESHOLD = 1.0
DEFAULT_BORDERLINE_BAND = 0.5
DEFAULT_SWEEP_BANDS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
FALSE_NEGATIVE = "false_negative"
HARD_NEGATIVE_CATEGORIES = frozenset({"false_positive", "near_boundary_benign"})
BASELINE_CORRECT = "baseline_correct"


def pct(n: int | float, total: int | float) -> float:
    return 0.0 if not total else float(n) / float(total)


def to_float(value: Any, *, bool_as_float: bool = False) -> float | None:
    if isinstance(value, bool):
        if bool_as_float:
            return 1.0 if value else 0.0
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def threshold(row: dict[str, Any]) -> float:
    thresholds = row.get("l1_thresholds")
    if isinstance(thresholds, dict):
        value = to_float(thresholds.get("l1"))
        if value is not None:
            return value
    return 0.0


def margin(row: dict[str, Any], field: str = "raw_score") -> float | None:
    value = to_float(row.get(field))
    if value is None:
        return None
    return value - threshold(row)


def current_effective_raw(row: dict[str, Any]) -> float | None:
    raw = to_float(row.get("raw_score"))
    if raw is None:
        return None
    unquoted = to_float(row.get("raw_unquoted_score"))
    raw_delta = to_float(row.get("raw_score_delta"))
    mention = (
        row.get("quote_detected") is True
        and raw_delta is not None
        and raw_delta > MENTION_RAW_DELTA_THRESHOLD
    )
    if mention and unquoted is not None:
        return unquoted
    return raw


def current_effective_margin(row: dict[str, Any]) -> float | None:
    value = current_effective_raw(row)
    if value is None:
        return None
    return value - threshold(row)


def blocks_current_effective(row: dict[str, Any]) -> bool:
    value = current_effective_raw(row)
    return value is not None and value >= threshold(row)


def blocks_raw_only(row: dict[str, Any]) -> bool:
    value = to_float(row.get("raw_score"))
    return value is not None and value >= threshold(row)


def blocks_borderline_squash(row: dict[str, Any], *, borderline_band: float) -> bool:
    if blocks_current_effective(row):
        return True

    raw_margin = margin(row, "raw_score")
    squash = to_float(row.get("raw_squash_score"))
    if raw_margin is None or squash is None:
        return False
    return abs(raw_margin) <= borderline_band and squash >= threshold(row)


def blocks_borderline_squash_gated(
    row: dict[str, Any],
    *,
    borderline_band: float,
    language: str | None,
) -> bool:
    if blocks_current_effective(row):
        return True
    if language is not None and str(row.get("language", "")).upper() != language.upper():
        return False
    return blocks_borderline_squash(row, borderline_band=borderline_band)


def blocks_max_raw_squash(row: dict[str, Any]) -> bool:
    raw = to_float(row.get("raw_score"))
    squash = to_float(row.get("raw_squash_score"))
    values = [v for v in (raw, squash) if v is not None]
    return bool(values) and max(values) >= threshold(row)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    description: str
    decide: Callable[[dict[str, Any]], bool]


def default_policy_specs(borderline_band: float = DEFAULT_BORDERLINE_BAND) -> list[PolicySpec]:
    """Return built-in candidate policies."""

    return [
        PolicySpec(
            "current_effective",
            "Current runtime-equivalent raw/unquoted decision.",
            blocks_current_effective,
        ),
        PolicySpec(
            "raw_only",
            "Use raw margin only.",
            blocks_raw_only,
        ),
        PolicySpec(
            "squash_when_raw_borderline",
            f"Use squash only when raw margin is within +/-{borderline_band}.",
            lambda row: blocks_borderline_squash(row, borderline_band=borderline_band),
        ),
        PolicySpec(
            "max_raw_squash",
            "Analysis-only upper bound: max(raw, squash).",
            blocks_max_raw_squash,
        ),
    ]


def borderline_squash_policy_spec(
    *,
    borderline_band: float,
    language: str | None = None,
) -> PolicySpec:
    if language is None:
        name = f"squash_borderline_band_{borderline_band:g}_all"
        description = f"Use squash only when raw margin is within +/-{borderline_band}."
    else:
        lang = language.upper()
        name = f"squash_borderline_band_{borderline_band:g}_{lang.lower()}"
        description = (
            f"Use squash only for {lang} rows when raw margin is within "
            f"+/-{borderline_band}."
        )
    return PolicySpec(
        name,
        description,
        lambda row: blocks_borderline_squash_gated(
            row,
            borderline_band=borderline_band,
            language=language,
        ),
    )


def concentration(rows: list[dict[str, Any]], field: str, top: int = 10) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row.get(field, "<missing>")) for row in rows)
    shares = [pct(count, total) for count in counts.values()]
    hhi = sum(share * share for share in shares)
    top_items = counts.most_common(top)
    return {
        "field": field,
        "n": total,
        "unique": len(counts),
        "top_share": pct(top_items[0][1], total) if top_items else 0.0,
        "top5_share": sum(pct(count, total) for _, count in top_items[:5]),
        "top10_share": sum(pct(count, total) for _, count in top_items[:10]),
        "hhi": hhi,
        "effective_n": None if hhi == 0 else 1.0 / hhi,
        "top": [
            {"value": value, "count": count, "share": pct(count, total)}
            for value, count in top_items
        ],
    }


def _breakdown(
    residual_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    decisions: dict[int, bool],
    sidecar_decisions: dict[int, bool],
    field: str,
) -> list[dict[str, Any]]:
    residual_groups: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    sidecar_groups: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)

    for i, row in enumerate(residual_rows):
        residual_groups[str(row.get(field, "<missing>"))].append((row, decisions[i]))
    for i, row in enumerate(sidecar_rows):
        if row.get("label") != "benign":
            continue
        sidecar_groups[str(row.get(field, "<missing>"))].append((row, sidecar_decisions[i]))

    values = sorted(set(residual_groups) | set(sidecar_groups))
    out: list[dict[str, Any]] = []
    for value in values:
        residual_group = residual_groups.get(value, [])
        sidecar_group = sidecar_groups.get(value, [])
        fn = [(r, d) for r, d in residual_group if r.get("residual_category") == FALSE_NEGATIVE]
        hard = [(r, d) for r, d in residual_group if r.get("residual_category") in HARD_NEGATIVE_CATEGORIES]
        sidecar_blocked = sum(1 for _, decision in sidecar_group if decision)
        recovered = sum(1 for _, decision in fn if decision)
        hard_blocked = sum(1 for _, decision in hard if decision)
        out.append({
            "value": value,
            "false_negative_n": len(fn),
            "false_negative_recovered": recovered,
            "false_negative_recovered_share": pct(recovered, len(fn)),
            "hard_negative_n": len(hard),
            "hard_negative_blocks": hard_blocked,
            "hard_negative_block_share": pct(hard_blocked, len(hard)),
            "sidecar_n": len(sidecar_group),
            "sidecar_blocks": sidecar_blocked,
            "sidecar_block_share": pct(sidecar_blocked, len(sidecar_group)),
        })
    return out


def _category_breakdown(
    residual_rows: list[dict[str, Any]],
    decisions: dict[int, bool],
    current_decisions: dict[int, bool],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[dict[str, Any], bool, bool]]] = defaultdict(list)
    for i, row in enumerate(residual_rows):
        groups[str(row.get("residual_category", "<missing>"))].append(
            (row, decisions[i], current_decisions[i])
        )

    out: list[dict[str, Any]] = []
    for category in sorted(groups):
        group = groups[category]
        blocks = sum(1 for _, decision, _ in group if decision)
        added_blocks = sum(1 for _, decision, current in group if decision and not current)
        out.append({
            "residual_category": category,
            "n": len(group),
            "blocks": blocks,
            "block_share": pct(blocks, len(group)),
            "added_blocks": added_blocks,
            "added_block_share": pct(added_blocks, len(group)),
        })
    return out


def evaluate_policy(
    residual_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    spec: PolicySpec,
) -> dict[str, Any]:
    """Evaluate one policy at its operating point."""

    decisions = {i: bool(spec.decide(row)) for i, row in enumerate(residual_rows)}
    current_decisions = {i: blocks_current_effective(row) for i, row in enumerate(residual_rows)}
    sidecar_decisions = {i: bool(spec.decide(row)) for i, row in enumerate(sidecar_rows)}
    benign_sidecar_rows = [
        (i, row) for i, row in enumerate(sidecar_rows)
        if row.get("label") == "benign"
    ]

    false_negative_rows = [
        (i, row) for i, row in enumerate(residual_rows)
        if row.get("residual_category") == FALSE_NEGATIVE
    ]
    hard_negative_rows = [
        (i, row) for i, row in enumerate(residual_rows)
        if row.get("residual_category") in HARD_NEGATIVE_CATEGORIES
    ]

    recovered = [(i, row) for i, row in false_negative_rows if decisions[i]]
    hard_blocks = [(i, row) for i, row in hard_negative_rows if decisions[i]]
    added_hard_blocks = [
        (i, row) for i, row in hard_negative_rows
        if decisions[i] and not current_decisions[i]
    ]
    sidecar_blocks = [
        (i, row) for i, row in benign_sidecar_rows
        if sidecar_decisions[i]
    ]

    recovered_rows = [row for _, row in recovered]

    return {
        "name": spec.name,
        "description": spec.description,
        "false_negative_n": len(false_negative_rows),
        "false_negative_recovered": len(recovered),
        "false_negative_recovered_share": pct(len(recovered), len(false_negative_rows)),
        "hard_negative_n": len(hard_negative_rows),
        "hard_negative_blocks": len(hard_blocks),
        "hard_negative_block_share": pct(len(hard_blocks), len(hard_negative_rows)),
        "added_hard_negative_blocks": len(added_hard_blocks),
        "added_hard_negative_block_share": pct(len(added_hard_blocks), len(hard_negative_rows)),
        "sidecar_total_n": len(sidecar_rows),
        "sidecar_n": len(benign_sidecar_rows),
        "sidecar_blocks": len(sidecar_blocks),
        "sidecar_block_share": pct(len(sidecar_blocks), len(benign_sidecar_rows)),
        "recovered_fn_source_concentration": concentration(recovered_rows, "source", top=10),
        "by_residual_category": _category_breakdown(residual_rows, decisions, current_decisions),
        "by_language": _breakdown(residual_rows, sidecar_rows, decisions, sidecar_decisions, "language"),
        "by_reason": _breakdown(residual_rows, sidecar_rows, decisions, sidecar_decisions, "reason"),
        "by_source": _breakdown(residual_rows, sidecar_rows, decisions, sidecar_decisions, "source"),
        "by_format_bin": _breakdown(residual_rows, sidecar_rows, decisions, sidecar_decisions, "format_bin"),
    }


def evaluate_policies(
    residual_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    specs: Iterable[PolicySpec],
) -> list[dict[str, Any]]:
    return [evaluate_policy(residual_rows, sidecar_rows, spec) for spec in specs]


def _category_row(policy_result: dict[str, Any], category: str) -> dict[str, Any]:
    for row in policy_result["by_residual_category"]:
        if row["residual_category"] == category:
            return row
    return {
        "residual_category": category,
        "n": 0,
        "blocks": 0,
        "block_share": 0.0,
        "added_blocks": 0,
        "added_block_share": 0.0,
    }


def borderline_squash_sweep(
    residual_rows: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    *,
    bands: Iterable[float] = DEFAULT_SWEEP_BANDS,
    languages: Iterable[str | None] = (None, "EN"),
) -> list[dict[str, Any]]:
    """Evaluate squash-borderline policies across window widths and gates."""

    rows: list[dict[str, Any]] = []
    for language in languages:
        for band in bands:
            result = evaluate_policy(
                residual_rows,
                sidecar_rows,
                borderline_squash_policy_spec(borderline_band=float(band), language=language),
            )
            near = _category_row(result, "near_boundary_benign")
            recovered = result["false_negative_recovered"]
            added_near = near["added_blocks"]
            rows.append({
                "language_gate": "ALL" if language is None else language.upper(),
                "borderline_band": float(band),
                "false_negative_recovered": recovered,
                "false_negative_recovered_share": result["false_negative_recovered_share"],
                "added_near_boundary_benign_blocks": added_near,
                "added_near_boundary_benign_share": near["added_block_share"],
                "sidecar_blocks": result["sidecar_blocks"],
                "sidecar_n": result["sidecar_n"],
                "sidecar_block_share": result["sidecar_block_share"],
                "min_fn_value_for_break_even": None if recovered == 0 else added_near / recovered,
                "source_hhi": result["recovered_fn_source_concentration"]["hhi"],
            })
    return rows


def rank_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Compute rank AUC for positive labels.

    Ties receive average rank. Returns None when only one class is present.
    """

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    indexed = sorted(enumerate(scores), key=lambda pair: pair[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def scalar_auc_for_field(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """AUC for false_negative vs residual hard negatives for one scalar field."""

    scores: list[float] = []
    labels: list[bool] = []
    for row in rows:
        category = row.get("residual_category")
        if category != FALSE_NEGATIVE and category not in HARD_NEGATIVE_CATEGORIES:
            continue
        value = to_float(row.get(field), bool_as_float=True)
        if value is None:
            continue
        scores.append(value)
        labels.append(category == FALSE_NEGATIVE)

    auc = rank_auc(scores, labels)
    return {
        "field": field,
        "n": len(scores),
        "auc": auc,
        "separation": None if auc is None else abs(auc - 0.5),
    }
