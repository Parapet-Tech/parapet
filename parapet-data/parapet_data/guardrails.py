"""Feature-coverage diagnostics for curated corpora."""

from __future__ import annotations


def check_feature_coverage(
    composition: dict,
    min_df: int,
    max_features: int,
    language_quota: dict[str, float] | None = None,
) -> list[str]:
    """Heuristic checks for likely feature loss before vectorization."""
    warnings: list[str] = []
    total = int(composition.get("total", 0))
    by_language_rows = composition.get("by_language", [])
    by_language = {
        str(row.get("name")): int(row.get("count", 0))
        for row in by_language_rows
        if row.get("name")
    }

    if not by_language or set(by_language.keys()) == {"EN"}:
        return warnings

    threshold = min_df * 10
    for lang in sorted(by_language.keys()):
        count = by_language[lang]
        if count < threshold:
            warnings.append(
                f"{lang}: {count} samples is below {threshold} (10x min_df={min_df}); "
                "n-grams may be pruned before training"
            )

    if min_df > 2:
        for lang in sorted(by_language.keys()):
            if lang == "EN":
                continue
            count = by_language[lang]
            if count < 200:
                warnings.append(
                    f"{lang}: {count} samples with min_df={min_df} risks sparse multilingual coverage; "
                    "use min_df<=2 for Tier-1 multilingual runs"
                )

    if language_quota and total > 0:
        for lang in sorted(language_quota.keys()):
            target = int(total * float(language_quota[lang]))
            actual = by_language.get(lang, 0)
            if target <= 0:
                continue
            if actual < target:
                short = target - actual
                pct = (short / target) * 100
                warnings.append(
                    f"{lang}: quota target {target}, actual {actual} (-{pct:.1f}%)"
                )

    utilization_floor = max(500, int(max_features * 0.05))
    if total > 0 and total < utilization_floor:
        warnings.append(
            f"Total samples {total} is low relative to max_features={max_features}; "
            f"feature space may be underutilized (heuristic floor={utilization_floor})"
        )

    return warnings
