"""Tests for feature coverage guardrails."""

from parapet_data.guardrails import check_feature_coverage


def _composition(by_language: dict[str, int]) -> dict:
    total = sum(by_language.values())
    return {
        "total": total,
        "by_language": [
            {"name": lang, "count": count, "pct": round((count / total) * 100, 2)}
            for lang, count in by_language.items()
        ] if total > 0 else [],
    }


def test_warns_when_language_below_threshold() -> None:
    warnings = check_feature_coverage(
        composition=_composition({"EN": 1000, "RU": 12}),
        min_df=5,
        max_features=10000,
    )
    assert any("RU: 12 samples is below 50" in w for w in warnings)


def test_no_warning_for_en_only() -> None:
    warnings = check_feature_coverage(
        composition=_composition({"EN": 1000}),
        min_df=5,
        max_features=10000,
    )
    assert warnings == []


def test_warning_includes_language_and_count() -> None:
    warnings = check_feature_coverage(
        composition=_composition({"EN": 500, "ZH": 15}),
        min_df=3,
        max_features=10000,
    )
    assert any("ZH: 15" in w for w in warnings)


def test_empty_when_support_is_sufficient() -> None:
    warnings = check_feature_coverage(
        composition=_composition({"EN": 2000, "RU": 400, "ZH": 300, "AR": 250}),
        min_df=2,
        max_features=10000,
    )
    assert warnings == []
