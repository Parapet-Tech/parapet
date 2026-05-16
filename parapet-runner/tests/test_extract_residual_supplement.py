"""Tests for parapet-runner/scripts/extract_residual_supplement.py."""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
from pathlib import Path

import yaml


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "extract_residual_supplement.py"
    )
    spec = importlib.util.spec_from_file_location("extract_residual_supplement", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_output_dir(case_name: str) -> Path:
    output_dir = Path("tests/.tmp_outputs") / "extract_residual" / case_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# infer_language
# ---------------------------------------------------------------------------


def test_infer_language_prefix_match():
    m = _load_module()
    assert m.infer_language("ar_attacks_corpus") == "AR"
    assert m.infer_language("ru_russian_prompt_injections") == "RU"
    assert m.infer_language("zh_attacks_novel") == "ZH"
    assert m.infer_language("en_attacks_merged") == "EN"


def test_infer_language_substring_match():
    """Multilingual sources where language is mid-stem (e.g. thewall_..._zh_)."""
    m = _load_module()
    assert m.infer_language("thewall_sql_injection_zh_attacks") == "ZH"
    assert m.infer_language("opensource_corpus_ar_subset") == "AR"
    assert m.infer_language("thewall_ru_jailbreaks") == "RU"


def test_infer_language_falls_through_to_none():
    m = _load_module()
    # No prefix or substring match => None (caller decides default).
    assert m.infer_language("opensource_global_attacks") is None
    assert m.infer_language("multilingual_pi_corpus") is None
    assert m.infer_language("") is None


def test_infer_language_prefix_takes_priority_over_none():
    m = _load_module()
    # Prefix matches even when the rest of the stem looks generic.
    assert m.infer_language("zh_anything_at_all") == "ZH"


# ---------------------------------------------------------------------------
# extract_entry
# ---------------------------------------------------------------------------


def test_extract_entry_uses_explicit_language_field():
    m = _load_module()
    out = m.extract_entry(
        {"content": "x", "label": "malicious", "language": "ru", "reason": "obfuscation",
         "source": "explicit_src"},
        "ar_some_stem",  # stem prefix would say AR, but explicit field wins
    )
    assert out == {
        "content": "x", "label": "malicious", "language": "RU",
        "reason": "obfuscation", "source": "explicit_src",
    }


def test_extract_entry_falls_back_to_stem_inference():
    m = _load_module()
    out = m.extract_entry(
        {"content": "y", "label": "benign"},  # no language field
        "ar_attacks_novel",
    )
    assert out["language"] == "AR"
    assert out["source"] == "ar_attacks_novel"  # falls back to stem


def test_extract_entry_defaults_to_en_when_inference_fails():
    m = _load_module()
    out = m.extract_entry(
        {"content": "z", "label": "benign"},
        "global_corpus",  # no prefix or substring match
    )
    assert out["language"] == "EN"


def test_extract_entry_uppercases_language():
    m = _load_module()
    out = m.extract_entry({"content": "q", "label": "benign", "language": "zh"}, "src")
    assert out["language"] == "ZH"


def test_extract_entry_preserves_reason_when_present():
    m = _load_module()
    out = m.extract_entry(
        {"content": "q", "label": "malicious", "language": "EN", "reason": "meta_probe"},
        "stem",
    )
    assert out["reason"] == "meta_probe"


def test_extract_entry_reason_is_none_when_missing():
    m = _load_module()
    out = m.extract_entry({"content": "q", "label": "malicious"}, "stem")
    assert out["reason"] is None


# ---------------------------------------------------------------------------
# sample_by_language
# ---------------------------------------------------------------------------


def _entries_for_lang(lang: str, n: int) -> list[dict]:
    return [{"content": f"{lang}-{i}", "label": "malicious", "language": lang,
             "source": "src", "reason": None} for i in range(n)]


def test_sample_by_language_respects_quota_when_pools_are_large_enough():
    m = _load_module()
    rng = random.Random(42)
    # Build pools well above quota
    entries = (_entries_for_lang("EN", 1000) + _entries_for_lang("RU", 200)
               + _entries_for_lang("ZH", 200) + _entries_for_lang("AR", 200))
    sampled = m.sample_by_language(entries, target=100,
                                   quota={"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07},
                                   rng=rng)
    counts = {"EN": 0, "RU": 0, "ZH": 0, "AR": 0}
    for r in sampled:
        counts[r["language"]] += 1
    assert counts["EN"] == 75
    assert counts["RU"] == 10
    assert counts["ZH"] == 8
    assert counts["AR"] == 7
    assert len(sampled) == 100


def test_sample_by_language_caps_at_pool_capacity():
    m = _load_module()
    rng = random.Random(42)
    # AR pool has only 3 entries but quota wants 7
    entries = _entries_for_lang("EN", 1000) + _entries_for_lang("AR", 3)
    sampled = m.sample_by_language(
        entries, target=100,
        quota={"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07}, rng=rng,
    )
    counts = {lang: 0 for lang in ("EN", "RU", "ZH", "AR")}
    for r in sampled:
        counts[r["language"]] += 1
    assert counts["AR"] == 3  # capped by pool, not 7


def test_sample_by_language_redistributes_shortfalls_to_surplus():
    m = _load_module()
    rng = random.Random(42)
    # RU/ZH/AR pools empty; EN has plenty.
    # Total target=100 with quota 75/10/8/7 → shortfall 10+8+7=25; backfill from EN.
    entries = _entries_for_lang("EN", 1000)
    sampled = m.sample_by_language(
        entries, target=100,
        quota={"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07}, rng=rng,
    )
    counts = {lang: 0 for lang in ("EN", "RU", "ZH", "AR")}
    for r in sampled:
        counts[r["language"]] += 1
    # All 100 land in EN (75 quota + 25 backfill).
    assert counts["EN"] >= 75
    assert counts["RU"] == 0 and counts["ZH"] == 0 and counts["AR"] == 0


def test_sample_by_language_deterministic_with_seed():
    m = _load_module()
    entries = _entries_for_lang("EN", 100) + _entries_for_lang("RU", 100)
    quota = {"EN": 0.75, "RU": 0.10, "ZH": 0.08, "AR": 0.07}

    s1 = m.sample_by_language(entries, target=20, quota=quota, rng=random.Random(42))
    s2 = m.sample_by_language(entries, target=20, quota=quota, rng=random.Random(42))
    s3 = m.sample_by_language(entries, target=20, quota=quota, rng=random.Random(99))

    ids1 = sorted(r["content"] for r in s1)
    ids2 = sorted(r["content"] for r in s2)
    ids3 = sorted(r["content"] for r in s3)

    assert ids1 == ids2
    assert ids1 != ids3  # different seed -> different sample (almost surely)


def test_sample_by_language_handles_empty_pool():
    m = _load_module()
    sampled = m.sample_by_language(
        [], target=100, quota={"EN": 1.0}, rng=random.Random(0),
    )
    assert sampled == []


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


def _setup_e2e(tmp_path: Path) -> dict:
    """Build a synthetic eval.json + verified-dir that main() can consume."""
    verified_dir = tmp_path / "verified"
    verified_dir.mkdir()

    # Two source files. EN attacks (FN candidates) + EN benign (FP candidates).
    en_attacks = [
        {"id": f"en_attacks_{i}", "content": f"attack-{i}", "label": "malicious",
         "language": "EN", "reason": "instruction_override", "source": "en_attacks"}
        for i in range(20)
    ]
    en_benign = [
        {"id": f"en_benign_{i}", "content": f"benign-{i}", "label": "benign",
         "language": "EN", "reason": None, "source": "en_benign"}
        for i in range(20)
    ]
    (verified_dir / "en_attacks.yaml").write_text(
        yaml.safe_dump(en_attacks, allow_unicode=True), encoding="utf-8"
    )
    (verified_dir / "en_benign.yaml").write_text(
        yaml.safe_dump(en_benign, allow_unicode=True), encoding="utf-8"
    )

    # eval.json: 5 FN (incorrect on attacks) + 5 FP (incorrect on benign).
    results = []
    for i in range(20):
        results.append({
            "case_id": f"en_attacks_{i}", "source": "en_attacks",
            "label": "malicious", "correct": i >= 5,  # first 5 are FN
        })
    for i in range(20):
        results.append({
            "case_id": f"en_benign_{i}", "source": "en_benign",
            "label": "benign", "correct": i >= 5,  # first 5 are FP
        })
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"results": results}), encoding="utf-8")

    output_path = tmp_path / "supplement.yaml"
    return {
        "eval_path": eval_path,
        "verified_dir": verified_dir,
        "output_path": output_path,
    }


def test_main_writes_balanced_supplement(monkeypatch, capsys):
    """End-to-end: synthetic eval.json + verified yamls → balanced supplement output."""
    m = _load_module()
    tmp_path = _new_output_dir("main_e2e")
    paths = _setup_e2e(tmp_path)

    import sys
    monkeypatch.setattr(sys, "argv", [
        "extract_residual_supplement.py",
        "--eval-json", str(paths["eval_path"]),
        "--verified-dir", str(paths["verified_dir"]),
        "--output", str(paths["output_path"]),
        "--total", "10",
        "--seed", "42",
    ])
    m.main()

    assert paths["output_path"].exists()
    with paths["output_path"].open(encoding="utf-8") as f:
        out = yaml.safe_load(f)
    assert isinstance(out, list)

    # Total ≤ available FN+FP (5 each = 10). Balanced: ~5 attacks + 5 benign.
    labels = [r["label"] for r in out]
    assert labels.count("malicious") <= 5
    assert labels.count("benign") <= 5
    # All EN
    assert all(r["language"] == "EN" for r in out)


def test_main_skips_unparseable_yaml(monkeypatch, capsys):
    """Malformed YAML in verified-dir should be skipped, not crash."""
    m = _load_module()
    tmp_path = _new_output_dir("main_skips_bad_yaml")
    paths = _setup_e2e(tmp_path)
    # Drop a malformed YAML file alongside the good ones.
    (paths["verified_dir"] / "en_broken.yaml").write_text(
        ":\n - this is not\n  - valid yaml\n   :\n", encoding="utf-8"
    )

    import sys
    monkeypatch.setattr(sys, "argv", [
        "extract_residual_supplement.py",
        "--eval-json", str(paths["eval_path"]),
        "--verified-dir", str(paths["verified_dir"]),
        "--output", str(paths["output_path"]),
        "--total", "4",
        "--seed", "0",
    ])
    # Should not raise; should produce output despite the broken file.
    m.main()
    assert paths["output_path"].exists()
