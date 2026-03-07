from __future__ import annotations

from pathlib import Path

import pytest

from parapet_runner.config import ThresholdPolicy, TrainConfig


def test_iteration_mode_rejects_cv_folds() -> None:
    with pytest.raises(ValueError, match="cv_folds=0"):
        TrainConfig(mode="iteration", cv_folds=3)


def test_iteration_mode_rejects_large_feature_set() -> None:
    with pytest.raises(ValueError, match="max_features <= 15000"):
        TrainConfig(mode="iteration", max_features=20_000)


def test_final_mode_requires_cv() -> None:
    with pytest.raises(ValueError, match="cv_folds >= 3"):
        TrainConfig(mode="final", cv_folds=0, max_features=25_000)


def test_final_mode_requires_feature_floor() -> None:
    with pytest.raises(ValueError, match="max_features >= 25000"):
        TrainConfig(mode="final", cv_folds=3, max_features=15_000)


def test_fixed_threshold_bounds() -> None:
    with pytest.raises(ValueError, match="threshold_value"):
        TrainConfig(
            mode="iteration",
            cv_folds=0,
            max_features=15_000,
            threshold_policy=ThresholdPolicy.FIXED,
            threshold_value=99.0,
        )


def test_variant_script_args_include_required_flags() -> None:
    cfg = TrainConfig(mode="iteration", cv_folds=0, max_features=15_000)
    args = cfg.to_train_script_args(
        attack_files=[Path("attacks.yaml")],
        benign_files=[Path("benign.yaml")],
        out_path=Path("out.rs"),
        holdout_path=Path("holdout.yaml"),
    )
    assert "--apply-l0-transform" in args
    assert "--specialist" in args


def test_variant_script_args_reject_empty_inputs() -> None:
    cfg = TrainConfig(mode="iteration", cv_folds=0, max_features=15_000)
    with pytest.raises(ValueError, match="attack_files"):
        cfg.to_train_script_args(
            attack_files=[],
            benign_files=[Path("benign.yaml")],
            out_path=Path("out.rs"),
            holdout_path=Path("holdout.yaml"),
        )
