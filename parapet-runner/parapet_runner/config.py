"""Training config contracts for runner orchestration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, model_validator


class ThresholdPolicy(str, Enum):
    """How the classifier decision threshold is selected."""

    FIXED = "fixed"
    CALIBRATE_F1 = "calibrate_f1"


class TrainConfig(BaseModel):
    """Config for one training run."""

    specialist: str = "generalist"
    analyzer: str = "char_wb"
    ngram_min: int = 3
    ngram_max: int = 5
    max_features: int = 25_000
    min_df: int = 5
    c: float = 0.1
    class_weight: Literal["balanced", "none"] = "none"
    max_iter: int = 100_000
    cv_folds: int = 0
    cv_max_samples: int = 120_000
    prune_threshold: float = 0.05
    seed: int = 42
    mode: Literal["iteration", "final"] = "iteration"
    squash_augment: bool = False
    apply_l0_transform: bool = True
    threshold_policy: ThresholdPolicy = ThresholdPolicy.FIXED
    threshold_value: float = Field(default=-0.5)

    @model_validator(mode="after")
    def validate_contract(self) -> "TrainConfig":
        if self.ngram_min <= 0 or self.ngram_max <= 0:
            raise ValueError("ngram_min and ngram_max must be positive")
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min cannot exceed ngram_max")
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.prune_threshold < 0:
            raise ValueError("prune_threshold must be >= 0")

        if self.mode == "iteration":
            if self.cv_folds != 0:
                raise ValueError("iteration mode requires cv_folds=0")
            if self.max_features > 25_000:
                raise ValueError("iteration mode expects max_features <= 25000")
        else:
            if self.cv_folds < 3:
                raise ValueError("final mode requires cv_folds >= 3")
            if self.max_features < 25_000:
                raise ValueError("final mode expects max_features >= 25000")

        if self.threshold_policy == ThresholdPolicy.FIXED:
            if not -10.0 <= self.threshold_value <= 10.0:
                raise ValueError("threshold_value must be in [-10, 10]")
        return self

    def to_train_script_args(
        self,
        *,
        attack_files: Sequence[Path],
        benign_files: Sequence[Path],
        out_path: Path,
        holdout_path: Path,
    ) -> list[str]:
        """Render CLI args for scripts/train_l1_specialist.py."""

        if not attack_files:
            raise ValueError("attack_files cannot be empty")
        if not benign_files:
            raise ValueError("benign_files cannot be empty")

        args: list[str] = [
            "--specialist",
            self.specialist,
            "--attack-files",
            *[str(p) for p in attack_files],
            "--benign-files",
            *[str(p) for p in benign_files],
            "--ngram-min",
            str(self.ngram_min),
            "--ngram-max",
            str(self.ngram_max),
            "--max-features",
            str(self.max_features),
            "--min-df",
            str(self.min_df),
            "--analyzer",
            self.analyzer,
            "--c",
            str(self.c),
            "--class-weight",
            self.class_weight,
            "--max-iter",
            str(self.max_iter),
            "--cv-folds",
            str(self.cv_folds),
            "--cv-max-samples",
            str(self.cv_max_samples),
            "--prune-threshold",
            str(self.prune_threshold),
            "--seed",
            str(self.seed),
            "--out",
            str(out_path),
            "--holdout-out",
            str(holdout_path),
        ]
        if self.squash_augment:
            args.append("--squash-augment")
        if self.apply_l0_transform:
            args.append("--apply-l0-transform")
        return args
