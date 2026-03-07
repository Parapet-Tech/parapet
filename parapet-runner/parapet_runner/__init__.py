"""Parapet runner package."""

from __future__ import annotations

from .config import ThresholdPolicy, TrainConfig
from .manifest import (
    BaselineFamily,
    CurationManifest,
    EvalResult,
    RunManifest,
    RuntimeIdentity,
    compute_metric_delta,
    compute_semantic_parity_hash,
)

_RUNNER_EXPORTS = {
    "BaselineRun",
    "CompositeBaselineProvider",
    "ExperimentDependencies",
    "ExperimentRunner",
    "F1GridSearchCalibrator",
    "ManifestSplitResolver",
    "OutputHashVerifier",
    "ParapetEvalEvaluator",
    "ParapetEvalPG2BaselineProvider",
    "ProtectAIBaselineProvider",
    "ResolvedSplits",
    "RuntimeIdentityCollector",
    "StaticSplitResolver",
    "TrainScriptTrainer",
    "YamlErrorAnalyzer",
}


def __getattr__(name: str):
    # Avoid importing parapet_runner.runner during package import. This keeps
    # `python -m parapet_runner.runner ...` from preloading the module.
    if name in _RUNNER_EXPORTS:
        from . import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaselineFamily",
    "BaselineRun",
    "CompositeBaselineProvider",
    "CurationManifest",
    "EvalResult",
    "ExperimentDependencies",
    "ExperimentRunner",
    "F1GridSearchCalibrator",
    "ManifestSplitResolver",
    "OutputHashVerifier",
    "ParapetEvalEvaluator",
    "ParapetEvalPG2BaselineProvider",
    "ProtectAIBaselineProvider",
    "ResolvedSplits",
    "RunManifest",
    "RuntimeIdentity",
    "RuntimeIdentityCollector",
    "StaticSplitResolver",
    "ThresholdPolicy",
    "TrainScriptTrainer",
    "TrainConfig",
    "YamlErrorAnalyzer",
    "compute_metric_delta",
    "compute_semantic_parity_hash",
]
