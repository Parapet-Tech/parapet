"""Shared types for P3 per-event detectors.

A detector scores ONE provenance-span event for surface_signal in [0,1] (how
strongly the event reads as a control-plane move to a single-event reader). It does
NOT produce liveness; liveness is mechanical and lives on the parapet_data.p3.carriers normalized
artifact. The cross-family invariant (D_gen.family not in D_eval families) is checked
on the `family` string, so keep family tags distinct per family.

See local-llm/.local/multiturn/detector_ensemble_spec.md sections 1 and 4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Protocol, runtime_checkable

# Family tags. D_gen is the MLX judge ("generative-mlx"); D_eval members are L1
# ("linear") and the embedding-distance detector ("embedding", later step). Keep
# these distinct so the cross-family invariant (D_gen.family not in D_eval families) holds.
FAMILY_GENERATIVE_MLX = "generative-mlx"
FAMILY_LINEAR = "linear"
FAMILY_EMBEDDING = "embedding"


@dataclass
class EventContext:
    """Light context for a per-event score. All optional; step 1 populates a subset."""
    function: Optional[str] = None        # tool being called: context only, NOT scored as text
    provenance_tag: Optional[str] = None  # principal_user | tool_output (unset in step 1)
    position: Optional[int] = None        # linear position in the carrier trace
    prior_text: Optional[str] = None      # preceding span text, for context-aware members


@dataclass
class DetectorResult:
    score: Optional[float]   # calibrated surface_signal in [0,1]; None on error
    family: str
    model_id: str
    detector_id: str
    rationale: Optional[str] = None
    error: Optional[str] = None
    # Provenance for subprocess-backed detectors (e.g. the L1 Rust binary); None for
    # in-process detectors. Records WHICH build produced the score.
    engine_sha: Optional[str] = None
    binary_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class Detector(Protocol):
    """A per-event surface_signal scorer.

    Detectors expose both single (`score`) and batch (`score_batch`) entry points.
    Batch matters for subprocess-backed detectors (the L1 binary), which must score
    many events in one process invocation, not one process per event.
    """
    family: str
    detector_id: str

    def score(self, event_text: str, context: Optional[EventContext] = None) -> DetectorResult:
        ...

    def score_batch(
        self,
        texts: list,
        contexts: Optional[list] = None,
    ) -> list:
        ...


def loop_score_batch(detector, texts: list, contexts: Optional[list] = None) -> list:
    """Default batch: loop `score`. For per-event detectors with no native batching."""
    ctxs = contexts if contexts is not None else [None] * len(texts)
    return [detector.score(t, c) for t, c in zip(texts, ctxs)]


def clamp_unit(x: float) -> float:
    """Clamp to [0,1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
