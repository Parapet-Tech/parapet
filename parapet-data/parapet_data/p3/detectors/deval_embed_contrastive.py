"""D_eval member: CONTRASTIVE embedding-distance detector (embedding family).

Two-sided successor to deval_embed.DEvalEmbed (adopted by author decision 2026-06-10
as the default embedding-side member after the discrimination-rescue diagnostic,
runs/p3_pilot/audit/deval_discrimination/REPORT.md). The one-sided member scored
"how close is this event to the nearest known control-plane payload", which is
style-dominated: native benign agentic action/comms text (tool outputs, transfer
memos, send-message arguments) sits close to control-plane payloads in embedding
space, producing a SYSTEMIC benign false-positive tail (39% of native benign events
over tau; 77.5% of benign carriers tripping the peak rule).

The contrastive member scores the per-event raw signal as a MARGIN:

    margin = max_cos(event, control_plane_refs) - max_cos(event, benign_refs)

where benign_refs are hard benign action/comms negatives mined from held-out
(reference-half) carriers. Subtracting the nearest-benign similarity removes the
shared "operational agentic text" component, which collapses the benign tail
(diagnostic: benign carrier peak-FP 77.5% -> 2.2%; naive-attack AUC 0.75 -> 0.977)
while leaving control-plane payload similarity intact.

Chunk-and-max: each event is chunked (injected chunk_fn, identity by default) and the
margin is the MAX over PER-CHUNK margins, so a control-plane chunk buried in a long
benign tool output still fires even when the rest of the span is benign-similar.
(Per-chunk margin, not per-side max-then-subtract: the latter lets benign chunks of
the same event cancel a hot chunk. Single-chunk events are identical either way.)
Both reference banks are chunked the same way at construction.

Reference governance: both banks come from the FROZEN contrastive reference artifact
(runs/p3_pilot/reference_set/contrastive_reference.json, p3-contrastive-reference/3):
control half unchanged from the v2 freeze; benign half mined from REF-half carriers
only, normalized near-duplicate-excluded against eval events and the attack pool.
The generator MUST NOT be conditioned on either bank.

KNOWN LIMIT (recorded, not a regression): cloaked slow-burn grafts optimized to read
as benign operational text are near-invisible to ANY similarity view, including this
one (margins at/below the benign median). This member fixes the benign side; attack
sensitivity to style-cloaked content is the job of the Option C content judge.

FAILS CLOSED exactly like DEvalEmbed: embed_fn error, wrong return count, zero
vector, dim mismatch, chunk failure -> error DetectorResults (score=None), never a
silent 0.0.
"""
from __future__ import annotations

from typing import Optional, Sequence

from parapet_data.p3.detectors.deval_embed import (
    ChunkFn,
    EmbedFn,
    _identity_chunk,
    _l2_normalize,
    calibrate,
)
from parapet_data.p3.detectors.interface import (
    FAMILY_EMBEDDING,
    DetectorResult,
    EventContext,
)

# Provisional sigmoid over the raw MARGIN (in [-2, 2]), fit on the 2026-06-10
# discrimination diagnostic (naive_attack vs eval-half native benign; class-imbalance
# squashed, rank order is the deliverable). Re-fit at the next tau lock.
CONTRAST_SIGMOID_A = 23.72
CONTRAST_SIGMOID_B = 0.16

DETECTOR_ID = "deval_embed_contrastive"
DEFAULT_MODEL_ID = "mlx-embed-unset"


def _build_bank(embed_fn: EmbedFn, chunk_fn: ChunkFn, texts: Sequence[str], name: str) -> list:
    """Chunk + embed + normalize one reference bank at construction. Misconfiguration
    (empty bank, backend failure, zero vector) fails LOUD here, never per-event."""
    kept = [t for t in (texts or []) if t and str(t).strip()]
    if not kept:
        raise ValueError(f"{name} reference texts must be non-empty")
    chunks = []
    for t in kept:
        chunks.extend(c for c in chunk_fn(t) if c and str(c).strip())
    if not chunks:
        raise ValueError(f"{name} reference texts produced no non-empty chunks")
    vecs = list(embed_fn(chunks))
    if len(vecs) != len(chunks):
        raise ValueError(f"embed_fn returned {len(vecs)} vectors for {len(chunks)} {name} chunks")
    bank = []
    for v in vecs:
        nv = _l2_normalize(v)
        if nv is None:
            raise ValueError(f"embed_fn produced an empty or zero {name} reference vector")
        bank.append(nv)
    dim = len(bank[0])
    if any(len(v) != dim for v in bank):
        raise ValueError(f"{name} reference embeddings have inconsistent dimensions")
    return bank


class DEvalEmbedContrastive:
    family = FAMILY_EMBEDDING
    detector_id = DETECTOR_ID

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        control_texts: Sequence[str],
        benign_texts: Sequence[str],
        model_id: str = DEFAULT_MODEL_ID,
        chunk_fn: Optional[ChunkFn] = None,
        sigmoid_a: float = CONTRAST_SIGMOID_A,
        sigmoid_b: float = CONTRAST_SIGMOID_B,
    ):
        if embed_fn is None:
            raise ValueError("embed_fn is required")
        overlap = {t for t in (control_texts or [])} & {t for t in (benign_texts or [])}
        if overlap:
            raise ValueError(
                f"{len(overlap)} texts appear in BOTH reference banks (misconfigured freeze)"
            )
        self.embed_fn = embed_fn
        self.chunk_fn = chunk_fn if chunk_fn is not None else _identity_chunk
        self.model_id = model_id
        self.sigmoid_a = sigmoid_a
        self.sigmoid_b = sigmoid_b
        self._ctrl = _build_bank(embed_fn, self.chunk_fn, control_texts, "control")
        self._ben = _build_bank(embed_fn, self.chunk_fn, benign_texts, "benign")
        self._dim = len(self._ctrl[0])
        if len(self._ben[0]) != self._dim:
            raise ValueError("control and benign reference banks have different dimensions")

    def _result(self, score, error, rationale=None) -> DetectorResult:
        return DetectorResult(
            score=score, family=self.family, model_id=self.model_id,
            detector_id=self.detector_id, rationale=rationale, error=error,
        )

    @staticmethod
    def _max_cos(qn: list, bank: list) -> float:
        best = -1.0
        for r in bank:
            dot = 0.0
            for a, b in zip(qn, r):
                dot += a * b
            if dot > best:
                best = dot
        return best

    def score(self, event_text: str, context: Optional[EventContext] = None) -> DetectorResult:
        return self.score_batch([event_text], [context])[0]

    def score_batch(self, texts: list, contexts=None) -> list:
        n = len(texts)
        results: list = [None] * n
        chunk_texts: list = []
        owner: list = []
        for i, t in enumerate(texts):
            if not t or not str(t).strip():
                results[i] = self._result(None, "empty_event_text")
                continue
            try:
                chunks = [c for c in self.chunk_fn(t) if c and str(c).strip()]
            except Exception as exc:
                results[i] = self._result(None, f"chunk_failed: {type(exc).__name__}")
                continue
            if not chunks:
                results[i] = self._result(None, "empty_event_text")
                continue
            for c in chunks:
                chunk_texts.append(c)
                owner.append(i)
        if not chunk_texts:
            return results

        try:
            vecs = list(self.embed_fn(chunk_texts))
        except Exception as exc:
            return self._fill(results, sorted(set(owner)), f"embed_failed: {type(exc).__name__}")
        if len(vecs) != len(chunk_texts):
            return self._fill(
                results, sorted(set(owner)),
                f"embed_count_mismatch: got {len(vecs)} for {len(chunk_texts)}",
            )

        by_event: dict = {}
        for pos, i in enumerate(owner):
            by_event.setdefault(i, []).append(pos)

        for i, positions in by_event.items():
            best_margin = None
            best_ctrl = best_ben = None
            err = None
            for pos in positions:
                qn = _l2_normalize(vecs[pos])
                if qn is None:
                    err = "embed_zero_vector"
                    break
                if len(qn) != self._dim:
                    err = f"embed_dim_mismatch: {len(qn)} != ref {self._dim}"
                    break
                c = self._max_cos(qn, self._ctrl)
                b = self._max_cos(qn, self._ben)
                m = c - b
                if best_margin is None or m > best_margin:
                    best_margin, best_ctrl, best_ben = m, c, b
            if err:
                results[i] = self._result(None, err)
                continue
            rationale = (f"contrastive margin {best_margin:.4f} "
                         f"(ctrl {best_ctrl:.4f} over {len(self._ctrl)} refs, "
                         f"benign {best_ben:.4f} over {len(self._ben)} refs)")
            if len(positions) > 1:
                rationale += f" ({len(positions)} chunks)"
            results[i] = self._result(
                calibrate(best_margin, self.sigmoid_a, self.sigmoid_b), None,
                rationale=rationale,
            )
        return results

    def _fill(self, results: list, idxs: list, error: str) -> list:
        for i in idxs:
            results[i] = self._result(None, error)
        return results
