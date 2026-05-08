# L2a: Historical Payload Analysis Slot

`L2a` is a legacy optional payload-analysis slot. It is not the current
strategic path after the 2026-05-08 inline-transformer closure decision.

Current target taxonomy:

- target `L1`: deterministic pattern gate
- target `L2`: lightweight lexical classifier, currently implemented under
  legacy `L1` names
- target `L3`: orthogonal sensors and deterministic router

See `strategy/layers.md` for the active layer map.

## Historical Role

`L2a` was originally Parapet's optional heavier-weight analysis slot for
untrusted payloads and routed traffic.

Typical inputs were:

- tool results
- retrieved documents
- API responses
- other untrusted payload-like text
- traffic routed from the lightweight classifier when deeper analysis was worth
  the latency

Prompt Guard 2 and later semantic-transformer candidates lived in or near this
slot during earlier research.

## Closure

The slot is closed as a hot-path strategy because:

1. Prompt Guard 2 is not the long-term Parapet-owned detection path.
2. MiniLM-class inline transformers only fit latency under constraints that
   failed residual effectiveness.
3. DeBERTa-class models missed the CPU latency budget by a wide margin.
4. The target runtime has no specialist/escalation branch.

Future work should not add a new model under `L2a` unless the layer strategy is
explicitly reopened. The active work is target `L3`: orthogonal deterministic
sensors over `L0`, `L1`, and `L2` evidence.
