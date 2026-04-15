# L2a: Optional Payload Analysis

L2a is Parapet's optional heavier-weight analysis slot for untrusted payloads and routed traffic.

Typical inputs:

- tool results
- retrieved documents
- API responses
- other untrusted payload-like text
- traffic escalated from `L1` when a deeper read is worth the latency

## Strategic Role

L2a exists to do work that `L1` should not try to do alone:

- disambiguate harder contextual cases
- analyze attack-like content inside larger payloads
- add a richer signal before final policy decisions

L2a is optional and should remain off by default unless the runtime budget and operating mode justify it.

## Current Direction

The active direction is away from Prompt Guard 2 and toward a Parapet-owned semantic implementation in the `L2a` slot.

Why:

- the old PG2 path is no longer the strategic bet
- recent research showed that lexical-only follow-on models help, but do not fully solve the harder benign families
- Parapet needs a Rust-first, CPU-viable semantic stage that fits the stack and can evolve with the rest of the system

Prompt Guard 2 should now be treated as historical background, not the canonical future of `L2a`.

## Current Candidate Shapes

The current working candidates are:

1. `L1 -> L2a`
2. `L1 -> L2b -> L2a`

Where:

- `L2a` is the semantic stage in the official stack
- `L2b` is an experimental lexical helper stage, not a canonical public layer

`L2b` may still be useful when:

- it reduces routed volume
- it improves precision on known lexical false-positive families
- it acts as a latency shield before semantic inference

## Runtime Requirements

The current `L2a` direction is designed around:

- Rust-owned inference and routing
- CPU-only operation
- explicit model assets
- bounded latency suitable for routed traffic, not full-stream always-on inference

The design target is an optional semantic stage that stays compatible with Parapet's existing deployment model.

## What L2a Should And Should Not Do

L2a should:

- analyze untrusted payload-like content that deserves more context than `L1` can provide
- complement deterministic layers such as `L3_inbound`
- improve stack behavior on context-sensitive cases without requiring a large-model dependency

L2a should not:

- replace `L0`, `L3_inbound`, `L3_outbound`, or `L5a`
- be treated as mandatory for every deployment
- depend on a GPU or an external inference service

## Status

Current status is research and implementation transition:

- PG2-based `L2a` is no longer the long-term strategy
- semantic `L2a` is the active implementation direction
- exact model choice, routing shape, and latency tradeoffs are still being validated

See `strategy/current_direction.md` for the program-level summary of why this change happened.
