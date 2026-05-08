# Layer Guide

This document defines the target Parapet layer taxonomy after the 2026-05-08
L2 closure decision. The implementation still contains legacy names in several
modules and config keys; use the mapping table below until those names are
renamed deliberately.

## Target Pipeline

Inbound request path:

1. `L0` normalization
2. `L1` deterministic pattern gate
   - hard block patterns
   - evidence-only regex/signature extraction
   - untrusted content-size policy
3. `L2` lightweight lexical classifier
   - current compiled char n-gram SVM
4. `L3` orthogonal sensors and router
   - entropy, structure, obfuscation, sizing, token-shape
   - consumes `L0`, `L1`, and `L2` outputs
   - returns only `allow` or `block`
5. `L4` multi-turn risk scoring
6. upstream model call

Outbound response path:

7. `L5` outbound tool-call constraint enforcement
8. `L6` output redaction

Runtime may short-circuit on a blocking layer. Evaluation and shadow runs should
still execute later layers where practical so attribution remains measurable.

## Current Implementation Mapping

| Target layer | Current implementation/config name | Current doc | Rename status |
| --- | --- | --- | --- |
| `L0` normalization | `L0` / `normalize` | `strategy/l0.md` | aligned |
| `L1` deterministic pattern gate | `L3_inbound`, `block_patterns`, `untrusted_content_policy` | `strategy/l3_inbound.md` | pending rename |
| `L2` lightweight lexical classifier | `L1`, `layers/l1.rs`, `L1Harness` | `strategy/l1.md` | pending rename |
| `L3` orthogonal sensors/router | planned; Python simulation first | local direction docs | not implemented |
| `L4` multi-turn risk scoring | `L4` | `strategy/l4.md` | aligned |
| `L5` outbound tool constraints | `L3_outbound` | `strategy/l3_outbound.md` | pending rename |
| `L6` output redaction | `L5a` | `strategy/l5a.md` | pending rename |

## Closed Slots

- Legacy `L2a` Prompt Guard / payload-analysis slot is not the strategic path.
- Inline semantic transformer L2 is closed for the hot path.
- There is no specialist/escalation branch in the target runtime taxonomy.

## Program Summary

See `strategy/current_direction.md` for the current strategy summary and
`implement/l2/direction/` for local experiment/audit notes.
