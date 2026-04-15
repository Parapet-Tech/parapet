# Layer Guide

Parapet request and response pipeline:

1. `L0` input normalization
2. `L1` lightweight lexical sensor
3. `L2a` optional payload analysis stage
4. `L3_inbound` block and evidence pattern scanning
5. `L4` multi-turn risk scoring
6. upstream model call
7. `L3_outbound` tool call constraint enforcement
8. `L5a` output redaction

Notes:

- `L1` is the always-fast first pass on untrusted message text.
- `L2a` is the optional heavier analysis slot for untrusted payloads and routed traffic. The active direction is away from Prompt Guard 2 and toward a Parapet-owned semantic path.
- `L2b` is not a canonical public layer. It is an experimental lexical helper stage that may sit between `L1` and `L2a` for precision recovery or latency shielding.

Layer docs:

- `strategy/l0.md`
- `strategy/l1.md`
- `strategy/l2a.md`
- `strategy/l3_inbound.md`
- `strategy/l3_outbound.md`
- `strategy/l4.md`
- `strategy/l5a.md`
- `strategy/current_direction.md`
