# Layer Guide

Parapet request/response pipeline:

1. `L0` input normalization
2. `L1` lightweight classifier
3. `L2a` data payload scanning (optional)
4. `L3_inbound` block/evidence pattern scanning
5. `L4` multi-turn risk scoring
6. upstream model call
7. `L3_outbound` tool call constraint enforcement
8. `L5a` output redaction

Layer docs:

- `strategy/l0.md`
- `strategy/l1.md`
- `strategy/l2a.md`
- `strategy/l3_inbound.md`
- `strategy/l3_outbound.md`
- `strategy/l4.md`
- `strategy/l5a.md`
