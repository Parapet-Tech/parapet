# L3 Inbound: Pattern Enforcement And Evidence

L3 inbound scans inbound messages for known injection patterns and enforces untrusted content-size policy.

It is a deterministic regex/policy layer with high precision and lower recall than model-based layers.

## Config

```yaml
untrusted_content_policy:
  max_length: 20000

tools:
  web_search:
    allowed: true
    result_policy:
      max_length: 8000

layers:
  L3_inbound:
    mode: block      # block | shadow
```

## What L3 inbound does

Pass 1:

1. Scans all message content against `block_patterns`.
2. Scans tool-call argument strings against `block_patterns`.
3. Supports `PatternAction::Block` and `PatternAction::Evidence`.
4. Only `Block` matches can enforce; `Evidence` matches are collected as signals.

Pass 2:

1. Enforces `untrusted_content_policy.max_length` on untrusted messages.
2. Uses per-tool `result_policy.max_length` override for tool messages when configured.

Pass 3:

1. Enforces max-length on untrusted spans inside trusted messages.

## Modes

1. `block`: return `403` on first blocking verdict.
2. `shadow`: log would-block events, do not block.

## Eval snapshot (L3-only baseline)

From `implement/v3/baseline_evals.md`:

1. Precision: `99.2%`
2. Recall: `14.1%`
3. FP rate: `0.11%`

This is expected behavior for a high-precision regex layer. L2a/L1/L4 provide additional coverage.
