# L5a: Output Redaction

Naming note: this is the current implementation doc for output redaction,
which maps to target `L6` in `strategy/layers.md`. The code and config still
use legacy `L5a` names until the runtime rename lands.

L5a scans model responses for canary tokens and sensitive patterns, then redacts matches as `[REDACTED]`.

## Config

Default policy (`use_default_sensitive_patterns: true`, the default when omitted):

```yaml
layers:
  L5a:
    mode: redact
    window_chars: 4096   # optional, streaming window (canary-only mode)
```

Canary-only incremental streaming policy (explicitly disables default sensitive regex and configures no custom sensitive regex):

```yaml
use_default_sensitive_patterns: false
sensitive_patterns: []

canary_tokens:
  - "{{CANARY_a8f3e9b1}}"

layers:
  L5a:
    mode: redact
    window_chars: 4096
```

## What L5a scans

Pass order:

1. canary tokens (case-insensitive substring match)
2. sensitive patterns (regex)

Each match is redacted and logged with pattern and position metadata.

## Paths covered

1. Non-streaming: full response JSON after outbound tool-call enforcement.
2. Streaming: SSE and plain-text response streams after outbound tool-call enforcement.

## Streaming operational semantics

1. **Default sensitive patterns disable incremental text delivery**: By default, `parapet: v1` loads built-in sensitive regex patterns (`use_default_sensitive_patterns: true`). Because regex matches can span arbitrary SSE delta boundaries and exceed `window_chars`, whenever any sensitive regex is active (built-in defaults or custom `sensitive_patterns`), streaming L5a buffers decoded text until the logical terminal boundary of the content segment (OpenAI `finish_reason` or `[DONE]`, Anthropic `content_block_stop`, or plain-text EOF) before scanning and releasing output. Incremental token-by-token text delivery is disabled while any sensitive regex is enabled.
2. **Canary-only incremental operation**: Incremental rolling-window text delivery (`window_chars`) is active only when no sensitive regex patterns are configured. Enabling canary-only incremental streaming requires explicitly setting `use_default_sensitive_patterns: false` and configuring no custom `sensitive_patterns` (`sensitive_patterns: []` or omitted).
3. **64 KiB decoded-content cap may suppress a response**: Streaming L5a enforces a `64 KiB` (`65,536` characters) cap on buffered decoded text (`MAX_STREAM_REDACTION_BUFFER_CHARS`) and raw SSE line length (`MAX_SSE_RAW_LINE_CHARS`). If buffered decoded content exceeds `65,536` characters before it can be released (for example, a long response buffered until terminal boundary while sensitive regex is enabled), L5a suppresses the decoded text content rather than releasing unredacted output or buffering unboundedly.
4. **Malformed or ambiguous SSE fails closed**: In SSE mode, malformed or ambiguous framing or payloads (invalid or non-object JSON `data:` lines, non-string or unexpected `delta` fields, unpaired `event:` lines, unrecognized SSE field lines, raw lines exceeding `64 KiB`, OpenAI/Anthropic protocol mixing, out-of-order terminal events, or streams truncated at EOF before a logical terminal boundary) transition the stream into `fail_closed_sse_state` and withhold buffered and subsequent output.
5. **User-visible symptom**: When L5a suppresses content due to the `64 KiB` decoded-content cap or SSE fail-closed handling, the downstream client receives an HTTP `200` stream with empty assistant text (only initial role/control frames and terminal `finish_reason`/`[DONE]`, or an early-ended stream) even though the upstream model produced output. Operators should check L5a streaming warnings to distinguish Parapet suppression from upstream model flakiness.

## Streaming suppression diagnostics

Because Parapet's `parapet-effect-receipt/v0` envelope settles L3-outbound tool-call candidates inside `StreamProcessor` before L5a output redaction runs (and one-shot CLI effect runs reject streaming at ingress with `effect_streaming_unsupported`), streaming L5a suppression is reported via structured `WARN` tracing logs (`layer="L5a"`, `mode="streaming"`, `verdict="suppress"`, message `"L5a streaming response content suppressed"`). Diagnostics are emitted **once per affected stream** (never per chunk) and never log response content, raw chunks, secrets, URLs, paths, or canary values.

| `reason_code` | `detail` values | Meaning |
|---|---|---|
| `l5a_stream_decoded_content_over_cap` | `sse_decoded_content_over_cap`, `plain_text_decoded_content_over_cap` | Buffered decoded text exceeded the `64 KiB` (`65,536`-character) streaming redaction buffer cap. |
| `l5a_stream_malformed_or_ambiguous_sse` | `overlong_sse_line`, `incomplete_sse_line_at_eof`, `unpaired_sse_event_line`, `malformed_sse_field_line`, `malformed_sse_data_json`, `malformed_or_ambiguous_sse_payload`, `sse_protocol_or_boundary_violation`, `unterminated_sse_content_at_eof`, `deferred_sse_event_cap_exceeded` | SSE stream entered `fail_closed_sse_state` due to malformed, ambiguous, overlong, protocol-violating, or truncated SSE framing/payloads. |
| `l5a_stream_upstream_error_interrupted` | `upstream_error_event_with_buffered_content` | Upstream SSE `error` event arrived while text content was still buffered or an Anthropic text block was open; buffered text was withheld before passing through the error event. |

## Notes

1. `mode: redact` enables redaction; other mode values effectively disable L5a behavior.
2. Streaming L5a redaction is skipped entirely when the upstream SSE response
   uses gzip or deflate content encoding. The engine requests identity encoding
   and records a warning and skip counter when an upstream still returns a
   compressed stream.
