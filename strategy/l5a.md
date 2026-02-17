# L5a: Output Redaction

L5a scans model responses for canary tokens and sensitive patterns, then redacts matches as `[REDACTED]`.

## Config

```yaml
layers:
  L5a:
    mode: redact
    window_chars: 4096   # optional, streaming window
```

## What L5a scans

Pass order:

1. canary tokens (case-insensitive substring match)
2. sensitive patterns (regex)

Each match is redacted and logged with pattern and position metadata.

## Paths covered

1. Non-streaming: full response JSON after outbound tool-call enforcement.
2. Streaming: rolling-window redaction over streamed content.

## Notes

1. `mode: redact` enables redaction; other mode values effectively disable L5a behavior.
2. Streaming responses with compressed chunk encodings can reduce boundary visibility for cross-chunk matches.
