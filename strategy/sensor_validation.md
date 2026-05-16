# Sensor Validation Policy

Observation sensors may be useful at runtime before public corpora contain
examples of every pattern they detect. That is allowed, but the validation
status must be explicit.

## Validation Classes

| Class | Meaning | Allowed Use |
|-------|---------|-------------|
| `real-corpus` | The sensor fires on tracked or reviewable corpus rows and has measured positives and negatives. | Eligible for precision/recall claims on that corpus. |
| `synthetic-regression` | The sensor is covered only by hand-built fixtures or simulations. | Eligible for deterministic behavior checks, but not corpus precision/recall claims. |
| `runtime-hypothesis` | The sensor targets traffic shapes expected at runtime but absent from current public corpora. | May ship if cheap and precision-first, but reports must say corpus validation is unmeasured. |

## Reporting Rules

- Do not report precision or recall for a detector on a corpus that does not
  contain the detector's target pattern.
- Synthetic fixtures prove implementation mechanics only. They do not establish
  prevalence, precision, or recall in real data.
- If a detector ships with `synthetic-regression` or `runtime-hypothesis`
  coverage, the audit summary must say so directly.
- Promotion from `runtime-hypothesis` to `real-corpus` requires at least one
  reviewable positive pattern source and one reviewable benign hard-negative
  source.

## Current Application

The mechanical blob `escape_sequence_blob` rule targets dense literal escape
text such as `\uXXXX` and `\xNN`. A scan of the tracked public schema and
parapet-data YAML/JSON/JSONL artifacts found no literal `\uXXXX` rows. Until a
reviewable corpus artifact exercises that pattern, `escape_sequence_blob`
should be described as synthetic-regression/runtime-hypothesis coverage rather
than real-corpus validated coverage.

This policy also applies to future zero-width rule splits. Script-dependent
joiner behavior needs real or reviewable hard negatives before precision claims
are made for multilingual text.

